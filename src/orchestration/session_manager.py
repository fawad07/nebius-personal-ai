import asyncio
import contextlib
import logging

from src.orchestration.input_gate import AlwaysOnGate
from src.orchestration.token_buffer import split_into_speech_segments
from src.orchestration.turn_state import TurnState, TurnStateMachine
from utils.logging_util import new_session_id, new_turn_id, session_id_var, turn_id_var

_FLOAT32_BYTES = 4


class SessionManager:
    """
    Owns the runtime pipeline components and drives an async capture ->
    understand -> respond -> speak loop for a single conversation session.

    Blocking model/IO calls (mic reads, speaker writes, STT, TTS, speaker
    embedding, emotion classification, and the combined mood+reply LLM call)
    are offloaded to worker threads via asyncio.to_thread so the event loop
    stays free to listen for barge-in speech while the assistant is talking.
    """

    def __init__(
        self,
        audio_in,
        audio_out,
        stt_service,
        speaker_identifier,
        emotion_classifier,
        conversation_manager,
        tts,
        vad,
        loop_delay_ms: int = 10,
        token_buffer_min_chars: int = 12,
        token_buffer_max_chars: int = 80,
        silence_duration_ms: int = 800,
        max_utterance_ms: int = 15000,
        barge_in_speech_chunks: int = 2,
        barge_in_enabled: bool = True,
        input_gate=None,
        on_partial_transcript=None,
        on_state_change=None,
        on_reply=None,
        profile_updater=None,
        auto_enroll=None,
    ):
        self.audio_in = audio_in
        self.audio_out = audio_out
        self.stt = stt_service
        self.speaker_identifier = speaker_identifier
        self.emotion_classifier = emotion_classifier
        self.convo_manager = conversation_manager
        self.tts = tts
        self.vad = vad
        self.loop_delay_ms = loop_delay_ms
        self.token_buffer_min_chars = token_buffer_min_chars
        self.token_buffer_max_chars = token_buffer_max_chars
        # Optional, config-gated speaker-profile adaptation run off the hot
        # path: drift-update for known speakers, auto-enroll for strangers.
        self.profile_updater = profile_updater
        self.auto_enroll = auto_enroll
        # Number of consecutive speech chunks required before a barge-in is
        # accepted, to reduce false triggers from the agent's own playback
        # leaking into the open mic (see _listen_for_barge_in). A real fix is
        # acoustic echo cancellation (Phase 1).
        self.barge_in_speech_chunks = max(1, barge_in_speech_chunks)
        # When disabled (e.g. push-to-talk), the assistant plays its reply to
        # completion without listening for a voice interruption.
        self.barge_in_enabled = barge_in_enabled
        # Gates the *start* of each turn: always-on VAD by default, or
        # push-to-talk. See src/orchestration/input_gate.py.
        self.input_gate = input_gate or AlwaysOnGate()

        chunk_ms = (audio_in.config.chunk_size / audio_in.config.rate) * 1000
        self.silence_chunks_needed = max(1, round(silence_duration_ms / chunk_ms))
        self.max_utterance_chunks = max(1, round(max_utterance_ms / chunk_ms))

        self.logger = logging.getLogger("session_manager")
        self.on_partial_transcript = on_partial_transcript or (
            lambda text: self.logger.info(f"[partial] {text}")
        )
        # Optional observer of each completed exchange (final user text + reply
        # + mood); used e.g. by the browser UI. Best-effort, never fatal.
        self.on_reply = on_reply
        self._running = False
        # Single-flight guard for partial transcripts: at most one in flight
        # so mid-utterance partial passes can't stack up and starve the pool.
        self._partial_task = None
        # Explicit, observable turn lifecycle (LISTENING/THINKING/SPEAKING/
        # INTERRUPTED). on_state_change lets a UI or metrics layer subscribe.
        self.state_machine = TurnStateMachine(on_change=on_state_change, logger=self.logger)

    def start(self):
        """Start the audio input/output streams and mark the session active."""
        self.audio_in.start()
        self.audio_out.start()
        self._running = True
        self.logger.info("Session started.")

    def stop(self):
        """Stop the audio input/output streams and mark the session inactive."""
        self._running = False
        self.audio_in.stop()
        self.audio_out.stop()
        self.state_machine.transition(TurnState.IDLE)
        self.logger.info("Session stopped.")

    @property
    def state(self) -> TurnState:
        """Current turn-lifecycle state (LISTENING/THINKING/SPEAKING/...)."""
        return self.state_machine.state

    def run_forever(self):
        """Synchronous entry point; drives the async pipeline to completion."""
        asyncio.run(self._run_forever_async())

    async def _run_forever_async(self):
        session_id_var.set(new_session_id())
        self.start()
        leading_chunk = None
        try:
            while self._running:
                # Only gate the start of a fresh turn; a barge-in carry-over
                # (leading_chunk) means the user is already mid-utterance.
                if leading_chunk is None:
                    if not await self.input_gate.wait():
                        break
                leading_chunk = await self.handle_turn(leading_chunk)
        finally:
            self.stop()

    async def _capture_utterance_async(self, leading_chunk: bytes | None = None) -> bytes | None:
        """
        Accumulate audio chunks with VAD-based endpointing: wait for speech
        to start, then keep capturing until trailing silence (or a max
        utterance length safety cap) is reached. Returns None if the chunk
        that triggered capture wasn't actually speech.

        If leading_chunk is provided (e.g. the chunk that triggered a
        barge-in during the previous turn's playback), it seeds the buffer
        instead of being discarded.
        """
        if leading_chunk is not None:
            chunk = leading_chunk
        else:
            chunk = await asyncio.to_thread(self.audio_in.capture_chunk)
            if not await asyncio.to_thread(self.vad.is_speech, chunk):
                return None

        frames = [chunk]
        silence_chunks = 0

        while silence_chunks < self.silence_chunks_needed and len(frames) < self.max_utterance_chunks:
            chunk = await asyncio.to_thread(self.audio_in.capture_chunk)
            frames.append(chunk)
            if await asyncio.to_thread(self.vad.is_speech, chunk):
                silence_chunks = 0
            else:
                silence_chunks += 1

            if len(frames) % self.silence_chunks_needed == 0:
                self._maybe_emit_partial_transcript(list(frames))

        await asyncio.to_thread(self.vad.reset)
        return b"".join(frames)

    def _maybe_emit_partial_transcript(self, frames: list[bytes]):
        """
        Schedule a partial transcript only if one isn't already running, so
        successive triggers during a long utterance don't pile up unbounded
        background STT passes competing for the worker pool.
        """
        if self._partial_task is not None and not self._partial_task.done():
            return
        self._partial_task = asyncio.create_task(self._emit_partial_transcript(frames))

    async def _emit_partial_transcript(self, frames: list[bytes]):
        """Best-effort partial transcription of audio captured so far."""
        audio_bytes = b"".join(frames)
        try:
            # Cheap, low-latency decode (beam_size=1) since this is a throwaway
            # mid-utterance hint, not the final transcription.
            partial_text = await asyncio.to_thread(self.stt.transcribe, audio_bytes, 1)
        except Exception as e:
            self.logger.debug(f"Partial transcript failed: {e}")
            return
        if partial_text:
            self.on_partial_transcript(partial_text)

    def _schedule_speaker_adaptation(self, speaker):
        """
        Fire-and-forget speaker-profile maintenance: update a known speaker's
        profile toward the new embedding (drift adaptation), or auto-enroll a
        stranger -- but only when the corresponding component is configured.
        Runs in a worker thread so it never blocks the response.
        """
        if self.profile_updater is None and self.auto_enroll is None:
            return
        asyncio.create_task(asyncio.to_thread(self._adapt_speaker, speaker))

    def _adapt_speaker(self, speaker):
        try:
            if speaker.name is not None:
                if self.profile_updater is not None:
                    self.profile_updater.update(speaker.name, speaker.embedding)
            elif self.auto_enroll is not None:
                self.auto_enroll.enroll_embedding(speaker.embedding)
        except Exception as e:  # best-effort; must not break the conversation
            self.logger.warning(f"Speaker profile adaptation failed: {e}")

    async def _listen_for_barge_in(self) -> bytes:
        """
        Continuously capture mic audio while the assistant is speaking.
        Returns the first chunk of a *sustained* speech burst once detected,
        so it can seed the next utterance capture instead of being discarded.

        Requiring several consecutive speech chunks (barge_in_speech_chunks)
        guards against the agent's own playback bleeding into an open mic and
        interrupting itself; proper acoustic echo cancellation is the real
        fix (Phase 1).
        """
        first_speech_chunk = None
        consecutive = 0
        while True:
            chunk = await asyncio.to_thread(self.audio_in.capture_chunk)
            if await asyncio.to_thread(self.vad.is_speech, chunk):
                if consecutive == 0:
                    first_speech_chunk = chunk
                consecutive += 1
                if consecutive >= self.barge_in_speech_chunks:
                    await asyncio.to_thread(self.vad.reset)
                    return first_speech_chunk
            else:
                consecutive = 0
                first_speech_chunk = None

    async def handle_turn(self, leading_chunk: bytes | None = None) -> bytes | None:
        """
        Process a single capture -> respond -> speak turn. Returns a leading
        audio chunk to seed the next turn's capture if a barge-in occurred,
        otherwise None.
        """
        turn_id_var.set(new_turn_id())

        self.state_machine.transition(TurnState.LISTENING)
        audio_bytes = await self._capture_utterance_async(leading_chunk)
        if audio_bytes is None:
            return None

        self.state_machine.transition(TurnState.THINKING)
        # Perception fan-out: transcription, speaker-ID, and emotion are
        # independent of one another, so run all three concurrently instead
        # of awaiting STT before the other two even start.
        stt_task = asyncio.create_task(asyncio.to_thread(self.stt.transcribe, audio_bytes))
        speaker_task = asyncio.create_task(
            asyncio.to_thread(self.speaker_identifier.identify_detailed, audio_bytes)
        )
        emotion_task = asyncio.create_task(asyncio.to_thread(self.emotion_classifier.classify, audio_bytes))
        text, speaker, emotion = await asyncio.gather(stt_task, speaker_task, emotion_task)

        if not text:
            return None
        self.logger.info(f"User said: {text!r}")

        speaker_group = speaker.group
        # Best-effort profile adaptation off the critical path.
        self._schedule_speaker_adaptation(speaker)

        # Single structured LLM call: infers mood/intent/sensitivity AND
        # generates the reply in one round trip (see ConversationManager.
        # respond_with_mood), instead of two serialized LLM calls.
        result = await asyncio.to_thread(
            self.convo_manager.respond_with_mood, text, emotion, speaker_group
        )
        mood = {
            "mood": result["mood"],
            "intent": result["intent"],
            "sensitivity": result["sensitivity"],
        }
        reply_text = result["reply"]

        return await self._speak_with_barge_in(text, reply_text, mood)

    async def _speak_with_barge_in(self, user_text: str, reply_text: str, mood: dict) -> bytes | None:
        self.state_machine.transition(TurnState.SPEAKING)

        # Barge-in disabled (e.g. push-to-talk): play the whole reply, no
        # concurrent mic listening.
        if not self.barge_in_enabled:
            await self._speak_reply(user_text, reply_text, mood)
            return None

        speak_task = asyncio.create_task(self._speak_reply(user_text, reply_text, mood))
        barge_task = asyncio.create_task(self._listen_for_barge_in())

        done, _ = await asyncio.wait({speak_task, barge_task}, return_when=asyncio.FIRST_COMPLETED)

        if speak_task in done:
            barge_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await barge_task
            speak_task.result()
            return None

        # barge_task finished first: user started speaking during playback.
        next_leading_chunk = barge_task.result()
        speak_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await speak_task
        self.state_machine.transition(TurnState.INTERRUPTED)
        self.logger.info("Barge-in detected; cancelling assistant playback.")
        return next_leading_chunk

    async def _speak_reply(self, user_text: str, reply_text: str, mood: dict):
        """
        Speak the (already fully-generated) reply in sentence-sized segments
        so playback can start on the first segment while later segments are
        still being synthesized, and so a barge-in can interrupt between --
        or within -- segments rather than only after the whole reply.
        """
        if self.on_reply is not None:
            try:
                self.on_reply(user_text, reply_text, mood)
            except Exception:
                self.logger.exception("on_reply callback failed.")

        segments = split_into_speech_segments(
            reply_text,
            min_chars=self.token_buffer_min_chars,
            max_chars=self.token_buffer_max_chars,
        )
        for segment in segments:
            await self._synthesize_and_play(segment, mood)

        self.convo_manager.update_history(
            user_text=user_text,
            reply_text=reply_text,
            mood=mood,
        )

    async def _synthesize_and_play(self, chunk_text: str, mood: dict):
        try:
            reply_audio = await asyncio.to_thread(self.tts.synthesize, chunk_text, mood)
            await self._play_audio_async(reply_audio)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")

    async def _play_audio_async(self, audio_bytes: bytes):
        """
        Write audio to the output stream in small sub-chunks so a barge-in
        cancellation can interrupt mid-utterance rather than only between
        whole TTS segments.
        """
        frame_bytes = self.audio_out.config.chunk_size * _FLOAT32_BYTES
        for i in range(0, len(audio_bytes), frame_bytes):
            await asyncio.to_thread(self.audio_out.play, audio_bytes[i:i + frame_bytes])
