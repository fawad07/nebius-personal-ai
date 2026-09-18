import logging

from utils.logging_util import session_id_var

logger = logging.getLogger("conversation")


class ConversationManager:
    """
    Handles conversation flow and persona behavior using a real LLM.
    Supports both full replies and streaming replies.
    """

    _DEFAULT_PERSONA = {
        "style": "supportive, calm, emotionally aware",
        "boundaries": (
            "You are an AI assistant. You do not form romantic or "
            "dependent relationships. You maintain healthy emotional boundaries."
        ),
        "tone_map": {
            "neutral": "balanced and clear",
            "calm": "soft and steady",
            "sad": "gentle and warm",
            "excited": "energetic and upbeat",
            "angry": "calm and de-escalating",
            "stressed": "reassuring and steady",
        },
    }

    def __init__(self, llm_client, persona_config: dict | None = None, store=None,
                 history_limit: int = 20, tools=None):
        self.llm = llm_client
        self.persona = self._build_persona(persona_config or {})
        # Optional persistence: when a store is provided, seed the in-memory
        # history from prior sessions so the agent carries continuity across
        # restarts instead of starting cold.
        self.store = store
        self.history_limit = history_limit
        self.history = store.recent(history_limit) if store else []
        if self.history:
            logger.info(f"Loaded {len(self.history)} prior conversation turn(s) from memory.")
        # Optional tool layer. When present and non-empty, the model may request
        # a tool via an "action" object in its structured reply.
        self.tools = tools if (tools and len(tools) > 0) else None

    def _build_persona(self, cfg: dict) -> dict:
        """
        Merge the ``conversation:`` config section over the built-in defaults
        so persona style, boundaries, and tone map are actually configurable
        (they were previously hardcoded and the config was ignored).
        """
        tone_map = dict(self._DEFAULT_PERSONA["tone_map"])
        tone_map.update(cfg.get("tone_map") or {})
        return {
            "style": cfg.get("persona_style") or self._DEFAULT_PERSONA["style"],
            "boundaries": cfg.get("boundaries") or self._DEFAULT_PERSONA["boundaries"],
            "tone_map": tone_map,
        }

    def _select_tone(self, mood: dict) -> str:
        mood_label = mood.get("mood", "neutral")
        return self.persona["tone_map"].get(mood_label, "balanced and clear")

    def _recent_history(self, limit: int = 6) -> str:
        formatted = []
        for h in self.history[-limit:]:
            formatted.append(f"User: {h['user']}\nAgent: {h['reply']}")
        return "\n".join(formatted)

    def _build_user_prompt(self, text: str, mood: dict, speaker_group: str) -> str:
        tone = self._select_tone(mood)
        recent_context = self._recent_history()

        return f"""
Speaker group: {speaker_group}
User mood: {mood.get('mood')}
User intent: {mood.get('intent')}
Sensitivity level: {mood.get('sensitivity')}
Current tone: {tone}

Recent conversation:
{recent_context}

User says: "{text}"
"""

    def respond_with_mood(self, text: str, emotion: dict, speaker_group: str) -> dict:
        """
        Single structured LLM call that infers mood/intent/sensitivity AND
        generates the reply in one round trip, instead of two serialized
        calls (mood inference, then reply generation). Cuts per-turn LLM
        latency and cost roughly in half.

        Returns a dict with keys: mood, intent, sensitivity, reply.
        """
        recent_context = self._recent_history()

        tools_block = ""
        action_key_line = ""
        if self.tools is not None:
            tools_block = f"""
You can use a tool to answer. Available tools:
{self.tools.describe()}

To use one, set "action" to {{"name": <tool>, "args": {{...}}}}; otherwise set "action" to null.

IMPORTANT tool rules — follow exactly:
- When the user shares a personal fact/preference ("remember…", "my favourite X is…",
  "I like…"), you MUST call remember_fact. Do not just say "got it" without the tool.
- When the user asks what you know/remember about them ("what's my favourite…?"),
  you MUST call recall_facts and answer from its result — never guess or say you
  don't know without calling it first.
- When the user says "take a note / note down" WITH content, call save_note with the
  actual content only (e.g. "buy milk"), never the words "take a note". If they say
  "take a note" with no content yet, ask what to note (action null).
- "what are my notes?" -> call list_notes.
- You have NO reminder, alarm, or scheduling ability. Never claim to have set a
  reminder. If asked, save it as a note instead and say you noted it.
Do not invent tools or arguments.
"""
            action_key_line = "\n- action: a tool request object, or null (see tools below)"

        system_prompt = f"""
You are a conversational AI assistant with an emotion/intent analysis module built in.

Persona:
- Style: {self.persona['style']}
- Boundaries: {self.persona['boundaries']}

You respond with emotional awareness, but keep healthy boundaries.
Keep replies concise, supportive, and context-aware.

You MUST return a single JSON object with exactly these keys:
- mood: one of [neutral, calm, stressed, sad, excited, angry]
- intent: one of [ask_for_help, venting, casual_chat, task_request]
- sensitivity: one of [low, medium, high]
- reply: the assistant's reply text to the user (plain text, no labels, no markdown){action_key_line}

Base mood/intent/sensitivity on the user's text, detected vocal emotion, speaker group,
and recent conversation context. Do not include any extra commentary.
{tools_block}"""
        user_prompt = f"""
Speaker group: {speaker_group}
Detected vocal emotion: {emotion.get('label')}

Recent conversation:
{recent_context}

User says: "{text}"
"""

        default = {
            "mood": "neutral",
            "intent": "casual_chat",
            "sensitivity": "medium",
            "reply": "I'm here with you. Let's take this one step at a time.",
        }

        # 1) Preferred path: one structured call yields mood + reply together.
        try:
            result = self.llm.generate_json(system_prompt, user_prompt)
        except Exception:
            logger.exception("Structured LLM call failed; trying plain-text fallback.")
            result = None

        if isinstance(result, dict) and result.get("reply"):
            reply = str(result["reply"]).strip()
            # If the model requested a tool, run it and let the model phrase the
            # result naturally (a second, cheap call only when a tool is used).
            action = result.get("action") if self.tools is not None else None
            if isinstance(action, dict) and self.tools.has(action.get("name", "")):
                reply = self._run_tool(text, action, base_reply=reply)
            return {
                "mood": result.get("mood", "neutral"),
                "intent": result.get("intent", "casual_chat"),
                "sensitivity": result.get("sensitivity", "medium"),
                "reply": reply,
            }

        # 2) Fallback: the model didn't return usable structured JSON (common
        #    with smaller local models). Rather than speak a canned line, make
        #    a plain-text call so the user still gets the model's real reply --
        #    just without the inline mood/intent analysis (defaults applied).
        fallback_reply = self._plain_reply_fallback(text, speaker_group)
        if fallback_reply:
            logger.warning("Structured JSON unavailable; used plain-text reply fallback.")
            return {
                "mood": "neutral",
                "intent": "casual_chat",
                "sensitivity": "medium",
                "reply": fallback_reply,
            }

        # 3) Last resort: the LLM is unreachable/unusable entirely.
        logger.warning("Plain-text fallback unavailable; using canned safe response.")
        return default

    def _run_tool(self, text: str, action: dict, base_reply: str) -> str:
        """
        Execute the tool the model requested and turn its raw result into a
        natural spoken reply. Falls back to appending the raw result to the
        model's original reply if the phrasing call fails.
        """
        name = action.get("name", "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        tool_result = self.tools.execute(name, args)
        logger.info("Tool %s(%s) -> %s", name, args, tool_result)

        try:
            system_prompt = f"""
You are {self.persona['style']}. You just used a tool to help the user.
Weave the tool's result into a short, natural spoken reply (plain text, no labels).
Never contradict yourself in one reply. If the tool found nothing but the recent
conversation already contains the answer, give that answer confidently and do NOT
mention that the tool found nothing.
"""
            user_prompt = (
                f'Recent conversation:\n{self._recent_history()}\n\n'
                f'User said: "{text}"\n'
                f'Tool "{name}" returned: {tool_result}\n'
                "Reply to the user naturally and briefly."
            )
            phrased = self.llm.generate_text(system_prompt, user_prompt).strip()
            return phrased or f"{base_reply} {tool_result}".strip()
        except Exception:
            logger.exception("Tool-result phrasing failed; appending raw result.")
            return f"{base_reply} {tool_result}".strip()

    def _plain_reply_fallback(self, text: str, speaker_group: str) -> str:
        """
        Best-effort plain-text reply used when structured JSON generation
        fails. Returns "" only if even a plain call can't produce text.
        """
        try:
            reply = self.respond(text, {"mood": "neutral"}, speaker_group)
            return reply.strip() if reply else ""
        except Exception:
            logger.exception("Plain-text reply fallback failed.")
            return ""

    def respond(self, text: str, mood: dict, speaker_group: str) -> str:
        system_prompt = f"""
You are a conversational AI assistant.

Persona:
- Style: {self.persona['style']}
- Boundaries: {self.persona['boundaries']}

You respond with emotional awareness, but keep healthy boundaries.
You keep replies concise, supportive, and context-aware.
Return ONLY the assistant's reply text, no JSON, no labels.
"""
        user_prompt = self._build_user_prompt(text, mood, speaker_group)

        try:
            reply = self.llm.generate_text(system_prompt, user_prompt)
            return reply.strip()
        except Exception:
            return "I'm here with you. Let's take this one step at a time."

    def stream_reply(self, text: str, mood: dict, speaker_group: str):
        """
        Yield reply tokens as they are generated.
        """
        system_prompt = f"""
You are a conversational AI assistant.

Persona:
- Style: {self.persona['style']}
- Boundaries: {self.persona['boundaries']}

You respond with emotional awareness, but keep healthy boundaries.
You keep replies concise, supportive, and context-aware.
Return ONLY the assistant's reply text, no JSON, no labels.
"""
        user_prompt = self._build_user_prompt(text, mood, speaker_group)

        try:
            for token in self.llm.stream_text(system_prompt, user_prompt):
                yield token
        except Exception:
            yield "I'm here with you. Let's take this one step at a time."

    def update_history(self, user_text: str, reply_text: str, mood: dict):
        self.history.append(
            {
                "user": user_text,
                "reply": reply_text,
                "mood": mood,
            }
        )
        # Persist the turn (best-effort) so it survives restarts.
        if self.store is not None:
            self.store.append(
                session_id=session_id_var.get(),
                user_text=user_text,
                reply_text=reply_text,
                mood=mood,
            )
        # Cap unbounded in-memory growth; only the recent tail is ever used
        # in prompts and the full record lives in the store.
        max_kept = max(self.history_limit, 50)
        if len(self.history) > max_kept:
            self.history = self.history[-max_kept:]

