"""
app.py — Gradio entry point for the Hugging Face Space (free SDK).

Wraps the exact same torch-free reasoning core the local app uses
(`webapp.agent_core.AgentCore`): NVIDIA Nemotron on Nebius Token Factory, plus
memory, notes, and Tavily web search. Voice is not needed here — this is the
hands-on demo judges can type into; the video shows the full voice pipeline.

Secrets (NEBIUS_API_KEY, TAVILY_API_KEY) come from the Space's secret store.
"""

from __future__ import annotations

import gradio as gr

from webapp.agent_core import AgentCore

# Hugging Face ZeroGPU (the only free hardware for Gradio Spaces) refuses to
# start unless it detects at least one @spaces.GPU function. This app does all
# inference via the remote Nebius API and never needs a GPU, so this is a tiny
# placeholder that satisfies the runtime check. Guarded so local runs (where the
# `spaces` package isn't installed) are unaffected.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():  # never called; existence satisfies ZeroGPU
        return True
except Exception:
    pass

core = AgentCore()

INTRO = (
    "Hi — I'm a private personal assistant that reasons with an **NVIDIA Nemotron** "
    "model on **Nebius Token Factory**. I remember facts you tell me, take and list "
    "notes, and search the web with **Tavily**. Try an example below, or just ask."
)

EXAMPLES = [
    "What's the tallest building in the world?",
    "Remember that my favourite colour is teal",
    "What's my favourite colour?",
    "Take a note: buy oat milk",
    "What are my notes?",
]


def _respond(message, chat, request: gr.Request = None):
    """One handler: take the textbox string directly (never re-read it from the
    chatbot, which Gradio 6 may hand back as a list), answer, append both."""
    message = str(message or "").strip()
    if not message:
        return "", chat or []
    chat = (chat or []) + [{"role": "user", "content": message}]
    session = getattr(request, "session_hash", None) or "public"
    try:
        answer = core.respond(session, message).get("reply", "")
    except Exception as e:  # never crash the UI mid-demo
        answer = f"(sorry — that turn failed: {e})"
    chat = chat + [{"role": "assistant", "content": answer}]
    return "", chat


with gr.Blocks(title="Nebius Personal AI") as demo:
    gr.Markdown(
        "# 🎙️ Nebius Personal AI\n"
        "Personal AI · **Nemotron on Nebius** · memory · notes · Tavily web search"
    )
    chatbot = gr.Chatbot(value=[{"role": "assistant", "content": INTRO}], height=440)
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask me something, or say 'take a note: …' / 'search the web for …'",
            show_label=False, autofocus=True, scale=8, container=False,
        )
        send = gr.Button("Send", variant="primary", scale=1)
    gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one")

    # Submit on Enter or the Send button.
    for trigger in (msg.submit, send.click):
        trigger(_respond, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
