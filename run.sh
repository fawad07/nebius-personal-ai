#!/bin/bash

# -------------------------
# Configuration
# -------------------------

APP_ENTRY="src/main.py"
APP_MODULE="src.main"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/agent.log"
ENV_FILE=".env"

DEBUG=false
NO_LOG=false
MODEL_OVERRIDE=""

# -------------------------
# Helper functions
# -------------------------

print_banner() {
    echo "========================================="
    echo "   Emotion-Aware Voice Agent - Launcher  "
    echo "========================================="
    echo
}

print_usage() {
    echo "Usage: ./run.sh [options]"
    echo
    echo "Options:"
    echo "  --debug           Run in debug mode (no log redirection, verbose output)"
    echo "  --no-log          Do not write logs to file, only console"
    echo "  --model <name>    Override LLM model (e.g., gpt-4o-mini)"
    echo "  --help            Show this help message"
    echo
}

load_env() {
    if [ -f "$ENV_FILE" ]; then
        echo "[launcher] Loading environment from $ENV_FILE"
        # shellcheck disable=SC2046
        export $(grep -v '^#' "$ENV_FILE" | xargs)
    fi
}

activate_venv() {
    if [ -d ".venv" ]; then
        echo "[launcher] Activating virtual environment (.venv)"
        # shellcheck disable=SC1091
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        echo "[launcher] Activating virtual environment (venv)"
        # shellcheck disable=SC1091
        source venv/bin/activate
    else
        echo "[launcher] No virtual environment found (.venv/ or venv/)."
    fi
}

# Read a top-level "key: value" string from config/settings.yaml (best effort).
# Strips an inline "# comment" and surrounding quotes so a commented value line
# (e.g. base_url: "https://..."  # note) yields just the value.
config_value() {
    local key="$1"
    grep -E "^[[:space:]]*${key}:" "config/settings.yaml" 2>/dev/null \
        | head -1 \
        | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^\"//; s/\"[[:space:]]*$//; s/[[:space:]]*$//"
}

# Fatal, actionable checks BEFORE we spend a minute loading models only to die.
preflight() {
    echo "[launcher] Running pre-flight checks..."
    local ok=true

    if [ ! -f "$APP_ENTRY" ]; then
        echo "  [ERROR] Entry file not found: $APP_ENTRY"
        ok=false
    fi

    # --- LLM provider: hosted OpenAI (needs key) vs local (needs a server) ---
    local provider base_url
    provider="$(config_value provider)"
    [ -z "$provider" ] && provider="openai"
    base_url="$(config_value base_url)"
    [ "$base_url" = "null" ] && base_url=""

    # Is the endpoint loopback (keyless local server) or remote (needs a key)?
    local is_local=false
    case "$base_url" in
        *localhost*|*127.0.0.1*|*0.0.0.0*) is_local=true ;;
    esac

    if [ "$provider" = "ollama" ] || { [ "$provider" = "openai_compatible" ] && [ "$is_local" = true ]; }; then
        [ -z "$base_url" ] && base_url="http://localhost:11434/v1"
        echo "  [ok] LLM provider: $provider (local endpoint: $base_url)."
        # Best-effort reachability check (curl optional).
        if command -v curl >/dev/null 2>&1; then
            if curl -sf -m 2 "${base_url%/v1}/api/tags" >/dev/null 2>&1 \
               || curl -sf -m 2 "${base_url}/models" >/dev/null 2>&1; then
                echo "  [ok] Local LLM endpoint reachable: $base_url"
            else
                echo "  [WARNING] Local LLM endpoint not reachable at $base_url"
                echo "            Start it first, e.g.:  ollama serve   (and: ollama pull <model>)"
            fi
        fi
    elif [ "$provider" = "openai_compatible" ]; then
        # Remote OpenAI-compatible endpoint (e.g. Nebius Token Factory): needs a key.
        if [ -n "$NEBIUS_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]; then
            echo "  [ok] LLM: $provider via $base_url (API key set)."
        else
            echo "  [ERROR] No API key for remote endpoint: $base_url"
            echo "          Set NEBIUS_API_KEY in $ENV_FILE (see .env.example)."
            ok=false
        fi
    elif [ -z "$OPENAI_API_KEY" ]; then
        echo "  [ERROR] OPENAI_API_KEY is not set (llm.provider = $provider)."
        if [ ! -f "$ENV_FILE" ]; then
            echo "          No .env file found. Create one:"
            echo "              cp .env.example .env    # then add your key"
        else
            echo "          Add it to $ENV_FILE:  OPENAI_API_KEY=sk-..."
        fi
        echo "          ...or switch to a local model: set llm.provider: \"ollama\" in config/settings.yaml"
        ok=false
    else
        echo "  [ok] OPENAI_API_KEY is set."
    fi

    # --- TTS engine (default: system voice; xtts needs a sample + Coqui TTS) ---
    local tts_engine
    tts_engine="$(config_value engine)"
    [ -z "$tts_engine" ] && tts_engine="system"
    if [ "$tts_engine" = "xtts" ]; then
        local voice_sample
        voice_sample="$(config_value voice_profile_path)"
        [ -z "$voice_sample" ] && voice_sample="data/samples/agent_voice.wav"
        if [ ! -f "$voice_sample" ]; then
            echo "  [WARNING] XTTS voice sample not found: $voice_sample (add a clean 6-10s mono WAV)."
        else
            echo "  [ok] XTTS voice sample present: $voice_sample"
        fi
        if ! python3 -c "import TTS" 2>/dev/null; then
            echo "  [WARNING] 'TTS' not importable. Install requirements-voiceclone.txt, or set tts.engine: system."
        fi
    else
        echo "  [ok] TTS engine: system voice."
        if [ "$(uname)" = "Darwin" ] && ! command -v say >/dev/null 2>&1; then
            echo "  [WARNING] macOS 'say' not found; system voice will be silent (text reply still shows)."
        fi
    fi

    # --- Core packages ---
    if ! python3 -c "import openai" 2>/dev/null; then
        echo "  [ERROR] Python package 'openai' not importable. Run: pip install -r requirements.txt"
        ok=false
    fi

    if [ "$ok" != true ]; then
        echo
        echo "[launcher] Pre-flight failed. Fix the [ERROR] items above and re-run."
        echo "           Tip: verify the model stack with:  python3 -m scripts.smoke_test --stt-model tiny"
        exit 1
    fi

    echo "[launcher] Pre-flight checks passed."
}

prepare_logging() {
    if [ "$NO_LOG" = true ]; then
        echo "[launcher] Logging disabled (NO_LOG=true)."
        return
    fi

    mkdir -p "$LOG_DIR"
    echo "[launcher] Logs will be written to $LOG_FILE"
}

run_app() {
    CMD="python3 -m $APP_MODULE"

    if [ -n "$MODEL_OVERRIDE" ]; then
        export LLM_MODEL_OVERRIDE="$MODEL_OVERRIDE"
        echo "[launcher] Overriding LLM model: $MODEL_OVERRIDE"
    fi

    echo "[launcher] Starting agent. Press Ctrl-C to stop."

    if [ "$DEBUG" = true ]; then
        echo "[launcher] DEBUG mode: output goes straight to this console."
        echo
        $CMD
        return
    fi

    if [ "$NO_LOG" = true ]; then
        echo "[launcher] Console-only mode (no log file)."
        echo
        $CMD
        return
    fi

    # Redirected mode: keep the console clean, but if the agent dies, show the
    # tail of the log right here instead of leaving the user staring at silence.
    echo "[launcher] Output is redirected to $LOG_FILE"
    echo "[launcher] Watch it live in another terminal:  tail -f $LOG_FILE"
    echo
    $CMD >> "$LOG_FILE" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[ERROR] Agent exited with code $status. Last lines of $LOG_FILE:"
        echo "-----------------------------------------------------------------"
        tail -n 20 "$LOG_FILE"
        echo "-----------------------------------------------------------------"
        echo "[launcher] For full console output next time, run:  ./run.sh --debug"
        exit $status
    fi
}

# -------------------------
# Parse CLI arguments
# -------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG=true
            shift
            ;;
        --no-log)
            NO_LOG=true
            shift
            ;;
        --model)
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "[launcher] Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# -------------------------
# Main
# -------------------------

print_banner
load_env
activate_venv
preflight
prepare_logging
run_app

