"""
telemetry_off sets the opt-out environment variables of every third-party
library this package can pull in, before those libraries load.

Import order matters: ``veridic_eval/__init__.py`` imports this module first,
so the variables below are in place before Phoenix, OpenTelemetry,
HuggingFace/transformers, RAGAS/LangChain, posthog, scarf, sentry, or promptfoo
are imported or spawned.

Env vars rather than avoided call sites: several of these libraries read their
opt-out flags at *import time* and start exporter or analytics threads during
import, so an API-level opt-out written later runs after the thread exists.

The evaluation itself reads a local Postgres and a local Ollama, so an outbound
call would come from one of these dependencies rather than from this package.
"""
from __future__ import annotations

import os

# Env flags are only applied once; re-import is a no-op.
_APPLIED = False

# Universal opt-out standard honoured by a growing number of tools.
# (https://consoledonottrack.com/)
_DO_NOT_TRACK = {
    "DO_NOT_TRACK": "1",
}

# Arize Phoenix - collects "basic web analytics" by default. Kill both the
# current and the historical variable names, and make sure we never point at
# a remote collector.
_PHOENIX = {
    "PHOENIX_TELEMETRY_ENABLED": "false",   # current opt-out flag
    "PHOENIX_ENABLE_TELEMETRY": "false",    # older/alt flag (belt & braces)
    "PHOENIX_DISABLE_TELEMETRY": "true",
    # If any Phoenix client is constructed, keep it pointed at localhost so a
    # stray span can never egress. We never launch the server anyway.
    "PHOENIX_HOST": os.environ.get("PHOENIX_HOST", "127.0.0.1"),
}

# OpenTelemetry SDK (Phoenix / OpenInference sit on top of this). Disabling the
# SDK outright means no span is ever created or exported, regardless of caller.
_OTEL = {
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "OTEL_TRACES_SAMPLER": "always_off",
    # empty endpoints => nothing to export to even if the above were ignored
    "OTEL_EXPORTER_OTLP_ENDPOINT": "",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "",
}

# HuggingFace / transformers - LettuceDetect loads a HF model. Disable HF Hub
# telemetry and advisory pings. (HF_HUB_OFFLINE is left to config so a first
# download can still happen; set it in .env once the model is cached.)
_HUGGINGFACE = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "DISABLE_TELEMETRY": "1",
}

# PostHog / Scarf / Sentry - used transitively by RAGAS, LangChain and friends
# for anonymous usage analytics and crash reporting.
_ANALYTICS_VENDORS = {
    # chromadb's variable, set defensively; chromadb is not a dependency (README)
    "ANONYMIZED_TELEMETRY": "false",
    "POSTHOG_DISABLED": "true",
    "SCARF_NO_ANALYTICS": "true",
    "DO_NOT_TRACK_SCARF": "true",
    "SENTRY_DSN": "",                  # empty DSN => Sentry never initialises
    "LANGCHAIN_TRACING_V2": "false",   # no LangSmith egress
    "LANGCHAIN_TRACING": "false",
    "RAGAS_DO_NOT_TRACK": "true",
}

# promptfoo - Node CLI we shell out to for deterministic abstention scoring.
# These are read by the child process; we also pass them explicitly when we
# spawn it (see abstention.py), so they hold even if the parent env is stripped.
_PROMPTFOO = {
    "PROMPTFOO_DISABLE_TELEMETRY": "1",
    "PROMPTFOO_DISABLE_ANALYTICS": "1",
    "PROMPTFOO_DISABLE_UPDATE": "1",
    "PROMPTFOO_DISABLE_SHARING": "1",
    "PROMPTFOO_DISABLE_REDTEAM_REMOTE_GENERATION": "1",
}

# Streamlit (cells_app.py) posts anonymous usage stats from the browser unless
# gatherUsageStats is off. The `streamlit run` server reads its config before it
# imports the script, so `.streamlit/config.toml` at the repo root is what
# actually holds for the UI; this group covers a programmatic launch and any
# child process that inherits the env.
_STREAMLIT = {
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
}

# The full set, in application order.
_ALL_GROUPS = (
    _DO_NOT_TRACK,
    _PHOENIX,
    _OTEL,
    _HUGGINGFACE,
    _ANALYTICS_VENDORS,
    _PROMPTFOO,
    _STREAMLIT,
)


def promptfoo_env() -> dict:
    """Env overrides to pass explicitly when spawning the promptfoo CLI."""
    return dict(_PROMPTFOO)


def disable_all_telemetry(force: bool = False) -> dict:
    """
    Set every telemetry opt-out env var. Existing user-set values are preserved
    unless ``force=True`` (we never want to *enable* telemetry a user disabled,
    but we do want to disable anything left at its phone-home default).

    Returns the dict of variables now in effect (for logging / auditing).
    """
    global _APPLIED
    applied: dict = {}
    for group in _ALL_GROUPS:
        for key, value in group.items():
            if force or key not in os.environ or _looks_like_optin(key):
                os.environ[key] = value
            applied[key] = os.environ[key]
    _APPLIED = True
    return applied


def _looks_like_optin(key: str) -> bool:
    """
    True if the current value of ``key`` looks like telemetry is *enabled*
    (e.g. left at a truthy default), so we override it to the off value.
    Only applied to the boolean-style enable flags.
    """
    enable_flags = {
        "PHOENIX_TELEMETRY_ENABLED",
        "PHOENIX_ENABLE_TELEMETRY",
        "ANONYMIZED_TELEMETRY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
    }
    if key not in enable_flags:
        return False
    return os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


# Apply immediately on import.
if not _APPLIED:
    disable_all_telemetry()
