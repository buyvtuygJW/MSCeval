"""Telemetry must be off the moment the package is imported."""
import os


def test_import_disables_telemetry():
    import veridic_eval  # noqa: F401  (import applies the kill switch)

    assert os.environ["DO_NOT_TRACK"] == "1"
    assert os.environ["PHOENIX_TELEMETRY_ENABLED"] == "false"
    assert os.environ["PHOENIX_ENABLE_TELEMETRY"] == "false"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"
    assert os.environ["OTEL_TRACES_EXPORTER"] == "none"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["ANONYMIZED_TELEMETRY"] == "false"
    assert os.environ["PROMPTFOO_DISABLE_TELEMETRY"] == "1"
    assert os.environ["SENTRY_DSN"] == ""


def test_promptfoo_env_is_all_off():
    from veridic_eval.telemetry_off import promptfoo_env

    env = promptfoo_env()
    assert env["PROMPTFOO_DISABLE_TELEMETRY"] == "1"
    assert env["PROMPTFOO_DISABLE_UPDATE"] == "1"
    assert env["PROMPTFOO_DISABLE_SHARING"] == "1"


def test_optin_flag_is_overridden():
    # A user who left telemetry enabled gets it forced off.
    os.environ["PHOENIX_TELEMETRY_ENABLED"] = "true"
    from veridic_eval.telemetry_off import disable_all_telemetry

    applied = disable_all_telemetry()
    assert applied["PHOENIX_TELEMETRY_ENABLED"] == "false"
    assert os.environ["PHOENIX_TELEMETRY_ENABLED"] == "false"
