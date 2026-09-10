"""Hosted entrypoint for the VERIDIC eval viewer (Streamlit Community Cloud).

Why this file has to sit in the repository root
-----------------------------------------------
`streamlit run` adds *only the entrypoint script's own directory* to `sys.path`:

    # streamlit/web/bootstrap.py
    def _fix_sys_path(main_script_path): sys.path.insert(0, os.path.dirname(main_script_path))

Nothing else in Streamlit touches `sys.path` -- in particular the working
directory is never added. So pointing the platform at `veridic_eval/cells_app.py`
puts `<repo>/veridic_eval/` on the path and every `from veridic_eval.x import y`
dies with `ModuleNotFoundError: No module named 'veridic_eval'`.

Community Cloud does not `pip install` your repo, it only installs
requirements.txt, so the package is never importable that way either. And
installing it (`pip install .`) would pull `lettucedetect -> torch` plus the
NVIDIA CUDA stack and blow the 1 GB free tier.

Keeping the entrypoint at the root solves it with zero install: the root lands
on `sys.path`, and `veridic_eval` resolves as a plain source package.

This is a launcher only -- it changes no application behaviour.
"""

import os
import sys

# The sidebar renders `settings.postgres_url` verbatim, and Settings is
# constructed at `veridic_eval.config` import time. On a public app the default
# ("postgresql://postgres:admin@localhost:6432/veridic_db") would be shown to
# every visitor, so neutralise it *before* the package is imported. The hosted
# viewer never opens a connection; it reads the committed JSON under out/.
os.environ.setdefault(
    "VERIDIC_EVAL_POSTGRES_URL", "postgresql://offline@snapshot/veridic_db"
)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from veridic_eval.cells_app import main  # noqa: E402

main()
