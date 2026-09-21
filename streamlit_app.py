"""Deploy entrypoint. Must stay in the repository root: `streamlit run` adds only
the entrypoint's own directory to sys.path, so an entrypoint inside veridic_eval/
cannot import veridic_eval.*
"""

import os
import sys

# The sidebar renders settings.postgres_url, which is built at config import time.
os.environ.setdefault(
    "VERIDIC_EVAL_POSTGRES_URL", "postgresql://offline@snapshot/veridic_db"
)

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from veridic_eval.cells_app import main  # noqa: E402

main()
