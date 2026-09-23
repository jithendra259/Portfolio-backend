import os
import sys

# Force UTF-8 on Windows command prompts to prevent UnicodeEncodeError with emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Prevent watchdog thread from triggering false GIL stall alerts on Render shared vCPU
os.environ.setdefault("LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Clean terminal logging: suppress verbose DEBUG logs (hpack, asyncio, httpx, etc.)
os.environ["LIVEKIT_LOG_LEVEL"] = "INFO"

import logging

# Ensure all standard Logger instances support .trace() and .dev() expected by LiveKit
if not hasattr(logging.Logger, "trace"):
    def _logger_trace(self, message, *args, **kwargs):
        if self.isEnabledFor(5):
            self._log(5, message, args, **kwargs)
    logging.Logger.trace = _logger_trace

if not hasattr(logging.Logger, "dev"):
    def _logger_dev(self, message, *args, **kwargs):
        if self.isEnabledFor(23):
            self._log(23, message, args, **kwargs)
    logging.Logger.dev = _logger_dev

from livekit import agents
from server import server

# Silence third-party network & HTTP request spam in the console
for _noisy_lib in ("httpx", "httpcore", "hpack", "urllib3"):
    logging.getLogger(_noisy_lib).setLevel(logging.WARNING)

for _info_lib in ("asyncio", "supabase_logger"):
    logging.getLogger(_info_lib).setLevel(logging.INFO)

if __name__ == "__main__":
    agents.cli.run_app(server)

