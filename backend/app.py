import os

# Prevent watchdog thread from triggering false GIL stall alerts on Render shared vCPU
os.environ.setdefault("LIVEKIT_AGENTS_LOOP_BLOCK_WARN_MS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from livekit import agents
from server import server

if __name__ == "__main__":
    agents.cli.run_app(server)

