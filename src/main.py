import os

# torch and CTranslate2 (faster-whisper's backend) each bundle their own
# libiomp5.dylib on macOS; loading both aborts the process with
# "OMP: Error #15" unless duplicate OpenMP runtimes are explicitly allowed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Beyond the duplicate-runtime abort, having torch and CTranslate2 both spin up
# multi-threaded OpenMP pools in one process segfaults (or deadlocks, depending
# on load order) on macOS. Pinning the OpenMP pool to a single thread makes the
# two coexist reliably. Override by exporting OMP_NUM_THREADS before launch.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.agent.agent import Agent


def main():
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()
