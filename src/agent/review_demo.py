from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Iterable
from pathlib import Path


DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_HOST = "127.0.0.1"
DEFAULT_FRONTEND_PORT = 3000
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the backend and frontend review UI for manual evaluation."
    )
    parser.add_argument("--backend-host", default=DEFAULT_BACKEND_HOST)
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-host", default=DEFAULT_FRONTEND_HOST)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="Seconds to wait for each server to become reachable.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the frontend URL in a browser automatically.",
    )
    parser.add_argument(
        "--skip-npm-install",
        action="store_true",
        help="Skip automatic frontend dependency installation when node_modules is missing.",
    )
    return parser


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frontend_dir() -> Path:
    return repo_root() / "frontend"


def backend_health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def frontend_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def build_backend_command(host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        "agent.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]


def build_frontend_command(host: str, port: int) -> list[str]:
    return [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(port),
    ]


def build_frontend_env(api_url: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["VITE_API_URL"] = api_url
    return env


def ensure_frontend_dependencies(skip_install: bool) -> None:
    node_modules = frontend_dir() / "node_modules"
    if node_modules.exists() or skip_install:
        return

    print("Installing frontend dependencies...", flush=True)
    subprocess.run(
        ["npm", "install"],
        cwd=frontend_dir(),
        check=True,
    )


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def ensure_port_available(host: str, port: int, label: str) -> None:
    if is_port_in_use(host, port):
        raise RuntimeError(f"{label} port {port} is already in use on {host}.")


def wait_for_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as error:  # pragma: no cover - live startup path
            last_error = error
        time.sleep(0.25)

    if last_error is None:
        raise TimeoutError(f"Timed out waiting for {url}")
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def stream_output(label: str, process: subprocess.Popen[str]) -> threading.Thread:
    def worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - live cleanup path
        process.kill()
        process.wait(timeout=5)


def start_process(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], threading.Thread]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return process, stream_output(label, process)


def monitor_processes(processes: Iterable[tuple[str, subprocess.Popen[str]]]) -> None:
    while True:
        for label, process in processes:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(f"{label} exited unexpectedly with code {return_code}.")
        time.sleep(0.5)


def main() -> int:
    args = build_parser().parse_args()

    backend_url = backend_health_url(args.backend_host, args.backend_port)
    ui_url = frontend_url(args.frontend_host, args.frontend_port)

    ensure_port_available(args.backend_host, args.backend_port, "Backend")
    ensure_port_available(args.frontend_host, args.frontend_port, "Frontend")
    ensure_frontend_dependencies(args.skip_npm_install)

    backend_process = None
    frontend_process = None

    try:
        print("Starting backend...", flush=True)
        backend_process, _ = start_process(
            label="backend",
            command=build_backend_command(args.backend_host, args.backend_port),
            cwd=repo_root(),
            env=os.environ.copy(),
        )
        wait_for_http(backend_url, args.timeout)

        print("Starting frontend...", flush=True)
        frontend_process, _ = start_process(
            label="frontend",
            command=build_frontend_command(args.frontend_host, args.frontend_port),
            cwd=frontend_dir(),
            env=build_frontend_env(f"http://{args.backend_host}:{args.backend_port}"),
        )
        wait_for_http(ui_url, args.timeout)

        print(f"Backend ready: {backend_url}", flush=True)
        print(f"Frontend ready: {ui_url}", flush=True)

        if not args.no_open:
            webbrowser.open(ui_url)

        print("Press Ctrl+C to stop both servers.", flush=True)
        monitor_processes(
            [
                ("backend", backend_process),
                ("frontend", frontend_process),
            ]
        )
    except KeyboardInterrupt:  # pragma: no cover - live path
        print("\nStopping review demo...", flush=True)
        return 0
    finally:
        if frontend_process is not None:
            terminate_process(frontend_process)
        if backend_process is not None:
            terminate_process(backend_process)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
