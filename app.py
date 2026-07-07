"""BoneSeg Studio — launcher.

Run:
    python app.py                 # opens http://127.0.0.1:7860 in your browser

The app is fully local: the FastAPI server binds to localhost only and no
data leaves the machine.
"""

from __future__ import annotations

import argparse
import socket
import threading
import webbrowser

import uvicorn

from boneseg import APP_NAME, __version__
from boneseg.config import AppConfig, ensure_runtime_dirs
from boneseg.logging_setup import get_logger, setup_logging
from boneseg.webui import create_app

logger = get_logger(__name__)


def find_free_port(host: str, start: int = 7860, tries: int = 50) -> int:
    """First bindable port from ``start`` upward."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}..{start + tries - 1}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} launcher")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=None,
                        help="port to bind. Omit to auto-select the first free "
                             "port from 7860 upward (default behavior).")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not auto-open the browser")
    args = parser.parse_args()

    setup_logging()
    ensure_runtime_dirs()
    port = args.port if args.port is not None else find_free_port(args.host)
    url = f"http://{args.host}:{port}"
    logger.info("Starting %s v%s at %s", APP_NAME, __version__, url)

    app = create_app(AppConfig())

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  {APP_NAME} v{__version__}  ->  {url}   (Ctrl+C to stop)\n")
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
