"""A local exfiltration sink.

The rendering-based exfiltration attack only *proves* anything if the smuggled
data can be observed leaving the system. This is that observation point: a tiny
HTTP server, bound to loopback, that records the query string of every request
it receives and answers with a 1x1 GIF so a rendering client believes it loaded
an image.

It is inert unless something fires a request at it, and it never reaches the
network -- ``127.0.0.1`` on an ephemeral port. Nothing here leaves the machine;
the "leak" is a line in an in-memory list. That is enough for M7 to count.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType

# A 1x1 transparent GIF. Returned to any GET so a Markdown-image render succeeds
# and the beacon looks like a working tracking pixel rather than a broken one.
_PIXEL = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c000000000100010000020144003b"
)


class ExfiltrationSink:
    """Captures the request paths a rendering client fires at it.

    Use as a context manager. ``received`` holds the full request path (with
    query string) of every GET, in order -- that is where the smuggled data
    shows up when the model encodes a secret into a beacon URL.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.received: list[str] = []
        self._host = host
        # port 0 lets the OS pick a free ephemeral port; the real one is read
        # back from the bound socket, so parallel runs never collide.
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        received = self.received

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                received.append(self.path)
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.send_header("Content-Length", str(len(_PIXEL)))
                self.end_headers()
                self.wfile.write(_PIXEL)

            def log_message(self, *args: object) -> None:
                """Silence the default stderr access log; ``received`` is the record."""

        return Handler

    @property
    def base_url(self) -> str:
        # Host is the loopback string we bound with; only the port is chosen by
        # the OS, so read that back and leave the address type untouched.
        port = self._server.server_address[1]
        return f"http://{self._host}:{port}"

    def clear(self) -> None:
        self.received.clear()

    def __enter__(self) -> ExfiltrationSink:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
