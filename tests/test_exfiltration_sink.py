"""The exfiltration sink, and the reason its port is fixed.

An exfiltration payload carries the sink's base URL in its own text. Chunk ids
are content-addressed, so an OS-assigned port gives the poison a different id,
a different embedding and a different rank on every run -- and exf-002, which
needs both the poison and the restricted note in the top-k, then lands or fails
depending on which port the OS handed out. Measured directly: the same
configuration on qwen2.5:1.5b scored 2/7 then 3/7 on consecutive runs, and 3/7
three times running once the port was pinned.
"""

from __future__ import annotations

import httpx
import pytest

from threatrag.security.attacks.sink import DEFAULT_SINK_PORT, ExfiltrationSink


def test_the_default_port_is_fixed_not_ephemeral() -> None:
    """The whole point. A zero here would silently restore run-to-run variance
    in every exfiltration measurement."""
    assert DEFAULT_SINK_PORT != 0


def test_the_same_port_yields_the_same_base_url() -> None:
    """Which is what keeps a poison document byte-identical across runs."""
    with ExfiltrationSink(port=0) as probe:
        free = int(probe.base_url.rsplit(":", 1)[1])

    with ExfiltrationSink(port=free) as first:
        first_url = first.base_url
    with ExfiltrationSink(port=free) as second:
        assert second.base_url == first_url


def test_an_unbindable_address_fails_loudly_rather_than_falling_back() -> None:
    """A silent fallback to an ephemeral port would reintroduce exactly the
    variance the fixed default removes, and the numbers would look ordinary --
    so the error names the flag that opts into it instead of guessing.

    Not tested with a *busy* port: ``HTTPServer`` sets SO_REUSEADDR, and on
    Windows that permits rebinding a port another socket already holds. Keeping
    the reuse is deliberate, because the sweep opens and closes this sink once
    per cell on the same fixed port and must not trip over TIME_WAIT between
    them.
    """
    with pytest.raises(OSError, match="--sink-port 0"):
        # TEST-NET-1, which this host does not own.
        ExfiltrationSink(host="192.0.2.1", port=24601)


def test_it_records_the_query_string_of_a_fired_beacon() -> None:
    with ExfiltrationSink(port=0) as sink:
        httpx.get(f"{sink.base_url}/verify?finding=41%20repositories", timeout=5)

        assert sink.received == ["/verify?finding=41%20repositories"]


def test_clear_resets_between_attacks() -> None:
    """The runner clears per attack, so one attack's beacon cannot be counted
    as the next attack's leak."""
    with ExfiltrationSink(port=0) as sink:
        httpx.get(f"{sink.base_url}/a?d=1", timeout=5)
        sink.clear()

        assert sink.received == []
