"""D3 -- remove the carrier an answer would use to phone home.

M4's exf-002 is the attack this answers. Run under IR-lead clearance, the model
stated a TLP:RED insider-risk finding, URL-encoded it, and emitted it as a
Markdown link; M2's own renderer issued the GET and the finding arrived in the
sink's log as a query parameter. Nothing about the answer text looked alarming.

The mechanism is the rendered URL, so that is what this removes: Markdown links
and images are reduced to their visible text, and bare URLs are dropped, unless
the host is allowlisted. What the answer *says* is untouched.

**This stops egress, not disclosure.** The analyst who ran exf-002 was cleared
to read that finding -- the breach was it leaving the system to a third party.
Strip the link and the secret still sits in the answer text, correctly, for a
reader entitled to it. A defence that also refused to state it would be a
different mitigation with a different cost, and conflating the two would hide
which one paid for the result.

The allowlist is where the trade-off actually lives. Empty, this severs every
outbound reference, including the attack.mitre.org and nvd.nist.gov links that
make a citation checkable by a human. That is a real utility loss for analysts,
and it is deliberately the default: a permissive default would make the defence
look free while quietly leaving open hosts an attacker can reach.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urlsplit

from threatrag.config import Config
from threatrag.domain.models import Answer
from threatrag.security.defenses.base import BaseDefense

# A Markdown image or link with an http(s) target -- the same shape the render
# simulation fires, because that is the shape a browser follows.
_MARKDOWN = re.compile(r"(!?)\[([^\]]*)\]\((https?://[^)\s]+)\)")
# A bare URL not already inside Markdown parentheses. Autolinking clients follow
# these too, so removing only the Markdown form would leave the carrier intact.
_BARE = re.compile(r"(?<![(\]])\bhttps?://[^\s<>)\]]+")


class EgressFilter(BaseDefense):
    """Strip outbound URLs from an answer unless their host is allowlisted."""

    name = "egress_filter"

    def __init__(self, allowed_hosts: Sequence[str] = ()) -> None:
        self._allowed = {host.lower().lstrip(".") for host in allowed_hosts}

    def _permitted(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower()
        # Subdomains of an allowlisted host are permitted; a suffix match alone
        # would let "evil-attack.mitre.org.example.com" through.
        return any(host == allowed or host.endswith(f".{allowed}") for allowed in self._allowed)

    def on_answer(self, answer: Answer) -> Answer:
        stripped: list[str] = []

        def _markdown(match: re.Match[str]) -> str:
            bang, label, url = match.groups()
            if self._permitted(url):
                return match.group(0)
            stripped.append(url)
            # The visible text survives, so the answer still reads as prose and
            # an image's alt text is not silently lost.
            return f"[image: {label}]" if bang else label

        def _bare(match: re.Match[str]) -> str:
            url = match.group(0)
            if self._permitted(url):
                return url
            stripped.append(url)
            return "[link removed]"

        text = _BARE.sub(_bare, _MARKDOWN.sub(_markdown, answer.text))
        if not stripped:
            return answer
        return answer.model_copy(
            update={"text": text, "stripped_urls": [*answer.stripped_urls, *stripped]}
        )


def build(config: Config) -> EgressFilter:
    return EgressFilter(config.defense_settings.egress_filter.allowed_hosts)
