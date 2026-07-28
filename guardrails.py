"""
Guardrails.

Policy is declared in ./guardrails_config (NeMo Guardrails Colang rails),
which is the source of truth for *what* is blocked. The functions below
are the enforcement layer actually called by rag.py: they mirror the
Colang flows one-to-one but run as plain Python for near-zero latency
(no extra LLM call needed to police every single message).

Covers:
    1. Domain restriction     -> enforced in rag.py via the Qdrant
                                 similarity threshold (see rag.py).
    2. Off-topic questions    -> same as above.
    3. Prompt injection       -> detect_prompt_injection()
    4. Jailbreak attempts     -> detect_jailbreak()
    5. Toxic language         -> detect_toxic_language()
    6. Hallucination prevention -> enforced in rag.py (never answer from
                                 outside retrieved context).
"""

import re

from logging_config import get_logger

logger = get_logger()

ADVERTISEMENT_MESSAGE = (
    "I'm here to help with our salon services only.\n\n"
    "✂ Visit our Men's Salon!\n"
    "• Professional Haircuts\n"
    "• Beard Styling\n"
    "• Hair Treatments\n"
    "• Premium Grooming\n"
    "• Friendly Staff\n\n"
    "We would love to see you!"
)

NOT_FOUND_MESSAGE = (
    "I couldn't find information about that in our salon knowledge base.\n"
    "Visit our salon for more information!"
)

INJECTION_BLOCKED_MESSAGE = (
    "I can't do that, but I'm happy to help with anything about our salon "
    "services, pricing, hours, or booking!"
)

TOXIC_LANGUAGE_MESSAGE = (
    "Let's keep things friendly! I'm happy to help with any questions about "
    "our salon services whenever you're ready."
)

# --- Prompt injection / jailbreak patterns -----------------------------------
_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)\s*instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above)\s*instructions",
    r"reveal (your|the)\s*(system|hidden)\s*prompt",
    r"show (me\s)?(your|the)\s*(system|hidden)\s*prompt",
    r"what (is|are) your (system|hidden) prompt",
    r"act as (chatgpt|a developer|the developer|dan)",
    r"you are now (dan|in developer mode|jailbroken)",
    r"developer mode",
    r"\bdan\b.*(mode|prompt)",
    r"pretend (you are|to be) (an? )?(unfiltered|unrestricted|uncensored)",
    r"bypass (your|the)\s*(guardrails|restrictions|rules|filters)",
    r"forget (all|your)\s*(previous\s*)?instructions",
    r"repeat (your|the)\s*(system\s*)?prompt",
    r"print (your|the)\s*(system\s*)?prompt",
]
_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# --- Toxic / abusive language (lightweight keyword screen) ------------------
_TOXIC_WORDS = {
    "idiot", "stupid", "dumb", "shut up", "fuck", "fucking", "shit",
    "bastard", "asshole", "bitch", "moron", "retard", "damn you",
}


def detect_prompt_injection(text: str) -> bool:
    """Detect prompt-injection / jailbreak attempts (covers requirements 3 & 4)."""
    return bool(_INJECTION_REGEX.search(text))


def detect_jailbreak(text: str) -> bool:
    """Alias kept for clarity; same detector handles both injection and jailbreaks."""
    return detect_prompt_injection(text)


def detect_toxic_language(text: str) -> bool:
    """Very lightweight toxicity screen for abusive language (requirement 5)."""
    lowered = text.lower()
    return any(word in lowered for word in _TOXIC_WORDS)


def apply_input_guardrails(message: str) -> str | None:
    """
    Run all input-side guardrail checks.

    Returns a canned safe response string if the message should be blocked,
    or None if the message is safe to continue to retrieval/LLM.
    """
    if detect_prompt_injection(message):
        logger.warning("Guardrail triggered: prompt injection / jailbreak attempt")
        return INJECTION_BLOCKED_MESSAGE

    if detect_toxic_language(message):
        logger.warning("Guardrail triggered: toxic language")
        return TOXIC_LANGUAGE_MESSAGE

    return None
