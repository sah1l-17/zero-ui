"""
Entity Extractor — separates intent detection from entity extraction.

Extracts structured entities (action, product_name, quantity) from user
queries, and provides a keyword-based intent resolver that disambiguates
when the embedding classifier returns low confidence.
"""

import re
from typing import Optional

# ─────────────────────────────────────────────────
# Action-verb buckets — each maps to an intent
# ─────────────────────────────────────────────────

_BROWSE_VERBS = (
    r"\b(show|find|search|browse|explore|look\s+for|display|view|list|"
    r"see|get\s+me|bring\s+up|what\s+do\s+you\s+have|what.?s\s+available|"
    r"what.?s\s+trending|what.?s\s+popular|top\s+rated|top\s+selling|"
    r"best\s+.+\s+available|best\s+.+\s+to\s+buy|anything\s+good)\b"
)

_ADD_CART_VERBS = (
    r"\b(add|put|place|keep|include)\b"
)

_CART_NOUNS = (
    r"\b(cart|basket|bag|shopping\s+(?:cart|bag|basket))\b"
)

_UPDATE_VERBS = (
    r"\b(increase|decrease|reduce|change|set|make|update|modify|adjust|"
    r"bump\s+up|correct|revise)\b"
)

_QUANTITY_WORDS = (
    r"\b(quantity|qty|units?|items?|pieces?|count|number)\b"
)

_REMOVE_VERBS = (
    r"\b(remove|delete|take\s+out|drop|discard|clear|cancel)\b"
)

_BUY_INTENT_VERBS = (
    r"\b(i\s+want\s+to\s+buy|i\s+want\s+to\s+purchase|i\s+want\s+to\s+get|"
    r"i\s+need|i\s+am\s+looking\s+for|i\s+plan\s+to\s+buy|"
    r"help\s+me\s+buy|help\s+me\s+find|i\s+want\s+(?:a|some)?)\b"
)

_ORDER_PLACE_VERBS = (
    r"\b(place|confirm|finalize|complete|proceed|checkout|pay\s+for|order\s+now|buy\s+now)\b"
)

_ORDER_CANCEL_VERBS = (
    r"\b(cancel|abort|stop|undo|revoke)\b"
)

_ORDER_NOUNS = (
    r"\b(order|purchase|checkout|payment)\b"
)

_ORDER_HISTORY_PHRASES = (
    r"\b(order\s+history|past\s+orders?|previous\s+orders?|my\s+orders?|"
    r"purchase\s+history|order\s+list|all\s+orders?)\b"
)

_ORDER_DETAILS_PHRASES = (
    r"\b(order\s+details?|details?\s+of\s+(?:my\s+)?order|"
    r"what\s+(?:was|is|are)\s+(?:in|of)\s+(?:my\s+)?(?:the\s+)?(?:first|second|third|fourth|fifth|last)?\s*order|"
    r"(?:price|cost|total|items?|products?|status|shipping)\s+(?:of|for|in)\s+(?:my\s+)?(?:the\s+)?order|"
    r"how\s+much\s+(?:was|is)\s+(?:my\s+)?order|"
    r"(?:first|second|third|fourth|fifth|last)\s+order\s+(?:price|details?|items?|total|status)|"
    r"ORD-[A-Z0-9]{8}|"
    r"(?:this|that)\s+order)\b"
)


# ─────────────────────────────────────────────────
# Quantity extraction
# ─────────────────────────────────────────────────

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def extract_quantity(text: str) -> Optional[int]:
    """
    Extract a numeric quantity from the text.
    Returns None if no quantity is mentioned (not 1 — the caller decides the default).
    """
    q = text.lower().strip()

    patterns = [
        r"(?:quantity|qty)\s*(?:to|=|:)?\s*(\d+)",
        r"(?:make\s+it|set\s+(?:it\s+)?to|change\s+(?:it\s+)?to)\s+(\d+)",
        r"(?:increase|decrease|reduce|update|bump\s+up)\s+(?:.*?\s+)?(?:to|by)\s+(\d+)",
        r"(\d+)\s*(?:items?|pieces?|units?|nos?\.?)",
        r"\b(?:to|by|=)\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))

    # Word numbers: "increase to three"
    for word, num in _WORD_TO_NUM.items():
        if re.search(rf"\b(?:to|by)\s+{word}\b", q):
            return num
        if re.search(rf"\b{word}\s+(?:items?|pieces?|units?)\b", q):
            return num

    return None


# ─────────────────────────────────────────────────
# Product-name extraction (strips action phrases)
# ─────────────────────────────────────────────────

def extract_product_name(text: str) -> Optional[str]:
    """
    Extract the product-name portion from a user query.
    Strips known action verbs, cart references, and quantity phrases.
    """
    q = text.lower().strip()

    # Remove leading action phrases
    leading = [
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:i\s+(?:want|need|would\s+like|wanna)\s+(?:to\s+)?)?",
        r"(?:add|put|place|keep|include|remove|delete|take\s+out|drop|discard|clear)\s+(?:the\s+)?",
        r"(?:show|find|search|browse|display|view|list|explore|get\s+me|bring\s+up)\s+(?:me\s+)?(?:the\s+)?(?:a\s+)?(?:some\s+)?",
        r"(?:increase|decrease|reduce|change|set|make|update|modify|adjust|bump\s+up|correct)\s+(?:the\s+)?(?:quantity\s+(?:of\s+)?)?",
    ]
    cleaned = q
    for pattern in leading:
        cleaned = re.sub(pattern, "", cleaned, count=1).strip()

    # Remove trailing cart / quantity / preposition phrases
    trailing = [
        r"\s+(?:to|in|into|from|out\s+of)\s+(?:my\s+)?(?:cart|basket|bag|shopping\s+(?:cart|bag|basket)).*$",
        r"\s+(?:from|in|out\s+of)\s+(?:my\s+)?(?:cart|basket|bag).*$",
        r"\s+(?:quantity|qty)\s*.*$",
        r"\s+(?:to|by|=)\s*\d+.*$",
        r"\s+(?:by\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*$",
    ]
    for pattern in trailing:
        cleaned = re.sub(pattern, "", cleaned).strip()

    # Remove stray pronouns / ordinals
    non_products = {
        "it", "this", "that", "these", "those", "one", "ones",
        "first", "second", "third", "fourth", "fifth", "last",
        "first one", "second one", "third one", "last one",
        "the first one", "the second one", "the third one", "the last one",
        "something", "a product", "some stuff", "all",
    }
    if cleaned in non_products:
        return None
    if len(cleaned) >= 2:
        return cleaned
    return None


# ─────────────────────────────────────────────────
# Keyword-based intent resolver
# ─────────────────────────────────────────────────

# Profile-related nouns — when these appear the entity extractor should
# stay out of it and let the profile keyword overrides in the router decide.
_PROFILE_NOUNS = (
    r"\b(profile|account\s+details|account\s+info|personal\s+info|personal\s+details"
    r"|my\s+details|my\s+info|my\s+account)"
    r"\b"
)

# Profile-field nouns that should NOT be confused with product quantity words
_PROFILE_FIELD_NOUNS = (
    r"\b(phone\s*(?:number)?|mobile\s*(?:number)?|contact\s*(?:number)?"
    r"|email\s*(?:address)?|e-?mail|address|delivery\s+address"
    r"|shipping\s+address|location)\b"
)


def resolve_intent_by_entities(query: str) -> Optional[str]:
    """
    Use keyword / regex heuristics to determine intent.
    Returns an intent tag string, or None if no strong signal found.

    Priority order (most specific → least specific):
      0. profile guard — bail out when profile terms are present
      1. update_cart  — quantity-change verbs + quantity words
      2. remove_from_cart — remove verbs + cart reference
      3. add_to_cart  — add verbs + explicit cart/basket/bag reference
      4. browse_product — browse/search/show verbs or buy-intent verbs
    """
    q = query.lower().strip()

    # ── Guard: if the query is about the user's profile, don't resolve here ──
    if re.search(_PROFILE_NOUNS, q):
        return None
    if re.search(_PROFILE_FIELD_NOUNS, q) and re.search(_UPDATE_VERBS, q):
        return None

    has_update_verb = bool(re.search(_UPDATE_VERBS, q))
    has_quantity_word = bool(re.search(_QUANTITY_WORDS, q))
    has_quantity_number = extract_quantity(q) is not None
    has_add_verb = bool(re.search(_ADD_CART_VERBS, q))
    has_cart_noun = bool(re.search(_CART_NOUNS, q))
    has_remove_verb = bool(re.search(_REMOVE_VERBS, q))
    has_browse_verb = bool(re.search(_BROWSE_VERBS, q))
    has_buy_intent = bool(re.search(_BUY_INTENT_VERBS, q))

    # ── update_cart: quantity-change verb + (quantity word OR number) ──
    if has_update_verb and (has_quantity_word or has_quantity_number):
        return "update_cart"

    # "increase/decrease/reduce ... by N" even without the word "quantity"
    if has_update_verb and re.search(r"\b(?:by|to)\s+(?:\d+|one|two|three|four|five)\b", q):
        return "update_cart"

    # ── order intents (checked BEFORE remove_from_cart so "cancel order" wins) ──
    has_order_place = bool(re.search(_ORDER_PLACE_VERBS, q))
    has_order_cancel = bool(re.search(_ORDER_CANCEL_VERBS, q))
    has_order_noun = bool(re.search(_ORDER_NOUNS, q))
    has_order_history = bool(re.search(_ORDER_HISTORY_PHRASES, q))
    has_order_details = bool(re.search(_ORDER_DETAILS_PHRASES, q))

    # order_details must be checked BEFORE cancel_order
    # because queries like "price of my order" should not trigger cancel
    if has_order_details and not has_order_cancel:
        return "order_details"
    # cancel_order must be checked BEFORE order_history
    # because "cancel my order" matches both _ORDER_CANCEL_VERBS and _ORDER_HISTORY_PHRASES
    if has_order_cancel and has_order_noun:
        return "cancel_order"
    if has_order_history:
        return "order_history"
    if has_order_place and has_order_noun:
        return "place_order"
    if has_order_place and not has_cart_noun and not has_browse_verb:
        return "place_order"

    # ── remove_from_cart: remove verb (+ optional cart reference) ──
    if has_remove_verb and has_cart_noun:
        return "remove_from_cart"
    if has_remove_verb and not has_cart_noun:
        return "remove_from_cart"

    # ── add_to_cart: add verb + explicit cart/basket/bag ──
    if has_add_verb and has_cart_noun:
        return "add_to_cart"

    # ── browse_product: search/show verbs OR buy-intent verbs WITHOUT cart ref ──
    if has_browse_verb:
        return "browse_product"
    if has_buy_intent and not has_cart_noun:
        return "browse_product"

    return None


def extract_entities(query: str) -> dict:
    """
    Extract all entities from a query in one call.

    Returns
    -------
    dict with keys:
        intent   : str | None   — keyword-resolved intent (or None)
        product  : str | None   — extracted product name
        quantity : int | None    — extracted quantity (None = not mentioned)
    """
    return {
        "intent": resolve_intent_by_entities(query),
        "product": extract_product_name(query),
        "quantity": extract_quantity(query),
    }
