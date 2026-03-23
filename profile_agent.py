"""
Profile Manager Agent — MongoDB-backed profile operations.
Called by the intent router when intent is one of:
  view_profile, update_profile
"""

import re
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

_mongo_client = MongoClient(MONGO_URI)
_db = _mongo_client["final_year"]
users_collection = _db["users"]

# Ensure one user document per username
users_collection.create_index("username", unique=True)


# ========================================
# Validation Helpers
# ========================================

def validate_email(email: str) -> bool:
    """Validate email format using a standard regex."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def validate_phone(phone: str) -> bool:
    """
    Validate phone number — must be 7-15 digits,
    optionally prefixed with + and country code.
    """
    cleaned = re.sub(r"[\s\-().]", "", phone.strip())
    return bool(re.match(r"^\+?\d{7,15}$", cleaned))


# ========================================
# Query Parsing
# ========================================

# Field aliases that map natural language tokens to canonical field names
_FIELD_ALIASES = {
    "phone": "phone",
    "phone number": "phone",
    "mobile": "phone",
    "mobile number": "phone",
    "contact": "phone",
    "contact number": "phone",
    "number": "phone",
    "cell": "phone",
    "email": "email",
    "email address": "email",
    "mail": "email",
    "e-mail": "email",
    "address": "address",
    "delivery address": "address",
    "shipping address": "address",
    "location": "address",
}

# Ordered by longest first so multi-word aliases match before single-word ones
_SORTED_ALIASES = sorted(_FIELD_ALIASES.keys(), key=len, reverse=True)


def extract_profile_updates(query: str) -> dict:
    """
    Parse natural language and extract profile fields to update.

    Handles patterns like:
      - "update my address to Delhi"
      - "change my phone number to 9876543210"
      - "set my email to abc@gmail.com"
      - "update my address to Pune and phone to 9999999999"
      - "change email to abc@gmail.com and address to Bangalore"

    Returns
    -------
    dict mapping canonical field names to their new values.
    Example: {"phone": "9876543210", "address": "Delhi"}
    """
    q = query.strip()
    q_lower = q.lower()
    updates: dict[str, str] = {}

    # Split on "and" to handle multi-field updates
    # e.g. "update address to Pune and phone to 9999999999"
    segments = re.split(r"\band\b", q, flags=re.IGNORECASE)

    for segment in segments:
        seg = segment.strip()
        seg_lower = seg.lower()

        matched_field = None
        value = None

        # Try each alias (longest first)
        for alias in _SORTED_ALIASES:
            pattern = (
                r"(?:update|change|set|modify|edit|correct|revise|make)?"
                r"\s*(?:my\s+)?"
                + re.escape(alias)
                + r"\s+(?:to|as|=|:)\s+(.+)"
            )
            match = re.search(pattern, seg, re.IGNORECASE)
            if match:
                matched_field = _FIELD_ALIASES[alias]
                value = match.group(1).strip()
                # Clean trailing punctuation
                value = re.sub(r"[.!?]+$", "", value).strip()
                break

        # Fallback: "<alias> <value>" without explicit "to"
        if not matched_field:
            for alias in _SORTED_ALIASES:
                pattern = (
                    r"(?:update|change|set|modify|edit|correct|revise|make)"
                    r"\s+(?:my\s+)?"
                    + re.escape(alias)
                    + r"\s+(.+)"
                )
                match = re.search(pattern, seg, re.IGNORECASE)
                if match:
                    matched_field = _FIELD_ALIASES[alias]
                    value = match.group(1).strip()
                    value = re.sub(r"[.!?]+$", "", value).strip()
                    break

        if matched_field and value:
            updates[matched_field] = value

    return updates


# ========================================
# Profile CRUD Operations
# ========================================

def view_profile(username: str) -> dict:
    """
    Fetch the user's profile from the users collection.
    Returns username, email, phone, and address.
    """
    user = users_collection.find_one({"username": username})

    if user is None:
        return {
            "status": "not_found",
            "message": "Profile not found. Please sign up or log in first.",
        }

    profile = {
        "username": user.get("username", ""),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone", user.get("contact", "")),
        "address": user.get("address", ""),
    }

    return {
        "status": "ok",
        "profile": profile,
    }


def update_profile(username: str, updates: dict) -> dict:
    """
    Update one or more profile fields for a user.

    Parameters
    ----------
    username : str
        The authenticated user's username.
    updates : dict
        Mapping of field names to new values.
        Allowed fields: phone, email, address.

    Returns
    -------
    dict with status, updated fields, and any validation errors.
    """
    ALLOWED_FIELDS = {"phone", "email", "address"}

    if not updates:
        return {
            "status": "no_fields",
            "message": "No valid fields detected to update. You can update: phone, email, or address.",
        }

    # Validate and filter
    valid_updates = {}
    errors = []

    for field, value in updates.items():
        if field not in ALLOWED_FIELDS:
            errors.append(f"'{field}' is not an updatable field.")
            continue

        if field == "email" and not validate_email(value):
            errors.append(f"Invalid email format: '{value}'.")
            continue

        if field == "phone" and not validate_phone(value):
            errors.append(f"Invalid phone number: '{value}'. Must be 7-15 digits.")
            continue

        if not value.strip():
            errors.append(f"Empty value for '{field}'.")
            continue

        valid_updates[field] = value.strip()

    if not valid_updates:
        return {
            "status": "validation_error",
            "errors": errors,
            "message": "No fields could be updated. " + " ".join(errors),
        }

    # Perform MongoDB update
    update_doc = {**valid_updates, "updated_at": datetime.utcnow()}
    result = users_collection.update_one(
        {"username": username},
        {"$set": update_doc},
    )

    if result.matched_count == 0:
        return {
            "status": "not_found",
            "message": "Profile not found. Please sign up or log in first.",
        }

    # Build response
    updated_fields = list(valid_updates.keys())
    field_summary = ", ".join(
        f"{field} → {valid_updates[field]}" for field in updated_fields
    )

    response = {
        "status": "updated",
        "updated_fields": valid_updates,
        "message": f"Profile updated successfully: {field_summary}.",
    }

    if errors:
        response["warnings"] = errors

    return response


# ========================================
# Profile Manager Agent (entry point)
# ========================================

def profile_manager_agent(intent: str, query: str, username: str) -> dict:
    """
    Main entry point called by the intent router.

    Parameters
    ----------
    intent : str
        One of view_profile, update_profile.
    query : str
        The user's raw message.
    username : str
        Authenticated username (from session).

    Returns
    -------
    dict with keys: action, result.
    """

    # ── view_profile ────────────────────────────────────────
    if intent == "view_profile":
        result = view_profile(username)
        return {"action": "view_profile", "result": result}

    # ── update_profile ──────────────────────────────────────
    if intent == "update_profile":
        updates = extract_profile_updates(query)
        result = update_profile(username, updates)
        return {"action": "update_profile", "result": result}

    # ── unknown intent fallback ─────────────────────────────
    return {
        "action": intent,
        "result": {
            "status": "unknown_intent",
            "message": f"Profile agent does not handle intent '{intent}'.",
        },
    }
