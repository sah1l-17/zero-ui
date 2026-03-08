"""
Cart Manager Agent — MongoDB-backed cart operations.
Called by the intent router when intent is one of:
  add_to_cart, remove_from_cart, cart_detail, update_cart
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
carts_collection = _db["carts"]

# Ensure one cart document per user
carts_collection.create_index("username", unique=True)


# ========================================
# 🛒 Cart CRUD Operations
# ========================================

def add_to_cart(username: str, product_row: dict, quantity: int = 1) -> dict:
    """
    Add a product to the user's cart.
    If the product already exists in the cart, increment its quantity.
    product_row must be a complete dict of the dataframe row.
    Uses 'product name' as the product_id key.
    """
    product_id = _product_id_from_row(product_row)

    cart = carts_collection.find_one({"username": username})

    if cart is None:
        # Create a new cart with this item
        carts_collection.insert_one({
            "username": username,
            "items": [
                {
                    "product_id": product_id,
                    "product_data": product_row,
                    "quantity": quantity,
                }
            ],
            "updated_at": datetime.utcnow(),
        })
        return {
            "status": "added",
            "product_id": product_id,
            "quantity": quantity,
            "message": f"Added '{product_id}' (qty {quantity}) to your cart.",
        }

    # Cart exists — check if product already present
    existing_item = None
    for item in cart.get("items", []):
        if item["product_id"] == product_id:
            existing_item = item
            break

    if existing_item:
        new_qty = existing_item["quantity"] + quantity
        carts_collection.update_one(
            {"username": username, "items.product_id": product_id},
            {
                "$set": {
                    "items.$.quantity": new_qty,
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        return {
            "status": "incremented",
            "product_id": product_id,
            "quantity": new_qty,
            "message": f"'{product_id}' already in cart — quantity updated to {new_qty}.",
        }
    else:
        carts_collection.update_one(
            {"username": username},
            {
                "$push": {
                    "items": {
                        "product_id": product_id,
                        "product_data": product_row,
                        "quantity": quantity,
                    }
                },
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
        return {
            "status": "added",
            "product_id": product_id,
            "quantity": quantity,
            "message": f"Added '{product_id}' (qty {quantity}) to your cart.",
        }


def remove_from_cart(username: str, product_id: str) -> dict:
    """Remove a product from the user's cart by product_id."""
    result = carts_collection.update_one(
        {"username": username},
        {
            "$pull": {"items": {"product_id": product_id}},
            "$set": {"updated_at": datetime.utcnow()},
        },
    )
    if result.modified_count:
        return {
            "status": "removed",
            "product_id": product_id,
            "message": f"Removed '{product_id}' from your cart.",
        }
    return {
        "status": "not_found",
        "product_id": product_id,
        "message": f"'{product_id}' was not found in your cart.",
    }


def _extract_item_price(product_data: dict) -> float:
    """Pull a numeric price from product_data, checking common key names."""
    import re as _re
    price_keys = (
        "price", "actual_price", "selling_price", "discounted_price",
        "cost", "amount", "mrp", "sale_price",
    )
    for key in price_keys:
        for k, v in product_data.items():
            if k.lower().replace(" ", "_") == key and v:
                cleaned = _re.sub(r"[^\d.]", "", str(v))
                try:
                    return float(cleaned)
                except (ValueError, TypeError):
                    pass
    return 0.0


def get_cart(username: str) -> dict:
    """Return all items in the user's cart with computed totals and ₹ prices."""
    cart = carts_collection.find_one({"username": username})
    if cart is None or len(cart.get("items", [])) == 0:
        return {
            "status": "empty",
            "items": [],
            "cart_total": "₹0",
            "message": "Your cart is empty.",
        }
    # Build a clean, LLM-friendly items list
    items = []
    cart_total = 0.0
    for item in cart["items"]:
        unit_price = _extract_item_price(item.get("product_data", {}))
        quantity = item.get("quantity", 1)
        subtotal = round(unit_price * quantity, 2)
        cart_total += subtotal
        items.append({
            "product_id": item["product_id"],
            "product_data": item["product_data"],
            "quantity": quantity,
            "unit_price": f"₹{unit_price:,.2f}",
            "subtotal": f"₹{subtotal:,.2f}",
        })
    return {
        "status": "ok",
        "items": items,
        "cart_total": f"₹{cart_total:,.2f}",
        "message": f"You have {len(items)} item(s) in your cart. Cart total: ₹{cart_total:,.2f}",
    }


def update_cart(username: str, product_id: str, quantity: int) -> dict:
    """Set the quantity of an existing cart item. If quantity <= 0, remove it."""
    if quantity <= 0:
        return remove_from_cart(username, product_id)

    result = carts_collection.update_one(
        {"username": username, "items.product_id": product_id},
        {
            "$set": {
                "items.$.quantity": quantity,
                "updated_at": datetime.utcnow(),
            }
        },
    )
    if result.modified_count:
        return {
            "status": "updated",
            "product_id": product_id,
            "quantity": quantity,
            "message": f"Quantity of '{product_id}' updated to {quantity}.",
        }
    return {
        "status": "not_found",
        "product_id": product_id,
        "message": f"'{product_id}' was not found in your cart.",
    }


# ========================================
# 🔍 Helpers
# ========================================

def _product_id_from_row(product_row: dict) -> str:
    """
    Derive a stable product_id from a product row dict.
    Prefers 'product name'; falls back to first non-empty value.
    """
    for key in ("product name", "name", "title", "product_name"):
        if key in product_row and product_row[key]:
            return str(product_row[key]).strip()
    # Fallback: first non-empty value
    for v in product_row.values():
        if v:
            return str(v).strip()
    return "unknown_product"


_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def extract_quantity(text: str) -> int:
    """
    Extract a numeric quantity from natural language.
    Examples:
      "make it 3"                  → 3
      "set quantity to 2"          → 2
      "increase to 5"              → 5
      "add 4 of these"             → 4
      "increase by one"            → 1
      "change quantity to three"   → 3
    Falls back to 1 if nothing found.
    """
    q = text.lower()

    # Digit-based patterns
    patterns = [
        r"(?:quantity|qty)\s*(?:to|=|:)?\s*(\d+)",
        r"(?:make\s+it|set\s+(?:it\s+)?to|change\s+(?:it\s+)?to)\s+(\d+)",
        r"(?:increase|decrease|reduce|update)\s+(?:.*?\s+)?(?:to|by)\s+(\d+)",
        r"(\d+)\s*(?:items?|pieces?|units?|nos?\.?)",
        r"\b(?:to|by|=)\s*(\d+)\b",
        r"\badd\s+(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))

    # Word-number patterns: "by one", "to three", "three items"
    for word, num in _WORD_TO_NUM.items():
        if re.search(rf"\b(?:to|by)\s+{word}\b", q):
            return num
        if re.search(rf"\b{word}\s+(?:items?|pieces?|units?)\b", q):
            return num
        # "quantity ... <word>" at the end
        if re.search(rf"(?:quantity|qty)\s+.*?\b{word}\b", q):
            return num

    return 1


def is_relative_change(text: str) -> bool:
    """
    Determine if the user wants a *relative* quantity change (increase/decrease BY N)
    vs an *absolute* change (set/change TO N).

    Examples:
      "increase quantity by one"      → True   (add 1 to current)
      "decrease quantity by 2"         → True   (subtract 2 from current)
      "set quantity to 3"             → False  (set to exactly 3)
      "change quantity to 5"          → False  (set to exactly 5)
      "make it 2"                     → False  (set to exactly 2)
      "increase quantity to 3"        → False  (set to exactly 3)
    """
    q = text.lower().strip()

    # Explicit "by N" → relative
    if re.search(r"\b(?:increase|decrease|reduce|bump\s+up|add|subtract)\b.*\bby\b", q):
        return True

    # "increase/decrease" without "to" → relative (e.g. "increase quantity")
    if re.search(r"\b(?:increase|decrease|reduce|bump\s+up)\b", q) and not re.search(r"\bto\b", q):
        return True

    # Everything else (set to, change to, make it, increase to) → absolute
    return False


def extract_product_name_from_query(query: str) -> str | None:
    """
    Try to pull a product name from an add/remove/update query.
    Strips common action phrases and returns the remainder.
    """
    q = query.lower().strip()
    # Strip leading action phrases
    action_phrases = [
        r"^(?:please\s+)?(?:i\s+(?:want|would like|wanna)\s+to\s+)?(?:add|put|place|include|keep)\s+(?:the\s+)?",
        r"^(?:please\s+)?(?:remove|delete|take\s+out|drop|discard|clear)\s+",
        r"^(?:please\s+)?(?:update|change|modify|edit|set|increase|decrease|reduce|adjust)\s+(?:the\s+)?(?:quantity\s+(?:of\s+)?)?",
    ]
    cleaned = q
    for pattern in action_phrases:
        cleaned = re.sub(pattern, "", cleaned, count=1)

    # Strip trailing phrases
    trailing = [
        r"\s+(?:to|in|into)\s+(?:my\s+)?(?:cart|basket|bag|shopping\s+(?:cart|bag|basket)).*$",
        r"\s+(?:from|in|out\s+of)\s+(?:my\s+)?(?:cart|basket|bag).*$",
        r"\s+(?:to|=)\s+\d+.*$",
        r"\s+(?:quantity|qty).*$",
    ]
    for pattern in trailing:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = cleaned.strip()
    # Filter out pure pronouns / ordinals (not product names)
    non_product_words = {
        "it", "this", "that", "these", "those", "one", "ones",
        "first", "second", "third", "fourth", "fifth", "last",
        "1st", "2nd", "3rd", "4th", "5th",
        "first one", "second one", "third one", "fourth one", "fifth one", "last one",
        "the first one", "the second one", "the third one", "the last one",
    }
    if cleaned in non_product_words:
        return None
    if len(cleaned) >= 2:
        return cleaned
    return None


def find_cart_item_by_query(username: str, query: str) -> str | None:
    """
    Given a free-text query, find the best matching product_id in the user's cart.
    Uses simple substring matching against product_id.
    """
    cart = carts_collection.find_one({"username": username})
    if not cart or not cart.get("items"):
        return None

    product_name = extract_product_name_from_query(query)
    search_term = (product_name or query).lower().strip()

    # Try exact match first
    for item in cart["items"]:
        if item["product_id"].lower() == search_term:
            return item["product_id"]

    # Substring match
    for item in cart["items"]:
        if search_term in item["product_id"].lower() or item["product_id"].lower() in search_term:
            return item["product_id"]

    # Word overlap match
    search_words = set(search_term.split())
    best_match = None
    best_overlap = 0
    for item in cart["items"]:
        item_words = set(item["product_id"].lower().split())
        overlap = len(search_words & item_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = item["product_id"]

    return best_match


# ========================================
# 🔗 Pronoun / Ordinal Reference Resolution
# ========================================

def resolve_product_reference(query: str, search_results: list,
                              session: dict | None = None) -> dict | None:
    """
    Resolve pronouns ('it', 'this') and ordinals ('first one', '2nd')
    to a concrete product from the most recent search results.
    Uses session['last_discussed_product_index'] for pronoun resolution
    when the user was previously discussing a specific product.
    Returns the matched product dict, or None.
    """
    if not search_results:
        return None

    q = query.lower().strip()

    # Ordinal mapping
    ordinal_map = {
        "first": 0, "1st": 0, "#1": 0, "number 1": 0, "number one": 0,
        "second": 0 + 1, "2nd": 1, "#2": 1, "number 2": 1, "number two": 1,
        "third": 2, "3rd": 2, "#3": 2, "number 3": 2, "number three": 2,
        "fourth": 3, "4th": 3, "#4": 3, "number 4": 3,
        "fifth": 4, "5th": 4, "#5": 4, "number 5": 4,
        "last": len(search_results) - 1,
    }

    for keyword, idx in ordinal_map.items():
        if keyword in q and idx < len(search_results):
            return search_results[idx]

    # Pronoun references → use last discussed product if tracked, else first result
    if re.search(r"\b(it|this|that)\b", q):
        discussed_idx = (session or {}).get("last_discussed_product_index")
        if discussed_idx is not None and 0 <= discussed_idx < len(search_results):
            return search_results[discussed_idx]
        return search_results[0]

    return None


# ========================================
# 🧩 Cart Manager Agent (entry point)
# ========================================

def cart_manager_agent(intent: str, query: str, username: str,
                       search_fn=None, session: dict | None = None) -> dict:
    """
    Main entry point called by the intent router.

    Parameters
    ----------
    intent : str
        One of add_to_cart, remove_from_cart, cart_detail, update_cart.
    query : str
        The user's raw message.
    username : str
        Authenticated username (from session).
    search_fn : callable, optional
        The search_products function from main.py (injected to avoid circular imports).
    session : dict, optional
        Session dict — used to check last_search_results for add_to_cart context.

    Returns
    -------
    dict with keys: action, result, and optionally product/cart data.
    """

    # ── add_to_cart ─────────────────────────────────────────
    if intent == "add_to_cart":
        quantity = extract_quantity(query)
        search_results = session.get("last_search_results", []) if session else []

        # 1) Resolve pronoun / ordinal references ("add it", "add the first one")
        resolved = resolve_product_reference(query, search_results, session=session)
        if resolved:
            product_row = {k: v for k, v in resolved.items() if k != "relevance_score"}
            result = add_to_cart(username, product_row, quantity)
            return {"action": "add_to_cart", "result": result}

        # 2) Extract a real product name from the query
        product_name_from_query = extract_product_name_from_query(query)

        # 3) Try matching the product name against recent search results
        if product_name_from_query and search_results:
            matched_product = None
            search_lower = product_name_from_query.lower()

            # Substring match
            for product in search_results:
                p_name = product.get("product name", product.get("name", "")).lower()
                if search_lower in p_name or p_name in search_lower:
                    matched_product = product
                    break

            # Word overlap fallback
            if not matched_product:
                search_words = set(search_lower.split())
                best_overlap = 0
                for product in search_results:
                    p_name = product.get("product name", product.get("name", "")).lower()
                    p_words = set(p_name.split())
                    overlap = len(search_words & p_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        matched_product = product
                if best_overlap == 0:
                    matched_product = None

            if matched_product:
                product_row = {k: v for k, v in matched_product.items() if k != "relevance_score"}
                result = add_to_cart(username, product_row, quantity)
                return {"action": "add_to_cart", "result": result}

        # 4) Run a product search as last resort
        if search_fn is not None and product_name_from_query:
            import pandas as pd
            results = search_fn(query)
            if isinstance(results, pd.DataFrame) and not results.empty:
                row = results.iloc[0]
                product_row = {
                    col: str(row[col])
                    for col in results.columns
                    if col != "similarity_score"
                }
                result = add_to_cart(username, product_row, quantity)
                return {"action": "add_to_cart", "result": result}

        return {
            "action": "add_to_cart",
            "result": {
                "status": "no_product",
                "message": "I couldn't find a matching product to add. Try searching for a product first, or mention the product name you'd like to add.",
            },
        }

    # ── remove_from_cart ────────────────────────────────────
    if intent == "remove_from_cart":
        product_id = find_cart_item_by_query(username, query)
        if product_id:
            result = remove_from_cart(username, product_id)
        else:
            result = {
                "status": "not_found",
                "message": "I couldn't identify which product to remove. Please mention the product name.",
            }
        return {"action": "remove_from_cart", "result": result}

    # ── cart_detail ─────────────────────────────────────────
    if intent == "cart_detail":
        result = get_cart(username)
        return {"action": "cart_detail", "result": result}

    # ── update_cart ─────────────────────────────────────────
    if intent == "update_cart":
        raw_quantity = extract_quantity(query)
        relative = is_relative_change(query)
        product_id = find_cart_item_by_query(username, query)

        # Resolve product_id — fallback to only cart item
        if not product_id:
            cart_data = get_cart(username)
            if cart_data["status"] == "ok" and len(cart_data["items"]) == 1:
                product_id = cart_data["items"][0]["product_id"]

        if product_id:
            if relative:
                # Read current quantity and compute new value
                cart_data = get_cart(username)
                current_qty = 0
                for item in cart_data.get("items", []):
                    if item["product_id"] == product_id:
                        current_qty = item["quantity"]
                        break
                # Detect increase vs decrease
                if re.search(r"\b(?:decrease|reduce|subtract|lower|less)\b", query.lower()):
                    new_qty = max(0, current_qty - raw_quantity)
                else:
                    new_qty = current_qty + raw_quantity
                result = update_cart(username, product_id, new_qty)
            else:
                # Absolute: set to the exact quantity
                result = update_cart(username, product_id, raw_quantity)
        else:
            result = {
                "status": "not_found",
                "message": "I couldn't identify which product to update. Please mention the product name.",
            }
        return {"action": "update_cart", "result": result}

    # ── unknown ─────────────────────────────────────────────
    return {
        "action": intent,
        "result": {"status": "error", "message": f"Unknown cart intent: {intent}"},
    }
