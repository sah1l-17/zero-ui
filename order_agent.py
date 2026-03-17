"""
Order Manager Agent — MongoDB-backed order operations.
Called by the intent router when intent is one of:
  place_order, cancel_order, order_history
"""

import re
import uuid
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

_mongo_client = MongoClient(MONGO_URI)
_db = _mongo_client["final_year"]
orders_collection = _db["orders"]
carts_collection = _db["carts"]
users_collection = _db["users"]

# Indexes
orders_collection.create_index("username")
orders_collection.create_index("created_at")
orders_collection.create_index("order_id", unique=True)


# ========================================
# 🔧 Helpers
# ========================================

def generate_order_id() -> str:
    """Generate a unique, human-readable order ID (e.g. ORD-A3F8B1)."""
    short_uuid = uuid.uuid4().hex[:8].upper()
    return f"ORD-{short_uuid}"


def calculate_total_amount(items: list) -> float:
    """
    Calculate total amount from a list of cart items.
    Each item: { "product_data": {...}, "quantity": int }
    Tries common price keys and strips currency symbols.
    """
    total = 0.0
    for item in items:
        price = _extract_price(item.get("product_data", {}))
        quantity = item.get("quantity", 1)
        total += price * quantity
    return round(total, 2)


def _extract_price(product_data: dict) -> float:
    """
    Pull a numeric price from the product_data dict.
    Checks common key names and strips currency symbols / commas.
    """
    price_keys = (
        "price", "actual_price", "selling_price", "discounted_price",
        "cost", "amount", "mrp", "sale_price",
    )
    for key in price_keys:
        for k, v in product_data.items():
            if k.lower().replace(" ", "_") == key and v:
                return _parse_number(str(v))
    # Fallback: first value that looks like a price
    for v in product_data.values():
        parsed = _parse_number(str(v))
        if parsed > 0:
            return parsed
    return 0.0


def _parse_number(text: str) -> float:
    """Parse a numeric string, stripping currency symbols and commas."""
    cleaned = re.sub(r"[^\d.]", "", text)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def extract_order_id_from_query(query: str) -> str | None:
    """
    Extract an order ID (e.g. ORD-A3F8B1) from the user's message.
    Falls back to any ORD-prefixed token, then any standalone hex-like token.
    """
    q = query.strip()

    # Pattern: ORD-XXXXXXXX (case-insensitive)
    match = re.search(r"\b(ORD-[A-Za-z0-9]{6,10})\b", q, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Looser pattern: "order <something>"
    match = re.search(r"\border\s+([A-Za-z0-9\-]{6,12})\b", q, re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if not candidate.startswith("ORD-"):
            candidate = f"ORD-{candidate}"
        return candidate

    return None


# ========================================
# 📦 Order CRUD Operations
# ========================================

def preview_order(username: str) -> dict:
    """
    Build a preview of what the order would look like WITHOUT placing it.
    Returns cart items, total amount, and the user's profile address so the
    chatbot can ask for confirmation before actually placing.
    """
    # 1) Fetch cart
    cart = carts_collection.find_one({"username": username})
    if cart is None or len(cart.get("items", [])) == 0:
        return {
            "status": "empty_cart",
            "message": "Your cart is empty. Add some products before placing an order.",
        }

    # 2) Fetch user address
    user = users_collection.find_one({"username": username})
    if user is None:
        return {
            "status": "user_not_found",
            "message": "User profile not found. Please sign up or log in first.",
        }

    address = user.get("address", "").strip()
    if not address:
        return {
            "status": "missing_address",
            "message": "No delivery address found on your profile. Please update your address before placing an order.",
        }

    # 3) Build preview
    items = cart["items"]
    total_amount = calculate_total_amount(items)

    item_summaries = []
    for item in items:
        name = item.get("product_id", "Unknown")
        qty = item.get("quantity", 1)
        price = _extract_price(item.get("product_data", {}))
        item_summaries.append({
            "name": name,
            "quantity": qty,
            "unit_price": f"₹{price:,.2f}",
            "subtotal": f"₹{price * qty:,.2f}",
        })

    return {
        "status": "preview",
        "items": item_summaries,
        "item_count": len(items),
        "total_amount": f"₹{total_amount:,.2f}",
        "shipping_address": address,
        "message": (
            f"Order preview: {len(items)} item(s) totalling ₹{total_amount:,.2f}, "
            f"shipping to: {address}. Please confirm to place the order."
        ),
    }


def place_order(username: str) -> dict:
    """
    Place an order from the user's current cart.
    Steps:
      1. Fetch cart — must have items.
      2. Fetch user address — must exist.
      3. Calculate total amount.
      4. Insert order document.
      5. Clear the cart.
    """
    # 1) Fetch cart
    cart = carts_collection.find_one({"username": username})
    if cart is None or len(cart.get("items", [])) == 0:
        return {
            "status": "empty_cart",
            "message": "Your cart is empty. Add some products before placing an order.",
        }

    # 2) Fetch user address
    user = users_collection.find_one({"username": username})
    if user is None:
        return {
            "status": "user_not_found",
            "message": "User profile not found. Please sign up or log in first.",
        }

    address = user.get("address", "").strip()
    if not address:
        return {
            "status": "missing_address",
            "message": "No delivery address found on your profile. Please update your address before placing an order.",
        }

    # 3) Build order
    items = cart["items"]
    total_amount = calculate_total_amount(items)
    order_id = generate_order_id()
    now = datetime.utcnow()

    order_doc = {
        "order_id": order_id,
        "username": username,
        "items": items,
        "total_amount": total_amount,
        "shipping_address": address,
        "status": "placed",
        "created_at": now,
        "updated_at": now,
    }

    # 4) Insert order
    orders_collection.insert_one(order_doc)

    # 5) Clear cart
    carts_collection.update_one(
        {"username": username},
        {
            "$set": {"items": [], "updated_at": now},
        },
    )

    return {
        "status": "placed",
        "order_id": order_id,
        "total_amount": total_amount,
        "shipping_address": address,
        "item_count": len(items),
        "message": (
            f"Order {order_id} placed successfully! "
            f"{len(items)} item(s) totalling ₹{total_amount:.2f} "
            f"will be shipped to: {address}."
        ),
    }


def cancel_order(username: str, order_id: str) -> dict:
    """
    Cancel an existing order.
    - Verifies the order belongs to the user.
    - Only allows cancellation if status is 'placed'.
    """
    order = orders_collection.find_one({"order_id": order_id})

    if order is None:
        return {
            "status": "not_found",
            "order_id": order_id,
            "message": f"Order '{order_id}' was not found.",
        }

    # Ownership check
    if order["username"] != username:
        return {
            "status": "unauthorized",
            "order_id": order_id,
            "message": "You are not authorized to cancel this order.",
        }

    # Status check
    if order["status"] == "cancelled":
        return {
            "status": "already_cancelled",
            "order_id": order_id,
            "message": f"Order '{order_id}' has already been cancelled.",
        }

    if order["status"] != "placed":
        return {
            "status": "not_cancellable",
            "order_id": order_id,
            "message": (
                f"Order '{order_id}' cannot be cancelled — "
                f"current status is '{order['status']}'."
            ),
        }

    # Cancel
    orders_collection.update_one(
        {"order_id": order_id},
        {
            "$set": {
                "status": "cancelled",
                "updated_at": datetime.utcnow(),
            }
        },
    )

    return {
        "status": "cancelled",
        "order_id": order_id,
        "message": f"Order '{order_id}' has been cancelled successfully.",
    }


def get_order_history(username: str) -> dict:
    """
    Fetch all orders for the user, sorted by created_at descending.
    Separates placed and cancelled orders for easy consumption.
    """
    cursor = orders_collection.find(
        {"username": username}
    ).sort("created_at", DESCENDING)

    orders = []
    placed = []
    cancelled = []

    for doc in cursor:
        # Extract product names from order items
        product_names = []
        for item in doc.get("items", []):
            name = item.get("product_id", None)
            if not name:
                name = item.get("product_data", {}).get("product name", "Unknown")
            product_names.append(name)

        order_summary = {
            "order_id": doc["order_id"],
            "item_count": len(doc.get("items", [])),
            "products": product_names,
            "total_amount": doc.get("total_amount", 0),
            "shipping_address": doc.get("shipping_address", ""),
            "status": doc["status"],
            "created_at": doc["created_at"].isoformat() if isinstance(doc["created_at"], datetime) else str(doc["created_at"]),
        }
        orders.append(order_summary)
        if doc["status"] == "placed":
            placed.append(order_summary)
        elif doc["status"] == "cancelled":
            cancelled.append(order_summary)

    if not orders:
        return {
            "status": "empty",
            "orders": [],
            "placed": [],
            "cancelled": [],
            "message": "You have no orders yet.",
        }

    return {
        "status": "ok",
        "orders": orders,
        "placed": placed,
        "cancelled": cancelled,
        "message": (
            f"You have {len(orders)} order(s) — "
            f"{len(placed)} placed, {len(cancelled)} cancelled."
        ),
    }


def get_order_details(username: str, order_id: str) -> dict:
    """
    Fetch detailed information for a specific order.
    Returns all order details including items, prices, status, dates, and shipping.
    """
    order = orders_collection.find_one({"order_id": order_id})

    if order is None:
        return {
            "status": "not_found",
            "order_id": order_id,
            "message": f"Order '{order_id}' was not found.",
        }

    # Ownership check
    if order["username"] != username:
        return {
            "status": "unauthorized",
            "order_id": order_id,
            "message": "You are not authorized to view this order.",
        }

    # Build detailed response
    items = order.get("items", [])
    item_summaries = []
    for item in items:
        name = item.get("product_id", "Unknown")
        qty = item.get("quantity", 1)
        price = _extract_price(item.get("product_data", {}))
        item_summaries.append({
            "name": name,
            "quantity": qty,
            "unit_price": f"₹{price:,.2f}",
            "subtotal": f"₹{price * qty:,.2f}",
        })

    return {
        "status": "ok",
        "order_id": order_id,
        "items": item_summaries,
        "item_count": len(items),
        "total_amount": f"₹{order.get('total_amount', 0):,.2f}",
        "shipping_address": order.get("shipping_address", ""),
        "order_status": order.get("status", "unknown"),
        "created_at": order["created_at"].isoformat() if isinstance(order["created_at"], datetime) else str(order["created_at"]),
        "message": (
            f"Order {order_id}: {len(items)} item(s), "
            f"Total: ₹{order.get('total_amount', 0):,.2f}, "
            f"Status: {order.get('status', 'unknown').title()}"
        ),
    }


def preview_cancel_order(username: str, order_id: str) -> dict:
    """
    Fetch the order details for a cancellation preview WITHOUT actually cancelling.
    Returns enough info for the chatbot to show a confirmation prompt.
    """
    order = orders_collection.find_one({"order_id": order_id})

    if order is None:
        return {
            "status": "not_found",
            "order_id": order_id,
            "message": f"Order '{order_id}' was not found.",
        }

    if order["username"] != username:
        return {
            "status": "unauthorized",
            "order_id": order_id,
            "message": "You are not authorized to cancel this order.",
        }

    if order["status"] == "cancelled":
        return {
            "status": "already_cancelled",
            "order_id": order_id,
            "message": f"Order '{order_id}' has already been cancelled.",
        }

    if order["status"] != "placed":
        return {
            "status": "not_cancellable",
            "order_id": order_id,
            "message": (
                f"Order '{order_id}' cannot be cancelled — "
                f"current status is '{order['status']}'."
            ),
        }

    # Build preview
    items = order.get("items", [])
    item_summaries = []
    for item in items:
        name = item.get("product_id", "Unknown")
        qty = item.get("quantity", 1)
        price = _extract_price(item.get("product_data", {}))
        item_summaries.append({
            "name": name,
            "quantity": qty,
            "unit_price": f"₹{price:,.2f}",
            "subtotal": f"₹{price * qty:,.2f}",
        })

    return {
        "status": "preview_cancel",
        "order_id": order_id,
        "items": item_summaries,
        "item_count": len(items),
        "total_amount": f"₹{order.get('total_amount', 0):,.2f}",
        "shipping_address": order.get("shipping_address", ""),
        "order_date": order["created_at"].isoformat() if isinstance(order["created_at"], datetime) else str(order["created_at"]),
        "message": (
            f"Are you sure you want to cancel order {order_id}? "
            f"{len(items)} item(s) totalling ₹{order.get('total_amount', 0):,.2f}. "
            f"Please confirm to proceed."
        ),
    }


# ========================================
# 🧩 Order Manager Agent (entry point)
# ========================================

def order_manager_agent(intent: str, query: str, username: str, order_id: str | None = None) -> dict:
    """
    Main entry point called by the intent router.

    Parameters
    ----------
    intent : str
        One of place_order, cancel_order, order_history, preview_order, preview_cancel.
    query : str
        The user's raw message.
    username : str
        Authenticated username (from session).
    order_id : str | None
        Pre-resolved order ID (e.g. from ordinal references). If provided,
        cancel_order / preview_cancel will use this instead of parsing the query.

    Returns
    -------
    dict with keys: action, result.
    """

    # ── preview_order (show summary before confirmation) ──
    if intent == "preview_order":
        result = preview_order(username)
        return {"action": "preview_order", "result": result}

    # ── place_order (only after user confirmation) ─────────
    if intent == "place_order":
        result = place_order(username)
        return {"action": "place_order", "result": result}

    # ── preview_cancel (show order details before confirming cancel) ──
    if intent == "preview_cancel":
        if not order_id:
            oid = extract_order_id_from_query(query)
            if not oid:
                latest = orders_collection.find_one(
                    {"username": username, "status": "placed"},
                    sort=[("created_at", DESCENDING)],
                )
                if latest:
                    oid = latest["order_id"]
                else:
                    return {
                        "action": "preview_cancel",
                        "result": {
                            "status": "no_cancellable_order",
                            "message": "You don't have any active orders that can be cancelled.",
                        },
                    }
            order_id = oid
        result = preview_cancel_order(username, order_id)
        return {"action": "preview_cancel", "result": result}

    # ── cancel_order (only after user confirmation) ─────────
    if intent == "cancel_order":
        if not order_id:
            order_id = extract_order_id_from_query(query)
        if not order_id:
            # No order ID provided — find the most recent cancellable order
            latest = orders_collection.find_one(
                {"username": username, "status": "placed"},
                sort=[("created_at", DESCENDING)],
            )
            if latest:
                order_id = latest["order_id"]
            else:
                return {
                    "action": "cancel_order",
                    "result": {
                        "status": "no_cancellable_order",
                        "message": "You don't have any active orders that can be cancelled.",
                    },
                }
        result = cancel_order(username, order_id)
        return {"action": "cancel_order", "result": result}

    # ── order_history ───────────────────────────────────────
    if intent == "order_history":
        result = get_order_history(username)
        return {"action": "order_history", "result": result}

    # ── order_details ──────────────────────────────────────
    if intent == "order_details":
        if not order_id:
            oid = extract_order_id_from_query(query)
            if not oid:
                # No order ID provided — caller should resolve from session
                return {
                    "action": "order_details",
                    "result": {
                        "status": "need_order_id",
                        "message": "Please specify which order you'd like details for.",
                    },
                }
            order_id = oid
        result = get_order_details(username, order_id)
        return {"action": "order_details", "result": result}

    # ── unknown ─────────────────────────────────────────────
    return {
        "action": intent,
        "result": {"status": "error", "message": f"Unknown order intent: {intent}"},
    }
