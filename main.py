"""
Main application with chatbot authentication and product search.
Note: Run scripts/build_embeddings.py first to generate embeddings!
"""

# ===============================
# Imports
# ===============================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pymongo import MongoClient
import bcrypt
import jwt
from groq import Groq
from cart_agent import cart_manager_agent
from order_agent import order_manager_agent
from profile_agent import profile_manager_agent
from entity_extractor import extract_entities, resolve_intent_by_entities

# ===============================
# 🔧 Environment & Config
# ===============================

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")

# ===============================
# 🗄️ MongoDB Setup
# ===============================

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["final_year"]
users_collection = db["users"]
users_collection.create_index("username", unique=True)

# ===============================
# 🤖 Groq Client
# ===============================

groq_client = Groq(api_key=GROQ_API_KEY)

# ===============================
# 🔧 Cache Configuration
# ===============================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings_cache")
PRODUCT_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "product_embeddings.npy")
CATEGORY_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "category_embeddings.npy")
PATTERN_EMBEDDINGS_FILE = os.path.join(CACHE_DIR, "pattern_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_index.bin")
DF_CACHE_FILE = os.path.join(CACHE_DIR, "df_cache.pkl")


def load_cache():
    """Load all embeddings and data from cache."""
    embeddings = np.load(PRODUCT_EMBEDDINGS_FILE)
    category_embeddings = np.load(CATEGORY_EMBEDDINGS_FILE)
    pattern_embeddings = np.load(PATTERN_EMBEDDINGS_FILE)
    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(DF_CACHE_FILE, "rb") as f:
        cached = pickle.load(f)

    return (
        cached["df"], embeddings, category_embeddings,
        cached["categories"], index,
        pattern_embeddings, cached["pattern_texts"], cached["labels"],
    )


# ===============================
# ⚡ Load Embeddings from Cache
# ===============================

if not os.path.exists(DF_CACHE_FILE):
    print("❌ Embeddings not found!")
    print("📝 Please run: python scripts/build_embeddings.py")
    exit(1)

print("⚡ Loading embeddings from cache...")
(df, embeddings, category_embeddings, categories,
 index, pattern_embeddings, pattern_texts, labels) = load_cache()
print("Total Products:", len(df))
print("FAISS index loaded with", index.ntotal, "products")


# ===============================
# Load Models (needed for runtime queries)
# ===============================

bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")


# ===============================
# Semantic Category Detection
# ===============================

# Keyword → category overrides for terms that the semantic model misclassifies
_CATEGORY_KEYWORD_MAP = {
    "clothing": [
        "jeans", "pants", "trousers", "denim", "chinos", "joggers", "cargo pants",
        "shorts", "leggings", "shirt", "t-shirt", "tshirt", "kurta", "kurti",
        "jacket", "hoodie", "sweater", "blazer", "suit", "dress", "skirt",
        "sweatshirt", "tracksuit", "pyjama", "pajama",
    ],
    "footwear": [
        "shoes", "sneakers", "sandals", "boots", "slippers", "heels",
        "loafers", "flip flops", "crocs", "floaters",
    ],
    "accessories": [
        "watch", "watches", "belt", "belts", "wallet", "wallets",
        "sunglasses", "cap", "hat", "scarf",
    ],
}


def detect_category_semantic(query):
    q = query.lower().strip()

    # 1) Keyword override — check before semantic matching
    for category, keywords in _CATEGORY_KEYWORD_MAP.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', q):
                return category

    # 2) Semantic fallback
    query_vector = bge_model.encode(
        "Represent this product category search query: " + q,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = np.dot(category_embeddings, query_vector)
    best_index = np.argmax(similarities)

    return categories[best_index]

# ===============================
# Query Embedding
# ===============================

def embed_query(query):
    return bge_model.encode(
        "Represent this product search query for retrieval: " + query.lower(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

# ===============================
# Product Search Function
# ===============================

def search_products(query, top_k=5):

    detected_category = detect_category_semantic(query)

    # Strict filtering
    filtered_indices = df.index[df["product type"] == detected_category].tolist()

    if len(filtered_indices) == 0:
        print("No products found in detected category.")
        return pd.DataFrame()

    query_vector = embed_query(query)

    scores, indices = index.search(
        np.array([query_vector]),
        50
    )

    SIMILARITY_THRESHOLD = 0.62  # adjust if needed

    valid_results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx in filtered_indices and score >= SIMILARITY_THRESHOLD:
            valid_results.append((idx, score))

    if len(valid_results) == 0:
        print("No relevant products found.")
        print("Detected Category:", detected_category)
        return pd.DataFrame()

    valid_results = sorted(valid_results, key=lambda x: x[1], reverse=True)[:top_k]

    result_indices = [x[0] for x in valid_results]
    result_scores = [x[1] for x in valid_results]

    results = df.iloc[result_indices].copy()
    results["similarity_score"] = result_scores

    print("Detected Category:", detected_category)

    return results


# ==============================
# Intent Prediction Function
# ==============================

def predict_intent(query, threshold=0.38):
    query_embedding = minilm_model.encode([query])

    similarities = cosine_similarity(query_embedding, pattern_embeddings)

    best_index = np.argmax(similarities)
    best_score = similarities[0][best_index]
    predicted_intent = labels[best_index]

    if best_score < threshold:
        return "fallback", float(best_score)

    return predicted_intent, float(best_score)


# ========================================
# 🔐 Authentication Helpers
# ========================================

def hash_pin(pin: str) -> str:
    """Hash a PIN using bcrypt."""
    return bcrypt.hashpw(pin.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_pin(pin: str, hashed: str) -> bool:
    """Verify a PIN against its bcrypt hash."""
    return bcrypt.checkpw(pin.encode("utf-8"), hashed.encode("utf-8"))


def create_jwt(username: str) -> str:
    """Create a JWT token valid for 24 hours."""
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=24),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_jwt_token(token: str) -> Optional[str]:
    """Verify JWT and return username if valid, else None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("username")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ========================================
# 🤖 Groq LLM Helper
# ========================================

_BREVITY_DIRECTIVE = (
    "IMPORTANT STYLE RULES: "
    "Be concise — use the fewest words needed to be clear and helpful. "
    "Maximum 1-2 short sentences for simple replies. "
    "No filler phrases, no over-explaining. "
    "Do NOT use numbered lists, bullet points, or markdown formatting unless showing product data. "
    "Sound natural and friendly, like a real shop assistant — not robotic. "
)


def groq_chat(system_prompt: str, user_message: str = "") -> str:
    """Generate a response using Groq LLM (llama-3.1-8b-instant)."""
    messages = [{"role": "system", "content": _BREVITY_DIRECTIVE + system_prompt}]
    if user_message:
        messages.append({"role": "user", "content": user_message})
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I'm having trouble right now. Please try again. ({e})"


# ========================================
# 🗂️ Session Management
# ========================================

sessions: dict = {}


def get_or_create_session(session_id: Optional[str]) -> tuple:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "state": "greeting",
        "flow": None,
        "data": {},
        "token": None,
        "username": None,
        "authenticated": False,
        "conversation_history": [],
        "last_search_results": None,
        "last_category": None,
    }
    return new_id, sessions[new_id]


def determine_auth_choice(message: str) -> Optional[str]:
    """Determine if user wants to login or signup from keywords."""
    msg = message.lower().strip()
    signup_kw = ["sign up", "signup", "register", "new account",
                 "create account", "new user", "create"]
    login_kw = ["log in", "login", "sign in", "signin",
                "existing", "already have", "returning"]
    for kw in signup_kw:
        if kw in msg:
            return "signup"
    for kw in login_kw:
        if kw in msg:
            return "login"
    return None


# ========================================
# 🔑 Auth Conversation State Machine
# ========================================

def handle_auth_flow(session: dict, message: str) -> dict:
    """Drive the login / signup conversation one step at a time."""
    state = session["state"]

    # ── GREETING ─────────────────────────────────────────────
    if state == "greeting":
        # Check if the very first message already indicates a choice
        choice = determine_auth_choice(message)
        if choice == "signup":
            session["flow"] = "signup"
            session["state"] = "signup_name"
            reply = groq_chat(
                "You are a friendly e-commerce shopping assistant. The user wants to create a new account. "
                "Welcome them and then ask ONLY for their FULL NAME. "
                "Do NOT ask for username, PIN, or any other detail yet. "
                "Keep it to 1-2 sentences. Ask only ONE question."
            )
            return {"reply": reply}
        elif choice == "login":
            session["flow"] = "login"
            session["state"] = "login_username"
            reply = groq_chat(
                "You are a friendly e-commerce shopping assistant. The user wants to log in. "
                "Welcome them back and ask ONLY for their USERNAME. "
                "Do NOT ask for PIN or any other detail yet. "
                "Keep it to 1-2 sentences. Ask only ONE question."
            )
            return {"reply": reply}
        else:
            reply = groq_chat(
                "You are a friendly e-commerce shopping assistant chatbot. "
                "The user has just connected for the first time. "
                "Greet them warmly and ask whether they would like to login or signup. "
                "Keep it brief, professional, and friendly. Ask only ONE question. "
                "Do not use bullet points or numbered lists."
            )
            session["state"] = "choosing"
            return {"reply": reply}

    # ── CHOOSING ─────────────────────────────────────────────
    if state == "choosing":
        choice = determine_auth_choice(message)
        if choice == "signup":
            session["flow"] = "signup"
            session["state"] = "signup_name"
            reply = groq_chat(
                "You are a friendly shopping assistant. The user chose to sign up. "
                "Ask ONLY for their FULL NAME. Do NOT ask for username, PIN, or anything else. "
                "Keep it to 1-2 sentences."
            )
            return {"reply": reply}
        elif choice == "login":
            session["flow"] = "login"
            session["state"] = "login_username"
            reply = groq_chat(
                "You are a friendly shopping assistant. The user chose to log in. "
                "Ask ONLY for their USERNAME. Do NOT ask for PIN or anything else. "
                "Keep it to 1-2 sentences."
            )
            return {"reply": reply}
        else:
            reply = groq_chat(
                "You are a friendly shopping assistant. The user said something "
                "but you couldn't tell if they want to login or signup. "
                "Politely ask them again whether they'd like to login or signup. "
                "Be brief. Ask only ONE question.",
                user_message=message,
            )
            return {"reply": reply}

    # ── SIGNUP: name ────────────────────────────────────────
    if state == "signup_name":
        name = message.strip()
        if len(name) < 2:
            reply = groq_chat(
                "You are helping a user sign up. The name they gave seems too short. "
                "Politely ask them to provide their full name again. Ask only ONE question."
            )
            return {"reply": reply}
        session["data"]["name"] = name
        session["state"] = "signup_username"
        reply = groq_chat(
            f"The user's name is {name}. You are helping them sign up. "
            "Acknowledge their name briefly, then ask ONLY for a UNIQUE USERNAME they'd like to use. "
            "Do NOT ask for PIN, contact, address or anything else. Keep it to 1-2 sentences."
        )
        return {"reply": reply}

    # ── SIGNUP: username ────────────────────────────────────
    if state == "signup_username":
        username = message.strip().lower()
        if len(username) < 3:
            reply = groq_chat(
                "You are helping a user sign up. The username they chose is too short "
                "(minimum 3 characters). Politely ask them to choose a longer username. "
                "Ask only ONE question."
            )
            return {"reply": reply}
        if users_collection.find_one({"username": username}):
            reply = groq_chat(
                f"You are helping a user sign up. The username '{username}' is already taken. "
                "Politely inform them and ask them to choose a different username. "
                "Ask only ONE question."
            )
            return {"reply": reply}
        session["data"]["username"] = username
        session["state"] = "signup_pin"
        reply = groq_chat(
            f"Username '{username}' is available! Now ask the user to create a 4-DIGIT NUMERIC PIN "
            "(exactly 4 digits, numbers only, e.g. 1234). This PIN will be their password. "
            "Do NOT ask for contact, address, or anything else. Keep it to 1-2 sentences."
        )
        return {"reply": reply}

    # ── SIGNUP: pin ─────────────────────────────────────────
    if state == "signup_pin":
        pin = message.strip()
        if not re.match(r"^\d{4}$", pin):
            reply = groq_chat(
                "You are helping a user sign up. The PIN they entered is invalid. "
                "It must be exactly 4 numeric digits (e.g. 1234). "
                "Politely ask them to try again. Ask only ONE question."
            )
            return {"reply": reply}
        session["data"]["pin"] = pin
        session["state"] = "signup_contact"
        reply = groq_chat(
            "PIN accepted! Now ask the user ONLY for their CONTACT NUMBER (phone number). "
            "Do NOT ask for address or anything else. Keep it to 1-2 sentences."
        )
        return {"reply": reply}

    # ── SIGNUP: contact ─────────────────────────────────────
    if state == "signup_contact":
        contact = message.strip()
        if len(contact) < 7:
            reply = groq_chat(
                "You are helping a user sign up. The contact number seems too short. "
                "Politely ask them to provide a valid phone number. Ask only ONE question."
            )
            return {"reply": reply}
        session["data"]["phone"] = contact
        session["state"] = "signup_address"
        reply = groq_chat(
            "Contact saved! Now ask the user ONLY for their DELIVERY ADDRESS. "
            "This is the last step of signup. Keep it to 1-2 sentences."
        )
        return {"reply": reply}

    # ── SIGNUP: address (final step) ────────────────────────
    if state == "signup_address":
        address = message.strip()
        if len(address) < 5:
            reply = groq_chat(
                "You are helping a user sign up. The address seems too short. "
                "Politely ask them to provide a complete delivery address. "
                "Ask only ONE question."
            )
            return {"reply": reply}
        session["data"]["address"] = address

        # Save to MongoDB
        user_doc = {
            "name": session["data"]["name"],
            "username": session["data"]["username"],
            "pin": hash_pin(session["data"]["pin"]),
            "phone": session["data"]["phone"],
            "address": session["data"]["address"],
            "created_at": datetime.utcnow(),
        }
        try:
            users_collection.insert_one(user_doc)
        except Exception as e:
            reply = groq_chat(
                f"You are helping a user sign up but an error occurred: {e}. "
                "Apologize and ask them to try again."
            )
            return {"reply": reply}

        # Generate JWT & mark authenticated
        token = create_jwt(session["data"]["username"])
        session["token"] = token
        session["username"] = session["data"]["username"]
        session["authenticated"] = True
        session["state"] = "authenticated"

        reply = groq_chat(
            f"You are a friendly shopping assistant. The user '{session['data']['name']}' "
            f"has successfully signed up with username '{session['data']['username']}'. "
            "Welcome them briefly and ask what they'd like to do. One sentence only."
        )
        return {"reply": reply, "token": token, "authenticated": True}

    # ── LOGIN: username ─────────────────────────────────────
    if state == "login_username":
        username = message.strip().lower()
        session["data"]["username"] = username
        session["state"] = "login_pin"
        reply = groq_chat(
            f"The user provided username '{username}'. Now ask ONLY for their 4-DIGIT PIN. "
            "Do NOT ask for anything else. Keep it to 1-2 sentences."
        )
        return {"reply": reply}

    # ── LOGIN: pin ──────────────────────────────────────────
    if state == "login_pin":
        pin = message.strip()
        username = session["data"]["username"]
        user = users_collection.find_one({"username": username})

        if not user:
            session["state"] = "choosing"
            session["data"] = {}
            reply = groq_chat(
                f"You are helping a user log in. The username '{username}' was not found. "
                "Politely inform them and ask whether they'd like to try a different "
                "username or sign up for a new account instead. Ask only ONE question."
            )
            return {"reply": reply}

        if not verify_pin(pin, user["pin"]):
            reply = groq_chat(
                "You are helping a user log in. The PIN they entered is incorrect. "
                "Politely let them know and ask them to try their PIN again. "
                "Ask only ONE question."
            )
            return {"reply": reply}

        # Successful login
        token = create_jwt(username)
        session["token"] = token
        session["username"] = username
        session["authenticated"] = True
        session["state"] = "authenticated"

        reply = groq_chat(
            f"You are a friendly shopping assistant. The user '{user['name']}' "
            "has successfully logged in. Welcome them back briefly and ask what they'd like to do. "
            "One sentence only."
        )
        return {"reply": reply, "token": token, "authenticated": True}

    # ── fallback (should not reach here) ────────────────────
    return {"reply": "Something went wrong. Please try again."}


# ========================================
# 🛍️ Authenticated Chat Handler
# ========================================

def groq_chat_with_history(system_prompt: str, history: list, user_message: str = "") -> str:
    """Generate a response using Groq LLM with full conversation history."""
    messages = [{"role": "system", "content": _BREVITY_DIRECTIVE + system_prompt}]

    # Add conversation history (keep last 10 turns to stay within token limits)
    messages.extend(history[-10:])

    if user_message:
        messages.append({"role": "user", "content": user_message})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"I'm having trouble right now. Please try again. ({e})"


def is_followup_query(query: str) -> bool:
    """Check if a query is a vague follow-up that lacks searchable product keywords."""
    followup_patterns = [
        # Price / cost questions
        r"\b(what|how much|whats|what's)\b.*(price|cost|rate|worth|expensive|cheap)",
        r"\b(price|cost)\b.*(this|that|it|these|those|first|second|third|last|one)",
        r"\bhow much\b",
        # Detail / info requests
        r"\b(tell|know|give|show|get)\b.*(more|detail|info|about|spec|feature)",
        r"\b(more|further)\b.*(detail|info|about)",
        # Pronouns referencing previous context
        r"\b(this|that|it|these|those)\b.*(good|available|stock|color|size|variant|model|brand|rating|review|deliver|ship)",
        r"\b(is it|is this|is that|are these|are those)\b",
        r"\bwhat about (this|that|it|these|those)\b",
        # Comparison / selection
        r"\bwhich (one|is better|should|do you)\b",
        r"\b(compare|difference|vs|versus|between)\b.*(them|two|these|those)",
        r"\b(first|second|third|last|1st|2nd|3rd)\b.*(one|option|product|phone|item)",
        # Generic follow-ups
        r"^(and|also|what about|how about|ok|okay)\b",
        r"\b(any (discount|offer|deal|coupon))\b",
        r"\b(available|in stock|delivery|shipping)\b.*(this|that|it)\b",
    ]
    q = query.lower().strip()
    for pattern in followup_patterns:
        if re.search(pattern, q):
            return True
    return False


# ── Ordinal detector for tracking which product the user is discussing ──

_FOLLOWUP_ORDINAL_MAP = {
    "first": 0, "1st": 0, "#1": 0, "number 1": 0, "number one": 0,
    "second": 1, "2nd": 1, "#2": 1, "number 2": 1, "number two": 1,
    "third": 2, "3rd": 2, "#3": 2, "number 3": 2, "number three": 2,
    "fourth": 3, "4th": 3, "#4": 3, "number 4": 3,
    "fifth": 4, "5th": 4, "#5": 4, "number 5": 4,
    "last": -1,
}


def _detect_discussed_product_index(query: str, results_count: int) -> int | None:
    """Detect if the user is referencing a specific product by ordinal.
    Returns the 0-based index, or None if no ordinal found."""
    q = query.lower().strip()
    for word, idx in _FOLLOWUP_ORDINAL_MAP.items():
        if re.search(r'\b' + re.escape(word) + r'\b', q):
            actual_idx = idx if idx >= 0 else results_count + idx
            if 0 <= actual_idx < results_count:
                return actual_idx
    return None


# ── Confidence threshold for the embedding classifier ──
INTENT_CONFIDENCE_THRESHOLD = 0.75


def override_intent_by_keywords(query: str, session: dict) -> str | None:
    """Keyword-based intent override using entity extraction.
    Returns an intent string if a keyword rule fires, else None."""
    q = query.lower().strip()

    # ── Profile intents (checked FIRST to prevent entity extractor misrouting) ──
    if re.search(r"\b(?:view|show|see|get|display)\b.*\b(?:profile|account|details|info)\b", q):
        return "view_profile"
    if re.search(r"\b(?:my\s+profile|my\s+account|my\s+details|my\s+info)\b", q) and not re.search(r"\b(?:update|change|edit|modify|set|correct|revise)\b", q):
        return "view_profile"
    if re.search(r"\b(?:update|change|edit|modify|set|correct|revise)\b.*\b(?:profile|account|details|info|address|phone|email|contact|number|mail|mobile)\b", q):
        return "update_profile"
    if re.search(r"\b(?:address|phone|email|contact|mobile)\b.*\b(?:to|=|:)\b", q):
        return "update_profile"
    if re.search(r"\b(?:i\s+want\s+to|i\s+need\s+to|i\s+would\s+like\s+to)\b.*\b(?:change|update|edit|modify)\b.*\b(?:profile|account|details|info|address|phone|email|contact)\b", q):
        return "update_profile"

    # ── Entity-based resolution (handles update_cart, browse_product, etc.) ──
    entity_intent = resolve_intent_by_entities(q)
    if entity_intent:
        return entity_intent

    # ── Fallbacks for pronoun / ordinal add-to-cart ("add it", "add the first one") ──
    if re.search(r"^(?:please\s+)?(?:add|put)\s+(?:it|this|that)\b", q):
        return "add_to_cart"
    if re.search(r"^(?:please\s+)?(?:add|put)\s+(?:the\s+)?(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b", q):
        return "add_to_cart"
    # "add <product name>" when we have recent search results (likely referencing them)
    if session.get("last_search_results") and re.match(r"^(?:please\s+)?(?:add|put)\s+", q):
        return "add_to_cart"
    # "yes add it" / "yes, add"
    if re.search(r"\byes\b.*\badd\b", q):
        return "add_to_cart"

    # ── Cart detail / total intents ──
    if re.search(r"\b(?:total|value|summary|worth)\b.*\b(?:cart|basket|bag)\b", q):
        return "cart_detail"
    if re.search(r"\b(?:cart|basket|bag)\b.*\b(?:total|value|summary|worth)\b", q):
        return "cart_detail"
    if re.search(r"\bhow\s+much\b.*\b(?:cart|basket|bag)\b", q):
        return "cart_detail"

    # ── Order intents ──
    if re.search(r"\b(?:place|confirm|finalize|complete|proceed)\b.*\b(?:order|purchase|checkout|payment)\b", q):
        return "place_order"
    if re.search(r"\b(?:checkout|place\s+order|confirm\s+order)\b", q):
        return "place_order"
    if re.search(r"\bcancel\b.*\border\b", q):
        return "cancel_order"
    if re.search(r"\b(?:order\s+history|past\s+orders|previous\s+orders|my\s+orders|purchase\s+history)\b", q):
        return "order_history"

    return None


# ── Ordinal / order-ID resolver for cancel commands ──

_ORDINAL_MAP = {
    "first": 0, "1st": 0, "one": 0, "1": 0,
    "second": 1, "2nd": 1, "two": 1, "2": 1,
    "third": 2, "3rd": 2, "three": 2, "3": 2,
    "fourth": 3, "4th": 3, "four": 3, "4": 3,
    "fifth": 4, "5th": 4, "five": 4, "5": 4,
    "sixth": 5, "6th": 5, "six": 5, "6": 5,
    "seventh": 6, "7th": 6, "seven": 6, "7": 6,
    "eighth": 7, "8th": 7, "eight": 7, "8": 7,
    "ninth": 8, "9th": 8, "nine": 8, "9": 8,
    "tenth": 9, "10th": 9, "ten": 9, "10": 9,
    "last": -1,
}


def _resolve_order_id_from_query(query: str, session: dict) -> str | None:
    """
    Try to figure out which order the user is referring to.
    Resolution order:
      1. Explicit ORD-XXXXX in the message
      2. Ordinal reference ("first", "second", "last") against the
         last displayed order list stored in session["last_order_list"]
      3. None (caller decides fallback)
    """
    from order_agent import extract_order_id_from_query

    # 1) Explicit order ID
    oid = extract_order_id_from_query(query)
    if oid:
        return oid

    # 2) Ordinal reference
    order_list = session.get("last_order_list", [])
    if order_list:
        q = query.lower().strip()
        for word, idx in _ORDINAL_MAP.items():
            if re.search(r'\b' + re.escape(word) + r'\b', q):
                try:
                    order = order_list[idx]  # supports negative index for "last"
                    return order.get("order_id")
                except IndexError:
                    pass

    return None


def _is_user_confirming(message: str) -> bool:
    """
    Determine whether the user's message is a confirmation or a refusal.
    Returns True for confirmations, False for anything else.
    """
    msg = message.lower().strip()
    confirm_keywords = [
        "yes", "yeah", "yep", "yup", "sure", "ok", "okay",
        "confirm", "confirmed", "place it", "place order", "place the order",
        "go ahead", "proceed", "do it", "let's go", "absolutely",
        "affirmative", "correct", "right", "definitely", "of course",
    ]
    deny_keywords = [
        "no", "nah", "nope", "cancel", "don't", "stop", "wait",
        "not now", "never mind", "nevermind", "hold on", "back",
    ]
    for kw in deny_keywords:
        if kw in msg:
            return False
    for kw in confirm_keywords:
        if kw in msg:
            return True
    return False


def handle_authenticated_chat(session: dict, username: str, message: str) -> dict:
    """Route an authenticated user's message through intent detection → agent → Groq."""

    history = session.setdefault("conversation_history", [])

    # ── Check if we are awaiting order placement confirmation ──
    if session.get("awaiting_order_confirmation"):
        confirmed = _is_user_confirming(message)
        session["awaiting_order_confirmation"] = False
        preview_data = session.pop("order_preview_data", {})

        if confirmed:
            agent_result = order_manager_agent(
                intent="place_order",
                query=message,
                username=username,
            )
            result_data = agent_result.get("result", {})
            if result_data.get("status") == "placed":
                reply = (
                    f"\u2705 **Order placed successfully!**\n\n"
                    f"**Order ID:** {result_data.get('order_id', 'N/A')}\n"
                    f"**Total:** \u20b9{result_data.get('total_amount', 0):,.2f}\n"
                    f"**Items:** {result_data.get('item_count', 0)}\n"
                    f"**Shipping to:** {result_data.get('shipping_address', 'N/A')}\n\n"
                    f"Thank you for your order!"
                )
            else:
                reply = result_data.get("message", "Something went wrong while placing the order.")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return {"reply": reply, "data": {"intent": "place_order", "order_result": result_data}}
        else:
            reply = groq_chat_with_history(
                "You are a helpful shopping assistant. The user decided NOT to place the order. "
                "Acknowledge politely and let them know their cart items are still saved. "
                "Ask if there's anything else they'd like to do. Be brief.",
                history,
                user_message=message,
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return {"reply": reply, "data": {"intent": "place_order_cancelled"}}

    # ── Check if we are awaiting cancel order confirmation ──
    if session.get("awaiting_cancel_confirmation"):
        confirmed = _is_user_confirming(message)
        session["awaiting_cancel_confirmation"] = False
        cancel_order_id = session.pop("cancel_order_id", None)

        if confirmed and cancel_order_id:
            agent_result = order_manager_agent(
                intent="cancel_order",
                query=message,
                username=username,
                order_id=cancel_order_id,
            )
            result_data = agent_result.get("result", {})
            if result_data.get("status") == "cancelled":
                reply = f"\u2705 Order **{result_data.get('order_id', '')}** has been cancelled successfully."
            else:
                reply = result_data.get("message", "Something went wrong while cancelling the order.")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return {"reply": reply, "data": {"intent": "cancel_order", "order_result": result_data}}
        else:
            reply = groq_chat_with_history(
                "You are a helpful shopping assistant. The user decided NOT to cancel the order. "
                "Acknowledge politely and let them know the order remains active. "
                "Ask if there's anything else they'd like to do. Be brief.",
                history,
                user_message=message,
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return {"reply": reply, "data": {"intent": "cancel_order_aborted"}}

    intent, confidence = predict_intent(message)
    print(f"[{username}] Intent: {intent} (confidence: {confidence:.2f})")

    # ── Entity extraction for disambiguation ──
    entities = extract_entities(message)
    print(f"[{username}] Entities: {entities}")

    # ── Keyword override: always fires if entity extractor has a strong signal ──
    override = override_intent_by_keywords(message, session)
    if override and override != intent:
        print(f"[{username}] Intent overridden: {intent} → {override}")
        intent = override
    # ── Low-confidence fallback: use entity-based intent when classifier is unsure ──
    elif confidence < INTENT_CONFIDENCE_THRESHOLD and entities["intent"]:
        print(f"[{username}] Low confidence ({confidence:.2f}), falling back to entity intent: {entities['intent']}")
        intent = entities["intent"]

    history = session.setdefault("conversation_history", [])

    # ── Keyword guard: route vague queries to follow-up if context exists ──
    if intent == "browse_product" and session.get("last_search_results") and is_followup_query(message):
        print(f"[{username}] Redirecting vague query to follow-up handler")

        # Track which product the user is discussing (by ordinal)
        search_results = session.get("last_search_results", [])
        discussed_idx = _detect_discussed_product_index(message, len(search_results))
        if discussed_idx is not None:
            session["last_discussed_product_index"] = discussed_idx

        reply = groq_chat_with_history(
            "You are a helpful shopping assistant. The user is asking a follow-up question "
            "about products that were previously shown.\n\n"
            "STRICT RULES:\n"
            "- ONLY use information from the product data provided below.\n"
            "- Do NOT invent, guess, or hallucinate ANY details (prices, specs, features, descriptions) that are not explicitly present in the data.\n"
            "- If a detail is not in the data, say you don't have that information.\n"
            "- Do NOT say the product was added to cart or mention cart operations.\n"
            "- Keep your response factual and based solely on the data below.\n\n"
            f"Previously shown products:\n{json.dumps(session['last_search_results'], indent=2)}",
            history,
            user_message=message,
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": "followup"}}

    # ── browse_product ──────────────────────────────────────
    if intent == "browse_product":
        results = search_products(message)
        category = detect_category_semantic(message)

        if isinstance(results, pd.DataFrame) and results.empty:
            reply = groq_chat_with_history(
                "You are a helpful shopping assistant. The user searched for a product but "
                "it is NOT available in our store's inventory. "
                "You MUST NOT suggest, list, or make up any products from your own knowledge. "
                "Simply tell the user we don't have that product right now and ask them to "
                "try searching for something else. Be brief — 1-2 sentences only.",
                history,
                user_message=message,
            )
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
            return {"reply": reply, "data": {"intent": intent, "category": category, "results": []}}

        product_list = []
        for _, row in results.iterrows():
            product = {
                col: str(row[col])
                for col in results.columns
                if col != "similarity_score" and str(row[col]).strip() not in ("", "nan")
            }
            product["relevance_score"] = f"{row['similarity_score']:.2f}"
            product_list.append(product)

        # Store results in session for follow-up questions
        session["last_search_results"] = product_list
        session["last_category"] = category

        reply = groq_chat_with_history(
            "You are a helpful shopping assistant. Present these search results concisely.\n\n"
            "STRICT RULES:\n"
            "- For each product show ONLY: name, price, and 1 key differentiator (color, RAM, brand, etc.) in ONE line.\n"
            "- All prices in Indian Rupees (\u20b9).\n"
            "- Do NOT invent details not in the data. Do NOT mention cart.\n"
            "- After listing, ask if they want more details on any item.\n\n"
            f"Products:\n{json.dumps(product_list, indent=2)}",
            history,
            user_message=message,
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": intent, "category": category, "results": product_list}}

    # ── profile_changes → remap to view_profile or update_profile ──
    if intent == "profile_changes":
        q_lower = message.lower()
        if re.search(r"\b(?:update|change|edit|modify|set|correct|revise)\b", q_lower):
            intent = "update_profile"
        else:
            intent = "view_profile"

    # ── profile intents (view_profile, update_profile) ──────
    if intent in ("view_profile", "update_profile"):
        agent_result = profile_manager_agent(
            intent=intent,
            query=message,
            username=username,
        )
        result_data = agent_result.get("result", {})

        if intent == "view_profile":
            if result_data.get("status") == "ok":
                p = result_data["profile"]
                reply = (
                    f"**Your Profile:**\n\n"
                    f"**Name:** {p.get('name', 'N/A')}\n"
                    f"**Username:** {p.get('username', 'N/A')}\n"
                    f"**Email:** {p.get('email', 'N/A') or 'Not set'}\n"
                    f"**Phone:** {p.get('phone', 'N/A') or 'Not set'}\n"
                    f"**Address:** {p.get('address', 'N/A') or 'Not set'}\n\n"
                    f"You can update your email, phone, or address anytime."
                )
            else:
                reply = result_data.get("message", "Could not retrieve your profile.")

        elif intent == "update_profile":
            if result_data.get("status") == "updated":
                fields = result_data.get("updated_fields", {})
                lines = [f"  - **{k.title()}:** {v}" for k, v in fields.items()]
                reply = "\u2705 Profile updated successfully!\n\n" + "\n".join(lines)
                if result_data.get("warnings"):
                    reply += "\n\n\u26a0\ufe0f Warnings: " + " ".join(result_data["warnings"])
            elif result_data.get("status") == "no_fields":
                reply = result_data.get("message", "No valid fields found to update. You can update: phone, email, or address.")
            elif result_data.get("status") == "validation_error":
                reply = result_data.get("message", "Validation failed. Please check your input.")
            else:
                reply = result_data.get("message", "Could not update your profile.")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": intent, "profile_result": result_data}}

    # ── logout ──────────────────────────────────────────────
    if intent == "logout":
        user = users_collection.find_one({"username": username})
        display_name = user["name"] if user else username
        reply = groq_chat_with_history(
            f"You are a friendly shopping assistant. The user '{display_name}' wants to log out. "
            "Say goodbye warmly and let them know they've been logged out successfully. Be brief.",
            history,
            user_message=message,
        )
        return {"reply": reply, "data": {"intent": intent}, "logout": True}

    # ── cart intents (add_to_cart, remove_from_cart, cart_detail, update_cart) ──
    cart_intents = ["add_to_cart", "remove_from_cart", "cart_detail", "update_cart"]
    if intent in cart_intents:
        agent_result = cart_manager_agent(
            intent=intent,
            query=message,
            username=username,
            search_fn=search_products,
            session=session,
        )
        result_data = agent_result.get("result", {})

        # ── Build response PROGRAMMATICALLY from actual MongoDB data ──
        # (No LLM involved for factual cart data to prevent hallucination)
        from cart_agent import get_cart as _get_cart_from_db
        current_cart = _get_cart_from_db(username)
        cart_items = current_cart.get("items", [])
        cart_total = current_cart.get("cart_total", "₹0")

        # Format cart display
        if cart_items:
            cart_lines = []
            for ci, item in enumerate(cart_items, 1):
                name = item.get("product_id", "Unknown")
                qty = item.get("quantity", 1)
                unit_price = item.get("unit_price", "N/A")
                subtotal = item.get("subtotal", "N/A")
                cart_lines.append(
                    f"  {ci}. **{name}** — Qty: {qty}, Unit Price: {unit_price}, Subtotal: {subtotal}"
                )
            cart_display = "\n".join(cart_lines) + f"\n\n**Cart Total: {cart_total}**"
        else:
            cart_display = None

        # Build reply based on operation type
        op_status = result_data.get("status", "error")
        product_name = result_data.get("product_id", "the product")

        if intent == "cart_detail":
            if cart_display:
                reply = f"Here's what's in your cart ({len(cart_items)} item(s)):\n\n{cart_display}"
            else:
                reply = "Your cart is empty. Browse our products to find something you like!"

        elif intent == "add_to_cart":
            if op_status == "added":
                reply = f"\u2705 **{product_name}** has been added to your cart!\n\nYour cart:\n{cart_display}"
            elif op_status == "incremented":
                reply = f"\u2705 **{product_name}** quantity updated in your cart.\n\nYour cart:\n{cart_display}"
            elif op_status == "no_product":
                reply = result_data.get("message", "Couldn't find a product to add. Try searching for it first.")
            else:
                reply = result_data.get("message", "Something went wrong while adding to cart.")

        elif intent == "remove_from_cart":
            if op_status == "removed":
                if cart_display:
                    reply = f"\u2705 **{product_name}** has been removed from your cart.\n\nYour cart:\n{cart_display}"
                else:
                    reply = f"\u2705 **{product_name}** has been removed. Your cart is now empty."
            else:
                reply = result_data.get("message", "Couldn't find that product in your cart.")

        elif intent == "update_cart":
            if op_status == "updated":
                reply = f"\u2705 **{product_name}** quantity updated.\n\nYour cart:\n{cart_display}"
            elif op_status == "removed":
                if cart_display:
                    reply = f"\u2705 **{product_name}** has been removed (quantity set to 0).\n\nYour cart:\n{cart_display}"
                else:
                    reply = f"\u2705 **{product_name}** has been removed. Your cart is now empty."
            else:
                reply = result_data.get("message", "Couldn't update the cart.")

        else:
            reply = result_data.get("message", "Cart operation completed.")

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": intent, "cart_result": result_data}}

    # ── order intents (place_order, cancel_order, order_history) ──
    order_intents = ["place_order", "cancel_order", "order_history"]
    if intent in order_intents:

        # ── place_order: show preview + ask for confirmation first ──
        if intent == "place_order":
            preview_result = order_manager_agent(
                intent="preview_order",
                query=message,
                username=username,
            )
            preview_data = preview_result.get("result", {})

            if preview_data.get("status") == "preview":
                # Store preview and set confirmation flag
                session["awaiting_order_confirmation"] = True
                session["order_preview_data"] = preview_data

                # Build programmatic order preview
                items_text = "\n".join(
                    f"  {i}. **{item['name']}** — Qty: {item['quantity']}, {item['unit_price']}"
                    for i, item in enumerate(preview_data.get("items", []), 1)
                )
                preview_text = (
                    f"**Order Summary:**\n"
                    f"{items_text}\n\n"
                    f"**Total:** {preview_data.get('total_amount', 'N/A')}\n"
                    f"**Shipping to:** {preview_data.get('shipping_address', 'N/A')}\n\n"
                    f"Would you like to confirm this order? Type **yes** to place or **no** to cancel."
                )
                reply = preview_text
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": reply})
                return {"reply": reply, "data": {"intent": "preview_order", "preview": preview_data}}
            else:
                # Preview failed (empty cart, missing address, etc.) — show error
                reply = preview_data.get("message", "Cannot place order right now. Please check your cart and profile.")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": reply})
                return {"reply": reply, "data": {"intent": "place_order", "order_result": preview_data}}

        # ── cancel_order: show preview + ask for confirmation first ──
        if intent == "cancel_order":
            resolved_id = _resolve_order_id_from_query(message, session)
            preview_result = order_manager_agent(
                intent="preview_cancel",
                query=message,
                username=username,
                order_id=resolved_id,
            )
            preview_data = preview_result.get("result", {})

            if preview_data.get("status") == "preview_cancel":
                session["awaiting_cancel_confirmation"] = True
                session["cancel_order_id"] = preview_data["order_id"]

                # Build programmatic cancel preview
                items_text = "\n".join(
                    f"  {i}. **{item['name']}** — Qty: {item['quantity']}, {item['unit_price']}"
                    for i, item in enumerate(preview_data.get("items", []), 1)
                )
                cancel_text = (
                    f"**Cancel Order {preview_data.get('order_id', '')}?**\n\n"
                    f"Items:\n{items_text}\n\n"
                    f"**Total:** {preview_data.get('total_amount', 'N/A')}\n"
                    f"**Order Date:** {preview_data.get('order_date', 'N/A')}\n"
                    f"**Shipping to:** {preview_data.get('shipping_address', 'N/A')}\n\n"
                    f"Type **yes** to confirm cancellation or **no** to keep the order."
                )
                reply = cancel_text
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": reply})
                return {"reply": reply, "data": {"intent": "preview_cancel", "preview": preview_data}}
            else:
                # Preview failed (not found, already cancelled, etc.)
                reply = preview_data.get("message", "Cannot cancel this order right now.")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": reply})
                return {"reply": reply, "data": {"intent": "cancel_order", "order_result": preview_data}}

        # ── order_history ──
        agent_result = order_manager_agent(
            intent=intent,
            query=message,
            username=username,
        )
        result_data = agent_result.get("result", {})

        if intent == "order_history":
            # Build programmatic order history display
            order_list = result_data.get("orders", [])
            if order_list:
                session["last_order_list"] = order_list
                placed_count = len(result_data.get("placed", []))
                cancelled_count = len(result_data.get("cancelled", []))
                header = f"You have {len(order_list)} order(s) — {placed_count} placed, {cancelled_count} cancelled.\n"
                order_lines = []
                for i, o in enumerate(order_list, 1):
                    products = o.get("products", [])
                    product_str = ", ".join(products) if products else f"{o.get('item_count', 0)} item(s)"
                    order_lines.append(
                        f"  {i}. **Order {o['order_id']}**\n"
                        f"     - Products: {product_str}\n"
                        f"     - Total: \u20b9{o.get('total_amount', 0):,.2f}\n"
                        f"     - Status: {o.get('status', 'unknown').title()}\n"
                        f"     - Date: {o.get('created_at', 'N/A')[:10]}\n"
                        f"     - Shipping: {o.get('shipping_address', 'N/A')}"
                    )
                reply = header + "\n".join(order_lines)
            else:
                reply = "You have no orders yet. Start shopping to place your first order!"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": intent, "order_result": result_data}}

    # ── track_order (placeholder) ───────────────────────────
    if intent == "track_order":
        reply = groq_chat_with_history(
            "You are a helpful shopping assistant. The user wants to track their order. "
            "This feature is being built and will be available soon. "
            "Politely inform them. Be brief.",
            history,
            user_message=message,
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": intent}}

    # ── follow-up about last search results ─────────────────
    if intent == "fallback" and session.get("last_search_results"):
        # Track ordinal references in follow-ups too
        search_results = session.get("last_search_results", [])
        discussed_idx = _detect_discussed_product_index(message, len(search_results))
        if discussed_idx is not None:
            session["last_discussed_product_index"] = discussed_idx

        reply = groq_chat_with_history(
            "You are a helpful shopping assistant. The user is asking a follow-up question "
            "about products that were previously shown.\n\n"
            "STRICT RULES:\n"
            "- ONLY use information from the product data provided below.\n"
            "- Do NOT invent, guess, or hallucinate ANY details (prices, specs, features, descriptions) that are not explicitly present in the data.\n"
            "- If the user asks about something not in the data, say you don't have that information.\n"
            "- Do NOT say the product was added to cart or mention cart operations unless the user explicitly asks.\n"
            "- Keep your response factual and based solely on the data below.\n\n"
            f"Previously shown products:\n{json.dumps(session['last_search_results'], indent=2)}",
            history,
            user_message=message,
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return {"reply": reply, "data": {"intent": "followup"}}

    # ── fallback (no prior results) ─────────────────────────
    reply = groq_chat_with_history(
        "You are a friendly shopping assistant. The user said something that doesn't "
        "match a specific action. Respond helpfully and let them know you can help with: "
        "browsing products, managing their cart, placing/tracking/cancelling orders, "
        "viewing order history, or updating their profile. Be brief.",
        history,
        user_message=message,
    )
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return {"reply": reply, "data": {"intent": "fallback"}}


# ========================================
# 🚀 Terminal Chatbot Loop
# ========================================

def run_chatbot():
    """Interactive terminal chatbot with login/signup → intent routing → Groq replies."""
    print("\n🚀 Chatbot Ready! (type 'exit' to quit)\n")

    session = {
        "state": "greeting",
        "flow": None,
        "data": {},
        "token": None,
        "username": None,
        "authenticated": False,
        "conversation_history": [],
        "last_search_results": None,
        "last_category": None,
    }

    # Generate initial greeting
    greeting = groq_chat(
        "You are a friendly e-commerce shopping assistant chatbot. "
        "The user has just connected for the first time. "
        "Greet them warmly and ask whether they would like to login or signup. "
        "Keep it brief, professional, and friendly. Ask only ONE question. "
        "Do not use bullet points or numbered lists."
    )
    session["state"] = "choosing"
    print(f"Bot: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break

        # ── Authenticated: route through intent detection ──
        if session.get("authenticated") and session.get("username"):
            result = handle_authenticated_chat(session, session["username"], user_input)

            print(f"\nBot: {result['reply']}\n")

            # Handle logout → reset session
            if result.get("logout"):
                session.update({
                    "state": "greeting",
                    "flow": None,
                    "data": {},
                    "token": None,
                    "username": None,
                    "authenticated": False,
                    "conversation_history": [],
                    "last_search_results": None,
                    "last_category": None,
                })
                # Show fresh greeting after logout
                greeting = groq_chat(
                    "You are a friendly e-commerce shopping assistant chatbot. "
                    "The user has just logged out and is back to the start. "
                    "Greet them and ask whether they would like to login or signup. "
                    "Keep it brief, professional, and friendly. Ask only ONE question."
                )
                session["state"] = "choosing"
                print(f"Bot: {greeting}\n")
            continue

        # ── Not authenticated: drive auth flow ──
        result = handle_auth_flow(session, user_input)
        print(f"\nBot: {result['reply']}\n")



# ========================================
# 🎤 Voice Chatbot Loop
# ========================================

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting for cleaner TTS output."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)   # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)        # *italic*
    text = re.sub(r'__(.+?)__', r'\1', text)        # __bold__
    text = re.sub(r'_(.+?)_', r'\1', text)          # _italic_
    text = re.sub(r'#+\s*', '', text)               # # headers
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text) # [link](url)
    text = re.sub(r'`(.+?)`', r'\1', text)          # `code`
    return text.strip()


def run_voice_chatbot():
    """Interactive voice chatbot — mirrors run_chatbot() but uses STT/TTS."""
    from voice_assistant import VoiceAssistant
    from ui_server import AssistantUIServer
    from ui_state import AssistantUIStateStore

    print("\n🎤 Voice Chatbot Starting...\n")

    # ── Start UI server for frontend ──
    ui_port = int(os.getenv("UI_PORT", "8000"))
    ui_server = AssistantUIServer(host="127.0.0.1", port=ui_port)
    ui_state = AssistantUIStateStore(on_change=ui_server.schedule_broadcast)
    ui_server.attach_store(ui_state)
    ui_server.start()
    ui_state.set(text="Starting assistant…", speaking=False)

    va = VoiceAssistant()

    session = {
        "state": "greeting",
        "flow": None,
        "data": {},
        "token": None,
        "username": None,
        "authenticated": False,
        "conversation_history": [],
        "last_search_results": None,
        "last_category": None,
    }

    def say(text):
        """Speak text via TTS with interrupt support (strips markdown first)."""
        print(f"Bot: {text}")
        clean = _strip_markdown(text)
        # Replace newlines with pauses for natural speech
        clean = re.sub(r'\n+', '. ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if clean:
            ui_state.set(speaking=True)
            done, reason = va.speak_sentences(
                clean,
                on_sentence=lambda s: ui_state.set(text=s, speaking=True),
            )
            ui_state.set(speaking=False)
            if not done:
                print("⚠️ Speech interrupted by user")

    def listen():
        """Listen for speech and return text, or None."""
        ui_state.set(text="Listening…", speaking=False)
        text, _ = va.listen_for_speech()
        if text:
            print(f"You: {text}")
        return text

    # Generate initial greeting
    greeting = groq_chat(
        "You are a friendly e-commerce shopping assistant chatbot. "
        "The user has just connected for the first time. "
        "Greet them warmly and ask whether they would like to login or signup. "
        "Keep it brief, professional, and friendly. Ask only ONE question. "
        "Do not use bullet points or numbered lists."
    )
    session["state"] = "choosing"
    say(greeting)

    try:
        while True:
            user_input = listen()

            if not user_input:
                say("I didn't catch that. Could you please repeat?")
                continue

            if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                say("Goodbye! Have a great day!")
                break

            # ── Authenticated: route through intent detection ──
            if session.get("authenticated") and session.get("username"):
                ui_state.set(text="Thinking…", speaking=False)
                result = handle_authenticated_chat(
                    session, session["username"], user_input
                )
                say(result["reply"])

                # Handle logout → reset session
                if result.get("logout"):
                    session.update({
                        "state": "greeting",
                        "flow": None,
                        "data": {},
                        "token": None,
                        "username": None,
                        "authenticated": False,
                        "conversation_history": [],
                        "last_search_results": None,
                        "last_category": None,
                    })
                    greeting = groq_chat(
                        "You are a friendly e-commerce shopping assistant chatbot. "
                        "The user has just logged out and is back to the start. "
                        "Greet them and ask whether they would like to login or signup. "
                        "Keep it brief, professional, and friendly. Ask only ONE question."
                    )
                    session["state"] = "choosing"
                    say(greeting)
                continue

            # ── Not authenticated: drive auth flow ──
            result = handle_auth_flow(session, user_input)
            say(result["reply"])

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        va.cleanup()
        ui_server.stop()
        print("🧹 Voice session ended.")


if __name__ == "__main__":
    import sys
    if "--voice" in sys.argv:
        run_voice_chatbot()
    else:
        run_chatbot()