import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import groq
import openai  # For whisper
import torch
import sounddevice as sd
import wave
import threading
import queue
import time
from TTS.api import TTS
from dotenv import load_dotenv
load_dotenv()
import os

# -----------------------
# Voice Configuration
# -----------------------
class VoiceAssistant:
    def __init__(self):
        # Initialize Whisper for STT (using openai-whisper)
        print("Loading Whisper model...")
        try:
            import whisper
            self.stt_model = whisper.load_model("base")  # You can use "small", "medium", "large" for better accuracy
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            self.stt_model = None
        
        # Initialize Coqui TTS
        print("Loading Coqui TTS model...")
        try:
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        except Exception as e:
            print(f"Warning: Could not load TTS model: {e}")
            self.tts = None
        
        # Audio settings
        self.sample_rate = 16000  # Whisper works best with 16kHz
        self.silence_threshold = 0.02  # Increased threshold for better detection
        self.silence_duration = 2.0  # seconds of silence to stop recording
        self.min_recording_duration = 1.0  # Minimum recording duration in seconds
        
        # TTS settings
        self.tts_speed = 1.5  # Speed multiplier (1.0 = normal, 1.5 = 50% faster, etc.)
        
        # Recording control
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
    def record_audio(self):
        """Record audio until silence is detected"""
        print("\n🎤 Listening... (Speak now)")
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())
        
        self.is_recording = True
        audio_data = []
        
        try:
            with sd.InputStream(callback=audio_callback,
                              channels=1,
                              samplerate=self.sample_rate,
                              blocksize=1024):
                
                silent_chunks = 0
                required_silent_chunks = int(self.silence_duration * self.sample_rate / 1024)
                min_chunks = int(self.min_recording_duration * self.sample_rate / 1024)
                chunk_count = 0
                
                print("🔴 Recording... (speak now)")
                
                while self.is_recording:
                    try:
                        chunk = self.audio_queue.get(timeout=5.0)
                        audio_data.append(chunk)
                        
                        # Check for silence
                        rms = np.sqrt(np.mean(chunk**2))
                        
                        # Debug: Show audio level
                        if chunk_count % 10 == 0:
                            print(f"Audio level: {rms:.4f}", end='\r')
                        
                        if rms < self.silence_threshold:
                            silent_chunks += 1
                        else:
                            silent_chunks = 0
                        
                        chunk_count += 1
                        
                        # Stop recording after sufficient silence and minimum recording time
                        if silent_chunks > required_silent_chunks and chunk_count > min_chunks:
                            print("\n⏹️ Recording stopped (silence detected)")
                            self.is_recording = False
                            
                    except queue.Empty:
                        print("\n⏹️ Recording timeout")
                        self.is_recording = False
                        break
                        
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
        
        if audio_data:
            total_audio = np.concatenate(audio_data, axis=0)
            print(f"✅ Recorded {len(total_audio)} samples ({len(total_audio)/self.sample_rate:.2f} seconds)")
            return total_audio
        
        print("❌ No audio data recorded")
        return None
    
    def speech_to_text(self):
        """Convert speech to text using Whisper"""
        if self.stt_model is None:
            print("❌ Error: Whisper model not loaded. Please install: pip install openai-whisper")
            return ""
        
        try:
            print("🎙️ Starting audio recording...")
            audio_data = self.record_audio()
            
            if audio_data is None or len(audio_data) == 0:
                print("❌ No audio recorded")
                return ""
            
            # Convert to the format Whisper expects (1D numpy array of float32)
            audio_np = audio_data.flatten().astype(np.float32)
            
            # Optionally save for debugging (you can remove this later)
            import soundfile as sf
            sf.write("temp_audio.wav", audio_np, self.sample_rate)
            print("💾 Audio saved (for debugging)")
            
            print("🔄 Transcribing with Whisper... (this may take a moment)")
            
            # Transcribe using Whisper - pass numpy array directly (no ffmpeg needed!)
            # This avoids the FileNotFoundError with ffmpeg
            result = self.stt_model.transcribe(
                audio_np,
                language='en',  # Set language to English
                fp16=False,  # Disable FP16 for CPU compatibility
                verbose=False  # Disable verbose output
            )
            
            text = result["text"].strip()
            
            if text:
                print(f"✅ 🗣 You said: {text}")
                return text
            else:
                print("❌ No speech detected in audio")
                return ""
                
        except Exception as e:
            print(f"❌ Speech recognition error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def text_to_speech(self, text):
        """Convert text to speech using Coqui TTS with adjustable speed"""
        if self.tts is None:
            print(f"🤖 Assistant: {text}")
            print("(TTS not available - install with: pip install TTS)")
            return
        
        try:
            print(f"🤖 Assistant: {text}")
            
            # Generate speech
            self.tts.tts_to_file(text=text, file_path="temp_speech.wav")
            
            # Load audio
            import soundfile as sf
            import librosa
            
            audio, original_sample_rate = sf.read("temp_speech.wav")
            
            # Speed up audio using librosa's time_stretch
            # speed > 1.0 makes it faster, < 1.0 makes it slower
            if self.tts_speed != 1.0:
                audio_fast = librosa.effects.time_stretch(audio, rate=self.tts_speed)
            else:
                audio_fast = audio
            
            # Play the sped-up audio
            sd.play(audio_fast, original_sample_rate)
            sd.wait()  # Wait until audio is done playing
            
            # Clean up
            import os
            if os.path.exists("temp_speech.wav"):
                os.remove("temp_speech.wav")
            
        except ImportError as e:
            # Fallback if librosa not installed - play at faster sample rate
            print(f"⚠️ librosa not installed. Using alternative speed method.")
            try:
                import soundfile as sf
                audio, original_sample_rate = sf.read("temp_speech.wav")
                
                # Play at higher sample rate to increase speed
                faster_sample_rate = int(original_sample_rate * self.tts_speed)
                sd.play(audio, faster_sample_rate)
                sd.wait()
                
                import os
                if os.path.exists("temp_speech.wav"):
                    os.remove("temp_speech.wav")
            except Exception as e2:
                print(f"TTS error: {e2}")
                print(f"Assistant: {text}")
                
        except Exception as e:
            print(f"TTS error: {e}")
            # Fallback: just print the text
            print(f"Assistant: {text}")

# -----------------------
# Load Intents (Your existing code)
# -----------------------
with open("intents.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

patterns = []
pattern_texts = []
for intent in data["intents"]:
    tag = intent["tag"]
    for p in intent["patterns"]:
        patterns.append(tag)
        pattern_texts.append(p.lower())

# -----------------------
# Load Sentence-Transformer model
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# Encode intent patterns
# -----------------------
intent_vectors = model.encode(pattern_texts, normalize_embeddings=True).astype("float32")
intent_index = faiss.IndexFlatIP(intent_vectors.shape[1])
intent_index.add(intent_vectors)

# -----------------------
# Load product database
# -----------------------
file_path = "All Products Data.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)

# Flatten all rows into text documents
documents = []
document_categories = []
for sheet_name, df in all_sheets.items():
    for _, row in df.iterrows():
        # Convert row to text
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        documents.append(f"[{sheet_name}] {row_text}")
        document_categories.append(sheet_name)

# -----------------------
# Encode product documents
# -----------------------
doc_vectors = model.encode(documents, normalize_embeddings=True).astype("float32")
doc_index = faiss.IndexFlatIP(doc_vectors.shape[1])
doc_index.add(doc_vectors)

# -----------------------
# Groq Client Setup
# -----------------------

groq_key = os.getenv("GROQ_API_KEY")

client = groq.Groq(api_key=groq_key)
# -----------------------
# Enhanced Conversation Memory (Your existing code)
# -----------------------
class ConversationMemory:
    def __init__(self):
        self.messages = []
        self.user_profile = {
            "name": None,
            "email": None,
            "phone": None,
            "address": None,
            "is_logged_in": False,
            "login_time": None
        }
        self.shopping_cart = []
        self.order_details = {
            "items": [],
            "shipping_address": None,
            "payment_method": None,
            "order_total": 0
        }
        self.user_preferences = {
            "preferred_brands": [],
            "price_range": None,
            "browsed_categories": [],
            "wishlisted_items": []
        }
        self.current_context = {
            "discussing_product": None,
            "last_product_shown": None,
            "last_category": None,
            "last_intent": None
        }
        self.order_history = []
        self.session_start = None
        self.session_end = None
        self.total_interactions = 0
    
    def add_message(self, role, content):
        from datetime import datetime
        self.messages.append({
            "role": role, 
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.total_interactions += 1
    
    def get_conversation_history(self, last_n=None):
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def update_user_profile(self, profile_data):
        self.user_profile.update(profile_data)
    
    def add_to_cart(self, item):
        self.shopping_cart.append(item)
        self.update_context({"last_action": f"Added {item} to cart"})
    
    def remove_from_cart(self, item):
        if item in self.shopping_cart:
            self.shopping_cart.remove(item)
            self.update_context({"last_action": f"Removed {item} from cart"})
    
    def get_cart_contents(self):
        return self.shopping_cart
    
    def update_order_details(self, order_data):
        self.order_details.update(order_data)
    
    def update_preferences(self, preferences):
        for key, value in preferences.items():
            if key in self.user_preferences:
                if isinstance(self.user_preferences[key], list):
                    if isinstance(value, list):
                        self.user_preferences[key].extend([v for v in value if v not in self.user_preferences[key]])
                    elif value not in self.user_preferences[key]:
                        self.user_preferences[key].append(value)
                else:
                    self.user_preferences[key] = value
    
    def update_context(self, context):
        self.current_context.update(context)
    
    def get_full_context_summary(self):
        context_parts = []
        
        if self.user_profile["is_logged_in"]:
            profile_info = f"User logged in"
            if self.user_profile["name"]:
                profile_info += f" as {self.user_profile['name']}"
            context_parts.append(profile_info)
        
        if self.shopping_cart:
            context_parts.append(f"Cart contains: {', '.join(self.shopping_cart)}")
        
        if self.order_details["shipping_address"]:
            context_parts.append(f"Shipping to: {self.order_details['shipping_address']}")
        if self.order_details["payment_method"]:
            context_parts.append(f"Payment method: {self.order_details['payment_method']}")
        
        if self.user_preferences["preferred_brands"]:
            context_parts.append(f"Preferred brands: {', '.join(self.user_preferences['preferred_brands'])}")
        if self.user_preferences["price_range"]:
            context_parts.append(f"Budget: {self.user_preferences['price_range']}")
        if self.user_preferences["browsed_categories"]:
            context_parts.append(f"Browsed categories: {', '.join(set(self.user_preferences['browsed_categories']))}")
        
        if self.current_context.get("discussing_product"):
            context_parts.append(f"Currently discussing: {self.current_context['discussing_product']}")
        if self.current_context.get("last_product_shown"):
            context_parts.append(f"Last product shown: {self.current_context['last_product_shown']}")
        
        if len(self.messages) > 0:
            context_parts.append(f"Total messages in session: {len(self.messages)}")
        
        return "; ".join(context_parts) if context_parts else "New conversation"
    
    def get_context_summary(self):
        return self.get_full_context_summary()
    
    def start_session(self):
        from datetime import datetime
        self.session_start = datetime.now()
    
    def end_session(self):
        from datetime import datetime
        self.session_end = datetime.now()
        self.user_profile["is_logged_in"] = False
    
    def get_session_summary(self):
        summary = {
            "total_messages": len(self.messages),
            "total_interactions": self.total_interactions,
            "cart_items": len(self.shopping_cart),
            "user_profile": self.user_profile,
            "preferences": self.user_preferences,
            "order_details": self.order_details,
            "session_duration": None
        }
        
        if self.session_start and self.session_end:
            duration = self.session_end - self.session_start
            summary["session_duration"] = str(duration)
        
        return summary

conversation_memory = ConversationMemory()

# -----------------------
# Enhanced Intent Classification (Your existing code)
# -----------------------
def keyword_match(user_input):
    text = user_input.lower()
    if any(word in text for word in ["cart", "buy", "purchase", "order"]):
        return "place_order"
    if any(word in text for word in ["update", "change", "edit", "modify"]):
        return "profile_changes"
    if any(word in text for word in ["price", "cost", "how much"]):
        return "price_inquiry"
    if any(word in text for word in ["feature", "spec", "specification"]):
        return "specifications"
    if any(word in text for word in ["recommend", "suggest", "what should"]):
        return "recommendation"
    return None

def classify_intent(user_input, top_k=3):
    rule_intent = keyword_match(user_input)
    if rule_intent:
        return rule_intent

    vec = model.encode([user_input.lower()], normalize_embeddings=True).astype("float32")
    scores, ids = intent_index.search(vec, top_k)
    if scores[0][0] < 0.5:
        return "product_inquiry"
    top_intents = [patterns[i] for i in ids[0]]
    return max(set(top_intents), key=top_intents.count)

# -----------------------
# Enhanced RAG Retrieval (Your existing code)
# -----------------------
def retrieve_docs(user_input, top_k=5, threshold=0.4):
    vec = model.encode([user_input], normalize_embeddings=True).astype("float32")
    scores, ids = doc_index.search(vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if score >= threshold:
            results.append({
                "content": documents[idx],
                "category": document_categories[idx],
                "score": score
            })
    return results

# -----------------------
# Salesman-style Response Generation (Your existing code)
# -----------------------
def generate_salesman_response(user_input, retrieved_docs, intent):
    conversation_history = conversation_memory.get_conversation_history()
    context_summary = conversation_memory.get_full_context_summary()
    
    retrieved_info = "\n".join([doc["content"] for doc in retrieved_docs[:3]])
    
    system_prompt = """You are a friendly and helpful sales assistant at an e-commerce store for visually impaired users. 
Your tone should be warm, conversational, and helpful. Always speak naturally like a human salesman.

Guidelines:
1. Be concise but informative - don't overwhelm with too much information at once
2. Keep the conversation to the point and relevant
3. Use natural language, not robotic lists
4. Give response in minimum words necessary to be clear and helpful
5. Ask follow-up questions to understand user needs better
6. Remember ENTIRE conversation context and user preferences from the beginning of the session
7. If you don't have information, be honest but offer alternatives
8. Always maintain a positive and helpful tone
9. Tailor recommendations based on user preferences and context
10. Maximum length of responses should be limited to 20 words but make sure you give complete information as expected by user if the response is long only then give long response.
11. REMEMBER EVERYTHING from this session - what items are in cart, user's address, payment preferences, and all previous discussions.

FULL SESSION CONTEXT: {context}

Available product information:
{product_info}

User's likely intent: {intent}"""

    messages = [
        {
            "role": "system",
            "content": system_prompt.format(
                context=context_summary,
                product_info=retrieved_info,
                intent=intent
            )
        }
    ]
    
    history_to_include = min(20, len(conversation_history))
    for msg in conversation_history[-history_to_include:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_input})
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )
        
        response = completion.choices[0].message.content
        
        conversation_memory.add_message("user", user_input)
        conversation_memory.add_message("assistant", response)
        
        update_user_preferences(user_input, response)
        
        return response
        
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request right now. Could you please try again?"

def update_user_preferences(user_input, response):
    """Extract potential user preferences and session data from conversation"""
    user_input_lower = user_input.lower()
    
    if any(keyword in user_input_lower for keyword in ["login", "log in", "sign in", "signed in"]):
        conversation_memory.user_profile["is_logged_in"] = True
        conversation_memory.start_session()
    
    name_patterns = ["my name is", "i am", "i'm", "call me"]
    for pattern in name_patterns:
        if pattern in user_input_lower:
            parts = user_input_lower.split(pattern)
            if len(parts) > 1:
                potential_name = parts[1].strip().split()[0] if parts[1].strip() else None
                if potential_name:
                    conversation_memory.update_user_profile({"name": potential_name})
    
    location_keywords = ["address", "location", "ship to", "deliver to", "shipping address"]
    if any(keyword in user_input_lower for keyword in location_keywords):
        for keyword in location_keywords:
            if keyword in user_input_lower:
                parts = user_input_lower.split(keyword)
                if len(parts) > 1:
                    potential_address = parts[1].strip()
                    if potential_address:
                        conversation_memory.update_user_profile({"address": potential_address})
                        conversation_memory.update_order_details({"shipping_address": potential_address})
    
    payment_methods = ["cash on delivery", "cod", "credit card", "debit card", "upi", "net banking", "paypal"]
    for method in payment_methods:
        if method in user_input_lower:
            conversation_memory.update_order_details({"payment_method": method})
    
    if any(keyword in user_input_lower for keyword in ["add to cart", "add this", "buy this", "purchase"]):
        for word in ["iphone", "phone", "laptop", "tv", "tablet", "watch", "headphone"]:
            if word in user_input_lower or word in response.lower():
                if word in user_input_lower:
                    context = user_input_lower
                else:
                    context = response.lower()
                
                if "iphone 14" in context:
                    conversation_memory.add_to_cart("iPhone 14")
                elif "iphone 15" in context:
                    conversation_memory.add_to_cart("iPhone 15")
                elif "iphone" in context:
                    conversation_memory.add_to_cart("iPhone")
                elif word == "phone" and "samsung" in context:
                    conversation_memory.add_to_cart("Samsung Phone")
                else:
                    conversation_memory.add_to_cart(word.capitalize())
                break
    
    if any(keyword in user_input_lower for keyword in ["cart", "in my cart", "what's in", "show cart", "view cart"]):
        conversation_memory.update_context({"last_action": "viewing_cart"})
    
    budget_keywords = ["budget", "price range", "affordable", "expensive", "cheap", "cost", "under", "below"]
    if any(keyword in user_input_lower for keyword in budget_keywords):
        conversation_memory.update_context({"discussing_budget": True})
        import re
        numbers = re.findall(r'\d+', user_input)
        if numbers:
            conversation_memory.update_preferences({"price_range": f"Around {numbers[0]}"})
    
    brands = ["apple", "samsung", "sony", "lg", "nike", "adidas", "mac", "colorbar", "oneplus", "xiaomi", "dell", "hp", "lenovo"]
    mentioned_brands = [brand for brand in brands if brand in user_input_lower or brand in response.lower()]
    if mentioned_brands:
        conversation_memory.update_preferences({"preferred_brands": mentioned_brands})
    
    categories = ["phone", "tv", "laptop", "tablet", "clothing", "footwear", "beauty", "fitness", "watch", "headphone", "speaker"]
    mentioned_categories = [cat for cat in categories if cat in user_input_lower or cat in response.lower()]
    if mentioned_categories:
        conversation_memory.update_preferences({"browsed_categories": mentioned_categories})
        conversation_memory.update_context({"interested_category": mentioned_categories[0]})
    
    if any(keyword in user_input_lower for keyword in ["place order", "checkout", "confirm order", "buy now"]):
        if conversation_memory.shopping_cart:
            order = {
                "items": conversation_memory.shopping_cart.copy(),
                "timestamp": conversation_memory.messages[-1].get("timestamp") if conversation_memory.messages else None
            }
            conversation_memory.order_history.append(order)
            conversation_memory.update_context({"last_action": "placed_order"})
    
    if any(keyword in user_input_lower for keyword in ["feature", "spec", "tell me about", "what about", "how about"]):
        for cat in categories:
            if cat in user_input_lower:
                conversation_memory.update_context({"discussing_product": cat})

# -----------------------
# Main Chat Function (Modified for Voice)
# -----------------------
def chat_with_salesman(user_input):
    intent = classify_intent(user_input)
    retrieved_docs = retrieve_docs(user_input)
    response = generate_salesman_response(user_input, retrieved_docs, intent)
    return response

# -----------------------
# Voice-Based Chat Loop
# -----------------------
def main():
    # Initialize voice assistant
    print("Initializing Voice Assistant...")
    voice = VoiceAssistant()
    
    # Check if voice features are available
    voice_available = voice.stt_model is not None and voice.tts is not None
    
    if voice_available:
        print("\n🎧 Voice Assistant Initialized!")
        print("🤖 Welcome to our store! I'm your friendly shopping assistant!")
        print("🎤 I'll listen to your voice and respond verbally!")
        print("💬 Speak naturally - I'll stop listening after 2 seconds of silence")
        print("❌ Say 'quit', 'exit', or 'goodbye' to end the conversation")
        print("💡 Alternatively, press 'T' to switch to text mode\n")
    else:
        print("\n⚠️ Voice features not available. Running in TEXT MODE.")
        print("🤖 Welcome to our store! I'm your friendly shopping assistant!")
        print("💬 Type your messages below")
        print("❌ Type 'quit', 'exit', or 'logout' to end the conversation\n")
        voice_available = False
    
    # Initialize session
    conversation_memory.start_session()
    conversation_memory.add_message("assistant", "Welcome to our store! I'm here to help you find the perfect products. What are you looking for today?")
    
    # Speak the welcome message if TTS is available
    if voice_available:
        voice.text_to_speech("Welcome to our store! I'm here to help you find the perfect products. What are you looking for today?")
    else:
        print("Chatbot: Welcome to our store! I'm here to help you find the perfect products. What are you looking for today?\n")
    
    # Allow switching between voice and text mode
    use_voice = voice_available
    
    while True:
        try:
            # Get user input
            if use_voice and voice_available:
                # Listen for user input
                user_text = voice.speech_to_text()
                
                if not user_text:
                    voice.text_to_speech("I didn't catch that. Could you please speak again, or press Enter to type instead.")
                    # Allow switching to text mode
                    try:
                        switch = input("Press Enter to type, or press any key + Enter to continue with voice: ").strip()
                        if switch == "":
                            use_voice = False
                            print("\n📝 Switched to TEXT mode. Type your message:\n")
                    except:
                        pass
                    continue
            else:
                # Text input mode
                user_text = input("You: ").strip()
                
                if not user_text:
                    print("Chatbot: I'm here to help! What would you like to know about our products?")
                    continue
            
            # Check for exit commands
            exit_words = ["quit", "exit", "stop", "bye", "goodbye", "logout", "log out"]
            if any(word in user_text.lower() for word in exit_words):
                # End session
                conversation_memory.end_session()
                session_summary = conversation_memory.get_session_summary()
                
                print(f"\n{'='*50}")
                print("SESSION SUMMARY")
                print(f"{'='*50}")
                print(f"Total interactions: {session_summary['total_interactions']}")
                print(f"Total messages: {session_summary['total_messages']}")
                print(f"Items in cart: {session_summary['cart_items']}")
                if session_summary['session_duration']:
                    print(f"Session duration: {session_summary['session_duration']}")
                print(f"{'='*50}\n")
                
                goodbye_msg = "Thank you for visiting! Feel free to come back if you need any help. Have a great day!"
                
                if use_voice and voice_available:
                    voice.text_to_speech(goodbye_msg)
                else:
                    print(f"Chatbot: {goodbye_msg}")
                break
            
            # Get response from salesman chatbot
            response = chat_with_salesman(user_text)
            
            # Output response
            if use_voice and voice_available:
                voice.text_to_speech(response)
            else:
                print(f"Chatbot: {response}\n")
            
        except KeyboardInterrupt:
            conversation_memory.end_session()
            goodbye_msg = "Thanks for stopping by! Hope to see you again soon!"
            
            if use_voice and voice_available:
                voice.text_to_speech(goodbye_msg)
            else:
                print(f"\nChatbot: {goodbye_msg}")
            break
        except Exception as e:
            error_msg = "I apologize, but I'm experiencing some technical difficulties. Please try again in a moment."
            print(f"Error: {e}")
            
            if use_voice and voice_available:
                voice.text_to_speech(error_msg)
            else:
                print(f"Chatbot: {error_msg}")

if __name__ == "__main__":
    main()