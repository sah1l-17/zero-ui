# Zero-UI: Voice-Based E-Commerce Shopping Assistant 🎙️🛍️

A conversational AI-powered shopping assistant designed for visually impaired users, combining voice interaction with advanced natural language understanding to provide a seamless e-commerce experience.

## 🌟 Features

### Voice Interface
- **Speech-to-Text (STT)**: OpenAI Whisper for accurate voice recognition
- **Text-to-Speech (TTS)**: Coqui TTS for natural-sounding voice responses
- **Adaptive Listening**: Automatic silence detection to stop recording
- **Speed Control**: Adjustable TTS playback speed for better comprehension

### Intelligent Shopping Assistant
- **Intent Classification**: ML-powered intent recognition using Sentence Transformers
- **Product Search**: Semantic search through product database using FAISS
- **Conversation Memory**: Maintains session context and user preferences
- **Smart Recommendations**: Personalized product suggestions based on user behavior
- **Multi-turn Dialogue**: Context-aware responses that understand full conversation history

### E-Commerce Features
- Shopping cart management (add/remove items)
- Order placement and tracking
- User profile management
- Order history and receipts
- Payment method selection
- Shipping address management
- User preference tracking

### Accessibility
- Designed specifically for visually impaired users
- Natural language interface (no complex commands)
- Audio-only interaction mode
- Fallback to text mode if needed
- Session memory for returning users

## 🏗️ Architecture

### Core Components

1. **Voice Processing** (`VoiceAssistant` class)
   - Audio recording with silence detection
   - Speech recognition using Whisper
   - Speech synthesis using Coqui TTS

2. **Intent Classification**
   - Keyword-based rule matching
   - Semantic similarity using Sentence Transformers
   - FAISS vector indexing for fast retrieval

3. **Document Retrieval**
   - RAG (Retrieval Augmented Generation) for product information
   - Vector-based similarity search
   - Multi-sheet Excel product database support

4. **Conversation Management** (`ConversationMemory` class)
   - Session tracking and context
   - User profile storage
   - Shopping cart management
   - Order history
   - Preference learning

5. **Response Generation**
   - Groq LLM integration (Llama 3.3 70B)
   - Context-aware prompting
   - Salesman-style conversational responses

## 📋 Requirements

```
sentence-transformers
faiss-cpu
pandas
numpy
groq
openpyxl
torch
transformers
tqdm
scikit-learn
openai-whisper
sounddevice
soundfile
TTS
python-dotenv
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sah1l-17/zero-ui.git
cd zero-ui
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from [https://console.groq.com](https://console.groq.com)

### 4. Download Required Models
The application will automatically download:
- Whisper model (on first run)
- Coqui TTS model (on first run)
- Sentence Transformer model (on first run)

## 📁 Project Structure

```
zero-ui/
├── main.py                          # Main application file
├── intents.json                     # Intent definitions for chatbot
├── requirements.txt                 # Python dependencies
├── All Products Data.xlsx           # Product database (required)
├── README.md                        # This file
└── .env                            # Environment variables (create this)
```

## 🎯 Usage

### Running the Application

```bash
python main.py
```

### Interaction Modes

#### Voice Mode (Default)
- The assistant will listen for your voice input
- Speak naturally - it stops listening after 2 seconds of silence
- Listen to audio responses from the assistant
- Press 'T' to switch to text mode if needed

#### Text Mode
- Type your queries and questions
- Useful when microphone isn't available
- Same intelligent responses as voice mode

### Example Interactions

**Finding Products**
```
You: Show me laptops under 50000
Assistant: I found several laptops within your budget...
```

**Shopping**
```
You: Add iPhone 15 to my cart
Assistant: Added iPhone 15 to your cart. Would you like...
```

**Checking Cart**
```
You: What's in my cart?
Assistant: Your cart contains: iPhone 15, and MacBook Pro...
```

**Placing Order**
```
You: Place my order
Assistant: Great! Let me confirm your order details...
```

## 🧠 Intent Types

The system recognizes 12+ intent categories:

- **sign_up**: Create new account
- **login**: Log into existing account
- **profile_changes**: Update personal information
- **add_to_cart**: Add items to shopping cart
- **remove_from_cart**: Remove items from cart
- **cart_detail**: View cart contents
- **place_order**: Checkout and place order
- **order_confirmation**: Get order confirmation
- **cancel_order**: Cancel existing order
- **track_order**: Track shipment status
- **download_invoice**: Get receipt/invoice
- **order_history**: View past purchases
- **logout**: End session

## 📊 Technical Details

### ML Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| STT | OpenAI Whisper (base) | Speech recognition |
| TTS | Coqui Tacotron2-DDC | Speech synthesis |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Semantic understanding |
| Vector DB | FAISS (IndexFlatIP) | Fast similarity search |
| LLM | Groq Llama-3.3-70B | Response generation |

### Session Memory

The assistant maintains detailed session information:
- User profile (name, email, phone, address)
- Shopping cart state
- Order details
- User preferences (brands, price range, categories)
- Full conversation history
- Session duration and interaction count

### Performance Optimizations

- **FAISS Indexing**: O(1) document retrieval for 1000+ products
- **Sentence Transformer Caching**: Reusable embeddings
- **Batch Processing**: Efficient batch encoding of text
- **Token Optimization**: Limited conversation history in prompts

## 🔧 Configuration

### Audio Settings (in `VoiceAssistant.__init__`)
```python
self.sample_rate = 16000           # Audio sampling rate
self.silence_threshold = 0.02      # RMS threshold for silence detection
self.silence_duration = 2.0        # Seconds of silence to stop recording
self.min_recording_duration = 1.0  # Minimum recording length
self.tts_speed = 1.5              # Speech playback speed multiplier
```

### Retrieval Settings (in `retrieve_docs`)
```python
top_k = 5          # Number of documents to retrieve
threshold = 0.4    # Similarity score threshold
```

### LLM Settings (in `generate_salesman_response`)
```python
temperature = 0.7  # Response creativity
max_tokens = 500   # Maximum response length
top_p = 0.9       # Nucleus sampling parameter
```

## 📦 Required Data Files

### Product Database
- **File**: `All Products Data.xlsx`
- **Format**: Multi-sheet Excel workbook
- **Content**: Product information (name, price, features, etc.)
- **Note**: Each sheet represents a product category

### Intent Definitions
- **File**: `intents.json`
- **Format**: JSON with intent tags and patterns
- **Content**: Pre-defined user intents and example utterances

## 🎓 How It Works

### 1. User Input Processing
```
Voice/Text Input → Silence Detection (if voice) → Text Normalization
```

### 2. Intent Classification
```
User Input → Keyword Matching → Semantic Similarity Search → Intent Tag
```

### 3. Context Retrieval
```
Intent + User Input → FAISS Search → Top-K Documents → Ranking by Score
```

### 4. Response Generation
```
Intent + Retrieved Docs + Conversation History → Groq LLM → Response
```

### 5. Output & Memory Update
```
Response → Text-to-Speech (if voice) → Update Session Memory
```

## 🐛 Troubleshooting

### Microphone Not Working
- Ensure your microphone is connected and working
- Check system audio settings
- Try switching to text mode with 'T'

### Poor Speech Recognition
- Speak clearly and at normal pace
- Reduce background noise
- Check microphone volume

### Models Won't Load
- Ensure all packages in `requirements.txt` are installed
- Check internet connection for model downloads
- Verify sufficient disk space (models ~2GB total)

### Product Database Not Found
- Ensure `All Products Data.xlsx` is in the project root
- Check file name spelling and format
- File should have multiple sheets for different categories

### API Errors
- Verify `GROQ_API_KEY` is set in `.env` file
- Check API quota and rate limits
- Test API connection with curl

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is part of a final year project. Check the repository for license details.

## 👤 Author

**Sahil** - [@sah1l-17](https://github.com/sah1l-17)

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Coqui AI for TTS
- Sentence Transformers for embeddings
- FAISS for vector search
- Groq for LLM API
- The accessibility community for inspiration

## 📞 Support

For issues, questions, or suggestions, please open an issue on the [GitHub repository](https://github.com/sah1l-17/zero-ui).

---
