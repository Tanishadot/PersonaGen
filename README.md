# PersonaGen: Emotionally Intelligent Chatbot with Feedback Loop

An advanced AI chatbot system that combines emotion detection, empathy scoring, and adaptive response generation to create emotionally intelligent conversations.

## 🌟 Features

- **Emotion Detection**: Uses RoBERTa-based emotion classification to detect user emotions
- **Empathy Scoring**: Multi-factor empathy scoring system with configurable thresholds
- **Feedback Loop**: Automatic response regeneration when empathy scores are below threshold
- **Multiple Personas**: Choose between Therapist, Friendly, or Professional communication styles
- **Memory System**: Maintains conversation context for more personalized responses
- **Real-time Metrics**: Track empathy scores, emotion patterns, and response diversity
- **Modern UI**: Clean, responsive React interface with real-time updates

## 🏗️ System Architecture

```
User Input → GPT-2 Generator → RoBERTa Emotion Classifier → Empathy Score Calculation
      ↓
If score < threshold → Refine Prompt → Regenerate Response (max 3 retries)
      ↓
Final Output with emotion detection and empathy metrics
```

## 📋 Requirements

- Python 3.8+
- Node.js 14+
- 8GB+ RAM (for model loading)
- Internet connection (for initial model download)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd C:\Users\User\CascadeProjects\PersonaGen
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the frontend development server
npm start
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
PersonaGen/
├── backend/
│   ├── main.py              # FastAPI application and endpoints
│   ├── models.py            # AI models (GPT-2, RoBERTa) and dataset loading
│   ├── utils.py             # Empathy scoring, evaluation metrics, memory management
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html       # HTML template
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Styling
│   │   ├── index.js         # React entry point
│   │   └── index.css        # Global styles
│   └── package.json         # Node.js dependencies
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables (Required for GPT-4)

Create a `.env` file in the project root:

```env
# OpenAI API Configuration (Required for GPT-4)
OPENAI_API_KEY=your_openai_api_key_here

# Generation Configuration
USE_GPT4=true
GPT4_MODEL=gpt-4o-mini

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Performance Configuration
MAX_RETRIES=3
EMPATHY_THRESHOLD=0.7
MAX_RESPONSE_LENGTH=150

# Model cache directory (optional)
HF_HOME=./cache

# Log level (optional)
LOG_LEVEL=INFO
```

### Getting OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key and add it to your `.env` file

### Model Selection

- **GPT-4** (Default): Higher quality responses, requires API key
- **GPT-2**: Local model, no API key required but lower quality

To use GPT-2 instead, set `USE_GPT4=false` in your `.env` file.

### Empathy Threshold Adjustment

The empathy threshold can be adjusted in the frontend using the slider (0.1 - 1.0):

- **0.8-1.0**: Very strict empathy requirements
- **0.6-0.8**: Balanced empathy and response quality
- **0.4-0.6**: More lenient empathy requirements
- **0.1-0.4**: Minimal empathy filtering

## 🎯 Available Personas

1. **Therapist**: Compassionate and empathetic, focused on emotional support
2. **Friendly**: Warm and cheerful, positive and encouraging
3. **Professional**: Clear, concise, and informative

## 📊 API Endpoints

### Chat Endpoint
```http
POST /chat
Content-Type: application/json

{
  "message": "I'm feeling really stressed about work",
  "persona": "therapist",
  "empathy_threshold": 0.6,
  "max_retries": 3
}
```

### Response Format
```json
{
  "response": "I understand that work stress can be overwhelming...",
  "detected_emotion": "sadness",
  "emotion_confidence": 0.85,
  "empathy_score": 0.78,
  "retry_count": 1,
  "persona": "therapist",
  "context_used": true,
  "bleu_score": 0.65
}
```

### Other Endpoints

- `GET /personas` - List available personas
- `GET /metrics` - Conversation metrics and statistics
- `GET /health` - System health check
- `DELETE /memory` - Clear conversation memory

## 🧠 Models and Datasets

### Pre-trained Models Used

1. **GPT-2**: `gpt2` - Response generation
2. **RoBERTa Emotion**: `j-hartmann/emotion-english-distilroberta-base` - Emotion classification

### Dataset

- **EmpatheticDialogues**: Automatically downloaded from HuggingFace on first run
- Used for training context and empathy pattern understanding

## 📈 Evaluation Metrics

The system implements multiple evaluation metrics:

- **Empathy Score**: Multi-factor scoring (0.0 - 1.0)
  - Empathy language usage (30%)
  - Emotional acknowledgment (25%)
  - Response length/completeness (20%)
  - Supportive language (15%)
  - Personalization (10%)

- **BLEU Score**: Response quality assessment
- **Response Diversity**: Avoids repetitive responses
- **Emotion Detection Confidence**: Model confidence in emotion classification

## 🔄 Feedback Loop Mechanism

1. Generate initial response
2. Detect emotion in user input
3. Calculate empathy score
4. If score < threshold:
   - Refine prompt with empathy instructions
   - Regenerate response
   - Repeat (max 3 retries)
5. Return best response with metrics

## 🛠️ Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check internet connection for initial download
   - Ensure sufficient RAM (8GB+ recommended)
   - Check virtual environment activation

2. **Frontend not connecting**:
   - Ensure backend is running on port 8000
   - Check for CORS issues
   - Verify proxy configuration in package.json

3. **Low empathy scores**:
   - Adjust empathy threshold downward
   - Try different persona
   - Check if context is being used

4. **Memory issues**:
   - Clear conversation memory using the button
   - Restart backend server
   - Check system resources

### Performance Optimization

- Use GPU acceleration if available (CUDA)
- Limit conversation history in memory
- Adjust model parameters for faster inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational and research purposes. Please ensure compliance with the licenses of the pre-trained models used.

## 🙏 Acknowledgments

- Hugging Face for the transformers library and datasets
- OpenAI for GPT-2
- RoBERTa emotion classification model authors
- EmpatheticDialogues dataset creators

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check console logs for error messages
4. Verify all dependencies are correctly installed

---

**Note**: The first run may take several minutes as models are downloaded and cached. Subsequent runs will be much faster.
