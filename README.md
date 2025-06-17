# Auto Prompt Optimization System

A comprehensive system for automated prompt optimization using multiple LLM providers with human feedback integration and rigorous evaluation metrics.

## 🚀 Overview

This system automatically optimizes prompts for JSON classification tasks through:
- **Multi-Model Support**: OpenAI, Anthropic, Groq, and Google models
- **Human Feedback Integration**: LangFuse-powered annotation and feedback collection
- **Rigorous Evaluation**: Train/Dev A/Dev B/Test data splitting with comprehensive metrics
- **FastAPI Backend**: RESTful API with real-time progress tracking
- **Streamlit Frontend**: User-friendly web interface

## 📁 Project Structure

```
PERSONAL_PROJECT/
├── Backend/                          # Core optimization system
│   ├── prompt_optimizer/            # Core optimization logic
│   │   ├── core/                   # Main optimization components
│   │   ├── models/                 # Data models and types
│   │   ├── optimizers/             # Optimization strategies
│   │   └── utils/                  # Utilities and helpers
│   ├── fastapi_optimization_system/ # FastAPI web service
│   │   ├── app/                    # FastAPI application
│   │   ├── main.py                 # FastAPI entry point
│   │   └── requirements.txt        # FastAPI dependencies
│   └── requirements.txt             # Backend dependencies
├── Frontend/                        # Streamlit web interface
│   ├── app.py                      # Streamlit application
│   └── requirements.txt            # Frontend dependencies
└── README.md                       # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- MongoDB (for token usage logging)
- Git

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PERSONAL_PROJECT
```

### 2. Backend Setup

#### Install Dependencies

```bash
cd Backend
pip install -r requirements.txt
```

#### Environment Configuration

1. Create environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```bash
# LLM API KEYS (at least one required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# LANGFUSE CONFIGURATION (for human feedback)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROJECT_ID=your_project_id_here

# DATABASE (optional - for logging)
MONGODB_URL=mongodb://localhost:27017/
MONGODB_DB_NAME=personal_project_log_usage
```

#### Start FastAPI Server

```bash
cd Backend/fastapi_optimization_system
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

### 3. Frontend Setup

#### Install Dependencies

```bash
cd Frontend
```

#### Start Streamlit App

```bash
streamlit run app.py
```

The web interface will be available at: http://localhost:8501

#### LangFuse (Human Feedback)
1. Visit https://cloud.langfuse.com/
2. Create project
3. Get public/secret keys from settings
4. Add to `.env` as `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`