# Auto Prompt Optimization FastAPI System

A FastAPI wrapper for the automated prompt optimization system that improves JSON responses with enum validation through systematic testing and human feedback integration.

## 🚀 Features

- **Automated Prompt Optimization**: Systematic optimization of prompts for JSON classification tasks
- **Multi-Model Support**: OpenAI, Anthropic, Groq, and Google models
- **Human Feedback Integration**: LangFuse-powered human annotation and feedback
- **Progress Tracking**: Real-time optimization progress monitoring
- **Rigorous Evaluation**: Train/Dev A/Dev B/Test data splitting with comprehensive metrics
- **RESTful API**: Clean FastAPI interface with comprehensive error handling

## 📋 System Flow

1. **Data Preparation**: Automatic dataset splitting (25% train, 35% dev_a, 20% dev_b, 20% test)
2. **Baseline Evaluation**: Performance measurement on train and dev_a datasets
3. **Intent Analysis**: Understanding optimization requirements using Claude
4. **Optimization Loop**: Iterative prompt improvement using selected optimizers
5. **Dev B Evaluation**: Testing on fresh data with human feedback collection
6. **Final Testing**: Performance validation on held-out test set

## 🏗️ Architecture

```
fastapi_optimization_system/
├── main.py                 # FastAPI application entry point
├── app/
│   ├── routes/            # API endpoints
│   ├── controllers/       # HTTP request handlers
│   ├── usecases/         # Business logic orchestration
│   ├── services/         # Core optimization service
│   ├── models/           # Pydantic models
│   ├── utils/            # Utilities and error handling
│   └── config/           # Configuration management
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🛠️ Installation

1. **Clone the repository and navigate to the FastAPI system:**
   ```bash
   cd fastapi_optimization_system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp env_template.txt .env
   # Edit .env with your API keys and configuration
   ```

4. **Required environment variables:**
   - `ANTHROPIC_API_KEY`: For Claude-powered optimization
   - `GROQ_API_KEY`: For Groq model support
   - `OPENAI_API_KEY`: For OpenAI model support (optional)
   - `LANGFUSE_SECRET_KEY` & `LANGFUSE_PUBLIC_KEY`: For human feedback

## 🚀 Usage

### Starting the Server

```bash
# Development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Main Optimization Endpoint
```http
POST /api/v1/optimize
```

**Request Example:**
```json
{
  "system_prompt": "You are a text classifier. Classify input according to the schema.",
  "user_prompt": "Classify this text for sentiment and intent.",
  "schema": {
    "sentiment": ["positive", "negative", "neutral"],
    "intent": ["complaint", "feedback", "question"]
  },
  "model_config": {
    "provider": "groq",
    "model_name": "llama-3.3-70b-versatile",
    "temperature": 0.2
  },
  "dataset": "classification_dataset",
  "max_iterations": 5,
  "improvement_threshold": 0.05,
  "enable_human_feedback": true
}
```

**Response:**
```json
{
  "request_id": "uuid-here",
  "status": "completed",
  "data_split": {
    "total_samples": 1000,
    "train_samples": 250,
    "dev_a_samples": 350,
    "dev_b_samples": 200,
    "test_samples": 200
  },
  "baseline_prompt": "Original prompt...",
  "baseline_metrics": {...},
  "total_iterations": 3,
  "iterations_history": [...],
  "best_prompt": "Optimized prompt...",
  "best_metrics": {...},
  "improvement_percentage": 15.2,
  "human_feedback_summary": {...},
  "test_metrics": {...},
  "deployment_recommendation": "deploy",
  "total_execution_time": 847.3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Progress Tracking
```http
GET /api/v1/optimize/{request_id}/status
```

#### Helper Endpoints
- `GET /api/v1/health` - Health check
- `GET /api/v1/models` - List supported models
- `GET /api/v1/examples` - Get request examples
- `POST /api/v1/validate` - Validate request format

## 📊 Optimization Process

### 1. Data Splitting
- **Train (25%)**: Baseline evaluation and intent analysis
- **Dev A (35%)**: Candidate prompt evaluation (hidden from optimizers)
- **Dev B (20%)**: Human feedback collection
- **Test (20%)**: Final evaluation

### 2. Optimization Iterations
Each iteration:
1. Generates candidate prompts using selected optimizers
2. Evaluates candidates on Dev A dataset
3. Selects best performing candidate
4. Tests on Dev B for human feedback
5. Updates optimization context

### 3. Human Feedback Integration
- Automatic trace creation in LangFuse
- Guided human annotation interface
- Feedback integration into next iteration
- Quality scoring and insights extraction

## 🔧 Configuration

### Model Providers
- **Groq**: Fast inference, good for development
- **Anthropic**: High quality, recommended for production
- **OpenAI**: Wide model selection
- **Google**: Gemini models support

### Optimization Parameters
- `max_iterations`: Maximum optimization rounds (1-10)
- `improvement_threshold`: Minimum improvement to continue (0.01-0.2)
- `enable_human_feedback`: Whether to collect human feedback

## 📝 Integration with Existing System

This FastAPI wrapper integrates seamlessly with the existing `complete_optimization_system` components:

- **DataManager**: Handles dataset loading and splitting
- **EvaluationEngine**: Performs metric calculations
- **OptimizationController**: Manages optimization strategy
- **HumanFeedbackIntegration**: Handles LangFuse integration

The core business logic remains unchanged - this is purely a FastAPI interface layer.

## 🧪 Example Use Cases

### Text Classification
```json
{
  "schema": {
    "sentiment": ["positive", "negative", "neutral"],
    "intent": ["complaint", "feedback", "question", "request"]
  }
}
```

### E-commerce Product Categorization
```json
{
  "schema": {
    "category": ["electronics", "clothing", "home", "books"],
    "price_range": ["budget", "mid", "premium"],
    "target_audience": ["kids", "adults", "seniors"]
  }
}
```

### Customer Support Routing
```json
{
  "schema": {
    "department": ["technical", "billing", "sales", "general"],
    "urgency": ["low", "medium", "high", "critical"],
    "complexity": ["simple", "moderate", "complex"]
  }
}
```

## 🔍 Monitoring and Debugging

### Progress Tracking
Monitor optimization progress in real-time:
```bash
curl http://localhost:8000/api/v1/optimize/{request_id}/status
```

### Logs
The system provides detailed logging for each optimization step:
- Data loading and splitting
- Baseline evaluation
- Intent analysis
- Optimization iterations
- Human feedback collection
- Final evaluation

### Error Handling
Comprehensive error handling with specific error types:
- `ValidationError`: Request validation issues
- `DatasetError`: Dataset loading/processing problems
- `ModelConfigurationError`: Model setup issues
- `OptimizationProcessError`: Optimization execution problems

## 📈 Performance Considerations

- **Async Processing**: Full async support for concurrent requests
- **Resource Cleanup**: Automatic cleanup of optimization resources
- **Progress Tracking**: Minimal memory footprint for status tracking
- **Error Recovery**: Graceful handling of partial failures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
1. Check the logs for detailed error messages
2. Verify environment variables are set correctly
3. Ensure all dependencies are installed
4. Check API key validity and permissions

## 🔮 Future Enhancements

- [ ] WebSocket support for real-time progress updates
- [ ] Async optimization queuing for high-load scenarios
- [ ] More optimization strategies and algorithms
- [ ] Enhanced metrics and visualization
- [ ] Automated A/B testing capabilities 