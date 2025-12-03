# Ask Your Docs — Agentic RAG 📄🤖

**Ask Your Docs** is an advanced Retrieval-Augmented Generation (RAG) application that transforms how you interact with PDF documents. Built with Streamlit, LangChain, and ChromaDB, it features **Hybrid Retrieval**, **Cross-Encoder Reranking**, and an **Agentic Mode** for multi-step reasoning.

## ✨ Features

- **📄 Multi-Document Support** — Upload and query multiple PDFs simultaneously
- **🤖 Agent Mode** — Multi-step reasoning for complex questions
- **🔀 Hybrid Search** — Combines BM25 keyword search with semantic vector search
- **📊 Smart Reranking** — Cross-encoder model improves result relevance
- **💬 Interactive Chat** — Streaming responses with source citations
- **🧪 Built-in Evaluation** — Test your pipeline with custom test cases
- **🔑 Easy API Key Setup** — Enter your OpenAI key directly in the UI

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ask-your-docs.git
cd ask-your-docs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser. Enter your OpenAI API key when prompted.

### Using Environment Variables (Optional)

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🖥️ Usage

1. **Enter API Key** — On first launch, enter your OpenAI API key
2. **Upload PDFs** — Use the sidebar to upload one or more PDF files
3. **Configure Settings** — Toggle Agent Mode, Hybrid Search, or Reranking
4. **Ask Questions** — Type your question in the chat input
5. **View Sources** — Expand "View Sources" to see citations

## ⚙️ Configuration Options

| Setting | Description |
|---------|-------------|
| **Agent Mode** | Enable multi-step reasoning for complex queries |
| **Hybrid Search** | Combine keyword (BM25) and semantic search |
| **Reranking** | Use cross-encoder to improve result quality |
| **Response Style** | Choose between Concise, Detailed, or ELI5 |

## 🧪 Evaluation

Test your RAG pipeline with custom test cases:

1. Create a JSON file with test cases (see `test_cases_template_1.json`)
2. Upload documents in the sidebar
3. Go to **Evaluation** section
4. Upload your test cases JSON
5. Click **Run Evaluation**

### Test Case Format

```json
[
  {
    "question": "What is the main topic?",
    "expected_answer": "The document discusses...",
    "expected_page": 1,
    "tags": ["overview"]
  }
]
```

## 🚢 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set `OPENAI_API_KEY` in Secrets (optional)
5. Deploy!

### Railway / Render / Heroku

The included `Procfile` works with these platforms:

```bash
# Railway
railway up

# Heroku
heroku create
git push heroku main
```

Set the `OPENAI_API_KEY` environment variable in your platform's dashboard.

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ask-your-docs .
docker run -p 8501:8501 ask-your-docs
```

## 📁 Project Structure

```
ask-your-docs/
├── app.py              # Main Streamlit application
├── config.py           # Configuration and prompts
├── storage.py          # ChromaDB vector store
├── ingestion.py        # PDF loading and chunking
├── retrieval.py        # Basic retrieval logic
├── hybrid_retrieval.py # Hybrid search + reranking
├── generation.py       # LLM response generation
├── agent.py            # RAG agent implementation
├── evaluation.py       # Evaluation framework
├── suggestions.py      # Auto-generate questions
├── utils.py            # Utilities and CSS
├── requirements.txt    # Python dependencies
├── Procfile           # Deployment config
└── .streamlit/
    └── config.toml     # Streamlit theme
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Invalid API key"** | Check your key at platform.openai.com |
| **"PDF appears empty"** | The PDF may be scanned images; OCR needed |
| **Slow first load** | Reranker model downloads on first use (~100MB) |
| **Rate limited** | Wait a moment or upgrade your OpenAI plan |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

Built with ❤️ using [Streamlit](https://streamlit.io), [LangChain](https://langchain.com), and [ChromaDB](https://www.trychroma.com)
