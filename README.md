# 📄 DocuChat AI 🤖

**Smart Document Search Engine + Conversational AI Chatbot**

DocuChat AI is an intelligent document processing and conversational AI application that combines semantic search capabilities with advanced RAG (Retrieval-Augmented Generation) technology. Upload documents or provide URLs, and chat with your content using natural language!

## ✨ Features

### 🔍 **Semantic Search**
- **Multi-format Support**: PDF, DOCX, PPTX, TXT files
- **OCR Integration**: Extract text from images within PDFs and web pages
- **Intelligent Highlighting**: Keywords highlighted in search results
- **Similarity Scoring**: Results ranked by relevance percentage
- **Multi-URL Processing**: Load content from multiple websites simultaneously

### 💬 **Conversational AI Chat**
- **Context-Aware Conversations**: Maintains chat history and context
- **RAG Technology**: Answers based on uploaded content
- **Fixed Chat Interface**: Modern floating chat input at bottom
- **Message History**: Persistent conversation memory
- **Real-time Responses**: Powered by Groq's fast LLM inference

### 🎨 **Modern UI/UX**
- **Streamlit Interface**: Clean, intuitive web application
- **Tabbed Navigation**: Separate tabs for search and chat
- **Responsive Design**: Works on desktop and mobile
- **Beautiful Chat Styling**: Gradient user messages, clean assistant responses
- **Progress Indicators**: Loading states and success/error messages

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama-3.3-70b-versatile)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain
- **OCR**: Tesseract + pdf2image
- **Web Scraping**: BeautifulSoup + requests

## 📦 Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR installed on your system

### 1. Clone the Repository
```bash
git clone https://github.com/vishwajeet-barade/Smart-Document-Search-Engine-Conversational-AI-Chatbot.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH environment variable
```

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=DocuChat-AI
```

## 🚀 Usage

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Upload Documents
- Use the sidebar to upload PDF, DOCX, PPTX, or TXT files
- Or enter multiple website URLs (one per line)

### 3. Semantic Search
- Go to the "🔎 Semantic Search" tab
- Enter your search query
- View highlighted results with similarity scores

### 4. Chat with Documents
- Go to the "💬 Chat with Document" tab
- Click "Build Knowledge Base"
- Start chatting with your documents!


### API Keys Required:
1. **Groq API Key**: Get from [Groq Console](https://console.groq.com/)
2. **HuggingFace Token**: Get from [HuggingFace](https://huggingface.co/settings/tokens)
3. **LangSmith API Key** (Optional): For tracing and monitoring

### Supported File Types:
- **Documents**: PDF, DOCX, PPTX, TXT
- **Images**: PNG, JPG, JPEG (via OCR)
- **Web Content**: Any accessible website URL

## 🎯 Use Cases

### 📚 **Education & Research**
- Analyze research papers and academic documents
- Create study guides from textbooks
- Q&A with course materials

### 💼 **Business & Corporate**
- Process contracts and legal documents
- Analyze reports and presentations
- Knowledge base for company policies

### 📖 **Content Analysis**
- Summarize articles and blog posts
- Extract insights from documentation
- Compare information across sources

### 🔬 **Technical Documentation**
- Navigate complex technical manuals
- Find specific implementation details
- Understand code documentation

## 🏗️ Architecture

```
DocuChat AI Architecture
├── Document Ingestion
│   ├── File Upload (PDF, DOCX, PPTX, TXT)
│   ├── URL Processing (Multiple URLs)
│   └── OCR Processing (Images in PDFs/Web)
├── Text Processing
│   ├── Document Chunking
│   ├── Embedding Generation (HuggingFace)
│   └── Vector Storage (FAISS)
├── Search Engine
│   ├── Semantic Search
│   ├── Similarity Scoring
│   └── Result Highlighting
└── Conversational AI
    ├── Context-Aware Retrieval
    ├── RAG Chain (LangChain)
    ├── Chat History Management
    └── Response Generation (Groq)
```

## 🙏 Acknowledgments

- **LangChain**: For the RAG framework
- **Groq**: For fast LLM inference
- **Streamlit**: For the amazing web framework
- **HuggingFace**: For embeddings and transformers
- **FAISS**: For efficient vector similarity search

---

*Transform your documents into intelligent conversations!*
