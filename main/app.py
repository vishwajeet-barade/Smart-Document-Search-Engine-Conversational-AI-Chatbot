import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,
    UnstructuredFileLoader, WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os
import re
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

groq_api_key = os.getenv("GROQ_API_KEY")

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

st.set_page_config(page_title="Smart Document Search + Chatbot", layout="wide")
st.title("📄 Smart Document Search Engine + Chatbot 🤖 - DocuChat AI ")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})

st.sidebar.markdown("### Upload Documents or Provide URLs")
uploaded_files = st.sidebar.file_uploader("Upload Documents", type=["pdf", "docx", "pptx", "txt"], accept_multiple_files=True)
st.sidebar.markdown("### OR Enter Website URLs")
url_input = st.sidebar.text_area("Enter URLs (one per line)", height=100, placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversational_rag_chain" not in st.session_state:
    st.session_state.conversational_rag_chain = None
if "store" not in st.session_state:
    st.session_state.store = {}

ocr_image_dir = "ocr_pages"
os.makedirs(ocr_image_dir, exist_ok=True)

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_images_text_from_pdf(path):
    ocr_texts = []
    images = convert_from_path(path)
    ocr_docs = []
    for idx, image in enumerate(images):
        ocr_text = extract_text_from_image(image)
        if ocr_text.strip():
            image_path = os.path.join(ocr_image_dir, f"page_{idx + 1}.png")
            image.save(image_path)
            ocr_docs.append(Document(
                page_content=ocr_text,
                metadata={"source": "OCR from PDF", "page": idx + 1, "image_path": image_path}
            ))
    return ocr_docs

def load_file(file):
    suffix = file.name.split(".")[-1]
    all_docs = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
        all_docs.extend(extract_images_text_from_pdf(tmp_path))

    elif suffix == "docx":
        loader = UnstructuredWordDocumentLoader(tmp_path)
        all_docs.extend(loader.load())

    elif suffix == "pptx":
        loader = UnstructuredPowerPointLoader(tmp_path)
        all_docs.extend(loader.load())

    else:
        loader = UnstructuredFileLoader(tmp_path)
        all_docs.extend(loader.load())

    return all_docs

def load_url_with_images_ocr(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    all_docs = docs.copy()

    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        image_tags = soup.find_all("img")
        for idx, img in enumerate(image_tags):
            img_url = img.get("src")
            if img_url:
                if img_url.startswith("/"):
                    from urllib.parse import urljoin
                    img_url = urljoin(url, img_url)
                try:
                    img_data = requests.get(img_url, timeout=5).content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        tmp_img.write(img_data)
                        image = Image.open(tmp_img.name)
                        ocr_text = extract_text_from_image(image)
                        if ocr_text.strip():
                            image_path = os.path.join(ocr_image_dir, f"url_img_{idx + 1}.png")
                            image.save(image_path)
                            all_docs.append(Document(
                                page_content=ocr_text,
                                metadata={"source": "OCR from URL Image", "page": idx + 1, "image_path": image_path}
                            ))
                except Exception as e:
                    print(f"OCR failed for image {img_url}: {e}")
    except Exception as e:
        print(f"Error parsing URL: {e}")

    return all_docs

def load_url(url):
    return load_url_with_images_ocr(url)

def preprocess_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

def build_vectorstore(docs):
    texts = preprocess_documents(docs)
    return FAISS.from_documents(texts, embedding_model)

def highlight_keywords(text, query):
    for word in query.split():
        pattern = re.compile(rf'\b({re.escape(word)})\b', flags=re.IGNORECASE)
        text = pattern.sub(r'<mark style="background-color: yellow;">\1</mark>', text)
    return text

def process_urls(urls_text):
    """Process multiple URLs from text input"""
    if not urls_text.strip():
        return []
    
    urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
    all_docs = []
    
    for url in urls:
        try:
            if url.startswith(('http://', 'https://')):
                docs = load_url(url)
                all_docs.extend(docs)
                st.success(f"✅ Successfully loaded: {url}")
            else:
                st.warning(f"⚠️ Invalid URL format: {url}")
        except Exception as e:
            st.error(f"❌ Failed to load {url}: {e}")
    
    return all_docs

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]
    
def setup_conversational_rag_chain(retriever):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

    context_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

tab1, tab2 = st.tabs(["🔎 Semantic Search", "💬 Chat with Document"])

with tab1:
    query = st.text_input("Enter your search query and press Search")
    search_button = st.button("Search")

    if search_button and query:
        all_docs = []
        if uploaded_files:
            for file in uploaded_files:
                docs = load_file(file)
                all_docs.extend(docs)

        if url_input:
            all_docs.extend(process_urls(url_input))

        if not all_docs:
            st.warning("Please upload files or enter a URL.")
        else:
            st.info("🔄 Processing documents and searching...")
            vectorstore = build_vectorstore(all_docs)
            results = vectorstore.similarity_search_with_score(query, k=4)

            st.success("✅ Top Matching Results:")
            for doc, distance in results:
                sim_percent = 1 / (1 + distance) * 100
                page_num = doc.metadata.get("page", "N/A")
                image_path = doc.metadata.get("image_path", None)
                highlighted_text = highlight_keywords(doc.page_content, query)

                st.markdown(f"""
                <div style="background-color:#1e1e1e;padding:15px;border-radius:10px;
                            box-shadow:2px 2px 5px rgba(0,0,0,0.6);margin-bottom:10px;
                            color:#ffffff;font-size:16px;">
                    <p style="margin:0;">{highlighted_text}</p>
                    <div style="text-align:right;font-size:12px;color:#bbbbbb;margin-top:10px;">
                        Similarity: {sim_percent:.2f}% | Page: {page_num}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if image_path:
                    st.image(image_path, caption=f"Image from Page {page_num}", use_column_width=True)

with tab2:
    st.subheader("Chat with your document 🤖")
    
    st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: calc(100% - 300px) !important;
        max-width: 900px !important;
        z-index: 999 !important;
        background: white !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        padding: 10px !important;
    }
    
    .stChatInputContainer > div {
        border-radius: 20px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        margin-bottom: 1rem !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        margin-left: 20% !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: #f8f9fa !important;
        border: 1px solid #e9ecef !important;
        margin-right: 20% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Build Knowledge Base"):
        all_docs = []
        if uploaded_files:
            for file in uploaded_files:
                docs = load_file(file)
                all_docs.extend(docs)

        if url_input:
            all_docs.extend(process_urls(url_input))

        if not all_docs:
            st.warning("Please upload files or enter URLs.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)
            st.session_state.vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()
            
            st.session_state.conversational_rag_chain = setup_conversational_rag_chain(st.session_state.retriever)
            
            st.success("✅ Knowledge base created!")

    chat_container = st.container()
    
    with chat_container:
        if st.session_state.conversational_rag_chain:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("💬 Ask your question about the document...", key="chat_input"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response = st.session_state.conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": "default_session"}},
                        )
                        
                        answer = response["answer"]
                        st.markdown(answer)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        st.rerun()
        else:
            st.info("Please build the knowledge base first to start chatting.")
        
    with st.sidebar:
        st.markdown("### Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.store = {}
            st.success("Chat history cleared!")
            st.rerun()
            
        if st.session_state.chat_history:
            st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
            
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)