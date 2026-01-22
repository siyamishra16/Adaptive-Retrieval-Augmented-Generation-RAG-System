# Adaptive-Retrieval-Augmented-Generation-RAG-System

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-0.2.16-green.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)

An intelligent document question-answering system that adapts its retrieval strategy based on the type of question being asked. Built with Streamlit, LangChain, Google's Gemini AI, and FAISS vector search.

## 🌟 What Makes This Special

Traditional RAG (Retrieval-Augmented Generation) systems use a one-size-fits-all approach to document retrieval. This system is different because it understands that different types of questions require different retrieval strategies, much like how a skilled researcher would approach various types of inquiries differently.

### The Four Intelligent Strategies

**🎯 Factual Strategy**: When you ask "What was the revenue in Q3 2023?", the system focuses on precision, using higher similarity thresholds to find exact information quickly and accurately.

**🔍 Analytical Strategy**: For questions like "How has the market evolved over the past year?", the system gathers comprehensive information from multiple sources and document sections to provide thorough analysis.

**💭 Opinion Strategy**: When exploring subjective topics such as "What are the different perspectives on remote work?", the system uses Maximal Marginal Relevance to ensure diverse viewpoints are represented.

**🎨 Contextual Strategy**: For personalized questions, the system incorporates user-specific context to tailor responses to individual needs and circumstances.

## 🚀 Key Features

- **Multi-PDF Support**: Upload and query multiple PDF documents simultaneously
- **Intelligent Query Classification**: Automatically determines the best retrieval strategy for each question
- **Adaptive Retrieval**: Four distinct strategies optimize for different question types
- **Response Evaluation**: Built-in scoring system to assess answer quality
- **Interactive Chat Interface**: User-friendly Streamlit interface with conversation history
- **Google Gemini Integration**: Leverages Google's latest AI models for both embeddings and generation
- **Real-time Processing**: Efficient document processing and vector storage with FAISS

## 🛠️ Technical Architecture

### Core Components

The system is built around several key components that work together to create an intelligent document assistant:

**Document Processing Pipeline**: Uses PyPDFLoader to extract text from PDFs, then employs RecursiveCharacterTextSplitter to create semantically meaningful chunks with overlapping boundaries to preserve context.

**Vector Storage System**: Implements FAISS (Facebook AI Similarity Search) for high-performance similarity search across document embeddings, using Google's embedding model for consistent semantic representation.

**Query Classification Engine**: Utilizes Google's Gemini Pro model to analyze query intent and classify questions into one of four strategic categories, enabling adaptive retrieval behavior.

**Adaptive Retrieval System**: Dynamically adjusts retrieval parameters, document count, and selection criteria based on the classified query type to optimize for different information needs.

**Response Generation**: Context-aware response generation that adapts its prompting strategy based on query type, ensuring appropriate tone and depth for different question types.

## 📋 Prerequisites

Before setting up the system, ensure you have the following:

- Python 3.8 or higher installed on your system
- A Google AI Studio API key (free tier available)
- At least 4GB of RAM for efficient document processing
- Internet connection for API calls and model downloads

## 🔧 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/intelligent-document-rag.git
cd intelligent-document-rag
```

### Step 2: Create Virtual Environment
Creating a virtual environment ensures package compatibility and prevents conflicts with other Python projects:

```bash
# Create virtual environment
conda create -p venv python=3.11 -y

# Activate virtual environment
# On Windows:
conda activate venv/
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```


### Step 4: Get Google AI Studio API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for use in the application

## 🎯 Usage

### Starting the Application

Launch the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the System

**Step 1: Configure API Key**
Enter your Google AI Studio API key in the sidebar. The system requires this for both embedding generation and response creation.

**Step 2: Upload Documents**
Upload one or more PDF documents using the file uploader. The system supports multiple files and will create a unified knowledge base from all uploaded documents.

**Step 3: Process Documents**
Click "Process Documents" to extract text, create chunks, and generate embeddings. This step typically takes 30-60 seconds depending on document size.

**Step 4: Ask Questions**
Enter your questions in natural language. The system will automatically classify your query and apply the appropriate retrieval strategy.

**Step 5: Review Results**
The system displays the generated response along with the query classification, evaluation score (if provided), and details about retrieved documents.

## 🔍 Understanding the Query Types

### Factual Questions
These seek specific, concrete information:
- "What was the company's revenue in 2023?"
- "When was the contract signed?"
- "What is the definition of machine learning?"

The system uses precision-focused retrieval with higher similarity thresholds to find exact answers quickly.

### Analytical Questions
These require comprehensive understanding and analysis:
- "How has the market changed over the past five years?"
- "What are the main challenges facing the industry?"
- "Compare the performance of different strategies."

The system retrieves more documents and diversifies sources to provide thorough analysis.

### Opinion Questions
These explore subjective topics and multiple perspectives:
- "What are the different views on remote work policies?
