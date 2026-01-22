# import streamlit as st
# import os
# import tempfile
# from typing import List, Dict, Any
# import google.generativeai as genai
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.schema import Document
# import numpy as np
# from datetime import datetime

# # Configure the page
# st.set_page_config(
#     page_title="Adaptive RAG System",
#     page_icon="📚",
#     layout="wide"
# )

# # Initialize session state
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = []
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# class AdaptiveRAGSystem:
#     def __init__(self, gemini_api_key: str):
#         """
#         Initialize the Adaptive RAG system with Gemini API.
#         This sets up our core components for document processing and query handling.
#         """
#         # Configure Gemini API - this is our language model for generating responses
#         genai.configure(api_key=gemini_api_key)
#         self.model = genai.GenerativeModel('gemini-pro')
        
#         # Initialize embeddings model - this converts text to numerical vectors
#         # We use a pre-trained model that's good at understanding semantic meaning
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/gemini-embedding-exp-03-07",
#             google_api_key=gemini_api_key
#         )
        
#         # Text splitter for breaking documents into manageable chunks
#         # This is crucial because we need to fit information into the model's context window
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,      # Each chunk will be about 1000 characters
#             chunk_overlap=200,    # 200 character overlap helps maintain context
#             length_function=len,
#             separators=["\n\n", "\n", " ", ""]  # Split on these boundaries in order
#         )
    
#     def process_document(self, pdf_file) -> List[Document]:
#         """
#         Process a single PDF file and extract text chunks.
#         This is the first step in our RAG pipeline - getting clean, structured text.
#         """
#         # Save uploaded file to temporary location
#         # Streamlit gives us file objects, but PyPDFLoader needs file paths
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(pdf_file.getvalue())
#             tmp_file_path = tmp_file.name
        
#         try:
#             # Load the PDF and extract text
#             loader = PyPDFLoader(tmp_file_path)
#             documents = loader.load()
            
#             # Split documents into chunks
#             # This is important because large documents need to be broken down
#             # for effective retrieval and to fit within model limits
#             chunks = self.text_splitter.split_documents(documents)
            
#             # Add metadata to help with retrieval
#             for i, chunk in enumerate(chunks):
#                 chunk.metadata.update({
#                     'source_file': pdf_file.name,
#                     'chunk_id': i,
#                     'total_chunks': len(chunks)
#                 })
            
#             return chunks
            
#         finally:
#             # Clean up temporary file
#             os.unlink(tmp_file_path)
    
#     def classify_query(self, query: str) -> str:
#         """
#         Classify the query into one of four types to determine retrieval strategy.
#         This is the 'adaptive' part - we adjust our approach based on what's being asked.
#         """
#         # Create a prompt that helps the model understand the classification task
#         classification_prompt = f"""
#         Classify the following query into one of these four categories:
        
#         1. FACTUAL: Questions asking for specific facts, numbers, dates, definitions, or concrete information
#         2. ANALYTICAL: Questions requiring comprehensive analysis, comparison, or exploration of multiple aspects
#         3. OPINION: Questions seeking different viewpoints, subjective opinions, or evaluative judgments
#         4. CONTEXTUAL: Questions that depend on user-specific context or personal circumstances
        
#         Query: {query}
        
#         Respond with only the category name (FACTUAL, ANALYTICAL, OPINION, or CONTEXTUAL).
#         """
        
#         try:
#             response = self.model.generate_content(classification_prompt)
#             query_type = response.text.strip().upper()
            
#             # Validate the response and provide fallback
#             if query_type not in ['FACTUAL', 'ANALYTICAL', 'OPINION', 'CONTEXTUAL']:
#                 return 'FACTUAL'  # Default fallback
            
#             return query_type
#         except Exception as e:
#             st.error(f"Error in query classification: {e}")
#             return 'FACTUAL'  # Fallback to factual for safety
    
#     def adaptive_retrieval(self, query: str, vector_store: FAISS, k: int = 4, 
#                           user_context: str = None) -> List[Document]:
#         """
#         Retrieve documents using different strategies based on query type.
#         This is where the adaptive magic happens - we adjust our retrieval approach.
#         """
#         if not vector_store:
#             return []
        
#         # First, classify the query to understand what strategy to use
#         query_type = self.classify_query(query)
        
#         # Now apply different retrieval strategies based on the query type
#         if query_type == 'FACTUAL':
#             # For factual queries, we want precise, specific information
#             # Use higher similarity threshold to get the most relevant chunks
#             docs = vector_store.similarity_search_with_score(query, k=k)
#             # Filter for higher relevance scores (lower distance means higher similarity)
#             filtered_docs = [doc for doc, score in docs if score < 0.5]
#             return filtered_docs[:k] if filtered_docs else [doc for doc, _ in docs[:k]]
        
#         elif query_type == 'ANALYTICAL':
#             # For analytical queries, we want comprehensive coverage
#             # Retrieve more documents to get different perspectives
#             docs = vector_store.similarity_search(query, k=k*2)  # Get more docs
#             # Try to get diverse sources by checking metadata
#             diverse_docs = self._diversify_sources(docs)
#             return diverse_docs[:k]
        
#         elif query_type == 'OPINION':
#             # For opinion queries, we want diverse viewpoints
#             # Use MMR (Maximal Marginal Relevance) to get diverse results
#             try:
#                 docs = vector_store.max_marginal_relevance_search(
#                     query, k=k, fetch_k=k*3  # Fetch more, then diversify
#                 )
#                 return docs
#             except:
#                 # Fallback to regular similarity search
#                 return vector_store.similarity_search(query, k=k)
        
#         elif query_type == 'CONTEXTUAL':
#             # For contextual queries, incorporate user context if available
#             enhanced_query = query
#             if user_context:
#                 enhanced_query = f"Context: {user_context}\nQuery: {query}"
            
#             docs = vector_store.similarity_search(enhanced_query, k=k)
#             return docs
        
#         else:
#             # Default fallback
#             return vector_store.similarity_search(query, k=k)
    
#     def _diversify_sources(self, documents: List[Document]) -> List[Document]:
#         """
#         Helper method to diversify document sources for better coverage.
#         This ensures we don't just get chunks from one part of one document.
#         """
#         source_groups = {}
        
#         # Group documents by source file
#         for doc in documents:
#             source = doc.metadata.get('source_file', 'unknown')
#             if source not in source_groups:
#                 source_groups[source] = []
#             source_groups[source].append(doc)
        
#         # Interleave documents from different sources
#         diversified = []
#         max_per_source = max(1, len(documents) // len(source_groups))
        
#         for source, docs in source_groups.items():
#             diversified.extend(docs[:max_per_source])
        
#         return diversified
    
#     def generate_response(self, query: str, retrieved_docs: List[Document], 
#                          query_type: str) -> str:
#         """
#         Generate a response using the retrieved documents and query type.
#         This is where we craft the final answer using our retrieved context.
#         """
#         if not retrieved_docs:
#             return "I couldn't find relevant information in the uploaded documents to answer your question."
        
#         # Prepare the context from retrieved documents
#         context = "\n\n".join([
#             f"Source: {doc.metadata.get('source_file', 'Unknown')}\n{doc.page_content}"
#             for doc in retrieved_docs
#         ])
        
#         # Create different prompts based on query type
#         if query_type == 'FACTUAL':
#             system_prompt = """You are a precise fact-finder. Provide accurate, specific information based on the given context. 
#             Focus on concrete facts, numbers, dates, and definitions. Be concise and direct."""
        
#         elif query_type == 'ANALYTICAL':
#             system_prompt = """You are a thorough analyst. Provide comprehensive analysis based on the given context. 
#             Explore different aspects, make connections, and provide detailed explanations. Consider multiple perspectives."""
        
#         elif query_type == 'OPINION':
#             system_prompt = """You are a balanced perspective-gatherer. Present different viewpoints and opinions from the context. 
#             Acknowledge varying perspectives and avoid taking a single stance. Present the range of opinions fairly."""
        
#         elif query_type == 'CONTEXTUAL':
#             system_prompt = """You are a context-aware assistant. Tailor your response to the specific context and user needs. 
#             Consider the user's situation and provide relevant, personalized information."""
        
#         else:
#             system_prompt = """You are a helpful assistant. Provide a clear, informative response based on the given context."""
        
#         # Construct the full prompt
#         full_prompt = f"""
#         {system_prompt}
        
#         Context from documents:
#         {context}
        
#         Question: {query}
        
#         Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
#         """
        
#         try:
#             response = self.model.generate_content(full_prompt)
#             return response.text
#         except Exception as e:
#             return f"Error generating response: {str(e)}"
    
#     def evaluate_response(self, query: str, response: str, reference_answer: str = None) -> float:
#         """
#         Evaluate the quality of the response.
#         This helps us understand how well our system is performing.
#         """
#         if not reference_answer:
#             # If no reference answer provided, do a basic quality check
#             evaluation_prompt = f"""
#             Evaluate the following AI response for a given query on a scale of 0 to 1:
            
#             Query: {query}
#             AI Response: {response}
            
#             Consider these factors:
#             - Relevance to the query
#             - Clarity and coherence
#             - Completeness of the answer
#             - Accuracy (if verifiable)
            
#             Provide only a score between 0 and 1 (e.g., 0.75).
#             """
#         else:
#             # If reference answer provided, compare against it
#             evaluation_prompt = f"""
#             You are an intelligent evaluation system. Compare the AI response to the reference answer.
            
#             Query: {query}
#             AI Response: {response}
#             Reference Answer: {reference_answer}
            
#             Scoring criteria:
#             - 1.0: Response is very close to or better than the reference answer
#             - 0.5: Response is partially aligned with the reference answer
#             - 0.0: Response is incorrect or unsatisfactory compared to the reference
            
#             Provide only a score between 0 and 1 (e.g., 0.86).
#             """
        
#         try:
#             response_eval = self.model.generate_content(evaluation_prompt)
#             score_text = response_eval.text.strip()
            
#             # Extract numerical score from response
#             import re
#             score_match = re.search(r'(\d+\.?\d*)', score_text)
#             if score_match:
#                 score = float(score_match.group(1))
#                 return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
#             else:
#                 return 0.5  # Default score if parsing fails
                
#         except Exception as e:
#             st.error(f"Error in evaluation: {e}")
#             return 0.5  # Default score on error

# # Streamlit UI
# def main():
#     st.title("📚 Adaptive RAG System")
#     st.markdown("Upload PDFs and ask questions using intelligent retrieval strategies")
    
#     # Sidebar for configuration
#     with st.sidebar:
#         st.header("Configuration")
        
#         # API Key input
#         gemini_api_key = st.text_input(
#             "Enter your Gemini API Key:",
#             type="password",
#             help="Get your API key from Google AI Studio"
#         )
        
#         if not gemini_api_key:
#             st.warning("Please enter your Gemini API key to continue.")
#             st.stop()
        
#         # File upload
#         st.header("Upload Documents")
#         uploaded_files = st.file_uploader(
#             "Choose PDF files",
#             type="pdf",
#             accept_multiple_files=True,
#             help="Upload one or more PDF documents"
#         )
        
#         # Process uploaded files
#         if uploaded_files:
#             if st.button("Process Documents"):
#                 with st.spinner("Processing documents..."):
#                     # Initialize the RAG system
#                     rag_system = AdaptiveRAGSystem(gemini_api_key)
                    
#                     # Process all uploaded files
#                     all_chunks = []
#                     processed_files = []
                    
#                     for uploaded_file in uploaded_files:
#                         try:
#                             chunks = rag_system.process_document(uploaded_file)
#                             all_chunks.extend(chunks)
#                             processed_files.append(uploaded_file.name)
#                             st.success(f"Processed: {uploaded_file.name}")
#                         except Exception as e:
#                             st.error(f"Error processing {uploaded_file.name}: {e}")
                    
#                     # Create vector store from all chunks
#                     if all_chunks:
#                         try:
#                             vector_store = FAISS.from_documents(
#                                 all_chunks,
#                                 rag_system.embeddings
#                             )
#                             st.session_state.vector_store = vector_store
#                             st.session_state.processed_files = processed_files
#                             st.session_state.rag_system = rag_system
#                             st.success("All documents processed successfully!")
#                         except Exception as e:
#                             st.error(f"Error creating vector store: {e}")
                    
#         # Show processed files
#         if st.session_state.processed_files:
#             st.header("Processed Files")
#             for file_name in st.session_state.processed_files:
#                 st.text(f"📄 {file_name}")
    
#     # Main chat interface
#     if st.session_state.vector_store is not None:
#         st.header("Chat with Your Documents")
        
#         # User context input (optional)
#         user_context = st.text_input(
#             "Additional Context (Optional):",
#             placeholder="Provide any relevant context for your question..."
#         )
        
#         # Query input
#         query = st.text_input(
#             "Ask a question:",
#             placeholder="What would you like to know about your documents?"
#         )
        
#         # Reference answer for evaluation (optional)
#         with st.expander("Evaluation (Optional)"):
#             reference_answer = st.text_area(
#                 "Reference Answer:",
#                 placeholder="Provide a reference answer to evaluate the system's response..."
#             )
        
#         if query:
#             if st.button("Get Answer"):
#                 with st.spinner("Analyzing query and retrieving information..."):
#                     try:
#                         # Get the RAG system from session state
#                         rag_system = st.session_state.rag_system
                        
#                         # Classify the query
#                         query_type = rag_system.classify_query(query)
                        
#                         # Retrieve relevant documents
#                         retrieved_docs = rag_system.adaptive_retrieval(
#                             query,
#                             st.session_state.vector_store,
#                             k=4,
#                             user_context=user_context if user_context else None
#                         )
                        
#                         # Generate response
#                         response = rag_system.generate_response(
#                             query,
#                             retrieved_docs,
#                             query_type
#                         )
                        
#                         # Evaluate response if reference answer is provided
#                         evaluation_score = None
#                         if reference_answer:
#                             evaluation_score = rag_system.evaluate_response(
#                                 query,
#                                 response,
#                                 reference_answer
#                             )
                        
#                         # Display results
#                         st.subheader("Results")
                        
#                         # Show query classification
#                         st.info(f"**Query Type:** {query_type}")
                        
#                         # Show response
#                         st.markdown("**Response:**")
#                         st.write(response)
                        
#                         # Show evaluation score if available
#                         if evaluation_score is not None:
#                             st.markdown(f"**Evaluation Score:** {evaluation_score:.2f}")
                            
#                             # Color code the score
#                             if evaluation_score >= 0.8:
#                                 st.success("Excellent response quality!")
#                             elif evaluation_score >= 0.6:
#                                 st.warning("Good response quality")
#                             else:
#                                 st.error("Response quality needs improvement")
                        
#                         # Show retrieved documents info
#                         with st.expander("Retrieved Documents Details"):
#                             for i, doc in enumerate(retrieved_docs):
#                                 st.write(f"**Document {i+1}:**")
#                                 st.write(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
#                                 st.write(f"Preview: {doc.page_content[:200]}...")
#                                 st.write("---")
                        
#                         # Add to chat history
#                         st.session_state.chat_history.append({
#                             'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                             'query': query,
#                             'query_type': query_type,
#                             'response': response,
#                             'evaluation_score': evaluation_score
#                         })
                        
#                     except Exception as e:
#                         st.error(f"Error processing query: {e}")
        
#         # Show chat history
#         if st.session_state.chat_history:
#             st.header("Chat History")
#             for i, chat in enumerate(reversed(st.session_state.chat_history)):
#                 with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
#                     st.write(f"**Time:** {chat['timestamp']}")
#                     st.write(f"**Query Type:** {chat['query_type']}")
#                     st.write(f"**Question:** {chat['query']}")
#                     st.write(f"**Answer:** {chat['response']}")
#                     if chat['evaluation_score'] is not None:
#                         st.write(f"**Evaluation Score:** {chat['evaluation_score']:.2f}")
    
#     else:
#         st.info("Please upload and process PDF documents to start chatting!")

# if __name__ == "__main__":
#     main()


import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class AdaptiveRAGSystem:
    def __init__(self, gemini_api_key: str):
        """
        Initialize the Adaptive RAG system with Gemini API.
        This sets up our core components for document processing and query handling.
        """
        # Configure Gemini API - this is our language model for generating responses
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize embeddings model - using Google's embedding model
        # This converts text to numerical vectors using Google's embedding service
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        # Text splitter for breaking documents into manageable chunks
        # This is crucial because we need to fit information into the model's context window
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Each chunk will be about 1000 characters
            chunk_overlap=200,    # 200 character overlap helps maintain context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Split on these boundaries in order
        )
    
    def process_document(self, pdf_file) -> List[Document]:
        """
        Process a single PDF file and extract text chunks.
        This is the first step in our RAG pipeline - getting clean, structured text.
        """
        # Save uploaded file to temporary location
        # Streamlit gives us file objects, but PyPDFLoader needs file paths
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Load the PDF and extract text
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents into chunks
            # This is important because large documents need to be broken down
            # for effective retrieval and to fit within model limits
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to help with retrieval
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source_file': pdf_file.name,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
            
            return chunks
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def classify_query(self, query: str) -> str:
        """
        Classify the query into one of four types to determine retrieval strategy.
        This is the 'adaptive' part - we adjust our approach based on what's being asked.
        """
        # Create a prompt that helps the model understand the classification task
        classification_prompt = f"""
        Classify the following query into one of these four categories:
        
        1. FACTUAL: Questions asking for specific facts, numbers, dates, definitions, or concrete information
        2. ANALYTICAL: Questions requiring comprehensive analysis, comparison, or exploration of multiple aspects
        3. OPINION: Questions seeking different viewpoints, subjective opinions, or evaluative judgments
        4. CONTEXTUAL: Questions that depend on user-specific context or personal circumstances
        
        Query: {query}
        
        Respond with only the category name (FACTUAL, ANALYTICAL, OPINION, or CONTEXTUAL).
        """
        
        try:
            response = self.model.generate_content(classification_prompt)
            query_type = response.text.strip().upper()
            
            # Validate the response and provide fallback
            if query_type not in ['FACTUAL', 'ANALYTICAL', 'OPINION', 'CONTEXTUAL']:
                return 'FACTUAL'  # Default fallback
            
            return query_type
        except Exception as e:
            st.error(f"Error in query classification: {e}")
            return 'FACTUAL'  # Fallback to factual for safety
    
    def adaptive_retrieval(self, query: str, vector_store: FAISS, k: int = 4, 
                          user_context: str = None) -> List[Document]:
        """
        Retrieve documents using different strategies based on query type.
        This is where the adaptive magic happens - we adjust our retrieval approach.
        """
        if not vector_store:
            return []
        
        # First, classify the query to understand what strategy to use
        query_type = self.classify_query(query)
        
        # Now apply different retrieval strategies based on the query type
        if query_type == 'FACTUAL':
            # For factual queries, we want precise, specific information
            # Use higher similarity threshold to get the most relevant chunks
            docs = vector_store.similarity_search_with_score(query, k=k)
            # Filter for higher relevance scores (lower distance means higher similarity)
            filtered_docs = [doc for doc, score in docs if score < 0.5]
            return filtered_docs[:k] if filtered_docs else [doc for doc, _ in docs[:k]]
        
        elif query_type == 'ANALYTICAL':
            # For analytical queries, we want comprehensive coverage
            # Retrieve more documents to get different perspectives
            docs = vector_store.similarity_search(query, k=k*2)  # Get more docs
            # Try to get diverse sources by checking metadata
            diverse_docs = self._diversify_sources(docs)
            return diverse_docs[:k]
        
        elif query_type == 'OPINION':
            # For opinion queries, we want diverse viewpoints
            # Use MMR (Maximal Marginal Relevance) to get diverse results
            try:
                docs = vector_store.max_marginal_relevance_search(
                    query, k=k, fetch_k=k*3  # Fetch more, then diversify
                )
                return docs
            except:
                # Fallback to regular similarity search
                return vector_store.similarity_search(query, k=k)
        
        elif query_type == 'CONTEXTUAL':
            # For contextual queries, incorporate user context if available
            enhanced_query = query
            if user_context:
                enhanced_query = f"Context: {user_context}\nQuery: {query}"
            
            docs = vector_store.similarity_search(enhanced_query, k=k)
            return docs
        
        else:
            # Default fallback
            return vector_store.similarity_search(query, k=k)
    
    def _diversify_sources(self, documents: List[Document]) -> List[Document]:
        """
        Helper method to diversify document sources for better coverage.
        This ensures we don't just get chunks from one part of one document.
        """
        source_groups = {}
        
        # Group documents by source file
        for doc in documents:
            source = doc.metadata.get('source_file', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        # Interleave documents from different sources
        diversified = []
        max_per_source = max(1, len(documents) // len(source_groups))
        
        for source, docs in source_groups.items():
            diversified.extend(docs[:max_per_source])
        
        return diversified
    
    def generate_response(self, query: str, retrieved_docs: List[Document], 
                         query_type: str) -> str:
        """
        Generate a response using the retrieved documents and query type.
        This is where we craft the final answer using our retrieved context.
        """
        if not retrieved_docs:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
        
        # Prepare the context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source_file', 'Unknown')}\n{doc.page_content}"
            for doc in retrieved_docs
        ])
        
        # Create different prompts based on query type
        if query_type == 'FACTUAL':
            system_prompt = """You are a precise fact-finder. Provide accurate, specific information based on the given context. 
            Focus on concrete facts, numbers, dates, and definitions. Be concise and direct."""
        
        elif query_type == 'ANALYTICAL':
            system_prompt = """You are a thorough analyst. Provide comprehensive analysis based on the given context. 
            Explore different aspects, make connections, and provide detailed explanations. Consider multiple perspectives."""
        
        elif query_type == 'OPINION':
            system_prompt = """You are a balanced perspective-gatherer. Present different viewpoints and opinions from the context. 
            Acknowledge varying perspectives and avoid taking a single stance. Present the range of opinions fairly."""
        
        elif query_type == 'CONTEXTUAL':
            system_prompt = """You are a context-aware assistant. Tailor your response to the specific context and user needs. 
            Consider the user's situation and provide relevant, personalized information."""
        
        else:
            system_prompt = """You are a helpful assistant. Provide a clear, informative response based on the given context."""
        
        # Construct the full prompt
        full_prompt = f"""
        {system_prompt}
        
        Context from documents:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
        """
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def evaluate_response(self, query: str, response: str, reference_answer: str = None) -> float:
        """
        Evaluate the quality of the response.
        This helps us understand how well our system is performing.
        """
        if not reference_answer:
            # If no reference answer provided, do a basic quality check
            evaluation_prompt = f"""
            Evaluate the following AI response for a given query on a scale of 0 to 1:
            
            Query: {query}
            AI Response: {response}
            
            Consider these factors:
            - Relevance to the query
            - Clarity and coherence
            - Completeness of the answer
            - Accuracy (if verifiable)
            
            Provide only a score between 0 and 1 (e.g., 0.75).
            """
        else:
            # If reference answer provided, compare against it
            evaluation_prompt = f"""
            You are an intelligent evaluation system. Compare the AI response to the reference answer.
            
            Query: {query}
            AI Response: {response}
            Reference Answer: {reference_answer}
            
            Scoring criteria:
            - 1.0: Response is very close to or better than the reference answer
            - 0.5: Response is partially aligned with the reference answer
            - 0.0: Response is incorrect or unsatisfactory compared to the reference
            
            Provide only a score between 0 and 1 (e.g., 0.86).
            """
        
        try:
            response_eval = self.model.generate_content(evaluation_prompt)
            score_text = response_eval.text.strip()
            
            # Extract numerical score from response
            import re
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            else:
                return 0.5  # Default score if parsing fails
                
        except Exception as e:
            st.error(f"Error in evaluation: {e}")
            return 0.5  # Default score on error

# Streamlit UI
def main():
    st.title("📚 Adaptive RAG System")
    st.markdown("Upload PDFs and ask questions using intelligent retrieval strategies")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key to continue.")
            st.stop()
        
        # File upload
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Initialize the RAG system
                    rag_system = AdaptiveRAGSystem(gemini_api_key)
                    
                    # Process all uploaded files
                    all_chunks = []
                    processed_files = []
                    
                    for uploaded_file in uploaded_files:
                        try:
                            chunks = rag_system.process_document(uploaded_file)
                            all_chunks.extend(chunks)
                            processed_files.append(uploaded_file.name)
                            st.success(f"Processed: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    # Create vector store from all chunks
                    if all_chunks:
                        try:
                            vector_store = FAISS.from_documents(
                                all_chunks,
                                rag_system.embeddings
                            )
                            st.session_state.vector_store = vector_store
                            st.session_state.processed_files = processed_files
                            st.session_state.rag_system = rag_system
                            st.success("All documents processed successfully!")
                        except Exception as e:
                            st.error(f"Error creating vector store: {e}")
                    
        # Show processed files
        if st.session_state.processed_files:
            st.header("Processed Files")
            for file_name in st.session_state.processed_files:
                st.text(f"📄 {file_name}")
    
    # Main chat interface
    if st.session_state.vector_store is not None:
        st.header("Chat with Your Documents")
        
        # User context input (optional)
        user_context = st.text_input(
            "Additional Context (Optional):",
            placeholder="Provide any relevant context for your question..."
        )
        
        # Query input
        query = st.text_input(
            "Ask a question:",
            placeholder="What would you like to know about your documents?"
        )
        
        # Reference answer for evaluation (optional)
        with st.expander("Evaluation (Optional)"):
            reference_answer = st.text_area(
                "Reference Answer:",
                placeholder="Provide a reference answer to evaluate the system's response..."
            )
        
        if query:
            if st.button("Get Answer"):
                with st.spinner("Analyzing query and retrieving information..."):
                    try:
                        # Get the RAG system from session state
                        rag_system = st.session_state.rag_system
                        
                        # Classify the query
                        query_type = rag_system.classify_query(query)
                        
                        # Retrieve relevant documents
                        retrieved_docs = rag_system.adaptive_retrieval(
                            query,
                            st.session_state.vector_store,
                            k=4,
                            user_context=user_context if user_context else None
                        )
                        
                        # Generate response
                        response = rag_system.generate_response(
                            query,
                            retrieved_docs,
                            query_type
                        )
                        
                        # Evaluate response if reference answer is provided
                        evaluation_score = None
                        if reference_answer:
                            evaluation_score = rag_system.evaluate_response(
                                query,
                                response,
                                reference_answer
                            )
                        
                        # Display results
                        st.subheader("Results")
                        
                        # Show query classification
                        st.info(f"**Query Type:** {query_type}")
                        
                        # Show response
                        st.markdown("**Response:**")
                        st.write(response)
                        
                        # Show evaluation score if available
                        if evaluation_score is not None:
                            st.markdown(f"**Evaluation Score:** {evaluation_score:.2f}")
                            
                            # Color code the score
                            if evaluation_score >= 0.8:
                                st.success("Excellent response quality!")
                            elif evaluation_score >= 0.6:
                                st.warning("Good response quality")
                            else:
                                st.error("Response quality needs improvement")
                        
                        # Show retrieved documents info
                        with st.expander("Retrieved Documents Details"):
                            for i, doc in enumerate(retrieved_docs):
                                st.write(f"**Document {i+1}:**")
                                st.write(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
                                st.write(f"Preview: {doc.page_content[:200]}...")
                                st.write("---")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'query_type': query_type,
                            'response': response,
                            'evaluation_score': evaluation_score
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
        
        # Show chat history
        if st.session_state.chat_history:
            st.header("Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Query {len(st.session_state.chat_history) - i}: {chat['query'][:50]}..."):
                    st.write(f"**Time:** {chat['timestamp']}")
                    st.write(f"**Query Type:** {chat['query_type']}")
                    st.write(f"**Question:** {chat['query']}")
                    st.write(f"**Answer:** {chat['response']}")
                    if chat['evaluation_score'] is not None:
                        st.write(f"**Evaluation Score:** {chat['evaluation_score']:.2f}")
    
    else:
        st.info("Please upload and process PDF documents to start chatting!")

if __name__ == "__main__":
    main()