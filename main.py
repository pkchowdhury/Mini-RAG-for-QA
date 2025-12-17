# Import Standard Libraries
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, Field

# Import Langchain Components
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

# Configuration and Logging Setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AgenticRAG")

# Initialize Azure Components

# Initialize Chat Model
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_CHAT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

# Initialize Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBED"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Global Vector Store Reference
vector_store = None
vector_store_path = "faiss_index"

# 2. Document Chunking & Ingestion 
def ingest_pdf(file_path: str):
    global vector_store
    logger.info(f"Processing PDF: {file_path}")
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from PDF")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        logger.info(f"Created {len(splits)} chunks")
        
        # 3. Store in FAISS
        logger.info("Creating Embeddings and Storing in FAISS...")
        vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Save vector store for persistence
        vector_store.save_local(vector_store_path)
        logger.info(f"Vector store saved to {vector_store_path}")
        logger.info("Ingestion Complete.")
        
        return len(splits)
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {str(e)}")
        raise

# Load existing vector store on startup
def load_vector_store():
    global vector_store
    if os.path.exists(vector_store_path):
        try:
            vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing vector store from {vector_store_path}")
            return True
        except Exception as e:
            logger.warning(f"Could not load vector store: {str(e)}")
            return False
    return False

# 4. Retrieval Logic (The Tool)
def retrieval_tool(query: str, k: int = 5) -> List[Document]:
    """
    Retrieves top-k most relevant document chunks.
    Increased k from 3 to 5 for better coverage.
    """
    if not vector_store:
        raise ValueError("Vector store not initialized. Upload a PDF first.")
    
    logger.info(f"Retrieving top-{k} chunks for query: {query[:50]}...")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} documents")
    
    return docs

# 5. Agent Components

# A. The Critic (Self-Reflection)
# Checks if retrieved documents are relevant to the query.
class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

parser = JsonOutputParser(pydantic_object=Grade)

critic_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing relevance of a retrieved document to a user question.
    Here is the retrieved document:
    \n\n {context} \n\n
    Here is the user question: {question}
    
    If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    
    {format_instructions}"""
)

critic_chain = critic_prompt | llm | parser

# B. The Generator (Final Answer)
generator_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If the context is not relevant or empty, say that you cannot answer based on the provided document.
    Keep the answer concise and accurate.
    
    Context: {context} 
    Question: {question} 
    Answer:"""
)

generator_chain = generator_prompt | llm

# Agent Control Loop with Enhanced Logging
def run_agentic_rag(question: str, debug: bool = False) -> Dict:
    """
    Run the agentic RAG pipeline with optional debug information.
    
    Returns:
        Dict with 'answer' and optional 'debug_info'
    """
    logger.info(f"🤖 Agent received: {question}")
    
    debug_info = {
        "total_retrieved": 0,
        "relevant_chunks": 0,
        "chunk_scores": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Step 1: Tool Calling (Retrieval)
    logger.info("🔍 Calling Retrieval Tool...")
    try:
        docs = retrieval_tool(question, k=5)  # Increased from 3 to 5
        debug_info["total_retrieved"] = len(docs)
    except ValueError as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return {
            "answer": "System not ready. Please upload a PDF first.",
            "debug_info": debug_info if debug else None
        }

    # Step 2: Self-Reflection (Critic)
    valid_context = []
    
    logger.info("Critic evaluating retrieved chunks...")
    for i, doc in enumerate(docs, 1):
        try:
            grade = critic_chain.invoke({
                "context": doc.page_content, 
                "question": question,
                "format_instructions": parser.get_format_instructions()
            })
            
            score = grade['binary_score'].lower()
            debug_info["chunk_scores"].append(score)
            
            if score == 'yes':
                logger.info(f"Chunk {i}/{len(docs)}: RELEVANT")
                valid_context.append(doc.page_content)
                debug_info["relevant_chunks"] += 1
            else:
                logger.info(f"Chunk {i}/{len(docs)}: NOT RELEVANT")
        except Exception as e:
            logger.error(f"Error evaluating chunk {i}: {str(e)}")
            debug_info["chunk_scores"].append("error")

    # Log summary
    logger.info(f"📊 Kept {len(valid_context)}/{len(docs)} chunks after reflection")

    # Step 3: Fallback Strategy - If no chunks are relevant, try re-retrieval with higher k
    if not valid_context:
        logger.warning("No relevant documents found. Trying broader retrieval...")
        try:
            # Try retrieving more documents (up to 10)
            docs_extended = retrieval_tool(question, k=10)
            debug_info["total_retrieved"] = len(docs_extended)
            
            for i, doc in enumerate(docs_extended[5:], len(docs)+1):  # Check new docs only
                try:
                    grade = critic_chain.invoke({
                        "context": doc.page_content, 
                        "question": question,
                        "format_instructions": parser.get_format_instructions()
                    })
                    
                    score = grade['binary_score'].lower()
                    debug_info["chunk_scores"].append(score)
                    
                    if score == 'yes':
                        logger.info(f"Extended Chunk {i}: RELEVANT")
                        valid_context.append(doc.page_content)
                        debug_info["relevant_chunks"] += 1
                except Exception as e:
                    logger.error(f"Error in extended evaluation: {str(e)}")
        except Exception as e:
            logger.error(f"Extended retrieval failed: {str(e)}")
    
    # Step 4: Final Decision & Generation
    if not valid_context:
        logger.warning("No relevant documents found after extended search")
        return {
            "answer": "I'm sorry, but the provided document does not contain information relevant to your question. Please try rephrasing your question or upload a different document.",
            "debug_info": debug_info if debug else None
        }
    
    logger.info("Generating Final Answer...")
    formatted_context = "\n\n".join(valid_context)
    
    try:
        response = generator_chain.invoke({
            "context": formatted_context, 
            "question": question
        })
        
        logger.info("Answer generated successfully")
        return {
            "answer": response.content,
            "debug_info": debug_info if debug else None
        }
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {
            "answer": "An error occurred while generating the answer. Please try again.",
            "debug_info": debug_info if debug else None
        }

# 6. API Implementation
app = FastAPI(title="Azure Agentic RAG API")

class QueryRequest(BaseModel):
    question: str
    debug: bool = False  # Optional debug flag

@app.on_event("startup")
async def startup_event():
    """Load vector store on startup if it exists"""
    load_vector_store()
    logger.info(" Server started")

@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and process a PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save temp file
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Trigger Ingestion
        num_chunks = ingest_pdf(temp_filename)
        
        # Cleanup
        os.remove(temp_filename)
        logger.info(f"Cleaned up temporary file: {temp_filename}")
        
        return {
            "message": "PDF processed and vector store ready.",
            "chunks_created": num_chunks
        }
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """Chat endpoint for asking questions"""
    if not vector_store:
        raise HTTPException(
            status_code=400, 
            detail="Please upload a PDF first."
        )
    
    logger.info(f"Received question: {request.question}")
    
    try:
        result = run_agentic_rag(request.question, debug=request.debug)
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_ready": vector_store is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)