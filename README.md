# Mini Agentic RAG for QA - Improved Version

## 🎯 Key Improvements

### 1. **Enhanced Error Handling**
- ✅ Chat input disabled until document is uploaded
- ✅ Comprehensive try-catch blocks for all API calls
- ✅ Graceful error messages with emojis for better UX

### 2. **Vector Store Persistence**
- ✅ FAISS index saved to disk (`faiss_index/`)
- ✅ Automatic loading on server startup
- ✅ Survives server restarts

### 3. **Improved Retrieval Logic**
- ✅ Increased initial retrieval from k=3 to k=5 chunks
- ✅ Fallback strategy: If no relevant chunks, tries k=10
- ✅ Better chance of finding relevant information

### 4. **Debug Mode**
- ✅ Optional "Show retrieval details" toggle in UI
- ✅ Displays:
  - Total chunks retrieved
  - Number of relevant chunks
  - Individual chunk relevance scores
- ✅ Helps understand system behavior

### 5. **Enhanced Logging**
- ✅ Structured logging with timestamps
- ✅ Emoji indicators for different stages (🔍 🧐 ✅ ❌)
- ✅ Summary statistics after reflection
- ✅ Better observability for debugging

### 6. **Better UX**
- ✅ Document name displayed in sidebar
- ✅ Chat history cleared when new document uploaded
- ✅ "Clear Chat History" button
- ✅ System status indicator
- ✅ Warning message when no document is loaded

### 7. **Robustness**
- ✅ File validation (PDF only)
- ✅ Health check endpoint (`/health`)
- ✅ Better error messages for users
- ✅ Handles edge cases (no relevant docs, API errors)

## 📋 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your Azure OpenAI credentials
nano .env
```

### 3. Run the Backend
```bash
python main.py
```
Server will start at `http://127.0.0.1:8000`

### 4. Run the Frontend
In a new terminal:
```bash
streamlit run app.py
```

## 🚀 Usage

1. **Upload Document**: Click "Choose a PDF file" in the sidebar
2. **Process**: Click "Process Document" button
3. **Wait**: System will show "Ready" when complete
4. **Ask Questions**: Type your question in the chat input
5. **Enable Debug** (Optional): Check "Show retrieval details" to see what's happening under the hood

## 🔍 Debug Mode Example

When enabled, you'll see:
```
Retrieved Chunks: 5
Relevant Chunks: 3
Chunk Relevance:
  - Chunk 1: yes
  - Chunk 2: no
  - Chunk 3: yes
  - Chunk 4: yes
  - Chunk 5: no
```

## 🏗️ Architecture

```
User Question
    ↓
Retrieval Tool (k=5 initially)
    ↓
Critic Evaluation (per chunk)
    ↓
No relevant chunks? → Try k=10
    ↓
Still none? → Return "cannot answer"
    ↓
Found relevant? → Generate Answer
    ↓
Return to User
```

## 📊 Logging Output Example

```
2024-12-17 10:30:15 - AgenticRAG - INFO - 🤖 Agent received: What is the main topic?
2024-12-17 10:30:15 - AgenticRAG - INFO - 🔍 Calling Retrieval Tool...
2024-12-17 10:30:15 - AgenticRAG - INFO - Retrieved 5 documents
2024-12-17 10:30:16 - AgenticRAG - INFO - 🧐 Critic evaluating retrieved chunks...
2024-12-17 10:30:16 - AgenticRAG - INFO - ✅ Chunk 1/5: RELEVANT
2024-12-17 10:30:17 - AgenticRAG - INFO - ❌ Chunk 2/5: NOT RELEVANT
2024-12-17 10:30:17 - AgenticRAG - INFO - ✅ Chunk 3/5: RELEVANT
2024-12-17 10:30:18 - AgenticRAG - INFO - ✅ Chunk 4/5: RELEVANT
2024-12-17 10:30:18 - AgenticRAG - INFO - ❌ Chunk 5/5: NOT RELEVANT
2024-12-17 10:30:18 - AgenticRAG - INFO - 📊 Kept 3/5 chunks after reflection
2024-12-17 10:30:19 - AgenticRAG - INFO - ✍️ Generating Final Answer...
2024-12-17 10:30:20 - AgenticRAG - INFO - ✅ Answer generated successfully
```


## 🔧 API Endpoints

### POST /upload
Upload and process a PDF document.

**Response:**
```json
{
  "message": "PDF processed and vector store ready.",
  "chunks_created": 45
}
```

### POST /chat
Ask a question about the uploaded document.

**Request:**
```json
{
  "question": "What is the main topic?",
  "debug": true  // optional
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "debug_info": {
    "total_retrieved": 5,
    "relevant_chunks": 3,
    "chunk_scores": ["yes", "no", "yes", "yes", "no"],
    "timestamp": "2024-12-17T10:30:20"
  }
}
```

### GET /health
Check system health and readiness.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_ready": true
}
```

## 🐛 Troubleshooting

### "Connection Error"
- Ensure `main.py` is running on port 8000
- Check if another service is using port 8000

### "No relevant documents found"
- Try rephrasing your question
- Ensure the PDF contains relevant information
- Enable debug mode to see what's being retrieved

### Vector store not persisting
- Check write permissions in project directory
- Ensure `faiss_index/` folder is created

## 📝 Notes

- Vector store persists in `faiss_index/` directory
- Each new document upload overwrites the previous index
- Debug mode adds minimal overhead (~100ms per query)
- Chat history is cleared when uploading a new document

## 🎨 Created By
Pallab Chowdhury
