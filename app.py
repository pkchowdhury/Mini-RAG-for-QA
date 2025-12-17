import streamlit as st
import requests

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Mini Agentic RAG for QA", page_icon="🤖")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

# Sidebar: Upload Document
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Ingesting and Chunking..."):
                try:
                    # Send file to backend
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success("Document processed successfully!")
                        st.session_state.system_ready = True
                        st.session_state.document_name = uploaded_file.name
                        # Clear previous chat history when new document is uploaded
                        st.session_state.messages = []
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Is 'main.py' running?")

    st.markdown("---")
    st.markdown("### Status")
    if st.session_state.system_ready:
        st.success("System: **Ready**")
        if st.session_state.document_name:
            st.info(f"Document: {st.session_state.document_name}")
    else:
        st.warning("System: **Waiting for PDF**")
    
    # Add debug mode toggle
    st.markdown("---")
    st.markdown("### Options")
    debug_mode = st.checkbox("Show retrieval details", value=False)

# Main Chat Interface
st.title("Mini Agentic RAG for QA")
st.caption("Created by Pallab Chowdhury")

# Display warning if system not ready
if not st.session_state.system_ready:
    st.info("Please upload a PDF document to get started!")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show debug info if available
        if debug_mode and message["role"] == "assistant" and "debug_info" in message:
            with st.expander("🔍 Retrieval Details"):
                debug = message["debug_info"]
                st.write(f"**Retrieved Chunks:** {debug.get('total_retrieved', 'N/A')}")
                st.write(f"**Relevant Chunks:** {debug.get('relevant_chunks', 'N/A')}")
                if "chunk_scores" in debug:
                    st.write("**Chunk Relevance:**")
                    for i, score in enumerate(debug["chunk_scores"], 1):
                        st.write(f"  - Chunk {i}: {score}")

# 2. Chat Input - Disabled when system not ready
if prompt := st.chat_input(
    "Ask a question about your document...", 
    disabled=not st.session_state.system_ready
):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(" Thinking...")
        
        try:
            # Call Backend API with debug flag
            response = requests.post(
                f"{API_URL}/chat", 
                json={"question": prompt, "debug": debug_mode}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "No answer received.")
                message_placeholder.markdown(answer)
                
                # Store message with debug info
                message_data = {"role": "assistant", "content": answer}
                if debug_mode and "debug_info" in result:
                    message_data["debug_info"] = result["debug_info"]
                
                st.session_state.messages.append(message_data)
                
                # Show debug info immediately if enabled
                if debug_mode and "debug_info" in result:
                    with st.expander("Retrieval Details"):
                        debug = result["debug_info"]
                        st.write(f"**Retrieved Chunks:** {debug.get('total_retrieved', 'N/A')}")
                        st.write(f"**Relevant Chunks:** {debug.get('relevant_chunks', 'N/A')}")
                        if "chunk_scores" in debug:
                            st.write("**Chunk Relevance:**")
                            for i, score in enumerate(debug["chunk_scores"], 1):
                                st.write(f"  - Chunk {i}: {score}")
                
            elif response.status_code == 400:
                 error_msg = "Please upload a document first."
                 message_placeholder.markdown(error_msg)
                 st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                error_msg = f"Error: {response.text}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.ConnectionError:
            error_msg = "Connection Error: Is the backend server running?"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add clear chat button
if st.session_state.messages:
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()