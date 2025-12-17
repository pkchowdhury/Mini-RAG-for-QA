import requests
import sys

# API URL
API_URL = "http://127.0.0.1:8000"

def upload_pdf(file_path):
    print(f"Uploading {file_path}...")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{API_URL}/upload", files={"file": f})
        
        if response.status_code == 200:
            print("Upload Success!")
            print(response.json())
        else:
            print(f"Upload Failed: {response.text}")
    except FileNotFoundError:
        print("Error: File not found. Check the path.")

def chat_loop():
    print("\n Chat System Ready. Type 'exit' to quit.")
    while True:
        question = input("\nUser: ")
        if question.lower() in ["exit", "quit"]:
            break
            
        try:
            response = requests.post(f"{API_URL}/chat", json={"question": question})
            if response.status_code == 200:
                print(f"Agent: {response.json()['answer']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        # Usage: python client.py upload my_document.pdf
        if len(sys.argv) < 3:
            print("Please provide a file path: python client.py upload <file.pdf>")
        else:
            upload_pdf(sys.argv[2])
    else:
        # Usage: python client.py
        print("Mini Agentic RAG for QA")
        print("1. To upload: python client.py upload path/to/doc.pdf")
        print("2. To chat:   Just run this script directly")
        chat_loop()