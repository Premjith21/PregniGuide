import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from prompt import build_prompt
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Load vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
)
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Handle both JSON and form data gracefully
        data = request.get_json(silent=True) or request.form

        question = data.get("question", "").strip()
        name = data.get("name", "").strip()
        pregnancy_week = data.get("pregnancy_week", "").strip()

        # Validate input
        if not name or not pregnancy_week or not question:
            return jsonify({"answer": "Please provide your name, pregnancy week, and question."}), 400

        # Retrieve context from vectorstore
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Build strict prompt
        prompt = build_prompt(question, context, name, pregnancy_week)

        # Call Groq API
        chat_completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
            top_p=1
        )

        answer = chat_completion.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
