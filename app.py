import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

# Embeddings (must match build_index.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# embeddings = OpenAIEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     api_key=os.getenv("DEEPINFRA_API_KEY"),
#     base_url="https://api.deepinfra.com/v1/openai"
# )



# Load FAISS index
vectorstore = FAISS.load_local(
    "office_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# DeepInfra LLM (OpenAI-compatible)
llm = ChatOpenAI(
    model="meta-llama/Llama-3.2-3B-Instruct",
    openai_api_key=os.getenv("DEEPINFRA_API_KEY"),
    openai_api_base="https://api.deepinfra.com/v1/openai",
    temperature=0.2
)

# Prompt template
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""")

# RAG pipeline using LCEL
def rag_answer(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({"context": context, "question": question})


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer = rag_answer(question)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
