import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate

# Load API key from .env
load_dotenv()

# Optional: Debug check
print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Starting PDF processing...")

    # Load PDF
    pdf_path = "/Users/webshar/Desktop/vectorstor-in-memory/Sample-pdf.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Store vectors
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local("faiss_index_gemini")

    # Load vectors back
    new_vectorstore = FAISS.load_local("faiss_index_gemini", embeddings, allow_dangerous_deserialization=True)

    #retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval_qa_chat")
    prompt = PromptTemplate.from_template("""
    Use the following context to answer the user's question. If you don't know the answer, say so.

    Context:
    {context}

    Question:
    {input}

    Answer:""")

    # Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Chain setup
    combine_docs_chain = create_stuff_documents_chain(llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    # Run the query
    result = retrieval_chain.invoke({"input": "Give me the any 3 sentences"})
    print("\nAnswer:\n", result["answer"])
