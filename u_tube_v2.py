import os
import re
from dotenv import load_dotenv

# === Load API keys ===
load_dotenv()
PPLX_API_KEY = os.getenv("PPLX_API_KEY")

# === Imports ===
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_perplexity import ChatPerplexity


# === Helper: Extract Video ID ===
def extract_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL."""
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


# === Step 1: Ingest YouTube Transcript ===
def ingest_youtube_video(video_url: str):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        print("No captions available for this video.")
        transcript = ""

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Build embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


# === Step 2: Build RAG Chain ===
def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatPerplexity(
        model="sonar-pro",
        temperature=0.7
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return " ".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    rag_chain = parallel_chain | prompt | llm | parser
    return rag_chain


# === Step 3: Chat with Memory ===
def chat_with_rag(rag_chain):
    print("\n🤖 RAGBOT is ready! Ask questions about the video. Type 'exit' to quit.\n")
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Format chat history for memory
        conversation_history = "\n".join(history)
        
        # Create combined query (history + new question)
        query_with_history = f"Conversation so far:\n{conversation_history}\n\nUser's latest question: {user_input}"

        # Pass the combined query to RAG
        answer = rag_chain.invoke(query_with_history)
        print(f"Bot: {answer}\n")

        # Update history
        history.append(f"User: {user_input}")
        history.append(f"Bot: {answer}")



# === Run Bot ===
if __name__ == "__main__":
    yt_url = input("📺 Enter YouTube Video Link: ")
    vector_store = ingest_youtube_video(yt_url)
    rag_chain = build_rag_chain(vector_store)
    chat_with_rag(rag_chain)
