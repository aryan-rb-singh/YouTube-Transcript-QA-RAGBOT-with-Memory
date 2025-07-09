import os
import re
os.environ["MISTRAL_API_KEY"] = "your_api_key_here"

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# === Function to Extract YouTube Video ID from URL ===
def extract_video_id(url):
    """
    Extracts the video ID from a full YouTube URL.
    """
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        return None

# === Streamlit UI ===
st.title("🎥 YouTube Video Q&A with Mistral AI")
st.markdown("Enter the **YouTube video URL** and ask a question about its content.")

video_url = st.text_input("YouTube Video URL", value="https://www.youtube.com/watch?v=Gfr50f6ZBvo")
question = st.text_input("Ask a Question", value="What is DeepMind?")

if st.button("Submit") and video_url and question:
    video_id = extract_video_id(video_url)
    
    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            st.success("Transcript fetched successfully.")

            # Split transcript into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            # Create vector store from chunks
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            # Prompt and Mistral setup
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            llm = ChatMistralAI(model="mistral-small", temperature=0.2)

            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })

            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            with st.spinner("Generating answer..."):
                answer = main_chain.invoke(question)
                st.subheader("🧠 Answer")
                st.write(answer)

        except TranscriptsDisabled:
            st.error("No captions available for this video.")
        except Exception as e:
            st.error(f"Error: {e}")
