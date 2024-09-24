import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load the NVIDIA model client
llm = ChatNVIDIA(
  model="meta/llama-3.1-70b-instruct",
  api_key=os.getenv("NVIDIA_API_KEY"), 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

# def vector_embeddings():
#   if "vectors" not in st.session_state:
#     st.session_state.embeddings=NVIDIAEmbeddings()
#     st.session_state.loader=PyPDFDirectoryLoader("/content/pdfs")
#     st.session_state.docs=st.session_state.loader.load()
#     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
#     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
#     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("/content/pdfs")
        
        # Load the documents
        st.session_state.docs = st.session_state.loader.load()
        if len(st.session_state.docs) == 0:
            st.error("No documents found in the directory.")
            return

        # Use text splitter to split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=75)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])

        # Check the documents after splitting
        if len(st.session_state.final_documents) == 0:
            st.error("No final documents available after splitting.")
            return
        else:
            print(f"Number of final documents: {len(st.session_state.final_documents)}")

        # Generate the embeddings and create FAISS index
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector index created successfully!")
        except Exception as e:
            st.error(f"Error creating FAISS index: {str(e)}")


st.title("NVIDIA NIM PDF RAG Chatbot Demo")
st.subheader("Document used: SQL Manual")

prompt=ChatPromptTemplate.from_template(
  """
  Answer the questions based on the provided context only.
  PLease provide the most accurate response based on the question
  <content>
  {context}
  <content>
  Question:{input}

  """
)

prompt1=st.text_input("Enter the question!")

if st.button("Document Embedding"):
  vector_embeddings()
  st.write("Faiss Vectorstore DB is ready using NVIDIA Embeddings!")

import time

if prompt1:
  document_chain=create_stuff_documents_chain(llm, prompt)
  retriever=st.session_state.vectors.as_retriever()
  retrieval_chain=create_retrieval_chain(retriever,document_chain)
  start=time.process_time()
  response=retrieval_chain.invoke({"input":prompt1})
  print("Response time:", time.process_time()-start)
  st.write(response['answer'])








