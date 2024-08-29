import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import pipeline

data = pd.read_csv("query.csv")
queries = data['User Query'].tolist()
responses = data['Response'].tolist()

documents = [Document(page_content=queries[i], metadata={"index": i}) for i in range(len(queries))]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)

llm = pipeline("text-generation", model="distilgpt2")

def retrieve(query, vector_store, responses):
    doc = vector_store.similarity_search(query, k=1)[0]
    response = responses[int(doc.metadata['index'])]
    return response

def conversational_chain(user_input, vector_store, llm, responses):
    context = retrieve(user_input, vector_store, responses)
    return context

def main():
    st.header('G-tec 🤖')

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask your question"):
        with st.chat_message("user", avatar='👨🏻'):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", 
                                          "avatar": '👨🏻',
                                          "content": prompt})
        response = conversational_chain(prompt, vector_store, llm, responses)
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", 
                                          "avatar": '🤖',
                                          "content": response})

if __name__ == '__main__':
    main()
