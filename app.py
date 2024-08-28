import streamlit as st
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from transformers import pipeline

# Load your dataset
data = pd.read_csv("query.csv")
queries = data['User Query'].tolist()
responses = data['Response'].tolist()

# Create documents with metadata containing the index
documents = [Document(page_content=queries[i], metadata={"index": i}) for i in range(len(queries))]

# Initialize embeddings and create FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize the language model
llm = pipeline("text-generation", model="distilgpt2")

# Define the retrieval function to get only 1 response
def retrieve(query, vector_store, responses):
    # Get the most similar document
    doc = vector_store.similarity_search(query, k=1)[0]
    
    # Retrieve the original response based on the document metadata
    response = responses[int(doc.metadata['index'])]
    return response

def conversational_chain(user_input, vector_store, llm, responses):
    # Retrieve the most relevant context
    context = retrieve(user_input, vector_store, responses)
    
    # Return the response directly without combining it with the user input
    return context

def main():
    st.header('G-tec 🤖')

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    # Chat input and response generation
    if prompt := st.chat_input("Ask your question"):
        with st.chat_message("user", avatar='👨🏻'):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", 
                                          "avatar": '👨🏻',
                                          "content": prompt})

        # Generate the response
        response = conversational_chain(prompt, vector_store, llm, responses)
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", 
                                          "avatar": '🤖',
                                          "content": response})

if __name__ == '__main__':
    main()
