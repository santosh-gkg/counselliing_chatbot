import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import BSHTMLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from typing import List

load_dotenv()


with st.sidebar:
    st.title("About")
    st.markdown("This is a chatbot for IIIT Lucknow btech counselling. The chatbot is trained on the previous conversations in the Telegram Counselling group. If no good answer found then students can refer the telegram group https://t.me/iiitlcounselling or the official website https://iiitl.ac.in for more information.")
    st.title("Created By:")
    st.image('santosh.png')
    st.write("Santosh Kumar a student of MSc AI & ML in IIIT Lucknow ")

## loading the google and groq api key form the .env file
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


st.title("IIIT Lucknow btech counselling chatbot")
st.markdown('Note: This is an experimental project the answers can be inaccurate')
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-it")




prompt=ChatPromptTemplate.from_template(
    """
    Answer the question of the students based on the context of the conversation and the previous replies to the similar questions.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors =FAISS.load_local('chat_history',st.session_state.embeddings,allow_dangerous_deserialization=True)
vector_embedding()
prompt1=st.text_input("Please enter your query")



import time
document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retriever_chain=create_retrieval_chain(retriever,document_chain) 
if prompt1:
    
    start=time.process_time()
    response=retriever_chain.invoke({'input':prompt1})
    st.write(response['answer'])
    st.markdown("The time taken to get the response is {}".format(time.process_time()-start))
    st.markdown("for further information refer the telegram group https://t.me/iiitlcounselling or the official website https://iiitl.ac.in. The answers can be outdated and they should refer the official website for the latest information.")

    # with st.expander("Document similarity search"):
    #     for i,doc in enumerate(response['context']):
    #         st.write(doc.page_content)
    #         st.write("-------------------------------")
    
        
