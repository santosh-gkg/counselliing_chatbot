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
    You are a female receptionist in the IIIT Lucknow counselling office.
    Answer the question of the students based on the context provided either from the college website or from the conversation of the counselling group. politely reply in a formal manner as a first person without indicating as if you are a chatbot.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors =FAISS.load_local('combined',st.session_state.embeddings,allow_dangerous_deserialization=True)
vector_embedding()
# prompt1=st.text_input("Please enter your query")



import time
document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retriever_chain=create_retrieval_chain(retriever,document_chain) 
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and display chat messages
if prompt := st.chat_input("How can i help you?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        start=time.process_time()
        time_msg="The time taken to get the response is {}".format(time.process_time()-start)
        further_info=" \n For further information refer the telegram group https://t.me/iiitlcounselling or the official website https://iiitl.ac.in. \n The answers can be outdated and they should refer the official website for the latest information."
        response=retriever_chain.invoke({'input':prompt})['answer']
        # response+=time_msg+further_info
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
        
