import streamlit as st
from database import *
import shelve
import os
import shutil

from langchain_openai import OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain.schema.document import Document
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

def load_chat_history():
    with shelve.open("conversation_history") as db:
        return db.get("messages",[])
    
def save_chat_history(messages):
    with shelve.open("conversation_history") as db:
        db["messages"] = messages



def get_response(context, question, llm):
    #encoded_image = encode_image(path)

    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an experienced advisor and international diplomat who is assisting the US government in foreign policy. You use natural language "
         "to answer questions based on structured data, unstructured data, and community summaries. You are thoughtful and thorough in your responses."
        ),
        (
            "user",
            """Answer the question only based on the following context:
            {context}


            Here is the question:
            {question}"""
        ),
        ]
        )
    
    chain = (
         {"context": lambda x: context, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
    )
    #response_text = model.invoke(prompt)
    response_text = chain.invoke(question)
    return response_text
    

embeddings = OpenAIEmbeddings()
llm = Ollama(model="llama3.2",temperature=0.5)


api_key = os.getenv("OPENAI_API_KEY")

#chunks = load_and_split()
#save_database(embeddings, chunks)
#db = load_database(embeddings, "DBs/Chroma")
#print("Ready to answer questions")

db =  None


st.title("Chat with Files")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

file_list = []
removed_files = []
current_file_names = []

with st.sidebar:

    previous_file_names = current_file_names
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1


    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []


    #uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
    previous_file_names = [file.name for file in st.session_state['uploaded_files']]

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])
    
    if st.button("Clear Databases"):
        for database_to_remove in os.listdir("DBs"):
            try:
                shutil.rmtree(os.path.join("DBs", database_to_remove))
            except:
                print("Error")
            # p = os.path.join("DBs", database_to_remove)
            # for file in os.listdir(p):
            #     print(file, type(file))
            #     os.remove(os.path.join(p, file))
            # os.rmdir(os.path.join("DBs", database_to_remove))
        print("===================\nDatabases cleared\n===================\n")
        st.write("Databases cleared")
        st.session_state["uploader_key"] += 1
        chromadb.api.client.SharedSystemClient.clear_system_cache()


    #st.session_state["uploader_key"] = 1

    uploaded_files = st.file_uploader(
        "Choose a PDF file", 
        accept_multiple_files=True,
        key=st.session_state["uploader_key"]
    )

    st.session_state['uploaded_files'] = uploaded_files


    #previous_file_names = [file.name for file in st.session_state['uploaded_files']]

    #print("a1231231",current_file_names)
    #previous_file_names = current_file_names

    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []

    print("asdasdasdasd",len(uploaded_files))
    print(current_file_names)
    print(previous_file_names)

    if len(uploaded_files) > 0:
        #print(f"\n\n {len(uploaded_files)} Uploaded files")
        print(uploaded_files)
        #print("\n\n\n")
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                file_list.append(uploaded_file.name)
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            #st.write(bytes_data)
            filename = uploaded_file.name
            db_path = os.path.join("DBs",filename)
            if not os.path.exists(db_path):# and st.session_state["uploader_key"] != 2:
                os.makedirs(db_path)
                file_text = extract_pdf_text(uploaded_file)
                print(type(file_text))
                doc = Document(file_text)
                print("Trying to save:",len(file_text),"at",db_path)
                save_database(embeddings, create_chunks([doc]), db_path)
                

    print("Current file names: ", current_file_names)
    print("Previous file names: ", previous_file_names)
    # Identify which files were removed by comparing lists
    for prev_file_name in previous_file_names:
        print("hi")
        print(prev_file_name)
        if prev_file_name not in current_file_names:
            print("Hello")
            #removed_files.append(prev_file_name)

        #if remove_file:
            #file_list.remove(file_name)
            shutil.rmtree(os.path.join("DBs", prev_file_name))
            chromadb.api.client.SharedSystemClient.clear_system_cache()




for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        context = query_database(prompt,db)
        print(context)
        full_response = get_response(context,prompt,llm)
        message_placeholder.markdown(full_response)   
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)

