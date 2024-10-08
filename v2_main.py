import streamlit as st
from database import *
import shelve
import os
import shutil

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain.memory import ChatMessageHistory


from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory.summary import ConversationSummaryMemory

import pickle
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
            "You are a helpful assistant. You are thoughtful and thorough in your responses."
        ),
        (
            "user",
            """You can use the following context to help you answer the question if there is some:
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
    
def save_document_list():

    with open("files_and_names.pkl",'wb') as file:
        pickle.dump(files_and_names, file)
        file.close()

def load_document_list():
    x = []

    try:
        with open("files_and_names.pkl",'rb') as file:
            files_and_names = pickle.load(file)
        x.append(files_and_names)
    except:
        return [{}]
    
    return x

def remove_from_database(filename):
    try:
        shutil.rmtree(os.path.join("DBs", filename))
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        return True
    except Exception as e:
        print("Error:",e)
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        return False

    
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = Ollama(model="llama3.2",temperature=0.6)


api_key = os.getenv("OPENAI_API_KEY")

db =  None

count = 1000
st.title("Chat with Files")
names = []
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "DBs" not in os.listdir():
    os.makedirs("DBs")


def initialize():    
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        with open("removed_files.pkl",'wb') as file:    
            pickle.dump([], file)
            file.close()
        st.session_state.initialized = True  # Mark initialization as done to avoid re-running
    else:
        st.write("Initialization already done.")

initialize()
removed_files = []
file_list = []

current_file_names = []
previous_file_names = []
conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))

files_and_names = load_document_list()[0]
print("Loaded files and names:",files_and_names)
model_name = "llama3.2"
with st.sidebar:

    #print("Removed file names: ", removed_files)

    st.write("Select Model")
    option = st.selectbox("Select Model",("Llama3.2", "GPT 4o", "GPT 4o Mini"))
    


    match option:
        case "Llama 3.2":
            model_name = "llama3.2"
            llm = Ollama(model=model_name,temperature=0.6)
            #conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))
            

        case "GPT 4o":
            model_name = "gpt-4o"
            llm = ChatOpenAI(model=model_name,temperature=0.6)
            #conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))

        case "GPT 4o Mini":
            model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name,temperature=0.6)
        

    if "memory" not in st.session_state:
        conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))
        st.session_state["memory"] = conversation
        st.session_state["history"] = ChatMessageHistory()
        memory = ConversationSummaryMemory.from_messages(
        llm=llm,
        chat_memory=st.session_state["history"],
        return_messages=True
        )
    else:
        conversation = st.session_state["memory"]
        st.session_state["history"] = ChatMessageHistory()
        memory = ConversationSummaryMemory.from_messages(
        llm=llm,
        chat_memory=st.session_state["history"],
        return_messages=True
        )
    

    number_of_results = st.number_input(
    "Number of results per database", value=20, placeholder="20"
    )


    previous_file_names = current_file_names
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1


    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []


    files_and_names = load_document_list()[0]
    previous_file_names = list(files_and_names.keys())


    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])
    
    if st.button("Clear Databases"):
        for database_to_remove in os.listdir("DBs"):
            try:
                shutil.rmtree(os.path.join("DBs", database_to_remove))
                print("removed:",database_to_remove)
            except:
                print("Error")
        print("===================\nDatabases cleared\n===================\n")
        st.write("Databases cleared")


        st.session_state["uploader_key"] += 1
        st.session_state["uploaded_files"] = []


        files_and_names = {}
        current_file_names = []
        previous_file_names = []
        uploaded_files = []
        removed_files = []

        #st.session_state['uploaded_files'] = uploaded_files
        st.session_state['uploaded_files'] = list(files_and_names.keys())
        print("uploaded files:",len(uploaded_files))
        save_document_list()
        files_ane_names = load_document_list()[0]
        chromadb.api.client.SharedSystemClient.clear_system_cache()

    uploaded_files = st.file_uploader(
        "Choose a PDF file", 
        accept_multiple_files=True,
        key=st.session_state["uploader_key"],
        type=["pdf"]
    )

    st.write("Number of files:",len(files_and_names.keys()))

    

    st.session_state['uploaded_files'] = list(files_and_names.keys())#uploaded_files

    save_document_list()



    current_file_names = list(files_and_names.keys())

    if len(uploaded_files) > 0:


        print("Here are the uploaded_files",uploaded_files)
        st.write("Loaded Files:")
        for uploaded_file in uploaded_files:

            bytes_data = uploaded_file.read()
            #else:
            #    continue
            filename = uploaded_file.name

            
            


            db_path = os.path.join("DBs",filename)


            if filename not in os.listdir("DBs") and filename not in files_and_names.keys():# and filename not in removed_files:
                files_and_names[filename] = uploaded_file

                print("fil",files_and_names.keys(),removed_files)
                with st.spinner('Saving to database...'):
                    os.makedirs(db_path)
                    file_text = extract_pdf_text(uploaded_file)
                    #print(type(file_text))
                    doc = Document(file_text)
                    print("Trying to save:",len(file_text),"at",db_path)
                    #st.write("trying to save:",filename)
                    save_database(embeddings, create_chunks([doc]), db_path)
                    save_document_list()

        uploaded_files = []
 
    st.session_state["uploaded_files"] = list(files_and_names.keys())
    for file_name in list(files_and_names):
        print("f",file_name)
        print(files_and_names.keys())
        print(st.session_state["uploaded_files"])
        col1, col2 = st.columns([0.9, 0.2])
        if file_name in os.listdir("DBs") and file_name in files_and_names.keys() and file_name in st.session_state["uploaded_files"]:
            print("Directory:",os.listdir("DBs"))
            #if file_name in files_and_names.keys():
            with col1:
                    st.write(file_name)
            with col2:
                count += 1

                remove_file = st.button("X",key=count)


            if remove_file:
                a = 0
                status = remove_from_database(file_name)
                if status:
                    st.write(f"Removed: {file_name}")
                    print("before:",files_and_names)
                    files_and_names.pop(file_name)
                    print("after:",files_and_names)
                    st.session_state["uploaded_files"].remove(file_name)
                    current_file_names.remove(file_name)
                    print("new:",current_file_names)
                    save_document_list()
                    st.rerun()
                else:
                    st.write(f"Error removing: {file_name}")
 
    st.write("Removed files:",removed_files)
    st.write("Current file names:",current_file_names)
    st.write("Previous file names:",previous_file_names)
    st.write("Files and names:",files_and_names)
    

    print("Removed file names: ", removed_files)
    print("Current file names: ", current_file_names)
    print("Previous file names: ", previous_file_names)
    # Identify which files were removed by comparing lists
    for prev_file_name in previous_file_names:
        if prev_file_name not in current_file_names:
            if prev_file_name not in removed_files:
                removed_files.append(prev_file_name)
            try:
                remove_from_database(prev_file_name)
            except Exception as e:
                print("Error removing file:",e)



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        context_list = []
        context = ""
        num_results_counter = 0

        
        with st.spinner("Searching Database..."):
            for file in current_file_names:
                db = load_database(embeddings, os.path.join("DBs", file))
                results, results_text = query_database(prompt,db,num_responses=number_of_results)
                num_results_counter += len(results)

                print(type(results))
                context += "\n\n New Document Source:\n"+results_text+"\n\n"
                #st.write("Results:",results_text)
            
        st.markdown(f"*Got {num_results_counter} results from database*")

        with st.spinner(f"Generating response..."):

            

            full_response = get_response(context,prompt,llm)
            message_placeholder.markdown(full_response)   
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)

