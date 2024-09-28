import streamlit as st
from database import *
import shelve
import os
import shutil

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms.ollama import Ollama


from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory.summary import ConversationSummaryMemory

import pickle
from langchain.schema.document import Document
import chromadb
import random

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
    with open('uploaded_files.pkl', 'wb') as file: 
        print("SAVING:",len(uploaded_files))
        s = pickle.dump(st.session_state["uploaded_files"],file)
        file.close()

    with open("file_names.pkl",'wb') as file:
        pickle.dump(names, file)
        file.close()

    with open("files_and_names.pkl",'wb') as file:
        pickle.dump(files_and_names, file)
        file.close()
    
    with open("removed_files.pkl",'wb') as file:    
        pickle.dump(removed_files, file)
        file.close()

def load_document_list():
    x = []
    try:
        with open('uploaded_files.pkl', 'rb') as file: 
            loaded = pickle.load(file) 
        print("LOADED FROM FILE",len(loaded))
        #st.session_state['uploaded_files'] = s
        #uploaded_files = s

        x.append(loaded)
        #return loaded
    except:
        return [[],[],{},[]]
    
    try:
        with open("file_names.pkl",'rb') as file:
            names = pickle.load(file)
        x.append(names)
    except:
        return [[],[],{},[]]
    
    try:
        with open("files_and_names.pkl",'rb') as file:
            files_and_names = pickle.load(file)
        x.append(files_and_names)
    except:
        return [[],[],{},[]]
    
    try:
        with open("removed_files.pkl",'rb') as file:
            removed_files = pickle.load(file)
        x.append(removed_files)

    except:
        return [[],[],{},[]]
    
    return x

def remove_from_database(filename):
    try:
        shutil.rmtree(os.path.join("DBs", filename))
    except FileNotFoundError:
        print("Error")
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    
embeddings = OpenAIEmbeddings()
llm = Ollama(model="llama3.2",temperature=0.6)


api_key = os.getenv("OPENAI_API_KEY")

#chunks = load_and_split()
#save_database(embeddings, chunks)
#db = load_database(embeddings, "DBs/Chroma")
#print("Ready to answer questions")

db =  None

count = 1000
st.title("Chat with Files")
names = []
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# @st.cache_resource
# def run_first():
#     global removed_files
#     #global file_list
#     #global current_file_names
#     global uploaded_files
#     global files_and_names
#     #global previous_file_names
#     uploaded_files = []
#     files_and_names = {}
#     removed_files = []

# run_first()

def initialize():
    #files_directory = 'predefined_files'
    
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        #file_paths = [os.path.join(files_directory, f) for f in os.listdir(files_directory) if f.endswith('.pdf')]
        #pdffiles = []  # This would be a list of file paths or file objects
        
        #for file_path in file_paths:
        #    pdffiles.append(file_path)  # Adjust this part to open the file if needed for processing
        with open("removed_files.pkl",'wb') as file:    
            pickle.dump([], file)
            file.close()
        st.session_state.initialized = True  # Mark initialization as done to avoid re-running
        
        #st.write(raw_text)
    else:
        st.write("Initialization already done.")

initialize()
#removed_files = initialize()
removed_files = []
#print(removed_files)
file_list = []

current_file_names = []
previous_file_names = []
conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))
# try:
#     with open('uploaded_files.pkl', 'rb') as file: 
#         s = pickle.load(file) 
#     print("LOADED FROM FILE",len(s))
#     #st.session_state['uploaded_files'] = s
# except:
#     pass
l = load_document_list()
loaded = l[0]
names = l[1]
files_and_names = l[2]
removed_files = l[3]

model_name = "llama3.2"
print("ldfnaldsasda")
with st.sidebar:

    #print("Removed file names: ", removed_files)

    st.write("Select Model")
    option = st.selectbox("Select Model",("Llama3.2", "GPT 4o", "GPT 4o Mini"))
    

    match option:
        case "Llama 3.2":
            model_name = "llama3.2"
            llm = Ollama(model=model_name,temperature=0.6)
            conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))

        case "GPT 4o":
            model_name = "gpt-4o-turbo"
            llm = ChatOpenAI(model=model_name,temperature=0.6)
            conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))

        case "GPT 4o Mini":
            model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name,temperature=0.6)
            conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))



    previous_file_names = current_file_names
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1


    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []

    # try:
    #     with open('uploaded_files.pkl', 'rb') as file: 
    #         loaded = pickle.load(file) 
    #     print("LOADED FROM FILE",len(s))
    #     #st.session_state['uploaded_files'] = s
    #     #uploaded_files = s
    # except:
    #     pass
    
    l = load_document_list()
    loaded = l[0]
    names = l[1]
    files_and_names = l[2]
    removed_files = l[3]
    #st.session_state['uploaded_files'] = s
    #uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
    previous_file_names = [file.name for file in st.session_state['uploaded_files']]

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
            # p = os.path.join("DBs", database_to_remove)
            # for file in os.listdir(p):
            #     print(file, type(file))
            #     os.remove(os.path.join(p, file))
            # os.rmdir(os.path.join("DBs", database_to_remove))
        print("===================\nDatabases cleared\n===================\n")
        st.write("Databases cleared")
        st.session_state["uploader_key"] += 1
        current_file_names = []
        previous_file_names = []
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        st.session_state["uploaded_files"] = []
        uploaded_files = []
        st.session_state['uploaded_files'] = uploaded_files
        save_document_list()
        file_list = []

        files_and_names = {}
        print("uploaded files:",len(uploaded_files))
        names = []
        save_document_list()
        l = load_document_list()
        loaded = l[0]
        names = l[1]
        files_and_names = l[2]
        removed_files = l[3]



    #st.session_state["uploader_key"] = 1

    uploaded_files = st.file_uploader(
        "Choose a PDF file", 
        accept_multiple_files=True,
        key=st.session_state["uploader_key"]
    )

    for z in loaded:
        if z not in uploaded_files:
            uploaded_files += loaded

    st.session_state['uploaded_files'] = uploaded_files
    # with open('uploaded_files.pkl', 'wb') as file: 
    #     print("SAVING:",len(uploaded_files))
    #     s = pickle.dump(st.session_state["uploaded_files"],file)
    #     file.close()
    save_document_list()


    #previous_file_names = [file.name for file in st.session_state['uploaded_files']]

    #print("a1231231",current_file_names)
    #previous_file_names = current_file_names

    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []
    
    print("asdasdasdasd",len(uploaded_files))
    #print(current_file_names)
    #print(previous_file_names)

    if len(uploaded_files) > 0:
        #print(f"\n\n {len(uploaded_files)} Uploaded files")
        #print(uploaded_files)
        #print("\n\n\n")
        st.write("Loaded Files:")
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                file_list.append(uploaded_file.name)
            bytes_data = uploaded_file.read()
            if uploaded_file.name not in removed_files:
                st.write("filename:", uploaded_file.name)
            #st.write(bytes_data)
            filename = uploaded_file.name
            names.append(filename)
            db_path = os.path.join("DBs",filename)
            print("FILENAME:",filename)
            files_and_names[filename] = uploaded_file
            print("FILES AND NAMES:",files_and_names)
            #if not os.path.exists(db_path):# and st.session_state["uploader_key"] != 2:
            #if filename not in os.listdir("DBs") and filename in files_and_names.keys() and filename not in removed_files:
            if filename not in os.listdir("DBs") and filename not in removed_files:
                print("fil",files_and_names.keys(),removed_files)
                with st.spinner('Saving to database...'):
                    os.makedirs(db_path)
                    file_text = extract_pdf_text(uploaded_file)
                    #print(type(file_text))
                    doc = Document(file_text)
                    print("Trying to save:",len(file_text),"at",db_path)
                    save_database(embeddings, create_chunks([doc]), db_path)


    #for file_name in file_list:
    #Need to use list conversion dict to avoid RuntimeError: dictionary changed size during iteration
    print("********BEFORE LOOP:",len(files_and_names.keys()))
    for file_name in file_list:#list(files_and_names):
        print("length:",len(files_and_names.keys()))
        col1, col2 = st.columns([0.9, 0.1])
        if file_name in os.listdir("DBs"):
            print(os.listdir("DBs"))
            print("*********IN DICT",len(files_and_names.keys()))
            with col1:
                #if file_name in files_and_names.keys():

                    st.write(file_name)
            with col2:
                count += 1

                remove_file = st.button("X",key=count)#, key=(file_list.index(file_name)+random.randint(1,50))*1000)
            if remove_file:
                print("*************removing",file_name,os.listdir("DBs"))
                # Remove the file name from the file_list if the button is clicked
                if file_name not in removed_files:
                    removed_files.append(file_name)
                print(type(file_name),"Uploaded files:",len(uploaded_files),"names:",len(file_list))
                print(uploaded_files)
                remove_from_database(file_name)
                print("REMOVED FROM FOLDER:",os.listdir("DBs"))
                #uploaded_files.remove(file_list.index(file_name))

                #del uploaded_files[file_list.index(file_name)]
                #del st.session_state["uploaded_files"][file_list.index(file_name)]
                print(files_and_names)
                print(files_and_names[file_name])
                uploaded_files.remove(files_and_names[file_name])
                st.session_state["uploaded_files"] = uploaded_files


                files_and_names.pop(file_name)

                print(len(files_and_names.keys()))
                


                print("New length:",len(uploaded_files),len(st.session_state["uploaded_files"]))
                current_file_names.remove(file_name)
                print(len(file_list))
                file_list.remove(file_name)
                print(len(file_list))
                #print("\n\n",current_file_names)
                #print(previous_file_names,"\n\n")

                save_document_list()
                l = load_document_list()
                loaded = l[0]
                names = l[1]
                files_and_names = l[2]
                removed_files = l[3]

                continue
                # current_file_names = []
                # previous_file_names = []
                # #st.session_state["uploaded_files"] = [file for file in st.session_state["uploaded_files"] if file.name != file_name]
                # print("Uploaded files:",len(uploaded_files),"names:",len(names))
                # print(st.session_state["uploaded_files"])
                # print(loaded)
                # if file_name in names:
                #     i = names.index(file_name)
                #     names.remove(file_name)
                # #$st.session_state["uploaded_files"].remove(i)
                # del st.session_state["uploaded_files"][i]
                # remove_from_database(file_name)
                # file_list.remove(file_name)
                # save_document_list()
                # l = load_document_list()
                # loaded = l[0]
                # names = l[1]

    print("Removed file names: ", removed_files)
    print("Current file names: ", current_file_names)
    print("Previous file names: ", previous_file_names)
    # Identify which files were removed by comparing lists
    for prev_file_name in previous_file_names:
        print("hi")
        print(prev_file_name)
        if prev_file_name not in current_file_names:
            print("Hello")
            #removed_files.append(prev_file_name)
            remove_from_database(prev_file_name)
        #if remove_file:
            #file_list.remove(file_name)
            #shutil.rmtree(os.path.join("DBs", prev_file_name))
            #chromadb.api.client.SharedSystemClient.clear_system_cache()




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
        context = []
        
        with st.spinner("Searching Database..."):
            for file in current_file_names:
                db = load_database(embeddings, os.path.join("DBs", file))
                info = query_database(prompt,db)
                print(type(info))
                context += info
            #context = query_database(prompt,db)
            print(len(context))

            full_response = get_response(context,prompt,llm)
            message_placeholder.markdown(full_response)   
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)

