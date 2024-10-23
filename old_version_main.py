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
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        return True
    except Exception as e:
        print("Error:",e)
        chromadb.api.client.SharedSystemClient.clear_system_cache()

        return False

    
embeddings = OpenAIEmbeddings()
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
print("***here are the removed files:",removed_files)
model_name = "llama3.2"
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
            model_name = "gpt-4o-2024-08-06"
            llm = ChatOpenAI(model=model_name,temperature=0.6)
            conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))

        case "GPT 4o Mini":
            model_name = "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name,temperature=0.6)
            conversation = ConversationChain(llm = llm, memory = ConversationSummaryMemory(llm=llm))


    number_of_results = st.number_input(
    "Number of results per database", value=20, placeholder="20"
    )


    previous_file_names = current_file_names
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1


    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []


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


        print("===================\nDatabases cleared\n===================\n")
        st.write("Databases cleared")
        st.session_state["uploader_key"] += 1
        current_file_names = []
        previous_file_names = []
        st.session_state["uploaded_files"] = []
        uploaded_files = []
        removed_files = []

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

        chromadb.api.client.SharedSystemClient.clear_system_cache()

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



    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []

    if len(uploaded_files) > 0:
        #print(f"\n\n {len(uploaded_files)} Uploaded files")
        #print(uploaded_files)
        #print("\n\n\n")
        st.write("Loaded Files:")
        for uploaded_file in uploaded_files:

            if uploaded_file is not None and uploaded_file.name not in removed_files:
                file_list.append(uploaded_file.name)
            #if uploaded_file.name not in removed_files:
                #st.write("filename:", uploaded_file.name)
                #st.write(bytes_data)
                bytes_data = uploaded_file.read()
            else:
                continue
            filename = uploaded_file.name

            if filename not in st.session_state["uploaded_files"] and filename not in removed_files:
                names.append(filename)
                db_path = os.path.join("DBs",filename)
                print("FILENAME:",filename)
                files_and_names[filename] = uploaded_file
                print("FILES AND NAMES:",files_and_names)
                #if not os.path.exists(db_path):# and st.session_state["uploader_key"] != 2:
                #if filename not in os.listdir("DBs") and filename in files_and_names.keys() and filename not in removed_files:
                if filename in removed_files:
                    print("REMOVED:",removed_files)
                if filename not in os.listdir("DBs") and filename not in removed_files:
                    print("fil",files_and_names.keys(),removed_files)
                    with st.spinner('Saving to database...'):
                        os.makedirs(db_path)
                        file_text = extract_pdf_text(uploaded_file)
                        #print(type(file_text))
                        doc = Document(file_text)
                        print("Trying to save:",len(file_text),"at",db_path)
                        st.write("trying to save:",filename)
                        save_database(embeddings, create_chunks([doc]), db_path)

    #for file_name in file_list:
    #Need to use list conversion dict to avoid RuntimeError: dictionary changed size during iteration
    print("********BEFORE LOOP:",len(files_and_names.keys()))
    files_removed = False
    for file_name in file_list:#list(files_and_names):
        print("length:",len(files_and_names.keys()))
        col1, col2 = st.columns([0.9, 0.2])
        if file_name in os.listdir("DBs"):
            print(os.listdir("DBs"))
            print("*********IN DICT",len(files_and_names.keys()),removed_files)
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
                
                z = remove_from_database(file_name)
                if z:
                    print("ASDKFUNASDKLXJFNALKSJNDLKA")
                    st.write(f"REMOVING FROM DB: {file_name}")
                    st.session_state["uploaded_files"].remove(files_and_names[file_name])
                    uploaded_files = st.session_state["uploaded_files"]

                    files_and_names.pop(file_name)
                    current_file_names.remove(file_name)
                    file_list.remove(file_name)
                    #a += 1
                    save_document_list()

                    print(len(file_list))
                    #print("\n\n",current_file_names)
                    #print(previous_file_names,"\n\n")

                    save_document_list()
                    l = load_document_list()
                    loaded = l[0]
                    names = l[1]
                    files_and_names = l[2]
                    removed_files = l[3]
                    files_removed = True
                    with col1:
                        st.write("")

                else:
                    print("Error removing file:",file_name)
                    #removed_files.remove(f)
                    #removed_files.remove(f)
                    #a += 1
                    #removed_files = []
                
                #st.rerun()
                #continue
    # if files_removed:
    #     st.rerun()
    #     files_removed = False
    #chromadb.api.client.SharedSystemClient.clear_system_cache()
    # a = 0
    # st.write("before Removed files:",removed_files)
    # while a < len(removed_files):
    #     f = removed_files[a]
    # #for f in removed_files:
        
    #     chromadb.api.client.SharedSystemClient.clear_system_cache()
    #     z = remove_from_database(f)
    #     if z:
    #         print("ASDKFUNASDKLXJFNALKSJNDLKA")
    #         st.write(f"REMOVING FROM DB: {f}")
    #         st.session_state["uploaded_files"].remove(files_and_names[f])
    #         uploaded_files = st.session_state["uploaded_files"]

    #         files_and_names.pop(f)
    #         current_file_names.remove(f)
    #         #file_list.remove(f)
    #         a += 1

    #     else:
    #         print("Error removing file:",f)
    #         #removed_files.remove(f)
    #         #removed_files.remove(f)
    #         a += 1    
    # #removed_files = []
    # save_document_list()

    
    #st.session_state["uploader_key"] += 1

    st.write("Removed files:",removed_files)
    st.write("Current file names:",current_file_names)
    st.write("Previous file names:",previous_file_names)
    st.write("Files and names:",files_and_names)
    

    print("Removed file names: ", removed_files)
    print("Current file names: ", current_file_names)
    print("Previous file names: ", previous_file_names)
    # Identify which files were removed by comparing lists
    for prev_file_name in previous_file_names:
        #print(prev_file_name)
        if prev_file_name not in current_file_names:
            if prev_file_name not in removed_files:
                removed_files.append(prev_file_name)
            
            try:
                remove_from_database(prev_file_name)
            except Exception as e:
                print("Error removing file:",e)

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
        context = ""
        num_results_counter = 0

        
        with st.spinner("Searching Database..."):
            for file in current_file_names:
                db = load_database(embeddings, os.path.join("DBs", file))
                results, results_text = query_database(prompt,db,num_responses=number_of_results)
                num_results_counter += len(results)
                print(type(results))
                context += "\n\n New Document Source:\n"+results_text+"\n\n"
            
        st.markdown(f"*Got {num_results_counter} results from database*")

        with st.spinner(f"Generating response..."):
            #print(len(context), type(context))
            #print(context)
            

            full_response = get_response(context,prompt,llm)
            message_placeholder.markdown(full_response)   
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)

