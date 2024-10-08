from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

#from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from langchain.text_splitter import TokenTextSplitter

import pdfplumber


def load_doc(path):
    doc_loader = PyPDFDirectoryLoader(path)
    return doc_loader.load()


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 100, length_function = len, is_separator_regex  = False)
    #text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=60)
    #documents = text_splitter.split_documents(documents)
    return text_splitter.split_documents(documents)

def extract_pdf_text(uploaded_file):
    total = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            total += page.extract_text()

    return total


def create_chunks(document,replace_newlines=False):#path, replace_newlines=False):
    chunks = split_docs(document)
    if replace_newlines == True:
        for i in range(len(chunks)):
            chunks[i].page_content = chunks[i].page_content.replace("\n","")
        return chunks
    
    return chunks


def save_database(embeddings, chunks, path):    
    database = Chroma.from_documents(chunks,embeddings,persist_directory=path)
    print(f"Saved {len(chunks)} chunks to Chroma")

def load_database(embeddings, path):
    database = Chroma(persist_directory=path,embedding_function=embeddings)
    return database


def query_database(query, database, num_responses = 20, similarity_threshold = 0.4):
    results = database.similarity_search_with_relevance_scores(query,k=num_responses)
    results_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    try:
        if results[0][1] < similarity_threshold:
            print("Could not find results")
    except:
        print("Error")
    return results, results_text