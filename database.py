from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

def load_doc(path):
    doc_loader = PyPDFDirectoryLoader(path)
    return doc_loader.load()


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 60, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(documents)

#def add_to_chroma(chunks: list[Document]):
    #database = Chroma(persist_directory=)
def create_chunks(path, replace_newlines=False):
    document = load_doc(path)
    chunks = split_docs(document)
    if replace_newlines == True:
        for i in range(len(chunks)):
            chunks[i].page_content = chunks[i].page_content.replace("\n","")
        return chunks
    
    return chunks
    
def save_database(embeddings, chunks, path):
    #embeddings = OllamaEmbeddings(model="llama3")
    database = Chroma.from_documents(chunks,embeddings,persist_directory=path)
    database.persist()
    print(f"Saved {len(chunks)} chunks to Chroma")
