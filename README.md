# chatbot-assistant


# LLM-Powered Chatbot Assistant
A self-hosted AI chatbot powered by a large language model (LLM) designed to assist with various tasks such as answering questions, summarizing text, and engaging in conversations—similar to ChatGPT.

## Key Features:
### LLM-Powered:
Leverages large language models to provide intelligent and conversational responses.

### File Upload Support:
Upload PDF documents directly to the assistant, which can read, interpret, and respond based on the content.

### Contextual Memory with RAG:
Incorporates a Retrieval-Augmented Generation (RAG) system, storing and retrieving relevant information from files and conversations using Chroma databases to ensure context-aware interactions.


## Uses:
- Personal productivity assistant for research, note-taking, and file-based queries.
- Local AI chatbot experience
- Enhanced context for conversations through persistent storage and retrieval of files and previous interactions.

## Setup:
Clone the repo and open the folder via terminal or command prompt. 

Make sure to set an environment variable OPENAI_API_KEY to your API key in your system or in an environment.


then run the chatbot with the following command:

```streamlit run v2_main.py```

