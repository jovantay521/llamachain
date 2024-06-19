# llamachain
## Description
Retreival Augmented Generation (RAG) application using langchain and llama3

## Getting Started
### Pre-requisites
- [ollama](https://ollama.com/download)
- [llama3](https://ollama.com/library/llama3)
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

### Installation
Clone this repository
```sh
git clone https://github.com/jovantay521/llamachain.git
```

Install Python 3.12.x ([pyenv installation](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation))
```sh
pyenv install 3.12
```
```sh
cd piggy
pyenv local 3.12
```
Set up virtual environment
```sh
python -m venv .venv
source .venv/bin/activate
```
Install required packages
```sh
pip install -r requirements.txt
```

### Updating environment variables
#### Required
Sign up for a [langsmith account](https://smith.langchain.com/) and generate an API key  

Replace the environment variable in main.py
```py
os.environ["LANGCHAIN_API_KEY"] = "..."
```

#### Optional
If LLM Model is on a different host, update the base url
```python
LLM = ChatOllama(base_url="...", model="llama3")
```

If using a different LLM model(e.g. llama3:70b), update the model variable
```py
LLM = ChatOllama(model="llama3:70b")
```

If your system has spare cpu threads, it is recommended to allocate them to ollama for faster generation of results
```py
LLM = ChatOllama(model="llama3", num_thread="...")
```

If using a different [embedding model](https://python.langchain.com/v0.2/docs/how_to/#embedding-models)
```py
oembed = OllamaEmbedding(model="...")
```

## Usage
main.py parses documents into the LLM which will be used to generate a response

Upload relevant documents to the documents directory
```sh
mv /path/to/documents ./documents
```

Run the program
```sh
python main.py
```

## Built using
- LLM: ollama, llama3, nomic-embed-text
- Document Loader: PyMuPDFReader, RapidOCR 
- Vectorstore: ChromaDB
- Langchain/Langsmith

## Credits
- Jovan (@jovantay521)
- Ziyang (@Zycranny)