# imports
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

# environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
os.environ["LANGCHAIN_PROJECT"] = "llamachain"

# llm
# LLM = ChatOllama(base_url="...", model="llama3:70b", num_thread=96)
LLM = ChatOllama(model="llama3b")

all_splits = []
directory_path = "documents"

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if os.path.isfile(file_path):
        try:
            print(f"=== loading document: {file_path} ===")
            # 1. Load document
            loader = PyMuPDFLoader(file_path, extract_images=True)
            docs = loader.load()

            # 2. Split document
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits.extend(text_splitter.split_documents(docs))

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# 3. store documents
print("=== storing documents ===")
# oembed = OllamaEmbeddings(base_url="...", model="nomic-embed-text", num_thread=96)
oembed = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# 4. retrieve documents
retriever = vectorstore.as_retriever(search_type="similarity")

system_prompt = (
    "You are a helpful assistant. Answer my questions to your best ability and keep the responses to 500 words"
    "\n\n"
    "{context}"
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 5. RAG chain
question_answer_chain = create_stuff_documents_chain(LLM, final_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("=== generating output ===")
results = rag_chain.invoke({"input": "What is eternalblue"})
print(results['answer'])