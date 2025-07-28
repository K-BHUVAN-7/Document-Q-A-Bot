from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

#LOADING THE DOCUMENTS

loader = PyPDFLoader("details.pdf")
document = loader.load()

#SPLITTING THE DATA INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
data = text_splitter.split_documents(document)

#LOADING INTO DATABASE

db = FAISS.from_documents(data, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

retriever = db.as_retriever()

prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":512}),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
