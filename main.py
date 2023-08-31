import os

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores import Chroma
from langchain.vectorstores import MyScale
from langchain.vectorstores import USearch
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.html'


def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf ',
        is_local: bool = False,
        question: str = 'Archivo HTML'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), UnstructuredHTMLLoader(f"./{default_doc_name}") if not is_local \
        else UnstructuredHTMLLoader(path)

    doc = loader.load_and_split()

    print(doc[-1])

    #db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())
    #db = MyScale.from_documents(doc, embedding=OpenAIEmbeddings())
    #db = USearch.from_documents(doc, embedding=OpenAIEmbeddings())
    db = FAISS.from_documents(doc, embedding=OpenAIEmbeddings())

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='map_reduce', retriever=db.as_retriever())

    st.write(qa.run(question))
    # print(qa.run(question))


def client():
    st.title('Manage LLM with LangChain')
    uploader = st.file_uploader('Upload HTML', type='html')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('HTML saved!!')

    question = st.text_input('Ejemplo: Dame un resumen del documento',
                             placeholder='Give response about your HTML ', disabled=not uploader)

    if st.button('Send Question'):
        if uploader:
            process_doc(
                path=default_doc_name,
                is_local=True,
                question=question
            )
        else:
            st.info('Loading default PDF')
            process_doc()


if __name__ == '__main__':
    client()
    # process_doc()

