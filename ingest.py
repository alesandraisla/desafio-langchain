import os
from typing import List
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document

# Carrega variáveis de ambiente
load_dotenv()

def process_pdf(pdf_path: str) -> List[Document]:
    """
    Processa um arquivo PDF e retorna uma lista de documentos divididos.
    """
    # Carrega o PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Divide o texto em chunks menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    return text_splitter.split_documents(documents)

def store_documents(documents: List[Document]):
    """
    Armazena os documentos no PostgreSQL usando pgvector.
    """
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "vectordb"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres")
    )

    # Inicializa embeddings
    embeddings = OpenAIEmbeddings()

    # Cria collection no banco de dados
    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="pdf_collection",
        connection_string=CONNECTION_STRING,
    )

def main():
    import sys
    if len(sys.argv) != 2:
        print("Uso: python ingest.py caminho/para/seu/arquivo.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Erro: O arquivo {pdf_path} não existe.")
        sys.exit(1)

    print("Processando PDF...")
    documents = process_pdf(pdf_path)
    print(f"Documento dividido em {len(documents)} partes.")

    print("Armazenando no banco de dados...")
    store_documents(documents)
    print("Documento processado e armazenado com sucesso!")

if __name__ == "__main__":
    main()