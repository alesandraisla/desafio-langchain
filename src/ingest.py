import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain.docstore.document import Document

# Carrega variáveis de ambiente
load_dotenv()

# Configurações
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def get_embeddings():
    """
    Retorna o modelo de embeddings configurado (OpenAI ou Gemini)
    """
    if os.getenv("USE_GEMINI", "false").lower() == "true":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return OpenAIEmbeddings(model="text-embedding-3-small")

def process_pdf(pdf_path: str) -> List[Document]:
    """
    Processa um arquivo PDF e retorna uma lista de documentos divididos.
    """
    # Carrega o PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Divide o texto em chunks menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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

    embeddings = get_embeddings()

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