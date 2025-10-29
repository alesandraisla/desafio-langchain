import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

# Carrega variáveis de ambiente
load_dotenv()

def get_embeddings():
    """
    Retorna o modelo de embeddings configurado (OpenAI ou Gemini)
    """
    if os.getenv("USE_GEMINI", "false").lower() == "true":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_vector_store() -> PGVector:
    """
    Retorna uma instância configurada do PGVector.
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
    
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="pdf_collection"
    )

def search_documents(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """
    Busca documentos similares à query no banco de dados.
    Retorna uma lista de tuplas (texto, score).
    """
    vectorstore = get_vector_store()
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    return [(doc.page_content, score) for doc, score in results]

if __name__ == "__main__":
    # Exemplo de uso
    query = input("Digite sua busca: ")
    results = search_documents(query)
    
    print("\nResultados encontrados:")
    for text, score in results:
        print(f"\nScore: {score:.4f}")
        print("Texto:", text)
        print("-" * 80)