from typing import Optional
import os
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Carrega variáveis de ambiente
load_dotenv()

class DatabaseConnection:
    def __init__(self):
        self.connection_string = PGVector.connection_string_from_db_params(
            driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            database=os.environ.get("POSTGRES_DB", "vectordb"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres")
        )
        self.embeddings = OpenAIEmbeddings()
        
    def get_vector_store(self) -> PGVector:
        """
        Retorna uma instância do PGVector para busca de documentos.
        """
        return PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name="pdf_collection"
        )

class QASystem:
    def __init__(self):
        db = DatabaseConnection()
        vectorstore = db.get_vector_store()
        
        # Inicializa o modelo de linguagem
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        # Cria a chain de conversação
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=False
        )
        
        self.chat_history = []

    def ask(self, question: str) -> Optional[str]:
        """
        Faz uma pergunta ao sistema e retorna a resposta.
        """
        if not question:
            return None

        result = self.qa_chain({
            "question": question,
            "chat_history": self.chat_history
        })

        # Atualiza o histórico de chat
        self.chat_history.append((question, result["answer"]))
        
        return result["answer"]