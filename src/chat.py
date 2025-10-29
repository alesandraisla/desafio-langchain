import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from search import search_documents

# Carrega variáveis de ambiente
load_dotenv()

# Template do prompt
PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def get_llm():
    """
    Retorna o modelo de linguagem configurado (OpenAI ou Gemini)
    """
    if os.getenv("USE_GEMINI", "false").lower() == "true":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    return ChatOpenAI(model="gpt-5-nano", temperature=0)

def format_context(results: List[tuple]) -> str:
    """
    Formata os resultados da busca em um contexto único.
    """
    return "\n\n".join([text for text, _ in results])

def main():
    print("Sistema de Perguntas e Respostas PDF")
    print("====================================")
    print("Digite 'sair' para encerrar o programa\n")

    # Inicializa o modelo
    llm = get_llm()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    while True:
        question = input("\nFaça sua pergunta: ").strip()
        
        if question.lower() == 'sair':
            print("\nAté logo!")
            break
        
        if not question:
            print("Por favor, faça uma pergunta.")
            continue
        
        try:
            # Busca documentos relevantes
            results = search_documents(question, k=10)
            
            if not results:
                print("\nRESPOSTA: Não encontrei nenhuma informação relevante para responder sua pergunta.")
                continue

            # Formata o contexto
            context = format_context(results)
            
            # Gera a resposta
            chain = prompt | llm
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            print("\nRESPOSTA:", response.content)
            
        except Exception as e:
            print(f"\nErro ao processar sua pergunta: {str(e)}")
            print("Por favor, tente novamente.")

if __name__ == "__main__":
    main()