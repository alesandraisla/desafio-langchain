from database import QASystem

def main():
    print("Sistema de Perguntas e Respostas PDF")
    print("====================================")
    print("Digite 'sair' para encerrar o programa\n")

    qa_system = QASystem()

    while True:
        question = input("\nFaça sua pergunta: ").strip()
        
        if question.lower() == 'sair':
            print("\nAté logo!")
            break
        
        if not question:
            print("Por favor, faça uma pergunta.")
            continue
        
        try:
            answer = qa_system.ask(question)
            print("\nRESPOSTA:", answer)
            
        except Exception as e:
            print(f"\nErro ao processar sua pergunta: {str(e)}")
            print("Por favor, tente novamente.")

if __name__ == "__main__":
    main()