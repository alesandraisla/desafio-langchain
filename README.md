# PDF Question Answering System

Este é um sistema de pergunta e resposta baseado em documentos PDF usando LangChain e PostgreSQL com pgVector.

## Requisitos

- Python 3.8+
- Docker e Docker Compose
- OpenAI API Key

## Configuração

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
```
Edite o arquivo `.env` com suas configurações

4. Inicie o banco de dados:
```bash
docker-compose up -d
```

5. Execute a ingestão do PDF:
```bash
python ingest.py seu_arquivo.pdf
```

6. Para fazer perguntas sobre o documento:
```bash
python query.py
```

## Estrutura do Projeto

```
.
├── docker-compose.yml    # Configuração do Docker
├── ingest.py            # Script para processar PDFs
├── query.py             # Interface CLI para perguntas
├── database.py          # Configuração do banco de dados
├── requirements.txt     # Dependências Python
└── .env                 # Variáveis de ambiente
```# desafio-langchain
