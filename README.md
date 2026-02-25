# GraphRAG com PDF, LM Studio e Neo4j

Projeto de estudo que implementa um pipeline de GraphRAG (Graph-based Retrieval-Augmented Generation) usando:

- **LM Studio** para LLM local (API compatível com OpenAI)
- **Neo4j** como banco de grafos
- **Streamlit** para interface interativa

## Funcionalidades

1. Upload de PDF
2. Extração de entidades e relacionamentos usando LLM
3. Armazenamento no Neo4j
4. Visualização dos dados em tabelas
5. Chat com perguntas sobre o grafo

## Como executar

### Pré-requisitos
- Python 3.10+
- Docker (para Neo4j) ou Neo4j instalado localmente
- LM Studio com um modelo de chat carregado (ex: gemma-2-9b-it)

### Passos

1. Clone o repositório
2. Crie e ative um ambiente virtual
3. Instale as dependências: `pip install -r requirements.txt`
4. Configure o Neo4j:

   ```bash
   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/senha123 -e NEO4J_PLUGINS='["apoc"]' -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' neo4j:5-community
   ```

### Inicie o LM Studio e ative o servidor local

### Execute o app: streamlit run app.py
