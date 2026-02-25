"""
GraphRAG com PDF + LM Studio (Local) + Neo4j
Projeto de estudo - Processamento de PDFs com grafos de conhecimento
"""

import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from py2neo import Graph
import pandas as pd

load_dotenv()

st.set_page_config(
    page_title="GraphRAG Local com LM Studio",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GraphRAG Local (LM Studio)")
st.markdown("""
**Tudo rodando na sua máquina:**
- ✅ LLM: Modelo de chat carregado no LM Studio (ex: gemma-2-9b-it)
- ✅ Embeddings: Modelo de embedding carregado no LM Studio (ex: nomic-embed-text-v1.5)
- ✅ Zero custo, zero rate limit, zero dependência de internet
""")

with st.sidebar:
    st.header("🔌 Conexões")

    lm_studio_url = st.text_input(
        "LM Studio URL",
        value="http://localhost:1234/v1",
        help="URL do servidor local do LM Studio (padrão: http://localhost:1234/v1)"
    )
    modelo_llm = st.text_input(
        "Modelo LLM",
        value="gemma-2-9b-it",
        help="Nome do modelo de chat carregado no LM Studio"
    )
    modelo_embedding = st.text_input(
        "Modelo Embedding",
        value="nomic-embed-text-v1.5",
        help="Nome do modelo de embedding carregado no LM Studio"
    )

    neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = st.text_input("Neo4j Usuário", value=os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_pass = st.text_input("Neo4j Senha", type="password", value=os.getenv("NEO4J_PASSWORD", "senha123"))

    if st.button("🔗 Testar Conexões"):
        try:
            llm = ChatOpenAI(
                base_url=lm_studio_url,
                api_key="lm-studio",
                model=modelo_llm,
                temperature=0
            )
            st.success(f"✅ LM Studio OK (modelo: {modelo_llm})")
        except Exception as e:
            st.error(f"❌ LM Studio: {e}")

        if neo4j_uri and neo4j_user and neo4j_pass:
            try:
                graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_pass))
                graph.run("RETURN 1")
                st.success("✅ Neo4j OK")
            except Exception as e:
                st.error(f"❌ Neo4j: {e}")

def process_pdf(file_bytes, file_name: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    return "\n".join([doc.page_content for doc in docs])

def chunk_text(text: str, chunk_size=2000, chunk_overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def extract_graph_from_chunks(chunks: List[str], llm: ChatOpenAI) -> List[Any]:
    from langchain.schema import Document
    docs = [Document(page_content=chunk) for chunk in chunks]
    transformer = LLMGraphTransformer(llm=llm)
    return transformer.convert_to_graph_documents(docs)

def populate_neo4j(neo4j_uri: str, user: str, password: str, graph_docs: List[Any]):
    graph = Neo4jGraph(url=neo4j_uri, username=user, password=password)
    graph.add_graph_documents(graph_docs)
    return graph

def run_cypher(neo4j_uri: str, user: str, password: str, query: str) -> List[Dict]:
    graph = Graph(neo4j_uri, auth=(user, password))
    return graph.run(query).data()

tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "🕸️ Visualizar Grafo", "💬 Chat com o Grafo"])

with tab1:
    st.header("Carregue seu PDF")
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])

    if uploaded_file and lm_studio_url and modelo_llm and neo4j_uri and neo4j_user and neo4j_pass:
        if st.button("🚀 Processar PDF e Construir Grafo"):
            with st.spinner("📖 Lendo PDF..."):
                text = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
                st.success(f"✅ PDF carregado: {len(text)} caracteres")

            with st.spinner("✂️ Dividindo em chunks..."):
                chunks = chunk_text(text)
                st.info(f"📦 {len(chunks)} chunks criados")

            with st.spinner(f"🤖 Inicializando LM Studio com {modelo_llm}..."):
                llm = ChatOpenAI(
                    base_url=lm_studio_url,
                    api_key="lm-studio",
                    model=modelo_llm,
                    temperature=0
                )

            with st.spinner("🕸️ Extraindo entidades e relacionamentos (pode levar alguns minutos)..."):
                graph_docs = extract_graph_from_chunks(chunks, llm)
                total_nodes = sum(len(gd.nodes) for gd in graph_docs)
                total_rels = sum(len(gd.relationships) for gd in graph_docs)
                st.success(f"🔍 Extraídos {total_nodes} nós e {total_rels} relacionamentos")

            with st.spinner("💾 Populando Neo4j..."):
                populate_neo4j(neo4j_uri, neo4j_user, neo4j_pass, graph_docs)
                st.success("✅ Grafo armazenado com sucesso!")

            st.session_state["graph_built"] = True
            st.session_state["neo4j_uri"] = neo4j_uri
            st.session_state["neo4j_user"] = neo4j_user
            st.session_state["neo4j_pass"] = neo4j_pass
            st.session_state["lm_studio_url"] = lm_studio_url
            st.session_state["modelo_llm"] = modelo_llm

    elif uploaded_file and (not lm_studio_url or not modelo_llm or not neo4j_uri):
        st.warning("⚠️ Preencha as credenciais do LM Studio e Neo4j na barra lateral.")
    else:
        st.info("Faça upload de um PDF para começar.")

with tab2:
    st.header("Visualização do Grafo")
    if "graph_built" in st.session_state and st.session_state["graph_built"]:
        query_nodes = "MATCH (n) RETURN n LIMIT 30"
        nodes = run_cypher(
            st.session_state["neo4j_uri"],
            st.session_state["neo4j_user"],
            st.session_state["neo4j_pass"],
            query_nodes
        )
        if nodes:
            df_nodes = pd.DataFrame([
                {"id": n["n"].get("id", "N/A"), "labels": list(n["n"].labels) if n["n"].labels else ["Entity"]}
                for n in nodes
            ])
            st.subheader("📌 Nós (amostra)")
            st.dataframe(df_nodes)
        else:
            st.info("Nenhum nó encontrado.")

        query_rels = "MATCH (a)-[r]->(b) RETURN a.id, b.id, type(r) AS relationship LIMIT 30"
        rels = run_cypher(
            st.session_state["neo4j_uri"],
            st.session_state["neo4j_user"],
            st.session_state["neo4j_pass"],
            query_rels
        )
        if rels:
            st.subheader("🔗 Relacionamentos (amostra)")
            st.dataframe(pd.DataFrame(rels))
        else:
            st.info("Nenhum relacionamento encontrado.")
    else:
        st.info("Primeiro, processe um PDF na aba 'Upload'.")

with tab3:
    st.header("💬 Faça perguntas sobre o grafo")
    if "graph_built" in st.session_state and st.session_state["graph_built"]:
        if "qa_chain" not in st.session_state:
            with st.spinner("Preparando motor de perguntas..."):
                graph = Neo4jGraph(
                    url=st.session_state["neo4j_uri"],
                    username=st.session_state["neo4j_user"],
                    password=st.session_state["neo4j_pass"]
                )
                llm = ChatOpenAI(
                    base_url=st.session_state["lm_studio_url"],
                    api_key="lm-studio",
                    model=st.session_state["modelo_llm"],
                    temperature=0.1
                )

                CYPHER_TEMPLATE = """
                Task: Generate Cypher query to answer the question based on the graph schema.
                Schema:
                {schema}
                Question: {question}
                Return only the Cypher statement, no explanations.
                """
                cypher_prompt = PromptTemplate(
                    template=CYPHER_TEMPLATE,
                    input_variables=["schema", "question"]
                )

                chain = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=graph,
                    cypher_prompt=cypher_prompt,
                    verbose=True,
                    allow_dangerous_requests=True,
                    return_intermediate_steps=True
                )
                st.session_state["qa_chain"] = chain

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ex: Quais entidades estão relacionadas ao tema principal?"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Consultando o grafo..."):
                    try:
                        result = st.session_state["qa_chain"].invoke({"query": prompt})
                        answer = result["result"]
                        cypher = result.get("intermediate_steps", [])[0].get("query", "N/A") if result.get("intermediate_steps") else "N/A"

                        st.markdown(answer)
                        with st.expander("🔍 Ver Cypher gerado"):
                            st.code(cypher, language="cypher")
                    except Exception as e:
                        st.error(f"Erro: {e}")
                        answer = "Desculpe, não consegui responder."

                st.session_state["messages"].append({"role": "assistant", "content": answer})
    else:
        st.info("Primeiro, processe um PDF na aba 'Upload'.")
        