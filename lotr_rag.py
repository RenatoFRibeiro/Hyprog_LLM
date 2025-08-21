#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOTROracle: um RAG simples em Python para fazer perguntas sobre o universo de Tolkien
- Ingestão de PDFs/TXT
- Indexação com ChromaDB (vetores)
- Geração com LLM local via Ollama (ex.: llama3.1)
Sem necessidade de chaves de API.

Uso rápido:
    # 1) Ingestão (cria o índice vetorial a partir dos ficheiros em ./data)
    python lotr_rag.py --ingest ./data

    # 2) Chat interativo sobre o conteúdo
    python lotr_rag.py --chat

    # 3) Pergunta única via CLI
    python lotr_rag.py --ask "Quem é Tom Bombadil?"
"""

import argparse
import os
import sys
from typing import List

# --- LangChain & afins ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# Para tipagem dos documentos
from langchain.schema import Document

os.environ["ANONYMIZED_TELEMETRY"] = "false"
PERSIST_DIR = os.environ.get("LOTRO_RAG_PERSIST_DIR", "./chroma_lotr")
EMBED_MODEL = os.environ.get("LOTRO_RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LOTRO_RAG_LLM_MODEL", "llama3.1:8b")  # pode ajustar para :70b se tiver máquina potente
CHUNK_SIZE = int(os.environ.get("LOTRO_RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("LOTRO_RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.environ.get("LOTRO_RAG_TOP_K", "4"))


def gather_files(data_dir: str) -> List[str]:
    """Lista todos os PDFs e TXTs de forma recursiva."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".pdf") or f.lower().endswith(".txt"):
                paths.append(os.path.join(root, f))
    if not paths:
        print(f"[!] Não encontrei PDFs/TXTs em: {data_dir}", file=sys.stderr)
    return paths


def load_documents(paths: List[str]) -> List[Document]:
    """Carrega ficheiros em Document(s) com metadados úteis (ex.: página)."""
    docs: List[Document] = []
    for p in paths:
        if p.lower().endswith(".pdf"):
            loader = PyPDFLoader(p)
            pdf_docs = loader.load()  # um por página
            for d in pdf_docs:
                # Normalizar info da fonte
                d.metadata["source"] = os.path.basename(p)
                d.metadata["page"] = d.metadata.get("page", None)
            docs.extend(pdf_docs)
        else:  # .txt
            # encoding='utf-8' evita erros com acentos
            loader = TextLoader(p, encoding="utf-8")
            txt_docs = loader.load()
            for d in txt_docs:
                d.metadata["source"] = os.path.basename(p)
                # TXT não tem número de página, mas podemos marcar como 1
                d.metadata["page"] = d.metadata.get("page", 1)
            docs.extend(txt_docs)
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Cria chunks para melhorar a recuperação sem partir frases no meio."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(splits: List[Document]) -> Chroma:
    """Cria/persiste a base vetorial com embeddings locais via Ollama."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vs.persist()
    return vs


def load_vectorstore() -> Chroma:
    """Carrega uma base vetorial já existente."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def format_docs_with_citations(docs: List[Document]) -> str:
    """Concatena texto dos docs e anexa tags de citação [fonte: ficheiro, pág X]."""
    formatted = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "desconhecido")
        page = d.metadata.get("page", "N/A")
        formatted.append(f"[{i}] (fonte: {source}, pág {page})\n{d.page_content}")
    return "\n\n".join(formatted)


def unique_sources(docs: List[Document]) -> List[str]:
    """Lista fontes únicas para mostrar ao utilizador."""
    seen = set()
    out = []
    for d in docs:
        src = d.metadata.get("source", "desconhecido")
        page = d.metadata.get("page", "N/A")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            out.append(f"{src} (pág {page})")
    return out


def make_llm():
    """Instancia o modelo de chat local via Ollama."""
    # Pode passar parâmetros como temperature, num_ctx, top_p, etc.
    return ChatOllama(model=LLM_MODEL, temperature=0.2)


SYSTEM_PROMPT = (
    "És um assistente especializado no legendário mundo de J.R.R. Tolkien "
    "(O Senhor dos Anéis, O Hobbit, O Silmarillion, Contos Inacabados, etc.). "
    "Responde sempre em PT-PT, de forma objectiva e factual.\n"
    "REGRAS:\n"
    "1) Usa APENAS o 'Contexto' fornecido. Se a resposta não estiver no contexto, diz claramente que não sabes.\n"
    "2) Inclui referências às fontes entre parêntesis rectos, por ex.: [fonte: livro, pág X].\n"
    "3) Evita spoilers muito explícitos, a não ser que o utilizador peça. Se fores dar spoilers, avisa.\n"
)


def answer_question(question: str, top_k: int = TOP_K) -> str:
    """Recupera contexto e pergunta ao LLM para responder, com citações."""
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "Não encontrei informação relevante nos documentos indexados."

    context = format_docs_with_citations(docs)
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Pergunta: {question}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Resposta (em PT-PT, com referências):"
    )

    llm = make_llm()
    # Chamada simples (sem streaming) para manter o script compacto
    resp = llm.invoke(prompt)
    # Também mostramos as fontes recuperadas (qualquer que seja a formatação do LLM)
    fontes = ", ".join(unique_sources(docs))
    return f"{resp.content}\n\nFontes consultadas: {fontes}"


def ingest_command(data_dir: str):
    print(f"[+] A indexar ficheiros de: {data_dir}")
    paths = gather_files(data_dir)
    docs = load_documents(paths)
    print(f"[+] {len(docs)} documento(s)/página(s) carregado(s). A criar chunks...")
    splits = split_documents(docs)
    print(f"[+] {len(splits)} chunk(s) gerado(s). A criar embeddings e a persistir ChromaDB em: {PERSIST_DIR}")
    build_vectorstore(splits)
    print("[✓] Ingestão concluída.")


def chat_loop():
    print("=== Chat LOTROracle (Ctrl+C para sair) ===")
    print("Dica: as respostas serão baseadas apenas no que foi indexado.")
    while True:
        try:
            q = input("\nPergunta> ").strip()
            if not q:
                continue
            ans = answer_question(q)
            print("\n" + "-" * 80 + "\n")
            print(ans)
            print("\n" + "-" * 80 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nAté à próxima!")
            break


def main():
    parser = argparse.ArgumentParser(description="RAG local sobre o universo de Tolkien (PDF/TXT).")
    parser.add_argument("--ingest", metavar="DIR", help="Pasta com PDFs/TXT para indexar (recursivo).")
    parser.add_argument("--ask", metavar="PERGUNTA", help="Faz uma pergunta única e imprime a resposta.")
    parser.add_argument("--chat", action="store_true", help="Inicia um chat interativo em terminal.")
    args = parser.parse_args()

    # Caminho padrão se não for fornecido
    data_dir = args.ingest or "./dataset"

    if not os.path.exists(data_dir):
        print(f"[!] Pasta '{data_dir}' não encontrada. Cria a pasta ou indica outra com --ingest <pasta>.", file=sys.stderr)
        sys.exit(1)

    ingest_command(data_dir)

    # Se só quis ingerir, e nada mais, termina aqui
    if args.ask is None and not args.chat and args.ingest:
        return

    # Garantir que existe índice antes de perguntar
    if not os.path.isdir(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print(f"[!] Índice não encontrado em {PERSIST_DIR}. Primeiro faça ingestão com --ingest.", file=sys.stderr)
        sys.exit(1)

    if args.ask:
        print(answer_question(args.ask))
    elif args.chat:
        chat_loop()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
