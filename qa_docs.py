from pypdf import PdfReader
import hashlib
from langchain_core.messages import AIMessage, HumanMessage
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


project_dir = Path(__file__).parent
data_dir = project_dir / "data"
markdown_files = sorted(data_dir.rglob("*.md"))
pdf_files = sorted(data_dir.rglob("*.pdf"))

if not markdown_files and not pdf_files:
    raise FileNotFoundError(f"目录中没有 Markdown 或 PDF 文件：{data_dir}")

documents = []
corpus_digest = hashlib.sha256()
corpus_digest.update(b"pdf-parser:pypdf-v1\0")

for file_path in markdown_files:
    source_name = file_path.relative_to(project_dir).as_posix()
    content = file_path.read_text(encoding="utf-8")

    corpus_digest.update(source_name.encode("utf-8"))
    corpus_digest.update(b"\0")
    corpus_digest.update(content.encode("utf-8"))
    corpus_digest.update(b"\0")

    documents.append(Document(
        page_content=content,
        metadata={"source": source_name},
    ))

for file_path in pdf_files:
    source_name = file_path.relative_to(project_dir).as_posix()
    pdf_bytes = file_path.read_bytes()

    corpus_digest.update(source_name.encode("utf-8"))
    corpus_digest.update(b"\0")
    corpus_digest.update(pdf_bytes)
    corpus_digest.update(b"\0")

    reader = PdfReader(str(file_path))
    
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
    
        documents.append(Document(
            page_content=page_text,
            metadata={
                "source": source_name,
                "page": page_number,
            },
        ))

corpus_hash = corpus_digest.hexdigest()
for document in documents:
    document.metadata["corpus_hash"] = corpus_hash
for document in documents:
    document.metadata["corpus_hash"] = corpus_hash

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10,
)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="bge-m3")
db_path = project_dir / "chroma_db"

vector_store = Chroma(
    collection_name="library_rules",
    embedding_function=embeddings,
    persist_directory=str(db_path),
)

existing_data = vector_store.get()
existing_ids = existing_data["ids"]
existing_metadata = existing_data["metadatas"] or []

index_is_current = (
    bool(existing_ids)
    and all(
        metadata.get("corpus_hash") == corpus_hash
        for metadata in existing_metadata
    )
)

if index_is_current:
    print("已加载磁盘中的向量索引。")
else:
    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(chunks)
    print("资料索引已创建或更新。")

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("请先在 .env 文件中设置 DEEPSEEK_API_KEY")

model = ChatOpenAI(
    model="qwen3.7-plus",
    base_url="http://10.11.7.197:32300/v1",
    api_key=api_key,
    temperature=0,
)

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "将用户当前问题改写成脱离对话也能理解的独立检索问题。"
        "只改写问题，不要回答，也不要添加用户没提过的事实。",
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "当前问题：{question}"),
])

rewrite_chain = rewrite_prompt | model

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是图书馆规则问答助手。请只根据提供的资料回答。"
        "如果资料中没有答案，请明确说明资料中没有相关信息，不要猜测。",
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "资料：\n{context}\n\n当前问题：{question}",
    ),
])

chain = prompt | model
history = []
min_relevance_score = 0.45

print("图书馆知识库问答已启动，输入“退出”结束。")

while True:
    question = input("\n请输入问题：").strip()

    if question.lower() in {"退出", "exit", "quit"}:
        print("问答结束。")
        break

    if not question:
        continue

    retrieval_question = question
    
    if history:
        rewritten = rewrite_chain.invoke({
            "history": history,
            "question": question,
        })
        retrieval_question = rewritten.content.strip()
        print(f"检索问题：{retrieval_question}")
    
    retrieved_with_scores = (
        vector_store.similarity_search_with_relevance_scores(
            retrieval_question,
            k=3,
        )
    )
    
    relevant_results = [
        (doc, score)
        for doc, score in retrieved_with_scores
        if score >= min_relevance_score
    ]
    
    if not relevant_results:
        print("\n资料中没有找到足够相关的信息。")
        continue
    
    retrieved_docs = []
    
    for index, (doc, score) in enumerate(relevant_results, start=1):
        retrieved_docs.append(doc)
        print(
            f"\n--- 检索片段 {index} | 相关度：{score:.3f} "
            f"| 来源：{doc.metadata.get('source')} ---"
        )
        print(doc.page_content)
    
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = []
    
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page")
    
        if page is None:
            citation = source
        else:
            citation = f"{source}（第 {page + 1} 页）"
    
        if citation not in sources:
            sources.append(citation)

    response = chain.invoke({
        "history": history,
        "context": context,
        "question": question,
    })

    print(f"\n回答：\n{response.content}")
    print(f"参考来源：{'、'.join(sources)}")

    history.extend([
        HumanMessage(content=question),
        AIMessage(content=response.content),
    ])

    history = history[-6:]  # 保留最近的三轮对话，每轮包含用户和AI消息各一条