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
document_path = project_dir / "data" / "knowledge.md"

source_name = "data/knowledge.md"
content = document_path.read_text(encoding="utf-8")
content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

document = Document(
    page_content=content,
    metadata={
        "source": source_name,
        "content_hash": content_hash,
    },
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10,
)
chunks = splitter.split_documents([document])

embeddings = OllamaEmbeddings(model="bge-m3")
db_path = project_dir / "chroma_db"

vector_store = Chroma(
    collection_name="library_rules",
    embedding_function=embeddings,
    persist_directory=str(db_path),
)

existing_data = vector_store.get(where={"source": source_name})
existing_ids = existing_data["ids"]
existing_metadata = existing_data["metadatas"] or []

index_is_current = (
    bool(existing_ids)
    and all(
        metadata.get("content_hash") == content_hash
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

print("图书馆知识库问答已启动，输入“退出”结束。")

while True:
    question = input("\n请输入问题：").strip()

    if question.lower() in {"退出", "exit", "quit"}:
        print("问答结束。")
        break

    if not question:
        continue

    retrieved_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "未知来源")
        for doc in retrieved_docs
    ))

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