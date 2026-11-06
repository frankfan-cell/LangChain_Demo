from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


project_dir = Path(__file__).parent
document_path = project_dir / "data" / "knowledge.md"

document = Document(
    page_content=document_path.read_text(encoding="utf-8"),
    metadata={"source": "data/knowledge.md"},
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=10,
)
chunks = splitter.split_documents([document])

embeddings = OllamaEmbeddings(model="bge-m3")
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)

question = "每位读者最多能借几本书？"
results = vector_store.similarity_search(question, k=2)

for index, result in enumerate(results, start=1):
    print(f"\n--- 检索结果 {index} | 来源：{result.metadata['source']} ---")
    print(result.page_content)