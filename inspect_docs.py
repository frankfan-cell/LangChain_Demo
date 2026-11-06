from pathlib import Path

from langchain_core.documents import Document
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

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- 文本块 {index} | 来源：{chunk.metadata['source']} ---")
    print(chunk.page_content)