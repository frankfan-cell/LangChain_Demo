from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


MIN_RELEVANCE_SCORE = 0.45

project_dir = Path(__file__).parent
db_path = project_dir / "chroma_db"

vector_store = Chroma(
    collection_name="library_rules",
    embedding_function=OllamaEmbeddings(model="bge-m3"),
    persist_directory=str(db_path),
)

test_cases = [
    ("周末的开放时间是什么？", "9:00 至 17:00"),
    ("逾期费怎么算？", "0.5 元"),
]

failed = 0

for question, expected_text in test_cases:
    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=3,
    )
    relevant_docs = [
        doc
        for doc, score in results
        if score >= MIN_RELEVANCE_SCORE
    ]
    retrieved_text = "\n".join(doc.page_content for doc in relevant_docs)
    passed = expected_text in retrieved_text

    print(f"[{'通过' if passed else '失败'}] {question}")
    if not passed:
        print(f"  预期包含：{expected_text}")
        print(f"  实际结果：\n{retrieved_text}")
        failed += 1

question = "图书馆附近有什么餐厅？"
results = vector_store.similarity_search_with_relevance_scores(question, k=3)
has_relevant_result = any(
    score >= MIN_RELEVANCE_SCORE
    for _, score in results
)

passed = not has_relevant_result
print(f"[{'通过' if passed else '失败'}] 无关问题应被阈值过滤：{question}")
if not passed:
    print("  检索结果中仍有分数达到阈值的片段。")
    failed += 1

print(f"\n结果：{len(test_cases) + 1 - failed}/{len(test_cases) + 1} 项通过")

if failed:
    raise SystemExit(1)