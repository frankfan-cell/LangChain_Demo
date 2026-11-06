import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


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

response = model.invoke("请用一句话介绍 LangChain。")
print(response.content)