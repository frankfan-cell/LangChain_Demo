# LangChain 图书馆知识库问答

这是一个用于学习 LangChain 的中文 RAG（检索增强生成）示例。程序读取 Markdown 和可提取文本的 PDF，将资料切分、向量化并存入 Chroma；用户提问时检索相关内容，再由聊天模型根据资料生成回答，并展示来源。

## 功能

- 读取 `data/` 下的 Markdown 和 PDF 文件。
- 将内容转换为 LangChain `Document`，切分为文本块。
- 使用 Ollama 的 `bge-m3` 生成嵌入向量。
- 使用 Chroma 持久化索引，并通过资料指纹检测源文件变化。
- 按相关度分数过滤检索结果，当前阈值为 `0.45`。
- 使用 OpenAI 兼容接口的聊天模型回答问题。
- 保留最近三轮对话，并将依赖上下文的追问改写为独立检索问题。
- 展示来源文件；PDF 引用会显示页码。
- 提供检索测试和中文 PDF 样本生成脚本。

## 工作流程

```text
Markdown / PDF
      |
      v
读取为 Document（正文 + 来源/页码元数据）
      |
      v
RecursiveCharacterTextSplitter 切分
      |
      v
Ollama bge-m3 生成向量
      |
      v
Chroma 持久化与语义检索
      |
      v
相关度过滤 -> 提示词与聊天模型 -> 回答及来源
```

多轮对话中的追问会先经过查询改写链，再将改写后的问题交给检索器。

## 项目结构

```text
LangChain_demo/
├── data/
│   ├── knowledge.md
│   ├── opening_hours.md
│   └── library_rules.pdf
├── chroma_db/              # 程序生成的 Chroma 向量索引
├── main.py                 # 最小聊天模型调用示例
├── inspect_docs.py         # 查看文档切分结果
├── search_docs.py          # 内存向量库检索示例
├── qa_docs.py              # 完整 RAG 命令行程序
├── test_retrieval.py       # 检索行为测试
└── create_sample_pdf.py    # 生成并验证中文 PDF 样本
```

## 环境准备

需要 Python、Ollama，以及一个可通过 OpenAI 兼容接口访问的聊天模型。项目使用 Ollama 的 `bge-m3` 生成嵌入向量。

在 Windows PowerShell 或 CMD 的项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install langchain-core langchain-openai langchain-text-splitters langchain-ollama langchain-chroma pypdf python-dotenv numpy reportlab
```

拉取嵌入模型：

```powershell
ollama pull bge-m3
```

确认 Ollama 可用：

```powershell
ollama list
```

在项目根目录创建 `.env` 文件：

```dotenv
DEEPSEEK_API_KEY=你的模型服务API_KEY
```

`.env` 已加入 `.gitignore`，不要将 API Key 提交到代码仓库。

`main.py` 和 `qa_docs.py` 使用 `ChatOpenAI` 连接 OpenAI 兼容服务。若要使用其他模型服务，请按实际情况调整脚本中的 `model`、`base_url` 和 API Key 配置。当前示例的 `DEEPSEEK_API_KEY` 是环境变量名，具体服务由 `base_url` 决定。

## 运行

在项目根目录运行以下命令：

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe inspect_docs.py
.\.venv\Scripts\python.exe search_docs.py
.\.venv\Scripts\python.exe create_sample_pdf.py
.\.venv\Scripts\python.exe qa_docs.py
```

问答程序启动后输入问题；输入 `退出`、`exit` 或 `quit` 结束。

运行检索测试：

```powershell
.\.venv\Scripts\python.exe test_retrieval.py
```

当前测试覆盖两个有答案的问题，以及一个无关问题是否会被相关度阈值过滤。测试只检查检索结果，不调用聊天模型。

## LangChain 知识点梳理

### 1. 聊天模型与提示词

`ChatOpenAI` 通过 OpenAI 兼容接口调用聊天模型。`ChatPromptTemplate` 将系统指令、对话历史和当前问题组织成模型输入。

### 2. Document 与元数据

LangChain 使用 `Document` 统一表示文档：`page_content` 保存正文，`metadata` 保存来源路径、PDF 页码等信息。检索后可以用这些元数据展示引用来源。

### 3. 文档加载

Markdown 由 Python 读取；PDF 由 `pypdf.PdfReader` 逐页提取文本，并包装为 `Document`。当前方式适用于包含可提取文本的 PDF；扫描图片版 PDF 需要 OCR，本项目未包含 OCR。

### 4. 文本切分

`RecursiveCharacterTextSplitter` 使用 `chunk_size=60`、`chunk_overlap=10`。较小块便于观察切分和检索；相邻块的重叠能降低切分边界造成上下文丢失的概率。实际项目应结合文档和检索效果调整参数。

### 5. 嵌入向量

`OllamaEmbeddings(model="bge-m3")` 将文本转换为向量。问题和文本块的向量用于语义相似度检索；嵌入模型与负责生成回答的聊天模型可以不同。

### 6. 向量存储与索引更新

`Chroma` 将向量持久化到 `chroma_db/`。程序对资料路径和内容计算 SHA-256 指纹；资料集合变化时删除旧索引并重建，否则复用已有索引。当前策略会在任一资料变化时重建整个资料集。

### 7. 检索与相关度阈值

`similarity_search_with_relevance_scores` 返回候选文本块和相关度分数。程序最多取 `k=3` 个结果，再用 `0.45` 过滤低相关结果。这个阈值是本示例根据测试数据选取的参数，并非通用标准；更换资料或嵌入模型后应重新测试。

### 8. LCEL 链

`prompt | model` 使用 LangChain Expression Language（LCEL）将提示模板和聊天模型组合成可调用的 Runnable 链，通过 `invoke()` 输入变量并取得结果。

### 9. 对话历史与查询改写

`MessagesPlaceholder` 将历史消息插入提示词。程序只保留最近三轮对话，历史仅保存在内存中，程序退出后会清空。存在历史时，查询改写链先将简短追问改写成独立检索问题；回答链再结合检索内容回答。

### 10. 来源与 PDF 页码

回答来源由检索到的 `Document.metadata` 生成。`pypdf` 页码从 `0` 开始，程序展示时加 `1`，转换成从第 1 页开始的用户习惯编号。

## 当前范围与限制

这是一个命令行学习项目，不包含 Agent、工具调用、Web 界面或持久化对话历史。PDF 文本提取不包含 OCR；向量检索阈值需要结合实际数据校准。修改任一资料文件后，程序会重建整个向量索引。
