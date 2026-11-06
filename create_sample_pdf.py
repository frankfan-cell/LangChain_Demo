from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


project_dir = Path(__file__).parent
output_path = project_dir / "data" / "library_rules.pdf"
output_path.parent.mkdir(parents=True, exist_ok=True)

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
page = canvas.Canvas(str(output_path), pagesize=A4)
page_width, page_height = A4
page.setTitle("图书馆借阅与开放规则")
page.setFont("STSong-Light", 18)
page.drawString(72, page_height - 72, "图书馆借阅与开放规则")
page.setFont("STSong-Light", 12)

lines = [
    "借阅规则",
    "每位读者最多可同时借阅 5 本书。",
    "借阅期限为 21 天。",
    "每本书可续借一次，续借期限为 21 天。",
    "如果该书已被其他读者预约，则不能续借。",
    "逾期归还的图书，每本每天收取 0.5 元逾期费。",
    "",
    "开放时间",
    "工作日开放时间为 8:00 至 20:00。",
    "周末开放时间为 9:00 至 17:00。",
    "",
    "本资料用于 LangChain PDF 加载与检索学习。",
]

y = page_height - 112
for line in lines:
    page.drawString(72, y, line)
    y -= 28

page.save()

reader = PdfReader(str(output_path))
extracted_text = "\n".join(item.extract_text() or "" for item in reader.pages)
print(f"已生成：{output_path}")
print(f"页数：{len(reader.pages)}")
print("文本提取验证：")
print(extracted_text)

if "周末开放时间为 9:00 至 17:00" not in extracted_text:
    raise SystemExit("PDF 文本验证失败")
