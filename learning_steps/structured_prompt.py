import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from pydantic_basics import InvestigationQuery     # 重複使用 M2.1 定義的工單格式

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

# 用 prompt「告訴」LLM 要回傳什麼格式
INSTRUCTIONS = """你是製造業設備調查助理。請把工程師的問題轉成 JSON，只輸出 JSON。
欄位：
- machine_id：機台編號，格式為 M-XX，例如「三號機」寫成 "M-03"
- days：查詢最近幾天（整數）。例如上週 = 7、最近一個月 = 30、兩週 = 14
- intent：只能是 "failure_count"、"failure_reason"、"repair_history" 其中之一
範例輸出：{"machine_id": "M-03", "days": 7, "intent": "failure_reason"}

工程師的問題："""

questions = [
    "M-03 上週為什麼一直停機？",
    "三號機最近一個月壞了幾次？",
    "M-05 的維修紀錄給我看一下，最近兩週就好",
]

for question in questions:
    print("\n問題：", question)
    response = client.models.generate_content(model=MODEL, contents=INSTRUCTIONS +question)   # 填空 1
    raw_text = response.text
    print("LLM 原始輸出：", repr(raw_text))

    try:
        data =   json.loads(raw_text)                   # 填空 2：第 1 層，文字 → dict
        query =  InvestigationQuery(**data)                   # 填空 3：第 2 層，dict → InvestigationQuery
        print("✅ 驗證通過：", query)
    except json.JSONDecodeError as error:
        print("❌ 第 1 層失敗（不是合法的 JSON）：", error)
    except ValidationError as error:
        print("❌ 第 2 層失敗（Pydantic）：", error.errors()[0]["msg"])
