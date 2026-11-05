import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types                      # ⭐ 新增：設定用的 class
from pydantic import ValidationError

from pydantic_basics import InvestigationQuery

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

# ⭐ prompt 變短：格式交給 API，這裡只留下業務規則
INSTRUCTIONS = """你是製造業設備調查助理，請解析工程師的問題。
規則：機台編號寫成 M-XX（例如三號機 = M-03）；上週 = 7 天、兩週 = 14 天、一個月 = 30 天。
工程師的問題："""

questions = [
    "M-03 上週為什麼一直停機？",
    "三號機最近一個月壞了幾次？",
    "M-05 的維修紀錄給我看一下，最近兩週就好",
]

for question in questions:
    print("\n問題：", question)

    response = client.models.generate_content(
        model=MODEL,
        contents=INSTRUCTIONS + question,
        config=types.GenerateContentConfig(          # ⭐ 新增：把格式交給 API
            response_mime_type="application/json",
            response_schema=InvestigationQuery,                    # 填空：要照哪一份「規格書」？
        ),
    )
    
    raw_text = response.text
    print("LLM 原始輸出：", repr(raw_text))

    # 驗證：和 structured_prompt.py 完全一樣，不能省
    try:
        data = json.loads(raw_text)
        query = InvestigationQuery(**data)
        print("✅ 驗證通過：", query)
    except json.JSONDecodeError as error:
        print("❌ 第 1 層失敗（不是合法的 JSON）：", error)
    except ValidationError as error:
        print("❌ 第 2 層失敗（Pydantic）：", error.errors()[0]["msg"])
