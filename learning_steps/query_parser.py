"""把工程師的自然語言問題，解析成經過驗證的 InvestigationQuery。"""

import json
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from pydantic_basics import InvestigationQuery

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

INSTRUCTIONS = """你是製造業設備調查助理，請解析工程師的問題。
規則：機台編號寫成 M-XX（例如三號機 = M-03）；上週 = 7 天、兩週 = 14 天、一個月 = 30 天。
工程師的問題："""


def parse_question(question):
    """把工程師的問題解析成 InvestigationQuery。

    Args:
        question: 工程師的問題（字串）
    Returns:
        InvestigationQuery 物件；API 錯誤或驗證失敗時回傳 None
    """
    # ① 呼叫 LLM，格式交給 API（API 錯誤 → None）
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=INSTRUCTIONS + question,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InvestigationQuery,
            ),
        )
    except errors.APIError as error:
        print(f"[API 錯誤 {error.code}] {error.message}")
        return None

    # ②③ 兩層驗證（失敗 → None）
    try:
        data = json.loads(response.text)                         # 填空 1：第 1 層，文字 → dict
        return InvestigationQuery(**data)                         # 填空 2：第 2 層，dict → InvestigationQuery，直接回傳
    except json.JSONDecodeError as error:
        print("[第 1 層失敗] 不是合法的 JSON：", error)
        return None
    except ValidationError as error:
        print("[第 2 層失敗] 內容不合格：", error.errors()[0]["msg"])
        return None


if __name__ == "__main__":
    for question in ["M-03 上週為什麼一直停機？", "三號機最近一個月壞了幾次？"]:
        query = parse_question(question)
        print(question, "→", query)
