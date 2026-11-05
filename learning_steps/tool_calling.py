"""M3: Tool Calling。LLM 決定要不要查資料, Python 負責檢查並執行工具。"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

# tools.py 在上一層（專案根目錄），把它加進 import 路徑
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools import count_failures  # noqa: E402

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

# ===== 工具說明書（給 LLM 看的）=====
# description 很重要：LLM 是讀這段文字來判斷「什麼時候該用這個工具」
COUNT_FAILURES_DECLARATION = types.FunctionDeclaration(
    name="count_failures",
    description="計算某台機台最近 N 天的故障次數。使用者問到故障次數、壞了幾次時使用。",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "machine_id": {"type": "string", "description": "機台編號，格式為 M-XX，例如 M-03"},
            "days": {"type": "integer", "description": "最近幾天，例如一週 = 7"},
        },
        "required": ["machine_id", "days"],
    },
)
CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[COUNT_FAILURES_DECLARATION])]
)

# ===== 允許清單：LLM 只能呼叫這裡面的工具 =====
AVAILABLE_TOOLS = {"count_failures": count_failures}


def run_tool(name, args):
    """依照 LLM 的要求執行工具；不在允許清單內的工具一律拒絕。"""
    tool_function = AVAILABLE_TOOLS.get(name)        # 填空 1：要用什麼去允許清單裡找？
    if tool_function is None:
        return {"error": f"不支援的工具：{name}"}
    try:
        return tool_function(**args)                  # 填空 2：執行工具，把 args 這個 dict 拆開傳進去
    except Exception as error:
        # 例如 LLM 給了錯誤型別的參數；把錯誤回傳給 LLM，而不是讓程式當掉
        return {"error": f"工具執行失敗：{error}"}


def answer(question):
    """回答一個問題；需要時呼叫一次工具。"""
    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    # ① 第 1 次呼叫：LLM 決定要不要用工具
    response = client.models.generate_content(model=MODEL, contents=history, config=CONFIG)

    # ② 不需要工具 → 直接回答
    if not response.function_calls:
        return response.text

    # ③ 需要工具：記錄 LLM 的要求 → Python 執行 → 記錄結果
    history.append(response.candidates[0].content)   # model 說「我要呼叫工具」
    call = response.function_calls[0]
    print(f"  [LLM 要求呼叫] {call.name}({call.args})")
    result = run_tool(call.name, call.args)
    print(f"  [工具結果] {result}")
    history.append(types.Content(
        role="user",
        parts=[types.Part.from_function_response(name=call.name, response=result)],
    ))

    # ④ 第 2 次呼叫：LLM 根據工具結果寫出回答
    final_response = client.models.generate_content(model=MODEL, contents=history, config=CONFIG)
    return final_response.text


if __name__ == "__main__":
    for question in [
        "M-03 最近一週壞了幾次？",       # 需要工具
        "M-99 最近一個月故障幾次？",     # 需要工具，但機台不存在
        "過熱停機通常是什麼原因？",       # 不需要工具（一般知識）
    ]:
        print("\n問題：", question)
        print("回答：", answer(question))
