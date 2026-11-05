"""M4: Agent Loop。LLM 一步步決定要查什麼, Python 執行工具，直到 LLM 可以回答為止。

與 M3（tool_calling.py）的差別：
- 工具從 1 個變成 3 個
- 工具呼叫從「最多 1 次」變成「重複，直到 LLM 給出最終回答」，並以 MAX_STEPS 設定上限
- 一次回覆中若要求多個工具，每一個都會執行
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import compare_with_baseline, count_failures, get_failure_types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")
client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

# 安全裝置：最多呼叫 LLM 幾次。最複雜的問題預期約 4 步（3 次工具 + 1 次回答），保留 1 步空間
MAX_STEPS = 5

# ===== 系統指示：整段對話都適用的角色與規則 =====
# 給 LLM「調查的方向」，但不寫死步驟（M4 設計決策：有彈性、有方向）
SYSTEM_INSTRUCTION = """你是製造業設備調查助理，協助產線工程師調查設備故障。
規則：
1. 所有數字（次數、倍數、停機時間）都必須來自工具結果，不可以自己推算或編造。
2. 調查「為什麼故障變多」時，建議先用 compare_with_baseline 確認是否真的變多，再用 get_failure_types 查故障類型。
3. 簡單的問題（例如只問次數）只呼叫需要的工具。
4. 工具回傳 error 時，如實告訴使用者，不要猜測。
5. 用繁體中文簡潔回答，並說明依據的數據。"""

# ===== 工具說明書（給 LLM 看的）=====
MACHINE_ID_SCHEMA = {"type": "string", "description": "機台編號，格式為 M-XX，例如 M-03"}

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="count_failures",
        description="計算某台機台最近 N 天的故障次數。只問次數時使用。",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "machine_id": MACHINE_ID_SCHEMA,
                "days": {"type": "integer", "description": "最近幾天，例如一週 = 7"},
            },
            "required": ["machine_id", "days"],
        },
    ),
    types.FunctionDeclaration(
        name="compare_with_baseline",
        description="比較某台機台最近 N 天的故障次數與平常（過去 90 天平均）的差異，回傳是平常的幾倍。判斷故障是否變多時使用。",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "machine_id": MACHINE_ID_SCHEMA,
                "recent_days": {"type": "integer", "description": "最近幾天，例如一週 = 7"},
            },
            "required": ["machine_id", "recent_days"],
        },
    ),
    types.FunctionDeclaration(
        name="get_failure_types",
        description="查詢某台機台最近 N 天的故障類型分布與停機分鐘數。調查故障原因時使用。",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "machine_id": MACHINE_ID_SCHEMA,
                "days": {"type": "integer", "description": "最近幾天，例如一週 = 7"},
            },
            "required": ["machine_id", "days"],
        },
    ),
]

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
)

# ===== 允許清單：LLM 只能呼叫這裡面的工具 =====
AVAILABLE_TOOLS = {
    "count_failures": count_failures,
    "compare_with_baseline": compare_with_baseline,
    "get_failure_types": get_failure_types,
}


def run_tool(name, args):
    """依照 LLM 的要求執行工具；不在允許清單內的工具一律拒絕。（與 M3 相同）"""
    tool_function = AVAILABLE_TOOLS.get(name)
    if tool_function is None:
        return {"error": f"不支援的工具：{name}"}
    try:
        return tool_function(**args)
    except Exception as error:
        # LLM 給的參數不可信任（缺欄位、型別錯誤）；把錯誤回傳給 LLM，而不是讓程式當掉
        return {"error": f"工具執行失敗：{error}"}


def run_agent(question):
    """讓 Agent 調查一個問題：反覆「LLM 決定 → 執行工具 → 回報結果」，直到 LLM 給出最終回答。

    Args:
        question: 工程師的問題（字串）
    Returns:
        最終回答（字串）；API 錯誤或超過 MAX_STEPS 時回傳說明訊息
    """
    history = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(1, MAX_STEPS + 1):
        # ① LLM 看目前所有資訊，決定下一步
        try:
            response = client.models.generate_content(model=MODEL, contents=history, config=CONFIG)
        except errors.APIError as error:
            return f"[API 錯誤 {error.code}] {error.message}"

        # ② 停止條件：LLM 不再要求工具 → 這就是最終回答
        if not response.function_calls:
            print(f"  [第 {step} 步] LLM 給出最終回答")
            return response.text

        # ③ 記錄 LLM 的要求（history 必須「要求 ↔ 結果」成對）
        history.append(response.candidates[0].content)

        # ④ 執行「每一個」要求的工具（LLM 可能一次要求多個），收集結果
        result_parts = []
        for call in response.function_calls:
            args = dict(call.args or {})   # 沒有參數時 args 可能是 None
            result = run_tool(call.name, args)
            print(f"  [第 {step} 步] 呼叫 {call.name}({args}) → {result}")
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))

        # ⑤ 把所有工具結果加進 history，回到迴圈開頭，讓 LLM 看到結果再判斷
        history.append(types.Content(role="user", parts=result_parts))

    # 安全裝置：跑完 MAX_STEPS 還沒得到最終回答，就停止，避免無限迴圈、無限花費
    return f"調查未完成：已達到最大步數（{MAX_STEPS}），請縮小問題範圍後再試一次。"


if __name__ == "__main__":
    for question in [
        "M-03 最近一週壞了幾次？",          # 預期：1 個工具
        "M-03 最近為什麼停機變多？",         # 預期：多個工具
    ]:
        print("\n問題：", question)
        print("回答：", run_agent(question))
