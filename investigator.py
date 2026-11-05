"""M6：Manufacturing AI Investigation Assistant（Agentic RAG）。

把 M4 的 Agent Loop 和 M5 的 RAG 結合：
- 資料庫工具（M3/M4）：回答「發生了什麼」，所有數字由 SQL / Python 計算
- search_documents（M5）：回答「該怎麼處理」，根據維修手冊與歷史維修紀錄
- LLM 自己決定要用哪些工具、搜尋什麼；Python 負責執行、記錄證據，並在最後檢查引用的來源是否真實存在
"""

import os
import re
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

import rag
from agent import AVAILABLE_TOOLS as DATABASE_TOOLS          # 重複使用 M4 的 3 個資料庫工具
from agent import TOOL_DECLARATIONS as DATABASE_TOOL_DECLARATIONS

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")
client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"

# 比 M4 多 1 步：查資料庫 → 搜尋文件 → 寫報告，保留緩衝
MAX_STEPS = 6

# ===== 系統指示：調查方向 + 固定的報告格式 + 引用規則 =====
SYSTEM_INSTRUCTION = """你是製造業設備調查助理，協助產線工程師調查設備故障並提出有依據的處理建議。

調查方向（依問題需要選用，簡單問題只用必要的工具）：
1. 問「為什麼變多」時，先用 compare_with_baseline 確認是否真的變多，再用 get_failure_types 查故障類型。
2. 問「怎麼處理」或「以前有沒有類似狀況」時，用 search_documents 搜尋維修文件；搜尋關鍵字要具體，例如「M-03 過熱 處理方式」。
3. 不要重複相同的搜尋。

資料規則：
- 所有數字必須來自資料庫工具結果，不可以自己推算或編造。
- 處理建議只能來自 search_documents 的結果；找不到文件時，明確寫「維修文件中沒有相關資訊」，不要用自己的知識補。
- 工具回傳 error 時，如實說明。

回答格式（需要調查時使用以下 4 段；簡單問題可以只回答重點並附來源）：
## 結論
## 數據依據
## 處理建議
## 參考來源

引用格式（每個重點後面都要標註，Python 會檢查）：
- 資料庫數據：【資料庫：工具名稱】，例如【資料庫：compare_with_baseline】
- 維修文件：【文件的 source_id】，必須和 search_documents 回傳的 source_id 完全相同

使用繁體中文，簡潔清楚。"""

# ===== 第 4 個工具：搜尋維修文件（包裝 M5 的 RAG 檢索）=====
SEARCH_DOCUMENTS_DECLARATION = types.FunctionDeclaration(
    name="search_documents",
    description="搜尋維修手冊與歷史維修紀錄，取得故障原因、處理步驟或過去的類似案例。需要處理建議或歷史案例時使用。",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜尋關鍵字，例如「過熱停機 處理方式」"},
        },
        "required": ["query"],
    },
)

# 文件索引：第一次搜尋時才建立（1 次 Embedding 呼叫），之後重複使用
DOCUMENT_INDEX = {}


def search_documents(query):
    """搜尋維修文件，回傳最相關的段落與 source_id；都不相關時回傳空結果。"""
    if not DOCUMENT_INDEX:
        DOCUMENT_INDEX["chunks"], DOCUMENT_INDEX["embeddings"] = rag.build_index()

    results = rag.search(query, DOCUMENT_INDEX["chunks"], DOCUMENT_INDEX["embeddings"])

    # 沿用 M5 的門檻：最相關的段落都不夠相關 → 回傳空結果，讓 LLM 明確說「沒有資料」
    if results[0][0] < rag.MIN_SIMILARITY:
        return {"query": query, "results": [], "message": "找不到相關的維修文件"}

    return {
        "query": query,
        "results": [
            {
                "source_id": f"{chunk['source']}｜{chunk['section']}",   # 引用時必須使用的來源名稱
                "similarity": round(score, 3),
                "text": chunk["text"],
            }
            for score, chunk in results
        ],
    }


CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    tools=[types.Tool(function_declarations=DATABASE_TOOL_DECLARATIONS + [SEARCH_DOCUMENTS_DECLARATION])],
)

# ===== 允許清單：3 個資料庫工具 + 搜尋文件 =====
AVAILABLE_TOOLS = {**DATABASE_TOOLS, "search_documents": search_documents}


def run_tool(name, args):
    """依照 LLM 的要求執行工具；不在允許清單內的工具一律拒絕。（與 M3、M4 相同）"""
    tool_function = AVAILABLE_TOOLS.get(name)
    if tool_function is None:
        return {"error": f"不支援的工具：{name}"}
    try:
        return tool_function(**args)
    except Exception as error:
        return {"error": f"工具執行失敗：{error}"}


def collect_evidence(tool_name, result):
    """從一次工具結果中，找出「可以被引用」的來源名稱。

    來源由 Python 記錄，而不是由 LLM 說了算：之後用來檢查 LLM 的引用是否真實存在。
    """
    if "error" in result:
        return set()
    if tool_name == "search_documents":
        return {item["source_id"] for item in result["results"]}
    return {f"資料庫：{tool_name}"}


def check_citations(answer, evidence):
    """找出回答中「本次調查沒有實際取得」的引用（例如 LLM 編造的手冊章節）。

    Returns:
        list：無法對應到證據的引用；空 list 代表所有引用都有依據
    """
    cited = re.findall(r"【(.+?)】", answer)
    return sorted({citation for citation in cited if citation not in evidence})


def run_investigation(question):
    """執行一次調查：Agent Loop + 證據記錄 + 引用檢查。

    Returns:
        dict：answer（報告）、evidence（實際取得的來源）、unverified_citations（無法驗證的引用）
    """
    history = [types.Content(role="user", parts=[types.Part(text=question)])]
    evidence = set()   # 本次調查「實際」取得的所有來源

    for step in range(1, MAX_STEPS + 1):
        try:
            response = client.models.generate_content(model=MODEL, contents=history, config=CONFIG)
        except errors.APIError as error:
            return {"answer": f"[API 錯誤 {error.code}] {error.message}", "evidence": [], "unverified_citations": []}

        # 停止條件：LLM 不再要求工具 → 最終報告
        if not response.function_calls:
            print(f"  [第 {step} 步] 產出報告")
            answer = response.text
            unverified = check_citations(answer, evidence)
            if unverified:
                # 不刪除 LLM 的回答，而是明確標示風險，讓工程師知道哪些內容不可採信
                answer += "\n\n⚠️ 系統檢查：以下引用無法對應到本次查詢的資料，請勿採信：" + "、".join(unverified)
            return {"answer": answer, "evidence": sorted(evidence), "unverified_citations": unverified}

        history.append(response.candidates[0].content)
        result_parts = []
        for call in response.function_calls:
            args = dict(call.args or {})
            result = run_tool(call.name, args)
            evidence |= collect_evidence(call.name, result)
            summary = [item["source_id"] for item in result["results"]] if call.name == "search_documents" and "results" in result else result
            print(f"  [第 {step} 步] {call.name}({args}) → {summary}")
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))
        history.append(types.Content(role="user", parts=result_parts))

    return {
        "answer": f"調查未完成：已達到最大步數（{MAX_STEPS}），請縮小問題範圍後再試一次。",
        "evidence": sorted(evidence),
        "unverified_citations": [],
    }


if __name__ == "__main__":
    for question in [
        "M-03 上週停機特別多，什麼原因？以前有類似狀況嗎？該怎麼處理？",   # Phase 0 的情境
        "M-04 最近一週壞了幾次？",                                          # 簡單問題：不需要搜尋文件
    ]:
        print("\n" + "=" * 60)
        print("問題：", question)
        result = run_investigation(question)
        print("\n" + result["answer"])
        print("\n[本次證據]", result["evidence"])
