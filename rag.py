"""M5：基本 RAG（不含 Agent）。根據維修手冊與維修紀錄回答問題，並附上來源。

流程是固定的，不由 LLM 決定：
  事前準備：讀取文件 → 依章節切段 → 每段轉成向量（Embedding）
  回答問題：問題轉成向量 → 找出最相似的 3 段 → 相似度太低就直接回覆找不到 → 否則交給 LLM 根據這 3 段回答
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")
client = genai.Client(api_key=api_key)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
EMBEDDING_MODEL = "gemini-embedding-2"
GENERATION_MODEL = "gemini-3.5-flash-lite"
TOP_K = 3              # 每次交給 LLM 的段落數（M5 設計決策）
MIN_SIMILARITY = 0.6   # 最相關的段落低於這個值 → 視為「找不到資料」，不呼叫 LLM（依實測分數調整）

# 回答規則：只能根據找到的段落，並註明來源（避免 LLM 用自己的知識補）
SYSTEM_INSTRUCTION = """你是製造業設備維修助理。
規則：
1. 只能根據「參考資料」回答，不可以使用參考資料以外的知識。
2. 每個重點後面標註來源編號，例如 [1]、[2]。
3. 參考資料中沒有答案時，直接回答「維修文件中沒有相關資訊」，不要猜測。
4. 用繁體中文簡潔回答。"""


# ===== 階段一：事前準備 =====

def load_chunks():
    """讀取 knowledge/ 底下所有 Markdown 文件，依「## 章節標題」切段。

    Returns:
        list of dict：每段包含 source（檔名）、section（章節標題）、text（章節全文）
    """
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        current_section = None
        current_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                # 遇到新的章節標題：先把上一個章節存起來
                if current_section:
                    chunks.append(make_chunk(path.name, current_section, current_lines))
                current_section = line.removeprefix("## ").strip()
                current_lines = []
            elif current_section:
                # 第一個「##」之前的文件標題（# ...）不屬於任何章節，略過
                current_lines.append(line)
        if current_section:
            chunks.append(make_chunk(path.name, current_section, current_lines))
    return chunks


def make_chunk(source, section, lines):
    """把一個章節整理成一段：章節標題也放進內文，讓搜尋時能比對到標題的意思。"""
    body = "\n".join(lines).strip()
    return {"source": source, "section": section, "text": f"{section}\n{body}"}


def embed_texts(texts):
    """把多段文字轉成向量（每段一列）。

    注意：每段文字要各自包成一個 Content，否則模型會把所有文字當成同一段，只回傳一個向量。
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[types.Content(parts=[types.Part(text=text)]) for text in texts],
    )
    return np.array([embedding.values for embedding in response.embeddings])


def build_index():
    """建立搜尋索引：所有段落 + 對應的向量。文件很少，直接放在記憶體，不使用向量資料庫。"""
    chunks = load_chunks()
    embeddings = embed_texts([chunk["text"] for chunk in chunks])   # 所有段落一次送出，只花 1 次 API 呼叫
    return chunks, embeddings


# ===== 階段二：回答問題 =====

def cosine_similarity(query_vector, matrix):
    """計算一個向量與矩陣中每一列的餘弦相似度：1 = 意思幾乎相同，越小越不相關。"""
    return (matrix @ query_vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector))


def search(question, chunks, embeddings, top_k=TOP_K):
    """找出與問題最相似的 top_k 段，依相似度由高到低排列。

    Returns:
        list of (similarity, chunk)
    """
    question_vector = embed_texts([question])[0]
    scores = cosine_similarity(question_vector, embeddings)
    best_indexes = np.argsort(scores)[::-1][:top_k]   # 由高到低排序，取前 top_k 個
    return [(float(scores[i]), chunks[i]) for i in best_indexes]


def build_prompt(question, results):
    """把找到的段落編號、附上來源，和問題組成給 LLM 的 prompt。"""
    references = "\n\n".join(
        f"[{number}] 來源：{chunk['source']}｜{chunk['section']}\n{chunk['text']}"
        for number, (_, chunk) in enumerate(results, start=1)
    )
    return f"參考資料：\n{references}\n\n問題：{question}"


def answer_with_rag(question, chunks, embeddings):
    """RAG 回答：檢索 → 檢查相關性 → 組合 prompt → LLM 生成。

    Returns:
        dict：answer（回答）、sources（使用的段落與相似度）
    """
    results = search(question, chunks, embeddings)

    # Python 把關：最相關的段落都不夠相關 → 不呼叫 LLM，避免它用自己的知識亂答（也省一次費用）
    if results[0][0] < MIN_SIMILARITY:
        return {"answer": "找不到相關的維修文件，此問題可能不在支援範圍內。", "sources": results}

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=build_prompt(question, results),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    return {"answer": response.text, "sources": results}


if __name__ == "__main__":
    chunks, embeddings = build_index()
    print(f"已建立索引：{len(chunks)} 段，向量維度 {embeddings.shape[1]}")

    for question in [
        "M-03 過熱停機該怎麼處理？",
        "機台溫度一直太高、一直跳保護，怎麼辦？",   # 沒有「過熱」兩字，測試語意搜尋
        "以前 M-03 有發生過類似的過熱問題嗎？",      # 應該找到維修紀錄
        "M-03 的主軸軸承型號是什麼？",               # 相關但文件中沒有
        "今天午餐吃什麼？",                          # 完全無關
    ]:
        result = answer_with_rag(question, chunks, embeddings)
        print("\n問題：", question)
        for score, chunk in result["sources"]:
            print(f"  [{score:.3f}] {chunk['source']}｜{chunk['section']}")
        print("回答：", result["answer"])
