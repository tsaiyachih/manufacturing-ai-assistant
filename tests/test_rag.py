"""測試 M5 RAG：切段、相似度、搜尋排序、相關性門檻。

Embedding 與 LLM 都用假的取代：不連網路、不花額度、結果固定。
"""

from unittest.mock import MagicMock

import numpy as np

import rag


def test_load_chunks_splits_by_section_and_keeps_source():
    """依「## 章節」切段，每段都要記錄來源檔名與章節名稱。"""
    chunks = rag.load_chunks()
    sections = [chunk["section"] for chunk in chunks]

    assert "§1 過熱停機" in sections
    assert all(chunk["source"].endswith(".md") for chunk in chunks)
    # 文件最上面的「# 標題」不屬於任何章節，不應該變成一段
    assert not any(chunk["section"].startswith("#") for chunk in chunks)


def test_cosine_similarity_basic_cases():
    """方向相同 = 1，互相垂直 = 0。"""
    query = np.array([1.0, 0.0])
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    scores = rag.cosine_similarity(query, matrix)
    assert np.allclose(scores, [1.0, 0.0])


def use_fake_embeddings(monkeypatch, question_vector):
    """把 embed_texts 換成假的：問題一律轉成指定的向量。"""
    monkeypatch.setattr(rag, "embed_texts", lambda texts: np.array([question_vector]))


FAKE_CHUNKS = [
    {"source": "a.md", "section": "過熱", "text": "過熱處理"},
    {"source": "b.md", "section": "潤滑", "text": "潤滑處理"},
    {"source": "c.md", "section": "開機", "text": "開機程序"},
]
FAKE_EMBEDDINGS = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])


def test_search_returns_most_similar_first(monkeypatch):
    """搜尋結果依相似度由高到低排列，並回傳 top_k 段。"""
    use_fake_embeddings(monkeypatch, [1.0, 0.0])   # 問題的方向最接近「過熱」那段

    results = rag.search("過熱怎麼辦", FAKE_CHUNKS, FAKE_EMBEDDINGS, top_k=2)

    assert [chunk["section"] for _, chunk in results] == ["過熱", "潤滑"]
    assert results[0][0] >= results[1][0]


def test_low_similarity_skips_llm(monkeypatch):
    """最相關的段落也低於門檻 → 直接回覆找不到，完全不呼叫 LLM。"""
    use_fake_embeddings(monkeypatch, [-1.0, 0.0])  # 和所有段落都不相似
    fake_client = MagicMock()
    monkeypatch.setattr(rag, "client", fake_client)

    result = rag.answer_with_rag("午餐吃什麼", FAKE_CHUNKS, FAKE_EMBEDDINGS)

    assert "找不到" in result["answer"]
    fake_client.models.generate_content.assert_not_called()


def test_prompt_contains_numbered_sources():
    """給 LLM 的 prompt 要有編號與來源，LLM 才能標註引用。"""
    results = [(0.9, FAKE_CHUNKS[0]), (0.7, FAKE_CHUNKS[1])]
    prompt = rag.build_prompt("過熱怎麼辦", results)

    assert "[1] 來源：a.md｜過熱" in prompt
    assert "[2] 來源：b.md｜潤滑" in prompt
    assert prompt.endswith("問題：過熱怎麼辦")
