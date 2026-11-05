"""測試 M6：搜尋文件工具、證據記錄、引用檢查、整合後的 Agent Loop。

Embedding 與 LLM 都用假的取代：不連網路、不花額度。
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

import investigator
import rag

FAKE_CHUNKS = [
    {"source": "manual.md", "section": "§1 過熱停機", "text": "過熱處理步驟"},
    {"source": "records.md", "section": "維修單 #1182", "text": "M-03 風扇積塵"},
    {"source": "manual.md", "section": "§2 刀具磨損", "text": "刀具處理步驟"},
]
FAKE_EMBEDDINGS = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])


@pytest.fixture
def fake_documents(monkeypatch):
    """用假的文件索引；問題一律轉成 [1, 0]，最接近「§1 過熱停機」。"""
    monkeypatch.setattr(investigator, "DOCUMENT_INDEX", {"chunks": FAKE_CHUNKS, "embeddings": FAKE_EMBEDDINGS})
    monkeypatch.setattr(rag, "embed_texts", lambda texts: np.array([[1.0, 0.0]]))


# ===== 搜尋文件工具 =====

def test_search_documents_returns_source_ids(fake_documents):
    result = investigator.search_documents("過熱怎麼處理")
    assert result["results"][0]["source_id"] == "manual.md｜§1 過熱停機"


def test_search_documents_returns_empty_when_not_relevant(fake_documents, monkeypatch):
    """問題和所有段落都不相似 → 回傳空結果，讓 LLM 說「沒有資料」。"""
    monkeypatch.setattr(rag, "embed_texts", lambda texts: np.array([[-1.0, 0.0]]))
    result = investigator.search_documents("午餐吃什麼")
    assert result["results"] == []


# ===== 證據記錄與引用檢查 =====

def test_collect_evidence_from_database_and_documents():
    assert investigator.collect_evidence("count_failures", {"failure_count": 7}) == {"資料庫：count_failures"}
    search_result = {"results": [{"source_id": "manual.md｜§1 過熱停機"}]}
    assert investigator.collect_evidence("search_documents", search_result) == {"manual.md｜§1 過熱停機"}
    # 工具失敗時沒有任何可引用的證據
    assert investigator.collect_evidence("count_failures", {"error": "找不到機台"}) == set()


def test_check_citations_flags_fabricated_source():
    """引用了本次沒有取得的來源（例如編造的 §9）→ 必須被標記出來。"""
    evidence = {"資料庫：count_failures", "manual.md｜§1 過熱停機"}
    answer = "故障 7 次【資料庫：count_failures】，請依步驟處理【manual.md｜§1 過熱停機】【manual.md｜§9 不存在】"
    assert investigator.check_citations(answer, evidence) == ["manual.md｜§9 不存在"]


def test_check_citations_passes_when_all_cited_sources_exist():
    evidence = {"資料庫：count_failures"}
    assert investigator.check_citations("故障 7 次【資料庫：count_failures】", evidence) == []


# ===== 整合：Agent Loop + 搜尋 + 引用檢查 =====

def make_tool_call_response(name, args):
    call = MagicMock()
    call.name = name
    call.args = args
    response = MagicMock()
    response.function_calls = [call]
    return response


def make_text_response(text):
    response = MagicMock()
    response.function_calls = None
    response.text = text
    return response


def test_investigation_warns_about_fabricated_citation(fake_documents, monkeypatch):
    """LLM 搜尋了文件，但最後的報告引用了一個沒有搜尋到的來源 → 報告附上警告。"""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        make_tool_call_response("search_documents", {"query": "過熱 處理"}),
        make_text_response("請清潔風扇【manual.md｜§1 過熱停機】，並更換軸承【manual.md｜§9 軸承更換】"),
    ]
    monkeypatch.setattr(investigator, "client", fake_client)

    result = investigator.run_investigation("M-03 過熱怎麼處理？")

    assert "manual.md｜§1 過熱停機" in result["evidence"]
    assert result["unverified_citations"] == ["manual.md｜§9 軸承更換"]
    assert "⚠️ 系統檢查" in result["answer"]
