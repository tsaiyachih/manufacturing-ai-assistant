"""測試 parse_question()：用假的 client 取代 Gemini，不連網路、不花額度。"""

from unittest.mock import MagicMock

import query_parser


def make_fake_client(llm_text):

    """建立一個假 client: 被呼叫時，回傳的 response.text 就是 llm_text。"""
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text=llm_text)
    return fake_client


def test_valid_json_returns_query(monkeypatch):
    """① LLM 回傳合格的 JSON → 回傳 InvestigationQuery 物件。"""
    fake = make_fake_client('{"machine_id": "M-03", "days": 7, "intent": "failure_reason"}')
    monkeypatch.setattr(query_parser, "client", fake)

    query = query_parser.parse_question("M-03 上週為什麼一直停機？")

    assert query.machine_id == "M-03"         # 填空 1
    assert query.days == 7               # 填空 2


def test_not_json_returns_none(monkeypatch):
    """② LLM 回傳的不是 JSON → 第 1 層擋下，回傳 None。"""
    fake = make_fake_client("好的，這是結果：M-03，7 天")
    monkeypatch.setattr(query_parser, "client", fake)

    query = query_parser.parse_question("M-03 上週為什麼一直停機？")

    assert query is None                    # 填空 3


def test_invalid_days_returns_none(monkeypatch):
    """③ JSON 格式正確，但 days = 0 不合規則 → 第 2 層擋下，回傳 None。"""
    fake = make_fake_client('{"machine_id": "M-03", "days": 0, "intent": "failure_reason"}')
    monkeypatch.setattr(query_parser, "client", fake)

    query = query_parser.parse_question("M-03 上週為什麼一直停機？")

    assert query is None                    # 填空 4
