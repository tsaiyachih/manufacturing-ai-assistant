"""測試 ask()：用假的 client 取代 Gemini，不連網路、不花額度。"""

from unittest.mock import MagicMock

from google.genai import errors

import chat_with_function


def test_success_adds_question_and_answer(monkeypatch):
    """成功時：回傳回答，history 增加「問題 + 回答」2 則。"""
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text="測試回答")
    monkeypatch.setattr(chat_with_function, "client", fake_client)

    history = []
    answer = chat_with_function.ask(history, "M-03 為什麼過熱？")

    assert answer == "測試回答"                   # 填空 1：應該回傳什麼？
    assert len(history) == 2             # 填空 2：history 應該有幾則？
    assert history[1]["role"] == "model"       # 填空 3：第 2 則（index 1）是誰說的？


def test_error_keeps_history_unchanged(monkeypatch):
    """API 失敗（429）時：回傳 None，history 不能被修改。"""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"code": 429, "message": "quota exceeded"}}
    )
    monkeypatch.setattr(chat_with_function, "client", fake_client)

    history = []
    answer = chat_with_function.ask(history, "M-03 為什麼過熱？")

    assert answer is None                  # 填空 4：失敗時回傳什麼？
    assert len(history) == 0            # 填空 5：history 應該有幾則？


def test_second_question_sends_full_history(monkeypatch):
    """第 2 輪送出時，要包含前一輪的問題和回答（這就是「記憶」）。"""
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text="測試回答")
    monkeypatch.setattr(chat_with_function, "client", fake_client)

    history = []
    chat_with_function.ask(history, "過熱的常見原因？")
    chat_with_function.ask(history, "第 2 點要怎麼檢查？")

    # call_args：假 client「最後一次」被呼叫時收到的參數
    sent_contents = fake_client.models.generate_content.call_args.kwargs["contents"]
    assert len(sent_contents) == 3       # 填空 6：第 2 輪送出了幾則訊息？
