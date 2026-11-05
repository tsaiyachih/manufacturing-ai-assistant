"""測試 M4：新工具 + Agent Loop。

- 工具測試：直接讀 factory.db（唯讀）。資料以「建立當天」為基準，跨日後請重新執行 create_database.py。
- Agent Loop 測試：用假的 client 模擬 LLM 的「多步驟」回覆，不連網路、不花額度。
"""

from unittest.mock import MagicMock

import agent
from tools import compare_with_baseline, get_failure_types


# ===== 工具測試 =====

def test_compare_with_baseline_detects_increase():
    """M-03 最近 7 天刻意加入了過熱故障，應該明顯高於平常。"""
    result = compare_with_baseline("M-03", 7)
    assert result["ratio"] > 1


def test_compare_with_baseline_rejects_invalid_days():
    """recent_days 超出範圍 → 回傳錯誤，而不是算出沒有意義的數字。"""
    assert "error" in compare_with_baseline("M-03", 90)


def test_get_failure_types_sorted_by_count():
    """故障類型依次數由多到少排列，M-03 最多的是過熱。"""
    result = get_failure_types("M-03", 7)
    counts = [item["count"] for item in result["by_type"]]
    assert counts == sorted(counts, reverse=True)
    assert result["by_type"][0]["failure_type"] == "過熱"


def test_new_tools_return_error_for_unknown_machine():
    assert "error" in compare_with_baseline("M-99", 7)
    assert "error" in get_failure_types("M-99", 7)


# ===== Agent Loop 測試：用假的 LLM 回覆 =====

def make_tool_call_response(*calls):
    """假的 LLM 回覆：要求呼叫一個或多個工具。calls 的格式為 (工具名稱, 參數 dict)。"""
    function_calls = []
    for name, args in calls:
        call = MagicMock()
        call.name = name          # 注意：MagicMock(name=...) 是設定 mock 自己的名字，所以要另外指定
        call.args = args
        function_calls.append(call)
    response = MagicMock()
    response.function_calls = function_calls
    return response


def make_text_response(text):
    """假的 LLM 回覆：不呼叫工具，直接給出最終回答。"""
    response = MagicMock()
    response.function_calls = None
    response.text = text
    return response


def use_fake_llm(monkeypatch, responses):
    """把 agent.client 換成假的；每次呼叫依序回傳 responses 裡的下一個回覆。"""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = responses
    monkeypatch.setattr(agent, "client", fake_client)
    return fake_client


def test_agent_stops_when_llm_answers_directly(monkeypatch):
    """LLM 第 1 步就回答 → 迴圈立刻停止，只呼叫 LLM 1 次。"""
    fake = use_fake_llm(monkeypatch, [make_text_response("不需要查資料的回答")])

    assert agent.run_agent("過熱是什麼？") == "不需要查資料的回答"
    assert fake.models.generate_content.call_count == 1


def test_agent_runs_tool_then_answers(monkeypatch):
    """第 1 步要求工具 → Python 執行 → 第 2 步回答；工具結果有送回給 LLM。"""
    fake = use_fake_llm(monkeypatch, [
        make_tool_call_response(("count_failures", {"machine_id": "M-03", "days": 7})),
        make_text_response("M-03 最近一週故障 7 次"),
    ])

    assert agent.run_agent("M-03 壞了幾次？") == "M-03 最近一週故障 7 次"
    assert fake.models.generate_content.call_count == 2

    # 第 2 次呼叫 LLM 時，history 的最後一則是工具結果，而且是資料庫查到的真實數字
    second_call_history = fake.models.generate_content.call_args.kwargs["contents"]
    tool_result = second_call_history[-1].parts[0].function_response.response
    assert "failure_count" in tool_result


def test_agent_executes_every_tool_in_one_step(monkeypatch):
    """LLM 一次要求 2 個工具 → 兩個都要執行，結果都要送回去。"""
    fake = use_fake_llm(monkeypatch, [
        make_tool_call_response(
            ("compare_with_baseline", {"machine_id": "M-03", "recent_days": 7}),
            ("get_failure_types", {"machine_id": "M-03", "days": 7}),
        ),
        make_text_response("結論"),
    ])

    agent.run_agent("M-03 為什麼停機變多？")

    second_call_history = fake.models.generate_content.call_args.kwargs["contents"]
    assert len(second_call_history[-1].parts) == 2


def test_agent_stops_at_max_steps(monkeypatch):
    """LLM 一直要求工具、永遠不回答 → 跑滿 MAX_STEPS 就停止，不會無限迴圈。"""
    endless_tool_calls = [
        make_tool_call_response(("count_failures", {"machine_id": "M-03", "days": 7}))
        for _ in range(agent.MAX_STEPS)
    ]
    fake = use_fake_llm(monkeypatch, endless_tool_calls)

    answer = agent.run_agent("M-03 壞了幾次？")

    assert "調查未完成" in answer
    assert fake.models.generate_content.call_count == agent.MAX_STEPS


def test_agent_rejects_unknown_tool_without_crashing(monkeypatch):
    """LLM 要求不在允許清單內的工具 → 不執行，把錯誤回傳給 LLM，迴圈繼續。"""
    fake = use_fake_llm(monkeypatch, [
        make_tool_call_response(("delete_all_data", {})),
        make_text_response("無法執行該操作"),
    ])

    assert agent.run_agent("把資料刪掉") == "無法執行該操作"
    second_call_history = fake.models.generate_content.call_args.kwargs["contents"]
    tool_result = second_call_history[-1].parts[0].function_response.response
    assert "不支援的工具" in tool_result["error"]
