"""測試 run_tool()：工具的安全與錯誤處理。

run_tool() 不會呼叫 LLM，所以不需要 Mock，直接呼叫即可。
這些測試讀取 factory.db（唯讀），執行前需先跑過 create_database.py。
"""

import tool_calling


def test_valid_call_returns_failure_count():
    """① 正常呼叫：回傳故障次數。"""
    result = tool_calling.run_tool("count_failures", {"machine_id": "M-03", "days": 7})
    assert "failure_count" in result
    assert result["failure_count"] >= 0


def test_unknown_tool_is_rejected():
    """② 不在允許清單內的工具：拒絕執行，回傳錯誤。"""
    result = tool_calling.run_tool("delete_all_data", {})
    assert "error" in result
    assert "不支援的工具" in result["error"]


def test_unknown_machine_returns_error_not_zero():
    """③ 機台不存在：回傳錯誤，而不是 failure_count = 0。"""
    result = tool_calling.run_tool("count_failures", {"machine_id": "M-99", "days": 7})
    assert "error" in result
    assert "failure_count" not in result


def test_missing_argument_returns_error_instead_of_crashing():
    """④ LLM 少給了參數：回傳錯誤，程式不當掉。"""
    result = tool_calling.run_tool("count_failures", {"machine_id": "M-03"})
    assert "error" in result
