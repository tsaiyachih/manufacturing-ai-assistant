from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class InvestigationQuery(BaseModel):
    """工程師問題解析後的結構化查詢條件。"""
    machine_id: str           # 填空 1：機台編號，是什麼型別？
    days: int = Field(ge=1, le=365)   # 填空 2：查詢最近幾天，限制在 1～365 天
    intent: Literal["failure_reason", "failure_count", "repair_history"]              # 填空 3：只能是下面 3 種之一：
                              #   "failure_count"   問故障次數
                              #   "failure_reason"  問故障原因
                              #   "repair_history"  問維修紀錄


if __name__ == "__main__":

# ===== 用 5 筆「假裝是 LLM 回傳」的資料測試 =====
    test_cases = [
        {"machine_id": "M-03", "days": 7, "intent": "failure_reason"},     # A
        {"machine_id": "M-03", "days": "7", "intent": "failure_count"},    # B
        {"machine_id": "M-03", "days": -3, "intent": "failure_reason"},    # C
        {"machine_id": "M-03", "days": 7, "intent": "weather"},            # D
        {"machine_id": "M-03", "intent": "failure_reason"},                # E
]

    for data in test_cases:
        try:
            query = InvestigationQuery(**data)
            print("✅ 通過：", query)
        except ValidationError as error:
            print("❌ 不合格：", error.errors()[0]["msg"])