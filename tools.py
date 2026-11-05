"""LLM 可以使用的工具。所有資料庫操作都是唯讀的。"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "factory.db"


def count_failures(machine_id, days):
    """計算某台機台最近 N 天的故障次數。

    Args:
        machine_id: 機台編號，例如 "M-03"
        days: 最近幾天
    Returns:
        dict: 成功時含 failure_count; 機台不存在時含 error
    """
    # Python 負責計算日期（M2 的設計原則）
    start_date = (date.today() - timedelta(days=days)).isoformat()

    # mode=ro：以「唯讀」模式開啟，就算程式有 bug，也不可能修改或刪除資料
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    try:
        # ① 機台存在嗎？
        machine = connection.execute(
            "SELECT machine_id FROM machines WHERE machine_id = ?", (machine_id,)
        ).fetchone()

        if machine is None:
            return {"error": f"找不到機台 {machine_id}"}

        # ② 計算故障次數
        sql = """
            SELECT COUNT(*) FROM failures
            WHERE machine_id = ? AND failure_time >= ?
        """
        count = connection.execute(sql, (machine_id, start_date)).fetchone()[0]
        return {"machine_id": machine_id, "days": days, "failure_count": count}
    finally:
        connection.close()   # 不管成功或失敗，最後都關閉連線


BASELINE_DAYS = 90   # 「平常」的定義：過去 90 天（M4 設計決策：用長期平均當基準）


def compare_with_baseline(machine_id, recent_days):
    """比較某台機台「最近 N 天」和「平常」的故障次數。

    「平常」= 過去 90 天中，扣掉最近 N 天之後的期間，換算成「每 N 天」的平均次數。
    所有計算都在 Python 完成，LLM 只負責解讀結果（不讓 LLM 做數學）。

    Args:
        machine_id: 機台編號，例如 "M-03"
        recent_days: 最近幾天，例如 7
    Returns:
        dict：recent_count、baseline_avg、ratio（最近是平常的幾倍）；錯誤時含 error
    """
    if not 1 <= recent_days < BASELINE_DAYS:
        return {"error": f"recent_days 必須介於 1 到 {BASELINE_DAYS - 1} 之間"}

    today = date.today()
    recent_start = (today - timedelta(days=recent_days)).isoformat()
    baseline_start = (today - timedelta(days=BASELINE_DAYS)).isoformat()

    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        machine = connection.execute(
            "SELECT machine_id FROM machines WHERE machine_id = ?", (machine_id,)
        ).fetchone()
        if machine is None:
            return {"error": f"找不到機台 {machine_id}"}

        # 最近 N 天的次數
        recent_count = connection.execute(
            "SELECT COUNT(*) FROM failures WHERE machine_id = ? AND failure_time >= ?",
            (machine_id, recent_start),
        ).fetchone()[0]

        # 基準期間：90 天前 ～ 最近 N 天之前（不和最近 N 天重疊，避免「拿自己跟自己比」）
        baseline_count = connection.execute(
            "SELECT COUNT(*) FROM failures WHERE machine_id = ? AND failure_time >= ? AND failure_time < ?",
            (machine_id, baseline_start, recent_start),
        ).fetchone()[0]
    finally:
        connection.close()

    # 換算成「每 N 天平均幾次」，才能和最近 N 天直接比較
    baseline_period_days = BASELINE_DAYS - recent_days
    baseline_avg = baseline_count / baseline_period_days * recent_days
    # 平常完全沒故障時無法計算倍數，回傳 None，而不是除以 0
    ratio = round(recent_count / baseline_avg, 1) if baseline_avg > 0 else None

    return {
        "machine_id": machine_id,
        "recent_days": recent_days,
        "recent_count": recent_count,
        "baseline_avg": round(baseline_avg, 2),
        "ratio": ratio,
    }


def get_failure_types(machine_id, days):
    """查詢某台機台最近 N 天的故障類型分布與停機時間。

    Args:
        machine_id: 機台編號，例如 "M-03"
        days: 最近幾天
    Returns:
        dict：total_failures 與 by_type（依次數由多到少排列）；錯誤時含 error
    """
    start_date = (date.today() - timedelta(days=days)).isoformat()

    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        machine = connection.execute(
            "SELECT machine_id FROM machines WHERE machine_id = ?", (machine_id,)
        ).fetchone()
        if machine is None:
            return {"error": f"找不到機台 {machine_id}"}

        rows = connection.execute(
            """
            SELECT failure_type, COUNT(*) AS failure_count, SUM(downtime_minutes) AS downtime_minutes
            FROM failures
            WHERE machine_id = ? AND failure_time >= ?
            GROUP BY failure_type
            ORDER BY failure_count DESC
            """,
            (machine_id, start_date),
        ).fetchall()
    finally:
        connection.close()

    by_type = [
        {"failure_type": failure_type, "count": count, "downtime_minutes": downtime}
        for failure_type, count, downtime in rows
    ]
    return {
        "machine_id": machine_id,
        "days": days,
        "total_failures": sum(item["count"] for item in by_type),
        "by_type": by_type,
    }


if __name__ == "__main__":
    for machine_id in ["M-03", "M-04", "M-99"]:
        print(machine_id, "→", count_failures(machine_id, 7))
    print(compare_with_baseline("M-03", 7))
    print(get_failure_types("M-03", 7))
