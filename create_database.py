"""建立模擬的工廠故障資料庫 factory.db(模擬資料，非真實公司資料）。

執行: python create_database.py
每次執行都會刪除舊的 factory.db, 重新產生最近 90 天的資料。
資料以「執行當天」為基準，所以隔幾天後想查「最近 7 天」，重新執行一次即可。
"""

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path # 這是用來取得 factory.db 的路徑，方便在不同作業系統上執行

DB_PATH = Path(__file__).parent / "factory.db"

RANDOM_SEED = 42          # 固定隨機種子：每次產生的故障模式都一樣，結果可重現
DAYS_OF_HISTORY = 90      # 產生最近 90 天的資料
DAILY_FAILURE_RATE = 0.08 # 每台機台每天約 8% 機率發生一次故障

MACHINES = [
    # (machine_id, production_line, machine_type)
    ("M-01", "A 線", "CNC 車床"),
    ("M-02", "A 線", "CNC 銑床"),
    ("M-03", "B 線", "CNC 銑床"),
    ("M-04", "B 線", "沖壓機"),
    ("M-05", "C 線", "射出成型機"),
]

# 故障類型與出現的相對比例
FAILURE_TYPES = ["過熱", "刀具磨損", "電氣異常", "潤滑不足"]
FAILURE_WEIGHTS = [0.25, 0.35, 0.2, 0.2]


def random_time_on(day):
    """回傳某一天中隨機的時間點（07:00～22:59，模擬兩班制的生產時段）。"""
    return datetime(day.year, day.month, day.day, random.randint(7, 22), random.randint(0, 59))


def generate_failures(today):
    """產生故障紀錄，回傳 list of tuple：(machine_id, failure_time, failure_type, downtime_minutes)。"""
    failures = []

    # ① 一般情況：每台機台、每一天，以固定機率發生故障
    for days_ago in range(DAYS_OF_HISTORY):
        day = today - timedelta(days=days_ago)
        for machine_id, _, _ in MACHINES:
            if random.random() < DAILY_FAILURE_RATE:
                failure_type = random.choices(FAILURE_TYPES, weights=FAILURE_WEIGHTS)[0]
                failures.append((machine_id, random_time_on(day), failure_type, random.randint(15, 120)))

    # ② 刻意加入的調查情境：M-03 最近 7 天「過熱」故障明顯增加，停機時間也比較長
    #    讓之後的調查助理有一個「異常」可以發現
    for days_ago in [0, 1, 2, 3, 5, 6]:
        day = today - timedelta(days=days_ago)
        failures.append(("M-03", random_time_on(day), "過熱", random.randint(60, 180)))

    return failures


def create_database():
    random.seed(RANDOM_SEED)
    today = date.today()

    # 每次都重新建立，避免舊資料殘留
    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    # 機台名冊：之後用來判斷「機台是否存在」（查不到機台 ≠ 機台沒有故障）
    cursor.execute("""
        CREATE TABLE machines (
            machine_id      TEXT PRIMARY KEY,
            production_line TEXT NOT NULL,
            machine_type    TEXT NOT NULL
        )
    """)
    cursor.executemany("INSERT INTO machines VALUES (?, ?, ?)", MACHINES)

    # 故障紀錄：時間存成 ISO 格式字串（YYYY-MM-DD HH:MM:SS），字串比較就等於時間比較
    cursor.execute("""
        CREATE TABLE failures (
            failure_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id       TEXT NOT NULL REFERENCES machines(machine_id),
            failure_time     TEXT NOT NULL,
            failure_type     TEXT NOT NULL,
            downtime_minutes INTEGER NOT NULL
        )
    """)
    failures = generate_failures(today)
    cursor.executemany(
        "INSERT INTO failures (machine_id, failure_time, failure_type, downtime_minutes) VALUES (?, ?, ?, ?)",
        [(m, t.isoformat(sep=" "), ft, d) for m, t, ft, d in failures],
    )

    connection.commit()
    connection.close()
    print(f"已建立 {DB_PATH.name}：{len(MACHINES)} 台機台、{len(failures)} 筆故障紀錄（基準日 {today}）")


if __name__ == "__main__":
    create_database()
