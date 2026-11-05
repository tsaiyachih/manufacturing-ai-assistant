"""pytest 共用設定：讓測試在全新 clone、沒有 API Key、沒有資料庫的環境也能執行。

1. 把專案根目錄與 learning_steps/ 加進 import 路徑（核心程式在根目錄，學習過程的程式在 learning_steps/）
2. 沒有 GEMINI_API_KEY 時，設一個假的 Key：各模組 import 時會建立 client，但測試一律用假的 LLM，
   不會真的呼叫 API。已經有真的 Key 時不覆蓋。
3. 每次測試開始前重建模擬資料庫 factory.db：資料以「執行當天」為基準，
   「最近 7 天」這類測試必須用當天產生的資料才會正確。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learning_steps"))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import create_database  # noqa: E402

create_database.create_database()
