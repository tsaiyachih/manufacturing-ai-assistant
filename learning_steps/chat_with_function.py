import os   # 讀取環境變數（os.environ）
import sys  # 用 sys.exit() 停止整支程式
from dotenv import load_dotenv  # 讀取 .env 檔案
from google import genai    # Google 的 SDK（呼叫層）
from google.genai import errors     # SDK 定義的錯誤類型：ClientError、ServerError

load_dotenv()

# 決策 A：Fail Fast，沒有 Key 就立刻停止
api_key = os.environ.get("GEMINI_API_KEY")    # .get()：找不到時回傳 None，不會 KeyError
if not api_key:                                       # 填空 1：什麼情況代表「沒有 Key」？
    sys.exit("找不到 GEMINI_API_KEY，請在 .env 檔案中設定。")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.5-flash-lite"


def ask(history, question):
    """送出新問題、取得回答；成功時才把問題和回答存進 history。

    Args:
        history: 對話紀錄（list）
        question: 使用者的新問題（字串）
    Returns:
        回答文字；失敗時回傳 None，且 history 不會被修改
    """
    user_message = {"role": "user", "parts": [{"text": question}]} #parts 是「一則訊息的內容」，用 list 裝

    try:
        # 決策 C：送出「history + 這次的問題」，但還不修改 history
        response = client.models.generate_content(model=MODEL, contents=history + [user_message])   # 填空 2
    except errors.ClientError as error:
        # 決策 B：429 是暫時的；其他 4xx 是請求本身有問題
        if error.code == 429:
            print(f"[錯誤 {error.code}] 請求太頻繁或免費額度已用完，請稍後再試。")
        else:
            print(f"[錯誤 {error.code}] 請求被拒絕，請檢查 API Key 或模型名稱：{error.message}")
        return None
    except errors.ServerError as error:
        print(f"[錯誤 {error.code}] Google 伺服器暫時有問題，請稍後再試。")
        return None

    # 決策 C：成功之後，才把問題和回答「一起」存進 history
    history.append(user_message)
    history.append({"role": "model", "parts": [{"text": response.text}]})
    return response.text


# ===== 主程式 =====
if __name__ == "__main__":
    history = []   
    print("【第 1 輪】\n", ask(history, "CNC 機台過熱停機的常見原因有哪些？請用 3 點簡短回答。"))
    print("\n【第 2 輪】\n", ask(history, "第 2 點要怎麼檢查？請簡短回答。"))
    print("\n【第 3 輪】\n", ask(history, "第一點呢？請簡短回答。"))
    print("\nhistory 共幾則訊息：", len(history))
