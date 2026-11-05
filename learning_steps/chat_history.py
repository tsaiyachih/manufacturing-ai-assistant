import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

# 對話紀錄：由 Python 自己保存，API 不會記得
history = []

# ===== 第 1 輪 =====
history.append({"role": "user", "parts": [{"text": "CNC 機台過熱停機的常見原因有哪些？請用 3 點簡短回答。"}]})


response_1 = client.models.generate_content(model=MODEL, contents=history)


print("【第 1 輪回答】\n", response_1.text)
print("第 1 輪 input tokens：", response_1.usage_metadata.prompt_token_count)
history.append({"role": "model", "parts": [{"text": response_1.text}]})


# ===== 第 2 輪 =====
history.append({"role": "user", "parts": 
                [{"text": "第 ２點要怎麼檢查？請簡短回答。"}]})

response_2 = client.models.generate_content(
    model=MODEL,
    contents=history
)

print("\n【第 2 輪回答】\n", response_2.text)
print("第 2 輪 input tokens：", response_2.usage_metadata.prompt_token_count)
print("history 共幾則訊息：", len(history))
