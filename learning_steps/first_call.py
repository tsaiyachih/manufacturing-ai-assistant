import os
from dotenv import load_dotenv
from google import genai

load_dotenv() #讀 .env → 放進 os.environ

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(

    model="gemini-3.5-flash-lite",
    contents="CNC 機台過熱停機的常見原因有哪些？請用 3 點簡短回答。",
 
)

print(response.text)
print(response.usage_metadata)

# 用 Python 透過 Google 的 SDK 呼叫 Gemini API。
# 程式從環境變數讀取 API Key，把模型名稱和問題送到 Google 的伺服器；
# 模型生成回答後，透過 HTTP Response 回傳。SDK 把回傳的 JSON 轉成 Python 物件，我再取出回答內容和 token 用量。