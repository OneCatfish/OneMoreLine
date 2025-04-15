from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com/beta",
)

messages = [
    {"role": "user", "content": "使用CAA开发现在要写代码：打开CATIA零件并保存创建步骤"},
    {"role": "assistant", "content": "```C++\n", "prefix": True}
]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    stop=["```"],
)
print(response.choices[0].message.content)