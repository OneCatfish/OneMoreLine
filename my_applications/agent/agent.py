from openai import OpenAI
import os
import subprocess
import getpass

class ChatAgent:
    def __init__(self, api_key=None):
        """
        初始化ChatAgent
        
        参数:
            api_key: DeepSeek API密钥，如果为None会尝试从环境变量获取
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in DEEPSEEK_API_KEY environment variable")
            
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.reset_conversation()
        self.safe_mode = True  # 安全模式，防止执行危险命令

    def reset_conversation(self):
        """重置对话历史到初始系统消息"""
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "你是一个智能助手，可以回答问题并执行简单的系统命令。"
                    "如果用户需要执行命令，请以 'cmd: ' 开头返回正确的cmd格式命令。"
                    "注意安全，不要执行危险命令。"
                )
            }
        ]

    def ask(self, prompt):
        """处理用户输入并返回响应"""
        if not prompt.strip():
            return "请输入有效内容"
            
        if prompt.lower() == "reset":
            self.reset_conversation()
            return "对话已重置"
            
        if prompt.lower() == "safe mode on":
            self.safe_mode = True
            return "安全模式已开启"
            
        if prompt.lower() == "safe mode off":
            self.safe_mode = False
            return "安全模式已关闭"

        try:
            # 添加用户消息到对话历史
            self.conversation_history.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.conversation_history,
                stream=False
            )
            
            # 获取助手的回复
            assistant_reply = response.choices[0].message.content
            print(f"Assistant: {assistant_reply}")
            
            # 检查是否需要执行命令
            if assistant_reply.strip().lower().startswith("cmd:"):
                if self.safe_mode:
                    return "安全模式下不能执行命令，请先关闭安全模式"
                command = assistant_reply[len("cmd:"):].strip()
                execution_result = self.execute_command(command)
                
                # 将命令执行结果添加到对话历史
                self.conversation_history.extend([
                    {"role": "assistant", "content": assistant_reply},
                    {"role": "user", "content": f"命令执行结果: {execution_result}"}
                ])
                return execution_result
            else:
                # 普通回复添加到对话历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_reply
                })
                return assistant_reply
                
        except Exception as e:
            return f"发生错误: {str(e)}"

    def execute_command(self, command):
        """执行系统命令并返回结果"""
        if not command:
            return "无有效命令"
            
        # 危险命令检查
        dangerous_keywords = ["rm", "del", "format", "shutdown", "restart", "kill"]
        if any(keyword in command.lower() for keyword in dangerous_keywords):
            return "拒绝执行可能危险的命令"
            
        try:
            # 根据系统类型选择不同的命令执行方式
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ["powershell", "-Command", command],
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=10
                )
            else:  # Unix-like
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=10,
                    executable="/bin/bash"
                )
                
            if result.returncode == 0:
                return result.stdout.strip() or "命令执行成功"
            else:
                return f"错误: {result.stderr.strip()}"
                
        except subprocess.TimeoutExpired:
            return "命令执行超时"
        except Exception as e:
            return f"执行错误: {str(e)}"


if __name__ == "__main__":
    try:
        api_key =  ""
        agent = ChatAgent(api_key)

        print("ChatAgent 已启动 (输入 'exit' 退出, 'reset' 重置对话)")
        print(f"安全模式: {'开启' if agent.safe_mode else '关闭'}")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() == "exit":
                    print("再见!")
                    break
                    
                response = agent.ask(user_input)
                print(f"Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n使用Ctrl+C退出，请输入'exit'退出程序")
                
    except Exception as e:
        print(f"初始化失败: {str(e)}")
