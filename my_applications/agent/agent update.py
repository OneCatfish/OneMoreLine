from openai import OpenAI
import os
import subprocess
import getpass
import platform
import shlex
import re
from typing import Optional, Tuple, List

class ChatAgent:
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化增强版 ChatAgent
        
        参数:
            api_key: DeepSeek API密钥，如果为None会尝试从环境变量获取
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in DEEPSEEK_API_KEY environment variable")
            
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.reset_conversation()
        self.safe_mode = True  # 安全模式，防止执行危险命令
        self.admin_mode = False  # 管理员模式，允许执行更多命令
        self.max_command_time = 30  # 命令最长执行时间(秒)
        self.current_dir = os.getcwd()  # 当前工作目录

    def reset_conversation(self):
        """重置对话历史到初始系统消息"""
        self.conversation_history = [
            {
                "role": "system",
                "content": (
                    "你是一个高级智能助手，可以回答问题并执行系统命令。\n"
                    "命令必须以以下格式之一返回:\n"
                    "1. `cmd: <command>` - 执行单条命令\n"
                    "2. `cmds: <command1> && <command2>` - 执行多条命令\n"
                    "3. `cmd-interactive: <command>` - 交互式执行命令\n"
                    "4. `file: <filename>` - 处理文件操作\n"
                    "5. `python: <code>` - 执行Python代码\n"
                    "注意安全，不要执行危险命令。"
                )
            }
        ]

    def ask(self, prompt: str) -> str:
        """处理用户输入并返回响应"""
        prompt = prompt.strip()
        if not prompt:
            return "请输入有效内容"
            
        # 处理特殊命令
        special_cmd_response = self._handle_special_commands(prompt)
        if special_cmd_response:
            return special_cmd_response

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
            
            # 检查并处理不同类型的命令
            command_response = self._process_commands(assistant_reply)
            if command_response:
                return command_response
                
            # 普通回复添加到对话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_reply
            })
            return assistant_reply
            
        except Exception as e:
            return f"发生错误: {str(e)}"

    def _handle_special_commands(self, prompt: str) -> Optional[str]:
        """处理特殊控制命令"""
        lower_prompt = prompt.lower()
        
        if lower_prompt == "reset":
            self.reset_conversation()
            return "对话已重置"
            
        if lower_prompt == "safe mode on":
            self.safe_mode = True
            self.admin_mode = False
            return "安全模式已开启"
            
        if lower_prompt == "safe mode off":
            self.safe_mode = False
            return "安全模式已关闭"
            
        if lower_prompt == "admin mode on":
            if not self.safe_mode:
                self.admin_mode = True
                return "管理员模式已开启 (谨慎使用!)"
            return "必须先关闭安全模式才能开启管理员模式"
            
        if lower_prompt == "admin mode off":
            self.admin_mode = False
            return "管理员模式已关闭"
            
        if lower_prompt == "pwd":
            return f"当前工作目录: {self.current_dir}"
            
        if lower_prompt.startswith("cd "):
            new_dir = prompt[3:].strip()
            return self._change_directory(new_dir)
            
        if lower_prompt == "status":
            return self._get_status()
            
        return None

    def _process_commands(self, assistant_reply: str) -> Optional[str]:
        """处理助手返回的各种命令"""
        reply = assistant_reply.strip()
        
        # 处理单条命令
        if reply.startswith("cmd:"):
            command = reply[len("cmd:"):].strip()
            return self._execute_single_command(command)
            
        # 处理多条命令
        elif reply.startswith("cmds:"):
            commands = reply[len("cmds:"):].strip()
            return self._execute_multiple_commands(commands)
            
        # 处理交互式命令
        elif reply.startswith("cmd-interactive:"):
            command = reply[len("cmd-interactive:"):].strip()
            return self._execute_interactive_command(command)
            
        # 处理文件操作
        elif reply.startswith("file:"):
            file_cmd = reply[len("file:"):].strip()
            return self._handle_file_operations(file_cmd)
            
        # 处理Python代码
        elif reply.startswith("python:"):
            code = reply[len("python:"):].strip()
            return self._execute_python_code(code)
            
        return None

    def _execute_single_command(self, command: str) -> str:
        """执行单条系统命令"""
        if not command:
            return "无有效命令"
            
        # 安全检查
        safety_check = self._check_command_safety(command)
        if not safety_check[0]:
            return safety_check[1]
            
        try:
            # 根据系统类型选择执行方式
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command", command],
                    cwd=self.current_dir,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=self.max_command_time
                )
            else:
                result = subprocess.run(
                    command,
                    cwd=self.current_dir,
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=self.max_command_time,
                    executable="/bin/bash"
                )
                
            output = result.stdout.strip() if result.stdout else ""
            error = result.stderr.strip() if result.stderr else ""
            
            if result.returncode == 0:
                return output or "命令执行成功"
            else:
                return f"命令执行失败 (退出码 {result.returncode}): {error}"
                
        except subprocess.TimeoutExpired:
            return f"命令执行超时 (超过 {self.max_command_time} 秒)"
        except Exception as e:
            return f"执行错误: {str(e)}"

    def _execute_multiple_commands(self, commands: str) -> str:
        """执行多条用 && 连接的命令"""
        if not commands:
            return "无有效命令"
            
        # 分割命令并检查安全性
        command_list = [cmd.strip() for cmd in commands.split("&&") if cmd.strip()]
        for cmd in command_list:
            safety_check = self._check_command_safety(cmd)
            if not safety_check[0]:
                return safety_check[1]
                
        # 合并执行
        try:
            if platform.system() == "Windows":
                full_command = " & ".join(command_list)
                shell_cmd = ["powershell", "-Command", full_command]
            else:
                full_command = " && ".join(command_list)
                shell_cmd = full_command
                
            result = subprocess.run(
                shell_cmd,
                cwd=self.current_dir,
                capture_output=True,
                text=True,
                shell=True,
                timeout=self.max_command_time * len(command_list)
            )
            
            output = result.stdout.strip() if result.stdout else ""
            error = result.stderr.strip() if result.stderr else ""
            
            if result.returncode == 0:
                return output or "所有命令执行成功"
            else:
                return f"命令执行失败 (退出码 {result.returncode}): {error}"
                
        except subprocess.TimeoutExpired:
            return f"命令执行超时 (超过 {self.max_command_time * len(command_list)} 秒)"
        except Exception as e:
            return f"执行错误: {str(e)}"

    def _execute_interactive_command(self, command: str) -> str:
        """交互式执行命令"""
        if not command:
            return "无有效命令"
            
        safety_check = self._check_command_safety(command)
        if not safety_check[0]:
            return safety_check[1]
            
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["start", "powershell", "-NoExit", "-Command", command],
                    cwd=self.current_dir,
                    shell=True,
                    check=True
                )
            else:
                subprocess.run(
                    ["x-terminal-emulator", "-e", "bash", "-c", f"{command}; exec bash"],
                    cwd=self.current_dir,
                    check=True
                )
            return "已启动交互式命令窗口"
        except Exception as e:
            return f"无法启动交互式命令: {str(e)}"

    def _handle_file_operations(self, file_cmd: str) -> str:
        """处理文件操作命令"""
        if not file_cmd:
            return "无有效文件操作命令"
            
        # 解析文件操作 (示例: "read example.txt" 或 "write test.txt 'content'")
        parts = shlex.split(file_cmd)
        if not parts:
            return "无效的文件操作格式"
            
        operation = parts[0].lower()
        
        try:
            if operation == "read" and len(parts) >= 2:
                # 读取文件
                with open(os.path.join(self.current_dir, parts[1]), 'r') as f:
                    return f.read()
                    
            elif operation == "write" and len(parts) >= 3:
                # 写入文件
                with open(os.path.join(self.current_dir, parts[1]), 'w') as f:
                    f.write(parts[2])
                return f"成功写入文件 {parts[1]}"
                
            elif operation == "list" and len(parts) >= 1:
                # 列出目录内容
                return "\n".join(os.listdir(self.current_dir))
                
            else:
                return f"不支持的文件操作: {operation}"
                
        except Exception as e:
            return f"文件操作错误: {str(e)}"

    def _execute_python_code(self, code: str) -> str:
        """执行Python代码"""
        if not code:
            return "无有效Python代码"
            
        # 安全检查
        if not self.admin_mode and any(
            kw in code.lower() 
            for kw in ["import os", "import sys", "subprocess", "exec(", "eval(", "open("]
        ):
            return "安全模式限制: 代码包含潜在危险操作"
            
        try:
            # 创建局部命名空间防止污染
            local_vars = {}
            global_vars = {"__builtins__": None}
            
            # 执行代码
            exec(code, global_vars, local_vars)
            
            # 获取非内置的变量作为输出
            output = "\n".join(
                f"{k}: {v}" 
                for k, v in local_vars.items() 
                if not k.startswith('_') and k not in global_vars
            )
            
            return output or "Python代码执行成功 (无输出)"
            
        except Exception as e:
            return f"Python代码执行错误: {str(e)}"

    def _change_directory(self, new_dir: str) -> str:
        """改变当前工作目录"""
        try:
            abs_path = os.path.abspath(os.path.join(self.current_dir, new_dir))
            if os.path.isdir(abs_path):
                self.current_dir = abs_path
                return f"工作目录已更改为: {self.current_dir}"
            else:
                return f"目录不存在: {abs_path}"
        except Exception as e:
            return f"更改目录错误: {str(e)}"

    def _check_command_safety(self, command: str) -> Tuple[bool, str]:
        """检查命令安全性"""
        if self.admin_mode:
            return (True, "")
            
        # 危险命令检查
        dangerous_patterns = [
            r"\b(rm|del|format|shutdown|restart|kill|mv|chmod|chown)\b",
            r"&{2,}|\|{2,}",  # 多个&&或||
            r";.*;",  # 多个命令分隔符
            r"`",  # 反引号
            r"\$\(.*\)",  # $()
            r">\s*/",  # 重定向到根目录
            r"\bdd\b",
            r"\b(wget|curl)\s.*\|\s*(sh|bash|python|perl)\b"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return (False, f"拒绝执行可能危险的命令: 检测到 '{pattern}'")
                
        # 限制访问系统关键目录
        sensitive_dirs = [
            "/", "/bin", "/sbin", "/usr/bin", "/usr/sbin", 
            "/etc", "/var", "/root", "/boot", 
            "C:\\", "C:\\Windows", "C:\\System32"
        ]
        
        for dir_path in sensitive_dirs:
            if dir_path.lower() in command.lower():
                return (False, f"拒绝访问系统关键目录: {dir_path}")
                
        return (True, "")

    def _get_status(self) -> str:
        """获取当前状态信息"""
        status = [
            f"安全模式: {'开启' if self.safe_mode else '关闭'}",
            f"管理员模式: {'开启' if self.admin_mode else '关闭'}",
            f"当前工作目录: {self.current_dir}",
            f"最大命令执行时间: {self.max_command_time}秒",
            f"操作系统: {platform.system()} {platform.release()}"
        ]
        return "\n".join(status)


def get_api_key() -> str:
    """安全获取API密钥"""
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        try:
            key = getpass.getpass("请输入DeepSeek API密钥(输入将不可见): ")
        except:
            key = input("请输入DeepSeek API密钥: ")
    return key


if __name__ == "__main__":
    try:
        api_key = get_api_key()
        agent = ChatAgent(api_key)

        print("增强版 ChatAgent 已启动")
        print("特殊命令: reset, safe mode on/off, admin mode on/off, pwd, cd <dir>, status")
        print(f"当前状态:\n{agent._get_status()}")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() == "exit":
                    print("再见!")
                    break
                    
                response = agent.ask(user_input)
                print(f"Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n提示: 使用 'exit' 退出程序")
                
    except Exception as e:
        print(f"初始化失败: {str(e)}")