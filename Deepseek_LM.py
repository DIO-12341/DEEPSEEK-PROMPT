# deepseek_client.py
from openai import OpenAI
from openai._exceptions import APIError, AuthenticationError, APIConnectionError, RateLimitError
from typing import Optional, List
import re
from conversation_manager import ConversationManager


class InvalidModelError(ValueError):
    """自定义模型错误类型 """
    pass


class DeepSeekClient:
    VALID_MODELS = ['deepseek-chat', 'deepseek-reasoner']
    COT_SYSTEM_PROMPT = (
        "你是一个严谨的 AI 助手，请使用 Chain-of-Thought 逐步推理并展示思考过程。"
        "每个推理步骤请以 'Step [序号]:' 开头（如 Step 1:），"
        "最终结论以 'Final Answer:' 开头"
    )

    def __init__(self, model: str, api_key: str):
        if model not in self.VALID_MODELS:
            raise InvalidModelError(f"无效模型，仅支持 {', '.join(self.VALID_MODELS)}")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model = model
        self.conversation = ConversationManager()
        self.cot_logs = []

    def _parse_cot_response(self, response: str) -> List[str]:
        """解析 COT 格式的响应"""
        steps = []
        step_pattern = r"(Step \d+: .*?)(?=\nStep |\nFinal Answer: |$)"
        final_answer_pattern = r"Final Answer: .*"

        steps.extend(re.findall(step_pattern, response, re.DOTALL))
        final_answer = re.findall(final_answer_pattern, response, re.DOTALL)
        if final_answer:
            steps.append(final_answer[0])

        return [s.strip() for s in steps if s.strip()]

    def _log_cot_steps(self, steps: List[str]):
        """记录 COT 过程日志"""
        print("\n=== COT 过程日志 ===")
        for idx, step in enumerate(steps, 1):
            log_entry = f"[Step {idx}] {step}"
            print(log_entry)
            self.cot_logs.append(log_entry)
        print("====================\n")

    def prompt(self,
               message: str,
               use_cot: bool = False,
               reset_conversation: bool = False,
               **kwargs) -> str:
        """
        执行对话请求
        :param message: 用户输入
        :param use_cot: 是否启用 Chain-of-Thought
        :param reset_conversation: 是否重置对话
        :param kwargs: 其他 OpenAI 参数
        :return: 模型响应内容
        """
        if reset_conversation:
            self.conversation.reset()

        if use_cot:
            self.conversation.initialize_cot(self.COT_SYSTEM_PROMPT)

        # 添加用户消息
        self.conversation.add_message("user", message)

        try:
            response = self.client.chat.completions.create(
                messages=self.conversation._history,
                model=self.model,
                **kwargs
            )
            assistant_response = response.choices[0].message.content

            # 添加助手响应
            self.conversation.add_message("assistant", assistant_response)

            # 处理 COT
            if use_cot:
                steps = self._parse_cot_response(assistant_response)
                self._log_cot_steps(steps)
                return self._extract_final_answer(steps) or assistant_response

            return assistant_response

        except AuthenticationError as e:
            raise Exception("API 密钥无效") from e
        except RateLimitError as e:
            raise Exception("请求速率超限") from e
        except APIConnectionError as e:
            raise Exception("网络连接失败") from e
        except APIError as e:
            raise Exception(f"API 错误: {e.message}") from e

    def _extract_final_answer(self, steps: List[str]) -> Optional[str]:
        """从 COT 步骤中提取最终答案"""
        for step in reversed(steps):
            if step.startswith("Final Answer:"):
                return step
        return None

    def get_conversation_history(self) -> List[dict]:
        """
        获取完整对话历史副本
        返回格式示例：
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮您？"}
        ]
        """
        return self.conversation.get_history()


