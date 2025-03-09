# conversation_manager.py
from typing import List, Dict

class ConversationManager:
    def __init__(self):
        self._history = []
        self._cot_activated = False

    def get_history(self) -> List[Dict]:
        """安全获取对话历史副本"""
        return [msg.copy() for msg in self._history]

    def add_message(self, role: str, content: str):
        """新增消息"""
        self._history.append({"role": role, "content": content})

    def reset(self):
        """清空所有对话记录"""
        self._history.clear()
        self._cot_activated = False

    def initialize_cot(self, system_prompt: str):
        """初始化 COT（保持不变）"""
        if not self._cot_activated:
            self._history.insert(0, {"role": "system", "content": system_prompt})
            self._cot_activated = True