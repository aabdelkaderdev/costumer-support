from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

class ConversationManager:
    def __init__(self, memory_type="buffer", llm=None):
        self.llm = llm

        if memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        elif memory_type == "summary":
            # For summary memory, LLM is required
            from langchain_groq import ChatGroq
            self.llm = llm or ChatGroq(model="qwen/qwen3-32b", temperature=0)
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )

    def add_user_message(self, message):
        """Add user message to memory"""
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message):
        """Add AI message to memory"""
        self.memory.chat_memory.add_ai_message(message)

    def get_conversation_history(self):
        """Get conversation history"""
        return self.memory.chat_memory.messages

    def clear(self):
        """Clear the conversation memory"""
        self.memory.clear()
