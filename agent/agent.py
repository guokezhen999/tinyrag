from typing import List, TypedDict, Annotated, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

class SpeakerMessage(TypedDict):
    speaker: str
    message: BaseMessage

def append_speaker_messages(left: List[SpeakerMessage], right: List[SpeakerMessage]) -> List[SpeakerMessage]:
    return left + right

class ConversationState(TypedDict):
    messages: Annotated[List[SpeakerMessage], append_speaker_messages]
    turn_count: int
    max_turns: int

class Agent:
    def __init__(self, name: str, system_message: str, model: BaseChatModel, tools: List[tool] = None):
        self.name = name
        self.system_message = system_message
        self.llm = model.bind_tools(tools) if tools else model
        self.tools = {tool.name: tool for tool in tools} if tools else {}

    def call_agent(self, state: ConversationState) -> BaseMessage:
        """ 接受当前对话，生成相应 """
        messages = [item['message'] for item in state['messages']]
        messages_for_llm = [SystemMessage(content=self.system_message)] + messages
        print(f"{self.name}正在思考...")
        response = self.llm.invoke(messages_for_llm)
        return response

    def call_agent_stream(self, state: ConversationState) -> Iterator[str]:
        """ 接受当前对话，以流式方式生成响应。 """
        messages = [item['message'] for item in state["messages"]]
        messages_for_llm = [SystemMessage(content=self.system_message)] + messages

        print(f"--- Agent: {self.name} is streaming... ---")
        stream = self.llm.stream(messages_for_llm)
        for chunk in stream:
            if chunk.content:
                yield chunk.content



