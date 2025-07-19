import os
import json
from typing import Iterator, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.agent import Agent, ConversationState, SpeakerMessage
from agent.tool import d2l_tool, search_tool

AVAILABLE_TOOLS = {
    "d2l_tool": d2l_tool,
    "google_search": search_tool
}

class AgentDialogService:
    """ 封装 LangGraph Agent 对话 """
    def __init__(self):
         print("--- AgentDialogService 已初始化（无状态模式） ---")

    def _get_llm(self, model_name: str) -> BaseChatModel:
        if model_name.startswith('gemini'):
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.environ.get("GOOGLE_API_KEY")
            )
        else:
            return ChatOpenAI(
                model=model_name,
                api_key=os.environ.get("SILICONFLOW_API_KEY"),
                base_url=os.environ.get("SILICONFLOW_BASE_URL"),
            )

    def _build_graph(self, asker: Agent, respondent: Agent) -> CompiledStateGraph:
        state_builder = StateGraph(ConversationState)

        def asker_node(state: ConversationState) -> dict:
            ai_message = asker.call_agent(state)
            return {"messages": [{
                "speaker": asker.name,
                "message": ai_message
            }]}

        def respondent_node(state: ConversationState) -> dict:
            ai_message = respondent.call_agent(state)
            return {"messages": [{
                "speaker": respondent.name,
                "message": ai_message
            }]}

        def tool_executor_node(state: ConversationState) -> dict:
            last_message = state['messages'][-1]['message']
            tool_outputs = []

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    if tool_name in respondent.tools:
                        tool_func = respondent.tools.get(tool_name)
                        output = tool_func.invoke(tool_input)
                        tool_message = ToolMessage(content=str(output), tool_call_id=tool_call["id"], name=tool_name)
                        tool_outputs.append({
                            "speaker": "tool",
                            "message": tool_message
                        })
            return {"messages": tool_outputs}

        def finalize_turn_node(state: ConversationState):
            return {"turn_count": state["turn_count"] + 1}

        def route_for_respondent(state: ConversationState) -> str:
            last_message = state['messages'][-1]['message']
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tool_executor"
            else:
                return "finalize_turn"

        def check_if_done(state: ConversationState) -> str:
            if state["turn_count"] >= state["max_turns"]:
                return END
            else:
                return "asker"

        state_builder.add_node("asker", asker_node)
        state_builder.add_node("respondent", respondent_node)
        state_builder.add_node("tool_executor", tool_executor_node)
        state_builder.add_node("finalize_turn", finalize_turn_node)

        state_builder.set_entry_point("asker")

        state_builder.add_edge("asker", "respondent")
        state_builder.add_conditional_edges(
            "respondent",
            route_for_respondent,
            {
                "tool_executor": "tool_executor",
                "finalize_turn": "finalize_turn",
                END: END
            }
        )
        state_builder.add_edge("tool_executor", "respondent")
        state_builder.add_conditional_edges(
            "finalize_turn",
            check_if_done,
            {
                "asker": "asker",
                END: END
            }
        )

        return state_builder.compile()

    def stream_dialog(self, config: Dict[str, Any]) -> Iterator[str]:
        try:
            llm_model_name = config.get('llm_model')
            llm = self._get_llm(llm_model_name)

            asker_config = config['asker']
            asker = Agent(
                name=asker_config['name'],
                system_message=asker_config['system_message'],
                model=llm,
                tools=[]
            )

            respondent_config = config['respondent']
            respondent_tools = [AVAILABLE_TOOLS[name] for name in respondent_config['tools'] if name in AVAILABLE_TOOLS]
            respondent = Agent(
                name=respondent_config['name'],
                system_message=respondent_config['system_message'],
                model=llm,
                tools=respondent_tools
            )
            print(respondent.tools)

            graph = self._build_graph(asker, respondent)
            initial_prompt = HumanMessage(content=config['initial_prompt'])
            initial_state: ConversationState = {
                "messages": [{"speaker": "User", "message": initial_prompt}],
                "turn_count": 0,
                "max_turns": int(config['max_turns'])
            }

            for update in graph.stream(initial_state, stream_mode='updates'):
                node_name, state_update = list(update.items())[0]

                print(f"--- Stream Update from Node: {node_name} ---")
                print(f"Update content: {state_update}")

                if "messages" not in state_update or not state_update["messages"]:
                    continue

                for speaker_message in state_update['messages']:
                    message_obj = speaker_message['message']
                    sender_name = speaker_message['speaker']

                    if isinstance(message_obj, (HumanMessage, ToolMessage)):
                        continue

                    payload = {}
                    if isinstance(message_obj, AIMessage):
                        if message_obj.tool_calls:
                            content = f"正在思考并准备调用工具: {[tc['name'] for tc in message_obj.tool_calls]}"
                        else:
                            content = message_obj.content
                        payload = {
                            "sender_name": sender_name,
                            "sender_type": "agent",
                            "content": content
                        }
                    print(payload)
                    if payload:
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_payload = {
                "sender_name": "系统错误",
                "sender_type": "system",
                "content": str(e)
            }
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'event_type': 'END'})}\n\n"
