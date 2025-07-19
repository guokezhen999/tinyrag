import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from agent.agent import Agent, ConversationState
from agent.tool import d2l_tool, search_tool

llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    api_key=os.environ.get("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )

asker = Agent(
    name="提问者",
    system_message="你是一个提问者，你需要对于深度学习的知识进行提问，一次只提一个具体问题，回答者是另一个AI。"
                   "你只需要提问，问题要简洁，不要进行对话的总结。不能有空的提问。",
    model=llm,
    tools=[]
)

respondent = Agent(
    name="回答者",
    system_message="你是一个回答者，需要根据已有对话和自身知识对提问者提出的问题进行回答。"
                   "请优先使用你被授予的工具来查找答案，如果工具无法提供答案，再使用你的通用知识。"
                   "回答要准确、专业，必须要有输出，不能有空的回复。",
    model=llm,
    tools=[d2l_tool, search_tool]
)

def asker_node(state: ConversationState) -> dict:
    """ 返回下一个agent需要的的state """
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

def finalize_turn_node(state: ConversationState) -> dict:
    """ 回合结束节点，负责更新回合次数 """
    return {"turn_count": state["turn_count"] + 1}


def route_for_respondent(state: ConversationState) -> str:
    """ respondent 的路由，选择调用工具或结束轮次 """
    last_message = state['messages'][-1]['message']
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"
    else:
        return "finalize_turn"

def check_if_done(state: ConversationState) -> str:
    """ finalize_turn 的路由，选择进行下一轮或者结束 """
    if state["turn_count"] >= state["max_turns"]:
        return END
    else:
        # 如果未达到最大轮数，则开始新一轮，轮到 Agent A
        return "asker"

if __name__ == '__main__':
    state_builder = StateGraph(ConversationState)

    # 节点
    state_builder.add_node("asker", asker_node)
    state_builder.add_node("respondent", respondent_node)
    state_builder.add_node("tool_executor",tool_executor_node)
    state_builder.add_node("finalize_turn", finalize_turn_node)

    # 入口
    state_builder.set_entry_point("asker")

    # 边
    state_builder.add_edge("asker", "respondent")

    state_builder.add_conditional_edges(
        "respondent",
        route_for_respondent,
        {
            "tool_executor": "tool_executor",
            "finalize_turn": "finalize_turn",
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

    app = state_builder.compile()

    # image_bytes = app.get_graph().draw_mermaid_png()
    # with open('../image/structure.png', "wb") as f:
    #     f.write(image_bytes)

    initial_message = HumanMessage(content="")
    initial_state: ConversationState = {
        "messages": [{'speaker': 'User', 'message': initial_message}],
        "turn_count": 0,
        "max_turns": 3,
    }

    final_state = app.invoke(initial_state)

    print(final_state)
    for speaker_message in final_state["messages"]:
        speaker = speaker_message['speaker']
        message = speaker_message['message']
        if isinstance(message, HumanMessage):
            print(f"人类: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"{speaker}: {message.content}")
            if message.tool_calls:
                print(f"  (请求工具: {[tc['name'] for tc in message.tool_calls]})")
        elif isinstance(message, ToolMessage):
            print(f"工具: {message.content}")