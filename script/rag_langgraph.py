from typing import List, TypedDict, Annotated
from operator import add

from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from rag.llm import PROMPT_TEMPLATE
from rag.vector_base import VectorStore
from rag.embeddings import BgeEmbedding

def get_deep_learning_knowledge(query: str) -> str:
    """
    这个 RAG 系统用于查询深度学习的知识。
    它会尝试在内部知识库中查找，并找到最相关的段落生成提示词。
    当需要查询深度学习相关知识时候，请调用这个函数。
    """
    vector = VectorStore()
    vector.load_vector(file='d2l')
    embedding = BgeEmbedding(device='mps')
    content = vector.query(query, EmbeddingModel=embedding, k=1)[0]
    prompt = PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE'].format(question=query, context=content)
    return prompt

rag_tool = Tool(
    name="deep_learning_knowledge",
    func=get_deep_learning_knowledge,
    description="当用户提问深度学习相关知识，请使用此工具进行查询。"
)

search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="google_search",
    func=search.run,
    description="当需要联网查询时，请使用此工具进行google搜索。"
)

tools = [rag_tool, search_tool]

class AgentState(TypedDict):
    input: str
    chat_history: Annotated[List[BaseMessage], add]

agent_llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
llm_with_tools = agent_llm.bind_tools(tools)

def run_agent_core(state: AgentState) -> AgentState:
    """ 核心决策逻辑，接受当前状态，调用 LLM 决定下一步"""
    print("--- 执行 Agent 核心决策 ---")
    print(state['chat_history'])
    response = llm_with_tools.invoke(state["chat_history"])
    state['chat_history'].append(response)
    return state

def execute_tools(state: AgentState) -> AgentState:
    """ 执行 Agent 决定调用的工具 """
    print("--- 执行工具 ---")
    last_message = state["chat_history"][-1]
    tool_messages = []

    if not last_message.tool_calls:
        print("警告: LLM 没有请求工具调用，但进入了工具执行节点。")
        return state

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        print(tool_call)

        tool_to_execute = next((t for t in tools if t.name == tool_name), None)
        if tool_to_execute:
            print(f"调用工具: {tool_name} with input: {tool_input}")
            tool_output = tool_to_execute.invoke(list(tool_input.values())[0])
            tool_messages.append(ToolMessage(
                content=tool_output,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            ))
            print(tool_messages)

    state["chat_history"].extend(tool_messages)
    return state

def should_continue(state: AgentState) -> str:
    """ 根据 Agent 核心的最新输出，决定下一步是继续执行工具还是结束。 """
    last_message = state["chat_history"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("--- 路由: 检测到工具调用，继续执行工具 ---")
        return "continue"
    else:
        print("--- 路由: 未检测到工具调用 (纯文本响应)，结束会话 ---")
        return "end"


workflow = StateGraph(AgentState)

workflow.add_node("agent_core", run_agent_core)
workflow.add_node("tool_node", execute_tools)

workflow.set_entry_point("agent_core")

# 从 agent_core 节点出发，根据 should_continue 的判断进行条件路由
workflow.add_conditional_edges(
    "agent_core",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

# 从 tool_node 节点出发，执行完工具后，总是返回 agent_core 节点
workflow.add_edge("tool_node", "agent_core")

app = workflow.compile()

prompt_1 = "请查询数据库，深度学习中的缩放点积注意力是什么？"
# prompt_1 = "联网搜索，今天上海的天气。"
print(f"\n用户: {prompt_1}")
final_state_1 = app.invoke({"chat_history": [HumanMessage(content=prompt_1)]})
print(f"\nAgent 最终答案: {final_state_1['chat_history'][-1].content}")










