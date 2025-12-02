from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from chemgraph.graphs.single_agent import BasicToolNode, route_tools
from chemgraph.state.active_space_state import ActiveSpaceState
from chemgraph.prompt.active_space_prompts import (
    active_space_reasoner_prompt,
    active_space_executor_prompt,
    active_space_critic_prompt,
)
from chemgraph.tools.pyscf_tools import (
    propose_active_space_guess,
    run_pyscf_casscf,
    evaluate_casscf_diagnostics,
)
#Right now each agent has the ability to call any tool.
def route_critic(state: ActiveSpaceState) -> str:
    """Decide whether to loop back for refinement or finish after the critic."""
    messages = state if isinstance(state, list) else state.get("messages", [])
    if not messages:
        return "done"

    last_message = messages[-1]
    content = getattr(last_message, "content", last_message)
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(str(chunk.get("text") or chunk.get("content") or chunk))
            else:
                parts.append(str(chunk))
        content_text = " ".join(parts)
    else:
        content_text = str(content)

    text = content_text.lower()
    if any(keyword in text for keyword in ("refine", "retry", "adjust", "loop back")):
        return "refine"
    if "needs_revision" in text or "did not converge" in text or "expand the active space" in text:
        return "refine"
    return "done"

def ReasonerAgent(state: ActiveSpaceState, llm: ChatOpenAI, system_prompt: str, tools: list):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def ExecutorAgent(state: ActiveSpaceState, llm: ChatOpenAI, system_prompt: str, tools: list):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def CriticAgent(state: ActiveSpaceState, llm: ChatOpenAI, system_prompt: str, tools: list):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    # If diagnostics already ran, skip to avoid repeat calls.
    has_diagnostics = any(
        getattr(m, "name", None) == "evaluate_casscf_diagnostics"
        or (isinstance(m, dict) and m.get("name") == "evaluate_casscf_diagnostics")
        for m in state.get("messages", [])
    )
    if has_diagnostics:
        return {"messages": [llm.invoke(messages)]}

    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_active_space_graph(
    llm: ChatOpenAI,
    reasoner_prompt: str = active_space_reasoner_prompt,
    executor_prompt: str = active_space_executor_prompt,
    critic_prompt: str = active_space_critic_prompt,
    tools: list = None,
):
    """Construct a three-agent active-space graph (Reasoner → Executor → Critic)."""
    checkpointer = MemorySaver()

    if tools is None:
        tools = [propose_active_space_guess, run_pyscf_casscf, evaluate_casscf_diagnostics]

    # Dedicated tool nodes per agent to keep control flow clear.
    reasoner_tools = BasicToolNode(tools=tools)
    executor_tools = BasicToolNode(tools=tools)
    critic_tools = BasicToolNode(tools=tools)

    graph_builder = StateGraph(ActiveSpaceState)

    graph_builder.add_node(
        "Reasoner",
        lambda state: ReasonerAgent(state, llm, system_prompt=reasoner_prompt, tools=tools),
    )
    graph_builder.add_node(
        "Executor",
        lambda state: ExecutorAgent(state, llm, system_prompt=executor_prompt, tools=tools),
    )
    graph_builder.add_node(
        "Critic",
        lambda state: CriticAgent(state, llm, system_prompt=critic_prompt, tools=tools),
    )

    graph_builder.add_node("reasoner_tools", reasoner_tools)
    graph_builder.add_node("executor_tools", executor_tools)
    graph_builder.add_node("critic_tools", critic_tools)

    graph_builder.add_edge(START, "Reasoner")
    # Reasoner phase: allow tool calls, then hand off to Executor.
    graph_builder.add_conditional_edges(
        "Reasoner",
        route_tools,
        {"tools": "reasoner_tools", "done": "Executor"},
    )
    graph_builder.add_edge("reasoner_tools", "Reasoner")
    # Executor phase: run computation, then pass to Critic.
    graph_builder.add_conditional_edges(
        "Executor",
        route_tools,
        {"tools": "executor_tools", "done": "Critic"},
    )
    graph_builder.add_edge("executor_tools", "Executor")
    # Critic phase: run diagnostics once, then finish.
    graph_builder.add_conditional_edges(
        "Critic",
        route_tools,
        {"tools": "critic_tools", "done": "CriticRouter"},
    )
    graph_builder.add_edge("critic_tools", "Critic")
    graph_builder.add_node("CriticRouter", lambda state: state)
    graph_builder.add_conditional_edges(
        "CriticRouter",
        route_critic,
        {"refine": "Reasoner", "done": END},
    )

    return graph_builder.compile(checkpointer=checkpointer)
