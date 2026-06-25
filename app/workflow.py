from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes.route import route
from .nodes.retrieve import retrieve
from .nodes.grade import grade
from .nodes.generate import generate
from .nodes.evaluate import evaluate
from .edges.decisions import after_grade, expand


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("expand", expand)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"expand": "expand", "generate": "generate"})
    g.add_edge("expand", "retrieve")
    g.add_edge("generate", "evaluate")
    g.add_edge("evaluate", END)

    return g.compile()


graph = build_graph()
