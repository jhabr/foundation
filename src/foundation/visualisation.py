import graphviz

from src.foundation.core import Scalar


"""
Inspired by https://github.com/karpathy/micrograd/tree/master/micrograd
"""


def trace(root: Scalar) -> tuple:
    nodes, edges = set(), set()

    def build(value: Scalar):
        if value not in nodes:
            nodes.add(value)

            for child in value.children:
                edges.add((child, value))
                build(child)

    build(value=root)
    return nodes, edges


def draw_graph(root: Scalar) -> graphviz.Digraph:
    graph = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root=root)

    for node in nodes:
        uid = str(id(node))
        graph.node(
            name=uid,
            label="{%s | data: %.4f | grad: %.4f}" % (node.label, node.data, node.grad),
            shape="record",
        )

        if node.operation:
            graph.node(name=uid + node.operation, label=node.operation)
            graph.edge(tail_name=uid + node.operation, head_name=uid)

    for node1, node2 in edges:
        graph.edge(tail_name=str(id(node1)), head_name=str(id(node2)) + node2.operation)

    return graph
