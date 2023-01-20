import graphviz

from src.main.core import Value


def trace(root: Value) -> tuple:
    nodes, edges = set(), set()

    def build(value: Value):
        if value not in nodes:
            nodes.add(value)

            for child in value.children:
                edges.add((child, value))
                build(child)

    build(value=root)
    return nodes, edges


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root=root)

    for node in nodes:
        uid = str(id(node))
        dot.node(
            name=uid,
            label="{%s | data: %.4f | grad: %.4f}" % (node.label, node.data, node.grad),
            shape="record",
        )

        if node.operation:
            dot.node(name=uid + node.operation, label=node.operation)
            dot.edge(tail_name=uid + node.operation, head_name=uid)

    for node1, node2 in edges:
        dot.edge(tail_name=str(id(node1)), head_name=str(id(node2)) + node2.operation)

    return dot
