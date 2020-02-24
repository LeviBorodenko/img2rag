import glob
import os.path as path

import graph_tool as gt
import numpy as np
import scipy.io as io
import sklearn.datasets as ds


def write_txt(graph_db, classes, name):
    f = open("datasets/" + name + "/" + name + "_A.txt", "w")
    f2 = open("datasets/" + name + "/" + name + "_graph_indicator.txt", "w")
    shift = 0

    i = 1
    for g in graph_db:
        for e in g.edges():
            v = g.vertex_index[e.source()] + shift + 1
            w = g.vertex_index[e.target()] + shift + 1

            f.write(str(v) + ", " + str(w) + "\n")
            f.write(str(w) + ", " + str(v) + "\n")

        for j, v in enumerate(g.vertices()):
            f2.write(str(i) + "\n")

        i += 1

        shift += g.num_vertices()

    f = open("datasets/" + name + "/" + name + "_graph_labels.txt", "w")
    # Write classes
    for c in classes:
        f.write(str(c) + "\n")

    labels = []
    for g in graph_db:
        for v in g.vertices():
            labels.append(g.vp.nl[v])

    el = []
    for g in graph_db:
        for e in g.edges():
            el.append(g.ep.ea[e][0])

    f = open("datasets/" + name + "/" + name + "_node_labels.txt", "w")
    # Write attributes
    for c in labels:
        f.write(str(c) + "\n")

    f = open("datasets/" + name + "/" + name + "_edge_attributes.txt", "w")
    # Write attributes
    for c in el:
        f.write(str(c) + "\n")


def read_txt(ds_name):
    s = "pygk/"
    s = ""
    pre = ""

    with open(
        s + "datasets/" + pre + ds_name + "/" + ds_name + "_graph_indicator.txt", "r"
    ) as f:
        graph_indicator = [int(i) - 1 for i in list(f)]
    f.closed
    #
    # # Nodes
    num_graphs = max(graph_indicator)
    node_indices = []
    offset = []
    c = 0
    #
    for i in range(num_graphs + 1):
        offset.append(c)
        c_i = graph_indicator.count(i)
        node_indices.append((c, c + c_i - 1))
        c += c_i

    graph_db = []
    vertex_list = []
    for i in node_indices:
        g = gt.Graph(directed=False)
        vertex_list_g = []
        for _ in range(i[1] - i[0] + 1):
            vertex_list_g.append(g.add_vertex())

        graph_db.append(g)
        vertex_list.append(vertex_list_g)

    # Edges
    with open(s + "datasets/" + pre + ds_name + "/" + ds_name + "_A.txt", "r") as f:
        edges = [i.split(",") for i in list(f)]
    f.closed

    edges = [(int(e[0].strip()) - 1, int(e[1].strip()) - 1) for e in edges]

    edge_indicator = []
    edge_list = []
    for e in edges:
        g_id = graph_indicator[e[0]]
        edge_indicator.append(g_id)
        g = graph_db[g_id]
        off = offset[g_id]

        # Avoid multigraph
        if not g.edge(e[0] - off, e[1] - off):
            edge_list.append(g.add_edge(e[0] - off, e[1] - off))

    # Node labels
    if path.exists(
        s + "datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt"
    ):
        with open(
            s + "datasets/" + pre + ds_name + "/" + ds_name + "_node_labels.txt", "r"
        ) as f:
            node_labels = [int(i) for i in list(f)]
        f.closed

        l_nl = []
        for i in range(num_graphs + 1):
            g = graph_db[graph_indicator[i]]
            l_nl.append(g.new_vertex_property("int"))

        for i, l in enumerate(node_labels):
            g_id = graph_indicator[i]
            off = offset[g_id]
            v = vertex_list[g_id][i - off]
            l_nl[g_id][v] = l
            #
            #    l_nl[g_id][v] = l

    # # Node Attributes
    # if path.exists(s + "datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt"):
    #     with open(s + "datasets/" + pre + ds_name + "/" + ds_name + "_node_attributes.txt", "r") as f:
    #         node_attributes = [map(float, i.split(',')) for i in list(f)]
    #     f.closed
    #
    #     l_na = []
    #     for i in range(0, num_graphs + 1):
    #         g = graph_db[graph_indicator[i]]
    #         l_na.append(g.new_vertex_property("vector<float>"))
    #
    #     for i, l in enumerate(node_attributes):
    #         g_id = graph_indicator[i]
    #         g = graph_db[g_id]
    #         off = offset[g_id]
    #         # l_na[g_id][vertex_list[g_id][i - off]] = l[0]
    #         l_na[g_id][vertex_list[g_id][i - off]] = np.array(l)
    #         g.vp.na = l_na[g_id]
    #
    # # Edge Labels
    # if path.exists("pygk/datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt"):
    #     with open("pygk/datasets/" + ds_name + "/" + ds_name + "_edge_labels.txt", "r") as f:
    #         edge_labels = [int(i) for i in list(f)]
    #     f.closed
    #
    #     l_el = []
    #     for i in range(num_graphs + 1):
    #         g = graph_db[graph_indicator[i]]
    #         l_el.append(g.new_edge_property("int"))
    #
    #     for i, l in enumerate(edge_labels):
    #         g_id = edge_indicator[i]
    #         g = graph_db[g_id]
    #
    #         l_el[g_id][edge_list[i]] = l
    #         g.ep.el = l_el[g_id]

    # Edge Attributes
    # if path.exists(s + "datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt"):
    #     with open(s + "datasets/" + ds_name + "/" + ds_name + "_edge_attributes.txt", "r") as f:
    #         edge_attributes = [map(float, i.split(',')) for i in list(f)]
    #     f.closed

    #     l_ea = []
    #     for i in range(num_graphs + 1):
    #         g = graph_db[graph_indicator[i]]
    #         l_ea.append(g.new_edge_property("vector<float>"))

    #     for i, l in enumerate(edge_attributes):
    #         g_id = edge_indicator[i]
    #         g = graph_db[g_id]

    #         l_ea[g_id][edge_list[i]] = l
    #         g.ep.ea = l_ea[g_id]

    # Classes
    with open(
        s + "datasets/" + pre + ds_name + "/" + ds_name + "_graph_labels.txt", "r"
    ) as f:
        classes = [int(i) for i in list(f)]
    f.closed

    return classes, classes
