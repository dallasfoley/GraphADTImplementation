"""
CSE 331 FS23 (Onsay)
Graph Project 7
"""

import heapq
import itertools
import math
import random
import time
import csv
from typing import TypeVar, Callable, Tuple, \
    List, Set, Dict, Any

import numpy as np
from queue import SimpleQueue
import heapq


T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')  # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, id_init: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param id_init: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = id_init
        self.adj = {}  # dictionary {id : weight} of outgoing edges
        self.visited = False  # boolean flag used in search algorithms
        self.x, self.y = x, y  # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY.
        Equality operator for Graph Vertex class.
        :param other: [Vertex] vertex to compare.
        :return: [bool] True if vertices are equal, else False.
        """
        if self.id != other.id:
            return False
        if self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        if self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        if self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        if set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Constructs string representation of Vertex object.
        :return: [str] string representation of Vertex object.
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]
        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    __str__ = __repr__

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

    # ============== Project 9 Vertex Methods Below ==============#
    def deg(self) -> int:
        """
        Returns the degree (number of outgoing edges from) of this Vertex.
        :return: [int] Degree of this vertex.
        """
        return len(self.adj)

    def get_outgoing_edges(self) -> Set[Tuple[str, float]]:
        """
        Returns a set of tuples (end_id, weight) or empty set if no adjacencies.
        :return: [set[tuple[str, float]]] Set of tuples of the form (end_id, weight).
        """
        return set(self.adj.items())

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Computes the Euclidean distance between this Vertex and another Vertex.
        :param other: [Vertex] Other Vertex to which we wish to compute distance.
        :return: [float] Euclidean distance between this Vertex and another Vertex.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Returns the taxicab distance between this Vertex and another Vertex.
        :param other: [Vertex] Other Vertex to which we wish to compute distance.
        :return: [float] Taxicab distance between this Vertex and another Vertex.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csvf: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csvf : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csvf, delimiter=',', dtype=str).tolist() if csvf else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex_by_id(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    __str__ = __repr__

    def plot(self) -> None:
        """
        DO NOT MODIFY
        Creates a plot a visual representation of the graph using matplotlib
        """

        if self.plot_show:
            import matplotlib.cm as cm
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_all_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_all_edges())
            max_weight = max([edge[2] for edge in self.get_all_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_all_edges()):
                origin = self.get_vertex_by_id(edge[0])
                destination = self.get_vertex_by_id(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_all_vertices()])
            y = np.array([vertex.y for vertex in self.get_all_vertices()])
            labels = np.array([vertex.id for vertex in self.get_all_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_all_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03 * max(x), y[j] - 0.03 * max(y), labels[j])

            # show plot
            plt.show()

            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, begin_id: str, end_id: str = None, weight: float = 1) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param begin_id: unique string id of starting vertex
        :param end_id: unique string id of ending vertex
        :param weight: weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(begin_id) is None:
            self.vertices[begin_id] = Vertex(begin_id)
            self.size += 1
        if end_id is not None:
            if self.vertices.get(end_id) is None:
                self.vertices[end_id] = Vertex(end_id)
                self.size += 1
            self.vertices.get(begin_id).adj[end_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: an n x n square matrix (list of lists) representing Graph as adjacency map
        :return: None
        """
        for i in range(1, len(matrix)):  # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):  # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        given a graph, creates an adjacency matrix of the type described in "construct_from_matrix"
        :return: Matrix
        """
        matrix = [[None] + [v_id for v_id in self.vertices]]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        given a (non-empty) graph, creates a csv file containing data necessary to reconstruct that graph
        :param filepath: location to save CSV
        :return: None
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

    # ============== Graph Methods from Project 9 ==============#

    def unvisit_vertices(self) -> None:
        """
        Resets all visited flags of vertices in Graph to false
        :return: None
        """
        for vertex in self.vertices.values():
            vertex.visited = False

    def get_vertex_by_id(self, v_id: str) -> Vertex:
        """
        Retrieves a Vertex in the Graph by a provided v_id
        :param vertex_id: unique string id of other vertex
        :return: Vertex object if found; else None
        """
        return self.vertices.get(v_id)

    def get_all_vertices(self) -> Set[Vertex]:
        """
        Returns a set of all vertices in the Graph
        :return: set(Vertex)
        """
        return set(self.vertices.values())

    def get_edge_by_ids(self, begin_id: str, end_id: str) -> Tuple[str, str, float]:
        """
        Returns the edge connecting the vertex with id begin_id to the vertex with id end_id
        If edge does not exist, returns None
        :return: tuple(begin_id, end_id, weight) or None if edge does not exist
        """
        if self.vertices.get(begin_id) and self.vertices.get(end_id):
            weight = self.vertices.get(begin_id).adj.get(end_id)
            return (begin_id, end_id, weight) if weight is not None else None

    def get_all_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Returns all edges in the graph
        :return: set(tuple(begin_id, end_id, weight)) or empty list if Graph is empty
        """
        return {(begin_id, end_id, v.adj[end_id])
                for begin_id, v in self.vertices.items() for end_id in v.adj}

    def build_path(self, back_edges: Dict[str, str], begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Given a dictionary of back-edges (a mapping of vertex id to predecessor vertex id),
        reconstruct the path from start_id to end_id and compute the total distance
        :param back_edges: Dictionary of back-edges, i.e., (key=vertex_id, value=predecessor_vertex_id) pairs
        :param start_id: Starting vertex ID string from which to construct path
        :param end_id: Ending vertex ID string to which to construct path
        :return: Tuple where first element is a list of vertex IDs specifying a path from start_id to end_id
                 and the second element is the cost of this path
        """
        path, dist = [end_id], 0
        while path[-1] != begin_id:
            path.append(back_edges.get(path[-1]))  # will construct path in O(V) - no calls to .insert(0)
            dist += self.get_edge_by_ids(path[-1], path[-2])[2]
        return list(reversed(path)), dist

    # ============== Modify Graph Methods Below ==============#
    def bfs(self, begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Performs a breadth-first search on the graph to find the shortest path between two vertices
        :param begin_id: The id of the vertex to start the search from
        :param end_id: The id of the vertex to end the search at
        :return: A tuple containing a list of the vertex ids in the path and the total weight of the path
        """
        if begin_id not in self.vertices or end_id not in self.vertices:
            return [], 0

        visited = {vertex_id: False for vertex_id in self.vertices}
        back_edges = {vertex_id: None for vertex_id in self.vertices}
        queue = SimpleQueue()
        queue.put(begin_id)
        visited[begin_id] = True

        while not queue.empty():
            current_id = queue.get()
            if current_id == end_id:
                break

            for neighbor_id in self.vertices[current_id].adj:
                if not visited[neighbor_id]:
                    visited[neighbor_id] = True
                    queue.put(neighbor_id)
                    back_edges[neighbor_id] = current_id

        path, total_weight = [], 0
        node = end_id

        while node:
            path.append(node)
            prev_node = back_edges[node]
            if prev_node:
                total_weight += self.vertices[prev_node].adj[node]
            node = prev_node

        path.reverse()
        return path if path[0] == begin_id else [], total_weight

    def a_star(self, begin_id: str, end_id: str,
               metric: Callable[[Vertex, Vertex], float]) -> Tuple[List[str], float]:
        """
        Performs an A* search on the graph to find the shortest path between two vertices
        :param begin_id: The id of the vertex to start the search from
        :param end_id: The id of the vertex to end the search at
        :param metric: The metric used to determine the distance between two vertices
        :return: A tuple containing a list of the vertex ids in the path and the total weight of the path
        """

        if begin_id not in self.vertices or end_id not in self.vertices:
            return [], 0

        open_set = PriorityQueue()
        open_set.push(0, self.get_vertex_by_id(begin_id))
        g_scores = {begin_id: 0}
        came_from = {begin_id: None}

        while not open_set.empty():
            current_vertex = open_set.pop()[1]

            if current_vertex.id == end_id:
                path = []
                while current_vertex:
                    path.append(current_vertex.id)
                    current_vertex = self.get_vertex_by_id(came_from.get(current_vertex.id))
                path.reverse()

                total_cost = sum(self.get_edge_by_ids(path[i], path[i + 1])[2] for i in range(len(path) - 1))
                return path, total_cost

            for adjacent_vertex, connecting_weight in current_vertex.adj.items():
                tentative_g = g_scores[current_vertex.id] + connecting_weight
                if tentative_g < g_scores.get(adjacent_vertex, float('inf')):
                    came_from[adjacent_vertex] = current_vertex.id
                    g_scores[adjacent_vertex] = tentative_g
                    f_score = tentative_g + metric(self.get_vertex_by_id(adjacent_vertex), self.get_vertex_by_id(end_id))

                    if adjacent_vertex in open_set.locator:
                        open_set.update(f_score, self.get_vertex_by_id(adjacent_vertex))
                    else:
                        open_set.push(f_score, self.get_vertex_by_id(adjacent_vertex))

        return [], 0




class PriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Created by Andrew McDonald
    Inspired by https://docs.python.org/2/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
            Construct an AStarPriorityQueue object
            """
        self.data = []  # underlying data list of priority queue
        self.locator = {}  # dictionary to locate vertices within priority queue
        self.counter = itertools.count()  # used to break ties in prioritization

    def __repr__(self) -> str:
        """
            Represent AStarPriorityQueue as a string
            :return: string representation of AStarPriorityQueue object
            """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
            Represent AStarPriorityQueue as a string
            :return: string representation of AStarPriorityQueue object
            """
        return repr(self)

    def empty(self) -> bool:
        """
            Determine whether priority queue is empty
            :return: True if queue is empty, else false
            """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
            Push a vertex onto the priority queue with a given priority
            :param priority: priority key upon which to order vertex
            :param vertex: Vertex object to be stored in the priority queue
            :return: None
            """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
            Remove and return the (priority, vertex) tuple with lowest priority key
            :return: (priority, vertex) tuple where priority is key,
            and vertex is Vertex object stored in priority queue
            """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]  # remove from locator dict
        vertex.visited = True  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)  # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
            Update given Vertex object in the priority queue to have new priority
            :param new_priority: new priority on which to order vertex
            :param vertex: Vertex object for which priority is to be updated
            :return: None
            """
        node = self.locator.pop(vertex.id)  # delete from dictionary
        node[-1] = None  # invalidate old node
        self.push(new_priority, vertex)  # push new node


def teleport(system_dict, start, end):
    """
    Given a system dictionary, a start vertex, and an end vertex, returns the cost of the shortest path between the
    start and end vertices. If no path exists, returns infinity.
    :param system_dict: A dictionary containing the system data
    :param start: A tuple containing the start galaxy and vertex
    :param end: A tuple containing the end galaxy and vertex
    :return: The cost of the shortest path between the start and end vertices
    """

    origin_galaxy, origin_vertex = start
    target_galaxy, target_vertex = end

    if origin_galaxy not in system_dict or target_galaxy not in system_dict:
        return float('inf')

    galaxy_network = Graph()
    for galaxy, galaxy_data in system_dict.items():
        galaxy_graph = galaxy_data["graph"]
        for vertex in galaxy_graph.get_all_vertices():
            for adjacent, path_weight in vertex.get_outgoing_edges():
                galaxy_network.add_to_graph((galaxy, vertex.id), (galaxy, adjacent), path_weight)

        teleportation_point = galaxy_data["departure_teleport"]
        for destination_galaxy in galaxy_data["departure_destinations"]:
            if destination_galaxy in system_dict:
                arrival_point = system_dict[destination_galaxy]["arrival_teleport"]
                galaxy_network.add_to_graph((galaxy, teleportation_point), (destination_galaxy, arrival_point), 0)

    if not galaxy_network.get_vertex_by_id((origin_galaxy, origin_vertex)) or not galaxy_network.get_vertex_by_id((target_galaxy, target_vertex)):
        return float('inf')

    shortest_path, travel_cost = galaxy_network.a_star((origin_galaxy, origin_vertex), (target_galaxy, target_vertex), Vertex.taxicab_distance)

    return travel_cost if shortest_path else float('inf')