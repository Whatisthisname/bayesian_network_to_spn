from collections import defaultdict
from math import sqrt
from typing import Dict, List, Set, Tuple
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from graphviz import Digraph
import matplotlib.pyplot as plt

class Node: 


    def __init__(self, name : str = "variable"):
        self.name = name
        self.children = []
        self.parents = []
        self.cache = cache()
        self.cache.factors_references = {self}
        self.cond_dist = lambda x: 0.0
        self.evidence = None

    def __radd__(self, other) -> "Node":
        return self.__add__(other)
    
    def __invert__(self) -> None:
        print("dabbed on", self)
    
    def __repr__(self) -> str:
        if self.parents == []:
            return f"{self.name}"
        else:
            return f"{self.name}|{','.join([p.name + ('=' + str(p.evidence) if p.evidence is not None else '') for p in self.parents])}"
        
    def add_children(self, children):
        if isinstance(children, List):
            for c in children:
                c.parents.append(self)
                self.children.append(c)
                c.cache = self.cache
        else:
            children.parents.append(self)
            self.children.append(children)
            children.cache = self.cache

    def deepcopy(self, copy_factors = False, __copied_nodes__ = None, __copied_map__ : dict = None) -> "Node":
        
        if copy_factors:
            all = []
            for factor in self.cache.factors_references:
                f = factor.deepcopy()
                all.append(f)
            
            for factor in all:
                factor.cache.factors_references = all
            return all

        is_outer_call = False
        if __copied_nodes__ is None:
            is_outer_call = True
            __copied_nodes__ = set()
            __copied_map__ = {}

        copy = Node(self.name)
        copy.evidence = self.evidence
        
        __copied_nodes__.add(self) # basically a proxy for copies.keys()
        __copied_map__[self] = copy
        
        for child in self.children:
            if child not in __copied_nodes__:
                child.deepcopy(__copied_nodes__= __copied_nodes__, __copied_map__= __copied_map__) # this will add the child to copied_nodes and copied_map
                
        for parent in self.parents:
            if parent not in __copied_nodes__: # use the existing copy
                parent.deepcopy(__copied_nodes__= __copied_nodes__, __copied_map__= __copied_map__)

            __copied_map__[parent].add_children(copy)

        if is_outer_call:
            copy.cache = cache()
            copy.cache.nodes = [__copied_map__[n] for n in self.get_nodes()]
            copy.cache.factors_references = set([copy.cache.nodes[0]])
            for n in copy.get_nodes():
                n.cache = copy.cache
            
        return copy
        
    def castrate_roots(self, across_factors = False) -> "Node":
        if across_factors:
            raise NotImplementedError()
            l = list(self.cache.factors_references)
            if len(l) == 1:
                return

            for factor in self.cache.factors_references:
                factor.castrate_roots(across_factors = False)
            return

        else:


            roots_children : Dict[Node, List[Node]] = defaultdict(lambda: [])

            for r in self.get_roots():
                for c in r.children:
                    roots_children[c] += r,
            

            new_roots : List[Node] = [] # these will all be new roots
            
            for c, root_parents in roots_children.items():
                
                if len(c.parents) == len(root_parents):
                    new_roots += c,



            handled = set()
            factors = [] # this will contain only a single root per connected pgm

            for n in new_roots:
                if n not in handled:
                    n : Node = n
                    factors.append(n)
                    n.cache = cache()
                    handled.update(n.get_nodes())
            
            if factors == []: # this means that all of the nodes in the pgm were root nodes
                return None
            else:
                #! add the new roots to one another (& operator)
                base_case = factors[0]
                for f in factors[1:]:
                    base_case = base_case & f
                
                return base_case
            
    # override & operator to join PGMs
    def __and__(self, other) -> None:
        if isinstance(other, Node):
            if other == self:
                raise ValueError("Cannot add a model with itself")
                        
            if other.get_nodes()[0] not in self.cache.factors_references:
                for o in other.cache.factors_references:
                    self.cache.factors_references.add(o)
                other.cache.factors_references = self.cache.factors_references
            
            return self

    def set_evidence(self, value : float | None) -> None:
        if np.any([p.evidence == None for p in self.parents]):
            raise ValueError("Cannot give evidence to a node who parent's haven't taken a value. (No interventional queries)")
         
        if isinstance(value, float) or isinstance(value, int) or value is None:
            self.evidence = value
        else:
            raise TypeError("value must be int, float or None")
        
        self.cache.roots = None

    def get_roots(self, include_deterministic=False, visited : Set = None, roots : Set = None) -> List["Node"]:
        
        if self.cache.roots is not None:
            return self.cache.roots

        return_only_roots = False
        if roots is None:
            return_only_roots = True
            roots = set()
        if visited is None:
            visited = set()

        if not include_deterministic:
            if self.evidence == None and np.all([p.evidence != None for p in self.parents]):
                roots.add(self)

        elif include_deterministic:
            if len(self.parents) == 0:
                roots.add(self)
        
        visited.add(self)
        
        for c in self.children:
            if c not in visited:
                visited, roots = c.get_roots(include_deterministic, visited, roots)
        
        for p in self.parents:
            if p in visited:
                continue
            if include_deterministic or (not include_deterministic and p.evidence != None):
                visited, roots = p.get_roots(include_deterministic,visited, roots)

        if return_only_roots:
            self.cache.roots = list(sorted(roots, key = lambda x: x.name))
            return self.cache.roots
        else:
            return visited, roots

    def get_scope(self, across_factors : bool = False) -> List[str]:
        
        if across_factors:
            all = []
            for factor in self.cache.factors_references:
                all.extend(factor.get_scope())
            return all
         
        nodes = self.get_nodes()
        scope = [n.name for n in nodes]

        return scope
    
    def get_nodes(self, across_factors : bool = False) -> List["Node"]:
        """returns all nodes of PGM in sorted primarily by topological order and then name"""
        if across_factors:
            all = []
            for factor in self.cache.factors_references:
                all.extend(factor.get_nodes())
            return all
        
        if self.cache.nodes is not None:
            return self.cache.nodes.copy()
        
        L = []
        S = self.get_roots().copy()
        removed_edges = defaultdict(lambda: 0)

        while len(S) > 0:
            n = S.pop(0)
            L.append(n)

            for c in sorted(n.children, key=lambda x: x.name):
                removed_edges[c] += 1
                if removed_edges[c] == len(c.parents):
                    S.append(c)
        
        self.cache.nodes = L
        return L.copy()
            
    def sample(self, n : int = 1000) -> pd.DataFrame:
        raise NotImplementedError()

    def get_graph(self, detailed = True) -> Digraph:
        
        G = Digraph(format='svg', graph_attr={'rankdir':'LR'})

        for f in sorted(self.cache.factors_references, key = lambda n: n.name):

            nodes = f.get_nodes()
            
            for n in nodes:
                label = n.name
                if detailed:
                    label += ':?' if n.evidence is None else ":"+str(n.evidence)
                G.node(n.name, label=label, shape='circle')

            visited = set()
            to_visit = f.get_roots().copy()

            while len(to_visit) > 0:
                r = to_visit.pop()
                visited.add(r)
                for c in r.children:
                    if c not in visited:
                        to_visit.append(c)
                
                for c in r.children:
                    G.edge(r.name, c.name)
        
        return G

    def partition_by_connected_components(self, across_factors = True):
        
        if across_factors:
            groups = []
            for f in sorted(self.cache.factors_references, key=lambda n:n.name):
                groups.extend(f.partition_by_connected_components(across_factors=False))
            return groups
        
        print("self:", self)
        
        groups = []
        found = set()

        for r in self.get_roots(include_deterministic=False):
            if r in found:
                continue
            

            print("r", r)

            L = []
            S = [r]
            removed_edges = defaultdict(lambda: 0)

            while len(S) > 0:
                n = S.pop(0)
                L.append(n)

                for c in sorted(n.children, key=lambda x: x.name):
                    removed_edges[c] += 1
                    if removed_edges[c] == len(c.parents):
                        S.append(c)
            print("L", L)
            print("S", S)
            found.update(L)
            groups.append(L)
        
        return [[n for n in C if np.all([p.evidence != None for p in n.parents])] for C in groups]
            

class cache:

    def __init__(self):
        self.roots : List = None
        self.nodes : List = None
        self.factors_references : Set = set()

class Distribution:
    
    def __init__(self, func):
        self.func = func
        
    # other > self
    def __lt__(self, other):
        if isinstance(other ,str):
            other = Node(name=other)
        
        elif not isinstance(other, Node):
            raise NotImplementedError()
        
        other.cond_dist = self.func
        return other

class ParentList:
    def __init__(self, *elems):
        self.elements = list(elems)

    # other | self
    def __ror__(self, other) -> Node:
        if isinstance(other, str):
            n = Node(name = other)
            for p in self.elements:
                p.add_children(n)
            return n
