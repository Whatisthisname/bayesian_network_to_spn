from collections import defaultdict
from math import sqrt
from typing import List, Set, Tuple
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from graphviz import Digraph
import matplotlib.pyplot as plt

class Norm:
    _og_mean : float
    """independent mean"""
    current_mean : float
    """mean after conditioning"""
    _og_sd : float
    """independent noise"""
    current_sd : float
    """sd after conditioning"""
    
    name : str
    children : List["Norm"]
    parents : List["Norm"]
    p_weights : List[float]

    noise : "Norm"
    is_noise : bool

    def __init__(self, mean : float, sd : float, name : str = "variable"):
        self._og_mean = float(mean)
        self.current_mean = float(mean)
        self._og_sd = float(sd)
        self.current_sd = float(sd)
        self.name = name
        self.children = []
        self.p_weights = []
        self.parents = []
        self.is_noise = False

        self.__cache__ = __cache__()

    def __neg__(self) -> "Norm":
        return -1 * self

    def __add__(self, other) -> "Norm":
        
        if isinstance(other, Norm):
            match self.is_noise, other.is_noise:
                case True, True:
                    n = Norm(mean = self._og_mean + other._og_mean, sd = sqrt(self._og_sd**2 + other._og_sd**2), name = "noise")
                    n.is_noise = True
                    return n
                case True, False:
                    return other + self
                case False, True:
                    i = __intermediate__()  
                    i.parents.add((self, 1))
                    i.noise_sd = other._og_sd
                    i.mean = other._og_mean
                    return i
                case False, False:
                    i = __intermediate__()
                    i.parents.add((self, 1))
                    i.parents.add((other, 1))
                    return i

        elif isinstance(other, __intermediate__):
            return other + self

        else:
            try:
                other = float(other)
            except:
                raise TypeError("Cannot add a Norm with", type(other), "type")
            if self.is_noise:
                n = Norm(mean = self._og_mean + other, sd = self._og_sd, name = "noise")
                n.is_noise = True
                return n
            else:
                i = __intermediate__()
                i.mean = other
                i.parents.add((self, 1))
            return i

    def __radd__(self, other) -> "Norm":
        return self.__add__(other)
    
    def __sub__(self, other) -> "Norm":
        return self + (-1 * other)
    
    def __rsub__(self, other) -> "Norm":
        return (-1 * self) + other

    def __mul__(self, other) -> "Norm":
        try:
            other = float(other)
        except:
            raise TypeError("Cannot multiply a Norm with", type(other), "type")
        
        if self.is_noise:
            n = Norm(mean = self._og_mean * other, sd = self._og_sd * other, name = "noise")
            n.is_noise = True
            return n
        else:
            i = __intermediate__()
            i.parents.add ((self, other))
            return i
    
    def __rmul__(self, other) -> "Norm":
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"{'#' if self.is_noise else ''}{self.name} ~ N({round(self.current_mean,3)}, {round(self.current_sd**2, 3)})"

    def __matmul__(self, other) -> "Norm":
        if isinstance(other, str):
            if self.is_noise:
                n = Norm(mean = self._og_mean, sd = self._og_sd, name = other)
                return n
            else:
                return (1.0 * self) @ other

    def __rmatmul__(self, other) -> "Norm":
        return self.__matmul__(other)

    def copy_subtree(self) -> "Norm":
        if self.is_noise:
            raise ValueError("Cannot copy noise node")
        copy = Norm(self._og_mean, self._og_sd, self.name)
        copy.current_sd = self.current_sd
        copy.current_mean = self.current_mean

        copy.children = [ c.copy_subtree() for c in self.children]
        for c, o in zip(copy.children, self.children):
            c.parents.append(copy)
            c.p_weights.append(o.p_weights[o.parents.index(self)])
        
        copy.parents = []
        copy.p_weights = []
        return copy

    def condition(self, value : float | None) -> None:
        if self.parents != []:
            raise ValueError("Cannot condition a non-root node")
        if self.is_noise:
            raise ValueError("Cannot condition noise node")
         
        if isinstance(value, float) or isinstance(value, int):
            self.current_mean = value
            self.current_sd = 0
        elif value is None:
            self.current_mean = self._og_mean + sum([p.current_mean * w for p, w in zip(self.parents, self.p_weights)])
            self.current_sd = self._og_sd + sqrt(sum([(p.current_sd**2) * w**2 for p, w in zip(self.parents, self.p_weights)]))
        else:
            raise TypeError("value must be a float or None")
        
        self.__recompute_params__()
    
    def get_roots(self, visited : Set = None, roots : Set = None) -> List["Norm"]:
        
        return_only_roots = False
        if roots is None:
            return_only_roots = True
            roots = set()
        if visited is None:
            visited = set()

        if self.parents == []:
            roots.add(self)
        
        visited.add(self)
        
        for c in self.children:
            if c not in visited:
                visited, roots = c.get_roots(visited, roots)
        
        for p in self.parents:
            if p not in visited:
                visited, roots = p.get_roots(visited, roots)

        if return_only_roots:
            return list(roots)
        else:
            return visited, roots

    def get_scope(self) -> List[str]:
        if self.__cache__.scope is not None:
            return self.__cache__.scope.copy()
        
        nodes = self.get_nodes()
        scope = [n.name for n in nodes if not n.is_noise]

        self.__cache__.scope = scope
        return scope.copy()
    
    def get_nodes(self) -> List["Norm"]:
        if self.__cache__.nodes is not None:
            return self.__cache__.nodes.copy()
        
        L = []
        S = self.get_roots().copy()
        removed_edges = defaultdict(lambda: 0)

        while len(S) > 0:
            n = S.pop(0)
            L.append(n)

            for c in n.children:
                removed_edges[c] += 1
                if removed_edges[c] == len(c.parents):
                    S.append(c)
        
        self.__cache__.nodes = L
        return L.copy()
    
    def get_µ(self) -> pd.Series:
        if self.__cache__.µ is not None:
            return self.__cache__.µ.copy()
        else:
            self.__recompute_params__()
            return self.get_µ()

    def get_Σ(self) -> pd.DataFrame:
        if self.__cache__.Σ is not None:
            return self.__cache__.Σ.copy()
        else:
            self.__recompute_params__()
            return self.get_Σ()
        
    def __recompute_params__(self) -> None:

        nodes = self.get_nodes()
        scope = [n.name for n in nodes]
        cov = pd.DataFrame(columns = scope, index = scope, dtype = float)
        cov = cov.fillna(0)

        mean = pd.Series(index = scope, dtype=float)
        mean.fillna(0)

        for i1, n1 in enumerate(nodes):
            n1 : Norm = n1
            if len(n1.parents) == 0:
                cov[n1.name][n1.name] = n1.current_sd**2
                mean[n1.name] = n1.current_mean
            else:
                cov[n1.name][n1.name] = n1._og_sd**2
                mean[n1.name] = n1._og_mean
                for p1, w1 in zip(n1.parents, n1.p_weights):
                    mean[n1.name] += w1 * mean[p1.name]
                    for p2, w2 in zip(n1.parents, n1.p_weights):
                        cov[n1.name][n1.name] += w1 * w2 * cov[p1.name][p2.name]

            for ic1, c1 in enumerate(n1.children):            
                
                # direct assignment of covariance to immediate children #! me to child, 0
                w1 = c1.p_weights[c1.parents.index(n1)]
                var = cov[n1.name][n1.name]
                cov[n1.name][c1.name] += w1*var
                cov[c1.name][n1.name] += w1*var

                for i2 in range(i1+1, len(nodes)): #! child to node
                    n2 : Norm = nodes[i2]
                    
                    # from covariance between n1 and n2, we can compute the covariance between n2 and c1 by multiplying by the weight
                    cov[c1.name][n2.name] += w1 * cov[n1.name][n2.name]
                    cov[n2.name][c1.name] += w1 * cov[n1.name][n2.name]

            for i2 in range(i1+1, len(nodes)): #! node to child, 1:n
                n2 : Norm = nodes[i2]
                for c2 in n2.children:
                    w2 = c2.p_weights[c2.parents.index(n2)] # how much c2 depends on n2
                    

                    # from covariance between n1 and n2, we can compute the covariance between n2 and c1 by multiplying by the weight
                    cov[n1.name][c2.name] += w2 * cov[n1.name][n2.name]
                    cov[c2.name][n1.name] += w2 * cov[n1.name][n2.name]

        for n in nodes:
            n : Norm = n
            n.current_sd = np.sqrt(cov[n.name][n.name])
            n.current_mean = mean[n.name]

        self.__cache__.Σ = cov
        self.__cache__.µ = mean

    def forward_sample(self, n : int = 1000) -> pd.DataFrame:
        nodes = self.get_nodes()
        scope = [n.name for n in nodes]
        samples = {n : [] for n in scope}

        roots = self.get_roots()
        for _ in range(n):
            
            # reset all nodes to their original distribution
            for r in roots:
                r.condition(None)
            
            for current in self.nodes:
                if current.current_sd == 0:
                    samples[current.name] += current.current_mean,
                else:
                    sample = scipy.stats.norm.rvs(loc = current.current_mean, scale = current.current_sd)
                    current.condition(sample)
                    samples[current.name] += sample,

        for n in nodes:
            n.condition(None)
                
        return pd.DataFrame(samples, columns = scope)

    def fit_data(self, data : pd.DataFrame) -> None:
        
        # for each root, estimate the mean as simply the mean of the data
        # and the variance as the variance of the data

        roots = self.get_roots()

        for r in roots:
            r.current_mean = data[r.name].mean()
            r.current_sd = data[r.name].std()
            # print(r)

        # each child is a conditional linear gaussian given its parents. Use linear regression to estimate the dependence on the parents
        # and the variance of the noise
        
        to_visit = set()
        for r in roots:
            to_visit = to_visit.union(r.children)
        
        visited = set()
        
        while len(to_visit) > 0:
            current = to_visit.pop()
            visited.add(current)

            # learn the weights
            X = data[[p.name for p in current.parents]]
            y = data[[current.name]]

            reg = LinearRegression().fit(X, y)
            current.current_mean = reg.intercept_[0] + sum([current.p_weights[i] * current.parents[i].current_mean for i in range(len(current.parents))])
            current.p_weights = reg.coef_[0]

            # learn the independent noise term by calulating the residuals of the linear regression
            predictions = reg.predict(X)
            error = y - predictions
            current._og_sd = sqrt(np.var(error,  ddof = len(current.parents)))
            current.current_sd = sqrt(current._og_sd**2 + sum([(current.p_weights[i]**2) * current.parents[i].current_sd**2 for i in range(len(current.parents))]))

            for c in current.children:
                if c not in visited:
                    to_visit.add(c)
   
    def get_graph(self, detailed = False) -> Digraph:
        
        if self.__cache__.Σ is None:
            self.__recompute_params__()

        G = Digraph(format='svg', graph_attr={'rankdir':'LR'})

        nodes = self.get_nodes()
        
        for n in nodes:
            if detailed:
                if n.parents == []:
                    G.node(n.name, label=f"{{{n.name}|{{{n.current_mean:.1f}\l|{n.current_sd**2}\l}}}}", shape = "record")
                else:
                    if n.current_sd == 0:
                        G.node(n.name, label=f"{{{n.name}|{{{n.current_mean:.1f}\l|{n.current_sd**2}\l}} }}", shape = "record")
                    else:
                        G.node(n.name, label=f"{{{n.name}|{{{n.current_mean:.1f}\l|{n.current_sd**2 - n._og_sd**2:.1f}+{n._og_sd**2:.1f}\l}}}}", shape = "record")
            else:
                G.node(n.name, label=n.name, shape='circle')

        visited = set()
        to_visit = self.get_roots().copy()

        while len(to_visit) > 0:
            r = to_visit.pop()
            visited.add(r)
            for c in r.children:
                if c not in visited:
                    to_visit.append(c)
            
            for c in r.children:
                G.edge(r.name, c.name, label = f'{c.p_weights[c.parents.index(r)]:.1f}')
        
        return G

    def cluster_roots_by_dependence(self) -> List[Set["Norm"]]:
        
        cov = self.get_Σ()
        roots = self.get_roots().copy()
        groups = {r.name : set([r]) for r in roots}
        nodes = self.get_nodes()

        for i1, r1 in enumerate(roots):
            col1 = cov[r1.name]
            for i2 in range(i1+1, len(roots)):
                r2 = roots[i2]    
                col2 = cov[r2.name]

                # if they both have a nonzero value in the same row, they are correlated
                if any([col1[r.name] != 0 and col2[r.name] != 0 for r in nodes]):
                    for r in groups[r2.name]:
                        groups[r1.name].add(r)

                    groups[r2.name] = groups[r1.name]
                    
        done = set()
        distinct = []
        for k, v in groups.items():
            if k in done:
                continue
            
            for r in v:
                done.add(r.name)
            distinct.append(v)
        return distinct


class __intermediate__:
    parents : Set[Tuple[Norm, float]]
    mean : float
    """independent mean"""
    noise_sd : float
    """irreducible noise"""

    def __init__(self):
        self.parents = set()
        self.mean = 0
        self.noise_sd = 0

    def __add__(self, other):
        if isinstance(other, __intermediate__):
            self.parents = self.parents.union(other.parents)
            self.mean += other.mean
            self.noise_sd = sqrt(self.noise_sd**2 + other.noise_sd**2)
            return self
        
        elif isinstance(other, Norm):
            self.mean += other.current_mean
            
            if other.is_noise:
                self.noise_sd = sqrt(self.noise_sd**2 + other._og_sd**2)
                return self
            else:
                self.parents.add((other, 1))
            return self
        
        else:
            try:
                other = float(other)
            except:
                raise TypeError("Cannot add intermediate to type " + str(type(other)))
            self.mean += other
            return self

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __matmul__(self, other):
        if isinstance(other, str):
            mean = sum([p[0].current_mean * p[1] for p in self.parents])
            var = sum([p[0].current_sd**2 * p[1]**2 for p in self.parents]) + self.noise_sd**2
            n = Norm(mean = self.mean, sd = 0, name= other)
            n.current_mean = mean + self.mean
            n._og_sd = self.noise_sd
            n.current_sd = sqrt(var)
            

            for p in self.parents:
                del p[0].__cache__
                p[0].__cache__ = n.__cache__
                p[0].children.append(n)
                n.p_weights.append(p[1])
                n.parents.append(p[0])
        return n

    def __rmatmul__(self, other):
        return self.__matmul__(other)


    def __mul__(self, other):
        try:
            other = float(other)
        except:
            raise TypeError("Cannot multiply intermediate by type " + str(type(other)))
        
        i = __intermediate__()
        i.parents = self.parents.copy()
        i.mean = self.mean * other
        i.noise_sd = self.noise_sd * other
        i.parents = set([(p[0], p[1] * other) for p in i.parents])
        return i
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Unnamed random variable. Define with @ operator."
    
class __cache__:

    def __init__(self):
        self.roots : List = None
        self.nodes : List = None
        self.scope : List = None
        self.Σ : pd.DataFrame = None
        self.µ : pd.Series = None

    def clear(self):
        self.roots = None
        self.nodes = None
        self.scope = None
        self.Σ = None
        self.µ = None


noise = Norm(0, 1, "noise")
noise.is_noise = True


