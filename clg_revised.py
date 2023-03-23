from collections import defaultdict
from math import sqrt
from typing import Dict, List, Set, Tuple
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
    p_weigh : List[float]

    is_noise : bool
    cache : "cache" = None

    def __init__(self, mean : float, sd : float, name : str = "variable"):
        self._og_mean = float(mean)
        self.current_mean = float(mean)
        self._og_sd = float(sd)
        self.current_sd = float(sd)
        self.name = name
        self.children = []
        self.p_weigh = []
        self.parents = []
        self.is_noise = False
        self.cache = cache()

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
                    i.parents.append((self, 1))
                    i.noise_sd = other._og_sd
                    i.mean = other._og_mean
                    return i
                case False, False:
                    i = __intermediate__()
                    i.parents.append((self, 1))
                    i.parents.append((other, 1))
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
                i.parents.append((self, 1))
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
            i.parents.append ((self, other))
            return i
    
    def __rmul__(self, other) -> "Norm":
        return self.__mul__(other)

    def __repr__(self) -> str:
        return f"{'#' if self.is_noise else ''}{self.name} ~ N({round(self.current_mean,3)}, {round(self.current_sd**2, 3)})"

    def __matmul__(self, other) -> "Norm":
        if isinstance(other, str):
            self.cache.must_recompute = True
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
            c.p_weigh.append(o.p_weigh[o.parents.index(self)])
        
        copy.parents = []
        copy.p_weigh = []
        return copy

    def deepcopy(self, copy_factors = False, __copied_nodes__ = None, __copied_map__ : dict = None) -> "Norm":
        
        if copy_factors:
            all = []
            for factor in self.cache.factors_references:
                f = factor.deepcopy()
                all.append(f)
            
            for factor in all:
                factor.cache.factors_references = all
            return all

        recalculate_params_at_the_end = False
        if __copied_nodes__ is None:
            recalculate_params_at_the_end = True
            __copied_nodes__ = set()
            __copied_map__ = {}

        copy = Norm(self._og_mean, self._og_sd, self.name)
        copy.is_noise = self.is_noise
        
        __copied_nodes__.add(self) # basically a proxy for copies.keys()
        __copied_map__[self] = copy
        
        for child in self.children:
            if child not in __copied_nodes__:
                child.deepcopy(__copied_nodes__= __copied_nodes__, __copied_map__= __copied_map__) # this will add the child to copied_nodes and copied_map
                
        for parent, w in zip(self.parents, self.p_weigh):
            
            if parent not in __copied_nodes__: # use the existing copy
                parent.deepcopy(__copied_nodes__= __copied_nodes__, __copied_map__= __copied_map__)

            __copied_map__[parent].children.append(copy)
            copy.parents.append(__copied_map__[parent])
            copy.p_weigh.append(w)

        if recalculate_params_at_the_end:
            copy.cache = cache()
            for n in copy.get_nodes():
                n.cache = copy.cache
            
            copy.__recompute_params__()
        
        return copy
        
    def castrate_roots(self, across_factors = False) -> "Norm":
        if across_factors:
            raise NotImplementedError()
            l = list(self.cache.factors_references)
            if len(l) == 1:
                return

            for factor in self.cache.factors_references:
                factor.castrate_roots(across_factors = False)
            return

        else:

            cov = self.get_Σ()
            new_root_ref = None

            roots_children : Dict[Norm, List[Norm]] = defaultdict(lambda: [])

            for r in self.get_roots():
                for c in r.children:
                    roots_children[c] += r,
            

            new_roots : List[Norm] = [] # these will all be new roots
            for c, root_parents in roots_children.items():
                
                if len(c.parents) == len(root_parents):
                    new_roots += c,

                ws = [c.p_weigh[c.parents.index(p)] for p in root_parents]
                mean_change = 0
                var_change = 0
                for i, p1 in enumerate(root_parents):
                    
                    p1.children.pop(p1.children.index(c)) # remove the child
                    p_idx = c.parents.index(p1)
                    c.parents.pop(p_idx) # remove the parent
                    c.p_weigh.pop(p_idx) # remove the weight

                    mean_change += ws[i] * p1.current_mean # absorb the mean
                    var_change += ws[i]**2 * p1.current_sd**2 # absorb the variance

                    for j in range(i+1, len(root_parents)):
                        p2 = root_parents[j]
                        var_change += 2 * ws[i] * ws[j] * cov[p1.name][p2.name]
                
            
                c._og_mean += mean_change# absorb the mean
                # print(f"{c.name} var change is", var_change)
                c._og_sd = np.sqrt(c._og_sd**2 + var_change) # absorb the variance


            handled = set()
            factors = [] # this will contain only a single root per connected pgm

            for n in new_roots:
                if n not in handled:
                    n : Norm = n
                    factors.append(n)
                    n.cache = cache()
                    n.__recompute_params__()
                    handled.update(n.get_nodes())
            
            if factors == []: # this means that all of the nodes in the pgm were root nodes
                return None
            else:
                #! add the new roots to one another (& operator)
                base_case = factors[0]
                for f in factors[1:]:
                    base_case = base_case & f
                
                return base_case #list(base_case.cache.factors_references)
            
    # override & operator to join PGMs
    def __and__(self, other) -> None:
        if isinstance(other, Norm):
            if other.is_noise:
                raise ValueError("Cannot add an independent noise node. Assign a name to the noise with the @ operator")
            if other == self:
                raise ValueError("Cannot add a model with itself")
            
            self.__recompute_params__()
            other.__recompute_params__()
            
            if other.get_nodes()[0] not in self.cache.factors_references:
                for o in other.cache.factors_references:
                    self.cache.factors_references.add(o)
                other.cache.factors_references = self.cache.factors_references
            
            return self

    def condition(self, value : float | None, recompute_covariance_and_mean = True) -> None:
        if self.parents != []:
            raise ValueError("Cannot condition a non-root node")
        if self.is_noise:
            raise ValueError("Cannot condition noise node")
         
        if isinstance(value, float) or isinstance(value, int):
            self.current_mean = value
            self.current_sd = 0
        elif value is None:
            self.current_mean = self._og_mean + sum([p.current_mean * w for p, w in zip(self.parents, self.p_weigh)])
            self.current_sd = self._og_sd + sqrt(sum([(p.current_sd**2) * w**2 for p, w in zip(self.parents, self.p_weigh)]))
        else:
            raise TypeError("value must be a float or None")
        
        self.cache.must_recompute = True
        if recompute_covariance_and_mean:
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
            return list(sorted(roots, key = lambda x: x.name))
        else:
            return visited, roots

    def get_scope(self, across_factors : bool = False) -> List[str]:
        
        if across_factors:
            all = []
            for factor in self.cache.factors_references:
                all.extend(factor.get_scope())
            return all
        
        if self.cache.scope is not None:
            return self.cache.scope.copy()
        
        nodes = self.get_nodes()
        scope = [n.name for n in nodes if not n.is_noise]

        self.cache.scope = scope
        return scope.copy()
    
    def get_nodes(self, across_factors : bool = False) -> List["Norm"]:
        
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
    
    def get_µ(self) -> pd.Series:
        if self.cache.µ is not None:
            return self.cache.µ.copy()
        else:
            self.__recompute_params__()
            return self.get_µ()

    def get_Σ(self) -> pd.DataFrame:
        if self.cache.Σ is not None:
            return self.cache.Σ.copy()
        else:
            self.__recompute_params__()
            return self.get_Σ()
        
    def __recompute_params__(self) -> None:
        if not self.cache.must_recompute:
            return
        
        self.cache.must_recompute = False
        if len(self.cache.factors_references) == 0:
            self.cache.factors_references = set([self.get_roots()[0]])
        
        nodes = self.get_nodes()
        scope = self.get_scope()
        cov = pd.DataFrame(columns = scope, index = scope, dtype = float)
        cov = cov.fillna(0)

        mean = pd.Series(index = scope, dtype=float)
        mean.fillna(0)

        for i1, n1 in enumerate(nodes):
            n1 : Norm = n1
            
            mean_update = 0
            cov_update = 0

            if len(n1.parents) == 0:
                cov_update = n1.current_sd**2
                mean_update = n1.current_mean
            else:
                p_w_zip = list(zip(n1.parents, n1.p_weigh) )
                cov_update = n1._og_sd**2
                mean_update = n1._og_mean

                for p1_i, (p1, w1) in enumerate(zip(n1.parents, n1.p_weigh)):
                    mean_update += w1 * mean[p1.name]
                    cov_update += w1**2  * cov[p1.name][p1.name]
                    for p2_i in range(p1_i + 1, len(p_w_zip)):
                        p2, w2 = p_w_zip[p2_i]
                        cov_update += 2 * w1 * w2 * cov[p1.name][p2.name]
                    
                # for p1, w1 in zip(n1.parents, n1.p_weigh):
                #     mean_update += w1 * mean[p1.name]
                #     for p2, w2 in zip(n1.parents, n1.p_weigh):
                #         cov_update += w1 * w2 * cov[p1.name][p2.name]

            cov[n1.name][n1.name] = cov_update
            mean[n1.name] = mean_update

            for ic1, c1 in enumerate(n1.children):            
                
                # direct assignment of covariance to immediate children #! me to child, 0
                w1 = c1.p_weigh[c1.parents.index(n1)]
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
                    w2 = c2.p_weigh[c2.parents.index(n2)] # how much c2 depends on n2
                    
                    # from covariance between n1 and n2, we can compute the covariance between n2 and c1 by multiplying by the weight
                    cov[n1.name][c2.name] += w2 * cov[n1.name][n2.name]
                    cov[c2.name][n1.name] += w2 * cov[n1.name][n2.name]

        for n in nodes:
            n : Norm = n
            n.current_sd = np.sqrt(cov[n.name][n.name])
            n.current_mean = mean[n.name]
            if n is not self:
                del n.cache
                n.cache = self.cache
            
        self.cache.Σ = cov
        self.cache.µ = mean

    def sample(self, n : int = 1000) -> pd.DataFrame:
        from scipy.stats import multivariate_normal    
        return pd.DataFrame(multivariate_normal.rvs(mean=self.get_μ(), cov=self.get_Σ(), size=n), columns = self.get_scope())

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
            current.current_mean = reg.intercept_[0] + sum([current.p_weigh[i] * current.parents[i].current_mean for i in range(len(current.parents))])
            current.p_weigh = reg.coef_[0]

            # learn the independent noise term by calulating the residuals of the linear regression
            predictions = reg.predict(X)
            error = y - predictions
            current._og_sd = sqrt(np.var(error,  ddof = len(current.parents)))
            current.current_sd = sqrt(current._og_sd**2 + sum([(current.p_weigh[i]**2) * current.parents[i].current_sd**2 for i in range(len(current.parents))]))

            for c in current.children:
                if c not in visited:
                    to_visit.add(c)
   
    def get_graph(self, detailed = False) -> Digraph:
        
        if self.cache.Σ is None:
            self.__recompute_params__()

        G = Digraph(format='svg', graph_attr={'rankdir':'LR'})


        for f in self.cache.factors_references:

            nodes = f.get_nodes()
            
            for n in nodes:
                if detailed:
                    
                    sd_text = f"{n.current_sd**2:.1f}"
                    mean_text = f"{n.current_mean:.1f}"

                    if n.parents != []: 
                        mean_text = f"{n._og_mean:.1f} + {n.current_mean - n._og_mean:.1f}"
                        sd_text = f"{n._og_sd**2:.1f} + {n.current_sd**2 - n._og_sd**2:.1f}"
                                
                    G.node(n.name, label=f"{{{n.name}|{{{mean_text}\l|{sd_text}\l}}}}", shape = "record")
                else:
                    G.node(n.name, label=n.name, shape='circle')

            visited = set()
            to_visit = f.get_roots().copy()

            while len(to_visit) > 0:
                r = to_visit.pop()
                visited.add(r)
                for c in r.children:
                    if c not in visited:
                        to_visit.append(c)
                
                for c in r.children:
                    w = c.p_weigh[c.parents.index(r)]
                    if detailed:
                        G.edge(r.name, c.name, label = f'{c.p_weigh[c.parents.index(r)]:.1f}')
                    else:
                        G.edge(r.name, c.name)
        
        return G

    def cluster_roots_by_dependence(self, across_factors = True) -> List[Set["Norm"]]:
        
        if across_factors:
            all = []
            for f in self.cache.factors_references:
                all.extend(f.cluster_roots_by_dependence(across_factors = False))
            return all
        
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
            distinct.append(sorted(v, key = lambda x: x.name))
        return distinct

class __intermediate__:
    parents : List[Tuple[Norm, float]]
    mean : float
    """independent mean"""
    noise_sd : float
    """irreducible noise"""

    def __init__(self):
        self.parents = []
        self.mean = 0
        self.noise_sd = 0

    def __add__(self, other):
        if isinstance(other, __intermediate__):
            # for p, w in other.parents:
            #     p.__cache__ = self.parents[0][0].__cache__
            self.parents = self.parents.union(other.parents)
            self.mean += other.mean
            self.noise_sd = sqrt(self.noise_sd**2 + other.noise_sd**2)
            return self
        
        elif isinstance(other, Norm):
            # self.mean += other.current_mean
            if other.is_noise:
                self.noise_sd = sqrt(self.noise_sd**2 + other._og_sd**2)
                return self
            else:
                self.parents.append((other, 1))
                # other.__cache__ = self.parents[0][0].__cache__
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
            
            
            # n.__cache__ = self.parents[0][0].__cache__
            n.cache.must_recompute = True
            for p in self.parents:

                p[0].children.append(n)
                n.p_weigh.append(p[1])
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
    
class cache:

    def __init__(self):
        self.roots : List = None
        self.nodes : List = None
        self.scope : List = None
        self.Σ : pd.DataFrame = None
        self.µ : pd.Series = None

        self.must_recompute : bool = True

        self.factors_references : Set = set()

noise = Norm(0, 1, "noise")
noise.is_noise = True