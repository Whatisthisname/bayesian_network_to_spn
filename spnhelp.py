from collections import defaultdict
from graphviz import Digraph
import numpy as np
from scipy.stats import norm
from typing import Tuple, List
import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState
from itertools import product
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Node, Sum, Product, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Gaussian, Uniform
from spn.algorithms.Inference import likelihood
from spn.algorithms.Marginalization import marginalize
import scipy.stats as stats


def sample_from_spn(spn : Node, amount : int) -> np.ndarray:
    rng = RandomState(None)
    samples = sample_instances(spn, np.full((amount, len(spn.scope)), np.nan), rng)
    return samples

def get_pdf_grid_values(spn : Node, w_h : Tuple[int, int], offset : np.array, resolution : int, pdf = None) -> Tuple[np.array, Tuple[int, int], np.array]:
    """
    param spn: SPN
    param w_h: (width, height) of the grid
    param offset: offset of the grid
    param resolution: resolution of the grid
    return: (data, (width, height), offset)
    """
    if pdf is None:
        pdf = lambda x : likelihood(spn, x)
    domain = np.mgrid[-w_h[0]/2.0:w_h[0]/2.0:resolution*1j, -w_h[1]/2.0:w_h[1]/2.0:resolution*1j].reshape(2, -1).T
    data = pdf(domain + offset).reshape(resolution, resolution)
    return data, w_h, offset

def show_data(info : Tuple, ax : plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    data, wxh, offset = info
    plt.imshow(data, origin='lower', extent=[-wxh[0]/2.0, wxh[0]/2.0, -wxh[1]/2.0, wxh[1]/2.0])
    plt.xlabel('Y')
    plt.ylabel('X')
    return ax

def gauss_to_spn_discretize(mean : float, sd : float, scope : int, crit, crit_param : float, sloped = False) -> Node:

    children = []
    weights = []
    for start, mid, end, weight, slope in gauss_discretization_params(mean, sd, crit_param, crit):
        if sloped:
            children += Slopyform(start=start, end=end, scope=scope, slope=slope),
        else:
            children += Uniform(start=start, end=end, scope=scope),
        weights += weight,
    
    s = Sum(children=children, weights=weights)
    assign_ids(s)
    rebuild_scopes_bottom_up(s)

    return s

def split_uniform_norm(start, end, mean, sd) -> Tuple[float, float, float]:
        
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)

        # middle_x = (start_x + end_x) / 2
        # middle = norm.cdf(middle_x, loc=mean, scale=sd)
    

        middle_x = mean + sd * norm.ppf(0.5*(start+end))
        middle = norm.cdf(middle_x, loc=mean, scale=sd)


        # print ("split ", start, end, " to ", start_x, middle_x, end_x)
        left_area = norm.cdf(middle_x, loc=mean, scale=sd) - norm.cdf(start_x, loc=mean, scale=sd)
        right_area = norm.cdf(end_x, loc=mean, scale=sd) - norm.cdf(middle_x, loc=mean, scale=sd)
        left_share = left_area / (left_area + right_area)
        left_share = 0.5
        return (start, middle, end), (left_share, 1-left_share)

def CRIT_uniform_bounded_ratio(start, end, weight, eps, mean, sd, alpha, ):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        height = weight / (end_x-start_x)

        return abs(np.log(height / (norm.pdf(start_x, loc=mean, scale=sd) / (1-alpha)))) < np.log(eps) and abs(np.log(height / (norm.pdf(end_x, loc=mean, scale=sd)/(1-alpha)))) < np.log(eps)

def CRIT_uniform_bounded_deviation(start, end, weight, eps, mean, sd, alpha, ):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        height = weight / (end_x-start_x)

        # print("s, e, h:", start_x, end_x, height, "p(s), p(e):", norm.pdf(start_x, loc=mean, scale=sd), norm.pdf(end_x, loc=mean, scale=sd))
        return abs(height - (norm.pdf(start_x, loc=mean, scale=sd)/(1-alpha))) < eps and abs(height - (norm.pdf(end_x, loc=mean, scale=sd)/(1-alpha))) < eps

def CRIT_even_partition(start, end, weight, eps, mean, sd, ):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        width = end_x - start_x

        return width < eps

def CRIT_slopyform_bounded_deviation(start, end, weight, bound, mean, sd, alpha):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        # just have to check the endpoints
        slope = gaussian_slope_at_point((start_x+end_x)*0.5, mean, sd) / weight

        if abs(slope) > 1/(0.5*(end_x-start_x)**2): # slope too big and we can't guarantee that the function is bounded, must split further
            # print("slope too big, splitting further")
            return False
        
        likelihoods = slopyform_pdf([start_x, end_x], start_x, end_x, slope) * weight
        
        # return abs(np.log(likelihoods[0] / norm.pdf(start_x, loc=mean, scale=sd))) < bound and abs(np.log(likelihoods[1] / norm.pdf(end_x, loc=mean, scale=sd))) < bound
        if abs(likelihoods[0] - (norm.pdf(start_x, loc=mean, scale=sd) /(1-alpha))) < bound and abs(likelihoods[1] - (norm.pdf(end_x, loc=mean, scale=sd)/(1-alpha))) < bound:
            # print(likelihoods.round(3), "close enough")
            # print(start_x.round(3), end_x.round(3), slope.round(3))
            return True
        else:
            
            # print("likelihoods too far off, splitting further")
            return False
        
def CRIT_slopyform_bounded_ratio(start, end, weight, bound, mean, sd, alpha):
    start_x = mean + sd * norm.ppf(start)
    end_x = mean + sd * norm.ppf(end)
    # just have to check the endpoints
    slope = gaussian_slope_at_point(start_x, mean, sd) / weight

    if abs(slope) > 1/(0.5*(end_x-start_x)**2): # slope too big and we can't guarantee that the function is bounded, must split further
        # print("slope too big, splitting further")
        return False
    
    likelihoods = slopyform_pdf([start_x, end_x], start_x, end_x, slope) * weight
    # print(likelihoods)
    # return abs(np.log(likelihoods[0] / norm.pdf(start_x, loc=mean, scale=sd))) < bound and abs(np.log(likelihoods[1] / norm.pdf(end_x, loc=mean, scale=sd))) < bound
    if abs(np.log(likelihoods[0] / (norm.pdf(start_x, loc=mean, scale=sd)/(1-alpha)))) < np.log(bound) and abs(np.log(likelihoods[1] / (norm.pdf(end_x, loc=mean, scale=sd))/(1-alpha))) < np.log(bound):
        return True
    else:
        # print("likelihoods too far off, splitting further")
        return False

def CRIT_nothing(*args):
    return True

def gaussian_slope_at_point(x, mean, sd):
    t1 = np.sqrt(2)
    t2 = np.sqrt(np.pi)
    t5 = sd ** 2
    t8 = x - mean
    t10 = t8 ** 2
    t14 = np.exp(-0.1e1 / t5 * t10 / 2)
    return -t14 * t8 / t5 / sd * t1 / t2 / 2

def gauss_discretization_params(mean : float, sd : float, crit_param : float, accept_split_criterion) -> List[Tuple[float, float, float, float, float]]:
    alpha = 0.5 # what fraction of the tail is left out
    # alpha = 0.1 # what fraction of the tail is left out

    output = []
    
    if accept_split_criterion == CRIT_even_partition:
        points = np.linspace(norm.ppf(alpha/2, loc=mean, scale=sd), norm.ppf(1-alpha/2, loc=mean, scale=sd), num=crit_param+1)
        ab = np.array(list(zip(points[:-1], points[1:])))
        widths = ab[:,1] - ab[:,0]
        mids = ab.mean(axis=1)

        ys = norm.pdf(mids, loc=mean, scale=sd)
        weights = widths * ys
        weights = weights / weights.sum()
        slopes = gaussian_slope_at_point(mids, mean, sd)/weights
        slopes /= (1-alpha)

        return list(zip(ab[:, 0], mids, ab[:,1], weights, slopes))

    start,end = alpha/2, 1-alpha/2
    left_infl, right_infl = norm.cdf(mean - sd, loc=mean, scale=sd) , norm.cdf(mean + sd, loc=mean, scale=sd)
    points = [start, left_infl, 0.5, right_infl, end]
    
    to_divide = []
    for a, b in zip(points, points[1:]):
        start, end = norm.ppf(a, loc=mean, scale=sd), norm.ppf(b, loc=mean, scale=sd)
        to_divide += ((a,b), (norm.cdf(end, loc=mean, scale=sd) - norm.cdf(start, loc=mean, scale=sd))/(1-alpha)),
    

    i = 0
    while len(to_divide) > 0:
        if i > 10000: 
            print("Warning: discretization split more than a 1000, aborting with 1000.")
            break
        i += 1
        (start, end), weight = to_divide.pop(0)

        if accept_split_criterion(start, end, weight, crit_param, mean, sd, alpha):
            left = mean + sd * norm.ppf(start)
            right = mean + sd * norm.ppf(end)
            mid = (left + right) / 2
            output += (mean + sd * norm.ppf(start), mid, mean + sd * norm.ppf(end), weight, (gaussian_slope_at_point(mid, mean, sd)/weight)/(1-alpha)),
        else:
            (left, middle, right), (w1, w2) = split_uniform_norm(start, end, mean, sd)
            to_divide += ((left, middle), weight * w1), ((middle, right), weight * w2)
    
    return output

import torch

def general_discretization_params(pdf, num : int, domain : Tuple[float, float], normalize = True) -> List[Tuple[float,float,float,float,float]]:
    """Returns a list of tuples (left, mid, right, weight, slope) for a given pdf after coarsification."""
    points = torch.linspace(domain[0], domain[1], steps=num+1)
    # width = points[1]- points[0]
    # points += torch.random.uniform(low = -width/2, high = width/2)
    ab = torch.tensor(list(zip(points[:-1], points[1:])))
    widths = ab[:,1] - ab[:,0]
    
    mids = ab.mean(axis=1)
    mids.requires_grad = True

    ys = pdf(mids)
    ys.sum().backward()
    slopes = mids.grad.numpy()

    weights = widths * ys.detach().numpy()
    Z = 1
    if normalize:
        Z = weights.sum()
        weights = weights / Z

    ab = ab.numpy()
    slopes /= (weights * Z).numpy()

    return list(zip(ab[:, 0], mids.detach().numpy(), ab[:,1], weights, slopes))

import clg as clg_lib
import pgm as pgm_lib

def clg_to_spn(clg : clg_lib.Norm, crit_param = 1.5, name_map = None, sloped = False, crit = CRIT_slopyform_bounded_ratio, disc_leaves = False):

    clg.__recompute_params__()

    # name mapping:
    rebuild_scope = (name_map == None)
    if name_map == None:
        name_map = {name : i for i, name in enumerate(clg.get_scope(across_factors = True))}

    root_clusters = clg.cluster_roots_by_dependence()
    
    global_factors = []

    for p in root_clusters:
        roots = list(p)

        if len(roots) == 1 and roots[0].children == []:
            # print("[] making a gaussian leaf from", roots[0].name)
            if disc_leaves:
                global_factors.append(gauss_to_spn_discretize(roots[0].current_mean, roots[0].current_sd, scope=name_map[roots[0].name], crit=crit, crit_param=crit_param, sloped=sloped))
            else:
                global_factors.append(Gaussian(mean=roots[0].current_mean, stdev=roots[0].current_sd, scope = name_map[roots[0].name]))

        else:
                        
            discs = [gauss_discretization_params(n.current_mean, n.current_sd, crit_param, crit) for n in roots]
            # print("[] discretizing", names, "and generating cartesian product of size", np.prod([len(d) for d in discs]))
            
            summands = []
            sum_weights = []

            for vals in product(*discs): # for each possible assignment of the cells of the roots (cartesian product)
                sub_factor = []
                sum_weight = 1
                copy = roots[0].deepcopy()
                
                for i, v in enumerate(vals): # condition each root on the assignment
                    start, mid, end, weight, slope = v
                    # print("[] set", copy.get_roots()[i].name, "to", v[1])

                    if sloped:
                        mid = 0.5*(start+end) - slope * (start-end)**3 / 12 # the mean of the slopyform bucket
                        copy.get_roots()[i].condition(mid, recompute_covariance_and_mean=False) # don't recompute the covariance and mean yet
                        sub_factor += Slopyform(start=start, end=end, slope=slope, scope = name_map[copy.get_roots()[i].name]),
                    else:
                        copy.get_roots()[i].condition(mid, recompute_covariance_and_mean=False) # don't recompute the covariance and mean yet
                        sub_factor += Uniform(start=start, end=end, scope = name_map[copy.get_roots()[i].name]),
                    sum_weight *= weight
            
                copy.get_roots()[0].__recompute_params__() # recompute the covariance matrix and means here

                # now, get the pgm of the children.
                child = copy.get_roots()[0].castrate_roots()
                sub_factor += clg_to_spn(child, crit_param = crit_param, name_map = name_map, sloped = sloped, crit=crit, disc_leaves=disc_leaves), #! recursive call

                summands += Product(children = sub_factor),
                # assign_ids(summands[-1])
                # rebuild_scopes_bottom_up(summands[-1])
                
                sum_weights += sum_weight,
                              
            s = Sum(children = summands, weights = sum_weights)
            # assign_ids(s)
            # rebuild_scopes_bottom_up(s)

            global_factors.append(s)

    if len(global_factors) == 1:
        prod = global_factors[0]
    else:
        prod = Product(global_factors)

    if rebuild_scope:
        assign_ids(prod)
        rebuild_scopes_bottom_up(prod)
    
    return prod

def plot_marginals(spn : Node, pgm : clg_lib.Norm | pgm_lib.Node, xs = None):

    fig, ax = plt.subplots()
    marginalized_spns : List[Node] = []
    nodes = pgm.get_nodes(across_factors = True)
    
    if xs is None:
        xs = np.linspace(-10, 10, 1000)

    nan_fill = np.full_like(xs, np.nan)
    import matplotlib.colors as col
    for i, n in enumerate(nodes):
        marginalized_spns += marginalize(spn, [i]),
        hsv_color = (i/len(nodes), 1, 1)
        color = col.hsv_to_rgb(hsv_color)
        
        if isinstance(pgm, clg_lib.Norm): # if it's a clg, we can plot the exact pdf as well
            ax.plot(xs, stats.norm.pdf(xs, n.current_mean, n.current_sd), label = f"p({n.name}) (exact)", linestyle =  (0, (1, 4)), c=color)
        
        likelihood_input = np.column_stack([nan_fill] * i + [xs.reshape(-1, 1)] + [nan_fill] * (len(nodes)-i-1))
        ax.plot(xs, likelihood(marginalized_spns[-1], likelihood_input), label = f"p({n.name}) (SPN)", c=color)
        
    ax.legend()
    return ax

if True: # parallel SPN

    from multiprocessing.pool import Pool
    def pgm_to_spn_parallel(pgm : clg_lib.Norm, eps =0.1, name_map = None, threads = 1):
        pgm.__recompute_params__()

        # name mapping:
        rebuild_scope = (name_map == None)
        if name_map == None:
            name_map = {name : i for i, name in enumerate(pgm.get_scope(across_factors = True))}

        root_clusters = pgm.cluster_roots_by_dependence()
        # print(root_clusters)
        
        global_factors = []

        for p in root_clusters:
            roots = list(p)

            if len(roots) == 1 and roots[0].children == []:
                # print("[] making a gaussian leaf from", roots[0].name)
                global_factors.append(Gaussian(mean=roots[0].current_mean, stdev=roots[0].current_sd, scope = name_map[roots[0].name]))

            else:
                            
                discs = [gauss_discretization_params(n.current_mean, n.current_sd, eps, CRIT_uniform_bounded_deviation) for n in roots]
                # print("[] discretizing", names, "and generating cartesian product of size", np.prod([len(d) for d in discs]))
                

                if threads == 1:
                    summands = []
                    sum_weights = []
                    for vals in product(*discs): # for each possible assignment of the cells of the roots (cartesian product)
                        node, weight = parallel_take_vals_and_return_spn(roots, name_map, vals, eps)
                        summands += node,
                        sum_weights += weight,
                else:
                    print("parallelizing!")
                    with Pool(threads) as p:
                        summands, sum_weights = zip(*p.starmap(parallel_take_vals_and_return_spn, [(roots, name_map, vals, eps) for vals in product(*discs)], chunksize=100))
                                
                s = Sum(children = summands, weights = sum_weights)
                global_factors.append(s)


        prod = Product(global_factors)

        if rebuild_scope:
            assign_ids(prod)
            rebuild_scopes_bottom_up(prod)
        
        return prod

    def parallel_take_vals_and_return_spn(roots, name_map, vals, eps):
        sub_factor = []
        sum_weight = 1
        copy = roots[0].deepcopy()
        
        for i, v in enumerate(vals): # condition each root on the assignment
            start, mid, end, weight = v
            # print("[] set", copy.get_roots()[i].name, "to", v[1])

            copy.get_roots()[i].condition(mid, recompute_covariance_and_mean=False) # don't recompute the covariance and mean yet


            sub_factor += Uniform(start=start, end=end, scope = name_map[copy.get_roots()[i].name]),
            sum_weight *= weight

        copy.get_roots()[0].__recompute_params__() # recompute the covariance matrix and means here

        # now, get the pgm of the children.
        child = copy.get_roots()[0].castrate_roots()
        sub_factor += pgm_to_spn_parallel(child, eps = eps, name_map = name_map, threads=1),

        return Product(children = sub_factor), sum_weight

if True: # SPN graph plotting
    def __get_label(node):
        type = node.__class__
        if type == Sum:
            return "➕"
        elif type == Product:
            return "✖️"
        elif type == Uniform:
            return f"𝒰({node.start:.2f},{node.end:.2f})"
        elif type == Slopyform:
            return f"𝒮(({node.start:.2f},{node.end:.2f}), {node.slope:.2f})"
        elif type == Gaussian:
            return f"𝒩({node.mean:.1f},{node.stdev:.1f}²)"
        else:
            return node.name[0] + "(🎲)"

    def get_spn_graph(spn, BN_or_RVnames, root : Node = None, G = None):
    
        if isinstance(BN_or_RVnames, (clg_lib.Norm, pgm_lib.Node)):
            name_map = {i:n.name for i, n in enumerate(BN_or_RVnames.get_nodes(across_factors = True))}
        else:
            name_map = {i:n for i, n in enumerate(BN_or_RVnames)}
        
        if root is None:
            G = Digraph(format='svg', graph_attr={'rankdir':'BT'})
            root = spn

            scope = ", ".join([name_map[v] for v in root.scope])
            label = __get_label(root)
            
            G.node(str(id(root)), label = f"{{{label} | {scope}}}", shape="Mrecord")

        
        if isinstance(root, Sum) or isinstance(root, Product):
            
            if isinstance(root, Sum):
                edge_labels = [f"{w:.2f}" for w in root.weights]
            else:
                edge_labels = ["" for _ in root.children]

            for c, l in zip(root.children, edge_labels):
                    
                scope = ", ".join([name_map[v] for v in c.scope])
                label = __get_label(c)
                id_ = str(id(c))
                
                G.node(id_, label = f"{{{label} | {scope}}}", shape="Mrecord")
                G.edge(str(id(c)), str(id(root)), label=l)
                get_spn_graph(spn, BN_or_RVnames, root=c, G=G)
        return G

if True: # Slopyform definition
    from spn.structure.leaves.parametric.Parametric import Leaf

    def slopyform_pdf (x, a, b, s) -> np.ndarray:
            const = (1 - s * (b - a) ** 2 / 2) / (b - a)
            return np.where(np.logical_and(a <= x, x <= b), (x - a) * s + const, 0)

    class Slopyform(Leaf):
        def __init__(self, start, end, slope, scope=None):
            Leaf.__init__(self, scope=scope)
            self.start = start
            self.end = end
            max_abs_of_slope = 1/(0.5*(end-start)**2)
            self.slope = min(max(slope, -max_abs_of_slope), max_abs_of_slope)

    def slopyform_node_likelihood(node, data=None, dtype=np.float64):
        probs = np.ones((data.shape[0], 1), dtype=dtype)

        probs[:] = slopyform_pdf(data[:, node.scope], node.start, node.end, node.slope)
        return probs

    from spn.algorithms.Inference import add_node_likelihood
    add_node_likelihood(Slopyform, slopyform_node_likelihood)


def bn_to_spn(pgm : Node, marginal_target : int = 25, a : float = 1, name_map = None, depth : int = 0, sloped = False):

    # name mapping:
    outer_loop = (name_map == None)
    if name_map == None:
        name_map = {name : i for i, name in enumerate(pgm.get_scope(across_factors = True))}

    # if outer_loop:
    root_clusters = pgm.partition_by_connected_components(across_factors = outer_loop)
        
    if root_clusters == []:
        return None
    
    independent_factors = []
    for p in root_clusters:
        roots = p

        # splits = t(depth-1, k=a, a=1, m=marginal_target)
        splits = marginal_target
        # splits = splits ** (1.0/len(roots))
        # splits = np.floor(splits + int(splits % 1 > np.random.rand())).astype(int)

        evidences = [torch.tensor([p.evidence for p in n.parents]) for n in roots]
        discs = [general_discretization_params(lambda x: n.dist.pdf(evi, x), num=splits, domain=n.dist.get_bounds(evi)) for (evi, n) in zip(evidences, roots)]
        # print("depth", depth, ", discretizing", [r.name for r in roots], f"into {splits} each")#, "and generating cartesian product of size", np.prod([len(d) for d in discs]))
        
        summands = []
        sum_weights = []

        for cell in product(*discs): # for each possible assignment of the cells of the roots (cartesian product)
            sub_factor = []
            sum_weight = 1
            
            # discretize the roots
            for i, dim in enumerate(cell): # condition each root on the assignment

                start, mid, end, weight, slope = dim

                if sloped:
                    mid = 0.5*(start+end) - slope * (start-end)**3 / 12 # the mean of the slopyform bucket
                    roots[i].set_evidence(mid)
                    sub_factor += Slopyform(start=start, end=end, slope=slope, scope = name_map[roots[i].name]),
                else:
                    roots[i].set_evidence(mid)
                    sub_factor += Uniform(start=start, end=end, scope = name_map[roots[i].name]),
                sum_weight *= weight
        
            if weight != 0:
                # now, get make the conditional distribution of their children (if there are children), given their values
                sub_spn = None
                if len(roots[0].children) > 0:
                    sub_spn = bn_to_spn(roots[0], marginal_target=marginal_target, a = a, name_map = name_map, depth= depth+1, sloped = sloped)
                if sub_spn != None:
                    sub_factor += sub_spn,

                if len(sub_factor) > 1:
                    prod = Product(children = sub_factor)
                else:
                    prod = sub_factor[0]
                summands += prod,
                sum_weights += sum_weight,

        for r in roots:
            r.set_evidence(None)
                            
        if len(summands ) == 1:
            s = summands[0]
        else:
            s = Sum(children = summands, weights = sum_weights)
        independent_factors.append(s)

    if len(independent_factors) == 1:
        prod = independent_factors[0]
    else:
        prod = Product(independent_factors)
    
    if outer_loop:
        assign_ids(prod)
        rebuild_scopes_bottom_up(prod)
    
    return prod


def get_error_bounds(clg : clg_lib.Norm, ratio_bounds : dict) -> dict:
        
    marginal_error_bounds = defaultdict(lambda: 1)
    depths = {node.name : 0 for node in clg.get_nodes()}
    ancestors = {node.name : set() for node in clg.get_nodes()}
    for node in clg.get_nodes():
        for c in node.children:
            depths[c.name] = max(depths[c.name], depths[node.name] + 1)
            ancestors[c.name].update(ancestors[node.name])
            ancestors[c.name].add(node.name)
        
        if node.name in ratio_bounds:
            marginal_error_bounds[node.name] = ratio_bounds[node.name]
        else:
            print("error: no log_likelihood bound for'", node.name, "'provided")
    
    print(np.array(list(ratio_bounds.values()))) # ±
    print("full evidence computation is off by a ratio of at most: ", np.prod(np.array(list(ratio_bounds.values()))))
    


    for node in clg.get_nodes():
        for a in ancestors[node.name]:
            marginal_error_bounds[node.name] *= ratio_bounds[a]

    print("depths", depths)
    print("ancestors", ancestors)

    return marginal_error_bounds
# hi there