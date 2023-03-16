import numpy as np
from scipy.stats import norm
from typing import Tuple, List
import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState

from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Node, Sum, Product, assign_ids, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Gaussian, Uniform
from spn.algorithms.Inference import likelihood

def sample_from_spn(spn : Node, amount : int) -> np.ndarray:
    rng = RandomState(None)
    samples = sample_instances(spn, np.full((amount, len(spn.scope)), np.nan), rng)
    return samples

def get_pdf_grid_values(spn : Node, w_h : Tuple[int, int], offset : np.array, resolution : int) -> Tuple[np.array, Tuple[int, int], np.array]:
    """
    param spn: SPN
    param w_h: (width, height) of the grid
    param offset: offset of the grid
    param resolution: resolution of the grid
    return: (data, (width, height), offset)
    """
    domain = np.mgrid[-w_h[0]/2.0:w_h[0]/2.0:resolution*1j, -w_h[1]/2.0:w_h[1]/2.0:resolution*1j].reshape(2, -1).T
    data = likelihood(spn, domain + offset).reshape(resolution, resolution)
    return data, w_h, offset

def show_data(info : Tuple, ax : plt.Axes = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    data, wxh, offset = info
    plt.imshow(data, origin='lower', extent=[-wxh[0]/2.0, wxh[0]/2.0, -wxh[1]/2.0, wxh[1]/2.0])
    plt.xlabel('Y')
    plt.ylabel('X')
    return ax

def gauss_to_spn_discretize(mean : float, sd : float, eps : float, scope : int, accept_split_criterion) -> Node:

    children = []
    weights = []
    for start, end, weight in gauss_discretization_params(mean, sd, eps, accept_split_criterion):        
        children += Uniform(start=start, end=end, scope=scope),
        weights += weight,
    
    s = Sum(children=children, weights=weights)
    assign_ids(s)
    rebuild_scopes_bottom_up(s)

    return s

def split_uniform(start, end, mean, sd) -> Tuple[float, float, float]:
        
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        middle_x = (start_x + end_x) / 2
        middle = norm.cdf(middle_x, loc=mean, scale=sd)
        # print ("split ", start, end, " to ", start_x, middle_x, end_x)
        left_area = norm.cdf(middle_x, loc=mean, scale=sd) - norm.cdf(start_x, loc=mean, scale=sd)
        right_area = norm.cdf(end_x, loc=mean, scale=sd) - norm.cdf(middle_x, loc=mean, scale=sd)
        left_share = left_area / (left_area + right_area)
        return (start, middle, end), (left_share, 1-left_share)

def split_until_bounded_log_likelihood(start, end, weight, eps, mean, sd):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        height = weight / (end_x-start_x)

        return abs(np.log(height / norm.pdf(start_x, loc=mean, scale=sd))) < eps and abs(np.log(height / norm.pdf(end_x, loc=mean, scale=sd))) < eps

def split_until_bounded_likelihood(start, end, weight, eps, mean, sd):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        height = weight / (end_x-start_x)

        # print("s, e, h:", start_x, end_x, height, "p(s), p(e):", norm.pdf(start_x, loc=mean, scale=sd), norm.pdf(end_x, loc=mean, scale=sd))
        return abs(height - norm.pdf(start_x, loc=mean, scale=sd)) < eps and abs(height - norm.pdf(end_x, loc=mean, scale=sd)) < eps

def split_until_at_most_eps_wide(start, end, weight, eps, mean, sd):
        start_x = mean + sd * norm.ppf(start)
        end_x = mean + sd * norm.ppf(end)
        height = weight / (end_x-start_x)

        width = end_x - start_x

        return width < eps

def gauss_discretization_params(mean : float, sd : float, eps : float, accept_split_criterion) -> List[Tuple[float, float, float]]:
    alpha = 0.001 # what fraction of the tail is left out
    output = []
            
    to_divide = [((0+alpha/2, 0.5), 0.5), ((0.5, 1-alpha/2), 0.5)]
    i = 0
    while len(to_divide) > 0:
        if i > 1000: 
            print("Warning: discretization split more than a 1000, aborting with 1000.")
            break
        i += 1
        (start, end), weight = to_divide.pop()

        if accept_split_criterion(start, end, weight, eps, mean, sd):
            output += (mean + sd * norm.ppf(start), mean + sd * norm.ppf(end), weight),
        else:
            (left, middle, right), (w1, w2) = split_uniform(start, end, mean, sd)
            to_divide += ((left, middle), weight * w1), ((middle, right), weight * w2)
    
    return output


import clg as clg_lib

def clg_to_spn(clg : clg_lib.CLG, eps : float, accept_split_criterion = split_until_bounded_likelihood, name_map = None) -> Node:
    assert len(clg.roots) == 1, "CLG must have exactly one root"
    
    is_leaf = (clg.roots[0].children == [])

    # name mapping:
    if name_map == None:
        name_map = {name : i for i, name in enumerate(clg.get_scope())}
    
    if is_leaf:
        return Gaussian(mean=clg.roots[0].current_mean, stdev=clg.roots[0].current_sd, scope=name_map[clg.roots[0].name])

    else:
        discretized_root = gauss_discretization_params(clg.roots[0].current_mean, clg.roots[0].current_sd, eps, accept_split_criterion)
        children = []
        weights = []
        for start, end, weight in discretized_root:
            root_copy = clg.roots[0].copy_subtree()
            root_copy.condition(0.5*(start+end))
            
            unif = Uniform(start=start, end=end, scope=name_map[clg.roots[0].name])

            for child in root_copy.children:   
                unif *= clg_to_spn(clg_lib.CLG([child]), eps, accept_split_criterion, name_map = name_map)
                
            children += unif,
            weights += weight,
        
        s = Sum(children=children, weights=weights)
        assign_ids(s)
        rebuild_scopes_bottom_up(s)
        return s


def clg_to_spn2(clg : clg_lib.CLG, eps : float, accept_split_criterion = split_until_bounded_likelihood, name_map = None) -> Node:
    assert len(clg.roots) == 1, "CLG must have exactly one root"
    
    is_leaf = (clg.roots[0].children == [])

    # name mapping:
    if name_map == None:
        name_map = {name : i for i, name in enumerate(clg.get_scope())}
    
    if is_leaf:
        return Gaussian(mean=clg.roots[0].current_mean, stdev=clg.roots[0].current_sd, scope=name_map[clg.roots[0].name])

    else:
        discretized_root = gauss_discretization_params(clg.roots[0].current_mean, clg.roots[0].current_sd, eps, accept_split_criterion)
        children = []
        weights = []
        for start, end, weight in discretized_root:
            root_copy = clg.roots[0].copy_subtree()
            root_copy.condition(0.5*(start+end))
            
            unif = Uniform(start=start, end=end, scope=name_map[clg.roots[0].name])

            for child in root_copy.children:   
                unif *= clg_to_spn(clg_lib.CLG([child]), eps, accept_split_criterion, name_map = name_map)
                
            children += unif,
            weights += weight,
        
        s = Sum(children=children, weights=weights)
        assign_ids(s)
        rebuild_scopes_bottom_up(s)
        return s


