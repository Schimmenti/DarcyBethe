import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.lib.function_base import delete
import scipy
from scipy import sparse
from scipy.special import lambertw, erfc,kn

import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from scipy.sparse import dok_matrix, csc_matrix
import argparse

debug_mode = False
def build_parents_recursive(source, parents,z, t):
    n = source + 1
    for idx in range(0, z):
        parents[n] = source
        if(t == 0):
            n += 1
        else:
            n = build_parents_recursive(n, parents, z, t-1)
    return n
def build_parents(z, t):
    inlet = 0
    source = 1
    parents = {}
    parents[inlet] = None
    parents[source] = inlet
    
    build_parents_recursive(source, parents, z, t)
    return parents
def overlap(end1,end2,parents,t):
    if(end1 == end2):
        return 1
    p1 = parents[end1]
    p2 = parents[end2]
    count = 1
    while(p1 != p2 and p1 != None and p2 != None):
        count += 1
        p1 = parents[p1]
        p2 = parents[p2]
    return (t+2-count)/(t+2)

def get_champions(site, max_index, is_outlet, tree, energy, champion_dict):
    champion_idx = -1
    champion_energy = np.infty
    if(len(tree[site]) == 0):
        champion_dict[site] = site
        return
    for r in range(site+1, max_index):
        if(is_outlet[r]):
            if(energy[r] < champion_energy):
                champion_idx = r
                champion_energy = energy[r]
    champion_dict[site] = champion_idx
    get_champions(tree[site][0], tree[site][1],is_outlet, tree, energy, champion_dict)
    get_champions(tree[site][1], max_index,is_outlet, tree, energy, champion_dict)

def build_champion_dict(ground_state, tree, energy, outlets):
    n_sites = len(tree.keys())
    outlet_flag = np.zeros(n_sites, dtype=bool)
    outlet_flag[outlets] = True
    champion_dict = {}
    champion_dict[0]= ground_state
    champion_dict[1] = ground_state
    get_champions(tree[1][0],tree[1][1], outlet_flag, tree, energy, champion_dict)
    get_champions(tree[1][1],n_sites, outlet_flag, tree, energy, champion_dict)
    return outlet_flag, champion_dict

def build_tree_recursive(source, tree, parents, weights, energy, outlets,levels, z, t, use_uniform = False):
    n = source + 1
    for idx in range(0, z):
        tree[n] = []
        tree[source].append(n)
        parents[n] = source
        if(use_uniform):
            weights[source][n] = np.random.rand()
        else:
            weights[source][n] = np.random.randn()/np.sqrt(12)
        energy[n] = weights[source][n] + energy[source]
        weights[n] = {}
        levels[t].append(n)
        if(t == 0):
            outlets.append(n)
            n += 1
        else:
            n = build_tree_recursive(n, tree, parents, weights, energy, outlets, levels, z, t-1, use_uniform)
    return n
def build_tree(z, t, use_uniform = False):
    inlet = 0 
    tree = {}
    tree[inlet] = []
    source = 1
    tree[inlet].append(source)
    tree[source] = []
    weights = {}
    weights[inlet] = {}
    if(use_uniform):
        weights[inlet][source] = np.random.rand()
    else:
        weights[inlet][source] = np.random.randn()/np.sqrt(12)
    weights[source] = {}
    energy = np.zeros(z**(t+2))
    energy[inlet] = 0
    energy[source] = weights[inlet][source]
    outlets = []
    parents = {}
    parents[inlet] = None
    parents[source] = inlet
    levels = {}
    for i in range(0,t+3):
        levels[i] = []
    levels[t+2].append(inlet)
    levels[t+1].append(source)
    build_tree_recursive(source, tree, parents, weights, energy, outlets,levels,z, t,use_uniform)
    return tree, parents, weights, energy, np.array(outlets),levels

def kirchhoff(sites_dict,opened_outlet_count,tree,weights, parents, inlet=0):
    opened_sites_count = len(sites_dict.keys())
    matrix_size = max(0,opened_sites_count-opened_outlet_count - 1)
    M = np.zeros((matrix_size,matrix_size))
    vu = np.zeros((matrix_size,2))
    for site, idx in sites_dict.items():
        if(idx != -1):
            #parent of site
            p = parents[site]
            #only the inlet has a non-zero v
            if(p==inlet):
                vu[idx,0] = 1
            #site - parent connection
            u_val = -weights[p][site]
            if(sites_dict[p] != -1):
                M[idx,sites_dict[p]] = -1
            #by default I have a connection with my parent
            M_val = 1
            for s in tree[site]:
                if(s in sites_dict):
                    M_val += 1
                    u_val += weights[site][s]
                    if(sites_dict[s] != -1):
                        M[idx, sites_dict[s]] = -1
            M[idx,idx] = M_val
            vu[idx,1] = u_val
    return np.linalg.solve(M,vu)

def add_path(sites_dict, outlet, opened_outlet_count, parents, inlet):
    if(not inlet in sites_dict):
        sites_dict[inlet] = -1
    
    opened_sites_count = len(sites_dict.keys())
    old_size = max(0,opened_sites_count-opened_outlet_count - 1)

    sites_dict[outlet] = -1 #we do not consider it
    last_site = parents[outlet]
    i = old_size

    while(True):
        if(last_site in sites_dict):
            break
        sites_dict[last_site] = i
        last_site = parents[last_site]
        i += 1
def open_path(bottom, parents, open_flag):
    if(open_flag[bottom]):
        return None
    else:
        open_flag[bottom] = True
        parent = parents[bottom]
        if(parent == None):
            return None
        else:
            if(open_flag[parent]):
                return parent
            else:
                return open_path(parent, parents, open_flag)
            return None
def close_path(bottom,top, parents, open_flag):
    if(bottom==top):
        return
    else:
        if(open_flag[bottom]):
            open_flag[bottom]=False
            close_path(parents[bottom], top, parents, open_flag)
        else:
            return
def compute_conducibility(tree, flag, src = 0):
    if(len(tree[src]) == 0):
        return 1
    l, r = tree[src][0],tree[src][1]
    if(flag[l]):
        left_contribute = compute_conducibility(tree, flag, l)
    else:
        left_contribute = 0
    if(flag[r]):
        right_contribute = compute_conducibility(tree, flag, r)
    else:
        right_contribute = 0
    return 1/(1+1/(left_contribute + right_contribute))
def linear_interpolate(x, x_table, y_table):
    #nearest values
    best_matches = np.argsort(np.abs(x-x_table), axis=1)
    idx_best =  best_matches[:,0]
    y_interp = []
    for i,idx in enumerate(idx_best):
        if(x<x_table[i,idx]):
            if(idx > 0 ):
                m = (y_table[i,idx]-y_table[i,idx-1])/(x_table[i,idx]-x_table[i,idx-1])
                y_interp.append(m*(x-x_table[i,idx-1])+y_table[i,idx-1])
            else:
                print("Shit")
        else:
            if(idx < x_table.shape[1]-1):
                m = (y_table[i,idx+1]-y_table[i,idx])/(x_table[i,idx+1]-x_table[i,idx])
                y_interp.append(m*(x-x_table[i,idx])+y_table[i,idx])
            else:
                print("Shit")
    y_interp = np.array(y_interp)
    y_interp = y_interp[np.isfinite(y_interp)]
    return np.mean(y_interp), np.std(y_interp)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', metavar='t', type=int,
                        help='Number of levels', default=5)
    parser.add_argument('--z', metavar='z', type=int,default=2,
                        help='Number of children in the tree. Not implemented (default 2).')
    parser.add_argument('--output',
                        help='Output file', default='darcy_out')

    parser.add_argument('--channels', type=int,
                        help='Number of channels to open', default=-1)

    parser.add_argument('--seed', metavar='seed', type=int,default=0,
                        help='Initial seed.')

    parser.add_argument('--use_uniform', metavar='use_uniform', type=int,default=0,
                        help='Use finite domain taus.')
    args = parser.parse_args()
    z=args.z
    t=args.t

    np.random.seed(args.seed)


    n_paths = z**(t+1)
    n_sites = z**(t+2)
    print("Building tree...")
    tree, parents, weights, energy, outlets,levels = build_tree(z, t, use_uniform=args.use_uniform > 0)
    print("Tree built.")
    inlet = 0
    ground_state = outlets[np.argmin(energy[outlets])]
    print("Ground state found.")
    _,champion_dict = build_champion_dict(ground_state, tree, energy, outlets)
    sites_dict = {}
    add_path(sites_dict, ground_state, 0, parents, inlet)
    print(sites_dict)
    solution = kirchhoff(sites_dict, 1, tree, weights, parents, inlet)
    a_1, b_1 = solution[sites_dict[1],0], solution[sites_dict[1],1]
    n_channels_opened = 1
    flux = (energy[ground_state]-b_1-a_1*energy[ground_state]) - weights[inlet][1]
    outlets_remaning = list(np.copy(outlets))
    outlets_remaning.remove(ground_state)
    outfile = open(args.output + '_seed=' + str(args.seed) + '.txt','w')
    outfile.write('%i %i %f %f %f %f\n' % (inlet,ground_state,energy[ground_state],flux, 0.0, energy[ground_state]))
    print("Ground state: ", ground_state)
    while(n_channels_opened < n_paths):
        print("Channels:", n_channels_opened)
        if(args.channels != -1 and n_channels_opened == args.channels):
            break
        min_pressure = np.infty
        min_start = -1
        min_stop = -1
        candidate_start = []
        candidate_stop = []
        for s_node in sites_dict.keys():
            if(sites_dict[s_node] == -1):
                continue
            left_champion = champion_dict[tree[s_node][0]]
            right_champion = champion_dict[tree[s_node][1]]
            if(not left_champion in sites_dict):
                candidate_start.append(s_node)
                candidate_stop.append(left_champion)
            if(not right_champion in sites_dict):
                candidate_start.append(s_node)
                candidate_stop.append(right_champion)
        print("Probing %i paths" % len(candidate_start))
        for start, stop in zip(candidate_start, candidate_stop):
            delta_energy = energy[stop]-energy[start]
            a_start, b_start = solution[sites_dict[start],0], solution[sites_dict[start],1]
            candidate_p = (delta_energy-b_start)/a_start
            if(candidate_p < min_pressure):
                min_start = start
                min_stop = stop
                min_pressure = candidate_p
        add_path(sites_dict, min_stop, n_channels_opened, parents, inlet)
        n_channels_opened += 1
        solution = kirchhoff(sites_dict,n_channels_opened , tree, weights, parents, inlet)
        a_1, b_1 = solution[sites_dict[1],0], solution[sites_dict[1],1]
        outlets_remaning.remove(min_stop)
        flux = (1-a_1)*min_pressure - b_1- weights[inlet][1]
        if(debug_mode):
            print(min_pressure)
            print('----------')
        outfile.write('%i %i %f %f %f %f\n' % (min_start,min_stop,min_pressure, flux, energy[min_start], energy[min_stop]))
        outfile.flush()
    outfile.close()