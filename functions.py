import snap
import numpy as np
from copy import deepcopy
from scipy.stats import bernoulli
from tqdm import tqdm
import statistics as stat
from heapq import nlargest
import random

def copy_graph(graph):
    """Creates a deep copy of a graph

    Args:
        graph: which graph to copy
        
    Returns:
        new_graph: a new graph object which is the same as the input graph
    """
    
    tmpfile = '.copy.bin'

    # Saving to tmp file
    FOut = snap.TFOut(tmpfile)
    graph.Save(FOut)
    FOut.Flush()

    # Loading to new graph
    FIn = snap.TFIn(tmpfile)
    graphtype = type(graph)
    new_graph = graphtype.New()
    new_graph = new_graph.Load(FIn)

    return new_graph


def gen_sample(G,p, noise_model='main'):
    """Creates a noisy sample graph of an underlying graph

    Args:
        G: the underlying graph
        p: probability that each vertex is hidden (this is needed only if noise_model='main')
        noise_model: which model to use when generating the sample. It can have the values 'main' for 
                        our main noise model, 'different_p_i' for the model where vertices hide different
                        portions of their incident edges and 'random_walk' for the random walk sampling method
        
    Returns:
        G_sample: the noisy sample
    """
        
    if noise_model == 'main':
        
        #-------------------------------------------main sample generation model------------------------------------
        G_sample = copy_graph(G)
        coins = bernoulli.rvs(p, size=G.GetNodes())
        r = snap.TIntIntH()
        i = 0
        for NI in G.Nodes():
            r[NI.GetId()] = coins[i]       #each node performs an independent bernoulli trial
            i += 1
        for e in G.Edges():
            if (r[e.GetSrcNId()] + r[e.GetDstNId()] == 2):
                G_sample.DelEdge(e.GetSrcNId(),e.GetDstNId())
                
                
    elif noise_model == 'different_p_i':
        #-------------------alternative sample generation model: different p_is-------------------------------------
        G_sample = copy_graph(G)
        probs =  np.random.uniform(low=0, high=0.7, size=G.GetNodes())

        probsHash = snap.TIntFltH()
        r = snap.TIntPrIntH()
        #initialize the edge hash
        for e in G.Edges():
            r[snap.TIntPr(e.GetSrcNId(),e.GetDstNId())] = 0
            r[snap.TIntPr(e.GetDstNId(),e.GetSrcNId())] = 0

        i = 0
        for NI in G.Nodes():
            probsHash[NI.GetId()] = probs[i]       #each node has its own p_i
            i += 1

        random_num = snap.TFlt
        for NI in G.Nodes():
            for k in NI.GetOutEdges():
                if random_num.GetRnd()<probsHash[NI.GetId()]: 
                    random_coin = 1
                else:
                    random_coin = 0
                r[snap.TIntPr(NI.GetId(),k)] += random_coin
                r[snap.TIntPr(k,NI.GetId())] += random_coin

        for e in G.Edges():
            edge_pair = snap.TIntPr(e.GetSrcNId(),e.GetDstNId())
            if (r[edge_pair] >= 1):
                G_sample.DelEdge(e.GetSrcNId(),e.GetDstNId())
            
            
    elif noise_model == 'random_walk': 
        #-------------------alternative sample generation model: random walk -------------------------------------
        Rnd = snap.TRnd(42)
        Rnd.Randomize()

        #initialize the sample graph as an empty graph
        G_sample = snap.TUNGraph.New()
        for NI in G.Nodes():
            G_sample.AddNode(NI.GetId())

        my_float = snap.TFlt
        my_int = snap.TInt
        #begin with a node iterator randomly initialized
        current = snap.TUNGraphNodeI(G.GetRndNI(Rnd))
        while True:        
            random_num = my_float.GetRnd()        #it is faster to use snap's random num generator
            if random_num < 0.15:
                #restart the walk with prob=0.15
                current = snap.TUNGraphNodeI(G.GetRndNI(Rnd))
            else:
                #select next vertex from the neighborhood
                old_visiting_node_id = current.GetId()
                degree = current.GetDeg()
                random_number = my_int.GetRnd(degree)
                random_neighbor_id = current.GetNbrNId(random_number)
                current = G.GetNI(random_neighbor_id)
                if not G_sample.IsEdge(old_visiting_node_id,current.GetId()) and not G_sample.IsEdge(current.GetId(),old_visiting_node_id):
                    G_sample.AddEdge(old_visiting_node_id,current.GetId())
            if G_sample.GetEdges() >= G.GetEdges() * 0.35: break
        
    return G_sample


def calculate_prob_distribution(G):
 
     """calculates a distribution over the vertices which will be later used to sample uniform wedges

    Args:
        G: input graph
        
    Returns:
        nodes_list: a list with all the node ids
        probabilities: the probability distribution over the nodes
        normalizing: the factor used to normalize the distribution
    """
    
    nodes_list = []
    probabilities = []
    normalizing = 0
    for NI in G.Nodes():
        nodes_list.append(NI.GetId())
        probb = 1.0*NI.GetDeg()*(NI.GetDeg()-1)/2
        probabilities.append(probb)
        normalizing += probb
    probabilities = [x/normalizing for x in probabilities]
    return nodes_list, probabilities, normalizing

def queries_only_estimator(G,num_of_queries,nodes_list, probabilities, normalizing):
 
     """implements the wedge sampling estimator

    Args:
        G: input graph
        num_of_queries: number of wedges to be sampled
        nodes_list: a list with all the node ids
        probabilities: the probability distribution over the nodes
        normalizing: the factor used to normalize the distribution: 
        
    Returns:
        estimator for the number of triangles
    """    
    
    count = 0
    my_int = snap.TInt  
    
    selected_vertices = np.random.choice(nodes_list, size=num_of_queries, replace=True, p=probabilities)
    for nid in selected_vertices:
        current = G.GetNI(nid)
        #select two nodes that are adjascent to current
        degree = current.GetDeg()
        random_number = my_int.GetRnd(degree)
        random_neighbor_id1 = current.GetNbrNId(random_number)
        while True:
            random_number = my_int.GetRnd(degree)
            random_neighbor_id2 = current.GetNbrNId(random_number)
            if random_neighbor_id2 != random_neighbor_id1:
                break
        if G.IsEdge(random_neighbor_id1, random_neighbor_id2) or G.IsEdge(random_neighbor_id2, random_neighbor_id1):
            count += 1
            

    return (1.0*count/num_of_queries)*(normalizing/3)
    
    
def naive_estimator(g,p,noise_model):
    if noise_model == 'main':
     """this implements the naive estimator that counts the triangles in the sample and performs a rescaling

    Args:
        g: sample graph
        p: the parameter of our main noise model (for the other noise model the value of p is not used)
        noise_model: can take the values 'main' for our main noise model or 'different_p_i' for the noise model
                        where each vertex hides a different fraction of its incident edges
        
    Returns:
        estimation: the naive estimation of the number of triangles
    """
    if noise_model == 'main':
        scaling = 1-3*p**2+2*p**3
    elif noise_model == 'different_p_i':
        scaling = 0.0994673703703704

    estimation = g.GetEdges()/scaling
    return estimation


def calc_error(measurements, actual_val): 
    """Calculates the percent error of a sequence of estimations

    Args:
        measurements: a list containing all the estimations
        actual_val: the real value
        
    Returns:
        percent_error: absolute relative error multiplied by 100
    """

    lst = []
    for x in measurements:
        lst.append(100*np.abs(x - actual_val)/actual_val)
    percent_error = stat.mean(lst)
    return percent_error


def calc_errors_stdev(measurements, actual_val):  
    """Calculates the standard deviation of the relative error for a sequence of estimations

    Args:
        measurements: a list containing all the estimations
        actual_val: the real value
        
    Returns:
        standdev: standard deviation of the error
    """

    lst = []
    for x in measurements:
        lst.append(100*np.abs(x - actual_val)/actual_val)
    standdev = np.sqrt(stat.variance(lst)/len(lst))          #divide by their number because we need the std of the mean
    return standdev


def select_queries(Gs,k):
    """Creates the query set consisting of the k highest degree vertices

    Args:
        Gs: a sample graph
        k: the size of the query set
        
    Returns:
        greatest_degree_ids: a list containing the ids of the k vertices having the greatest degrees in Gs
    """

    result_degree = snap.TIntV()
    snap.GetDegSeqV(Gs, result_degree)
    lst = []
    nodeids = []
    for v in Gs.Nodes():
        nodeids.append(v.GetId())
    for i in range(0, result_degree.Len()):
        lst.append((nodeids[i],result_degree[i]))
        
    greatest_degree_lst = nlargest(k, lst, key=lambda e:e[1])
    greatest_degree_ids = [el[0] for el in greatest_degree_lst]
    
    return greatest_degree_ids


def select_random_queries(G,k):
    """Creates a random query set of size k

    Args:
        G: the graph (in order to know the nodes' ids in case they are not consecutive integers)
        k: size of the query set
        
    Returns:
        random_query_set: a list of size k with ids of randomly selected nodes
    """
    
    lst = []
    for v in G.Nodes():
        lst.append(v.GetId())
    random.shuffle(lst)
    random_query_set = lst[:k]
    return random_query_set



def keep_final_queries(V_q,G,k,filtering):
    """Filters the query set by keeping only a fraction of its vertices having the highest degrees in the underlying graph

    Args:
        V_q: the query set
        G: the underlying graph
        k: size of the query set
        filtering: what fraction of the query set to eventually keep
        
    Returns:
        greatest_degree_ids: the reduced query set after the filtering
    """
    
    knew = int(round(k*filtering))
    lst = []
    
    for NI in G.Nodes():
        if NI.GetId() in V_q:
            lst.append((NI.GetId(), NI.GetOutDeg()))
    greatest_degree_lst = nlargest(knew, lst, key=lambda e:e[1])
    greatest_degree_ids = [el[0] for el in greatest_degree_lst]
    return greatest_degree_ids


def estimator_edges(G,Gs,k,p, use_one_sample=False,filtering=0.5, random_queries=False):
    """Implements the estimator for the number of edges (assuming our main model for the sample generation)

    Args:
        G: the underlying graph
        Gs: the sample graph
        k: size of the query set
        use_one_sample: if True uses the same sample to both determine the query set and perform the estimation
        filtering: the factor of the query set that will be kept after filtering out the low degree vertices
        random_queries: if True performs random queries
        
    Returns:
        count: estimator for the number of edges
    """
    
    if random_queries:
        V_q = select_random_queries(G,k)
    else:
        if use_one_sample:
            myset = select_queries(Gs,k)
            V_q = keep_final_queries(myset,G,k,filtering)
        else:
            Gss = gen_sample(G,p,noise_model='main')
            V_q = select_queries(Gss,k)
       
    count = 0
    for e in G.Edges():
        u = e.GetSrcNId()
        v = e.GetDstNId()
            
        if u in V_q or v in V_q:
            count += np.float(1)
            
        elif Gs.IsEdge(u, v) or Gs.IsEdge(v, u):
            count += np.float(1.0)/(1-p**2)
    
    
    return np.float(count)


def estimator_triangles(G,Gs,k,noise_model,p, use_one_sample=False,filtering=0.5, random_queries=False, learn_weights=False):
    """Implements the estimator for the number of triangles

    Args:
        G: the underlying graph
        Gs: the sample graph
        k: size of the sample graph
        noise_model: which noise model to use. 'main' for our main noise model, 'different_p_i' for the model 
            where each vertex hides different fraction of its incident edges and 'random_walk' for the random walk sample
        filtering: the factor of the query set that will be kept after filtering out the low degree vertices
        random_queries: if True performs random queries
        learn_weights: if True uses the empirical fractions of survived edges and triangles as scaling factors instead of the theoretically correct
        
        
    Returns:
        estim: estimation of the number of triangles
    """
        
    if random_queries:
        V_q = select_random_queries(G,k)
    else:
        if use_one_sample:
            myset = select_queries(Gs,k)
            V_q = keep_final_queries(myset,G,k,filtering)
        else:
            Gss = gen_sample(G,p,noise_model)
            V_q = select_queries(Gss,k)

    
    #first count the triangles that have no nodes in V_q
    g = copy_graph(Gs)
    V_q_snap = snap.TIntV()
    for x in V_q:
        V_q_snap.Add(x)
    snap.DelNodes(g, V_q_snap)
    t0 = snap.GetTriads(g)
    
    #count the triangles with exactly 3 nodes in V_q
    g = copy_graph(G) 
    V_q_complement_snap = snap.TIntV()
    for NI in G.Nodes():
        x = NI.GetId()
        if x not in V_q:
            V_q_complement_snap.Add(x)
    snap.DelNodes(g, V_q_complement_snap)
    t3 = snap.GetTriads(g)    
    
    
    #---------------------determine the weights that will scale the counts--------------------------------------
    if learn_weights or noise_model=='random_walk':
        g_known = copy_graph(g)
        gs_known = copy_graph(Gs)  
        snap.DelNodes(gs_known, V_q_complement_snap)

        for NI in G.Nodes():
            if NI.GetId() in V_q:
                for k in NI.GetOutEdges():
                    if k not in V_q:
                        if not g_known.IsNode(k): g_known.AddNode(k)    #to avoid adding same node multiple times
                        g_known.AddEdge(k,NI.GetId())
                        if Gs.IsEdge(k,NI.GetId()) or Gs.IsEdge(NI.GetId(),k):
                            if not gs_known.IsNode(k): gs_known.AddNode(k)
                            gs_known.AddEdge(k,NI.GetId())                    

        trig_actual = 1.0* snap.GetTriads(g_known)
        edges_actual = 1.0*g_known.GetEdges()
        trig_realized = 1.0* snap.GetTriads(gs_known)
        edges_realized = 1.0*gs_known.GetEdges()

        fraction_for_edges = edges_realized/edges_actual
        fraction_for_triangles = trig_realized/trig_actual
    else:
        if noise_model == 'main':
            fraction_for_edges = 1-p**2
            fraction_for_triangles = 1-3*p**2+2*p**3
        elif noise_model == 'different_p_i':
            #these have been calculated before-hand
            fraction_for_edges = 0.42250000000000004 
            fraction_for_triangles = 0.0994673703703704
    #-----------------------------------------------------------------------------------------------------------
    
    
    #in order to count the edges with 2 nodes in V_q:
    #first add to g all nodes from V_q_complement_snap (no edges)
    for v in V_q_complement_snap:
        g.AddNode(v)
    #now we need to count the triangles which have exactly one 2 nodes inside V_q
    #to do so, first count the triangles in the graph that have all edges adjacent to V_q and subtract the number
    #of trianlges having all edges in V_q
    for EI in G.Edges():
        u = EI.GetSrcNId()
        v = EI.GetDstNId()
        if (u in V_q and v not in V_q):
            g.AddEdge(v,u)
        if (v in V_q and u not in V_q):
            g.AddEdge(v,u)
    # the subtraction    
    t2 = snap.GetTriads(g) - t3             
        
    #before counting the triangles that have exactly one node in V_q, 
    #we take the induced graph of G from the set of nodes having distance <= 1 from V_q
    g = copy_graph(G)
    
    NIdV = snap.TIntV()
    for x in V_q:
        NIdV.Add(x)
    for NI in g.Nodes():
        for k in NI.GetOutEdges():

            if k in V_q:
                nodeid = NI.GetId()
                if nodeid not in V_q:
                    NIdV.Add(NI.GetId())
                    break
    SubGraph = snap.GetSubGraph(g, NIdV)
    snap.DelNodes(SubGraph, V_q_snap)
    
    t1=0
    for EI in SubGraph.Edges():
        u = EI.GetSrcNId()
        v = EI.GetDstNId()
        if Gs.IsEdge(u,v) and u not in V_q and v not in V_q:
            for w in V_q:
                if (G.IsEdge(w,u) or G.IsEdge(u,w)) and (G.IsEdge(w,v) or G.IsEdge(v,w)):
                    t1 += 1
    estim = t2+ t3 + t1/fraction_for_edges + t0/fraction_for_triangles
    return estim
