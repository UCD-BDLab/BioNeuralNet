import copy
import getopt
import sys, os
from bioneuralnet.clustering.hybrid_clustering.Hybrid_Louvain import ManualLouvain

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'leidenalg-igraph', 'build', 'lib.linux-x86_64-3.8'))
# louvaun aka name
import leidenalg as leiden
import igraph as ig
# from OutputGrabber import *
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from time import time
from datetime import datetime


def ReadDataset():
    print('Reading dataset...')
    read_time = time()
    dataset = pd.read_excel('X.xlsx')
    target = pd.read_excel('Y.xlsx')
    # G = nx.read_edgelist('GFEV1ac110.edgelist', data=(('weight', float),))
    G = ig.Graph.Read_Ncol('GFEV1ac110.edgelist', directed=False)
    for node in G.vs:
        node['name'] = int(node['name'])

    target_list = target.iloc[:, 1].values.tolist()
    dataset_list = dataset.iloc[:, 1:].T.values.tolist()

    scaler = StandardScaler()
    dataset_list_scaled = scaler.fit_transform(dataset.iloc[:, 1:]).T.tolist()
    read_time = time() - read_time
    print('Dataset loaded.')
    print('Read time: {}'.format(read_time))

    return G, dataset_list, dataset_list_scaled, target_list, list(dataset.columns[1:])


def RunAlgorithm(k3, k4):
    print('Starting partitioning...')
    partition_time = time()

    optimiser = leiden.Optimiser()
    partitions = []
    partial_partition = leiden.HybridVertexParition(G, weights='weight',
                                                    dataset=dataset_list_scaled, target=target_list, k3=k3, k4=k4)
    partition_agg = partial_partition.aggregate_partition()
    while optimiser.move_nodes(partition_agg) > 0:
        partial_partition.from_coarse_partition(partition_agg)
        partitions += [copy.copy(partial_partition)]
        partition_agg = partition_agg.aggregate_partition()

    partition_time = time() - partition_time
    print('Partitioned.')
    print('Partition time: {}'.format(partition_time))

    return partitions


def GetStats(G, partition, dataset_list, dataset_list_scaled, target_list, name_list):
    corr = [0 for _ in range(len(partition))]
    pval = [0 for _ in range(len(partition))]
    corr_scaled = [0 for _ in range(len(partition))]
    pval_scaled = [0 for _ in range(len(partition))]
    true_index = [0 for _ in range(len(partition))]
    node_names = [0 for _ in range(len(partition))]

    for i, community in enumerate(partition):
        subset = []
        subset_scaled = []
        true_comm = []
        names = []
        for node_index in community:
            index = int(G.vs[int(node_index)]['name'])
            subset += [dataset_list[index]]
            subset_scaled += [dataset_list_scaled[index]]
            true_comm += [index]
            names += [name_list[index]]

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(pd.DataFrame(subset).T)
        pc1 = [j for j, in pc1]

        pca_scaled = PCA(n_components=1)
        pc1_scaled = pca_scaled.fit_transform(pd.DataFrame(subset).T)
        pc1_scaled = [j for j, in pc1_scaled]

        corr_val, pval_val = pearsonr(pc1, target_list)
        corr[i] = corr_val
        pval[i] = pval_val

        corr_val, pval_val = pearsonr(pc1_scaled, target_list)
        corr_scaled[i] = corr_val
        pval_scaled[i] = pval_val

        true_index[i] = true_comm
        node_names[i] = names

    return corr, pval, corr_scaled, pval_scaled, true_index, node_names


def PrintAllStats(index, partition, corr, pval, corr_scaled, pval_scaled, true_index):
    print(color.BOLD + color.BLUE + '------------------------------------------------------------------' + color.END)
    print(color.BOLD + color.BLUE + 'Level {}'.format(index) + color.END)
    for i in range(len(partition)):
        print('Community {}'.format(i))
        print('Members: {}'.format(partition[i]))
        print('Real Members: {}'.format(true_index[i]))
        print('Correlation: {}\nP Value: {}'.format(corr[i], pval[i]))
        print('Scaled Correlation: {}\nScaled P Value: {}'.format(corr_scaled[i], pval_scaled[i]))
        print('\n\n\n')


def PrintTopKStats(k, partition, corr, pval, corr_scaled, pval_scaled, true_index):
    print(color.BOLD + color.BLUE + '------------------------------------------------------------------' + color.END)
    print(color.BOLD + color.BLUE + 'Top {}'.format(k) + color.END)
    for cluster in sorted(list(zip(partition, list(range(len(partition))), corr, pval, corr_scaled, pval_scaled, true_index)),
                          key=lambda x: abs(x[2]), reverse=True)[:min(k, len(partition))]:
        print('Community {}'.format(cluster[1]))
        print('Members: {}'.format(cluster[0]))
        print('Real Members: {}'.format(cluster[6]))
        print('Correlation: {}\nP Value: {}'.format(cluster[2], cluster[3]))
        print('Scaled Correlation: {}\nScaled P Value: {}'.format(cluster[4], cluster[5]))
        print('\n\n\n')


def SaveStats(filename, partition, corr, pval, corr_scaled, pval_scaled, true_index, node_names, directory='Results'):
    df = pd.DataFrame(columns=['Cluster Number', 'Partition Members', 'Real Members', 'Member Names', 'Correlation',
                               'P Value', 'Scaled Correlation', 'Scaled P Value'])
    for i, cluster in enumerate(
            sorted(list(zip(partition, list(range(len(partition))), corr, pval, corr_scaled, pval_scaled, true_index, node_names)),
                   key=lambda x: abs(x[2]), reverse=True)):
        entry = {'Cluster Number': cluster[1],
                 'Partition Members': cluster[0],
                 'Real Members': cluster[6],
                 'Member Names': cluster[7],
                 'Correlation': cluster[2],
                 'P Value': cluster[3],
                 'Scaled Correlation': cluster[4],
                 'Scaled P Value': cluster[5]}
        df = df.append(entry, ignore_index=True)

    if not os.path.isdir(directory):
        os.mkdir(directory)

    df.to_csv('{}/{}'.format(directory, filename), index=False)


# sweep-cut fucntion for simultaneous PageRank method
def sweep_cut (p, G):
    cond_res = []
    corr_res = []
    cond_corr_res = []

    cluster =  set()
    min_cut , min_cond , len_clus, min_cond_corr  = len(p), 1, 0, 2
    vec = sorted([(p[i]/G.degree(weight='weight')[str(i)]  , i) for i in p.keys()], reverse=True)
    
    for i, (val, node) in enumerate(vec): # val is normalized PR score
        if val == 0: break
        else:
            cluster.add(node)
    
        if len(G.nodes()) > len(cluster):
            cluster_cond = nx.conductance(G, cluster , weight= 'weight')
            cluster_cond = round (cluster_cond, 3)
            cond_res.append(cluster_cond)


            Nodes = [int(k) for k in cluster]
            cluster_corr, corr_pvalue  = Phen_omics_corr(Nodes)
            corr_res.append (round(cluster_corr, 3))
            cluster_corr = (-1) * abs(round(cluster_corr,3))

            # setting K value for composite correlation-coductance function
            k =0.9 
            cond_corr  = round((1-k)*cluster_cond + k*cluster_corr, 3)
            cond_corr_res.append(cond_corr)
            
              
            if (cond_corr < min_cond_corr):
                min_cond_corr , min_cut = cond_corr , i
                len_clus = len(cluster)
                cond = cluster_cond
                corr = - cluster_corr
                cor_pval = corr_pvalue 
                
    
    return ([vec[i][1] for i in range(min_cut+1)], len_clus, cond, corr, round (min_cond_corr,3), cor_pval)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Hybrid functions

# Calculating omics-phyenotype Pearson correlation
# Nodes is a list of nodes indeces a vector[]
def Phen_omics_corr (Nodes):
    
    # Subsetting the Omics data from the PageRank subent
    # B is the omics data
    B_sub = B.iloc[:, Nodes ]

    # Scaling the sebset data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(B_sub)

    # applying PCA to the subset data
    pca = PCA(n_components = 1)
    g1 = pca.fit_transform(scaled)

    g1 = [i for i, in g1]

    # Subsetting the phenotype data
    g2 = Y.iloc[:,0]

    # obtaining the omics-phenotype correlation
    corr, pvalue =  pearsonr(g1, g2)
    corr = round (corr,2)
    p_value = format(pvalue,'.3g')
    corr_pvalue = "(".join([str(corr),str(p_value)+')'])
    return(corr, corr_pvalue)

def generate_weighted_personalization(nodes, max_alpha):
    total_corr = Phen_omics_corr(nodes)[0]
    corr_contribution = [0 for _ in range(len(nodes))]
    for i in range(len(nodes)):
#         print('Total: {}, Excluding {}: {}'.format(total_corr, nodes[i], Phen_omics_corr(nodes[:i] + nodes[i+1:])[0]))
        corr_contribution[i] = abs(Phen_omics_corr(nodes[:i] + nodes[i+1:])[0]) - abs(total_corr)
        
#     print('Contributions: {}'.format(corr_contribution))
    weighted_personalization = {}
#     print('Max Correlation Contribution: {}'.format(max(corr_contribution, key=abs)))
    for pair in zip(nodes, corr_contribution):
#         print('Node {} contribution: {}'.format(pair[0], pair[1]))
        weighted_personalization[str(pair[0])] = max_alpha * pair[1]/max(corr_contribution, key=abs)
        
#     print('Weighted personalization: {}'.format(weighted_personalization))
    return weighted_personalization
def calculate_conductance(G, subgraph):
    S = [str(item) for item in subgraph]
    T = [item for item in G.nodes if item not in S]
    return nx.algorithms.cuts.conductance(G, S, T), nx.conductance(G, nx.Graph(nx.subgraph(G, S)), weight='weight')



def ManualLouvain(G, dataset, target, k3, k4):
    print('Starting partitioning...')
    partition_time = time()

    #Hybredvertex
    optimiser = leiden.Optimiser()
    partitions = []
    partial_partition = leiden.HybridVertexParition(G, weights='weight',
                                                    dataset=dataset, target=target, k3=k3, k4=k4)
    partition_agg = partial_partition.aggregate_partition()
    while optimiser.move_nodes(partition_agg) > 0:
        partial_partition.from_coarse_partition(partition_agg)
        partitions += [copy.copy(partial_partition)]
        partition_agg = partition_agg.aggregate_partition()

    partition_time = time() - partition_time
    print('Partitioned.')
    print('Partition time: {}'.format(partition_time))


if __name__ == "__main__":

    # global variables.
    # Reading dataset
    print('Reading dataset...')
    print(os.path.join(os.path.dirname(sys.path[0]),'leidenalg-igraph','build','lib.linux-x86_64-3.8'))
    read_time = time()
    dataset = pd.read_excel('X.xlsx')
    target = pd.read_excel('Y.xlsx')
    G = ig.Graph.Read_Ncol('GFEV1ac110.edgelist', directed=False)
    for node in G.vs:
        node['name'] = int(node['name'])

    target_list = target.iloc[:, 1].values.tolist()
    dataset_list = dataset.iloc[:, 1:].T.values.tolist()

    scaler = StandardScaler()
    dataset_list_scaled = scaler.fit_transform(dataset_list).tolist()
    read_time = time() - read_time
    print('Dataset loaded.')
    print('Read time: {}'.format(read_time))

    # Variables for pagerank
    # apply PageRank to the graph
    alpha = 0.9
    seeds = 94

    # Loading data
    B = pd.read_excel('X.xlsx')
    B.drop(B.columns[0], axis =1, inplace = True)

    Y = pd.read_excel('Y.xlsx')
    Y.drop(Y.columns[0], axis = 1, inplace = True)

    G2 = nx.read_edgelist('GFEV1ac110.edgelist', data=(('weight', float),))



    # g is the graph, dataset_list is the list of omics data, dataset_list_scaled is the scaled omics data, target_list is the phenotype data, name_list is the list of node names
    all_partitions = ManualLouvain(G, dataset_list_scaled, target_list, 0.2, 0.8)
    partition = all_partitions[-1]
    corr, pval, corr_scaled, pval_scaled, true_index, node_names = GetStats(G, partition, dataset_list, dataset_list_scaled, target_list, name_list)
    candidates = sorted(zip(corr, true_index), key=lambda x: abs(x[0]), reverse=True)

    for candidate in candidates:
        if len(candidate[1]) > 1:
            top_candidate = candidate[1]
            print('Top cluster: {}\nCorrelation: {}'.format(candidate[1], candidate[0]))
            break

    p = nx.pagerank(G2, personalization=generate_weighted_personalization(candidates, alpha), max_iter=100000, tol=1e-06)
    nodes, n, cond, corr, min_corr, pval = sweep_cut(p, G2)

    # Never generated an actual subgraph. 
    # Cluster is a list of node indexes
    # Sweep return a list of indices.

####################################################################
    k3 = float(sys.argv[1])
    k4 = float(sys.argv[2])

    G, dataset_list, dataset_list_scaled, target_list, name_list = ReadDataset()
    # This is a list of clusters.
    partitions = RunAlgorithm(k3, k4)

    for i, partition in enumerate(partitions):
        corr, pval, corr_scaled, pval_scaled, true_index, node_names = GetStats(G, partition, dataset_list, dataset_list_scaled, target_list, name_list)
        PrintAllStats(i, partition, corr, pval, corr_scaled, pval_scaled, true_index)
        PrintTopKStats(20, partition, corr, pval, corr_scaled, pval_scaled, true_index)
        timestamp = datetime.now()
        SaveStats('{}_{}_Level {}_Max {}_{}.csv'.format(k3, k4, i, max(corr, key=abs), timestamp.strftime('%m%d%y-%H%M%S')),
                  partition, corr, pval, corr_scaled, pval_scaled, true_index, node_names)


## Subset of nodes

    test_nodes = [11, 206, 303, 406, 1421, 1805]
    edge_threshold = 0.09

    # p = nx.pagerank(G, personalization=generate_weighted_personalization(test_nodes, alpha), max_iter=100000, tol=1e-06)
    p = nx.pagerank(G, personalization=generate_weighted_personalization(test_nodes, alpha), max_iter=100000, tol=1e-06)
    nodes, n, cond, corr, min_corr, pval = sweep_cut(p, G)

    print('Seed node {}:'.format(test_nodes))
    print('Number of nodes: {}'.format(n))
    print('Cond: {}'.format(cond))
    print('Correlation: {}'.format(corr))
    print('Minimum Correlation: {}'.format(min_corr))
    print('Corr/PVal: {}'.format(pval))
    print('\n\n\n')

    subgraph = nx.Graph(G.subgraph(nodes))
    edge_weights = nx.get_edge_attributes(subgraph, 'weight')
    subgraph.remove_edges_from((e for e, w in edge_weights.items() if w < 0.09))
    subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
    corr, pval = Phen_omics_corr([int(node) for node in list(subgraph.nodes)])

    print('After pruning:')
    print('Number of nodes: {}'.format(len(list(subgraph.nodes))))
    print('Correlation: {}'.format(corr))


    ## READ METHODS from 

    #hybrid approach pagerank -> louvain -> pagerank -> 