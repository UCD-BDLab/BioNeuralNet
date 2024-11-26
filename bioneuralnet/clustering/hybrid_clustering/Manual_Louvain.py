import copy
import sys, os
base_path = os.path.join('home', 'mohamed', 'Documents')
sys.path.append(os.path.join(base_path,'leidenalg-igraph','build','lib.linux-x86_64-3.8'))
import leidenalg as leiden
import igraph as ig
import networkx as nx
# from OutputGrabber import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from time import time, sleep
from datetime import datetime
import ipyparallel as ipp
import matplotlib.pyplot as plt
from bioneuralnet.clustering.hybrid_clustering.Hybrid_Louvain import SaveStats, GetStats, ReadDataset

# Calculating omics-phyenotype Pearson correlation
def Phen_omics_corr (Nodes):
    
    # Subsetting the Omics data from the PageRank subent
    B_sub = B.iloc[:, Nodes ]

    # Scaling the sebset data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(B_sub)

    # applying PCA to the subset data
    pca = PCA(n_components = 1)
    g1 = pca.fit_transform(scaled)

    g1 = [i for i, in g1]

    g2 = Y.iloc[:,0]

    # obtaining the omics-phenotype correlation
    corr, pvalue =  pearsonr(g1, g2)
    corr = round (corr,2)
    p_value = format(pvalue,'.3g')
    corr_pvalue = "(".join([str(corr),str(p_value)+')'])
    return(corr, corr_pvalue)

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

def generate_weighted_personalization(nodes, max_alpha):
    total_corr = Phen_omics_corr(nodes)[0]
    corr_contribution = [0 for _ in range(len(nodes))]
    for i in range(len(nodes)):
        print('Total: {}, Excluding {}: {}'.format(total_corr, nodes[i], Phen_omics_corr(nodes[:i] + nodes[i+1:])[0]))
        corr_contribution[i] = abs(Phen_omics_corr(nodes[:i] + nodes[i+1:])[0]) - abs(total_corr)
        
    print('Contributions: {}'.format(corr_contribution))
    weighted_personalization = {}
#     print('Max Correlation Contribution: {}'.format(max(corr_contribution, key=abs)))
    for pair in zip(nodes, corr_contribution):
        print('Node {} contribution: {}'.format(pair[0], pair[1]))
        weighted_personalization[str(pair[0])] = max_alpha * (pair[1]/max(corr_contribution, key=abs)) if abs(max(corr_contribution, key=abs)) > 1e-6 else max_alpha
        
    print('Weighted personalization: {}'.format(weighted_personalization))
    return weighted_personalization

def calculate_conductance(G, subgraph):
    S = [str(item) for item in subgraph]
    T = [item for item in G.nodes if item not in S]
    return nx.algorithms.cuts.conductance(G, S, T), nx.conductance(G, nx.Graph(nx.subgraph(G, S)), weight='weight')

def ManualLouvain(G, dataset, target, k3, k4):
    print('Starting partitioning...')
    partition_time = time()

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

    return partitions

if __name__ == '__main__':
    ###### Reading dataset
#     print('Reading dataset...')
#     print(os.path.join(os.path.dirname(sys.path[0]),'leidenalg-igraph','build','lib.linux-x86_64-3.8'))
#     read_time = time()
#     dataset = pd.read_excel('X.xlsx')
#     target = pd.read_excel('Y.xlsx')
#     G = ig.Graph.Read_Ncol('GFEV1ac110.edgelist', directed=False)
#     ig_subgraph = G
#     for node in G.vs:
#         node['name'] = int(node['name'])

#     target_list = target.iloc[:, 1].values.tolist()
#     dataset_list = dataset.iloc[:, 1:].T.values.tolist()

#     scaler = StandardScaler()
#     dataset_list_scaled = scaler.fit_transform(dataset_list).tolist()
#     read_time = time() - read_time
#     print('Dataset loaded.')
#     print('Read time: {}'.format(read_time))
    G, dataset_list, dataset_list_scaled, target_list, name_list = ReadDataset()
    ig_subgraph = G
    k3 = float(sys.argv[1])
    k4 = float(sys.argv[2])
    
    
    ###### Variables for pagerank
    # apply PageRank to the graph
    alpha = 0.04

    # Loading data
    B = pd.read_excel('X.xlsx')
    B.drop(B.columns[0], axis =1, inplace = True)

    Y = pd.read_excel('Y.xlsx')
    Y.drop(Y.columns[0], axis = 1, inplace = True)

    G2 = nx.read_edgelist('GFEV1ac110.edgelist', data=(('weight', float),))
    subgraph = G2
    graph_size = len(list(G2.nodes))
    
    ###### Other variables
    directory = 'ResultsHybrid9_{}'.format(k4)
    PRResults = pd.DataFrame(columns=['Seed Nodes', 'Cluster', 'Number of Nodes', 'Conductance', 'Correlation', 'Minimum Correlation', 'Corr/PVal'])
    cont = True
    run = 1
    keepSingletons = False
    
    
    ###### Running initial louvain
    # all_partitions = ManualLouvain(G, dataset_list_scaled, target_list, 0.2, 0.8)
    # partition = all_partitions[-1]
    # corr, pval, corr_scaled, pval_scaled, true_index, node_names = GetStats(G, partition, dataset_list, dataset_list_scaled, target_list, name_list)
    # candidates = sorted(zip(corr, true_index), key=lambda x: abs(x[0]), reverse=True)

    
    if not os.path.isdir(directory):
        os.mkdir(directory)


    ###### Main loop
    while cont:
        #### Louvain
        all_partitions = ManualLouvain(ig_subgraph, dataset_list_scaled, target_list, k3, k4)
        
        for i, partition in enumerate(all_partitions):
            corr, pval, corr_scaled, pval_scaled, true_index, node_names = GetStats(ig_subgraph, partition, dataset_list, dataset_list_scaled, target_list, name_list)
            # PrintAllStats(i, partition, corr, pval, corr_scaled, pval_scaled, true_index)
            # PrintTopKStats(20, partition, corr, pval, corr_scaled, pval_scaled, true_index)
            timestamp = datetime.now()
            SaveStats('Round {}_{}_{}_Level {}_Max {}_{}.csv'.format(run, k3, k4, i, max(corr, key=abs), timestamp.strftime('%m%d%y-%H%M%S')),
                      partition, corr, pval, corr_scaled, pval_scaled, true_index, node_names, directory)
            
        if not len(all_partitions) > 0:
            break
            
        partition = all_partitions[-1]
        corr, pval, corr_scaled, pval_scaled, true_index, node_names = GetStats(ig_subgraph, partition, dataset_list, dataset_list_scaled, target_list, name_list)
        candidates = sorted(zip(corr, true_index), key=lambda x: abs(x[0]), reverse=True)

        top_candidate = None
        for candidate in candidates:
            if len(candidate[1]) > 1 or keepSingletons:
                top_candidate = candidate[1]
                print('Top cluster: {}\nCorrelation: {}'.format(candidate[1], candidate[0]))
                break
        
        #### PageRank
        if top_candidate is not None:
            p = nx.pagerank(subgraph, personalization=generate_weighted_personalization(top_candidate, alpha), max_iter=100000, tol=1e-06)
            for edge in subgraph.edges(data=True):
                print(edge[2]['weight'])
            print('Nodes: {}'.format(list(subgraph.nodes)))
            print('Seed: {}'.format(top_candidate))
            # p = nx.pagerank(subgraph, personalization={str(node): alpha for node in top_candidate}, max_iter=100000, tol=1e-06)
            nodes, n, cond, corr, min_corr, pval = sweep_cut(p, subgraph)

            entry = {'Seed Nodes': top_candidate,
                     'Cluster': [int(node) for node in nodes],
                     'Number of Nodes': n, 
                     'Conductance': cond, 
                     'Correlation': corr, 
                     'Minimum Correlation': min_corr, 
                     'Corr/PVal': pval}
            PRResults = PRResults.append(entry, ignore_index=True)
            PRResults.to_csv(os.path.join(directory, 'PRResults.csv'))
            


            if len(nodes) < graph_size and len(nodes) > 0:
                graph_size = len(nodes)
                subgraph = subgraph.subgraph(nodes)
                ig_subgraph = ig_subgraph.subgraph(ig_subgraph.vs.select([i for i, v in enumerate(ig_subgraph.vs) if v['name'] in [int(node) for node in nodes]]))
                run += 1
            else:
                cont = False

#     if not os.path.isdir(directory):
#         os.mkdir(directory)
        
#     PRResults.to_csv(os.path.join(directory, 'PRResults.csv'))