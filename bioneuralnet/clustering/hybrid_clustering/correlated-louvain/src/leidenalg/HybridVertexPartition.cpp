//
// Created by Mohamed Abdel-Hafiz on 10/7/21.
//

#include "HybridVertexPartition.h"
#include <cmath>

#ifdef DEBUG
#include <iostream>
using std::cerr;
using std::endl;
#endif

HybridVertexPartition::HybridVertexPartition(Graph* graph,
                                                     vector<size_t> const& membership) :
        MutableVertexPartition(graph,
                               membership)
{ }

HybridVertexPartition::HybridVertexPartition(Graph* graph) :
        MutableVertexPartition(graph)
{ }

HybridVertexPartition::HybridVertexPartition(Graph* graph, std::vector< std::vector<double> >  input_dataset, std::vector<double> target,
                                             double k1, double k2, double k3, double k4) :
        MutableVertexPartition(graph), dataset(input_dataset), target(target), k1(k1), k2(k2), k3(k3), k4(k4)
{
//    igraph_t* rawGraph = this->get_graph()->get_igraph();
////    for (int i=0; i<this->get_graph()->vcount(); i++) {
////        igraph_real_t nodeName = VAN(rawGraph, "name", i);
////        printf("%d\n", nodeName);
////    }
//    for (int i=0; i<this->get_graph()->ecount(); i++) {
//        printf("1\n");
//        printf("Null check: %d\n", rawGraph == NULL? 1 : 0);
//        igraph_real_t edgeWeight = EAN(rawGraph, "weight", i);
//        printf("2\n");
//        printf("%d\n", edgeWeight);
//    }
//    printf("3\n");
//    for (int i=0; i < this->_membership.size(); i++) {
//        printf("%d\n", this->_membership[i]);
//    }
//    printf("Here4\n");
//    printf("%d\n", raw_dataset.size());
////    this->dataset.reserve(raw_dataset.size());
//    printf("%d\n", this->n_communities());
//    this->dataset = std::vector< std::vector< std::vector<double> > >(this->n_communities());
//    printf("%d\n", input_dataset.size());
//    printf("%d\n", this->dataset.size());
//    printf("Here5\n");
//    for (int i=0; i<input_dataset.size(); i++) {
//        printf("Here6\n");
//        printf("%d\n", input_dataset[i].size());
//        this->dataset[i].push_back(input_dataset[i]);
//        printf("Here8\n");
//    }
//    printf("Here7\n");
}

HybridVertexPartition::HybridVertexPartition(Graph* graph, vector<size_t> const& membership,
                                             std::vector< std::vector<double> >  input_dataset, std::vector<double> target,
                                             double k1, double k2, double k3, double k4) :
        MutableVertexPartition(graph, membership), dataset(input_dataset), target(target), k1(k1), k2(k2), k3(k3), k4(k4)
{
//    printf("Here1\n");
//    printf("%d\n%d\n", raw_dataset.size(), membership.size());
////    this->dataset.reserve(raw_dataset.size());
//    this->dataset = std::vector< std::vector< std::vector<double> > >(this->n_communities());
//    printf("%d\n", this->dataset.size());
//    for (int i=0; i<membership.size(); i++) {
//        printf("Here2\n");
//        this->dataset[membership[i]].push_back(input_dataset[membership[i]]);
//    }
//    printf("Here3\n");
}

HybridVertexPartition::~HybridVertexPartition()
{ }

HybridVertexPartition* HybridVertexPartition::create(Graph* graph)
{
    printf("Creating1\n");
    return new HybridVertexPartition(graph, this->dataset, this->target, this->k1, this->k2, this->k3, this->k4);
    printf("Created\n");
}

HybridVertexPartition* HybridVertexPartition::create(Graph* graph, vector<size_t> const& membership)
{
    printf("Creating2\n");
    return new HybridVertexPartition(graph, membership, this->dataset, this->target, this->k1, this->k2, this->k3, this->k4);
    printf("Created\n");
}

/*****************************************************************************
  Returns the difference in modularity if we move a node to a new community
*****************************************************************************/
double HybridVertexPartition::diff_move(size_t v, size_t new_comm)
{
#ifdef DEBUG
    cerr << "double ModularityVertexPartition::diff_move(" << v << ", " << new_comm << ")" << endl;
#endif
    size_t old_comm = this->_membership[v];
    double diff = 0.0;
    double total_weight = this->graph->total_weight()*(2.0 - this->graph->is_directed());
    if (total_weight == 0.0)
        return 0.0;
    if (new_comm != old_comm)
    {
#ifdef DEBUG
        cerr << "\t" << "old_comm: " << old_comm << endl;
#endif
        double w_to_old = this->weight_to_comm(v, old_comm);
#ifdef DEBUG
        cerr << "\t" << "w_to_old: " << w_to_old << endl;
#endif
        double w_from_old = this->weight_from_comm(v, old_comm);
#ifdef DEBUG
        cerr << "\t" << "w_from_old: " << w_from_old << endl;
#endif
        double w_to_new = this->weight_to_comm(v, new_comm);
#ifdef DEBUG
        cerr << "\t" << "w_to_new: " << w_to_new << endl;
#endif
        double w_from_new = this->weight_from_comm(v, new_comm);
#ifdef DEBUG
        cerr << "\t" << "w_from_new: " << w_from_new << endl;
#endif
        double k_out = this->graph->strength(v, IGRAPH_OUT);
#ifdef DEBUG
        cerr << "\t" << "k_out: " << k_out << endl;
#endif
        double k_in = this->graph->strength(v, IGRAPH_IN);
#ifdef DEBUG
        cerr << "\t" << "k_in: " << k_in << endl;
#endif
        double self_weight = this->graph->node_self_weight(v);
#ifdef DEBUG
        cerr << "\t" << "self_weight: " << self_weight << endl;
#endif
        double K_out_old = this->total_weight_from_comm(old_comm);
#ifdef DEBUG
        cerr << "\t" << "K_out_old: " << K_out_old << endl;
#endif
        double K_in_old = this->total_weight_to_comm(old_comm);
#ifdef DEBUG
        cerr << "\t" << "K_in_old: " << K_in_old << endl;
#endif
        double K_out_new = this->total_weight_from_comm(new_comm) + k_out;
#ifdef DEBUG
        cerr << "\t" << "K_out_new: " << K_out_new << endl;
#endif
        double K_in_new = this->total_weight_to_comm(new_comm) + k_in;
#ifdef DEBUG
        cerr << "\t" << "K_in_new: " << K_in_new << endl;
      cerr << "\t" << "total_weight: " << total_weight << endl;
#endif
        double diff_old = (w_to_old - k_out*K_in_old/total_weight) + \
               (w_from_old - k_in*K_out_old/total_weight);
#ifdef DEBUG
        cerr << "\t" << "diff_old: " << diff_old << endl;
#endif
        double diff_new = (w_to_new + self_weight - k_out*K_in_new/total_weight) + \
               (w_from_new + self_weight - k_in*K_out_new/total_weight);
#ifdef DEBUG
        cerr << "\t" << "diff_new: " << diff_new << endl;
#endif
        diff = diff_new - diff_old;
#ifdef DEBUG
        cerr << "\t" << "diff: " << diff << endl;
#endif
    }
#ifdef DEBUG
    cerr << "exit double ModularityVertexPartition::diff_move((" << v << ", " << new_comm << ")" << endl;
    cerr << "return " << diff << endl << endl;
#endif
    double m;
    if (this->graph->is_directed())
        m = this->graph->total_weight();
    else
        m = 2*this->graph->total_weight();

//    double k1 = 0.1, k2 = 0.9; // Weights for correlation portion of quality measure
//    double k3 = 0.5, k4 = 0.5; // Weights for correlation (k3) vs modularity (k4) in overall quality measure
    double pearson_diff = 0;
    try {
        pearson_diff = this->calculate_correlation_diff(v, new_comm);
//        printf("%f\n", pearson_diff);
    } catch (const alglib::ap_error& e) {
        printf("alglib::error: ");
//        printf(e.msg.c_str());
        printf("\n");
        throw;
    }
    printf("k3: %f\tk4: %f\n", this->k3, this->k4);
    printf("diff: %f\t p_diff: %f\t total: %f\n v: %d\t new_comm: %d\tN_Comm: %d\tmembership.size: %d\n", diff, pearson_diff, (this->k3*(diff/m)) + (this->k4*pearson_diff), v, new_comm, this->n_communities(), this->_membership.size());
    return (this->k3*(diff/m)) + (this->k4*pearson_diff);
}


/*****************************************************************************
  Give the modularity of the partition.

  We here use the unscaled version of modularity, in other words, we don"t
  normalise by the number of edges.
******************************************************************************/
double HybridVertexPartition::quality()
{
#ifdef DEBUG
    cerr << "double ModularityVertexPartition::quality()" << endl;
#endif
    double mod = 0.0;
    double corr = 0.0;
//    double k1 = 0.1, k2 = 0.9; // Weights for correlation portion of quality measure
//    double k3 = 0.5, k4 = 0.5; // Weights for correlation (k3) vs modularity (k4) in overall quality measure

    printf("Test4\n");
    std::vector<size_t> membership = this->membership();
    printf("Membership:\n");
    for (unsigned long i=0; i < membership.size(); i++) {
        printf("%d\n", membership[i]);
    }
    printf("\\Membership\n");

    double m;
    if (this->graph->is_directed())
        m = this->graph->total_weight();
    else
        m = 2*this->graph->total_weight();

    if (m == 0)
        return 0.0;

    for (size_t c = 0; c < this->n_communities(); c++)
    {
        double w = this->total_weight_in_comm(c);
        double w_out = this->total_weight_from_comm(c);
        double w_in = this->total_weight_to_comm(c);
#ifdef DEBUG
        size_t csize = this->csize(c);
      cerr << "\t" << "Comm: " << c << ", size=" << csize << ", w=" << w << ", w_out=" << w_out << ", w_in=" << w_in << "." << endl;
#endif
        mod += w - w_out*w_in/((this->graph->is_directed() ? 1.0 : 4.0)*this->graph->total_weight());

        // Calculating pearson correlation
        std::vector<size_t> members;
        for (int i=0; i<_membership.size(); i++) {
            if (this->_membership[i] == c) {
                members.push_back(i);
            }
        }

        alglib::real_2d_array subset, V;
        alglib::real_1d_array s2, pc, alg_target;

        subset.setlength(this->target.size(), members.size());
        for (int i=0; i<members.size(); i++) {
            std::vector<int> node_ids = this->get_graph()->get_features(members[i]);
            for (int j=0; j<node_ids.size(); j++)
                for (int k=0; j<this->dataset[node_ids[j]].size(); j++) {
                    subset[k][i] = this->dataset[node_ids[j]][k];
                }
        }

        V.setlength(members.size(), 1);
        s2.setlength(1);
        alglib::pcatruncatedsubspace(subset, this->dataset[0].size(), members.size(), 1, 0.0006, 100000, s2, V);

        pc.setlength(this->dataset[0].size());
        alg_target.setlength(this->target.size());
        for (int i=0; i<this->target.size(); i++) {
            alg_target[i] = this->target[i];
        }
        for (int i=0; i<this->dataset[0].size(); i++) {
            pc[0] = 0;
            for (int j=0; j<members.size(); j++) {
                pc[i] += V(j, 0) * subset(i, j);
            }
        }

        corr += alglib::pearsoncorr2(pc, alg_target);
    }
    double q = (2.0 - this->graph->is_directed())*mod;
#ifdef DEBUG
    cerr << "exit double ModularityVertexPartition::quality()" << endl;
    cerr << "return " << q/m << endl << endl;
#endif

    return (this->k3*(q/m)) + (this->k4*corr);
}

double HybridVertexPartition::calculate_correlation_diff(size_t v, size_t new_comm) {
//    printf("In diff func.\n");
    printf("Num comms: %d\n", this->get_graph()->get_features().size());
    size_t old_comm = this->_membership[v];
    std::vector<size_t> old_members = std::vector<size_t>();
    std::vector<size_t> new_members = std::vector<size_t>();
    for (int i=0; i < this->_membership.size(); i++) {
        if (this->_membership[i] == old_comm) {
            old_members.push_back(i);
        }
        if (this->_membership[i] == new_comm || i == v) {
            new_members.push_back(i);
        }
    }
    alglib::real_2d_array old_including, old_excluding, new_including, new_excluding;
//    printf("\n\nStart\n");
//    printf("Dataset Size: %d\nFeature Size: %d\n", this->dataset.size(), this->dataset[v].size());
//    printf("Node: %d\nOld Community: %d, Size: %d\nNew Community: %d, Size: %d\n", v, old_comm, old_members.size(), new_comm, new_members.size());
    old_including.setlength(this->dataset[v].size(), old_members.size());
//    printf("Done.");
    if (old_members.size() > 1)
        old_excluding.setlength(this->dataset[v].size(), old_members.size() - 1);
    new_including.setlength(this->dataset[v].size(), new_members.size());
    if (new_members.size() > 1)
        new_excluding.setlength(this->dataset[v].size(), new_members.size() - 1);
    int old_offset = 0, new_offset = 0;
    for (int i=0; i<old_members.size(); i++) {
//        printf("1\n");
        std::vector<int> node_ids = this->get_graph()->get_features(old_members[i]);
//        printf("2\n");
        for (int j=0; j<node_ids.size(); j++) {
//            printf("3\n");
            for (int k=0; k<this->dataset[node_ids[j]].size(); k++) {
//                printf("4\n");
                old_including[k][i] = this->dataset[node_ids[j]][k];
                if (old_members.size() > 1) {
                    if (old_members[i] != v) {
                        old_excluding[k][i - old_offset] = this->dataset[node_ids[j]][k];
                    } else {
                        old_offset = 1;
                    }
                }
            }
        }
    }
//    printf("5\n");
    for (int i=0; i<new_members.size(); i++) {
//        printf("6\n");
        std::vector<int> node_ids = this->get_graph()->get_features(new_members[i]);
//        printf("7\n");
        for (int j=0; j<node_ids.size(); j++) {
//            printf("8\n");
            for (int k=0; k<this->dataset[node_ids[j]].size(); k++) {
//                printf("9\n");
                new_including[k][i] = this->dataset[node_ids[j]][k];
                if (new_members.size() > 1) {
                    if (new_members[i] != v) {
                        new_excluding[k][i - new_offset] = this->dataset[node_ids[j]][k];
                    } else {
                        new_offset = 1;
                    }
                }
            }
        }
    }
//    printf("10\n");
    alglib::real_1d_array s2_old_i, s2_old_e, s2_new_i, s2_new_e;
    alglib::real_2d_array V_old_i, V_old_e, V_new_i, V_new_e;
    s2_old_i.setlength(1);
    if (old_members.size() > 1)
        s2_old_e.setlength(1);
    s2_new_i.setlength(1);
    if (new_members.size() > 1)
        s2_new_e.setlength(1);
    V_old_i.setlength(old_members.size(), 1);
    if (old_members.size() > 1)
        V_old_e.setlength(old_members.size() - 1, 1);
    V_new_i.setlength(new_members.size(), 1);
    if (new_members.size() > 1)
        V_new_e.setlength(new_members.size() - 1, 1);

    alglib::pcatruncatedsubspace(old_including, this->dataset[v].size(), old_members.size(), 1, 0.0006, 100000, s2_old_i, V_old_i);
    if (old_members.size() > 1)
        alglib::pcatruncatedsubspace(old_excluding, this->dataset[v].size(), old_members.size() - 1, 1, 0.0006, 100000, s2_old_e, V_old_e);
    alglib::pcatruncatedsubspace(new_including, this->dataset[v].size(), new_members.size(), 1, 0.0006, 100000, s2_new_i, V_new_i);
    if (new_members.size() > 1)
        alglib::pcatruncatedsubspace(new_excluding, this->dataset[v].size(), new_members.size() - 1, 1, 0.0006, 100000, s2_new_e, V_new_e);

    alglib::real_1d_array alg_target, pc_old_i, pc_old_e, pc_new_i, pc_new_e;
    alg_target.setlength(this->target.size());
    pc_old_i.setlength(this->dataset[v].size());
    if (old_members.size() > 1)
        pc_old_e.setlength(this->dataset[v].size());
    pc_new_i.setlength(this->dataset[v].size());
    if (new_members.size() > 1)
        pc_new_e.setlength(this->dataset[v].size());

    for (int i=0; i<this->target.size(); i++) {
        alg_target[i] = this->target[i];
    }

    for (int i=0; i<this->dataset[v].size(); i++) {
        pc_old_i[i] = 0;
        for (int j=0; j<old_members.size(); j++) {
            pc_old_i[i] += V_old_i(j, 0) * old_including(i, j);
        }
    }
    if (old_members.size() > 1) {
        for (int i = 0; i < this->dataset[v].size(); i++) {
            pc_old_e[i] = 0;
            for (int j = 0; j < old_members.size() - 1; j++) {
                pc_old_e[i] += V_old_e(j, 0) * old_excluding(i, j);
            }
        }
    }
    for (int i=0; i<this->dataset[v].size(); i++) {
        pc_new_i[i] = 0;
        for (int j=0; j<new_members.size(); j++) {
            pc_new_i[i] += V_new_i(j, 0) * new_including(i, j);
        }
    }
    if (new_members.size() > 1) {
        for (int i = 0; i < this->dataset[v].size(); i++) {
            pc_new_e[i] = 0;
            for (int j = 0; j < new_members.size() - 1; j++) {
                pc_new_e[i] += V_new_e(j, 0) * new_excluding(i, j);
            }
        }
    }

    double p_old_i, p_old_e, p_new_i, p_new_e;
    p_old_i = alglib::pearsoncorr2(pc_old_i, alg_target);
    p_old_e = old_members.size() > 1 ? alglib::pearsoncorr2(pc_old_e, alg_target) : 0;
    p_new_i = alglib::pearsoncorr2(pc_new_i, alg_target);
    p_new_e = new_members.size() > 1 ? alglib::pearsoncorr2(pc_new_e, alg_target) : 0;

    printf("poi: %f\tpoe: %f\tpni: %f\tpne: %f\n", p_old_i, p_old_e, p_new_i, p_new_e);

//    return (abs(p_old_e) - abs(p_old_i)) + (abs(p_new_i) - abs(p_new_e)); // delta_old + delta_new
    return ((0.5*(1-p_old_e)) - (0.5*(1-p_old_i))) + ((0.5*(1-p_new_i)) - (0.5*(1-p_new_e)));
//    return ((0.5*(1+abs(p_old_e))) - (0.5*(1+abs(p_old_i)))) + ((0.5*(1+abs(p_new_i))) - (0.5*(1+abs(p_new_e))));
}