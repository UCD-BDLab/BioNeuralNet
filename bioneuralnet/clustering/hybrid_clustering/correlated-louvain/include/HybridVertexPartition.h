//
// Created by Mohamed Abdel-Hafiz on 10/7/21.
//

#ifndef LEIDENALG_IGRAPH_HYBRIDVERTEXPARTITION_H
#define LEIDENALG_IGRAPH_HYBRIDVERTEXPARTITION_H

#include "MutableVertexPartition.h"
#include "vector"
#include "dataanalysis.h"
#include "statistics.h"
#include "ap.h"

class HybridVertexPartition : public MutableVertexPartition
{
public:
    HybridVertexPartition(Graph* graph,
                              vector<size_t> const& membership);
    HybridVertexPartition(Graph* graph);
    HybridVertexPartition(Graph* graph, std::vector< std::vector<double> >  input_dataset, std::vector<double> target,
                          double k1, double k2, double k3, double k4);
    HybridVertexPartition(Graph* graph,
                          vector<size_t> const& membership,
                          std::vector< std::vector<double> >  input_dataset, std::vector<double> target,
                          double k1, double k2, double k3, double k4);
    virtual ~HybridVertexPartition();
    virtual HybridVertexPartition* create(Graph* graph);
    virtual HybridVertexPartition* create(Graph* graph, vector<size_t> const& membership);

    virtual double diff_move(size_t v, size_t new_comm);
    virtual double quality();
    virtual double calculate_correlation_diff(size_t v, size_t new_comm);
//    void move_node(size_t v,size_t new_comm);

protected:
    std::vector< std::vector<double> > dataset;
    std::vector<double> target;
    double k1, k2, k3, k4;
private:
};

#endif //LEIDENALG_IGRAPH_HYBRIDVERTEXPARTITION_H
