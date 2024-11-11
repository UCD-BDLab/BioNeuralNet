from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--gnn_model', dest='gnn_model', default='GCN', type=str,
                        help='GNN Model Class Name [GCN, GAT, SAGE, GIN]')

    parser.add_argument('--repeat_num', dest='repeat_num', default=10, type=int,
                        help='Number of Repeats')
    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='Run on GPU')
    parser.add_argument('--cuda', dest='cuda', default='0', type=str,
                        help='CUDA Device')
    parser.add_argument('--tune', dest='tune', action='store_true',
                        help='Tune the Model Hyperparameters')

    parser.add_argument('--dataset_dir', dest='dataset_dir', type=str,
                        help='Dataset Directory (Subjects MultiOmics & Associated Networks)')

    args = parser.parse_args()
    return args
