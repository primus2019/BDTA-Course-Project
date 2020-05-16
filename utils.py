import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import glob


def label_encoding(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''Convert a scipy sparse matrix to a torch sparse tensor.'''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset, seed=None):
    if seed:
        torch.manual_seed(seed)
        rg = np.random.default_rng(seed)
    raw_X_Y = np.genfromtxt('data/{}/{}.content'.format(dataset, dataset), dtype=np.dtype(str))
    idx = np.array(raw_X_Y[:, 0], dtype=np.dtype(str))
    X = sp.csr_matrix(raw_X_Y[:, 1:-1], dtype=np.float32)
    Y = label_encoding(raw_X_Y[:, -1])

    # build graph
    idx_map = {j: i for i, j in enumerate(idx)}
    raw_A = np.genfromtxt('data/{}/{}.cites'.format(dataset, dataset), dtype=np.dtype(str))
    
    diff = np.setdiff1d(raw_A.flatten(), idx)
    A_pairs = raw_A[~np.isin(raw_A, diff).any(axis=1)]
    A_pairs = np.vectorize(idx_map.get)(A_pairs).astype(np.int32)
    A = sp.coo_matrix((np.ones(A_pairs.shape[0]), (A_pairs[:, 0], A_pairs[:, 1])),
                        shape=(Y.shape[0], Y.shape[0]),
                        dtype=np.float32)

    # build symmetric Adjacency matrix
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)

    X = normalize(X)
    A = normalize(A + sp.eye(A.shape[0]))

    sample_num = X.shape[0]

    print('Train indices: 0 ~ {};'.format(int(sample_num * 0.1)),
        'Valid indices: {} ~ {};'.format(int(sample_num * 0.1), int(sample_num * 0.33)),
        'Test indices: {} ~ {};'.format(int(sample_num * 0.33), sample_num - 1))

    idx_all = list(range(sample_num))
    rg.shuffle(idx_all)
    idx_train = idx_all[: int(sample_num * 0.1)]
    idx_val = idx_all[int(sample_num * 0.1): int(sample_num * 0.33)]
    idx_test = idx_all[int(sample_num * 0.33): ]

    X = torch.FloatTensor(np.array(X.todense()))
    Y = torch.LongTensor(np.where(Y)[1])
    A = sparse_mx_to_torch_sparse_tensor(A)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, X, Y, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_rand_split_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/randsplit/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/randsplit/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    print(labels.shape)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # labels = np.apply_along_axis(lambda r: np.where(r)[0], 1, labels).flatten()
    labels = [np.where(r)[0][0] for r in labels]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]

    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    A = sparse_mx_to_torch_sparse_tensor(adj)
    X = torch.FloatTensor(labels)
    Y = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return A, X, Y, idx_train, idx_val, idx_test