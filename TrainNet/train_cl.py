import cleanlab.classification
import numpy as np
import torch
from sklearn.model_selection import cross_val_predict
from skorch import NeuralNetClassifier
from cleanlab.filter import find_label_issues
from Models.ToyModel import MLP


def clean_lab(sample_index, net, dataset, num_crossval_folds=10, set_test=False):
    # for efficiency, num_crossval_folds values like 5 or 10 will generally work better
    Dataset = dataset(sample_index)
    labels = [Dataset.labels[indice] for indice in range(len(sample_index))]
    samples = torch.from_numpy(np.array(
        [np.array(Dataset.samples[indice]) for indice in range(len(sample_index))]
    ))
    pred_probs = cross_val_predict(
        net,
        samples,
        labels,
        cv=num_crossval_folds,
        method="predict_proba",
    )

    prune_list = find_label_issues(labels, pred_probs, )
    list_name = r'..\tmp\tmp_list_prune.npy'
    np.save(list_name, prune_list)

    rank_list = cleanlab.rank.get_label_quality_scores(labels, pred_probs)
    list_name = r'..\tmp\tmp_list_rank.npy'
    np.save(list_name, rank_list)

    unprune_list = np.array(
        [sample_index[i] for i in range(len(sample_index)) if not prune_list[i]]
    )

    if set_test:
        # 挑选test_set
        selected_list = np.random.choice(unprune_list, size=int(0.2 * len(labels)))
        list_name = r'..\tmp\tmp_list_test.npy'
        np.save(list_name, selected_list)

    elif not set_test:
        # 挑选certain set
        list_name = r'..\tmp\tmp_list_good.npy'
        np.save(list_name, unprune_list)

        # 挑选test set


def train_cl(index, dataset):
    #
    aux_net = MLP(96 * 64, 2, cl=True).double()
    aux_net = NeuralNetClassifier(aux_net)
    # you can edit your own aux_net
    clean_lab(sample_index=index, net=aux_net, dataset=dataset, set_test=False)

    index = list(np.load(r'..\tmp\tmp_list_good.npy'))
    aux_net = MLP(96 * 64, 2, cl=True).double()
    aux_net = NeuralNetClassifier(aux_net)
    clean_lab(sample_index=index, net=aux_net, dataset=DatasetTemplate, set_test=True)


if __name__ == '__main__':
    from Dataset.DatasetTemplate import DatasetTemplate

    rand_seed = 5
    generator = torch.Generator().manual_seed(rand_seed)
    index = list(range(0, 15705))
    dataset = DatasetTemplate
    train_cl(index=index, dataset=dataset)
