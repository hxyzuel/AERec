import numpy as np

def ndcg_atk_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    # pred_data = r

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def recall_atk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k

    """
    right_pred = r[:, :k].sum(1)
    # right_pred = r.sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    correct_pred_count = right_pred / recall_n
    correct_pred_count[np.isnan(correct_pred_count)] = 0
    recall = np.sum(correct_pred_count)
    return recall

def get_label(test_data, pred_data):
    """
    :param test_data: the collection of positively rated items in Test dataset for each user
    :param pred_data: the collection of items considered as positive by prediction
    :return test_data ranking & pred_data ranking
    """
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        # == [_ in groundTrue for _ in predictTopK]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')