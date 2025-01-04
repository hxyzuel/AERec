import json

from model.mf import MF
from model.keras import MF as KerasMF


def main():
    with open('config.json', 'r') as f:
        args = json.load(f)

    mode = "gpu"
    dataset_name = "Books"
    data_path = "data/" + dataset_name + '/ratings.dat'
    dim = 1  # 1,2,3,4,5
    # alpha = args['alpha']  # 1e-2,1e-3,1e-4,1e-5
    beta = 1e-2  # 1e-2,1e-3,1e-4,1e-5
    epoch = 500  # 500
    num_rec_items = 50
    verbose = 1

    for run in [1, 2, 3]:
        for alpha in [1e-2]:
            if mode == 'gpu':
                mf = KerasMF(data_path, dataset_name)
            else:
                mf = MF(data_path, dataset_name)
            mf.train(dim, alpha, beta, epoch, verbose=verbose)
            mf.recommend(run, num_rec_items)


if __name__ == '__main__':
    main()
