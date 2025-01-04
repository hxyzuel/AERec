import json

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *


class MF:
    def __init__(self, data_path, dataset_name):
        self.data = np.genfromtxt(data_path,
                                  delimiter='\t',
                                  skip_header=True,
                                  dtype=np.int32)
        self.model = None
        self.result = None
        self.dataset_name = dataset_name

    def train(self,
              dim=2,
              alpha=1e-3,
              beta=1e-4,
              epoch=300,
              batch=1000,
              verbose=2):
        self.build(dim, alpha, beta)
        self._train(epoch,
                    batch,
                    verbose)
        self.predict()

    def predict(self):
        predicted = self.model.predict(
            [self.data[:, 0], self.data[:, 1]]
        ).reshape(-1)
        pd.DataFrame({'actual': self.data[:, 2],
                      'predicted': predicted}) \
            .to_csv(f'result/{self.dataset_name}/predict_keras.csv', index=False)

    def recommend(self, run,num_rec_items):
        """
        do recommend num_rec_items items excluding the observed items.
        """
        user_mtx, item_mtx = self.model.get_weights()
        predicted = np.inner(user_mtx, item_mtx) * -1
        predicted[self.data[:, 0], self.data[:, 1]] *= 0
        self.result = pd.DataFrame(predicted.argsort()[:, :num_rec_items],
                                   columns=['top%s' % i
                                            for i in range(1, num_rec_items + 1)],
                                   index=np.arange(len(user_mtx)))
        self.result.to_csv(f'result/{self.dataset_name}/ecommend_keras.csv')

        predicted = predicted.argsort()[:, :num_rec_items]
        test_path = f'data/{self.dataset_name}/test.txt'
        ground_truth = []
        with open(test_path) as f:
            lines = f.readlines()
            for line in lines[1:]:
                iids = line.rstrip().split(' ')[1:]
                temp = []
                for iid in iids:
                    temp.append(int(iid))
                ground_truth.append(temp)

        recalls = []
        ndcgs = []
        sorted_items = predicted
        r = get_label(ground_truth, sorted_items)
        for k in [5, 10, 20, 50]:
            recall = recall_atk(ground_truth, r, k)/len(ground_truth)
            ndcg = ndcg_atk_r(ground_truth, r, k)/len(ground_truth)
            recalls.append(recall)
            ndcgs.append(ndcg)

        metrics = {}
        metrics['recalls'] = recalls
        metrics['ndcgs'] = ndcgs
        metrics_path = f'result/{self.dataset_name}/metrics{run}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def build(self, dim, alpha, beta):
        users, items, _ = self.data.max(axis=0)

        user_input = tf.keras.layers.Input((1,), name='user')
        user_vec = self.embedding(user_input, users, dim, beta, 'user_vec')
        item_input = tf.keras.layers.Input((1,), name='item')
        item_vec = self.embedding(item_input, items, dim, beta, 'item_vec')
        outputs = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])

        model = tf.keras.models.Model([user_input, item_input], outputs)
        adam = tf.keras.optimizers.Adam(alpha)
        model.compile(adam, 'mse')

        model.summary()
        self.model = model

    def _train(self, epoch, batch, verbose=2):
        self.model.fit([self.data[:, 0], self.data[:, 1]],
                       self.data[:, 2],
                       epochs=epoch,
                       verbose=verbose,
                       batch_size=batch,
                       shuffle=False)

    def embedding(self,
                  last_layer,
                  input_dim,
                  latent_dim,
                  beta,
                  name):
        input_length = 1
        regularizer = tf.keras.regularizers.l2(beta)
        initializer = tf.keras \
            .initializers \
            .RandomNormal()
        embedding = tf.keras.layers.Embedding(
            input_dim + 1,
            latent_dim,
            input_length=input_length,
            embeddings_initializer=initializer,
            embeddings_regularizer=regularizer)(last_layer)
        return tf.keras.layers.Flatten(name=name)(embedding)
