import numpy as np
from sklearn.manifold import TSNE
import pickle

import matplotlib.pyplot as plt

def main():
    dims = 64
    x_train, y_train = load_dataset(dims)

    x_embed = TSNE(n_components=2).fit_transform(x_train.T)
    print(x_embed.shape)

    y_train = y_train[0]

    plt.figure()
    plt.plot(x_embed[y_train == 1, -2], x_embed[y_train == 1, -1], 'bx', label='sars-cov-2')
    plt.plot(x_embed[y_train == 0, -2], x_embed[y_train == 0, -1], 'go', label='normal')
    plt.legend()
    plt.savefig("./covid/plots/covid_d_tsne_embed.png")


def load_dataset(dims):
    x_train = pickle.load(open("./covid/" + str(dims) + "/x_train_d.p","rb"))
    y_train = pickle.load(open("./covid/" + str(dims) + "/y_train_d.p","rb"))
 
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    y_train = y_train.reshape(y_train.shape[0],1).T

    # flatten x
    x_train = x_train.reshape(x_train.shape[0],-1).T

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train

if __name__ == "__main__":
    main()