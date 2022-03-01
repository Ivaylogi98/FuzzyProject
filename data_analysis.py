import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    df1 =  np.array(pd.read_csv('Datasets/Gas/batch1.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df2 =  np.array(pd.read_csv('Datasets/Gas/batch2.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df3 =  np.array(pd.read_csv('Datasets/Gas/batch3.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df4 =  np.array(pd.read_csv('Datasets/Gas/batch4.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df5 =  np.array(pd.read_csv('Datasets/Gas/batch5.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df6 =  np.array(pd.read_csv('Datasets/Gas/batch6.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df7 =  np.array(pd.read_csv('Datasets/Gas/batch7.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df8 =  np.array(pd.read_csv('Datasets/Gas/batch8.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df9 =  np.array(pd.read_csv('Datasets/Gas/batch9.dat',  sep=' |:', header=None, engine='python'))[:, ::2]
    df10 = np.array(pd.read_csv('Datasets/Gas/batch10.dat', sep=' |:', header=None, engine='python'))[:, ::2]
    df = np.vstack([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])

    X_train, X_test, y_train, y_test = train_test_split(df[:,1:], df[:,0], test_size=0.2, random_state=42)

    pca = PCA(n_components=128)
    PCA_train = pca.fit_transform(df[:, 1:])

    PC_values = np.arange(pca.n_components) + 1
    plt.plot(PC_values[:10], pca.explained_variance_ratio_[:10], 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 11   
    ax.plot(PCA_train[0:2564,0], PCA_train[0:2564,1], PCA_train[0:2564,2], 'o', markersize=2.5, label='Ethanol')
    ax.plot(PCA_train[2565:5490,0], PCA_train[2565:5490,1], PCA_train[2565:5490,2], 'o', markersize=2.5, label='Ethylene')
    ax.plot(PCA_train[5491:7131,0], PCA_train[5491:7131,1], PCA_train[5491:7131,2], 'o', markersize=2.5, label='Ammonia')
    ax.plot(PCA_train[7132:9067,0], PCA_train[7132:9067,1], PCA_train[7132:9067,2], 'o', markersize=2.5, label='Acetaldehyde')
    ax.plot(PCA_train[9068:12076,0], PCA_train[9068:12076,1], PCA_train[9068:12076,2], 'o', markersize=2.5, label='Acetone')
    ax.plot(PCA_train[12077:13909,0], PCA_train[12077:13909,1], PCA_train[12077:13909,2], 'o', markersize=2.5, label='Toluene')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(loc='upper right')

    plt.savefig(f'Images/PCA.png')
    plt.show()


    # from sklearn.manifold import TSNE

    # for p in range(70, 100, 10):
    #     i = 2000
    #     t_sne = TSNE(n_components=3, verbose=1, perplexity=p, n_iter=i) #change perplexity for better result
    #     TSNE_train = t_sne.fit_transform(df[:, 1:])


    #     fig = plt.figure(figsize=(8,8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     plt.rcParams['legend.fontsize'] = 11   
    #     ax.plot(TSNE_train[0:2564,0], TSNE_train[0:2564,1], TSNE_train[0:2564,2], 'o', markersize=2.5, label='Ethanol')
    #     ax.plot(TSNE_train[2565:5490,0], TSNE_train[2565:5490,1], TSNE_train[2565:5490,2], 'o', markersize=2.5, label='Ethylene')
    #     ax.plot(TSNE_train[5491:7131,0], TSNE_train[5491:7131,1], TSNE_train[5491:7131,2], 'o', markersize=2.5, label='Ammonia')
    #     ax.plot(TSNE_train[7132:9067,0], TSNE_train[7132:9067,1], TSNE_train[7132:9067,2], 'o', markersize=2.5, label='Acetaldehyde')
    #     ax.plot(TSNE_train[9068:12076,0], TSNE_train[9068:12076,1], TSNE_train[9068:12076,2], 'o', markersize=2.5, label='Acetone')
    #     ax.plot(TSNE_train[12077:13909,0], TSNE_train[12077:13909,1], TSNE_train[12077:13909,2], 'o', markersize=2.5, label='Toluene')
    #     ax.set_xlabel('PC1')
    #     ax.set_ylabel('PC2')
    #     ax.set_zlabel('PC3')
    #     ax.legend(loc='upper right')

    #     plt.savefig(f'Images/TSNE_p-{p}_i-{i}.png')
        # plt.show()

if __name__ == "__main__":
    main()