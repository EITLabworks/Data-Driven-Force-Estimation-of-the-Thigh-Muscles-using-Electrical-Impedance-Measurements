import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_PCA(
    signal: np.ndarray,
    n_cmp: int = 2,
    feature: list = [],
    feature_label: str = "Feature",
    plot: bool = True,
):
    pca = PCA(n_cmp)
    P = pca.fit(signal).transform(signal)

    if n_cmp == 2 and plot:
        plt.figure(figsize=(6, 4))
        if len(feature) == 0:
            plt.scatter(P[:, 0], P[:, 1])
        else:
            norm = mcolors.Normalize(vmin=np.min(feature), vmax=np.max(feature))
            plt.scatter(P[:, 0], P[:, 1], c=feature, cmap="viridis", norm=norm)
            plt.colorbar(label=feature_label)
        plt.grid()
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.show()

    if n_cmp == 3 and plot:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection="3d")
        if len(feature) == 0:
            sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2])
        else:
            norm = mcolors.Normalize(vmin=np.min(feature), vmax=np.max(feature))
            sc = ax.scatter(
                P[:, 0], P[:, 1], P[:, 2], c=feature, cmap="viridis", norm=norm
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(feature_label)

        # Achsentitel setzen
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.set_zlabel("component 3")
        plt.show()


def compute_TSNE(
    X: np.ndarray,
    feature: list,
    feature_label: str = "Feature",
    plot: bool = True,
):

    X_reshaped = X.reshape(X.shape[0], -1)
    X_reshaped = np.nan_to_num(X_reshaped)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_reshaped)

    plt.figure(figsize=(6, 4))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=feature, cmap="jet", alpha=0.7)
    plt.colorbar(label=feature_label)
    plt.title("t-SNE")
    plt.show()
