import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation(corr, feats_name, output_path):
    """
    PLot correlation between features.
    :param corr: Correlation matrix.
    :param feats_name: Features name.
    :param output_path: Plot output path
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(corr, annot=True, ax=ax, xticklabels=feats_name, yticklabels=feats_name)
    fig.suptitle("Correlation matrix")
    fig.tight_layout()
    fig.savefig(output_path)


def plot_hist_panel(feats, feats_name, output_path, log_scale_y=False):
    """
    PLot a panel with histograms for each feature.
    :param feats: Features.
    :param feats_name: Features name.
    :param output_path: PLot output path.
    :param log_scale_y: Whether to apply log on y scale.
    """
    w = 5
    h = len(feats_name) // w + 1
    fig, ax = plt.subplots(h, w, figsize=(10, 8))
    for i in range(h):
        j = 0
        while i * w + j < len(feats_name) and j < w:
            sbplt = ax[i, j]
            if log_scale_y:
                sbplt.set_yscale('log')
            sbplt.hist(feats[:, j + i * w], bins=100)
            sbplt.set_title(feats_name[j + i * w])
            j += 1

    fig.tight_layout()

    fig.text(0.4, 0, "Specific feature range of values")
    fig.text(0, 0.4, "Occurrences of the value", rotation=90)

    fig.savefig(output_path)


def plot_loss(iters, tr_loss, output_path):
    """
    PLot the training loss.
    :param iters: Number of iterations.
    :param tr_loss: Loss.
    :param output_path: Output path for the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(iters, tr_loss, '-b', label='train loss')
    ax.set_title("Training loss")
    ax.set_xlabel("Nr iteration")
    ax.set_ylabel("Loss")

    fig.legend(loc='upper right')
    fig.tight_layout()

    fig.savefig(output_path)


def plot_pca(x1, x2, y, output_path):
    """
    Visulize data in 2D.
    :param x1: First feature.
    :param x2: Second feature.
    :param y: Labels.
    :param output_path: Plot output path.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.scatterplot(x=x1, y=x2, hue=y, s=5)
    fig.tight_layout()
    fig.savefig(output_path)


def cross_validation_visualization_lambdas(lambdas, f1_tr, f1_te, output_path):
    """
    Visualize the f1 score for training and validation for a range of lambdas.
    :param lambdas: Regularization parameters array.
    :param f1_tr: F1 score for training
    :param f1_te: F1 score for validation
    :param output_path: Plot output_path
    """
    plt.semilogx(lambdas, f1_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambdas, f1_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("F1 Score")
    plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(output_path)


def bias_variance_decomposition_visualization(degrees, f1_tr, f1_te, output_path):
    """
    Visualize the bias variance decomposition.
    :param degrees:
    :param f1_tr:
    :param f1_te:
    :param output_path:
    :return:
    """
    f1_tr_mean = np.expand_dims(np.mean(f1_tr, axis=0), axis=0)
    f1_te_mean = np.expand_dims(np.mean(f1_te, axis=0), axis=0)
    plt.plot(
        degrees,
        f1_tr.T,
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    plt.plot(
        degrees,
        f1_te.T,
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.3)
    plt.plot(
        degrees,
        f1_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=1)
    plt.plot(
        degrees,
        f1_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.xlim(1, 15)
    plt.ylim(0.2, 0.9)
    plt.xlabel("degree")
    plt.ylabel("f1 score")
    plt.legend(loc=1)
    plt.title("Bias-Variance Decomposition")
    plt.savefig(output_path)
