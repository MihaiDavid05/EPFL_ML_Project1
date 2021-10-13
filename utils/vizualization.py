import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist_panel(feats, feats_name, output_path):
    """
    PLot a panel with histograms for each feature.
    :param feats:
    :param feats_name:
    :param output_path:
    :return:
    """
    h = 6
    w = 5
    fig, ax = plt.subplots(h, w, figsize=(10, 8))
    for i in range(h):
        for j in range(w):
            sbplt = ax[i, j]
            sbplt.set_yscale('log')
            sbplt.hist(feats[:, j + i * w], bins=100)
            sbplt.set_title(feats_name[j + i * w])

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
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.scatterplot(x1, x2, hue=y, s=5)
    fig.tight_layout()
    fig.savefig(output_path)
