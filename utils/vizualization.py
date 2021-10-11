import matplotlib.pyplot as plt


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
    plt.yscale('log', nonpositive='clip')
    for i in range(h):
        for j in range(w):
            sbplt = ax[i, j]
            sbplt.hist(feats[:, j + i * w], bins=100)
            sbplt.set_title(feats_name[j + i * w])

    fig.tight_layout()

    fig.text(0.4, 0, "Specific feature range of values")
    fig.text(0, 0.4, "Occurrences of the value", rotation=90)

    fig.savefig(output_path)


def check_relevant_samples():
    # TODO: I do not know what this should do
    pass


def plot_loss_accuracy(losses, accs):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # TODO: Implement this and create folder for each experiment
    pass
