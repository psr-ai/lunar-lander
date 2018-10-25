import matplotlib.pyplot as plt


def show_histogram(data, bin_width, title, x_label, y_label):
    pruned_data = [int(d) for d in data]
    plt.hist(pruned_data, bins=range(min(pruned_data), max(pruned_data) + bin_width, bin_width))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
