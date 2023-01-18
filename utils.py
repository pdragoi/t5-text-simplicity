import matplotlib.pylab as plt
from typing import List, Tuple
import os

def save_plot(
    plots: List[Tuple[List[float], str]],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.figure()
    for data, label in plots:
        plt.plot(data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()