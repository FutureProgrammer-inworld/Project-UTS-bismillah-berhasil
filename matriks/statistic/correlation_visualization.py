import numpy as np
from matriks.matrix import Matrix
import matplotlib.pyplot as plt

def plot_correlation_matrix(matrix, labels):
    """
    Menampilkan heatmap dari matriks korelasi.
    """
    if isinstance(matrix, Matrix):
        data = matrix.data
    else:
        data = matrix

    data = np.array(data, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Korelasi")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Heatmap Korelasi Antar Variabel", fontsize=14, fontweight="bold")

    # Tampilkan nilai di dalam sel
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()
