# matriks/operations/regresi_linier.py
from matriks.operations.transpose import transpose
from matriks.operations.inverse import inverse
from matriks.operations.multiplier import multiply_matrices
from matriks.matrix import Matrix

def regresi_linier(X, y):
    """
    Menghitung regresi linier sederhana atau berganda menggunakan metode OLS.
    Rumus: β = (XᵀX)⁻¹ Xᵀy
    """

    # Pastikan y berbentuk kolom
    if isinstance(y, Matrix):
        y_data = y.data
    elif isinstance(y, list):
        # Kalau list 1D, ubah jadi kolom
        if not isinstance(y[0], list):
            y_data = [[val] for val in y]
        else:
            y_data = y
    else:
        raise TypeError("y harus berupa Matrix atau list.")

    Xt = transpose(X)
    XtX = multiply_matrices(Xt, X)
    XtX_inv = inverse(XtX)
    XtY = multiply_matrices(Xt, Matrix(y_data))
    beta = multiply_matrices(XtX_inv, XtY)

    return beta


def prediksi(X, beta):
    """Menghitung nilai prediksi y_hat = X * beta"""
    return multiply_matrices(X, beta)


def evaluasi(y_asli, y_pred):
    """Menghitung metrik evaluasi: SSE, MSE, dan R²"""

    # ubah ke list 1D
    y_asli = [val[0] if isinstance(val, list) else val for val in y_asli]
    y_pred = [val[0] if isinstance(val, list) else val for val in y_pred]

    n = len(y_asli)
    mean_y = sum(y_asli) / n
    residuals = [y_asli[i] - y_pred[i] for i in range(n)]

    # SSE (Sum of Squared Errors)
    SSE = sum(r**2 for r in residuals)

    # SST (Total Sum of Squares)
    SST = sum((y - mean_y)**2 for y in y_asli)

    # R² dan MSE
    R2 = 1 - (SSE / SST)
    MSE = SSE / n

    return {
        "SSE": SSE,
        "MSE": MSE,
        "R2": R2,
        "residuals": residuals
    }

def pilih_variabel_xy(matrix_data):
    header = matrix_data.data[0]
    rows = matrix_data.data[1:]  # data tanpa header

    print("=== Pilih Variabel X dan Y ===")
    print("Struktur Matriks:")
    for i, row in enumerate(matrix_data.data[:10]):  # tampilkan sebagian
        print(f"{i+1}: {row}")
    print(f"\nTerdapat {len(header)} kolom (indeks 0–{len(header)-1})")

    # input bisa berupa indeks atau nama kolom
    kol_y = input("Pilih kolom untuk variabel Y (indeks atau nama): ")
    if not kol_y.isdigit():
        if kol_y not in header:
            raise ValueError(f"Kolom '{kol_y}' tidak ditemukan dalam header.")
        kol_y = header.index(kol_y)
    else:
        kol_y = int(kol_y)

    kol_x = input("Pilih kolom untuk variabel X (indeks atau nama, pisahkan dengan koma): ").split(",")
    kol_x_final = []
    for x in kol_x:
        x = x.strip()
        if not x.isdigit():
            if x not in header:
                raise ValueError(f"Kolom '{x}' tidak ditemukan dalam header.")
            kol_x_final.append(header.index(x))
        else:
            kol_x_final.append(int(x))

    # buat matriks X dan vektor y
    X_data = [[row[i] for i in kol_x_final] for row in rows]
    y_data = [[row[kol_y]] for row in rows]

    return Matrix(X_data), Matrix(y_data)

