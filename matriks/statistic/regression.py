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
def pilih_variabel_xy(matrix_data, col_y_index=None, col_x_indices=None):
    """
    Fungsi fleksibel untuk memilih variabel X dan Y.
    - Jika col_y_index dan col_x_indices dikirim (mode web/backend), langsung pakai itu.
    - Jika tidak dikirim (mode CLI), maka minta input manual dari pengguna.
    """

    header = getattr(matrix_data, 'header', [f"X{i+1}" for i in range(matrix_data.cols)])

    # ---------------------------
    # Mode CLI (interaktif)
    # ---------------------------
    if col_y_index is None or col_x_indices is None:
        print("=== Pilih Variabel X dan Y ===")
        print(f"Terdapat {len(header)} kolom (indeks 0–{len(header)-1})")
        print("Header:", header)
        print("\nContoh 5 baris pertama:")
        for i, row in enumerate(matrix_data.data[:5]):
            print(f"{i+1}: {row}")

        kol_y = input("Pilih kolom untuk variabel Y (indeks atau nama): ")
        if not kol_y.isdigit():
            if kol_y not in header:
                raise ValueError(f"Kolom '{kol_y}' tidak ditemukan dalam header.")
            col_y_index = header.index(kol_y)
        else:
            col_y_index = int(kol_y)

        kol_x = input("Pilih kolom untuk variabel X (indeks atau nama, pisahkan dengan koma): ").split(",")
        col_x_indices = []
        for x in kol_x:
            x = x.strip()
            if not x.isdigit():
                if x not in header:
                    raise ValueError(f"Kolom '{x}' tidak ditemukan dalam header.")
                col_x_indices.append(header.index(x))
            else:
                col_x_indices.append(int(x))

    # ---------------------------
    # Ambil kolom sesuai indeks
    # ---------------------------
    y_data = [[row[col_y_index]] for row in matrix_data.data]
    y = Matrix(y_data)
    y_name = header[col_y_index] if col_y_index < len(header) else f"Kolom {col_y_index}"

    X_data = [[row[i] for i in col_x_indices] for row in matrix_data.data]
    X = Matrix(X_data)
    X_names = [header[i] if i < len(header) else f"Kolom {i}" for i in col_x_indices]

    return X, y, X_names, y_name


