# matriks/utilities.py

def print_matrix(matrix):
    """
    Mencetak isi dari objek matriks.
    """
    for row in matrix.data:
        print(row)
        
def find_determinant(matrix):
    """
    Menghitung determinan dari sebuah matriks.
    Fungsi ini hanya bekerja untuk matriks 2x2.
    """
    # Periksa apakah matriks adalah matriks 2x2
    if matrix.rows != 2 or matrix.cols != 2:
        raise ValueError("Fungsi ini hanya mendukung matriks 2x2.")

    # Hitung determinan (ad - bc)
    determinant = (matrix.data[0][0] * matrix.data[1][1]) - (matrix.data[0][1] * matrix.data[1][0])
    return determinant

def is_square(matrix):
    """
    Memeriksa apakah sebuah matriks adalah matriks persegi.
    """
    return matrix.rows == matrix.cols

def is_symmetric(matrix):
    """
    Memeriksa apakah sebuah matriks adalah matriks simetris.
    Matriks simetris adalah matriks persegi yang sama dengan transposenya.
    """
    # Periksa apakah matriks adalah matriks persegi
    if not is_square(matrix):
        return False

    # Periksa apakah elemen (i, j) sama dengan elemen (j, i)
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            if matrix.data[i][j] != matrix.data[j][i]:
                return False

    return True
