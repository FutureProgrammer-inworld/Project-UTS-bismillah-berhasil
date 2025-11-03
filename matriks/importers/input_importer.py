# matriks/importers/input_importer.py
from matriks.matrix import Matrix

def import_from_input(raw_input=None):
    """
    Mengimpor data matriks baik dari input web (string)
    maupun input manual di terminal.
    Format input:
        - Setiap baris dipisahkan dengan newline (\n)
        - Elemen tiap baris dipisahkan dengan spasi atau koma
    """
    # === MODE WEB ===
    if raw_input is not None:
        try:
            # Pisahkan baris input berdasarkan newline
            rows = raw_input.strip().split("\n")
            data = []

            for row in rows:
                # Pisahkan elemen dengan koma atau spasi
                elemen = [float(x) for x in row.replace(",", " ").split()]
                data.append(elemen)

            return Matrix(data)
        except Exception as e:
            raise ValueError(f"Input manual tidak valid: {e}")
   # === MODE TERMINAL ===
    """Membuat matriks berdasarkan input pengguna."""
    try:
        baris = int(input("Masukkan jumlah baris: "))
        kolom = int(input("Masukkan jumlah kolom: "))
    except ValueError:
        print("Input harus berupa angka!")
        return None

    data = []
    for i in range(baris):
        while True:
            baris_input = input(f"Masukkan elemen baris ke-{i+1} (pisahkan dengan spasi): ").strip()
            elemen = baris_input.split()

            if len(elemen) != kolom:
                print(f"Jumlah elemen harus {kolom}, silakan ulangi.")
                continue

            try:
                data.append([float(x) for x in elemen])
                break
            except ValueError:
                print("Pastikan semua elemen berupa angka.")
                continue

    print("\nMatriks berhasil dibuat!")
    for row in data:
        print(row)

    return Matrix(data)
