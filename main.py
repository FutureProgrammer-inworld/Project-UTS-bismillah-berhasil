# matriks/main.py
from matriks.matrix import Matrix
from matriks.operations.adder import add_matrices
from matriks.operations.multiplier import multiply_matrices
from matriks.operations.transpose import transpose
from matriks.operations.inverse import inverse
from matriks.statistic.correlation import mean, correlation, correlation_matrix
from matriks.statistic.regression import regresi_linier, prediksi, evaluasi
from matriks.statistic.correlation_visualization import plot_correlation_matrix
from matriks.statistic.regression_visualization import plot_regresi
from matriks.utilities import print_matrix
from matriks.exporters.csv_exporter import export_to_csv
from matriks.exporters.json_exporter import export_to_json
from matriks.importers.csv_importer import import_from_csv
from matriks.importers.json_importer import import_from_json
from matriks.importers.input_importer import import_from_input

def tampilkan_menu():
    print("\n=== MENU OPERASI MATRIKS ===")
    print("1. Penjumlahan matriks")
    print("2. Perkalian matriks")
    print("3. Transpose matriks")
    print("4. Invers matriks")
    print("5. Korelasi")
    print("6. Regresi Linier (OLS)")
    print("7. Ekspor matriks ke CSV")
    print("8. Ekspor matriks ke JSON")
    print("0. Keluar")

def pilih_matriks():
    """Memilih sumber matriks (manual / CSV / JSON)"""
    print("\nPilih sumber matriks:")
    print("1. Input manual")
    print("2. Impor dari CSV")
    print("3. Impor dari JSON")
    pilihan = input("Masukkan pilihan: ")

    if pilihan == "1":
        return import_from_input()
    elif pilihan == "2":
        path = input("Masukkan nama file CSV: ")
        return import_from_csv(path)
    elif pilihan == "3":
        path = input("Masukkan nama file JSON: ")
        return import_from_json(path)
    else:
        print("Pilihan tidak valid.")
        return None

if __name__ == "__main__":
    matriks_aktif = None

    while True:
        tampilkan_menu()
        pilihan = input("\nPilih menu (0-10): ")

        if pilihan == "0":
            print("Terima kasih! Program selesai.")
            break

        elif pilihan == "1":
            print("\n=== Penjumlahan Matriks ===")
            print("Masukkan matriks pertama:")
            A = pilih_matriks()
            print("Masukkan matriks kedua:")
            B = pilih_matriks()
            if A and B:
                hasil = add_matrices(A, B)
                print("Hasil penjumlahan:")
                print_matrix(hasil)

        elif pilihan == "2":
            print("\n=== Perkalian Matriks ===")
            print("Masukkan matriks pertama:")
            A = pilih_matriks()
            print("Masukkan matriks kedua:")
            B = pilih_matriks()
            if A and B:
                hasil = multiply_matrices(A, B)
                print("Hasil perkalian:")
                print_matrix(hasil)

        elif pilihan == "3":
            print("\n=== Transpose Matriks ===")
            print("Masukkan matriks:")
            matriks = pilih_matriks()
            hasil = transpose(matriks)
            print("Hasil transpose:")
            print_matrix(hasil)

        elif pilihan == "4":
            print("\n=== Invers Matriks ===")
            print("Masukkan matriks:")
            matriks = pilih_matriks()
            hasil = inverse(matriks)
            print("Hasil invers:")
            print_matrix(hasil)


        elif pilihan == "5":
            print("\n=== Korelasi Antar Semua Kolom (Correlation Matrix) ===")
            print("Pilih sumber data:")

            sumber = input("1. Input manual\n2. Impor dari CSV\n3. Impor dari JSON\nMasukkan pilihan: ")

            if sumber == "2":
                from matriks.importers.csv_importer import import_from_csv
                nama_file = input("Masukkan nama file CSV: ")
                data = import_from_csv(nama_file)
                print(f"Matriks berhasil diimpor dari {nama_file}")

                from matriks.statistic.correlation import correlation_matrix
                header, corr_mat = correlation_matrix(data.data)

                print("\n=== Tabel Korelasi ===")
                print("     " + "  ".join(f"{h[:6]:>8}" for h in header))
                for i, row in enumerate(corr_mat.data):
                    print(f"{header[i][:6]:>6} " + "  ".join(f"{val:8.3f}" for val in row))
                    plot_correlation_matrix(corr_mat, header)

        elif pilihan == "6":
            print("\n=== Regresi Linier (OLS) ===")
            print("Masukkan sumber data matriks (X dan Y akan dipilih dari sini):")
            data_matriks = pilih_matriks()

            if data_matriks:
                from matriks.statistic.regression import pilih_variabel_xy
                X, y = pilih_variabel_xy(data_matriks)

                beta = regresi_linier(X, y)
                print("\nKoefisien β:")
                print_matrix(beta)

                y_pred = prediksi(X, beta)
                print("\nPrediksi ŷ:")
                print_matrix(y_pred)

                hasil_eval = evaluasi(y.data, y_pred.data)
                print("\nEvaluasi Model:")
                print("  SSE =", hasil_eval["SSE"])
                print("  MSE =", hasil_eval["MSE"])
                print("  R²  =", hasil_eval["R2"])
                plot_regresi(X, y, beta, judul="Hubungan antara X dan Y (Regresi Linier)")

        elif pilihan == "7":
            if matriks_aktif:
                nama_file = input("Masukkan nama file CSV untuk ekspor: ")
                export_to_csv(matriks_aktif, nama_file)
            else:
                print("Belum ada matriks aktif untuk diekspor!")

        elif pilihan == "8":
            if matriks_aktif:
                nama_file = input("Masukkan nama file JSON untuk ekspor: ")
                export_to_json(matriks_aktif, nama_file)
            else:
                print("Belum ada matriks aktif untuk diekspor!")

        else:
            print("Pilihan tidak dikenal, silakan coba lagi.")

