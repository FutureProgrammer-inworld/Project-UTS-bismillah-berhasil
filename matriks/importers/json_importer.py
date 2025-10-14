# matriks/importers/json_importer.py
import json
from matriks.matrix import Matrix

def import_from_json(nama_file):
    """Mengimpor data matriks dari file JSON (berformat list of lists)."""
    with open(nama_file, 'r') as f:
        data = json.load(f)

    # pastikan data berupa list of lists
    if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
        raise ValueError("Format JSON tidak valid, harus berupa list of lists.")

    print(f"Matriks berhasil diimpor dari {nama_file}")
    return Matrix(data)
