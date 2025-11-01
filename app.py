# app.py
from flask import Flask, render_template, request, jsonify, make_response
import json
import numpy as np
import io
import base64

# Menggunakan backend Matplotlib Agg (non-interaktif)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Impor semua modul Matriks
from matriks.matrix import Matrix
from matriks.operations.adder import add_matrices
from matriks.operations.multiplier import multiply_matrices
from matriks.operations.transpose import transpose
from matriks.operations.inverse import inverse
from matriks.statistic.correlation import correlation_matrix, visualize_correlation_matrix
from matriks.statistic.regression import (
    regresi_linier, prediksi, evaluasi, visualize_regression, pilih_variabel_xy_web
)
from matriks.importers.csv_importer import import_from_csv
from matriks.importers.json_importer import import_from_json
from matriks.importers.input_importer import import_from_input
from matriks.exporters.csv_exporter import export_to_csv
from matriks.exporters.json_exporter import export_to_json
from matriks.utilities import format_matrix_for_html, format_table_for_html

app = Flask(__name__)

# --- Fungsi Pembantu Matriks Global ---

def get_matrix_from_request(req_data, key="matrix_a"):
    """Membantu mendapatkan objek Matrix dari input user di web."""
    source = req_data.get(f'{key}_source')
    content = req_data.get(f'{key}_content')
    
    if not content:
        raise ValueError(f"Konten Matriks ({key}) tidak boleh kosong.")

    if source == 'manual':
        return import_from_input(content)
    elif source == 'csv':
        return import_from_csv(content)
    elif source == 'json':
        return import_from_json(content)
    else:
        raise ValueError("Sumber matriks tidak valid.")

def matrix_to_json_response(matrix):
    """Mengkonversi objek Matrix ke format JSON yang dapat dikirim ke frontend."""
    return {
        "header": getattr(matrix, 'header', [f"X{i+1}" for i in range(matrix.cols)]),
        "data": matrix.data,
        "rows": matrix.rows,
        "cols": matrix.cols,
        "html": format_matrix_for_html(matrix.data)
    }

# --- Routing Utama ---

@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoints Operasi Matriks ---

@app.route('/api/add', methods=['POST'])
def api_add():
    try:
        data = request.json
        A = get_matrix_from_request(data, key="matrix_a")
        B = get_matrix_from_request(data, key="matrix_b")
        
        result = add_matrices(A, B)
        return jsonify({
            "success": True,
            "result": matrix_to_json_response(result)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/multiply', methods=['POST'])
def api_multiply():
    try:
        data = request.json
        A = get_matrix_from_request(data, key="matrix_a")
        B = get_matrix_from_request(data, key="matrix_b")
        
        result = multiply_matrices(A, B)
        return jsonify({
            "success": True,
            "result": matrix_to_json_response(result)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/transpose', methods=['POST'])
def api_transpose():
    try:
        data = request.json
        A = get_matrix_from_request(data, key="matrix_a")
        
        result = transpose(A)
        return jsonify({
            "success": True,
            "result": matrix_to_json_response(result)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/inverse', methods=['POST'])
def api_inverse():
    try:
        data = request.json
        A = get_matrix_from_request(data, key="matrix_a")
        
        result = inverse(A)
        return jsonify({
            "success": True,
            "result": matrix_to_json_response(result)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# --- API Endpoint Analisis Statistik ---

@app.route('/api/correlation', methods=['POST'])
def api_correlation():
    try:
        data = request.json
        A = get_matrix_from_request(data, key="matrix_a")
        
        names, corr_mat = correlation_matrix(A)
        
        # Hitung Visualisasi
        plot_base64 = visualize_correlation_matrix(corr_mat.data, names)

        # Format tabel untuk HTML
        table_html = format_table_for_html(names, corr_mat.data)

        return jsonify({
            "success": True,
            "names": names,
            "matrix_html": table_html,
            "plot_base64": plot_base64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/regression', methods=['POST'])
def api_regression():
    try:
        data = request.json
        A = get_matrix_from_request(data, key="matrix_a")
        
        # Mendapatkan indeks kolom dari form web
        col_y_index = int(data.get('col_y_index'))
        # col_x_indices adalah string dipisahkan koma, konversi ke list of int
        col_x_indices = [int(i.strip()) for i in data.get('col_x_indices').split(',')]

        X, y, X_names, y_name = pilih_variabel_xy_web(A, col_y_index, col_x_indices)

        beta = regresi_linier(X, y)
        y_pred = prediksi(X, y.data)
        hasil_eval = evaluasi(y.data, y_pred.data)

        # Hitung Visualisasi
        plot_base64 = visualize_regression(X.data, y.data, y_pred.data, beta.data)

        # Format koefisien Beta untuk tampilan
        beta_names = ["Intercept"] + X_names 
        beta_html = '<h4 class="text-lg font-semibold mb-2">Koefisien Regresi (β)</h4>'
        beta_html += '<ul class="list-disc ml-5 space-y-1">'
        
        beta_html += f'<li><span class="font-bold">Intercept</span>: {beta.data[0][0]:.4f}</li>'
        for i in range(len(X_names)):
            beta_html += f'<li><span class="font-bold">{X_names[i]}</span>: {beta.data[i+1][0]:.4f}</li>'
        beta_html += '</ul>'

        return jsonify({
            "success": True,
            "beta_html": beta_html,
            "evaluation": {
                "R2": f"{hasil_eval['R2']:.4f}",
                "MSE": f"{hasil_eval['MSE']:.4f}",
                "SSE": f"{hasil_eval['SSE']:.4f}",
            },
            "plot_base64": plot_base64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    # Jalankan app di port 8040 jika dijalankan tanpa Docker (untuk testing)
    app.run(debug=True, port=8040)
