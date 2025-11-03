from flask import Flask, render_template, request, jsonify
import json
import csv
import io
import numpy as np
import base64

# Matplotlib non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import modul matriks
from matriks.matrix import Matrix
from matriks.operations.adder import add_matrices
from matriks.operations.multiplier import multiply_matrices
from matriks.operations.transpose import transpose
from matriks.operations.inverse import inverse
from matriks.statistic.regression import (
    regresi_linier, prediksi, evaluasi, pilih_variabel_xy
)
from matriks.statistic.regression_visualization import plot_regresi
from matriks.utilities.formatter import format_matrix_for_html

app = Flask(__name__)

# ================================
# Helper: membaca konten CSV/Manual/JSON dari request
# ================================
def get_matrix_from_request(req_data, key="matrix_a"):
    source = (req_data.get(f'{key}_source') or '').lower()
    content = req_data.get(f'{key}_content', '')

    # Jika file upload (HTML form multipart)
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename:
            content = file.read().decode('utf-8')
            source = 'csv'

    content = content.strip()
    if not content:
        raise ValueError(f"Konten Matriks ({key}) tidak boleh kosong.")

    # MANUAL
    if source == 'manual':
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        data = []
        for line in lines:
            line = line.replace(',', ' ')
            parts = [p for p in line.split() if p]
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                raise ValueError(f"Format angka salah di baris: '{line}'")
        if len({len(r) for r in data}) != 1:
            raise ValueError("Semua baris harus memiliki jumlah kolom sama.")
        return Matrix(data)

    # CSV
    elif source == 'csv':
        f = io.StringIO(content)
        reader = csv.reader(f)
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("CSV kosong atau tidak valid.")

        first = rows[0]
        header = None
        numeric_first_row = True
        try:
            [float(x) for x in first]
        except Exception:
            numeric_first_row = False

        if numeric_first_row:
            numeric_rows = rows
        else:
            header = [h.strip() for h in first]
            numeric_rows = rows[1:]

        data = []
        for r in numeric_rows:
            try:
                row = [float(x) for x in r if x.strip() != ""]
                data.append(row)
            except ValueError:
                raise ValueError("CSV berisi nilai non-numerik pada baris data.")

        if len({len(r) for r in data}) != 1:
            raise ValueError("Semua baris CSV harus punya jumlah kolom sama.")

        m = Matrix(data)
        if header:
            setattr(m, 'header', header)
        return m

    # JSON
    elif source == 'json':
        try:
            parsed = json.loads(content)
        except Exception:
            raise ValueError("JSON tidak valid.")
        if isinstance(parsed, dict) and 'data' in parsed:
            data = parsed['data']
            m = Matrix([[float(x) for x in row] for row in data])
            if 'header' in parsed:
                setattr(m, 'header', parsed['header'])
            return m
        elif isinstance(parsed, list):
            return Matrix([[float(x) for x in row] for row in parsed])
        else:
            raise ValueError("Format JSON tidak dikenali.")
    else:
        raise ValueError("Sumber matriks tidak valid (manual/csv/json).")


# ================================
# Convert Matrix ke JSON
# ================================
def matrix_to_json_response(matrix):
    return {
        "header": getattr(matrix, 'header', [f"X{i+1}" for i in range(matrix.cols)]),
        "data": matrix.data,
        "rows": matrix.rows,
        "cols": matrix.cols,
        "html": format_matrix_for_html(matrix.data)
    }


# ================================
# ROUTES
# ================================
@app.route('/')
def index():
    return render_template('index.html')


# --- OPERASI MATRIKS ---
@app.route('/api/add', methods=['POST'])
def api_add():
    try:
        data = request.form.to_dict() if request.form else request.json
        A = get_matrix_from_request(data, key="matrix_a")
        B = get_matrix_from_request(data, key="matrix_b")
        result = add_matrices(A, B)
        return jsonify({"success": True, "result": matrix_to_json_response(result)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/multiply', methods=['POST'])
def api_multiply():
    try:
        data = request.form.to_dict() if request.form else request.json
        A = get_matrix_from_request(data, key="matrix_a")
        B = get_matrix_from_request(data, key="matrix_b")
        result = multiply_matrices(A, B)
        return jsonify({"success": True, "result": matrix_to_json_response(result)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/transpose', methods=['POST'])
def api_transpose():
    try:
        data = request.form.to_dict() if request.form else request.json
        A = get_matrix_from_request(data, key="matrix_a")
        result = transpose(A)
        return jsonify({"success": True, "result": matrix_to_json_response(result)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/inverse', methods=['POST'])
def api_inverse():
    try:
        data = request.form.to_dict() if request.form else request.json
        A = get_matrix_from_request(data, key="matrix_a")
        result = inverse(A)
        return jsonify({"success": True, "result": matrix_to_json_response(result)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# --- REGRESI LINIER ---
@app.route('/api/regression', methods=['POST'])
def api_regression():
    try:
        # Jika file diupload dari form
        if 'file' in request.files:
            file = request.files['file']
            content = file.read().decode('utf-8')
            # tambahkan matrix_a_content agar bisa diproses
            data = {
                "matrix_a_source": "csv",
                "matrix_a_content": content,  # <-- penting!
                "col_y_index": request.form.get("col_y_index"),
                "col_x_indices": request.form.get("col_x_indices")
            }
        else:
            data = request.json

        # ✅ pastikan ada konten
        if not data.get("matrix_a_content"):
            raise ValueError("File CSV tidak terbaca atau kosong.")

        # --- proses regresi
        A = get_matrix_from_request(data, key="matrix_a")
        col_y_index = int(data.get('col_y_index'))
        col_x_indices = [
            int(i.strip())
            for i in data.get('col_x_indices', '').split(',')
            if i.strip() != ''
        ]

        X, y, X_names, y_name = pilih_variabel_xy(A, col_y_index, col_x_indices)
        beta = regresi_linier(X, y)
        y_pred = prediksi(X, y.data)
        hasil_eval = evaluasi(y.data, y_pred.data)
        plot_base64 = plot_regresi(X.data, y.data, y_pred.data, beta.data)

        # --- tampilkan hasil
        beta_html = '<h4 class="text-lg font-semibold mb-2">Koefisien Regresi (β)</h4>'
        beta_html += '<ul class="list-disc ml-5 space-y-1">'
        beta_html += f'<li><b>Intercept</b>: {beta.data[0][0]:.4f}</li>'
        for i, name in enumerate(X_names):
            beta_html += f'<li><b>{name}</b>: {beta.data[i+1][0]:.4f}</li>'
        beta_html += '</ul>'

        return jsonify({
            "success": True,
            "beta_html": beta_html,
            "evaluation": {
                "R2": f"{hasil_eval['R2']:.4f}",
                "MSE": f"{hasil_eval['MSE']:.4f}",
                "SSE": f"{hasil_eval['SSE']:.4f}"
            },
            "plot_base64": plot_base64
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400



# ================================
# Jalankan server
# ================================
if __name__ == '__main__':
    app.run(debug=True, port=8040)
