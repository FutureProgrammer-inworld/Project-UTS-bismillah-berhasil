from flask import Flask, render_template, request, jsonify
import json
import csv
import io
import numpy as np
import base64
import pandas as pd
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
    regresi_linier, prediksi, evaluasi
)
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
def regression_auto():
    try:
        # --- Baca CSV dan ambil kolom numerik
        df = pd.read_csv(request.files['file']).select_dtypes(include='number')
        X_data = df.iloc[:, :-1].values.tolist()  # semua kolom kecuali terakhir
        y_data = df.iloc[:, -1].values.reshape(-1, 1).tolist()  # kolom terakhir = Y

        # --- Konversi ke Matrix
        X = Matrix(X_data)
        y = Matrix(y_data)

        # --- Jalankan regresi linier
        beta = regresi_linier(X, y)
        y_pred = prediksi(X, beta)
        hasil_eval = evaluasi(y.data, y_pred.data)

        # --- Plot scatter + garis regresi
        import io, base64, matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        y_actual = [row[0] for row in y.data]
        y_predict = [row[0] for row in y_pred.data]

        # Scatter plot Y aktual vs Y prediksi
        plt.scatter(y_actual, y_predict, color='blue', label='Prediksi')

        # Garis regresi (Y_actual vs Y_actual=Y_pred)
        min_y, max_y = min(y_actual), max(y_actual)
        plt.plot([min_y, max_y], [min_y, max_y], color='red', linestyle='--', label='Garis Ideal')
        
        plt.xlabel("Y Aktual")
        plt.ylabel("Y Prediksi")
        plt.title("Plot Regresi Linier")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # --- Hasil
        beta_html = "<pre>" + "\n".join([f"β{i} = {val[0]:.4f}" for i, val in enumerate(beta.data)]) + "</pre>"

        return jsonify({
            "success": True,
            "beta_html": beta_html,
            "evaluation": hasil_eval,
            "plot_base64": plot_base64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ================================
# Jalankan server
# ================================
if __name__ == '__main__':
    app.run(debug=True, port=8040)
