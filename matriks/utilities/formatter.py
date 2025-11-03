# matriks/utilities/formatter.py
def to_string(matrix):
    """Mengubah matriks menjadi string dengan format baris-kolom."""
    result = []
    for row in matrix.data:
        result.append(" ".join(map(str, row)))
    return "\n".join(result) # Perbaikan: Gunakan join untuk format yang benar

# matriks/utilities/formatter.py

def format_matrix_for_html(matrix_data):
    """
    Mengonversi data matriks (list of lists) menjadi HTML table string.
    """
    html = '<table class="table table-bordered table-sm">'
    for row in matrix_data:
        html += '<tr>'
        for val in row:
            html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</table>'
    return html


def format_table_for_html(headers, data):
    """
    Mengonversi tabel dengan header ke HTML table string.
    Cocok untuk menampilkan hasil korelasi.
    """
    html = '<table class="table table-bordered table-sm">'
    
    # Header
    html += '<thead><tr><th></th>'
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead>'
    
    # Body
    html += '<tbody>'
    for i, row in enumerate(data):
        html += f'<tr><th>{headers[i]}</th>'
        for val in row:
            html += f'<td>{val:.4f}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    
    return html
