# Menggunakan image Python 3.11 versi slim untuk ukuran yang lebih kecil dan stabilitas
FROM python:3.11-slim

# Mengatur environment variable untuk Gunicorn agar log keluar secara real-time
ENV PYTHONUNBUFFERED 1

# Menginstal dependensi sistem yang diperlukan oleh Matplotlib dan dependensi lainnya
# Ini penting agar Matplotlib dan paket ilmiah lainnya dapat diinstal dengan benar.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpng-dev \
    libfreetype6-dev \
    pkg-config \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Mengatur direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt dan menginstal dependensi Python
# Langkah ini dipecah untuk memanfaatkan cache Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file aplikasi, termasuk app.py dan direktori templates
COPY . /app

# Port yang akan digunakan oleh aplikasi (Gunicorn)
EXPOSE 5000

# Perintah untuk menjalankan aplikasi menggunakan Gunicorn
# 'app:app' berarti menjalankan fungsi 'app' dari modul 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]