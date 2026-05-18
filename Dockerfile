# 1. Gunakan base image Python yang ringan dan stabil
FROM python:3.12-slim

# 2. Atur env variable untuk mengoptimalkan kinerja Python di Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# 3. Tetapkan direktori kerja utama di dalam container
WORKDIR /app

# 4. Install dependensi sistem dasar jika diperlukan (seperti gcc/curl untuk healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Salin requirements.txt terlebih dahulu untuk efisiensi caching build Docker
COPY requirements.txt /app/

# 6. Install dependensi Python (tanpa menyimpan cache pip agar image tetap ramping)
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Salin seluruh isi folder project ke dalam container di /app/
# Catatan: Kita tidak perlu menyalin folder "dataset" mentah karena data yang dibutuhkan 
# untuk runtime (CSV & model) sudah berada di dalam folder ini (di data/ dan models/)
COPY . /app/

# 8. Buka port aplikasi (default Cloud Run adalah 8080)
EXPOSE 8080

# 9. Jalankan aplikasi FastAPI menggunakan python script
# PENTING: Jalankan dari folder /app agar path relatif ter-resolve dengan benar
CMD ["python", "app.py"]
