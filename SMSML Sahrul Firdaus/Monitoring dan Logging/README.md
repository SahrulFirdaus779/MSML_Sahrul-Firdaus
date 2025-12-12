# Monitoring dan Logging

Struktur ini menyiapkan servis inference + metriks Prometheus dan panduan bukti yang diminta rubric. Pastikan dashboard Grafana memakai nama akun Dicoding Anda.

## Cara cepat menjalankan stack lokal
1. Aktifkan environment yang berisi dependensi (lihat `Membangun_model/requirements.txt`).
2. Jalankan servis inference (menyediakan `/predict`, `/health`, `/metrics`):
   ```powershell
   cd "c:\MSML_Sahrul Firdaus\Monitoring dan Logging"
   python inference.py --model-path "..\Membangun_model\tuning_artifacts\rf_grid_model.joblib" --preprocess-path "..\Eksperimen_SML_Sahrul-Firdaus.-main\preprocessing\data_clustering_preprocessing\preprocess_pipeline.joblib" --host 0.0.0.0 --port 8000
   ```
3. Jalankan Prometheus untuk men-scrape inference + exporter:
   ```powershell
   prometheus --config.file=prometheus.yml
   ```
4. Jalankan Grafana, buat data source Prometheus (URL `http://localhost:9090`) lalu bangun dashboard bernama akun Dicoding Anda. Tambahkan panel untuk metriks yang sama dengan Prometheus.
5. Tambah rules alerting di Grafana (misal latency tinggi atau error rate > 1%).
6. Simpan tangkapan layar ke folder bukti sesuai rubric.

## Isi penting
- `inference.py` : FastAPI servis inference + endpoint `/metrics` dengan metriks:
  - `inference_requests_total` (labels: endpoint, status)
  - `inference_request_latency_seconds` (histogram)
  - `prediction_class_total` (per kelas)
  - `prediction_confidence` (histogram dari probabilitas max)
  - `preprocess_duration_seconds`, `inference_duration_seconds`
  - `inference_errors_total` untuk error.
- `prometheus_exporter.py` : Exporter ringan (CPU/mem + umur model/pipeline) di port 9101.
- `prometheus.yml` : Job scrape untuk inference (8000) dan exporter (9101).
- `1.bukti_serving` s.d. `6.bukti alerting Grafana` : tempat menyimpan screenshot sesuai urutan rubric.

## Catatan
- Path default diasumsikan dijalankan dari folder ini. Sesuaikan jika memindahkan artefak.
- Jika belum ada, tambahkan paket `fastapi`, `uvicorn`, dan `prometheus_client` ke environment. 
