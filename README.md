# E-Commerce Package QC — Computer Vision

Klasifikasi kondisi fisik paket dari foto ke dalam 3 kelas: **Aman**, **Rusak**, **Lainnya** (DATASCAPE 2026). Metrik: Macro-F1.

## Isi repo

| File | Isi |
|---|---|
| `DATASCAPE2026_Package_QC_Pipeline.ipynb` | Notebook pipeline lengkap: split → audit data → embedding CLIP + DINOv2 → MLP → CLIP multi-crop → ConvNeXt-tiny (2 putaran) → ensemble → `submission.csv` |
| `split/train.csv`, `split/val.csv` | Split stratified 85/15 (seed 42), path relatif terhadap folder data. Notebook membuat ulang file yang sama secara otomatis. |

## Cara menjalankan (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zhuhnn/-E-Commerce-Package-QC-Computer-Vision/blob/claude/quirky-darwin-kriudh/DATASCAPE2026_Package_QC_Pipeline.ipynb)

1. Buka notebook di Colab, pilih `Runtime → Change runtime type → GPU`.
2. `Runtime → Run all`, lalu izinkan akses Google Drive.
3. Dataset diunduh otomatis dari Hugging Face. Hasil kerja disimpan di `MyDrive/package-qc/`, jadi kalau sesi putus cukup **Run all** lagi: tahap yang sudah selesai dilewati dan training CNN lanjut dari epoch terakhir.
4. File akhir: `MyDrive/package-qc/submission.csv`.

Saklar penting di sel konfigurasi (§0.1):
- `RUN[...]`: menyalakan/mematikan tiap tahap. Eksperimen pembanding (zero-shot, LogReg, kNN, SVM, ViT-L/14, EfficientNet, YOLO) mati secara default.
- `USE_PSEUDO_OVERRIDE = False`: override label foto test yang identik dengan foto train. Nyalakan hanya bila aturan lomba mengizinkan.

## Dataset
Dataset ada di Hugging Face: https://huggingface.co/datasets/Zhuhn/Package-QC-dataset

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Zhuhn/Package-QC-dataset", repo_type="dataset", local_dir="data")
```
