# Deep Learning - Comparative Analysis of Generative Model

Repositori ini berisi implementasi PyTorch dan pipeline evaluasi terstandarisasi untuk proyek **Deep Learning (D6421) — Generative Models**. Proyek ini mencakup dua fokus utama:

1. **Pembuktian Teoretis dan Empiris Lima Tahap** mengenai ketidakmampuan bobot ($W^{(l)}$) dan fitur laten ($z_{\text{MLP}}$) dari model klasifikasi terawasi (*MLP Classifier*) untuk melakukan rekonstruksi presisi maupun sintesis citra baru.
2. **Studi Komparatif Empiris Enam Arsitektur Generatif** (*Vanilla Autoencoder*, *Sparse Autoencoder*, *Variational Autoencoder [VAE]*, *Vanilla GAN*, *Deep Convolutional GAN [DCGAN]*, dan *Denoising Diffusion Probabilistic Model [DDPM]*) pada dataset **Fashion-MNIST**.

## Daftar Isi

* [Struktur Repositori](#struktur-repositori)
* [Protokol Standardisasi Eksperimen](#protokol-standardisasi-eksperimen)
* [Ringkasan Arsitektur Model](#ringkasan-arsitektur-model)
* [Hasil Evaluasi Kuantitatif](#hasil-evaluasi-kuantitatif)
* [Instalasi dan Cara Menjalankan](#instalasi-dan-cara-menjalankan)
* [Kompilasi Visualisasi Utama](#kompilasi-visualisasi-utama)

## Struktur Repositori

```text
Comparative-Analysis-of-Generative-Model_FashionMNIST/
├── notebooks/
│   └── Comparative_Generative_Models_FashionMNIST.ipynb   # Notebook utama eksperimen end-to-end
├── src/                                                   # (Opsional) Modul skrip PyTorch terpisah
│   ├── dataset.py                                         # DataLoader, splitting 80:10:10, & normalisasi
│   ├── models.py                                          # Definisi MLP, AE, Sparse AE, VAE, GAN, DCGAN, DDPM
│   ├── metrics.py                                         # Evaluasi MSE, PSNR, SSIM, FID, & GPU Latency
│   └── visualize.py                                       # Generator 5 figur visualisasi komparatif
├── outputs/
│   └── figures/                                           # Hasil render resolusi tinggi Figur 1 s.d. Figur 5
├── requirements.txt                                       # Daftar dependensi pustaka Python
└── README.md                                              # Dokumentasi proyek
```

## Protokol Standardisasi Eksperimen

Untuk memastikan perbedaan performa murni mencerminkan *inductive bias* dan perumusan matematis tiap model, seluruh eksperimen dijalankan di bawah konfigurasi berikut:

* **Dataset:** `torchvision.datasets.FashionMNIST` ($70.000$ citra *grayscale* berukuran $1 \times 28 \times 28$, $10$ kelas pakaian).
* **Pembagian Dataset (*Split*):** **80% Train** ($56.000$ citra) : **10% Validation** ($7.000$ citra) : **10% Test** ($7.000$ citra). Seluruh metrik akhir dievaluasi secara eksklusif pada *Test Set*.
* **Konfigurasi Batch & Reproduktibilitas:** `batch_size = 128` (`shuffle=True` pada Train, `shuffle=False` pada Val/Test) dan `batch_size = 64` untuk pengukuran *inference latency*, dengan penguncian *random seed* tetap (`torch.manual_seed(42)` dan `np.random.seed(42)`).
* **Normalisasi Piksel:** Rentang `[0, 1]` untuk *MLP Classifier* dan keluarga *Autoencoder* (aktivasi output `Sigmoid`), serta rentang `[-1, 1]` ($x' = 2x - 1$) untuk keluarga *GAN* dan *DDPM* (aktivasi output `Tanh` / *zero-mean Gaussian noise*).

## Ringkasan Arsitektur Model

| Model | Paradigma | Struktur Utama & Dimensi Laten | Fungsi Loss Objektif | Epoch |
| :--- | :--- | :--- | :--- | :---: |
| **MLP Classifier** | Diskriminatif ($P(y \mid x)$) | `784 -> 256 -> 64 -> 10` (`ReLU`) | *Cross-Entropy Loss* | 20 |
| **Vanilla AE** | Rekonstruktif Deterministik | `784 -> 256 -> 64 -> 256 -> 784` (`ReLU`, `Sigmoid`) | *Pixel-wise MSE Loss* | 25 |
| **Sparse AE** | Rekonstruktif + *Sparsity* | `784 -> 256 -> 784` (`ReLU` + penalti $L_1$, `Sigmoid`) | $\text{MSE} + \lambda \Vert h \Vert_1$ | 25 |
| **VAE** | Generatif Probabilistik | Encoder `784->256->(64,64)`, Decoder `64->256->784` | *ELBO* ($\text{MSE} + D_{\text{KL}}$) | 25 |
| **Vanilla GAN** | Generatif Adversarial (MLP) | G: `z -> 256 -> 512 -> 784`, D: `784 -> 512 -> 256 -> 1` | *Minimax BCE Loss* | 30 |
| **DCGAN** | Generatif Konvolusional | G: `ConvTranspose2d` (`z=100`), D: `Conv2d` + `BatchNorm2d` | *Minimax BCE Loss* | 30 |
| **DDPM** | Generatif Difusi Iteratif | U-Net (`64-128-256`) + *Sinusoidal Time Embedding* ($T=300$) | *Noise Prediction MSE* | 20 |

## Hasil Evaluasi Kuantitatif

### 1. Pembuktian Batasan Fitur Laten *MLP Classifier* (Test Acc: `88.50%`)

| Model Rekonstruksi | Sumber Representasi Laten | Test MSE ($\downarrow$) | Test PSNR ($\uparrow$) | Test SSIM ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: |
| **Frozen MLP Decoder** | $z_{\text{MLP}}$ (diskriminatif, beku) | `0.017486` | `17.57 dB` | `0.7087` |
| **Vanilla Autoencoder** | $z_{\text{AE}}$ (rekonstruktif) | `0.008250` | `20.84 dB` | `0.8344` |
| **Sparse Autoencoder** | $z_{\text{SAE}}$ (rekonstruktif + *sparsity*) | **`0.003709`** | **`24.31 dB`** | **`0.8935`** |
| **Variational Autoencoder (VAE)** | $z_{\text{VAE}}$ (probabilistik) | `0.017678` | `17.53 dB` | `0.6947` |

### 2. Komparasi Menyeluruh Keenam Model Generatif (*Test Set*)

| Arsitektur Model | Dimensi Laten | Training Time / Epoch (s) | Latency (ms / batch 64) | Test MSE ($\downarrow$) | Test PSNR (dB) ($\uparrow$) | Test SSIM ($\uparrow$) | FID Score ($\downarrow$) | Kestabilan Training |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Vanilla Autoencoder** | `64` | 7.50 | 0.27 | 0.008250 | 20.84 | 0.8344 | N/A (*Latent Voids*) | Stabil / Konvergen |
| **Sparse Autoencoder** | `64` / `256` | **6.73** | **0.16** | **0.003709** | **24.31** | **0.8935** | N/A (*Latent Voids*) | Stabil / Konvergen |
| **Variational Autoencoder (VAE)** | `64` | 7.79 | **0.16** | 0.017678 | 17.53 | 0.6947 | 101.95 | Stabil / Konvergen |
| **Vanilla GAN (MLP)** | `100` | 8.64 | 0.19 | N/A | N/A | N/A | 149.70 | Sangat Rentan Osilasi / *Mode Collapse* |
| **Deep Convolutional GAN (DCGAN)** | `100` | 17.42 | 7.84 | N/A | N/A | N/A | **40.77** | Cukup Stabil (Sensitif Arsitektur) |
| **Denoising Diffusion (DDPM)** | $28 \times 28$ | 52.89 | 5108.56 | N/A | N/A | N/A | 50.22 | Sangat Stabil |

## Instalasi dan Cara Menjalankan

### 1. Kloning Repositori & Instalasi Dependensi

```bash
git clone https://github.com/Mycticount-X/Comparative-Analysis-of-Generative-Model_FashionMNIST.git
cd Comparative-Analysis-of-Generative-Model_FashionMNIST

# Buat virtual environment (disarankan)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Instal seluruh pustaka yang diperlukan
pip install -r requirements.txt
```

*(Catatan: Jika menggunakan GPU NVIDIA lokal, instal `torch` dan `torchvision` sesuai versi CUDA Anda terlebih dahulu sebelum menjalankan `pip install -r requirements.txt`.)*

### 2. Eksekusi Eksperimen

* **Via Google Colab / Jupyter Notebook:**
  Buka berkas `notebooks/Comparative_Generative_Models_FashionMNIST.ipynb`, pilih runtime **GPU (T4 atau lebih tinggi)**, lalu jalankan seluruh sel secara berurutan (*Run All*). Dataset Fashion-MNIST akan diunduh otomatis melalui `torchvision.datasets`.
* **Via Skrip Modular (jika menjalankan folder `src/`):**
  ```bash
  python src/train_all.py --seed 42 --batch_size 128
  python src/evaluate.py --latency_batch 64 --fid_samples 1024
  ```

## Kompilasi Visualisasi Utama

Eksekusi pipeline menghasilkan lima figur komparatif utama di direktori `outputs/figures/`:

1. **Figur 1 — Grid Rekonstruksi Side-by-Side:** Perbandingan 8 citra *Ground Truth* terhadap rekonstruksi Vanilla AE, Sparse AE, dan VAE.
2. **Figur 2 — Grid Random Generative Sampling ($4 \times 4$):** Sampel citra sintetis baru dari prior $z \sim \mathcal{N}(0, I)$ untuk VAE, Vanilla GAN, DCGAN, dan DDPM.
3. **Figur 3 — Bukti Empiris Batasan Bobot & Fitur Laten MLP Classifier:** Proyeksi balik transpos bobot ($W^T$), rekonstruksi *Frozen Decoder* ($z_{\text{MLP}}$ vs. $z_{\text{AE}}$), serta uji *prior sampling* ruang laten diskriminatif.
4. **Figur 4 — Linear Latent Space Interpolation:** Transisi morfologi $z(\alpha) = (1-\alpha)z_A + \alpha z_B$ untuk $\alpha \in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]$ antarmodel.
5. **Figur 5 — Training Loss Curves & DDPM Denoising Trajectory:** Kurva konvergensi seluruh model beserta visualisasi *reverse diffusion* langkah demi langkah ($x_T \to x_0$).