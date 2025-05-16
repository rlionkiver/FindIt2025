**ENGLISH**

# Application Risk Assessment Model

## Overview

This notebook implements a machine learning model to assess the risk associated with applications, specifically focusing on the `coppaRisk` target variable.  The model performs several preprocessing steps, feature selection, and employs a Gradient Boosting Classifier as the baseline model.  Further experimentation with other models (XGBoost, LightGBM, CatBoost) is conducted to optimize performance.

## Dependencies

The following Python libraries are required to run this notebook:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn` (`sklearn`)
  * `imblearn`
  * `xgboost`
  * `lightgbm`
  * `catboost`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost lightgbm catboost
```

## Data

The notebook uses the following datasets:

  * `train.csv`:  The training dataset containing features used to predict `coppaRisk`.
  * `test.csv`: The test dataset for which `coppaRisk` predictions are made.
  * `target.csv`:  Contains the `coppaRisk` labels corresponding to the training data.

These datasets should be placed in a directory named `dataset_findIt/` relative to the notebook's location.

## Methodology

The methodology employed in this notebook can be broken down into the following key stages:

1.  **Data Loading and Preprocessing:**

      * Loads the datasets (`train.csv`, `test.csv`, `target.csv`).
      * Imputes missing values using the median for numerical columns and the most frequent value for categorical columns.
      * Applies logarithmic transformations to skewed numerical features (`downloads`, `size`, `ratingCount`).
      * Creates new features such as `downloads_per_rating`, `rating_ratio`, and `adSpent_sqrt`.
      * Handles outliers by clipping numerical features to a specified range (5th to 95th percentile, with a 3\*IQR threshold).

2.  **Feature Selection:**

      * Uses `SelectKBest` with the `f_classif` (ANOVA F-value) method to select the most relevant numerical features.
      * Prints feature importance scores and p-values.
      * Ensures consistency of features between the training and testing sets.

3.  **Data Preparation and Encoding:**

      * Splits the training data into training and validation sets (85/15 split, stratified).
      * Applies SMOTE (Synthetic Minority Oversampling Technique) and RandomUnderSampler to address class imbalance in the target variable.
      * Encodes categorical features using `LabelEncoder`.
      * Scales numerical features using `StandardScaler`.

4.  **Model Training and Evaluation:**

      * Trains a baseline `GradientBoostingClassifier`.
      * Evaluates the model's performance using metrics such as accuracy, F1-score, recall, and ROC AUC.
      * Tunes the classification threshold to optimize the F1-score.
      * Experiments with other models (`XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`) using `RandomizedSearchCV` for hyperparameter tuning.
      * Compares the performance of all models.

5.  **Prediction and Submission:**

      * Makes predictions on the test dataset using the best performing model.
      * Generates a submission file in the required format.

## Results

The notebook includes visualizations and performance metrics for each model, allowing for a clear comparison of their effectiveness.  The best performing model is selected based on the F1-score, and a detailed classification report is provided.

## How to Run

1.  Ensure all required libraries are installed.
2.  Place the datasets (`train.csv`, `test.csv`, `target.csv`) in the `dataset_findIt/` directory.
3.  Run the notebook cells sequentially.

-----

**BAHASA INDONESIA**

# Model Penilaian Risiko Aplikasi

## Ringkasan

Notebook ini mengimplementasikan model machine learning untuk menilai risiko yang terkait dengan aplikasi, khususnya berfokus pada variabel target `coppaRisk`. Model ini melakukan beberapa langkah prapemrosesan, pemilihan fitur, dan menggunakan Gradient Boosting Classifier sebagai model dasar. Eksperimen lebih lanjut dengan model lain (XGBoost, LightGBM, CatBoost) dilakukan untuk mengoptimalkan kinerja.

## Dependensi

Library Python berikut diperlukan untuk menjalankan notebook ini:

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `seaborn`
  * `scikit-learn` (`sklearn`)
  * `imblearn`
  * `xgboost`
  * `lightgbm`
  * `catboost`

Anda dapat menginstal library ini menggunakan pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost lightgbm catboost
```

## Data

Notebook ini menggunakan dataset berikut:

  * `train.csv`: Dataset pelatihan yang berisi fitur-fitur yang digunakan untuk memprediksi `coppaRisk`.
  * `test.csv`: Dataset pengujian yang akan diprediksi `coppaRisk`.
  * `target.csv`: Berisi label `coppaRisk` yang sesuai dengan data pelatihan.

Dataset ini harus ditempatkan dalam direktori bernama `dataset_findIt/` relatif terhadap lokasi notebook.

## Metodologi

Metodologi yang digunakan dalam notebook ini dapat diuraikan menjadi beberapa tahapan utama:

1.  **Pemuatan dan Prapemrosesan Data:**

      * Memuat dataset (`train.csv`, `test.csv`, `target.csv`).
      * Mengisi nilai yang hilang menggunakan median untuk kolom numerik dan nilai yang paling sering muncul untuk kolom kategorikal.
      * Menerapkan transformasi logaritmik pada fitur numerik yang miring (`downloads`, `size`, `ratingCount`).
      * Membuat fitur baru seperti `downloads_per_rating`, `rating_ratio`, dan `adSpent_sqrt`.
      * Menangani outlier dengan membatasi fitur numerik ke rentang tertentu (persentil ke-5 hingga ke-95, dengan ambang batas 3\*IQR).

2.  **Pemilihan Fitur:**

      * Menggunakan `SelectKBest` dengan metode `f_classif` (nilai F ANOVA) untuk memilih fitur numerik yang paling relevan.
      * Mencetak skor pentingnya fitur dan nilai p.
      * Memastikan konsistensi fitur antara set pelatihan dan pengujian.

3.  **Persiapan dan Encoding Data:**

      * Membagi data pelatihan menjadi set pelatihan dan validasi (split 85/15, stratified).
      * Menerapkan SMOTE (Synthetic Minority Oversampling Technique) dan RandomUnderSampler untuk mengatasi ketidakseimbangan kelas dalam variabel target.
      * Mengkodekan fitur kategorikal menggunakan `LabelEncoder`.
      * Menskalakan fitur numerik menggunakan `StandardScaler`.

4.  **Pelatihan dan Evaluasi Model:**

      * Melatih model dasar `GradientBoostingClassifier`.
      * Mengevaluasi kinerja model menggunakan metrik seperti akurasi, skor F1, recall, dan ROC AUC.
      * Menyesuaikan ambang batas klasifikasi untuk mengoptimalkan skor F1.
      * Melakukan eksperimen dengan model lain (`XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`) menggunakan `RandomizedSearchCV` untuk tuning hyperparameter.
      * Membandingkan kinerja semua model.

5.  **Prediksi dan Pengajuan:**

      * Membuat prediksi pada dataset pengujian menggunakan model dengan kinerja terbaik.
      * Menghasilkan file pengajuan dalam format yang diperlukan.

## Hasil

Notebook ini mencakup visualisasi dan metrik kinerja untuk setiap model, yang memungkinkan perbandingan efektivitasnya secara jelas. Model dengan kinerja terbaik dipilih berdasarkan skor F1, dan laporan klasifikasi terperinci disediakan.

## Cara Menjalankan

1.  Pastikan semua library yang diperlukan telah diinstal.
2.  Tempatkan dataset (`train.csv`, `test.csv`, `target.csv`) di direktori `dataset_findIt/`.
3.  Jalankan sel-sel notebook secara berurutan.
