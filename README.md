<div align="center">
  
# 💻 Laptop Price Prediction (LPP)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### [🚀 View Live Demo](https://laptop-price-prediction-by-jsh.streamlit.app/)

An end-to-end Machine Learning project to predict laptop prices based on hardware specifications, featuring a fully trained Random Forest model and a premium web interface powered by Streamlit.

</div>

---

## 📝 Table of Contents
- [Live Demo](https://laptop-price-prediction-by-jsh.streamlit.app/)
- [About the Project](#-about-the-project)
- [Project Architecture](#-project-architecture)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#%EF%B8%8F-prerequisites)
- [Installation \& Setup](#%EF%B8%8F-installation--setup)
- [How to Use](#-how-to-use)
  - [Streamlit Web App (Recommended)](#-streamlit-web-app-recommended)
  - [Interactive Web App (Gradio)](#interactive-web-app-gradio)
- [Model Details \& Performance](#-model-details--performance)

---

## 📖 About the Project

The digital marketplace for laptops is vast and continually changing. The primary goal of this repository is to provide a robust machine learning pipeline capable of estimating the price of a laptop based on its specs such as the Processor, RAM, OS, Storage capacity, and customer ratings.

**Key Highlights:**
- **Data Preprocessing Pipeline:** Robust handling of missing continuous data (using median imputation) and encoding of categorical features via `LabelEncoder`.
- **Model Comparison:** Evaluated multiple regression models, finding the optimal balance of variance and bias.
- **Interactive UI:** A premium Streamlit interface for a professional user experience.
- **Legacy Support:** Gradio interface still available within the notebooks for development testing.

---

## 🏗 Project Architecture

1. **Data Ingestion:** Loads `laptops.csv` into a Pandas DataFrame.
2. **Data Cleaning:** Drops irrelevant identifiers (`Unnamed: 0`, `img_link`).
3. **Imputation:** Fills `NaN` values in `rating`, `no_of_ratings`, and `no_of_reviews` with median values.
4. **Encoding:** Transforms string categorical variables (`name`, `processor`, `ram`, `os`, `storage`) into numeric formats.
5. **Modeling:** Splits data into Train/Test sets and trains Regressor models.
6. **Deployment:** Exposes the winning model (Random Forest) via a premium Streamlit Web App.

---

## 📂 Repository Structure

```text
LPP/
├── assets/
│   ├── Colab screenshot.png       # View of the development environment
│   └── Gradio UI.png              # Preview of the prediction interface
├── data/
│   └── laptops.csv                # Raw dataset containing laptop specs and prices
├── notebooks/
│   └── LPP_final.ipynb            # Core Jupyter/Colab notebook with all logic
└── README.md                      # Project documentation (You are here)
```

---

## ⚙️ Prerequisites

To run this project locally, ensure you have the following installed:
- **Python** 3.8 or higher.
- **Jupyter Notebook** or **JupyterLab** (or alternatively, a Google account to use Google Colab).

---

## 🛠️ Installation & Setup

### Option 1: Using Google Colab (Recommended)
This requires zero local setup.
1. Download `LPP_final.ipynb` and `laptops.csv` from this repository.
2. Navigate to [Google Colab](https://colab.research.google.com/).
3. Click **File -> Upload notebook** and upload `LPP_final.ipynb`.
4. On the left sidebar, click the **Files** icon (folder) and upload `laptops.csv` into the Colab environment's current working directory.

### Option 2: Local Environment Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/RaidenX2905/LPP.git
   cd LPP
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install required dependencies:**
   You can install the necessary packages directly using pip:
   ```bash
   pip install pandas numpy scikit-learn gradio matplotlib seaborn jupyter
   ```

---

## 🚀 How to Use

1. Launch your Jupyter environment by running:
   ```bash
   jupyter notebook notebooks/LPP_final.ipynb
   ```
2. Ensure that the notebook is loading the `laptops.csv` file from the correct relative path (`../data/laptops.csv`). Adjust the path in the Pandas `read_csv` cell if necessary based on where your kernel is running.
3. **Run All Cells:** Go to `Cell -> Run All` (or `Runtime -> Run all` in Colab). This will re-train the models and evaluate them on the spot.

### 🌐 Streamlit Web App (Recommended)

Run the main application for a premium experience:
```bash
streamlit run app.py
```
Or visit the live link: [laptop-price-prediction-by-jsh.streamlit.app](https://laptop-price-prediction-by-jsh.streamlit.app/)

### Interactive Web App (Gradio)

At the very bottom of the notebook, a Gradio interface cell will execute.
- **Local URL:** A local link (e.g., `http://127.0.0.1:7860`) will be generated.
- **Public URL:** Gradio will also generate a temporary public `.gradio.live` link that you can share with others for 72 hours.

Simply input the laptop specifications into the dropdowns and text boxes, click **Submit**, and the Random Forest model will output the estimated price instantly.

---

## 📊 Model Details & Performance

During the development phase, three regression algorithms were benched:
1. **Linear Regression:** Struggled to capture the non-linear relationships between complex specs and pricing.
2. **Decision Tree Regressor:** Prone to overfitting on the training data.
3. **Random Forest Regressor:** The final selected model. By utilizing an ensemble of decision trees, it successfully generalized the data and achieved an $R^2$ Score of **~0.88**, indicating high confidence in its price estimations.
