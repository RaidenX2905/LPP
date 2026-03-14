# Laptop Price Prediction (LPP)

This repository contains a Google Colab notebook for a **Laptop Price Prediction** machine learning project. The primary goal of this project is to predict laptop prices based on various hardware specifications and ratings.

## Overview

The project involves end-to-end machine learning pipeline steps:
1. **Data Loading**: Using the `laptops.csv` dataset.
2. **Data Preprocessing**:
   - Dropping irrelevant columns such as `Unnamed: 0` and `img_link`.
   - Handling missing values: Replacing missing values in continuous features (`rating`, `no_of_ratings`, and `no_of_reviews`) with their respective medians.
   - Encoding categorical features: Applying `LabelEncoder` to categorical inputs like `name`, `processor`, `ram`, `os`, and `storage`.
3. **Model Training & Evaluation**:
   The dataset is split into training and testing sets. Three different regression models have been trained and evaluated:
   - **Linear Regression**
   - **Decision Tree Regressor**
   - **Random Forest Regressor**
4. **Results**:
   The **Random Forest Regressor** achieved the highest $R^2$ score (approximately 0.88) and and was chosen as the final model for predictions.
5. **Web Interface**:
   A **Gradio** interface has been integrated into the notebook to allow users to interactively predict new laptop prices using the trained Random Forest model.

## Repository Contents
- **`LPP_final.ipynb`**: The main Google Colab Jupyter Notebook containing all the code for data processing, model training, and the Gradio UI.
- **`laptops.csv`**: The dataset used for training the model.
- **`Colab screenshot.png`**: A screenshot showing the notebook environment.
- **`Gradio UI.png`**: A screenshot showcasing the interactive Gradio predictions interface.

## How to use
1. Open the `LPP_final.ipynb` in [Google Colab](https://colab.research.google.com/) or a local Jupyter Notebook environment.
2. Ensure you have the `laptops.csv` in the same directory (or upload it to your Colab session).
3. Run all the cells sequentially.
4. The final cell will launch the Gradio interface. You can interact with it directly in the notebook output or open it in a new browser tab using the generated public link.

## Dependencies
- pandas
- numpy
- scikit-learn
- gradio
- matplotlib / seaborn (for any visualizations within the notebook)
