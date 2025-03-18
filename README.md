# PTC for AR

This project aims to predict comfort levels using physiological signals and SHAP (SHapley Additive exPlanations) for model interpretability.

## Project Structure

- `DL_PTC_AR.py`: Contains the deep learning model for predicting comfort levels.
- `SHAP_PTC_AR.py`: Contains the SHAP analysis for model interpretability.
- `data_PCT_AR.csv`: Dataset used for training and evaluation.
- `PCT_EDAandHRV.pth`: Saved model weights.

## Requirements

- Python 3.8+
- PyTorch
- pandas
- scikit-learn
- seaborn
- shap

## Dataset
The dataset is located in the project directory and is named data_PCT_AR.csv. It contains the following features:
- HF: High Frequency component of heart rate variability.
- LF: Low Frequency component of heart rate variability.
- SCR: Skin Conductance Response.
- SCL: Skin Conductance Level.
- SKT: Skin Temperature.
- BMI: Body Mass Index.
- AR: Indicates health status, where:
  - `0`: Healthy (No allergic rhinitis)
  - `1`: Allergic rhinitis patient
- Comfort:  The target variable representing the comfort level, which is the output the model aims to predict.

# Neural Network for Comfort Level Prediction

## Model Architecture
The model is a neural network with the following architecture:
- **Input Layer**: 8 features
- **Hidden Layers**: [403, 287]
- **Output Layer**: 5 classes
- **Activation Function**: ReLU
- **Dropout Rate**: 0.4
- **Batch Normalization**: Enabled

## SHAP Analysis
The SHAP analysis provides both global and local interpretability of the model. The following plots are generated:
- **Beeswarm Plots**: Display feature importance across all classes
- **Force Plot**: Illustrates the impact of each feature on a single prediction
- **Waterfall Plot**: Shows the contribution of each feature to a specific prediction

## Acknowledgements
This project utilizes the following libraries:
- **PyTorch** - For building and training the neural network
- **SHAP** - For explainability and feature importance analysis
- **scikit-learn** - For data preprocessing and evaluation metrics
- **seaborn** - For data visualization
