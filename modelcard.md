---
# MODEL CARD

# Model ID: Hydroponic_Tomato_Optimization_Model

<!-- Provide a quick summary of what the model is/does. -->

**Model Summary:**

This model aims to develop an AI system for optimizing hydroponic tomato cultivation by determining optimal nutrient uptake, water usage, and greenhouse gas emissions. It leverages data-driven insights to minimize waste and maximize efficiency, contributing to sustainable agriculture.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

{{ model_description | default("", true) }}

- **Developed by:** Gizem Yüksel
- **Model date:** 17.02.2024
- **Model type:** Machine Learning
- **Language(s):** Python


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

Direct Use: The model is directly used to optimize hydroponic tomato cultivation.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

The model may not provide accurate results at very high temperatures or extreme conditions because it does not have enough data.

### Recommendations

Users should be made aware of the risks, biases, and limitations of the model:

When entering ambient conditions values, entering values close to optimal values allows for a much more accurate result.

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

**Training features:**
EC limit, number of days after translation, temperature and water rate.


#### Training Hyperparameters

To refine the model, particularly for predicting CO2 and CH4 emissions accurately, we employed Randomized Grid Search for hyperparameter tuning. This method efficiently explores the hyperparameter space without exhaustive computation, leading to enhanced model precision in estimating greenhouse gas emissions.


## Evaluation

Utilizing the Random Forest algorithm and Randomized Grid Search significantly enhances hydroponic tomato cultivation's sustainability and efficiency. By optimizing nutrient compositions, establishing ideal electrical conductivity limits, and managing water usage, we minimize resource wastage and environmental impact. This project aligns technological advancements with environmental responsibility, showcasing a model for future sustainable farming practices.

## Testing Data, Factors & Metrics

**Metrics**

Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
R-squared (R²): Represents the proportion of variance in the target variable explained by the model.

- Mean Squared Error (MSE) for EC Limit Prediction: 0.023
- R-squared (R2) for EC Limit Prediction: 0.85
- Mean Squared Error (MSE) for Nutrient Concentrations Prediction: [0.015, 0.009, 0.012, 0.018, 0.022]
- R-squared (R2) for Nutrient Concentrations Prediction: [0.75, 0.89, 0.82, 0.79, 0.68]
- Mean Squared Error (MSE) for Water Consumption Prediction: 0.034
- R-squared (R2) for Water Consumption Prediction: 0.92
- Mean Squared Error (MSE) for N2O Emission Prediction: 0.008
- R-squared (R2) for N2O Emission Prediction: 0.94
- Mean Squared Error (MSE) for CO2 Emission Prediction: 0.012
- R-squared (R2) for CO2 Emission Prediction: 0.88
- Mean Squared Error (MSE) for CH4 Emission Prediction: 0.019
- R-squared (R2) for CH4 Emission Prediction: 0.80



