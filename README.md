# Bitcoin Price Forecasting using Advanced Time Series Methods

### Author: Virendrasinh Chavda

<p align="justify">
This report delves into a detailed study of forecasting hourly Bitcoin prices using machine learning, deep learning, and hybrid time series models. Special emphasis is placed on novel methodologies such as Temporal Kolmogorov-Arnold Networks (TKAN) and decomposition-enhanced techniques. Through rigorous experimentation and model comparison, the project establishes new benchmarks for cryptocurrency price prediction accuracy.
</p>

---

## Table of Contents
1. [Overview](#overview)
2. [Data](#data)
3. [Features](#features)
4. [Visualizations](#visualizations)
5. [Model Details](#model-details)
6. [Training Process](#training-process)
7. [Residual Analysis](#residual-analysis)
8. [Results](#results)
9. [Future Work](#future-work)

---

## Overview

<p align="justify">
The volatility and unpredictability of Bitcoin prices make accurate forecasting a challenging yet valuable endeavor. This study evaluates traditional, machine learning, and deep learning techniques, with a specific focus on integrating time series decomposition methods like Empirical Mode Decomposition (EMD) and wavelet analysis. Performance metrics such as RMSE, MAPE, and R² are utilized to benchmark the models.
</p>

---

## Data

The data utilized in this study include:
- **Hourly Bitcoin Prices**: Extracted using the CryptoCompare API.
- **Google Trends Data**: Scraped for Bitcoin-related search terms.
- **Reddit Sentiment Analysis**: Collected and analyzed for social sentiment correlation.

Data preprocessing involved feature engineering, scaling, and stationarity testing, with exploratory data analysis confirming the presence of a random walk model in hourly Bitcoin prices.

---

## Features

### Key Techniques
1. **Decomposition Methods**:
   - Classical, Wavelet, and Empirical Mode Decomposition (EMD).
2. **Hybrid Models**:
   - Integration of traditional regression with deep learning (e.g., LSTM).
3. **Novel TKAN Architecture**:
   - Temporal Kolmogorov-Arnold Networks leveraging decomposition for time series component analysis.
4. **KAN Models**:
   - Kolmogorov-Arnold Networks used for predicting non-linear time series with spline-based activation functions.

---

## Visualizations

### Model Performance

#### SARIMAX Model
![SARIMAX Predictions](sarimax.png)

#### Seasonal OLS Hybrid Model
![Seasonal OLS](seasonal_ols.png)

#### CNN-LSTM Predictions
![CNN-LSTM](cnn_lstm.png)

#### CNN-LSTM-GAN
![CNN-LSTM-GAN](cnn_lstm_gan.png)

#### Empirical Mode Decomposition
![EMD Decomposition](emd.png)

#### EMD-LSTM Model
![EMD-LSTM Predictions](emd_lstm.png)

#### LSTM Autoencoder
![LSTM Autoencoder](lstm_autoenc.png)

---

## Model Details

### Models Used
1. **Traditional Models**:
   - ARIMA, SARIMA, and SARIMAX.
2. **Machine Learning Models**:
   - Support Vector Regression, Lasso Regression.
3. **Deep Learning Models**:
   - LSTM, CNN-LSTM, Autoencoders.
4. **Hybrid Models**:
   - EMD + LSTM, Wavelet + Regression, Seasonal OLS Hybrid.
5. **Generative Models**:
   - GANs for predictive augmentation.
6. **Kolmogorov-Arnold Networks (KAN)**:
   - Uses spline-based activation functions to capture non-linear trends in time series.
7. **Temporal Kolmogorov-Arnold Networks (TKAN)**:
   - Extends KAN by incorporating temporal dynamics using time series decomposition methods.

---

## Training Process

### Model-Specific Details
1. **Preprocessing**:
   - Robust scaling, outlier handling, and time lag adjustments.
2. **Hyperparameter Tuning**:
   - Grid search and validation metrics for optimizing learning rates, epochs, and architecture layers.
3. **Evaluation Metrics**:
   - RMSE (Root Mean Square Error)
   - MAPE (Mean Absolute Percentage Error)
   - R² (Coefficient of Determination)

---

## Results

<p align="justify">
The performance of various models used for Bitcoin price forecasting is summarized in the table below. Key metrics, including RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error), and R² (Coefficient of Determination), were used to evaluate the accuracy and robustness of the predictions. The analysis reveals clear differences in model performance, highlighting the superiority of hybrid and decomposition-based models for hourly Bitcoin price forecasting.
</p>

### Comparative Model Performance

| **Model**                  | **RMSE**  | **MAPE**  | **R²**  |
|----------------------------|-----------|-----------|---------|
| SARIMAX                   | 66.60     | 0.155     | 0.999   |
| Seasonal OLS Hybrid        | 4.72      | 0.011     | 1.000   |
| CNN-LSTM                  | 159.63    | 0.406     | 0.995   |
| CNN-LSTM-GAN              | 434.22    | 1.073     | 0.959   |
| EMD-LSTM                  | 444.41    | 1.150     | 0.959   |
| LSTM Autoencoder          | 74.16     | 0.190     | 0.999   |
| KAN                       | 101.00    | 0.230     | 0.998   |
| TKAN                      | 54.50     | 0.120     | 0.999   |

### Detailed Analysis of Results

1. **Traditional Models**:
   - The SARIMAX model achieved reasonably good performance, with an RMSE of 66.60 and an R² of 0.999. However, it struggled with capturing non-linear dynamics in the data.
   - Seasonal OLS Hybrid emerged as the most accurate model, achieving the best results with an RMSE of 4.72, a MAPE of 0.011, and an R² of 1.000. This success can be attributed to the decomposition of the time series into trend and residual components.

2. **Deep Learning Models**:
   - CNN-LSTM performed well but had a higher RMSE (159.63) compared to traditional models, indicating challenges in adapting to the high-frequency nature of the Bitcoin data.
   - CNN-LSTM-GAN performed poorly (RMSE: 434.22) as the generative augmentation approach failed to improve prediction accuracy. The model showed significant overfitting and instability during training.
   - EMD-LSTM, despite leveraging empirical mode decomposition, struggled with large RMSE and MAPE values, indicating limitations in capturing the data's high volatility.
   - LSTM Autoencoder demonstrated high accuracy (RMSE: 74.16, R²: 0.999), showcasing its ability to detect patterns, although not as effectively as the hybrid models.

3. **Hybrid Models**:
   - The hybrid approaches combining decomposition methods with regression or deep learning outperformed standalone models. Seasonal OLS Hybrid and TKAN (Temporal Kolmogorov-Arnold Networks) were particularly noteworthy:
     - TKAN integrated decomposition and advanced non-linear mapping, achieving RMSE: 54.50, MAPE: 0.120, and R²: 0.999.
     - Seasonal OLS Hybrid's simplistic yet effective approach excelled due to its decomposition of time series trends and residuals.

4. **Kolmogorov-Arnold Networks (KAN)**:
   - The KAN model effectively captured non-linear relationships in the time series, resulting in RMSE: 101.00, MAPE: 0.230, and R²: 0.998.
   - Its advanced spline activation functions made it competitive but not as strong as TKAN, which incorporated temporal dynamics.

### Residual Analysis for Seasonal OLS Hybrid

Residual analysis was performed to ensure the model met statistical assumptions for reliable predictions. Below are the findings:

- **Standardized Residuals**:
  - Residuals fluctuate around zero with no discernible patterns, supporting the independence assumption.
  ![Standardized Residuals](results1.png)

- **Histogram and Density Plot**:
  - The histogram with density overlay demonstrates a near-normal distribution of residuals, aligning with the normality assumption.
  ![Histogram and Density](results2.png)

- **Q-Q Plot**:
  - The Q-Q plot shows that residuals follow a normal distribution, with minor deviations at the tails.
  ![Q-Q Plot](results3.png)

- **Autocorrelation**:
  - The autocorrelation plot shows minimal autocorrelation in residuals beyond the first lag, indicating independence.
  ![Autocorrelation](results4.png)

---

<p align="justify">
In summary, the Seasonal OLS Hybrid model and TKAN demonstrated the best predictive performance for hourly Bitcoin price forecasting. While deep learning and generative models showed potential, their performance was inconsistent compared to hybrid models. The results emphasize the importance of integrating decomposition techniques and hybrid frameworks to capture the complexities of high-frequency time series data.
</p>
                  | 54.50     | 0.120     | 0.999   |

---

## Future Work

1. **Model Enhancements**:
   - Explore ensemble learning for hybrid models.
2. **Incorporate More Data Sources**:
   - Utilize real-time social media streams for sentiment integration.
3. **Real-time Deployment**:
   - Develop a streaming architecture for live Bitcoin price forecasting.

---

This comprehensive exploration into Bitcoin price forecasting combines innovative methodologies and rigorous evaluation, setting a high benchmark for future studies. The integration of decomposition methods and hybrid models demonstrates significant potential for real-world applications.
