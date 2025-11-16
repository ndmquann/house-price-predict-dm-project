# Data Mining Application: House Price Prediction (CO3029)  🏡 
This is the project for the **Data Mining (CO3029)** course at **Ho Chi Minh City University of Technology VNU**. This project focuces on building a regression model to predict housing price using well-knowned **Ames Housing Dataset** from Kaggle.

# Team member and Responsibilities
| No. | Full Name | Student ID | Responsibilities |
| :--- | :--- | :--- | :--- |
| 1 | Nguyễn Phan Duy Bảo | 2210244 | Data Preprocessing, Feature Engineering, Report Writing |
| 2 | Phan Lê Hiếu | 2211006 | Model Training & Evaluation, Report Writing |
| 3 | Nguyễn Đỗ Minh Quân | 2312833 | Data Preprocessing, Feature Engineering, Report Compilation, Slide Design |

# Project Introduction

**1. Motivation**
   - **High Practicality:** Predicting real estate prices is a problem of great practical value, influencing the decisions of buyers, sellers, and financial institutions.
   - **Classic KDD Problem:** This is a benchmark dataset on Kaggle that presents many typical KDD challenges.
   - **Appropriate Complexity:** The dataset contains 79 features (both quantitative and qualitative), many missing values, and complex relationships, requiring the application of in-depth KDD techniques.

**2. Problem Objective**
   - To build a predictive model capable of accurately estimating the final sale price (SalePrice) of a house based on its descriptive features.

**3. Dataset**
   - **Source:** [Ames Housing Dataset (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
   - **Scale:** 1460 training samples and 1459 test samples.
   - **Features:** 79 features (numerical and categorical) describing homes in Ames, Iowa.
   - **Target Variable:** SalePrice.

# Data Mining Workflow
Our project followed a systematic KDD pipeline:

**1. Data Preprocessing**
- **Handling Missing Values:**
  - Filled "None" for categorical features where NA means the absence of that feature (e.g., Alley, PoolQC, FireplaceQu).
  - Filled 0 for numerical features where NA meant "none" (e.g., BsmtFinSF1, GarageArea).
  - Imputed LotFrontage using the mean value grouped by Neighborhood.
  - Imputed remaining categorical features (e.g., MSZoning, KitchenQual) with their mode.
 
- **Handling Skewness:**
  - The target variable SalePrice was heavily right-skewed. We applied a log transformation (np.log1p) to normalize its distribution.
  - Applied the Yeo-Johnson transformation to other skewed numerical features (e.g., 1stFlrSF).
    
- **Data Type Conversion:**
  - Converted numerical features that are actually categorical into strings (e.g., MSSubClass, MoSold).
  - Encoded ordinal features (e.g., ExterQual) into numerical values based on their rank (e.g., "Po" -> 1, "Fa" -> 2, ...).
 
**2. Feature Engineering**

We generated new features to improve model performance:
- **Feature Simplification:**
  - Grouped ordinal features into simpler categories (e.g., OverallQual 1-10 was grouped into "bad", "average", "good").
  - Created binary flags for presence/absence (e.g., haspool, hasgarage, hasbsmt).
 
- **Feature Combination:** Created new, more meaningful composite features:
  - TotalBath: Sum of all bathrooms.
  - AllSF: Total square footage (GrLivArea + TotalBsmtSF).
  - HouseAge: YrSold - YearBuilt.

- **Polynomial Features:**
  - To capture non-linear relationships, we created 2nd-degree, 3rd-degree, and square-root features for the top 10 features most correlated with SalePrice (e.g., OverallQual-s2, GrLivArea-3).

**3. Encoding & Scaling**

- One-Hot Encoding: Used pd.get_dummies to convert all remaining nominal categorical features into dummy variables.
- Scaling: Applied RobustScaler to the entire feature set. This was chosen for its robustness to outliers, which were present in the data.

**4. Models Applied**

Given the high dimensionality of the data after encoding (which can lead to Overfitting and Multicollinearity), we used Regularized Linear Regression models.
1. **Lasso Regression (L1 Regularization):** Performs automatic feature selection by shrinking irrelevant feature weights to zero.
2. **Ridge Regression (L2 Regularization):** Effectively handles multicollinearity by shrinking coefficients.
3. **ElasticNet (L1 + L2 Hybrid):** A combination that balances the benefits of both Lasso and Ridge.
   
We used the Cross-Validation versions of each model (e.g., LassoCV) to automatically tune their hyperparameters.

**5. Evaluation Metric**

The primary metric used was the Root Mean Squared Logarithmic Error (RMSLE).
- This is the official metric for the Kaggle competition.
- It measures the relative error (percentage) rather than the absolute error.
- It is less sensitive to outliers, which is ideal for this dataset.
  
$$RMSLE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(log(p_{i}+1)-log(a_{i}+1))^{2}}$$

# Experimental Results
We ran two experiments: **(1) Without Feature Engineering (FE)** and **(2) With FE**.

**Experiment 1: Without Feature Engineering**

| Model | RMSE (10-folds) | Std (10-folds) | Best Alpha | Features Dropped | Public Score (Kaggle) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Lasso | 0.121788 | 0.038904 | 0.001 | 199 / 268 | 0.12449 | 
| Ridge | 0.131615 | 0.035006 | 0.01 | 0 / 268 | 0.13613 | 
| ElasticNet | 0.119080 | 0.038372 | 0.001 | 161 / 268 | 0.12462 |

**Experiment 2: With Feature Engineering**
- **Improvement:** Feature Engineering successfully improved the performance of all models.
- **Total Features after FE:** 345.

| Model | RMSE (10-folds) | Std (10-folds) | Best Alpha | Features Dropped | Public Score (Kaggle) | 
| :--- | :--- | :--- | :--- | :--- | :--- |
| Lasso | 0.120397 | 0.046050 | 0.001 | 252 / 345 | 0.12090 | 
| Ridge | 0.133522 | 0.034551 | 0.005 | 0 / 345 | 0.13588 | 
| ElasticNet | 0.119105 | 0.043043 | 0.001 | 218 / 345 | 0.12095 | 

# Conclusion
1. **Best Model: Lasso Regression (with Feature Engineering)** achieved the best predictive performance on the Kaggle public leaderboard (RMSLE = 0.12090).
2. **Interpretability:**
   - **Lasso & ElasticNet** provided logical and interpretable results. The top factors driving price were Location (e.g., Neighborhood_Crawfor), Total Area (e.g., AllSF-2), and Overall Quality (e.g., OverallQual-s3).
   - **Ridge** completely failed at interpretability. The severe multicollinearity (worsened by FE) caused illogical coefficients (e.g., AllSF having a large negative weight while AllSF-2 had a large positive one), making it unusable for explanation.
3. **Key Discovery (from Data Mining):** The most important finding came from the Lasso model's feature selection. It automatically dropped original linear features (like GrLivArea) and selected the polynomial features (like GrLivArea-3 and OverallQual-s3) instead.

This revealed a critical insight: **The relationship between price and key factors like Size and Quality is not linear, but exponential**. Price increases at an accelerating rate as quality and size increase.
