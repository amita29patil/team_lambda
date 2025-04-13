import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

#read file
stock_and_econ_cleaned = pd.read_excel("stock_and_econ_cleaned.xlsx")

# Fill missing values with mean
numeric_columns = stock_and_econ_cleaned.select_dtypes(include=[np.number]).columns
stock_and_econ_cleaned[numeric_columns] = stock_and_econ_cleaned[numeric_columns].fillna(
    stock_and_econ_cleaned[numeric_columns].mean()
)

# List of sector columns
sector_columns = [
    "Sector_Electrical Utilities & IPPs",
    "Sector_Food & Tobacco",
    "Sector_Healthcare Equipment & Supplies",
    "Sector_Hotels & Entertainment Services",
    "Sector_Insurance",
    "Sector_Investment Banking & Investment Services",
    "Sector_Machinery, Equipment & Components",
    "Sector_Media & Publishing",
    "Sector_Oil & Gas",
    "Sector_Other",
    "Sector_Pharmaceuticals",
    "Sector_Professional & Commercial Services",
    "Sector_Residential & Commercial REIT",
    "Sector_Semiconductors & Semiconductor Equipment",
    "Sector_Software & IT Services"
]

# Convert sector columns to numeric
stock_and_econ_cleaned[sector_columns] = stock_and_econ_cleaned[sector_columns].astype(float)

# Check for remaining missing values
print(f"Number of missing values: {stock_and_econ_cleaned.isna().sum().sum()}")

# Create quarter features
quarter_mapping = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
stock_and_econ_cleaned['Quarter_Num'] = stock_and_econ_cleaned['Quarter'].map(quarter_mapping)
stock_and_econ_cleaned['Quarter_sin'] = np.sin(2 * np.pi * stock_and_econ_cleaned['Quarter_Num'] / 4)
stock_and_econ_cleaned['Quarter_cos'] = np.cos(2 * np.pi * stock_and_econ_cleaned['Quarter_Num'] / 4)

# Save to Excel
stock_and_econ_cleaned.to_excel("stock_and_econ_cleaned.xlsx", index=False)

# Create PE Category numeric mapping
pe_mapping = {
    "Low": 0,
    "Good": 1,
    "High": 2,
    "Craziness": 3
}
stock_and_econ_cleaned['PE_Category_Num'] = stock_and_econ_cleaned['PE_Category'].map(pe_mapping).fillna(0)

# Save updated data
stock_and_econ_cleaned.to_excel("stock_and_econ_cleaned.xlsx", index=False)

# PCA Analysis
# Select numeric columns excluding Label
pca_data = stock_and_econ_cleaned.select_dtypes(include=[np.number]).drop('Label', axis=1, errors='ignore')

# Scale the data
scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(pca_data_scaled)

# Print PCA summary
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

# Print column names
print("\nColumn names:", stock_and_econ_cleaned.columns.tolist())

# Convert Label to categorical
stock_and_econ_cleaned['Label'] = pd.Categorical(stock_and_econ_cleaned['Label'])

# Split data into train and test sets
X = stock_and_econ_cleaned.drop('Label', axis=1)
y = stock_and_econ_cleaned['Label']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Create the logistic regression model with elastic net (alpha=0.5)
# Cs is the inverse of regularization strength (C = 1/lambda)
logistic_cv = LogisticRegressionCV(
    Cs=10,  # number of C values to try
    cv=5,   # number of cross-validation folds
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.5],  # alpha=0.5 in R's glmnet
    max_iter=1000
)

# Fit the model
logistic_cv.fit(X_train, y_train)

# Get the mean cross-validation scores
mean_scores = logistic_cv.scores_[1].mean(axis=0)  # [1] for class 1 scores

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.log(1/logistic_cv.Cs_), mean_scores)
plt.xlabel('log(lambda)')
plt.ylabel('Binomial Deviance')
plt.title('Cross-Validation Results')
plt.grid(True)
plt.show()

# Print the best lambda value
best_lambda = 1/logistic_cv.C_[0]
print(f"Best lambda value: {best_lambda}") 