# Utilizing Machine Learning to Predict Which Stocks Warren Buffett is Likely to Buy Based on Past Investment History- A Predictive Modeling Approach

### Team Lambda
Eliel Polanco |
Amita Patil |
Hunter Huberdeau | 
Hayden Realmuto

## Abstract -add results section (XGboost best performing model, with random forest to follow)
Team Lambda explores the use of machine learning to anticipate the stock purchasing behavior of renowned investor Warren Buffett. Conventional studies traditionally focus on general stock forecasting or portfolio optimization, while our approach emphasizes explainability and investor behavior modeling. By analyzing historical stock selections alongside company-level financial indicators, our team aims to model the underlying patterns that drive Buffett’s investment decisions. Drawing on the principles of value investing, we hypothesize that specific financial metrics—such as low price-to-earnings ratios, strong revenue growth, and high dividend yields can serve as reliable predictors of future acquisitions. We applied dimensionality reduction using Principal Component Analysis (PCA) and built predictive models using XGBoost, Random Forest, and Elastic Net Linear Regression. The resulting final models will function as a decision-support tool, empowering individual investors with forward-looking insights into one of the most closely followed investment strategies in modern finance.

### Data Sets:
Bloomberg Terminal - Time series historical data of financial and economic metrics.

DATAROMA (https://www.dataroma.com) - Portfolio activity. 

### Features and Observations: 
#### Qualitative: 
Sector, Industry, Sentiment 

#### Financials: 
Return on Equity (ROE), Return on Assets (ROA), Gross Profit Margin, Operating Profit Margin, Net Profit Margin, EBITDA Margin, Price-to-earnings (P/E), Price-to-book (P/B), Price-to-Free Cash Flow (P/FCF), Dividend Yield, Dividend Payout Ratio, Interest Coverage Ratio, Free Cash Flow, Operating Cash Flow, Inventory Turnover, Earnings Growth Rates, Book Value Growth, Return on Invested Capital (ROIC), Working Capital Metric, Capital Expenditure (CapEx), Property Plant and Equipment (PPE), Retained Earnings, Quick Ratio. 

#### Economic: 
Federal Funds Rate, 10-Year Treasury Yield, Inflation Rate (CPI, PPI), GDP Growth Rate, Unemployment Rate, Consumer Confidence Index, Corporate Credit Spreads (Investment Grade vs High Yield), Yield Curve Shape (2s/10s spread), S&P 500 PE Ratio, Volatility (VIX Index), Household Debt-to-Income Ratio. 

## Codebase Information

#### How to run our code:
# UPDATE THIS
Our code is structured in chronological order for the most streamlined approach for displaying our EDA, feature engineering, and modeling approach. Starting with notebook 01... and continuing through notebook xx... will provide a complete and comprehensive analysis

#### Code Folder structure :
 - `notebooks/`: Jupyter notebooks for each step of the pipeline.
- `src/`: Source code for reusable functions (e.g., data loading, model evaluation).
- `Data/`: Folder for datasets and data splits.
- `models/`: Folder for saved models.



