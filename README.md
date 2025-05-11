# Utilizing Machine Learning to Predict Which Stocks Warren Buffett is Likely to Buy Based on Past Investment History- A Predictive Modeling Approach

### Team Lambda
Eliel Polanco |
Amita Patil |
Hunter Huberdeau | 
Hayden Realmuto

## Abstract
Can we predict which stocks influential investors like Warren Buffett will buy next? Team Lambda’s innovative project combines behavioral finance and machine learning to explore this question by modeling Buffett's historical investment decisions using stock-level financial fundamentals and macroeconomic indicators. We hypothesize that Buffett’s stock purchases are influenced by key financial indicators such as low price-to-earnings (P/E) ratios, high return on equity (ROE), low debt-to-equity (D/E) ratios, and steady profitability. We predict that stocks exhibiting these traits during a given quarter are more likely to be added to his portfolio in that same period. We engineered a dataset of over 39,000 quarterly stock observations from 2007 to 2024, incorporating company metrics (e.g., P/E, ROE, Profit Margin) and economic conditions (e.g., GDP, CPI). We applied dimensionality reduction using Principal Component Analysis (PCA) and built predictive models using XGBoost, Random Forest, and Elastic Net Linear Regression. After applying 1 to 1 nearest neighbor matching and hyper-parameter tuning, XGBoost had the best model performance achieving an F1-score of 0.71 and an AUC of 0.82. Our findings confirm that features central to Buffett's value investing style are strong predictors and that predictive models can offer signal as timely insight into potential buys.

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

Our code is structured in chronological order for the most streamlined approach for displaying our EDA, feature engineering, and modeling approach. For the complete analysis, please run all numbered 'notebook' files from 01-07. For running all of our models, use the 'src' folder and run the evaluate_models_notebook.jpynb file. 

#### Code Folder structure :
 - `notebooks/`: Jupyter notebooks for each step of the pipeline.
- `src/`: Source code for reusable functions (e.g., data loading, model evaluation).
- `Data/`: Folder for datasets and data splits.
- `models/`: Folder for saved models.



