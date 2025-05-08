# Utilizing Machine Learning to Predict Which Stocks Warren Buffett is Likely to Buy Based on Past Investment History- A Predictive Modeling Approach

### Team Lambda
Eliel Polanco |
Amita Patil |
Hunter Huberdeau | 
Hayden Realmuto

## Abstract
Team Lambda explores the use of machine learning to anticipate the stock purchasing behavior of renowned investor Warren Buffett. By analyzing historical stock selections alongside company-level financial indicators, our team aims to model the underlying patterns that drive Buffett’s investment decisions. Drawing on the principles of value investing, we hypothesize that specific financial metrics—such as low price-to-earnings ratios, strong revenue growth, and high dividend yields—can serve as reliable predictors of future acquisitions. Unlike prior studies that focus on general stock forecasting or portfolio optimization, our approach emphasizes explainability and investor behavior modeling. The resulting predictive framework aspires to function as a decision-support tool, empowering individual investors with forward-looking insights into one of the most closely followed investment strategies in modern finance.

#### Question:
Are investors like Warren Buffett likely to buy specific stocks next based on historical data and known investment behaviors?

#### Hypothesis: 
If an individual's investment philosophy can be reverse-engineered, we can predict their next stock purchase. Investor decisions are influenced by a multitude of factors including financial metrics, economic data, and market sentiment. By analyzing the historical tendencies of a great investor -- we aim to build a model that can predict an investor's next stock purchase.

#### Predictions:
1. Historical Purchase Behavior: If an investor has a pattern of purchasing stocks in a particular industry (e.g., technology), they are likely to continue purchasing stocks from the same industry.
2. Market Sentiment: Positive news and market sentiment surrounding a particular stock could increase the probability of an investor choosing that stock next. 
3. Value Investing: Stocks deemed undervalued by valuation metrics like the price-to-earnings ratio (P/E) and price-to-book ratio (P/B) will have a higher probability than overvalued stocks. 

#### Data Sets:
Bloomberg Terminal - Time series historical data of financial and economic metrics.

DATAROMA (https://www.dataroma.com) - Portfolio activity. 

#### Features and Observations: 
##### Qualitative: 
Sector, Industry, Sentiment 

##### Financials: 
Return on Equity (ROE), Return on Assets (ROA), Gross Profit Margin, Operating Profit Margin, Net Profit Margin, EBITDA Margin, Price-to-earnings (P/E), Price-to-book (P/B), Price-to-Free Cash Flow (P/FCF), Dividend Yield, Dividend Payout Ratio, Interest Coverage Ratio, Free Cash Flow, Operating Cash Flow, Inventory Turnover, Earnings Growth Rates, Book Value Growth, Return on Invested Capital (ROIC), Working Capital Metric, Capital Expenditure (CapEx), Property Plant and Equipment (PPE), Retained Earnings, Quick Ratio. 

##### Economic: 
Federal Funds Rate, 10-Year Treasury Yield, Inflation Rate (CPI, PPI), GDP Growth Rate, Unemployment Rate, Consumer Confidence Index, Corporate Credit Spreads (Investment Grade vs High Yield), Yield Curve Shape (2s/10s spread), S&P 500 PE Ratio, Volatility (VIX Index), Household Debt-to-Income Ratio. 


##### Code Folder structure :
 - `notebooks/`: Jupyter notebooks for each step of the pipeline.
- `src/`: Source code for reusable functions (e.g., data loading, model evaluation).
- `Data/`: Folder for datasets and data splits.
- `models/`: Folder for saved models.



