import yfinance as yf


# Download stock data then export as CSV
data_df = yf.download("AAPL", start="2018-01-01", end="2020-09-10")
data_df.to_csv('./aapl.csv')
