import pandas as pd
from sqlalchemy import create_engine

def get_db_fundamentals(ticker, db):
	sql_query = f"""SELECT * FROM fundamentals
		JOIN stock ON stock.id = fundamentals.stock_id
		WHERE stock.ticker = '{ticker}'"""
	df = pd.read_sql(sql_query, db)
	df = df[['longbusinesssummary', 'sector', 'sharesoutstanding', 'marketcap', 'forwardpe', 'dividendyield', 'beta', 'previousclose', 'averagevolume']]
	df.columns = ['longBusinessSummary', 'sector', 'sharesOutstanding', 'marketCap', 'forwardPE', 'dividendYield', 'beta', 'previousClose', 'averageVolume']
	return df

def get_db_price(ticker, db):
	sql_query = f"""SELECT * FROM price
		JOIN stock ON stock.id = price.stock_id
		WHERE stock.ticker = '{ticker}'"""
	df = pd.read_sql(sql_query, db)
	df = df[['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'date_price']]
	df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
	return df