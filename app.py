# Import libraries
import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import requests
import os, time, metadata_parser
from streamlit_lottie import st_lottie
from sqlalchemy import create_engine
from PIL import Image
from plot_setup import finastra_theme
from OptimizePortfolio import optimize_portfolio, calculate_portfolio, getCompanyName
from chart import areaChart, candlestickChart, gaugeChart, pieChart
from db import config
import user
from download_data import Data

db = create_engine(config.opt)
display_cols = ["DATE", "SourceCommonName", "URL", "Tone"]

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache(show_spinner=False, suppress_st_warning=True)
def filter_company_data(df_company, esg_categories, start, end):
    #Filter E,S,G Categories
    comps = []
    for i in esg_categories:
        X = df_company[df_company[i] == True]
        comps.append(X)
    df_company = pd.concat(comps)
    df_company = df_company[df_company.DATE.between(start, end)]
    return df_company

@st.cache(show_spinner=False, suppress_st_warning=True,
          allow_output_mutation=True)
def load_data():
    data = Data().read()
    companies = data["data"].Organization.sort_values().unique().tolist()
    companies.insert(0,"Select a Company")
    return data, companies

@st.cache(show_spinner=False,suppress_st_warning=True)
def filter_publisher(df_company,publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company


def get_melted_frame(data_dict, frame_names, keepcol=None, dropcol=None):
    if keepcol:
        reduced = {k: df[keepcol].rename(k) for k, df in data_dict.items()
                   if k in frame_names}
    else:
        reduced = {k: df.drop(columns=dropcol).mean(axis=1).rename(k)
                   for k, df in data_dict.items() if k in frame_names}
    df = (pd.concat(list(reduced.values()), axis=1).reset_index().melt("date")
            .sort_values("date").ffill())
    df.columns = ["DATE", "ESG", "Score"]
    return df.reset_index(drop=True)


def filter_on_date(df, start, end, date_col="DATE"):
    df = df[(df[date_col] >= pd.to_datetime(start)) &
            (df[date_col] <= pd.to_datetime(end))]
    return df


def get_clickable_name(url):
    try:
        T = metadata_parser.MetadataParser(url=url, search_head_only=True)
        title = T.metadata["og"]["title"].replace("|", " - ")
        return f"[{title}]({url})"
    except:
        return f"[{url}]({url})"

def main():
	hms = """
		<style>
		#MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		</style>
	"""
	alt.themes.register("finastra", finastra_theme)
	alt.themes.enable("finastra")
	violet, fuchsia = ["#694ED6", "#C137A2"]
	icon_path = os.path.join("images", "esg_ai_logo.png")
	st.set_page_config(page_title="SEG AI", page_icon=icon_path,
					layout='centered', initial_sidebar_state="collapsed")
	_, logo, _ = st.beta_columns(3)
	logo.image(icon_path, width=200)
	style = ("text-align:center; padding: 0px; font-family: arial black;, "
			"font-size: 400%")
	title = f"<h1 style='{style}'>SEG<sup>AI</sup></h1>"
	st.write(title, unsafe_allow_html=True)
	subheading_style = ("text-align:center; padding: 0px")
	subheading = f"<h3 style='{subheading_style}'>The Responsible Investor</h3>"
	st.write(subheading, unsafe_allow_html=True)
	candle_image = Image.open('./images/candleStick.png')
	st.markdown(hms, unsafe_allow_html=True)

	st.markdown("<h1 style='text-align: center; color: red;'></h1>", unsafe_allow_html=True)

	functionality = st.sidebar.selectbox('What would you like to do?',
		('Track Individual Stocks', 'ESG Performance'))
		# , 'Optimize My Portfolio'))
	
	with st.spinner(text="Fetching Data..."):
		data, companies = load_data()
	
	df_conn = data["conn"]
	df_data = data["data"]
	embeddings = data["embed"]

	if (functionality == 'Track Individual Stocks'):
		st.markdown('<h2 style="display: flex;justify-content: center;">Track Individual Stocks</h2>', unsafe_allow_html=True)
		ticker = st.text_input('Enter ticker symbol', value='AMD')
		companyName = getCompanyName(ticker)
		df = user.get_db_price(ticker, db)
		st.markdown(f"<h3 style='display: flex;justify-content: center;'>{companyName}</h3>", unsafe_allow_html=True)
		plot = areaChart(df, ticker)
		st.plotly_chart(plot)
		gauge = gaugeChart(df, ticker)
		st.plotly_chart(gauge)
		
		st.subheader(f"Fundamental Analysis of {companyName}")
		with st.beta_expander("What is Fundamental Analysis?"):
			st.write("""Fundamental analysis (FA) is a method of **measuring a security's intrinsic value** 
				by examining related economic and financial factors. These factors include macroeconomic 
				factors such as the state of the economy and industry conditions to microeconomic factors 
				like the effectiveness of the company's management. The **end goal** is to arrive at a number 
				that an investor can compare with a security's current price **in order to see whether the 
				security is undervalued or overvalued.**""")
		
		info = user.get_db_fundamentals(ticker, db)
		st.write(f"**_Business Summary_**: {info['longBusinessSummary'].values[0]}")
		st.write(f"**_Sector_**: {info['sector'].values[0]}")
		st.write(f"**_Shares Outstanding_**: {info['sharesOutstanding'].values[0]}")
		with st.beta_expander("Shares Outstanding"):
			st.write("""Shares outstanding refer to a company's stock currently held by all its 
				shareholders, including share blocks held by institutional investors and restricted 
				shares owned by the company’s officers and insiders.""")
		st.write(f"**_Market Capitalization_**: {info['marketCap'].values[0]}")
		with st.beta_expander("Market Capitalization"):
			st.write("""Market Capitalization is the total dollar value of all of a company's 
				outstanding shares. It is a measure of corporate size.""")
			st.text('Market Capital = Current Market Price * Number Of Shares Outstanding')
		st.write(f"**_Price-to-Earnings (P/E) Ratio_**: {info['forwardPE'].values[0]}")
		with st.beta_expander("P/E Ratio"):
			st.write("""The **price-to-earnings (P/E) ratio** is a metric that helps investors 
				determine the market value of a stock compared to the company's earnings. The P/E 
				ratio shows what the market is willing to pay today for a stock based on its past 
				or future earnings. The P/E ratio is important because it provides a measuring stick 
				for comparing whether a stock is overvalued or undervalued.""")
			st.write("""A **high** P/E ratio could mean that a stock's price is expensive relative to 
				earnings and **possibly overvalued**. Conversely, a **low** P/E ratio might indicate that 
				the **current stock price is cheap relative to earnings**.""")
			st.text('P/E = Average Common Stock Price / Net Income Per Share')
			st.write("""The **Forward P/E** uses forecasted earnings to calculate P/E for the next fiscal 
				year. If the earnings are expected to grow in the future, the forward P/E will be lower 
				than the current P/E.""")
			st.text('Forward P/E = Current Market Price / Forecasted Earnings Per Share')
		st.write(f"**_Dividend Yield_**: {info['dividendYield'].values[0]}")
		with st.beta_expander("Dividend Yield"):
			st.write("""The dividend yield, expressed as a percentage, is a financial ratio 
				(dividend/price) that shows how much a company pays out in dividends each year 
				relative to its stock price.""")
			st.text('Dividend Yield = Annual Dividend Per Share / Price Per Share')
			st.write("""New companies that are relatively small, but still growing quickly, may pay a 
				lower average dividend than mature companies in the same sectors. In general, mature 
				companies that aren't growing very quickly pay the highest dividend yields.""")
		st.write(f"**_Beta_**: {info['beta'].values[0]}")
		with st.beta_expander("Beta"):
			st.write("""Beta is a measure of the volatility—or systematic risk—of a security or portfolio 
				compared to the market as a whole. It effectively describes the activity of a security's 
				returns as it responds to swings in the market.""")
			st.write("If a stock has a beta of **1.0**, it indicates that its price activity is strongly correlated with the market.")
			st.write("""A beta value that is **less than 1.0** means that the security is theoretically 
				less volatile than the market. Including this stock in a portfolio makes it less risky 
				than the same portfolio without the stock.""")
			st.write("""A beta that is greater than 1.0 indicates that the security's price is 
				theoretically more volatile than the market. For example, if a stock's beta is 
				1.2, it is assumed to be 20% more volatile than the market. Technology stocks 
				and small cap stocks tend to have higher betas than the market benchmark.""")
			st.write("""A negative beta shows that the asset inversely follows the market, 
				meaning it decreases in value if the market goes up and increases if the market goes down.""")
	
	elif (functionality == 'Optimize My Portfolio'):
		st.markdown('<h2 style="display: flex;justify-content: center;">Optimize My Portfolio</h2>', unsafe_allow_html=True)
		lottie_url = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_TWo1Pn.json")
		st_lottie(lottie_url, height=300)	
		index = st.sidebar.selectbox('Select Which Companies to Evaluate', 
			('S&P100','S&P500', 'NASDAQ-100'))
		portfolio_val = int(st.sidebar.text_input("Enter Amount to Invest", value=10000))
		strategy = st.sidebar.selectbox("Select Allocation Strategy",
			('Optimize Return & Risk', 'Minimize Risk', 'Custom Risk', 'Custom Return'))
		
		if (index == 'S&P500'):
			st.subheader('S&P 500')
			st.write('''The S&P 500, or simply the S&P, is a stock market index that measures the 
				stock performance of 500 large companies listed on stock exchanges in the United 
				States. It is one of the most commonly followed equity indices. The S&P 500 index 
				is a capitalization-weighted index and the 10 largest companies in the index account 
				for 27.5% of the market capitalization of the index. The 10 largest companies in the 
				index, in order of weighting, are Apple Inc., Microsoft, Amazon.com, Facebook, Tesla, 
				Inc., Alphabet Inc. (class A & C), Berkshire Hathaway, Johnson & Johnson, and JPMorgan 
				Chase & Co., respectively.''')
			portfolio = pd.read_csv("S&P500.csv", index_col="Date")
		
		elif (index == 'S&P100'):
			st.subheader('S&P 100')
			st.write('''The S&P 100 Index is a stock market index of United States stocks maintained 
				by Standard & Poor's. It is a subset of the S&P 500 and includes 101 (because one of 
				its component companies has 2 classes of stock) leading U.S. stocks. Constituents of 
				the S&P 100 are selected for sector balance and represent about 67% of the market 
				capitalization of the S&P 500 and almost 54% of the market capitalization of the U.S. 
				equity markets as of December 2020. The stocks in the S&P 100 tend to be the largest 
				and most established companies in the S&P 500.''')
			portfolio = pd.read_csv("SP100index.csv", index_col="Date")
			with st.beta_expander("The S&P 100 consists of:"):
				tickers = portfolio.columns
				for ticker in tickers:
					st.write(f"* {getCompanyName(ticker)}")
		
		elif (index == 'NASDAQ-100'):
			st.subheader('NASDAQ-100')
			st.write('''The NASDAQ-100 is a stock market index made up of 102 equity securities issued 
				by 100 of the largest non-financial companies listed on the Nasdaq stock market.''')
			portfolio = pd.read_csv("NASDAQ.csv", index_col="Date")
			with st.beta_expander("The NASDAQ-100 consists of:"):
				tickers = portfolio.columns
				for ticker in tickers:
					st.write(f"* {getCompanyName(ticker)}")
		
		elif (index == 'Dow Jones Industrial Average (DJIA)'):
			st.subheader('Dow Jones Industrial Average (DJIA)')
			st.write('''The Dow Jones Industrial Average (DJIA), Dow Jones, or simply the Dow, is a
				stock market index that measures the stock performance of 30 large companies listed 
				on stock exchanges in the United States. It is one of the most commonly followed 
				equity indices. First calculated on May 26, 1896, the index is the second-oldest 
				among the U.S. market indices (after the Dow Jones Transportation Average). It was 
				created by Charles Dow, the editor of The Wall Street Journal and the co-founder of 
				Dow Jones & Company, and named after him and his business associate, statistician 
				Edward Jones. Although the word industrial appears in the name of the index, several 
				of the constituent companies operate in sectors of the economy other than heavy industry.''')
			portfolio = pd.read_csv("DJIA.csv", index_col="Date")
			with st.beta_expander("The DJIA consists of:"):
				tickers = portfolio.columns
				for ticker in tickers:
					st.write(f"* {getCompanyName(ticker)}")
		
		if (strategy == 'Optimize Return & Risk'):
			expectedReturns, volatility, ratio, allocation, leftover = optimize_portfolio(portfolio, portfolio_val)
		
		elif (strategy == 'Minimize Risk'):
			expectedReturns, volatility, ratio, allocation, leftover = optimize_portfolio(portfolio, portfolio_val, method="min_risk")
		
		elif (strategy == 'Custom Risk'):
			target = st.sidebar.slider("Maximise return for a chosen target risk", 15, 50)
			expectedReturns, volatility, ratio, allocation, leftover = optimize_portfolio(portfolio, portfolio_val, method="custom_risk", custom_value=target/100)
		
		elif (strategy == 'Custom Return'):
			target = st.sidebar.slider("Minimize risk for a chosen target return", 15, 50)
			expectedReturns, volatility, ratio, allocation, leftover = optimize_portfolio(portfolio, portfolio_val, method="custom_return", custom_value=target/100)
		
		portfolio_df = calculate_portfolio(portfolio, allocation)
		st.subheader('Suggested Portfolio')
		st.write(portfolio_df)
		st.write(f'**Expected annual return**: {str(round(expectedReturns*100, 2))}%')
		st.write(f'**Annual Volatility**: {str(round(volatility*100, 2))}%')
		st.write(f'**Funds Remaining**: ${str(round(leftover, 2))}')
		st.plotly_chart(pieChart(portfolio_df))
	elif functionality == 'ESG Performance':
		st.sidebar.title("Filter Options")
		date_place = st.sidebar.empty()
		esg_categories = st.sidebar.multiselect("Select News Categories",
												["E", "S", "G"], ["E", "S", "G"])
		pub = st.sidebar.empty()
		num_neighbors = st.sidebar.slider("Number of Connections", 1, 20, value=8)

		company = st.selectbox("Select a Company to Analyze", companies)

		if company and company != "Select a Company":
			###### FILTER ######
			df_company = df_data[df_data.Organization == company]
			diff_col = f"{company.replace(' ', '_')}_diff"
			esg_keys = ["E_score", "S_score", "G_score"]
			esg_df = get_melted_frame(data, esg_keys, keepcol=diff_col)
			ind_esg_df = get_melted_frame(data, esg_keys, dropcol="industry_tone")
			tone_df = get_melted_frame(data, ["overall_score"], keepcol=diff_col)
			ind_tone_df = get_melted_frame(data, ["overall_score"],
										dropcol="industry_tone")
		
			###### DATE WIDGET ######
			start = df_company.DATE.min()
			end = df_company.DATE.max()
			selected_dates = date_place.date_input("Select a Date Range",value=[start, end], min_value=start, max_value=end, key=None)
			time.sleep(0.8)  #Allow user some time to select the two dates -- hacky :D
			start, end = selected_dates
			
			###### FILTER DATA ######
			df_company = filter_company_data(df_company, esg_categories,start, end)
			esg_df = filter_on_date(esg_df, start, end)
			ind_esg_df = filter_on_date(ind_esg_df, start, end)
			tone_df = filter_on_date(tone_df, start, end)
			ind_tone_df = filter_on_date(ind_tone_df, start, end)
			date_filtered = filter_on_date(df_data, start, end)
			
			###### PUBLISHER SELECT BOX ######
			publishers = df_company.SourceCommonName.sort_values().unique().tolist()
			publishers.insert(0, "all")
			publisher = pub.selectbox("Select Publisher", publishers)
			df_company = filter_publisher(df_company, publisher)
			
			###### CHART: ESG RADAR ######
			# col1, col2 = st.beta_columns((1, 3))
			# st.markdown("---")
			avg_esg = data["ESG"]
			avg_esg.rename(columns={"Unnamed: 0": "Type"}, inplace=True)
			avg_esg.replace({"T": "Overall", "E": "Environment",
							"S": "Social", "G": "Governance"}, inplace=True)
			avg_esg["Industry Average"] = avg_esg.mean(axis=1)

			radar_df = avg_esg[["Type", company, "Industry Average"]].melt("Type",
				value_name="score", var_name="entity")
			
			radar = px.line_polar(radar_df, r="score", theta="Type",
				color="entity", line_close=True, hover_name="Type",
				hover_data={"Type": True, "entity": True, "score": ":.2f"},
				color_discrete_map={"Industry Average": fuchsia, company: violet})

			radar.update_layout(template=None,
								polar={
									"radialaxis": {"showticklabels": False,
													"ticks": ""},
									   "angularaxis": {"showticklabels": True,
									   					"rotation":180,
									                   "ticks": ""}
									},
								legend={"title": None, "yanchor": "middle",
										"orientation": "h"},
								title={"text": "<b>ESG Scores</b>",
									}
								)
			radar.update_layout(showlegend=True)
			st.plotly_chart(radar, use_container_width=True)


			# st.markdown("---")
			URL_Expander_trends = st.beta_expander(f"Trends Over Time - {company.title()}", False)
			col1, col2 = URL_Expander_trends.beta_columns((1, 2))
			metric_options = ["Tone", "NegativeTone", "PositiveTone", "Polarity",
							"ActivityDensity", "WordCount", "Overall Score",
							"ESG Scores"]
			line_metric = col1.radio("Choose Metric", options=metric_options)

			if line_metric == "ESG Scores":
				# Get ESG scores
				esg_df["WHO"] = company.title()
				ind_esg_df["WHO"] = "Industry Average"
				esg_plot_df = pd.concat([esg_df, ind_esg_df]
										).reset_index(drop=True)
				esg_plot_df.replace({"E_score": "Environment", "S_score": "Social",
									"G_score": "Governance"}, inplace=True)

				metric_chart = alt.Chart(esg_plot_df, title="Trends Over Time"
										).mark_line().encode(
					x=alt.X("yearmonthdate(DATE):O", title="DATE"),
					y=alt.Y("Score:Q"),
					color=alt.Color("ESG", sort=None, legend=alt.Legend(
						title=None, orient="top")),
					strokeDash=alt.StrokeDash("WHO", sort=None, legend=alt.Legend(
						title=None, symbolType="stroke", symbolFillColor="gray",
						symbolStrokeWidth=4, orient="top")),
					tooltip=["DATE", "ESG", alt.Tooltip("Score", format=".5f")]
					)
			else:
				if line_metric == "Overall Score":
					line_metric = "Score"
					tone_df["WHO"] = company.title()
					ind_tone_df["WHO"] = "Industry Average"
					plot_df = pd.concat([tone_df, ind_tone_df]).reset_index(drop=True)
				else:
					df1 = df_company.groupby("DATE")[line_metric].mean(
						).reset_index()
					df2 = filter_on_date(df_data.groupby("DATE")[line_metric].mean(
						).reset_index(), start, end)
					df1["WHO"] = company.title()
					df2["WHO"] = "Industry Average"
					plot_df = pd.concat([df1, df2]).reset_index(drop=True)
				metric_chart = alt.Chart(plot_df, title="Trends Over Time"
										).mark_line().encode(
					x=alt.X("yearmonthdate(DATE):O", title="DATE"),
					y=alt.Y(f"{line_metric}:Q", scale=alt.Scale(type="linear")),
					color=alt.Color("WHO", legend=None),
					strokeDash=alt.StrokeDash("WHO", sort=None,
						legend=alt.Legend(
							title=None, symbolType="stroke", symbolFillColor="gray",
							symbolStrokeWidth=4, orient="top",
							),
						),
					tooltip=["DATE", alt.Tooltip(line_metric, format=".3f")]
					)
			metric_chart = metric_chart.properties(
				height=340,
				width=250
			).interactive()
			col2.altair_chart(metric_chart, use_container_width=True)


			###### CHART: DOCUMENT TONE DISTRIBUTION #####
			# st.markdown("---")
			URL_Expander_doc_tone = st.beta_expander(f"Document Tone Distribution - {company.title()}", False)
			dist_chart = alt.Chart(df_company, title="Document Tone "
									"Distribution").transform_density(
					density='Tone',
					as_=["Tone", "density"]
				).mark_area(opacity=0.5,color="purple").encode(
						x=alt.X('Tone:Q', scale=alt.Scale(domain=(-10, 10))),
						y='density:Q',
						tooltip=[alt.Tooltip("Tone", format=".3f"),
									alt.Tooltip("density:Q", format=".4f")]
					).properties(
						height=325,
					).configure_title(
						dy=-20
					).interactive()
			URL_Expander_doc_tone.markdown("### <br>", unsafe_allow_html=True)
			URL_Expander_doc_tone.altair_chart(dist_chart,use_container_width=True)


			###### CHART: SCATTER OF ARTICLES OVER TIME #####
			# st.markdown("---")
			URL_Expander_article_scatter = st.beta_expander(f"Article Tone Distribution - {company.title()}", False)
			scatter = alt.Chart(df_company, title="Article Tone").mark_circle().encode(
				x="NegativeTone:Q",
				y="PositiveTone:Q",
				size="WordCount:Q",
				color=alt.Color("Polarity:Q", scale=alt.Scale()),
				tooltip=[alt.Tooltip("Polarity", format=".3f"),
						alt.Tooltip("NegativeTone", format=".3f"),
						alt.Tooltip("PositiveTone", format=".3f"),
						alt.Tooltip("DATE"),
						alt.Tooltip("WordCount", format=",d"),
						alt.Tooltip("SourceCommonName", title="Site")]
				).properties(
					height=450
				).interactive()
			URL_Expander_article_scatter.altair_chart(scatter, use_container_width=True)

			###### NUMBER OF NEIGHBORS TO FIND #####
			neighbor_cols = [f"n{i}_rec" for i in range(num_neighbors)]
			company_df = df_conn[df_conn.company == company]
			neighbors = company_df[neighbor_cols].iloc[0]


			###### CHART: 3D EMBEDDING WITH NEIGHBORS ######
			# st.markdown("---")
			URL_Expander_graph_neighbors = st.beta_expander(f"Company Connections - {company.title()}", False)
			color_f = lambda f: f"Company: {company.title()}" if f == company else (
				"Connected Company" if f in neighbors.values else "Other Company")
			embeddings["colorCode"] = embeddings.company.apply(color_f)
			point_colors = {company: violet, "Connected Company": fuchsia,
							"Other Company": "lightgrey"}
			fig_3d = px.scatter_3d(embeddings, x="0", y="1", z="2",
								color='colorCode',
								color_discrete_map=point_colors,
								opacity=0.4,
								hover_name="company",
								hover_data={c: False for c in embeddings.columns},
								)
			fig_3d.update_layout(legend={"orientation": "h",
										"yanchor": "bottom",
										"title": None},
								title={"text": "<b>Company Connections</b>",
										"x": 0.5, "y": 0.9,
										"xanchor": "center",
										"yanchor": "top",
										"font": {"family": "Futura", "size": 23}},
								scene={"xaxis": {"visible": False},
										"yaxis": {"visible": False},
										"zaxis": {"visible": False}},
								margin={"l": 0, "r": 0, "t": 0, "b": 0},
								)
			URL_Expander_graph_neighbors.plotly_chart(fig_3d, use_container_width=True)

			###### CHART: NEIGHBOR SIMILIARITY ######
			neighbor_conf = pd.DataFrame({
				"Neighbor": neighbors,
				"Confidence": company_df[[f"n{i}_conf" for i in
										range(num_neighbors)]].values[0]})
			conf_plot = alt.Chart(neighbor_conf, title="Connected Companies"
								).mark_bar().encode(
				x="Confidence:Q",
				y=alt.Y("Neighbor:N", sort="-x"),
				tooltip=["Neighbor", alt.Tooltip("Confidence", format=".3f")],
				color=alt.Color("Confidence:Q", scale=alt.Scale(), legend=None)
			).properties(
				height=25 * num_neighbors + 100
			).configure_axis(grid=False)
			URL_Expander_graph_neighbors.altair_chart(conf_plot, use_container_width=True)

			###### DISPLAY DATA ######
			# st.markdown("---")
			URL_Expander = st.beta_expander(f"View {company.title()} Data", False)
			metric_options = ["All", "Positive Tone", "Negative Tone"]
			tone_metric = URL_Expander.radio("Choose Metric", options=metric_options)

			if tone_metric == "All":
				URL_Expander.write(f"### {len(df_company):,d} Matching All Articles for " + company.title())
				URL_Expander.write(df_company[display_cols])
			elif tone_metric == "Positive Tone":
				URL_Expander.write(f"### {len(df_company):,d} Matching Positive Tone Articles for " + company.title())
				URL_Expander.write(df_company[df_company["Tone"] > 0][display_cols])
			elif tone_metric == "Negative Tone":
				URL_Expander.write(f"### {len(df_company):,d} Matching Negative Tone Articles for " + company.title())
				URL_Expander.write(df_company[df_company["Tone"] < 0][display_cols])
			
	elif functionality == "Presentation":
		st.write("Introduction")
		st.write("Problem Statement")
		st.write("Technology & Architecture")

if __name__=='__main__':
	main()
