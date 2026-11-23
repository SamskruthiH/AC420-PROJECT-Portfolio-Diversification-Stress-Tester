import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

st.set_page_config(page_title='Portfolio Diversification Stress Tester', page_icon='📊', layout='wide')
st.title('Portfolio Diversification Stress Tester')
st.caption('Explore portfolio concentration, return correlations, volatility, and historical drawdowns. Educational use only.')

TRADING_DAYS = 252
STRESS_WINDOWS = {
    'COVID-19 selloff (Feb–Mar 2020)': ('2020-02-19', '2020-03-23'),
    '2022 bear market (Jan–Oct 2022)': ('2022-01-03', '2022-10-12'),
    'Custom range': None,
}

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, start, end):
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # Supports current yfinance's price-field-first column arrangement.
        prices = raw['Close'] if 'Close' in raw.columns.get_level_values(0) else raw.xs('Close', axis=1, level=1)
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices.reindex(columns=list(tickers)).apply(pd.to_numeric, errors='coerce')

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sectors(tickers):
    sectors = {}
    for ticker in tickers:
        try:
            sectors[ticker] = yf.Ticker(ticker).info.get('sector') or 'Unknown'
        except Exception:
            sectors[ticker] = 'Unknown'
    return sectors

with st.sidebar:
    st.header('Portfolio setup')
    ticker_text = st.text_input('Stock tickers (comma-separated)', 'AAPL, MSFT, JNJ, JPM')
    tickers = list(dict.fromkeys(x.strip().upper() for x in ticker_text.split(',') if x.strip()))
    if not 2 <= len(tickers) <= 12:
        st.error('Enter 2 to 12 distinct stock tickers.')
        st.stop()
    st.caption('Adjust weights below; values are normalized to 100%.')
    raw_weights = {}
    for ticker in tickers:
        raw_weights[ticker] = st.number_input(f'{ticker} weight', min_value=0.0, max_value=100.0,
                                              value=round(100 / len(tickers), 2), step=1.0, key=f'weight_{ticker}')
    if sum(raw_weights.values()) <= 0:
        st.error('At least one stock must have a positive weight.')
        st.stop()
    weights = pd.Series(raw_weights, dtype=float)
    weights /= weights.sum()
    lookback = st.selectbox('Analysis period', ['2y', '5y', '10y'], index=1)
    benchmark = st.text_input('Benchmark for beta', value='SPY').strip().upper()
    if not benchmark:
        st.error('Enter a benchmark ticker.')
        st.stop()
    st.divider()
    st.subheader('Optional WRDS sector mapping')
    st.caption('Upload a CSV with columns ticker,sector. Do not upload restricted WRDS records to a public repository.')
    sector_file = st.file_uploader('Sector mapping CSV', type=['csv'])

end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
start_date = end_date - pd.DateOffset(years=int(lookback[:-1]))
# Fetch enough history for portfolio analysis and historical stress tests independently.
with st.spinner('Downloading historical prices...'):
    prices = download_prices(tuple(tickers + ([benchmark] if benchmark not in tickers else [])),
                             start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
if prices.empty:
    st.error('No price data returned. Check the ticker symbols or retry later.')
    st.stop()

missing = [ticker for ticker in tickers if ticker not in prices or prices[ticker].dropna().empty]
if missing:
    st.error('No data found for: ' + ', '.join(missing))
    st.stop()

# Use only common observed dates: never forward-fill missing historical prices into returns.
asset_prices = prices[tickers].dropna(how='any')
if len(asset_prices) < 65:
    st.error('Not enough overlapping price history. Try different tickers.')
    st.stop()
returns = asset_prices.pct_change().dropna()
w = weights.reindex(tickers).to_numpy()
covariance = returns.cov().to_numpy() * TRADING_DAYS
portfolio_vol = float(np.sqrt(max(w @ covariance @ w, 0)))
