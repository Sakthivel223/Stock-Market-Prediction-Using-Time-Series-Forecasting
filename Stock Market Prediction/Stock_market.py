import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Constants for date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Title and sidebar
st.title("Stock Forecast Webapp")

# Stock selection
stocks = ("GOOG", "AAPL", "MSFT", "GME", "TSLA", "AMZN", "NFLX")
selected_stock = st.selectbox("Select dataset for prediction", stocks)
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Load data function with caching
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    # Ensure the data has at least 2 valid rows and no NaN values
    data.dropna(inplace=True)
    if len(data) < 2:
        raise ValueError("Insufficient data for forecast.")
    return data

# Load the data
data = load_data(selected_stock)

# Prepare data for Prophet
df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})  # Rename for Prophet

# Create and fit the Prophet model
m = Prophet()  # Create Prophet instance
m.fit(df_train)  # Fit with training data

# Create forecast
future = m.make_future_dataframe(periods=period)  # Future data points
forecast = m.predict(future)  # Predict the future

# Sidebar for page navigation
page = st.sidebar.radio("Select a Page", ("Home", "Forecast", "Graph"))

# Home section: Display raw data
if page == "Home":
    st.subheader("Raw Stock Data")
    st.write(data[["Date", "Open", "Close", "High", "Low", "Volume"]].tail())

# Forecast section: Display forecast data as a table
elif page == "Forecast":
    st.subheader("Forecast Data Table")
    st.write(forecast.tail())  # Display the forecast tail

# Graph section: Raw data and forecast plots
elif page == "Graph":
    # Raw stock data plot
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Stock Open"))
    fig_raw.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Stock Close"))
    fig_raw.update_layout(
        title="Time Series Data with Range Slider",
        xaxis_rangeslider_visible=True,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig_raw)

    # Forecast plot with Prophet
    st.subheader("Forecast Plot")
    fig_forecast = plot_plotly(m, forecast)  # Plot forecast with Prophet
    st.plotly_chart(fig_forecast)

    # Forecast components plot
    st.subheader("Forecast Components")
    fig_components = m.plot_components(forecast)  # Trend, seasonality, etc.
    st.write(fig_components)