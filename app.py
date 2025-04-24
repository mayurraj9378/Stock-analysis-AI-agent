import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Retrieve API key from environment variables

COMMON_STOCKS = {
    'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'META': 'META', 'NETFLIX': 'NFLX',
    'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS', 'HDFC': 'HDFCBANK.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
    'ICICIBANK': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS'
}

st.set_page_config(page_title="Stocks Analysis AI Agents", page_icon="📈", layout="wide")

try:
    from phi.agent.agent import Agent
    from phi.model.groq import Groq
    from phi.tools.yfinance import YFinanceTools
    from phi.tools.duckduckgo import DuckDuckGo
    from phi.tools.googlesearch import GoogleSearch
    PHI_IMPORTED = True
except ImportError as e:
    PHI_IMPORTED = False
    import_error_msg = str(e)

def initialize_agents():
    if not PHI_IMPORTED:
        st.error(f"Required package 'phi' is not installed or could not be imported: {import_error_msg}")
        return False

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
        return False

    if not st.session_state.get('agents_initialized', False):
        try:
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Search the web for information",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[GoogleSearch(fixed_max_results=5), DuckDuckGo(fixed_max_results=5)]
            )
            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Providing financial insights",
                model=Groq(api_key=GROQ_API_KEY),
                tools=[YFinanceTools()]
            )
            st.session_state.multi_ai_agent = Agent(
                name='Stock Market Agent',
                role='Stock market analysis specialist',
                model=Groq(api_key=GROQ_API_KEY),
                team=[st.session_state.web_agent, st.session_state.finance_agent]
            )
            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Agent initialization error: {str(e)}")
            return False
    return True

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

def create_price_chart(hist_data, symbol, theme):
    chart_bg = '#1e1e1e' if theme == 'Dark' else '#ffffff'
    grid_color = '#333333' if theme == 'Dark' else '#e6e6e6'
    text_color = '#e0e0e0' if theme == 'Dark' else '#212529'
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index, open=hist_data['Open'],
        high=hist_data['High'], low=hist_data['Low'],
        close=hist_data['Close'], name='OHLC',
        increasing_line=dict(color='#26a69a'),
        decreasing_line=dict(color='#ef5350')
    ))
    
    # Add volume as a bar chart
    colors = ['#26a69a' if close >= open else '#ef5350' for open, close in zip(hist_data['Open'], hist_data['Close'])]
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['Volume'],
        name='Volume',
        marker=dict(color=colors, opacity=0.7),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title={
            'text': f'{symbol} Price & Volume',
            'font': {'size': 22, 'color': text_color}
        },
        xaxis_rangeslider_visible=False,
        height=600,
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        font={'color': text_color},
        hovermode='x unified',
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation='h', y=1.02),
        yaxis=dict(
            title='Price',
            side='left',
            gridcolor=grid_color
        ),
        yaxis2=dict(
            title='Volume',
            side='right',
            overlaying='y',
            showgrid=False
        )
    )
    return fig

def main():
    # Sidebar for navigation and settings
    with st.sidebar:
        st.title("📊 Options")
        # Improved theme toggle with a more professional look
        theme = st.radio(
            "Application Theme",
            ["Dark"],
            horizontal=True,
            index=0,
            help="Choose the application theme"
        )
    
    # Main header with professional styling
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <h1 style="margin-bottom: 0;">Stocks Analysis AI Agents</h1>
        <p style="margin-top: 0;">Powered by Groq LLM & yfinance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced CSS for more professional styling
    if theme == "Dark":
        st.markdown("""
        <style>
        .main, .stApp {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #4caf50;
        }
        .card {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .chart-container {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .metrics-container {
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }
        .metric-box {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            flex: 1;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .metric-value {
            color: #4caf50;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #9e9e9e;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton button {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        .stButton button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }
        .section-header {
            color: #4caf50;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        /* Style for input fields and selectbox */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1e1e1e;
            border-color: #333;
        }
        .stTextInput input {
            background-color: #1e1e1e;
            border-color: #333;
            color: #e0e0e0;
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1a1a1a;
        }
        .st-cb, .st-bq, .st-aj, .st-c0 {
            background-color: #1e1e1e;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main, .stApp {
            background-color: #f8f9fa;
            color: #212529;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #0366d6;
        }
        .card {
            background: linear-gradient(145deg, #ffffff, #f5f7fa);
            color: #212529;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .chart-container {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .metrics-container {
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }
        .metric-box {
            background: linear-gradient(145deg, #ffffff, #f5f7fa);
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            flex: 1;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .metric-value {
            color: #0366d6;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #6c757d;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton button {
            background-color: #0366d6;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        .stButton button:hover {
            background-color: #0353b4;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        .section-header {
            color: #0366d6;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Stock selection section with improved styling
    st.markdown("<h3 class='section-header'>Select Stock</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_input = st.selectbox("Select Company Name", 
                                 options=[""] + list(COMMON_STOCKS.keys()), 
                                 help="Select a company or enter a custom stock symbol below.")
    with col2:
        custom_input = st.text_input("Or enter custom stock symbol", 
                                    value="", 
                                    help="e.g., AAPL, TSLA")

    symbol = None
    if stock_input:
        symbol = COMMON_STOCKS.get(stock_input.upper())
    if custom_input:
        symbol = custom_input.strip()

    if st.button("Analyze Stock", use_container_width=True):
        if not symbol:
            st.error("Please select or enter a stock symbol")
            return

        if initialize_agents():
            with st.spinner("Fetching and analyzing stock data..."):
                info, hist = get_stock_data(symbol)

                if info and hist is not None:
                    # Enhanced metrics display
                    st.markdown("<h3 class='section-header'>Key Metrics</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Current Price
                    with col1:
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 'N/A'
                        if current_price != 'N/A':
                            current_price = f"${current_price:,.2f}"
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{current_price}</div>
                            <div class='metric-label'>Current Price</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Market Cap
                    with col2:
                        market_cap = info.get('marketCap')
                        if market_cap:
                            if market_cap >= 1e12:
                                market_cap_str = f"${market_cap/1e12:.2f}T"
                            elif market_cap >= 1e9:
                                market_cap_str = f"${market_cap/1e9:.2f}B"
                            elif market_cap >= 1e6:
                                market_cap_str = f"${market_cap/1e6:.2f}M"
                            else:
                                market_cap_str = f"${market_cap:,.0f}"
                        else:
                            market_cap_str = 'N/A'
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{market_cap_str}</div>
                            <div class='metric-label'>Market Cap</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # P/E Ratio
                    with col3:
                        forward_pe = info.get('forwardPE')
                        pe_str = f"{forward_pe:.2f}" if forward_pe else 'N/A'
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{pe_str}</div>
                            <div class='metric-label'>Forward P/E</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendation
                    with col4:
                        recommendation = info.get('recommendationKey')
                        recommendation_text = recommendation.title() if recommendation else 'N/A'
                        st.markdown(f"""
                        <div class='metric-box'>
                            <div class='metric-value'>{recommendation_text}</div>
                            <div class='metric-label'>Recommendation</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Enhanced chart container
                    st.markdown("<h3 class='section-header'>Price Chart</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                    st.plotly_chart(create_price_chart(hist, symbol, theme), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Company overview with enhanced styling
                    if 'longBusinessSummary' in info and info['longBusinessSummary']:
                        st.markdown("<h3 class='section-header'>Company Overview</h3>", unsafe_allow_html=True)
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.write(info['longBusinessSummary'])
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional financial metrics
                    st.markdown("<h3 class='section-header'>Financial Details</h3>", unsafe_allow_html=True)
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("#### Growth & Valuation")
                        
                        # Create a dataframe for growth metrics
                        growth_metrics = {
                            'Metric': ['Revenue Growth (YoY)', 'EPS Growth (YoY)', 'Profit Margin', '52 Week High', '52 Week Low'],
                            'Value': [
                                f"{info.get('revenueGrowth', 'N/A') * 100:.2f}%" if info.get('revenueGrowth') is not None else 'N/A',
                                f"{info.get('earningsGrowth', 'N/A') * 100:.2f}%" if info.get('earningsGrowth') is not None else 'N/A',
                                f"{info.get('profitMargins', 'N/A') * 100:.2f}%" if info.get('profitMargins') is not None else 'N/A',
                                f"${info.get('fiftyTwoWeekHigh', 'N/A')}" if info.get('fiftyTwoWeekHigh') is not None else 'N/A',
                                f"${info.get('fiftyTwoWeekLow', 'N/A')}" if info.get('fiftyTwoWeekLow') is not None else 'N/A'
                            ]
                        }
                        st.table(pd.DataFrame(growth_metrics))
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metrics_col2:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("#### Dividend & Trading Info")
                        
                        # Create a dataframe for dividend metrics
                        dividend_metrics = {
                            'Metric': ['Dividend Yield', 'Dividend Rate', 'Ex-Dividend Date', 'Average Volume', 'Beta'],
                            'Value': [
                                f"{info.get('dividendYield', 'N/A') * 100:.2f}%" if info.get('dividendYield') is not None else 'N/A',
                                f"${info.get('dividendRate', 'N/A')}" if info.get('dividendRate') is not None else 'N/A',
                                info.get('exDividendDate', 'N/A'),
                                f"{info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') is not None else 'N/A',
                                f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') is not None else 'N/A'
                            ]
                        }
                        st.table(pd.DataFrame(dividend_metrics))
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # AI Analysis Section (Would connect to your AI agents)
                    st.markdown("<h3 class='section-header'>AI Analysis</h3>", unsafe_allow_html=True)
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # This is where your AI analysis would go
                    # For example:
                    if st.button("Generate AI Analysis", key="ai_analysis"):
                        st.write("AI analysis processing...")
                        # Here you would call your AI agents
                        # response = st.session_state.multi_ai_agent.run(f"Analyze the recent performance and prospects of {symbol}.")
                        # st.write(response)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                else:
                    st.error("Failed to retrieve stock data. Please check the stock symbol and try again.")

if __name__ == "__main__":
    main()