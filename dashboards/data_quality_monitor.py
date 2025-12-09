"""
Data Quality Monitoring Dashboard
Bronze & Silver Layer Visual Inspection

Run with: streamlit run dashboards/data_quality_monitor.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import configuration (handles Docker and local environments)
from src.config import settings

# Database connection - uses settings that work both in Docker and locally
DATABASE_URL = settings.database_url

@contextmanager
def get_db_session():
    """Get database session"""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()

# Page configuration
st.set_page_config(
    page_title="Ahold Options - Data Quality Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quality-badge-good {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .quality-badge-warning {
        background-color: #ffc107;
        color: black;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .quality-badge-danger {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Ahold Options - Data Quality Monitor</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
layer_view = st.sidebar.radio("Select Layer", ["Bronze Layer", "Silver Layer", "Comparison"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Refresh Data")
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

# Get database session
@st.cache_resource
def get_db():
    return get_db_session()

# Bronze Layer Functions
@st.cache_data(ttl=300)
def get_bronze_overview():
    """Get Bronze layer overview statistics"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(DISTINCT DATE(scraped_at)) as scrape_days,
                MIN(scraped_at) as earliest_scrape,
                MAX(scraped_at) as latest_scrape,
                COUNT(DISTINCT expiry_date) as unique_expiries,
                AVG(CASE WHEN bid IS NOT NULL AND ask IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100 as bid_ask_coverage_pct
            FROM bronze_fd_options
        """)).fetchone()
        
        return {
            'total_records': result[0],
            'unique_tickers': result[1],
            'scrape_days': result[2],
            'earliest_scrape': result[3],
            'latest_scrape': result[4],
            'unique_expiries': result[5],
            'bid_ask_coverage_pct': float(result[6])
        }

@st.cache_data(ttl=300)
def get_bronze_daily_scrapes():
    """Get daily scrape counts"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                DATE(scraped_at) as scrape_date,
                COUNT(*) as record_count,
                COUNT(DISTINCT expiry_date) as expiry_count,
                AVG(CASE WHEN bid IS NOT NULL AND ask IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100 as bid_ask_pct
            FROM bronze_fd_options
            WHERE scraped_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(scraped_at)
            ORDER BY scrape_date DESC
        """)).fetchall()
        
        return pd.DataFrame(result, columns=['scrape_date', 'record_count', 'expiry_count', 'bid_ask_pct'])

@st.cache_data(ttl=300)
def get_bronze_latest_sample():
    """Get latest Bronze records sample"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                ticker,
                option_type,
                strike,
                expiry_date,
                bid,
                ask,
                laatste as last_price,
                volume,
                open_interest,
                underlying_price,
                scraped_at
            FROM bronze_fd_options
            WHERE scraped_at = (SELECT MAX(scraped_at) FROM bronze_fd_options)
            ORDER BY expiry_date, strike
            LIMIT 50
        """)).fetchall()
        
        return pd.DataFrame(result, columns=[
            'Ticker', 'Type', 'Strike', 'Expiry', 'Bid', 'Ask', 
            'Last', 'Volume', 'OI', 'Underlying', 'Scraped At'
        ])

# Silver Layer Functions
@st.cache_data(ttl=300)
def get_silver_overview():
    """Get Silver layer overview statistics"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                COUNT(*) as total_options,
                COUNT(DISTINCT trade_date) as trade_days,
                COUNT(implied_volatility) as with_iv,
                COUNT(CASE WHEN greeks_valid = TRUE THEN 1 END) as greeks_valid_count,
                COUNT(CASE WHEN greeks_valid = FALSE THEN 1 END) as greeks_rejected_count,
                ROUND(CAST(AVG(CASE WHEN greeks_valid = TRUE THEN 1.0 ELSE 0.0 END) * 100 AS NUMERIC), 2) as greeks_valid_pct,
                MIN(trade_date) as earliest_trade,
                MAX(trade_date) as latest_trade,
                AVG(risk_free_rate_used) * 100 as avg_risk_free_rate
            FROM silver_options
        """)).fetchone()
        
        return {
            'total_options': result[0],
            'trade_days': result[1],
            'with_iv': result[2],
            'greeks_valid': result[3],
            'greeks_rejected': result[4],
            'greeks_valid_pct': float(result[5]),
            'earliest_trade': result[6],
            'latest_trade': result[7],
            'avg_risk_free_rate': float(result[8]) if result[8] else 0
        }

@st.cache_data(ttl=300)
def get_silver_daily_quality():
    """Get daily Silver quality metrics"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                trade_date,
                COUNT(*) as total,
                COUNT(CASE WHEN greeks_valid = TRUE THEN 1 END) as valid,
                COUNT(CASE WHEN greeks_status = 'rejected' THEN 1 END) as rejected,
                ROUND(CAST(AVG(CASE WHEN greeks_valid = TRUE THEN 1.0 ELSE 0.0 END) * 100 AS NUMERIC), 1) as valid_pct,
                ROUND(CAST(AVG(implied_volatility) * 100 AS NUMERIC), 2) as avg_iv,
                ROUND(CAST(AVG(risk_free_rate_used) * 100 AS NUMERIC), 2) as avg_rate
            FROM silver_options
            WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY trade_date
            ORDER BY trade_date DESC
        """)).fetchall()
        
        return pd.DataFrame(result, columns=[
            'trade_date', 'total', 'valid', 'rejected', 'valid_pct', 'avg_iv', 'avg_rate'
        ])

@st.cache_data(ttl=300)
def get_silver_iv_distribution():
    """Get IV distribution for valid Greeks"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                implied_volatility * 100 as iv_pct,
                option_type,
                days_to_expiry
            FROM silver_options
            WHERE greeks_valid = TRUE
              AND implied_volatility IS NOT NULL
              AND trade_date >= CURRENT_DATE - INTERVAL '7 days'
        """)).fetchall()
        
        return pd.DataFrame(result, columns=['iv_pct', 'option_type', 'days_to_expiry'])

@st.cache_data(ttl=300)
def get_silver_greeks_sample():
    """Get latest Silver Greeks sample"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                ticker,
                option_type,
                strike,
                expiry_date,
                trade_date,
                mid_price,
                underlying_price,
                days_to_expiry,
                implied_volatility * 100 as iv_pct,
                delta,
                gamma,
                theta,
                vega,
                risk_free_rate_used * 100 as rf_pct,
                greeks_valid,
                greeks_status
            FROM silver_options
            WHERE trade_date = (SELECT MAX(trade_date) FROM silver_options)
              AND greeks_valid = TRUE
            ORDER BY expiry_date, strike
            LIMIT 50
        """)).fetchall()
        
        return pd.DataFrame(result, columns=[
            'Ticker', 'Type', 'Strike', 'Expiry', 'Trade Date', 'Mid Price',
            'Underlying', 'DTE', 'IV %', 'Delta', 'Gamma', 'Theta', 'Vega',
            'RF Rate %', 'Valid', 'Status'
        ])

@st.cache_data(ttl=300)
def get_silver_greeks_time_series():
    """Get Greeks time series for visualization"""
    with get_db_session() as session:
        result = session.execute(text("""
            SELECT 
                trade_date,
                option_type,
                AVG(delta) as avg_delta,
                AVG(gamma) as avg_gamma,
                AVG(implied_volatility * 100) as avg_iv
            FROM silver_options
            WHERE greeks_valid = TRUE
              AND trade_date >= CURRENT_DATE - INTERVAL '30 days'
              AND ABS(moneyness - 1.0) < 0.05  -- Near ATM only
            GROUP BY trade_date, option_type
            ORDER BY trade_date, option_type
        """)).fetchall()
        
        return pd.DataFrame(result, columns=[
            'trade_date', 'option_type', 'avg_delta', 'avg_gamma', 'avg_iv'
        ])

@st.cache_data(ttl=300)
def get_atm_iv_trends():
    """Get ATM and neighboring strikes IV trends over time"""
    with get_db_session() as session:
        result = session.execute(text("""
            WITH latest_underlying AS (
                SELECT DISTINCT ON (trade_date, expiry_date)
                    trade_date,
                    expiry_date,
                    underlying_price
                FROM silver_options
                WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY trade_date, expiry_date, updated_at DESC
            ),
            atm_options AS (
                SELECT 
                    s.trade_date,
                    s.expiry_date,
                    s.option_type,
                    s.strike,
                    s.implied_volatility * 100 as iv_pct,
                    lu.underlying_price,
                    s.moneyness,
                    s.days_to_expiry,
                    -- Classify strikes relative to ATM
                    CASE 
                        WHEN ABS(s.moneyness - 1.0) < 0.02 THEN 'ATM'
                        WHEN s.moneyness < 0.95 THEN 'OTM-2'
                        WHEN s.moneyness < 0.98 THEN 'OTM-1'
                        WHEN s.moneyness > 1.05 THEN 'ITM+2'
                        WHEN s.moneyness > 1.02 THEN 'ITM+1'
                        ELSE 'Near-ATM'
                    END as strike_class
                FROM silver_options s
                JOIN latest_underlying lu 
                    ON s.trade_date = lu.trade_date 
                    AND s.expiry_date = lu.expiry_date
                WHERE s.greeks_valid = TRUE
                  AND s.implied_volatility IS NOT NULL
                  AND s.trade_date >= CURRENT_DATE - INTERVAL '30 days'
                  AND s.days_to_expiry BETWEEN 20 AND 60  -- Focus on front month
                  AND ABS(s.moneyness - 1.0) < 0.10  -- Within 10% of ATM
            )
            SELECT 
                trade_date,
                expiry_date,
                option_type,
                strike_class,
                ROUND(AVG(iv_pct)::NUMERIC, 2) as avg_iv,
                ROUND(AVG(underlying_price)::NUMERIC, 2) as underlying_price,
                COUNT(*) as count
            FROM atm_options
            GROUP BY trade_date, expiry_date, option_type, strike_class
            ORDER BY trade_date DESC, expiry_date, 
                     CASE strike_class 
                         WHEN 'OTM-2' THEN 1
                         WHEN 'OTM-1' THEN 2
                         WHEN 'ATM' THEN 3
                         WHEN 'Near-ATM' THEN 4
                         WHEN 'ITM+1' THEN 5
                         WHEN 'ITM+2' THEN 6
                     END,
                     option_type
        """)).fetchall()
        
        return pd.DataFrame(result, columns=[
            'trade_date', 'expiry_date', 'option_type', 'strike_class', 
            'avg_iv', 'underlying_price', 'count'
        ])

# ==================== BRONZE LAYER VIEW ====================
if layer_view == "Bronze Layer":
    st.header("🥉 Bronze Layer - Raw Data Quality")
    st.markdown("Visual inspection of raw scraped options data")
    
    # Get data
    overview = get_bronze_overview()
    daily_scrapes = get_bronze_daily_scrapes()
    latest_sample = get_bronze_latest_sample()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{overview['total_records']:,}")
        st.metric("Scrape Days", f"{overview['scrape_days']:,}")
    
    with col2:
        st.metric("Unique Tickers", overview['unique_tickers'])
        st.metric("Unique Expiries", overview['unique_expiries'])
    
    with col3:
        st.metric("Earliest Scrape", overview['earliest_scrape'].strftime('%Y-%m-%d'))
        st.metric("Latest Scrape", overview['latest_scrape'].strftime('%Y-%m-%d'))
    
    with col4:
        bid_ask_badge = "good" if overview['bid_ask_coverage_pct'] > 80 else "warning" if overview['bid_ask_coverage_pct'] > 50 else "danger"
        st.metric("Bid/Ask Coverage", f"{overview['bid_ask_coverage_pct']:.1f}%")
    
    st.markdown("---")
    
    # Daily scrape trends
    st.subheader("📈 Daily Scrape Volume (Last 30 Days)")
    
    if not daily_scrapes.empty:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Record Count per Day", "Bid/Ask Coverage %"),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_scrapes['scrape_date'],
                y=daily_scrapes['record_count'],
                mode='lines+markers',
                name='Records',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_scrapes['scrape_date'],
                y=daily_scrapes['bid_ask_pct'],
                mode='lines+markers',
                name='Bid/Ask %',
                line=dict(color='#2ca02c', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Coverage %", row=2, col=1)
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Latest scrape sample table
    st.subheader("📋 Latest Scrape Sample (Top 50 Records)")
    
    if not latest_sample.empty:
        # Format numeric columns
        latest_sample['Strike'] = latest_sample['Strike'].apply(lambda x: f"€{float(x):.2f}")
        latest_sample['Bid'] = latest_sample['Bid'].apply(lambda x: f"€{float(x):.4f}" if pd.notna(x) else "—")
        latest_sample['Ask'] = latest_sample['Ask'].apply(lambda x: f"€{float(x):.4f}" if pd.notna(x) else "—")
        latest_sample['Last'] = latest_sample['Last'].apply(lambda x: f"€{float(x):.4f}" if pd.notna(x) else "—")
        latest_sample['Underlying'] = latest_sample['Underlying'].apply(lambda x: f"€{float(x):.2f}" if pd.notna(x) else "—")
        
        st.dataframe(latest_sample, use_container_width=True, height=600)
    else:
        st.warning("No Bronze data available")
    
    # Daily summary table
    st.markdown("---")
    st.subheader("📊 Daily Scrape Summary (Last 30 Days)")
    
    if not daily_scrapes.empty:
        daily_scrapes_display = daily_scrapes.copy()
        daily_scrapes_display['scrape_date'] = pd.to_datetime(daily_scrapes_display['scrape_date']).dt.strftime('%Y-%m-%d')
        daily_scrapes_display['bid_ask_pct'] = daily_scrapes_display['bid_ask_pct'].apply(lambda x: f"{x:.1f}%")
        daily_scrapes_display.columns = ['Date', 'Records', 'Expiries', 'Bid/Ask Coverage']
        st.dataframe(daily_scrapes_display, use_container_width=True, height=400)

# ==================== SILVER LAYER VIEW ====================
elif layer_view == "Silver Layer":
    st.header("🥈 Silver Layer - Greeks Quality & Analytics")
    st.markdown("Deduplicated data with calculated Greeks and quality metrics")
    
    # Get data
    overview = get_silver_overview()
    daily_quality = get_silver_daily_quality()
    iv_dist = get_silver_iv_distribution()
    greeks_sample = get_silver_greeks_sample()
    greeks_ts = get_silver_greeks_time_series()
    
    # Overview metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Options", f"{overview['total_options']:,}")
        st.metric("Trade Days", f"{overview['trade_days']:,}")
    
    with col2:
        st.metric("With IV", f"{overview['with_iv']:,}")
        quality_pct = overview['greeks_valid_pct']
        delta_color = "normal" if quality_pct > 50 else "inverse"
        st.metric("Quality Pass Rate", f"{quality_pct:.1f}%", delta=f"{quality_pct:.1f}%", delta_color=delta_color)
    
    with col3:
        st.metric("Valid Greeks", f"{overview['greeks_valid']:,}", delta="✓ Passed QC")
        st.metric("Rejected", f"{overview['greeks_rejected']:,}", delta="✗ Failed QC", delta_color="inverse")
    
    with col4:
        st.metric("Earliest Trade", overview['earliest_trade'].strftime('%Y-%m-%d'))
        st.metric("Latest Trade", overview['latest_trade'].strftime('%Y-%m-%d'))
    
    with col5:
        st.metric("Avg Risk-Free Rate", f"{overview['avg_risk_free_rate']:.2f}%")
    
    st.markdown("---")
    
    # Greeks Quality Over Time
    st.subheader("📈 Daily Greeks Quality Metrics (Last 30 Days)")
    
    if not daily_quality.empty:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Greeks Validation (Count)",
                "Quality Pass Rate (%)",
                "Average Implied Volatility (%)"
            ),
            vertical_spacing=0.1
        )
        
        # Valid vs Rejected
        fig.add_trace(
            go.Scatter(
                x=daily_quality['trade_date'],
                y=daily_quality['valid'],
                mode='lines+markers',
                name='Valid',
                line=dict(color='#28a745', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_quality['trade_date'],
                y=daily_quality['rejected'],
                mode='lines+markers',
                name='Rejected',
                line=dict(color='#dc3545', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        # Pass rate
        fig.add_trace(
            go.Scatter(
                x=daily_quality['trade_date'],
                y=daily_quality['valid_pct'],
                mode='lines+markers',
                name='Pass Rate',
                line=dict(color='#1f77b4', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1, 
                      annotation_text="50% Target")
        
        # Average IV
        fig.add_trace(
            go.Scatter(
                x=daily_quality['trade_date'],
                y=daily_quality['avg_iv'],
                mode='lines+markers',
                name='Avg IV',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)
        fig.update_yaxes(title_text="IV %", row=3, col=1)
        fig.update_layout(height=900, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # IV Distribution
    st.subheader("📊 Implied Volatility Distribution (Last 7 Days)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not iv_dist.empty:
            fig = px.histogram(
                iv_dist,
                x='iv_pct',
                color='option_type',
                nbins=50,
                title="IV Distribution by Option Type",
                labels={'iv_pct': 'Implied Volatility (%)', 'count': 'Count'},
                color_discrete_map={'Call': '#1f77b4', 'Put': '#ff7f0e'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not iv_dist.empty:
            fig = px.box(
                iv_dist,
                x='option_type',
                y='iv_pct',
                color='option_type',
                title="IV Box Plot by Type",
                labels={'iv_pct': 'Implied Volatility (%)', 'option_type': 'Type'},
                color_discrete_map={'Call': '#1f77b4', 'Put': '#ff7f0e'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ATM IV Trends
    st.subheader("🎯 ATM & Neighboring Strikes IV Trends (20-60 DTE)")
    st.markdown("Track implied volatility across ATM and neighboring strikes to spot IV skew patterns")
    
    atm_iv = get_atm_iv_trends()
    
    if not atm_iv.empty:
        # Get latest expiry for IV smile chart
        latest_date = atm_iv['trade_date'].max()
        latest_expiry = atm_iv[atm_iv['trade_date'] == latest_date]['expiry_date'].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # IV Time Series by Strike Class
            st.markdown(f"**IV Trends Over Time** (Latest Expiry: {latest_expiry.strftime('%Y-%m-%d')})")
            
            # Filter for calls only to reduce clutter, and ATM + neighbors
            atm_calls = atm_iv[
                (atm_iv['option_type'] == 'Call') & 
                (atm_iv['strike_class'].isin(['OTM-1', 'ATM', 'ITM+1']))
            ].copy()
            
            if not atm_calls.empty:
                fig = px.line(
                    atm_calls,
                    x='trade_date',
                    y='avg_iv',
                    color='strike_class',
                    title="ATM Call IV Trends (OTM-1, ATM, ITM+1)",
                    labels={'avg_iv': 'Implied Volatility (%)', 'trade_date': 'Date', 'strike_class': 'Strike'},
                    color_discrete_map={
                        'OTM-2': '#d62728',
                        'OTM-1': '#ff7f0e', 
                        'ATM': '#2ca02c',
                        'Near-ATM': '#bcbd22',
                        'ITM+1': '#1f77b4',
                        'ITM+2': '#9467bd'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ATM call data available for this period")
        
        with col2:
            # IV Smile/Skew - Latest Date
            st.markdown(f"**Current IV Smile** (Latest: {latest_date.strftime('%Y-%m-%d')})")
            
            latest_iv = atm_iv[atm_iv['trade_date'] == latest_date].copy()
            
            if not latest_iv.empty:
                # Order by strike class properly
                strike_order = ['OTM-2', 'OTM-1', 'ATM', 'Near-ATM', 'ITM+1', 'ITM+2']
                latest_iv['strike_class'] = pd.Categorical(
                    latest_iv['strike_class'], 
                    categories=strike_order, 
                    ordered=True
                )
                latest_iv = latest_iv.sort_values('strike_class')
                
                fig = px.line(
                    latest_iv,
                    x='strike_class',
                    y='avg_iv',
                    color='option_type',
                    markers=True,
                    title=f"IV Smile (Expiry: {latest_expiry.strftime('%b %d')})",
                    labels={'avg_iv': 'Implied Volatility (%)', 'strike_class': 'Strike Class'},
                    color_discrete_map={'Call': '#1f77b4', 'Put': '#ff7f0e'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent IV data available")
        
        # Summary table of latest ATM IVs
        st.markdown("**📊 Latest ATM IV Summary**")
        
        latest_summary = atm_iv[atm_iv['trade_date'] == latest_date].copy()
        if not latest_summary.empty:
            pivot_table = latest_summary.pivot_table(
                index='strike_class',
                columns='option_type',
                values='avg_iv',
                aggfunc='mean'
            ).round(2)
            
            # Reorder rows
            strike_order = ['OTM-2', 'OTM-1', 'ATM', 'Near-ATM', 'ITM+1', 'ITM+2']
            pivot_table = pivot_table.reindex([s for s in strike_order if s in pivot_table.index])
            
            # Add underlying price
            underlying = latest_summary['underlying_price'].iloc[0]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(pivot_table, use_container_width=True)
            with col2:
                st.metric("Underlying Price", f"€{underlying:.2f}")
                if 'ATM' in pivot_table.index:
                    atm_call_iv = pivot_table.loc['ATM', 'Call'] if 'Call' in pivot_table.columns else None
                    atm_put_iv = pivot_table.loc['ATM', 'Put'] if 'Put' in pivot_table.columns else None
                    if atm_call_iv and atm_put_iv:
                        st.metric("ATM Call IV", f"{atm_call_iv:.2f}%")
                        st.metric("ATM Put IV", f"{atm_put_iv:.2f}%")
                        st.metric("Put-Call Skew", f"{(atm_put_iv - atm_call_iv):.2f}%")
    else:
        st.info("No ATM IV data available for the last 30 days")
    
    st.markdown("---")
    
    # Greeks Time Series
    st.subheader("📈 ATM Greeks Time Series (Last 30 Days)")
    
    if not greeks_ts.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Delta by type
            fig = px.line(
                greeks_ts,
                x='trade_date',
                y='avg_delta',
                color='option_type',
                title="Average Delta (ATM Options)",
                labels={'avg_delta': 'Delta', 'trade_date': 'Date'},
                color_discrete_map={'Call': '#1f77b4', 'Put': '#ff7f0e'}
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="0.5")
            fig.add_hline(y=-0.5, line_dash="dash", line_color="gray", annotation_text="-0.5")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gamma
            fig = px.line(
                greeks_ts,
                x='trade_date',
                y='avg_gamma',
                color='option_type',
                title="Average Gamma (ATM Options)",
                labels={'avg_gamma': 'Gamma', 'trade_date': 'Date'},
                color_discrete_map={'Call': '#1f77b4', 'Put': '#ff7f0e'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Latest Greeks Sample Table
    st.subheader("📋 Latest High-Quality Greeks (Top 50)")
    
    if not greeks_sample.empty:
        # Format numeric columns
        greeks_sample['Strike'] = greeks_sample['Strike'].apply(lambda x: f"€{float(x):.2f}")
        greeks_sample['Mid Price'] = greeks_sample['Mid Price'].apply(lambda x: f"€{float(x):.4f}" if pd.notna(x) else "—")
        greeks_sample['Underlying'] = greeks_sample['Underlying'].apply(lambda x: f"€{float(x):.2f}" if pd.notna(x) else "—")
        greeks_sample['IV %'] = greeks_sample['IV %'].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—")
        greeks_sample['Delta'] = greeks_sample['Delta'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
        greeks_sample['Gamma'] = greeks_sample['Gamma'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
        greeks_sample['Theta'] = greeks_sample['Theta'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
        greeks_sample['Vega'] = greeks_sample['Vega'].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "—")
        greeks_sample['RF Rate %'] = greeks_sample['RF Rate %'].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—")
        
        st.dataframe(greeks_sample, use_container_width=True, height=600)
    else:
        st.warning("No Silver data with valid Greeks available")
    
    # Daily quality summary table
    st.markdown("---")
    st.subheader("📊 Daily Quality Summary (Last 30 Days)")
    
    if not daily_quality.empty:
        daily_display = daily_quality.copy()
        daily_display['trade_date'] = pd.to_datetime(daily_display['trade_date']).dt.strftime('%Y-%m-%d')
        daily_display['valid_pct'] = daily_display['valid_pct'].apply(lambda x: f"{float(x):.1f}%")
        daily_display['avg_iv'] = daily_display['avg_iv'].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—")
        daily_display['avg_rate'] = daily_display['avg_rate'].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "—")
        daily_display.columns = ['Date', 'Total', 'Valid', 'Rejected', 'Pass Rate', 'Avg IV', 'Avg RF Rate']
        st.dataframe(daily_display, use_container_width=True, height=400)

# ==================== COMPARISON VIEW ====================
else:  # Comparison
    st.header("⚖️ Bronze vs Silver - Data Flow & Quality")
    st.markdown("Compare raw data ingestion with processed analytics")
    
    # Get overview data
    bronze_overview = get_bronze_overview()
    silver_overview = get_silver_overview()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥉 Bronze Layer")
        st.metric("Total Records", f"{bronze_overview['total_records']:,}")
        st.metric("Scrape Days", f"{bronze_overview['scrape_days']:,}")
        st.metric("Bid/Ask Coverage", f"{bronze_overview['bid_ask_coverage_pct']:.1f}%")
        st.info("**Bronze** stores all raw scrapes (multiple per day)")
    
    with col2:
        st.subheader("🥈 Silver Layer")
        st.metric("Total Options", f"{silver_overview['total_options']:,}")
        st.metric("Trade Days", f"{silver_overview['trade_days']:,}")
        st.metric("Greeks Valid", f"{silver_overview['greeks_valid_pct']:.1f}%")
        st.info("**Silver** deduplicates and enriches with Greeks")
    
    st.markdown("---")
    
    # Data flow visualization
    st.subheader("📊 Data Pipeline Flow")
    
    flow_data = {
        'Stage': ['Bronze (Raw)', 'Silver (Deduplicated)', 'Silver (Valid Greeks)'],
        'Count': [
            bronze_overview['total_records'],
            silver_overview['total_options'],
            silver_overview['greeks_valid']
        ]
    }
    flow_df = pd.DataFrame(flow_data)
    
    fig = go.Figure(data=[
        go.Funnel(
            y=flow_df['Stage'],
            x=flow_df['Count'],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        )
    ])
    fig.update_layout(height=400, title="Data Pipeline Funnel")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quality comparison metrics
    st.subheader("🎯 Quality Metrics Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dedup_ratio = (silver_overview['total_options'] / bronze_overview['total_records']) * 100
        st.metric(
            "Deduplication Ratio",
            f"{dedup_ratio:.1f}%",
            help="Percentage of Bronze records that make it to Silver after deduplication"
        )
    
    with col2:
        greeks_coverage = (silver_overview['greeks_valid'] / silver_overview['total_options']) * 100
        st.metric(
            "Greeks Coverage",
            f"{greeks_coverage:.1f}%",
            help="Percentage of Silver records with valid Greeks"
        )
    
    with col3:
        end_to_end = (silver_overview['greeks_valid'] / bronze_overview['total_records']) * 100
        st.metric(
            "End-to-End Quality",
            f"{end_to_end:.1f}%",
            help="Percentage of Bronze records that result in valid Greeks"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Ahold Options Data Quality Monitor | Updated: {}</p>
        <p>Bronze: Raw scraped data | Silver: Deduplicated + Greeks | Gold: Analytics</p>
    </div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
