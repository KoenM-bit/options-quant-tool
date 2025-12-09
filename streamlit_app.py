"""
Streamlit Dashboard for GEX Positioning Trends
Visualizes gamma exposure, strike roles, and market positioning evolution
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import pyarrow.parquet as pq
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="GEX Positioning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Data path configuration
DATA_DIR = os.getenv('PARQUET_DATA_DIR', 'data/parquet')

@st.cache_data
def load_data():
    """Load the GEX positioning trends data"""
    parquet_path = Path(DATA_DIR) / 'gold_gex_positioning_trends.parquet'
    if not parquet_path.exists():
        st.error(f"❌ Data file not found: {parquet_path}")
        st.info("Please ensure the Parquet files have been exported.")
        st.stop()
    df = pq.read_table(str(parquet_path)).to_pandas()
    return df

@st.cache_data
def load_weekly_gex():
    """Load the weekly GEX data for additional context"""
    parquet_path = Path(DATA_DIR) / 'gold_gamma_exposure_weekly.parquet'
    df = pq.read_table(str(parquet_path)).to_pandas()
    return df

def format_currency(value):
    """Format value in millions of euros"""
    return f"€{value/1e6:.1f}M"

def get_role_color(role):
    """Get color for strike role"""
    colors = {
        'Strong Support': '#00CC66',
        'Support': '#66FF99',
        'Flip Point': '#FFD700',
        'Resistance': '#FF9966',
        'Strong Resistance': '#FF4444'
    }
    return colors.get(role, '#808080')

def main():
    st.title("📊 Gamma Exposure Positioning Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        weekly_gex = load_weekly_gex()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Expiry date selection
    expiry_dates = sorted(df['expiry_date'].unique())
    selected_expiry = st.sidebar.selectbox(
        "Select Expiry Date",
        expiry_dates,
        index=len(expiry_dates) - 1 if len(expiry_dates) > 0 else 0,
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )
    
    # Filter data for selected expiry
    expiry_df = df[df['expiry_date'] == selected_expiry].copy()
    
    # Get latest week
    latest_week = expiry_df['week_ending'].max()
    latest_data = expiry_df[expiry_df['week_ending'] == latest_week]
    
    # Strike selection
    available_strikes = sorted(expiry_df['strike'].unique())
    selected_strikes = st.sidebar.multiselect(
        "Focus Strikes (for detailed analysis)",
        available_strikes,
        default=available_strikes[:3] if len(available_strikes) >= 3 else available_strikes
    )
    
    # Show underlying price context
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Market Context")
    days_to_expiry = (selected_expiry - latest_week.date()).days
    st.sidebar.metric("Days to Expiry", days_to_expiry)
    
    # === OVERVIEW SECTION ===
    st.header(f"📈 Overview: {selected_expiry.strftime('%Y-%m-%d')} Expiry")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gamma_wall = latest_data[latest_data['is_gamma_wall'] == True]
        if not gamma_wall.empty:
            gw_strike = gamma_wall.iloc[0]['strike']
            gw_gex = gamma_wall.iloc[0]['net_dealer_gex']
            st.metric(
                "🎯 Gamma Wall",
                f"Strike {gw_strike:.0f}",
                format_currency(gw_gex)
            )
        else:
            st.metric("🎯 Gamma Wall", "N/A")
    
    with col2:
        flip_points = latest_data[latest_data['is_flip_point'] == True]
        if not flip_points.empty:
            fp_strike = flip_points.iloc[0]['strike']
            st.metric("⚡ Flip Point", f"Strike {fp_strike:.0f}")
        else:
            st.metric("⚡ Flip Point", "N/A")
    
    with col3:
        total_net_gex = latest_data['net_dealer_gex'].sum()
        regime = "Stabilizing" if total_net_gex > 0 else "Amplifying"
        st.metric(
            "🌊 Market Regime",
            regime,
            format_currency(total_net_gex)
        )
    
    with col4:
        total_oi = latest_data['call_oi'].sum() + latest_data['put_oi'].sum()
        pc_ratio = latest_data['put_oi'].sum() / latest_data['call_oi'].sum()
        st.metric(
            "📊 Put/Call Ratio",
            f"{pc_ratio:.2f}",
            f"{total_oi:.0f} OI"
        )
    
    st.markdown("---")
    
    # === GEX EVOLUTION CHART ===
    st.header("📉 GEX Evolution Over Time")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create time series chart for selected strikes
        fig_evolution = go.Figure()
        
        for strike in selected_strikes:
            strike_data = expiry_df[expiry_df['strike'] == strike].sort_values('week_ending')
            
            fig_evolution.add_trace(go.Scatter(
                x=strike_data['week_ending'],
                y=strike_data['net_dealer_gex'] / 1e6,
                mode='lines+markers',
                name=f"Strike {strike:.0f}",
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        # Add zero line
        fig_evolution.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_evolution.update_layout(
            title="Net Dealer GEX by Strike",
            xaxis_title="Week Ending",
            yaxis_title="Net GEX (€M)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)
    
    with col2:
        st.subheader("📊 Latest Week Breakdown")
        
        for strike in selected_strikes:
            strike_latest = latest_data[latest_data['strike'] == strike]
            if not strike_latest.empty:
                row = strike_latest.iloc[0]
                with st.expander(f"Strike {strike:.0f} - {row['strike_role']}", expanded=True):
                    st.metric("Net GEX", format_currency(row['net_dealer_gex']))
                    st.metric("Call OI", f"{row['call_oi']:.0f}")
                    st.metric("Put OI", f"{row['put_oi']:.0f}")
                    st.metric("P/C Ratio", f"{row['put_call_ratio']:.2f}")

                    # New: Flow and bias metrics
                    st.metric("Buyer Flow Bias", f"{row.get('buyer_flow_bias', 50):.0f}/100")
                    st.metric("Break Bias %ile (week)", f"{row.get('break_bias_percentile_week', 0):.0f}")
                    st.metric("Hold Bias %ile (week)", f"{row.get('hold_bias_percentile_week', 0):.0f}")

                    # New: ΔGEX decomposition snapshot
                    if pd.notna(row.get('net_gex_change', None)):
                        oi = row.get('net_gex_change_from_oi', 0.0)
                        gamma = row.get('net_gex_change_from_gamma', 0.0)
                        total = oi + gamma if (oi is not None and gamma is not None) else None
                        if total not in (None, 0):
                            oi_pct = abs(oi) / (abs(oi) + abs(gamma)) * 100 if (abs(oi) + abs(gamma)) > 0 else 0
                            gamma_pct = 100 - oi_pct
                            st.metric("ΔGEX (WoW)", format_currency(row['net_gex_change']))
                            st.metric("From OI (share)", f"{oi_pct:.0f}%")
                            st.metric("From γ (share)", f"{gamma_pct:.0f}%")

                    if row['is_gamma_wall']:
                        st.success("🎯 Gamma Wall")
                    if row['is_flip_point']:
                        st.warning("⚡ Flip Point")
    
    st.markdown("---")
    
    # === STRIKE ROLE HEATMAP ===
    st.header("🎨 Strike Role Evolution Heatmap")
    
    # Create role mapping for numeric representation
    role_map = {
        'Strong Resistance': -2,
        'Resistance': -1,
        'Flip Point': 0,
        'Support': 1,
        'Strong Support': 2
    }
    
    # Prepare data for heatmap
    pivot_data = expiry_df.pivot_table(
        index='strike',
        columns='week_ending',
        values='strike_role',
        aggfunc='first'
    )
    
    # Convert roles to numeric values
    numeric_pivot = pivot_data.map(lambda x: role_map.get(x, 0))
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=numeric_pivot.values,
        x=[d.strftime('%m/%d') for d in numeric_pivot.columns],
        y=numeric_pivot.index,
        colorscale=[
            [0, '#FF4444'],      # Strong Resistance
            [0.25, '#FF9966'],   # Resistance
            [0.5, '#FFD700'],    # Flip Point
            [0.75, '#66FF99'],   # Support
            [1, '#00CC66']       # Strong Support
        ],
        colorbar=dict(
            title="Role",
            tickmode='array',
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=['Strong Resist', 'Resistance', 'Flip Point', 'Support', 'Strong Support']
        ),
        text=pivot_data.values,
        hovertemplate='Strike: %{y}<br>Week: %{x}<br>Role: %{text}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title="How Strike Roles Changed Week-by-Week",
        xaxis_title="Week Ending",
        yaxis_title="Strike Price",
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # === OI FLOW ANALYSIS ===
    st.header("💰 Open Interest Flow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Call Open Interest Trends")
        
        fig_call_oi = go.Figure()
        
        for strike in selected_strikes:
            strike_data = expiry_df[expiry_df['strike'] == strike].sort_values('week_ending')
            
            fig_call_oi.add_trace(go.Bar(
                x=strike_data['week_ending'],
                y=strike_data['call_oi'],
                name=f"Strike {strike:.0f}",
                text=strike_data['call_oi'],
                textposition='auto'
            ))
        
        fig_call_oi.update_layout(
            barmode='group',
            xaxis_title="Week Ending",
            yaxis_title="Call OI",
            height=350
        )
        
        st.plotly_chart(fig_call_oi, use_container_width=True)
    
    with col2:
        st.subheader("Put Open Interest Trends")
        
        fig_put_oi = go.Figure()
        
        for strike in selected_strikes:
            strike_data = expiry_df[expiry_df['strike'] == strike].sort_values('week_ending')
            
            fig_put_oi.add_trace(go.Bar(
                x=strike_data['week_ending'],
                y=strike_data['put_oi'],
                name=f"Strike {strike:.0f}",
                text=strike_data['put_oi'],
                textposition='auto'
            ))
        
        fig_put_oi.update_layout(
            barmode='group',
            xaxis_title="Week Ending",
            yaxis_title="Put OI",
            height=350
        )
        
        st.plotly_chart(fig_put_oi, use_container_width=True)
    
    st.markdown("---")
    
    # === DETAILED STRIKE ANALYSIS ===
    st.header("🔍 Detailed Strike Analysis")
    
    for strike in selected_strikes:
        strike_data = expiry_df[expiry_df['strike'] == strike].sort_values('week_ending')
        
        if not strike_data.empty:
            latest_strike = strike_data.iloc[-1]
            first_strike = strike_data.iloc[0]
            
            with st.expander(f"📌 Strike {strike:.0f} - {latest_strike['strike_role']}", expanded=False):
                
                # Metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    gex_change_pct = latest_strike.get('net_gex_pct_change', 0)
                    st.metric(
                        "Net GEX",
                        format_currency(latest_strike['net_dealer_gex']),
                        f"{gex_change_pct:+.1f}% WoW" if pd.notna(gex_change_pct) else None
                    )
                
                with col2:
                    call_change = latest_strike.get('call_oi_change', 0)
                    st.metric(
                        "Call OI",
                        f"{latest_strike['call_oi']:.0f}",
                        f"{call_change:+.0f}" if pd.notna(call_change) else None
                    )
                
                with col3:
                    put_change = latest_strike.get('put_oi_change', 0)
                    st.metric(
                        "Put OI",
                        f"{latest_strike['put_oi']:.0f}",
                        f"{put_change:+.0f}" if pd.notna(put_change) else None
                    )
                
                with col4:
                    st.metric(
                        "P/C Ratio",
                        f"{latest_strike['put_call_ratio']:.2f}",
                        f"{latest_strike['market_regime']}"
                    )
                
                with col5:
                    st.metric(
                        "Trend",
                        latest_strike['conviction_trend'],
                        latest_strike['gex_trend']
                    )
                
                # Historical trend chart
                fig_strike_detail = go.Figure()
                
                # Add Net GEX
                fig_strike_detail.add_trace(go.Scatter(
                    x=strike_data['week_ending'],
                    y=strike_data['net_dealer_gex'] / 1e6,
                    mode='lines+markers',
                    name='Net GEX',
                    line=dict(width=3, color='blue'),
                    marker=dict(size=10),
                    yaxis='y'
                ))
                
                # New: area chart for ΔGEX from OI vs γ if fields exist
                if 'net_gex_change_from_oi' in strike_data.columns and 'net_gex_change_from_gamma' in strike_data.columns:
                    fig_decomp = go.Figure()

                    fig_decomp.add_trace(go.Bar(
                        x=strike_data['week_ending'],
                        y=strike_data['net_gex_change_from_oi'] / 1e6,
                        name='ΔGEX from OI',
                        marker_color='teal',
                        opacity=0.7
                    ))

                    fig_decomp.add_trace(go.Bar(
                        x=strike_data['week_ending'],
                        y=strike_data['net_gex_change_from_gamma'] / 1e6,
                        name='ΔGEX from γ (time/vol)',
                        marker_color='orange',
                        opacity=0.7
                    ))

                    fig_decomp.update_layout(
                        title=f"Strike {strike:.0f} ΔGEX Decomposition (OI vs γ)",
                        xaxis_title="Week Ending",
                        yaxis_title="ΔGEX (€M)",
                        barmode='relative',
                        height=300,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig_decomp, use_container_width=True)

                # Add Call/Put OI on secondary axis
                fig_strike_detail.add_trace(go.Bar(
                    x=strike_data['week_ending'],
                    y=strike_data['call_oi'],
                    name='Call OI',
                    marker_color='lightgreen',
                    yaxis='y2',
                    opacity=0.6
                ))
                
                fig_strike_detail.add_trace(go.Bar(
                    x=strike_data['week_ending'],
                    y=strike_data['put_oi'],
                    name='Put OI',
                    marker_color='lightcoral',
                    yaxis='y2',
                    opacity=0.6
                ))
                
                fig_strike_detail.update_layout(
                    title=f"Strike {strike:.0f} Complete History",
                    xaxis_title="Week Ending",
                    yaxis=dict(title="Net GEX (€M)", side='left'),
                    yaxis2=dict(title="Open Interest", side='right', overlaying='y'),
                    hovermode='x unified',
                    height=300
                )
                
                st.plotly_chart(fig_strike_detail, use_container_width=True)
                
                # Role shift detection
                role_shifts = strike_data[strike_data['role_shifted'] == True]
                if not role_shifts.empty:
                    st.warning(f"⚠️ **Role Shifts Detected:** {len(role_shifts)} times")
                    for _, shift in role_shifts.iterrows():
                        st.write(f"  • {shift['week_ending'].strftime('%Y-%m-%d')}: "
                                f"{shift['prev_strike_role']} → {shift['strike_role']}")
    
    st.markdown("---")
    
    # === DATA TABLE ===
    st.header("📋 Raw Data Table")
    
    # Show latest week data for all strikes
    display_cols = [
        'strike', 'call_oi', 'put_oi', 'net_dealer_gex', 
        'put_call_ratio', 'strike_role', 'market_regime',
        'conviction_trend', 'gex_trend', 'is_gamma_wall', 'is_flip_point'
    ]
    
    display_data = latest_data[display_cols].copy()
    display_data['net_dealer_gex'] = display_data['net_dealer_gex'].apply(lambda x: f"€{x/1e6:.2f}M")
    display_data = display_data.sort_values('strike')
    
    st.dataframe(
        display_data,
        use_container_width=True,
        height=400
    )
    
    # Download button
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=expiry_df.to_csv(index=False),
        file_name=f"gex_positioning_{selected_expiry.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
