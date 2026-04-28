import streamlit as st
import pandas as pd
import plotly.express as px
try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from cassandra.cluster import Cluster
except ImportError:
    Cluster = None
import os

# --- Configuration ---
st.set_page_config(page_title="Sales Intelligence Dashboard", layout="wide")
st.title("📊 Sales Forecasting & Demand Intelligence System")

CASSANDRA_HOST = '127.0.0.1'
CASSANDRA_KEYSPACE = 'sales_intelligence'
MONGO_URI = 'mongodb://127.0.0.1:27017/'
MONGO_DB = 'sales_intelligence'

@st.cache_resource
def init_connections():
    cass_session = None
    mongo_db = None
    
    try:
        if Cluster:
            cluster = Cluster([CASSANDRA_HOST])
            cass_session = cluster.connect(CASSANDRA_KEYSPACE)
    except Exception:
        pass
        
    try:
        if MongoClient:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            mongo_db = client[MONGO_DB]
    except Exception:
        pass
        
    return cass_session, mongo_db

cass_session, mongo_db = init_connections()

# --- Data Fetching ---
@st.cache_data(ttl=60)
def load_historical_data():
    if cass_session:
        try:
            rows = cass_session.execute("SELECT * FROM historical_sales")
            df = pd.DataFrame(list(rows))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df
        except Exception as e:
            st.warning(f"Failed to read from Cassandra: {e}")
            
    # Fallback to CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    csv_path_fallback = os.path.join(os.path.dirname(__file__), '..', 'data', 'store1_2017.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(csv_path_fallback):
        df = pd.read_csv(csv_path_fallback)
    else:
        return pd.DataFrame()
        
    df['date'] = pd.to_datetime(df['date'])
    # Sample it for frontend performance if it's huge
    if len(df) > 100000:
        df = df[df['date'].dt.year == 2017]
    return df

@st.cache_data(ttl=60)
def load_forecasts():
    if mongo_db is not None:
        try:
            cursor = mongo_db.forecasts.find({}, {'_id': 0})
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                df['forecast_date'] = pd.to_datetime(df['forecast_date'])
                return df
        except Exception:
            pass
            
    # Mock fallback if DB is not available
    import datetime
    today = datetime.date(2017, 8, 15)
    dates = [today + datetime.timedelta(days=i) for i in range(1, 8)]
    mock_f = []
    for d in dates:
        mock_f.append({'family': 'AUTOMOTIVE', 'store_nbr': 1, 'forecast_date': pd.to_datetime(d), 'predicted_sales': 5 + d.day%3})
        mock_f.append({'family': 'GROCERY I', 'store_nbr': 1, 'forecast_date': pd.to_datetime(d), 'predicted_sales': 3000 - d.day%100})
    return pd.DataFrame(mock_f)

@st.cache_data(ttl=60)
def load_alerts():
    if mongo_db is not None:
        try:
            cursor = mongo_db.alerts.find({}, {'_id': 0})
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                df['alert_date'] = pd.to_datetime(df['alert_date'])
                return df
        except Exception:
            pass
            
    # Mock fallback alerts
    return pd.DataFrame([
        {'family': 'GROCERY I', 'store_nbr': 1, 'alert_date': pd.to_datetime('2017-08-16'), 'alert_type': 'RESTOCK', 'message': 'High demand predicted (3000). Check stock levels.'},
        {'family': 'AUTOMOTIVE', 'store_nbr': 1, 'alert_date': pd.to_datetime('2017-08-17'), 'alert_type': 'SPIKE', 'message': 'Demand Spike: 35% increase expected.'}
    ])


# --- Main UI ---
hist_df = load_historical_data()
forecast_df = load_forecasts()
alerts_df = load_alerts()

if hist_df.empty:
    st.error("No historical data available. Ensure Cassandra is running or train.csv exists.")
    st.stop()

# --- Top KPIs ---
total_sales = hist_df['sales'].sum()
total_products = hist_df['family'].nunique()
total_stores = hist_df['store_nbr'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Historical Sales", f"{total_sales:,.0f}")
col2.metric("Product Families", total_products)
col3.metric("Stores", total_stores)

st.markdown("---")

col_main, col_sidebar = st.columns([3, 1])

with col_main:
    st.subheader("📈 Sales Trends & Forecasts")
    
    # Filter controls
    selected_store = st.selectbox("Select Store", hist_df['store_nbr'].unique())
    selected_product = st.selectbox("Select Product Family", hist_df['family'].unique())
    
    # Filter Data
    h_filtered = hist_df[(hist_df['store_nbr'] == selected_store) & (hist_df['family'] == selected_product)].copy()
    h_filtered = h_filtered.sort_values('date')
    
    # Plotly Chart
    fig = px.line(h_filtered, x='date', y='sales', title=f"Sales for {selected_product} in Store {selected_store}")
    fig.data[0].name="Historical Sales"
    fig.data[0].showlegend=True
    
    if not forecast_df.empty:
        f_filtered = forecast_df[(forecast_df['store_nbr'] == selected_store) & (forecast_df['family'] == selected_product)].copy()
        if not f_filtered.empty:
            f_filtered = f_filtered.sort_values('forecast_date')
            fig.add_scatter(x=f_filtered['forecast_date'], y=f_filtered['predicted_sales'], 
                            mode='lines+markers', name='Forecasted Sales',
                            line=dict(dash='dash', color='orange'))
            
    st.plotly_chart(fig, use_container_width=True)

with col_sidebar:
    st.markdown("### 🚨 Recent Alerts")
    with st.container(height=250):
        if not alerts_df.empty:
            for _, row in alerts_df.iterrows():
                icon = "📦" if row['alert_type'] == 'RESTOCK' else "🔥" if row['alert_type'] == 'SPIKE' else "📉"
                st.markdown(f"**{icon} {row['alert_type']}**: {row['family']} (Store {row['store_nbr']})")
        else:
            st.success("No alerts generated.")
        
st.markdown("---")
col_raw1, col_raw2 = st.columns(2)
with col_raw1:
    st.subheader("📋 Raw Data View")
    st.dataframe(h_filtered.tail(10))

with col_raw2:
    if not forecast_df.empty:
        st.subheader("🔮 Forecast Data")
        st.dataframe(forecast_df.tail(10))

st.markdown("---")
st.subheader("🔮 Interactive Predictor")
st.write(f"Test out 'What-if' scenarios for **{selected_product}** at **Store {selected_store}** using a dynamic AI trained instantly on the data above!")

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    pred_date = st.date_input("Select Date to Predict", value=pd.to_datetime('2017-09-01').date())
with col_p2:
    is_promo = st.checkbox("Is it a Promotion Day?")
with col_p3:
    st.write("") # spacing
    st.write("")
    if st.button("Predict Sales!", use_container_width=True, type="primary"):
        from sklearn.ensemble import RandomForestRegressor
        
        # Train a quick model on selected data
        train_data = h_filtered.copy()
        train_data['day_of_year'] = train_data['date'].dt.dayofyear
        train_data['day_of_week'] = train_data['date'].dt.dayofweek
        train_data['is_promo'] = train_data['onpromotion'].apply(lambda x: 1 if x > 0 else 0)
        
        X = train_data[['day_of_year', 'day_of_week', 'is_promo']]
        y = train_data['sales']
        
        if len(train_data) > 10:
            with st.spinner("Training model instantly..."):
                rf = RandomForestRegressor(n_estimators=20, random_state=42)
                rf.fit(X, y)
                
                # Predict
                pred_X = pd.DataFrame({
                    'day_of_year': [pred_date.timetuple().tm_yday],
                    'day_of_week': [pred_date.weekday()],
                    'is_promo': [1 if is_promo else 0]
                })
                predicted_val = rf.predict(pred_X)[0]
                
            st.success(f"### Predicted Sales: **{predicted_val:,.0f}** units")
            st.balloons()
        else:
            st.warning("Not enough historical data to train the interactive model. Select a product with more data.")
