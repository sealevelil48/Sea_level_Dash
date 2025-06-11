# ---------------------------
# BASE IMPORTS & CONFIG
# ---------------------------
import os
import gc
import logging
import warnings
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, select, and_, text, MetaData, Table, Column, TypeDecorator, String, exc, func
from sqlalchemy.types import TypeEngine
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from dotenv import load_dotenv

import plotly.graph_objs as go
import plotly.express as px

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
from dash import dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from flask import Flask
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific SAWarnings
warnings.filterwarnings("ignore",
                        category=exc.SAWarning,
                        message="Did not recognize type 'point' of column 'locations'")

# ---------------------------
# DATABASE CONFIGURATION
# ---------------------------
DB_URI = os.getenv('DB_URI', "postgresql://postgres:sealevel123@localhost:5432/Test2-SeaLevels")
engine = create_engine(DB_URI, pool_pre_ping=True, pool_size=15, max_overflow=30, pool_recycle=3600)


class PointType(TypeDecorator):
    """Handles PostgreSQL POINT type"""
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return f"{value[0]},{value[1]}" if value else None

    def process_result_value(self, value, dialect):
        return tuple(map(float, value.split(','))) if value else None

    # Initialize with error handling
try:
        metadata = MetaData()

        M = Table('Monitors_info2', metadata,
                  Column('Tab_TabularTag', String),  # Explicitly defining for join
                  autoload_with=engine,
                  extend_existing=True)

        L = Table('Locations', metadata,
                  Column('locations', PointType()),  # PointType is now defined and accessible
                  Column('Station', String),  # Explicitly defining Station column
                  autoload_with=engine,
                  extend_existing=True)

        S = Table('SeaTides', metadata,
                  autoload_with=engine,
                  extend_existing=True)

except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        M, L, S = None, None, None

class SeaLevelDashboard:
    def __init__(self, engine):
        """Initialize dashboard with resource management"""
        self.engine = engine  # Store engine as an instance attribute
        self.df_melted = pd.DataFrame()
        self.stations_cache = None
        self.last_stations_fetch = None
        self.current_graph_data = None  # Will store the figure object for graph export
        self.current_table_df_for_export = pd.DataFrame()  # Will store DataFrame for table export
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.scaler = MinMaxScaler()

        # Initialize models
        self.prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        self.iso_forest = IsolationForest(contamination=0.01, random_state=42)
        self.lstm_model = self.build_lstm_model()

        # Flask server setup
        self.flask_server = Flask(__name__)
        self.flask_server.after_request(self.add_cors_headers)
        self.load_initial_data()

    # ---------------------------
    # DATABASE OPERATIONS
    # ---------------------------
    def load_initial_data(self):
        """Initialize core dataset"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            # Load data for the graph/table initially
            self.df_melted = self.load_data_from_db(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                station='All Stations',  # Default station
                data_source='default'  # Default data source
            )
            logging.info("Initial data loaded successfully")
        except Exception as e:
            logging.error(f"Initial data load failed: {e}")
            self.df_melted = pd.DataFrame()

    def fetch_station_locations(self):
        """Fetches distinct station locations as DataFrame"""
        try:
            # Check cache first
            if hasattr(self, 'station_locations_cache') and self.station_locations_cache is not None:
                if (datetime.now() - self.last_stations_locations_fetch) < timedelta(hours=1):
                    return self.station_locations_cache

            logging.info("Fetching station locations from DB for map display...")

            # Build query to extract point coordinates using text-based SQL
            query = text("""
                SELECT DISTINCT "Station", 
                       (locations::point)[0] AS latitude, 
                       (locations::point)[1] AS longitude
                FROM "Locations" 
                WHERE locations IS NOT NULL
            """)

            # Execute using SQLAlchemy connection
            with self.engine.connect() as connection:
                result = connection.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            if df.empty:
                logging.warning("No station location data fetched for map")
                return pd.DataFrame(columns=["Station", "longitude", "latitude"])

            # Cache results
            self.station_locations_cache = df
            self.last_stations_locations_fetch = datetime.now()
            return df

        except Exception as e:
            logging.error(f"Error fetching locations: {e}")
            return pd.DataFrame(columns=["Station", "longitude", "latitude"])


    def load_data_from_db(self, start_date=None, end_date=None, station=None, data_source='default'):
        """Optimized database query execution"""
        if self.engine is None or M is None or L is None or S is None:
            logging.error("Database engine or tables not loaded. Cannot query data.")
            return pd.DataFrame()

        try:
            sql_query_obj = self.build_query(start_date, end_date, station, data_source)
            if sql_query_obj is None:
                logging.error("build_query returned None. Cannot execute query.")
                return pd.DataFrame()

            df_chunks = []
            # CORRECTED: Use SQLAlchemy connection properly
            with self.engine.connect() as connection:
                # Execute the query and get a result proxy
                result = connection.execute(sql_query_obj)

                # Fetch data in chunks
                while True:
                    chunk = result.fetchmany(10000)
                    if not chunk:
                        break
                    df_chunk = pd.DataFrame(chunk, columns=result.keys())
                    df_chunks.append(df_chunk)

            if not df_chunks:
                return pd.DataFrame()
            return pd.concat(df_chunks, ignore_index=True)

        except Exception as e:
            logging.error(f"Data load error: {e}")
            return pd.DataFrame()

    def build_query(self, start_date, end_date, station, data_source):
        """SQL query construction"""
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if station and station != 'All Stations':
            params['station'] = station

        stmt = None  # Initialize stmt to None
        table_for_date_filter = None  # This variable will hold the correct table for date column access
        date_col_name = None  # Initialize date_col_name

        if data_source == 'tides':
            cols_to_select = self.tides_columns()
            table_for_date_filter = S  # For 'tides', the date column is in table S
            date_col_name = 'Date'
            stmt = select(*cols_to_select).select_from(S)  # Initial query for tides

        else:  # This is the 'default' data_source
            cols_to_select = self.default_columns()
            join_condition = M.c.Tab_TabularTag == L.c.Tab_TabularTag
            stmt = select(*cols_to_select).select_from(M.join(L, join_condition))
            table_for_date_filter = M  # For 'default', the date column 'Tab_DateTime' is in table M
            date_col_name = 'Tab_DateTime'

        # Apply date filters using the correctly assigned table_for_date_filter
        if start_date:
            stmt = stmt.where(
                table_for_date_filter.c[date_col_name] >= params['start_date']
            )
        if end_date:
            stmt = stmt.where(
                table_for_date_filter.c[date_col_name] <= params['end_date']
            )

        # Apply station filter if 'default' data_source and station is selected
        if data_source == 'default' and 'station' in params:
            # The 'Station' column is in the 'L' (Locations) table
            stmt = stmt.where(L.c.Station == params['station'])
        elif data_source == 'tides' and 'station' in params:
            # For 'tides' source, station filtering should be on S.c.Station
            stmt = stmt.where(S.c.Station == params['station'])


        # Add ordering, ensuring date_col_name is defined before trying to use it
        if date_col_name and table_for_date_filter is not None:
            stmt = stmt.order_by(table_for_date_filter.c[date_col_name])

        return stmt

    def build_where_clause(self, start_date, end_date, station, date_column_selectable):
        """Condition builder for SQL queries"""
        conditions = []
        if start_date:
            conditions.append(date_column_selectable >= text(':start_date'))
        if end_date:
            conditions.append(date_column_selectable <= text(':end_date'))
        if station and station != 'All Stations':
            # This method just gets the conditions.
            # The original code used text('"Station" = :station'). This refers to a column named "Station".
            # If L.c.Station and S.c.Station are indeed named 'Station' in the DB, it's fine.
            # Let's assume this is the case.
            conditions.append(text('"Station" = :station'))  # Reverting to original simple text for now

        return and_(*conditions) if conditions else True  # True means no conditions effectively

    def default_columns(self):
        # Ensure these columns exist in M and L respectively and can be joined.
        return [M.c.Tab_DateTime, L.c.Station, M.c.Tab_Value_mDepthC1,
                M.c.Tab_Value_monT2m,
                #L.c.X, L.c.Y, L.c.Longitude, L.c.Latitude
                ]

    def tides_columns(self):
        return [S.c.Date, S.c.Station, S.c.HighTide, S.c.HighTideTime,
                S.c.HighTideTemp, S.c.LowTide, S.c.LowTideTime,
                S.c.LowTideTemp, S.c.MeasurementCount]

    # ---------------------------
    # DATA PROCESSING & LSTM
    # ---------------------------
    def create_sequences(self, data, seq_length=24):
        """Create time series sequences for LSTM"""
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            targets.append(data[i + seq_length])
        return np.array(sequences), np.array(targets)

    def prepare_data_for_lstm(self, df, column='Tab_Value_mDepthC1', seq_length=24):
        """Prepare normalized LSTM data"""
        if df.empty or column not in df.columns or len(
                df) < seq_length + 1:  # Need at least seq_length + 1 for one target
            logging.warning(f"Not enough data for LSTM. Need at least {seq_length + 1} points. Got {len(df)}.")
            # Return empty arrays or raise error, consistent with create_sequences
            return np.array([]), np.array([]), np.array([]), np.array([]), self.scaler

        values = df[column].values.reshape(-1, 1)
        normalized = self.scaler.fit_transform(values)
        X, y = self.create_sequences(normalized, seq_length)

        if len(X) == 0:  # If create_sequences returned empty (e.g. len(data) was exactly seq_length)
            logging.warning("LSTM sequence creation resulted in empty X. Check data length and seq_length.")
            return np.array([]), np.array([]), np.array([]), np.array([]), self.scaler

        train_size = int(len(X) * 0.8)
        return X[:train_size], X[train_size:], y[:train_size], y[train_size:], self.scaler

    def build_lstm_model(self, seq_length=24):
        """LSTM model architecture"""
        model = keras.Sequential([
            keras.layers.LSTM(50, activation='relu',
                              input_shape=(seq_length, 1),
                              return_sequences=True),
            keras.layers.LSTM(50, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # ---------------------------
    # PREDICTION AND UTILITY METHODS
    # ---------------------------
    @lru_cache(maxsize=4)
    def get_prediction_data(self, station): # Cache results for a few recent stations
        """Get last 365 days of data for predictions"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        # Ensure data_source is 'default' for sea level predictions
        return self.load_data_from_db(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            station=station,
            data_source='default'
        )

    @lru_cache(maxsize=2)  # Cache for the last 2 station predictions
    def arima_predict(self, station):
        """Generates ARIMA predictions."""
        try:
            # Fetch relevant data for the prediction
            df = self.get_prediction_data(station)
            if df.empty or 'Tab_Value_mDepthC1' not in df.columns:
                logging.warning(f"No data available for ARIMA prediction for station {station}.")
                return None

            # Sort by DateTime to ensure correct time series order
            df['Tab_DateTime'] = pd.to_datetime(df['Tab_DateTime'])
            df = df.sort_values('Tab_DateTime')
            df = df.set_index('Tab_DateTime')


            # --- NEW: Resample data for ARIMA ---
            # Resample to hourly data, taking the mean for each hour
            # Change 'H' to 'D' for daily if your data is very high frequency and hourly is still too much.
            series_to_predict = df['Tab_Value_mDepthC1'].resample('h').mean().dropna()

            # Ensure enough data points after resampling
            if len(series_to_predict) < 20:
                logging.warning(
                    f"Not enough valid data points ({len(series_to_predict)}) after resampling for ARIMA prediction for station {station}.")
                return None

            model = ARIMA(series_to_predict, order=(5, 1, 0))
            model_fit = model.fit()
            return model_fit.forecast(steps=240)  # Forecast 240 hours
        except Exception as e:
            logging.error(f"ARIMA prediction failed for station {station}: {str(e)}")
            return None

            # Ensure the series is stationary or apply differencing as part of the model order
            # For simplicity, using (5,1,0) as in old version, implying 1st order differencing
            model = ARIMA(df['Tab_Value_mDepthC1'], order=(5, 1, 0))
            model_fit = model.fit()
            return model_fit.forecast(steps=240) # Predict next 240 steps (e.g., 10 days at 1-hour intervals)
        except Exception as e:
            logging.error(f"ARIMA prediction failed for station {station}: {str(e)}")
            return None

    @lru_cache(maxsize=2)
    def prophet_predict(self, station):
        """Generates Prophet predictions (cached) with data resampling)."""
        try:
            df = self.get_prediction_data(station)
            if df.empty or 'Tab_Value_mDepthC1' not in df.columns:
                logging.warning(
                    f"Prophet: No data available from get_prediction_data for prediction for station {station}.")
                return None

            df['Tab_DateTime'] = pd.to_datetime(df['Tab_DateTime'])
            df = df.sort_values('Tab_DateTime')
            df = df.set_index('Tab_DateTime')

            prophet_df = df['Tab_Value_mDepthC1'].resample('h').mean().reset_index()
            prophet_df = prophet_df.rename(columns={'Tab_DateTime': 'ds', 'Tab_Value_mDepthC1': 'y'})[
                ['ds', 'y']].dropna()

            logging.info(f"Prophet: Data points after resampling and dropna for station {station}: {len(prophet_df)}")

            if len(prophet_df) < 50:
                logging.warning(
                    f"Prophet: Not enough valid data points ({len(prophet_df)}) after resampling for prediction for station {station}. Prophet needs at least 50 points for reliable results.")
                return None

            # Initialize forecast to None or an empty DataFrame before trying to assign it
            forecast = None
            model = Prophet(yearly_seasonality=True, daily_seasonality=True, growth='linear')
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=240, freq='h')

            # Check if future dataframe is empty. Prophet might have issues predicting for empty 'future'.
            if future.empty:
                logging.warning(f"Prophet: Future dataframe is empty for station {station}. Cannot generate forecast.")
                return None

            forecast = model.predict(future)

            # --- NEW CHECKS ---
            if forecast is None or forecast.empty or 'yhat' not in forecast.columns:
                logging.warning(
                    f"Prophet: Forecast result is invalid (None, empty, or missing 'yhat') for station {station}.")
                return None
            if forecast['yhat'].isnull().all():
                logging.warning(f"Prophet: All 'yhat' values in forecast are NaN for station {station}.")
                return None

            logging.info(
                f"Prophet: Forecast generated for station {station}. Length: {len(forecast)}, First 'ds': {forecast['ds'].min()}, Last 'ds': {forecast['ds'].max()}")
            logging.debug(f"Prophet Forecast Head for {station}:\n{forecast[['ds', 'yhat']].head()}")
            logging.debug(f"Prophet Forecast Tail for {station}:\n{forecast[['ds', 'yhat']].tail()}")
            logging.debug(f"Prophet Forecast yhat NaNs for {station}: {forecast['yhat'].isnull().sum()}")

            return forecast[['ds', 'yhat']]
        except Exception as e:
            # If an error happens *after* `forecast` is assigned but before return,
            # it will be caught here. If it happens before, the 'forecast' variable might still be unassigned
            # but the `except` block catches it.
            logging.error(f"Prophet prediction failed for station {station}: {str(e)}")
            return None

    def get_stations(self):
        """Fetches station list from the database."""
        if L is not None:
            try:
                # Cache station list for a period if it doesn't change often
                if self.stations_cache is not None and \
                        (self.last_stations_fetch is None or datetime.now() - self.last_stations_fetch < timedelta(
                            hours=1)):
                    return self.stations_cache

                with engine.connect() as connection:
                    # Assuming 'Station' column exists in your Locations table 'L'
                    query = select(L.c.Station).distinct().order_by(L.c.Station)
                    result = connection.execute(query)
                    stations = [row[0] for row in result if row[0] is not None]
                    self.stations_cache = ['All Stations'] + stations
                    self.last_stations_fetch = datetime.now()
                    return self.stations_cache
            except Exception as e:
                logging.error(f"Error fetching stations: {e}")
                return ['All Stations', 'ErrorFetchingStations']
        return ['All Stations', 'Station1_Placeholder', 'Station2_Placeholder']

        # DETECT_ANOMALIES METHOD
    def detect_anomalies(self, df):
        """Detect anomalies using Isolation Forest"""
        if df.empty or 'Tab_Value_mDepthC1' not in df.columns:
            return df

        # Prepare data for anomaly detection
        X = df[['Tab_Value_mDepthC1']].values

        # Detect anomalies
        pred = self.iso_forest.fit_predict(X)
        df['anomaly'] = np.where(pred == -1, -1, 0)  # -1 = anomaly, 0 = normal

        return df


    def add_analysis_features(self, df):
        """Adds analytical features (e.g., rolling averages) to the DataFrame."""
        if df.empty or 'Tab_Value_mDepthC1' not in df.columns:
            return df
        # Initialize anomaly column if it doesn't exist
        if 'anomaly' not in df.columns:
            df['anomaly'] = 0  # Default to not anomaly

        for window, col_name in [(3, 'rolling_avg_3h'), (6, 'rolling_avg_6h'), (24, 'rolling_avg_24h')]:
            df[col_name] = df['Tab_Value_mDepthC1'].rolling(window=window, min_periods=1).mean()

        return df

    def _default_map_frame(self, message="Map service is currently unavailable. Please try again later."):
        """Provides a default map frame or error message if GovMap integration fails."""
        return html.Div(html.P(message, style={'textAlign': 'center', 'color': 'white', 'fontSize': '1.2em'}),
                        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'height': '100%'})

    # ---------------------------
    # VISUALIZATION ENGINE
    # ---------------------------
    def create_sea_level_graph(self, df, trendline_period, show_outliers,
                               show_predictions, prediction_models, station, analysis_type):
        """Main visualization renderer for sea level data."""
        fig = go.Figure()

        if df.empty or 'Tab_DateTime' not in df.columns or 'Tab_Value_mDepthC1' not in df.columns:
            logging.warning("DataFrame is empty or missing required columns for graph.")
            fig.update_layout(title_text="No data available to display.",
                              plot_bgcolor='#4E5D6C', paper_bgcolor='#2B3E50', font=dict(color='white'))
            return fig

        # Ensure 'Tab_DateTime' is datetime
        df['Tab_DateTime'] = pd.to_datetime(df['Tab_DateTime'])
        df = df.sort_values('Tab_DateTime')  # Sort before processing

        # Add analysis features (like rolling averages)
        df = self.add_analysis_features(df)

        # Station traces
        if "Station" in df.columns and station == 'All Stations':
            for station_name, station_data in df.groupby("Station"):
                fig.add_trace(go.Scattergl(
                    x=station_data["Tab_DateTime"],
                    y=station_data["Tab_Value_mDepthC1"],
                    mode="lines",
                    name=str(station_name),
                    hoverinfo="x+y+name",
                    legendgroup="stations",
                    showlegend = True
                ))
        else:
            fig.add_trace(go.Scattergl(
                x=df["Tab_DateTime"],
                y=df["Tab_Value_mDepthC1"],
                mode="lines",
                name=f"Sea Level Data ({station if station != 'All Stations' else 'All Stations'})",
                hoverinfo="x+y",
                legendgroup="stations",
                showlegend=True
            ))

        # Analytical features (rolling averages, etc.)
        self.add_analytical_traces(fig, df, analysis_type)

        # Trendline
        if trendline_period != "none":
            self.add_trendline(fig, df, trendline_period)

        # Predictions
        if show_predictions: # Assuming show_predictions is a boolean or list that evaluates to True
            self.add_predictions(fig, prediction_models, station, df)

        # Anomalies
        if show_outliers: # Assuming show_outliers is a boolean or list that evaluates to True
            self.add_anomalies(fig, df)

        return self.style_figure(fig)

    def create_tides_graph(self, df):
        """Create a line graph for tides data."""
        fig = go.Figure()

        if df.empty or 'Date' not in df.columns or 'HighTide' not in df.columns or 'LowTide' not in df.columns:
            logging.warning("DataFrame is empty or missing required columns for tides graph.")
            fig.update_layout(title_text="No tides data available to display.",
                              plot_bgcolor='#4E5D6C', paper_bgcolor='#2B3E50', font=dict(color='white'))
            return fig

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # High Tide trace
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["HighTide"],
            mode="lines",
            name="High Tide (m)",
            line=dict(color='deepskyblue'),
            hoverinfo="x+y+name"
        ))

        # Low Tide trace
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["LowTide"],
            mode="lines",
            name="Low Tide (m)",
            line=dict(color='lightcoral'),
            hoverinfo="x+y+name"
        ))

        fig.update_layout(
            title_text="Tides Over Time",
            xaxis_title="Date",
            yaxis_title="Tide Level (m)",
            showlegend=True
        )
        return self.style_figure(fig)


    def add_analytical_traces(self, fig, df, analysis_type):
        """Add analytical features to plot"""
        if df.empty:
            return

        avg_config = {
            'rolling_avg_3h': ('3-Hour Avg', 'violet'),
            'rolling_avg_6h': ('6-Hour Avg', 'cyan'),
            'rolling_avg_24h': ('24-Hour Avg', 'magenta')
        }

        if analysis_type == 'all':
            for col, (name, color) in avg_config.items():
                if col in df.columns:
                    fig.add_trace(go.Scattergl(
                        x=df['Tab_DateTime'], y=df[col],
                        name=name, line=dict(color=color, width=2),
                        legendgroup="analytics"
                    ))
        elif analysis_type in avg_config:
            if analysis_type in df.columns:
                fig.add_trace(go.Scattergl(
                    x=df['Tab_DateTime'], y=df[analysis_type],
                    name=avg_config[analysis_type][0],
                    line=dict(color=avg_config[analysis_type][1], width=2),
                    legendgroup="analytics"
                ))

    def add_trendline(self, fig, df, trendline_period):
        """Calculate and add trendline"""
        if df.empty or 'Tab_DateTime' not in df.columns or 'Tab_Value_mDepthC1' not in df.columns:
            return

        end_date = df['Tab_DateTime'].max()
        start_date_trend = {
            "last_decade": end_date - pd.DateOffset(years=10),
            "last_two_decades": end_date - pd.DateOffset(years=20),
            "all": df['Tab_DateTime'].min()
        }.get(trendline_period, end_date - pd.DateOffset(years=10))

        mask = (df['Tab_DateTime'] >= start_date_trend) & (df['Tab_DateTime'] <= end_date)
        reg_df = df[mask].copy()

        if len(reg_df) > 1:
            x_numeric = reg_df['Tab_DateTime'].astype(np.int64) // 10 ** 9

            y_numeric = reg_df['Tab_Value_mDepthC1']
            valid_indices = ~np.isnan(x_numeric) & ~np.isinf(x_numeric) & \
                            ~np.isnan(y_numeric) & ~np.isinf(y_numeric)

            if np.sum(valid_indices) > 1:
                x_numeric_clean = x_numeric[valid_indices]
                y_numeric_clean = y_numeric[valid_indices]

                # Changed to degree 2 for quadratic trendline, as per old version
                degree = 2
                coeffs = np.polyfit(x_numeric_clean, y_numeric_clean, degree)
                trend_values = np.polyval(coeffs, x_numeric_clean)

                fig.add_trace(go.Scattergl(
                    x=reg_df['Tab_DateTime'][valid_indices],
                    y=trend_values,
                    mode="lines",
                    name="Trendline",
                    line=dict(dash="dash", color="grey"),
                    legendgroup="analytics"
                ))
            else:
                logging.warning("Not enough valid data points to calculate trendline after cleaning NaNs/Infs.")
        else:
            logging.warning(f"Not enough data points ({len(reg_df)}) in the selected period for trendline.")

    def add_predictions(self, fig, models, station, df):
        """Add prediction traces"""
        if df.empty or 'Tab_DateTime' not in df.columns:
            return

        last_date = df['Tab_DateTime'].max()
        if pd.isna(last_date):
            logging.warning("Cannot generate predictions as last date is NaT.")
            return

        future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=240, freq='h')

        if 'arima' in models:
            arima_pred_values = self.arima_predict(station)
            if arima_pred_values is not None and len(arima_pred_values) == len(future_dates):
                fig.add_trace(go.Scattergl(
                    x=future_dates,
                    y=arima_pred_values,
                    mode="lines",
                    name="ARIMA Forecast",
                    line=dict(dash="dot", color="lime")
                ))

        if 'prophet' in models:
            prophet_pred_df = self.prophet_predict(station)
            if prophet_pred_df is not None and not prophet_pred_df.empty and \
                    'ds' in prophet_pred_df.columns and 'yhat' in prophet_pred_df.columns:
                fig.add_trace(go.Scattergl(
                    x=prophet_pred_df['ds'],
                    y=prophet_pred_df['yhat'],
                    mode="lines",
                    name="Prophet Forecast",
                    line=dict(dash="dot", color="orange")
                ))

    def add_anomalies(self, fig, df):
        """Highlight anomalies"""
        if df.empty or 'anomaly' not in df.columns or \
                'Tab_DateTime' not in df.columns or 'Tab_Value_mDepthC1' not in df.columns:
            return

        anomalies_df = df[df['anomaly'] == -1]
        if not anomalies_df.empty:
            fig.add_trace(go.Scattergl(
                x=anomalies_df['Tab_DateTime'],
                y=anomalies_df['Tab_Value_mDepthC1'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', symbol='x', size=8)
            ))

    def style_figure(self, fig):
        """Apply consistent styling"""
        fig.update_layout(
            title_text="Sea Level Over Time",
            xaxis_title="Date",
            yaxis_title="Sea Level (m)",
            showlegend=True,
            plot_bgcolor='#4E5D6C',
            paper_bgcolor='#2B3E50',
            font=dict(color='white'),
            xaxis=dict(linecolor='#7FD1AE', gridcolor='#4E5D6C'),
            yaxis=dict(linecolor='#7FD1AE', gridcolor='#4E5D6C'),
            hovermode='x unified',
        )
        return fig

    # ---------------------------
    # TABLE COMPONENTS & EXPORT
    # ---------------------------
    def create_table_content(self, df_records, quarry_option):
        """Generate styled data table for Dash DataTable"""
        if not df_records:
            return [], [{"name": "Status", "id": "Status"}]

        df_display = pd.DataFrame(df_records)

        if df_display.empty:
            return [], [{"name": "Status", "id": "Status"}]

        # Prepare data with user-friendly headers for display
        df_processed_for_display = self.prepare_table_for_export(df_display.copy(), quarry_option)

        self.current_table_df_for_export = df_processed_for_display.copy()

        # Define column mapping for display
        column_mapping = {}
        if quarry_option == 'tides':
            column_mapping = {
                "Date": "Date",
                "Station": "Station",
                "HighTide": "High Tide (m)",
                "HighTideTime": "High Tide Time",
                "HighTideTemp": "High Tide Temp (°C)",
                "LowTide": "Low Tide (m)",
                "LowTideTime": "Low Tide Time",
                "LowTideTemp": "Low Tide Temp (°C)",
                "MeasurementCount": "Measurement Count"
            }
        else: # Default sea level data
            column_mapping = {
                "Tab_DateTime": "Date/Time",
                "Station": "Station",
                "Tab_Value_mDepthC1": "Sea Level (m)",
                "Tab_Value_monT2m": "Water Temp (°C)",
                "Tab_Value_mDepth": "Depth (m)",
                #"X": "X Coordinate",
                #"Y": "Y Coordinate",
                #"Longitude": "Longitude",
                #"Latitude": "Latitude"
            }

        # Create columns for DataTable using 'id' for original column name and 'name' for display name
        dt_columns = [{"name": column_mapping.get(col, col), "id": col} for col in df_display.columns]

        # Sort columns to ensure consistent display order based on mapping order
        # This approach ensures columns defined in column_mapping appear first in their defined order.
        def get_order(col_id):
            if quarry_option == 'tides':
                map_keys = list(column_mapping.keys())
            else:
                map_keys = list(column_mapping.keys())
            try:
                return map_keys.index(col_id)
            except ValueError:
                return len(map_keys) + df_display.columns.tolist().index(col_id) # Columns not in mapping go after

        dt_columns_sorted = sorted(dt_columns, key=lambda x: get_order(x['id']))

        return df_display.to_dict('records'), dt_columns_sorted # Pass original df.to_dict('records') for data,
                                                               # and sorted columns for display
    def _format_table_cell(self, value):
        """Type-aware cell formatting - for manual HTML table, less so for DataTable"""
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        return str(value)

    def prepare_table_for_export(self, df, quarry_option):
        """Prepare data for display in DataTable and for download with user-friendly names."""
        if df.empty:
            return pd.DataFrame()

        column_mapping = {}
        if quarry_option == 'tides':
            column_mapping = {
                "Date": "Date",
                "Station": "Station",
                "HighTide": "High Tide (m)",
                "HighTideTime": "High Tide Time",
                "HighTideTemp": "High Tide Temp (°C)",
                "LowTide": "Low Tide (m)",
                "LowTideTime": "Low Tide Time",
                "LowTideTemp": "Low Tide Temp (°C)",
                "MeasurementCount": "Measurement Count"
            }
        else: # Default sea level data
            column_mapping = {
                "Tab_DateTime": "Date/Time",
                "Station": "Station",
                "Tab_Value_mDepthC1": "Sea Level (m)",
                "Tab_Value_monT2m": "Water Temp (°C)",
                "Tab_Value_mDepth": "Depth (m)",
                #"X": "X Coordinate",
                #"Y": "Y Coordinate",
                #"Longitude": "Longitude",
                #"Latitude": "Latitude"
            }

        df_renamed = df.rename(columns=column_mapping)

        # Filter to only include columns that exist in the DataFrame after renaming
        # This will ensure correct column order for export as well.
        # Create a list of desired columns in order, then filter by existence in df_renamed
        ordered_cols = [column_mapping.get(col, col) for col in df.columns]
        existing_cols = [col for col in ordered_cols if col in df_renamed.columns]
        df_export = df_renamed[existing_cols]

        # Apply formatting for export (e.g., rounding floats, datetime string)
        for col in df_export.columns:
            if pd.api.types.is_float_dtype(df_export[col]):
                df_export[col] = df_export[col].round(3)
            # Date/Time column was already parsed by pd.read_sql
            # For export, we ensure it's a string
            if pd.api.types.is_datetime64_any_dtype(df_export[col]):
                df_export[col] = df_export[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df_export

    # ---------------------------
    # MAP COMPONENTS
    # ---------------------------
    def serve_map_html(self):
        """Serve GovMap integration via an iframe."""
        try:
            center_x = 176505
            center_y = 662250

            govmap_url = (
                f"https://www.govmap.gov.il/map.html?"
                f"bb=1&zb=1&in=1&"
                f"c={center_x},{center_y}&"
                f"z=0&b=0&"
                f"lay=SEA_LEVEL" # This layer name should be verified with GovMap documentation
            )

            return f"""
            <iframe id='ifrMap'
                frameborder='0'
                scrolling='no'
                marginheight='0'
                marginwidth='0'
                width='100%'
                height='100%'
                src='{govmap_url}'>
            </iframe>
            """
        except Exception as e:
            logging.error(f"GovMap serving error: {e}")
            return self._default_map_frame(f"Map service error: {str(e)}")

    def handle_israel_map_view(self, df):
        """Generates OSM map with proper DataFrame handling"""
        try:
            # Get station locations
            station_locations_df = self.fetch_station_locations()

            if station_locations_df.empty:
                return self.create_empty_map("No station location data available")

            # Merge station data with latest values
            if not df.empty:
                df['Tab_DateTime'] = pd.to_datetime(df['Tab_DateTime'])
                latest_values = df.groupby('Station').agg(
                    latest_value=('Tab_Value_mDepthC1', 'last'),
                    last_update=('Tab_DateTime', 'last')
                ).reset_index()

                df_for_map = pd.merge(station_locations_df, latest_values, on='Station', how='left')

                if 'last_update' in df_for_map.columns:
                    df_for_map['last_update'] = df_for_map['last_update'].dt.strftime('%Y-%m-%d %H:%M').fillna('N/A')
                else:
                    df_for_map['last_update'] = 'N/A'

                df_for_map['latest_value'] = df_for_map['latest_value'].fillna('N/A')
            else:
                df_for_map = station_locations_df.copy()
                df_for_map['latest_value'] = 'N/A'
                df_for_map['last_update'] = 'N/A'

            # Ensure coordinates are valid
            df_for_map['lat'] = pd.to_numeric(df_for_map.get('latitude'), errors='coerce')
            df_for_map['lon'] = pd.to_numeric(df_for_map.get('longitude'), errors='coerce')
            df_for_map.dropna(subset=['lat', 'lon'], inplace=True)

            if df_for_map.empty:
                return self.create_empty_map("No valid coordinates after processing")

            # Dynamic Marker Sizing Based on Sea Level
            df_for_map['latest_value_num'] = pd.to_numeric(df_for_map['latest_value'], errors='coerce')
            sea_level_min = df_for_map['latest_value_num'].min()
            sea_level_max = df_for_map['latest_value_num'].max()
            min_size = 6
            max_size = 20

            if pd.isna(sea_level_min) or sea_level_max - sea_level_min == 0:
                df_for_map['marker_size'] = max_size
            else:
                df_for_map['marker_size'] = min_size + (
                        (df_for_map['latest_value_num'] - sea_level_min) /
                        (sea_level_max - sea_level_min)
                ) * (max_size - min_size)
                # Replace NaNs in marker size with a default size (e.g. min_size)
                df_for_map['marker_size'] = df_for_map['marker_size'].fillna(min_size)
                df_for_map['latest_value_formatted'] = df_for_map['latest_value_num'].map(
                    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")

            # Create Map
            center_lat = 31.5
            center_lon = 34.75
            zoom_level = 6

            fig = px.scatter_mapbox(
                df_for_map,
                lat="lat",
                lon="lon",
                size="marker_size",
                size_max=20,
                hover_name="Station",
                custom_data=["latest_value_formatted", "last_update"],  # Pass formatted data
                color_discrete_sequence=["#5b7fde"],
                zoom=zoom_level,
                center={"lat": center_lat, "lon": center_lon},
                height=600
            )

            fig.update_layout(
                mapbox_style="open-street-map",
                title='Sea Level Monitoring Stations in Israel',
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                plot_bgcolor='#4E5D6C',
                paper_bgcolor='#2B3E50',
                font=dict(color='white'),
                title_font=dict(size=20, color='white'),
                title_x=0.5,
                uirevision=True,
            )

            fig.update_mapboxes(
                accesstoken=None,
                zoom=zoom_level,
                center={"lat": center_lat, "lon": center_lon}
            )

            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                              "Sea Level (m): %{customdata[0]}<br>" +
                              "Last Update: %{customdata[1]}<extra></extra>"
            )

            return fig

        except Exception as e:
            logging.error(f"OSM Map error: {e}")
            return self.create_empty_map(f"Error generating map: {str(e)}")


    def create_empty_map(self, message):
        """Create empty map with error message"""
        fig = go.Figure()
        fig.update_layout(
            title_text=message,
            plot_bgcolor='#4E5D6C',
            paper_bgcolor='#2B3E50',
            font=dict(color='white'))
        return fig

    # ---------------------------
    # APP SETUP & UI LAYOUT
    # ---------------------------
    def create_app(self):
        """Configure Dash application"""
        app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
            server=self.flask_server
        )
        app.title = "Sea Level Dashboard"

        self.flask_server.add_url_rule('/mapframe', 'mapframe_content', self.serve_map_html)

        app.layout = self._create_layout()
        self.register_optimized_callbacks(app)
        return app

    def _create_layout(self):
        """Responsive UI layout"""
        return dbc.Container(
            fluid=True,
            className="dash-flex-container",
            style={"minHeight": "100vh", "display": "flex", "flexDirection": "column", "padding": "15px",
                   "backgroundColor": "#2B3E50"},
            children=[
                # Header Row
                dbc.Row(
                    [
                        html.Div(
                            html.Img(
                                src="/assets/Mapi_Logo2.png",
                                style={"height": "100px", "position": "absolute", "left": "15px", "top": "15px"}
                            ),
                        ),
                        dbc.Col(
                            html.H1(
                                'Sea Level Visualization Dashboard',
                                className="text-center mb-4 py-3 text-white",
                                style={"fontSize": "clamp(1.5rem, 4vw, 2.5rem)", "width": "100%"}
                            ),
                            width=12,
                            style={"position": "relative"}
                        )
                    ],
                    style={"position": "relative", "marginBottom": "20px", "paddingTop": "120px"}
                ),

                # Main Content Row
                dbc.Row(
                    className="flex-grow-1 g-3",
                    style={"minHeight": "0", "overflow": "hidden"},
                    children=[
                        # Filters Column
                        dbc.Col(
                            xs=12, md=3,
                            className="filters-column h-100",
                            id="filters-column",
                            style={"display": "flex", "flexDirection": "column"},
                            children=[
                                dbc.Card(
                                    className="h-100 shadow-lg",
                                    style={"backgroundColor": "#4E5D6C", "borderColor": "#7FD1AE"},
                                    children=[
                                        dbc.CardHeader("Filters & Controls",
                                                       className="bg-info text-white py-2",
                                                       style={"fontSize": "1.1rem"}),
                                        dbc.CardBody(
                                            className="overflow-auto text-white",
                                            ##style={"maxHeight": "calc(100vh - 250px)"},
                                            children=[
                                                html.Label("Date Range:", className="font-weight-bold mb-1"),
                                                dcc.DatePickerRange(
                                                    id='date-range',
                                                    min_date_allowed=datetime(2015, 1, 1),
                                                    max_date_allowed=datetime.now() + timedelta(days=1),
                                                    initial_visible_month=datetime.now(),
                                                    start_date=(datetime.now() - timedelta(days=7)).strftime(
                                                        '%Y-%m-%d'),
                                                    end_date=datetime.now().strftime('%Y-%m-%d'),
                                                    className="mb-3 w-100",
                                                    display_format='YYYY-MM-DD'
                                                ),

                                                html.Label("Station Selection:", className="font-weight-bold mb-1"),
                                                dcc.Dropdown(
                                                    id='station-dropdown',
                                                    options=[{'label': s, 'value': s} for s in self.get_stations()],
                                                    value='All Stations',
                                                    className="mb-3",
                                                ),

                                                html.Label("Data View:", className="font-weight-bold mb-1"),
                                                dcc.Dropdown(
                                                    id='view-dropdown',
                                                    options=[
                                                        {'label': "Line Graph", "value": "graph"},
                                                        {'label': "Table View", "value": "table"},
                                                        {'label': "GovMap (Israel)", "value": "govmap"},
                                                        {'label': "OSM Stations Map", "value": "israel_map"}
                                                    ],
                                                    value="graph",
                                                    className="mb-3"
                                                ),

                                                html.Label("Data Type:", className="font-weight-bold mb-1"),
                                                dcc.Dropdown(
                                                    id='quarry-dropdown',
                                                    options=[
                                                        {'label': 'Default Sensor Data', 'value': 'default'},
                                                        {'label': 'Tidal Data', 'value': 'tides'}
                                                    ],
                                                    value='default',
                                                    className="mb-3"
                                                ),

                                                html.Label("Trendline Period:", className="font-weight-bold mb-1"),
                                                dcc.Dropdown(
                                                    id='trendline-dropdown',
                                                    options=[
                                                        {'label': 'No Trendline', 'value': 'none'},
                                                        {'label': 'All Period', 'value': 'all'},
                                                        {'label': 'Last Two Decades', 'value': 'last_two_decades'},
                                                        {'label': 'Last Decade', 'value': 'last_decade'}
                                                    ],
                                                    value='none',
                                                    className="mb-3"
                                                ),

                                                html.Label("Analysis Type:", className="font-weight-bold mb-1"),
                                                dcc.Dropdown(
                                                    id='analysis-type',
                                                    options=[
                                                        {'label': 'None', 'value': 'none'},
                                                        {'label': '3-Hour Rolling Avg', 'value': 'rolling_avg_3h'},
                                                        {'label': '6-Hour Rolling Avg', 'value': 'rolling_avg_6h'},
                                                        {'label': '24-Hour Rolling Avg', 'value': 'rolling_avg_24h'},
                                                        {'label': 'All Rolling Averages', 'value': 'all'}
                                                    ],
                                                    value='none',
                                                    className="mb-3"
                                                ),

                                                dbc.Row([
                                                    dbc.Col(dbc.Checklist(
                                                        id='show-anomalies',
                                                        options=[{'label': 'Show Anomalies', 'value': 'show'}],
                                                        value=[],
                                                        switch=True,
                                                        className="mb-2"
                                                    ), xs=12, md=6),

                                                    html.Div(id='placeholder-for-removed-show-predictions')
                                                ]),
                                                html.Label("Prediction Models:", className="font-weight-bold mt-3"),
                                                dbc.Checklist(
                                                    id='prediction-models',
                                                    options=[
                                                        {'label': ' ARIMA', 'value': 'arima'},
                                                        {'label': ' Prophet', 'value': 'prophet'}
                                                    ],
                                                    value=[],
                                                    switch=True,
                                                    inline=True
                                                ),

                                                dbc.Row([
                                                    dbc.Col(dbc.Button(
                                                        "Export Graph",
                                                        id="export-graph-btn",
                                                        color="primary",
                                                        className="w-100 mb-2"
                                                    ), xs=12, sm=6),
                                                    dbc.Col(dbc.Button(
                                                        "Export Table",
                                                        id="export-table-btn",
                                                        color="primary",
                                                        className="w-100"
                                                    ), xs=12, sm=6)
                                                ], className="g-2")
                                            ]
                                        )
                                    ]
                                )
                            ]
                        ),

                        # Main Content Column (Graph/Table/Map)
                        dbc.Col(
                            xs=12, md=9,
                            className="content-column h-100",
                            style={"minHeight": "450px", "display": "flex", "flexDirection": "column"},
                            children=[
                                dbc.Card(
                                    className="h-100 shadow-sm",
                                    style={"backgroundColor": "#4E5D6C", "borderColor": "#7FD1AE"},
                                    children=[
                                        dbc.CardBody(
                                            className="h-100 p-2",
                                            style={"overflow": "hidden", "display": "flex", "flexDirection": "column"},
                                            children=[
                                                # Graph Container
                                                html.Div(
                                                    id="graph-container",
                                                    children=[dcc.Graph(id="line-graph", style={"height": "100%"},config={"scrollZoom": True})],
                                                    style={"display": "block", "flexGrow": "1", "minHeight": "400px"}
                                                ),

                                                # Table Container
                                                html.Div(
                                                    id="table-container",
                                                    style={"display": "none", "flexGrow": "1", "overflow": "auto"},
                                                    children=[
                                                        dash_table.DataTable(
                                                            id='data-table',
                                                            page_current=0,
                                                            page_size=15,
                                                            page_action='native',
                                                            sort_action='native',
                                                            filter_action='native',
                                                            style_table={'overflowX': 'auto',
                                                                         'backgroundColor': '#2B3E50',
                                                                         'border': '1px solid #7FD1AE',
                                                                         'minHeight': '400px'
                                                            },
                                                            style_header={
                                                                'backgroundColor': '#5bc0de',
                                                                'color': 'black',
                                                                'fontWeight': 'bold',
                                                                'border': '1px solid #5bc0de'
                                                            },
                                                           style_data_conditional=[
                                                           {
                                                                'if': {'row_index': 'odd'},
                                                                'backgroundColor': '#4E5D6C'
                                                            },
                                                            {
                                                                'if': {'row_index': 'even'},
                                                                'backgroundColor': '#2B3E50'
                                                            }
                                                           ],
                                                            style_cell={
                                                                'color': 'white',
                                                                'textAlign': 'left',
                                                                'padding': '10px',
                                                                'border': '1px solid #7FD1AE'
                                                            },
                                                        )
                                                    ]
                                                ),

                                                # GovMap Container (using an iframe that points to the Flask route)
                                                html.Div(
                                                    id="govmap-container",
                                                    style={"display": "none", "flexGrow": "1", "minHeight": "400px"},
                                                    children=[
                                                        html.Iframe(
                                                            id='govmap-iframe',
                                                            src='/mapframe',
                                                            style={"height": "100%", "width": "100%", "border": "none"}
                                                        )
                                                    ]
                                                ),
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                # Stores and Download
                dcc.Store(id='current_graph_or_table_data'),
                dcc.Download(id="download-data")
            ]
        )

    # ---------------------------
    # CALLBACKS & INTERACTIVITY
    # ---------------------------
    def register_optimized_callbacks(self, app):
        """Register all application callbacks"""

        # View visibility control
        @app.callback(
            [Output("graph-container", "style"),
             Output("table-container", "style"),
             Output("govmap-container", "style")],
            [Input("view-dropdown", "value")]
        )
        def update_view_visibility(view):
            graph_style = {"display": "none", "flexGrow": "1", "minHeight": "400px"}
            table_style = {"display": "none", "flexGrow": "1", "overflow": "auto", "minHeight": "400px"}
            govmap_style = {"display": "none", "flexGrow": "1", "minHeight": "400px"}

            if view == "graph" or view == "israel_map":
                graph_style["display"] = "block"
            elif view == "table":
                table_style["display"] = "block"
            elif view == "govmap":
                govmap_style["display"] = "block"

            return graph_style, table_style, govmap_style

        # Core data updates for graph, table data store, and OSM map
        @app.callback(
            [Output("line-graph", "figure"),  # Used for both line graph and OSM map
             Output('current_graph_or_table_data', 'data')],  # Stores data for table and OSM map logic
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("station-dropdown", "value"),
             Input("quarry-dropdown", "value"),
             Input("view-dropdown", "value"),
             Input("show-anomalies", "value"),
             Input("prediction-models", "value"),
             Input("trendline-dropdown", "value"),
             Input("analysis-type", "value"),
             ]
        )
        def update_main_content(start_date, end_date, station, quarry_option, view,
                                show_anomalies_val, prediction_models_val, trendline_period, analysis_type_val):


            # Load data from DB
            df = self.load_data_from_db(start_date, end_date, station, quarry_option)

            #  Section to handle Anomaly detection
            if 'show' in show_anomalies_val and quarry_option == 'default' and not df.empty:
                df = self.detect_anomalies(df.copy())
            else:
                # Ensure anomaly column exists even if not showing anomalies
                if 'anomaly' not in df.columns:
                    df['anomaly'] = 0


            # Store df for table use or other processing (like OSM map if it needs full df)
            df_records = df.to_dict('records') if not df.empty else []

            if view == "israel_map":
                # For OSM map, generate map figure
                fig = self.handle_israel_map_view(df.copy())  # Pass a copy of df
            elif view == "graph":
                if quarry_option == "tides":
                    fig = self.create_tides_graph(df.copy())
                else:
                    fig = self.create_sea_level_graph(df.copy(), trendline_period,
                                                      'show' in show_anomalies_val,
                                                      bool(prediction_models_val),
                                                      prediction_models_val,
                                                      station, analysis_type_val)
            else:  # For table or govmap view, graph is hidden, return empty/default figure
                fig = go.Figure().update_layout(
                    plot_bgcolor='#4E5D6C', paper_bgcolor='#2B3E50', font=dict(color='white'),
                    title_text="Select 'Line Graph' or 'OSM Stations Map' to view visualization."
                )

            return fig, df_records

        # Callback to update DataTable
        @app.callback(
            [Output('data-table', 'data'),
             Output('data-table', 'columns')],
            [Input('current_graph_or_table_data', 'data')],
            [State("quarry-dropdown", "value")]
        )
        def update_data_table(df_records, quarry_option):
            if not df_records:
                return [], [{"name": "Status", "id": "Status", "type": "text"}]

            data, columns = self.create_table_content(df_records, quarry_option)
            return data, columns

        # Export handlers
        @app.callback(
            Output("download-data", "data"),
            [Input("export-graph-btn", "n_clicks"),
             Input("export-table-btn", "n_clicks")],
            [State("quarry-dropdown", "value")], # To pass to prepare_table_for_export
            prevent_initial_call=True
        )
        def handle_exports(graph_clicks, table_clicks, quarry_option):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == "export-graph-btn":
                if self.current_graph_data:
                    try:
                        if isinstance(self.current_graph_data, go.Figure):
                            img_bytes = self.current_graph_data.to_image(format="png", width=1200, height=700, scale=1)
                            return dcc.send_bytes(img_bytes, "sea_level_graph.png")
                        else:
                            logging.error("Graph data for export is not a Plotly Figure object.")
                            return no_update
                    except Exception as e:
                        logging.error(f"Error exporting graph: {e}")
                        return no_update
                return no_update


            elif button_id == "export-table-btn":
                if not self.current_table_df_for_export.empty:
                    # Use the stored df directly, as it's already prepared for export
                    return dcc.send_data_frame(
                        self.current_table_df_for_export.to_csv,
                        "sea_level_data.csv",
                        index=False
                    )
                return no_update

            return no_update

        # Station list updates (if stations are dynamic or searchable)
        @app.callback(
            Output('station-dropdown', 'options'),
            [Input('station-dropdown', 'search_value')], # Triggered by search value to enable dynamic search
            prevent_initial_call=False
        )
        def update_station_options(search_value):
            all_stations = self.get_stations()
            if not search_value:
                return [{'label': s, 'value': s} for s in all_stations]
            # Filter stations based on search_value
            filtered_stations = [s for s in all_stations if search_value.lower() in s.lower()]
            return [{'label': s, 'value': s} for s in filtered_stations]

    # ---------------------------
    # CLEANUP & MAINTENANCE
    # ---------------------------
    def __del__(self):
        """Resource cleanup handler"""
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)
                logging.info("ThreadPoolExecutor shut down.")
        except Exception as e:
            logging.error(f"Error shutting down ThreadPoolExecutor: {e}")


    def add_cors_headers(self, response):
        """CORS headers for Flask server (if Dash app is accessed from different origin)"""
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response


# ---------------------------
# APPLICATION BOOTSTRAP
# ---------------------------
if __name__ == "__main__":
    dashboard = SeaLevelDashboard(engine)
    app_instance = dashboard.create_app()

    app_instance.server.run(host="localhost", port=8052, debug=True)

    try:
        if engine:
            engine.dispose()
            logging.info("Global SQLAlchemy engine disposed.")
    except NameError:
        logging.info("Global SQLAlchemy engine was not defined or already handled.")
    except Exception as e:
        logging.error(f"Error disposing global SQLAlchemy engine: {e}")