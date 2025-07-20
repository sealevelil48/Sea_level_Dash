# ---------------------------
# BASE IMPORTS & CONFIG
# ---------------------------
import os
import gc
import re
import logging
import warnings
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
from dash.dependencies import ClientsideFunction
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from flask import Flask, request, jsonify, Response
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from pyproj import Transformer

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
DB_URI = os.environ['DB_URI']
if not DB_URI:
    raise RuntimeError("❌ DB_URI not set. Please check your .env file or environment variables.")
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
        self.stats_data = {
            'current_level': None,
            '24h_change': None,
            'avg_temp': None,
            'anomalies': None
        }

        # Initialize models
        self.prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        self.iso_forest = IsolationForest(contamination=0.01, random_state=42)

        # Flask server setup
        self.flask_server = Flask(__name__)
        self.flask_server.after_request(self.add_cors_headers)
        self.setup_api()  # Initialize API endpoints
        self.load_initial_data()

    def setup_api(self):
        """Add API endpoints to Flask server"""

        @self.flask_server.route('/api/stations', methods=['GET'])
        def get_stations():
            try:
                stations = self.get_stations()
                return jsonify({'stations': stations})
            except Exception as e:
                logging.error(f"API Error: {e}")
                return jsonify({'error': str(e)}), 500

        # Get Yesterday Data Route & Get Live Data Route :
        @self.flask_server.route('/api/yesterday/<station>', methods=['GET'])
        def get_yesterday_station_data(station):
            try:
                data = self.get_yesterday_data(station)
                return jsonify({'station': station, 'data': data})
            except Exception as e:
                logging.error(f"API Error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.flask_server.route('/api/live', methods=['GET'])
        @self.flask_server.route('/api/live/<station>', methods=['GET'])
        def get_live_station_data(station=None):
            try:
                data = self.get_live_data(station)
                return jsonify({'station': station or 'all', 'data': data})
            except Exception as e:
                logging.error(f"API Error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.flask_server.route('/api/data', methods=['GET'])
        def get_data():
            try:
                station = request.args.get('station', 'All Stations')
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')
                data_source = request.args.get('data_source', 'default')

                # Validate parameters
                if station == 'All Stations':
                    station = None

                # New: Check if station is required for default data source
                if data_source == 'default' and not station:
                    return jsonify({
                        'error': 'Station parameter is required for default data source'
                    }), 400

                df = self.load_data_from_db(start_date, end_date, station, data_source)

                if df.empty:
                    # New: More specific 404 response
                    return jsonify({
                        'message': 'No data found',
                        'details': {
                            'station': station or 'All Stations',
                            'start_date': start_date,
                            'end_date': end_date,
                            'data_source': data_source
                        }
                    }), 404

                # Convert to JSON with ISO date formatting
                return df.to_json(orient='records', date_format='iso')
            except Exception as e:
                logging.error(f"API Error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.flask_server.route('/api/predictions', methods=['GET'])
        def get_predictions():
            try:
                station = request.args.get('station')
                model = request.args.get('model', 'all')

                # New: Better validation
                if not station:
                    return jsonify({
                        'error': 'Station parameter is required'
                    }), 400

                results = {}

                if model in ['arima', 'all']:
                    arima_pred = self.arima_predict(station)
                    results['arima'] = arima_pred if arima_pred else None

                if model in ['prophet', 'all']:
                    prophet_pred = self.prophet_predict(station)
                    if prophet_pred is not None and not prophet_pred.empty:
                        results['prophet'] = prophet_pred.to_dict(orient='records')
                    else:
                        results['prophet'] = None

                return jsonify(results)
            except Exception as e:
                logging.error(f"API Error: {e}")
                return jsonify({'error': str(e)}), 500

        @self.flask_server.route('/api/stations/map', methods=['GET'])
        def get_station_map():
            try:
                data = self.get_govmap_station_data()
                return jsonify(data)
            except Exception as e:
                logging.error(f"API Error: {e}")
                return jsonify({'error': str(e)}), 500

    # ---------------------------
    # EXPORT UTILITIES
    # ---------------------------
    def generate_export_filename(self, station, start_date, end_date, extension="png"):
        """Generate sanitized filename with consistent naming pattern"""
        # If station is None or empty, use a placeholder
        station = station or "AllStations"

        # Sanitize station name - keep only alphanumeric, underscore, and hyphen
        sanitized_station = re.sub(r'[^\w\-]', '', station)

        # Format dates consistently
        def format_date(date_str):
            if not date_str:
                return "NODATE"
            try:
                # Handle full datetime strings or just dates
                if ' ' in date_str:
                    date_part = date_str.split(' ')[0]
                    return datetime.strptime(date_part, '%Y-%m-%d').strftime('%Y-%m-%d')
                return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
            except:
                # Fallback sanitization
                return re.sub(r'[^\w\-]', '_', date_str)[:20]

        sanitized_start = format_date(start_date)
        sanitized_end = format_date(end_date)

        return f"sea_level_{sanitized_station}_{sanitized_start}_to_{sanitized_end}.{extension}"

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
            if hasattr(self, 'station_locations_cache') and self.station_locations_cache is not None:
                if (datetime.now() - self.last_stations_locations_fetch) < timedelta(hours=1):
                    return self.station_locations_cache

            logging.info("Fetching station locations from DB for map display.")

            query = text("""
                SELECT DISTINCT "Station", 
                       (locations::point)[0] AS latitude, 
                       (locations::point)[1] AS longitude
                FROM "Locations" 
                WHERE locations IS NOT NULL
            """)

            with self.engine.connect() as connection:
                result = connection.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            if df.empty:
                logging.warning("No station location data fetched for map")
                return pd.DataFrame(columns=["Station", "longitude", "latitude"])

            # 🛠 Clean up parentheses from coordinates
            df['latitude'] = df['latitude'].astype(str).str.replace("(", "", regex=False).str.replace(")", "",
                                                                                                      regex=False).astype(
                float)
            df['longitude'] = df['longitude'].astype(str).str.replace("(", "", regex=False).str.replace(")", "",
                                                                                                        regex=False).astype(
                float)

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
            conditions.append(text('"Station" = :station'))  # Reverting to original simple text for now

        return and_(*conditions) if conditions else True  # True means no conditions effectively

    def default_columns(self):
        # Ensure these columns exist in M and L respectively and can be joined.
        return [M.c.Tab_DateTime, L.c.Station, M.c.Tab_Value_mDepthC1,
                M.c.Tab_Value_monT2m,
                # L.c.X, L.c.Y, L.c.Longitude, L.c.Latitude
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
    def get_prediction_data(self, station):  # Cache results for a few recent stations
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
            forecast = model_fit.forecast(steps=240)  # Forecast 240 hours
            return forecast.tolist()  # Convert to list for JSON serialization
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
                return pd.DataFrame()

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
                return pd.DataFrame()

            # Initialize forecast to None or an empty DataFrame before trying to assign it
            forecast = None
            model = Prophet(yearly_seasonality=True, daily_seasonality=True, growth='linear')
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=240, freq='h')

            # Check if future dataframe is empty. Prophet might have issues predicting for empty 'future'.
            if future.empty:
                logging.warning(f"Prophet: Future dataframe is empty for station {station}. Cannot generate forecast.")
                return pd.DataFrame()

            forecast = model.predict(future)

            # --- NEW CHECKS ---
            if forecast is None or forecast.empty or 'yhat' not in forecast.columns:
                logging.warning(
                    f"Prophet: Forecast result is invalid (None, empty, or missing 'yhat') for station {station}.")
                return pd.DataFrame()
            if forecast['yhat'].isnull().all():
                logging.warning(f"Prophet: All 'yhat' values in forecast are NaN for station {station}.")
                return pd.DataFrame()

            logging.info(
                f"Prophet: Forecast generated for station {station}. Length: {len(forecast)}, First 'ds': {forecast['ds'].min()}, Last 'ds': {forecast['ds'].max()}")
            logging.debug(f"Prophet Forecast Head for {station}:\n{forecast[['ds', 'yhat']].head()}")
            logging.debug(f"Prophet Forecast Tail for {station}:\n{forecast[['ds', 'yhat']].tail()}")
            logging.debug(f"Prophet Forecast yhat NaNs for {station}: {forecast['yhat'].isnull().sum()}")

            return forecast[['ds', 'yhat']]
        except Exception as e:
            logging.error(f"Prophet prediction failed for station {station}: {str(e)}")
            return pd.DataFrame()

    def get_stations(self):
        """Fetches station list from the database."""
        if L is not None:
            try:
                # Fix cache logic - corrected version:
                if (self.stations_cache is not None and
                        self.last_stations_fetch is not None and
                        (datetime.now() - self.last_stations_fetch) < timedelta(hours=1)):
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

    # Get Yesterday & Get Live Data
    def get_yesterday_data(self, station):
        """Get yesterday's data for a specific station"""
        try:
            # Calculate yesterday's date
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)

            # Create datetime objects for start and end of yesterday
            start_date = datetime.combine(yesterday, datetime.min.time())
            end_date = datetime.combine(yesterday, datetime.max.time())

            # Format dates as strings
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

            # Load data from DB
            df = self.load_data_from_db(
                start_date=start_str,
                end_date=end_str,
                station=station,
                data_source='default'
            )

            return df.to_dict('records')
        except Exception as e:
            logging.error(f"Error getting yesterday's data for {station}: {e}")
            return []

    def get_live_data(self, station=None):
        """Get the latest reading for all stations or a specific station"""
        try:
            # Subquery: latest timestamp per station
            latest_time_subq = (
                select([
                    L.c.Station,
                    func.max(M.c.Tab_DateTime).label('max_time')
                ])
                .select_from(M.join(L, M.c.Tab_TabularTag == L.c.Tab_TabularTag))
                .group_by(L.c.Station)
                .subquery()
            )

            # Main query: only numeric fields
            stmt = (
                select([
                    L.c.Station,
                    M.c.Tab_Value_mDepthC1,
                    M.c.Tab_DateTime
                ])
                .select_from(
                    M.join(L, M.c.Tab_TabularTag == L.c.Tab_TabularTag)
                    .join(latest_time_subq, and_(
                        L.c.Station == latest_time_subq.c.Station,
                        M.c.Tab_DateTime == latest_time_subq.c.max_time
                    ))
                )
            )

            if station:
                stmt = stmt.where(L.c.Station == station)

            with self.engine.connect() as conn:
                result = conn.execute(stmt)
                columns = result.keys()
                rows = result.fetchall()

            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logging.error(f"Error getting live data: {e}")
            return []

    def get_govmap_station_data(self):
        try:
            # Step 1: Load station coordinates
            locations_df = self.fetch_station_locations()

            # Step 2: Load latest sea level values
            live_data = self.get_live_data()
            live_df = pd.DataFrame(live_data)

            # Step 3: Merge latest values into location data
            if not live_df.empty:
                merged = pd.merge(
                    locations_df,
                    live_df[['Station', 'Tab_Value_mDepthC1', 'Tab_DateTime']],
                    on='Station',
                    how='left'
                )
                merged['latest_value'] = merged['Tab_Value_mDepthC1']
                merged['last_update'] = pd.to_datetime(
                    merged['Tab_DateTime'], errors='coerce'
                ).dt.strftime('%Y-%m-%d %H:%M')
            else:
                merged = locations_df.copy()
                merged['latest_value'] = None
                merged['last_update'] = None

            # Step 4: Remove rows missing critical coordinate info
            merged = merged.dropna(subset=['longitude', 'latitude'])

            # Step 5: Convert WGS84 → ITM using pyproj (preserves row alignment)
            transformer = Transformer.from_crs("epsg:4326", "epsg:2039", always_xy=True)
            merged[['x', 'y']] = merged.apply(
                lambda row: pd.Series(transformer.transform(row['longitude'], row['latitude'])),
                axis=1
            )

            # Step 6: Final cleanup
            merged = merged.dropna(subset=['x', 'y', 'latest_value'])
            merged = merged.sort_values(by='Station')

            # Step 7: Return clean structured output
            return merged[['Station', 'x', 'y', 'latest_value', 'last_update']].to_dict(orient='records')

        except Exception as e:
            logging.error(f"Error preparing station data for GovMap: {e}")
            return []

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

    def serve_map_html(self):
        """Serves GovMap HTML page"""
        try:
            html = """
<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GovMap View</title>
        <script src="https://www.govmap.gov.il/govmap/api/govmap.api.js"></script>
        <style>
            html, body, #map-container {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
            }
        </style>
    </head>
    <body>
        <div id="map-container"></div>

        <script>
            govmap.token = "your-real-production-token-here"; // Replace this!

            govmap.createMap('map-container', {
                layers: [],
                center: { x: 176505, y: 662250 },
                zoom: 0,
                basemap: '2',
                isPopupOpen: false
            });

            fetch('/api/stations/map')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status} - ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(stations => {
                    console.log("GovMap Stations:", stations);

                    if (!stations || !stations.length) {
                        alert("No station data available.");
                        return;
                    }

                    const wkts = [];
                    const names = [];
                    const tooltips = [];
                    const bubbleHTML = [];

                    stations.forEach(s => {
                        if (!s || s.x == null || s.y == null) return; // skip broken entries

                        const name = s.Station || "Unknown";
                        const x = Number(s.x);
                        const y = Number(s.y);
                        const val = s.latest_value !== null ? Number(s.latest_value).toFixed(3) : "N/A";
                        const date = s.last_update || "N/A";

                        wkts.push(`POINT(${x} ${y})`);
                        names.push(name);

                        tooltips.push(`Station: ${name}\nSea Level: ${val} m\nLast Update: ${date}`);

                        bubbleHTML.push(`
                            <div>
                                <strong>Station:</strong> ${name}<br/>
                                <strong>Sea Level:</strong> ${val} m<br/>
                                <strong>Last Update:</strong> ${date}
                            </div>`);
                    });

                    govmap.displayGeometries({
                        wkts,
                        names,
                        geometryType: govmap.drawType.Point,
                        clearExisting: true,
                        data: {
                            tooltips: tooltips,
                            bubbleHTML: bubbleHTML
                        }
                        // defaultSymbol: undefined  <-- default blue dot marker
                    });
                })
                .catch(err => {
                    console.error("GovMap station fetch error:", err);
                    alert("GovMap failed to load stations: " + err.message);
                });
        </script>
    </body>
    </html>
    """
            return Response(html, mimetype='text/html')
        except Exception as e:
            logging.error(f"GovMap mapframe error: {e}")
            return self._default_map_frame(f"Map service error: {str(e)}")

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
                              plot_bgcolor='#142950', paper_bgcolor='#0c1c35', font=dict(color='white'))
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
                    showlegend=True
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
        if show_predictions:  # Assuming show_predictions is a boolean or list that evaluates to True
            self.add_predictions(fig, prediction_models, station, df)

        # Anomalies
        if show_outliers:  # Assuming show_outliers is a boolean or list that evaluates to True
            self.add_anomalies(fig, df)

        return self.style_figure(fig)

    def create_tides_graph(self, df):
        """Create a line graph for tides data."""
        fig = go.Figure()

        if df.empty or 'Date' not in df.columns or 'HighTide' not in df.columns or 'LowTide' not in df.columns:
            logging.warning("DataFrame is empty or missing required columns for tides graph.")
            fig.update_layout(title_text="No tides data available to display.",
                              plot_bgcolor='#142950', paper_bgcolor='#0c1c35', font=dict(color='white'))
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
            plot_bgcolor='#142950',
            paper_bgcolor='#0c1c35',
            font=dict(color='white'),
            xaxis=dict(
                linecolor='#7FD1AE',
                gridcolor='#1e3c72',
                gridwidth=0.5
            ),
            yaxis=dict(
                linecolor='#7FD1AE',
                gridcolor='#1e3c72',
                gridwidth=0.5
            ),
            hovermode='x unified',
            legend=dict(
                bgcolor='#1e3c70',
                font=dict(size=10)
            )
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
        else:  # Default sea level data
            column_mapping = {
                "Tab_DateTime": "Date/Time",
                "Station": "Station",
                "Tab_Value_mDepthC1": "Sea Level (m)",
                "Tab_Value_monT2m": "Water Temp (°C)",
                "Tab_Value_mDepth": "Depth (m)",
                # "X": "X Coordinate",
                # "Y": "Y Coordinate",
                # "Longitude": "Longitude",
                # "Latitude": "Latitude"
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
                return len(map_keys) + df_display.columns.tolist().index(col_id)  # Columns not in mapping go after

        dt_columns_sorted = sorted(dt_columns, key=lambda x: get_order(x['id']))

        return df_display.to_dict('records'), dt_columns_sorted  # Pass original df.to_dict('records') for data,
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
        else:  # Default sea level data
            column_mapping = {
                "Tab_DateTime": "Date/Time",
                "Station": "Station",
                "Tab_Value_mDepthC1": "Sea Level (m)",
                "Tab_Value_monT2m": "Water Temp (°C)",
                "Tab_Value_mDepth": "Depth (m)",
                # "X": "X Coordinate",
                # "Y": "Y Coordinate",
                # "Longitude": "Longitude",
                # "Latitude": "Latitude"
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
                plot_bgcolor='#142950',
                paper_bgcolor='#0c1c35',
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
            plot_bgcolor='#142950',
            paper_bgcolor='#0c1c35',
            font=dict(color='white'))
        return fig

    # ---------------------------
    # STATS CALCULATION
    # ---------------------------
    def calculate_stats(self, df):
        """Calculate statistics for display in cards"""
        stats = {
            'current_level': None,
            '24h_change': None,
            'avg_temp': None,
            'anomalies': None
        }

        if not df.empty:
            try:
                # Current Level
                if 'Tab_Value_mDepthC1' in df.columns:
                    stats['current_level'] = df['Tab_Value_mDepthC1'].iloc[-1]

                # 24h Change
                if 'Tab_Value_mDepthC1' in df.columns and len(df) > 1:
                    now_val = df['Tab_Value_mDepthC1'].iloc[-1]
                    yesterday_val = df['Tab_Value_mDepthC1'].iloc[0]
                    stats['24h_change'] = now_val - yesterday_val

                # Average Temperature
                if 'Tab_Value_monT2m' in df.columns:
                    stats['avg_temp'] = df['Tab_Value_monT2m'].mean()

                # Anomalies
                if 'anomaly' in df.columns:
                    stats['anomalies'] = df[df['anomaly'] == -1].shape[0]

            except Exception as e:
                logging.error(f"Error calculating stats: {e}")

        return stats

    # ---------------------------
    # APP SETUP & UI LAYOUT
    # ---------------------------
    def create_app(self):
        """Configure Dash application"""
        app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, 'assets/style.css'],
            meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
            server=self.flask_server,
            assets_folder='assets'  # Point to folder containing style.css
        )
        app.title = "Sea Level Dashboard"

        # Add GovMap route
        self.flask_server.add_url_rule('/mapframe', 'mapframe_content', self.serve_map_html)

        app.layout = self._create_layout()
        self.register_optimized_callbacks(app)
        return app

    def _create_layout(self):
        """Responsive UI layout with new design"""
        return html.Div(
            className="dash-container",
            style={
                'backgroundColor': '#0c1c35',
                'minHeight': '100vh',
                'fontFamily': 'Segoe UI, sans-serif',
                'padding': '0',
                'margin': '0',
                'zoom': '85%'  # Scale down the entire dashboard
            },
            children=[
                # Header Section
                html.Div(
                    className="header",
                    style={
                        'backgroundColor': '#0a172c',
                        'padding': '10px 20px',
                        'display': 'flex',
                        'alignItems': 'center',
                        'borderBottom': '1px solid #1e3c72'
                    },
                    children=[
                        html.Img(
                            src="/assets/Mapi_Logo2.png",
                            style={'height': '60px', 'marginRight': '15px'}
                        ),
                        html.H1(
                            'Sea Level Monitoring System',
                            style={
                                'color': 'white',
                                'margin': '0',
                                'fontSize': '24px',
                                'flexGrow': '1',
                                'fontWeight': 'bold'
                            }
                        ),
                        html.Div(
                            id="current-time",
                            style={
                                'color': '#4dabf5',
                                'fontWeight': 'bold',
                                'fontSize': '18px'
                            }
                        )
                    ]
                ),

                # Main Content Area
                html.Div(
                    className="main-content",
                    style={
                        'display': 'flex',
                        'padding': '20px',
                        'gap': '20px',
                        'minHeight': 'auto'
                    },
                    children=[
                        # Left Panel - Filters (full height)
                        html.Div(
                            className="filters-column",
                            style={
                                'backgroundColor': '#142950',
                                'width': '300px',
                                'padding': '20px',
                                'borderRadius': '12px',
                                'boxShadow': '0 2px 5px rgba(0,0,0,0.3)',
                                'overflowY': 'visible',
                                # Allow overflow for long content (scrollbar will appear if needed)
                                'height': 'auto',  # Allow height to adjust based on content

                            },
                            children=[

                                html.H3(
                                    "Data Filters",
                                    style={
                                        'color': 'white',
                                        'borderBottom': '1px solid #2a4a8c',
                                        'paddingBottom': '10px',
                                        'marginTop': '0',
                                        'overflow': 'visible'
                                    }
                                ),

                                html.Label("Date Range:", className="font-weight-bold mb-1",
                                           style={'color': '#a0c8f0'}),
                                dcc.DatePickerRange(
                                    id='date-range',
                                    min_date_allowed=datetime(2015, 1, 1),
                                    max_date_allowed=datetime.now() + timedelta(days=1),
                                    initial_visible_month=datetime.now(),
                                    start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                                    end_date=datetime.now().strftime('%Y-%m-%d'),
                                    className="mb-3 w-100",
                                    display_format='YYYY-MM-DD',
                                    style={
                                        'backgroundColor': 'white',
                                        'border': 'none',
                                        'color': 'black',
                                        'fontWeight': 'bold'
                                    }
                                ),

                                html.Label("Station Selection:", className="font-weight-bold mb-1",
                                           style={'color': '#a0c8f0'}),
                                dcc.Dropdown(
                                    id='station-dropdown',
                                    options=[{'label': s, 'value': s} for s in self.get_stations()],
                                    value='All Stations',
                                    className="mb-3",
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black'
                                    }
                                ),

                                html.Label("Data Type:", className="font-weight-bold mb-1", style={'color': '#a0c8f0'}),
                                dcc.Dropdown(
                                    id='quarry-dropdown',
                                    options=[
                                        {'label': 'Default Sensor Data', 'value': 'default'},
                                        {'label': 'Tidal Data', 'value': 'tides'}
                                    ],
                                    value='default',
                                    className="mb-3",
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black'
                                    }
                                ),

                                html.Label("Trendline Period:", className="font-weight-bold mb-1",
                                           style={'color': '#a0c8f0'}),
                                dcc.Dropdown(
                                    id='trendline-dropdown',
                                    options=[
                                        {'label': 'No Trendline', 'value': 'none'},
                                        {'label': 'All Period', 'value': 'all'},
                                        {'label': 'Last Two Decades', 'value': 'last_two_decades'},
                                        {'label': 'Last Decade', 'value': 'last_decade'}
                                    ],
                                    value='none',
                                    className="mb-3",
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black'
                                    }
                                ),

                                html.Label("Analysis Type:", className="font-weight-bold mb-1",
                                           style={'color': '#a0c8f0'}),
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
                                    className="mb-3",
                                    style={
                                        'backgroundColor': 'white',
                                        'color': 'black'
                                    }
                                ),

                                dbc.Row([
                                    dbc.Col(dbc.Checklist(
                                        id='show-anomalies',
                                        options=[{'label': 'Show Anomalies', 'value': 'show'}],
                                        value=[],
                                        switch=True,
                                        className="mb-2",
                                        style={'color': '#a0c8f0'}
                                    ), xs=12, md=6),
                                ]),
                                html.Label("Prediction Models:", className="font-weight-bold mt-3",
                                           style={'color': '#a0c8f0'}),
                                dbc.Checklist(
                                    id='prediction-models',
                                    options=[
                                        {'label': ' ARIMA', 'value': 'arima'},
                                        {'label': ' Prophet', 'value': 'prophet'}
                                    ],
                                    value=[],
                                    switch=True,
                                    inline=True,
                                    style={'color': '#a0c8f0'}
                                ),

                                # Export buttons with same size
                                html.Div(
                                    className="export-buttons",
                                    style={
                                        'display': 'flex',
                                        'gap': '10px',
                                        'marginTop': '20px'
                                    },
                                    children=[
                                        dbc.Button(
                                            "Export Graph",
                                            id="export-graph-btn",
                                            className="w-50",
                                            style={
                                                'backgroundColor': '#1e6bc4',
                                                'border': 'none',
                                                'borderRadius': '6px',
                                                'fontWeight': 'bold',
                                                'flex': '1'
                                            }
                                        ),
                                        dbc.Button(
                                            "Export Table",
                                            id="export-table-btn",
                                            className="w-50",
                                            style={
                                                'backgroundColor': '#1e6bc4',
                                                'border': 'none',
                                                'borderRadius': '6px',
                                                'fontWeight': 'bold',
                                                'flex': '1'
                                            }
                                        )
                                    ]
                                )
                            ]
                        ),

                        # Right Panel - Content
                        html.Div(
                            className="content-area",
                            style={
                                'flex': '1',
                                'display': 'flex',
                                'flexDirection': 'column',
                                'gap': '20px'
                            },
                            children=[
                                # Stats Cards Row
                                html.Div(
                                    className="kpi-row",
                                    style={
                                        'display': 'flex',
                                        'gap': '10px',
                                        'flexWrap': 'wrap'
                                    },
                                    children=[
                                        self._create_stat_card("Current Level", "current-level", "#1e3c72"),
                                        self._create_stat_card("24h Change", "24h-change", "#2a4a8c"),
                                        self._create_stat_card("Avg. Temp", "avg-temp", "#1e3c72"),
                                        self._create_stat_card("Anomalies", "anomalies", "#2a4a8c"),
                                    ]
                                ),

                                # Tabs View
                                dbc.Tabs(
                                    id="view-tabs",
                                    active_tab="graph-tab",
                                    className="tab-header",
                                    children=[
                                        dbc.Tab(
                                            label="Graph View",
                                            tab_id="graph-tab",
                                            className="tab-panel",
                                            children=dbc.Card(
                                                className="graph-panel",
                                                children=dbc.CardBody(
                                                    dcc.Graph(
                                                        id="line-graph",
                                                        style={'height': '100%'},
                                                        config={'scrollZoom': True}
                                                    )
                                                )
                                            )
                                        ),
                                        dbc.Tab(
                                            label="Table View",
                                            tab_id="table-tab",
                                            className="tab-panel",
                                            children=dbc.Card(
                                                className="table-panel",
                                                children=dbc.CardBody(
                                                    dash_table.DataTable(
                                                        id='data-table',
                                                        page_current=0,
                                                        page_size=15,
                                                        page_action='native',
                                                        sort_action='native',
                                                        filter_action='native',
                                                        style_table={
                                                            'overflowX': 'auto',
                                                            'height': '100%',
                                                            'backgroundColor': '#142950'
                                                        },
                                                        style_header={
                                                            'backgroundColor': '#1e3c72',
                                                            'color': 'white',
                                                            'fontWeight': 'bold',
                                                            'border': '1px solid #2a4a8c'
                                                        },
                                                        style_data_conditional=[
                                                            {
                                                                'if': {'row_index': 'odd'},
                                                                'backgroundColor': '#1e3c72'
                                                            },
                                                            {
                                                                'if': {'row_index': 'even'},
                                                                'backgroundColor': '#2a4a8c'
                                                            }
                                                        ],
                                                        style_cell={
                                                            'color': 'white',
                                                            'textAlign': 'left',
                                                            'padding': '10px',
                                                            'border': '1px solid #2a4a8c'
                                                        },
                                                    )
                                                )
                                            )
                                        ),
                                        dbc.Tab(
                                            label="Map View",
                                            tab_id="map-tab",
                                            className="tab-panel",
                                            children=dbc.Card(
                                                className="map-panel",
                                                children=[
                                                    dbc.CardHeader(
                                                        dbc.Tabs(
                                                            id="map-type-tabs",
                                                            active_tab="osm-tab",
                                                            className="mb-3",
                                                            children=[
                                                                dbc.Tab(label="OSM Map", tab_id="osm-tab"),
                                                                dbc.Tab(label="GovMap", tab_id="govmap-tab"),
                                                            ]
                                                        )
                                                    ),
                                                    dbc.CardBody(
                                                        id="map-container",
                                                        style={'height': '500px'}
                                                    )
                                                ]
                                            )
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),

                # Stores and Download
                dcc.Store(id='current_graph_or_table_data'),
                dcc.Download(id="download-data"),
                dcc.Store(id='dummy-output-for-clientside'),
                dcc.Store(id='export-info-store'),
                dcc.Store(id='export-filename-store'),
                dcc.Store(id='stats-data-store'),

                # Interval for updating time
                dcc.Interval(
                    id='interval-component',
                    interval=1000,  # in milliseconds
                    n_intervals=0
                )
            ]
        )

    def _create_stat_card(self, title, id_suffix, color):
        """Create a statistic card component"""
        return html.Div(
            id=f"stat-card-{id_suffix}",
            className="kpi-card",
            style={
                'backgroundColor': color,
                'padding': '10px',
                'borderRadius': '8px',
                'flex': '1',
                'minWidth': '200px',
                'textAlign': 'center',
            },
            children=[
                html.Div(title, className="kpi-label", style={'fontSize': '0.9rem'}),
                html.Div(
                    id=f"stat-value-{id_suffix}",
                    className="kpi-value",
                    style={'fontSize': '1.4rem'}
                )
            ]
        )

    # ---------------------------
    # CALLBACKS & INTERACTIVITY
    # ---------------------------
    def register_optimized_callbacks(self, app):
        """Register all application callbacks"""

        # Update current time
        @app.callback(
            Output('current-time', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_time(n):
            return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Core data updates for graph, table data store, and stats
        @app.callback(
            [Output("line-graph", "figure"),
             Output('data-table', 'data'),
             Output('data-table', 'columns'),
             Output('stats-data-store', 'data')],  # Store stats data
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("station-dropdown", "value"),
             Input("quarry-dropdown", "value"),
             Input("show-anomalies", "value"),
             Input("prediction-models", "value"),
             Input("trendline-dropdown", "value"),
             Input("analysis-type", "value")]
        )
        def update_main_content(start_date, end_date, station, quarry_option,
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

            # Calculate stats
            stats = self.calculate_stats(df)

            if quarry_option == "tides":
                fig = self.create_tides_graph(df.copy())
            else:
                fig = self.create_sea_level_graph(df.copy(), trendline_period,
                                                  'show' in show_anomalies_val,
                                                  bool(prediction_models_val),
                                                  prediction_models_val,
                                                  station, analysis_type_val)
                # Prediction error feedback system
                prediction_errors = []

                if prediction_models_val:
                    if 'arima' in prediction_models_val:
                        arima_pred = self.arima_predict(station)
                        if not arima_pred:
                            prediction_errors.append("ARIMA: Not enough data to generate forecast")

                    if 'prophet' in prediction_models_val:
                        prophet_pred = self.prophet_predict(station)
                        if prophet_pred.empty:
                            prediction_errors.append("Prophet: Not enough data to generate forecast")

                # Add error annotations to figure if any
                if prediction_errors:
                    fig.add_annotation(
                        x=0.5,
                        y=0.95,  # Position at top of graph
                        xref="paper",
                        yref="paper",
                        text="<br>".join(prediction_errors),
                        showarrow=False,
                        bgcolor="rgba(200, 0, 0, 0.7)",
                        font=dict(size=14, color="white"),
                        align="center"
                    )

            # Store the current figure for export
            self.current_graph_data = fig

            # Prepare table content
            if not df_records:
                table_data, table_columns = [], [{"name": "Status", "id": "Status", "type": "text"}]
            else:
                table_data, table_columns = self.create_table_content(df_records, quarry_option)

            return fig, table_data, table_columns, stats

        # Update stats cards
        @app.callback(
            [Output('stat-value-current-level', 'children'),
             Output('stat-value-24h-change', 'children'),
             Output('stat-value-avg-temp', 'children'),
             Output('stat-value-anomalies', 'children'),
             Output('stat-card-24h-change', 'className')],  # Add class for color
            [Input('stats-data-store', 'data')]
        )
        def update_stats_cards(stats):
            if not stats:
                return ["N/A", "N/A", "N/A", "N/A", "kpi-card"]

            current_level = f"{stats['current_level']:.3f} m" if stats['current_level'] is not None else "N/A"

            # Return plain text value instead of styled component
            change_value = f"{stats['24h_change']:+.3f} m" if stats['24h_change'] is not None else "N/A"

            avg_temp = f"{stats['avg_temp']:.1f}°C" if stats['avg_temp'] is not None else "N/A"
            anomalies = str(stats['anomalies']) if stats['anomalies'] is not None else "N/A"

            # Determine color class based on value
            color_class = "kpi-card"
            if stats['24h_change'] is not None:
                color_class += " green" if stats['24h_change'] >= 0 else " red"

            return [current_level, change_value, avg_temp, anomalies, color_class]

        # Update map view
        @app.callback(
            Output('map-container', 'children'),
            [Input('map-type-tabs', 'active_tab'),
             Input('station-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_map_view(active_tab, station, start_date, end_date):
            if active_tab == 'govmap-tab':
                # Return GovMap iframe
                return html.Iframe(
                    src='/mapframe',
                    style={'width': '100%', 'height': '100%', 'border': 'none'}
                )
            else:
                # Return OSM map
                df = self.load_data_from_db(start_date, end_date, station, 'default')
                return dcc.Graph(
                    figure=self.handle_israel_map_view(df),
                    style={'height': '100%', 'width': '100%'}
                )

        # Table Export Handler
        @app.callback(
            Output("download-data", "data"),
            [Input("export-table-btn", "n_clicks")],
            [State('station-dropdown', 'value'),
             State('date-range', 'start_date'),
             State('date-range', 'end_date')],
            prevent_initial_call=True
        )
        def export_table(n_clicks, station, start_date, end_date):
            if not n_clicks:
                return no_update

            if self.current_table_df_for_export.empty:
                logging.warning("No table data available for export.")
                return no_update

            # Generate consistent filename
            filename = self.generate_export_filename(
                station,
                start_date,
                end_date,
                "csv"  # CSV extension for tables
            )
            logging.info(f"Exporting table data to {filename}...")

            return dcc.send_data_frame(
                self.current_table_df_for_export.to_csv,
                filename,
                index=False
            )

        # Station list updates (if stations are dynamic or searchable)
        @app.callback(
            Output('station-dropdown', 'options'),
            [Input('station-dropdown', 'search_value')],  # Triggered by search value to enable dynamic search
            prevent_initial_call=False
        )
        def update_station_options(search_value):
            all_stations = self.get_stations()
            if not search_value:
                return [{'label': s, 'value': s} for s in all_stations]
            # Filter stations based on search_value
            filtered_stations = [s for s in all_stations if search_value.lower() in s.lower()]
            return [{'label': s, 'value': s} for s in filtered_stations]

        # Export-Info-Store callback:

        @app.callback(
            Output('export-info-store', 'data'),
            [Input('export-graph-btn', 'n_clicks'),
             Input('export-table-btn', 'n_clicks')],  # Added table button
            [State('station-dropdown', 'value'),
             State('date-range', 'start_date'),
             State('date-range', 'end_date')]
        )
        def store_export_info(graph_clicks, table_clicks, station, start_date, end_date):
            ctx = callback_context
            if not ctx.triggered:
                return no_update

            # Get which button was clicked
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Only store when either button is clicked
            if button_id in ['export-graph-btn', 'export-table-btn']:
                return {
                    'station': station or "AllStations",
                    'start_date': start_date,
                    'end_date': end_date
                }
            return no_update

        # Export-Filename-Store callback
        @app.callback(
            Output('export-filename-store', 'data'),
            Input('export-info-store', 'data')
        )
        def generate_filename(export_info):
            if not export_info:
                return None
            return self.generate_export_filename(
                export_info['station'],
                export_info['start_date'],
                export_info['end_date'],
                "png"  # PNG extension for graphs
            )

        # Combined Graph Export Clientside Callback
        app.clientside_callback(
            """
            function(n_clicks, export_filename) {
                if (!n_clicks || !export_filename) return window.dash_clientside.no_update;

                setTimeout(function () {
                    const plotContainer = document.getElementById('line-graph');
                    if (!plotContainer) {
                        alert("Graph container not found.");
                        return;
                    }

                    const graphDiv = plotContainer.getElementsByClassName('js-plotly-plot')[0];
                    if (!graphDiv || !graphDiv.data || graphDiv.data.length === 0) {
                        alert("Graph is not rendered yet.");
                        return;
                    }

                    Plotly.toImage(graphDiv, {
                        format: 'png',
                        width: 1280,
                        height: 720,
                        scale: 2
                    }).then(url => {
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = export_filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }).catch(err => {
                        console.error("Export failed:", err);
                        alert("Export failed. See console for details.");
                    });
                }, 500);

                return window.dash_clientside.no_update;
            }
            """,
            Output("dummy-output-for-clientside", "data"),
            [Input("export-graph-btn", "n_clicks")],
            [State('export-filename-store', 'data')]
        )

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

    app_instance.server.run(host="Sea-Level-Dash-Local", port=8052, debug=True)

    try:
        if engine:
            engine.dispose()
            logging.info("Global SQLAlchemy engine disposed.")
    except NameError:
        logging.info("Global SQLAlchemy engine was not defined or already handled.")
    except Exception as e:
        logging.error(f"Error disposing global SQLAlchemy engine: {e}")