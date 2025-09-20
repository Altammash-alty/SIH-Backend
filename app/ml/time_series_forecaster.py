"""
Time Series Forecasting using ARIMA and Prophet
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """Time series forecasting for disease patterns"""
    
    def __init__(self):
        self.arima_model = None
        self.prophet_model = None
        self.is_trained = False
        self.performance_metrics = {}
        self.forecast_horizon = 7  # days
    
    async def train(self, health_data: pd.DataFrame) -> Dict[str, Any]:
        """Train time series forecasting models"""
        try:
            # Prepare time series data
            ts_data = self._prepare_time_series_data(health_data)
            
            if ts_data.empty or len(ts_data) < 10:
                logger.warning("Insufficient data for time series training")
                return {'status': 'insufficient_data'}
            
            # Split data for validation
            split_idx = int(len(ts_data) * 0.8)
            train_data = ts_data[:split_idx]
            test_data = ts_data[split_idx:]
            
            # Train ARIMA model
            logger.info("Training ARIMA model...")
            arima_result = await self._train_arima(train_data)
            
            # Train Prophet model
            logger.info("Training Prophet model...")
            prophet_result = await self._train_prophet(train_data)
            
            # Evaluate models
            arima_metrics = self._evaluate_arima(arima_result, test_data)
            prophet_metrics = self._evaluate_prophet(prophet_result, test_data)
            
            self.performance_metrics = {
                'arima': arima_metrics,
                'prophet': prophet_metrics
            }
            
            self.is_trained = True
            logger.info("Time series forecasting models trained successfully")
            
            return {
                'status': 'success',
                'performance_metrics': self.performance_metrics,
                'forecast_horizon_days': self.forecast_horizon
            }
            
        except Exception as e:
            logger.error(f"Error training time series forecaster: {e}")
            raise
    
    async def predict(self, health_data: pd.DataFrame) -> Dict[str, Any]:
        """Make time series predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare time series data
            ts_data = self._prepare_time_series_data(health_data)
            
            if ts_data.empty:
                return {'error': 'No time series data available'}
            
            # Make ARIMA predictions
            arima_forecast = await self._predict_arima(ts_data)
            
            # Make Prophet predictions
            prophet_forecast = await self._predict_prophet(ts_data)
            
            # Combine predictions
            combined_forecast = self._combine_forecasts(arima_forecast, prophet_forecast)
            
            return {
                'arima_forecast': arima_forecast,
                'prophet_forecast': prophet_forecast,
                'combined_forecast': combined_forecast,
                'forecast_horizon_days': self.forecast_horizon
            }
            
        except Exception as e:
            logger.error(f"Error making time series prediction: {e}")
            raise
    
    def _prepare_time_series_data(self, health_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data from health reports"""
        if health_data.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime if needed
        if 'report_timestamp' in health_data.columns:
            health_data['date'] = pd.to_datetime(health_data['report_timestamp']).dt.date
        else:
            health_data['date'] = pd.to_datetime(health_data['created_at']).dt.date
        
        # Aggregate daily case counts
        daily_cases = health_data.groupby('date').size().reset_index(name='case_count')
        daily_cases['date'] = pd.to_datetime(daily_cases['date'])
        daily_cases = daily_cases.sort_values('date')
        
        # Fill missing dates with 0 cases
        date_range = pd.date_range(
            start=daily_cases['date'].min(),
            end=daily_cases['date'].max(),
            freq='D'
        )
        ts_data = pd.DataFrame({'date': date_range})
        ts_data = ts_data.merge(daily_cases, on='date', how='left')
        ts_data['case_count'] = ts_data['case_count'].fillna(0)
        
        return ts_data
    
    async def _train_arima(self, train_data: pd.DataFrame) -> Any:
        """Train ARIMA model"""
        try:
            # Simple ARIMA(1,1,1) model
            model = ARIMA(train_data['case_count'], order=(1, 1, 1))
            fitted_model = model.fit()
            self.arima_model = fitted_model
            return fitted_model
        except Exception as e:
            logger.warning(f"ARIMA training failed: {e}")
            return None
    
    async def _train_prophet(self, train_data: pd.DataFrame) -> Any:
        """Train Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_data = train_data[['date', 'case_count']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            model.fit(prophet_data)
            self.prophet_model = model
            return model
        except Exception as e:
            logger.warning(f"Prophet training failed: {e}")
            return None
    
    async def _predict_arima(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Make ARIMA predictions"""
        if self.arima_model is None:
            return {'error': 'ARIMA model not available'}
        
        try:
            # Make forecast
            forecast = self.arima_model.forecast(steps=self.forecast_horizon)
            forecast_ci = self.arima_model.get_forecast(steps=self.forecast_horizon).conf_int()
            
            # Create future dates
            last_date = ts_data['date'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(self.forecast_horizon)]
            
            return {
                'dates': [d.isoformat() for d in future_dates],
                'predictions': forecast.tolist(),
                'lower_bound': forecast_ci.iloc[:, 0].tolist(),
                'upper_bound': forecast_ci.iloc[:, 1].tolist(),
                'model': 'ARIMA'
            }
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return {'error': str(e)}
    
    async def _predict_prophet(self, ts_data: pd.DataFrame) -> Dict[str, Any]:
        """Make Prophet predictions"""
        if self.prophet_model is None:
            return {'error': 'Prophet model not available'}
        
        try:
            # Create future dataframe
            future = self.prophet_model.make_future_dataframe(periods=self.forecast_horizon)
            
            # Make forecast
            forecast = self.prophet_model.predict(future)
            
            # Extract forecast for future dates only
            future_forecast = forecast.tail(self.forecast_horizon)
            
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in future_forecast['ds']],
                'predictions': future_forecast['yhat'].tolist(),
                'lower_bound': future_forecast['yhat_lower'].tolist(),
                'upper_bound': future_forecast['yhat_upper'].tolist(),
                'model': 'Prophet'
            }
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return {'error': str(e)}
    
    def _combine_forecasts(self, arima_forecast: Dict, prophet_forecast: Dict) -> Dict[str, Any]:
        """Combine ARIMA and Prophet forecasts"""
        try:
            if 'error' in arima_forecast and 'error' in prophet_forecast:
                return {'error': 'Both models failed'}
            
            if 'error' in arima_forecast:
                return prophet_forecast
            
            if 'error' in prophet_forecast:
                return arima_forecast
            
            # Simple average of predictions
            arima_preds = arima_forecast['predictions']
            prophet_preds = prophet_forecast['predictions']
            
            combined_preds = [(a + p) / 2 for a, p in zip(arima_preds, prophet_preds)]
            
            return {
                'dates': arima_forecast['dates'],
                'predictions': combined_preds,
                'model': 'Ensemble (ARIMA + Prophet)'
            }
        except Exception as e:
            logger.error(f"Forecast combination failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_arima(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ARIMA model performance"""
        if model is None:
            return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}
        
        try:
            # Make predictions on test data
            predictions = model.forecast(steps=len(test_data))
            actual = test_data['case_count'].values
            
            return {
                'mae': float(mean_absolute_error(actual, predictions)),
                'mse': float(mean_squared_error(actual, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(actual, predictions))),
                'r2': float(r2_score(actual, predictions))
            }
        except Exception as e:
            logger.warning(f"ARIMA evaluation failed: {e}")
            return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}
    
    def _evaluate_prophet(self, model: Any, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate Prophet model performance"""
        if model is None:
            return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}
        
        try:
            # Make predictions on test data
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Get predictions for test period
            test_predictions = forecast.tail(len(test_data))['yhat'].values
            actual = test_data['case_count'].values
            
            return {
                'mae': float(mean_absolute_error(actual, test_predictions)),
                'mse': float(mean_squared_error(actual, test_predictions)),
                'rmse': float(np.sqrt(mean_squared_error(actual, test_predictions))),
                'r2': float(r2_score(actual, test_predictions))
            }
        except Exception as e:
            logger.warning(f"Prophet evaluation failed: {e}")
            return {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics
