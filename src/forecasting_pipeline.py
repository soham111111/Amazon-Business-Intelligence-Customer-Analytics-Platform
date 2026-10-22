import pandas as pd
import numpy as np
import time
import logging
import warnings
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import Prophet, with a clean fallback indicator
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet is not available in the current environment. Fallback mechanisms will be used.")

class ForecastingPipeline:
    """
    Hierarchical Demand Forecasting Pipeline for 2026.
    Level 1: Categories (Prophet / Benchmarks)
    Level 2: Top Products (Prophet / Holt-Winters / Benchmarks)
    Level 3: Remaining Products (XGBoost / Category Share Fallback)
    """

    def __init__(self, biz_df: pd.DataFrame, cat_df: pd.DataFrame, prod_df: pd.DataFrame):
        self.biz_df = biz_df.copy()
        self.cat_df = cat_df.copy()
        self.prod_df = prod_df.copy()
        
        # Standardize dates
        for df in [self.biz_df, self.cat_df, self.prod_df]:
            df['OrderMonth'] = pd.to_datetime(df['OrderMonth'])
            
        self.evaluation_results = []

    def evaluate_univariate_models(self, series: pd.Series, dates: pd.Series, name: str) -> Tuple[str, pd.DataFrame]:
        """
        Evaluates Prophet, Holt-Winters, SARIMA, and XGBoost on a train/validation split.
        Train: first 48 months, Validation: last 12 months.
        Returns the name of the best performing model and evaluation metrics.
        """
        df_temp = pd.DataFrame({'ds': dates, 'y': series.values}).sort_values('ds').reset_index(drop=True)
        n = len(df_temp)
        
        if n < 24:
            # Insufficient data for rigorous evaluation, default to Holt-Winters or linear model
            return "Holt-Winters", pd.DataFrame()
            
        split_idx = n - 12
        train = df_temp.iloc[:split_idx]
        test = df_temp.iloc[split_idx:]
        
        test_y = test['y'].values
        metrics = []

        # -------------------------------------------------------------
        # Model 1: Prophet
        # -------------------------------------------------------------
        if PROPHET_AVAILABLE:
            t0 = time.time()
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
                # Suppress Prophet logs
                import os
                import sys
                logging.getLogger('prophet').setLevel(logging.ERROR)
                
                m.fit(train)
                future = m.make_future_dataframe(periods=12, freq='M', include_history=False)
                forecast = m.predict(future)
                pred_y = forecast['yhat'].values
                t1 = time.time()
                metrics.append(self._calculate_metrics("Prophet", test_y, pred_y, t1 - t0))
            except Exception as e:
                logger.error(f"Prophet evaluation failed for {name}: {str(e)}")
                
        # -------------------------------------------------------------
        # Model 2: Holt-Winters (Exponential Smoothing)
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            m = ExponentialSmoothing(train['y'].values, seasonal='add', seasonal_periods=12).fit()
            pred_y = m.forecast(12)
            # Clip negative predictions
            pred_y = np.clip(pred_y, 0, None)
            t1 = time.time()
            metrics.append(self._calculate_metrics("Holt-Winters", test_y, pred_y, t1 - t0))
        except Exception as e:
            logger.error(f"Holt-Winters evaluation failed for {name}: {str(e)}")

        # -------------------------------------------------------------
        # Model 3: SARIMA
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            m = SARIMAX(train['y'].values, order=(1,1,1), seasonal_order=(1,1,1,12), 
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred_y = m.forecast(12)
            pred_y = np.clip(pred_y, 0, None)
            t1 = time.time()
            metrics.append(self._calculate_metrics("SARIMA", test_y, pred_y, t1 - t0))
        except Exception as e:
            logger.error(f"SARIMA evaluation failed for {name}: {str(e)}")

        # -------------------------------------------------------------
        # Model 4: XGBoost
        # -------------------------------------------------------------
        t0 = time.time()
        try:
            # Build lag and rolling features
            # Create feature engineering matrix
            df_feats = df_temp.copy()
            for lag in [1, 2, 3, 12]:
                df_feats[f"Lag{lag}"] = df_feats['y'].shift(lag)
            df_feats["RollingMean_3"] = df_feats['y'].shift(1).rolling(3, min_periods=1).mean()
            df_feats["RollingMedian_3"] = df_feats['y'].shift(1).rolling(3, min_periods=1).median()
            df_feats["RollingStd_3"] = df_feats['y'].shift(1).rolling(3, min_periods=1).std().fillna(0)
            df_feats["Month"] = df_feats['ds'].dt.month
            df_feats["Trend"] = np.arange(n)
            
            # Split
            train_feats = df_feats.iloc[12:split_idx]
            test_feats = df_feats.iloc[split_idx:]
            
            feature_cols = [c for c in train_feats.columns if c not in ['ds', 'y']]
            
            X_train = train_feats[feature_cols]
            y_train = train_feats['y'].values
            
            xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)
            
            # Recursive multi-step prediction on the test set to avoid lookahead bias
            pred_y = []
            history = list(df_temp.iloc[:split_idx]['y'].values)
            
            for step in range(12):
                # Prepare features for the step
                curr_month = test.iloc[step]['ds'].dt.month
                curr_trend = split_idx + step
                # Lags
                lag_1 = history[-1]
                lag_2 = history[-2]
                lag_3 = history[-3]
                lag_12 = history[-12]
                
                # Rolling stats
                roll_3_arr = history[-3:]
                roll_mean = np.mean(roll_3_arr)
                roll_med = np.median(roll_3_arr)
                roll_std = np.std(roll_3_arr)
                
                X_step = pd.DataFrame([{
                    "Lag1": lag_1, "Lag2": lag_2, "Lag3": lag_3, "Lag12": lag_12,
                    "RollingMean_3": roll_mean, "RollingMedian_3": roll_med, "RollingStd_3": roll_std,
                    "Month": curr_month, "Trend": curr_trend
                }])
                
                pred_val = max(0.0, float(xgb.predict(X_step[feature_cols])[0]))
                pred_y.append(pred_val)
                history.append(pred_val)
                
            t1 = time.time()
            metrics.append(self._calculate_metrics("XGBoost", test_y, np.array(pred_y), t1 - t0))
        except Exception as e:
            logger.error(f"XGBoost evaluation failed for {name}: {str(e)}")

        eval_df = pd.DataFrame(metrics)
        eval_df['Series'] = name
        
        # Pick best model (lowest RMSE)
        if not eval_df.empty:
            # Sort by RMSE
            best_model = eval_df.sort_values('RMSE').iloc[0]['Model']
        else:
            best_model = "Holt-Winters"
            
        return best_model, eval_df

    def _calculate_metrics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, elapsed_time: float) -> Dict[str, Any]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
        r2 = r2_score(y_true, y_pred)
        return {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "TrainingTime": elapsed_time
        }

    def train_and_forecast_univariate(self, df_series: pd.DataFrame, name: str, best_model: str) -> pd.DataFrame:
        """
        Trains the selected best model on all available history and forecasts 24 months ahead
        (covering 2025 and 2026), returning the 2026 portion.
        """
        df_temp = df_series[['OrderMonth', 'Sales']].rename(columns={'OrderMonth': 'ds', 'Sales': 'y'}).sort_values('ds').reset_index(drop=True)
        n = len(df_temp)
        
        # Generate 2025 and 2026 dates (24 months)
        last_date = df_temp['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=24, freq='MS')
        
        forecast_sales = []
        lower_bounds = []
        upper_bounds = []
        
        # Let's get predictions
        if best_model == "Prophet" and PROPHET_AVAILABLE:
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
                m.fit(df_temp)
                future = pd.DataFrame({'ds': future_dates})
                forecast = m.predict(future)
                forecast_sales = forecast['yhat'].clip(lower=0).values
                lower_bounds = forecast['yhat_lower'].clip(lower=0).values
                upper_bounds = forecast['yhat_upper'].clip(lower=0).values
            except Exception as e:
                logger.error(f"Prophet forecast failed for {name}, falling back to Holt-Winters: {e}")
                best_model = "Holt-Winters"
                
        if best_model == "Holt-Winters" or (best_model == "Prophet" and not PROPHET_AVAILABLE):
            try:
                m = ExponentialSmoothing(df_temp['y'].values, seasonal='add', seasonal_periods=12).fit()
                forecast_sales = m.forecast(24)
                forecast_sales = np.clip(forecast_sales, 0, None)
                # Compute mock intervals based on train residual standard error
                resid_std = np.std(m.resid)
                lower_bounds = np.clip(forecast_sales - 1.96 * resid_std, 0, None)
                upper_bounds = forecast_sales + 1.96 * resid_std
            except Exception as e:
                logger.error(f"Holt-Winters forecast failed for {name}, using naive fallback: {e}")
                forecast_sales = np.array([df_temp['y'].iloc[-12:].mean()] * 24)
                lower_bounds = forecast_sales * 0.7
                upper_bounds = forecast_sales * 1.3
                
        if best_model == "SARIMA":
            try:
                m = SARIMAX(df_temp['y'].values, order=(1,1,1), seasonal_order=(1,1,1,12),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                res = m.get_forecast(24)
                forecast_sales = np.clip(res.predicted_mean, 0, None)
                conf = res.conf_int(alpha=0.05)
                lower_bounds = np.clip(conf[:, 0], 0, None)
                upper_bounds = conf[:, 1]
            except Exception as e:
                logger.error(f"SARIMA forecast failed, using naive fallback: {e}")
                forecast_sales = np.array([df_temp['y'].iloc[-12:].mean()] * 24)
                lower_bounds = forecast_sales * 0.7
                upper_bounds = forecast_sales * 1.3
                
        if best_model == "XGBoost":
            try:
                # Regressor trained on complete history
                df_feats = df_temp.copy()
                for lag in [1, 2, 3, 12]:
                    df_feats[f"Lag{lag}"] = df_feats['y'].shift(lag)
                df_feats["RollingMean_3"] = df_feats['y'].shift(1).rolling(3, min_periods=1).mean()
                df_feats["RollingMedian_3"] = df_feats['y'].shift(1).rolling(3, min_periods=1).median()
                df_feats["RollingStd_3"] = df_feats['y'].shift(1).rolling(3, min_periods=1).std().fillna(0)
                df_feats["Month"] = df_feats['ds'].dt.month
                df_feats["Trend"] = np.arange(n)
                
                train_feats = df_feats.iloc[12:]
                feature_cols = [c for c in train_feats.columns if c not in ['ds', 'y']]
                
                xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
                xgb.fit(train_feats[feature_cols], train_feats['y'].values)
                
                # Recursive execution
                history = list(df_temp['y'].values)
                pred_y = []
                for step in range(24):
                    curr_month = future_dates[step].month
                    curr_trend = n + step
                    
                    lag_1 = history[-1]
                    lag_2 = history[-2]
                    lag_3 = history[-3]
                    lag_12 = history[-12]
                    
                    roll_3 = history[-3:]
                    roll_mean = np.mean(roll_3)
                    roll_med = np.median(roll_3)
                    roll_std = np.std(roll_3)
                    
                    X_step = pd.DataFrame([{
                        "Lag1": lag_1, "Lag2": lag_2, "Lag3": lag_3, "Lag12": lag_12,
                        "RollingMean_3": roll_mean, "RollingMedian_3": roll_med, "RollingStd_3": roll_std,
                        "Month": curr_month, "Trend": curr_trend
                    }])
                    
                    pred_val = max(0.0, float(xgb.predict(X_step[feature_cols])[0]))
                    pred_y.append(pred_val)
                    history.append(pred_val)
                
                forecast_sales = np.array(pred_y)
                # Compute mock intervals based on residuals std
                y_train_pred = xgb.predict(train_feats[feature_cols])
                resid_std = np.std(train_feats['y'].values - y_train_pred)
                lower_bounds = np.clip(forecast_sales - 1.96 * resid_std, 0, None)
                upper_bounds = forecast_sales + 1.96 * resid_std
            except Exception as e:
                logger.error(f"XGBoost forecast failed, using naive fallback: {e}")
                forecast_sales = np.array([df_temp['y'].iloc[-12:].mean()] * 24)
                lower_bounds = forecast_sales * 0.7
                upper_bounds = forecast_sales * 1.3
                
        # Build output dataframe covering both 2025 and 2026
        fc_df = pd.DataFrame({
            "OrderMonth": future_dates,
            "Forecast_Sales": forecast_sales,
            "Forecast_Sales_Lower": lower_bounds,
            "Forecast_Sales_Upper": upper_bounds,
            "BestModel": best_model
        })
        
        # Return all 24-month forecasts (2025 and 2026)
        return fc_df.reset_index(drop=True)

    def run_level1_categories(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes Level 1: Category Demand Forecasting.
        Returns:
        - category_forecasts: Forecasts for all categories for 2026.
        - metrics_report: Evaluation table.
        """
        logger.info("Starting LEVEL 1: Category Demand Forecasting...")
        categories = self.cat_df['Category'].unique()
        
        forecast_results = []
        metrics_reports = []
        
        for cat in categories:
            cat_data = self.cat_df[self.cat_df['Category'] == cat].sort_values('OrderMonth')
            logger.info(f"Evaluating models for category: {cat}")
            
            # We forecast BOTH Sales quantity AND Revenue
            # 1. Sales quantity
            best_model_sales, eval_df_sales = self.evaluate_univariate_models(
                cat_data['Sales'], cat_data['OrderMonth'], f"Category_{cat}_Sales"
            )
            if not eval_df_sales.empty:
                metrics_reports.append(eval_df_sales)
                
            fc_sales = self.train_and_forecast_univariate(cat_data, f"Category_{cat}_Sales", best_model_sales)
            
            # 2. Revenue
            # Let's map same best model or separate best model for revenue
            cat_rev_data = cat_data.rename(columns={'Revenue': 'Sales', 'Sales': 'Quantity_orig'})
            best_model_rev, eval_df_rev = self.evaluate_univariate_models(
                cat_rev_data['Sales'], cat_rev_data['OrderMonth'], f"Category_{cat}_Revenue"
            )
            if not eval_df_rev.empty:
                metrics_reports.append(eval_df_rev)
                
            fc_rev = self.train_and_forecast_univariate(cat_rev_data, f"Category_{cat}_Revenue", best_model_rev)
            
            # Merge sales & revenue forecasts
            fc_merged = fc_sales.rename(columns={
                'Forecast_Sales': 'Forecast_Sales',
                'Forecast_Sales_Lower': 'Forecast_Sales_Lower',
                'Forecast_Sales_Upper': 'Forecast_Sales_Upper'
            })
            fc_merged['Forecast_Revenue'] = fc_rev['Forecast_Sales']
            fc_merged['Forecast_Revenue_Lower'] = fc_rev['Forecast_Sales_Lower']
            fc_merged['Forecast_Revenue_Upper'] = fc_rev['Forecast_Sales_Upper']
            fc_merged['Category'] = cat
            fc_merged['BestModel_Revenue'] = best_model_rev
            
            forecast_results.append(fc_merged)
            
        cat_fc_df = pd.concat(forecast_results, ignore_index=True)
        reports_df = pd.concat(metrics_reports, ignore_index=True) if metrics_reports else pd.DataFrame()
        
        return cat_fc_df, reports_df

    def run_level2_top_products(self) -> pd.DataFrame:
        """
        Executes Level 2: Top Selling Products Forecasting.
        Top products are modeled individually.
        """
        logger.info("Starting LEVEL 2: Top Selling Products Forecasting...")
        product_ids = self.prod_df['ProductID'].unique()
        
        forecast_results = []
        
        for pid in product_ids:
            p_data = self.prod_df[self.prod_df['ProductID'] == pid].sort_values('OrderMonth')
            p_name = p_data['ProductName'].iloc[0]
            cat_name = p_data['Category'].iloc[0]
            
            logger.info(f"Forecasting top product: {pid} ({p_name[:20]})")
            
            # Forecast Sales
            best_model_sales, _ = self.evaluate_univariate_models(
                p_data['Sales'], p_data['OrderMonth'], f"Product_{pid}_Sales"
            )
            fc_sales = self.train_and_forecast_univariate(p_data, f"Product_{pid}_Sales", best_model_sales)
            
            # Forecast Revenue
            p_rev_data = p_data.rename(columns={'Revenue': 'Sales', 'Sales': 'Quantity_orig'})
            best_model_rev, _ = self.evaluate_univariate_models(
                p_rev_data['Sales'], p_rev_data['OrderMonth'], f"Product_{pid}_Revenue"
            )
            fc_rev = self.train_and_forecast_univariate(p_rev_data, f"Product_{pid}_Revenue", best_model_rev)
            
            # Merge
            fc_merged = fc_sales.rename(columns={
                'Forecast_Sales': 'Forecast_Sales',
                'Forecast_Sales_Lower': 'Forecast_Sales_Lower',
                'Forecast_Sales_Upper': 'Forecast_Sales_Upper'
            })
            fc_merged['Forecast_Revenue'] = fc_rev['Forecast_Sales']
            fc_merged['Forecast_Revenue_Lower'] = fc_rev['Forecast_Sales_Lower']
            fc_merged['Forecast_Revenue_Upper'] = fc_rev['Forecast_Sales_Upper']
            fc_merged['ProductID'] = pid
            fc_merged['ProductName'] = p_name
            fc_merged['Category'] = cat_name
            
            forecast_results.append(fc_merged)
            
        top_prod_fc_df = pd.concat(forecast_results, ignore_index=True)
        return top_prod_fc_df

    def run_level3_remaining_products(self, data_preprocessor: Any, top_product_fc: pd.DataFrame, category_fc: pd.DataFrame) -> pd.DataFrame:
        """
        Executes Level 3: Long-tail/Remaining Products Forecasting.
        Remaining products are modeled using XGBoost.
        For products with sparse/insufficient monthly history, their forecasts
        are derived from parent category forecasts using historical share ratio.
        """
        logger.info("Starting LEVEL 3: Remaining Products Forecasting...")
        
        # 1. Identify which products are NOT in Top Products
        # Load raw data from processor to get all active products
        raw_df = data_preprocessor.df
        all_products = raw_df['ProductID'].unique()
        top_products = set(self.prod_df['ProductID'].unique())
        remaining_products = [p for p in all_products if p not in top_products]
        
        if not remaining_products:
            logger.info("No remaining products to forecast in Level 3.")
            return pd.DataFrame()
            
        logger.info(f"Found {len(remaining_products)} remaining products.")
        
        # We need to construct monthly data for remaining products
        # Get start/end dates from history
        min_month = self.prod_df['OrderMonth'].min()
        max_month = self.prod_df['OrderMonth'].max()
        all_months = pd.period_range(start=min_month, end=max_month, freq='M')
        
        # Group and construct monthly series for all remaining products
        prod_index = pd.MultiIndex.from_product([remaining_products, all_months], names=['ProductID', 'OrderMonth'])
        
        # We can extract the subset of remaining products
        rem_raw = raw_df[raw_df['ProductID'].isin(remaining_products)]
        
        rem_agg = rem_raw.groupby(['ProductID', 'OrderMonth']).agg(
            Sales=('Quantity', 'sum'),
            Revenue=('TotalAmount', 'sum')
        ).reindex(prod_index, fill_value=0).reset_index()
        
        rem_agg['OrderMonth'] = rem_agg['OrderMonth'].dt.to_timestamp()
        
        # Join product metadata
        product_meta = raw_df[raw_df['ProductID'].isin(remaining_products)].groupby('ProductID').agg(
            ProductName=('ProductName', 'first'),
            Category=('Category', 'first')
        ).reset_index()
        rem_agg = rem_agg.merge(product_meta, on='ProductID', how='left')
        
        forecast_results = []
        
        # We define sparse history as: non-zero sales in less than 12 months, or total quantity sold < 15 units
        for pid in remaining_products:
            p_data = rem_agg[rem_agg['ProductID'] == pid].sort_values('OrderMonth')
            p_name = p_data['ProductName'].iloc[0]
            cat_name = p_data['Category'].iloc[0]
            
            total_quantity = p_data['Sales'].sum()
            nonzero_months = (p_data['Sales'] > 0).sum()
            
            # Category forecast dates for 2026
            cat_fc_subset = category_fc[category_fc['Category'] == cat_name].sort_values('OrderMonth').reset_index(drop=True)
            
            if total_quantity < 15 or nonzero_months < 12:
                # -------------------------------------------------------------
                # Hierarchy Share Fallback (Sparse Data)
                # -------------------------------------------------------------
                # Compute product's average monthly share relative to total parent category sales
                # We fetch category historical aggregates
                cat_hist = self.cat_df[self.cat_df['Category'] == cat_name].sort_values('OrderMonth').reset_index(drop=True)
                
                # Merge product sales and category sales to get ratio
                shares = []
                for _, row in p_data.iterrows():
                    val = row['Sales']
                    c_val = cat_hist[cat_hist['OrderMonth'] == row['OrderMonth']]['Sales'].values
                    c_val = c_val[0] if len(c_val) > 0 else 0
                    if c_val > 0:
                        shares.append(val / c_val)
                    else:
                        shares.append(0.0)
                        
                avg_share = np.mean(shares) if len(shares) > 0 else 0.0
                
                # Derive revenue ratio as well
                rev_shares = []
                for _, row in p_data.iterrows():
                    val = row['Revenue']
                    c_val = cat_hist[cat_hist['OrderMonth'] == row['OrderMonth']]['Revenue'].values
                    c_val = c_val[0] if len(c_val) > 0 else 0
                    if c_val > 0:
                        rev_shares.append(val / c_val)
                    else:
                        rev_shares.append(0.0)
                avg_rev_share = np.mean(rev_shares) if len(rev_shares) > 0 else 0.0
                
                # Scale 2026 Category Forecasts
                derived_fc = cat_fc_subset.copy()
                derived_fc['Forecast_Sales'] = derived_fc['Forecast_Sales'] * avg_share
                derived_fc['Forecast_Sales_Lower'] = derived_fc['Forecast_Sales_Lower'] * avg_share
                derived_fc['Forecast_Sales_Upper'] = derived_fc['Forecast_Sales_Upper'] * avg_share
                derived_fc['Forecast_Revenue'] = derived_fc['Forecast_Revenue'] * avg_rev_share
                derived_fc['Forecast_Revenue_Lower'] = derived_fc['Forecast_Revenue_Lower'] * avg_rev_share
                derived_fc['Forecast_Revenue_Upper'] = derived_fc['Forecast_Revenue_Upper'] * avg_rev_share
                
                derived_fc['ProductID'] = pid
                derived_fc['ProductName'] = p_name
                derived_fc['BestModel'] = "Category Share Fallback"
                
                keep_cols = [
                    'OrderMonth', 'Forecast_Sales', 'Forecast_Sales_Lower', 'Forecast_Sales_Upper', 
                    'Forecast_Revenue', 'Forecast_Revenue_Lower', 'Forecast_Revenue_Upper', 
                    'ProductID', 'ProductName', 'Category'
                ]
                forecast_results.append(derived_fc[keep_cols])
            else:
                # -------------------------------------------------------------
                # XGBoost Forecasting for Sufficient Data Remaining Products
                # -------------------------------------------------------------
                # Sales
                fc_sales = self.train_and_forecast_univariate(p_data, f"Product_{pid}_Sales", "XGBoost")
                # Revenue
                p_rev_data = p_data.rename(columns={'Revenue': 'Sales', 'Sales': 'Quantity_orig'})
                fc_rev = self.train_and_forecast_univariate(p_rev_data, f"Product_{pid}_Revenue", "XGBoost")
                
                fc_merged = fc_sales.rename(columns={
                    'Forecast_Sales': 'Forecast_Sales',
                    'Forecast_Sales_Lower': 'Forecast_Sales_Lower',
                    'Forecast_Sales_Upper': 'Forecast_Sales_Upper'
                })
                fc_merged['Forecast_Revenue'] = fc_rev['Forecast_Sales']
                fc_merged['Forecast_Revenue_Lower'] = fc_rev['Forecast_Sales_Lower']
                fc_merged['Forecast_Revenue_Upper'] = fc_rev['Forecast_Sales_Upper']
                fc_merged['ProductID'] = pid
                fc_merged['ProductName'] = p_name
                fc_merged['Category'] = cat_name
                
                logger.info(f"Fitted XGBoost for product: {pid}")
                forecast_results.append(fc_merged)
                
        rem_prod_fc_df = pd.concat(forecast_results, ignore_index=True)
        return rem_prod_fc_df

if __name__ == "__main__":
    pass
