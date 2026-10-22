import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class InventoryOptimizationEngine:
    """
    Inventory Optimization Engine.
    Generates inventory replenishment recommendations based on forecasted 2026 demand
    and historical variability. Calculate safety stocks, reorder points, priorities,
    and lead times.
    """
    
    @staticmethod
    def generate_recommendations(
        forecast_df: pd.DataFrame, 
        historical_df: pd.DataFrame, 
        lead_time_months: int = 2,
        service_level_z: float = 1.645
    ) -> pd.DataFrame:
        """
        Input:
        - forecast_df: DataFrame with ['OrderMonth', 'ProductID' or 'CategoryName', 'Forecast_Sales' / 'Quantity' / 'Revenue']
          Must contain forecasts for 2026.
        - historical_df: DataFrame with historical monthly aggregates to calculate standard deviation and average.
        - lead_time_months: Default procurement lead time (e.g., 2 months).
        - service_level_z: Z-score for safety stock (1.645 corresponds to 95% service level).
        
        Returns:
        - DataFrame containing monthly recommendations with priority, safety stock, status, lead time, etc.
        """
        logger.info("Generating inventory planning recommendations...")
        
        # Check if ProductID or Category is the key
        group_key = 'ProductID' if 'ProductID' in forecast_df.columns else ('Category' if 'Category' in forecast_df.columns else None)
        if group_key is None:
            raise ValueError("Forecast DataFrame must contain 'ProductID' or 'Category' column.")
            
        # Ensure date columns are datetimes
        forecast_df = forecast_df.copy()
        forecast_df['OrderMonth'] = pd.to_datetime(forecast_df['OrderMonth'])
        historical_df = historical_df.copy()
        historical_df['OrderMonth'] = pd.to_datetime(historical_df['OrderMonth'])
        
        # Calculate historical stats (mean and std of Sales) for each items
        hist_stats = historical_df.groupby(group_key).agg(
            HistMeanSales=('Sales', 'mean'),
            HistStdSales=('Sales', 'std')
        ).reset_index()
        
        # Merge stats into forecast
        df = forecast_df.merge(hist_stats, on=group_key, how='left')
        df['HistStdSales'] = df['HistStdSales'].fillna(df['HistMeanSales'] * 0.25).fillna(1.0) # fallback
        df['HistMeanSales'] = df['HistMeanSales'].fillna(1.0)
        
        # 1. Safety Stock = Z * Historical Std Dev * sqrt(LeadTime)
        df['SafetyStock'] = np.ceil(service_level_z * df['HistStdSales'] * np.sqrt(lead_time_months)).astype(int)
        # Ensure a minimum safety stock of 5 units
        df['SafetyStock'] = df['SafetyStock'].clip(lower=5)
        
        # 2. Expected Demand Increase: compared to the historical average
        df['ExpectedDemandIncrease'] = df['Forecast_Sales'] - df['HistMeanSales']
        df['ExpectedDemandIncreasePercent'] = (df['ExpectedDemandIncrease'] / df['HistMeanSales']) * 100.0
        
        # 3. Inventory Status
        df['InventoryStatus'] = 'Maintain Inventory'
        df.loc[df['ExpectedDemandIncreasePercent'] > 10.0, 'InventoryStatus'] = 'Increase Inventory'
        df.loc[df['ExpectedDemandIncreasePercent'] < -10.0, 'InventoryStatus'] = 'Reduce Inventory'
        
        # 4. Priority Level
        # Priority increases if demand is projected to spike sharply above normal capacity
        df['Priority'] = 'Medium'
        df.loc[df['ExpectedDemandIncreasePercent'] <= -10.0, 'Priority'] = 'Low'
        df.loc[df['ExpectedDemandIncreasePercent'] > 10.0, 'Priority'] = 'High'
        # Critical if increase is over 25% and demand is highly variable or high volume
        df.loc[
            (df['ExpectedDemandIncreasePercent'] > 25.0) & 
            ((df['HistStdSales'] / df['HistMeanSales']) > 0.3), 
            'Priority'
        ] = 'Critical'
        
        # 5. Procurement Lead Time and Procurement Month
        # Recommended Procurement Month is Lead Time months prior to the forecast month
        df['LeadTimeMonths'] = lead_time_months
        df['ProcurementMonth'] = df['OrderMonth'] - pd.DateOffset(months=lead_time_months)
        df['ProcurementMonthName'] = df['ProcurementMonth'].dt.strftime('%B %Y')
        df['ForecastMonthName'] = df['OrderMonth'].dt.strftime('%B %Y')
        
        # Clean columns structure
        keep_cols = [
            group_key, 'OrderMonth', 'ForecastMonthName', 'Forecast_Sales', 
            'HistMeanSales', 'ExpectedDemandIncrease', 'ExpectedDemandIncreasePercent',
            'SafetyStock', 'InventoryStatus', 'Priority', 'LeadTimeMonths', 'ProcurementMonthName'
        ]
        
        # Add ProductName/Category if ProductID was grouped and exists in forecast
        for meta_col in ['ProductName', 'Category']:
            if meta_col in forecast_df.columns:
                keep_cols.insert(1, meta_col)
                
        return df[keep_cols]

if __name__ == "__main__":
    pass
