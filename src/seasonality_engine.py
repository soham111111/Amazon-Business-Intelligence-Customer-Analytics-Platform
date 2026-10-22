import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SeasonalityEngine:
    """
    Analyzes seasonality, peaks, off-seasons, and trends for categories and top products.
    Generates actionable business inventory recommendations.
    """
    
    @staticmethod
    def analyze_category_seasonality(cat_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates for each category:
        - Monthly average demand
        - Seasonality Index (Monthly Avg / Overall Avg)
        - Peak Month & Lowest Month
        - YoY Growth
        - Demand Variability (Coefficient of Variation)
        """
        logger.info("Analyzing category seasonality...")
        df = cat_df.copy()
        df['MonthNum'] = df['OrderMonth'].dt.month
        df['Year'] = df['OrderMonth'].dt.year
        
        results = []
        categories = df['Category'].unique()
        
        for cat in categories:
            cat_data = df[df['Category'] == cat]
            
            # Overall mean
            overall_mean_sales = cat_data['Sales'].mean()
            if overall_mean_sales == 0:
                overall_mean_sales = 1e-5
                
            # Monthly averages
            monthly_avgs = cat_data.groupby('MonthNum')['Sales'].mean()
            # Reindex to ensure 1-12 range
            monthly_avgs = monthly_avgs.reindex(range(1, 13), fill_value=0)
            
            # Seasonality Index
            seasonal_indices = monthly_avgs / overall_mean_sales
            
            peak_month_num = seasonal_indices.idxmax()
            lowest_month_num = seasonal_indices.idxmin()
            
            # Month Names
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            
            # Calculate YoY Growth
            yearly_sales = cat_data.groupby('Year')['Sales'].sum()
            yoy_growth = 0.0
            if len(yearly_sales) >= 2:
                years_sorted = sorted(yearly_sales.index)
                last_year = years_sorted[-1]
                prev_year = years_sorted[-2]
                if yearly_sales[prev_year] > 0:
                    yoy_growth = (yearly_sales[last_year] - yearly_sales[prev_year]) / yearly_sales[prev_year]
            
            # Demand Variability (Coefficient of Variation)
            sales_std = cat_data['Sales'].std()
            sales_mean = cat_data['Sales'].mean()
            variability = (sales_std / sales_mean) if sales_mean > 0 else 0.0
            
            # Recommendations
            peak_m_name = month_names[peak_month_num]
            low_m_name = month_names[lowest_month_num]
            
            recommendation = (
                f"Increase procurement 2 months before peak in {peak_m_name}. "
                f"Hold safety stock of ~{int(overall_mean_sales * 0.25)} units during high demand. "
                f"Scale back warehouse operations in {low_m_name} (lowest month)."
            )
            
            for m in range(1, 13):
                results.append({
                    "Category": cat,
                    "MonthNum": m,
                    "MonthName": month_names[m],
                    "MonthlyAvgDemand": monthly_avgs[m],
                    "SeasonalityIndex": seasonal_indices[m],
                    "PeakMonth": peak_m_name,
                    "LowestMonth": low_m_name,
                    "YoYGrowth": yoy_growth,
                    "DemandVariability": variability,
                    "BusinessRecommendation": recommendation
                })
                
        return pd.DataFrame(results)

    @staticmethod
    def analyze_product_seasonality(prod_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates for each top product:
        - Peak Month
        - Off-Season (Months with Seasonality Index < 0.75)
        - Seasonality Strength (Std Dev of Seasonality Index)
        - Trend (Linear Regression slope of sales)
        - Business Recommendation
        """
        logger.info("Analyzing top product seasonality...")
        df = prod_df.copy()
        df['MonthNum'] = df['OrderMonth'].dt.month
        df['Year'] = df['OrderMonth'].dt.year
        
        results = []
        product_ids = df['ProductID'].unique()
        
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 
            5: 'May', 6: 'June', 7: 'July', 8: 'August', 
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
        for pid in product_ids:
            p_data = df[df['ProductID'] == pid].copy()
            p_name = p_data['ProductName'].iloc[0]
            cat_name = p_data['Category'].iloc[0]
            
            # Sort by date for trend
            p_data = p_data.sort_values('OrderMonth')
            
            # Overall mean
            overall_mean_sales = p_data['Sales'].mean()
            if overall_mean_sales == 0:
                overall_mean_sales = 1e-5
                
            # Monthly averages
            monthly_avgs = p_data.groupby('MonthNum')['Sales'].mean().reindex(range(1, 13), fill_value=0)
            
            # Seasonality Index
            seasonal_indices = monthly_avgs / overall_mean_sales
            
            peak_month_num = seasonal_indices.idxmax()
            off_season_months = seasonal_indices[seasonal_indices < 0.75].index.tolist()
            off_season_names = [month_names[m] for m in off_season_months]
            
            # Seasonality Strength
            strength = seasonal_indices.std()
            
            # Trend calculation (Slope)
            # Create a simple index x=[0,1,2,...]
            n_obs = len(p_data)
            if n_obs > 1:
                x = np.arange(n_obs)
                y = p_data['Sales'].values
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0.0
                
            trend_desc = "Declining" if slope < -0.05 else ("Expanding" if slope > 0.05 else "Stable")
            
            # Custom Recommendations
            peak_m_name = month_names[peak_month_num]
            off_m_str = ", ".join(off_season_names[:3]) if off_season_names else "None"
            
            rec_actions = []
            if trend_desc == "Declining":
                rec_actions.append("Reduce procurement volumes to avoid write-off inventory.")
            elif trend_desc == "Expanding":
                rec_actions.append("Increase general buffer capacity by 15-20% to capture demand growth.")
                
            if strength > 0.25:
                rec_actions.append(f"Highly seasonal product. Procure early ahead of {peak_m_name} peak.")
            else:
                rec_actions.append("Demand is flat. Maintain consistent cycle inventory.")
                
            if off_season_names:
                rec_actions.append(f"Scale down replenishment orders in off-season ({off_m_str}).")
                
            recommendation = " ".join(rec_actions)
            
            for m in range(1, 13):
                results.append({
                    "ProductID": pid,
                    "ProductName": p_name,
                    "Category": cat_name,
                    "MonthNum": m,
                    "MonthName": month_names[m],
                    "MonthlyAvgDemand": monthly_avgs[m],
                    "SeasonalityIndex": seasonal_indices[m],
                    "PeakMonth": peak_m_name,
                    "OffSeason": ", ".join(off_season_names) if off_season_names else "None",
                    "SeasonalityStrength": strength,
                    "Trend": trend_desc,
                    "Slope": slope,
                    "BusinessRecommendation": recommendation
                })
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test script runs
    pass
