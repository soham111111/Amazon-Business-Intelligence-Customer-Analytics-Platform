import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.data_preprocessing import DataPreprocessor
from src.forecasting_pipeline import ForecastingPipeline
from src.seasonality_engine import SeasonalityEngine
from src.inventory_engine import InventoryOptimizationEngine
from src.insights_generator import BusinessInsightsGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing demand forecasting and business intelligence pipeline (2025–2026)…")

    for d in ["Data", "outputs", "reports", "models"]:
        os.makedirs(d, exist_ok=True)

    raw_data_path = os.path.join("Data", "Amazon_transaction_data.csv")
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data file not found: {raw_data_path}")
        return

    # ── STEP 1: Preprocess ───────────────────────────────────────────────────
    preprocessor = DataPreprocessor(raw_data_path)
    preprocessor.engineer_date_features()
    biz_agg, cat_agg, prod_agg = preprocessor.get_monthly_aggregates()
    preprocessor.df.to_csv("Data/amazon_transactions_clean.csv", index=False)
    logger.info("Saved cleaned transaction data.")

    # ── STEP 2: Hierarchical Forecasting ─────────────────────────────────────
    pipeline = ForecastingPipeline(biz_agg, cat_agg, prod_agg)

    # Level 1 – Categories (returns 2025 + 2026)
    cat_fc_df, metrics_df = pipeline.run_level1_categories()
    cat_fc_df.to_csv("outputs/category_forecast.csv", index=False)
    # year-split exports for Power BI
    for yr in [2025, 2026]:
        cat_fc_df[cat_fc_df['OrderMonth'].dt.year == yr].to_csv(f"outputs/category_forecast_{yr}.csv", index=False)
    metrics_df.to_csv("reports/model_comparison.csv", index=False)
    logger.info("Saved Level 1 Category Forecasts.")

    # Level 2 – Top Products
    top_prod_fc_df = pipeline.run_level2_top_products()
    # Level 3 – Remaining Products
    rem_prod_fc_df = pipeline.run_level3_remaining_products(preprocessor, top_prod_fc_df, cat_fc_df)

    product_fc_df = pd.concat([top_prod_fc_df, rem_prod_fc_df], ignore_index=True)
    product_fc_df.to_csv("outputs/product_forecast.csv", index=False)
    for yr in [2025, 2026]:
        product_fc_df[product_fc_df['OrderMonth'].dt.year == yr].to_csv(f"outputs/product_forecast_{yr}.csv", index=False)
    logger.info("Saved combined Product Level Forecasts.")

    # Business level roll-up
    biz_fc_df = cat_fc_df.groupby('OrderMonth').agg({
        'Forecast_Sales': 'sum', 'Forecast_Sales_Lower': 'sum', 'Forecast_Sales_Upper': 'sum',
        'Forecast_Revenue': 'sum', 'Forecast_Revenue_Lower': 'sum', 'Forecast_Revenue_Upper': 'sum'
    }).reset_index()
    biz_fc_df.to_csv("outputs/business_forecast.csv", index=False)
    for yr in [2025, 2026]:
        biz_fc_df[biz_fc_df['OrderMonth'].dt.year == yr].to_csv(f"outputs/business_forecast_{yr}.csv", index=False)

    # ── STEP 3: Seasonality ──────────────────────────────────────────────────
    cat_seasonality  = SeasonalityEngine.analyze_category_seasonality(cat_agg)
    prod_seasonality = SeasonalityEngine.analyze_product_seasonality(prod_agg)
    cat_seasonality.to_csv("reports/category_seasonality.csv", index=False)
    prod_seasonality.to_csv("reports/product_seasonality.csv", index=False)
    logger.info("Executed Seasonality Analysis.")

    # ── STEP 4: Inventory Engine ─────────────────────────────────────────────
    product_inventory_plan = InventoryOptimizationEngine.generate_recommendations(
        forecast_df=product_fc_df,
        historical_df=_to_hist(preprocessor.df, group_by='ProductID'),
        lead_time_months=2
    )
    product_inventory_plan.to_csv("outputs/inventory_plan.csv", index=False)

    cat_inventory_plan = InventoryOptimizationEngine.generate_recommendations(
        forecast_df=cat_fc_df,
        historical_df=cat_agg,
        lead_time_months=2
    )
    cat_inventory_plan.to_csv("outputs/category_inventory_plan.csv", index=False)
    logger.info("Executed Inventory Replenishment Engine.")

    # ── STEP 5: Business Insights ────────────────────────────────────────────
    all_insights = BusinessInsightsGenerator.generate_all_insights(
        historical_df=preprocessor.df,
        category_forecast_df=cat_fc_df,
        product_forecast_df=product_fc_df,
        category_seasonality=cat_seasonality,
        product_seasonality=prod_seasonality
    )
    insights_df = pd.DataFrame(all_insights)
    insights_df.to_csv("reports/business_insights.csv", index=False)
    logger.info("Saved dynamic business insights.")

    # ── STEP 6: Executive Summary ─────────────────────────────────────────────
    exec_summary = build_executive_summary(preprocessor.df, cat_fc_df, product_fc_df, cat_seasonality, prod_seasonality, cat_inventory_plan, insights_df)
    exec_summary.to_csv("reports/executive_summary.csv", index=False)
    logger.info("Saved Executive Summary.")

    # ── STEP 7: Forecast Charts ──────────────────────────────────────────────
    _generate_static_charts(biz_agg, biz_fc_df)

    logger.info("══════ PIPELINE COMPLETE (2025–2026) ══════")


def build_executive_summary(hist_df, cat_fc, prod_fc, cat_seas, prod_seas, inv_plan, insights_df):
    """Builds a structured executive summary table for the reports/ directory."""
    hist_df  = hist_df.copy().reset_index(drop=True)
    hist_df['OrderDate'] = pd.to_datetime(hist_df['OrderDate'])
    last_hist_year = hist_df['OrderDate'].dt.year.max()

    cat_fc = cat_fc.copy().reset_index(drop=True)
    cat_fc['OrderMonth'] = pd.to_datetime(cat_fc['OrderMonth'])
    cat_fc['Category'] = cat_fc['Category'].astype(str)

    prod_fc = prod_fc.copy().reset_index(drop=True)
    prod_fc['OrderMonth'] = pd.to_datetime(prod_fc['OrderMonth'])
    prod_fc['ProductID'] = prod_fc['ProductID'].astype(str)

    cat_seas = cat_seas.copy().reset_index(drop=True)
    cat_seas['Category'] = cat_seas['Category'].astype(str)

    prod_seas = prod_seas.copy().reset_index(drop=True)
    prod_seas['ProductID'] = prod_seas['ProductID'].astype(str)

    inv_plan = inv_plan.copy().reset_index(drop=True)

    rows = []

    for yr in [2025, 2026]:
        fc_yr  = cat_fc[cat_fc['OrderMonth'].dt.year == yr].copy()
        pfc_yr = prod_fc[prod_fc['OrderMonth'].dt.year == yr].copy()

        # --- category growth
        try:
            cat_rev       = fc_yr.groupby('Category', as_index=True)['Forecast_Revenue'].sum()
            hist_cat_rev  = hist_df[hist_df['OrderDate'].dt.year == last_hist_year].groupby('Category')['TotalAmount'].sum()
            cat_growth    = ((cat_rev - hist_cat_rev) / hist_cat_rev.clip(lower=1)) * 100
            highest_cat   = cat_growth.idxmax() if not cat_growth.empty else "N/A"
            declining_cat = cat_growth.idxmin() if not cat_growth.empty else "N/A"
        except Exception:
            highest_cat = declining_cat = "N/A"

        # --- product growth
        try:
            prod_rev       = pfc_yr.groupby('ProductID', as_index=True)['Forecast_Revenue'].sum()
            hist_prod_rev  = hist_df[hist_df['OrderDate'].dt.year == last_hist_year].groupby('ProductID')['TotalAmount'].sum()
            prod_growth    = ((prod_rev - hist_prod_rev) / hist_prod_rev.clip(lower=1)) * 100
            highest_prod   = prod_growth.idxmax() if not prod_growth.empty else "N/A"
            declining_prod = prod_growth.idxmin() if not prod_growth.empty else "N/A"
        except Exception:
            highest_prod = declining_prod = "N/A"

        # --- seasonality
        try:
            cat_strength      = cat_seas.groupby('Category', as_index=True)['SeasonalityIndex'].std()
            most_seasonal_cat = cat_strength.idxmax() if not cat_strength.empty else "N/A"
        except Exception:
            most_seasonal_cat = "N/A"

        try:
            seas_col           = 'SeasonalityStrength' if 'SeasonalityStrength' in prod_seas.columns else 'SeasonalityIndex'
            prod_strength      = prod_seas.groupby('ProductID', as_index=True)[seas_col].mean()
            most_seasonal_prod = str(prod_strength.idxmax()) if not prod_strength.empty else "N/A"
        except Exception:
            most_seasonal_prod = "N/A"

        # --- revenue month
        try:
            biz_monthly       = fc_yr.groupby('OrderMonth', as_index=True)['Forecast_Revenue'].sum()
            highest_rev_month = biz_monthly.idxmax().strftime('%B') if not biz_monthly.empty else "N/A"
            lowest_rev_month  = biz_monthly.idxmin().strftime('%B') if not biz_monthly.empty else "N/A"
        except Exception:
            highest_rev_month = lowest_rev_month = "N/A"

        # --- risk
        try:
            high_risk_cats   = inv_plan[inv_plan['Priority'] == 'Critical']['Category'].value_counts() if 'Category' in inv_plan.columns else pd.Series()
            highest_risk_cat = str(high_risk_cats.index[0]) if not high_risk_cats.empty else "N/A"
        except Exception:
            highest_risk_cat = "N/A"

        try:
            high_risk_prods   = inv_plan[inv_plan['Priority'] == 'Critical']['ProductID'].value_counts() if 'ProductID' in inv_plan.columns else pd.Series()
            highest_risk_prod = str(high_risk_prods.index[0]) if not high_risk_prods.empty else "N/A"
        except Exception:
            highest_risk_prod = "N/A"

        total_fc_rev   = fc_yr['Forecast_Revenue'].sum()
        total_fc_sales = fc_yr['Forecast_Sales'].sum()
        hist_yr_rev    = hist_df[hist_df['OrderDate'].dt.year == last_hist_year]['TotalAmount'].sum()
        growth_pct     = ((total_fc_rev - hist_yr_rev) / max(hist_yr_rev, 1)) * 100

        rows.append({
            "ForecastYear"         : yr,
            "ForecastRevenue"      : round(total_fc_rev, 2),
            "ForecastQuantity"     : round(total_fc_sales, 0),
            "OverallGrowthPct"     : round(growth_pct, 2),
            "HighestGrowthCategory": highest_cat,
            "DecliningCategory"    : declining_cat,
            "HighestGrowthProduct" : highest_prod,
            "DecliningProduct"     : declining_prod,
            "MostSeasonalCategory" : most_seasonal_cat,
            "MostSeasonalProduct"  : most_seasonal_prod,
            "HighestRevenueMonth"  : highest_rev_month,
            "LowestRevenueMonth"   : lowest_rev_month,
            "HighestRiskCategory"  : highest_risk_cat,
            "HighestRiskProduct"   : highest_risk_prod,
        })

    return pd.DataFrame(rows)


def _to_hist(raw_df, group_by='ProductID'):
    raw_df = raw_df.copy()
    raw_df['OrderMonth'] = pd.to_datetime(raw_df['OrderDate']).dt.to_period('M')
    df_group = raw_df.groupby([group_by, 'OrderMonth']).agg({'Quantity': 'sum', 'TotalAmount': 'sum'}).reset_index()
    df_group = df_group.rename(columns={'Quantity': 'Sales', 'TotalAmount': 'Revenue'})
    df_group['OrderMonth'] = df_group['OrderMonth'].dt.to_timestamp()
    return df_group


def _generate_static_charts(hist_biz, fc_biz):
    plt.figure(figsize=(14, 6))
    plt.plot(hist_biz['OrderMonth'], hist_biz['Sales'], label='Historical (2020–2024)', color='#232f3e', linewidth=2)
    plt.plot(fc_biz['OrderMonth'], fc_biz['Forecast_Sales'], label='Forecast (2025–2026)', color='#ff9900', linestyle='--', linewidth=2.5)
    plt.fill_between(fc_biz['OrderMonth'], fc_biz['Forecast_Sales_Lower'], fc_biz['Forecast_Sales_Upper'],
                     color='#ff9900', alpha=0.15, label='95% CI')
    plt.title('Amazon Platform — Demand Path: 5-Year History + 2025–2026 Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Timeline'); plt.ylabel('Monthly Units Sold')
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(); plt.tight_layout()
    plt.savefig("outputs/forecast_charts.png", dpi=150); plt.close()
    logger.info("Saved forecast chart to outputs/forecast_charts.png")


if __name__ == "__main__":
    main()
