"""
AI Executive Summary Generator (src/executive_summary.py)
Generates rich, structured management-level reporting from forecast, inventory, and seasonality data.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ExecutiveSummaryGenerator:
    """
    Compiles an executive-grade report from all generated artefacts.
    Covers: KPIs, growth, risk, seasonality, opportunities, recommendations.
    """

    def __init__(
        self,
        hist_df: pd.DataFrame,
        cat_fc: pd.DataFrame,
        prod_fc: pd.DataFrame,
        cat_seas: pd.DataFrame,
        prod_seas: pd.DataFrame,
        inv_plan: pd.DataFrame,
        insights_df: pd.DataFrame,
        model_comparison: pd.DataFrame,
    ):
        self.hist = hist_df.copy()
        self.hist['OrderDate'] = pd.to_datetime(self.hist['OrderDate'])
        self.last_hist_year = self.hist['OrderDate'].dt.year.max()

        self.cat_fc   = cat_fc.copy();  self.cat_fc['OrderMonth']  = pd.to_datetime(self.cat_fc['OrderMonth'])
        self.prod_fc  = prod_fc.copy(); self.prod_fc['OrderMonth'] = pd.to_datetime(self.prod_fc['OrderMonth'])
        self.cat_seas  = cat_seas.copy()
        self.prod_seas = prod_seas.copy()
        self.inv_plan  = inv_plan.copy()
        self.insights  = insights_df.copy()
        self.model_cmp = model_comparison.copy() if model_comparison is not None and not model_comparison.empty else pd.DataFrame()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _fc_for_year(self, year: int) -> pd.DataFrame:
        return self.cat_fc[self.cat_fc['OrderMonth'].dt.year == year]

    def _prod_fc_for_year(self, year: int) -> pd.DataFrame:
        return self.prod_fc[self.prod_fc['OrderMonth'].dt.year == year]

    def _hist_rev_by_cat(self) -> pd.Series:
        return self.hist[self.hist['OrderDate'].dt.year == self.last_hist_year].groupby('Category')['TotalAmount'].sum()

    def _hist_rev_by_prod(self) -> pd.Series:
        return self.hist[self.hist['OrderDate'].dt.year == self.last_hist_year].groupby('ProductID')['TotalAmount'].sum()

    def _cat_growth(self, year: int) -> pd.Series:
        fc_rev  = self._fc_for_year(year).groupby('Category')['Forecast_Revenue'].sum()
        hist_rev = self._hist_rev_by_cat()
        return ((fc_rev - hist_rev) / hist_rev.clip(lower=1)) * 100

    def _prod_growth(self, year: int) -> pd.Series:
        fc_rev   = self._prod_fc_for_year(year).groupby('ProductID')['Forecast_Revenue'].sum()
        hist_rev = self._hist_rev_by_prod()
        return ((fc_rev - hist_rev) / hist_rev.clip(lower=1)) * 100

    # ── public methods ────────────────────────────────────────────────────────

    def kpi_cards(self) -> Dict[str, Any]:
        """Returns a dict of headline KPI values shown at the top of the Executive page."""
        total_hist_rev = self.hist['TotalAmount'].sum()
        total_orders   = self.hist['OrderID'].nunique()
        total_customers= self.hist['CustomerID'].nunique()

        fc_rev_2025 = self._fc_for_year(2025)['Forecast_Revenue'].sum()
        fc_rev_2026 = self._fc_for_year(2026)['Forecast_Revenue'].sum()
        hist_yr_rev = self.hist[self.hist['OrderDate'].dt.year == self.last_hist_year]['TotalAmount'].sum()

        growth_2025 = ((fc_rev_2025 - hist_yr_rev) / max(hist_yr_rev, 1)) * 100
        growth_2026 = ((fc_rev_2026 - hist_yr_rev) / max(hist_yr_rev, 1)) * 100

        avg_mape = self.model_cmp['MAPE'].mean() if not self.model_cmp.empty else None

        return {
            "total_hist_revenue": total_hist_rev,
            "total_orders": total_orders,
            "total_customers": total_customers,
            "forecast_rev_2025": fc_rev_2025,
            "forecast_rev_2026": fc_rev_2026,
            "growth_2025_pct": growth_2025,
            "growth_2026_pct": growth_2026,
            "avg_mape": avg_mape,
        }

    def top_growing_categories(self, year: int = 2026, n: int = 10) -> pd.DataFrame:
        g = self._cat_growth(year).sort_values(ascending=False).head(n).reset_index()
        g.columns = ['Category', 'GrowthPct']
        return g

    def top_declining_categories(self, year: int = 2026, n: int = 5) -> pd.DataFrame:
        g = self._cat_growth(year).sort_values(ascending=True).head(n).reset_index()
        g.columns = ['Category', 'GrowthPct']
        return g

    def top_growing_products(self, year: int = 2026, n: int = 10) -> pd.DataFrame:
        g = self._prod_growth(year).sort_values(ascending=False).head(n).reset_index()
        g.columns = ['ProductID', 'GrowthPct']
        # Enrich with names
        name_map = self.prod_fc[['ProductID', 'ProductName', 'Category']].drop_duplicates('ProductID')
        return g.merge(name_map, on='ProductID', how='left')

    def top_declining_products(self, year: int = 2026, n: int = 5) -> pd.DataFrame:
        g = self._prod_growth(year).sort_values(ascending=True).head(n).reset_index()
        g.columns = ['ProductID', 'GrowthPct']
        name_map = self.prod_fc[['ProductID', 'ProductName', 'Category']].drop_duplicates('ProductID')
        return g.merge(name_map, on='ProductID', how='left')

    def most_seasonal_categories(self, n: int = 5) -> pd.DataFrame:
        s = self.cat_seas.groupby('Category')['SeasonalityIndex'].std().sort_values(ascending=False).head(n).reset_index()
        s.columns = ['Category', 'SeasonalityStrength']
        return s

    def highest_risk_items(self, n: int = 10) -> pd.DataFrame:
        if 'Priority' not in self.inv_plan.columns:
            return pd.DataFrame()
        critical = self.inv_plan[self.inv_plan['Priority'].isin(['Critical', 'High'])].copy()
        critical['PriorityScore'] = critical['Priority'].map({'Critical': 2, 'High': 1})
        critical = critical.sort_values('PriorityScore', ascending=False)
        id_col = 'ProductID' if 'ProductID' in critical.columns else 'Category'
        return critical[[id_col, 'Priority', 'SafetyStock', 'InventoryStatus']].drop_duplicates(id_col).head(n)

    def revenue_by_month(self, year: int = 2026) -> pd.DataFrame:
        fc = self._fc_for_year(year).copy()
        fc['MonthName'] = fc['OrderMonth'].dt.strftime('%B')
        m = fc.groupby(['OrderMonth', 'MonthName'])['Forecast_Revenue'].sum().reset_index()
        return m.sort_values('OrderMonth')

    def opportunities(self) -> List[str]:
        """Top 5 business opportunities extracted from insights and growth data."""
        opps = []
        top_cats = self.top_growing_categories(2026, 3)
        for _, row in top_cats.iterrows():
            opps.append(f"📈 **{row['Category']}** is projected to grow by **{row['GrowthPct']:.1f}%** — consider expanding product range and promotional spend.")
        top_prods = self.top_growing_products(2026, 2)
        for _, row in top_prods.iterrows():
            name = row.get('ProductName', row['ProductID'])
            opps.append(f"🏷️ **{str(name)[:40]}** shows strong demand growth — increase safety stock and ensure supplier contracts are renewed early.")
        return opps[:5]

    def risks(self) -> List[str]:
        """Top 5 business risks."""
        risk_items = self.highest_risk_items(5)
        risks_list = []
        id_col = 'ProductID' if 'ProductID' in risk_items.columns else 'Category'
        for _, row in risk_items.iterrows():
            risks_list.append(f"🚨 **{row[id_col]}** has **{row['Priority']}** inventory risk — action required before procurement deadline.")
        declining = self.top_declining_categories(2026, 2)
        for _, row in declining.iterrows():
            risks_list.append(f"📉 **{row['Category']}** demand is declining ({row['GrowthPct']:.1f}%) — reduce procurement volumes to avoid write-offs.")
        return risks_list[:5]

    def recommendations(self) -> List[str]:
        """Generates 8–12 executive inventory and procurement recommendations."""
        recs = []
        top_seas = self.most_seasonal_categories(3)
        peak_map = self.cat_seas.groupby('Category')['PeakMonth'].first()
        for _, row in top_seas.iterrows():
            cat = row['Category']
            peak = peak_map.get(cat, 'peak season')
            recs.append(f"📦 Increase **{cat}** inventory 2 months before **{peak}** to avoid stockout losses.")
        for _, row in self.top_growing_categories(2026, 3).iterrows():
            recs.append(f"🔼 Expand procurement budget for **{row['Category']}** by ~15% in 2025 ahead of 2026 demand peak.")
        for _, row in self.top_declining_categories(2026, 2).iterrows():
            recs.append(f"🔽 Phase down **{row['Category']}** order volumes — demand is declining at {abs(row['GrowthPct']):.1f}%.")
        high_vol_prods = self.inv_plan[self.inv_plan.get('Priority', pd.Series()) == 'Critical'] if 'Priority' in self.inv_plan else pd.DataFrame()
        if not high_vol_prods.empty:
            id_col = 'ProductID' if 'ProductID' in high_vol_prods.columns else 'Category'
            recs.append(f"🛡️ Maintain higher safety stock for **{high_vol_prods[id_col].nunique()}** Critical-priority products — buffer against demand spikes.")
        recs.append("🏭 Negotiate multi-year supplier contracts for top-10 growing categories to lock in pricing ahead of demand surge.")
        return recs[:10]

    def forecast_confidence(self) -> str:
        if self.model_cmp.empty:
            return "N/A"
        avg_mape = self.model_cmp['MAPE'].mean()
        if avg_mape < 10:
            return f"High ({100 - avg_mape:.1f}% accuracy, MAPE={avg_mape:.1f}%)"
        elif avg_mape < 20:
            return f"Moderate ({100 - avg_mape:.1f}% accuracy, MAPE={avg_mape:.1f}%)"
        else:
            return f"Low ({100 - avg_mape:.1f}% accuracy, MAPE={avg_mape:.1f}%)"

    def executive_conclusion(self) -> str:
        kpi = self.kpi_cards()
        g26 = kpi['growth_2026_pct']
        conf = self.forecast_confidence()
        top_cat = self.top_growing_categories(2026, 1)['Category'].iloc[0] if not self.top_growing_categories(2026, 1).empty else "N/A"
        return (
            f"The Amazon platform is forecast to achieve **{g26:.1f}% revenue growth** in 2026 compared to {self.last_hist_year} base. "
            f"**{top_cat}** leads category expansion. Forecast confidence is **{conf}**. "
            f"Key actions: prioritize procurement for high-growth and high-risk categories, maintain safety buffers for volatile products, "
            f"and leverage peak-season windows identified in the seasonality analysis to maximize revenue capture."
        )
