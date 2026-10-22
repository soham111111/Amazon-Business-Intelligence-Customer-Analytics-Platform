"""
AI Business Assistant (src/ai_business_assistant.py)
Retrieves facts from pre-generated artefacts and composes natural-language answers.
Does NOT retrain or re-run any forecasting model.
"""
import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class AIBusinessAssistant:
    """
    Rule-based, retrieval-driven business QA system.
    Understands ~20 categories of questions about forecasts, seasonality,
    inventory, risks, and business insights.
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
        self.hist      = hist_df.copy();       self.hist['OrderDate'] = pd.to_datetime(self.hist['OrderDate'])
        self.cat_fc    = cat_fc.copy();        self.cat_fc['OrderMonth']  = pd.to_datetime(self.cat_fc['OrderMonth'])
        self.prod_fc   = prod_fc.copy();       self.prod_fc['OrderMonth'] = pd.to_datetime(self.prod_fc['OrderMonth'])
        self.cat_seas  = cat_seas.copy()
        self.prod_seas = prod_seas.copy()
        self.inv_plan  = inv_plan.copy()
        self.insights  = insights_df.copy()
        self.model_cmp = model_comparison.copy() if model_comparison is not None and not model_comparison.empty else pd.DataFrame()

        self._last_hist_year = self.hist['OrderDate'].dt.year.max()
        self._build_indexes()

    # ── Pre-compute indexes ───────────────────────────────────────────────────

    def _build_indexes(self):
        # Category growth for 2025 and 2026
        hist_cat = self.hist[self.hist['OrderDate'].dt.year == self._last_hist_year].groupby('Category')['TotalAmount'].sum()
        self._cat_growth = {}
        for yr in [2025, 2026]:
            fc_cat = self.cat_fc[self.cat_fc['OrderMonth'].dt.year == yr].groupby('Category')['Forecast_Revenue'].sum()
            g = ((fc_cat - hist_cat) / hist_cat.clip(lower=1)) * 100
            self._cat_growth[yr] = g.to_dict()

        # Product growth
        hist_prod = self.hist[self.hist['OrderDate'].dt.year == self._last_hist_year].groupby('ProductID')['TotalAmount'].sum()
        self._prod_growth = {}
        for yr in [2025, 2026]:
            fc_prod = self.prod_fc[self.prod_fc['OrderMonth'].dt.year == yr].groupby('ProductID')['Forecast_Revenue'].sum()
            g = ((fc_prod - hist_prod) / hist_prod.clip(lower=1)) * 100
            self._prod_growth[yr] = g.to_dict()

        # Seasonality indexes
        self._cat_peak = self.cat_seas.groupby('Category')['PeakMonth'].first().to_dict()
        self._cat_seasonality_strength = self.cat_seas.groupby('Category')['SeasonalityIndex'].std().to_dict()
        self._prod_peak = self.prod_seas.groupby('ProductID')['PeakMonth'].first().to_dict()
        self._prod_trend = self.prod_seas.groupby('ProductID')['Trend'].first().to_dict()
        self._prod_name  = self.prod_fc.groupby('ProductID')['ProductName'].first().to_dict() if 'ProductName' in self.prod_fc.columns else {}

        # Inventory priorities
        if not self.inv_plan.empty and 'Priority' in self.inv_plan.columns:
            id_col = 'ProductID' if 'ProductID' in self.inv_plan.columns else 'Category'
            self._critical_items = self.inv_plan[self.inv_plan['Priority'] == 'Critical'][id_col].unique().tolist()
            self._high_items     = self.inv_plan[self.inv_plan['Priority'] == 'High'][id_col].unique().tolist()
            self._reduce_items   = self.inv_plan[self.inv_plan['InventoryStatus'] == 'Reduce Inventory'][id_col].unique().tolist()
        else:
            self._critical_items = []; self._high_items = []; self._reduce_items = []

    # ── Intent Detection ──────────────────────────────────────────────────────

    def _detect_intent(self, q: str) -> str:
        q = q.lower()
        if re.search(r'grow|increase|expand|best invest|highest.*growth|top.*grow', q): return 'top_growing'
        if re.search(r'declin|fall|reduc|worst|slowest', q):                           return 'top_declining'
        if re.search(r'invest|recommend|buy|procure|stock.*up',  q):                   return 'investment'
        if re.search(r'critical|urgent|stockout|risk|danger',    q):                   return 'risk'
        if re.search(r'safe.*reduc|can.*cut|slow mov',           q):                   return 'reduce_stock'
        if re.search(r'seasonal|peak|season|holiday',            q):                   return 'seasonality'
        if re.search(r'highest.*month|month.*highest|which month', q):                 return 'monthly_peak'
        if re.search(r'summar.*electronics|electronics.*summar', q):                   return 'cat_summary:Electronics'
        if re.search(r'summar.*clothing|clothing.*summar',       q):                   return 'cat_summary:Clothing'
        if re.search(r'summar.*home|kitchen.*summar',            q):                   return 'cat_summary:Home & Kitchen'
        if re.search(r'summar.*sport|sport.*summar',             q):                   return 'cat_summary:Sports'
        if re.search(r'management.*summar|executive.*summar|overall.*summar|generat.*summar', q): return 'exec_summary'
        if re.search(r'opportunit',                              q):                   return 'opportunities'
        if re.search(r'model|accuracy|mape|rmse|prophet|xgboost', q):                 return 'model_info'
        if re.search(r'safety.*stock|buffer',                    q):                   return 'safety_stock'
        if re.search(r'forecast.*2025|2025.*forecast',           q):                   return 'forecast_2025'
        if re.search(r'forecast.*2026|2026.*forecast',           q):                   return 'forecast_2026'
        return 'general'

    # ── Response Builder ──────────────────────────────────────────────────────

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Returns a structured response dict with keys:
        forecast, explanation, seasonality, business_insight,
        inventory_recommendation, risk_level, confidence_score, suggested_action
        """
        intent = self._detect_intent(question)
        handler = {
            'top_growing':   self._handle_top_growing,
            'top_declining': self._handle_top_declining,
            'investment':    self._handle_investment,
            'risk':          self._handle_risk,
            'reduce_stock':  self._handle_reduce,
            'seasonality':   self._handle_seasonality,
            'monthly_peak':  self._handle_monthly_peak,
            'exec_summary':  self._handle_exec_summary,
            'opportunities': self._handle_opportunities,
            'model_info':    self._handle_model_info,
            'safety_stock':  self._handle_safety_stock,
            'forecast_2025': lambda q: self._handle_forecast_year(q, 2025),
            'forecast_2026': lambda q: self._handle_forecast_year(q, 2026),
            'general':       self._handle_general,
        }.get(intent, self._handle_general)

        # Category-specific summaries
        if intent.startswith('cat_summary:'):
            cat = intent.split(':')[1]
            return self._handle_cat_summary(question, cat)

        return handler(question)

    # ── Intent Handlers ───────────────────────────────────────────────────────

    def _handle_top_growing(self, q: str) -> Dict:
        g26 = sorted(self._cat_growth.get(2026, {}).items(), key=lambda x: x[1], reverse=True)
        top_cats = g26[:5]
        lines = "\n".join([f"  {i+1}. **{c}**: +{v:.1f}% revenue growth" for i, (c, v) in enumerate(top_cats)])
        g_prod = sorted(self._prod_growth.get(2026, {}).items(), key=lambda x: x[1], reverse=True)
        top_prods = g_prod[:3]
        prod_lines = "\n".join([f"  {i+1}. **{self._prod_name.get(p, p)}** ({p}): +{v:.1f}%" for i, (p, v) in enumerate(top_prods)])
        return self._response(
            forecast=f"Top 5 growing categories for 2026:\n{lines}",
            explanation="Growth is computed by comparing 2026 forecast revenue against the last historical year.",
            seasonality=f"Peak seasons for {top_cats[0][0] if top_cats else 'N/A'}: {self._cat_peak.get(top_cats[0][0] if top_cats else '', 'N/A')}",
            insight=f"Top growing products:\n{prod_lines}",
            inv_rec="Increase procurement and inventory buffers for all high-growth categories.",
            risk="Low – demand is growing; stockout risk if procurement is not increased.",
            confidence=self._mape_to_conf(),
            action="Expand supplier contracts and pre-position inventory before peak season."
        )

    def _handle_top_declining(self, q: str) -> Dict:
        g26 = sorted(self._cat_growth.get(2026, {}).items(), key=lambda x: x[1])
        worst = g26[:5]
        lines = "\n".join([f"  {i+1}. **{c}**: {v:.1f}% revenue change" for i, (c, v) in enumerate(worst)])
        return self._response(
            forecast=f"Categories with declining demand in 2026:\n{lines}",
            explanation="Products are ranked by lowest projected revenue change versus historical baseline.",
            seasonality="Off-season purchasing patterns may amplify decline further.",
            insight="Declining categories often signal market saturation or changing consumer preferences.",
            inv_rec="Reduce procurement orders by 15–25% for declining categories. Avoid overstock.",
            risk="High – excess stock leads to write-offs and carrying cost losses.",
            confidence=self._mape_to_conf(),
            action="Phase down order volumes and negotiate flexible supplier terms."
        )

    def _handle_investment(self, q: str) -> Dict:
        g26 = sorted(self._cat_growth.get(2026, {}).items(), key=lambda x: x[1], reverse=True)
        top3_cats = [c for c, _ in g26[:3]]
        g_prod = sorted(self._prod_growth.get(2026, {}).items(), key=lambda x: x[1], reverse=True)
        top3_prods = [(p, self._prod_name.get(p, p), v) for p, v in g_prod[:3]]
        cat_str  = ", ".join(top3_cats)
        prod_str = "\n".join([f"  • {name} ({pid}): +{v:.1f}%" for pid, name, v in top3_prods])
        return self._response(
            forecast=f"Highest investment opportunities:\n**Categories**: {cat_str}\n**Products**:\n{prod_str}",
            explanation="Investment priority is ranked by 2026 forecast revenue growth over historical base.",
            seasonality="Align investment timing with seasonal peaks of recommended categories.",
            insight="Investing in high-growth categories ahead of demand peaks maximizes ROI.",
            inv_rec="Increase safety stock and procurement budget for identified categories by 10–20%.",
            risk="Medium – early investment carries demand uncertainty risk.",
            confidence=self._mape_to_conf(),
            action="Secure sourcing contracts for top-3 categories Q4 2024 (2 months before 2025 peaks)."
        )

    def _handle_risk(self, q: str) -> Dict:
        critical_count = len(self._critical_items)
        high_count = len(self._high_items)
        sample_critical = ", ".join([str(x) for x in self._critical_items[:5]])
        return self._response(
            forecast=f"**{critical_count} Critical** and **{high_count} High** priority items identified for 2025–2026.",
            explanation="Risk is computed from forecast volatility, safety stock shortfall, and demand change %.",
            seasonality="Seasonal peaks amplify stockout risk for critical items if replenishment is delayed.",
            insight=f"Critical items include: {sample_critical or 'None identified'}",
            inv_rec="Maintain 2x safety stock for Critical items. Trigger emergency reorder if stock falls below buffer.",
            risk="Critical",
            confidence=self._mape_to_conf(),
            action="Immediately review procurement timelines for all Critical-priority items. Escalate to supply chain team."
        )

    def _handle_reduce(self, q: str) -> Dict:
        sample = ", ".join([str(x) for x in self._reduce_items[:5]]) or "None identified"
        return self._response(
            forecast=f"**{len(self._reduce_items)} items** recommended for inventory reduction.",
            explanation="These items have declining forecast demand below historical average.",
            seasonality="These items' demand does not increase significantly during any seasonal window.",
            insight=f"Sample items safe to reduce: {sample}",
            inv_rec="Reduce reorder quantities by 20–30%. Consider liquidating slow-moving SKUs.",
            risk="Low – reducing over-stock protects working capital.",
            confidence=self._mape_to_conf(),
            action="Pause auto-replenishment for these items and clear existing stock via promotions."
        )

    def _handle_seasonality(self, q: str) -> Dict:
        strength = sorted(self._cat_seasonality_strength.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = "\n".join([f"  {i+1}. **{cat}** (strength={v:.2f}, peak={self._cat_peak.get(cat, 'N/A')})" for i, (cat, v) in enumerate(strength)])
        return self._response(
            forecast=f"Most seasonal categories:\n{lines}",
            explanation="Seasonality strength = std deviation of monthly seasonality index. Higher = more seasonal.",
            seasonality=lines,
            insight="Strong seasonality requires advance procurement planning to avoid peak-period shortages.",
            inv_rec="Trigger early restock 2 months before each category's peak month.",
            risk="High (for seasonal products) if procurement is not timed correctly.",
            confidence=self._mape_to_conf(),
            action="Update procurement calendar to align with above peak months. Set automated reorder alerts."
        )

    def _handle_monthly_peak(self, q: str) -> Dict:
        biz_fc = self.cat_fc.copy()
        biz_monthly = biz_fc.groupby('OrderMonth')['Forecast_Revenue'].sum().sort_values(ascending=False)
        peak_month = biz_monthly.index[0].strftime('%B %Y') if not biz_monthly.empty else "N/A"
        low_month  = biz_monthly.index[-1].strftime('%B %Y') if not biz_monthly.empty else "N/A"
        top5 = biz_monthly.head(5).reset_index()
        lines = "\n".join([f"  {r['OrderMonth'].strftime('%B %Y')}: ${r['Forecast_Revenue']:,.0f}" for _, r in top5.iterrows()])
        return self._response(
            forecast=f"**Highest revenue month**: {peak_month}\n**Lowest revenue month**: {low_month}\n\nTop 5 months:\n{lines}",
            explanation="Based on aggregated category-level revenue forecasts for 2025–2026.",
            seasonality=f"Peak at {peak_month} aligns with historical seasonal trends.",
            insight="Ensure maximum inventory availability and logistics capacity in peak months.",
            inv_rec=f"Pre-position stock by 2 months before {peak_month}.",
            risk="Medium – logistics bottlenecks during peak months increase delivery risk.",
            confidence=self._mape_to_conf(),
            action=f"Coordinate with warehouse and logistics teams to scale operations in Q leading into {peak_month}."
        )

    def _handle_cat_summary(self, q: str, cat: str) -> Dict:
        fc = self.cat_fc[self.cat_fc['Category'] == cat]
        if fc.empty:
            return self._response(forecast=f"No forecast data for category: {cat}", explanation="Category not found.", seasonality="N/A", insight="N/A", inv_rec="N/A", risk="N/A", confidence="N/A", action="Check category name and re-run.")
        rev_2025 = fc[fc['OrderMonth'].dt.year == 2025]['Forecast_Revenue'].sum()
        rev_2026 = fc[fc['OrderMonth'].dt.year == 2026]['Forecast_Revenue'].sum()
        peak = self._cat_peak.get(cat, 'N/A')
        g26 = self._cat_growth.get(2026, {}).get(cat, 0)
        return self._response(
            forecast=f"**{cat}** Forecast:\n  2025 Revenue: ${rev_2025:,.2f}\n  2026 Revenue: ${rev_2026:,.2f}\n  YoY Growth vs baseline: {g26:+.1f}%",
            explanation=f"Prophet/Holt-Winters model trained on 5-years of {cat} monthly data.",
            seasonality=f"Demand peaks in **{peak}**.",
            insight=f"{'Growing category – increase procurement.' if g26 > 0 else 'Declining category – review procurement strategy.'}",
            inv_rec=f"Increase stock 2 months before {peak}. {'Expand budget.' if g26 > 5 else 'Maintain current levels.'}",
            risk="Low" if g26 > 0 else "Medium",
            confidence=self._mape_to_conf(),
            action=f"{'Scale up procurement for ' + cat if g26 > 0 else 'Reduce procurement orders cautiously.'}"
        )

    def _handle_exec_summary(self, q: str) -> Dict:
        kpi = self._kpi_snapshot()
        return self._response(
            forecast=kpi,
            explanation="Consolidated from hierarchical Prophet + XGBoost forecast models.",
            seasonality="Category-specific peaks are identified and used for procurement scheduling.",
            insight="Top opportunities: growing categories. Top risks: critical-priority inventory items.",
            inv_rec="Focus procurement on high-growth, high-risk categories. Maintain safety stock buffers.",
            risk="Overall business risk: Medium (balanced growth with manageable volatility).",
            confidence=self._mape_to_conf(),
            action="Execute the 2025 inventory procurement plan. Review quarterly against actuals."
        )

    def _handle_opportunities(self, q: str) -> Dict:
        g26 = sorted(self._cat_growth.get(2026, {}).items(), key=lambda x: x[1], reverse=True)
        top = g26[:5]
        lines = "\n".join([f"  {i+1}. **{cat}** (+{v:.1f}%)" for i, (cat, v) in enumerate(top)])
        return self._response(
            forecast=f"Top 5 Business Opportunities for 2025–2026:\n{lines}",
            explanation="Ranked by projected revenue growth vs historical baseline.",
            seasonality="Leverage peak months of each category to maximize revenue capture.",
            insight="Opportunities are highest in categories with strong demand growth AND manageable variability.",
            inv_rec="Expand procurement budgets by 10–20% for top-opportunity categories.",
            risk="Low-Medium",
            confidence=self._mape_to_conf(),
            action="Brief senior leadership on these 5 categories and allocate additional CapEx in next budget cycle."
        )

    def _handle_model_info(self, q: str) -> Dict:
        if self.model_cmp.empty:
            info = "Model comparison data not available."
        else:
            best = self.model_cmp.sort_values('RMSE').groupby('Series').first().reset_index()
            counts = best['Model'].value_counts()
            info = "\n".join([f"  • **{m}**: best for {n} series" for m, n in counts.items()])
            avg_mape = self.model_cmp['MAPE'].mean()
            info += f"\n\nAverage MAPE across all series: **{avg_mape:.1f}%**"
        return self._response(
            forecast="Forecasting Model Summary:\n" + info,
            explanation="The hierarchical pipeline benchmarks Prophet, Holt-Winters, SARIMA, and XGBoost on a 12-month holdout.",
            seasonality="Prophet is preferred for series with clear seasonal patterns.",
            insight="XGBoost handles remaining products with engineered lag and rolling features.",
            inv_rec="Higher MAPE categories may require additional safety stock buffer.",
            risk="Low – ensemble architecture with automatic fallback mechanisms.",
            confidence=self._mape_to_conf(),
            action="Monitor monthly actuals vs forecast. Re-run pipeline quarterly for updated predictions."
        )

    def _handle_safety_stock(self, q: str) -> Dict:
        if self.inv_plan.empty or 'SafetyStock' not in self.inv_plan.columns:
            return self._response(forecast="Safety stock data not available.", explanation="N/A", seasonality="N/A", insight="N/A", inv_rec="N/A", risk="N/A", confidence="N/A", action="Run run_pipeline.py to generate inventory plan.")
        total_ss = self.inv_plan['SafetyStock'].sum()
        critical_ss = self.inv_plan[self.inv_plan['Priority'] == 'Critical']['SafetyStock'].mean() if 'Priority' in self.inv_plan.columns else 0
        return self._response(
            forecast=f"Total recommended safety stock across all products: **{total_ss:,.0f} units**\nAverage for Critical items: **{critical_ss:.0f} units/month**",
            explanation="Safety stock = Z(1.645) × σ(demand) × √(lead time). Based on 95% service level.",
            seasonality="Increase safety stock multiplier by 1.5x during seasonal peak months.",
            insight="High-volatility products have disproportionately higher safety stock requirements.",
            inv_rec="Maintain minimum safety stock at all times. Automate alerts at 50% buffer breach.",
            risk="High if safety stock drops below recommended level during peak season.",
            confidence=self._mape_to_conf(),
            action="Ensure warehouse space and capital reserves account for recommended safety stock levels."
        )

    def _handle_forecast_year(self, q: str, year: int) -> Dict:
        fc_yr = self.cat_fc[self.cat_fc['OrderMonth'].dt.year == year]
        rev_yr = fc_yr['Forecast_Revenue'].sum()
        qty_yr = fc_yr['Forecast_Sales'].sum()
        top_cat = fc_yr.groupby('Category')['Forecast_Revenue'].sum().idxmax() if not fc_yr.empty else "N/A"
        return self._response(
            forecast=f"**{year} Forecast Summary**\n  Total Revenue: ${rev_yr:,.2f}\n  Total Quantity: {qty_yr:,.0f} units\n  Top Category: {top_cat}",
            explanation=f"Generated by hierarchical forecasting models trained on 5-year historical data (2020–2024), projecting into {year}.",
            seasonality=f"Seasonal patterns from history are extrapolated into {year} by Prophet's Fourier seasonality components.",
            insight=f"{'2025 serves as a transition year. Trends established in 2025 will amplify into 2026.' if year==2025 else '2026 represents the full-growth horizon with accelerated demand in key categories.'}",
            inv_rec="Align inventory procurement schedule with 2025 forecast peaks to avoid shortages.",
            risk="Medium – forecast accuracy is highest in the nearest 12 months.",
            confidence=self._mape_to_conf(),
            action=f"Use this forecast as Q1 {year} planning baseline. Revisit monthly."
        )

    def _handle_general(self, q: str) -> Dict:
        insight_samples = []
        if not self.insights.empty:
            insight_samples = self.insights.head(3)['Description'].tolist()
        sample_txt = "\n".join([f"  • {s[:120]}" for s in insight_samples])
        return self._response(
            forecast="I can answer questions about: growth forecasts, declining products, investment opportunities, risk analysis, seasonality peaks, monthly demand, inventory safety stock, and model performance.",
            explanation="Please ask something specific, for example: 'Which category will grow the most?' or 'Show critical priority products'.",
            seasonality=f"Sample insights:\n{sample_txt}",
            insight="Try asking: 'Summarize Electronics forecast' or 'Which products need higher safety stock?'",
            inv_rec="Re-phrase your query with a specific category, product, or KPI topic.",
            risk="N/A",
            confidence=self._mape_to_conf(),
            action="Ask a specific business question above."
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _mape_to_conf(self) -> str:
        if self.model_cmp.empty: return "N/A"
        avg_mape = self.model_cmp['MAPE'].mean()
        return f"{max(0, 100 - avg_mape):.1f}% confidence (avg MAPE={avg_mape:.1f}%)"

    def _kpi_snapshot(self) -> str:
        rev25 = self.cat_fc[self.cat_fc['OrderMonth'].dt.year == 2025]['Forecast_Revenue'].sum()
        rev26 = self.cat_fc[self.cat_fc['OrderMonth'].dt.year == 2026]['Forecast_Revenue'].sum()
        hist_rev = self.hist['TotalAmount'].sum()
        g26 = ((rev26 - self.hist[self.hist['OrderDate'].dt.year==self._last_hist_year]['TotalAmount'].sum()) /
               max(self.hist[self.hist['OrderDate'].dt.year==self._last_hist_year]['TotalAmount'].sum(), 1)) * 100
        return (f"Historical Revenue (2020–{self._last_hist_year}): ${hist_rev:,.2f}\n"
                f"2025 Forecast Revenue: ${rev25:,.2f}\n"
                f"2026 Forecast Revenue: ${rev26:,.2f}\n"
                f"2026 Growth vs Base: {g26:+.1f}%")

    @staticmethod
    def _response(forecast, explanation, seasonality, insight, inv_rec, risk, confidence, action) -> Dict[str, str]:
        return {
            "Forecast": forecast,
            "Explanation": explanation,
            "Seasonality": seasonality,
            "Business Insight": insight,
            "Inventory Recommendation": inv_rec,
            "Risk Level": risk,
            "Confidence Score": confidence,
            "Suggested Action": action,
        }
