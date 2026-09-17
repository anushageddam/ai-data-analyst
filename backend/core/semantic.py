import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple


class SemanticEngine:
    """
    Understands the dataset's business domain and semantic relationships.
    Classifies fields into Measures, Dimensions, Time Dimensions, and Identifiers.
    Detects business domains (Sales, SAP Procurement, Finance, HR, Logistics, Generic).
    """

    DOMAIN_SIGNATURES = {
        "Procurement / SAP ERP": {
            "keywords": ["po", "vendor", "material", "plant", "purchase_group", "net_value", "procurement", "requisition", "supplier", "rfq"],
            "primary_measure_candidates": ["net_value", "po_value", "amount", "spend", "total_price", "value", "quantity"],
            "primary_dim_candidates": ["vendor", "material_group", "material", "plant", "purchase_group", "business_unit", "department"],
            "time_candidates": ["po_date", "order_date", "delivery_date", "doc_date", "date"]
        },
        "Sales & Commerce": {
            "keywords": ["sales", "revenue", "customer", "product", "profit", "order", "discount", "segment", "store", "deal"],
            "primary_measure_candidates": ["sales", "revenue", "profit", "amount", "total", "net_sales", "turnover"],
            "primary_dim_candidates": ["category", "sub_category", "product_name", "product", "region", "segment", "customer_name", "city", "country"],
            "time_candidates": ["order_date", "sale_date", "invoice_date", "ship_date", "date", "year", "month"]
        },
        "Finance & Accounting": {
            "keywords": ["ebitda", "expense", "budget", "cost_center", "ledger", "asset", "liability", "cash_flow", "operating", "tax"],
            "primary_measure_candidates": ["amount", "budget", "actual", "variance", "expense", "cost", "margin", "ebitda"],
            "primary_dim_candidates": ["cost_center", "account", "department", "entity", "category", "project"],
            "time_candidates": ["period", "fiscal_year", "date", "month", "quarter"]
        },
        "Human Resources": {
            "keywords": ["employee", "salary", "compensation", "tenure", "headcount", "attrition", "job_title", "performance_rating"],
            "primary_measure_candidates": ["salary", "compensation", "bonus", "tenure", "age", "rating"],
            "primary_dim_candidates": ["department", "job_title", "location", "gender", "education", "level"],
            "time_candidates": ["hire_date", "exit_date", "review_date", "date"]
        },
        "Inventory & Logistics": {
            "keywords": ["inventory", "stock", "warehouse", "reorder", "shipment", "carrier", "lead_time", "sku", "transit"],
            "primary_measure_candidates": ["stock_level", "quantity_on_hand", "reorder_point", "lead_time_days", "shipping_cost"],
            "primary_dim_candidates": ["warehouse", "carrier", "item_category", "location", "status"],
            "time_candidates": ["ship_date", "arrival_date", "date"]
        }
    }

    def __init__(self, df: pd.DataFrame, profiler_output: Dict[str, Any]):
        self.df = df
        self.columns_by_type = profiler_output["columns_by_type"]
        self.column_profiles = {cp["name"]: cp for cp in profiler_output["columns"]}
        self.domain = self._detect_domain()
        self.semantic_mapping = self._classify_semantic_roles()

    def _detect_domain(self) -> str:
        """Determines the business domain using column names and data distribution."""
        col_names = [str(c).lower() for c in self.df.columns]
        scores = {}

        for domain, sig in self.DOMAIN_SIGNATURES.items():
            score = 0
            for kw in sig["keywords"]:
                for c in col_names:
                    if kw in c:
                        score += 3
                    elif any(part in kw for part in c.split('_')):
                        score += 1
            scores[domain] = score

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] >= 3:
            return best_domain
        return "General Business Analytics"

    def _classify_semantic_roles(self) -> Dict[str, Any]:
        """Classifies each column into analytical roles."""
        measures = []
        dimensions = []
        time_dimensions = []
        identifiers = []

        for col, profile in self.column_profiles.items():
            ctype = profile["type"]
            col_lower = str(col).lower()

            if ctype == "ID":
                identifiers.append(col)
            elif ctype == "DateTime":
                time_dimensions.append(col)
            elif ctype == "Numerical":
                # Check if it might be an ID or year
                if col_lower in ["year", "yr"] and profile["min"] and profile["min"] > 1900 and profile["max"] < 2100:
                    time_dimensions.append(col)
                elif profile["cardinality_ratio"] > 0.95 and not any(k in col_lower for k in ["sales", "profit", "amount", "price", "cost", "value", "qty", "quantity", "spend"]):
                    identifiers.append(col)
                else:
                    measures.append(col)
            elif ctype in ["Categorical", "Boolean"]:
                # If cardinality is too high (> 0.85 total rows), could be identifier
                if profile["cardinality_ratio"] > 0.85 and len(self.df) > 15:
                    identifiers.append(col)
                else:
                    dimensions.append(col)

        # Fallback date detection if none identified as DateTime
        if not time_dimensions:
            for col in self.df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ["date", "time", "day", "month", "period", "year"]):
                    time_dimensions.append(col)
                    if col in dimensions:
                        dimensions.remove(col)
                    elif col in measures:
                        measures.remove(col)

        # Pick primary time dimension
        primary_time = None
        if time_dimensions:
            # Sort by candidates from detected domain
            sig = self.DOMAIN_SIGNATURES.get(self.domain, {})
            time_cands = sig.get("time_candidates", [])
            for cand in time_cands:
                match = next((t for t in time_dimensions if cand in t.lower()), None)
                if match:
                    primary_time = match
                    break
            if not primary_time:
                primary_time = time_dimensions[0]

        # Pick primary measure
        primary_measure = None
        if measures:
            sig = self.DOMAIN_SIGNATURES.get(self.domain, {})
            measure_cands = sig.get("primary_measure_candidates", [])
            for cand in measure_cands:
                match = next((m for m in measures if cand in m.lower()), None)
                if match:
                    primary_measure = match
                    break
            if not primary_measure:
                # Pick numeric column with highest variance / scale
                primary_measure = measures[0]

        # Pick secondary measures
        secondary_measures = [m for m in measures if m != primary_measure]

        # Pick primary dimension
        primary_dim = None
        if dimensions:
            sig = self.DOMAIN_SIGNATURES.get(self.domain, {})
            dim_cands = sig.get("primary_dim_candidates", [])
            for cand in dim_cands:
                match = next((d for d in dimensions if cand in d.lower()), None)
                if match:
                    primary_dim = match
                    break
            if not primary_dim:
                # Pick dimension with clean cardinality (between 3 and 25 categories)
                suitable = [d for d in dimensions if 2 <= self.column_profiles[d]["unique_count"] <= 30]
                primary_dim = suitable[0] if suitable else dimensions[0]

        # Select meaningful filter dimensions (cardinality between 2 and 35)
        filter_dimensions = []
        for d in dimensions:
            ucnt = self.column_profiles[d]["unique_count"]
            if 2 <= ucnt <= 35 and not self.column_profiles[d]["is_constant"]:
                filter_dimensions.append(d)

        return {
            "domain": self.domain,
            "primary_time": primary_time,
            "time_dimensions": time_dimensions,
            "primary_measure": primary_measure,
            "secondary_measures": secondary_measures,
            "all_measures": measures,
            "primary_dimension": primary_dim,
            "all_dimensions": dimensions,
            "filter_dimensions": filter_dimensions[:5],  # Top 5 most useful filters
            "identifiers": identifiers
        }

    def get_understanding_summary(self) -> Dict[str, Any]:
        """Returns a user-friendly high level summary of the data semantic model."""
        return {
            "detected_domain": self.domain,
            "primary_time_dimension": self.semantic_mapping["primary_time"],
            "primary_measure": self.semantic_mapping["primary_measure"],
            "primary_dimension": self.semantic_mapping["primary_dimension"],
            "measures_count": len(self.semantic_mapping["all_measures"]),
            "dimensions_count": len(self.semantic_mapping["all_dimensions"]),
            "time_dimensions_count": len(self.semantic_mapping["time_dimensions"]),
            "filter_dimensions": self.semantic_mapping["filter_dimensions"],
            "semantic_roles": self.semantic_mapping
        }
