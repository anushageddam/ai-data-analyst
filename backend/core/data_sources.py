import os
import io
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class BaseDataSource:
    """Abstract base class for all YOO PROJECT data sources."""
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        raise NotImplementedError


class FileDataSource(BaseDataSource):
    """Handles CSV and Excel file ingestion with sheet detection."""
    def __init__(self, file_bytes: bytes, filename: str, sheet_name: Optional[str] = None):
        self.file_bytes = file_bytes
        self.filename = filename
        self.sheet_name = sheet_name
        self.is_excel = filename.lower().endswith(('.xlsx', '.xls'))
        self.is_csv = filename.lower().endswith('.csv')
        self._df: Optional[pd.DataFrame] = None
        self._sheets: List[str] = []

    def get_sheets(self) -> List[str]:
        if not self.is_excel:
            return []
        if not self._sheets:
            try:
                excel_file = pd.ExcelFile(io.BytesIO(self.file_bytes))
                self._sheets = excel_file.sheet_names
            except Exception as e:
                self._sheets = ["Sheet1"]
        return self._sheets

    def load(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        target_sheet = sheet_name or self.sheet_name

        if self.is_csv:
            # Try utf-8 first, fallback to latin1
            try:
                df = pd.read_csv(io.BytesIO(self.file_bytes), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(self.file_bytes), encoding="latin1")
        elif self.is_excel:
            sheets = self.get_sheets()
            selected = target_sheet if target_sheet in sheets else (sheets[0] if sheets else 0)
            df = pd.read_excel(io.BytesIO(self.file_bytes), sheet_name=selected)
        else:
            raise ValueError(f"Unsupported file format: {self.filename}")

        # Clean column names (strip spaces, replace special chars if problematic)
        df.columns = [str(c).strip() for c in df.columns]
        self._df = df
        return df

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "size_bytes": len(self.file_bytes),
            "is_excel": self.is_excel,
            "sheets": self.get_sheets(),
            "selected_sheet": self.sheet_name or (self.get_sheets()[0] if self.is_excel else None)
        }


class SQLDataSource(BaseDataSource):
    """
    Modular enterprise data source for SAP -> CDS -> ADF -> SQL Server / Azure SQL pipeline.
    Ready for connection string integration without re-architecting the application.
    """
    def __init__(self, connection_string: str, query: str, source_name: str = "Enterprise SQL"):
        self.connection_string = connection_string
        self.query = query
        self.source_name = source_name

    def load(self) -> pd.DataFrame:
        # In enterprise deployment, uses sqlalchemy / pyodbc
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(self.connection_string)
            df = pd.read_sql(self.query, engine)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as e:
            raise RuntimeError(f"Enterprise SQL pipeline connection error: {str(e)}")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source_type": "SQL_ENTERPRISE",
            "pipeline": "SAP -> CDS -> ADF -> SQL Server -> YOO PROJECT",
            "source_name": self.source_name,
            "query": self.query
        }


def load_sample_dataset(sample_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads pre-built enterprise datasets for instantaneous user exploration."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    
    samples = {
        "sap_procurement": {
            "file": os.path.join(data_dir, "sap_procurement.csv"),
            "name": "SAP_Procurement_S4HANA.csv",
            "domain_hint": "Procurement / SAP ERP"
        },
        "enterprise_sales": {
            "file": os.path.join(data_dir, "enterprise_sales.csv"),
            "name": "Global_Enterprise_Sales.csv",
            "domain_hint": "Sales & Profit Margin"
        },
        "sales_basic": {
            "file": os.path.join(data_dir, "sales.csv"),
            "name": "Quick_Sales_Sample.csv",
            "domain_hint": "Sales"
        }
    }

    if sample_id not in samples:
        sample_id = "sap_procurement"

    cfg = samples[sample_id]
    if os.path.exists(cfg["file"]):
        df = pd.read_csv(cfg["file"])
        df.columns = [str(c).strip() for c in df.columns]
        meta = {
            "filename": cfg["name"],
            "sample_id": sample_id,
            "domain_hint": cfg["domain_hint"],
            "size_bytes": os.path.getsize(cfg["file"]),
            "is_excel": False,
            "sheets": []
        }
        return df, meta
    else:
        raise FileNotFoundError(f"Sample dataset file not found: {cfg['file']}")
