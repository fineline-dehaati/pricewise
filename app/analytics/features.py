import pandas as pd, numpy as np
from typing import Dict, List, Tuple, Optional

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detect column types in a price dataset.
    Returns a mapping of detected column types to actual column names.
    """
    column_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for date columns
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'created', 'updated']):
            if _is_date_column(df[col]):
                column_mapping['date'] = col
                continue
        
        # Check for ID columns
        if any(keyword in col_lower for keyword in ['id', 'sku', 'code', 'ref', 'number']):
            if _is_id_column(df[col]):
                column_mapping['item_id'] = col
                continue
        
        # Check for name/description columns
        if any(keyword in col_lower for keyword in ['name', 'title', 'description', 'product', 'item']):
            if _is_name_column(df[col]):
                column_mapping['item_name'] = col
                continue
        
        # Check for price/amount columns
        if any(keyword in col_lower for keyword in ['price', 'cost', 'amount', 'value', 'rate', 'fee']):
            if _is_price_column(df[col]):
                column_mapping['price'] = col
                continue
    
    # If we didn't find a date column, try to infer from datetime columns
    if 'date' not in column_mapping:
        for col in df.columns:
            if _is_date_column(df[col]):
                column_mapping['date'] = col
                break
    
    # If we didn't find an ID column, use the first column that looks like an ID
    if 'item_id' not in column_mapping:
        for col in df.columns:
            if _is_id_column(df[col]):
                column_mapping['item_id'] = col
                break
    
    # If we didn't find a name column, use the first string column
    if 'item_name' not in column_mapping:
        for col in df.columns:
            if _is_name_column(df[col]):
                column_mapping['item_name'] = col
                break
    
    # If we didn't find a price column, use the first numeric column
    if 'price' not in column_mapping:
        for col in df.columns:
            if _is_price_column(df[col]):
                column_mapping['price'] = col
                break
    
    return column_mapping

def _is_date_column(series: pd.Series) -> bool:
    """Check if a column contains date-like data."""
    if series.dtype == 'object':
        # Try to convert to datetime with better format inference
        try:
            # Sample a few rows to test date parsing
            sample = series.dropna().head(10)
            if len(sample) > 0:
                # Try common date formats first
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        pd.to_datetime(sample, format=fmt, errors='raise')
                        return True
                    except:
                        continue
                # Fallback to auto-detection
                pd.to_datetime(sample, errors='raise')
                return True
        except:
            return False
    return pd.api.types.is_datetime64_any_dtype(series)

def _is_id_column(series: pd.Series) -> bool:
    """Check if a column contains ID-like data."""
    if series.dtype == 'object':
        # Check if it's mostly unique and contains alphanumeric patterns
        unique_ratio = series.nunique() / len(series)
        return unique_ratio > 0.8 and series.str.match(r'^[A-Za-z0-9\-_]+$').mean() > 0.7
    return False

def _is_name_column(series: pd.Series) -> bool:
    """Check if a column contains name-like data."""
    if series.dtype == 'object':
        # Check if it contains text with spaces and is not too unique
        has_spaces = series.str.contains(' ').mean() > 0.3
        unique_ratio = series.nunique() / len(series)
        return has_spaces and unique_ratio < 0.9
    return False

def _is_price_column(series: pd.Series) -> bool:
    """Check if a column contains price-like data."""
    if pd.api.types.is_numeric_dtype(series):
        # Check if values are positive and reasonable for prices
        positive_ratio = (series > 0).mean()
        reasonable_range = ((series >= 0.01) & (series <= 1000000)).mean()
        return positive_ratio > 0.8 and reasonable_range > 0.8
    return False

def normalize_schema(df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize a dataframe to use standard column names.
    Returns normalized dataframe and column mapping.
    """
    if column_mapping is None:
        column_mapping = detect_column_types(df)
    
    # Create a copy and rename columns
    normalized_df = df.copy()
    
    # Rename columns to standard names
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    normalized_df = normalized_df.rename(columns=reverse_mapping)
    
    # Ensure required columns exist, create defaults if missing
    if 'date' not in normalized_df.columns:
        # Create a dummy date column if none exists
        normalized_df['date'] = pd.Timestamp.now()
    
    if 'item_id' not in normalized_df.columns:
        # Create sequential IDs if none exist
        normalized_df['item_id'] = [f'LOCATION_{i:04d}' for i in range(len(normalized_df))]
    
    if 'item_name' not in normalized_df.columns:
        # Use item_id as name if none exist
        normalized_df['item_name'] = normalized_df['item_id']
    
    if 'price' not in normalized_df.columns:
        raise ValueError("No price column detected in the dataset")
    
    # Convert date column to datetime with format inference
    try:
        # Try common formats first
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
            try:
                normalized_df['date'] = pd.to_datetime(normalized_df['date'], format=fmt, errors='raise')
                break
            except:
                continue
        else:
            # Fallback to auto-detection
            normalized_df['date'] = pd.to_datetime(normalized_df['date'])
    except Exception as e:
        print(f"⚠️  Date parsing warning: {e}")
        # Create a dummy date if parsing fails
        normalized_df['date'] = pd.Timestamp.now()
    
    # Ensure price is numeric
    normalized_df['price'] = pd.to_numeric(normalized_df['price'], errors='coerce')
    
    # Rename columns to user-friendly names
    column_rename_map = {
        'date': 'datetime',
        'item_id': 'Location_Id',
        'item_name': 'Location_Name'
    }
    
    # Only rename columns that exist
    existing_columns = {old: new for old, new in column_rename_map.items() if old in normalized_df.columns}
    normalized_df = normalized_df.rename(columns=existing_columns)
    
    return normalized_df, column_mapping

def coerce_schema(df: pd.DataFrame)->pd.DataFrame:
    cols={c.lower():c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    date_col=pick("date","datetime","ts")
    price_col=pick("price","close","value")
    item_id=pick("item_id","sku","id")
    item_name=pick("item_name","product","name")
    out=pd.DataFrame()
    out["datetime"]=pd.to_datetime(df[date_col]) if date_col else pd.to_datetime(df.iloc[:,0],errors="coerce")
    out["Location_Id"]=df[item_id] if item_id else df.index.astype(str)
    out["Location_Name"]=df[item_name] if item_name else out["Location_Id"]
    out["price"]=pd.to_numeric(df[price_col],errors="coerce") if price_col else np.nan
    return out

def basic_profile(df: pd.DataFrame)->Dict:
    # Handle both old and new column names
    date_col = 'datetime' if 'datetime' in df.columns else 'date'
    id_col = 'Location_Id' if 'Location_Id' in df.columns else 'item_id'
    
    return {
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "date_min": str(df[date_col].min()),
        "date_max": str(df[date_col].max()),
        "items": int(df[id_col].nunique())
    }

def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to prevent PyArrow serialization issues"""
    df_clean = df.copy()
    
    # Convert datetime columns to string to avoid PyArrow issues
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Check if it's a datetime column
            if df_clean[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                df_clean[col] = df_clean[col].astype(str)
    
    # Ensure numeric columns are properly typed
    for col in df_clean.columns:
        if col == 'price' and df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            except:
                pass
    
    return df_clean
