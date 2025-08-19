import pandas as pd
def top_mom_jumps(df: pd.DataFrame, top_n:int=10)->pd.DataFrame:
    d=df.copy()
    d["date"]=pd.to_datetime(d["date"])
    d=d.sort_values("date")
    d["month"]=d["date"].dt.to_period("M").dt.to_timestamp()
    g=d.groupby(["item_id","item_name","month"],as_index=False)["price"].mean()
    g["mom_pct"]=g.groupby("item_id")["price"].pct_change()
    return g.sort_values("mom_pct",ascending=False).head(top_n)
