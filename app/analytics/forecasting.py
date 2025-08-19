import pandas as pd
from prophet import Prophet

def prepare_ts(df: pd.DataFrame):
    d=df.copy()
    d["date"]=pd.to_datetime(d["date"])
    d=d.groupby("date",as_index=False)["price"].mean()
    return d.rename(columns={"date":"ds","price":"y"}).dropna()

def prophet_forecast(ts_df: pd.DataFrame, periods:int=90):
    m=Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True)
    m.fit(ts_df)
    future=m.make_future_dataframe(periods=periods)
    fcst=m.predict(future)
    return fcst[["ds","yhat","yhat_lower","yhat_upper"]]
