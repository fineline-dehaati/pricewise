import pandas as pd
from nl2code import guarded_pandas_plan
from analytics.anomalies import top_mom_jumps
from analytics.forecasting import prepare_ts, prophet_forecast
from charts import line_chart_from_df

def route_query(df: pd.DataFrame, question: str):
    q = question.lower()
    warnings, figs = [], []

    if any(k in q for k in ["forecast","predict","future","next"]):
        ts = prepare_ts(df)
        fcst = prophet_forecast(ts)
        figs.append(line_chart_from_df(ts, y="y", title="Actuals"))
        return ("Prophet forecast", None, fcst, figs, warnings)

    if any(k in q for k in ["anomaly","outlier","jump","spike"]):
        out = top_mom_jumps(df, top_n=10)
        return ("Top MoM jumps", None, out, figs, warnings)

    plan, code, result, generated_fig = guarded_pandas_plan(df, question)
    
    # Add generated figure to figs list if it exists
    if generated_fig is not None:
        figs.append(generated_fig)
        print(f"📊 Added generated figure to display list")
    
    print(f"🔍 Router received: plan={type(plan)}, code={type(code)}, result={type(result)}, generated_fig={type(generated_fig)}")
    
    return (plan, code, result, figs, warnings)
