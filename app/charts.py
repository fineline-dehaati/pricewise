import matplotlib.pyplot as plt
def line_chart_from_df(df,x="ds",y="y",title="Line Chart"):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(df[x],df[y]); ax.set_title(title); ax.grid(True,alpha=0.3)
    return fig
