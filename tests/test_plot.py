import pandas as pd
from rltrading.plot import plot_summary


def test_plot():
    df = pd.read_csv('./data/AAPL.csv')
    df = df.sort_values('Date')
    summary = pd.read_csv("./data/res_APPL.csv")

    plot_summary(df, summary)
