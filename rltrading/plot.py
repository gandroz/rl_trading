import matplotlib.pyplot as plt


def plot_summary(df, summary):
    plt.plot(df.Close.values)
    # buy signals
    x_buy = summary[summary.action=="buy"].step.values
    y_buy = df.Close.iloc[x_buy]
    plt.scatter(x_buy, y_buy, marker="^", c="green")
    # sell signals
    x_sell = summary[(summary.action=="sell")&(summary.shares_held>0)].step.values
    y_sell = df.Close.iloc[x_sell]
    plt.scatter(x_sell, y_sell, marker="v", c="red")

    plt.show()