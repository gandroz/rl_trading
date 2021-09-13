import matplotlib.pyplot as plt


def plot_summary(df, summary):
    plt.plot(df.Close.values)
    # buy signals
    x_buy = summary[summary.action=="buy"].step.values
    y_buy = df.Close.iloc[x_buy] * 1.1
    plt.scatter(x_buy, y_buy, marker="v")
    # sell signals
    x_sell = summary[summary.action=="sell"].step.values
    y_sell = df.Close.iloc[x_sell] * 0.9
    plt.scatter(x_sell, y_sell, marker="^")

    plt.show()