import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_prediction_regression(
    X, results_dictionary, target_name, index_name="", report_type="platform"
):
    """
    Function to plot the results of the regression model
    """

    # specify if the plot is used by the platform, or it is downloaded by the user
    if report_type == "platform":
        cl = "white"
    elif report_type == "client":
        cl = "black"

    else:
        raise ValueError("Colour not selected!")

    # store the results' dictionary in a dataframe
    results_df = pd.DataFrame(results_dictionary)

    y = results_df.prediction

    figure = plt.figure(figsize=(12, 7), edgecolor=cl)  # figure parameters
    plt.plot(
        X.index.values,
        y,
        "o",
        markersize=10,
        linewidth=3,
        c="orange",
        label=target_name,
    )
    # TODO: implement confidence interval

    # title, x-axis, y-axis names
    plt.title("Predictions", fontsize=16, color=cl)
    plt.xlabel(index_name, fontsize=16, color=cl)
    plt.ylabel(target_name, fontsize=16, color=cl)
    # # legend
    legend = plt.legend(facecolor="grey", fontsize=16, framealpha=0.7)
    legend.get_frame().set_edgecolor("grey")
    plt.setp(legend.get_texts(), color=cl)
    # parameters for the figure axis and ticks
    plt.tick_params(axis="both", which="major", labelsize=14, colors=cl)
    plt.xticks(fontsize=15, color=cl, rotation=45)
    plt.yticks(fontsize=15, color=cl)
    plt.tight_layout()
    plt.grid(color=cl, linestyle=":", alpha=0.6)

    return figure


def plot_prediction_classification(
    X, results_dictionary, target_name, index_name="", report_type="platform"
):
    """Function to plot the prediction probability"""

    # colors
    # specify if the plot is used by the platform or it is downloaded by the user
    if report_type == "platform":
        cl = "white"
    elif report_type == "client":
        cl = "black"

    else:
        raise ValueError("Colour not selected!")

    # confidence display value
    z_value = 2

    # store the results' dictionary in a dataframe
    results_df = pd.DataFrame(results_dictionary)

    figure = plt.figure(figsize=(15, 6))

    # set figure limits
    plt.ylim((0, 100))
    plt.xlim((0, len(results_df) + 1))

    plt.title("Class probability prediction", fontsize=16, color=cl)
    plt.xlabel(index_name, fontsize=16, color=cl)
    plt.ylabel(target_name, fontsize=16, color=cl)

    plt.plot(
        results_df.decision_threshold * 100,
        "--",
        c="orange",
        linewidth=3,
        label="{}% decision threshold".format(
            results_dictionary["decision_threshold"] * 100
        ),
    )

    for idx, i in results_df.iterrows():
        # compute the predicted probability and thr CI at each step
        y = i.prediction_event_prob
        # CI = y + i.prediction_CI * z_value

        if i.prediction_event_prob < np.int(
            i.decision_threshold * 100 / 2
        ):  # probability of non-event less than threshold/2
            random_error = np.random.choice(np.arange(1, 3, 0.15))
            markers, caps, bars = plt.errorbar(
                idx, y, yerr=random_error * z_value, markersize=4, marker=">", c=cl
            )
            [bar.set_alpha(0.2) for bar in bars]

        elif (
            i.prediction_event_prob < i.decision_threshold * 100
        ):  # probability of non-event and associated error bar
            markers, caps, bars = plt.errorbar(
                idx, y, yerr=i.prediction_CI * z_value, markersize=4, marker=">", c=cl
            )
            [bar.set_alpha(0.2) for bar in bars]

        else:  # probability of event and associated error bar
            markers, caps, bars = plt.errorbar(
                idx,
                y,
                yerr=i.prediction_CI * z_value,
                marker="X",
                markersize=13,
                mfc="red",
                c="orange",
            )
            [bar.set_alpha(0.7) for bar in bars]

    plt.tick_params(axis="both", which="major", labelsize=14, colors=cl)
    plt.xticks(fontsize=15, color=cl)
    plt.yticks(fontsize=15, color=cl)
    plt.tight_layout()

    legend = plt.legend(facecolor="grey", fontsize=16, framealpha=0.7)
    legend.get_frame().set_edgecolor("grey")
    plt.setp(legend.get_texts(), color=cl)

    return figure
