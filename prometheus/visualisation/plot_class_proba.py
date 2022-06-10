import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_proba_anomaly(results_dictionary, report_type):
    # specify if the plot is used by the platform, or it is downloaded by the user
    if report_type == "platform":
        cl = "white"
    elif report_type == "client":
        cl = "black"

    else:
        raise ValueError("Colour not selected!")

    # confidence display value
    z_value = 2

    # store the results dictionary in a dataframe
    results_df = pd.DataFrame(results_dictionary)

    figure = plt.figure(figsize=(15, 6))

    plt.ylim((0, 100))
    plt.xlim((0, len(results_df) + 1))

    plt.xlabel("Sample no.", fontsize=16, color=cl)
    plt.ylabel("Probability of event [%]", fontsize=16, color=cl)

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

        if i.prediction_event_prob < np.int(
            i.decision_threshold * 100 / 2
        ):  # probability of non-event les than threshold/2
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

    legend = plt.legend(fontsize=14, framealpha=0.0)
    plt.setp(legend.get_texts(), color=cl)

    return figure
