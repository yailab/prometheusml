"""Module that includes the plot functions for the training pipeline"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from prometheus.algorithms.utils import compute_class


def _find_matthews_threshold(
    p_valid, y_valid, try_all=False, verbose=False, return_for_plot=False
):
    p_valid, y_valid = np.array(p_valid), np.array(y_valid)

    best_threshold = 0
    best_score = -2
    totry = np.arange(0, 1, 0.05) if try_all is False else np.arange(0, 1, 0.01)

    mcc = []
    for t in totry:
        score = matthews_corrcoef(y_valid, p_valid > t)
        mcc.append(score)

        if score > best_score:
            best_score = score
            best_threshold = t

    if verbose is True:
        print("Best score: ", round(best_score, 5), " @ threshold ", best_threshold)

    if return_for_plot:
        return np.array(mcc), np.array(totry)

    else:
        return best_threshold


def _plot_confusion_matrix(
    y: np.ndarray, y_hat_proba: np.ndarray, decision_threshold: float, cl: str
) -> matplotlib.figure.Figure:
    """Plot the confusion matrix"""

    y_hat = compute_class(
        y_hat_proba, decision_threshold
    )  # transform model's predicted probs to class labels
    confusion_m = confusion_matrix(y, y_hat)  # create confusion matrix

    grid_spec = {"width_ratios": (0.9, 0.05)}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_spec, figsize=(8, 6))

    # create the heatmap
    ax = sns.heatmap(
        confusion_m,
        ax=ax,
        annot=True,
        fmt=".0f",
        cmap=cm.OrRd,
        annot_kws={"size": 16},
        yticklabels=["True positive (TP)", "True negative (TN)"],
        xticklabels=["True positive (TP)", "True negative (TN)"],
        cbar_ax=cbar_ax,
    )

    ax.set_title("Confusion Matrix", fontsize=15, color=cl)  # set title
    ax.tick_params(
        axis="x", labelsize=15, colors=cl, labelcolor=cl
    )  # set x-axis labels
    ax.tick_params(
        axis="y", labelsize=15, colors=cl, labelrotation=90, labelcolor=cl
    )  # set y-axis labels

    ax.set_ylabel("True labels", fontsize=15, color=cl)
    ax.set_xlabel("Predicted labels", fontsize=15, color=cl)

    # adjust the colorbar
    cbar_ax.tick_params(labelsize=15, color=cl, labelcolor=cl)
    fig.tight_layout()

    plt.close(fig)  # remove the figure instance and allow it to be garbage collected

    return fig


def _plot_precision_recall_threshold(y: np.ndarray) -> matplotlib.figure.Figure:
    pass


def _plot_ROC_AUC_and_precision_recall_vs_threshold(
    y: np.ndarray, y_hat_proba: np.ndarray, cl: str, **kwargs
):
    """
    Function to plot ROC_AUC and the precision and recall versus the decision threshold for a classifier
    """

    # TODO: Refactor it, they seems to be things calculated twice
    # compute the matthews coefficient for each threshold
    mcc, d_th = _find_matthews_threshold(
        y_hat_proba.flatten()[1::2],
        y,
        try_all=False,
        verbose=True,
        return_for_plot=True,
    )

    # ROC-AUC and Precision - Recall based on the decision threshold
    threshold_vector = np.arange(0, 1, 0.05)
    # since the random forest is output in a probability and simply decide based on
    # the probability threshold when selected at the prediction, once can look across
    # the probability spectrum i.e. 0% to 100%

    precision_th = []  # precision based on the threshold
    recall_th = []  # recall based on the threshold
    fpr = []  # false positive rate
    tpr = []  # true positive rate

    for i in threshold_vector:
        y_hat_threshold = compute_class(y_hat_proba, decision_threshold=i)

        # calculate the confusion matrix and extract
        tn, fp, fn, tp = confusion_matrix(y, y_hat_threshold).ravel()

        # calculate precision and recall
        precision_th.append(tp / (tp + fp) * 100)
        recall_th.append(tp / (tp + fn) * 100)

        # calculate the true positive rate and the false positive rate
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)

        fpr.append(FPR)
        tpr.append(TPR)

    # ROC-AUC curve
    fpr.append(0)
    tpr.append(0)
    roc_auc = plt.figure(figsize=(12, 7), edgecolor=cl)  # figure parameters
    plt.plot(fpr, tpr, "--", linewidth=3, c="orange")
    plt.plot([0, 1], [0, 1], "--", c=cl)  # Dashed diagonal
    # title, x-axis, y-axis names
    plt.title("ROC-AUC curve", fontsize=16, color=cl)
    plt.xlabel("False Positive", fontsize=16, color=cl)
    plt.ylabel("True Positive Rate", fontsize=16, color=cl)
    # legend
    legend = plt.legend(fontsize=16, framealpha=0.0)
    plt.setp(legend.get_texts(), color=cl)
    # parameters for the figure axis and ticks
    plt.tick_params(axis="both", which="major", labelsize=14, colors=cl)
    plt.xticks(fontsize=15, color=cl)
    plt.yticks(fontsize=15, color=cl)
    plt.tight_layout()
    plt.grid(color=cl, linestyle=":")  # , alpha=0.6)

    # Precision/Recall as a function of the decision threshold
    fig_precision_recall_th, ax1 = plt.subplots(
        figsize=(12, 7), edgecolor=cl
    )  # figure parameters
    lns1 = ax1.plot(
        threshold_vector * 100,
        precision_th,
        "o-",
        markersize=7,
        c="orange",
        label="Precision",
    )
    lns2 = ax1.plot(
        threshold_vector * 100, recall_th, "o-", markersize=7, c=cl, label="Sensitivity"
    )
    ax2 = ax1.twinx()
    lns3 = ax2.plot(
        d_th * 100, mcc, "o-", markersize=7, c="sienna", label="Phi coefficient"
    )

    decisio_th = kwargs["decision_threshold"] * 100  # convert to percentage
    lns4 = ax1.plot(
        np.ones(len(threshold_vector)) * decisio_th,
        np.linspace(
            min(precision_th + recall_th),
            max(precision_th + recall_th),
            len(threshold_vector),
        ),
        "--",
        linewidth=2.5,
        color="red",
        label="{}% threshold".format(int(decisio_th)),
    )
    # title, x-axis, y-axis names
    ax1.set_title(
        "Precision and recall versus the decision threshold", fontsize=16, color=cl
    )
    ax1.set_xlabel("Probability threshold [%]", fontsize=16, color=cl)
    ax1.set_ylabel("Precision | Recall [%]", fontsize=16, color=cl)
    ax2.set_ylabel("Phi coefficient", fontsize=16, color=cl)

    # added these 4 lines into a single legend
    lns = lns1 + lns2 + lns3 + lns4
    labs = [line.get_label() for line in lns]
    legend = ax1.legend(
        lns,
        labs,
        fontsize=16,
        edgecolor=(0.26, 0.26, 0.26),
        facecolor=(0.26, 0.26, 0.26),
        framealpha=0.6,
        loc="lower center",
    )
    plt.setp(legend.get_texts(), color=cl)

    # # parameters for the figure axis and ticks
    ax1.tick_params(axis="both", which="major", labelsize=14, colors=cl)
    ax2.tick_params(axis="both", which="major", labelsize=14, colors=cl)

    # calculate the minimum to adjust axis
    min_axis = np.min([np.nanmin(precision_th), np.nanmin(recall_th)])

    ax1.set_xticks(np.arange(0, 101, 5))  # , labelsize=15, color=cl)
    ax2.set_xticks(np.arange(0, 101, 5))  # , labelsize=15, color=cl)
    ax1.set_yticks(np.arange(min_axis, 101, 10))  # , labelsize=15, color=cl)
    ax2.set_yticks(np.round(np.arange(-1.0, 1.0, 0.2), 2))

    ax1.grid(color=cl, linestyle=":")  # , alpha=0.6)
    # ax2.grid(color=cl, linestyle=':')  # , alpha=0.6)

    fig_precision_recall_th.tight_layout()

    plt.close("all")  # remove the figure instance and allow it to be garbage collected

    return roc_auc, fig_precision_recall_th


def plot_prediction_regression_cv(
    X,
    y,
    y_hat,
    no_of_model_parameters: int,
    target_name: str,
    index_name: str = "",
    report_type: str = "platform",
) -> matplotlib.figure.Figure:
    """
    Function to plot the results of the regression model

    The confidence interval is computed, based on:
    *
    Machine learning approaches for estimation of prediction interval for the model output
    Durga L. Shrestha *, Dimitri P. Solomatine
    *
    """

    # specify if the plot is used by the platform or it is downloaded by the user
    if report_type == "platform":
        cl = "white"
    elif report_type == "client":
        cl = "black"

    else:
        raise ValueError("Colour not selected!")

    # TODO: compute the variance based on the algo and only use Durga's method cited above,
    #  if std prediction not available
    # compute the variance
    sum_errs = np.sum((y - y_hat) ** 2)
    p = no_of_model_parameters  # degrees of freedom

    if len(y) == p:
        p = 1
    else:
        pass

    var_hat = abs(1 / (len(y) - p)) * sum_errs

    stdev = np.sqrt(var_hat)
    z_value = 2  # this is the 95% confidence interval
    # compute the confidence interval for each sample
    ci = np.ones(len(y_hat)) * z_value * stdev

    # pick the first 500 points to plot the results
    no_of_entries_to_plot = 500
    X_plot = X[:no_of_entries_to_plot]
    y_plot = y[:no_of_entries_to_plot]
    y_hat_plot = y_hat[:no_of_entries_to_plot]
    ci_plot = ci[:no_of_entries_to_plot]

    figure = plt.figure(figsize=(12, 7), edgecolor=cl)  # figure parameters
    plt.plot(
        X_plot.index.values,
        y_plot,
        "o",
        markersize=10,
        alpha=0.7,
        linewidth=3,
        c="red",
        label="True " + target_name,
    )
    plt.plot(
        X_plot.index.values,
        y_hat_plot,
        "o",
        markersize=10,
        linewidth=3,
        c="orange",
        label="Predicted " + target_name,
    )
    plt.errorbar(
        X_plot.index.values,
        y_hat_plot,
        yerr=ci_plot,
        marker="",
        ls="",
        capsize=3,
        c=cl,
        label="Confidence interval",
    )

    # title, x-axis, y-axis names
    plt.title("Predictions", fontsize=16, color=cl)
    plt.xlabel(index_name, fontsize=16, color=cl)
    plt.ylabel(target_name, fontsize=16, color=cl)
    # legend
    legend = plt.legend(facecolor="grey", fontsize=16, framealpha=0.7)
    legend.get_frame().set_edgecolor("grey")
    plt.setp(legend.get_texts(), color=cl)

    # parameters for the figure axis and ticks
    plt.tick_params(axis="both", which="major", labelsize=14, colors=cl)
    plt.xticks(fontsize=15, color=cl, rotation=45)
    plt.yticks(fontsize=15, color=cl)
    plt.tight_layout()
    plt.grid(color=cl, linestyle=":", alpha=0.6)

    # create the error plot
    histogram_error = plt.figure(figsize=(10, 10))

    # calculate prcent error
    percent_error = (y_hat - y) / y * 100
    # determine number of bins as 10% of data
    no_of_bins = int(len(y_hat) * 0.1)

    # check for inf/-inf and replace with nan and drop
    percent_error.replace([np.inf, -np.inf], np.nan, inplace=True)
    percent_error.dropna(axis=0, inplace=True)

    if percent_error.empty:
        raise ValueError("Error dataset is empty!")
    else:
        pass

    n, bins, patches = plt.hist(
        percent_error,
        bins=no_of_bins,
        edgecolor="orange",
        color="orange",
        label="Model % error",
    )
    plt.axvline(0, color="white", lw=3, linestyle="dashed", label="0% error")

    plt.title("Histogram of model percentage error", fontsize=16, color=cl)
    plt.xlabel("Model error [%]", fontsize=16, color=cl)
    plt.ylabel("No. of errors in each bin", fontsize=16, color=cl)
    plt.xticks(fontsize=15, color=cl, rotation=45)
    plt.yticks(fontsize=15, color=cl)
    plt.grid(color=cl, linestyle=":", alpha=0.6)
    legend = plt.legend(facecolor="grey", fontsize=16, framealpha=0.7)
    legend.get_frame().set_edgecolor("grey")
    plt.setp(legend.get_texts(), color=cl)

    return figure, histogram_error


def plot_prediction_classification_cv(
    y: np.ndarray,
    y_hat_proba: np.ndarray,
    decision_threshold: float,
    report_type: str = "platform",
) -> matplotlib.figure.Figure:
    """
    Function to plot the results of the classification model

    The confidence interval is computed, based on:
    *
    Machine learning approaches for estimation of prediction interval for the model output
    Durga L. Shrestha *, Dimitri P. Solomatine
    *
    """

    # specify if the plot is used by the platform or it is downloaded by the user
    if report_type == "platform":
        cl = "white"
    elif report_type == "client":
        cl = "black"

    else:
        raise ValueError("Colour not selected!")

    confusion_matrix_fig = _plot_confusion_matrix(
        y, y_hat_proba, decision_threshold, cl
    )  # create confusion figure

    # create roc_auc curve and precision vs recall curves
    _, fig_precision_recall_th = _plot_ROC_AUC_and_precision_recall_vs_threshold(
        y, y_hat_proba, cl, decision_threshold=decision_threshold
    )

    return confusion_matrix_fig, fig_precision_recall_th
