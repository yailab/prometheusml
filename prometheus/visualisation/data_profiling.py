import pandas as pd
from pandas_profiling import ProfileReport


def profile_dataset(data: pd.DataFrame) -> str:
    """
    :param data: dataset in pandas dataframe
    :return:
    """

    # subsample for memory constraints
    no_samples = min(data.shape[0], 100000)
    data = data.sample(no_samples)

    profile = ProfileReport(
        data,
        minimal=True,
        progress_bar=False,
        samples=None,
        duplicates=None,
        title="",
        explorative=True,
        html={
            "style": {
                "full_width": True,
                "theme": "united",  # available theme options: ‘bootswatch’ ,'flatly','united'
                "primary_color": "orange",
            }
        },
        plot={
            "correlation": {"cmap": "Wistia", "bad": "#000000"},
            "missing": {"cmap": "Wistia"},
        },
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": False},
            "kendall": {"calculate": False},
            "phi_k": {"calculate": False},
            "cramers": {"calculate": False},
        },
        missing_diagrams={
            "heatmap": True,
            "dendrogram": False,
            "matrix": False,
            "count": False,
        },
    )

    return profile.to_html()
