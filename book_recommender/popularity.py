import pandas as pd


def bayesian_weighted_rating(
    average_rating: pd.Series,
    rating_count: pd.Series,
    global_mean: float,
    minimum_votes: float,
) -> pd.Series:
    votes = rating_count.astype(float)
    ratings = average_rating.astype(float)

    return (
        (votes / (votes + minimum_votes)) * ratings
        + (minimum_votes / (votes + minimum_votes)) * global_mean
    )