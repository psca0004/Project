
"""Metric calculations for evaluating apple-count predictions."""


def calculate_mae(actual_counts, predicted_counts):
    """Calculate mean absolute error between actual and predicted counts."""
    if len(actual_counts) != len(predicted_counts):
        raise ValueError("Actual and predicted counts must have equal lengths.")

    if len(actual_counts) == 0:
        raise ValueError("At least one count is required.")

    errors = [
        abs(actual - predicted)
        for actual, predicted in zip(actual_counts, predicted_counts)
    ]

    return sum(errors) / len(errors)


def calculate_relative_mae(actual_counts, predicted_counts):
    """Calculate MAE as a percentage of the mean actual count."""
    mae = calculate_mae(actual_counts, predicted_counts)
    mean_actual = sum(actual_counts) / len(actual_counts)

    if mean_actual == 0:
        raise ValueError("Mean actual count must not be zero.")

    return (mae / mean_actual) * 100