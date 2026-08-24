"""
recommendation_engine.py
"""


def generate_recommendation(
    denial_probability,
    top_driver=None
):

    if denial_probability >= 0.80:
        return (
            "High denial risk. Review diagnosis, CPT coding "
            "and medical necessity documentation."
        )

    if denial_probability >= 0.60:
        return (
            "Moderate denial risk. Verify coding accuracy "
            "before submission."
        )

    if denial_probability >= 0.40:
        return (
            "Low-moderate denial risk. Additional review recommended."
        )

    return (
        "Low denial risk. Ready for submission."
    )