import shap


class ShapEngine:

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, df):

        shap_values = self.explainer.shap_values(df)

        explanations = []

        for i in range(len(df)):

            features = []

            for feature, value in zip(
                df.columns,
                shap_values[i]
            ):

                features.append(
                    {
                        "feature": feature,
                        "shap_value": round(float(value), 4),
                        "impact": (
                            "increase_risk"
                            if value > 0
                            else "decrease_risk"
                        )
                    }
                )

            features.sort(
                key=lambda x: abs(x["shap_value"]),
                reverse=True
            )

            explanations.append(features[:5])

        return explanations