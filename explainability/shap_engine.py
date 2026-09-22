import shap
import shap
import numpy as np


# class ShapEngine:

#     def __init__(self, model):

#         self.explainer = shap.TreeExplainer(model)

#     def explain(self, df, top_n=5):

#         shap_values = self.explainer.shap_values(df)

#         explanations = []

#         for row_idx in range(len(df)):

#             features = []

#             for col_idx, feature in enumerate(df.columns):

#                 shap_val = float(
#                     shap_values[row_idx][col_idx]
#                 )

#                 features.append(
#                     {
#                         "feature": feature,
#                         "value": str(
#                             df.iloc[row_idx][feature]
#                         ),
#                         "shap_value": round(
#                             shap_val,
#                             4
#                         ),
#                         "impact": (
#                             "increase_risk"
#                             if shap_val > 0
#                             else "decrease_risk"
#                         )
#                     }
#                 )

#             features.sort(
#                 key=lambda x: abs(
#                     x["shap_value"]
#                 ),
#                 reverse=True
#             )

#             explanations.append(
#                 features[:top_n]
#             )

#         return explanations

class ShapEngine:

    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain_multiclass(
        self,
        df,
        predicted_classes,
        original_df=None,
        top_n=5
    ):

        shap_values = self.explainer.shap_values(df)
        

        explanations = []

        # New SHAP format
        if hasattr(shap_values, "shape"):

            for row_idx in range(len(df)):

                cls = int(predicted_classes[row_idx])

                row_features = []

                for col_idx, feature in enumerate(df.columns):

                    shap_val = float(
                        shap_values[row_idx][col_idx][cls]
                    )

                    row_features.append({
                        "feature": feature,
                        "value": str(df.iloc[row_idx][feature]),
                        "shap_value": round(shap_val, 4),
                        "impact":
                            "increase_reason_probability"
                            if shap_val > 0
                            else "decrease_reason_probability"
                    })

                row_features.sort(
                    key=lambda x: abs(x["shap_value"]),
                    reverse=True
                )

                explanations.append(
                    row_features[:top_n]
                )

        # Old SHAP format
        else:

            for row_idx in range(len(df)):

                cls = int(predicted_classes[row_idx])

                row_features = []

                for col_idx, feature in enumerate(df.columns):

                    shap_val = float(
                        shap_values[cls][row_idx][col_idx]
                    )

                    if original_df is not None:
                        display_value = original_df.iloc[row_idx][feature]
                    else:
                        display_value = df.iloc[row_idx][feature]

                    row_features.append({
                        "feature": feature,
                        "value": str(display_value),
                        "shap_value": round(shap_val, 4),
                        "impact":
                            "increase_reason_probability"
                            if shap_val > 0
                            else "decrease_reason_probability"
                    })

                row_features.sort(
                    key=lambda x: abs(x["shap_value"]),
                    reverse=True
                )

                explanations.append(
                    row_features[:top_n]
                )

        return explanations


    def interaction_values(self, df):

        interaction_matrix = (
            self.explainer
            .shap_interaction_values(df)
        )

        results = []

        for row_idx in range(len(df)):

            row_interactions = []

            feature_names = list(df.columns)

            n_features = len(feature_names)

            for i in range(n_features):

                for j in range(i + 1, n_features):

                    strength = abs(
                        float(
                            interaction_matrix
                            [row_idx][i][j]
                        )
                    )

                    row_interactions.append(
                        {
                            "feature_1":
                                feature_names[i],

                            "feature_2":
                                feature_names[j],

                            "interaction_strength":
                                round(
                                    strength,
                                    4
                                )
                        }
                    )

            row_interactions.sort(
                key=lambda x:
                x["interaction_strength"],
                reverse=True
            )

            results.append(
                row_interactions[:5]
            )

        return results