
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Cardiac Phenotype Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. LOAD TRAINED MODEL
# ==========================================================

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "cardiac_phenotype_model.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please check that "
        "cardiac_phenotype_model.joblib exists inside the models folder."
    )
    st.stop()

try:
    bundle = load_model()

    model = bundle["model"]
    model_name = bundle["model_name"]

    feature_columns = bundle["feature_columns"]
    numeric_columns = bundle["numeric_columns"]
    categorical_columns = bundle["categorical_columns"]

    defaults = bundle["defaults"]
    categories = bundle["categories"]
    phenotype_classes = bundle["phenotype_classes"]
    

except Exception as error:
    st.error(f"Unable to load model bundle: {error}")
    st.stop()


# ==========================================================
# 3. APPLICATION HEADER
# ==========================================================

st.title("❤️ Cardiac Phenotype Prediction")

st.subheader(
    "Machine Learning-Based Cardiac Adverse Event Prediction"
)

st.markdown(
    """
    This application predicts potential cardiac phenotype categories
    from the entered patient and adverse-event report parameters.
    """
)

with st.sidebar:
    st.header("About the Model")

    st.write(f"**Selected model:** {model_name}")

    st.write(
        f"**Input features:** {len(feature_columns)}"
    )

    st.write(
        f"**Phenotype classes:** {len(phenotype_classes)}"
    )

    st.divider()

    st.caption(
        "Research prototype. Model scores are not validated "
        "clinical risk estimates and must not be used for "
        "diagnosis or treatment decisions."
    )

st.info(
    "Enter the available information below. "
    "For missing information, select the corresponding NaN checkbox."
)

st.divider()


# ==========================================================
# 4. INPUT FORM
# ==========================================================

st.header("Patient / Report Information")

st.caption(
    "Fields are displayed in the same order as the model's "
    "training features. Use the arrows for numerical values "
    "or select an available categorical option."
)

input_values = {}

with st.form("cardiac_prediction_form"):

    # Three-column layout
    input_columns = st.columns(3)

    for i, col in enumerate(feature_columns):

        with input_columns[i % 3]:

            st.markdown(f"**{col}**")

            # --------------------------------------------------
            # Missing value checkbox
            # --------------------------------------------------

            is_missing = st.checkbox(
                "NaN / Missing",
                value=False,
                key=f"missing_{col}"
            )

            # --------------------------------------------------
            # NUMERICAL FEATURES
            # --------------------------------------------------

            if col in numeric_columns:

                default_value = defaults.get(col, 0.0)

                if pd.isna(default_value):
                    default_value = 0.0

                default_value = float(default_value)

                # Age gets a dedicated customer-friendly control
                if col.lower() == "age":

                    value = st.number_input(
                        "Age (years)",
                        min_value=0,
                        max_value=120,
                        value=int(min(120, max(0, default_value))),
                        step=1,
                        disabled=is_missing,
                        key=f"input_{col}"
                    )

                # Weight
                elif col.lower() in ["wt", "weight"]:

                    value = st.number_input(
                        "Weight",
                        min_value=0.0,
                        value=max(0.0, default_value),
                        step=0.1,
                        format="%.1f",
                        disabled=is_missing,
                        key=f"input_{col}"
                    )

                # Dose amount
                elif col.lower() in ["dose_amt", "dose_amount"]:

                    value = st.number_input(
                        "Dose Amount",
                        value=default_value,
                        step=0.1,
                        format="%.2f",
                        disabled=is_missing,
                        key=f"input_{col}"
                    )

                # Frequency
                elif col.lower() in ["dose_freq", "frequency"]:

                     value = st.number_input(
                       "Dose Frequency",
                       min_value=0,
                       max_value=100,
                       value=int(max(0, default_value)),
                       step=1,
                       disabled=is_missing,
                       key=f"input_{col}"
                     )

                # All other numeric columns
                else:

                    value = st.number_input(
                        f"Enter {col}",
                        value=default_value,
                        step=1.0,
                        disabled=is_missing,
                        key=f"input_{col}"
                    )

            # --------------------------------------------------
            # CATEGORICAL FEATURES
            # --------------------------------------------------

            elif col in categorical_columns:

                options = categories.get(col, [])

                default_value = defaults.get(col)

                if options:

                    # Ensure default is present in the dropdown
                    if default_value not in options:
                        default_index = 0
                    else:
                        default_index = options.index(default_value)

                    value = st.selectbox(
                        f"Select {col}",
                        options=options,
                        index=default_index,
                        disabled=is_missing,
                        key=f"input_{col}"
                    )

                else:

                    # Text input fallback if no options exist
                    value = st.text_input(
                        f"Enter {col}",
                        value=str(default_value or ""),
                        disabled=is_missing,
                        key=f"input_{col}"
                    )

            else:

                value = None

            # Store a genuine missing value when selected
            if is_missing:
                input_values[col] = np.nan
            else:
                input_values[col] = value

    st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Cardiac Phenotypes",
        type="primary",
        use_container_width=True
    )


# ==========================================================
# 5. PREDICTION
# ==========================================================

if predict_button:

    # Create one-row input dataframe
    input_df = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    st.divider()

    st.header("Submitted Input Data")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # Make prediction
    # ------------------------------------------------------

    try:

        scores = np.asarray(
            model.predict_proba(input_df)
        )

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        scores = scores[0]

        if len(scores) != len(phenotype_classes):
            st.error(
                "Prediction output does not match the number "
                "of phenotype classes in the saved model."
            )
            st.stop()

        if not np.all(np.isfinite(scores)):
            st.error(
                "The model returned invalid scores. "
                "Please check the entered input values."
            )
            st.stop()

        # --------------------------------------------------
        # Create results dataframe
        # --------------------------------------------------

        results_df = pd.DataFrame({
            "Cardiac Phenotype": phenotype_classes,
            "Model Score": scores
        })

        results_df = results_df.sort_values(
            by="Model Score",
            ascending=False
        ).reset_index(drop=True)

        st.success("Prediction completed successfully!")

        # --------------------------------------------------
        # TOP 3 PREDICTIONS
        # --------------------------------------------------

        st.header("Top Predicted Cardiac Phenotypes")

        for rank, row in results_df.head(3).iterrows():

            phenotype = row["Cardiac Phenotype"]
            score = float(row["Model Score"])

            st.markdown(
                f"### {rank + 1}. {phenotype}"
            )

            st.write(
                f"Model score: {score:.4f} "
                f"({score * 100:.2f}%)"
            )

            st.progress(
                float(np.clip(score, 0.0, 1.0))
            )

        # --------------------------------------------------
        # ALL PHENOTYPE RESULTS
        # --------------------------------------------------

        st.divider()

        st.header("All Phenotype Scores")

        display_df = results_df.copy()

        display_df["Model Score"] = (
            display_df["Model Score"].round(4)
        )

        display_df["Score (%)"] = (
            results_df["Model Score"] * 100
        ).round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # DOWNLOAD RESULTS
        # --------------------------------------------------

        csv_data = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="cardiac_phenotype_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ==========================================================
# 6. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Cardiac Phenotype Prediction | Machine Learning Research Project"
)

st.caption(
    "This application is intended for research and educational "
    "purposes only, not for clinical decision-making."
)
