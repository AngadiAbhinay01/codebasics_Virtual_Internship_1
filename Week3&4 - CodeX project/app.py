from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CodeX Beverage: Price Prediction", layout="wide")

# ------------------------------------------------------------------
# 1. Load the saved model and encoding rules (done once, then cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load(Path(__file__).parent / "model_artifacts.pkl")

art = load_artifacts()
model = art["model"]
feature_columns = art["feature_columns"]
label_maps = art["label_maps"]
onehot_cols = art["onehot_cols"]
classes = art["target_classes"]
opts = art["options"]

# ------------------------------------------------------------------
# 2. Ordered dropdown lists (so they appear in a natural order)
# ------------------------------------------------------------------
INCOME_LEVELS = ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L"]
FREQUENCY = ["0-2 times", "3-4 times", "5-7 times"]
AWARENESS = ["0 to 1", "2 to 4", "above 4"]

def sort_by_ml(values):
    def key(v):
        m = re.search(r"(\d+)\s*ml", v)
        return int(m.group(1)) if m else 0
    return sorted(values, key=key)

def sort_by_level(values):
    order = {"low": 0, "medium": 1, "high": 2}
    return sorted(values, key=lambda v: order.get(v.split()[0].lower(), 99))

# ------------------------------------------------------------------
# 3. Same feature engineering as the notebook
# ------------------------------------------------------------------
CF_MAP = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
AB_MAP = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
ZONE_MAP = {"Urban": 3, "Metro": 4, "Rural": 1, "Semi-Urban": 2}
INCOME_MAP = {"<10L": 1, "10L - 15L": 2, "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5}

def get_age_group(age):
    if age <= 25: return "18-25"
    if age <= 35: return "26-35"
    if age <= 45: return "36-45"
    if age <= 55: return "46-55"
    if age <= 70: return "56-70"
    return "70+"

def build_features(r):
    """Turn one respondent (dict) into a one-row DataFrame the model understands."""
    row = {c: 0 for c in feature_columns}

    # engineered features
    row["cf_ab_score"] = round((CF_MAP[r["consume_frequency(weekly)"]]
                                + AB_MAP[r["awareness_of_other_brands"]]) / 2, 2)
    row["zas_score"] = ZONE_MAP[r["zone"]] * INCOME_MAP.get(r["income_levels"], 0)
    row["bsi"] = int(r["current_brand"] != "Established"
                     and r["reasons_for_choosing_brands"] in ("Price", "Quality"))

    # label-encoded columns
    row["age_group"] = label_maps["age_group"][get_age_group(r["age"])]
    for col in ["income_levels", "health_concerns",
                "consume_frequency(weekly)", "preferable_consumption_size"]:
        row[col] = label_maps[col][r[col]]

    # one-hot columns (a missing dummy = the dropped baseline category)
    for col in onehot_cols:
        dummy = f"{col}_{r[col]}"
        if dummy in row:
            row[dummy] = 1

    return pd.DataFrame([row])[feature_columns]

# ------------------------------------------------------------------
# 4. User interface
# ------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>CodeX Beverage: Price Prediction</h1>",
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
age = c1.number_input("Age", min_value=18, max_value=70, value=28, step=1)
gender = c2.selectbox("Gender", opts["gender"])
zone = c3.selectbox("Zone", opts["zone"])
occupation = c4.selectbox("Occupation", opts["occupation"])

c1, c2, c3, c4 = st.columns(4)
income = c1.selectbox("Income Level (In L)", INCOME_LEVELS)
frequency = c2.selectbox("Consume Frequency(weekly)", FREQUENCY)
brand = c3.selectbox("Current Brand", opts["current_brand"])
size = c4.selectbox("Preferable Consumption Size",
                    sort_by_ml(opts["preferable_consumption_size"]))

c1, c2, c3, c4 = st.columns(4)
awareness = c1.selectbox("Awareness of other brands", AWARENESS)
reason = c2.selectbox("Reasons for choosing brands", opts["reasons_for_choosing_brands"])
flavor = c3.selectbox("Flavor Preference", opts["flavor_preference"])
channel = c4.selectbox("Purchase Channel", opts["purchase_channel"])

c1, c2, c3, c4 = st.columns(4)
packaging = c1.selectbox("Packaging Preference", opts["packaging_preference"])
health = c2.selectbox("Health Concerns", sort_by_level(opts["health_concerns"]))
situation = c3.selectbox("Typical Consumption Situations",
                         opts["typical_consumption_situations"])

# ------------------------------------------------------------------
# 5. Prediction
# ------------------------------------------------------------------
if st.button("Calculate Price Range"):
    respondent = {
        "age": age,
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income_levels": income,
        "consume_frequency(weekly)": frequency,
        "current_brand": brand,
        "preferable_consumption_size": size,
        "awareness_of_other_brands": awareness,
        "reasons_for_choosing_brands": reason,
        "flavor_preference": flavor,
        "purchase_channel": channel,
        "packaging_preference": packaging,
        "health_concerns": health,
        "typical_consumption_situations": situation,
    }

    X_new = build_features(respondent)
    probs = model.predict_proba(X_new)[0]
    best = int(probs.argmax())

    st.subheader(f"Predicted Price Range: ₹{classes[best]}")
    st.caption(f"Model confidence for this range: {probs[best]:.0%}")

    prob_df = (
        pd.DataFrame({"Price range": classes, "Probability": probs})
        .sort_values("Price range", key=lambda s: s.str.split("-").str[0].astype(int))
        .set_index("Price range")
    )
    st.bar_chart(prob_df)