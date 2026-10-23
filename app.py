
import json
from pathlib import Path

import requests
import streamlit as st


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "models" / "model_features.json"

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Pune House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD FEATURE CONFIGURATION
# --------------------------------------------------

@st.cache_data
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Feature configuration not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    config = load_config()
except (FileNotFoundError, json.JSONDecodeError) as error:
    st.error(str(error))
    st.stop()


FEATURES = config["features"]
NUMERIC_COLUMNS = config.get("numeric_columns", [])
CATEGORICAL_COLUMNS = config.get("categorical_columns", [])
AMENITY_COLUMNS = config.get("amenity_columns", [])

# Show only amenities that are actual model features.
AMENITY_COLUMNS = [
    col for col in AMENITY_COLUMNS if col in FEATURES
]


# --------------------------------------------------
# 3. PAGE HEADER
# --------------------------------------------------

st.title("🏠 Pune House Price Prediction")
st.write(
    "Enter property details to estimate the price "
    "of a residential property in Pune."
)

st.caption(
    "This tool provides a machine-learning estimate, "
    "not an official property valuation."
)

st.divider()


# --------------------------------------------------
# 4. PROPERTY DETAILS FORM
# --------------------------------------------------

with st.form("property_form"):

    st.subheader("📍 Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq. ft.)",
            min_value=100,
            max_value=50000,
            value=1000,
            step=50
        )

        bedrooms = st.selectbox(
            "Bedrooms (BHK)",
            options=[1, 2, 3, 4, 5, 6, 7, 8]
        )

        bathroom = st.selectbox(
            "Bathrooms",
            options=[1, 2, 3, 4, 5, 6]
        )

        balconies = st.selectbox(
            "Balconies",
            options=[0, 1, 2, 3, 4]
        )

        locality = st.text_input(
            "Locality",
            value="Wakad",
            help="Enter a locality name used in the dataset."
        )

    with col2:
        floor = st.selectbox(
            "Floor",
            options=["Not specified"] + [str(i) for i in range(0, 51)]
        )

        age = st.selectbox(
            "Property Age",
            options=[
                "Not specified",
                "New",
                "Old"
            ]
        )

        facing = st.selectbox(
            "Facing",
            options=[
                "Not specified",
                "East",
                "West",
                "North",
                "South",
                "North-East",
                "North-West",
                "South-East",
                "South-West"
            ]
        )

        opensides = st.selectbox(
            "Open Sides",
            options=["Not specified", 1, 2, 3, 4]
        )

        neworold = st.selectbox(
            "Property Category",
            options=[
                "Not specified",
                "New",
                "Resale",
                "Old"
            ]
        )

    st.divider()
    st.subheader("🏡 Property Type")

    property_type = st.selectbox(
        "Select Property Type",
        options=[
            "Apartment",
            "Residential",
            "Villa",
            "Independent",
            "Penthouse"
        ]
    )

    st.divider()
    st.subheader("✨ Additional Rooms")

    addi_rooms = st.selectbox(
        "Additional Rooms",
        options=[
            "Not specified",
            "Study Room",
            "Pooja Room",
            "Servant Room",
            "Store Room"
        ]
    )

    st.divider()
    st.subheader("🛠️ Amenities")

    st.write(
        "Select all amenities available in the property. "
        "You can leave this empty if none are specified."
    )

    selected_amenities = st.multiselect(
        "Available Amenities",
        options=AMENITY_COLUMNS
    )

    submitted = st.form_submit_button(
        "🔍 Predict House Price",
        use_container_width=True
    )


# --------------------------------------------------
# 5. BUILD MODEL INPUT
# --------------------------------------------------

if submitted:

    if not locality.strip():
        st.error("Please enter the property locality.")
        st.stop()

    property_data = {
        "Area": area,
        "Bedrooms": bedrooms,
        "bathroom": bathroom,
        "balconies": balconies,
        "floor": floor,
        "opensides": opensides,
        "Age": age,
        "addi_rooms": addi_rooms,
        "facing": facing,
        "locality": locality.strip(),
        "neworold": neworold,
        "Residential": int(property_type == "Residential"),
        "Villa": int(property_type == "Villa"),
        "Independent": int(property_type == "Independent"),
        "Penthouse": int(property_type == "Penthouse"),
    }

    # Include every amenity as an individual feature.
    for amenity in AMENITY_COLUMNS:
        property_data[amenity] = (
            "Yes" if amenity in selected_amenities else "No"
        )

    # Send only features expected by the trained model.
    # Unknown features are not sent to the API.
    property_data = {
        feature: property_data.get(feature, "Not specified")
        for feature in FEATURES
    }

    # Use the same convention as the model training script:
    # optional numeric values should be missing, not text.
    for col in NUMERIC_COLUMNS:
        if property_data.get(col) == "Not specified":
            property_data[col] = None

    for col in CATEGORICAL_COLUMNS:
        if property_data.get(col) == "Not specified":
            property_data[col] = None

    # --------------------------------------------------
    # 6. CALL FASTAPI
    # --------------------------------------------------

    with st.spinner("Predicting property price..."):

        try:
            response = requests.post(
                API_URL,
                json=property_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()["prediction"]

                predicted_price = result["predicted_price"]
                price_lakhs = result["predicted_price_in_lakhs"]
                price_crores = result["predicted_price_in_crores"]

                st.divider()
                st.subheader("📊 Estimated Property Price")

                price_col1, price_col2, price_col3 = st.columns(3)

                price_col1.metric(
                    "Estimated Price",
                    f"₹{predicted_price:,.0f}"
                )

                price_col2.metric(
                    "Price in Lakhs",
                    f"₹{price_lakhs:.2f} Lakh"
                )

                price_col3.metric(
                    "Price in Crores",
                    f"₹{price_crores:.3f} Crore"
                )

                st.success("Prediction completed successfully!")

                with st.expander("View submitted property details"):
                    st.json(property_data)

            else:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Please start the backend in another terminal."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. "
                "Please try again."
            )

        except (KeyError, ValueError) as error:
            st.error(f"Invalid API response: {error}")
