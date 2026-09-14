import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).parent / "dataset"

DINING_AREAS = {
    "San Luis Potosi": (22.15, -100.98),
    "Cuernavaca": (18.92, -99.23),
    "Ciudad Victoria": (23.75, -99.16),
}


@st.cache_data
def load_data():
    ratings = pd.read_csv(DATA_DIR / "rating_final.csv")
    places = pd.read_csv(DATA_DIR / "geoplaces2.csv", encoding="utf-8")
    cuisines = pd.read_csv(DATA_DIR / "chefmozcuisine.csv")
    payments = pd.read_csv(DATA_DIR / "chefmozaccepts.csv")
    parking = pd.read_csv(DATA_DIR / "chefmozparking.csv")
    profiles = pd.read_csv(DATA_DIR / "userprofile.csv")

    for frame in [places, cuisines, payments, parking, profiles]:
        frame.replace("?", np.nan, inplace=True)

    return ratings, places, cuisines, payments, parking, profiles


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).replace("_", " ").replace("�", "o").strip()


def composite_rating(row):
    return 0.5 * row["rating"] + 0.3 * row["food_rating"] + 0.2 * row["service_rating"]


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def prepare_places(ratings, places, cuisines, payments, parking):
    ratings = ratings.copy()
    ratings["composite"] = ratings.apply(composite_rating, axis=1)
    global_mean = ratings["composite"].mean()

    quality = ratings.groupby("placeID").agg(
        raw_quality=("composite", "mean"),
        review_count=("composite", "size"),
        food_avg=("food_rating", "mean"),
        service_avg=("service_rating", "mean"),
    )
    quality["quality"] = (
        quality["review_count"] * quality["raw_quality"] + 7 * global_mean
    ) / (quality["review_count"] + 7) / 2

    place_data = places.merge(quality, left_on="placeID", right_index=True, how="left")
    place_data[["raw_quality", "quality", "food_avg", "service_avg"]] = place_data[
        ["raw_quality", "quality", "food_avg", "service_avg"]
    ].fillna(global_mean / 2)
    place_data["review_count"] = place_data["review_count"].fillna(0).astype(int)

    cuisine_map = cuisines.groupby("placeID")["Rcuisine"].apply(list).to_dict()
    payment_map = payments.groupby("placeID")["Rpayment"].apply(list).to_dict()
    parking_map = parking.groupby("placeID")["parking_lot"].apply(list).to_dict()

    place_data["cuisines"] = place_data["placeID"].map(lambda x: cuisine_map.get(x, []))
    place_data["payments"] = place_data["placeID"].map(lambda x: payment_map.get(x, []))
    place_data["parking_options"] = place_data["placeID"].map(lambda x: parking_map.get(x, ["unknown"]))
    return place_data, ratings


def cosine_similarity(left, right):
    common = left.index.intersection(right.index)
    if common.empty:
        return 0
    dot = float((left.loc[common] * right.loc[common]).sum())
    norm = np.linalg.norm(left.values) * np.linalg.norm(right.values)
    return dot / norm if norm else 0


def item_based_scores(user_id, ratings, place_data):
    matrix = ratings.pivot_table(index="userID", columns="placeID", values="composite")
    own = matrix.loc[user_id].dropna() if user_id in matrix.index else pd.Series(dtype=float)
    liked = own[own >= 1]
    scores = {}

    for place_id in place_data["placeID"]:
        candidate = matrix[place_id].dropna() if place_id in matrix else pd.Series(dtype=float)
        weighted = total = evidence = 0
        for rated_place_id, rating in liked.items():
            if rated_place_id == place_id or rated_place_id not in matrix:
                continue
            sim = cosine_similarity(candidate, matrix[rated_place_id].dropna())
            if sim <= 0.05:
                continue
            weighted += sim * rating
            total += sim
            evidence += 1
        fallback = float(place_data.loc[place_data["placeID"] == place_id, "quality"].iloc[0])
        scores[place_id] = (min(max((weighted / total) / 2, 0), 1) if total else fallback, evidence)
    return scores


def preference_fit(place, selected_cuisines, price):
    signals = []
    if selected_cuisines:
        matches = set(place["cuisines"]).intersection(selected_cuisines)
        signals.append((1 if matches else 0.15, 4))
    if price != "Any":
        place_price = clean(place["price"]).lower()
        target = price.lower()
        rank = {"low": 1, "medium": 2, "high": 3}
        if place_price == target:
            signals.append((1, 3))
        elif rank.get(place_price, 2) < rank.get(target, 2):
            signals.append((0.7, 3))
        else:
            signals.append((0.18, 3))
    if not signals:
        return 0.65
    return sum(score * weight for score, weight in signals) / sum(weight for _, weight in signals)


def score_recommendations(method, user_id, area, radius, selected_cuisines, price, require_parking):
    ratings, places, cuisines, payments, parking, _ = load_data()
    place_data, ratings = prepare_places(ratings, places, cuisines, payments, parking)
    item_scores = item_based_scores(user_id, ratings, place_data)
    area_lat, area_lon = DINING_AREAS[area]
    rows = []

    for _, place in place_data.iterrows():
        distance = haversine(area_lat, area_lon, place["latitude"], place["longitude"])
        parking_available = not any(x in ["none", "unknown"] for x in place["parking_options"])
        if distance > radius or (require_parking and not parking_available):
            continue

        fit = preference_fit(place, selected_cuisines, price)
        convenience = max(0, min(1, 1 - distance / max(radius * 1.15, 2)))
        item_score, evidence = item_scores[place["placeID"]]
        quality = place["quality"]

        if method == "Item-Based Collaborative Filtering":
            final_score = 100 * (item_score * 0.72 + quality * 0.18 + fit * 0.10)
            reason = f"{evidence} similar restaurant signals" if evidence else "Quality fallback due to sparse item history"
        elif method == "Location-Aware Recommendation":
            final_score = 100 * (convenience * 0.58 + fit * 0.25 + quality * 0.17)
            reason = f"{distance:.1f} km from selected dining area"
        else:
            final_score = 100 * (quality * 0.34 + item_score * 0.28 + fit * 0.24 + convenience * 0.14)
            reason = "Balanced quality, taste, preference fit, and distance"

        rows.append(
            {
                "Restaurant": clean(place["name"]),
                "City": clean(place["city"]) or clean(place["state"]),
                "Score": round(final_score, 1),
                "Rating": round(place["raw_quality"], 2),
                "Reviews": int(place["review_count"]),
                "Distance km": round(distance, 2),
                "Price": clean(place["price"]).title(),
                "Cuisines": ", ".join(clean(c) for c in place["cuisines"][:3]) or "Not listed",
                "Evidence": reason,
            }
        )

    return pd.DataFrame(rows).sort_values("Score", ascending=False)


st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="wide")
st.title("Restaurant Recommendation System")
st.caption("Choose one of three recommender methods and compare how the restaurant ranking changes.")

ratings, places, cuisines, _, _, profiles = load_data()
all_cuisines = sorted(cuisines["Rcuisine"].dropna().unique())
user_ids = sorted(profiles["userID"].dropna().unique())

with st.sidebar:
    st.header("Dining brief")
    method = st.selectbox(
        "Recommendation method",
        [
            "Hybrid Recommendation",
            "Item-Based Collaborative Filtering",
            "Location-Aware Recommendation",
        ],
    )
    user_id = st.selectbox("Matched user profile", user_ids, index=0)
    area = st.selectbox("Dining area", list(DINING_AREAS.keys()))
    radius = st.slider("Travel radius", 1, 40, 8)
    selected_cuisines = set(st.multiselect("Cuisine mood", all_cuisines, default=["Mexican"] if "Mexican" in all_cuisines else []))
    price = st.selectbox("Budget", ["Any", "low", "medium", "high"], index=2)
    require_parking = st.checkbox("Require parking")

recommendations = score_recommendations(method, user_id, area, radius, selected_cuisines, price, require_parking)

col1, col2, col3 = st.columns(3)
col1.metric("Restaurants", len(places))
col2.metric("Ratings", len(ratings))
col3.metric("Eligible results", len(recommendations))

st.subheader(method)
if method == "Item-Based Collaborative Filtering":
    st.info("Uses restaurant-to-restaurant cosine similarity from shared user ratings.")
elif method == "Location-Aware Recommendation":
    st.info("Uses Haversine distance and ranks nearby restaurants higher while keeping preference filters.")
else:
    st.info("Combines Bayesian quality, item-based taste, preference fit, and convenience.")

if recommendations.empty:
    st.warning("No restaurants match these constraints. Increase radius or relax filters.")
else:
    st.dataframe(recommendations.head(20), use_container_width=True, hide_index=True)
    st.bar_chart(recommendations.head(10), x="Restaurant", y="Score")

with st.expander("Show calculation evidence"):
    st.markdown(
        """
        **Composite rating**

        `Composite = 0.50 * rating + 0.30 * food_rating + 0.20 * service_rating`

        **Item-Based CF**

        `Similarity(A, B) = (A dot B) / (||A|| * ||B||)`

        `Item score = weighted average of restaurants the user already rated, weighted by restaurant similarity`

        **Location-Aware**

        `Convenience = 1 - distance / max(radius * 1.15, 2)`

        `Location score = 58% convenience + 25% preference fit + 17% quality`

        **Hybrid**

        `Final score = 100 * (0.34 quality + 0.28 item taste + 0.24 preference fit + 0.14 convenience)`
        """
    )
