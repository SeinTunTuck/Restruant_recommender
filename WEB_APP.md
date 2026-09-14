# Savor recommender

An explainable restaurant recommender and decision-support dashboard built from the nine CSV files in this folder. The project now includes:

- A custom Vite web app (`index.html`, `app.js`, `styles.css`)
- A Streamlit sharing app (`streamlit_app.py`)
- A Canva presentation source file (`canva-restaurant-recommender.html`)

Visitors begin with a dinner brief: occasion, cuisines, dining area, budget, travel radius, and practical needs. Dataset user IDs remain internal in the Vite app; the interface matches the brief to anonymous historical taste patterns instead of asking the visitor to impersonate a recorded user.

## Run locally

### Vite web app

```bash
npm install
npm run dev
```

Open the local address shown by Vite. To create a deployable static bundle, run `npm run build`; the output in `dist/` includes all nine datasets.

### Streamlit app

```bash
pip3 install -r requirements.txt
streamlit run streamlit_app.py
```

If the `streamlit` command is not available:

```bash
python3 -m streamlit run streamlit_app.py
```

Open the local address shown by Streamlit, usually `http://localhost:8501`.

## Recommendation methods

The app supports three selectable methods.

### 1. Item-Based Collaborative Filtering

This method compares restaurants with other restaurants. Each restaurant is represented as a vector of user ratings, and cosine similarity measures how similar two restaurants are.

```text
Similarity(A, B) = (A dot B) / (||A|| * ||B||)
```

The predicted item score is a weighted average of restaurants the matched user already rated, weighted by restaurant-to-restaurant similarity.

### 2. Location-Aware Recommendation

This method uses the selected dining area and restaurant coordinates. Distance is calculated with the Haversine formula, then converted into a convenience score.

```text
Convenience = 1 - distance / max(radius * 1.15, 2)
Location score = 58% convenience + 25% preference fit + 17% quality
```

### 3. Hybrid Recommendation

The hybrid method combines rating evidence, collaborative taste, preference fit, and location convenience.

```text
Final score = 100 * (
  0.34 * quality +
  0.28 * item taste +
  0.24 * preference fit +
  0.14 * convenience
)
```

## Evidence and calculations

The base rating evidence is a composite rating:

```text
Composite rating = 0.50 * overall rating + 0.30 * food rating + 0.20 * service rating
```

Restaurant quality uses Bayesian averaging to reduce small-sample bias:

- A restaurant with very few perfect ratings should not automatically outrank a restaurant with many strong ratings.
- The global rating average is used as a prior.
- The app uses a seven-rating prior in the Vite implementation.

Travel radius, parking, accessibility, and alcohol switches are hard constraints. Users can tune the four decision weights, inspect evidence confidence, search and sort results, and compare up to three venues side by side.

## Sharing

For easiest sharing, deploy the Streamlit app:

1. Push the project to GitHub.
2. Go to https://share.streamlit.io/.
3. Create a new app from the GitHub repository.
4. Set the main file path to `streamlit_app.py`.
5. Deploy and share the generated public URL.

## Privacy

All Vite calculations run locally in the browser. The Streamlit version loads the included CSV files from the deployed repository. No source data is modified by either app.
