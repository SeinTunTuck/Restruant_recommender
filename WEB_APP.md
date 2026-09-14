# Savor recommender

An explainable restaurant recommender and decision-support dashboard built from the nine CSV files in this folder.

Visitors begin with a dinner brief—occasion, cuisines, dining area, budget, travel radius, and practical needs. Dataset user IDs remain internal; the interface matches the brief to anonymous historical taste patterns instead of asking the visitor to impersonate a recorded user.

## Run locally

```bash
npm install
npm run dev
```

Open the local address shown by Vite. To create a deployable static bundle, run `npm run build`; the output in `dist/` includes all nine datasets.

## Recommendation model

Each restaurant receives a weighted score from four normalized signals:

- **Quality:** Bayesian-adjusted overall, food, and service ratings. A seven-rating prior reduces small-sample bias.
- **Taste:** item predictions from positively correlated diners, falling back to venue quality when collaborative evidence is sparse.
- **Fit:** cuisine, budget, payment, ambience, and dress-preference compatibility.
- **Convenience:** distance from the visitor's selected dining area, calculated with the Haversine formula.

Travel radius, parking, accessibility, and alcohol switches are hard constraints. Users can tune the four decision weights, inspect evidence confidence, search and sort results, and compare up to three venues side by side.

All calculations run locally in the browser. No source data is uploaded or modified.
