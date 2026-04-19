# Sip&Site: Sydney Hospitality Location Intelligence

> Smart location scouting for F&B founders. Submit your business profile, get your top Sydney suburbs ranked by data, not guesswork.

---

## What It Does

Sip&Site helps hospitality entrepreneurs find the best location for their business in Sydney. Input your business type, budget, target customer, priorities, and the platform returns ranked suburb recommendations backed by real geospatial and demographic data.

Each recommendation includes:
- A composite location score
- Foot traffic, competitor count, and rent estimates
- Live maps of nearby restaurants and the recommended hex site
- Sydney demographic context (income and population density percentiles)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| Scoring Engine | H3 hexagonal grid (resolution 8), Pandas, NumPy |
| Geospatial Data | OpenStreetMap via Overpass API |
| Frontend | Vanilla HTML/CSS/JS, Tailwind CSS, Leaflet.js |
| Data | ABS Census, Sydney F&B rental data, OSM POIs |

---

## Project Structure

```
sydney-location-scout-datathon/
├── app.py                        # FastAPI backend
├── scorer.py                     # Location scoring engine
├── feature_df_v2.parquet         # Processed feature dataset
├── sydney_fnb_rental_sourced.csv # Rental data
├── front-end/
│   ├── design1fix.html           # Landing page
│   ├── design1fixquote.html      # Quote / input form
│   └── top6_dynamic.html         # Results page
```

---

## How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn pandas numpy h3 pyarrow
```

### 2. Start the backend

```bash
cd sydney-location-scout-datathon
uvicorn app:app --reload --port 8000
```

### 3. Open the frontend

Open `front-end/design1fix.html` in your browser (or serve via Live Server in VS Code).

Make sure the backend is running on `http://localhost:8000` before submitting the quote form.

---

## API

### `POST /api/quote`

Returns top N scored location recommendations.

**Request body:**
```json
{
  "business_type": "Cafe",
  "budget_min": 500,
  "budget_max": 1500,
  "foot_traffic_importance": 4,
  "competition_comfort": 2,
  "transit_importance": 3,
  "target_customers": ["students", "professionals"],
  "age_group": "25-34",
  "price_point": "mid",
  "top_n": 6
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "rank": 1,
      "sa2_name": "Paddington - Moore Park",
      "score": 0.81,
      "stats": { ... },
      "reasons": [ ... ],
      "hex_boundary": [ ... ]
    }
  ]
}
```

---

## Team

Built for the USYD Datathon.
