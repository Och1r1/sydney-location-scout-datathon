import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from math import radians, sin, cos, asin, sqrt
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
ABS_DIR  = BASE_DIR / "abs_data/2021_GCP_SA2_for_NSW_short-header/2021 Census GCP Statistical Area 2 for NSW"
SHP_PATH = BASE_DIR / "abs_data/SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp"
FEATURE_PARQUET = BASE_DIR / "feature_df_v2.parquet"
SCORED_PARQUET  = BASE_DIR / "scored_v2.parquet"
RENTAL_CSV      = BASE_DIR / "sydney_fnb_rental_sourced.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H3_RES = 8

NON_COMMERCIAL_IDS = {
    '4bf58dd8d48988d1f0931735','62d587aeda6648532de2b88c','4bf58dd8d48988d12b951735',
    '52f2ab2ebcbc57f1066b8b3b','50aa9e094b90af0d42d5de0d','5267e4d9e4b0ec79466e48c6',
    '5267e4d9e4b0ec79466e48c9','530e33ccbcbc57f1066bbff7','5345731ebcbc57f1066c39b2',
    '63be6904847c3692a84b9bb7','4d4b7105d754a06373d81259','5267e4d9e4b0ec79466e48c7',
    '4bf58dd8d48988d132951735','52f2ab2ebcbc57f1066b8b4c','50aaa4314b90af0d42d5de10',
    '58daa1558bbb0b01f18ec1fa','63be6904847c3692a84b9bb8','4f2a23984b9023bd5841ed2c',
    '5267e4d9e4b0ec79466e48d1','4f2a25ac4b909258e854f55f','5267e4d9e4b0ec79466e48c8',
    '52741d85e4b0d5d1e3c6a6d9','4bf58dd8d48988d1f7931735','4f4531504b9074f6e4fb0102',
    '4cae28ecbf23941eb1190695','4bf58dd8d48988d1f9931735','5bae9231bedf3950379f89c5',
    '530e33ccbcbc57f1066bbff8','530e33ccbcbc57f1066bbfe4','52f2ab2ebcbc57f1066b8b54',
    '5267e4d8e4b0ec79466e48c5','53e0feef498e5aac066fd8a9','4bf58dd8d48988d130951735',
    '530e33ccbcbc57f1066bbff3','5bae9231bedf3950379f89c3','4bf58dd8d48988d12a951735',
    '52e81612bcbc57f1066b7a24','530e33ccbcbc57f1066bbff9',
}

COMPLEMENTARY_MAP = {
    'Cafe, Coffee, and Tea House': [
        'Office', 'Education', 'Fashion Retail', 'Health and Beauty Service',
    ],
    'Restaurant': [
        'Lodging', 'Bar', 'Performing Arts Venue', 'Fashion Retail', 'Movie Theater',
    ],
    'Bar': [
        'Restaurant', 'Lodging', 'Night Club', 'Office', 'Performing Arts Venue',
    ],
}

ANCHOR_RULES = {
    'rail':       ('Transport Hub', {'Rail Station', 'Metro Station', 'Light Rail Station', 'Tram Station'}),
    'bus':        ('Transport Hub', {'Bus Station'}),
    'university': ('Education',     {'College and University'}),
    'school':     ('Education',     {'Primary and Secondary School', 'Preschool', 'Private School'}),
    'office':     ('Office',        None),
    'shopping':   ('Shopping Mall', None),
    'hospital':   ('Hospital',      None),
    'hotel':      ('Lodging',       {'Hotel', 'Resort'}),
    'park':       ('Park',          None),
}

RETAIL_LEVEL2 = {
    'Fashion Retail', 'Food and Beverage Retail', 'Furniture and Home Store',
    'Computers and Electronics Retail', 'Cosmetics Store', 'Sporting Goods Retail',
    'Automotive Retail', 'Miscellaneous Store', 'Hardware Store', 'Pharmacy',
    'Convenience Store', 'Department Store', 'Shopping Mall', 'Gift Store',
    'Bookstore', 'Pet Supplies Store', 'Flower Store', 'Boutique',
    'Newsagent', 'Eyecare Store', 'Toy Store', 'Discount Store',
}

FOOD_BEV_LEVEL2 = {
    'Restaurant', 'Bar', 'Dessert Shop', 'Cafe, Coffee, and Tea House',
}

# ---------------------------------------------------------------------------
# Scoring weights per business type
# Keys must match the form's business_type dropdown options.
# All weights in each profile must sum to 1.0.
# ---------------------------------------------------------------------------
WEIGHT_PROFILES = {
    "Specialty Cafe": {
        'low_saturation_score':       0.18,
        'complementary_score':        0.15,
        'income_score':               0.12,
        'low_competition_score':      0.10,
        'office_proximity_score':     0.08,
        'pop_density_score':          0.08,
        'rail_proximity_score':       0.07,
        'foot_traffic_score':         0.07,
        'diversity_score':            0.05,
        'university_proximity_score': 0.05,
        'age_fit_score':              0.05,
    },
    "Fine Dining Restaurant": {
        'income_score':               0.22,  # affluent clientele critical
        'low_saturation_score':       0.15,
        'complementary_score':        0.12,
        'hotel_proximity_score':      0.10,  # tourists + business travellers
        'foot_traffic_score':         0.08,
        'low_competition_score':      0.08,
        'age_fit_score':              0.07,  # skews older for fine dining
        'rail_proximity_score':       0.06,
        'pop_density_score':          0.06,
        'diversity_score':            0.03,
        'university_proximity_score': 0.03,
    },
    "Casual Dining": {
        'foot_traffic_score':         0.18,
        'low_saturation_score':       0.15,
        'pop_density_score':          0.13,
        'complementary_score':        0.12,
        'low_competition_score':      0.10,
        'rail_proximity_score':       0.08,
        'income_score':               0.08,
        'diversity_score':            0.06,
        'age_fit_score':              0.05,
        'university_proximity_score': 0.03,
        'office_proximity_score':     0.02,
    },
    "Cocktail Bar / Lounge": {
        'low_saturation_score':       0.16,
        'income_score':               0.16,
        'complementary_score':        0.14,  # restaurants, hotels, venues nearby
        'foot_traffic_score':         0.10,
        'hotel_proximity_score':      0.10,
        'low_competition_score':      0.09,
        'rail_proximity_score':       0.08,
        'diversity_score':            0.07,
        'age_fit_score':              0.05,
        'pop_density_score':          0.03,
        'university_proximity_score': 0.02,
    },
    "Bistro": {
        'low_saturation_score':       0.17,
        'complementary_score':        0.14,
        'foot_traffic_score':         0.13,
        'income_score':               0.12,
        'pop_density_score':          0.10,
        'low_competition_score':      0.09,
        'rail_proximity_score':       0.08,
        'office_proximity_score':     0.07,
        'diversity_score':            0.05,
        'age_fit_score':              0.03,
        'university_proximity_score': 0.02,
    },
}

# ---------------------------------------------------------------------------
# Sydney area definitions
# Each area maps to a list of suburb name fragments. We match by substring
# against SA2 names, so e.g. "Newtown" matches "Newtown - Camperdown - Darlington".
# These lists are based on common usage and the LGA boundaries that locals recognise.
# ---------------------------------------------------------------------------
SYDNEY_DISTRICTS = {
    "Inner West": [
        "Newtown", "Camperdown", "Darlington",
        "Marrickville", "Sydenham", "Petersham",
        "Leichhardt", "Annandale", "Lilyfield",
        "Balmain", "Rozelle", "Birchgrove",
        "Glebe", "Forest Lodge", "Ultimo",
        "Stanmore", "Enmore",
        "Ashfield", "Summer Hill", "Haberfield",
        "Dulwich Hill", "Lewisham", "Hurlstone Park",
        "Erskineville", "Alexandria", "St Peters", "Tempe",
    ],
    "Eastern Suburbs": [
        "Bondi", "Bondi Junction", "Bronte", "Tamarama",
        "Coogee", "Clovelly", "Randwick", "Kingsford", "Kensington",
        "Maroubra", "Pagewood", "Eastgardens", "Daceyville",
        "Waverley", "Rose Bay", "Vaucluse", "Watsons Bay", "Dover Heights",
        "Double Bay", "Darling Point", "Edgecliff", "Point Piper",
        "Paddington", "Woollahra", "Centennial Park", "Queens Park",
        "Bellevue Hill",
    ],
    "North Shore": [
        "North Sydney", "Lavender Bay", "Kirribilli", "McMahons Point",
        "Crows Nest", "Waverton", "Wollstonecraft", "St Leonards",
        "Cammeray", "Naremburn", "Northbridge",
        "Mosman", "Cremorne", "Neutral Bay", "Cammeray",
        "Chatswood", "Artarmon", "Willoughby", "Roseville", "Lindfield",
        "Killara", "Gordon", "Pymble", "Turramurra",
        "Wahroonga", "Waitara", "Hornsby",
        "Lane Cove", "Greenwich", "Longueville",
        "Castle Cove", "Castlecrag",
    ],
    "CBD/Inner City": [
        "Sydney (North)", "Sydney (South)", "Sydney - Haymarket",
        "Millers Point", "Dawes Point", "The Rocks", "Walsh Bay",
        "Pyrmont", "Ultimo",
        "Surry Hills", "Darlinghurst", "Potts Point", "Elizabeth Bay",
        "Rushcutters Bay", "Woolloomooloo",
        "Redfern", "Waterloo", "Zetland",
        "Chippendale",
    ],
    "Northern Beaches": [
        "Manly", "Fairlight", "Balgowlah", "Seaforth", "Clontarf",
        "Dee Why", "Brookvale", "Curl Curl", "Freshwater", "Harbord",
        "Collaroy", "Narrabeen", "Cromer",
        "Mona Vale", "Warriewood", "Bayview", "Newport",
        "Avalon", "Bilgola", "Whale Beach", "Palm Beach",
        "Frenchs Forest", "Belrose", "Forestville", "Davidson",
        "Terrey Hills", "Duffys Forest",
    ],
    "Western Sydney": [
        "Parramatta", "Harris Park", "Granville", "Merrylands", "Holroyd",
        "Westmead", "Wentworthville", "Pendle Hill", "Toongabbie",
        "Auburn", "Lidcombe", "Berala", "Regents Park",
        "Blacktown", "Quakers Hill", "Kings Langley", "Doonside", "Marayong",
        "Mount Druitt", "St Marys", "Werrington",
        "Penrith", "Cambridge Park", "Kingswood", "Emu Plains",
        "Liverpool", "Cabramatta", "Fairfield", "Smithfield", "Wetherill Park",
        "Bankstown", "Chester Hill", "Yagoona", "Punchbowl",
        "Campbelltown", "Macquarie Fields", "Minto", "Leumeah", "Glenfield",
        "Ingleburn",
    ],
}

# Map form business_type values to Foursquare Level 2 category for competitor counting
BUSINESS_TYPE_TO_FSQ = {
    "Specialty Cafe":          "Cafe, Coffee, and Tea House",
    "Fine Dining Restaurant":  "Restaurant",
    "Casual Dining":           "Restaurant",
    "Cocktail Bar / Lounge":   "Bar",
    "Bistro":                  "Restaurant",
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))


def is_commercial(cat_ids):
    if not isinstance(cat_ids, (list, tuple, np.ndarray)) or len(cat_ids) == 0:
        return False
    return not any(cid in NON_COMMERCIAL_IDS for cid in cat_ids)


def first_labels(labels):
    if not isinstance(labels, (list, tuple, np.ndarray)) or len(labels) == 0:
        return (None, None)
    parts = labels[0].split(' > ')
    return (parts[0] if len(parts) >= 1 else None,
            parts[1] if len(parts) >= 2 else None)


def all_level2s(labels):
    if not isinstance(labels, (list, tuple, np.ndarray)):
        return []
    out = []
    for lbl in labels:
        parts = lbl.split(' > ')
        if len(parts) >= 2:
            out.append(parts[1])
    return out


def matches_anchor(row, level2_name, level3_set):
    if row['level2'] != level2_name:
        return False
    if level3_set is None:
        return True
    labels = row['fsq_category_labels']
    if not isinstance(labels, (list, tuple, np.ndarray)):
        return False
    for lbl in labels:
        parts = lbl.split(' > ')
        if len(parts) >= 3 and parts[2] in level3_set:
            return True
    return False


def _minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


# ---------------------------------------------------------------------------
# Feature building (used when recomputing from raw Foursquare data)
# ---------------------------------------------------------------------------

def build_features(target_hex, target_category_level2, hex_to_pois, df, anchor_coords, k=1):
    ring = h3.grid_disk(target_hex, k)

    pois_in_ring = []
    for h in ring:
        if h in hex_to_pois.groups:
            pois_in_ring.append(hex_to_pois.get_group(h))
    if not pois_in_ring:
        return None
    local = pd.concat(pois_in_ring)

    total_density = len(local)
    competitors   = (local['level2'] == target_category_level2).sum()

    level2_counts = local['level2'].value_counts()
    p = level2_counts / level2_counts.sum()
    diversity = -(p * np.log(p + 1e-12)).sum()

    complementary = sum(
        (local['level2'] == c).sum()
        for c in COMPLEMENTARY_MAP.get(target_category_level2, [])
    )

    centre_lat, centre_lon = h3.cell_to_latlng(target_hex)

    comp_mask = df['level2'] == target_category_level2
    if comp_mask.any():
        comp_pts = df.loc[comp_mask, ['latitude', 'longitude']].values
        dists = np.array([haversine_km(centre_lat, centre_lon, la, lo)
                          for la, lo in comp_pts])
        nearest_competitor_km = dists.min()
    else:
        nearest_competitor_km = np.nan

    anchor_dists = {}
    for anchor_name, pts in anchor_coords.items():
        if len(pts) == 0:
            anchor_dists[f'dist_{anchor_name}_km'] = np.nan
            continue
        dists = np.array([haversine_km(centre_lat, centre_lon, la, lo)
                          for la, lo in pts])
        anchor_dists[f'dist_{anchor_name}_km'] = dists.min()

    return {
        'h3': target_hex,
        'centre_lat': centre_lat,
        'centre_lon': centre_lon,
        'total_density': total_density,
        'competitors': competitors,
        'complementary': complementary,
        'diversity_entropy': diversity,
        'unique_categories': level2_counts.size,
        'nearest_competitor_km': nearest_competitor_km,
        **anchor_dists,
    }


def commercial_counts(hex_id, hex_to_pois, k=1):
    retail_hex = food_hex = retail_ring = food_ring = 0

    if hex_id in hex_to_pois.groups:
        sub = hex_to_pois.get_group(hex_id)
        retail_hex = sub['level2'].isin(RETAIL_LEVEL2).sum()
        food_hex   = sub['level2'].isin(FOOD_BEV_LEVEL2).sum()

    for h in h3.grid_disk(hex_id, k):
        if h in hex_to_pois.groups:
            sub = hex_to_pois.get_group(h)
            retail_ring += sub['level2'].isin(RETAIL_LEVEL2).sum()
            food_ring   += sub['level2'].isin(FOOD_BEV_LEVEL2).sum()

    return retail_hex, food_hex, retail_ring, food_ring


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(df):
    out = df.copy()

    for col in ['total_density', 'complementary', 'competitors', 'unique_categories']:
        out[f'{col}_log'] = np.log1p(out[col])

    out['foot_traffic_score']  = _minmax(out['total_density_log'])
    out['complementary_score'] = _minmax(out['complementary_log'])
    out['diversity_score']     = _minmax(out['diversity_entropy'])
    out['variety_score']       = _minmax(out['unique_categories_log'])
    out['low_competition_score'] = 1 - _minmax(out['competitors_log'])

    saturation = out['competitors'] / (out['complementary'] + 5)
    out['low_saturation_score'] = 1 - _minmax(saturation)

    for anchor in ['rail', 'bus', 'university', 'school',
                   'office', 'shopping', 'hospital', 'hotel', 'park']:
        col = f'dist_{anchor}_km'
        if col in out.columns:
            out[f'{anchor}_proximity_score'] = 1 - _minmax(out[col])

    out['income_score'] = _minmax(
        np.log1p(df['median_hhd_income_weekly'].fillna(df['median_hhd_income_weekly'].median()))
    )
    out['pop_density_score'] = _minmax(
        np.log1p(df['pop_density'].fillna(df['pop_density'].median()))
    )

    age_fit = 1 - (df['median_age'].fillna(35) - 35).abs() / 40
    out['age_fit_score'] = age_fit.clip(0, 1)

    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def apply_weights(scored_df, weights):
    score = sum(scored_df[col] * w for col, w in weights.items() if col in scored_df.columns)
    return score


def apply_commercial_filter(feature_df, scored_df):
    in_hex  = (feature_df['retail_hex'] >= 2) | (feature_df['food_hex'] >= 2)
    in_ring = (feature_df['retail_count'] >= 15) | (feature_df['food_count'] >= 10)
    viable  = in_hex & in_ring
    result  = scored_df.copy()
    result.loc[~viable.values, 'score'] = 0
    return result


# ---------------------------------------------------------------------------
# Geographic diversity selection
# ---------------------------------------------------------------------------

def diverse_top_n(scored_df, n=10, min_distance_km=2.0):
    ranked = scored_df.sort_values('score', ascending=False).reset_index(drop=True)
    picks  = []
    for _, row in ranked.iterrows():
        if len(picks) >= n:
            break
        if all(haversine_km(row['centre_lat'], row['centre_lon'],
                            p['centre_lat'], p['centre_lon']) >= min_distance_km
               for p in picks):
            picks.append(row)
    return pd.DataFrame(picks)


# ---------------------------------------------------------------------------
# Human-readable explanation
# ---------------------------------------------------------------------------

def explain_hex(feature_row, all_features):
    reasons = []

    def percentile(series, value):
        return (series < value).mean()

    pct_density       = percentile(all_features['total_density'],   feature_row['total_density'])
    pct_complementary = percentile(all_features['complementary'],   feature_row['complementary'])
    pct_competitors   = percentile(all_features['competitors'],     feature_row['competitors'])
    saturation_val    = feature_row['competitors'] / (feature_row['complementary'] + 5)
    pct_saturation    = percentile(
        all_features['competitors'] / (all_features['complementary'] + 5), saturation_val
    )

    if pct_density > 0.80:
        reasons.append(
            f"High foot traffic - {int(feature_row['total_density'])} POIs within 1km "
            f"(top {int((1-pct_density)*100)}% in Sydney)"
        )

    if pct_complementary > 0.75:
        reasons.append(
            f"Strong complementary neighbours — "
            f"{int(feature_row['complementary'])} offices/schools/shops nearby"
        )

    if feature_row['competitors'] == 0:
        reasons.append("No direct competitors within 1km")
    elif pct_competitors < 0.50:
        reasons.append(
            f"Manageable competition — {int(feature_row['competitors'])} competitors "
            f"(below Sydney median)"
        )

    if pct_saturation < 0.30:
        reasons.append(
            f"Underserved for this category - low saturation "
            f"(bottom {int(pct_saturation*100)}% in Sydney)"
        )

    dist_rail = feature_row.get('dist_rail_km')
    if pd.notna(dist_rail):
        if dist_rail < 0.4:
            reasons.append(f"{int(dist_rail*1000)}m from nearest rail station")
        elif dist_rail < 1.0:
            reasons.append(f"{dist_rail:.1f}km from nearest rail station")

    dist_office = feature_row.get('dist_office_km')
    if pd.notna(dist_office) and dist_office < 0.3:
        reasons.append(f"Surrounded by offices ({int(dist_office*1000)}m to nearest)")

    dist_uni = feature_row.get('dist_university_km')
    if pd.notna(dist_uni) and dist_uni < 0.5:
        reasons.append(f"Near university campus ({dist_uni:.1f}km)")

    dist_hotel = feature_row.get('dist_hotel_km')
    if pd.notna(dist_hotel) and dist_hotel < 0.3:
        reasons.append(f"Near hotels ({int(dist_hotel*1000)}m) — tourism / business travellers")

    retail = feature_row.get('retail_count', 0)
    if retail >= 15:
        reasons.append(f"Established retail precinct — {int(retail)} shops nearby")
    elif retail >= 5:
        reasons.append(f"Active commercial strip — {int(retail)} shops nearby")

    income = feature_row.get('median_hhd_income_weekly')
    if pd.notna(income):
        pct_income = percentile(all_features['median_hhd_income_weekly'], income)
        if pct_income > 0.75:
            reasons.append(
                f"Affluent neighbourhood — ${int(income):,}/week median income "
                f"(top {int((1-pct_income)*100)}% in Sydney)"
            )
        elif pct_income < 0.25:
            reasons.append(f"Budget-conscious area — ${int(income):,}/week median income")

    age = feature_row.get('median_age')
    if pd.notna(age) and 28 <= age <= 42:
        reasons.append(f"Target demographic — median age {int(age)}")

    density = feature_row.get('pop_density')
    if pd.notna(density) and percentile(all_features['pop_density'], density) > 0.80:
        reasons.append(f"Dense residential catchment — {int(density):,} people per km²")

    return reasons


# ---------------------------------------------------------------------------
# Budget → rent conversion
# Frontend sends two number fields (AUD total investment).
# Heuristic: investment / (60 sqm * 10 years) = affordable annual rent/sqm.
# ---------------------------------------------------------------------------

def investment_to_rent(investment_aud: float) -> float:
    return round(investment_aud / (60 * 10), 2)


def apply_rent_filter(
    feature_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    budget_min: float = None,
    budget_max: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero out scores for hexes outside the user's affordable rent band."""
    if not RENTAL_CSV.exists():
        return feature_df, scored_df

    rental = pd.read_csv(RENTAL_CSV)
    rent_lookup = rental.groupby("suburb")["avg_rent_sqm_per_year_aud"].max()

    def suburb_rent(sa2_name):
        if not isinstance(sa2_name, str):
            return None
        for suburb in rent_lookup.index:
            if suburb.lower() in sa2_name.lower() or sa2_name.lower() in suburb.lower():
                return float(rent_lookup[suburb])
        return None  # unknown suburb → skip filter

    rents = feature_df['sa2_name'].apply(suburb_rent)
    result = scored_df.copy()

    if budget_max is not None:
        max_rent = investment_to_rent(budget_max)
        too_expensive = rents.apply(lambda r: r is not None and r > max_rent)
        result.loc[too_expensive.values, 'score'] = 0

    if budget_min is not None:
        min_rent = investment_to_rent(budget_min)
        too_cheap = rents.apply(lambda r: r is not None and r < min_rent)
        result.loc[too_cheap.values, 'score'] = 0

    return feature_df, result


# ---------------------------------------------------------------------------
# Dynamic weight builder
# Starts from the base profile and adjusts weights based on user sliders /
# selections. Re-normalises so weights always sum to 1.0.
# ---------------------------------------------------------------------------

# Maps target_customer multi-select values to score columns
CUSTOMER_SCORE_MAP = {
    "office workers": "office_proximity_score",
    "students":       "university_proximity_score",
    "tourists":       "hotel_proximity_score",
    "shoppers":       "shopping_proximity_score",
    "locals":         "pop_density_score",
}

# Maps price_point to the income_score weight multiplier
PRICE_POINT_MULTIPLIER = {
    "budget":    0.5,
    "mid-range": 1.0,
    "premium":   2.0,
}

def build_dynamic_weights(
    business_type: str = None,
    is_first_business: bool = None,
    competition_comfort: int = None,   # 1–5: 1=avoid, 5=thrive
    foot_traffic_importance: int = None,  # 1–5
    target_customers: list = None,     # e.g. ["students", "office workers"]
    age_group: str = None,             # "20s", "30s", "40s+"
    price_point: str = None,           # "budget", "mid-range", "premium"
    transit_importance: int = None,    # 1–5
    near_shopping: bool = None,
) -> dict:
    # Start from base profile
    weights = dict(WEIGHT_PROFILES.get(business_type, WEIGHT_PROFILES["Specialty Cafe"]))

    def slider_scale(value, base, low=0.02, high=0.22):
        """Map a 1–5 slider to a weight between low and high, centred at base."""
        if value is None:
            return base
        return low + (high - low) * (value - 1) / 4

    # Competition comfort: low value → penalise competition more
    if competition_comfort is not None:
        weights['low_competition_score'] = slider_scale(
            6 - competition_comfort, weights.get('low_competition_score', 0.10))
        weights['low_saturation_score'] = slider_scale(
            6 - competition_comfort, weights.get('low_saturation_score', 0.18))

    # Foot traffic importance
    if foot_traffic_importance is not None:
        weights['foot_traffic_score'] = slider_scale(
            foot_traffic_importance, weights.get('foot_traffic_score', 0.07))

    # Transit importance
    if transit_importance is not None:
        weights['rail_proximity_score'] = slider_scale(
            transit_importance, weights.get('rail_proximity_score', 0.07))

    # First business → care more about low competition and saturation
    if is_first_business is True:
        weights['low_competition_score'] = weights.get('low_competition_score', 0.10) * 1.5
        weights['low_saturation_score']  = weights.get('low_saturation_score', 0.18) * 1.3

    # Target customers → boost relevant proximity scores
    if target_customers:
        for customer in target_customers:
            col = CUSTOMER_SCORE_MAP.get(customer.lower())
            if col:
                weights[col] = weights.get(col, 0.03) + 0.06

    # Price point → adjust income score
    if price_point:
        multiplier = PRICE_POINT_MULTIPLIER.get(price_point.lower(), 1.0)
        weights['income_score'] = weights.get('income_score', 0.12) * multiplier

    # Age group → boost age_fit_score weight so age-matched suburbs rank higher
    AGE_GROUP_TARGET = {"20s": 25, "30s": 35, "40s+": 45}
    if age_group and age_group in AGE_GROUP_TARGET:
        weights['age_fit_score'] = weights.get('age_fit_score', 0.05) + 0.06

    # Near shopping hard boost
    if near_shopping is True:
        weights['shopping_proximity_score'] = weights.get('shopping_proximity_score', 0.03) + 0.08

    # Re-normalise so weights sum to 1.0
    total = sum(weights.values())
    weights = {k: round(v / total, 6) for k, v in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# Main entry point — call this from the API
# ---------------------------------------------------------------------------

def score_locations(
    business_type: str = None,
    district: str = None,
    budget_min: float = None,          # AUD total investment lower bound
    budget_max: float = None,          # AUD total investment upper bound
    vision: str = None,                # reserved for future Claude API use
    is_first_business: bool = None,
    competition_comfort: int = None,   # 1–5
    foot_traffic_importance: int = None,  # 1–5
    target_customers: list = None,
    age_group: str = None,
    price_point: str = None,
    transit_importance: int = None,    # 1–5
    near_shopping: bool = None,
    top_n: int = 10,
):
    """
    Score Sydney hexes against user inputs. All parameters are optional —
    omitting one simply skips that filter/weighting dimension.

    Returns a list of dicts ready to JSON-serialise.
    """
    feature_df = pd.read_parquet(FEATURE_PARQUET)
    scored_df  = normalise(feature_df)

    # --- Build weights dynamically from all user inputs ---
    weights = build_dynamic_weights(
        business_type=business_type,
        is_first_business=is_first_business,
        competition_comfort=competition_comfort,
        foot_traffic_importance=foot_traffic_importance,
        target_customers=target_customers,
        age_group=age_group,
        price_point=price_point,
        transit_importance=transit_importance,
        near_shopping=near_shopping,
    )
    scored_df['score'] = apply_weights(scored_df, weights)

    # --- Commercial viability filter (always applied) ---
    scored_df = apply_commercial_filter(feature_df, scored_df)
    scored_df['centre_lat'] = feature_df['centre_lat'].values
    scored_df['centre_lon'] = feature_df['centre_lon'].values
    scored_df['sa2_name']   = feature_df['sa2_name'].values
    scored_df['h3']         = feature_df['h3'].values

    # --- Budget filter (only if at least one bound is given) ---
    if budget_min is not None or budget_max is not None:
        feature_df, scored_df = apply_rent_filter(
            feature_df, scored_df, budget_min=budget_min, budget_max=budget_max)

    # --- District filter (STRICT — only SA2s inside the chosen Sydney area) ---
    if district and district.strip() and district.lower() != "anywhere":
        allowed_sa2s = SYDNEY_DISTRICTS.get(district.strip())
        if allowed_sa2s:
            mask = feature_df['sa2_name'].apply(
                lambda name: any(s.lower() in str(name).lower() for s in allowed_sa2s)
            )
            scored_filtered  = scored_df[mask.values]
            feature_filtered = feature_df[mask.values]
        else:
            # Unknown district name → fall back to all Sydney rather than returning nothing
            print(f"⚠️  Unknown district '{district}' — falling back to all of Sydney")
            scored_filtered  = scored_df
            feature_filtered = feature_df
    else:
        scored_filtered  = scored_df
        feature_filtered = feature_df

    top = diverse_top_n(scored_filtered, n=top_n, min_distance_km=1.5)

    # Load rental CSV once for result enrichment
    rental_df = pd.read_csv(RENTAL_CSV) if RENTAL_CSV.exists() else None

    # Compute Sydney-wide distributions once for chart overlays
    income_vals = feature_df['median_hhd_income_weekly'].dropna().values
    inc_counts, inc_bins = np.histogram(income_vals, bins=15)
    income_hist_data = {
        "bins":   [round(float(b)) for b in inc_bins.tolist()],
        "counts": [int(c) for c in inc_counts.tolist()],
    }
    density_vals = feature_df['pop_density'].dropna().values
    den_counts, den_bins = np.histogram(density_vals, bins=15)
    density_hist_data = {
        "bins":   [round(float(b)) for b in den_bins.tolist()],
        "counts": [int(c) for c in den_counts.tolist()],
    }

    results = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        hex_mask    = feature_df['h3'] == row['h3']
        feature_row = feature_df[hex_mask].iloc[0]
        reasons     = explain_hex(feature_row, feature_df)

        # H3 hex boundary polygon for Leaflet map
        try:
            boundary_pts = h3.cell_to_boundary(str(feature_row['h3']))
            hex_boundary = [[round(lat, 5), round(lon, 5)] for lat, lon in boundary_pts]
        except Exception:
            hex_boundary = []

        rental_info = None
        if rental_df is not None:
            sa2 = feature_row.get('sa2_name', '')
            for _, r in rental_df.iterrows():
                if isinstance(sa2, str) and (
                    r['suburb'].lower() in sa2.lower() or sa2.lower() in r['suburb'].lower()
                ):
                    rental_info = {
                        "suburb":      r['suburb'],
                        "rent_sqm_yr": r['avg_rent_sqm_per_year_aud'],
                        "zone":        r.get('zone', ''),
                    }
                    break

        results.append({
            "rank":        rank,
            "sa2_name":    feature_row.get('sa2_name', 'Unknown'),
            "score":       round(float(row['score']), 3),
            "lat":         round(float(row['centre_lat']), 5),
            "lon":         round(float(row['centre_lon']), 5),
            "maps_url":    f"https://www.google.com/maps/@{row['centre_lat']:.5f},{row['centre_lon']:.5f},17z",
            "reasons":     reasons,
            "rental":      rental_info,
            "vision_echo": vision or None,
            "hex_boundary": hex_boundary,
            "stats": {
                "foot_traffic":  int(feature_row['total_density']),
                "competitors":   int(feature_row['competitors']),
                "complementary": int(feature_row['complementary']),
                "median_income": int(feature_row['median_hhd_income_weekly'])
                                 if pd.notna(feature_row.get('median_hhd_income_weekly')) else None,
                "pop_density":   int(feature_row['pop_density'])
                                 if pd.notna(feature_row.get('pop_density')) else None,
                "dist_rail_km":  round(float(feature_row['dist_rail_km']), 2)
                                 if pd.notna(feature_row.get('dist_rail_km')) else None,
            },
            "income_distribution": {
                **income_hist_data,
                "location_value": int(feature_row['median_hhd_income_weekly'])
                                  if pd.notna(feature_row.get('median_hhd_income_weekly')) else None,
            },
            "pop_distribution": {
                **density_hist_data,
                "location_value": int(feature_row['pop_density'])
                                  if pd.notna(feature_row.get('pop_density')) else None,
            },
        })

    return results
