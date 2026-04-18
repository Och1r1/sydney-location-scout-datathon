from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scorer import score_locations

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuoteRequest(BaseModel):
    business_type: str = None
    district: str = None
    budget_min: float = None
    budget_max: float = None
    vision: str = None
    is_first_business: bool = None
    competition_comfort: int = None      # 1–5
    foot_traffic_importance: int = None  # 1–5
    target_customers: list = None        # e.g. ["students", "office workers"]
    age_group: str = None                # "20s", "30s", "40s+"
    price_point: str = None              # "budget", "mid-range", "premium"
    transit_importance: int = None       # 1–5
    near_shopping: bool = None
    top_n: int = 10


@app.post("/api/quote")
def get_quote(req: QuoteRequest):
    results = score_locations(
        business_type=req.business_type,
        district=req.district,
        budget_min=req.budget_min,
        budget_max=req.budget_max,
        vision=req.vision,
        is_first_business=req.is_first_business,
        competition_comfort=req.competition_comfort,
        foot_traffic_importance=req.foot_traffic_importance,
        target_customers=req.target_customers,
        age_group=req.age_group,
        price_point=req.price_point,
        transit_importance=req.transit_importance,
        near_shopping=req.near_shopping,
        top_n=req.top_n,
    )
    return {"recommendations": results}
