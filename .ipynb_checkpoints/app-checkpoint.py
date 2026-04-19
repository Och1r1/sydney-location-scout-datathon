from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scorer import score_locations
from typing import Optional, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuoteRequest(BaseModel):
    business_type: Optional[str] = None
    district: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    vision: Optional[str] = None
    is_first_business: Optional[bool] = None
    competition_comfort: Optional[int] = None
    foot_traffic_importance: Optional[int] = None
    target_customers: Optional[List[str]] = None
    age_group: Optional[str] = None
    price_point: Optional[str] = None
    transit_importance: Optional[int] = None
    near_shopping: Optional[bool] = None
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
