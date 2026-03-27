from pydantic import BaseModel
from typing import Dict, List, Optional


class GroupStat(BaseModel):
    count: int
    positive_rate: float


class AnalyzeRequest(BaseModel):
    target_col: str
    sensitive_col: str
    audience: str = "ngo"


class BiasReport(BaseModel):
    sensitive_col: str
    target_col: str
    group_stats: Dict[str, GroupStat]
    spd: float
    di: float
    bias_detected: bool
    severity: str
    top_features: List[str]
    explanation: str
    fixes: List[str]