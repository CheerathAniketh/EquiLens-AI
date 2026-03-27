# EquiLens AI
### AI-powered bias detection for non-technical users

> "Amazon's hiring AI downgraded women's CVs. COMPAS 
> flagged Black defendants at 2× the rate. These failures 
> could have been caught. EquiLens catches them."

## The Problem
AI makes life-changing decisions about jobs, loans, and 
healthcare. When trained on biased historical data, these 
systems don't just repeat discrimination — they amplify 
it at scale, silently, with no accountability.

## The Solution
EquiLens gives any organization — NGO, school, small 
business — the ability to audit their data for bias 
before it causes harm. No data science degree required.

## How It Works
1. Upload your CSV dataset
2. Select your target column and sensitive attribute
3. EquiLens analyzes bias using SPD, DI, and EOD metrics
4. SHAP explains which features are causing the bias
5. Gemini translates everything into plain language
6. What-if panel shows impact of removing biased features

## Real User Story
Priya runs an NGO in Pune distributing scholarships. 
She uploads her dataset, selects gender as the sensitive 
attribute, and discovers rural girls are approved at 34% 
the rate of urban boys — a Disparate Impact of 0.34, far 
below the legal threshold of 0.8. Gemini explains this in 
plain language and suggests fixes. Priya downloads the 
audit report and shares it with her board. All in under 
5 minutes.

## Tech Stack
- Backend: FastAPI (Python)
- Bias Metrics: SPD, Disparate Impact, Equalized Odds
- Explainability: SHAP
- AI Layer: Gemini API (explanation + audience toggle)
- Deployment: Google Cloud Run

## Fairness Metrics
| Metric | Threshold | Meaning |
|--------|-----------|---------|
| Disparate Impact | < 0.8 = biased | Legal standard (EEOC) |
| Statistical Parity Difference | > 0.1 = biased | Outcome gap between groups |
| Equalized Odds | > 0.1 = biased | Error rate gap between groups |

## What Makes EquiLens Different
Every existing tool — IBM AI Fairness 360, Fairlearn, 
Aequitas — outputs p-values and confusion matrices that 
only data scientists can interpret. EquiLens translates 
those results into plain language tuned to who's reading:
- Student → learning-oriented explanation
- NGO worker → policy implication
- Policy maker → legal risk framing

## SDG Alignment
EquiLens directly addresses UN Sustainable Development 
Goal 10 — Reduced Inequalities. By making AI fairness 
auditing accessible to organizations without data science 
teams, we prevent algorithmic discrimination before it 
reaches production.


## Status
🚧 Active development 

## Live Demo
🔗 Coming soon

## Setup
```bash
git clone https://github.com/CheerathAniketh/EquiLens-AI
cd EquiLens-AI/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## License
MIT