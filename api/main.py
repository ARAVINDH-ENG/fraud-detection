from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import predict, review, metrics
from database.db import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FraudShield Detection System",
    description="Real-time fraud detection with risk-based decision engine, SHAP explainability, and human-in-the-loop review",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(predict.router)
app.include_router(review.router)
app.include_router(metrics.router)

# Serve dashboard at http://127.0.0.1:8000/dashboard
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}