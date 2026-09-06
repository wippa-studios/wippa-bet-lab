from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from betlab.api.routers import datasets, strategies, experiments, paper
from betlab.api.routers.api_models import router as models_router

app = FastAPI(title="Wippa Bet Lab API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(paper.router, prefix="/api/paper", tags=["paper"])
app.include_router(models_router, prefix="/api/models", tags=["models"])


@app.get("/")
def root():
    return {"name": "Wippa Bet Lab API", "version": "0.1.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}
