from betlab.api.routers.datasets import router as datasets_router
from betlab.api.routers.strategies import router as strategies_router
from betlab.api.routers.experiments import router as experiments_router
from betlab.api.routers.paper import router as paper_router

__all__ = [
    "datasets_router",
    "strategies_router",
    "experiments_router",
    "paper_router",
]
