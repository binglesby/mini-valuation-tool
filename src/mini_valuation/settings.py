from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_VERSION: str = "2025-09-26"
    CACHE_DIR: str = "data/cache"
    TIMEOUT_S: int = 20

    # Valuation defaults (adjustable in UI)
    TAX_RATE: float = 0.25
    DEFAULT_WACC: float = 0.08
    DEFAULT_TERMINAL_G: float = 0.02

    # Multiples fallback
    FALLBACK_PE: float = 20.0

    # DCF heuristics
    CAPEX_PCT_SALES: float = 0.05
    DELTA_WC_PCT_SALES: float = 0.01
    BETA: float = 1.0
    RF: float = 0.04
    ERP: float = 0.05


@lru_cache(None)
def get_settings() -> Settings:
    return Settings()
