"""Custom exceptions for agricultural market intelligence integrations."""


class MarketIntegrationError(Exception):
    """Base exception for agricultural market services."""
    pass


class CommodityNotFoundError(MarketIntegrationError):
    """Raised when a requested commodity cannot be resolved in reference catalogs."""
    pass


class MandiNotFoundError(MarketIntegrationError):
    """Raised when a target APMC mandi is not found or recognized."""
    pass


class MarketAPIError(MarketIntegrationError):
    """Raised when an external market data provider request fails or times out."""
    pass


class InsufficientDataError(MarketIntegrationError):
    """Raised when historical observations are too sparse for statistical analytics or forecasting."""
    pass
