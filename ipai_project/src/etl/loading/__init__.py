"""Loading sub-package.

Exposes the warehouse loader at the package level.
"""

from src.etl.loading.warehouse_loader import WarehouseLoader

__all__ = ["WarehouseLoader"]
