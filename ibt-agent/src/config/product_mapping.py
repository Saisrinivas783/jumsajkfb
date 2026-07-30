"""Product ID mapping configuration for NCCT ID to Product ID conversion."""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ProductMapping:
    """Product mapping configuration."""
    product_id: str
    plan: str
    brochure: str
    description: str

# Product ID mappings based on plan and brochure combinations
PRODUCT_MAPPINGS = {
    "1": ProductMapping(
        product_id="1",
        plan="standard/basic",
        brochure="fehb",
        description="Standard/Basic FEHB Plan"
    ),
    "4": ProductMapping(
        product_id="4", 
        plan="standard/basic",
        brochure="fehb",
        description="Standard/Basic FEHB Plan"
    ),
    "6": ProductMapping(
        product_id="6",
        plan="blue focus",
        brochure="fehb", 
        description="Blue Focus FEHB Plan"
    ),
    "7": ProductMapping(
        product_id="7",
        plan="standard/basic",
        brochure="pshb",
        description="Standard/Basic PSHB Plan"
    ),
    "8": ProductMapping(
        product_id="8",
        plan="standard/basic", 
        brochure="pshb",
        description="Standard/Basic PSHB Plan"
    ),
    "9": ProductMapping(
        product_id="9",
        plan="blue focus",
        brochure="pshb",
        description="Blue Focus PSHB Plan"
    )
}