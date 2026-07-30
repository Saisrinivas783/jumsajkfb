"""Unit tests for product mapping functionality."""

import pytest
from src.config.product_mapping import (
    PRODUCT_MAPPINGS,
    ProductMapping
)

class TestProductMapping:
    """Tests for product ID mapping functionality."""
    
    def test_product_mappings_structure(self):
        """Test that all product mappings have required fields."""
        for product_id, mapping in PRODUCT_MAPPINGS.items():
            assert mapping.product_id == product_id
            assert mapping.plan in ['standard/basic', 'blue focus']
            assert mapping.brochure in ['fehb', 'pshb']
            assert len(mapping.description) > 0
    
    def test_product_mappings_content(self):
        """Test that product mappings contain expected entries."""
        expected_ids = ['1', '4', '6', '7', '8', '9']
        assert sorted(PRODUCT_MAPPINGS.keys()) == sorted(expected_ids)
    
    def test_fehb_products(self):
        """Test FEHB product mappings."""
        fehb_products = {k: v for k, v in PRODUCT_MAPPINGS.items() if v.brochure == 'fehb'}
        assert len(fehb_products) == 3
        assert '1' in fehb_products
        assert '4' in fehb_products
        assert '6' in fehb_products
    
    def test_pshb_products(self):
        """Test PSHB product mappings."""
        pshb_products = {k: v for k, v in PRODUCT_MAPPINGS.items() if v.brochure == 'pshb'}
        assert len(pshb_products) == 3
        assert '7' in pshb_products
        assert '8' in pshb_products
        assert '9' in pshb_products
    
    def test_standard_basic_plans(self):
        """Test standard/basic plan mappings."""
        standard_products = {k: v for k, v in PRODUCT_MAPPINGS.items() if v.plan == 'standard/basic'}
        assert len(standard_products) == 4
        assert '1' in standard_products
        assert '4' in standard_products
        assert '7' in standard_products
        assert '8' in standard_products
    
    def test_blue_focus_plans(self):
        """Test blue focus plan mappings."""
        focus_products = {k: v for k, v in PRODUCT_MAPPINGS.items() if v.plan == 'blue focus'}
        assert len(focus_products) == 2
        assert '6' in focus_products
        assert '9' in focus_products
    
    def test_product_mapping_dataclass(self):
        """Test ProductMapping dataclass functionality."""
        mapping = ProductMapping(
            product_id="test",
            plan="test_plan",
            brochure="test_brochure",
            description="test_description"
        )
        assert mapping.product_id == "test"
        assert mapping.plan == "test_plan"
        assert mapping.brochure == "test_brochure"
        assert mapping.description == "test_description"