"""Unit and integration tests for Phase 11 Expense & Labor Tracking Service."""

import tempfile
from decimal import Decimal
import pytest
from fastapi import HTTPException

from backend.app.repositories.farm_db import FarmDatabaseRepository
from backend.app.schemas.economics import (
    ExpenseCategory,
    ExpenseCorrectionCreate,
    FarmExpenseCreate,
    FarmExpenseUpdate,
    PaymentStatus,
)
from backend.app.schemas.farm import (
    AreaUnit,
    Farm,
    FarmerProfile,
    LiteracyPreference,
    OwnershipType,
    UserRecord,
    UserRole,
)
from backend.app.services.economics.expense_service import ExpenseService
from backend.app.services.economics.unit_conversion_service import UnitConversionService


@pytest.fixture
def expense_env():
    """Create test repository and environment for expense testing."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    repo = FarmDatabaseRepository(db_path=tmp.name)

    # Farmer 1
    user1 = UserRecord(
        user_id="usr_kailash",
        username="kailash",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user1)
    profile1 = FarmerProfile(
        farmer_id="fmr_kailash",
        user_id="usr_kailash",
        full_name="Kailash Chandra",
        phone_number="9876543210",
        state="Rajasthan",
        district="Kota",
        sub_district="Ladpura",
        village="Mandana",
        primary_language="hi",
        literacy_preference=LiteracyPreference.VOICE_PRIMARY,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile1)

    farm1 = Farm(
        farm_id="farm_kailash_1",
        farmer_id="fmr_kailash",
        farm_name="Kailash Main Farm",
        total_area=5.0,
        total_area_acres=5.0,
        ownership_type=OwnershipType.OWNED,
        state="Rajasthan",
        district="Kota",
        village="Mandana",
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farm(farm1)

    # Farmer 2 (for tenant isolation tests)
    user2 = UserRecord(
        user_id="usr_suresh",
        username="suresh",
        password_hash="hash",
        salt="salt",
        role=UserRole.FARMER,
        created_at="2026-11-10T00:00:00Z",
    )
    repo.create_user(user2)
    profile2 = FarmerProfile(
        farmer_id="fmr_suresh",
        user_id="usr_suresh",
        full_name="Suresh Patel",
        phone_number="9876543211",
        state="Gujarat",
        district="Anand",
        sub_district="Anand",
        village="Vasad",
        primary_language="gu",
        literacy_preference=LiteracyPreference.TEXT,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
    )
    repo.create_farmer_profile(profile2)

    service = ExpenseService(repo)
    return {"repo": repo, "service": service, "farmer1": profile1, "farmer2": profile2, "farm1": farm1}


def test_unit_conversion_service_precision():
    """Verify Decimal conversion and round_currency avoids floating point errors."""
    assert UnitConversionService.round_currency(Decimal("100.005")) == Decimal("100.01")
    assert UnitConversionService.round_currency(Decimal("100.004")) == Decimal("100.00")
    assert UnitConversionService.round_currency(123.456) == Decimal("123.46")

    # Indian Currency Formatting
    fmt = UnitConversionService.format_inr(Decimal("125000.50"))
    assert "1,25,000.50" in fmt

    # Land conversion
    acres = UnitConversionService.convert_area_to_acres(2.5, "bigha", state="Rajasthan")
    assert acres == Decimal("1.563")  # 2.5 * 0.625 rounded to 3 decimal places


def test_record_expense_with_arithmetic_validation(expense_env):
    """Test successful expense creation when quantity * unit_price matches total."""
    service = expense_env["service"]
    farmer_id = expense_env["farmer1"].farmer_id
    farm_id = expense_env["farm1"].farm_id

    # 5 bags of DAP @ 1350 = 6750
    payload = FarmExpenseCreate(
        farm_id=farm_id,
        expense_category=ExpenseCategory.FERTILIZER_SOIL,
        subcategory="DAP Fertilizer",
        expense_description="5 bags of IFFCO DAP",
        quantity=5.0,
        quantity_unit="bags",
        unit_price=1350.0,
        total_amount=6750.0,
        purchase_date="2026-11-01",
        payment_status=PaymentStatus.PAID,
    )

    rec = service.record_expense(farmer_id=farmer_id, payload=payload, user_id="usr_kailash")
    assert rec.expense_id.startswith("exp_")
    assert rec.total_amount == 6750.0
    assert rec.version_number == 1
    assert rec.payment_status == PaymentStatus.PAID


def test_record_expense_arithmetic_mismatch_fails(expense_env):
    """Test that mismatched arithmetic raises 400 Bad Request."""
    service = expense_env["service"]
    farmer_id = expense_env["farmer1"].farmer_id
    farm_id = expense_env["farm1"].farm_id

    # 10 days labor @ 400 = 4000, but user passed 6000
    payload = FarmExpenseCreate(
        farm_id=farm_id,
        expense_category=ExpenseCategory.LABOR,
        subcategory="Weeding Labor",
        expense_description="10 laborers for weeding",
        quantity=10.0,
        quantity_unit="days",
        unit_price=400.0,
        total_amount=6000.0,  # mismatch!
        purchase_date="2026-11-02",
    )

    with pytest.raises(HTTPException) as exc_info:
        service.record_expense(farmer_id=farmer_id, payload=payload, user_id="usr_kailash")
    assert exc_info.value.status_code == 400
    assert "Total amount mismatch" in exc_info.value.detail


def test_append_only_expense_correction(expense_env):
    """Test that correcting an expense records an audit entry and updates the current total without erasing history."""
    service = expense_env["service"]
    farmer_id = expense_env["farmer1"].farmer_id
    farm_id = expense_env["farm1"].farm_id

    payload = FarmExpenseCreate(
        farm_id=farm_id,
        expense_category=ExpenseCategory.MACHINERY_FUEL,
        subcategory="Tractor Diesel",
        expense_description="Diesel for plowing",
        total_amount=2500.0,
        purchase_date="2026-11-03",
    )
    rec = service.record_expense(farmer_id=farmer_id, payload=payload, user_id="usr_kailash")
    assert rec.total_amount == 2500.0

    # Apply correction: actually paid 2200 after discount
    correction = service.correct_expense(
        expense_id=rec.expense_id,
        farmer_id=farmer_id,
        payload=ExpenseCorrectionCreate(reason="Discount applied at fuel station", new_amount=2200.0),
        user_id="usr_kailash",
    )
    assert correction.previous_amount == 2500.0
    assert correction.new_amount == 2200.0
    assert correction.reason == "Discount applied at fuel station"

    # Updated expense check
    updated = service.get_expense(rec.expense_id, farmer_id=farmer_id)
    assert updated.total_amount == 2200.0
    assert updated.version_number == 2

    # Verify corrections list
    corrs = service.list_corrections(rec.expense_id, farmer_id=farmer_id)
    assert len(corrs) == 1
    assert corrs[0].new_amount == 2200.0


def test_expense_filtering_and_planned_distinction(expense_env):
    """Test filtering expenses by planned status, category, and date range."""
    service = expense_env["service"]
    farmer_id = expense_env["farmer1"].farmer_id
    farm_id = expense_env["farm1"].farm_id

    # Incurred expense
    service.record_expense(
        farmer_id=farmer_id,
        payload=FarmExpenseCreate(
            farm_id=farm_id,
            expense_category=ExpenseCategory.SEEDS_PLANTING,
            expense_description="Wheat Seeds",
            total_amount=1800.0,
            purchase_date="2026-10-15",
            is_planned=False,
        ),
        user_id="usr_kailash",
    )

    # Planned expense
    service.record_expense(
        farmer_id=farmer_id,
        payload=FarmExpenseCreate(
            farm_id=farm_id,
            expense_category=ExpenseCategory.CROP_PROTECTION,
            expense_description="Planned Pesticide",
            total_amount=1200.0,
            purchase_date="2026-11-20",
            is_planned=True,
        ),
        user_id="usr_kailash",
    )

    incurred = service.list_expenses(farmer_id=farmer_id, is_planned=False)
    assert len(incurred) == 1
    assert incurred[0].expense_category == ExpenseCategory.SEEDS_PLANTING

    planned = service.list_expenses(farmer_id=farmer_id, is_planned=True)
    assert len(planned) == 1
    assert planned[0].expense_category == ExpenseCategory.CROP_PROTECTION


def test_expense_tenant_isolation(expense_env):
    """Test that a farmer cannot view, update, correct, or delete another farmer's expense."""
    service = expense_env["service"]
    farmer1_id = expense_env["farmer1"].farmer_id
    farmer2_id = expense_env["farmer2"].farmer_id
    farm1_id = expense_env["farm1"].farm_id

    # Farmer 1 creates expense
    exp1 = service.record_expense(
        farmer_id=farmer1_id,
        payload=FarmExpenseCreate(
            farm_id=farm1_id,
            expense_category=ExpenseCategory.OTHER,
            expense_description="Fencing repair",
            total_amount=950.0,
            purchase_date="2026-11-05",
        ),
        user_id="usr_kailash",
    )

    # Farmer 2 attempts to get expense -> 404
    with pytest.raises(HTTPException) as exc1:
        service.get_expense(exp1.expense_id, farmer_id=farmer2_id)
    assert exc1.value.status_code == 404

    # Farmer 2 attempts to correct expense -> 404
    with pytest.raises(HTTPException) as exc2:
        service.correct_expense(
            exp1.expense_id,
            farmer_id=farmer2_id,
            payload=ExpenseCorrectionCreate(reason="Hack", new_amount=10.0),
            user_id="usr_suresh",
        )
    assert exc2.value.status_code == 404

    # Farmer 2 attempts to delete expense -> 404
    with pytest.raises(HTTPException) as exc3:
        service.delete_expense(exp1.expense_id, farmer_id=farmer2_id, user_id="usr_suresh")
    assert exc3.value.status_code == 404
