"""Durable SQLite database repository for Farmer Identity, Farms, Plots, Crops, Soil, and Activities."""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.schemas.farm import (
    ActivityStatus,
    ActivityType,
    AreaUnit,
    CropCycle,
    CropCycleStatus,
    CropGrowthStage,
    CropSeason,
    Farm,
    FarmActivity,
    FarmerProfile,
    IrrigationMethod,
    IrrigationProfile,
    LiteracyPreference,
    OwnershipType,
    Plot,
    PumpingSource,
    SoilProfile,
    SoilType,
    UserRecord,
    UserRole,
    WaterAvailability,
)

logger = get_logger("FarmDBRepository")


# Area conversion helper to standardized acres
AREA_TO_ACRES = {
    AreaUnit.ACRE: 1.0,
    AreaUnit.HECTARE: 2.47105,
    AreaUnit.BIGHA: 0.625,  # Standard north-Indian bigha benchmark (approx 0.625 acre / 2500 sq m)
    AreaUnit.GUNTHA: 0.025,  # 1 acre = 40 gunthas
}


def normalize_area_to_acres(area: float, unit: AreaUnit) -> float:
    """Normalize regional area unit to standard acres for agronomic calculation."""
    factor = AREA_TO_ACRES.get(unit, 1.0)
    return round(area * factor, 4)


class FarmDatabaseRepository:
    """Thread-safe SQLite repository enforcing strict multi-user tenancy and relational integrity."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.settings = get_settings()
        self.db_path = db_path or self.settings.farm_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a contextual connection with WAL mode and foreign key enforcement."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize relational schema with indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT
            );
            """)

            # 2. Farmer Profiles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS farmer_profiles (
                farmer_id TEXT PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                full_name TEXT NOT NULL,
                phone_number TEXT,
                preferred_language TEXT NOT NULL DEFAULT 'hi-IN',
                experience_years INTEGER,
                literacy_preference TEXT NOT NULL DEFAULT 'voice_primary',
                kisan_credit_card_holder INTEGER DEFAULT 0,
                soil_health_card_available INTEGER DEFAULT 0,
                profile_completion_pct REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 3. Farms
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS farms (
                farm_id TEXT PRIMARY KEY,
                farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
                farm_name TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                tehsil_taluka TEXT,
                village TEXT,
                latitude REAL,
                longitude REAL,
                total_area REAL NOT NULL,
                area_unit TEXT NOT NULL DEFAULT 'acre',
                total_area_acres REAL NOT NULL,
                ownership_type TEXT NOT NULL DEFAULT 'owned',
                is_active_farm INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 4. Farm Plots
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS farm_plots (
                plot_id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL REFERENCES farms(farm_id) ON DELETE CASCADE,
                farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
                plot_name TEXT NOT NULL,
                area REAL NOT NULL,
                area_unit TEXT NOT NULL DEFAULT 'acre',
                area_acres REAL NOT NULL,
                soil_type TEXT NOT NULL DEFAULT 'alluvial',
                irrigation_source TEXT NOT NULL DEFAULT 'Borewell',
                drainage_type TEXT NOT NULL DEFAULT 'Good',
                slope TEXT NOT NULL DEFAULT 'Flat',
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 5. Crop Cycles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS crop_cycles (
                cycle_id TEXT PRIMARY KEY,
                plot_id TEXT NOT NULL REFERENCES farm_plots(plot_id) ON DELETE CASCADE,
                farm_id TEXT NOT NULL REFERENCES farms(farm_id) ON DELETE CASCADE,
                farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
                crop_name TEXT NOT NULL,
                variety_name TEXT,
                season TEXT NOT NULL DEFAULT 'kharif',
                sowing_date TEXT NOT NULL,
                transplanting_date TEXT,
                estimated_harvest_date TEXT,
                actual_harvest_date TEXT,
                current_stage TEXT NOT NULL DEFAULT 'sowing_germination',
                stage_confirmed_by_farmer INTEGER NOT NULL DEFAULT 0,
                target_yield_quintals REAL,
                actual_yield_quintals REAL,
                status TEXT NOT NULL DEFAULT 'active',
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 6. Soil Profiles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS soil_profiles (
                soil_id TEXT PRIMARY KEY,
                plot_id TEXT UNIQUE NOT NULL REFERENCES farm_plots(plot_id) ON DELETE CASCADE,
                farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
                soil_type TEXT NOT NULL DEFAULT 'alluvial',
                ph_level REAL,
                organic_carbon_pct REAL,
                nitrogen_kg_ha REAL,
                phosphorus_kg_ha REAL,
                potassium_kg_ha REAL,
                electrical_conductivity REAL,
                micronutrients_json TEXT,
                is_lab_tested INTEGER NOT NULL DEFAULT 0,
                test_date TEXT,
                soil_health_card_number TEXT,
                testing_lab_name TEXT,
                updated_at TEXT NOT NULL
            );
            """)

            # 7. Irrigation Profiles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS irrigation_profiles (
                irrigation_id TEXT PRIMARY KEY,
                plot_id TEXT UNIQUE NOT NULL REFERENCES farm_plots(plot_id) ON DELETE CASCADE,
                farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
                primary_source TEXT NOT NULL DEFAULT 'Borewell',
                secondary_source TEXT,
                irrigation_method TEXT NOT NULL DEFAULT 'drip',
                water_availability TEXT NOT NULL DEFAULT 'adequate',
                pumping_source TEXT NOT NULL DEFAULT 'electric_grid',
                notes TEXT,
                updated_at TEXT NOT NULL
            );
            """)

            # 8. Farm Activities
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS farm_activities (
                activity_id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL REFERENCES farms(farm_id) ON DELETE CASCADE,
                plot_id TEXT REFERENCES farm_plots(plot_id) ON DELETE SET NULL,
                cycle_id TEXT REFERENCES crop_cycles(cycle_id) ON DELETE SET NULL,
                farmer_id TEXT NOT NULL REFERENCES farmer_profiles(farmer_id) ON DELETE CASCADE,
                activity_type TEXT NOT NULL,
                activity_title TEXT NOT NULL,
                description TEXT,
                scheduled_date TEXT NOT NULL,
                completed_date TEXT,
                status TEXT NOT NULL DEFAULT 'planned',
                stage_tag TEXT,
                cost_incurred REAL,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # Performance & Isolation Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_farms_farmer ON farms(farmer_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plots_farm ON farm_plots(farm_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_plots_farmer ON farm_plots(farmer_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_plot ON crop_cycles(plot_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_farm ON crop_cycles(farm_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_farmer ON crop_cycles(farmer_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_farm ON farm_activities(farm_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_farmer ON farm_activities(farmer_id);")

        logger.info(f"Initialized FarmDatabaseRepository at {self.db_path}")

    # =========================================================================
    # USER & AUTH REPOSITORY METHODS
    # =========================================================================

    def create_user(self, user: UserRecord) -> UserRecord:
        """Insert user credentials into database."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, username, password_hash, salt, role, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user.user_id, user.username, user.password_hash, user.salt, user.role.value, user.created_at, user.last_login),
            )
        return user

    def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """Retrieve user credentials by unique username."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                return None
            return UserRecord(
                user_id=row["user_id"],
                username=row["username"],
                password_hash=row["password_hash"],
                salt=row["salt"],
                role=UserRole(row["role"]),
                created_at=row["created_at"],
                last_login=row["last_login"],
            )

    def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        """Retrieve user credentials by user_id."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                return None
            return UserRecord(
                user_id=row["user_id"],
                username=row["username"],
                password_hash=row["password_hash"],
                salt=row["salt"],
                role=UserRole(row["role"]),
                created_at=row["created_at"],
                last_login=row["last_login"],
            )

    def update_user_last_login(self, user_id: str, last_login: str) -> None:
        """Update last login timestamp."""
        with self.get_connection() as conn:
            conn.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (last_login, user_id))

    # =========================================================================
    # FARMER PROFILE REPOSITORY METHODS
    # =========================================================================

    def create_farmer_profile(self, profile: FarmerProfile) -> FarmerProfile:
        """Create farmer profile bound to user_id."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO farmer_profiles (
                    farmer_id, user_id, full_name, phone_number, preferred_language,
                    experience_years, literacy_preference, kisan_credit_card_holder,
                    soil_health_card_available, profile_completion_pct, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.farmer_id,
                    profile.user_id,
                    profile.full_name,
                    profile.phone_number,
                    profile.preferred_language,
                    profile.experience_years,
                    profile.literacy_preference.value,
                    1 if profile.kisan_credit_card_holder else 0,
                    1 if profile.soil_health_card_available else 0,
                    profile.profile_completion_pct,
                    profile.created_at,
                    profile.updated_at,
                ),
            )
        return profile

    def get_farmer_profile_by_user_id(self, user_id: str) -> Optional[FarmerProfile]:
        """Retrieve profile by user_id."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM farmer_profiles WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                return None
            return self._row_to_farmer_profile(row)

    def get_farmer_profile(self, farmer_id: str) -> Optional[FarmerProfile]:
        """Retrieve profile by farmer_id."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM farmer_profiles WHERE farmer_id = ?", (farmer_id,)).fetchone()
            if not row:
                return None
            return self._row_to_farmer_profile(row)

    def update_farmer_profile(self, profile: FarmerProfile) -> FarmerProfile:
        """Update profile fields and completion score."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE farmer_profiles SET
                    full_name = ?,
                    phone_number = ?,
                    preferred_language = ?,
                    experience_years = ?,
                    literacy_preference = ?,
                    kisan_credit_card_holder = ?,
                    soil_health_card_available = ?,
                    profile_completion_pct = ?,
                    updated_at = ?
                WHERE farmer_id = ?
                """,
                (
                    profile.full_name,
                    profile.phone_number,
                    profile.preferred_language,
                    profile.experience_years,
                    profile.literacy_preference.value,
                    1 if profile.kisan_credit_card_holder else 0,
                    1 if profile.soil_health_card_available else 0,
                    profile.profile_completion_pct,
                    profile.updated_at,
                    profile.farmer_id,
                ),
            )
        return profile

    def _row_to_farmer_profile(self, row: sqlite3.Row) -> FarmerProfile:
        return FarmerProfile(
            farmer_id=row["farmer_id"],
            user_id=row["user_id"],
            full_name=row["full_name"],
            phone_number=row["phone_number"],
            preferred_language=row["preferred_language"],
            experience_years=row["experience_years"],
            literacy_preference=LiteracyPreference(row["literacy_preference"]),
            kisan_credit_card_holder=bool(row["kisan_credit_card_holder"]),
            soil_health_card_available=bool(row["soil_health_card_available"]),
            profile_completion_pct=float(row["profile_completion_pct"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # MULTI-FARM REPOSITORY METHODS (Strict Multi-User Tenancy)
    # =========================================================================

    def create_farm(self, farm: Farm) -> Farm:
        """Create new farm belonging to farmer_id."""
        with self.get_connection() as conn:
            # If this farm is marked active, unset previous active farms for this farmer
            if farm.is_active_farm:
                conn.execute(
                    "UPDATE farms SET is_active_farm = 0 WHERE farmer_id = ?",
                    (farm.farmer_id,),
                )

            conn.execute(
                """
                INSERT INTO farms (
                    farm_id, farmer_id, farm_name, state, district, tehsil_taluka,
                    village, latitude, longitude, total_area, area_unit, total_area_acres,
                    ownership_type, is_active_farm, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    farm.farm_id,
                    farm.farmer_id,
                    farm.farm_name,
                    farm.state,
                    farm.district,
                    farm.tehsil_taluka,
                    farm.village,
                    farm.latitude,
                    farm.longitude,
                    farm.total_area,
                    farm.area_unit.value,
                    farm.total_area_acres,
                    farm.ownership_type.value,
                    1 if farm.is_active_farm else 0,
                    farm.notes,
                    farm.created_at,
                    farm.updated_at,
                ),
            )
        return farm

    def get_farm(self, farm_id: str, farmer_id: str) -> Optional[Farm]:
        """Fetch farm enforcing tenant ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM farms WHERE farm_id = ? AND farmer_id = ?",
                (farm_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_farm(row)

    def get_farms_by_farmer(self, farmer_id: str) -> List[Farm]:
        """List all farms owned by the farmer."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM farms WHERE farmer_id = ? ORDER BY is_active_farm DESC, created_at ASC",
                (farmer_id,),
            ).fetchall()
            return [self._row_to_farm(r) for r in rows]

    def get_active_farm(self, farmer_id: str) -> Optional[Farm]:
        """Get the currently selected active farm for farmer context."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM farms WHERE farmer_id = ? AND is_active_farm = 1 LIMIT 1",
                (farmer_id,),
            ).fetchone()
            if row:
                return self._row_to_farm(row)
            # If no farm explicitly active, check if farmer has exactly 1 farm
            all_farms = conn.execute(
                "SELECT * FROM farms WHERE farmer_id = ?",
                (farmer_id,),
            ).fetchall()
            if len(all_farms) == 1:
                return self._row_to_farm(all_farms[0])
            return None

    def set_active_farm(self, farm_id: str, farmer_id: str) -> Optional[Farm]:
        """Set a farm as the active farm for this farmer."""
        with self.get_connection() as conn:
            # Verify farm belongs to farmer
            row = conn.execute(
                "SELECT * FROM farms WHERE farm_id = ? AND farmer_id = ?",
                (farm_id, farmer_id),
            ).fetchone()
            if not row:
                return None

            conn.execute("UPDATE farms SET is_active_farm = 0 WHERE farmer_id = ?", (farmer_id,))
            conn.execute("UPDATE farms SET is_active_farm = 1, updated_at = ? WHERE farm_id = ? AND farmer_id = ?",
                         (datetime.now().isoformat(), farm_id, farmer_id))

        return self.get_farm(farm_id, farmer_id)

    def update_farm(self, farm: Farm) -> Farm:
        """Update farm record enforcing tenant ownership."""
        with self.get_connection() as conn:
            if farm.is_active_farm:
                conn.execute("UPDATE farms SET is_active_farm = 0 WHERE farmer_id = ?", (farm.farmer_id,))

            conn.execute(
                """
                UPDATE farms SET
                    farm_name = ?,
                    state = ?,
                    district = ?,
                    tehsil_taluka = ?,
                    village = ?,
                    latitude = ?,
                    longitude = ?,
                    total_area = ?,
                    area_unit = ?,
                    total_area_acres = ?,
                    ownership_type = ?,
                    is_active_farm = ?,
                    notes = ?,
                    updated_at = ?
                WHERE farm_id = ? AND farmer_id = ?
                """,
                (
                    farm.farm_name,
                    farm.state,
                    farm.district,
                    farm.tehsil_taluka,
                    farm.village,
                    farm.latitude,
                    farm.longitude,
                    farm.total_area,
                    farm.area_unit.value,
                    farm.total_area_acres,
                    farm.ownership_type.value,
                    1 if farm.is_active_farm else 0,
                    farm.notes,
                    farm.updated_at,
                    farm.farm_id,
                    farm.farmer_id,
                ),
            )
        return farm

    def delete_farm(self, farm_id: str, farmer_id: str) -> bool:
        """Delete farm and cascade to plots and activities (enforcing tenant ownership)."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM farms WHERE farm_id = ? AND farmer_id = ?",
                (farm_id, farmer_id),
            )
            return cursor.rowcount > 0

    def _row_to_farm(self, row: sqlite3.Row) -> Farm:
        return Farm(
            farm_id=row["farm_id"],
            farmer_id=row["farmer_id"],
            farm_name=row["farm_name"],
            state=row["state"],
            district=row["district"],
            tehsil_taluka=row["tehsil_taluka"],
            village=row["village"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            total_area=float(row["total_area"]),
            area_unit=AreaUnit(row["area_unit"]),
            total_area_acres=float(row["total_area_acres"]),
            ownership_type=OwnershipType(row["ownership_type"]),
            is_active_farm=bool(row["is_active_farm"]),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # PLOT / FIELD REPOSITORY METHODS (Tenant Enforced)
    # =========================================================================

    def create_plot(self, plot: Plot) -> Plot:
        """Create plot within farm."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO farm_plots (
                    plot_id, farm_id, farmer_id, plot_name, area, area_unit, area_acres,
                    soil_type, irrigation_source, drainage_type, slope, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plot.plot_id,
                    plot.farm_id,
                    plot.farmer_id,
                    plot.plot_name,
                    plot.area,
                    plot.area_unit.value,
                    plot.area_acres,
                    plot.soil_type.value,
                    plot.irrigation_source,
                    plot.drainage_type,
                    plot.slope,
                    plot.notes,
                    plot.created_at,
                    plot.updated_at,
                ),
            )
        return plot

    def get_plot(self, plot_id: str, farmer_id: str) -> Optional[Plot]:
        """Fetch plot verifying tenant ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM farm_plots WHERE plot_id = ? AND farmer_id = ?",
                (plot_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_plot(row)

    def get_plots_by_farm(self, farm_id: str, farmer_id: str) -> List[Plot]:
        """List plots within a specific farm for this farmer."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM farm_plots WHERE farm_id = ? AND farmer_id = ? ORDER BY created_at ASC",
                (farm_id, farmer_id),
            ).fetchall()
            return [self._row_to_plot(r) for r in rows]

    def get_plots_by_farmer(self, farmer_id: str) -> List[Plot]:
        """List all plots across all farms owned by this farmer."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM farm_plots WHERE farmer_id = ? ORDER BY created_at ASC",
                (farmer_id,),
            ).fetchall()
            return [self._row_to_plot(r) for r in rows]

    def update_plot(self, plot: Plot) -> Plot:
        """Update plot verifying tenant ownership."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE farm_plots SET
                    plot_name = ?,
                    area = ?,
                    area_unit = ?,
                    area_acres = ?,
                    soil_type = ?,
                    irrigation_source = ?,
                    drainage_type = ?,
                    slope = ?,
                    notes = ?,
                    updated_at = ?
                WHERE plot_id = ? AND farmer_id = ?
                """,
                (
                    plot.plot_name,
                    plot.area,
                    plot.area_unit.value,
                    plot.area_acres,
                    plot.soil_type.value,
                    plot.irrigation_source,
                    plot.drainage_type,
                    plot.slope,
                    plot.notes,
                    plot.updated_at,
                    plot.plot_id,
                    plot.farmer_id,
                ),
            )
        return plot

    def delete_plot(self, plot_id: str, farmer_id: str) -> bool:
        """Delete plot verifying tenant ownership."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM farm_plots WHERE plot_id = ? AND farmer_id = ?",
                (plot_id, farmer_id),
            )
            return cursor.rowcount > 0

    def _row_to_plot(self, row: sqlite3.Row) -> Plot:
        return Plot(
            plot_id=row["plot_id"],
            farm_id=row["farm_id"],
            farmer_id=row["farmer_id"],
            plot_name=row["plot_name"],
            area=float(row["area"]),
            area_unit=AreaUnit(row["area_unit"]),
            area_acres=float(row["area_acres"]),
            soil_type=SoilType(row["soil_type"]),
            irrigation_source=row["irrigation_source"],
            drainage_type=row["drainage_type"],
            slope=row["slope"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # CROP CYCLE REPOSITORY METHODS (Tenant Enforced & Historical Preservation)
    # =========================================================================

    def create_crop_cycle(self, cycle: CropCycle) -> CropCycle:
        """Create new crop cycle bound to plot and farm."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO crop_cycles (
                    cycle_id, plot_id, farm_id, farmer_id, crop_name, variety_name,
                    season, sowing_date, transplanting_date, estimated_harvest_date,
                    actual_harvest_date, current_stage, stage_confirmed_by_farmer,
                    target_yield_quintals, actual_yield_quintals, status, notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle.cycle_id,
                    cycle.plot_id,
                    cycle.farm_id,
                    cycle.farmer_id,
                    cycle.crop_name,
                    cycle.variety_name,
                    cycle.season.value,
                    cycle.sowing_date,
                    cycle.transplanting_date,
                    cycle.estimated_harvest_date,
                    cycle.actual_harvest_date,
                    cycle.current_stage.value,
                    1 if cycle.stage_confirmed_by_farmer else 0,
                    cycle.target_yield_quintals,
                    cycle.actual_yield_quintals,
                    cycle.status.value,
                    cycle.notes,
                    cycle.created_at,
                    cycle.updated_at,
                ),
            )
        return cycle

    def get_crop_cycle(self, cycle_id: str, farmer_id: str) -> Optional[CropCycle]:
        """Retrieve crop cycle verifying tenant ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM crop_cycles WHERE cycle_id = ? AND farmer_id = ?",
                (cycle_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_crop_cycle(row)

    def get_active_crop_cycles_by_farm(self, farm_id: str, farmer_id: str) -> List[CropCycle]:
        """Retrieve active crop cycles for a farm."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM crop_cycles WHERE farm_id = ? AND farmer_id = ? AND status = 'active' ORDER BY sowing_date DESC",
                (farm_id, farmer_id),
            ).fetchall()
            return [self._row_to_crop_cycle(r) for r in rows]

    def get_active_crop_cycle_by_plot(self, plot_id: str, farmer_id: str) -> Optional[CropCycle]:
        """Retrieve current active crop cycle for a plot."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM crop_cycles WHERE plot_id = ? AND farmer_id = ? AND status = 'active' ORDER BY sowing_date DESC LIMIT 1",
                (plot_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_crop_cycle(row)

    def get_all_crop_cycles_by_farmer(self, farmer_id: str, status: Optional[str] = None) -> List[CropCycle]:
        """Retrieve crop cycles across all farms (with optional status filter) preserving rotation history."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM crop_cycles WHERE farmer_id = ? AND status = ? ORDER BY sowing_date DESC",
                    (farmer_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM crop_cycles WHERE farmer_id = ? ORDER BY sowing_date DESC",
                    (farmer_id,),
                ).fetchall()
            return [self._row_to_crop_cycle(r) for r in rows]

    def update_crop_cycle(self, cycle: CropCycle) -> CropCycle:
        """Update cycle fields (stage, harvest date, yield) without overwriting history."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE crop_cycles SET
                    crop_name = ?,
                    variety_name = ?,
                    season = ?,
                    sowing_date = ?,
                    transplanting_date = ?,
                    estimated_harvest_date = ?,
                    actual_harvest_date = ?,
                    current_stage = ?,
                    stage_confirmed_by_farmer = ?,
                    target_yield_quintals = ?,
                    actual_yield_quintals = ?,
                    status = ?,
                    notes = ?,
                    updated_at = ?
                WHERE cycle_id = ? AND farmer_id = ?
                """,
                (
                    cycle.crop_name,
                    cycle.variety_name,
                    cycle.season.value,
                    cycle.sowing_date,
                    cycle.transplanting_date,
                    cycle.estimated_harvest_date,
                    cycle.actual_harvest_date,
                    cycle.current_stage.value,
                    1 if cycle.stage_confirmed_by_farmer else 0,
                    cycle.target_yield_quintals,
                    cycle.actual_yield_quintals,
                    cycle.status.value,
                    cycle.notes,
                    cycle.updated_at,
                    cycle.cycle_id,
                    cycle.farmer_id,
                ),
            )
        return cycle

    def _row_to_crop_cycle(self, row: sqlite3.Row) -> CropCycle:
        # Calculate days since sowing
        sowing_d = row["sowing_date"]
        days_elapsed = 0
        try:
            d0 = datetime.strptime(sowing_d, "%Y-%m-%d").date()
            days_elapsed = max(0, (datetime.now().date() - d0).days)
        except Exception:
            pass

        return CropCycle(
            cycle_id=row["cycle_id"],
            plot_id=row["plot_id"],
            farm_id=row["farm_id"],
            farmer_id=row["farmer_id"],
            crop_name=row["crop_name"],
            variety_name=row["variety_name"],
            season=CropSeason(row["season"]),
            sowing_date=row["sowing_date"],
            transplanting_date=row["transplanting_date"],
            estimated_harvest_date=row["estimated_harvest_date"],
            actual_harvest_date=row["actual_harvest_date"],
            current_stage=CropGrowthStage(row["current_stage"]),
            estimated_stage=CropGrowthStage(row["current_stage"]),  # Enriched by service
            days_since_sowing=days_elapsed,
            stage_confirmed_by_farmer=bool(row["stage_confirmed_by_farmer"]),
            target_yield_quintals=row["target_yield_quintals"],
            actual_yield_quintals=row["actual_yield_quintals"],
            status=CropCycleStatus(row["status"]),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # SOIL & IRRIGATION REPOSITORY METHODS
    # =========================================================================

    def save_soil_profile(self, profile: SoilProfile) -> SoilProfile:
        """Create or update soil profile for a plot (tenant enforced)."""
        with self.get_connection() as conn:
            micronutrients_str = json.dumps(profile.micronutrients or {})
            conn.execute(
                """
                INSERT INTO soil_profiles (
                    soil_id, plot_id, farmer_id, soil_type, ph_level, organic_carbon_pct,
                    nitrogen_kg_ha, phosphorus_kg_ha, potassium_kg_ha, electrical_conductivity,
                    micronutrients_json, is_lab_tested, test_date, soil_health_card_number,
                    testing_lab_name, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plot_id) DO UPDATE SET
                    soil_type = excluded.soil_type,
                    ph_level = excluded.ph_level,
                    organic_carbon_pct = excluded.organic_carbon_pct,
                    nitrogen_kg_ha = excluded.nitrogen_kg_ha,
                    phosphorus_kg_ha = excluded.phosphorus_kg_ha,
                    potassium_kg_ha = excluded.potassium_kg_ha,
                    electrical_conductivity = excluded.electrical_conductivity,
                    micronutrients_json = excluded.micronutrients_json,
                    is_lab_tested = excluded.is_lab_tested,
                    test_date = excluded.test_date,
                    soil_health_card_number = excluded.soil_health_card_number,
                    testing_lab_name = excluded.testing_lab_name,
                    updated_at = excluded.updated_at
                WHERE soil_profiles.farmer_id = excluded.farmer_id
                """,
                (
                    profile.soil_id,
                    profile.plot_id,
                    profile.farmer_id,
                    profile.soil_type.value,
                    profile.ph_level,
                    profile.organic_carbon_pct,
                    profile.nitrogen_kg_ha,
                    profile.phosphorus_kg_ha,
                    profile.potassium_kg_ha,
                    profile.electrical_conductivity,
                    micronutrients_str,
                    1 if profile.is_lab_tested else 0,
                    profile.test_date,
                    profile.soil_health_card_number,
                    profile.testing_lab_name,
                    profile.updated_at,
                ),
            )
        return profile

    def get_soil_profile_by_plot(self, plot_id: str, farmer_id: str) -> Optional[SoilProfile]:
        """Fetch soil profile for plot verifying tenant ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM soil_profiles WHERE plot_id = ? AND farmer_id = ?",
                (plot_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_soil_profile(row)

    def _row_to_soil_profile(self, row: sqlite3.Row) -> SoilProfile:
        micro = {}
        if row["micronutrients_json"]:
            try:
                micro = json.loads(row["micronutrients_json"])
            except Exception:
                pass

        return SoilProfile(
            soil_id=row["soil_id"],
            plot_id=row["plot_id"],
            farmer_id=row["farmer_id"],
            soil_type=SoilType(row["soil_type"]),
            ph_level=row["ph_level"],
            organic_carbon_pct=row["organic_carbon_pct"],
            nitrogen_kg_ha=row["nitrogen_kg_ha"],
            phosphorus_kg_ha=row["phosphorus_kg_ha"],
            potassium_kg_ha=row["potassium_kg_ha"],
            electrical_conductivity=row["electrical_conductivity"],
            micronutrients=micro,
            is_lab_tested=bool(row["is_lab_tested"]),
            test_date=row["test_date"],
            soil_health_card_number=row["soil_health_card_number"],
            testing_lab_name=row["testing_lab_name"],
            updated_at=row["updated_at"],
        )

    def save_irrigation_profile(self, profile: IrrigationProfile) -> IrrigationProfile:
        """Create or update irrigation profile for a plot (tenant enforced)."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO irrigation_profiles (
                    irrigation_id, plot_id, farmer_id, primary_source, secondary_source,
                    irrigation_method, water_availability, pumping_source, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plot_id) DO UPDATE SET
                    primary_source = excluded.primary_source,
                    secondary_source = excluded.secondary_source,
                    irrigation_method = excluded.irrigation_method,
                    water_availability = excluded.water_availability,
                    pumping_source = excluded.pumping_source,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                WHERE irrigation_profiles.farmer_id = excluded.farmer_id
                """,
                (
                    profile.irrigation_id,
                    profile.plot_id,
                    profile.farmer_id,
                    profile.primary_source,
                    profile.secondary_source,
                    profile.irrigation_method.value,
                    profile.water_availability.value,
                    profile.pumping_source.value,
                    profile.notes,
                    profile.updated_at,
                ),
            )
        return profile

    def get_irrigation_profile_by_plot(self, plot_id: str, farmer_id: str) -> Optional[IrrigationProfile]:
        """Fetch irrigation profile for plot verifying tenant ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM irrigation_profiles WHERE plot_id = ? AND farmer_id = ?",
                (plot_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_irrigation_profile(row)

    def _row_to_irrigation_profile(self, row: sqlite3.Row) -> IrrigationProfile:
        return IrrigationProfile(
            irrigation_id=row["irrigation_id"],
            plot_id=row["plot_id"],
            farmer_id=row["farmer_id"],
            primary_source=row["primary_source"],
            secondary_source=row["secondary_source"],
            irrigation_method=IrrigationMethod(row["irrigation_method"]),
            water_availability=WaterAvailability(row["water_availability"]),
            pumping_source=PumpingSource(row["pumping_source"]),
            notes=row["notes"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # FARM ACTIVITIES & CALENDAR REPOSITORY METHODS (Tenant Enforced)
    # =========================================================================

    def create_farm_activity(self, activity: FarmActivity) -> FarmActivity:
        """Insert scheduled or due farm activity."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO farm_activities (
                    activity_id, farm_id, plot_id, cycle_id, farmer_id, activity_type,
                    activity_title, description, scheduled_date, completed_date,
                    status, stage_tag, cost_incurred, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    activity.activity_id,
                    activity.farm_id,
                    activity.plot_id,
                    activity.cycle_id,
                    activity.farmer_id,
                    activity.activity_type.value,
                    activity.activity_title,
                    activity.description,
                    activity.scheduled_date,
                    activity.completed_date,
                    activity.status.value,
                    activity.stage_tag,
                    activity.cost_incurred,
                    activity.notes,
                    activity.created_at,
                    activity.updated_at,
                ),
            )
        return activity

    def get_farm_activity(self, activity_id: str, farmer_id: str) -> Optional[FarmActivity]:
        """Get activity by ID verifying tenant ownership."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM farm_activities WHERE activity_id = ? AND farmer_id = ?",
                (activity_id, farmer_id),
            ).fetchone()
            if not row:
                return None
            return self._row_to_farm_activity(row)

    def get_activities_by_farm(
        self,
        farm_id: str,
        farmer_id: str,
        status: Optional[str] = None,
    ) -> List[FarmActivity]:
        """List activities for a farm, optionally filtered by status."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM farm_activities WHERE farm_id = ? AND farmer_id = ? AND status = ? ORDER BY scheduled_date ASC",
                    (farm_id, farmer_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM farm_activities WHERE farm_id = ? AND farmer_id = ? ORDER BY scheduled_date ASC",
                    (farm_id, farmer_id),
                ).fetchall()
            return [self._row_to_farm_activity(r) for r in rows]

    def get_activities_by_farmer(
        self,
        farmer_id: str,
        status: Optional[str] = None,
    ) -> List[FarmActivity]:
        """List activities across all farms owned by this farmer."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM farm_activities WHERE farmer_id = ? AND status = ? ORDER BY scheduled_date ASC",
                    (farmer_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM farm_activities WHERE farmer_id = ? ORDER BY scheduled_date ASC",
                    (farmer_id,),
                ).fetchall()
            return [self._row_to_farm_activity(r) for r in rows]

    def update_farm_activity(self, activity: FarmActivity) -> FarmActivity:
        """Update activity fields (completion date, status, notes) verifying tenant ownership."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE farm_activities SET
                    activity_title = ?,
                    description = ?,
                    scheduled_date = ?,
                    completed_date = ?,
                    status = ?,
                    stage_tag = ?,
                    cost_incurred = ?,
                    notes = ?,
                    updated_at = ?
                WHERE activity_id = ? AND farmer_id = ?
                """,
                (
                    activity.activity_title,
                    activity.description,
                    activity.scheduled_date,
                    activity.completed_date,
                    activity.status.value,
                    activity.stage_tag,
                    activity.cost_incurred,
                    activity.notes,
                    activity.updated_at,
                    activity.activity_id,
                    activity.farmer_id,
                ),
            )
        return activity

    def delete_farm_activity(self, activity_id: str, farmer_id: str) -> bool:
        """Delete activity verifying tenant ownership."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM farm_activities WHERE activity_id = ? AND farmer_id = ?",
                (activity_id, farmer_id),
            )
            return cursor.rowcount > 0

    def _row_to_farm_activity(self, row: sqlite3.Row) -> FarmActivity:
        return FarmActivity(
            activity_id=row["activity_id"],
            farm_id=row["farm_id"],
            plot_id=row["plot_id"],
            cycle_id=row["cycle_id"],
            farmer_id=row["farmer_id"],
            activity_type=ActivityType(row["activity_type"]),
            activity_title=row["activity_title"],
            description=row["description"],
            scheduled_date=row["scheduled_date"],
            completed_date=row["completed_date"],
            status=ActivityStatus(row["status"]),
            stage_tag=row["stage_tag"],
            cost_incurred=row["cost_incurred"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Global singleton instance helper
_farm_repo: Optional[FarmDatabaseRepository] = None


def get_farm_repo(db_path: Optional[str] = None) -> FarmDatabaseRepository:
    """Get or create singleton repository instance."""
    global _farm_repo
    if _farm_repo is None or db_path is not None:
        _farm_repo = FarmDatabaseRepository(db_path=db_path)
    return _farm_repo
