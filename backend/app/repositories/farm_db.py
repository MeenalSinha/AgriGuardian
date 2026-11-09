"""Durable SQLite database repository for Farmer Identity, Farms, Plots, Crops, Soil, and Activities."""

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
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
from backend.app.schemas.analytics import (
    AdvisoryCase,
    CaseFollowUpRecord,
    OfficerTerritoryScope,
)
from backend.app.schemas.expert import (
    ExpertAccountStatus,
    ExpertCredentialRecord,
    ExpertDiagnosisAssessment,
    ExpertProfile,
    ExpertRecommendationVersion,
    ExpertReviewComment,
    ExpertSpecializationItem,
    KnowledgeImprovementProposal,
    DiagnosisDecision,
    RecommendationDecision,
    BiologicalControlItem,
    ChemicalInterventionItem,
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

            # 9. Advisory Cases (Relational persistence replacing legacy CSV row indexes)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS advisory_cases (
                case_id TEXT PRIMARY KEY,
                farmer_id TEXT,
                farmer_name TEXT NOT NULL,
                farm_id TEXT,
                crop TEXT NOT NULL,
                location TEXT NOT NULL,
                state TEXT,
                district TEXT,
                symptoms TEXT,
                risk TEXT NOT NULL DEFAULT 'Low',
                field_visit_required INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                advisory_hi TEXT,
                advisory_en TEXT,
                steps_json TEXT,
                weather_summary TEXT,
                market_summary TEXT,
                image_analysis_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 10. Field Officer Territory Assignments
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS field_officer_assignments (
                assignment_id TEXT PRIMARY KEY,
                officer_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                block TEXT,
                assigned_at TEXT NOT NULL
            );
            """)

            # 11. Case Follow-Ups (Audit log for field visits and agronomist interactions)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_follow_ups (
                follow_up_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES advisory_cases(case_id) ON DELETE CASCADE,
                officer_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                officer_name TEXT NOT NULL,
                notes TEXT NOT NULL,
                action_taken TEXT,
                status_transition TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 12. Phase 8 Expert Profiles
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_profiles (
                expert_id TEXT PRIMARY KEY,
                user_id TEXT UNIQUE NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                full_name TEXT NOT NULL,
                email TEXT,
                phone_number TEXT,
                institution TEXT NOT NULL,
                designation TEXT NOT NULL DEFAULT 'Agronomist',
                experience_years INTEGER NOT NULL DEFAULT 0,
                specialization_json TEXT NOT NULL DEFAULT '{}',
                verification_status TEXT NOT NULL DEFAULT 'pending_verification',
                verified_by_admin_id TEXT REFERENCES users(user_id),
                verified_at TEXT,
                active_caseload INTEGER NOT NULL DEFAULT 0,
                total_reviewed_cases INTEGER NOT NULL DEFAULT 0,
                accuracy_rating REAL NOT NULL DEFAULT 5.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 13. Phase 8 Expert Credentials
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_credentials (
                credential_id TEXT PRIMARY KEY,
                expert_id TEXT NOT NULL REFERENCES expert_profiles(expert_id) ON DELETE CASCADE,
                institution_name TEXT NOT NULL,
                degree_or_certification TEXT NOT NULL,
                registration_number TEXT NOT NULL,
                document_url TEXT,
                issued_date TEXT,
                expiry_date TEXT,
                is_verified INTEGER NOT NULL DEFAULT 0,
                verified_by_admin TEXT REFERENCES users(user_id),
                verified_at TEXT
            );
            """)

            # 14. Phase 8 Expert Review Cases
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_review_cases (
                review_id TEXT PRIMARY KEY,
                case_id TEXT UNIQUE NOT NULL REFERENCES advisory_cases(case_id) ON DELETE CASCADE,
                priority TEXT NOT NULL DEFAULT 'medium',
                status TEXT NOT NULL DEFAULT 'queued',
                eligibility TEXT NOT NULL DEFAULT 'recommended',
                trigger_reasons_json TEXT NOT NULL DEFAULT '[]',
                requires_dual_review INTEGER NOT NULL DEFAULT 0,
                assigned_expert_id TEXT REFERENCES expert_profiles(expert_id),
                secondary_expert_id TEXT REFERENCES expert_profiles(expert_id),
                claimed_at TEXT,
                reviewed_at TEXT,
                secondary_reviewed_at TEXT,
                sla_deadline TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 15. Phase 8 Expert Diagnosis Assessments
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_diagnosis_assessments (
                assessment_id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL REFERENCES expert_review_cases(review_id) ON DELETE CASCADE,
                case_id TEXT NOT NULL REFERENCES advisory_cases(case_id) ON DELETE CASCADE,
                expert_id TEXT NOT NULL REFERENCES expert_profiles(expert_id),
                decision TEXT NOT NULL,
                revised_crop TEXT,
                revised_disease TEXT,
                confidence_score REAL NOT NULL DEFAULT 1.0,
                differential_diagnosis_json TEXT NOT NULL DEFAULT '[]',
                pathology_notes TEXT NOT NULL,
                assessed_at TEXT NOT NULL
            );
            """)

            # 16. Phase 8 Expert Recommendation Versions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_recommendation_versions (
                version_id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL REFERENCES expert_review_cases(review_id) ON DELETE CASCADE,
                case_id TEXT NOT NULL REFERENCES advisory_cases(case_id) ON DELETE CASCADE,
                expert_id TEXT NOT NULL REFERENCES expert_profiles(expert_id),
                version_number INTEGER NOT NULL DEFAULT 1,
                decision TEXT NOT NULL,
                cultural_management_json TEXT NOT NULL DEFAULT '[]',
                biological_control_json TEXT NOT NULL DEFAULT '[]',
                chemical_control_json TEXT NOT NULL DEFAULT '[]',
                pre_harvest_interval_days INTEGER,
                safety_disclaimers_json TEXT NOT NULL DEFAULT '[]',
                rationale TEXT NOT NULL,
                farmer_summary_notes TEXT,
                field_officer_instructions TEXT,
                requires_secondary_review INTEGER NOT NULL DEFAULT 0,
                is_approved INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            """)

            # 17. Phase 8 Expert Review Comments
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_review_comments (
                comment_id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL REFERENCES expert_review_cases(review_id) ON DELETE CASCADE,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

            # 18. Phase 8 Knowledge Improvement Proposals
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_improvement_proposals (
                proposal_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES advisory_cases(case_id) ON DELETE CASCADE,
                expert_id TEXT NOT NULL REFERENCES expert_profiles(expert_id),
                original_ai_disease TEXT NOT NULL,
                expert_confirmed_disease TEXT NOT NULL,
                crop TEXT NOT NULL,
                correction_category TEXT NOT NULL,
                reason_for_override TEXT NOT NULL,
                suggested_kb_fact_or_rule TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_by_admin TEXT,
                created_at TEXT NOT NULL
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_farmer ON advisory_cases(farmer_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_dist ON advisory_cases(district);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_created ON advisory_cases(created_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_officer_assign ON field_officer_assignments(officer_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_follow_ups_case ON case_follow_ups(case_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expert_user ON expert_profiles(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expert_status ON expert_profiles(verification_status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_case ON expert_review_cases(case_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_expert ON expert_review_cases(assigned_expert_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON expert_review_cases(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_priority ON expert_review_cases(priority);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_diag_review ON expert_diagnosis_assessments(review_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rec_ver_review ON expert_recommendation_versions(review_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_review ON expert_review_comments(review_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proposals_status ON knowledge_improvement_proposals(status);")

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

    list_farms_by_farmer = get_farms_by_farmer

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

    list_plots_by_farm = get_plots_by_farm

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
        pending_only: bool = False,
    ) -> List[FarmActivity]:
        """List activities for a farm, optionally filtered by status or pending only."""
        with self.get_connection() as conn:
            if pending_only:
                rows = conn.execute(
                    "SELECT * FROM farm_activities WHERE farm_id = ? AND farmer_id = ? AND status IN ('due', 'planned') ORDER BY scheduled_date ASC",
                    (farm_id, farmer_id),
                ).fetchall()
            elif status:
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

    list_activities_by_farm = get_activities_by_farm

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

    # =========================================================================
    # ADVISORY CASES & CASE REGISTER METHODS (Replacing Legacy CSV Row Indexes)
    # =========================================================================

    def create_advisory_case(
        self,
        case: Optional[AdvisoryCase] = None,
        farmer_name: Optional[str] = None,
        crop: Optional[str] = None,
        location: Optional[str] = None,
        farmer_id: Optional[str] = None,
        farm_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        symptoms: str = "",
        risk: str = "Low",
        field_visit_required: bool = False,
        status: str = "pending",
        advisory_hi: str = "",
        advisory_en: str = "",
        steps: Optional[List[str]] = None,
        weather_summary: str = "",
        market_summary: str = "",
        image_analysis: Optional[Dict[str, Any]] = None,
        case_id: Optional[str] = None,
    ) -> AdvisoryCase:
        """Insert or replace relational advisory case with stable case_id."""
        if case is None:
            now_str = datetime.now().isoformat()
            cid = case_id or f"cas_{uuid.uuid4().hex[:12]}"
            case = AdvisoryCase(
                case_id=cid,
                farmer_id=farmer_id,
                farmer_name=farmer_name or "Farmer",
                farm_id=farm_id,
                crop=crop or "General Crop",
                location=location or "Territory",
                state=state,
                district=district,
                symptoms=symptoms,
                risk=risk,
                field_visit_required=field_visit_required,
                status=status,
                advisory_hi=advisory_hi,
                advisory_en=advisory_en,
                steps=steps or [],
                weather_summary=weather_summary,
                market_summary=market_summary,
                image_analysis=image_analysis,
                created_at=now_str,
                updated_at=now_str,
            )

        with self.get_connection() as conn:
            steps_str = json.dumps(case.steps or [])
            img_str = json.dumps(case.image_analysis or {})
            conn.execute(
                """
                INSERT OR REPLACE INTO advisory_cases (
                    case_id, farmer_id, farmer_name, farm_id, crop, location,
                    state, district, symptoms, risk, field_visit_required,
                    status, advisory_hi, advisory_en, steps_json, weather_summary,
                    market_summary, image_analysis_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case.case_id,
                    case.farmer_id,
                    case.farmer_name,
                    case.farm_id,
                    case.crop,
                    case.location,
                    case.state,
                    case.district,
                    case.symptoms,
                    case.risk,
                    1 if case.field_visit_required else 0,
                    case.status,
                    case.advisory_hi,
                    case.advisory_en,
                    steps_str,
                    case.weather_summary,
                    case.market_summary,
                    img_str,
                    case.created_at,
                    case.updated_at,
                ),
            )
        return case

    def get_advisory_case(self, case_id: str) -> Optional[AdvisoryCase]:
        """Fetch advisory case by stable ID."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM advisory_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not row:
                return None
            return self._row_to_advisory_case(row)

    def list_advisory_cases_for_farmer(self, farmer_id: str, limit: int = 50) -> List[AdvisoryCase]:
        """List advisory cases belonging to a specific farmer."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM advisory_cases WHERE farmer_id = ? ORDER BY created_at DESC LIMIT ?",
                (farmer_id, limit),
            ).fetchall()
            return [self._row_to_advisory_case(r) for r in rows]

    def list_advisory_cases_for_territory(
        self,
        districts: List[str],
        crop: Optional[str] = None,
        status: Optional[str] = None,
        risk: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AdvisoryCase]:
        """List cases matching officer's assigned districts with optional filters."""
        start = start_date or date_from
        end = end_date or date_to
        with self.get_connection() as conn:
            query = "SELECT * FROM advisory_cases WHERE 1=1"
            params: List[Any] = []

            if districts:
                dist_clauses = [f"LOWER(district) = ?" for _ in districts]
                loc_clauses = [f"LOWER(location) LIKE ?" for _ in districts]
                query += f" AND ({' OR '.join(dist_clauses + loc_clauses)})"
                params.extend([d.lower() for d in districts])
                params.extend([f"%{d.lower()}%" for d in districts])

            if crop:
                query += " AND LOWER(crop) = ?"
                params.append(crop.lower())

            if status:
                query += " AND LOWER(status) = ?"
                params.append(status.lower())

            if risk:
                query += " AND LOWER(risk) = ?"
                params.append(risk.lower())

            if start:
                query += " AND created_at >= ?"
                params.append(start)

            if end:
                query += " AND created_at <= ?"
                params.append(end)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_advisory_case(r) for r in rows]

    def count_advisory_cases_for_territory(
        self,
        districts: List[str],
        crop: Optional[str] = None,
        status: Optional[str] = None,
        risk: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> int:
        """Count cases matching territory and criteria."""
        start = start_date or date_from
        end = end_date or date_to
        with self.get_connection() as conn:
            query = "SELECT COUNT(*) as cnt FROM advisory_cases WHERE 1=1"
            params: List[Any] = []

            if districts:
                dist_clauses = [f"LOWER(district) = ?" for _ in districts]
                loc_clauses = [f"LOWER(location) LIKE ?" for _ in districts]
                query += f" AND ({' OR '.join(dist_clauses + loc_clauses)})"
                params.extend([d.lower() for d in districts])
                params.extend([f"%{d.lower()}%" for d in districts])

            if crop:
                query += " AND LOWER(crop) = ?"
                params.append(crop.lower())

            if status:
                query += " AND LOWER(status) = ?"
                params.append(status.lower())

            if risk:
                query += " AND LOWER(risk) = ?"
                params.append(risk.lower())

            if start:
                query += " AND created_at >= ?"
                params.append(start)

            if end:
                query += " AND created_at <= ?"
                params.append(end)

            row = conn.execute(query, params).fetchone()
            return row["cnt"] if row else 0

    def update_advisory_case_status(
        self,
        case_id: str,
        status: Optional[str] = None,
        new_status: Optional[str] = None,
    ) -> bool:
        """Update case workflow status."""
        target_status = status or new_status
        if not target_status:
            raise ValueError("Must provide either status or new_status.")
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE advisory_cases SET status = ?, updated_at = ? WHERE case_id = ?",
                (target_status, datetime.now().isoformat(), case_id),
            )
            return cursor.rowcount > 0

    def create_case_follow_up(
        self,
        follow_up: Optional[CaseFollowUpRecord] = None,
        case_id: Optional[str] = None,
        officer_id: Optional[str] = None,
        officer_name: Optional[str] = None,
        notes: Optional[str] = None,
        action_taken: Optional[str] = None,
        status_transition: Optional[str] = None,
    ) -> CaseFollowUpRecord:
        """Add an audit log entry for a field officer follow-up visit."""
        if follow_up is None:
            if not case_id or not officer_id:
                raise ValueError("Must provide either follow_up or case_id and officer_id.")
            follow_up = CaseFollowUpRecord(
                follow_up_id=f"flw_{uuid.uuid4().hex[:12]}",
                case_id=case_id,
                officer_id=officer_id,
                officer_name=officer_name or "Field Officer",
                notes=notes or "",
                action_taken=action_taken,
                status_transition=status_transition,
                created_at=datetime.now().isoformat(),
            )
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO case_follow_ups (
                    follow_up_id, case_id, officer_id, officer_name, notes, action_taken, status_transition, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    follow_up.follow_up_id,
                    follow_up.case_id,
                    follow_up.officer_id,
                    follow_up.officer_name,
                    follow_up.notes,
                    follow_up.action_taken,
                    follow_up.status_transition,
                    follow_up.created_at,
                ),
            )
            if follow_up.status_transition:
                conn.execute(
                    "UPDATE advisory_cases SET status = ?, updated_at = ? WHERE case_id = ?",
                    (follow_up.status_transition, datetime.now().isoformat(), follow_up.case_id),
                )
        return follow_up

    def list_case_follow_ups(self, case_id: str) -> List[CaseFollowUpRecord]:
        """List all follow-up notes and actions for a case."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM case_follow_ups WHERE case_id = ? ORDER BY created_at ASC",
                (case_id,),
            ).fetchall()
            return [
                CaseFollowUpRecord(
                    follow_up_id=r["follow_up_id"],
                    case_id=r["case_id"],
                    officer_id=r["officer_id"],
                    officer_name=r["officer_name"],
                    notes=r["notes"],
                    action_taken=r["action_taken"],
                    status_transition=r["status_transition"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    # =========================================================================
    # FIELD OFFICER ASSIGNMENT METHODS
    # =========================================================================

    def assign_field_officer_territory(
        self,
        assignment: Optional[OfficerTerritoryScope] = None,
        officer_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        block: Optional[str] = None,
    ) -> OfficerTerritoryScope:
        """Assign an administrative territory to a field officer."""
        if assignment is None:
            if not officer_id or not state or not district:
                raise ValueError("Must provide either assignment or officer_id, state, and district.")
            assignment = OfficerTerritoryScope(
                assignment_id=f"asg_{uuid.uuid4().hex[:12]}",
                officer_id=officer_id,
                state=state,
                district=district,
                block=block,
                assigned_at=datetime.now().isoformat(),
            )
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO field_officer_assignments (assignment_id, officer_id, state, district, block, assigned_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    assignment.assignment_id,
                    assignment.officer_id,
                    assignment.state,
                    assignment.district,
                    assignment.block,
                    assignment.assigned_at,
                ),
            )
        return assignment

    def get_field_officer_territory(self, officer_id: str) -> List[OfficerTerritoryScope]:
        """Fetch all assigned territories for a field officer."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM field_officer_assignments WHERE officer_id = ? ORDER BY assigned_at ASC",
                (officer_id,),
            ).fetchall()
            return [
                OfficerTerritoryScope(
                    assignment_id=r["assignment_id"],
                    officer_id=r["officer_id"],
                    state=r["state"],
                    district=r["district"],
                    block=r["block"],
                    assigned_at=r["assigned_at"],
                )
                for r in rows
            ]

    def remove_field_officer_territory(self, assignment_id: str) -> bool:
        """Remove a territory assignment."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM field_officer_assignments WHERE assignment_id = ?", (assignment_id,))
            return cursor.rowcount > 0

    # =========================================================================
    # PLATFORM OPERATIONAL AGGREGATIONS (Admin Dashboard)
    # =========================================================================

    def get_platform_user_counts(self) -> Dict[str, int]:
        """Count users grouped by role."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT role, COUNT(*) as cnt FROM users GROUP BY role").fetchall()
            result = {"total": 0, "farmer": 0, "field_officer": 0, "agronomist": 0, "administrator": 0}
            total = 0
            for r in rows:
                role_val = r["role"]
                cnt = r["cnt"]
                total += cnt
                if role_val in result:
                    result[role_val] = cnt
            result["total"] = total
            return result

    def get_active_users_count_30d(self) -> int:
        """Count users who have logged in within the past 30 days."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        with self.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE last_login >= ?", (cutoff,)).fetchone()
            return row["cnt"] if row else 0

    def get_platform_total_farms(self) -> int:
        """Total farms registered across all farmers."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM farms").fetchone()
            return row["cnt"] if row else 0

    def get_platform_total_crops(self) -> int:
        """Total active crop cycles across all farms."""
        with self.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM crop_cycles WHERE status = 'active'").fetchone()
            return row["cnt"] if row else 0

    # =========================================================================
    # HISTORICAL CSV ADVISORY MIGRATION
    # =========================================================================

    def migrate_csv_advisories_to_db(self, csv_path: str = "data/advisories.csv") -> int:
        """Migrate legacy CSV advisories into relational advisory_cases table with stable IDs."""
        if not os.path.exists(csv_path):
            return 0

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if df.empty:
                return 0

            imported = 0
            with self.get_connection() as conn:
                for idx, row in df.iterrows():
                    ts = str(row.get("timestamp", datetime.now().isoformat()))
                    case_id = f"cas_legacy_{idx}_{ts.replace(':', '').replace('-', '')[:14]}"
                    
                    exists = conn.execute("SELECT 1 FROM advisory_cases WHERE case_id = ?", (case_id,)).fetchone()
                    if exists:
                        continue

                    loc_raw = str(row.get("location", "India"))
                    state_val = None
                    district_val = None
                    if "," in loc_raw:
                        parts = [p.strip() for p in loc_raw.split(",")]
                        district_val = parts[0]
                        state_val = parts[1] if len(parts) > 1 else None
                    else:
                        district_val = loc_raw

                    steps_raw = row.get("steps", "[]")
                    img_raw = row.get("image_analysis", "{}")

                    conn.execute(
                        """
                        INSERT INTO advisory_cases (
                            case_id, farmer_name, crop, location, state, district,
                            symptoms, risk, field_visit_required, status, advisory_hi,
                            advisory_en, steps_json, weather_summary, market_summary,
                            image_analysis_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            case_id,
                            str(row.get("farmer_name", "Farmer")),
                            str(row.get("crop", "General")),
                            loc_raw,
                            state_val,
                            district_val,
                            str(row.get("symptoms", "")),
                            str(row.get("risk", "Low")),
                            1 if bool(row.get("field_visit_required", False)) else 0,
                            str(row.get("status", "pending")),
                            str(row.get("advisory_hi", "")),
                            str(row.get("advisory_en", "")),
                            str(steps_raw),
                            str(row.get("weather_summary", "")),
                            str(row.get("market_summary", "")),
                            str(img_raw),
                            ts,
                            ts,
                        ),
                    )
                    imported += 1
            if imported > 0:
                logger.info(f"Migrated {imported} legacy CSV advisories into relational advisory_cases.")
            return imported
        except Exception as e:
            logger.error(f"Error migrating legacy CSV advisories: {e}")
            return 0

    def _row_to_advisory_case(self, row: sqlite3.Row) -> AdvisoryCase:
        steps_val = []
        if row["steps_json"]:
            try:
                steps_val = json.loads(row["steps_json"])
            except Exception:
                pass

        img_val = None
        if row["image_analysis_json"]:
            try:
                img_val = json.loads(row["image_analysis_json"])
            except Exception:
                pass

        return AdvisoryCase(
            case_id=row["case_id"],
            farmer_id=row["farmer_id"],
            farmer_name=row["farmer_name"],
            farm_id=row["farm_id"],
            crop=row["crop"],
            location=row["location"],
            state=row["state"],
            district=row["district"],
            symptoms=row["symptoms"] or "",
            risk=row["risk"] or "Low",
            field_visit_required=bool(row["field_visit_required"]),
            status=row["status"] or "pending",
            advisory_hi=row["advisory_hi"] or "",
            advisory_en=row["advisory_en"] or "",
            steps=steps_val if isinstance(steps_val, list) else [],
            weather_summary=row["weather_summary"] or "",
            market_summary=row["market_summary"] or "",
            image_analysis=img_val,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # PHASE 8: AGRICULTURAL EXPERT IDENTITY & REPOSITORY METHODS
    # =========================================================================

    def _row_to_credential(self, row: sqlite3.Row) -> ExpertCredentialRecord:
        """Convert credential database row to schema."""
        return ExpertCredentialRecord(
            credential_id=row["credential_id"],
            expert_id=row["expert_id"],
            institution_name=row["institution_name"],
            degree_or_certification=row["degree_or_certification"],
            registration_number=row["registration_number"],
            document_url=row["document_url"],
            issued_date=row["issued_date"],
            expiry_date=row["expiry_date"],
            is_verified=bool(row["is_verified"]),
            verified_by_admin=row["verified_by_admin"],
            verified_at=row["verified_at"],
        )

    def _row_to_expert_profile(
        self, row: sqlite3.Row, credentials: Optional[List[ExpertCredentialRecord]] = None
    ) -> ExpertProfile:
        """Convert expert profile database row to schema."""
        spec_data = {}
        if row["specialization_json"]:
            try:
                spec_data = json.loads(row["specialization_json"])
            except Exception:
                pass
        specializations = ExpertSpecializationItem(**spec_data) if spec_data else ExpertSpecializationItem()

        return ExpertProfile(
            expert_id=row["expert_id"],
            user_id=row["user_id"],
            full_name=row["full_name"],
            email=row["email"],
            phone_number=row["phone_number"],
            institution=row["institution"],
            designation=row["designation"],
            experience_years=row["experience_years"],
            specializations=specializations,
            credentials=credentials or [],
            verification_status=ExpertAccountStatus(row["verification_status"]),
            verified_by_admin_id=row["verified_by_admin_id"],
            verified_at=row["verified_at"],
            active_caseload=row["active_caseload"],
            total_reviewed_cases=row["total_reviewed_cases"],
            accuracy_rating=row["accuracy_rating"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_expert_profile(self, profile: ExpertProfile) -> ExpertProfile:
        """Insert a new expert profile."""
        spec_json = json.dumps(profile.specializations.model_dump())
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_profiles (
                    expert_id, user_id, full_name, email, phone_number,
                    institution, designation, experience_years, specialization_json,
                    verification_status, verified_by_admin_id, verified_at,
                    active_caseload, total_reviewed_cases, accuracy_rating,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    profile.expert_id,
                    profile.user_id,
                    profile.full_name,
                    profile.email,
                    profile.phone_number,
                    profile.institution,
                    profile.designation,
                    profile.experience_years,
                    spec_json,
                    profile.verification_status.value,
                    profile.verified_by_admin_id,
                    profile.verified_at,
                    profile.active_caseload,
                    profile.total_reviewed_cases,
                    profile.accuracy_rating,
                    profile.created_at,
                    profile.updated_at,
                ),
            )
            # Insert initial credentials if present
            for cred in profile.credentials:
                conn.execute(
                    """
                    INSERT INTO expert_credentials (
                        credential_id, expert_id, institution_name,
                        degree_or_certification, registration_number, document_url,
                        issued_date, expiry_date, is_verified, verified_by_admin, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        cred.credential_id,
                        profile.expert_id,
                        cred.institution_name,
                        cred.degree_or_certification,
                        cred.registration_number,
                        cred.document_url,
                        cred.issued_date,
                        cred.expiry_date,
                        1 if cred.is_verified else 0,
                        cred.verified_by_admin,
                        cred.verified_at,
                    ),
                )
        return self.get_expert_profile(profile.expert_id) or profile

    def get_expert_profile(self, expert_id: str) -> Optional[ExpertProfile]:
        """Fetch expert profile by expert_id with verified credentials."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM expert_profiles WHERE expert_id = ?;", (expert_id,)
            ).fetchone()
            if not row:
                return None
            cred_rows = conn.execute(
                "SELECT * FROM expert_credentials WHERE expert_id = ?;", (expert_id,)
            ).fetchall()
            credentials = [self._row_to_credential(cr) for cr in cred_rows]
            return self._row_to_expert_profile(row, credentials)

    def get_expert_profile_by_user_id(self, user_id: str) -> Optional[ExpertProfile]:
        """Fetch expert profile by linked user_id."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM expert_profiles WHERE user_id = ?;", (user_id,)
            ).fetchone()
            if not row:
                return None
            cred_rows = conn.execute(
                "SELECT * FROM expert_credentials WHERE expert_id = ?;", (row["expert_id"],)
            ).fetchall()
            credentials = [self._row_to_credential(cr) for cr in cred_rows]
            return self._row_to_expert_profile(row, credentials)

    def list_expert_profiles(self, status: Optional[str] = None) -> List[ExpertProfile]:
        """List expert profiles filtered optionally by verification status."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM expert_profiles WHERE verification_status = ? ORDER BY created_at DESC;",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM expert_profiles ORDER BY created_at DESC;"
                ).fetchall()
            
            result = []
            for r in rows:
                cred_rows = conn.execute(
                    "SELECT * FROM expert_credentials WHERE expert_id = ?;", (r["expert_id"],)
                ).fetchall()
                creds = [self._row_to_credential(cr) for cr in cred_rows]
                result.append(self._row_to_expert_profile(r, creds))
            return result

    def update_expert_profile(self, expert_id: str, updates: Dict[str, Any]) -> Optional[ExpertProfile]:
        """Update selected fields of an expert profile."""
        with self.get_connection() as conn:
            set_clauses = []
            params = []
            for key, val in updates.items():
                if key == "specializations" and isinstance(val, (dict, ExpertSpecializationItem)):
                    set_clauses.append("specialization_json = ?")
                    params.append(
                        json.dumps(val.model_dump() if isinstance(val, ExpertSpecializationItem) else val)
                    )
                elif key in (
                    "full_name", "email", "phone_number", "institution", "designation",
                    "experience_years", "active_caseload", "total_reviewed_cases", "accuracy_rating",
                    "verification_status", "verified_by_admin_id", "verified_at"
                ):
                    set_clauses.append(f"{key} = ?")
                    params.append(val.value if isinstance(val, Enum) else val)

            if not set_clauses:
                return self.get_expert_profile(expert_id)

            set_clauses.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(expert_id)

            query = f"UPDATE expert_profiles SET {', '.join(set_clauses)} WHERE expert_id = ?;"
            conn.execute(query, tuple(params))
        return self.get_expert_profile(expert_id)

    def verify_expert(
        self,
        expert_id: str,
        admin_id: str,
        status: ExpertAccountStatus,
        rejection_reason: Optional[str] = None,
    ) -> bool:
        """Admin action to verify, reject, or suspend an agronomist."""
        now_str = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE expert_profiles
                SET verification_status = ?, verified_by_admin_id = ?, verified_at = ?, updated_at = ?
                WHERE expert_id = ?;
                """,
                (status.value, admin_id, now_str, now_str, expert_id),
            )
            # If approved, also mark credentials as verified
            if status == ExpertAccountStatus.ACTIVE_VERIFIED:
                conn.execute(
                    """
                    UPDATE expert_credentials
                    SET is_verified = 1, verified_by_admin = ?, verified_at = ?
                    WHERE expert_id = ?;
                    """,
                    (admin_id, now_str, expert_id),
                )
            return cursor.rowcount > 0

    def add_expert_credential(self, cred: ExpertCredentialRecord) -> ExpertCredentialRecord:
        """Attach a verified academic or institutional credential."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_credentials (
                    credential_id, expert_id, institution_name,
                    degree_or_certification, registration_number, document_url,
                    issued_date, expiry_date, is_verified, verified_by_admin, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    cred.credential_id,
                    cred.expert_id,
                    cred.institution_name,
                    cred.degree_or_certification,
                    cred.registration_number,
                    cred.document_url,
                    cred.issued_date,
                    cred.expiry_date,
                    1 if cred.is_verified else 0,
                    cred.verified_by_admin,
                    cred.verified_at,
                ),
            )
        return cred

    def get_expert_credentials(self, expert_id: str) -> List[ExpertCredentialRecord]:
        """Fetch all credentials for an expert."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM expert_credentials WHERE expert_id = ? ORDER BY is_verified DESC;",
                (expert_id,),
            ).fetchall()
            return [self._row_to_credential(r) for r in rows]

    # =========================================================================
    # PHASE 8: CASE REVIEW WORKFLOW REPOSITORY METHODS
    # =========================================================================

    def create_review_case(
        self,
        case_id: str,
        priority: str,
        eligibility: str,
        trigger_reasons: Optional[List[str]] = None,
        requires_dual_review: bool = False,
        sla_deadline: Optional[str] = None,
        assigned_expert_id: Optional[str] = None,
    ) -> str:
        """Route an advisory case to the expert review queue."""
        review_id = f"rev_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now().isoformat()
        triggers_json = json.dumps(trigger_reasons or [])

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_review_cases (
                    review_id, case_id, priority, status, eligibility,
                    trigger_reasons_json, requires_dual_review, assigned_expert_id,
                    sla_deadline, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    review_id,
                    case_id,
                    priority,
                    "assigned" if assigned_expert_id else "queued",
                    eligibility,
                    triggers_json,
                    1 if requires_dual_review else 0,
                    assigned_expert_id,
                    sla_deadline,
                    now_str,
                    now_str,
                ),
            )
            # Update advisory case status to awaiting review
            conn.execute(
                "UPDATE advisory_cases SET status = 'awaiting_expert_review', updated_at = ? WHERE case_id = ?;",
                (now_str, case_id),
            )
            if assigned_expert_id:
                conn.execute(
                    "UPDATE expert_profiles SET active_caseload = active_caseload + 1 WHERE expert_id = ?;",
                    (assigned_expert_id,),
                )
        return review_id

    def get_review_case(self, review_id: str) -> Optional[Dict[str, Any]]:
        """Fetch expert review case record with details."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM expert_review_cases WHERE review_id = ?;", (review_id,)
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def get_review_case_by_advisory_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Fetch review record for an advisory case."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM expert_review_cases WHERE case_id = ?;", (case_id,)
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def list_review_cases(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        expert_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query review cases matching filtering criteria."""
        with self.get_connection() as conn:
            query = "SELECT * FROM expert_review_cases WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if priority:
                query += " AND priority = ?"
                params.append(priority)
            if expert_id:
                query += " AND (assigned_expert_id = ? OR secondary_expert_id = ?)"
                params.extend([expert_id, expert_id])
            query += " ORDER BY created_at DESC;"
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    def update_review_case(self, review_id: str, updates: Dict[str, Any]) -> bool:
        """Update expert review case fields."""
        with self.get_connection() as conn:
            clauses = []
            params = []
            for k, v in updates.items():
                if k in (
                    "status", "priority", "assigned_expert_id", "secondary_expert_id",
                    "claimed_at", "reviewed_at", "secondary_reviewed_at", "sla_deadline"
                ):
                    clauses.append(f"{k} = ?")
                    params.append(v.value if isinstance(v, Enum) else v)
            if not clauses:
                return False
            clauses.append("updated_at = ?")
            params.append(datetime.now().isoformat())
            params.append(review_id)
            query = f"UPDATE expert_review_cases SET {', '.join(clauses)} WHERE review_id = ?;"
            cursor = conn.execute(query, tuple(params))
            return cursor.rowcount > 0

    def claim_review_case(self, review_id: str, expert_id: str) -> bool:
        """Allow an eligible agronomist to claim an unassigned queued case."""
        now_str = datetime.now().isoformat()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE expert_review_cases
                SET assigned_expert_id = ?, status = 'claimed', claimed_at = ?, updated_at = ?
                WHERE review_id = ? AND assigned_expert_id IS NULL AND status = 'queued';
                """,
                (expert_id, now_str, now_str, review_id),
            )
            if cursor.rowcount > 0:
                conn.execute(
                    "UPDATE expert_profiles SET active_caseload = active_caseload + 1 WHERE expert_id = ?;",
                    (expert_id,),
                )
                return True
            return False

    def save_diagnosis_assessment(self, assessment: ExpertDiagnosisAssessment) -> None:
        """Record expert plant pathology assessment."""
        diff_json = json.dumps(assessment.differential_diagnosis)
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_diagnosis_assessments (
                    assessment_id, review_id, case_id, expert_id, decision,
                    revised_crop, revised_disease, confidence_score,
                    differential_diagnosis_json, pathology_notes, assessed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    decision = excluded.decision,
                    revised_crop = excluded.revised_crop,
                    revised_disease = excluded.revised_disease,
                    confidence_score = excluded.confidence_score,
                    differential_diagnosis_json = excluded.differential_diagnosis_json,
                    pathology_notes = excluded.pathology_notes,
                    assessed_at = excluded.assessed_at;
                """,
                (
                    assessment.assessment_id,
                    assessment.review_id,
                    assessment.case_id,
                    assessment.expert_id,
                    assessment.decision.value,
                    assessment.revised_crop,
                    assessment.revised_disease,
                    assessment.confidence_score,
                    diff_json,
                    assessment.pathology_notes,
                    assessment.assessed_at,
                ),
            )

    def get_diagnosis_assessment(self, review_id: str) -> Optional[ExpertDiagnosisAssessment]:
        """Fetch expert diagnosis assessment for a review case."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM expert_diagnosis_assessments WHERE review_id = ? ORDER BY assessed_at DESC LIMIT 1;",
                (review_id,),
            ).fetchone()
            if not row:
                return None
            diff_diag = []
            if row["differential_diagnosis_json"]:
                try:
                    diff_diag = json.loads(row["differential_diagnosis_json"])
                except Exception:
                    pass
            return ExpertDiagnosisAssessment(
                assessment_id=row["assessment_id"],
                review_id=row["review_id"],
                case_id=row["case_id"],
                expert_id=row["expert_id"],
                decision=DiagnosisDecision(row["decision"]),
                revised_crop=row["revised_crop"],
                revised_disease=row["revised_disease"],
                confidence_score=row["confidence_score"],
                differential_diagnosis=diff_diag,
                pathology_notes=row["pathology_notes"],
                assessed_at=row["assessed_at"],
            )

    def save_recommendation_version(self, rec: ExpertRecommendationVersion) -> None:
        """Preserve immutable recommendation revision authored by agronomist."""
        cult_json = json.dumps(rec.cultural_management)
        bio_json = json.dumps([b.model_dump() if hasattr(b, "model_dump") else b for b in rec.biological_control])
        chem_json = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in rec.chemical_control])
        safety_json = json.dumps(rec.safety_disclaimers)

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_recommendation_versions (
                    version_id, review_id, case_id, expert_id, version_number,
                    decision, cultural_management_json, biological_control_json,
                    chemical_control_json, pre_harvest_interval_days, safety_disclaimers_json,
                    rationale, requires_secondary_review, is_approved, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    rec.version_id,
                    rec.review_id,
                    rec.case_id,
                    rec.expert_id,
                    rec.version_number,
                    rec.decision.value,
                    cult_json,
                    bio_json,
                    chem_json,
                    rec.pre_harvest_interval_days,
                    safety_json,
                    rec.rationale,
                    1 if rec.requires_secondary_review else 0,
                    1,
                    rec.created_at,
                ),
            )

    def get_latest_recommendation_version(self, review_id: str) -> Optional[ExpertRecommendationVersion]:
        """Fetch the most recent recommendation version for a review case."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM expert_recommendation_versions WHERE review_id = ? ORDER BY version_number DESC LIMIT 1;",
                (review_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_recommendation_version(row)

    def list_recommendation_versions(self, review_id: str) -> List[ExpertRecommendationVersion]:
        """Fetch all versions of treatment recommendations for a review case."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM expert_recommendation_versions WHERE review_id = ? ORDER BY version_number ASC;",
                (review_id,),
            ).fetchall()
            return [self._row_to_recommendation_version(r) for r in rows]

    def _row_to_recommendation_version(self, row: sqlite3.Row) -> ExpertRecommendationVersion:
        """Convert row to ExpertRecommendationVersion schema."""
        cultural = []
        bio = []
        chem = []
        safety = []
        try:
            if row["cultural_management_json"]:
                cultural = json.loads(row["cultural_management_json"])
            if row["biological_control_json"]:
                bio_raw = json.loads(row["biological_control_json"])
                bio = [BiologicalControlItem(**b) if isinstance(b, dict) else b for b in bio_raw]
            if row["chemical_control_json"]:
                chem_raw = json.loads(row["chemical_control_json"])
                chem = [ChemicalInterventionItem(**c) if isinstance(c, dict) else c for c in chem_raw]
            if row["safety_disclaimers_json"]:
                safety = json.loads(row["safety_disclaimers_json"])
        except Exception:
            pass

        return ExpertRecommendationVersion(
            version_id=row["version_id"],
            review_id=row["review_id"],
            case_id=row["case_id"],
            expert_id=row["expert_id"],
            version_number=row["version_number"],
            decision=RecommendationDecision(row["decision"]),
            cultural_management=cultural,
            biological_control=bio,
            chemical_control=chem,
            pre_harvest_interval_days=row["pre_harvest_interval_days"],
            safety_disclaimers=safety,
            rationale=row["rationale"],
            requires_secondary_review=bool(row["requires_secondary_review"]),
            created_at=row["created_at"],
        )

    def add_review_comment(self, comment: ExpertReviewComment) -> None:
        """Add communication or query comment on a case."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO expert_review_comments (
                    comment_id, review_id, author_id, author_name, author_role, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    comment.comment_id,
                    comment.review_id,
                    comment.author_id,
                    comment.author_name,
                    comment.author_role,
                    comment.content,
                    comment.created_at,
                ),
            )

    def list_review_comments(self, review_id: str) -> List[ExpertReviewComment]:
        """List chronological comments on a review case."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM expert_review_comments WHERE review_id = ? ORDER BY created_at ASC;",
                (review_id,),
            ).fetchall()
            return [
                ExpertReviewComment(
                    comment_id=r["comment_id"],
                    review_id=r["review_id"],
                    author_id=r["author_id"],
                    author_name=r["author_name"],
                    author_role=r["author_role"],
                    content=r["content"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def create_knowledge_proposal(self, proposal: KnowledgeImprovementProposal) -> None:
        """Record agronomist feedback proposing knowledge base refinement."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_improvement_proposals (
                    proposal_id, case_id, expert_id, original_ai_disease,
                    expert_confirmed_disease, crop, correction_category,
                    reason_for_override, suggested_kb_fact_or_rule, status,
                    reviewed_by_admin, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    proposal.proposal_id,
                    proposal.case_id,
                    proposal.expert_id,
                    proposal.original_ai_disease,
                    proposal.expert_confirmed_disease,
                    proposal.crop,
                    proposal.correction_category,
                    proposal.reason_for_override,
                    proposal.suggested_kb_fact_or_rule,
                    proposal.status,
                    proposal.reviewed_by_admin,
                    proposal.created_at,
                ),
            )

    def list_knowledge_proposals(self, status: Optional[str] = None) -> List[KnowledgeImprovementProposal]:
        """List proposals for administrative knowledge curation."""
        with self.get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM knowledge_improvement_proposals WHERE status = ? ORDER BY created_at DESC;",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM knowledge_improvement_proposals ORDER BY created_at DESC;"
                ).fetchall()
            return [
                KnowledgeImprovementProposal(
                    proposal_id=r["proposal_id"],
                    case_id=r["case_id"],
                    expert_id=r["expert_id"],
                    original_ai_disease=r["original_ai_disease"],
                    expert_confirmed_disease=r["expert_confirmed_disease"],
                    crop=r["crop"],
                    correction_category=r["correction_category"],
                    reason_for_override=r["reason_for_override"],
                    suggested_kb_fact_or_rule=r["suggested_kb_fact_or_rule"],
                    status=r["status"],
                    reviewed_by_admin=r["reviewed_by_admin"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def update_knowledge_proposal_status(self, proposal_id: str, status: str, admin_id: str) -> bool:
        """Admin action on a knowledge improvement proposal."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE knowledge_improvement_proposals SET status = ?, reviewed_by_admin = ? WHERE proposal_id = ?;",
                (status, admin_id, proposal_id),
            )
            return cursor.rowcount > 0



# Global singleton instance helper
_farm_repo: Optional[FarmDatabaseRepository] = None


def get_farm_repo(db_path: Optional[str] = None) -> FarmDatabaseRepository:
    """Get or create singleton repository instance."""
    global _farm_repo
    if _farm_repo is None or db_path is not None:
        _farm_repo = FarmDatabaseRepository(db_path=db_path)
    return _farm_repo


# Alias for backward and forward compatibility
FarmDatabase = FarmDatabaseRepository

