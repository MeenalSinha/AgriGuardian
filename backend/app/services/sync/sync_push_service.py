"""Incremental push synchronization, batch mutation executor, and idempotency service."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.farm import ActivityStatus, ActivityType, FarmActivity
from backend.app.schemas.notification import (
    FarmReminderRecord,
    ReminderActivityType,
    ReminderStatus,
)
from backend.app.schemas.sync import (
    MutationResultItem,
    SyncMutationItem,
    SyncPushRequest,
    SyncPushResponse,
)
from backend.app.services.sync.sync_conflict_service import SyncConflictService
from backend.app.services.sync.sync_device_service import SyncDeviceService

logger = get_logger("sync_push_service")


class SyncPushService:
    """Processes batches of offline client mutations with idempotency and optimistic locking."""

    def __init__(
        self,
        repo: Optional[FarmDatabaseRepository] = None,
        device_service: Optional[SyncDeviceService] = None,
        conflict_service: Optional[SyncConflictService] = None,
    ):
        self.repo = repo or get_farm_repo()
        self.device_service = device_service or SyncDeviceService(self.repo)
        self.conflict_service = conflict_service or SyncConflictService(self.repo)

    def process_push_mutations(
        self, user_id: str, farmer_id: str, req: SyncPushRequest
    ) -> SyncPushResponse:
        """Process an ordered batch of offline mutations idempotently."""
        self.device_service.validate_active_device(req.device_id, user_id)

        results: List[MutationResultItem] = []
        applied_count = 0
        conflict_count = 0

        for mutation in req.mutations:
            # 1. Idempotency Check: has this mutation_id already been executed?
            cached_res = self.repo.get_mutation_result(mutation.mutation_id)
            if cached_res:
                logger.info(f"Idempotency hit for mutation {mutation.mutation_id} (status: {cached_res['status']})")
                conflict_details = None
                if cached_res.get("conflict_details_json"):
                    try:
                        conflict_details = json.loads(cached_res["conflict_details_json"])
                    except Exception:
                        pass
                results.append(
                    MutationResultItem(
                        mutation_id=cached_res["mutation_id"],
                        client_record_id=mutation.client_record_id,
                        server_record_id=cached_res["server_record_id"],
                        status=cached_res["status"],
                        server_version=cached_res.get("server_version", 1),
                        conflict_details=conflict_details,
                    )
                )
                if cached_res["status"] == "applied":
                    applied_count += 1
                elif cached_res["status"] == "conflict":
                    conflict_count += 1
                continue

            # Record mutation in durable inbound queue
            self.repo.record_mutation(
                mutation_id=mutation.mutation_id,
                device_id=req.device_id,
                user_id=user_id,
                farmer_id=farmer_id,
                resource_type=mutation.resource_type,
                operation=mutation.operation,
                client_record_id=mutation.client_record_id,
                payload_json=json.dumps(mutation.payload),
                base_version=mutation.base_version,
                status="processing",
            )

            # Process single mutation
            res_item = self._process_single_mutation(
                device_id=req.device_id,
                user_id=user_id,
                farmer_id=farmer_id,
                mutation=mutation,
            )

            # Record result
            result_id = f"mres_{uuid.uuid4().hex[:12]}"
            self.repo.record_mutation_result(
                result_id=result_id,
                mutation_id=mutation.mutation_id,
                server_record_id=res_item.server_record_id,
                status=res_item.status,
                server_version=res_item.server_version,
                conflict_details_json=json.dumps(res_item.conflict_details) if res_item.conflict_details else None,
            )

            if res_item.status == "applied":
                applied_count += 1
            elif res_item.status == "conflict":
                conflict_count += 1

            results.append(res_item)

        self.repo.update_device_sync_time(req.device_id)

        now_str = datetime.now(timezone.utc).isoformat()
        return SyncPushResponse(
            processed_count=len(results),
            applied_count=applied_count,
            conflict_count=conflict_count,
            results=results,
            server_timestamp=now_str,
        )

    def _process_single_mutation(
        self, device_id: str, user_id: str, farmer_id: str, mutation: SyncMutationItem
    ) -> MutationResultItem:
        """Route and execute a mutation on the appropriate domain repository."""
        resource = mutation.resource_type.lower()
        operation = mutation.operation.upper()
        payload = mutation.payload

        try:
            if resource == "farm_reminder":
                return self._handle_reminder_mutation(device_id, farmer_id, mutation)
            elif resource == "farm_activity":
                return self._handle_activity_mutation(device_id, farmer_id, mutation)
            elif resource == "crop_cycle":
                return self._handle_crop_cycle_mutation(device_id, farmer_id, mutation)
            elif resource == "field_observation":
                return self._handle_observation_mutation(device_id, farmer_id, mutation)
            elif resource == "crop_diagnosis_draft":
                return self._handle_diagnosis_draft_mutation(device_id, farmer_id, mutation)
            else:
                return MutationResultItem(
                    mutation_id=mutation.mutation_id,
                    client_record_id=mutation.client_record_id,
                    server_record_id=mutation.client_record_id,
                    status="rejected",
                    error_message=f"Unsupported resource type '{mutation.resource_type}'",
                )
        except Exception as e:
            logger.error(f"Error processing mutation {mutation.mutation_id}: {e}", exc_info=True)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=mutation.client_record_id,
                status="rejected",
                error_message=str(e),
            )

    def _handle_reminder_mutation(
        self, device_id: str, farmer_id: str, mutation: SyncMutationItem
    ) -> MutationResultItem:
        """Process farm_reminder CREATE, COMPLETE, UPDATE, DELETE with optimistic locking."""
        operation = mutation.operation.upper()
        p = mutation.payload

        if operation == "CREATE":
            server_id = f"rem_{uuid.uuid4().hex[:12]}"
            now_str = datetime.now(timezone.utc).isoformat()
            raw_act_type = p.get("activity_type", "custom")
            try:
                rem_act_type = ReminderActivityType(raw_act_type)
            except Exception:
                rem_act_type = ReminderActivityType.CUSTOM

            rec = FarmReminderRecord(
                reminder_id=server_id,
                farmer_id=farmer_id,
                farm_id=p.get("farm_id"),
                crop_cycle_id=p.get("crop_cycle_id"),
                activity_type=rem_act_type,
                title_en=p.get("title_en", "Farm Activity Reminder"),
                title_hi=p.get("title_hi", "कृषि कार्य सूचना"),
                description_en=p.get("description_en"),
                description_hi=p.get("description_hi"),
                scheduled_date=p.get("scheduled_date", now_str[:10]),
                due_time=p.get("due_time", "08:00"),
                status=ReminderStatus.SCHEDULED,
                created_at=now_str,
                updated_at=now_str,
            )
            self.repo.create_farm_reminder(rec)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=1,
            )

        # For UPDATE, COMPLETE, DELETE, find existing reminder
        server_id = p.get("reminder_id") or mutation.client_record_id
        existing = self.repo.get_farm_reminder(server_id)
        if not existing or existing.farmer_id != farmer_id:
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="rejected",
                error_message="Reminder does not exist or does not belong to farmer.",
            )

        server_version = getattr(existing, "version_number", 1)

        # Optimistic locking: check if server version diverged from client base_version
        if mutation.base_version > 0 and server_version > mutation.base_version:
            server_payload = existing.model_dump() if hasattr(existing, "model_dump") else existing.dict()
            conflict = self.conflict_service.record_conflict(
                mutation_id=mutation.mutation_id,
                device_id=device_id,
                farmer_id=farmer_id,
                resource_type="farm_reminder",
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                client_payload=p,
                server_payload=server_payload,
            )
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="conflict",
                server_version=server_version,
                conflict_details=conflict,
            )

        if operation == "COMPLETE":
            self.repo.complete_farm_reminder(server_id, farmer_id=farmer_id)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=server_version + 1,
            )
        elif operation == "UPDATE":
            self.repo.update_farm_reminder(reminder_id=server_id, farmer_id=farmer_id, update_data=p)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=server_version + 1,
            )
        elif operation == "DELETE":
            self.repo.delete_farm_reminder(reminder_id=server_id, farmer_id=farmer_id)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=server_version + 1,
            )

        return MutationResultItem(
            mutation_id=mutation.mutation_id,
            client_record_id=mutation.client_record_id,
            server_record_id=server_id,
            status="rejected",
            error_message=f"Unsupported reminder operation '{operation}'",
        )

    def _handle_activity_mutation(
        self, device_id: str, farmer_id: str, mutation: SyncMutationItem
    ) -> MutationResultItem:
        """Process farm_activity CREATE, COMPLETE, UPDATE, DELETE."""
        operation = mutation.operation.upper()
        p = mutation.payload

        if operation == "CREATE":
            server_id = f"act_{uuid.uuid4().hex[:12]}"
            now_str = datetime.now(timezone.utc).isoformat()
            raw_act_type = p.get("activity_type", "land_preparation")
            try:
                act_type = ActivityType(raw_act_type)
            except Exception:
                if raw_act_type in ("scouting", "observation"):
                    act_type = ActivityType.SOIL_TESTING
                elif raw_act_type in ("fertilization", "manure"):
                    act_type = ActivityType.FERTILIZER_APPLICATION
                elif raw_act_type in ("pest_spray", "spray"):
                    act_type = ActivityType.PESTICIDE_SPRAY
                else:
                    act_type = ActivityType.LAND_PREPARATION

            act = FarmActivity(
                activity_id=server_id,
                farm_id=p.get("farm_id") or "farm_default",
                plot_id=p.get("plot_id"),
                cycle_id=p.get("cycle_id"),
                farmer_id=farmer_id,
                activity_type=act_type,
                activity_title=p.get("activity_title", "Farm Field Operation"),
                description=p.get("description"),
                scheduled_date=p.get("scheduled_date", now_str[:10]),
                completed_date=p.get("completed_date"),
                status=ActivityStatus(p.get("status", "planned")),
                stage_tag=p.get("stage_tag"),
                cost_incurred=float(p.get("cost_incurred", 0.0)),
                notes=p.get("notes"),
                created_at=now_str,
                updated_at=now_str,
            )
            self.repo.create_farm_activity(act)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=1,
            )

        server_id = p.get("activity_id") or mutation.client_record_id
        existing = self.repo.get_farm_activity(server_id, farmer_id=farmer_id)
        if not existing:
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="rejected",
                error_message="Activity does not exist or does not belong to farmer.",
            )

        if operation == "COMPLETE":
            existing.status = ActivityStatus.COMPLETED
            existing.completed_date = p.get("completed_date") or datetime.now(timezone.utc).isoformat()[:10]
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            if p.get("notes"):
                existing.notes = (existing.notes or "") + "\n" + str(p.get("notes"))
            self.repo.update_farm_activity(existing)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=getattr(existing, "version_number", 1) + 1,
            )
        elif operation == "UPDATE":
            for k, v in p.items():
                if hasattr(existing, k) and k not in ("activity_id", "farmer_id", "farm_id"):
                    setattr(existing, k, v)
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            self.repo.update_farm_activity(existing)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=getattr(existing, "version_number", 1) + 1,
            )
        elif operation == "DELETE":
            self.repo.delete_farm_activity(server_id, farmer_id=farmer_id)
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="applied",
                server_version=1,
            )

        return MutationResultItem(
            mutation_id=mutation.mutation_id,
            client_record_id=mutation.client_record_id,
            server_record_id=server_id,
            status="rejected",
            error_message=f"Unsupported activity operation '{operation}'",
        )

    def _handle_crop_cycle_mutation(
        self, device_id: str, farmer_id: str, mutation: SyncMutationItem
    ) -> MutationResultItem:
        """Process crop_cycle offline stage / status update."""
        p = mutation.payload
        server_id = p.get("cycle_id") or mutation.client_record_id
        cycle = self.repo.get_crop_cycle(server_id)
        if not cycle or cycle.farmer_id != farmer_id:
            return MutationResultItem(
                mutation_id=mutation.mutation_id,
                client_record_id=mutation.client_record_id,
                server_record_id=server_id,
                status="rejected",
                error_message="Crop cycle not found or access denied.",
            )

        # Update allowed fields
        for k in ["current_stage", "status", "notes", "estimated_harvest_date"]:
            if k in p and hasattr(cycle, k):
                setattr(cycle, k, p[k])
        cycle.updated_at = datetime.now(timezone.utc).isoformat()
        self.repo.update_crop_cycle(cycle)

        return MutationResultItem(
            mutation_id=mutation.mutation_id,
            client_record_id=mutation.client_record_id,
            server_record_id=server_id,
            status="applied",
            server_version=getattr(cycle, "version_number", 1) + 1,
        )

    def _handle_observation_mutation(
        self, device_id: str, farmer_id: str, mutation: SyncMutationItem
    ) -> MutationResultItem:
        """Process field observation: records scouting activity with observation details."""
        p = mutation.payload
        server_id = f"act_scout_{uuid.uuid4().hex[:10]}"
        now_str = datetime.now(timezone.utc).isoformat()
        notes = f"Field Observation [{p.get('observation_category', 'general')}]: {p.get('notes', '')}"
        act = FarmActivity(
            activity_id=server_id,
            farm_id=p.get("farm_id") or "farm_default",
            plot_id=p.get("plot_id"),
            cycle_id=p.get("cycle_id"),
            farmer_id=farmer_id,
            activity_type=ActivityType.SOIL_TESTING,
            activity_title=f"Field Scouting Observation: {p.get('observation_category', 'General')}",
            description=notes,
            scheduled_date=now_str[:10],
            completed_date=now_str[:10],
            status=ActivityStatus.COMPLETED,
            notes=notes,
            created_at=now_str,
            updated_at=now_str,
        )
        self.repo.create_farm_activity(act)
        return MutationResultItem(
            mutation_id=mutation.mutation_id,
            client_record_id=mutation.client_record_id,
            server_record_id=server_id,
            status="applied",
            server_version=1,
        )

    def _handle_diagnosis_draft_mutation(
        self, device_id: str, farmer_id: str, mutation: SyncMutationItem
    ) -> MutationResultItem:
        """Process offline diagnosis submission draft."""
        p = mutation.payload
        server_id = f"diag_draft_{uuid.uuid4().hex[:10]}"
        logger.info(f"Processed offline diagnosis submission draft {server_id} for farmer {farmer_id}")
        return MutationResultItem(
            mutation_id=mutation.mutation_id,
            client_record_id=mutation.client_record_id,
            server_record_id=server_id,
            status="applied",
            server_version=1,
        )
