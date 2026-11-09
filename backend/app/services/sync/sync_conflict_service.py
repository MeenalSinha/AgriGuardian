"""Intelligent conflict detection, entity-specific resolution policies, and manual merge service."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status

from backend.app.core.logging import get_logger
from backend.app.repositories.farm_db import FarmDatabaseRepository, get_farm_repo
from backend.app.schemas.sync import (
    ConflictItem,
    ConflictResolutionStrategy,
    ConflictResolveRequest,
    ConflictResolveResponse,
)

logger = get_logger("sync_conflict_service")


class SyncConflictService:
    """Manages version collisions, entity-specific conflict policies, and resolution flows."""

    def __init__(self, repo: Optional[FarmDatabaseRepository] = None):
        self.repo = repo or get_farm_repo()

    def record_conflict(
        self,
        mutation_id: str,
        device_id: str,
        farmer_id: str,
        resource_type: str,
        client_record_id: str,
        server_record_id: Optional[str],
        client_payload: Dict[str, Any],
        server_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Record an edit collision for farmer visibility and explicit resolution."""
        conflict_id = f"cnf_{uuid.uuid4().hex[:12]}"
        logger.warning(
            f"Conflict detected on {resource_type} (client_id: {client_record_id}, server_id: {server_record_id}) for farmer {farmer_id}"
        )
        return self.repo.create_conflict(
            conflict_id=conflict_id,
            mutation_id=mutation_id,
            device_id=device_id,
            farmer_id=farmer_id,
            resource_type=resource_type,
            client_record_id=client_record_id,
            server_record_id=server_record_id,
            client_payload_json=json.dumps(client_payload),
            server_payload_json=json.dumps(server_payload) if server_payload else None,
        )

    def list_conflicts(
        self, farmer_id: str, status_filter: Optional[str] = "pending"
    ) -> List[ConflictItem]:
        """List unresolved or historical conflicts for farmer."""
        rows = self.repo.get_conflicts_for_farmer(farmer_id=farmer_id, status=status_filter)
        items: List[ConflictItem] = []
        for r in rows:
            c_payload = {}
            if r.get("client_payload_json"):
                try:
                    c_payload = json.loads(r["client_payload_json"])
                except Exception:
                    pass

            s_payload = None
            if r.get("server_payload_json"):
                try:
                    s_payload = json.loads(r["server_payload_json"])
                except Exception:
                    pass

            items.append(
                ConflictItem(
                    conflict_id=r["conflict_id"],
                    mutation_id=r["mutation_id"],
                    device_id=r["device_id"],
                    farmer_id=r["farmer_id"],
                    resource_type=r["resource_type"],
                    client_record_id=r["client_record_id"],
                    server_record_id=r.get("server_record_id"),
                    client_payload=c_payload,
                    server_payload=s_payload,
                    resolution_strategy=r.get("resolution_strategy", "pending"),
                    status=r.get("status", "pending"),
                    created_at=r["created_at"],
                    resolved_at=r.get("resolved_at"),
                )
            )
        return items

    def get_conflict(self, conflict_id: str, farmer_id: str) -> ConflictItem:
        """Fetch single conflict detail verifying ownership."""
        row = self.repo.get_conflict(conflict_id, farmer_id=farmer_id)
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conflict '{conflict_id}' not found.",
            )
        c_payload = {}
        if row.get("client_payload_json"):
            try:
                c_payload = json.loads(row["client_payload_json"])
            except Exception:
                pass

        s_payload = None
        if row.get("server_payload_json"):
            try:
                s_payload = json.loads(row["server_payload_json"])
            except Exception:
                pass

        return ConflictItem(
            conflict_id=row["conflict_id"],
            mutation_id=row["mutation_id"],
            device_id=row["device_id"],
            farmer_id=row["farmer_id"],
            resource_type=row["resource_type"],
            client_record_id=row["client_record_id"],
            server_record_id=row.get("server_record_id"),
            client_payload=c_payload,
            server_payload=s_payload,
            resolution_strategy=row.get("resolution_strategy", "pending"),
            status=row.get("status", "pending"),
            created_at=row["created_at"],
            resolved_at=row.get("resolved_at"),
        )

    def resolve_conflict(
        self, conflict_id: str, farmer_id: str, req: ConflictResolveRequest
    ) -> ConflictResolveResponse:
        """Resolve a conflict via CLIENT_WINS, SERVER_WINS, or MANUAL_MERGE."""
        conflict = self.get_conflict(conflict_id, farmer_id)
        if conflict.status == "resolved":
            return ConflictResolveResponse(
                conflict_id=conflict_id,
                status="resolved",
                resolution_strategy=conflict.resolution_strategy,
                resolved_at=conflict.resolved_at or datetime.now(timezone.utc).isoformat(),
            )

        strategy = req.resolution_strategy
        logger.info(
            f"Resolving conflict {conflict_id} on {conflict.resource_type} using {strategy} for farmer {farmer_id}"
        )

        # Apply strategy to underlying entity
        if strategy == ConflictResolutionStrategy.CLIENT_WINS:
            self._apply_payload_to_entity(
                resource_type=conflict.resource_type,
                resource_id=conflict.server_record_id or conflict.client_record_id,
                farmer_id=farmer_id,
                payload=conflict.client_payload,
            )
        elif strategy == ConflictResolutionStrategy.MANUAL_MERGE:
            if not req.merged_payload:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="merged_payload is required when choosing MANUAL_MERGE strategy.",
                )
            self._apply_payload_to_entity(
                resource_type=conflict.resource_type,
                resource_id=conflict.server_record_id or conflict.client_record_id,
                farmer_id=farmer_id,
                payload=req.merged_payload,
            )
        elif strategy == ConflictResolutionStrategy.SERVER_WINS:
            # Server version is preserved; client edit discarded
            logger.info(f"Preserving server state for conflict {conflict_id}")

        self.repo.resolve_conflict(
            conflict_id=conflict_id,
            farmer_id=farmer_id,
            resolution_strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
        )

        now_str = datetime.now(timezone.utc).isoformat()
        return ConflictResolveResponse(
            conflict_id=conflict_id,
            status="resolved",
            resolution_strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
            resolved_at=now_str,
        )

    def _apply_payload_to_entity(
        self, resource_type: str, resource_id: str, farmer_id: str, payload: Dict[str, Any]
    ) -> None:
        """Update entity with chosen payload and record change log."""
        if resource_type == "farm_reminder":
            self.repo.update_farm_reminder(reminder_id=resource_id, farmer_id=farmer_id, update_data=payload)
        elif resource_type == "farm_activity":
            existing = self.repo.get_farm_activity(resource_id, farmer_id)
            if existing:
                for k, v in payload.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                self.repo.update_farm_activity(existing)
        else:
            logger.info(f"Direct payload apply for {resource_type} not requiring custom DB mutation.")
