"""Competition implementation primitives.

The Phase 0 module intentionally depends only on the Python standard library so
its security and feasibility gates can run before the validator stack is
installed.
"""

from .phase0 import (
    IMAGE_SIZE_LIMIT_BYTES,
    CompetitionItemResult,
    CompetitionScore,
    GateResult,
    GateStatus,
    ImageSizeLimitExceeded,
    RawPatSubmission,
    SecretRedactor,
    allocate_batch_cost,
    calculate_competition_scores,
    enforce_image_size,
    run_local_phase0_gates,
)
from .config import CompetitionConfig, CompetitionManifest, load_manifest
from .manager import CompetitionManager
from .repository import CompetitionRepository
from .state import CompetitionState, ContenderState
from .contracts import CompetitionCompressionRequest, CompetitionCompressionResponse
from .intake import (
    CompetitionSubmissionIntakeService,
    RepositoryIntake,
    RepositorySubmission,
    pinned_repository_source,
)
from .qualification import WarmupQualifier
from .validation import RepositoryStaticValidator, ValidationReason, ValidationStatus
from .build import (
    BuildEvidence,
    BuildReason,
    BuildRequest,
    CompetitionBuildService,
    ModalImageBuildBackend,
    ModalImageBuilder,
    TrustedBuildError,
    TrustedImageBuilder,
)
from .modal_runner import CompetitionModalRunner, ModalSandboxBackend
from .execution import CompetitionExecutionCoordinator
from .media_contracts import (
    CompetitionScoringMedia,
    CompetitionTaskAdapter,
    UpscalingCompetitionAdapterStub,
    UpscalingEvaluationIndexItemStub,
)
from .enrollment import (
    CompetitionEnrollmentDispatcher,
    CompetitionMinerEndpoint,
)
from .artifact_backup import (
    CompetitionArtifactBackupError,
    CompetitionArtifactBackupResult,
    CompetitionArtifactBackupService,
    CompetitionDatabaseBackupError,
    CompetitionDatabaseBackupResult,
)

__all__ = [
    "IMAGE_SIZE_LIMIT_BYTES",
    "CompetitionItemResult",
    "CompetitionScore",
    "GateResult",
    "GateStatus",
    "ImageSizeLimitExceeded",
    "RawPatSubmission",
    "SecretRedactor",
    "allocate_batch_cost",
    "calculate_competition_scores",
    "enforce_image_size",
    "run_local_phase0_gates",
    "CompetitionConfig",
    "CompetitionManifest",
    "CompetitionManager",
    "CompetitionRepository",
    "CompetitionState",
    "ContenderState",
    "load_manifest",
    "CompetitionCompressionRequest",
    "CompetitionCompressionResponse",
    "CompetitionSubmissionIntakeService",
    "RepositoryIntake",
    "RepositorySubmission",
    "pinned_repository_source",
    "RepositoryStaticValidator",
    "ValidationReason",
    "ValidationStatus",
    "WarmupQualifier",
    "BuildEvidence",
    "BuildReason",
    "BuildRequest",
    "CompetitionBuildService",
    "ModalImageBuildBackend",
    "ModalImageBuilder",
    "TrustedBuildError",
    "TrustedImageBuilder",
    "CompetitionModalRunner",
    "ModalSandboxBackend",
    "CompetitionExecutionCoordinator",
    "CompetitionScoringMedia",
    "CompetitionTaskAdapter",
    "UpscalingCompetitionAdapterStub",
    "UpscalingEvaluationIndexItemStub",
    "CompetitionEnrollmentDispatcher",
    "CompetitionMinerEndpoint",
    "CompetitionArtifactBackupError",
    "CompetitionArtifactBackupResult",
    "CompetitionArtifactBackupService",
    "CompetitionDatabaseBackupError",
    "CompetitionDatabaseBackupResult",
]
