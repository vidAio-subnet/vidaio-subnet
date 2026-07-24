# Compression Competition Mode Implementation Plan

- Status: Core Phases 0–5 implemented; Phase 0 still has production-only gates,
  and production remains gated by the outstanding Phase 0, 6, 7, and 8 work
- Initial scope: compression competitions only
- Prepared: 2026-07-13
- Last revised: 2026-07-24

## 1. Objective

Add a weekly, validator-operated competition mode alongside the existing inference mode. Existing synthetic and organic inference must continue unchanged while miners may independently opt into a compression competition by submitting a private GitHub repository.

The validator will:

1. Open registration every Thursday.
2. Poll opted-in miners for a repository submission for a configurable window (30-minute polling for 24 hours by default).
3. Pin and validate each submission as it arrives; at the submission deadline,
   privately back up the final pinned snapshots before any build can begin.
4. Run submissions in validator-owned, network-isolated Modal compute with an immutable source-video Volume mounted read-only and a contender-specific output Volume mounted read-write. The source video is also the trusted scorer reference.
5. Evaluate inputs in batches of five, score every input, track runtime and cost, and resume safely after restarts.
6. Wait until the configured competition end time before finalizing the winner, allowing a recorded maintainer tie-break decision during the waiting period and using Git commit metadata as the deterministic fallback.
7. Feed the latest eligible podium into the 60% compression inference / 20% upscaling inference / 20% compression competition weight policy without mixing competition records into inference tables; split the competition pool 70% / 20% / 10% across first, second, and third place.

This plan deliberately leaves entry fees and competition types beyond compression as extension points. Reproducibility publishing and alpha-stake entry eligibility are mandatory final phases.

## Implementation summary so far

### Phase 0: safety and scoring foundation

Phase 0 established the competition's core safety and scoring rules:

- Contender images have a single 25 GB rejection threshold.
- Competition workloads must run offline in validator-controlled Modal execution.
- Raw GitHub PATs are confined to the miner protocol boundary and redacted from persistence and representative logging/export surfaces.
- Runtime costs are recorded and allocated to individual evaluation items using a documented method.
- Final scoring is 60% absolute compression/VMAF media score, 25% contender-relative cost efficiency, and 15% length-weighted completion. Longer videos have more influence, failed items contribute zero, and the cost component is normalized against the cheapest valid contender for each evaluation item.

The local safety/scoring tests and live Modal isolation/billing checks pass.
Modal's direct build API does not attest final image size, so production
activation requires an explicit acknowledgement of that limitation. The
stricter quota-attested builder contract remains available where required.

### Phase 1: competition control system

Phase 1 built the restart-safe control plane that later execution phases will use:

- A versioned manifest defines each competition's schedule, resources, scoring
  rules, source dataset, and unique ID. Changes remain allowed while the
  competition is non-terminal and are recorded as digest-bound audit events.
- Miners can run inference mode, competition mode, or both without mixing their state.
- Versioned invitation and submission protocols let miners opt in and submit private repositories.
- A restart-safe validator dispatcher snapshots serving miner identities, sends
  invitations on entry to `ENROLLING`, persists opt-in and attempt metadata, and
  polls participating miners at the manifest interval until finalisation.
- Competitions, contenders, evaluation records, batches, reviews, and events use competition-only SQLite tables rather than inference history tables.
- A persisted state machine moves competitions through enrollment, submission finalisation, validation, building, evaluation, scoring, waiting, and completion.
- Database leases and optimistic transitions allow the validator to resume safely after restarts and prevent two schedulers from advancing the same competition concurrently.
- A database-level constraint permits only one active competition at a time.
- Competition mode is disabled by default, so existing synthetic and organic inference behavior remains unchanged.

### Phase 2: miner solution and repository intake

Phase 2 makes it possible for a miner to prepare a real contender and for a
validator to accept it without trusting mutable repository state:

- The compression service exposes `/health` and a local-path `/compress` contract
  while keeping the existing inference API backward compatible.
- Competition inputs must remain below `/evaluation-inputs`, outputs below
  `/output`, remote I/O is disabled, and successful outputs must be valid AV1
  MP4 files that preserve dimensions/timing and are smaller than their inputs.
- A redistribution-safe `compression_warmup_input.mp4` fixture and shared
  batch-of-one preflight verify the exact request/response and media contract.
- The miner SDK creates a sanitized standalone repository, then builds and runs
  it in a disposable validator-shaped Modal Sandbox before publication. The
  warmup fixture is uploaded to a read-only input Volume; a different Volume is
  mounted read-write at `/output`. Runtime networking is blocked, no secrets or
  OIDC identity are attached, CPU/GPU/timeouts are SDK-controlled, and the
  Sandbox and Volumes are cleaned up afterward.
- `validate` and any `publish` without a matching receipt require an
  authenticated Modal SDK session. A
  checksum-, resource-, and age-bound receipt lets `publish` reuse the exact
  successful validation for 24 hours by default instead of running the same
  Sandbox twice.
  Publishing only requests the GitHub PAT after qualification, creates or
  verifies a private GitHub repository, and pushes through ephemeral askpass.
- Git commits use the account authenticated by the PAT, with its GitHub no-reply
  identity. Existing private repositories are updated by fetching their default
  branch and adding a fast-forward child commit; no force-push is used.
- Validator intake validates protocol bindings, clones once, removes the remote,
  pins commit/tree/timestamp metadata, makes source read-only, and rejects unsafe
  layouts or clear secret/obfuscation/download patterns with stable reason codes.
- At the end of submission collection, the validator creates a deterministic
  archive and JSON inventory of the final pinned revisions and uploads both to
  a private S3 bucket without provider-specific object ACL overrides. SQLite
  records the backup state, checksum, object keys, and private `s3://` folder
  path. Validation/building cannot begin until the bucket and both inherited
  object ACLs have no public grants and the objects pass size verification; a
  failed backup remains in `FINALIZING_SUBMISSIONS` and is retried after restart
  or on the next tick.

In the bigger picture, Phases 0–2 now provide the rules, security boundaries,
communication contracts, database/scheduler, a working miner submission path,
and a validator-equivalent warmup environment.

### Phase 3 summary: trusted build and validator-owned Modal execution

Phase 3 moves that security contract to the validator side:

- A trusted-build boundary accepts an image only when allowlisted builder
  evidence binds it to the pinned Git tree, supplies an immutable image ID and
  SHA-256 digest, proves that the 25 GB quota was enforced during the hostile
  build, and measures a final size no greater than 25 GB. The production Modal
  path separately records `MODAL_UNATTESTED` evidence because Modal does not
  currently supply that size attestation, and requires an explicit operator
  acknowledgement before it can run.
- Accepted build evidence is persisted on the contender and is immutable.
- The validator-owned Modal adapter uses the recorded image ID and ignores all
  miner resource declarations. It fixes the service command, GPU, CPU hard
  limit, lifetime, environment, network block, empty secrets/OIDC, empty public
  ports, one read-only `/evaluation-inputs` mount, and one contender-specific
  read-write `/output` mount. No other contender or validator Volume is mounted;
  compression scoring reuses the immutable source input as its reference.
- Every new or recovered sandbox must pass a validator-supplied active probe:
  DNS, direct-IP, and HTTPS access must fail; the input mount must reject a
  write; the output mount must accept a write; reference paths and cloud/GitHub
  credentials must be absent. A separate trusted probe records the GPU
  model/count and cgroup/CPU-affinity allocation actually present so accounting
  never assumes that Modal fulfilled the requested hardware literally. Failure
  forces immediate termination.
- A competition-only `competition_sandboxes` table records each lifecycle
  generation, image and resource policy, external IDs, isolation report,
  health, expiry, and termination reason. The supervisor reattaches after a
  process restart, reuses one warm sandbox sequentially, and replaces it before
  Modal's 24-hour ceiling using the same image and output Volume.
- Batch calls run only over localhost through `sandbox.exec`. A call failure or
  supervisor timeout terminates the sandbox so a subsequent attempt gets a
  clean generation. Calls for different contenders are designed to run in
  parallel; calls for one contender remain sequential within its warm Sandbox.
- The live execution coordinator advances accepted pinned submissions through
  `FINALIZING_SUBMISSIONS`, `VALIDATING`, `BUILDING`, and `EVALUATING`; records
  build start/success/rejection events; builds contenders concurrently; and
  starts or recovers one isolated Sandbox per successfully built contender in
  parallel. Both operations are bounded by `max_parallel_contenders`.
- Testnet and production operators select the `modal` backend. It records new
  images as `MODAL_ACCEPTED` and requires a separate acknowledgement that Modal
  does not attest the final 25 GB image size. Existing
  `DEVELOPMENT_ACCEPTED` rows remain accepted for migration only.

In the bigger picture, Phases 0–3 now provide the rules, security boundaries,
control plane, submission path, trusted image contract, and restart-safe offline
execution environment. Phase 4 can focus on dispatching the private benchmark,
validating outputs, scoring every item, and accounting for cost without also
having to invent the sandbox security model. Later phases finalize rankings and
weights, publish reproducibility bundles, and enforce alpha-stake entry criteria.

Production Modal execution is deliberately gated by
`COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true`. This makes the
unavailable measurement visible in configuration and persisted evidence. A
deployment can replace this acknowledgement with an isolated quota-enforcing
builder that satisfies the stricter trusted-build evidence contract.

## 2. Original repository baseline

- `neurons/validator.py` starts four long-lived tasks: synthetic inference, organic inference, weight setting, and W&B maintenance. `start_synthetic_epoch()` discovers a miner's inference task through `TaskWarrantProtocol`.
- `vidaio_subnet_core/protocol.py` defines only inference task types (`COMPRESSION` and `UPSCALING`) and has no competition enrollment or submission protocol.
- `vidaio_subnet_core/base/miner.py` attaches the inference and organic job/poll handlers to the miner axon.
- `neurons/miner.py` selects HTTP or miner-owned Modal inference through `MINER_PROCESSING_BACKEND`. Its task warrant is currently a single module-level value.
- `miner/modal_workers.py` is a miner-owned Modal app named `vidaio-miner-workers`. It controls its own secrets, GPUs, CPUs, timeouts, functions, and writable volumes. This is suitable for miner-operated inference, but not for validator-operated execution of untrusted competition code.
- `miner/compression/app.py` already accepts local input paths, but always chooses output paths inside one shared service directory. Competition mode needs explicit, validated per-input output paths.
- SQLite persistence is owned by `MinerManager` and currently contains `miner_metadata`, `miner_performance_history`, and emission snapshot data.
- `run.sh` starts four inference scoring endpoints plus scheduler and gateway processes; there is no competition dispatcher or competition scoring worker.

## 3. Non-negotiable design boundaries

### 3.1 Preserve inference mode

Competition protocols, state, queues, tables, and weights must be separate from inference equivalents. A miner can configure `inference`, `competition`, or `both`; `both` is the recommended default for the repository's miner template. Failure or maintenance in competition mode must not stop synthetic or organic inference.

### 3.2 The validator owns the security boundary

Do **not** execute `modal deploy` directly against an untrusted contender repository and do not import its `modal_workers.py` in the validator process. A Python Modal file can execute code at import time and can otherwise request arbitrary resources, secrets, volumes, or network access.

Instead:

- A validator-owned launcher builds a pinned image and creates the Modal Sandbox or restricted function.
- The validator sets the GPU, CPU hard limit, timeout, network policy, volume mounts, app name, and tags.
- The submitted `modal_workers.py` remains a required, readable contract adapter and local self-test entrypoint. Resource declarations in it are statically checked, but are never treated as authoritative.
- No validator secrets are exposed during build or execution.

Modal currently supports outbound blocking with `block_network=True`, CPU hard limits using a `(request, limit)` tuple, Sandbox GPU selection, execution timeouts, and read-only Volume mounts. These controls must be applied by trusted validator code, not contender code. See the official [restricted execution](https://modal.com/docs/guide/restricted-access), [Sandbox API](https://modal.com/docs/sdk/py/latest/modal.Sandbox), [Sandbox resource limits](https://modal.com/docs/guide/sandbox-resources), and [Volume mount options](https://modal.com/docs/guide/volumes) documentation.

### 3.3 SQLite is authoritative; Redis is transient

SQLite stores lifecycle state, pinned submissions, evaluation attempts, scores, and final results. Redis is used only for work dispatch, response notification, leases, and idempotency acceleration. Every Redis job must be reconstructible from SQLite after a process restart or Redis flush.

### 3.4 Fail closed

A submission is rejected if the validator cannot prove that its image, resource request, routes, repository revision, network policy, volume isolation, or output mapping complies with the competition manifest.

## 4. Proposed lifecycle

Use an explicit persisted state machine:

```text
SCHEDULED
  -> ENROLLING
  -> FINALIZING_SUBMISSIONS
  -> VALIDATING
  -> BUILDING
  -> EVALUATING
  -> SCORING
  -> AWAITING_END_TIME
  -> COMPLETED

Any pre-completion state may also enter FAILED or CANCELLED.
Individual contenders have their own ACCEPTED, REJECTED, BUILT, RUNNING,
SCORED, and FAILED states so one bad repository does not abort the competition.
```

Default weekly timeline, expressed in UTC:

- Thursday at `competition_start_time`: move to `ENROLLING` and send invitations.
- Thursday through Friday: poll interested miners every `contender_ping_interval` until `contender_finalisation_time`.
- At finalisation: stop accepting or changing submissions, privately archive
  every final pinned repository snapshot and its inventory to S3, then validate
  and build only after the backup is verified.
- Remaining competition window: evaluate contenders continuously and score responses asynchronously.
- If evaluation finishes early: move to `AWAITING_END_TIME`, publish the provisional ranking and review queue, and accept audited human eligibility/tie-break decisions until the review deadline.
- At `competition_end_time` (normally the following Thursday): reconcile scores and costs, apply tie-breakers, persist and announce the winner.

`Validator.run_competition()` will be a fifth task created in `neurons/validator.py::main()`. It should wake at a short configurable scheduler interval, acquire a database-backed competition lease, and advance any due state. It must not contain a single week-long sleep; that would make restarts and configuration changes unsafe.

## 5. Competition manifest

Add versioned manifests under `competitions/manifests/`. Validate them with a new
Pydantic `CompetitionManifest` before a competition is inserted or updated.
Store the normalized manifest JSON and SHA-256 digest in SQLite. An edit to a
non-terminal competition updates its scheduler-facing columns and appends the
old/new digests plus field-level changes to the event log; terminal competition
manifests remain immutable.

Proposed initial manifest:

```yaml
schema_version: 2
scoring_version: "3"
competition_id: compression-2026-w29
competition_type: COMPRESSION
competition_start_time: 2026-07-16T00:00:00Z
contender_ping_interval: 30m
contender_finalisation_time: 2026-07-17T00:00:00Z
human_review_deadline: 2026-07-22T23:30:00Z
competition_end_time: 2026-07-23T00:00:00Z

required_routes:
  - /compress
allowed_gpus:
  - L4
  - L40S
  - RTX-PRO-6000
max_cpu_cores: 32
requested_cpu_cores: 16

container_size_limit_gb: 25
evaluation_batch_size: 5
evaluation_batched_run_timeout: 10m
scoring_batched_run_timeout: 5m
min_video_length: 5s
max_video_length: 1h
length_weight_exponent: 2.02
required_output_codec: AV1
vmaf_threshold: 90.0
vmaf_sample_count: 10
minimum_compression_ratio: 1.25
scoring_seed: 20260716
warmup_input_path: competitions/fixtures/compression_warmup_input.mp4

boss:
  repository_path: competition_boss/compression-current
  boss_hotkey: <ss58-hotkey>

evaluation_input_volume_name: vidaio-competition-compression-2026-w29-inputs
evaluation_index_path: /validator-evaluation/index.json
output_volume_prefix: vidaio-competition-compression-2026-w29-output
max_parallel_contenders: 4
max_attempts_per_item: 2

scoring_factors:
  quality: 0.6
  cost_efficiency: 0.25
  length_coverage: 0.15
  runtime: 0.00
cost_floor_usd: 0.000001
score_precision: 8
```

Validation rules:

- All timestamps are timezone-aware UTC, ordered start < finalisation <= human review deadline < end, and the default production start is Thursday.
- Competition IDs match a conservative slug format and are unique.
- Initial `competition_type` is exactly `COMPRESSION` and `required_routes` is exactly `[/compress]`.
- Batch size is 1-5; the initial default is 5.
- `length_weight_exponent` is finite and in `(0, 10]`; omitted fields default
  to `1.0` to preserve the original logarithmic weighting.
- Video durations are positive and min <= max.
- Requested GPUs are a non-empty subset of the platform allowlist.
- Requested CPUs are positive and never exceed 32; the trusted launcher sets a hard CPU limit as well as a request.
- Per-batch inference and scoring timeouts are independent, positive durations
  and cannot exceed the platform safety ceiling.
- The built image's measured uncompressed size must not exceed the single 25 GB rejection threshold.
- The required codec is AV1, the VMAF threshold is in `[0, 100]`, the minimum compression ratio is at least `1.0`, sample counts are positive, and the scoring seed is fixed for deterministic frame selection.
- The warmup fixture exists, is readable, is excluded from scored evaluation IDs, and is suitable for proving dimension/timing preservation and real size reduction.
- Scoring factors are known keys, non-negative, and sum to 1. Absolute media
  score receives 60%, contender-relative cost efficiency receives 25%, and
  length-weighted completion coverage receives 15%; runtime is recorded but has
  no direct score weight. The cost floor must be positive.
- `boss.repository_path` and `boss.boss_hotkey` must either both be null (no
  boss contender) or both be set. The repository path is relative to and
  contained beneath the validator repository root. It points to a complete,
  sanitized repository produced by `miner/competition_sdk.py` export, so the
  directory contains `.vidaio-sdk-export`, `competition_solution.json`,
  `requirements.txt`, `miner/modal_workers.py`,
  `scripts/competition_modal_build.py`, and the warmup fixture. Absolute paths,
  `..`, symlink/path escapes, incomplete exports, and paths outside the validator
  checkout are rejected. At submission finalisation, the configured export is
  snapshotted and checksummed as contender `<boss_hotkey>-boss`, then included
  in the private pre-evaluation artifact archive under that folder name. Later
  filesystem changes cannot change the boss evaluated for that competition.
- The immutable source-input Volume and evaluation index already exist and pass preflight checks before enrollment begins. The scorer reads the same source bytes as the contender; there is no separate hidden reference Volume for compression.

## 6. Miner protocols and configuration

Add a separate competition namespace rather than overloading `TaskWarrantProtocol`:

- `CompetitionType`: initially `COMPRESSION` only.
- `CompetitionInvitationProtocol`: validator sends competition ID, type, manifest digest, registration deadline, and protocol version; miner returns `participating`, supported type, and optional refusal reason.
- `CompetitionSubmissionProtocol`: validator polls a participating miner and includes the persisted status, reason, pinned commit, and revision number from its previous intake attempt; the miner returns competition ID, submission status, private GitHub HTTPS URL, and a short-lived read-only credential once ready.

Both protocols must include enough data to reject stale, replayed, or cross-competition responses. The validator verifies the response hotkey against the queried axon and records the UID only as a point-in-time chain attribute; hotkey plus competition ID is the durable contender identity.

Changes:

- `vidaio_subnet_core/protocol.py`: enums, request/response models, strict URL/status validation, and protocol versioning.
- `vidaio_subnet_core/base/miner.py`: attach abstract forward, blacklist, and priority handlers for invitation and submission polling.
- `neurons/miner.py`: add `MINER_MODES=inference,competition`, `MINER_COMPETITION_TYPES=COMPRESSION`, repository URL, credential source, and competition handlers. Keep all existing inference handlers attached when inference mode is enabled.
- `miner/.env.template` and `miner/README.md`: document dual-mode configuration and private-repository credential requirements.

Credential handling requirements:

- `CompetitionSubmissionProtocol` returns the raw GitHub PAT as explicitly selected for the initial release. Require a fine-grained, repository-scoped, contents-read-only PAT with an expiry; do not support GitHub App tokens or deploy keys in this version.
- Never log, serialize to SQLite/Redis/W&B, or place the token in a clone URL or Git configuration.
- Pass it to Git via an ephemeral askpass environment, remove the repository origin after clone, and erase the in-memory credential reference immediately.
- Redact common token formats from subprocess output and exception messages.
- Document that the product has accepted the risk of returning a raw PAT through the miner protocol. Exercise the actual Axon/Dendrite transport in security tests, minimize token lifetime, and never echo the synapse payload in debug logs, traces, metrics, or exception serialization.

## 7. Submission pinning and validation

Create a competition artifact root outside tracked source, configurable through `CompetitionConfig.artifact_root`:

```text
competition_artifacts/
  <competition_id>/
    manifest.normalized.json
    dataset_index.sha256
    contenders/
      <hotkey_slug>/
        source/                 # pinned, read-only clone
        submission.json         # no credential
        validation_report.json
        build_report.json
```

Clone workflow:

1. Accept only `https://github.com/<owner>/<repo>.git` private repository URLs for the first release.
2. Clone into a newly created directory with symlink/path-escape defenses and a maximum repository transfer size.
3. Resolve and persist HEAD commit SHA, tree SHA, committer timestamp (`%cI`), repository URL hash, and clone time.
4. Remove credentials and the origin remote, disable hooks, and make the pinned tree read-only.
5. While the competition is `ENROLLING`, continue polling contenders with `ACCEPTED`, `REJECTED`, or `REVIEW_REQUIRED` submissions. A new repository or HEAD commit is cloned and validated in a temporary sibling, then atomically replaces the prior artifact and increments `submission_revision`; the old artifact is deleted only after SQLite commits the new metadata. A database failure restores the prior artifact.
6. Never pull or re-clone for that contender after finalisation; retries use the same pinned tree and image digest.
7. On entry to `FINALIZING_SUBMISSIONS`, archive the normalized manifest and
   every DB-selected pinned contender directory under
   `competition_artifacts/<competition_id>/`. Require a private destination
   bucket and upload the archive and inventory without object ACL overrides
   beneath `competition_artifacts/<competition_id>/submission-snapshots/<manifest-digest-prefix>/`.
   This preserves compatibility with providers such as Backblaze B2, where
   objects inherit bucket visibility. The archive filename also contains the
   competition ID. Verify object sizes and reject any public grant in the bucket
   or inherited object ACL before recording `COMPLETED` in SQLite. A Backblaze
   application key therefore needs `readBuckets`, `readFiles`, and `writeFiles`.
8. Output and persist only the private
   `s3://<bucket>/<prefix>/<competition_id>/.../` path. Do not create, log, or
   return public or presigned HTTP URLs. A failed/partial upload records a safe
   reason and blocks the transition to `VALIDATING`; retrying overwrites the
   same deterministic private keys.

Automated validation should produce reason-coded results:

- Required files exist, including `miner/modal_workers.py`, the compression app, lock/requirements file, and the competition solution descriptor.
- `/compress` contract is declared and passes schema tests with local input and output paths.
- No broken symlinks, path escapes, Git submodules, LFS pointer placeholders, nested repositories, or files exceeding configured limits.
- Python parses successfully; Docker/build descriptors use an approved initial format.
- Reject committed secrets and obvious credential material.
- Flag compiled-only payloads, encrypted archives, dynamic download/`eval` patterns, minified/generated source, and a low readable-source ratio.

"Readable/non-obfuscated" cannot be proven perfectly with static analysis. During submission collection, automated checks reject clear unsafe forms, including minified source, embedded encoded loaders, known obfuscator runtimes, NUL-bearing source, and opaque executable/code extensions such as `.pyc`, `.pyd`, `.so`, `.pyz`, `.wasm`, and their supported-platform equivalents. A failure persists the contender as `REJECTED`; a clean repository is persisted as `ACCEPTED`. The row retains a safe `reason_code` and `reason_detail`, and the next validator poll returns that feedback to the miner so it can publish a correction before finalisation. Other ambiguous eligibility findings may still mark a contender `REVIEW_REQUIRED`; an operator resolves those during the review window, and the decision identity and reason are retained in the competition event log.

## 8. Miner template and route contract

Keep `miner/` as a working dual-mode reference implementation. Add a documented competition profile without removing its current inference deployment path.

The competition route request must use explicit volume paths:

```json
{
  "competition_id": "compression-2026-w29",
  "hotkey": "<ss58>",
  "batch_id": "batch-000042",
  "items": [
    {
      "evaluation_id": "video-0042",
      "input_path": "/evaluation-inputs/video-0042.mp4",
      "output_path": "/output/video-0042.mp4",
      "codec": "AV1",
      "vmaf_threshold": 90.0
    }
  ]
}
```

The response returns one result per evaluation ID, preserving order but keyed explicitly:

```json
{
  "batch_id": "batch-000042",
  "results": [
    {
      "evaluation_id": "video-0042",
      "success": true,
      "output_path": "/output/video-0042.mp4",
      "processing_started_at": "...",
      "processing_finished_at": "...",
      "runtime_seconds": 31.25,
      "error": null
    }
  ]
}
```

Required service changes:

- Add caller-supplied `output_paths` alongside `video_paths` in `miner/compression/app.py`, with equal-length validation.
- In competition mode, accept only normalized paths below `/evaluation-inputs` for input and `/output` for output; reject URLs, traversal, symlinks, overwrites, and duplicate output paths.
- Require compression outputs to be smaller than their source and preserve source dimensions, frame count/duration tolerance, AV1 encoding profile, YUV pixel format, MP4 container, and sample aspect ratio before accepting an output for scoring. The scorer separately applies the requested VMAF target through the locked absolute media-score curve.
- Set `DISABLE_REMOTE_IO=true`; do not inject S3 credentials or any miner/validator secret.
- Do not clean up evaluation inputs. Only create outputs in the assigned contender volume.
- Preserve existing URL input/output behavior when the service runs in inference mode.
- Add `/health` and route-contract tests that require no internet access.

Upscaling competition support is out of scope for the initial release. Do not change the existing upscaling inference route contract solely for this work; add an upscaling competition later through the competition-type registry without changing the compression lifecycle or persistence model.

### Preliminary qualification and warmup

Before a contender becomes eligible for the real evaluation dataset, the trusted launcher must:

1. Start the pinned image with the same offline, CPU/GPU, and Volume restrictions used in evaluation.
2. Confirm that `/health` and `/compress` are exposed on localhost.
3. Invoke `/compress` with `warmup_input_path`, an explicit local output path, AV1, and the manifest's VMAF target.
4. Verify a non-empty output at the requested path, unchanged dimensions, valid frame count/duration, required AV1/MP4 encoding, a smaller file size, and no writes outside the assigned output mount. The stock warmup proves mechanics only; its VMAF result is not scored.
5. Reject the contender with a reason code if any check fails. The warmup never contributes to score or cost ranking.

The repository includes a small redistribution-cleared synthetic clip at `competitions/fixtures/compression_warmup_input.mp4`. It is a five-second, 720p-or-lower MP4 suitable for a quick compression check. `competition_preflight.py` fails before enrollment if it is absent or invalid, and the miner SDK uploads it as `/evaluation-inputs/compression_warmup_input.mp4` before mounting the input Volume read-only. The warmup remains reserved and unscored.

## 9. Trusted Modal execution

Add a validator-owned module, proposed as `vidaio_subnet_core/competition/modal_runner.py`.

For every contender it will:

1. Build or resolve the pinned contender image without importing contender Python into the validator.
2. Record the immutable image ID/digest and measured image size.
3. Create a uniquely named app/sandbox such as `vidaio-cmp-<competition>-<hotkey-prefix>` with full IDs in tags and structured logs.
4. Select one GPU from the manifest allowlist; reject any contender attempt to control this value.
5. Set `cpu=(requested_cpu_cores, min(requested_cpu_cores, 32))` so 32 is a hard limit, not merely a reservation.
6. Mount only the immutable source-video Volume at `/evaluation-inputs` read-only. The scorer later mounts this same Volume as its reference; no separate reference Volume exists for compression.
7. Mount exactly one contender-specific Volume at `/output` read-write. Never mount another contender's volume or the parent namespace.
8. Start with `block_network=True`, no secrets, no identity token, and a trusted readiness probe.
9. Enforce both the batch call timeout and a slightly larger supervisor timeout; explicitly terminate a failed or expired sandbox.

The route can be hosted on localhost inside the sandbox and invoked through `sandbox.exec`, avoiding a public endpoint. If a Modal connect token/tunnel is used, it must be validator-only, rotated with the sandbox, and tested while outbound traffic remains blocked.

To reduce cold starts, maintain one warm sandbox per actively evaluated contender and submit the next batch immediately after the prior batch returns. Modal Sandboxes have a maximum lifetime of 24 hours, so the supervisor must checkpoint progress and replace a sandbox before expiry. The replacement uses the same pinned image and volumes; cold-start time is measured separately and excluded from item runtime.

Build each pinned contender revision once and persist its immutable Modal Image
ID; batch dispatch must never invoke an image build. For the reference FFmpeg
compression layout, keep the expensive native toolchain/requirements in a stable cached
base and add the customized Python service plus trusted preflight as small final
image layers. Python-only revisions therefore reuse the native compilation,
while Dockerfile, native dependency, model, or requirements changes invalidate
the base. Sandbox rollover reuses the recorded image ID rather than rebuilding.

### Build-size enforcement spike

Official Modal documentation exposes image building and image IDs, but the current public material does not establish a dependable API for enforcing an arbitrary 25 GB image-size limit during a hostile build. Before implementing contender builds, complete a time-boxed spike that proves one of these fail-closed paths:

1. Build an OCI image in an isolated, quota-limited builder, inspect its uncompressed size, push by digest to a validator registry, and use the pinned digest from Modal; or
2. Obtain a supported Modal image-size/build quota API and verify it against intentionally oversized test images.

The builder may have network access but receives no secrets beyond a narrowly scoped registry upload credential. It must have wall-clock, CPU, memory, disk, and download/bandwidth budgets. Runtime execution remains fully offline. Do not launch production competitions until oversized builds are demonstrably stopped before unbounded spend.

## 10. Dataset handling

Add a validator CLI, proposed as `scripts/competition_dataset.py`, with `prepare`, `validate`, `upload`, and `seal` commands.

The evaluation index contains an immutable evaluation ID, source relative path, byte size and checksum, duration/frame metadata, codec metadata, and scoring parameters for every video. Each original source is both the contender input and the trusted VMAF reference. It lives once in the read-only input Modal Volume; contenders can read it but cannot replace it. Preflight will:

- reject videos outside `min_video_length` and `max_video_length`;
- reject missing, duplicate, unreadable, or checksum-mismatched source files;
- record source dimensions, frame count, duration, pixel format, and display aspect ratio for later output validation;
- upload sources to the manifest-named read-only input Volume;
- write the normalized index and checksum;
- remount/read back a sample and verify input read-only enforcement;
- mark the dataset sealed in SQLite before enrollment begins.

Evaluation jobs reference IDs and normalized paths from this index only. Maintainers can replace data between competitions by creating a new competition ID/volume; a running competition's data is immutable.

## 11. Dispatch, Redis, and scoring

Create dedicated processes:

- `services/competition/dispatcher.py`: maintains warm contender sandboxes, leases batches, invokes `/compress`, validates output files, and enqueues per-item scoring work.
- `services/competition/scoring_worker.py`: reads the original source and contender output outside the untrusted sandbox, applies deterministic compression validations, computes VMAF and compression effectiveness, and writes the raw metrics needed for cost- and length-aware finalization transactionally.
- Optional `services/competition/reconciler.py`: requeues expired leases, finalizes aggregates, and reconciles delayed Modal billing data.

Suggested Redis keys:

```text
competition:<id>:dispatch                 # stream
competition:<id>:score                    # stream
competition:<id>:lease:<hotkey>:<batch>   # expiring ownership
competition:<id>:response:<hotkey>:<evaluation_id>
competition:<id>:heartbeat:<worker>
```

Use `competition_id + hotkey + evaluation_id + attempt` as the idempotency key. A transaction must claim pending SQLite items before execution. A duplicate result may update observability timestamps but must not create a second score.

Scheduling policy:

- One sequential batch stream per contender keeps that contender warm.
- Different contenders run concurrently, bounded by `max_parallel_contenders` and a validator-wide cost circuit breaker.
- As soon as a batch response is persisted, dispatch the next batch for that contender; scoring happens independently.
- Timeouts, missing files, invalid outputs, sandbox crashes, storage/scoring
  errors, and expired leases become terminal reason-coded item failures.
  Automatic evaluation replay is disabled; a confirmed validator-infrastructure
  incident requires the explicit allowlisted operator repair workflow.
- Every contender is attempted on every indexed evaluation input. After retries are exhausted, a failed item receives zero for media score, cost efficiency, and completion; failures are not excluded and do not trigger whole-contender disqualification.

Reuse the current synthetic compression validation pipeline: the AV1/MP4 and output-size checks, unchanged-resolution validation, frame-count and duration tolerances, and VMAF quality gate. Wrap it in a competition-specific adapter so inference history is not mutated. Persist deterministic VMAF sample selections, source/output checksums, and the pinned scoring algorithm version and configuration in the normalized manifest.

### Competition score formula

For reference, current inference scoring computes a content-length component as `log(1 + duration_seconds) / log(321)`, combines quality and length 50/50, and applies an exponential final transform. For competition mode, length should instead make longer evaluation entries more influential and make failure on a long video more costly.

For evaluation item `i`, define:

```text
length_weight_i = (
    log(1 + duration_i)
    / log(1 + manifest.max_video_length_seconds)
) ^ manifest.length_weight_exponent

compression_ratio_i = source_size_i / output_size_i

compression_rate_i = output_size_i / source_size_i

media_score_i, compression_component_i, vmaf_component_i, reason_i =
    services.scoring.scoring_function.calculate_compression_score(
        vmaf_score_i,
        compression_rate_i,
        item.vmaf_threshold,
    )

effective_cost_i =
    max(reconciled_or_estimated_cost_i, manifest.cost_floor_usd)

minimum_valid_cost_i =
    min(effective_cost_i across contenders with a valid scored result for item i)

cost_efficiency_i =
    minimum_valid_cost_i / effective_cost_i
    or 0 for an invalid/failed result

completed_i = 1 for a valid scored result, otherwise 0
```

The absolute media curve is shared directly with inference scoring and is locked by `scoring_version`: 70% compression and 30% VMAF, a five-point VMAF soft zone, a compression component reaching 1 at 20x, a logarithmic bonus above 20x, and a combined score normalized and capped at 1. A zero media score makes the item failed and zeroes every component.

Use reconciled per-item cost when available by the finalization cutoff;
otherwise use the locked estimated cost. Apply the configured cost floor before
selecting the minimum and persist which source was used. Every cheapest valid
result for an item receives cost efficiency 1.0; a valid result costing twice
that minimum receives 0.5. Failed and invalid outputs receive zero and are
excluded from the minimum, so they cannot lower valid contenders' scores.

Compute three length-weighted aggregates over the complete evaluation index:

The example exponent `2.02` makes a 30-minute item contribute approximately 10
times the weight of a 10-second item. The exponent applies to the common item
weight used by media score, cost efficiency, and completion coverage; it does
not alter the top-level `scoring_factors` split.

```text
media_score_aggregate = sum(length_weight_i * media_score_i) / sum(length_weight_i)
cost_aggregate = sum(length_weight_i * cost_efficiency_i) / sum(length_weight_i)
length_coverage = sum(length_weight_i * completed_i) / sum(length_weight_i)

final_score =
    (0.60 * media_score_aggregate) +
    (0.25 * cost_aggregate) +
    (0.15 * length_coverage)
```

Media score is not normalized against another contender, but cost deliberately
is. Adding or removing a cheaper valid contender can change existing
contenders' cost components and final scores. The 60% media component combines
compression and perceptual quality on an absolute scale. The 25% cost component
rewards efficiency relative to the finalized field. The remaining 15% rewards
successful coverage weighted toward longer videos; length is not a free score
shared equally by every contender. All source/output sizes, VMAF values, media
subcomponents, costs, the finalized evaluated contender set, and raw duration
weights must be retained so the final score can be reproduced exactly.

## 12. Persistence model

Define the competition-only SQLAlchemy models in
`vidaio_subnet_core/competition/models.py`. Bootstrap SQLite through the
explicit, idempotent baseline in `vidaio_subnet_core/competition/migrations.py`
rather than relying only on `create_all`. Before production, squash development
schema changes into that baseline and test the complete resulting schema.

### `competitions`

- `competition_id` primary key, type, schema/scoring versions
- manifest JSON and digest
- lifecycle status and reason
- start, submission finalisation, and end timestamps
- input Volume name, index, and checksum (legacy databases may retain a reference-Volume alias pointing to the same input Volume)
- optional manifest-configured boss repository path, snapshotted tree checksum,
  boss hotkey, and human-review deadline
- winner hotkey/UID-at-finalisation and finalized timestamp
- created/updated timestamps and scheduler lease fields

### `contender_metadata`

Composite unique key: `(competition_id, hotkey)`.

- UID/coldkey snapshots, participation status, `is_boss`, and optional boss
  source-path/snapshot metadata
- repository URL hash/display-safe value; never the token
- pinned commit/tree SHA and latest commit timestamp
- validation/build status and reason codes
- image digest/size, Modal app/sandbox/volume identifiers
- item totals (pending/success/failure), aggregate final score, VMAF, compression ratio/effectiveness, quality aggregate, cost-efficiency aggregate, and length-weighted coverage
- estimated/reconciled cost, active runtime, cold-start runtime
- final rank and eligibility

The boss owner's submitted contender remains `<boss_hotkey>`, so both solutions
are evaluated independently. Finalisation ranks only the higher score for that
payout hotkey; the other row retains its score but is marked ineligible with
`LOWER_SCORING_SOLUTION_FOR_HOTKEY`. A boss win is therefore explicit in the
SQLite snapshot as `hotkey=<boss_hotkey>-boss, is_boss=1, final_rank=1`, while a
submitted-solution win is `hotkey=<boss_hotkey>, is_boss=0, final_rank=1`.

### `contender_performance_history`

One row per evaluation data point and attempt, not merely per batch:

- competition ID, hotkey, evaluation ID, batch ID, attempt, idempotency key
- input/output checksums and sizes
- requested codec, duration/length weight, source/output sizes, compression ratio, VMAF threshold/score, compression-effectiveness component, raw cost, cost-efficiency component, completion value, and item status
- handler runtime excluding cold start, queue time, cold-start attribution
- estimated cost, reconciled cost, currency, and cost attribution method
- Modal app/sandbox/input identifiers
- status, reason-coded error, timestamps, and scoring version

### Supporting tables

- `competition_evaluation_items`: immutable dataset rows and current dispatch/score state.
- `competition_batches`: batch lifecycle, sandbox invocation, timeout, aggregate resource use, and reconciliation state.
- `competition_human_reviews`: operator identity, contender(s), eligibility decision or tied-hotkey ordering, reason, creation time, superseded state, and an integrity hash. Reviews never contain PATs.
- `competition_events`: modifiable lifecycle/audit events with redacted payload hashes.
- `competition_eligibility_checks`: competition/hotkey, threshold, observed alpha stake, UID, metagraph block/hash, check stage, timestamp, outcome, and stable reason code.
- `competition_audit_artifacts`: expected S3 key, checksum, size, content type, upload/read-back state, retry state, and inclusion in the sealed audit inventory.

Indexes are required on competition/status, contender/status, evaluation/status, lease expiry, and `(competition_id, hotkey, evaluation_id)`.

## 13. Runtime and cost accounting

Measure the complete validator-observed batch invocation with a monotonic clock.
Because the Sandbox is provisioned for that entire interval, persist batch
`active_runtime_seconds` equal to `wall_runtime_seconds`, regardless of whether
individual outputs pass validation or scoring.

Exact per-input cloud cost is not inherently observable when five inputs share one concurrently billed GPU container. Implement two fields and make the distinction explicit:

- `estimated_cost_usd`: probe the GPU model/count and CPU allocation actually present in the running Sandbox, apply the locked public Sandbox rates to the complete measured batch wall time, then allocate equally to every attempted item. Never price from the manifest's requested GPU or CPU values: Modal may place a request on upgraded hardware. Failed items retain their shares so total batch and contender estimates do not discard consumed compute; automatic evaluation replay is disabled.
- `reconciled_cost_usd`: populate later from Modal billing reports when the account/API supports the necessary granularity, then allocate with the same documented method.

The locked 2026-07-21 public GPU rates per second are B300 `$0.001972`, B200 `$0.001736`, H200 `$0.001261`, H100 `$0.001097`, RTX PRO 6000 `$0.000842`, A100 80 GB `$0.000694`, A100 40 GB `$0.000583`, L40S `$0.000542`, A10 `$0.000306`, L4 `$0.000222`, and T4 `$0.000164`. Modal's separate Sandbox CPU rate is `$0.00003942` per physical core per second. Source: [Modal pricing](https://modal.com/pricing).

Modal's current billing documentation describes programmatic reports on Team/Enterprise plans, delayed availability, app/resource breakdown, and app tags—not guaranteed per-input billing. Modal also documents that some automatic GPU upgrades retain the requested SKU's invoice rate. Competition estimates intentionally use observed hardware for consistent contender cost scoring; `reconciled_cost_usd` remains the invoice-aligned field. Therefore the implementation must not label an allocated estimate as an exact item invoice. See [Modal billing reports and attribution](https://modal.com/docs/guide/billing) and [GPU allocation behavior](https://modal.com/docs/guide/gpu).

Tag apps/sandboxes with competition ID and hotkey, and include batch/evaluation IDs in structured logs. This makes every run searchable in Modal without relying on a shared `vidaio-miner-workers` name.

## 14. Final ranking and tie-breakers

When evaluation finishes early and the competition enters `AWAITING_END_TIME`, generate a provisional ranking and a review packet containing score components, failures, cost source, static readability findings, pinned commit metadata, and exact tie groups. Add `scripts/competition_review.py` with `list`, `set-eligibility`, and `set-tiebreak` commands. Every decision records an operator identity, requires a reason, and is append-only/superseding rather than silently edited.

A human tie-break applies only to the exact tied hotkeys named in the review and only if recorded before the review deadline. It cannot change metrics or reorder non-tied contenders. Ambiguous readability findings can be resolved eligible/ineligible through the same review flow. If no decision is recorded, finalization remains deterministic.

At `competition_end_time`, inside one transaction:

1. Require every evaluation item to be terminal, including zero-scored failures, or mark the competition failed for manual resolution.
2. Apply the latest recorded eligibility decisions and recompute all aggregates from immutable history rows; do not trust cached summary columns.
3. Rank by normalized aggregate final score at manifest `score_precision`.
4. For an exact tie, apply a valid human tie-break order recorded during `AWAITING_END_TIME`.
5. If no human tie-break exists, compare the pinned repository's latest commit **committer timestamp**; the earlier timestamp wins.
6. If still tied, compare the pinned commit SHA lexicographically to make the result deterministic.
7. Persist ranks, winner, exact ranking inputs, review references, and a finalization event before exposing the winner to weight calculation.

Commit dates are author-controlled and can be backdated. The requested rule is implemented, but commit time should not be treated as a strong anti-cheating signal.

## 15. Weight integration

Refactor `MinerManager.weights` into a policy composer with explicit inputs:

- current inference compression/upscaling allocations;
- active competition state;
- most recently finalized eligible compression competition podium;
- burn policy.

Once at least one compression competition has finalized, use this top-level emissions split:

- 60%: compression inference pool, retaining its existing internal ranking policy.
- 20%: upscaling inference pool, retaining its existing internal ranking policy.
- 20%: the latest eligible compression competition podium, split 70% to the winner, 20% to the runner-up, and 10% to third place.

The competition podium keeps the 20% allocation until a later competition finalizes and atomically replaces it. During a challenger competition, the previous podium therefore remains incumbent. Competition data and aggregates come only from the separate competition SQLite tables; do not copy competition results into `miner_metadata` or `miner_performance_history`.

Before the first competition podium exists, or if its winner is no longer registered/eligible, fall back to the existing 80% compression inference / 20% upscaling inference split so no weight is burned or stranded. At all times—including active evaluation—burn weight is exactly zero. Add tests proving total normalization, no duplicate UID entries when a podium miner also participates in inference, the 60/20/20 split, the 70/20/10 podium ladder, the 80/20 fallback, boss continuity, and atomic podium replacement.

If the competition winner also earns an inference allocation under the same UID, sum its pool contributions by UID before Bittensor weight conversion and normalization; never emit duplicate UID rows.

### Manifest-configured boss repository

Support many immutable competition IDs and historical manifests, but allow only
one competition to be actively enrolling/building/evaluating subnet-wide in the
initial implementation. A new manifest may set `boss.repository_path` to a
directory in the validator checkout that was created by
`miner/competition_sdk.py` export. The directory is a complete contender
repository root—not a model/checkpoint path—and therefore includes
`miner/modal_workers.py`, the compression service, descriptor, requirements,
build worker, and warmup fixture.

When a boss path is configured, the manager associates the export with the
manifest's non-null `boss_hotkey`, applies the normal registration and eligibility
checks to that hotkey, copies the export into the competition's immutable
artifacts, records its tree checksum, and adds it as `is_boss=true`. From that
point it follows the normal contender pipeline: static validation, trusted build,
warmup, and fresh evaluation on the new benchmark. It does not reuse old scores,
import a prior competition's source, or require a GitHub PAT. If the boss wins,
its configured hotkey retains the 20% allocation. If a challenger wins,
finalization atomically promotes the challenger; the operator can place that
winner's next SDK-shaped export in the repository path selected by a later
manifest. Null path and hotkey values omit the boss from that competition while
the incumbent still retains its allocation until finalization.

## 16. Process and documentation changes

- Add competition dispatcher, scoring worker, and reconciler to `run.sh` with independent PM2 names and restart policies.
- Add health endpoints or heartbeat keys so the validator does not enter evaluation without live workers.
- Add graceful shutdown that releases leases and detaches/terminates sandboxes as appropriate.
- Extend `docs/validator_setup.md` with Modal SDK installation, `modal setup`, environment selection, billing budget/circuit-breaker configuration, Volume preparation, registry/build setup, required account tier for billing reports, and a preflight command.
- Extend `miner/README.md` with dual-mode configuration, repository template layout, self-tests, credential safety, local-path route schemas, resource limits, and rejection reasons.
- Document the required warmup clip at `competitions/fixtures/compression_warmup_input.mp4`, its licensing/format constraints, and the preflight failure shown when it is missing.
- Add an operator runbook for cancelling a competition, rejecting a contender, rotating credentials, rebuilding only from a pinned revision, recovering Redis, resuming after validator restart, and cleaning up Modal resources/Volumes.

## 17. Proposed code layout

```text
competitions/
  manifests/
  schemas/manifest.schema.json
vidaio_subnet_core/competition/
  config.py
  models.py
  manager.py
  repository.py
  validation.py
  build.py
  modal_runner.py
  dataset.py
  scoring.py
  weights.py
  eligibility.py
  artifact_publisher.py
services/competition/
  dispatcher.py
  scoring_worker.py
  reconciler.py
scripts/
  competition_dataset.py
  competition_preflight.py
  competition_review.py
  competition_audit.py
tests/competition/
  ...
```

`CompetitionManager` should be injected into `Validator` and should depend on narrow adapters for chain queries, Git, Redis, SQLite, Modal, and time. This permits deterministic unit tests without a live subnet, GitHub, Redis, or Modal account.

## 18. Delivery phases and exit criteria

### Phase 0: feasibility and security gates

- Prove that hostile builds are stopped at the approved 25 GB image threshold and that runtime Modal execution is offline.
- Exercise raw-PAT submission through the actual protocol and prove redaction across application, Bittensor, PM2, Git, Redis, SQLite, W&B, and exception logs.
- Prove available billing report granularity and document item cost allocation.
- Validate the 60/25/15 absolute-media/relative-cost/length-coverage formula against representative short and long evaluation mixes.

Exit: all security/cost gates have reproducible tests; no production build proceeds on assumptions.

Implementation status (2026-07-14): the local gates, scoring/allocation contracts, test suite, and bounded live-Modal probe are implemented. The local suite passes. The live dev probe also passes offline network isolation, credential absence, cleanup, and hourly billing access. Phase 0 remains blocked for production because an end-to-end trusted builder enforcing the 25 GB limit has not yet been proven, and the PAT canary must be repeated through the production submission path once Phase 1 creates it. See `docs/competition_phase0_feasibility_report.md`; its blockers are mandatory rather than deferred assumptions.

### Phase 1: contracts, configuration, and persistence

- Add manifest model/loader and competition configuration.
- Add protocols and dual-mode miner handlers.
- Add state machine, SQL models, an explicit schema baseline, and event log.
- Add Thursday scheduling and restart/resume behavior behind `COMPETITION_MODE_ENABLED=false`.

Exit: unit tests can advance a fake-clock competition through enrollment and resume every state after process recreation without touching inference.

Implementation status (2026-07-16): complete. Versioned compression manifests,
invitation/submission protocols, dual-mode miner handlers, restart-safe validator
invitation and submission polling, competition-only SQLite models, an explicit
idempotent schema baseline, redacted events, database leases, Thursday
scheduling, audited non-terminal manifest revisions, and restart/resume behavior
are implemented behind `COMPETITION_MODE_ENABLED=false`. Enrollment attempts and poll
cadence are persisted so a validator restart neither loses opted-in miners nor
polls them on every scheduler heartbeat. The focused Phase 0/1 suite advances a
fake-clock competition through all lifecycle states while recreating the manager
at each boundary and verifies that inference tables are never created or accessed
by the competition repository. The pre-production migration history has been
squashed into schema version 1; it creates the final current tables, constraints,
and indexes without the retired event update/delete guards.

### Phase 2: miner template and repository intake

- Implement explicit local input/output route contracts.
- Update the working `miner/` template and self-tests.
- Implement secure polling, clone, pinning, credential redaction, and static validation.
- Add the documented stock warmup fixture and enforce `/health` plus local-path `/compress` qualification before evaluation.

Exit: a template private repository is accepted and pinned; malicious/path-escape/oversized/obfuscated fixtures are rejected with stable reason codes.

Implementation status (revised 2026-07-20): implemented. The miner template now declares and serves a backward-compatible, local-only competition route; validates caller-owned input/output paths and AV1 MP4 media; and includes a synthetic five-second warmup fixture plus a shared miner/validator batch-of-one preflight. A miner SDK builds a credential-free standalone export and qualifies that exact export in a disposable Modal Sandbox before publication. It places `compression_warmup_input.mp4` in a read-only `/evaluation-inputs` Volume, mounts a separate read-write `/output` Volume, blocks runtime networking, attaches no secrets or identity token, enforces SDK-selected GPU/CPU/timeout controls, invokes the service only on localhost, and tears down the Sandbox and Volumes. Modal login is mandatory for validation and any publication without a matching receipt. A successful validation produces a checksum-, resource-, and age-bound local receipt so `publish` can reuse that exact result for 24 hours by default. After qualification, the SDK creates or verifies a private GitHub repository and pushes through ephemeral askpass without persisting the PAT. Commits use the authenticated GitHub account's no-reply identity, and `--use-existing --refresh` updates an existing private repository with a fast-forward child commit rather than force-pushing. Validator-side intake validates poll bindings, clones through ephemeral askpass, removes the Git remote, captures immutable commit/tree/timestamp metadata, writes credential-free artifacts, makes source read-only, and statically rejects unsafe or obfuscated repositories with stable reason codes. SQLite stores each result and safe reason detail; protocol v3 returns it on the next poll. Accepted, rejected, and review-required contenders may replace their repository or HEAD commit until finalisation through a rollback-safe artifact swap with a revision event. Ambiguous non-obfuscation findings enter `REVIEW_REQUIRED`. Validator-owned live build coordination, sandbox supervision, and restart recovery are delivered in the Phase 3 record below. The production quota-enforcing builder remains an external deployment gate; the miner SDK is the executable reference contract for it.

### Phase 3 delivery record: trusted build and Modal isolation

- Implement the validated build path and size enforcement selected in Phase 0.
- Implement validator-owned Modal sandbox creation, naming/tags, GPU/CPU caps, offline runtime, read-only input and isolated output volumes.
- Add timeout, teardown, warm reuse, 24-hour rollover, and restart recovery.

Exit: integration tests prove no outbound DNS/IP access, no write to evaluation data, no access to another contender's output, hard CPU/GPU selection, build-size rejection, and forced termination at timeout.

Implementation status (2026-07-16): validator-side implementation complete.
Database schema version 5 includes restart-safe enrollment attempt
metadata alongside immutable build evidence and sandbox lifecycle generations.
The trusted build service enforces source binding, allowlisted
builder identity, during-build quota evidence, immutable digest/ID, and the exact
25 GB boundary. Direct Modal builds additionally run in isolated client
processes and contender-specific Apps under the manifest's
`modal_build_timeout` deadline, which defaults to 600 seconds when omitted. A
deadline breach force-kills the client, requests remote App termination, and is
persisted as terminal `BUILD_TIMEOUT`, allowing other parallel builds and the
scheduler to continue. The Modal supervisor owns all resources and mounts,
performs an active DNS/IP/HTTPS, credential, reference-visibility, and read/write
isolation probe, persists external IDs before qualification, reattaches after restart,
reuses warm sandboxes, rolls them before 24 hours, and force-terminates failures.
The live validator now connects these components: accepted pinned repositories
are built on Modal in parallel, persisted with immutable image bindings,
advanced to `EVALUATING`, and launched through the sandbox supervisor in
parallel. Both concurrency stages obey `max_parallel_contenders`. Testnet and
production use the same `modal` backend and new builds are labelled
`MODAL_ACCEPTED`. Because Modal does not provide final-image size attestation,
activation requires the explicit
`COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true` acknowledgement;
the persisted evidence remains labelled `MODAL_UNATTESTED`. Focused Phase 1–3
tests pass.

### Phase 4: evaluation, scoring, and accounting

- Add dataset CLI/index sealing.
- Add dispatcher, Redis leases/streams, scoring worker, and reconciliation.
- Reuse `services/scoring/scoring_function.py` directly for absolute media
  scoring and implement per-item contender-relative cost efficiency and
  length-weighted completion coverage with 60/25/15 final weights.
- Add per-item runtime, estimated cost, and billing reconciliation fields.

Exit: a multi-contender test competition processes all items exactly once despite worker/Redis/validator restarts, keeps contenders warm between batches, and reproduces aggregate scores from history.

Implementation status (2026-07-16): the validator now prepares, validates,
uploads, read-back verifies, and immutably seals a normalized evaluation index;
the sealing command can initialize a new competition database and register the
manifest as `SCHEDULED` before the validator's first boot, while retaining the
ability to seal a matching competition later during `EVALUATING`;
claims batches transactionally in SQLite; expires abandoned leases; rejects late
or duplicate results; keeps one sequential batch stream per warm contender while
running different contenders concurrently; commits Sandbox output Volumes before
read-back; and validates checksums, dimensions, duration, frame count, aspect
ratio, AV1/MP4 encoding, compression ratio, and deterministic VMAF. Validator-
measured batch wall time is allocated equally across items using locked public
Modal Sandbox rates and the GPU/CPU allocation probed from the running Sandbox,
so neither contender-reported timing nor a substituted Modal GPU can distort cost
scores. Requested and allocated resources are retained separately. Raw item
history reproduces the 60/25/15 media-score, cost-efficiency, and length-weighted
coverage aggregate before the lifecycle advances through `SCORING` to
`AWAITING_END_TIME`.

SQLite is the authoritative queue and lease store in this delivery. Redis is not
required for correctness and can be added later only as a reconstructible
dispatch accelerator; flushing or losing Redis therefore cannot lose Phase 4
work. Focused tests cover restart recovery, expired claims, stale delivery,
attempt exhaustion, exact-once scoring, media validation, aggregate reproduction,
Modal Volume path translation, and the end-to-end Phase 4 lifecycle transition.

### Phase 5: finalization and weights

- Implement provisional ranking, audited human eligibility/tie-break review, and commit-date/SHA fallback.
- Add `AWAITING_END_TIME` review behavior and winner publication.
- Integrate 60% compression inference / 20% upscaling inference / 20% latest competition podium, split that podium pool 70% / 20% / 10%, retain zero burn, and keep the pre-podium 80/20 fallback.
- Add boss-contender snapshotting from the manifest-configured SDK export
  directory onto the new benchmark.

Exit: tests cover early completion, ties, ineligible contenders, multiple UIDs/hotkey changes, weight normalization, zero burn, and unchanged legacy inference behavior.

Implementation status (completed 2026-07-24): implemented.

Implemented:

- The scheduler holds scored competitions in `AWAITING_END_TIME` and completes
  them only when `competition_end_time` is reached.
- Completion ranks eligible scored contenders transactionally, applies a
  recorded order only to a complete exact-score tie group, and otherwise falls
  back to the earlier pinned committer timestamp, commit SHA, and hotkey.
  Final ranks, the winner hotkey/UID snapshot, podium metadata, and the
  completion event are persisted together.
- `scripts/competition_review.py` can list review history, resolve the narrow
  readability hold during `VALIDATING`, disqualify a contender through
  `AWAITING_END_TIME`, and order an exact tie. Reviews require a non-empty
  operator identity and reason, carry an integrity hash, and support
  superseding tie orders. A scored disqualification rebuilds relative-cost
  components and aggregates from persisted history without rerunning media
  evaluation. Its `list` command exposes a database-derived provisional/final
  packet containing ranks, score components, failures, cost sources, static
  validation, pinned commits, exact tie groups, review queue, and review
  history. The packet shares the completion ranking implementation and does not
  persist `final_rank` while the competition is still awaiting its end time.
- A manifest-configured boss SDK export is validated, copied into the immutable
  private submission snapshot under a separate contender identity, and then
  follows the normal build/evaluation/ranking path. The better of a boss
  solution and a submitted solution sharing the same payout hotkey is the only
  one ranked for that hotkey. Before enrollment,
  `scripts/competition_preflight.py --validate-boss` reads the supplied
  manifest's boss path/hotkey, enforces path containment and complete SDK-export
  shape, and applies the same objective static validation policy used for miner
  submissions.
- The inference validator can read the competition SQLite database, select the
  latest completed podium atomically, resolve its hotkeys against the current
  metagraph, merge duplicate inference/competition UIDs, retain the 80/20
  pre-podium fallback, and apply the 60/20/20 and 70/20/10 allocations with
  zero burn. Focused tests cover final ranking, exact ties, disqualification
  recalculation, boss/submission deduplication, podium shares, UID merging,
  winner disappearance, current-metagraph hotkey/UID resolution, missing lower
  podium members, the pre-podium legacy split, zero burn, normalization, and
  atomic incumbent replacement. A validator permit is not a reward-exclusion
  criterion because the chain may assign one to a miner based on its alpha
  stake.

### Phase 6: operator rollout

- Update `run.sh`, setup docs, dashboards/alerts, cleanup, and runbooks.
- Run a dev manifest with seconds/minutes instead of days.
- Run a shadow competition with no weight effect.
- Exercise rollback and budget controls, but do not enable production until Phases 7 and 8 pass.

Exit: dev/shadow sign-off, budget alarms enabled, rollback exercised, and the validator is ready for the final audit and entry-eligibility gates.

### Phase 7: public audit and reproducibility publishing

- In competition mode, make `neurons/miner.py` accept invitation and submission requests only from the hotkey currently occupying subnet UID 0 by default. Resolve UID 0 from the locally synced metagraph and compare its hotkey with `synapse.dendrite.hotkey`; never trust a UID claimed in the payload. Keep inference-mode blacklist behavior unchanged. A development override may exist but must be explicit and unsafe for production.
- Add a validator-owned `CompetitionArtifactPublisher`; untrusted contender code never receives S3 credentials and never performs the audit upload.
- Make post-competition publication of the pinned source and generated outputs explicit in the invitation/manifest terms. A miner that opts in consents to this audit publication; no source bundle is published before the competition ends.
- Publish a credential-free, checksum-addressed reproduction bundle to a configured S3 bucket. Use this stable layout:

```text
competitions/<competition_id>/
  manifest.normalized.json
  manifest.sha256
  dataset/index.json
  inputs/<input_id>/input.mp4
  contenders/<contender_hotkey>/
    source/source.tar.zst
    source/submission.json
    validation/validation_report.json
    build/build_report.json
    evaluations/<input_id>/
      output.mp4
      result.json
      score.json
  ranking/provisional.json
  ranking/final.json
  accounting/costs.json
  reviews/reviews.json
  events/events.jsonl
  inventory.json
  COMPLETE
```

- `source.tar.zst` contains the pinned, credential-free contender solution; `output.mp4` is the miner-produced compressed result. The original input is also the scoring reference. That input, raw deterministic metrics, score components, failures, cost source, ranking inputs, human decisions, and recorded events must be sufficient to reproduce every score and final rank.
- Build `inventory.json` from object path, byte size, media metadata, and SHA-256 checksum. Upload `COMPLETE` only after every required object has been uploaded and read back successfully. Never publish PATs, clone environment, presigned URLs, validator/miner secrets, private Git metadata containing credentials, or unrelated S3 objects.
- Keep the audit prefix private while evaluation is active so benchmark inputs cannot leak. Publish or grant auditor access only after finalization and successful sealing.
- Add an independent `competition_audit verify` command that downloads a bundle, validates inventory/checksums, recomputes item and aggregate scores, and confirms the final ordering without reading the live validator database.
- Define bucket encryption, lifecycle/retention, access policy, retry, multipart-upload cleanup, and redaction rules. A failed audit upload keeps the competition result provisional and cannot silently publish an incomplete bundle.

Exit: tests prove non-UID-0 validators cannot call competition handlers, UID-0 hotkey rotation follows the synced metagraph, inference handlers are unchanged, PAT canaries never reach S3, interrupted uploads never create `COMPLETE`, and a clean machine reproduces all contender scores and the final ranking solely from the published bundle.

### Phase 8: alpha-stake competition entry criteria

- Increment the manifest schema version and add `minimum_alpha_stake`. It must be
  finite, non-negative, and denominated consistently with the subnet metagraph's
  normalized alpha-stake value. Any non-terminal update follows the same audited
  manifest-revision policy; eligibility already locked at submission
  finalisation is not retroactively rewritten.
- At invitation response and again at submission finalisation, resolve the contender hotkey to its current metagraph UID and read its alpha stake from the subnet metagraph. The miner may participate only when it is registered, the responding hotkey matches the queried axon, and its alpha stake is at least the manifest threshold.
- Persist the threshold, observed alpha stake, hotkey, UID snapshot, metagraph block number, block hash when available, check timestamp, and pass/fail reason in competition-only eligibility records and the event log. Do not copy these decisions into inference performance tables.
- Lock entry eligibility at submission finalisation. A stake change afterward does not rewrite the running competition's entrant set, but the miner is checked again for every later competition. Human review cannot override a failed objective stake threshold.
- Apply the same threshold to the hotkey associated with a configured boss by
  default. Any future boss exemption must be an explicit versioned manifest
  policy rather than an implicit special case.
- Expose clear refusal/rejection reason codes such as `NOT_REGISTERED`, `HOTKEY_MISMATCH`, `ALPHA_STAKE_UNAVAILABLE`, and `ALPHA_STAKE_BELOW_MINIMUM` without exposing other wallet-sensitive data.
- Enable the first production competition only after the audit bundle and alpha-stake gates both pass in a shadow run.

Exit: fake-metagraph and dev-subnet tests cover exact-threshold acceptance, below-threshold rejection, UID/hotkey movement, missing/malformed stake values, block-pinned evidence, restart determinism, finalisation recheck, boss handling, and proof that inference scoring and weights remain unchanged before final podium integration.

## 19. Test matrix

At minimum, add:

- Manifest boundary, timestamp, duration, 25 GB threshold, scoring-factor,
  boss-path containment/export-shape, warmup, allowlist, and immutable-digest
  tests.
- Scheduler tests for Thursday start, missed wake-up, duplicate loop, clock
  skew, restart, one-active-competition enforcement, and sequential new-ID boss
  competitions sourced from configured SDK exports.
- Protocol tests for opt-out, stale IDs, unsupported type, replay, malformed URLs, raw-PAT redaction, missing/expired credentials, and hotkey mismatch.
- Git tests for token redaction, hooks, submodules, LFS, symlinks, path traversal, branch mutation after finalisation, and commit timestamp capture.
- Static validation fixtures for readable template, known obfuscator runtimes/extensions, obfuscated-only code, archives, secrets, dynamic execution, and oversized repositories/images; intake tests cover rejection feedback, corrected/new-repository resubmission, atomic artifact cleanup/rollback, and the finalisation cutoff.
- Route tests for warmup qualification, missing `/compress`, batches 1-5, input/output length mismatch, traversal, symlink escape, URL rejection offline, partial result, duplicate output, and inference backward compatibility.
- Modal integration tests for network block, read-only source input, per-contender output isolation, GPU allowlist, CPU hard cap, timeout, unique tags/logs, warm reuse, and 24-hour replacement.
- Redis/SQLite tests for lease expiry, duplicate delivery, retry exhaustion, crash between output and enqueue, Redis loss, and exact resume.
- Scoring regression tests against existing compression fixtures,
  unchanged-resolution and encoding checks, output-size reduction, the VMAF
  soft/hard zones, absolute compression/VMAF media scoring, cheapest-valid
  per-item cost normalization, population-dependent recomputation,
  length-weighted completion, failures-as-zero, 60/25/15 factors,
  deterministic sampling, per-item records, aggregate reproduction, and
  deterministic ties.
- Human-review tests for recorded operator identity, mandatory reason, eligibility resolution, exact-tie scope, deadline enforcement, superseding decisions, and deterministic no-input fallback.
- Weight tests for zero burn, 60/20/20 allocation, 80/20 pre-winner/ineligible-winner fallback, total normalization, duplicate winner/inference UID merging, boss continuity, and atomic replacement.
- Audit tests for the UID-0-only competition blacklist, metagraph hotkey rotation, inference-handler isolation, deterministic S3 paths, multipart retry/cleanup, required artifacts, checksum inventory, PAT/secret canaries, `COMPLETE` atomicity, and score/ranking reproduction from a clean machine.
- Entry-criteria tests for exact/below minimum alpha stake, unavailable or non-finite stake, registration and hotkey mismatch, block-pinned snapshots, finalisation recheck, post-lock stake changes, boss eligibility, restart behavior, and stable reason codes.

## 20. Deferred extension points

- `CompetitionEntryFeePolicy` interface and tables for a future monetary entry fee; alpha-stake eligibility is no longer deferred.
- Competition-type registry so upscaling or future tasks can supply their own route schema, dataset validator, scorer, and resource policy without modifying the compression lifecycle engine. The inactive typed stub in `media_contracts.py` fixes upscaling's three physical roles as high-resolution ground truth, downsampled reference/contender input, and miner-processed output. Activation must keep ground truth unavailable to contender code while making it available to the trusted scorer; it requires a new manifest version and cannot reuse compression's two-file assumption.
- Cryptographic signatures, transparency logs, and multi-validator mirroring beyond the mandatory checksum-addressed S3 audit bundle.

## 21. Approved product decisions

1. The contender image rejection threshold is 25 GB, with no separate soft/hard limit.
2. After the first finalized podium, emissions are split 60% compression inference, 20% upscaling inference, and 20% latest compression competition podium. The competition pool is divided 70% / 20% / 10% across first, second, and third place. The podium remains incumbent until atomically replaced; before a podium exists, use the current 80/20 inference split. Burn remains zero.
3. Final competition scoring is 60% absolute media score, 25%
   contender-relative cost efficiency, and 15% length-weighted completion
   coverage. Longer videos have greater influence through logarithmic duration
   weights; runtime is recorded but is not a direct factor. For each evaluation
   item, the cheapest valid contender receives cost efficiency 1.0 and other
   valid contenders receive `minimum_valid_cost / own_cost`; failed and invalid
   outputs receive zero and do not define the minimum.
4. The miner returns a raw, read-only, repository-scoped GitHub PAT through `CompetitionSubmissionProtocol`; strict ephemeral handling and redaction are mandatory.
5. Clear readability/obfuscation violations are rejected automatically. Ambiguous cases and exact score ties can receive an audited, reasoned human decision during the review window; commit time and SHA remain deterministic fallbacks.
6. Every contender is evaluated across every input. Terminal item failures score zero and do not disqualify the whole contender. A separate unscored stock-video warmup must prove `/compress` and local path I/O before evaluation begins.
7. Competitions use unique immutable IDs. The initial scheduler permits one
   active competition at a time. A manifest can select a complete
   `competition_sdk.py` export directory in the validator checkout as the boss;
   the validator snapshots it and evaluates it against new contenders on the
   new dataset without importing source from a prior competition.
8. Competition protocol requests are accepted by miners only from the hotkey currently at subnet UID 0 by default. The validator publishes a credential-free S3 reproduction bundle containing the manifest, original dataset inputs, pinned contender sources, miner outputs, metrics, costs, reviews, events, and final ranking under stable competition/hotkey/input paths.
9. A miner must meet the manifest's minimum alpha-stake threshold at opt-in and submission finalisation. The block-pinned decision is persisted; human review cannot override a failed stake threshold, and eligibility is locked for that competition after finalisation.
