# Validator Competition Checklist

Use this checklist to create and operate every new compression competition.
Complete the validator host setup in
[`validator_setup.md`](validator_setup.md) first. The current contract and
timeout formulas are in [`competitions.md`](competitions.md), the design and
rollout record is in
[`competition_mode_implementation_plan.md`](competition_mode_implementation_plan.md),
and the sample manifest is
[`competitions/manifests/examples/compression-competition.json`](../competitions/manifests/examples/compression-competition.json).

Competition mode is independent from inference mode and is disabled by default.
Use a new `competition_id` whenever the schedule, sealed dataset, query contract,
or another immutable live-competition input changes.

## Competition details and sign-off owners

- [ ] Competition ID: `____________________________`
- [ ] Manifest path: `____________________________`
- [ ] Manifest SHA-256/digest: `____________________________`
- [ ] Dataset source directory: `____________________________`
- [ ] Dataset index path: `____________________________`
- [ ] Dataset index digest: `____________________________`
- [ ] Absolute competition database URL: `____________________________`
- [ ] Modal environment: `____________________________`
- [ ] Private artifact bucket/prefix: `____________________________`
- [ ] Technical owner: `____________________________`
- [ ] Security reviewer: `____________________________`
- [ ] Competition operator: `____________________________`

## 1. Establish the production gate

- [ ] Confirm `competition_type=COMPRESSION`. Do not configure an upscaling
  competition until its separate input/ground-truth contract is implemented.
- [ ] Decide whether this is a development, shadow, testnet, or
  reward-affecting production competition.
- [ ] Keep `COMPETITION_MODE_ENABLED=false` until every pre-enrollment item in
  this checklist passes.
- [ ] Confirm the validator-owned build and Sandbox path is the only path that
  executes contender code. Never import contender Python into the validator or
  run `modal deploy` against a contender repository.
- [ ] Record explicit approval for
  `COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true`. This
  acknowledges that direct Modal builds do not attest the manifest's 25 GB
  final-image limit.
- [ ] Confirm the production UID-0 authorization, PAT-redaction canary,
  alpha-stake eligibility, and post-competition public-audit gates required by
  local policy. The implementation plan lists these as later rollout phases;
  do not assume they exist merely because evaluation works.
- [ ] Set a competition spend budget and alert threshold covering image builds,
  Sandbox GPU/CPU time, Volumes, storage, and retries.
- [ ] Document stop, rollback, and incident owners.

## 2. Create the manifest

- [ ] Copy the sample into the scanned manifest directory:

  ```bash
  cp competitions/manifests/examples/compression-competition.json \
    competitions/manifests/compression-YYYY-wWW.json
  ```

- [ ] Assign a unique `competition_id`.
- [ ] Set `schema_version` to the version supported by the deployed validator.
- [ ] Assign a stable `scoring_version` identifier for the scoring contract used
  by this manifest; changing it does not itself change the scoring algorithm.
- [ ] Set `competition_start_time` to a Thursday UTC.
- [ ] Verify this ordering:
  `competition_start_time < contender_finalisation_time <= human_review_deadline < competition_end_time`.
- [ ] Leave enough enrollment time for polling, correction, and republishing.
- [ ] Confirm `required_routes`, `required_output_codec`, video-length limits,
  compression ratio, VMAF criteria, scoring seed, and scoring factors.
- [ ] Confirm scoring factors sum to exactly 1 and record the intended economic
  effect of each factor.
- [ ] Confirm the selected non-negative media, cost-efficiency, and coverage
  factors express the intended policy, sum to one, and retain 0% direct runtime.
- [ ] Confirm the deployed scoring implementation matches the contract denoted
  by `scoring_version`, including the absolute media curve and
  contender-relative per-item cost normalization.
- [ ] Confirm final aggregate scoring includes the complete finalized evaluated
  contender set; cost components cannot be finalized incrementally.
- [ ] Select only supported GPUs: `L4`, `L40S`, and/or `RTX-PRO-6000`.
- [ ] Set requested and maximum CPU cores, container-size policy,
  `modal_build_timeout`, and `max_parallel_contenders`. Treat
  `max_attempts_per_item` as a legacy compatibility field; evaluation batches
  are not retried automatically.
- [ ] Set `evaluation_batch_size`; the current intended batch size is five.
- [ ] Treat `evaluation_batched_run_timeout` as the full minimum miner deadline
  for every batch, including a final or explicitly operator-requeued batch smaller than
  `evaluation_batch_size`.
- [ ] Treat `scoring_batched_run_timeout` as the full minimum scoring deadline
  for every batch, including partial batches.
- [ ] Verify sandbox-forward failures, storage/scoring failures, and expired
  leases become terminal item failures and cannot produce an automatic second
  claim. Use the allowlisted, audited repair command only for a confirmed
  validator-infrastructure incident.
- [ ] Confirm dynamic deadlines may exceed those two minimums. Do not reduce or
  prorate either configured floor.
- [ ] Review the dynamic timeout examples in
  [`competitions.md`](competitions.md#batch-and-timeout-fields), including the
  two-minute minimum allocation for every video shorter than ten minutes.
- [ ] Choose unique, bounded `evaluation_input_volume_name` and
  `output_volume_prefix` values containing the new competition ID.
- [ ] Set `evaluation_index_path` and verify it matches the dataset tool.
- [ ] Set both `boss.repository_path` and `boss.boss_hotkey`, or set both to
  `null`. If used, verify the path is a sanitized SDK export contained beneath
  the validator repository root; the boss preflight below is mandatory.
- [ ] Review the normalized manifest diff and get operator approval before
  enrollment.

## 3. Configure the validator processes

- [ ] Use an absolute SQLite URL. The dataset CLI, competition process, and
  inference process must reference exactly the same database:

  ```env
  COMPETITION_DATABASE_URL=sqlite:////absolute/path/to/competition_database.db
  ```

- [ ] Configure the competition process:

  ```env
  COMPETITION_MODE_ENABLED=true
  COMPETITION_DATABASE_URL=sqlite:////absolute/path/to/competition_database.db
  COMPETITION_ARTIFACT_ROOT=competition_artifacts
  COMPETITION_MANIFEST_GLOB=competitions/manifests/*.json
  COMPETITION_SCHEDULER_INTERVAL_SECONDS=30
  COMPETITION_LEASE_TTL_SECONDS=120
  COMPETITION_NETWORK_TIMEOUT_SECONDS=30
  COMPETITION_MAX_CONCURRENT_REQUESTS=32
  COMPETITION_EXECUTION_ENABLED=true
  COMPETITION_BUILD_BACKEND=modal
  COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true
  COMPETITION_MODAL_ENVIRONMENT=main
  ```

- [ ] Configure a dedicated private competition-artifact bucket and credentials.
  Do not share the inference bucket if another service clears that bucket:

  ```env
  COMPETITION_ARTIFACT_BACKUP_BUCKET=replace-with-private-bucket
  COMPETITION_ARTIFACT_BACKUP_PREFIX=competition_artifacts
  COMPETITION_ARTIFACT_BACKUP_REGION=us-east-1
  COMPETITION_ARTIFACT_BACKUP_ENDPOINT_URL=
  COMPETITION_ARTIFACT_BACKUP_ACCESS_KEY_ID=
  COMPETITION_ARTIFACT_BACKUP_SECRET_ACCESS_KEY=
  ```

- [ ] Confirm the bucket is private, supports read-back verification, and no
  presigned URL is published.
- [ ] Ensure the inference process receives the same
  `COMPETITION_DATABASE_URL`; otherwise competition rewards are excluded from
  its weight calculation.
- [ ] Confirm `run.sh`/PM2 launches separate `--validator-mode inference` and
  `--validator-mode competition` processes.
- [ ] Restart both processes with `--update-env` only after the manifest,
  database, dataset, credentials, and preflight are ready.

## 4. Authenticate and probe Modal

- [ ] Authenticate the OS user that runs the competition process:

  ```bash
  modal setup
  ```

- [ ] For a headless PM2 process, provide `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET` in its environment.
- [ ] Confirm the token stays in the trusted host and is never attached to a
  contender Sandbox.
- [ ] Confirm the configured Modal environment exists.
- [ ] Run the bounded isolation probe before enabling live execution:

  ```bash
  python scripts/competition_modal_phase0_probe.py \
    --environment main \
    --skip-billing
  ```

- [ ] Confirm the probe proves blocked outbound networking, absent credentials,
  cleanup, and the expected GPU/CPU controls.
- [ ] Omit `--skip-billing` only when the workspace supports billing reports and
  that separate capability must be checked.
- [ ] Inspect Modal for orphaned Apps, Sandboxes, or Volumes from a failed
  probe.

## 5. Prepare and seal the private dataset

- [ ] Select private source MP4s within the manifest duration limits.
- [ ] Provide at least `evaluation_batch_size` distinct physical videos if a
  full batch is expected.
- [ ] Confirm each source becomes exactly one query. Do not create multiple
  CRF/VBR/VMAF variants from the same source.
- [ ] Create the manifest's input Volume in the selected environment:

  ```bash
  INPUT_VOLUME=vidaio-competition-compression-YYYY-wWW-inputs
  modal volume create --env main "$INPUT_VOLUME"
  ```

- [ ] Confirm `INPUT_VOLUME` exactly equals
  `evaluation_input_volume_name`.
- [ ] Do not pre-create contender output Volumes. The validator creates separate
  v2 output Volumes automatically.
- [ ] Define absolute working values:

  ```bash
  MANIFEST=/absolute/path/to/competitions/manifests/compression-YYYY-wWW.json
  SOURCE_DIR=/absolute/path/to/private-evaluation-videos
  INDEX=/tmp/compression-YYYY-wWW-index.json
  DATABASE_URL=sqlite:////absolute/path/to/competition_database.db
  ```

- [ ] Prepare the deterministic index:

  ```bash
  python scripts/competition_dataset.py prepare \
    --manifest "$MANIFEST" \
    --source-dir "$SOURCE_DIR" \
    --index "$INDEX"
  ```

- [ ] Inspect the index and confirm:

  - [ ] every row references a different source video;
  - [ ] the batchable source count is sufficient;
  - [ ] CRF/VBR modes and VMAF/bitrate targets are distributed as expected;
  - [ ] dimensions, duration, frame count, codec, pixel format, aspect ratio,
    file size, and SHA-256 metadata are present.

- [ ] Validate all local source files:

  ```bash
  python scripts/competition_dataset.py validate \
    --manifest "$MANIFEST" \
    --source-dir "$SOURCE_DIR" \
    --index "$INDEX"
  ```

- [ ] Upload and read-back verify the index and every source:

  ```bash
  python scripts/competition_dataset.py upload \
    --manifest "$MANIFEST" \
    --source-dir "$SOURCE_DIR" \
    --index "$INDEX" \
    --environment main
  ```

- [ ] Seal the exact digest into the competition database:

  ```bash
  python scripts/competition_dataset.py seal \
    --manifest "$MANIFEST" \
    --index "$INDEX" \
    --environment main \
    --database-url "$DATABASE_URL"
  ```

- [ ] Record the printed index checksum and confirm the database manifest digest
  matches the file the scheduler will scan.
- [ ] Treat the sealed index as immutable. To change the index, dataset, or API
  contract, create a new competition ID and require miners to publish a matching
  SDK export.

## 6. Run preflight and enrollment readiness

- [ ] Run the manifest/warmup preflight:

  ```bash
  python scripts/competition_preflight.py \
    competitions/manifests/compression-YYYY-wWW.json
  ```

- [ ] Confirm status `ACCEPTED`, the manifest digest, and valid warmup metadata.
- [ ] If the manifest configures a boss, validate the exact manifest-selected
  export before enrollment:

  ```bash
  python scripts/competition_preflight.py \
    competitions/manifests/compression-YYYY-wWW.json \
    --validate-boss
  ```

- [ ] Confirm the boss result is `ACCEPTED` and records the expected
  `boss_hotkey`, manifest-relative `repository_path`, resolved path, tree
  checksum, file count, and byte count. `REVIEW_REQUIRED` and `REJECTED` are
  non-zero preflight failures.
- [ ] Confirm the stock five-second warmup fixture exists and is not part of the
  scored dataset.
- [ ] Confirm the database and artifact root are backed up before enrollment.
- [ ] Confirm only one intended active competition is in scope.
- [ ] Confirm serving miner axons and the metagraph snapshot can be collected.
- [ ] Confirm the scheduler clock is UTC-synchronized.
- [ ] Confirm operator dashboards/log collection redact raw PATs and submission
  payloads.
- [ ] Start/restart both PM2 validator processes with their final environment.

## 7. Monitor enrollment and submission finalisation

- [ ] Confirm the transition to `ENROLLING` occurs at the intended time.
- [ ] Confirm invitations and immediate submission polls are sent to the
  snapshotted serving miners.
- [ ] Confirm `NOT_READY` miners are repolled at `contender_ping_interval`.
- [ ] Track each contender as `INVITED`, `ACCEPTED`, `REJECTED`,
  `REVIEW_REQUIRED`, or withdrawn/not ready.
- [ ] Verify every received repository URL is private GitHub HTTPS, every PAT is
  handled ephemerally, and no PAT appears in SQLite, artifacts, logs, Git
  remotes, traces, Redis, W&B, exceptions, or backups.
- [ ] Confirm each accepted submission has immutable commit SHA, tree SHA,
  committer timestamp, URL hash, safe display value, revision, and static
  validation result.
- [ ] Confirm corrected submissions atomically replace prior artifacts only
  while enrollment remains open.
- [ ] Resolve every `REVIEW_REQUIRED` row with a signed human decision before
  the review deadline.
- [ ] At `contender_finalisation_time`, confirm the state advances through
  `FINALIZING_SUBMISSIONS` and accepted revisions stop changing.
- [ ] Confirm the private contender archive and `inventory.json` are complete,
  read-back verified, and contain no credentials before build/evaluation.

## 8. Monitor build and isolated execution

- [ ] In `VALIDATING`, confirm every finalized repository is checked against the
  static policy and warmup contract.
- [ ] In `BUILDING`, confirm builds use the pinned revision, contender-specific
  Apps, manifest build timeout, validator-selected resources, and bounded
  concurrency.
- [ ] Investigate every `BUILD_TIMEOUT` and verify best-effort App cancellation
  did not leave an active Modal App.
- [ ] Confirm every accepted image ID is immutable and associated with the
  correct contender revision.
- [ ] In `EVALUATING`, confirm each contender Sandbox has no secrets/OIDC,
  blocked egress, read-only input, separate read-write v2 output, no public
  endpoint, and localhost-only route access.
- [ ] Confirm every full batch contains `evaluation_batch_size` distinct source
  videos; partial retry/final batches may be smaller but must still contain no
  duplicated source.
- [ ] Confirm sampled contenders have identical first-attempt membership for
  each persisted canonical batch index, and retries never include an evaluation
  from a later canonical batch. Treat this as a cost-scoring fairness
  requirement because equal-share per-item runtime and cost depend on the video
  length/workload composition of the batch.
- [ ] For sampled batches, independently calculate and compare:

  - [ ] miner invocation deadline = maximum of the dynamic four-lane estimate
    and the full `evaluation_batched_run_timeout` floor;
  - [ ] execution lease = invocation deadline plus the validator recovery and
    persistence allowance;
  - [ ] scoring deadline = maximum of sequential 200-FPS-at-4K VMAF time plus
    scoring overhead and the full `scoring_batched_run_timeout` floor.

- [ ] Confirm every sub-ten-minute video receives at least two minutes of miner
  processing allowance.
- [ ] Confirm `competition_sandboxes.batch_timeout_seconds` records the latest
  invocation deadline, while each `competition_batches.timeout_seconds` records
  its attempt-specific execution lease.
- [ ] Confirm `competition_batches.scoring_timeout_seconds` records the effective
  scoring deadline and may be lower than the miner invocation deadline when
  200-FPS VMAF is faster.
- [ ] Confirm each requested output parent is created by the validator and each
  result matches its assigned path and position exactly.
- [ ] Confirm completed contenders terminate promptly with
  `volume_retained=true`; do not delete their Volumes automatically.

## 9. Validate scoring, ranking, and completion

- [ ] Confirm outputs are independently checked for AV1/MP4, unchanged
  dimensions/timing, YUV pixel format, square pixels, expected path, size, and
  checksum before scoring.
- [ ] Confirm invalid/missing outputs fail closed and retain measurable metrics
  where safe.
- [ ] Confirm VMAF is computed against the matching original source and the
  query-specific target feeds the absolute compression/VMAF media-score curve.
- [ ] Confirm runtime and cost use validator-measured batch wall time and
  observed GPU/CPU allocation, not contender claims.
- [ ] Confirm each history row receives
  `batch_wall_runtime / attempted_item_count` and
  `batch_estimated_cost / attempted_item_count`; acknowledge these equal-share
  values depend on canonical batch composition rather than individual measured
  video runtime.
- [ ] Confirm retries preserve prior attempts and increase attempt numbers.
- [ ] Confirm the transition sequence reaches `SCORING`, then
  `AWAITING_END_TIME`, only after terminal results exist for every eligible
  contender/item.
- [ ] Generate and retain the read-only provisional review packet:

  ```bash
  python scripts/competition_review.py \
    --database-url "$DATABASE_URL" \
    list --competition-id compression-YYYY-wWW
  ```

- [ ] Confirm the packet includes every provisional rank, score component,
  failure, cost source, static-validation result, pinned commit, exact tie
  group, pending review, and prior review record without assigning persisted
  `final_rank`.
- [ ] Review aggregate factor scores, score precision, tie-break sequence,
  duplicate-hotkey boss handling, and audited human tie-breaks.
- [ ] Before weight integration, independently verify the podium, winner
  hotkey, score ordering, and configured competition/inference allocation.
- [ ] Keep the previous completed podium active until the new competition is
  truly `COMPLETED`.
- [ ] At `competition_end_time`, confirm the integrity-checked SQLite snapshot
  uploads privately and its size/SHA-256 read-back verification is recorded.
- [ ] Record the final state, ranking, score digest, database snapshot path, and
  audit event IDs.

## 10. Recovery and incident handling

- [ ] If infrastructure fails before dispatch, leave the competition fail-closed
  in `EVALUATING`; do not convert all contenders to zero.
- [ ] Before an audited requeue, stop the competition validator process so it
  cannot race the repair.
- [ ] Requeue only allowlisted validator-infrastructure failures:

  ```bash
  python scripts/competition_repair.py requeue-infrastructure \
    --competition-id compression-YYYY-wWW \
    --database-url "$DATABASE_URL" \
    --reason-code SANDBOX_START_FAILED \
    --apply
  ```

- [ ] Restart the validator and confirm old attempts remain `REQUEUED`, new
  attempt numbers increase, and an `EVALUATION_INFRASTRUCTURE_REQUEUED` event is
  present.
- [ ] Never automatically delete a non-empty output Volume. If an empty
  mistakenly created v1 Volume blocks execution, stop all users of it before
  explicitly replacing it with v2 as documented in
  [`validator_setup.md`](validator_setup.md#modal-authentication-for-competition-execution).
- [ ] On manifest/database artifact divergence, treat SQLite as authoritative
  and verify the scheduler's preserved revision and divergence event.
- [ ] Escalate any credential leak, unexpected network access, unpinned code,
  digest mismatch, cross-contender Volume access, unexplained cost spike, or
  unauthorized competition request immediately.

## 11. Closeout

- [ ] Winner hotkey/contender ID: `____________________________`
- [ ] Final score/rank digest: `____________________________`
- [ ] Final database backup URI and SHA-256: `____________________________`
- [ ] Submission archive URI and inventory digest: `____________________________`
- [ ] Modal resources reviewed/retained/deleted by: `____________________________`
- [ ] Credentials rotated/revoked at (UTC): `____________________________`
- [ ] Incident and exception record: `____________________________`
- [ ] Technical owner sign-off: `____________________________`
- [ ] Security reviewer sign-off: `____________________________`
- [ ] Competition operator sign-off: `____________________________`
