# Compression competitions

This document is the primary technical and operational guide to compression
competitions. It consolidates the durable decisions from
[`competition_mode_implementation_plan.md`](competition_mode_implementation_plan.md)
and describes the behavior implemented by the current codebase.

Competition mode runs alongside synthetic and organic inference. Its protocols,
state machine, database tables, artifacts, execution, scoring, and weights are
kept separate from inference so a competition failure cannot corrupt inference
history or stop inference work.

## Competition objective

A validator runs a scheduled benchmark against private miner submissions:

1. Open enrollment and invite eligible miners.
2. Poll participating miners for private GitHub repositories.
3. Pin, validate, and privately archive the final repository revisions.
4. Build each accepted revision once and run it in validator-owned, isolated
   Modal compute.
5. Dispatch the immutable private video dataset in batches, validate every
   output, and score it outside the untrusted sandbox.
6. Persist raw attempts, costs, metrics, failures, reviews, and rankings so work
   survives restarts and results can be reproduced.
7. Finalize the ranking at the scheduled end time and feed the latest eligible
   podium into the subnet weight policy.

The initial and currently supported competition type is `COMPRESSION`.

## Lifecycle

Competition state is authoritative in SQLite:

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

Any pre-completion state may enter FAILED or CANCELLED.
```

- `SCHEDULED`: the manifest and competition record exist, but enrollment has
  not started.
- `ENROLLING`: invitations are sent and participating miners are polled at
  `contender_ping_interval`.
- `FINALIZING_SUBMISSIONS`: repository revisions are frozen and their private
  backup must complete before validation continues.
- `VALIDATING`: pinned source is checked for the required layout, readable
  source, route contract, unsafe constructs, and credential leakage.
- `BUILDING`: the validator builds or resolves an immutable contender image.
- `EVALUATING`: accepted contenders receive sequential batches in fresh,
  batch-scoped sandboxes. Different contenders run concurrently.
- `SCORING`: raw item histories are converted into reproducible aggregates.
- `AWAITING_END_TIME`: scored aggregates are available in SQLite until the
  configured end time. Review operators may disqualify a contender or order an
  exact rounded-score tie before `human_review_deadline`; score-affecting
  disqualifications rebuild the aggregates. The review CLI exposes a
  provisional ranking packet without assigning persisted `final_rank` before
  completion.
- `COMPLETED`: final ranks, winner, and reward recipients are immutable.

The scheduler uses short restart-safe ticks rather than a week-long sleep.
Database leases and optimistic transitions prevent two schedulers from
advancing the same competition concurrently. The initial policy permits only
one active competition at a time.

Competition invitation and submission handlers on miners authorize the
requesting dendrite hotkey against the current on-chain subnet-owner hotkey.
They reject every other caller by default; the development override is unsafe
for production. This restriction is competition-only and does not change the
existing inference blacklist behavior.

If a serving miner does not expose a competition invitation or submission
route, the validator records `CONTENDER_NOT_FOUND` with the missing route in
the detail. This applies whether transport reports an explicit HTTP 404 or
returns an HTTP 200 synapse with untouched protocol response defaults; neither
case is treated as a response for a different competition.

## Manifest

The executable reference is the
[`compression-competition.json`](../competitions/manifests/examples/compression-competition.json)
sample manifest. Manifests are normalized and digest-bound in SQLite. A
non-terminal revision is audited; terminal competition manifests are immutable.
Every new benchmark must use a new `competition_id`.

### Schedule

| Field | Meaning |
| --- | --- |
| `competition_id` | Unique conservative slug, normally including year/week. |
| `competition_start_time` | UTC enrollment start. Any day of the week is valid. |
| `contender_ping_interval` | Poll cadence for participating miners. |
| `minimum_alpha_stake` | Minimum current metagraph alpha stake required at invitation and again when submissions are finalised; defaults to `0`. A miner must remain at or above it through the end of enrollment. |
| `contender_finalisation_time` | Submission freeze and private backup boundary. |
| `human_review_deadline` | Last time an eligibility or exact-tie review may be recorded. |
| `competition_end_time` | Final ranking and winner publication boundary. |

The required ordering is:

```text
competition_start_time
  < contender_finalisation_time
  <= human_review_deadline
  < competition_end_time
```

A miner below `minimum_alpha_stake` remains in the invitation queue and is
retried at `contender_ping_interval` until enrollment ends. If its stake reaches
the threshold in time, it can opt in and submit normally. The finalisation check
still rejects any pinned submission that is below the threshold at the freeze.
Likewise, a miner that declines because competition mode or the requested
competition type is unavailable is reinvited at `contender_ping_interval` while
enrollment remains open, allowing it to restart with corrected configuration.

### Execution resources and safety

| Field | Meaning |
| --- | --- |
| `allowed_gpus` | Validator allowlist for an optional miner `SANDBOX_GPU` preference. The first entry is the fallback. |
| `requested_cpu_cores` | CPU request and hard limit fallback when a miner omits `SANDBOX_CPUS`. |
| `max_cpu_cores` | Manifest ceiling, currently no greater than 32. |
| `container_size_limit_gb` | Single image rejection threshold, currently 25 GB. |
| `modal_build_timeout` | Deadline for the validator-owned image build process. |
| `max_parallel_contenders` | Maximum contenders built or evaluated concurrently. |
| `max_attempts_per_item` | Legacy compatibility field. Automatic evaluation retries are disabled; only an explicit audited operator repair may create another attempt. |

During a refreshed SDK export, optional `SANDBOX_GPU` and `SANDBOX_CPUS` values
from `miner/.env` are copied as non-secret `sandbox_resources` metadata in
`competition_solution.json`; the `.env` file itself remains excluded. When
`--manifest` is supplied, the SDK and shared preflight read `allowed_gpus` and
`max_cpu_cores` from that exact manifest rather than using a built-in SKU list
or CPU ceiling. Intake independently rejects a GPU outside `allowed_gpus` or
CPUs above `max_cpu_cores`. The trusted launcher applies accepted preferences
and otherwise uses `allowed_gpus[0]` and `requested_cpu_cores`. Modal may still
allocate different hardware, so requested and observed resources are persisted
separately.

### Batch and timeout fields

The sample manifest contains:

```json
{
  "evaluation_batch_size": 5,
  "evaluation_batched_run_timeout": "10m",
  "scoring_batched_run_timeout": "5m"
}
```

- `evaluation_batch_size` is the maximum number of videos sent in one miner
  request. The supported range is 1–5. Sealing assigns every evaluation ID an
  immutable zero-based canonical batch index and position using this value.
- `canonical_batch_index` identifies that immutable competition-wide group.
  It is persisted on `competition_evaluation_items`, `competition_batches`,
  and `contender_performance_history`, so operators can compare membership
  directly without inferring it from dispatch identifiers.
- `batch_id` is a contender-specific dispatch execution ID. Different
  contenders therefore have different `batch_id` values for the same
  `canonical_batch_index`, while all history rows from one dispatch share both
  values.
- `evaluation_batched_run_timeout` is the **minimum miner invocation deadline**
  for every batch.
- `scoring_batched_run_timeout` is the **minimum validator scoring-service
  deadline** for every batch. It is not miner runtime.
- A timeout is a maximum allowed wall-clock deadline, not a required runtime.
  Miners and the scorer may return earlier.
- The full configured minimum applies to final or explicitly operator-requeued
  batches smaller than `evaluation_batch_size`; it is never prorated.
- Workload calculations may increase either deadline but can never reduce the
  manifest minimum.

The effective deadlines are:

```text
miner_timeout =
  max(dynamic_encoding_timeout, evaluation_batched_run_timeout)

scoring_timeout =
  max(dynamic_vmaf_timeout, scoring_batched_run_timeout)
```

### Dynamic miner timeout

The model supports the allowed L4, L40S, and RTX PRO 6000 choices. A miner may
request one allowed GPU, but should inspect the observed allocation because
Modal can upgrade the requested hardware. The validator grants time from sealed
video metadata.

Constants:

```text
reference resolution       = 3840 × 2160 (4K)
chunk duration             = 10 minutes
4K processing per chunk    = 2 minutes
parallel videos            = 4
per-video processing floor = 2 minutes
I/O/startup allowance      = 2 minutes per batch
database recovery grace    = 2 minutes per batch
```

For video `i`:

```text
resolution_scale_i =
  width_i × height_i / (3840 × 2160)

chunks_i =
  ceil(duration_seconds_i / 600)

video_work_i =
  max(
    120,
    chunks_i × 120 × resolution_scale_i
  )
```

Every started chunk receives its full allowance, including videos shorter than
10 minutes and final partial chunks. The absolute 120-second floor means a short
720p or 1080p video does not receive less processing time after resolution
scaling.

Video workloads are scheduled longest-first across four parallel video lanes.
The largest lane load is the estimated encoding makespan:

```text
dynamic_encoding_timeout =
  four_lane_makespan(video_work) + 120

effective_miner_timeout =
  max(dynamic_encoding_timeout, manifest minimum)

execution_lease =
  effective_miner_timeout + 120
```

The additional lease grace is validator recovery time. It does not extend the
deadline passed to miner code.

### Dynamic scoring timeout

The scoring service currently evaluates returned items sequentially. VMAF is
modeled at 200 FPS for 4K, with pixel-proportional throughput:

```text
vmaf_work_i =
  frame_count_i / 200 × resolution_scale_i

dynamic_vmaf_timeout =
  sum(vmaf_work_i) + 120

effective_scoring_timeout =
  max(dynamic_vmaf_timeout, scoring_batched_run_timeout)
```

The 120-second allowance covers Volume reads, media probing, and scoring process
overhead. Only outputs eligible for scoring contribute to the scoring batch.

### Timeout examples

Using the sample manifest's 10-minute miner minimum and 5-minute scorer minimum:

**Five short videos**

Each video receives the two-minute floor. Four run on the first wave and the
fifth runs on the second:

```text
encoding makespan       = 4 minutes
dynamic miner timeout   = 6 minutes
effective miner timeout = max(6m, 10m) = 10 minutes
execution lease         = 12 minutes
```

A final partial batch containing one short video still receives the full
10-minute miner deadline, not one fifth of it.

**Five one-hour 4K videos at 30 FPS**

```text
per-video encoding work = 6 chunks × 2m = 12 minutes
encoding makespan       = 24 minutes
dynamic miner timeout   = 26 minutes
effective miner timeout = 26 minutes
execution lease         = 28 minutes

frames per video        = 108,000
VMAF work per video     = 108,000 / 200 = 9 minutes
sequential VMAF work    = 45 minutes
dynamic scoring timeout = 47 minutes
effective scoring       = 47 minutes
```

VMAF is faster per video in this example, but the current scoring endpoint runs
the five items sequentially while encoding is modeled with four-way
parallelism.

### Timeout persistence

| Location | Current relevance |
| --- | --- |
| `competition_sandboxes.batch_timeout_seconds` | Miner invocation deadline for that batch-scoped sandbox generation. It is audit/diagnostic metadata; enforcement occurs through the backend invocation argument. |
| `competition_batches.timeout_seconds` | Execution lease duration for that batch attempt, including recovery grace. |
| `competition_batches.lease_expires_at` | Authoritative absolute execution lease expiry used for recovery. |
| `competition_batches.scoring_timeout_seconds` | Effective scoring deadline for the batch. |
| `competition_batches.scoring_expires_at` | Authoritative absolute scoring expiry used for recovery. |

A contender has a fresh sandbox generation for every batch, while batch history
belongs primarily in `competition_batches`. Any future removal or rename of the sandbox timeout
column requires an explicit schema migration.

### Media and scoring fields

| Field | Meaning |
| --- | --- |
| `min_video_length`, `max_video_length` | Allowed source duration range. |
| `length_weight_exponent` | Positive exponent applied to the normalized logarithmic duration weight. Defaults to `1.0` for manifests that omit it; `2.02` makes a 30-minute video weigh approximately 10 times a 10-second video. |
| `required_output_codec` | Currently `AV1`. |
| `vmaf_threshold` | Quality target used by the absolute media-score curve. Results more than five points below it receive zero; the five-point soft zone receives partial media score. |
| `vmaf_sample_count` | Deterministic scoring sample configuration. |
| `minimum_compression_ratio` | Minimum valid source/output size ratio. |
| `scoring_seed` | Deterministic dataset query and scoring selections. |
| `scoring_version` | Stable scoring-contract identifier. Version `4` adds trusted VBR bitrate enforcement. |
| `scoring_factors` | Manifest-selected weights for absolute media score (legacy key `quality`), contender-relative cost efficiency, and completion coverage. `runtime` is reserved and must remain zero. |
| `cost_floor_usd` | Positive floor used in cost-efficiency division. |
| `score_precision` | Normalization precision used for ranking and exact ties. |

The default scoring split is 60% absolute media score, 25% contender-relative
cost efficiency, and 15% length-weighted completion coverage. A manifest may
select a different non-negative split as long as the weights sum to one.
The manifest and persistence schema retain `quality`/`quality_aggregate` as
legacy compatibility names for media score. Runtime is measured and costed but
has no defined direct score component, so its manifest weight must remain zero.

### Dataset and Volume fields

| Field | Meaning |
| --- | --- |
| `evaluation_input_volume_name` | Immutable source-video Volume mounted read-only to contenders. |
| `evaluation_index_path` | Normalized sealed index inside the input Volume. |
| `output_volume_prefix` | Prefix for contender-specific writable output Volumes. |
| `warmup_input_path` | Trusted, unscored route qualification clip. |

For compression, the immutable source video is also the trusted VMAF reference;
there is no separate reference Volume.

### Optional boss contender

`boss.repository_path` and `boss.boss_hotkey` must both be null or both be set.
The path must name a complete, sanitized `competition_sdk.py` export contained
inside the validator checkout. At finalisation it is snapshotted and checksummed
as a distinct boss contender, then follows the same validation, build, warmup,
evaluation, scoring, and objective eligibility rules as submissions.

Before enrollment, run the manifest preflight with `--validate-boss`. The flag
requires a configured boss, resolves the path from the supplied manifest,
enforces checkout containment and complete SDK-export shape, and applies the
same objective static source validation used for miner submissions. Any
`REJECTED` or `REVIEW_REQUIRED` result makes the command exit non-zero.

The boss owner's submitted solution and boss solution are scored independently.
Only the higher-scoring solution for the payout hotkey remains rank-eligible.

## Miner participation and repository intake

Competition invitation and submission protocols are separate from inference
protocols. Durable contender identity is `(competition_id, hotkey)`; UID and
coldkey are point-in-time chain snapshots.

The validator synchronizes its metagraph, including `alpha_stake` / `AS`, every
`METAGRAPH_REFRESH_INTERVAL_SECONDS` (1,800 seconds / 30 minutes by default).
When the competition enters `FINALIZING_SUBMISSIONS`, the validator checks
every submitted miner using the latest metagraph snapshot recorded before the
transition, rejects ineligible submissions, and then locks the contender set.

The initial repository credential contract uses a raw, fine-grained,
repository-scoped, contents-read-only GitHub PAT:

- It must be short-lived and is returned only through the submission protocol.
- It must never enter SQLite, Redis, W&B, logs, clone URLs, Git configuration,
  reports, or published artifacts.
- Git receives it through ephemeral askpass.
- The validator removes the Git remote after cloning and redacts token-shaped
  values from errors.
- The pinned commit SHA, tree SHA, committer timestamp, safe repository display,
  and repository URL hash are persisted.

During enrollment, an accepted, rejected, or review-required contender may
submit a new repository or HEAD revision. The new revision is cloned and
validated in a temporary sibling and atomically replaces the prior artifact only
after persistence succeeds. No submission changes are accepted after
finalisation.

Within one competition, each non-boss GitHub account and canonical repository
may belong to only one miner hotkey. Owner and repository matching is
case-insensitive and stored as hashes rather than raw identity fields. A second
hotkey attempting to reuse either identity is rejected with
`GITHUB_ACCOUNT_ALREADY_SUBMITTED` or `GITHUB_REPOSITORY_ALREADY_SUBMITTED`.
The same hotkey may refresh its own submission during `ENROLLING`; boss rows are
excluded from this miner-identity constraint.

The final pinned source snapshots and inventory are uploaded to a private S3
prefix before validation/building proceeds. Partial or publicly accessible
backups fail closed and remain retryable from
`FINALIZING_SUBMISSIONS`.

Before upload, the backup service rejects public bucket ACL grants, wildcard
bucket-policy statements that allow object reads, and—when the provider exposes
AWS's Public Access Block API—any bucket that does not enable all four controls.
Uploaded objects are read back and their ACLs, sizes, and checksums are checked.
On S3-compatible providers that do not expose bucket-policy or Public Access
Block APIs, the service logs an explicit warning; operators must independently
verify equivalent provider policy. `database_archive_path` is non-secret member
metadata inside the archive and inventory, and may appear in validator logs. It
does not expose the SQLite bytes or create a separately addressable S3 object.

## Miner route contract

The competition route is local-filesystem-only and runs offline. A request
contains one to five items:

```json
{
  "competition_id": "compression-2026-w29",
  "hotkey": "<ss58-hotkey>",
  "batch_id": "batch-example",
  "items": [
    {
      "evaluation_id": "input-00001-vbr-vmaf89-8mbps",
      "input_path": "/evaluation-inputs/inputs/example.mp4",
      "output_path": "/output/evaluations/input-00001.mp4",
      "codec": "AV1",
      "codec_mode": "VBR",
      "target_bitrate": 8000000,
      "vmaf_threshold": 89.0
    }
  ]
}
```

The response preserves item order and returns the produced local path:

```json
{
  "results": [
    {
      "output_path": "/output/evaluations/input-00001.mp4"
    }
  ]
}
```

Inputs must remain under `/evaluation-inputs`; outputs must remain under
`/output`. URLs, traversal, symlink escapes, duplicate outputs, remote I/O, and
writes outside the assigned output Volume are rejected.

Before accessing the private benchmark, every contender runs the stock,
unscored warmup through the same local route and isolation policy.

## Trusted build and sandbox boundary

Untrusted contender Python is never imported into the validator and never
controls Modal declarations. The validator-owned launcher:

- binds the image to the pinned Git tree and immutable image ID/digest;
- enforces or records the 25 GB image-size boundary;
- applies the validated, pinned miner GPU/CPU preferences or manifest fallbacks;
- starts with blocked networking, no secrets or identity token, and no public
  ports;
- mounts only the active canonical batch subdirectory of the immutable source
  Volume read-only and a fresh batch-specific subdirectory of that contender's
  output Volume read-write;
- invokes the service over localhost through `sandbox.exec`;
- records the actual GPU model/count and CPU allocation from a trusted probe;
- terminates a sandbox after a failed invocation or supervisor timeout;
- records restart-safe identities and never reuses a generation for another
  batch.

Every new or recovered generation must prove:

- outbound DNS, direct-IP, and HTTPS access fail;
- the input mount rejects writes;
- the output mount accepts writes;
- the sealed global `index.json` is absent from the batch-scoped input mount;
- no reference-only path or another contender Volume is visible;
- GitHub, cloud, Modal, and validator credentials are absent.

Modal does not currently provide the strict final-image size attestation
required by the strongest builder contract. Production Modal activation is
therefore gated by the explicit
`COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true`
acknowledgement and persists `MODAL_UNATTESTED` evidence.

## Dataset preparation and sealing

The dataset CLI is `scripts/competition_dataset.py`:

The four commands are mandatory and must run in the shown order. Each command
requires the receipt produced by its predecessor.

```bash
python scripts/competition_dataset.py prepare \
  --manifest competitions/manifests/examples/compression-competition.json \
  --source-dir /absolute/path/to/private-videos \
  --index /tmp/competition-index.json

python scripts/competition_dataset.py validate \
  --manifest competitions/manifests/examples/compression-competition.json \
  --source-dir /absolute/path/to/private-videos \
  --index /tmp/competition-index.json

python scripts/competition_dataset.py upload \
  --manifest competitions/manifests/examples/compression-competition.json \
  --source-dir /absolute/path/to/private-videos \
  --index /tmp/competition-index.json \
  --environment main

python scripts/competition_dataset.py seal \
  --manifest competitions/manifests/examples/compression-competition.json \
  --index /tmp/competition-index.json \
  --environment main \
  --database-url sqlite:////absolute/path/to/competition.db
```

`prepare` probes every MP4 and records immutable size, checksum, duration,
dimensions, frame count, codec, pixel format, and aspect ratio. Each physical
video becomes one compression query. CRF/VBR mode, VMAF target, and any VBR
bitrate are selected deterministically from the manifest seed. A full batch
contains distinct source paths. Missing or unknown media metadata, invalid
aspect ratios, ffprobe-reported media errors, non-MP4 containers, multiple
video streams,
manifest duration violations, and index-to-media metadata mismatches are flagged.
The CLI asks the operator to type `yes` or `no` before removing every evaluation
variant that references a flagged source and continuing. `--yes` performs this
exclusion non-interactively. `prepare` writes `<index>.pipeline.json`, which binds
the manifest and index digests to the source directory and file identities.
`validate` checks that receipt and advances it only if the index contract and
source identities still match. `upload` requires the validated receipt and
performs one remote size/checksum read-back. `seal` requires the upload receipt
and checks the remote index digest without downloading and probing every video a
second time.
Sealing orders evaluation IDs deterministically,
persists their canonical batch index and position, and records the batch count
in the dataset-sealed event. Every contender therefore receives the same
first-attempt batch membership. `evaluation_batch_size` cannot change after
sealing. This is a scoring-fairness invariant, not only a dispatch convenience:
the equal-share runtime and cost attributed to an item depend on the duration
and processing time of the other videos in its batch. Comparing contenders that
received different video compositions would therefore compare different
workloads for the same per-item cost component.

`prepare` performs the local full-media probe, hashing, metadata, and duration
checks once. `validate` confirms that the prepared index and source file
identities have not changed. `prepare` records each physical source path as
`batches/<zero-padded-index>/inputs/<relative-source-path>` in the global index.
`upload` writes exactly those batch-scoped sources; it does not create duplicate
root `inputs/` objects. Each batch directory contains only its assigned source
paths and never contains the global index. The upload is verified by read-back.
The uploaded layout cannot be narrowed or replaced; upload read-back failure
requires a new competition ID and Volume. `seal` requires the remote digest and
pipeline receipt to match before inserting immutable evaluation rows in
SQLite. A sealed index cannot be replaced; use a new competition ID and Volume.

## Dispatch, terminal failures, and restart safety

Each claimed batch starts a fresh Sandbox whose `/evaluation-inputs` mount is a
Modal Volume `sub_path` for exactly that canonical batch. Its `/output` mount is
also scoped to that dispatch ID. The Sandbox is terminated after the invocation,
including successful invocations; contender processes, local files, inputs, and
derived outputs therefore cannot persist into a later batch. The trusted scorer
continues to read the full input and contender output Volumes outside the
untrusted Sandbox.

Before claiming work, a disposable Sandbox mounts one randomly selected,
already-existing canonical batch read-only and proves that the root index is
absent. Using an existing batch is required because Modal cannot create a missing
`sub_path` through a read-only mount. The disposable output mount uses a fresh
writable subdirectory and is isolated from real batch outputs.

Contender code must not process an evaluation before the validator begins that
batch's measured invocation, retain an input or derived encode across batches,
or return an output generated outside the measured invocation. The filesystem
and lifecycle controls enforce this rule rather than relying only on policy.

SQLite is the authoritative queue and lease store. Redis is not required for
correctness; if added as a dispatch accelerator, every key must remain
reconstructible from SQLite.

- One contender receives sequential batches so its sandbox remains warm.
- Different contenders run concurrently up to `max_parallel_contenders`.
- Claims select the earliest incomplete canonical batch.
- A claim creates one attempt row per evaluation and one batch row.
- Idempotency is
  `competition_id + hotkey + evaluation_id + attempt`.
- Late or duplicate outcomes cannot revive an expired batch.
- Miner invocation errors, storage/scoring errors, and expired execution or
  scoring leases are terminal evaluation failures and are not automatically
  dispatched again. This prevents a contender from reducing its recorded cost
  by deliberately failing selected videos and processing them on a cheaper
  replay.
- After a confirmed validator-infrastructure incident, an operator may use the
  allowlisted repair command to requeue affected rows explicitly. The old
  attempt remains `REQUEUED`, the new attempt number increases, and an
  `EVALUATION_INFRASTRUCTURE_REQUEUED` audit event records the intervention.
- Terminal failures contribute zero to quality, cost efficiency, and completion
  rather than disqualifying the entire contender.
- Expired execution leases use `BATCH_LEASE_EXPIRED`; expired scoring leases use
  `SCORING_LEASE_EXPIRED`.
- A validator restart resumes from persisted claims, batches, histories, image
  bindings, sandbox generations, and output Volumes.

## Output validation and scoring

The trusted scorer reads both source and contender output outside the sandbox.
A result must:

- exist at the exact assigned output path;
- be an AV1 MP4 smaller than its source;
- preserve the source audio streams, dimensions, duration, frame count, pixel
  format constraints, and sample aspect ratio within allowed tolerances;
- match the immutable source checksum and evaluation metadata;
- for VBR items, report a video-stream bitrate (falling back to container
  bitrate) no greater than 110% of the requested target;
- receive a positive absolute media score and meet the manifest's minimum
  compression ratio.

For item `i`:

```text
length_weight_i =
  (
    log(1 + duration_i)
    / log(1 + manifest.max_video_length_seconds)
  ) ^ manifest.length_weight_exponent

compression_ratio_i =
  source_size_i / output_size_i

compression_rate_i =
  output_size_i / source_size_i

(media_score_i, compression_component_i, vmaf_component_i, reason_i) =
  services.scoring.scoring_function.calculate_compression_score(
    vmaf_score_i,
    compression_rate_i,
    item.vmaf_threshold
  )

effective_cost_i =
  max(reconciled_or_estimated_cost_i, manifest.cost_floor_usd)

minimum_valid_cost_i =
  min(effective_cost_i across contenders with a valid scored result for item i)

cost_efficiency_i =
  minimum_valid_cost_i / effective_cost_i
```

`calculate_compression_score` is the same absolute curve used by compression
inference. It weights compression 70% and VMAF 30%, gives a compression
component of 1 at 20x, adds a logarithmic compression bonus above 20x, and
normalizes/caps the combined media score at 1. VMAF values from five points
below the item target up to the target occupy a quadratic soft-recovery zone.
At or above the target, VMAF contributes from 0.7 at the target to 1 at VMAF
100. A zero media score fails the item and therefore gives zero for every
component.

VBR validation follows the synthetic-compression encoding contract. Missing
bitrate metadata fails the item. The requested bitrate is an upper target, so
the 10% tolerance is enforced only as a ceiling: an achieved bitrate below the
target remains valid and is then scored by the same compression/VMAF media
curve as CRF. A rejected VBR item records zero media, compression, and VMAF
components with `OUTPUT_BITRATE_MISSING` or
`OUTPUT_BITRATE_EXCEEDS_TARGET` as its reason.

Invalid items receive zero components and are excluded from the minimum-cost
reference so a cheap invalid output cannot reduce valid contenders' scores. An
output whose audio-stream count or encoded-audio fingerprint differs from the
source records a zero media score and
`media_score_reason=OUTPUT_AUDIO_NOT_PRESERVED` before failing the item. For
each item, every cheapest valid contender receives cost efficiency `1.0`; a
valid contender costing twice that minimum receives `0.5`. Media score remains
on the absolute curve implemented by the deployed scoring code.
`scoring_version` is the manifest's stable identifier for that scoring contract;
changing the identifier alone does not dynamically select another algorithm.
Components are then aggregated over the complete dataset. The implemented
schema retains `quality_aggregate` as a compatibility copy of
`media_score_aggregate`:

The exponent changes how strongly video duration separates item weights. An
exponent of `1.0` preserves the original logarithmic curve. The example
manifest uses `2.02`, for which a 30-minute video has approximately 10 times
the weight of a 10-second video. The same persisted item weight is used by
media score, cost efficiency, and length coverage aggregation.

```text
media_score_aggregate =
  sum(length_weight_i × media_score_i)
  / sum(length_weight_i)

cost_efficiency_aggregate =
  sum(length_weight_i × cost_efficiency_i) / sum(length_weight_i)

length_coverage =
  sum(length_weight_i × completed_i) / sum(length_weight_i)

final_score =
  scoring_factors.quality × media_score_aggregate
  + scoring_factors.cost_efficiency × cost_efficiency_aggregate
  + scoring_factors.length_coverage × length_coverage
```

Compression ratio and VMAF are not normalized against participating contenders.
Cost efficiency deliberately is: adding or removing a valid cheaper contender
can change existing contenders' cost components and final scores. Consequently,
the final cost normalization must use the complete finalized evaluated
contender set. Failed and invalid items never define the minimum.

Therefore compression ratio and VMAF both directly affect ranking. Raw
checksums, sizes, VMAF, compression ratios, media/compression/VMAF components,
duration weights, failures, scoring version, cost source, and the finalized
evaluated contender set are retained so aggregates can be reproduced from
history.

## Runtime and cost accounting

The validator measures the complete sandbox invocation with a monotonic clock.
Batch `active_runtime_seconds` equals measured wall time from fresh Sandbox
creation through isolation/resource/readiness probes, contender execution,
output sync, and termination. This matches the billed Sandbox lifecycle; cold
start is included rather than attributed outside the batch.
The supervisor polls both the active command and parent Sandbox. If the Sandbox
finishes first, the batch stops with `SANDBOX_DISAPPEARED` and records only the
time observed before disappearance instead of waiting for the command deadline.

Exact per-input cloud cost is not observable when inputs share one concurrently
billed sandbox. The assigned per-item runtime is
`batch_wall_runtime / attempted_item_count`, and the estimated batch cost is
divided by the same count. These are equal-share accounting values, not measured
individual runtimes. Their value depends on batch composition: a long or
compute-intensive video can dominate batch wall time and increase the runtime
and cost assigned to every other video in that batch.

To keep this composition effect identical across contenders, dataset sealing
persists canonical batches and every contender receives the same evaluation IDs
in the same positions. Automatic batch replay is disabled, so a miner cannot
drop a long video on one invocation and process it later to change the normal
cost-accounting population. Any exceptional operator requeue is explicit,
allowlisted, and audit logged.

The current documented allocation:

1. Probe the actual GPU model/count and CPU cores.
2. Apply locked public Modal GPU and Sandbox CPU rates to complete batch wall
   time.
3. Allocate the batch estimate equally across all attempted items, including
   failed items.
4. Store invoice-aligned `reconciled_cost_usd` separately when sufficiently
   granular billing data becomes available.

Estimated cost must never be described as an exact per-input invoice.

## Ranking, review, and weights

At `competition_end_time`, the implementation ranks independently calculated,
already precision-rounded final scores:

1. Require every evaluation item to be terminal.
2. Exclude contenders already marked objectively ineligible.
3. Rank by final score.
4. Apply a recorded review order to a complete exact-score tie group, if one
   exists.
5. Otherwise prefer the earlier pinned commit committer timestamp.
6. If still tied, compare commit SHA and then hotkey lexicographically.

In the production validator, evaluated batch outcomes are retained in memory
for the complete competition round. Once every expected contender/item result
is present and no systemic infrastructure failure blocks the round, item
histories, aggregates, score components, and the transition to
`AWAITING_END_TIME` are committed in one transaction. This deferred round
commit prevents auditors from observing a partially persisted,
population-dependent cost ranking. A process loss before that transaction
leaves no partial scoring round to treat as authoritative; the round is run
again from its persisted pre-round state.

### Human review

`REVIEW_REQUIRED` is deliberately narrow. It currently occurs only when
executable-scope submission content matches remote-download behavior:
`requests`/`httpx` calls, `urllib.request`, or shell `curl`/`wget` with an HTTP
URL. A rejecting finding always takes precedence, so a repository that also
contains an objective violation is `REJECTED`, not review-required. Opaque
executables, obfuscation patterns, `eval`/`exec`, committed credentials,
modified SDK tooling, path escapes, invalid descriptors, and the other
fail-closed checks are objective rejections.

The complete set of objective static-validation rejections currently emitted
before `BUILDING` is:

| Reason code | Rejection condition |
| --- | --- |
| `REQUIRED_FILE_MISSING` | A required competition file or an explicit/locked dependency file is missing. |
| `INVALID_SOLUTION_DESCRIPTOR` | `competition_solution.json` is unreadable or invalid, declares the wrong competition type/schema, disables local-path I/O, or names the wrong entrypoint, preflight, or SDK file. A repository file that cannot be read is also reported under this code. |
| `REQUIRED_ROUTE_MISSING` | The solution descriptor does not declare both `/health` and `/compress`. |
| `PATH_ESCAPE` | A symlink resolves outside the pinned repository. |
| `BROKEN_SYMLINK` | A symlink target does not resolve or a path disappears during validation. |
| `SUBMODULE_NOT_ALLOWED` | The repository contains `.gitmodules`; Git submodules are not permitted. |
| `NESTED_REPOSITORY` | Nested Git metadata is present below the repository root. |
| `GIT_LFS_POINTER` | A committed file is a Git LFS pointer rather than the referenced content. |
| `FILE_TOO_LARGE` | An individual file exceeds the static validator's 50 MB limit. |
| `REPOSITORY_TOO_LARGE` | Total repository content exceeds the static validator's 2 GB limit. |
| `INVALID_PYTHON` | A Python source file cannot be parsed. |
| `COMMITTED_SECRET` | Content matches a supported GitHub token, AWS access-key, or private-key signature. |
| `DYNAMIC_EXECUTION` | Executable-scope content uses `eval()` or `exec()`. |
| `ENCRYPTED_ARCHIVE` | A ZIP or JAR contains encrypted entries. |
| `OBFUSCATION_REVIEW` | Despite the legacy name, this is a hard rejection. It covers opaque executable extensions, known obfuscation runtimes, NUL bytes, lines longer than 1,000 bytes, more than 12 Python statements on one line, and sufficiently large embedded encoded-loader payloads. |
| `SDK_TOOL_MODIFIED` | Canonical `miner/common_preflight.py` or `miner/competition_sdk.py` content does not match an audited digest. |

`COMPILED_ONLY_CODE` remains in the reason-code enum for compatibility but is
not currently emitted. Compiled or otherwise opaque executable files are
rejected as `OBFUSCATION_REVIEW`. `REMOTE_DOWNLOAD_REVIEW` is the only currently
emitted non-rejecting static finding. If it appears alongside any rejection,
the overall validation status is `REJECTED`.

During `VALIDATING`, any non-disqualified `REVIEW_REQUIRED` submission holds the
competition in that state. A review operator must either accept or reject its
readability review, or manually disqualify it. Acceptance means the reviewed
remote-download usage was found readable and compliant; it does not waive any
objective validation failure.

Every decision is stored in `competition_human_reviews` with the operator
identity, affected contenders, structured decision, reason, timestamp,
supersession link where applicable, and an integrity hash. A matching
`competition_events` row records its effect. Reviews are accepted only through
`human_review_deadline` and never after a competition becomes terminal.

The supported review actions are:

- Resolve a pending readability decision during `VALIDATING`.
- Manually disqualify a pinned contender during `VALIDATING`, `BUILDING`,
  `EVALUATING`, or `AWAITING_END_TIME`. New work stops for that contender and
  its active batch sandbox is terminated by the scheduler. Disqualification can narrow
  eligibility but cannot reinstate an objectively invalid contender.
- Order every contender in one exact rounded-score tie group during
  `AWAITING_END_TIME`. Non-tied contenders cannot be reordered. A newer order
  for the same tie supersedes the previous record.

Manual disqualification preserves raw attempts, costs, media measurements, and
failure history. If provisional scores already exist, the validator excludes
the contender and recomputes all derived item components and contender
aggregates from the remaining eligible set. This is required because the
minimum valid cost for each evaluation is contender-relative. The competition
row is marked `scores_need_recalculation` before that rebuild and the scheduler
retries the database transaction after a restart until the flag is cleared.
This is aggregate recalculation, not evaluation: it reads the persisted latest
terminal histories and does not rebuild an image, start or invoke a Modal
sandbox, re-encode a video, read output media, or rerun VMAF. The stored status,
duration weight, VMAF, compression ratio, and estimated/reconciled cost are
sufficient.

After the first eligible compression podium exists, subnet emissions are:

```text
60% compression inference
20% upscaling inference
20% latest compression competition podium
    - 70% of the competition pool to first
    - 20% to second
    - 10% to third
```

Before the first eligible podium, use the existing 80% compression / 20%
upscaling inference split. Burn remains zero. A podium remains incumbent until
atomically replaced by a later finalized competition. Contributions for the
same registered UID are merged before normalization.

The current lookup resolves podium hotkeys against the current metagraph and
disables the competition pool when rank 1 is absent. A validator permit does
not exclude a miner from podium rewards because the chain may assign one based
on alpha stake. Missing lower podium members return their unused shares to
compression inference.

## Persistence and observability

Competition-only tables include:

- `competition_schema_migrations`
- `competitions`
- `contender_metadata`
- `competition_evaluation_items`
- `competition_batches`
- `competition_sandboxes`
- `contender_performance_history`
- `competition_human_reviews`
- `competition_events`

Before production, the historical development migrations were squashed into
schema version 1. Schema version 2 adds the optional contender Sandbox GPU/CPU
preferences. Existing version-1 databases upgrade additively; fresh databases
record both migrations. Databases containing the retired development migration
history are intentionally rejected.

For batch inspection, group dispatches by
`competition_id + hotkey + canonical_batch_index`. Use `batch_id` to correlate
one concrete Sandbox invocation with its item-history rows; do not use it to
compare canonical membership across contenders.

SQLite contains no GitHub PATs. Structured logs include competition ID, hotkey,
batch/evaluation identifiers, reason codes, and redacted safe detail. Expected
operator logs include:

- `Competition state transition`
- `Dispatching competition batch`
- `Competition batch persisted`
- `Contender Sandbox terminated after evaluation`
- `Competition Phase 4 complete`

## Operator controls

Competition mode remains opt-in and should be enabled only after manifests,
dataset, database, Modal authentication, budget controls, and production gates
are verified. Important configuration includes:

- `COMPETITION_MODE_ENABLED`
- `COMPETITION_MANIFEST_GLOB`
- `COMPETITION_DATABASE_URL`
- `COMPETITION_EXECUTION_ENABLED=true`
- `COMPETITION_BUILD_BACKEND=modal`
- `COMPETITION_MODAL_ENVIRONMENT`
- `COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION`
- private artifact backup bucket/prefix and credentials

Run preflight before enrollment:

```bash
python scripts/competition_preflight.py \
  competitions/manifests/examples/compression-competition.json
```

If the manifest configures a boss, validate that exact export too:

```bash
python scripts/competition_preflight.py \
  competitions/manifests/compression-YYYY-wWW.json \
  --validate-boss
```

Use an absolute SQLite URL and the same database for dataset sealing and the
validator process. A relative URL resolved from different working directories
can make a successfully sealed competition appear to have no dataset.

Use the review CLI from a host with access to that same competition database.
Every mutation requires an operator identity, a reason, and `--apply`:

```bash
python scripts/competition_review.py \
  --database-url sqlite:////absolute/path/video_subnet_validator.db \
  resolve-readability \
  --competition-id compression-2026-w29 \
  --hotkey <contender-hotkey> \
  --decision accept \
  --operator reviewer@example.com \
  --reason "Pinned package URL is readable and used only during image build" \
  --apply

python scripts/competition_review.py \
  --database-url sqlite:////absolute/path/video_subnet_validator.db \
  disqualify \
  --competition-id compression-2026-w29 \
  --hotkey <contender-hotkey> \
  --operator reviewer@example.com \
  --reason "Published competition rule violation" \
  --apply

python scripts/competition_review.py \
  --database-url sqlite:////absolute/path/video_subnet_validator.db \
  order-exact-tie \
  --competition-id compression-2026-w29 \
  --hotkey <first-hotkey> \
  --hotkey <second-hotkey> \
  --operator reviewer@example.com \
  --reason "Documented exact-tie review outcome" \
  --apply
```

Inspect the full audit trail without mutating it:

```bash
python scripts/competition_review.py \
  --database-url sqlite:////absolute/path/video_subnet_validator.db \
  list --competition-id compression-2026-w29
```

The `list` result is the complete review packet. It contains provisional ranks,
score components, item totals and failures, estimated/reconciled cost source,
static-validation status, pinned commit metadata, exact tie groups and their
decision source, pending review queues, and the full review history. During
`AWAITING_END_TIME` these ranks are computed without writing `final_rank`; the
same ranking implementation is reused transactionally at `COMPLETED`.

For focused dataset operations and troubleshooting, see
[`competition_phase4_dataset_runbook.md`](competition_phase4_dataset_runbook.md).
For installation and PM2 configuration, see
[`validator_setup.md`](validator_setup.md).
Use [`miner_competition_checklist.md`](miner_competition_checklist.md) for each
contender submission and
[`validator_competition_checklist.md`](validator_competition_checklist.md) for
each new competition.

## Production readiness and roadmap

The historical implementation plan also records production and extension
requirements around the implemented competition engine:

- **Public audit/reproducibility:** publish a credential-free,
  checksum-addressed post-competition bundle containing the manifest, dataset,
  pinned source, outputs, metrics, costs, reviews, events, and final ranking.
  Publication must be atomic through an inventory and final `COMPLETE` marker.
- **Implemented owner authorization:** competition protocol requests are
  accepted only from the on-chain subnet-owner hotkey by default.
- **Implemented entry guards:** invitations report the observed and required
  alpha stake; an ineligible miner cannot opt in. Every submitted miner is
  checked again using the latest metagraph snapshot recorded before
  finalisation. GitHub accounts or repositories already used by another
  contender in the competition are rejected.
- **Future competition types:** use a type adapter/registry. Upscaling requires
  separate high-resolution ground truth unavailable to contenders and cannot
  reuse compression's two-media assumption.
- **Future entry fees:** remain a separate policy extension.

Production enablement should follow a dev competition, a shadow competition
with no weight effect, budget/rollback exercises, and completion of every
mandatory security and eligibility gate.

## Source map

| Area | Path |
| --- | --- |
| Manifest model | `vidaio_subnet_core/competition/config.py` |
| Lifecycle manager | `vidaio_subnet_core/competition/manager.py` |
| Persistence and leases | `vidaio_subnet_core/competition/repository.py` |
| Human review CLI | `scripts/competition_review.py` |
| Models and migrations | `vidaio_subnet_core/competition/models.py`, `migrations.py` |
| Enrollment and intake | `enrollment.py`, `intake.py`, `validation.py` |
| Trusted build | `build.py` |
| Sandbox supervision | `modal_runner.py` |
| Dataset index and Volume access | `dataset.py` |
| Batch execution | `execution.py` |
| Dynamic timeout model | `timeouts.py` |
| Item and aggregate scoring | `scoring.py`, `scoring_api.py` |
| Dataset CLI | `scripts/competition_dataset.py` |
| Preflight | `scripts/competition_preflight.py` |
| Competition tests | `tests/competition/` |
