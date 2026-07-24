# Running Validator

---

## Machine Requirements

To achieve optimal results, we recommend the following setup:

- **Operating System**: Ubuntu 24.04 LTS or higher
  [Learn more about Ubuntu 24.04 LTS](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- **GPU**: NVIDIA RTX A6000 with 48GB VRAM and 30 CPU Cores

---

## Bootstrap System Dependencies

The `bootstrap.sh` script at the repository root automates the installation of core system-level dependencies:

- **NVIDIA GPU drivers** (default version 535)
- **Docker** and the **NVIDIA Container Toolkit**
- **Python 3.11**
- Base utilities (git, curl, wget, etc.)

Run the script **as root** with the `-E` flag to preserve environment variables:

```bash
sudo -E ./bootstrap.sh
```

> **Note:** If you encounter `dpkg` lock issues (common on platforms like TensorDock), wait ~15 minutes and re-run the script. The script will **automatically reboot** the machine if a new NVIDIA driver was installed.

#### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_DRIVER_VERSION` | `535` | NVIDIA driver version to install |

Once the bootstrap completes (and any reboot finishes), proceed with the rest of this guide.

---

## Install PM2 (Process Manager)

**PM2** is used to manage and monitor the validator process. If you haven’t installed PM2 yet, follow these steps:

1. Install `npm` and PM2:
   ```bash
   sudo apt update
   sudo apt install npm -y
   sudo npm install pm2 -g
   pm2 update
   ```

2. For more details, refer to the [PM2 Documentation](https://pm2.io/docs/runtime/guide/installation/).

---

## Install Redis

1. Install 'redis'
   ```bash
   sudo apt update
   sudo apt install redis-server
   sudo systemctl start redis
   sudo systemctl enable redis-server
   sudo systemctl status redis
   ```


## Install Project Dependencies

### Prerequisites

- **Python**: Version 3.10 or higher
- **pip**: Python package manager
- **virtualenv** (optional): For dependency isolation

---

### 1. Clone the Repository

Clone the project repository to your local machine:
```bash
git clone https://github.com/vidaio-subnet/vidaio-subnet.git
cd vidaio-subnet
```

---

### 2. Set Up a Virtual Environment (Recommended)

Create and activate a virtual environment to isolate project dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  
```

---

### 3. Install the Package and Dependencies

Install the project and its dependencies using `pip`:
```bash
pip install -e .
```

---

### 4. Configure Environment Variables

To configure environment variables, follow these steps:

1. Create a `.env` file in the project root directory by referencing the provided `.env.template` file:
   ```bash
   cp .env.template .env
   ```
2. Set up a bucket in cloud storage. The base miner code utilizes MinIO to connect with cloud storage services, so you'll need to prepare your bucket using a platform that supports MinIO integration, such as Backblaze. Alternatively, you can modify the code to suit your specific requirements. *IMPORTANT*: Note that currently the `region` of the storage is hardcoded, and must be adjusted in `vidaio_subnet_core/utilities/storage_client.py` for corresponding storage, such as AWS.
3. Add the required variables to the `.env` file. For example:
   ```env
   BUCKET_NAME="S3 buckent name"
   BUCKET_COMPATIBLE_ENDPOINT="S3 bucket endpoint"
   BUTKET_COMPATIBLE_ACCESS_KEY="S3 bucket personal access key"
   BUCKET_COMPATIBLE_SECRET_KEY="S3 bucket personal secret key"
   PEXELS_API_KEY="Your Pexels account api key"
   WANDB_API_KEY="Your WANDB account api key"
   ```

4. Ensure that the bucket is configured with the appropriate permissions to allow file uploads and enable public access for downloads via presigned URLs.

5. Create your Pexels API key and replace it. (https://www.pexels.com/)
6. Once the `.env` file is properly configured, the application will use the specified credentials for S3 bucket, Pexels and Wandb.


---

## Install FFMPEG

FFMPEG is required for processing video files. Install it using the following commands:
```bash
sudo apt update
sudo apt install ffmpeg -y
```

For more details, refer to the [FFMPEG Documentation](https://www.ffmpeg.org/download.html#build-linux).

---

## Install VMAF

To enable video quality validation, install **VMAF** by following the steps below to set up a clean virtual environment, install dependencies, and compile the tool.

---

Clone the VMAF repository into the working root directory of your `vidaio-subnet` package:

```bash
git clone https://github.com/vidAio-subnet/vmaf.git
cp vmaf_utils/Dockerfile vmaf/Dockerfile
cp vmaf_utils/Dockerfile.ffmpeg vmaf/Dockerfile.ffmpeg
cd vmaf
git stash && git reset --hard 332dde62838d91d8b5216e9822de58851f2fd64f && git stash apply
docker build -t vmaf .
docker build -t vmaf_ffmpeg:latest -f Dockerfile.ffmpeg .
```

---
## Competition scheduler and Modal execution

Competition mode is independently disabled by default. With it disabled, the
validator does not create competition tables, scan manifests, or change any
synthetic/organic inference task.

For each new competition, complete
[`validator_competition_checklist.md`](validator_competition_checklist.md).
The consolidated competition contract and timeout formulas are in
[`competitions.md`](competitions.md).

To prepare a competition scheduler, copy the example manifest from
`competitions/manifests/examples/` into `competitions/manifests/`, update its
ID, timestamps, and Volume names, and provide the stock warmup fixture. Then set:

```env
COMPETITION_MODE_ENABLED=true
COMPETITION_DATABASE_URL=sqlite:///video_subnet_validator.db
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

`COMPETITION_DATABASE_URL` must be visible to both PM2 validator processes.
The competition-mode process writes scores and final rankings to that database;
the inference-mode process opens the same database only as its competition
ranking source when composing on-chain weights. Its inference performance data
continues to use `video_subnet_validator.db`. If the variable is absent from the
inference process, competition rewards are disabled and weights retain the
80% compression / 20% upscaling inference fallback. Restart both processes with
updated environment variables after changing the URL.

The four execution settings above enable the validator-owned Modal build and
Sandbox path in testnet or production. `modal` builds each pinned contender
repository directly on Modal, records its immutable image ID as
`MODAL_ACCEPTED`, and starts its validator-controlled offline Sandbox. Builds
and Sandbox startup run concurrently, bounded by the manifest's
`max_parallel_contenders` value.

When `competition_end_time` moves a competition to `COMPLETED`, the validator
creates an online, integrity-checked SQLite snapshot and uploads it privately to
`s3://<bucket>/<artifact-prefix>/<competition-id>/final/<manifest-digest>/<sqlite-filename>`.
The source path and filename come from `COMPETITION_DATABASE_URL`. The object
size and SHA-256 are verified after upload, and a successful upload is recorded
in the audit event stream. Failed uploads are logged and retried on later
scheduler ticks.

For a manifest-configured boss, the pre-evaluation artifact archive contains
the immutable boss export as `contenders/<boss_hotkey>-boss` and any submission
from the same hotkey as `contenders/<boss_hotkey>`. Both are scored; only the
higher-scoring row is ranked. The final SQLite snapshot preserves `is_boss` and
the concrete contender ID, so it identifies which solution won even though
`competitions.winner_hotkey` remains the real payout hotkey.

Before enrollment, validate the boss configuration and exact SDK export selected
by the manifest:

```bash
python scripts/competition_preflight.py \
  competitions/manifests/compression-YYYY-wWW.json \
  --validate-boss
```

The command must report `ACCEPTED`. It fails non-zero when the boss is missing
from the manifest, its path escapes the validator checkout, the export is
incomplete, or the normal objective static source policy returns
`REVIEW_REQUIRED` or `REJECTED`.

Modal does not currently provide final-image size attestation for this build
path. Setting
`COMPETITION_ACCEPT_MODAL_BUILD_WITHOUT_SIZE_ATTESTATION=true` is an explicit
operator acknowledgement of that limitation; it is not a claim that the 25 GB
limit was measured. Deployments that require attested size enforcement can
still integrate the stricter trusted-builder interface.

Every direct Modal contender build uses the manifest's `modal_build_timeout`
duration, defaulting to `10m` when that field is omitted. Builds run in separate
client processes and use contender-specific Modal Apps, so one stalled or
malicious Dockerfile cannot block another contender's parallel build. When the
deadline is exceeded, the validator kills the build client, asks Modal to stop
that isolated App, records the contender as rejected with `BUILD_TIMEOUT`, and
continues the `BUILDING` phase. Remote App cancellation is best-effort, so
operators should investigate any App that Modal still reports as active after a
`BUILD_TIMEOUT` event.

### Modal authentication for competition execution

The scheduler and enrollment path above do not need Modal credentials. The
validator-owned execution path does: it creates Modal Sandboxes, looks up the
manifest's input Volume, and creates a separate output Volume for each
contender. The Modal SDK is installed by `pip install -e .` because it is listed
in `requirements.txt`.

Contender output Volumes are created automatically as Modal v2 Volumes. V2 is
required because the validator explicitly commits each completed batch with
`sync` before reading it; this Sandbox commit operation is unavailable for v1
Volumes. Do not pre-create generated output Volumes with Modal's default v1
setting.

Generated names retain a bounded namespace from `output_volume_prefix`, the
actual `competition_id`, and recognizable leading/trailing characters from the
miner hotkey plus a collision-resistant suffix. For example:

```text
vidaio-compression-2026-w30-5FRUjvY1AxpK-bLqFmJLo-bad385
```

The database remains authoritative for the complete hotkey-to-Volume mapping.
Contender code only performs the requested transformation; it is not trusted to
validate or score its output. Every returned output is committed to the
contender's output Volume and independently inspected by the validator. A
competition result contains only its validator-assigned `output_path` (or
`null` when no output was produced); it cannot supply success, failure reason,
timing, evaluation identity, hotkey, or batch identity. Results are matched to
request items by position and exact output path. A valid file is scored
normally; an invalid file is `FAILED` with zero score while retaining any
measurable size, compression, VMAF, validator-measured runtime, and cost
metrics. Missing outputs remain ordinary failures with no stored file.
As soon as one contender has terminal results for every evaluation item, the
validator terminates that contender's Sandbox without waiting for slower miners.
Later scheduler ticks exclude completed contenders instead of warming them
again. This does not delete the input Volume, contender output Volumes, or any
files stored in them. Competition Volumes persist until an operator manually
deletes them.

If an empty v1 output Volume already exists under a generated contender name,
stop the validator and any Modal Apps using it, then replace it explicitly:

```bash
modal volume delete --env main --yes "$OUTPUT_VOLUME"
modal volume create --env main --version 2 "$OUTPUT_VOLUME"
```

Never delete a non-empty output Volume as an automatic repair. Modal does not
provide an in-place v1-to-v2 upgrade; migrate data that must be retained to a
separately named v2 Volume first.

Authenticate the operating-system user that runs the validator:

```bash
modal setup
```

For a headless host or a validator managed by PM2, provide a Modal token in the
validator process environment instead. Do not put this token in a miner
submission or expose it to a contender Sandbox.

```env
MODAL_TOKEN_ID=your-modal-token-id
MODAL_TOKEN_SECRET=your-modal-token-secret
```

Make sure the selected Modal environment exists and contains the read-only
input Volume named by `evaluation_input_volume_name` in the competition
manifest. The validator uses the environment selected by
`COMPETITION_MODAL_ENVIRONMENT` (`dev` by default). Its launch policy passes
`secrets=[]`, disables OIDC identity, and
blocks outbound networking, so the validator's Modal token remains in the
trusted host process rather than inside contender code.

For example, create the manifest's input Volume before starting the validator:

```bash
INPUT_VOLUME=vidaio-competition-compression-2026-w30-inputs
modal volume create \
  --env main \
  "$INPUT_VOLUME"
```

`INPUT_VOLUME` must exactly match the manifest's
`evaluation_input_volume_name`.

Creating the Volume does **not** upload the evaluation dataset. An empty Volume
causes the validator to remain in `EVALUATING` while its Sandboxes continue to
answer `/health`. Phase 4 dispatch starts only after the Volume contains a
read-back-verified index and source files and that exact index digest is sealed
in the competition database.

### Prepare, upload, and seal the Phase 4 dataset

Choose the manifest that the validator will load, a private directory containing
the source MP4 files, a temporary local index path, and the validator's actual
SQLite URL:

```bash
MANIFEST=/absolute/path/to/competitions/manifests/compression-2026-w30.json
SOURCE_DIR=/absolute/path/to/private-evaluation-videos
INDEX=/tmp/compression-2026-w30-index.json
DATABASE_URL=sqlite:////absolute/path/to/video_subnet_validator.db
```


```bash
MANIFEST=/root/old_root/root/vidaio-subnet-optim/competitions/manifests/compression-competition.json
SOURCE_DIR=/root/old_root/root/vidaio-subnet-optim/sample_data
INDEX=/tmp/compression-2026-w32-index.json
DATABASE_URL=sqlite:////root/old_root/root/vidaio-subnet-optim/competition_test.db
```

Run all four operations from the same deployed repository and virtual
environment as the validator:

```bash
python scripts/competition_dataset.py prepare \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --index "$INDEX"

python scripts/competition_dataset.py validate \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --index "$INDEX"

python scripts/competition_dataset.py upload \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --index "$INDEX" \
  --environment dev

python scripts/competition_dataset.py seal \
  --manifest "$MANIFEST" \
  --index "$INDEX" \
  --environment dev \
  --database-url "$DATABASE_URL"
```

`prepare` probes every MP4 and records immutable size, checksum, duration,
dimensions, frame count, codec, pixel format, and aspect ratio metadata.
Each source becomes exactly one compression evaluation row. Its CRF or VBR mode
and VMAF target (85, 89, or 93) are selected by deterministic pseudorandom choice
seeded from the manifest; VBR queries also receive a 5, 8, or 10 Mbps target.
Rebuilding an unchanged index therefore produces the same query set. A full
five-item batch always contains five distinct physical videos.
`validate` verifies the local files. `upload` uploads the index and sources to
the manifest's input Volume and reads back every object to verify its checksum.
It refuses to replace a different index. `seal` verifies that the remote and
local index digests match before writing immutable evaluation rows to SQLite.
If the database does not exist yet, `seal` creates it, applies the competition
migrations, and registers the manifest in `SCHEDULED` state. If the competition
already exists, it requires the database manifest digest to match and therefore
cannot silently replace the validator's configured revision.

Because the sealed index and contender API contract are immutable for a live
competition, use a new competition ID for this one-query-per-video format and
require miners to republish with the matching SDK (`--refresh` when upgrading
an existing prepared repository).

This means the entire sequence can run while the validator is stopped. On its
first boot the validator finds the pre-registered, pre-sealed competition and
starts it automatically when `competition_start_time` is due. Keep the manifest
file in `COMPETITION_MANIFEST_GLOB` and point `COMPETITION_DATABASE_URL` at the
same database passed to `seal`.

Use an absolute SQLite URL as shown above, or prove that the CLI and PM2 use the
same working directory. Sealing a different relative database file will leave
the running validator waiting for a dataset even though the CLI reported
success. Preloading is recommended, but not mandatory: the validator may also
start with an empty Volume and wait in `EVALUATING`; running `upload` and `seal`
later unblocks it on the next execution cycle. After late sealing, restart or
wait for that cycle:

```bash
PM2_APP=video-validator-testnet
pm2 restart "$PM2_APP" --update-env
```

Expected progress logs include `Dispatching competition batch`, `Competition
batch persisted`, `EVALUATING -> SCORING`, `SCORING -> AWAITING_END_TIME`, and
`Competition Phase 4 complete`. Each contender logs `Contender Sandbox
terminated after evaluation` with `trigger=contender_dataset_complete` and
`volume_retained=true` as soon as its own work is done. Dataset
checksum mismatches and missing outputs remain fail-closed in `EVALUATING` with
a reason-coded log. See
[`competition_phase4_dataset_runbook.md`](competition_phase4_dataset_runbook.md)
for the focused operator procedure.

Competition execution and scoring deadlines are derived from each batch's
sealed media metadata. Encoding is modeled as four parallel video lanes, with
each video split into 10-minute chunks. A 10-minute 4K chunk takes two minutes;
every started chunk receives that full allowance, including videos shorter than
10 minutes and final partial chunks. Every video also has an absolute two-minute
processing floor before parallel scheduling, so resolution scaling cannot reduce
a short video below that allowance. Work above the floor scales linearly with
pixel count. VMAF scoring is modeled sequentially at 200 FPS for 4K and also
scales with pixel count. Both processing deadlines include two minutes for
Volume I/O and startup, and the database execution lease includes another two
minutes for validator recovery and outcome persistence. The Sandbox row records
the most recently applied invocation deadline, while each batch row records its
attempt-specific execution lease and scoring deadline.

The manifest's `evaluation_batched_run_timeout` and
`scoring_batched_run_timeout` are minimum deadlines, not fixed estimates.
Dynamic workload calculations may increase them but never reduce them. The full
minimum also applies to partial final or explicitly operator-requeued batches
and is not prorated by item count. See
[`competitions.md`](competitions.md#batch-and-timeout-fields) for the formulas,
persistence semantics, and worked examples.

The validator creates every `/output/evaluations/<batch-id>` directory before
calling the contender's `/compress` route. A Sandbox request failure is logged
with its reason code and redacted detail; the persisted summary reports the
actual post-attempt status and a per-reason count. If every terminal item failed
for validator-infrastructure reasons, evaluation stays fail-closed in
`EVALUATING` instead of scoring all contenders as zero.
Sandbox creation, isolation, and readiness are a pre-dispatch gate: failure to
warm any eligible contender pauses dispatch before claims are created, so a
startup outage does not consume evaluation attempts. Startup and recovery
details are also retained in the redacted `SANDBOX_CLOSED` audit event.

After fixing validator infrastructure, an operator can requeue those exhausted
attempts without deleting their history. Stop the validator first so the repair
does not race its scheduler, use the same absolute SQLite URL, then restart it:

```bash
python scripts/competition_repair.py requeue-infrastructure \
  --competition-id compression-2026-w30 \
  --database-url "$DATABASE_URL" \
  --reason-code SANDBOX_START_FAILED \
  --apply

pm2 restart "$PM2_APP" --update-env
```

The repair accepts only allowlisted infrastructure reason codes, records an
`EVALUATION_INFRASTRUCTURE_REQUEUED` audit event, retains old attempts as
`REQUEUED`, and returns an `AWAITING_END_TIME` competition to `EVALUATING`.
New attempts continue with increasing attempt numbers.

Before enabling competition execution, verify authentication and the
offline Sandbox policy with the bounded live probe (this creates a short-lived
Modal Sandbox and may incur a small charge):

```bash
python scripts/competition_modal_phase0_probe.py \
  --environment main \
  --skip-billing
```

Omit `--skip-billing` only when validating billing-report capability. Billing
reports require a supported Modal Team or Enterprise workspace; they are not
required for the network-isolation probe.

> **Current implementation status:** the live validator now orchestrates
> submission finalisation, static-validation acceptance, image building,
> isolated Sandbox startup, durable evaluation dispatch, output validation,
> VMAF and cost scoring, and the transition to `AWAITING_END_TIME`. Batches are
> sequential per warm contender while different contenders run concurrently up
> to `max_parallel_contenders`. SQLite is authoritative for claims, lease
> expiry, terminal failure persistence, and exact-once result persistence;
> Redis is not required for recovery. Sandbox-forward, storage/scoring, and
> lease-expiry failures are not automatically retried. Existing
> `DEVELOPMENT_ACCEPTED` rows remain readable for migration.

Competition cost scoring locks Modal's public rates as of 2026-07-21: B300
`$0.001972/s`, B200 `$0.001736/s`, H200 `$0.001261/s`, H100 `$0.001097/s`, RTX
PRO 6000 `$0.000842/s`, A100 80 GB `$0.000694/s`, A100 40 GB `$0.000583/s`,
L40S `$0.000542/s`, A10 `$0.000306/s`, L4 `$0.000222/s`, and T4 `$0.000164/s`.
Modal's separate Sandbox CPU rate is `$0.00003942/physical-core/s`. The validator
probes the GPU model/count and CPU allocation inside every running Sandbox,
persists them separately from the request, and estimates the full batch wall time
from that observed allocation. Batch `active_runtime_seconds` is
therefore equal to `wall_runtime_seconds`; failed validation, failed scoring,
and other terminally failed items still consumed the Sandbox and retain their
equal share of runtime and estimated cost. Automatic evaluation replay is
disabled, preventing a contender from deliberately failing long videos and
processing them later to alter the normal cost aggregate. Per-item compression
ratio, VMAF, checksums, and media sizes are retained on failed rows whenever
those measurements could be completed safely. A metric remains null only when
the output could not be read, probed, or compared sufficiently to produce it. It
does not attempt to reproduce the final Modal invoice: memory above the request,
image builds, Volume storage, credits, reservations, taxes, CPU utilization
reconciliation, and unrelated workspace usage are excluded. Some Modal
automatic GPU upgrades retain the requested SKU's invoice rate; invoice-aligned
data belongs in
`reconciled_cost_usd`, while competition estimates consistently price the
observed hardware.

Competition scoring version 3 keeps each item's media score absolute: it comes
directly from `services/scoring/scoring_function.py` using its measured
compression rate, measured VMAF, and sealed VMAF target. Cost efficiency is
relative for each evaluation item. The cheapest valid contender receives 1.0
and every other valid contender receives `minimum_valid_cost / own_cost`.
Failed and invalid outputs receive zero and cannot define the minimum. Because
cost is population-dependent, aggregate scoring must use the complete finalized
evaluated contender set.

The start time must be Thursday UTC. At startup the validator normalizes and
hashes each manifest and persists it in competition-only SQLite tables. If a
manifest with the same ID changes while its competition is non-terminal, the
validator updates the database fields used by the scheduler, appends a
`MANIFEST_UPDATED` audit event containing the old/new digests and field-level
changes, archives the prior normalized manifest below
`competition_artifacts/<competition-id>/manifest.revisions/`, and makes the new
normalized manifest current. Completed, failed, and cancelled competitions stay
immutable and require a new competition ID.

If SQLite and `manifest.normalized.json` contain different revisions after an
interrupted update, SQLite remains authoritative. The validator preserves the
divergent file as `manifest.revisions/observed-<sha256>.json`, appends a
`MANIFEST_ARTIFACT_DIVERGENCE` audit event with the observed/database/requested
digests, and atomically restores the requested manifest as current.

Startup fails closed if the configured warmup fixture is absent. The control
scheduler advances persisted lifecycle state. If both the edited start and
contender-finalisation times are already past, one scheduler tick advances the
competition through `ENROLLING` to `FINALIZING_SUBMISSIONS`. No invitations are
sent in that case because the collection window has already closed. Otherwise,
on entry to `ENROLLING`, the validator snapshots serving miner axons,
sends each miner a competition invitation, and immediately polls opt-ins for a
submission. `NOT_READY` miners are polled again at `contender_ping_interval`
until `contender_finalisation_time`; invitation and polling attempts survive
validator restarts. A `READY` response is cloned and pinned immediately so the
raw repository credential is never persisted. Trusted builds and evaluation
remain separate fail-closed lifecycle gates.

Only `COMPRESSION` manifests are active today. The code reserves a future
upscaling media contract with three distinct roles: (i) high-resolution ground
truth held for trusted scoring, (ii) a downsampled reference mounted as the
miner input, and (iii) the miner-processed output. The upscaling adapter remains
an explicit `NotImplementedError` stub; do not configure or advertise an
upscaling competition until its manifest, route, isolation policy, and scorer
are implemented and tested.

Keep `COMPETITION_MODE_ENABLED=false` in production unless the outstanding
trusted-builder Phase 0 gate is satisfied or the operator has explicitly
accepted the unattested direct-Modal build limitation described above.

## Running the Validator with PM2

To run the validator, use the following command:

```bash
pm2 start run.sh --name vidaio_v_autoupdater -- --wallet.name [Your_Wallet_Name] --wallet.hotkey [Your_Hotkey_Name] --subtensor.network finney --netuid 85 --axon.port [port] --logging.debug
```

### Parameters:
- **`--wallet.name`**: Replace `[Your_Wallet_Name]` with your wallet name.
- **`--wallet.hotkey`**: Replace `[Your_Hotkey_Name]` with your hotkey name.
- **`--subtensor.network`**: Specify the target network (e.g., `finney`).
- **`--netuid`**: Specify the network UID (e.g., `85`).
- **`--axon.port`**: Replace `[port]` with the desired port number.
- **`--logging.debug`**: Enables debug-level logging for detailed output.

---
