# Competition Phase 4 dataset runbook

Phase 4 does not dispatch until the evaluation input Volume contains a verified
index and that exact index digest is sealed in the competition SQLite database.
Creating an empty Modal Volume is therefore only the first step.

Set these shell variables to the manifest the validator will load, the directory
containing the private source MP4 files, and a local generated index path:

```bash
MANIFEST=/absolute/path/to/compression-2026-w30.json
SOURCE_DIR=/absolute/path/to/private-evaluation-videos
INDEX=/tmp/compression-2026-w30-index.json
DATABASE_URL=sqlite:////absolute/path/to/video_subnet_validator.db
```

Prepare and locally verify the immutable index:

```bash
python scripts/competition_dataset.py prepare \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --index "$INDEX"

python scripts/competition_dataset.py validate \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --index "$INDEX"
```

For compression competitions, `prepare` creates one evaluation query per
physical source MP4. It deterministically selects CRF or VBR and a VMAF floor
of 85, 89, or 93 using the manifest's `scoring_seed`; VBR queries also receive
a deterministic random target of 5, 8, or 10 Mbps. Consequently, 20 source
videos produce 20 evaluation rows. The format remains evaluation index schema
version 2; validators continue to read older indexes for sealed competitions.

Upload to the manifest's `evaluation_input_volume_name` in Modal `main`. The
command verifies the uploaded index and every source object by reading them back.
It refuses to replace a different index already present in the Volume.

```bash
python scripts/competition_dataset.py upload \
  --manifest "$MANIFEST" \
  --source-dir "$SOURCE_DIR" \
  --index "$INDEX" \
  --environment main
```

Seal the same digest in the validator database only after upload succeeds:

```bash
python scripts/competition_dataset.py seal \
  --manifest "$MANIFEST" \
  --index "$INDEX" \
  --environment main \
  --database-url "$DATABASE_URL"
```

The validator does not need to be running. For a new database, `seal` applies
the competition schema baseline, registers the manifest as `SCHEDULED`, and
seals the evaluation rows. On first boot, configure `COMPETITION_DATABASE_URL`
with this same URL and ensure the manifest remains matched by
`COMPETITION_MANIFEST_GLOB`; the competition will start automatically when its
start time is due.

The late-loading workflow remains supported. If the validator was started with
an empty input Volume, run `upload` and `seal` against its existing database.
The manifest digest must match, and an `EVALUATING` competition resumes on its
next execution cycle.

A sealed index is immutable. Do not try to apply the one-query-per-video format
to an already sealed competition; prepare a fresh index and competition ID.
Miners must publish from the matching SDK revision because VBR requests add
`codec_mode` and `target_bitrate` to the competition route contract.

On the next scheduler cycle, every accepted contender receives sequential
batches of at most five distinct source videos. Different contenders run
concurrently up to the manifest's `max_parallel_contenders`. A restart resumes
from SQLite. An expired in-flight batch is marked terminally failed and is not
automatically dispatched again. A confirmed validator-infrastructure incident
can only be replayed through the explicit, audited operator repair command.

Expected logs include `Dispatching competition batch`, `Competition batch
persisted`, both `EVALUATING -> SCORING` and `SCORING -> AWAITING_END_TIME`, and
finally `Competition Phase 4 complete`. Each contender Sandbox is terminated as
soon as that hotkey finishes every dataset item, without waiting for slower
contenders, and logs `trigger=contender_dataset_complete` plus
`volume_retained=true`. Input and contender output Volumes and their files
persist until manually deleted. A checksum mismatch, missing index, or missing
output remains in `EVALUATING` with a reason-coded log instead of silently
scoring incomplete data.
