# Miner Competition Checklist

Use this checklist for every new compression competition. Complete the host
setup in [`miner_setup.md`](miner_setup.md) first. The detailed competition
contract is in [`competitions.md`](competitions.md), its design history is in
[`competition_mode_implementation_plan.md`](competition_mode_implementation_plan.md),
and the executable miner instructions are in
[`miner/README.md`](../miner/README.md#competition-mode-phase-1-contract).

The validator, not the miner, chooses the evaluation videos, CRF/VBR mode,
quality target, bitrate target, batch composition, GPU allowance, CPU allowance,
and deadlines. A valid solution must handle every request independently; it
must not assume that a batch contains repeated versions of one source video.

## Competition details

- [ ] Competition ID: `____________________________`
- [ ] Manifest path: `____________________________`
- [ ] Contender finalisation time (UTC): `____________________________`
- [ ] Competition end time (UTC): `____________________________`
- [ ] Repository: `https://github.com/____________________________.git`
- [ ] Prepared commit SHA: `____________________________`
- [ ] Miner hotkey: `____________________________`
- [ ] Operator/contact: `____________________________`

## 1. Confirm the competition contract

- [ ] Obtain the exact manifest from the competition operator.
- [ ] Confirm that `competition_type` is `COMPRESSION`. Upscaling competitions
  are not implemented.
- [ ] Confirm that the manifest uses the SDK/API contract supported by this
  checkout.
- [ ] Read the schedule, allowed GPUs, CPU limits, build timeout, batch size,
  minimum runtime/scoring timeout floors, output codec, scoring factors, and
  submission deadline.
- [ ] Confirm that the competition uses one query per physical source video.
  A full five-item batch must contain five distinct input videos.
- [ ] Plan for randomly assigned `CRF` or `VBR` queries. VMAF targets are selected
  from 85, 89, and 93; VBR queries also receive a 5, 8, or 10 Mbps target.
- [ ] Do not rely on the validator's dynamic deadline as a performance target.
  It is a maximum allowance, and actual wall time and cost are scored.

## 2. Prepare the miner host

- [ ] Complete the system, Python, wallet, PM2, and service setup in
  [`miner_setup.md`](miner_setup.md).
- [ ] Pull the operator-required code revision and activate the miner virtual
  environment.
- [ ] Install the competition SDK dependencies:

  ```bash
  pip install -r miner/requirements.txt
  ```

- [ ] Authenticate the local OS user to the Modal environment used for
  qualification:

  ```bash
  uvx modal setup
  ```

- [ ] Confirm the miner hotkey is registered and the axon is reachable on the
  target subnet.
- [ ] Confirm the miner process is running from the repository root and inference
  mode still works if it remains enabled.

## 3. Implement the solution

- [ ] Make competition changes under `miner/` and keep
  `competition_solution.json` at the repository root.
- [ ] Preserve the required `/health` and `/compress` routes.
- [ ] Accept batches containing one through five unique items.
- [ ] Accept validator-owned local inputs below `/evaluation-inputs`.
- [ ] Write only the assigned, new `.mp4` paths below `/output`.
- [ ] Return exactly one ordered result for each request item, containing only
  its assigned output path.
- [ ] Support both `codec_mode=CRF` and `codec_mode=VBR`, including the supplied
  VMAF and target-bitrate criteria.
- [ ] Produce a smaller AV1 MP4 while preserving source dimensions, timing,
  YUV pixel format, and square pixels.
- [ ] Run with `DISABLE_REMOTE_IO=true`. Do not fetch URLs, call external
  services, expose ports, or depend on runtime credentials.
- [ ] Reject path traversal, symlink escapes, duplicate output paths, and
  overwrites.
- [ ] Handle every video independently. Do not cache or reuse an encode merely
  because another item has similar criteria.
- [ ] Keep processing within the manifest's declared GPU/CPU limits and make no
  assumption about which allowed GPU the validator assigns.

## 4. Keep the export safe and reviewable

- [ ] Do not put `.env`, wallet data, GitHub tokens, Modal tokens, S3
  credentials, validator credentials, or other secrets in the solution.
- [ ] Do not include Git submodules, Git LFS dependencies, archives, compiled
  executables, opaque loaders, obfuscated/minified source, dynamic downloads,
  or code that executes submitted strings.
- [ ] Ensure all required dependencies are pinned or otherwise reproducible
  without private package credentials.
- [ ] Keep the repository readable enough for static review.
- [ ] Confirm the source export remains below the repository transfer limit and
  the built image is intended to remain below the manifest's container limit.

## 5. Run static preflight

- [ ] Run the manifest and repository preflight from the repository root:

  ```bash
  python scripts/competition_preflight.py \
    competitions/manifests/examples/compression-competition.json \
    --repository .
  ```

- [ ] Confirm the final JSON status is `ACCEPTED`.
- [ ] Resolve every rejection or review reason before continuing.

## 6. Prepare and qualify the exact export

- [ ] Create the sanitized standalone export:

  ```bash
  python miner/competition_sdk.py prepare \
    --repository your-github-user/private-compressor
  ```

- [ ] After any source change, rebuild the SDK-marked export with `--refresh`.
- [ ] Validate the exact export using the operator-provided manifest:

  ```bash
  python miner/competition_sdk.py validate \
    --repository your-github-user/private-compressor \
    --manifest competitions/manifests/examples/compression-competition.json
  ```

- [ ] Use `--modal-environment`, `--modal-gpu`, `--modal-cpu`,
  `--modal-cpu-limit`, or `--modal-timeout` only when the operator specifies
  different qualification resources.
- [ ] Confirm the Modal test proves blocked outbound networking, no attached
  secrets or identity token, read-only input, isolated output, localhost-only
  routes, AV1 media validity, and cleanup of its temporary Sandbox and Volumes.
- [ ] Record the `ACCEPTED` result, export path, tree checksum, resource
  configuration, estimated cost, and validation receipt.
- [ ] Publish within the receipt's default 24-hour validity period, or expect a
  new validation. Use `--revalidate` when a fresh run is required.

## 7. Publish the private repository

- [ ] Use a publishing credential that can create/update the private repository.
  Enter it only at the SDK's hidden prompt, or provide it through the environment
  variable selected with `--pat-env`.
- [ ] Publish the already-qualified export:

  ```bash
  python miner/competition_sdk.py publish \
    --repository your-github-user/private-compressor \
    --manifest competitions/manifests/examples/compression-competition.json
  ```

- [ ] For a repository that already exists, update it with a normal
  fast-forward child commit:

  ```bash
  python miner/competition_sdk.py publish \
    --repository your-github-user/private-compressor \
    --manifest competitions/manifests/examples/compression-competition.json \
    --use-existing \
    --refresh
  ```

- [ ] Confirm the SDK reports `PUBLISHED`, the repository is private, and the
  printed canonical HTTPS `.git` URL and commit are correct.
- [ ] Never force-push the competition repository.
- [ ] Create a separate, expiring, fine-grained, repository-scoped,
  contents-read-only PAT for validator submission. Do not reuse the publishing
  credential.

## 8. Configure competition participation

- [ ] Set the miner modes and submission configuration in the miner process
  environment:

  ```env
  MINER_MODES=inference,competition
  MINER_COMPETITION_TYPES=COMPRESSION
  MINER_COMPETITION_REPOSITORY_URL=https://github.com/your-github-user/private-compressor.git
  MINER_COMPETITION_GITHUB_PAT_ENV=MINER_COMPETITION_GITHUB_PAT
  MINER_COMPETITION_GITHUB_PAT=replace-with-read-only-expiring-token
  ```

- [ ] Keep `inference` in `MINER_MODES` if the miner should continue serving
  ordinary inference.
- [ ] Confirm the PAT appears in no URL, Git configuration, command-line
  argument, log, trace, W&B record, Redis value, SQLite row, or committed file.
- [ ] Restart the miner with its updated environment:

  ```bash
  pm2 restart video-miner --update-env
  ```

- [ ] Check the miner log for competition-mode startup and confirm inference
  handlers remain attached when dual mode is selected.

## 9. Enrollment and finalisation

- [ ] Keep the miner online throughout `ENROLLING`.
- [ ] Watch each validator poll result for `ACCEPTED`, `REJECTED`,
  `REVIEW_REQUIRED`, or `NOT_RECEIVED`, including its reason code and pinned
  revision.
- [ ] If rejected, fix the source, rebuild with `--refresh`, revalidate,
  republish, and keep the same configured repository URL unless intentionally
  changing repositories.
- [ ] If `REVIEW_REQUIRED`, resolve the stated ambiguity with the competition
  operator before the human-review deadline.
- [ ] Verify the validator has accepted and pinned the intended commit before
  `contender_finalisation_time`.
- [ ] Stop changing the repository after finalisation. The accepted revision is
  immutable for that competition.
- [ ] Revoke the read-only PAT as soon as the operator confirms submission
  finalisation and that the intended commit is pinned. Private archival uses
  the validator's pinned local copy and does not need the PAT.
- [ ] Retain the private repository until the operator confirms archival is
  complete.

## 10. Completion record

- [ ] Final accepted commit SHA: `____________________________`
- [ ] Validator submission revision: `____________________________`
- [ ] Validator status/reason: `____________________________`
- [ ] PAT revoked at (UTC): `____________________________`
- [ ] Final competition state/rank: `____________________________`
- [ ] Notes: `________________________________________________________`
