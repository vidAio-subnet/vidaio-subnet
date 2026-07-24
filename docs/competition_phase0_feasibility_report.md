# Competition mode Phase 0 feasibility report

Date: 2026-07-14
Scope: compression competition mode only
Decision: **Phase 0 is implemented but not cleared for production.**

## Result summary

| Gate | Result | Evidence and remaining work |
|---|---|---|
| 25 GB hostile-build limit | Partial | The exact 25,000,000,000-byte boundary accepts 25 GB and rejects 25 GB plus one byte. A POSIX hard file-size limit stopped a hostile writer at its configured boundary. The local Docker daemon was unavailable, and neither Docker's external daemon nor Modal's image builder inherits this process limit, so an end-to-end trusted-builder proof is still required. |
| Offline Modal runtime | Passed | A live dev Sandbox blocked direct-IP and HTTPS egress. No Modal account credential or OIDC identity token was present inside it. The CPU-only probe used bounded CPU/memory, no attached secrets or GPU, and a 60-second timeout; afterward the app was stopped with zero tasks. |
| Raw PAT boundary | Partial | The raw PAT is present only in the wire DTO. Representative application, Bittensor debug, PM2, Git environment, Redis, SQLite, W&B, and exception values are redacted or use a PAT-free persistent DTO. The test must be repeated through the production Bittensor submission handler and real configured log/export sinks when those Phase 1 components exist. |
| Billing granularity and allocation | Passed | Live workspace access returned a valid hourly report (zero rows were expected for the newly created dev environment). The API groups by Modal object, resource, and requested tags rather than native evaluation-item cost. The implemented allocation assigns a batch bill by active-runtime share, uses an equal share if every runtime is zero, and preserves the exact billed `Decimal` total. |
| Competition formula | Passed | Tests prove 60% length-weighted compression effectiveness, 25% per-item contender-relative cost efficiency, and 15% length-weighted completion coverage. The cheapest valid result for an item receives cost efficiency 1.0; failed and invalid results receive zero and cannot define that minimum. The Phase 4 scorer supplies zero effectiveness below the VMAF floor. Longer inputs carry greater logarithmic weight, terminal failures contribute zero, and every contender must have one terminal row for every manifest input. |

The dependency-free local runner still reports its live-only gate as blocked by design. The separate machine-readable live result is in `docs/competition_phase0_modal_report.json` and passes both network isolation and billing access. Overall production readiness remains false because the trusted hostile-builder proof and actual protocol/sink PAT canary are not complete.

## Implemented artifacts

- `vidaio_subnet_core/competition/phase0.py`: policy gates, redaction boundary, approved scoring formula, and bill allocation.
- `scripts/competition_phase0.py`: dependency-free local runner and JSON report generator.
- `scripts/competition_modal_phase0_probe.py`: bounded live Modal network-isolation and billing capability probe.
- `tests/competition/test_phase0.py`: local reproducible contract tests.
- `docs/competition_phase0_modal_report.json`: sanitized live Modal evidence.

## Reproduce the local evidence

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.competition.test_phase0 -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/competition_phase0.py
```

On the recorded workstation, all 12 unit tests passed. The local gate runner proved the formula and local primitives and reported partial results for image enforcement and PAT integration. The live Modal gate was then executed separately and passed.

## Run the live Modal gate

Do not paste the token into logs, source files, CLI arguments, or this report. Supply a dev-workspace token through the process environment and run:

```bash
export MODAL_TOKEN_ID='...'
export MODAL_TOKEN_SECRET='...'
env UV_CACHE_DIR=/private/tmp/vidaio-modal-phase0-uvcache \
  uv run --python 3.12 --with 'modal>=1.4.0' \
  python scripts/competition_modal_phase0_probe.py --environment dev
unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
```

The probe creates one short-lived CPU-only Sandbox with no attached secrets or GPU. It attempts direct-IP and HTTPS egress, verifies account credentials and an OIDC identity token are absent inside the Sandbox, and then requests the last two complete hourly billing intervals. Billing reports may require a Team or Enterprise workspace. If the workspace has no `dev` environment, create it first or explicitly select the intended non-production environment.

The probe was API-checked and executed against Modal SDK 1.5.2 on 2026-07-14. A dedicated `dev` environment was created because the workspace initially contained only `main`. The runtime network probe was repeated after Phase 3 with explicit DNS, direct-IP, and HTTPS attempts; all three failed while Modal account credentials remained absent. Relevant current Modal documentation: [network controls](https://modal.com/docs/guide/sandbox-networking), [resource limits](https://modal.com/docs/guide/sandbox-resources), [workspace billing reports](https://modal.com/docs/sdk/py/latest/modal.Workspace), and [token environment variables](https://modal.com/docs/sdk/py/latest/modal.config).

## Production blockers

1. Select and prove a trusted build mechanism that enforces the 25 GB limit while layers are being created. Post-build inspection is not sufficient because it does not cap hostile build-time disk or spend. Modal's public Image API does not expose a dependable arbitrary image-size quota/measurement for this gate.
2. Repeat the PAT canary through the real Bittensor submission handler and configured PM2, Git, Redis, SQLite, W&B, and exception paths after Phase 1 creates that handler.

Phase 1 contract work may continue behind the default-off feature flag, but no contender build or production competition should run until the trusted-builder gate is cleared. The PAT integration repeat is a mandatory pre-production gate once the submission path exists.
