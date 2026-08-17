# Inference Miner Checklist

Use this checklist to configure a miner for ordinary validator inference. Complete
the host setup in [`miner_setup.md`](miner_setup.md) first. Competition enrollment
is a separate workflow; use
[`miner_competition_checklist.md`](miner_competition_checklist.md) for that.

An inference miner advertises exactly one processing task to validators. Starting
an upscaling or compression service does not change the advertised task.

## 1. Choose the advertised inference task

- [ ] Decide whether this miner will receive `UPSCALING` or `COMPRESSION`
  inference work.
- [ ] Open `neurons/miner.py` and find the module-level `warrant_task` setting
  near the top of the file.
- [ ] For an upscaling inference miner, keep:

  ```python
  warrant_task = TaskType.UPSCALING
  ```

- [ ] For a compression inference miner, change it to:

  ```python
  warrant_task = TaskType.COMPRESSION
  ```

- [ ] Confirm that `MINER_MODES` includes `inference`. For example:

  ```env
  MINER_MODES=inference
  ```

  Use `MINER_MODES=inference,competition` only if this process should also
  participate in competitions.

`MINER_MODES` controls which protocol handlers are enabled; it does not select
the advertised inference task. The validator reads `warrant_task` through
`TaskWarrantProtocol` and uses it to place the miner in the upscaling or
compression inference cohort.

## 2. Configure the shared miner requirements

- [ ] Install the repository and Python dependencies as described in
  [`miner_setup.md`](miner_setup.md).
- [ ] Confirm the wallet hotkey is registered on the target subnet.
- [ ] Create `miner/.env` from `miner/.env.template` and keep all secrets out of
  Git.
- [ ] Configure the S3-compatible storage variables used to upload processed
  outputs.
- [ ] Create `MINER_SHARED_DIR` and ensure the miner process and local processing
  container can both read and write it.
- [ ] Choose one processing backend:
  - `MINER_PROCESSING_BACKEND=http` for local Docker services.
  - `MINER_PROCESSING_BACKEND=modal` for deployed Modal workers.
- [ ] Confirm the miner's public axon port is reachable while processing service
  ports remain localhost-only.

## 3. Start the selected processing service

### Upscaling inference

- [ ] Start exactly one upscaling implementation from the `miner` directory:

  ```bash
  docker compose --profile upscaling-video2x up -d upscaling-video2x
  ```

  or:

  ```bash
  docker compose --profile upscaling-ffmpeg up -d upscaling-ffmpeg
  ```

- [ ] Set `MINER_UPSCALING_SERVICE_URL` to the selected service. Video2X uses
  `http://localhost:8003` by default; FFmpeg upscaling uses
  `http://localhost:8005`.
- [ ] Confirm the selected service is healthy:

  ```bash
  curl -sf http://localhost:8003/health
  curl -sf http://localhost:8005/health
  ```

  Only run the check for the implementation you started.

### Compression inference

- [ ] Change `warrant_task = TaskType.UPSCALING` to
  `warrant_task = TaskType.COMPRESSION` in `neurons/miner.py`. Starting the
  compression service without this edit leaves the miner advertised as an
  upscaling miner.
- [ ] Start the compression service from the `miner` directory:

  ```bash
  docker compose up -d compression
  ```

- [ ] Confirm `MINER_COMPRESSION_SERVICE_URL` points to the service. Its default
  is `http://localhost:8004`.
- [ ] Confirm the service is healthy:

  ```bash
  curl -sf http://localhost:8004/health
  ```

### Modal inference backend

- [ ] Deploy the appropriate workers using the instructions in
  [`miner/README.md`](../miner/README.md#modal-serverless-workers).
- [ ] For upscaling, set `MINER_MODAL_UPSCALING_FUNCTION` to
  `upscale_video2x` or `upscale_ffmpeg`.
- [ ] For compression, keep `MINER_MODAL_COMPRESSION_FUNCTION=compress` unless
  the deployment uses another compatible router entrypoint.
- [ ] Confirm the miner process has valid Modal authentication and can resolve
  the configured app and function.

The `warrant_task` edit is required for compression with either backend.

## 4. Validate before going live

- [ ] Check that the miner source still parses:

  ```bash
  python3 -m py_compile neurons/miner.py
  ```

- [ ] From the repository root, start the miner using one of the following
  methods after replacing every bracketed value.

  Direct process:

  ```bash
  python3 neurons/miner.py \
    --wallet.name [Your_Wallet_Name] \
    --wallet.hotkey [Your_Hotkey_Name] \
    --subtensor.network finney \
    --netuid 85 \
    --axon.port [port] \
    --logging.debug
  ```

  PM2-managed process:

  ```bash
  PYTHONPATH=. pm2 start neurons/miner.py \
    --name video-miner \
    --interpreter python3 -- \
    --wallet.name [Your_Wallet_Name] \
    --wallet.hotkey [Your_Hotkey_Name] \
    --subtensor.network finney \
    --netuid 85 \
    --axon.port [port] \
    --logging.debug
  ```

- [ ] If PM2 already manages the process, restart it after changing
  `warrant_task` or environment variables:

  ```bash
  pm2 restart video-miner --update-env
  ```

- [ ] Check the miner log for `TaskWarrantRequest` messages. A newly started
  process must answer them with the task selected in `warrant_task`.
- [ ] Check for the matching workload log:
  - Upscaling: `Receiving ... Request` followed by a returned processed URL.
  - Compression: `Receiving CompressionRequest` or `Receiving CompressionJob`
    followed by a successful result.
- [ ] Confirm a processed output can be uploaded and its returned presigned URL
  remains valid for the configured expiry.
- [ ] Watch for service timeouts, HTTP 429 backpressure, failed downloads,
  failed uploads, or storage cleanup errors.

## 5. Record the live configuration

- [ ] Advertised task: `UPSCALING` / `COMPRESSION`
- [ ] Miner commit SHA: `____________________________`
- [ ] Processing backend: `http` / `modal`
- [ ] Processing implementation/function: `____________________________`
- [ ] Wallet and hotkey: `____________________________`
- [ ] Network and netuid: `____________________________`
- [ ] Axon port: `____________________________`
- [ ] Health check completed at (UTC): `____________________________`
- [ ] Notes: `________________________________________________________`
