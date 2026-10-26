# Raspberry Pi 5 reproducible training benchmark (issue #18)

This is an **infrastructure and measurement** workflow for the `big-refactor`
candidate. It runs the existing `examples/held_out.rs` portable integer
training path and its **independent scalar FP32 reference**. It neither adds
NEON nor alters the approved integer-only training boundary (ADR-002).
A Docker build, QEMU run, or x86_64 run is **not a Pi 5 measurement**.

## 1. Build locally / image contents

Build natively on ARM64 or with Buildx emulation (emulation is *validation only*):

```sh
git switch big-refactor
# After this PR is reviewed and merged; before merge use this issue branch.
docker buildx build --platform linux/arm64 --load \
  -f Dockerfile.benchmark \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  -t integers-pi5:local .
docker image inspect integers-pi5:local --format '{{.Size}} bytes (local uncompressed image layers)'
docker image inspect integers-pi5:local --format '{{.Id}}'
```

The Dockerfile builds the locked Rust release example, records `rustc
--version` and the build flags, and uses a slim Python 3 runtime with **no
Rust toolchain**. An isolated dataset stage fetches the four canonical MNIST
IDX gzip files from the CVDF mirror, verifies pinned compressed SHA256,
validates uncompressed checksums against the archived #15 baseline, and
converts **once** to a common signed-i8 pixel representation. Both training
backends read precisely the same pixel mantissas and byte-for-byte labels.
The signed mantissa is interpreted with dyadic shift 7; FP32 independently
decodes the same inputs. Targets retain the existing one-hot 127/128 convention.
No weights, backend-specific input augmentations, or tuning are included.

`/opt/dataset` contains four prepared files, `dataset-manifest.json`,
`dataset-manifest.sha256`, and `DATASET_NOTICE.txt`. Their SHA256 checks
run **before each applicable benchmark** (train artifacts in validation-only;
train and official-test artifacts in explicitly selected final-test mode);
tampering fails closed. Validation-only mode never opens or hashes official
test bytes. Dataset source,
compressed and IDX hashes, preprocessing, 55,000/5,000 stratified split
(seed `81985529216486895`), and original official 10k test identity are
machine-readable. Manifest generation does not include wall time or host
information. Preprocessed data is read-only, stored without duplicated IDX
files, and never fetched at Job startup. License information and attribution:
[MNIST via Keras](https://keras.io/api/datasets/mnist/)
(CC BY-SA 3.0), [original authors and CVDF mirror](https://github.com/cvdfoundation/mnist).
The image includes a modification and license notice. Do not transfer the
dataset's license to the (separately undecided) Rust library licensing policy.

On PR #43's original GitHub Actions ARM64/QEMU build (2026-10-25), `docker image inspect .Size` reported **67,416,792 bytes** for the locally loaded image. This is a reproducible CI observation for that image build, **not** compressed GHCR transfer size or measured Pi process memory.

Image **compressed registry transfer size** must be measured after publish;
do not equate it with local `docker inspect .Size`, process RSS, or memory
limits. Record pull duration separately:

```sh
time docker pull ghcr.io/philsupertramp/integers-benchmark@sha256:REPLACE_DIGEST
docker image inspect ghcr.io/philsupertramp/integers-benchmark@sha256:REPLACE_DIGEST --format '{{.Size}} bytes'
```

## 2. GitHub Actions publishing

Workflow: `.github/workflows/pi5-benchmark-image.yml`.

The image pipeline uses GitHub-hosted native `ubuntu-24.04-arm`
runners for both validation and publication; these are **not** physical
Raspberry Pi 5 devices, so their timings are not Pi measurements.
Buildx persists its intermediate layers to the GitHub Actions cache
(`type=gha,scope=pi5-benchmark-arm64,mode=max`), including the separate
Rust dependency-warmup layer and checksum-verified prepared MNIST stage.
The publish job restores this cache rather than recompiling every
layer through QEMU. Cache population is best-effort and may miss after
eviction or dependency changes; the first run is a cold build. The
publish job still performs a separate build, and a cached build is not
proof of byte-for-byte equivalence to the validation image. Source
changes recompile the application crate but should reuse dependency
compilation; Cargo lockfile/toolchain changes invalidate the warmup layer.


PRs targeting `big-refactor`: run Python integrity tests, build the ARM64
image using Buildx/QEMU, and run a **tiny, offline, paired** native ARM64 smoke with
`--network none`. **No PR build pushes.**

A **successful** validation build on a push to `big-refactor` (or explicitly
dispatched workflow on that branch) allows the separate GHCR publication job.
Only that job has `packages: write`; authentication uses the ephemeral
`GITHUB_TOKEN`. Commit-SHA tags take the form

```text
ghcr.io/philsupertramp/integers-benchmark:sha-<FULL_GIT_COMMIT_SHA>
```

**Do not assume an image exists until the publish job succeeds.** Take the
published `sha256:...` from the workflow summary or GHCR, then deploy using
`ghcr.io/philsupertramp/integers-benchmark@sha256:...`. Digests are the
actual immutable byte identity; tags can theoretically be overwritten.
If GHCR package visibility defaults to private, the maintainer must enable
public package visibility or supply an authorized `imagePullSecret` to the
Job; no registry secret is embedded in the repository.

For a self-built private image, `docker buildx build ... --push` to an
authorized private registry and update the Job image reference. Rights and
distribution policy must be checked before any external redistribution.

## 3. Select and label the **actual** Raspberry Pi 5

```sh
kubectl get nodes -o wide
kubectl describe node REPLACE_WITH_PI5_NODE
# Verify physical device identity out of band (SSH to the Pi):
cat /proc/device-tree/model
uname -m
# Only label after confirming it is a Raspberry Pi 5 and ARM64:
kubectl label node REPLACE_WITH_PI5_NODE benchmark.integers.dev/pi5=true --overwrite
kubectl get nodes -l 'kubernetes.io/arch=arm64,benchmark.integers.dev/pi5=true' -o wide
```

ARM64 architecture alone **does not** identify a Pi 5. Do not copy a node
name, namespace or storage class from these examples into your cluster without
checking it. Job `nodeSelector` requires both labels; pod scheduling fails
rather than silently running on another ARM64 machine. Record OS/kernel,
firmware, core count, frequency governor, ambient and load conditions on the
host separately; do not assume the container can read all host sensors.

## 4. Configure the Job (no dataset PVC required)

Copy `deployments/pi5/benchmark-job.yaml` and replace the image in **both**
`image` and `BENCHMARK_IMAGE` env with the exact same **digest**. The
Downward API does not expose the running container digest; that env variable
is a *declared image reference*, not an independently discovered identity.

Default workload: integer + FP32, seed 1337, dense 784→128→128→10, one
validation-only epoch, 128 fixed training and 128 validation images sampled
after the canonical split, batch 32, LR shift 7. This is a **feasibility
test**, not an accuracy threshold or a full MNIST experiment.
The initial Job requests 1 CPU/1 GiB and limits 2 CPU/3 GiB; adjust based on
Pi resources and competing workloads. CPU scheduling, governor, kernel
thermal throttling, and cluster contention can invalidate fair timing.

Use `args` to select another predeclared seed, `--architecture cnn`, epochs,
or budget. **CNN can be slow on a Pi.** The existing CNN is
Conv(1→4,3) → ReLU → pool → Conv(4→8,3) → ReLU → pool → Dense(64) →
Dense(10), matched to a separate FP32 implementation. Do not rerun hours of
CNN/full MNIST in routine CI.

Use `--validation-only` with `--train-limit`/`--validation-limit` for
development and timing studies (official test **not opened/evaluated**).
An intentionally frozen full-data regression removes both sample limits and
replaces `--validation-only` with `--final-test`, e.g. `--epochs 20`
and the predeclared 1337/2027/4242 seeds as separate Jobs. The official
10k MNIST test was already consumed by #15 and #35 and is **not pristine**
for an independent generalization claim. Never select hyperparameters using
its scores. `activeDeadlineSeconds` must be raised if a justified large run
needs more than 3600 seconds.

The `/results` mount is a writable `emptyDir`, scoped to the Job pod; it
is not durable after pod deletion. To keep files across restarts, replace
`emptyDir` with a writable PVC **for results only** and arrange volume
ownership to permit UID/GID 65532. The container is non-root, non-privileged,
has no writable root filesystem, and needs no hostPath or dataset PVC.

For an approved external/prepared dataset: mount a read-only volume over
`/opt/dataset` or use `--data-dir /another/path`. It must contain exactly
the four prepared input/label files and their manifest/hash, adhering to the
documented schema/stratified protocol; the launcher rejects wrong checksums.
For a *different dataset protocol*, implement a separate validated version
rather than disguising non-MNIST bytes under the MNIST identifier.

## 5. Run and read logs

```sh
# Choose an existing namespace; no namespace is hard-coded.
NS=default
kubectl -n "$NS" apply -f deployments/pi5/benchmark-job.yaml
kubectl -n "$NS" get pods -l app.kubernetes.io/name=integers-pi5-benchmark -o wide
kubectl -n "$NS" logs -f job/integers-pi5-benchmark --timestamps
kubectl -n "$NS" wait --for=condition=complete job/integers-pi5-benchmark --timeout=3600s
```

Progress logs show each backend/epoch and end with one `BENCHMARK_RESULT=`
JSON line summarizing validation, legitimate test metrics, wall time,
dataset-loading time, and process RSS. An error causes nonzero Job exit.
Nothing downloads, preprocesses, or accesses the network at runtime;
`docker run --network none` is used by CI to test this property after the
image has been built/pulled.

## 6. Retrieve complete JSON from logs after Job completion

**The default results volume is an ephemeral `emptyDir`. Once the Job's
only container exits, `kubectl exec` cannot access its files, even if the
completed Pod still exists.** Older instructions incorrectly suggested that
approach. The image now prints a SHA256-verified, gzip-compressed, chunked
archive of the complete `benchmark-result.json` to standard output after
the ordinary human-readable `BENCHMARK_RESULT=` summary line. Chunk size
is deliberately bounded for Kubernetes/CRI log systems. Both validation-only
and final-test results are included; the archive appears only after both
backend runs and report verification finish.

```sh
NS=default
mkdir -p pi5-results
kubectl -n "$NS" logs job/integers-pi5-benchmark > pi5-results/job.log
python3 experiments/extract_pi5_results.py pi5-results/job.log \
  --output pi5-results/benchmark-result.json
python3 -m json.tool pi5-results/benchmark-result.json >/dev/null
```

The extractor validates chunk count/order, uncompressed size, SHA256 and JSON
integrity; a truncated Job log is an **error**, never a silent partial result.
For a complete final result, keep the completed Job Pod and its logs until
you have archived the output. Logs are not guaranteed persistent forever:
log rotation/retention policies can remove them. For long-running experiments
that need stronger retention, mount a real **results PVC** or upload reports
through a separately authorized collector. The embedded input dataset still
requires no PVC.

For **older images** (before this archive change), only the compact
`BENCHMARK_RESULT=` log line survives by default. Full JSON files from a
finished, single-container Job with only `emptyDir` cannot be retrieved
using `kubectl exec`. Rerun with the updated image or use a results PVC.

A run emits `benchmark-result.json` (summary and both full records) plus
one per-backend JSON. The schema and data-integrity tests live in
`tests/test_pi5_dataset.py`. Important result fields:
`git_commit_sha`, `image_reference`, `rustc`, `build_options`, CPU
architecture/OS/hardware identity, `dataset_manifest_sha256`, dataset
sources/preprocessed artifact hashes/split definitions, seed, model and
hyperparameters, train/validation/test sample counts, validation curve and
accuracy, `test_evaluated` and nullable `test_accuracy_pct`,
`dataset_loading_ms`, `model_initialization_ms`, per-epoch
`training_ms`/`validation_ms`, `test_evaluation_ms`,
`wall_ms`, `process_elapsed_ms`, and `peak_rss_kb`.

## 7. Measurement scope and reproducibility

- **Process memory:** `peak_rss_kb` uses Linux `/proc/self/status`
  `VmHWM` in each fresh backend subprocess (the baseline's established
  whole-process peak methodology). It is **not** Kubernetes/container memory.
  `machine.memory_limit_bytes` records a cgroup **limit**, never a
  measured allocation. Container peak is `not_measured`.
- **Timing:** `entrypoint_setup_ms` is manifest/hash verification from the
  Python launcher start, **not container startup or image-pull time**.
  `dataset_loading_ms` is data load, deterministic split selection and
  optional subsampling inside the Rust process; `model_initialization_ms`
  is separate. Rust `wall_ms` spans epochs (training+validation) and optional
  test evaluation. `process_elapsed_ms` covers one entire Rust process.
  No time from downloading/preprocessing (build-only) is included.
- **Hardware:** `machine.hardware_model` uses device-tree when visible or
  `/proc/cpuinfo` fallback. Temperatures are read only if container sensors
  are exposed and otherwise `not_measured`. Firmware throttling, CPU
  frequency and energy consumption are explicitly `not_measured` by default;
  capture `vcgencmd get_throttled` and `vcgencmd measure_temp` on the Pi
  separately before/after if needed, with timestamps and permissions.
  No power or energy benefit is claimed.
- **Comparability:** Every backend executes with exactly the same image,
  checked manifest, original labels, canonical 55k/5k split algorithm/seed,
  selected samples, input shift, model seed, architecture, training budget,
  batch and update schedule. FP32 has independent scalar kernels and
  continuous operations; integer uses dyadic quantization and STE.
  Baseline code maturity differs, so ratios are **not intrinsic hardware**
  instruction efficiency comparisons. The second sequential run may see a
  warmer CPU/page cache than the first; repeat paired experiments, reverse
  order in a separately reviewed study, and log governor/thermals/cluster load.
- **Same bytes, different executions:** Pin the image **digest**, label the
  same physical Pi, retain the full JSON with manifest hash and raw source
  hashes, specify a stable seed/budget/configuration, and keep CPU limits and
  ambient conditions comparable. Running several seeds adds model variance;
  it does not multiply the independent official test population.

## 8. Fast verification (no real Pi claimed)

```sh
python3 -m unittest discover -s tests -p 'test_pi5_dataset.py' -v
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo check --locked --all-targets
# Native ARM64 build + offline smoke runs in the dedicated GH Actions workflow.
```

All ARM64 CI timings are **hosted ARM64 diagnostics only** (not Pi 5 measurements). Report actual Pi 5
training accuracy, duration, process RSS, sensor availability and controlled
comparison results **after** running the versioned image on the real device.

## 9. Submit all full-development-set seeds with one Indexed Job

Use `deployments/pi5/benchmark-full-validation-indexed.yaml` for the
**55,000 train / 5,000 validation, 5 epoch**, dense-MLP development-set
benchmark with seeds 1337, 2027 and 4242. The independent integer and
FP32 backends run sequentially inside each seed Pod. Kubernetes submits
all 3 completions at once but starts **only one Pod at a time**
(`completionMode: Indexed`, `completions: 3`, `parallelism: 1`).
This avoids overlap between the three heavy benchmark Pods on the same Pi.
The Kubernetes controller injects `JOB_COMPLETION_INDEX` (0, 1, 2);
the runner selects the matching seed and rejects missing/out-of-range values.
The single-run smoke Job above is unchanged.

The experiment stays **validation-only**; the official 10,000 MNIST test
observations are not opened/evaluated. The 14,400-second Job deadline covers
the ENTIRE three-seed series, rather than each Pod. With `backoffLimit: 0`,
a failed seed stops the overall study; preserve logs for diagnosis.
No TTL is set; Pods/logs survive completion until explicitly removed or
log retention rotates them.

Replace BOTH image references (`image` and `BENCHMARK_IMAGE`) with the
same **published ARM64 image digest** built from the commit containing
`--seed-list`. Confirm Pi node labels and sufficient free resources.

```sh
NS=default
kubectl -n "$NS" apply -f deployments/pi5/benchmark-full-validation-indexed.yaml
kubectl -n "$NS" get job integers-pi5-full-validation -w
# Separately inspect the three sequential Pods:
kubectl -n "$NS" get pods -l batch.kubernetes.io/job-name=integers-pi5-full-validation -w
# Once all three have completed (could be hours):
kubectl -n "$NS" wait --for=condition=complete \
  job/integers-pi5-full-validation --timeout=14400s
```

Then retrieve **all three complete, verified JSON reports and their logs**
in one command:

```sh
python3 experiments/collect_pi5_indexed.py \
  --job integers-pi5-full-validation \
  --namespace "$NS" \
  --out-dir pi5-results/full-development
```

The collector lists only this Job's Pods, resolves each seed from the
completion-index **annotation** (works without relying on the optional
index label or Pod name parsing), requires all 3 Pods to have succeeded,
extracts each checksum-verified log archive, and verifies the paired backend
seed/index and validation-only 55k/5k protocol. It fails on missing,
unfinished, duplicate or invalid results rather than silently accepting
partial experiments. Logs remain subject to cluster retention. Archive them
before deleting a Pod.

Changing seeds requires changing `completions`, `--seed-list`, and the
collector's `--seeds` together (the supplied collector specifically
validates the predeclared 55k/5k development workload). Use a *new Job name*
for a new experiment: important fields of an existing Kubernetes Job are
immutable.

Serialization prevents these benchmark Pods competing with **each other**,
but not with unrelated K3s workloads or the Pi's CPU governor and thermals.
The integer→FP32 order still requires separate control experiments for
an order-independent speed comparison; do not infer energy savings.

## 10. Opt-in training path profiling (issue #18)

Use `deployments/pi5/benchmark-profile-job.yaml` with the **same published
ARM64 image digest in both image references**, built from the commit that adds
`--profile-training`. This runs one fixed seed (1337), 8192 training and
1042 validation observations, dense MLP, batch 32, 2 epochs, *validation
only*. It does NOT touch official test data and does NOT change the normal
unprofiled three-seed Indexed Job. Do not compare profiled wall times to
unprofiled wall times as if overhead were zero.

```sh
kubectl apply -f deployments/pi5/benchmark-profile-job.yaml
kubectl wait --for=condition=complete job/integers-pi5-training-profile --timeout=3600s
kubectl logs job/integers-pi5-training-profile > pi5-training-profile.log
python3 experiments/extract_pi5_results.py pi5-training-profile.log \
  --output pi5-training-profile.json
python3 experiments/summarize_pi5_profile.py pi5-training-profile.json
```

Each run's per-epoch JSON includes `training_profile` (null without the
flag). In opt-in mode it contains `training_wall_ns`, `batches`,
`samples`, `preparation_ns` (input/target materialization),
`zero_grad_ns`, `forward_ns`, `loss_metrics_ns` (prediction, loss
gradient), `backward_ns`, `update_ns` and `other_ns`
(shuffle, Python-unrelated Rust loop/timer overhead and unaccounted work).
All timings use Rust `Instant` wall-clock nanoseconds, **not CPU cycles**.
Integer calls forward/backward **per batch**; the scalar FP32 reference does
so **per individual sample**, so measured percentages indicate high-level
where time is spent but do not isolate architecture-specific primitive
efficiency. No arithmetic, clipping, update schedule or input order is
modified. The ordinary benchmark path remains uninstrumented.

The launcher refuses missing/malformed profiling records. Its SHA256 archive
and extractor are unchanged. CI runs the same small synthetic/prepared-data
fixture both with and without profiling and compares all accuracy predictions
and confusion matrices exactly, as well as the other numerical metrics.
Run the benchmark on the actual Pi for performance numbers: QEMU timings
are *not* Pi evidence. The profile's self-time includes stopwatch overhead;
`other_ns` is an unassigned residual, not a concrete hotspot. For a
lower-level diagnosis of the dominant phase, follow up with targeted CPU
sampling (`perf`/flamegraph if permitted) and hardware clock measurements.
Profile runs do not measure throttling, energy or container peak memory.

## 11. Portable backward accumulation optimization and per-layer attribution (#47)

The `big-refactor` candidate's `Linear::backward_one` now takes a **same
dyadic scale** fast path for gradient accumulation. When scales match,
`add(a,b)` is exactly `(a.v.saturating_add(b.v), a.s)` without a rounding
RNG draw; otherwise the original `add()` is invoked unchanged. Both weight
gradient and input gradient are still computed in the **same order** with the
same `mul()` calls. No public gradient return values, clipping, tensor
precision, or optimizer behavior are removed. A frozen pre-optimization
backward reference and exhaustive mixed-scale/saturation fixtures check
bit-identical gradients, accumulators and next RNG state.

With `--profile-training`, integer records also expose
`training_profile.backward_layers`: array of
`{index, name, elapsed_ns}`, indexed in *forward layer order*. The model
executes these timed calls in reverse order. These measurements are
nested inside `backward_ns` and include timer overhead; **do not sum them
again when computing total training**. There are five layer positions:
0 Linear (784→128), 1 ReLU, 2 Linear (128→128), 3 ReLU,
4 Linear (128→10). The existing FP32 sample-wise reference returns an
empty backward-layer list. The summary script prints layer attribution when
available.

To evaluate speed without clobbering the **already collected** physical
Pi baseline `profile.json` from commit `0539065a97bfe75fbfb7d7c1eb1171527c0bdd28`,
publish the new ARM64 image from a merged branch and replace both image
references in `deployments/pi5/benchmark-profile-optimized-job.yaml`
with the identical immutable digest:

```sh
kubectl apply -f deployments/pi5/benchmark-profile-optimized-job.yaml
kubectl wait --for=condition=complete job/integers-pi5-training-profile-optimized --timeout=3600s
kubectl logs job/integers-pi5-training-profile-optimized > profile-optimized.log
python3 experiments/extract_pi5_results.py profile-optimized.log \
  --output profile-optimized.json
python3 experiments/summarize_pi5_profile.py profile-optimized.json
python3 experiments/compare_pi5_backward.py profile.json profile-optimized.json
```

Match the exact seed 1337, 2 epochs, 8192 train / 1042 validation,
batch 32, LR shift 7, dataset manifest, compiler/release flags, node,
CPU quota and hardware conditions. The comparison script rejects
mismatched run settings, wrong backend order and official-test access.
It reports relative phase timings, accuracy and process RSS rather than
declaring success automatically. A measured Pi speed improvement still
requires reproducing across runs and checking thermal/frequency/host
contention; CI/QEMU timings prove neither acceleration nor power savings.
If the optimization has no benefit, revert it rather than changing
semantics for a speculative improvement.
