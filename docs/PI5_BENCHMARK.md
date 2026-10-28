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


## 12. Fast multi-seed optimization profiles: one submit, one collect

This is a **separate, reusable** profiled seed-sweep runner and does not
modify the 55k/5k full-development Indexed Job. It is intended for repeated
performance investigations such as the scoped-RNG Linear backward experiment.
It reuses the existing indexed-image CLI and checksummed
`BENCHMARK_ARCHIVE_V1` Pod logs. Python standard library + local kubectl
are sufficient: no new Kubernetes service, RBAC permissions or PVC.

**Publish an immutable ARM64 image first.** PR-only validation builds are
not published: the image must be published through the protected
big-refactor workflow after merging the candidate. Use the digest from
the publication summary (not a mutable commit tag or the example below).

Submit a 5-seed, validation-only, instrumented dense training sweep:

```sh
python3 experiments/pi5_seed_sweep.py submit \
  --job integers-pi5-rng-sweep-a \
  --namespace default \
  --image ghcr.io/philsupertramp/integers-benchmark@sha256:REPLACE_64_HEX_DIGEST \
  --seeds 1337,2026,2027,4242,42069 \
  --epochs 2 --train-limit 8192 --validation-limit 1042 \
  --batch-size 32 --lr-shift 7
kubectl -n default get job integers-pi5-rng-sweep-a -w
```

The script creates an Indexed Job with **all five completions scheduled
immediately but `parallelism: 1`**. This avoids benchmark Pods
running against each other on the Pi; it does not control unrelated cluster
workloads. Each seed runs integer then FP32 within a Pod. Node selectors
require a labelled physical Pi 5 and ARM64. A single global deadline scales
with the number of seeds (3600 seconds per seed in the overall Job budget);
Pods and logs are intentionally retained. Existing Job names are never
overwritten: submission uses `kubectl create`, not `apply`.
To inspect the complete Job without creating anything, append `--dry-run`.

After all completions succeed, **one command** retrieves every seed's raw
Pod log, verifies its gzip/base64 SHA256 archive against its JSON report,
enforces same-image/dataset/compiler and experiment configuration, and emits
aggregates with mean and sample standard deviation:

```sh
python3 experiments/pi5_seed_sweep.py collect \
  --job integers-pi5-rng-sweep-a --namespace default \
  --out-dir pi5-results/rng-sweep-a
cat pi5-results/rng-sweep-a/summary.md
```

Output files are `seed-<seed>.log`, `seed-<seed>.json`,
`summary.json`, and `summary.md`. The report includes
individual seed accuracy, phase durations, backward layer attribution,
mean/sample-SD across initializations, process RSS, image identity and
dataset-manifest SHA256. The collector discovers seed order, experimental
dimensions and image from the **actual Job spec**, so there is no second
`--seeds` argument to accidentally mismatch. It rejects unfinished,
missing, duplicate or failed Pods; mismatched ownership or seeds;
truncated/bad archives; test-set access; non-Pi hosts; parameter drift; and
existing output directories. No files are saved for a partially verified set.

For **seed-matched paired comparisons**, run another sweep with identical
seeds/configuration and a different published image and Job name, then pass
the original sweep output as baseline:

```sh
python3 experiments/pi5_seed_sweep.py collect \
  --job integers-pi5-rng-sweep-b \
  --out-dir pi5-results/rng-sweep-b \
  --baseline-dir pi5-results/rng-sweep-a
```

This additionally reports mean per-seed percentage changes for training,
backward, forward, updates and preparation, using the existing
`compare_pi5_backward.py` protocol gate. Baseline files must exist
for **every** matching seed. This is **not** an interleaved randomized A/B
test: distinct seeds estimate initialization sensitivity; genuine hardware
speed claims still need repeated seed-matched runs with temperature, clock,
governor and contention measurements. Profiled and unprofiled durations
must not be treated as interchangeable. No official MNIST test images are
opened in this workflow.

Offline tests:

```sh
python3 -m unittest discover -s tests -p 'test_pi5_seed_sweep.py' -v
```

## 13. Scoped PRNG multiplication investigation (#47)

The portable candidate uses `with_rng_state` inside `Linear::backward_one`
so its two dyadic products per connection draw from one local XorShift64
state. `mul_with_rng` shares the **same** exponent checks, exact i64 product,
stochastic-rounding implementation and i32 saturation as public `mul`.
The scoped guard commits consumed draws on normal exit or unwind. It does
**not** discard zero-product draws, change update order, skip input gradients,
or change the public `Module` API. A single caller thread must not invoke
other thread-local RNG functions or nest another scope inside this closure.

Run `cargo test --locked --lib` and `cargo test --locked --test dyadic_regression`;
scoped/unscoped unit oracles assert identical gradients and subsequent RNG
state, including the large-shift rounding branch. The existing Pi profile
measures complete backward kernels; this implementation has **no physical Pi
speed result until a new immutable ARM64 image is published and compared**.
A PR validation image built on a GitHub-hosted ARM64 runner tests the native
architecture but is not Raspberry Pi performance evidence.

Use a **new** validation-only K3s Job identity, and compare at least seed 1337
against commit `44ab8d02943ffe7fa87be4083abdfd9f0a41b0d3` (the original
same-scale-accumulation change). Pin image digest, dataset manifest and CPU
settings; capture thermal and clock conditions where possible. Run repeated,
interleaved A/B comparisons to distinguish optimization from contention and
thermal drift. Do not merge PR #9 on the basis of this experiment.


## 14. Scoped RNG in Linear forward — paired Pi benchmark protocol (#47)

The next portable optimization extends the existing `with_rng_state` lease
from `Linear::backward_one` to `Linear::forward_one`.
Each sample's original output-major / input-minor traversal stays unchanged,
including calls for zero-mantissa operands. `mul_with_rng` shares the
exact stochastic dyadic rounding logic; only RNG state acquisition moves
outside the repeated inner product loop. `add` aligns to the finer
exponent without consuming RNG; `requantize(acc, acc.s, ...)`
does not change the exponent and also cannot draw RNG. Do not nest a
thread-local RNG call inside the new scoped closure.

A frozen pre-scoping reference under
`src/nn/linear.rs::forward_scoped_rng_tests`
compares **exact** outputs, clipping flags, and the next RNG draw across
1×1, 5×3, and 64×32 matrices; shifts q=0,7,31,64,128; 4/8/31 output
bits; 17 consecutive samples per case, zero/signed-extreme/unequal-exponent
inputs, and saturation. Public single-sample and batch paths and their
clipping/output caches are also checked. Existing backward oracles stay.

The native ARM64 release-mode microbenchmark compares 64 forward calls
through `Linear(784→128)` with the frozen TLS baseline within the
**same binary**, and asserts outputs, clip flags and next RNG parity:

```sh
cargo test --locked --release --lib benchmark_forward_scoped_rng_vs_tls -- --ignored --nocapture
```

The hosted ARM64 result is only a diagnostic, **not** Raspberry Pi proof.

For physical Pi performance, use
`experiments/pi5_seed_sweep.py` and the prior 5-seed baseline
(commit `57332082bb04b1c0bfc0a6d8e0e69b8c42fc61d6`,
GHCR image digest
`sha256:bb839ebbf724e1791e897bedfe1ee62d5dc3fb67bf97bd6f5fb111b4be56a121`).
Reference means for 1337,2026,2027,4242,42069: integer forward
14.859s, backward 20.868s, total integer training 36.764s.
Use **identical ordered seeds** and workloads (2 epochs, 8,192/1,042,
batch32, lr_shift7, same Pi and 2-CPU quota), different immutable
candidate image/Job name, and original results as
`--baseline-dir` for accurate per-seed comparisons. Check
thermal state, clocks, scheduling and contention across interleaved
measurements. Candidate image is published **after** merging to
`big-refactor`, never by PR validation. The official test
set must remain unevaluated.

## 15. Training-only first-layer input-gradient elision (#47, PR #53)

### Why this work is unused

The first layer receives input pixels that are not trainable parameters.
`Sequential::backward_batch` used to compute their gradients anyway;
the training loop then discarded the result. Only the **first** layer may
omit the input gradient: intermediate layers still need it for earlier
trainable parameters. Public backward methods remain unchanged.

The new explicit opt-in `backward_batch_discard_input_grad` plus
timed equivalent skips the unused input-gradient tensor allocation and
accumulation for the first `Linear` only. The conservative
implementation preserves the original XorShift64 draws: shifts 1–63
consume one draw even on zero products; q=0 consumes none; q>=64 falls
back to actual multiplication because draw count depends on the product.
Parameter gradients, optimizer updates and following RNG remain equal.

Frozen backward oracle tests cover shifts 0,1,7,31,63,64,128; signed
extremes; mixed scales; saturation; multiple minibatches; optimizer
momentum; and repeated multi-layer training.

Native ARM64 release-mode *diagnostic* (not physical Pi performance):

```sh
cargo test --locked --release --lib benchmark_discard_unused_input_grad -- --ignored --nocapture
```

### Numerical differences are allowed with evidence

Exact seeded-output matching is **not** mandatory for future training
optimizations. The experimental no-draw variant entirely skips the
discarded multiplication's RNG consumption and may change subsequent
weight-gradient stochastic rounding, model updates and training
trajectories. It is currently **benchmark-only**, not used by training:

```sh
cargo test --locked --release --lib benchmark_relaxed_rng_elision -- --ignored --nocapture
```

This upper bound measures potential **speed**, not model quality.
Before allowing production numerical divergence, validate the revised
rounding policy and compare paired multi-seed training curves, held-out
validation accuracy and class recall, instability diagnostics, integer
and FP32 timings. Set non-inferiority thresholds **before** examining
outcomes; do not require any particular seed to reproduce identical
predictions, but require comparable or better overall quality and
performance. Do not tune against the official test set.

### Real Pi validation protocol

The previous five-seed Pi 5 baseline is commit
`b0ce98372e500a6b621abc0007e46ca2861fc4ab` and image
`sha256:2c8f32322dcb4916c38b09f3123fb4e0043a009f094881be99d395aa33c8adf9`.
Integer means: 32.555 s training, 10.792 s forward,
20.720 s backward, 17.616 s first Linear backward.
Match original image dataset manifest and seeds
1337,2026,2027,4242,42069; train/validation 8192/1042,
two epochs, batch32, lr shift7, 2-core quota. Use immutable
post-merge image and per-seed comparison, preferably repeated
interleaved A/B trials with temperature, clocks, throttling and
contention measurements. Never merge PR #9 because of a benchmark.

## 16. Interleaved physical Pi A/B gate (#54)

### Why we are doing this

The post-PR #52 and post-PR #53 sweeps used the same five MNIST seeds
but ran at different times; they do not establish a confidence interval
for a *hardware* speedup. For PR #53 we therefore preregister a paired,
**single-device mirrored ABBA/BAAB experiment** using only immutable images:

- A: PR #52, commit `b0ce98372e500a6b621abc0007e46ca2861fc4ab`,
  `ghcr.io/philsupertramp/integers-benchmark@sha256:2c8f32322dcb4916c38b09f3123fb4e0043a009f094881be99d395aa33c8adf9`
- B: PR #53, commit `cb36399f7bbacb6239490a0ff8fad50c06347c1f`,
  `ghcr.io/philsupertramp/integers-benchmark@sha256:1345a53c6976736524894cf7bb07f7ff5cb8392e8afdcb14a374b4db8a699bcc`

Both images contain the **same release binary compiler and benchmark runner**
and the same pinned MNIST dataset manifest
`8042b6e40973be57d479b8eb3673f39b89f1cdb11e05db3ab74fecd814023a5d`.
Keep 8192 training / 1042 validation, batch32, two epochs, lr shift7,
2-CPU quota and 3 GiB memory limit. Do **not** evaluate official test data.

### Execute on a machine with `kubectl` access to the actual Pi cluster

Run from a checkout containing `experiments/pi5_interleaved_ab.py`
(e.g. this PR's branch). No image build, privileged Pod, or namespace
mutation beyond creating new benchmark Jobs is required. Use a **fresh**
short lowercase run ID for each experiment; existing Jobs/results are
never deleted, reused without validation, or overwritten.

```sh
python3 experiments/pi5_interleaved_ab.py plan \
  --run-id ab52-53-01 --namespace default --dir pi5-results/ab52-53-01

# Inspect the immutable 20-trial plan before running.
cat pi5-results/ab52-53-01/plan.json

# Runs the serial Jobs, waits, validates each archived report and prints
# comparison results. Repeat this exact command after an interruption:
python3 experiments/pi5_interleaved_ab.py run \
  --dir pi5-results/ab52-53-01

# Offline recomputation (no cluster access):
python3 experiments/pi5_interleaved_ab.py analyze \
  --dir pi5-results/ab52-53-01
```

The tool generates exactly twenty independent one-seed Jobs:
`ABBA` for seeds 1337, 2027, 42069, and `BAAB` for 2026, 4242.
Each Job runs **both** integer and independent FP32 comparisons in the
existing order, and finishes before the next is created.
The tool preserves per-trial `seed-*.json`, `seed-*.log`, `summary.*`
and `trial.json` including Job/Pod/node UIDs. Source SHA, image digest,
cross-arm hardware, dataset, model configuration and numerical validity
are validated before a complete paired report is written. The 20 Jobs
remain in K3s so failure investigation is possible; no automatic cleanup.

On any failure, the CLI exits nonzero. The already verified trials
are retained. Fix the underlying cause or allocate a new run ID;
never change the plan after creation. A repeat `run` checks existing
Kubernetes Jobs and validated local artifacts before proceeding.
If the cluster is unavailable, `plan` and `analyze` still work.

### Analysis and preregistered acceptance

For each seed, average the two A and two B observations **within that
seed**. Across the five seed-block ratios B/A, report geometric mean and
the two-sided Student-t 95% confidence interval for mean log(B/A),
df=4. This estimates a version change while balancing linear drift
within each block; it is **not** a guarantee against nonlinear DVFS,
scheduler noise or environmental confounding.

Performance gate passes only if:

- Geometric mean integer training B/A <=0.95 (**>=5%** improvement)
  **and** 95% CI upper bound <1.00;
- Mean integer final validation accuracy B−A >=−1.0 percentage point,
  and no individual seed loses more than 3.0 percentage points;
- No nonfinite/extreme model outputs or nonfinite FP32 parameters,
  no test-set access, consistent pinning and exactly one physical Pi node;
- FP32 negative-control training drift <=5% in magnitude; no
  seed's integer peak process RSS rises by more than 10%.

A failure is labeled **inconclusive or regressed**, not silently converted
into an optimization success. The report also includes per-phase
forward/backward/first Linear, training and FP32 drift. Accuracy is
allowed to differ per seed if the overall quality gate is satisfied.

The immutable old images record CPU temperature at process boundaries;
they do **not** record CPU frequency, actual throttling or energy.
The new analyzer marks those fields `not_measured` rather than
inventing data. If the negative control or temperature reveals drift,
repeat with another run ID and investigate contention; obtain cgroup
throttling and continuous CPU-frequency telemetry in a **separate**
monitoring change before making claims about energy efficiency.
These five seed-blocks are a focused engineering gate, not a
large-scale proof of cross-device performance generalization.

### Keep tooling CI cheap

```sh
python3 -m unittest discover -s tests -p 'test_pi5_interleaved_ab.py' -v
python3 -m py_compile experiments/pi5_interleaved_ab.py
```

The dedicated seed-sweep tooling workflow checks these offline.
A real physical Pi result must be supplied from the completed cluster
run. It cannot be inferred from GitHub-hosted ARM64 CI.
