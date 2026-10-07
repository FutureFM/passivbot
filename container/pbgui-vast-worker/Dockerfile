# PBGui Vast GPU worker built from this PB8 fork.
#
# Mirrors PBGui's setup/vast_gpu_benchmark/Dockerfile + Dockerfile.calibration, but
# installs the checked-out fork instead of cloning upstream. Build context layout
# (prepared by .github/workflows/pbgui-vast-worker.yml):
#   passivbot/        full checkout of the fork at PB8_REVISION (with .git)
#   cloud_worker.py   PBGui's worker, from the PBGui release that will drive rentals
FROM rust:1.90-bookworm AS rust
FROM python:3.12-bookworm AS build
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
ARG PB8_REVISION
COPY passivbot/ /opt/passivbot/
RUN test -n "$PB8_REVISION" \
    && cd /opt/passivbot && test "$(git rev-parse HEAD)" = "$PB8_REVISION" \
    && git rev-parse HEAD > /opt/pb8-revision \
    && python -m pip install --no-cache-dir '.[full,gpu-cuda]' \
    && python -m pip freeze > /opt/pb8-pip-freeze.txt
# pip builds the extension without PB8's stamping step; record the source fingerprint.
RUN cd /opt/passivbot && PYTHONPATH=/opt/passivbot/src python -c 'import passivbot_rust; from rust_utils import source_fingerprint, stamp_compiled_extensions, verify_loaded_runtime_extension; stamp_compiled_extensions(source_fingerprint()); state=verify_loaded_runtime_extension(); assert state["runtime_compiled_source_stamp"] == state["expected_source_fingerprint"] and state["runtime_compiled_source_stamp"]'
# Split large GPU libraries into separate runtime layers (Vast pulls retry per layer).
RUN python -c 'import pathlib,shutil; source=pathlib.Path("/usr/local/lib/python3.12/site-packages"); target=pathlib.Path("/opt/runtime-site"); target.mkdir(); [(shutil.copytree(p,target/p.name,symlinks=True) if p.is_dir() else shutil.copy2(p,target/p.name)) for p in source.iterdir() if p.name not in ("nvidia","torch","triton")]; shutil.copytree(source/"nvidia/cu13", "/opt/runtime-cu13", symlinks=True, ignore=shutil.ignore_patterns("libcublasLt.so.13"))'

FROM python:3.12-bookworm AS runtime
COPY --from=build /opt/runtime-site/ /usr/local/lib/python3.12/site-packages/
COPY --from=build /usr/local/lib/python3.12/site-packages/torch/ /usr/local/lib/python3.12/site-packages/torch/
COPY --from=build /usr/local/lib/python3.12/site-packages/triton/ /usr/local/lib/python3.12/site-packages/triton/
RUN apt-get update && apt-get install -y --no-install-recommends rsync \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/__init__.py /usr/local/lib/python3.12/site-packages/nvidia/__init__.py
COPY --from=build /opt/runtime-cu13/ /usr/local/lib/python3.12/site-packages/nvidia/cu13/
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13 /usr/local/lib/python3.12/site-packages/nvidia/cu13/lib/libcublasLt.so.13
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/cudnn/ /usr/local/lib/python3.12/site-packages/nvidia/cudnn/
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/nccl/ /usr/local/lib/python3.12/site-packages/nvidia/nccl/
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/nvshmem/ /usr/local/lib/python3.12/site-packages/nvidia/nvshmem/
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/cusparselt/ /usr/local/lib/python3.12/site-packages/nvidia/cusparselt/
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/ /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/
COPY --from=build /usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/ /usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/
COPY --from=build /usr/local/bin/ /usr/local/bin/
COPY --from=build /opt/passivbot/ /opt/passivbot/
COPY --from=build /opt/pb8-revision /opt/pb8-pip-freeze.txt /opt/
COPY cloud_worker.py /opt/pbgui/worker.py
LABEL org.opencontainers.image.title="PBGui PB8 GPU Worker (fork)"
WORKDIR /work
CMD ["bash"]
