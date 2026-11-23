# Build after creating the archive with scripts/environment/pack_environment.sh.
ARG BASE_IMAGE=registry.h.pjlab.org.cn/library/ml-base@sha256:b3a5303f5c0b9dea21a7f80939901e2f0dcb84018e7ef7740f3b52de95f466be
FROM ${BASE_IMAGE}

ENV PATH="/opt/sc-evae/bin:${PATH}" \
    CONDA_PREFIX=/opt/sc-evae \
    CONDA_DEFAULT_ENV=sc-evae \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    MPLBACKEND=Agg \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ADD extracts the conda-pack tarball directly into the image.
ADD artifacts/sc-evae-py312-torch291-cu128-linux-x86_64.tar.gz /opt/sc-evae/
COPY scripts/environment/unpack_environment.py /tmp/sc-evae-unpack.py
RUN /opt/sc-evae/bin/python /tmp/sc-evae-unpack.py \
    && rm /tmp/sc-evae-unpack.py \
    && /opt/sc-evae/bin/python -m pip check

WORKDIR /workspace/sc-evae
COPY pyproject.toml README.md LICENSE environment.yml conda-linux-64.lock.txt requirements-linux-64.lock.txt ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY assets/ ./assets/
RUN python scripts/environment/check_environment.py

# The base image activates its own conda base environment in interactive Bash.
RUN printf '%s\n' '' \
    '# Activate the project environment after the base image shell setup.' \
    'if command -v conda >/dev/null 2>&1; then' \
    '    conda activate /opt/sc-evae' \
    'else' \
    '    export PATH="/opt/sc-evae/bin:$PATH"' \
    '    export CONDA_PREFIX=/opt/sc-evae' \
    'fi' >> /root/.bashrc \
    && printf '%s\n' '' \
    'export PATH="/opt/sc-evae/bin:$PATH"' \
    'export CONDA_PREFIX=/opt/sc-evae' \
    'export CONDA_DEFAULT_ENV=sc-evae' >> /root/.profile

CMD ["bash"]
