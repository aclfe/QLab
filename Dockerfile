# ----------------------------
# QLab Dockerfile
# The environment has been setup in a python 3.11
# The container will have a non-root user (a widely accepted and recommended security practice)
# We'll use an entrypoint script to handle checks and tasks before launching
# ----------------------------

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    git curl wget unzip \
    libgl1 libglib2.0-0 \
    libxkbcommon-x11-0 libxkbcommon0 \
    libx11-xcb1 libxrender1 libxext6 \
    libfontconfig1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# the non root user
RUN useradd -ms /bin/bash qlabuser
USER qlabuser
WORKDIR /app


COPY --chown=qlabuser:qlabuser requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# I'll start adding docs
COPY --chown=qlabuser:qlabuser src ./src
COPY --chown=qlabuser:qlabuser README.md ./
COPY --chown=qlabuser:qlabuser scripts ./scripts

RUN chmod +x ./scripts/entrypoint.sh

ENV MPLBACKEND=QtAgg

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import sys; import importlib.util; sys.exit(0 if importlib.util.find_spec('src') else 1)"


ENTRYPOINT ["./scripts/entrypoint.sh"]

CMD ["python", "-m", "src"]

#docker build -t qlab .

#xhost +local:docker
#docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix qlab
