# Base image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ make \
    git curl \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY src ./src
COPY README.md ./

ENV MPLBACKEND=QtAgg

CMD ["python", "-m", "src"]


#docker build -t qlab .

#xhost +local:docker
#docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix qlab
