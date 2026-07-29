# BoneSeg Studio — application image.
#
# Ships CODE ONLY. The trained model weights are deliberately NOT baked into
# this image (see .dockerignore). Provide them at runtime with a bind mount and
# point BONESEG_MODEL_PATH at the mounted file, e.g.:
#
#   docker build -t boneseg .
#   docker run --rm -p 7860:7860 \
#       -v /path/to/models:/models:ro \
#       boneseg
#
# The container serves the local web UI on http://localhost:7860

FROM python:3.11-slim

# Native libraries a few Python wheels load at import time:
#   libgl1 / libglib2.0-0 -> opencv-python
#   libgomp1              -> torch / scikit-image (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first, in their own layer. Docker caches this layer and
# only rebuilds it when requirements.txt changes — so code edits rebuild fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source. Weights, caches and runtime dirs are excluded
# by .dockerignore, so nothing large or private ends up in the image.
COPY . .

# Where the app looks for the checkpoint INSIDE the container. The file itself
# is supplied at runtime via the bind mount above — never copied into the image.
ENV BONESEG_MODEL_PATH=/models/model367b3/best_bone_model.pth

EXPOSE 7860

# 0.0.0.0 makes the port reachable from outside the container; --no-browser
# because there is no desktop browser to open inside a container.
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
