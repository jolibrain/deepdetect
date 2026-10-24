# Local verification extension; build the regular CPU/GPU image first.
ARG DEEPDETECT_IMAGE=deepdetect_cpu:latest
FROM ${DEEPDETECT_IMAGE}
USER root
# Keep the generic image's Torch/torchvision pair fixed while resolving upstream dependencies.
RUN python3 -m pip freeze | grep -E '^(torch|torchvision)==' > /tmp/torch-constraints.txt && \
    python3 -m pip install --break-system-packages -c /tmp/torch-constraints.txt \
        'faster-coco-eval>=1.6.5' PyYAML tensorboard scipy calflops transformers
COPY tests/pytorch_rtdetr_http.py tests/pytorch_prediction_http.py /opt/deepdetect/tests/
USER dd
