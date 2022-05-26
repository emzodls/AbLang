FROM mambaorg/micromamba:0.23.3
COPY --chown=$MAMBA_USER:$MAMBA_USER pytorch_env.yml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER pytorch_pip.txt /tmp/requirements.txt

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
RUN pip install -r /tmp/requirements.txt