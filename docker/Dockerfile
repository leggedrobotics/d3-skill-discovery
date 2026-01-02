# we use the basic isaaclab image as the base
FROM nvcr.io/nvidia/isaac-lab:2.2.0 AS base

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ /workspace/d3-skill-discovery

# Install D3 skill discovery packages
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd /workspace/d3-skill-discovery"
RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install -e /workspace/d3-skill-discovery/source/d3_rsl_rl
RUN --mount=type=cache,target=/root/.cache/pip \
    /isaac-sim/python.sh -m pip install -e /workspace/d3-skill-discovery/source/d3_skill_discovery/

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace/d3-skill-discovery
