ARG  CODE_VERSION=3.9.12-slim-bullseye
ARG  BASE_IMAGE_SHA=sha256:b952de95d8c615422760bbd5d8ff2e82d3fe453d1c1eb9649c0a97263c25e3dc
FROM python:${CODE_VERSION}@${BASE_IMAGE_SHA} AS base
LABEL maintainer="yannis.antonopoulos@yailab.com"
ARG API_ENV=dev
# if you forked PrometheusML, you can pass in your own GitLab username to use your fork
# i.e. gl_username=myname
ARG gl_username=yailab
ENV prometheus_dir="home/prometheus/prometheusml"

# Setup env
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Install tzdata and configure timezone (fix for tests which try to read from "/etc/localtime")
    && apt-get -y install tzdata \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git iproute2 procps iproute2 lsb-release \
    #
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Clone PrometheusML repo
RUN mkdir -p "$prometheus_dir" \
    && git clone "https://gitlab.com/$gl_username/prometheusml.git" "$prometheus_dir" \
    && cd "$prometheus_dir" \
    && git remote add upstream "https://gitlab.com/yailab/prometheusml.git" \
    && git pull upstream main

# Set up pipenv environment
# DO NOT use `PIPENV_VERSION` as name! It breaks it!
ENV PY_PIPENV_VERSION=2022.4.21

# Install pipenv
RUN pip3 install --upgrade pip && \
    pip3 install pipenv==${PY_PIPENV_VERSION}

WORKDIR "$prometheus_dir"

# Install python dependencies in /.venv
RUN export PIPENV_VENV_IN_PROJECT=1 && \
    if [ "$API_ENV" = "dev" ] ; then pipenv install --dev ; \
    elif [ "$API_ENV" = "prod" ] ; then pipenv lock && pipenv install --deploy ; \
    else { echo Environment [$API_ENV] not supported! Only development and production.; exit 1; } ; \
    fi

# Add installed python packages to PATH
ENV PATH=".venv/bin:$PATH"

# make uploads dir
RUN mkdir uploads

# Expose this port
EXPOSE 5000/tcp

# make boot.sh executable
RUN chmod a+x boot.sh

# define the command when container is started
ENTRYPOINT ["./boot.sh"]
