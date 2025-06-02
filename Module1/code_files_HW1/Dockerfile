FROM docker.io/pytorch/pytorch

# Set up a new non-root user
ARG GID=1000
ARG UID=1000
RUN groupadd -r myuser -g $GID && \
    useradd -u $UID -r -m -g myuser -s /bin/bash -c "Docker image user" myuser

# Make this the default user
USER myuser

# Make /workdir the default working directory
WORKDIR /workdir

# Install Python dependencies
COPY requirements.txt /
RUN pip install --no-warn-script-location -r /requirements.txt
