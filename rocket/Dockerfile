FROM registry.gitlab.hpi.de/akita/i/pyod:0.2.5

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"
ENV NUMBA_CACHE_DIR=/tmp
# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY DatasetConnector.py /app/
COPY algorithm.py .
