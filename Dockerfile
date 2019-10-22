FROM conda/miniconda3

ENV DISPLAY=localhost:0

RUN mkdir /code && \
    mkdir /code/radvel && \
    apt-get --yes update && \
    apt-get install --yes gcc g++

ADD . /code/radvel
WORKDIR /code/radvel

RUN pip install --no-cache-dir . && \
    python setup.py build_ext -i && \
    pip install --no-cache-dir nose coveralls celerite

CMD nosetests radvel --with-coverage --cover-package=radvel && \
    coveralls
