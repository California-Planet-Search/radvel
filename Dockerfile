FROM conda/miniconda3

ENV TERM=xterm
ENV TERMINFO=/etc/terminfo

RUN mkdir /code && \
    mkdir /code/radvel && \
    apt-get --yes update && \
    apt-get install --yes gcc g++ git && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

WORKDIR /code/radvel

RUN pip install --no-cache-dir nose coveralls pybind11 && \
    conda install -y -c conda-forge celerite && \
    pip install --no-cache-dir . && \
    python setup.py build_ext -i

CMD nosetests radvel --with-coverage --cover-package=radvel
