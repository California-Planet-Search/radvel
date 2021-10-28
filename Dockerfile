FROM conda/miniconda3

ENV TERM=xterm
ENV TERMINFO=/etc/terminfo
ENV PYTHONDONTWRITEBYTECODE=true
ENV COVERALLS_REPO_TOKEN=7ZpQ0LQWM2PNl5iu7ZndyFEisQnZow8oT


RUN mkdir /code && \
    mkdir /code/radvel && \
    apt-get --yes update && \
    apt-get install --yes gcc git pkg-config libhdf5-100 libhdf5-dev && \
    apt-get clean && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda update -n base -c defaults conda && \
    conda install --yes nomkl numpy pybind11 coveralls nose && \
    conda install --yes -c conda-forge celerite && \
    conda clean -afy

WORKDIR /code/radvel
ADD requirements.txt /code/radvel/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ADD . /code/radvel


CMD python setup.py build_ext -i  && \
    pip install --no-cache-dir --no-deps .  && \
    nosetests radvel --with-coverage --cover-package=radvel && \
    coveralls
