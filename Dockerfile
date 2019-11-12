FROM conda/miniconda3

ENV TERM=xterm
ENV TERMINFO=/etc/terminfo
ENV COVERALLS_REPO_TOKEN=7ZpQ0LQWM2PNl5iu7ZndyFEisQnZow8oT


RUN mkdir /code && \
    mkdir /code/radvel && \
    apt-get --yes update && \
    apt-get install --yes gcc g++ git && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda update -n base -c defaults conda && \
    pip install --no-cache-dir nose coveralls pybind11 && \
    conda install -y -c conda-forge celerite

WORKDIR /code/radvel
ADD requirements.txt /code/radvel/requirements.txt

RUN conda install -y --file requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

ADD . /code/radvel

RUN pip install --no-cache-dir . && \
    python setup.py build_ext -i

CMD nosetests radvel --with-coverage --cover-package=radvel && \
    coveralls
