FROM apache/airflow:2.9.1-python3.10

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gnupg \
    curl \
    apt-transport-https \
    software-properties-common \
    unixodbc-dev && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y DEBIAN_FRONTEND=noninteractive apt-get install -y msodbcsql17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

ENV AIRFLOW_HOME=/opt/airflow
ENV PATH="${AIRFLOW_HOME}/bin:${PATH}"

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]