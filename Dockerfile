FROM apache/airflow:2.8.0-python3.10

USER root

# Install system dependencies including Chromium for Selenium
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    postgresql-client \
    build-essential \
    chromium \
    chromium-driver \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements first (for better layer caching)
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy project files
COPY --chown=airflow:airflow src /opt/airflow/src
COPY --chown=airflow:airflow dags /opt/airflow/dags
COPY --chown=airflow:airflow dbt /opt/airflow/dbt
COPY --chown=airflow:airflow scripts /opt/airflow/scripts

# Install DBT dependencies
RUN cd /opt/airflow/dbt/ahold_options && \
    dbt deps --profiles-dir .

# Set Python path
ENV PYTHONPATH="/opt/airflow:/opt/airflow/src:${PYTHONPATH}"

# Create directories
RUN mkdir -p /opt/airflow/logs /opt/airflow/plugins /opt/airflow/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD airflow jobs check --job-type SchedulerJob --hostname "${HOSTNAME}" || exit 1

WORKDIR /opt/airflow
