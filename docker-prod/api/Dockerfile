FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn>=21.2

COPY . .

# --preload builds the index once in the master and forks, instead of every
# worker embedding the same six articles on boot. See the roadmap note in
# DECISIONS.md: start-up calls the embeddings API, so this is also one fewer
# way for a provider outage to stop a replica booting.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--preload", \
     "--timeout", "180", "--access-logfile", "-", "app:create_app()"]
