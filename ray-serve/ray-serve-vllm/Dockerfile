FROM rayproject/ray-ml:nightly.250616.74d958-gpu
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER root

RUN chown -R ray /app

USER ray

COPY serve-chat-completion.py .

ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD []
