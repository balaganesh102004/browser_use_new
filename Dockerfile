FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl xvfb x11vnc \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libxcomposite1 libxrandr2 libxdamage1 libxss1 \
    libasound2 libxshmfence1 libgbm1 libx11-xcb1 \
    libx11-dev libdrm2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install chromium

COPY . .
RUN mkdir -p /app/shared

ENV DISPLAY=:99 
ENV Xvfb=:99 
ENV -screen 0 1920x1080x24
ENV PYTHONUNBUFFERED=1
ENV CHROMIUM_FLAGS="--no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage"
ENV PLAYWRIGHT_SKIP_BROWSER_GC=1

CMD ["sleep", "infinity"]