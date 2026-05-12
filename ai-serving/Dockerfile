FROM rayproject/ray:latest-py312-gpu

WORKDIR /app

COPY pyproject.toml /app/

RUN pip install uv --no-cache-dir
RUN uv pip install -r pyproject.toml --no-cache --system

ADD data /app/data  
ADD src /app/src

CMD ["serve", "run", "serving.app:rayApp"]
