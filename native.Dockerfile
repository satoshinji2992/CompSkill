# syntax=docker/dockerfile:1
# Build with: docker buildx build --build-context sra=./third_party/SR-Agents \
#   -f native.Dockerfile -t compskill-toolqa-native --load .
FROM python:3.11-slim
RUN pip install --no-cache-dir torch sentence-transformers pandas numpy scipy networkx \
    sympy latex2sympy2 requests openai pyyaml tqdm
COPY --from=sra /src/sragents /opt/sra/src/sragents
COPY src/skill_experience/toolqa/native_worker.py /opt/native_worker.py
ENV PYTHONPATH=/opt/sra/src PYTHONDONTWRITEBYTECODE=1
USER 10001:10001
ENTRYPOINT ["python", "-u", "/opt/native_worker.py"]
