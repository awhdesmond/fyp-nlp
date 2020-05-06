FROM python:3.6

WORKDIR /pinocchio-nlp

EXPOSE 8080

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY *.py ./
ENTRYPOINT [ "python3", "src/server.py" ]