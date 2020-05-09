FROM python:3.6

WORKDIR /pinocchio-nlp

EXPOSE 5000

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
ENTRYPOINT [ "python3", "src/server.py" ]
