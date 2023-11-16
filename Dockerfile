FROM python:3.10

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

EXPOSE 3000

ENTRYPOINT [ "python" ]

CMD [ "app.py", "0.0.0.0:3000" ]