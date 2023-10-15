FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
RUN pipenv install flask
RUN pipenv install scikit-learn
RUN pipenv install waitress
COPY ["model2_predict.py", "./"]
WORKDIR /app

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "model2_predict:app"]


