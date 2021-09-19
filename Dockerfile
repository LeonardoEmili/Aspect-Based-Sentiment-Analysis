FROM python:3.7-stretch

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy model

COPY model model

# copy code

COPY hw2 hw2
ENV PYTHONPATH hw2

# standard cmd

CMD [ "python", "hw2/app.py" ]
