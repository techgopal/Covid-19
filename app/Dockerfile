FROM python:3.8
WORKDIR /streamlit
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /streamlit
CMD streamlit run streamlit/app.py --server.port $PORT