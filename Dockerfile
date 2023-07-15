FROM google/cloud-sdk:410.0.0

WORKDIR /Cyberbullying
COPY . /Cyberbullying

RUN apt update -y && \
    apt-get update && \
    pip install --upgrade pip && \ 
    apt-get install ffmpeg libsm6 libxext6  -y


RUN pt-get install apt-transfport-https ca-certificates gnupg -y
RUN apt install python3 -y

RUN pip install -r requirements.txt && \ 
    pip install -e .

CMD ["python3", "app.py"]