FROM python:3.9.19-slim
RUN apt-get update && apt-get install -y sox
WORKDIR /Reface_task/
COPY . .
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/riffusion/riffusion-hobby.git

ADD requirements.txt /Reface_task/
RUN pip install numpy typing-extensions

RUN pip install -r /Reface_task/requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "Riffusion_app.py"]