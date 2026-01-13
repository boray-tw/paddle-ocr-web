FROM paddlepaddle/paddle:3.2.2-gpu-cuda12.9-cudnn9.9

ARG UID=2000

# install dependencies
RUN pip install \
  --no-cache-dir \
  --root-user-action ignore \
  paddleocr[doc-parser]==3.3.2 \
  fastapi==0.128.0 \
  uvicorn==0.40.0 \
  python-multipart==0.0.21

# IP and port to host the backend app
ENV APP_HOST="0.0.0.0"
ENV APP_PORT="8010"

# initialize the app directory
RUN mkdir /app
WORKDIR /app
COPY main.py ./

# run the app as a normal user
RUN useradd -u ${UID} -m paddle
USER paddle

# suppress PaddleOCR warnings when writing to $HOME (/root by default)
ENV HOME=/home/paddle

CMD [ "python", "./main.py" ]
