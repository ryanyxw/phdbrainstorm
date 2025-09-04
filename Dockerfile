FROM gcr.io/ai2-beaker-core/public/d2p4h60orjbbosetkkog:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage

# Copy the file `main.py` to `/stage/main.py`
# You might need multiple of these statements to copy all the files you need for your experiment.
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN cd lm-evaluation-harness && pip install -e .

RUN cd litgpt && pip install -e '.[extra]'



