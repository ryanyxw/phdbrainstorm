FROM gcr.io/ai2-beaker-core/public/d2p4h60orjbbosetkkog:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /stage

# Copy the `requirements.txt` to `/stage/requirements.txt/` and then install them.
# We do this first because it's slow and each of these commands are cached in sequence.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the file `main.py` to `/stage/main.py`
# You might need multiple of these statements to copy all the files you need for your experiment.
COPY . .


