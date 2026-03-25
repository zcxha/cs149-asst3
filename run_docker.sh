docker run --rm -it \
  --runtime=nvidia \
  --gpus all \
  -v "$PWD":/workspace \
  gtx650-cuda102