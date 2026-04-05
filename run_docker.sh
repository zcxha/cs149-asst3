docker run -it \
  --name asst3-cuda \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --runtime=nvidia \
  --gpus all \
  -p 127.0.0.1:12345:12345 \
  -v "$PWD":/workspace \
  -w /workspace \
  gtx650-cuda102 bash
