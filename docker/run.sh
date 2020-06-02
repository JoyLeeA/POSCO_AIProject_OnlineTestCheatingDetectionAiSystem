nvidia-docker run -it --rm \
    -v /data/JoyLeeA/posca:/app \
    -p 1028:1028 \
    --name posca \
    JoyLeeA/posca $@
    bash
