if [ "${1}" == "" ]; then
    nvidia-docker build -t JoyLeeA/posca .
else
    nvidia-docker build -t JoyLeeA/posca:${1} .
fi
