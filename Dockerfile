FROM ubuntu:latest
LABEL authors="donald"

ENTRYPOINT ["top", "-b"]