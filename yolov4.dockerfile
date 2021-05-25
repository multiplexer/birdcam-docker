FROM multiplexer/yolov4:latest
EXPOSE 8080
USER darknet
WORKDIR /home/darknet/darknet
CMD ["python3", "main.py"]
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]