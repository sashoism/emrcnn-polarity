FROM nvcr.io/nvidia/pytorch:24.03-py3
RUN rm -rf /workspace/*

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN apt-get update && apt-get install -y libgl1

ENV VIRTUAL_ENV=/venv
RUN pip install uv
RUN uv venv --system-site-packages ${VIRTUAL_ENV}
RUN uv pip install opencv-python scikit-image scikit-learn imantics