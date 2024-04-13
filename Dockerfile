FROM nvcr.io/nvidia/pytorch:24.03-py3
RUN rm -rf /workspace/*

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN apt-get update && apt-get install -y libgl1

RUN pip install uv
RUN uv venv --system-site-packages /venv
RUN VIRTUAL_ENV=/venv uv pip install opencv-python scikit-image scikit-learn imantics