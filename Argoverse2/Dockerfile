FROM nvcr.io/nvidia/pytorch:23.08-py3
ADD . .
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
# python3 rawTOinterm.py --data-dir /main/Argoverse\ Dataset/val/ --save-dir /main/Argoverse\ Dataset/val_interm --type "val"
CMD ["python3", "rawTOinterm.py", "--data-dir", "/main/Argoverse Dataset/val/", "--save-dir", "/main/Argoverse Dataset/val_interm", "--type", "val"]