# MAYOLO

### Introduction
SAM-CompAug: Segment Anything Model-Assisted Geometric Transformation Framework for Part Detection Data Augmentation

### Framework Structure
![](data/images/yolo-facev2.jpg)

### Requirments
Create a Python Virtual Environment.   
```shell
conda create -n {name} python=x.x
```

Enter Python Virtual Environment.   
```shell
conda activate {name}
```

Install pytorch in *[this](https://pytorch.org/get-started/previous-versions/)*.  
```shell 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other python package.   
```shell
pip install -r requirements.txt
```
   
### Step-Through Example
#### Downloaded Dataset

#### Dataset
Download the [Part-ID] dataset. 
```shell
# You can modify convert.py and voc_label.py if needed.
python3 data/convert.py
python3 data/voc_label.py
```




## Preweight
| Model            | mAP@50  | mAP@50-95 | P/%  | R/%  |weight|
|------------------|-------|--------|-------|------------|------|
| MAYOLOv8         | 97.1  | 90.2 | 97.5 |93.6 |[MAYOLOv8n.pt]
| MAYOLOv12        | 97.6 | 90.6| 97.8 | 94.4|[MAYOLOv12n.pt]|


#### Training
```shell
python train.py --weights preweight.pt    
                --data data/partid.yaml    
                --cfg models/mayolov8.yaml     
                --batch-size 8   
                --epochs 200
```


#### Evaluate   
```shell
python3 test_mayolo.py --weights 'your test model' --img-size 640
```


### Finetune
see in *[https://github.com/ultralytics/yolov5/issues/607](https://github.com/ultralytics/yolov5/issues/607)*
```shell
# Single-GPU
python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve

# Multi-GPU
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  nohup python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --device $i --evolve > evolve_gpu_$i.log &
done

# Multi-GPU bash-while (not recommended)
for i in 0 1 2 3 4 5 6 7; do
  sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting GPU '$i'...' &&
  "$(while true; do nohup python train.py... --device $i --evolve 1 > evolve_gpu_$i.log; done)" &
done
```




