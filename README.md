# YOLO-v1
Form sractch implementation of the YOLO-v1 object detection algorithm. 

Paper: https://arxiv.org/pdf/1506.02640.pdf

## Installation
* Download the dataset [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
* Install requirements 
* Export PYTHONPATH: `export PYTHONPATH=/yourpath/yolo-v1`

## Train
* Run `python src/test/test_trainer.py`


## Some Results
Image example. 
* Red: gt
* green/blue: box1, box2 predictions


![alt text](https://github.com/aldipiroli/yolo-v1/blob/develop/img/example.png)


## TODOs
- [ ] Training scheaduler as in paper
- [ ] Load pre-trained model
- [ ] Tensorboard integration 
- [ ] Provide fully trained model




