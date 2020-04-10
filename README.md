- Prerequisite: cuda 10.0, cudnn 7, python 3.6 installed
- Install dependencies 
  pip3 install -r requirement.txt 
- Download fasion mnist data to ./data/fashion folder
- For training from scratch, run: "python3 train.py", model will be saved to model folder.
  For training from saved checkpoint, run: "python3 train.py --checkpoint ./model_trained/model.ckpt-30000" to resume from provided checkpoint. Or run "python3 train.py --checkpoint [/path/to/your/saved/checkpoint]" to resume from your own saved checkpoint. Newly trained model will be saved at the same directory of provided checkpoint
  
  Some explanations for choosing model/parameter can be found in Report file.
  
  To view metrics, run "tensorboard --logdir=model --port [port] --host [host]"
- For inference, simple run "python3 test.py" to predict fasion mnist dataset. 
