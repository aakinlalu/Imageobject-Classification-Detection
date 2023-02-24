# Object Detection Model
The model has been fine-tuned with dataset of classes: 

**Labels**:
- *Bin*, 
- *Bottles*, 
- *Cars*, 
- *Computer*,
- *Face mask*,
- *Fire Extinguisher*,
-  *Powerpoint*

## Model location 
Model should be loaded from https endpoints. After loading the json file the function will make requests for corresponding .bin files that the json file references.

In this case, model is loaded to github and url is made available to the api. S3 can be used instead.

### Native File System (Node.js only)
use `file://path/to/model.json`

## Javascript Libraries 
ml5.js and p5.js are used instead of tensorflow.js. Although, ml5.js is built on tensorflowjs. They are high level and easy to use.



