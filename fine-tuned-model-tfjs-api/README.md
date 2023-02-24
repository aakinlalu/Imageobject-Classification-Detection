# Object Detection Model
The model has been fine-tuned with dataset of classes: 

**Labels**:
- *apple*
- *avocado*
- *banana*
- *cherry*
- *date*
- *grape*
- guava
- *kiwifruit*
- *lemon*
- *lime*
- *mango*
- *orange*
- *peach*
- *pear*
- *pearl*
- *pineapple*
- *pomegranate*
- *strawberry*
- *watermelon*

## Model location 
Model should be loaded from https endpoints. After loading the json file the function will make requests for corresponding .bin files that the json file references.

In this case, model is loaded to github and url is made available to the api. S3 can be used instead.

### Native File System (Node.js only)
use `file://path/to/model.json`

## Javascript Libraries 
Tensorflowjs is used for prediction.
