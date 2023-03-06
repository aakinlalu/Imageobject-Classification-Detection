import * as tf from '@tensorflow/tfjs';
tf.setBackend('cpu');
tf.setBackend('webgl');

const demoSession = document.getElementById('demos');
const video = document.getElementById('webcam');
const liveview = document.getElementById('liveView');
const enableWebcamButton = document.getElementById('webcamButton');
const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)

var vidWidth = 0;
var vidHeight = 0;
var xStart = 0;
var yStart = 0;

const names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

//Function Loads the GraphModel type model of
const asyncLoadModel = async (model_url)=> {
    model = await tf.loadGraphModel(model_url);
    console.log('Model loaded');
    //Enable start button:
    demoSession.classList.remove('invisible');
    // enableWebcamButton.innerHTML = 'Start camera';
}

var model = undefined;
// model_url = 'https://raw.githubusercontent.com/aakinlalu/object-classifier-tng/main/fine-tuned-model-tfjs-api/best_web_model/model.json';
model_url ='https://raw.githubusercontent.com/aakinlalu/object-classifier-tng/main/fine-tuned-model-tfjs-api-small/fine_yolov8n_320_web_model/model.json';


// Check if webcam access is supported.
const hasGetUserMedia = () => {
  return !!(navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia);
};

// Enable the live webcam view and start classification.
const enableCam = (event) => {
  if (!model) {
      return;
  }
  // Hide the button once clicked.
  enableWebcamButton.classList.add('removed');

  //getUsermedia parameters to force video but not audio.
  const constraints = {
      video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment'
      }}).then((stream) => {
      video.srcObject = stream;
      video.onloadedmetadata = () => {
          vidWidth = video.videoHeight;
          vidHeight = video.videoWidth;
          //The start position of the video (from top left corner of the viewport)
          xStart = Math.floor((vw - vidWidth) / 2);
          yStart = (Math.floor((vh - vidHeight) / 2) >= 0) ? (Math.floor((vh - vidHeight) / 2)) : 0;
          video.play();
      video.addEventListener('loadeddata', predictWebcam);
      }
  });
};


//Call load function
asyncLoadModel(model_url);

// Keep a reference of all the child elements we create 
// so we can remove them easily on each render 
let children = [];

// If webcam supported, add event listener to button for when user
//want to activate it 
if (hasGetUserMedia()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}


//Perform prediction based on webcam using Layer model model:
// Prediction loop!
const predictWebcam = () => {

  let [modelWidth, modelHeight] = model.inputs[0].shape.slice(1, 3);

  console.log('modelWidth: ', modelWidth, 'modelHeight: ', modelHeight)

  // tf.tidy will clean up any GPU memory we used when this function is done.
  const input = tf.tidy(() => {
      tfimg= tf.image.resizeBilinear(tf.browser.fromPixels(video), [modelWidth, modelHeight])
      return tfimg.transpose([0,1,2]).expandDims();
    });

    input.print()
    console.log('input: ', input.shape)


  // Now let's start classifying a frame in the stream.
  model.executeAsync(input).then((predictions) => {
       console.log(predictions.shape)
       console.log(predictions.squeeze().arraySync())

      const [boxes, scores, classes, valid_detections] = predictions;
      const boxes_data = boxes.dataSync();
      const scores_data = scores.dataSync();
      const classes_data = classes.dataSync();
      const valid_detections_data = valid_detections.dataSync()[0];

      // console.log("boxes_data: ", boxes_data);
      // console.log("scores_data: ", scores_data);
      // console.log("classes_data: ", classes_data);
      // console.log("valid_detections_data: ", valid_detections_data);

      // tf.dispose(predictions);

      // Remove any highlighting we did previous frame.
      for (let i = 0; i < children.length; i++) {
          liveview.removeChild(children[i]);
      }
      children.splice(0);

      // Now lets loop through predictions and draw them to the live view if
      // they have a high confidence score.
      for (let n = 0; n < predictions.length; n++) {
          // If we are over 66% sure we are sure we classified it right, draw it!
          const score = scores_data[n].toFixed(2);
          const classLabel = names[classes_data[n]];
          let [x1, y1, x2, y2] = boxes_data.slice(n * 4, (n + 1) * 4);

          // console.log("score: ", score);
          // console.log("classLabel: ", classLabel);
          // console.log("x1: ", x1);
          // console.log("y1: ", y1);
          // console.log("x2: ", x2);
          // console.log("y2: ", y2);

          if (score > 0.2) {
          
              const p = document.createElement('p');
              p.innerText = classLabel  + ' - with '
                  + Math.round(parseFloat(score) * 100)
                  + '% confidence.';
              

              p.style = 'left: ' + (x1+vidWidth) + 'px;' + 
                  'top: ' + y1 + 'px;' +
                  'width: ' + (x2 - x1) + 'px' +
                  'height: ' + (y2 - y1) + 'px;';


              //Draw the actual bounding box
              const highlighter = document.createElement('div');
              highlighter.setAttribute('class', 'highlighter');
              highlighter.style = 'left: ' + (x1 + vidWidth) + 'px; + top; '
                  + y1 + 'px; width: '
                  + (x2 - x1) + 'px; height: '
                  + ((y2 - y1) +vidHeight) + 'px;';

              liveview.appendChild(highlighter);
              liveview.appendChild(p);

              //Store the child elements we create so we can remove them next time
              children.push(highlighter);
              children.push(p);
          }

          tf.dispose(predictions);
  }

  // Call this function again to keep predicting when the browser is ready.
  window.requestAnimationFrame(predictWebcam);

  });

};
 



