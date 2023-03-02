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

const CLASS_NAMES = ["Face mask","Car","Powerpoint","Computer","Bottles","Fire Extinguisher","Bin"];


async function loadMobileNetFeatureModel() {
    const URL = 'https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
    const mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  }

//Function Loads the GraphModel type model of
const asyncLoadModel = async (model_url)=> {
    model = await tf.loadLayersModel(model_url);
    console.log('Model loaded');
    model.summary();
    //Enable start button:
    demoSession.classList.remove('invisible');
    // enableWebcamButton.innerHTML = 'Start camera';
}

var model = undefined;
model_url =  'https://raw.githubusercontent.com/aakinlalu/object-classifier-tng/main/fine-tuned-model-ml5js-api/image_model/model.json';


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
        //   vidWidth = video.videoHeight;
        //   vidHeight = video.videoWidth;
        //   //The start position of the video (from top left corner of the viewport)
        //   xStart = Math.floor((vw - vidWidth) / 2);
        //   yStart = (Math.floor((vh - vidHeight) / 2) >= 0) ? (Math.floor((vh - vidHeight) / 2)) : 0;
          video.play();
      video.addEventListener('loadeddata', predictWebcam);
      }
  });
};

loadMobileNetFeatureModel()
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

  // tf.tidy will clean up any GPU memory we used when this function is done.
   tf.tidy(() => { 
    // Capture the frame from the webcam.
     let videoFrameAsTensor = tf.browser.fromPixels(video).div(255)
     let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [224, 224], true)
     let imageFeatures = loadMobileNetFeatureModel.predict(resizedTensorFrame.expandDims())
     let prediction =model.predict(imageFeatures).squeeze();

     let highestIndex = prediction.argMax().arraySync();

     let predictionArray = prediction.arraySync();

     if (predictionArray[highestIndex] > 0.7) {

     console.log("prediction: ", CLASS_NAMES[highestIndex], Math.floor(predictionArray[highestIndex] * 100));

     }

      // return tf.image.resizeBilinear(tf.browser.fromPixels(video), [224, 224], true)
      //   .div(255).expandDims(0);
    });

    window.requestAnimationFrame(predictWebcam);


};
 



