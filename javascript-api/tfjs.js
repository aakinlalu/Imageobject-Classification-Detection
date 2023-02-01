const tf = require('@tensorflow/tfjs');

async function loadModel() {
    try {
    const model = await tf.loadLayersModel('best_web_model/model.json');
    return model;
    } catch (err) {
    console.log(err);
    }
}

async function predict() {
    const model = await loadModel();
    console.log(model);
}

predict();