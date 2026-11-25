
import * as tf from '@tensorflow/tfjs';

const MODEL_PATH = '/models/yolov5n_web_model/model.json';
const LABELS_PATH = '/models/yolov5n_web_model/labels.json';
const CLASS_THRESHOLD = 0.5;

/**
 * Carrega o modelo YOLOv5n e suas classes.
 *
 * @returns {Promise<{model: tf.GraphModel, labels: Array, classThreshold: number}>}
 */
async function loadModelAndLabels() {
    // Inicializa o TensorFlow.js antes de carregar o modelo.
    await tf.ready();

    // Carrega as classes utilizadas para identificar os objetos.
    const response = await fetch(LABELS_PATH);

    if (!response.ok) {
        throw new Error(`Falha ao carregar labels: ${response.status}`);
    }

    const labels = await response.json();

    // Carrega o modelo exportado no formato TensorFlow.js GraphModel.
    const model = await tf.loadGraphModel(MODEL_PATH);

    // Executa uma inferência fictícia para inicializar os recursos do modelo.
    const dummyInput = tf.ones(model.inputs[0].shape);

    try {
        const warmupOutput = await model.executeAsync(dummyInput);

        // Libera os tensores gerados durante o warm-up.
        tf.dispose(warmupOutput);
    } finally {
        dummyInput.dispose();
    }

    // Disponibiliza o modelo e suas configurações para o Worker.
    return {
        model,
        labels,
        classThreshold: CLASS_THRESHOLD
    };
}

export { loadModelAndLabels };
