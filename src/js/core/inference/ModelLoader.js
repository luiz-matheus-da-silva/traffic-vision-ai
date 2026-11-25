// ============================================================================
// ModelLoader.js — Carrega o modelo de IA e a lista de classes
// ============================================================================

import * as tf from '@tensorflow/tfjs';

// Caminhos dos arquivos dentro da pasta `public/` do projeto.
// model.json descreve a rede; os arquivos .bin guardam os pesos (o "conhecimento").
const MODEL_PATH = '/models/yolov5n_web_model/model.json';
// labels.json: lista com o nome de cada classe que o modelo conhece.
const LABELS_PATH = '/models/yolov5n_web_model/labels.json';
import { MODEL_CONFIG } from '../../config/ModelConfig.js';

const CLASS_THRESHOLD = MODEL_CONFIG.confidenceThreshold;

/**
 * Carrega o modelo YOLOv5n e suas classes.
 *
 * @returns {Promise<{model: tf.GraphModel, labels: string[], classThreshold: number}>}
 */
async function loadModelAndLabels() {
    // Garante que o TensorFlow.js terminou de iniciar.
    await tf.ready();

    // Baixa a lista de nomes das classes.
    const response = await fetch(LABELS_PATH);

    // response.ok é false se o servidor respondeu com erro (ex.: 404).
    if (!response.ok) {
        throw new Error(`Failed to load labels: ${response.status}`);
    }

    const labels = await response.json(); // converte o texto JSON em array

    // Baixa e monta o modelo. "GraphModel" é o formato de modelo convertido
    // para rodar no navegador.
    const model = await tf.loadGraphModel(MODEL_PATH);

    // WARM-UP (aquecimento): a primeira execução de um modelo é lenta, pois a
    // GPU precisa preparar tudo. Rodamos uma vez com dados falsos (tudo = 1)
    // para que as inferências reais já comecem rápidas.
    // model.inputs[0].shape é o formato de entrada esperado, ex.: [1, 640, 640, 3].
    const warmupInput = tf.ones(model.inputs[0].shape);

    try {
        const warmupOutput = await model.executeAsync(warmupInput);

        // O resultado do aquecimento não interessa; só liberamos a memória.
        tf.dispose(warmupOutput);
    } finally {
        warmupInput.dispose();
    }

    // Devolve tudo que o Worker precisa.
    return {
        model,
        labels,
        classThreshold: CLASS_THRESHOLD
    };
}

export { loadModelAndLabels };
