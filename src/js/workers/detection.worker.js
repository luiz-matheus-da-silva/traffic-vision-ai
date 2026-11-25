
import * as tf from '@tensorflow/tfjs';

import { loadModelAndLabels } from '../core/inference/ModelLoader.js';
import { preprocessImage } from '../core/inference/ImagePreprocessor.js';
import { runInference } from '../core/inference/InferenceEngine.js';

let model = null;
let labels = [];
let classThreshold = 0.5;

/**
 * Inicializa o TensorFlow.js e carrega o modelo.
 */
async function initialize() {
    try {
        // Configura o backend utilizado para executar as operações.
        await tf.setBackend('webgl');
        await tf.ready();

        // Carrega o modelo e suas configurações.
        const loaded = await loadModelAndLabels();

        model = loaded.model;
        labels = loaded.labels;
        classThreshold = loaded.classThreshold;

        // Informa à Main Thread que o modelo está disponível.
        self.postMessage({
            type: 'model-loaded',
            message: 'YOLOv5n carregado com sucesso.'
        });

    } catch (error) {
        // Comunica falhas de inicialização à Main Thread.
        self.postMessage({
            type: 'error',
            message: error.message
        });
    }
}

/**
 * Recebe frames da Main Thread e executa a inferência.
 */
self.onmessage = async ({ data }) => {
    // Ignora mensagens que não representam frames.
    if (data.type !== 'process-frame') return;

    // Impede inferências antes do carregamento do modelo.
    if (!model) {
        self.postMessage({
            type: 'error',
            message: 'Modelo ainda não está carregado.'
        });

        return;
    }

    try {
        // Converte o frame recebido em um tensor compatível com o modelo.
        const tensor = preprocessImage(data.frame, 640, 640);

        // Executa a inferência e obtém as saídas do YOLOv5n.
        const outputs = await runInference(model, tensor);

        const results = [];

        // Inspeciona o formato e os primeiros valores de cada saída.
        for (const [index, output] of outputs.entries()) {
            results.push({
                index,
                shape: output.shape,
                values: Array.from(await output.data()).slice(0, 20)
            });

            // Libera os tensores após extrair os dados.
            output.dispose();
        }

        // Devolve os resultados para a Main Thread.
        self.postMessage({
            type: 'inference-completed',
            outputs: results
        });

    } catch (error) {
        // Encaminha erros de processamento para a aplicação.
        self.postMessage({
            type: 'error',
            message: error.message
        });
    }
};

// Notifica a Main Thread assim que o Worker é inicializado.
self.postMessage({
    type: 'worker-ready',
    message: 'Detection Worker inicializado.'
});

// Inicia o carregamento do modelo independentemente do recebimento de frames.
initialize();
