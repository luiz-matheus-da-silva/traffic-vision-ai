// ============================================================================
// InferenceEngine.js — Executa o modelo
//
// "Inferência" = usar um modelo já treinado para fazer uma previsão
// (aqui: encontrar veículos numa imagem).
// ============================================================================

/**
 * Executa a inferência de um frame utilizando o YOLOv5n.
 *
 * @param {tf.GraphModel} model - Modelo previamente carregado.
 * @param {tf.Tensor} inputTensor - Frame pré-processado.
 * @returns {Promise<tf.Tensor[]>} Tensores retornados pelo modelo.
 */
async function runInference(model, inputTensor) {
    try {
        // executeAsync roda o modelo sem travar a thread. Modelos com operações
        // dinâmicas (como o NMS do YOLO) exigem a versão assíncrona.
        const output = await model.executeAsync(inputTensor);

        // O modelo pode devolver um tensor só ou uma lista de tensores.
        // Padronizamos para SEMPRE devolver uma lista (array).
        return Array.isArray(output) ? output : [output];

    } finally {
        // O tensor de entrada não é mais necessário; liberamos a memória dele.
        // O `finally` garante isso mesmo se a inferência der erro.
        inputTensor.dispose();
    }
}

export { runInference };
