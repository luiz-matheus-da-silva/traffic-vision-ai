
/**
 * Executa a inferência de um frame utilizando o YOLOv5n.
 *
 * @param {tf.GraphModel} model - Modelo previamente carregado.
 * @param {tf.Tensor} tensor - Frame pré-processado.
 * @returns {Promise<tf.Tensor[]>} Tensores retornados pelo modelo.
 */
async function runInference(model, tensor) {
    try {
        // Executa o modelo sobre o frame recebido.
        const output = await model.executeAsync(tensor);

        // Padroniza a saída para facilitar o processamento posterior.
        return Array.isArray(output) ? output : [output];

    } finally {
        // Libera o tensor de entrada após a inferência.
        tensor.dispose();
    }
}

export { runInference };
