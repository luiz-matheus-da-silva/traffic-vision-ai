// ============================================================================
// ImagePreprocessor.js — Prepara a imagem para o modelo
//
// A IA não "enxerga" imagens: ela trabalha com números. Aqui convertemos o
// frame em um tensor (uma matriz de números) no formato exato que o YOLOv5n espera.
// ============================================================================

import * as tf from '@tensorflow/tfjs';

/**
 * Converte um frame em tensor e prepara sua entrada para o YOLOv5n.
 *
 * @param {ImageBitmap|HTMLVideoElement|HTMLCanvasElement} input - Frame extraído do vídeo.
 * @param {number} targetHeight - Altura esperada pelo modelo.
 * @param {number} targetWidth - Largura esperada pelo modelo.
 * @returns {tf.Tensor4D} Tensor no formato [1, altura, largura, 3] com valores de 0 a 1.
 */
function preprocessImage(input, targetHeight, targetWidth) {
    // tf.tidy() apaga automaticamente os tensores intermediários criados
    // dentro dele, mantendo só o que é retornado. Isso evita vazar memória.
    return tf.tidy(() => {
        // Lê os pixels da imagem. Resultado: [altura, largura, 3],
        // onde 3 = canais de cor (R, G, B), cada um de 0 a 255.
        const image = tf.browser.fromPixels(input);

        return tf.image
            // Ajusta para o tamanho que o modelo exige (ex.: 640x640).
            .resizeBilinear(image, [targetHeight, targetWidth])

            // Normaliza: divide por 255 para os valores ficarem entre 0 e 1.
            .div(255)

            // Modelos processam "lotes" (batch) de imagens. Adicionamos uma
            // dimensão na frente: [640, 640, 3] vira [1, 640, 640, 3] (lote de 1 imagem).
            .expandDims(0);
    });
}

export { preprocessImage };
