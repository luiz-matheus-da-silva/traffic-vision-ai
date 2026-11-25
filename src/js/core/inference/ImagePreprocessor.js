
import * as tf from '@tensorflow/tfjs';

/**
 * Converte um frame em tensor e prepara sua entrada para o YOLOv5n.
 *
 * @param {ImageBitmap|HTMLVideoElement|HTMLCanvasElement} input - Frame extraído do vídeo.
 * @param {number} height - Altura esperada pelo modelo.
 * @param {number} width - Largura esperada pelo modelo.
 * @returns {tf.Tensor4D} Tensor normalizado no formato [1, altura, largura, 3].
 */
function preprocessImage(input, height, width) {
    return tf.tidy(() => {
        // Converte o frame em um tensor RGB.
        const image = tf.browser.fromPixels(input);

        // Redimensiona a imagem e normaliza os pixels para [0, 1].
        return tf.image
            .resizeBilinear(image, [height, width])
            .div(255)

            // Adiciona a dimensão do batch exigida pelo modelo.
            .expandDims(0);
    });
}

export { preprocessImage };
