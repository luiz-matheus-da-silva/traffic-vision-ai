
import * as tf from '@tensorflow/tfjs';

/**
 * Calcula as dimensões necessárias para redimensionar a imagem
 * preservando sua proporção original (aspect ratio).
 *
 * O espaço restante será preenchido com padding para atingir
 * as dimensões esperadas pelo modelo.
 *
 * @param {number} originalWidth - Largura original da imagem.
 * @param {number} originalHeight - Altura original da imagem.
 * @param {number} targetWidth - Largura esperada pelo modelo.
 * @param {number} targetHeight - Altura esperada pelo modelo.
 * @returns {Object} Dimensões redimensionadas e informações de padding.
 */
function calculateResizeDimensions(
    originalWidth,
    originalHeight,
    targetWidth,
    targetHeight
) {
    // Calcula a escala necessária para encaixar a imagem
    // dentro das dimensões do modelo sem distorcer sua proporção.
    const scale = Math.min(
        targetWidth / originalWidth,
        targetHeight / originalHeight
    );

    // Calcula as dimensões finais após o redimensionamento.
    const resizedWidth = Math.round(originalWidth * scale);
    const resizedHeight = Math.round(originalHeight * scale);

    // Calcula quanto espaço será preenchido em cada direção.
    const paddingX = Math.floor((targetWidth - resizedWidth) / 2);
    const paddingY = Math.floor((targetHeight - resizedHeight) / 2);

    return {
        scale,
        resizedWidth,
        resizedHeight,
        paddingX,
        paddingY
    };
}

/**
 * Converte um frame em tensor e prepara sua entrada para o YOLOv5n.
 *
 * Preserva a proporção original da imagem utilizando Letterbox:
 * redimensiona a imagem e adiciona padding até atingir o tamanho do modelo.
 * Sem letterbox, esticar um vídeo 16:9 para 1:1 deformaria os veículos e
 * pioraria a detecção.
 *
 * @param {ImageBitmap|HTMLVideoElement|HTMLCanvasElement} input - Frame original.
 * @param {number} targetWidth - Largura esperada pelo modelo.
 * @param {number} targetHeight - Altura esperada pelo modelo.
 * @param {{x: number, y: number, width: number, height: number}|null} region -
 *        Recorte opcional do frame. Analisar um recorte amplia os objetos
 *        pequenos e distantes, que ficam com poucos pixels na imagem inteira.
 * @returns {{tensor: tf.Tensor4D, metadata: Object}} Tensor e metadados do processamento.
 */
function preprocessImage(input, targetWidth, targetHeight, region = null) {
    // Obtém as dimensões originais do frame.
    const originalWidth = input.width || input.videoWidth;
    const originalHeight = input.height || input.videoHeight;

    if (!originalWidth || !originalHeight) {
        throw new Error('Invalid dimensions for image processing.');
    }

    // Sem recorte, a região é o frame inteiro.
    const crop = region ?? {
        x: 0,
        y: 0,
        width: originalWidth,
        height: originalHeight
    };

    // Calcula o redimensionamento proporcional e o padding necessário.
    const resizeMetadata = calculateResizeDimensions(
        crop.width,
        crop.height,
        targetWidth,
        targetHeight
    );

    const {
        scale,
        resizedWidth,
        resizedHeight,
        paddingX,
        paddingY
    } = resizeMetadata;

    // Executa as operações com tensores e descarta automaticamente
    // os tensores intermediários que não forem retornados.
    const tensor = tf.tidy(() => {
        // Converte os pixels do frame em um tensor RGB.
        // Formato: [altura, largura, 3].
        const fullImage = tf.browser.fromPixels(input);

        // Extrai apenas a região analisada (ou mantém o frame inteiro).
        const image = region
            ? fullImage.slice(
                [crop.y, crop.x, 0],
                [crop.height, crop.width, 3]
            )
            : fullImage;

        // Redimensiona a imagem preservando sua proporção original.
        const resizedImage = tf.image.resizeBilinear(
            image,
            [resizedHeight, resizedWidth]
        );

        // Adiciona padding para atingir as dimensões exigidas pelo YOLOv5n.
        // O valor 114 representa o preenchimento cinza utilizado
        // com frequência no pré-processamento do YOLOv5.
        const paddedImage = tf.pad(
            resizedImage,
            [
                [paddingY, targetHeight - resizedHeight - paddingY],
                [paddingX, targetWidth - resizedWidth - paddingX],
                [0, 0]
            ],
            114
        );

        // Normaliza os valores RGB de 0-255 para 0-1.
        // Adiciona a dimensão batch: [H, W, 3] -> [1, H, W, 3].
        return paddedImage
            .div(255)
            .expandDims(0);
    });

    // Retorna também os metadados necessários para converter
    // as coordenadas das detecções para o frame original.
    return {
        tensor,
        metadata: {
            originalWidth,
            originalHeight,
            cropX: crop.x,
            cropY: crop.y,
            targetWidth,
            targetHeight,
            scale,
            resizedWidth,
            resizedHeight,
            paddingX,
            paddingY
        }
    };
}

export { preprocessImage };
