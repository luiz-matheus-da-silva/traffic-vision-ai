
/**
 * DetectionProcessor.js — Pós-processamento das detecções.
 *
 * O YOLOv5n retorna arrays numéricos contendo coordenadas, confiança
 * e identificadores das classes detectadas.
 *
 * Este módulo transforma esses resultados em objetos estruturados,
 * valida as detecções e recupera as coordenadas do frame original,
 * desfazendo a transformação aplicada pelo Letterbox.
 */

/**
 * Converte as coordenadas normalizadas do modelo para coordenadas
 * em pixels do vídeo original.
 *
 * O Letterbox adiciona padding à imagem para preencher o espaço
 * esperado pelo modelo. Por isso, precisamos remover esse padding
 * e desfazer a escala aplicada durante o pré-processamento.
 *
 * @param {Object} boundingBox - Coordenadas normalizadas do modelo.
 * @param {Object} metadata - Metadados do pré-processamento.
 * @returns {Object} Bounding box em pixels no frame original.
 */
function restoreOriginalCoordinates(boundingBox, metadata) {
    const {
        originalWidth,
        originalHeight,
        cropX = 0,
        cropY = 0,
        targetWidth,
        targetHeight,
        scale,
        paddingX,
        paddingY
    } = metadata;

    // Converte as coordenadas normalizadas (0 a 1) para pixels da entrada do
    // modelo (0 a 640). O modelo exportado já devolve [xMin, yMin, xMax, yMax].
    const modelXMin = boundingBox.xMin * targetWidth;
    const modelYMin = boundingBox.yMin * targetHeight;
    const modelXMax = boundingBox.xMax * targetWidth;
    const modelYMax = boundingBox.yMax * targetHeight;

    // Remove o padding e desfaz o redimensionamento proporcional.
    // Soma o deslocamento do recorte para voltar ao frame completo.
    const xMin = (modelXMin - paddingX) / scale + cropX;
    const yMin = (modelYMin - paddingY) / scale + cropY;
    const xMax = (modelXMax - paddingX) / scale + cropX;
    const yMax = (modelYMax - paddingY) / scale + cropY;

    // Limita as coordenadas às dimensões do vídeo original.
    return {
        xMin: Math.max(0, Math.min(xMin, originalWidth)),
        yMin: Math.max(0, Math.min(yMin, originalHeight)),
        xMax: Math.max(0, Math.min(xMax, originalWidth)),
        yMax: Math.max(0, Math.min(yMax, originalHeight))
    };
}

/**
 * Transforma as saídas do YOLOv5n em detecções estruturadas.
 *
 * @param {Object} data - Dados extraídos dos tensores do modelo.
 * @param {number[]} data.boxes - Coordenadas das bounding boxes.
 * @param {number[]} data.scores - Confiança de cada detecção.
 * @param {number[]} data.classes - Identificadores das classes.
 * @param {number} data.numDetections - Quantidade de detecções válidas.
 * @param {string[]} labels - Nomes das classes disponíveis no modelo.
 * @param {number} classThreshold - Confiança mínima para aceitar uma detecção.
 * @param {Object} metadata - Metadados do pré-processamento Letterbox.
 * @returns {Array<Object>} Lista de detecções válidas.
 */
function processDetections(
    { boxes, scores, classes, numDetections },
    labels,
    classThreshold,
    metadata
) {
    const detections = [];

    // Limita a quantidade de iterações aos dados realmente disponíveis.
    const detectionCount = Math.min(
        Math.floor(numDetections),
        scores.length,
        classes.length,
        Math.floor(boxes.length / 4)
    );

    for (let detectionIndex = 0; detectionIndex < detectionCount; detectionIndex++) {
        const confidence = scores[detectionIndex];
        const classId = classes[detectionIndex];

        // Descarta resultados inválidos ou abaixo da confiança mínima.
        if (
            !Number.isFinite(confidence) ||
            confidence < classThreshold ||
            !Number.isInteger(classId) ||
            classId < 0
        ) {
            continue;
        }

        // Recupera o nome correspondente ao identificador da classe.
        const className = labels[classId];

        if (!className) continue;

        // Extrai as quatro coordenadas da bounding box.
        const [xMin, yMin, xMax, yMax] = boxes.slice(
            detectionIndex * 4,
            (detectionIndex + 1) * 4
        );

        // Verifica se todas as coordenadas são números válidos.
        if (![xMin, yMin, xMax, yMax].every(Number.isFinite)) {
            continue;
        }

        // Descarta caixas sem largura ou altura.
        if (xMax <= xMin || yMax <= yMin) {
            continue;
        }

        // Recupera as coordenadas em pixels do vídeo original.
        const boundingBox = restoreOriginalCoordinates(
            { xMin, yMin, xMax, yMax },
            metadata
        );

        // Descarta caixas que ficaram inválidas após a conversão.
        if (
            boundingBox.xMax <= boundingBox.xMin ||
            boundingBox.yMax <= boundingBox.yMin
        ) {
            continue;
        }

        // Adiciona a detecção estruturada à lista de resultados.
        detections.push({
            classId,
            className,
            confidence,
            boundingBox
        });
    }

    return detections;
}

/**
 * Calcula a interseção sobre união (IoU) entre duas caixas.
 */
function calculateIoU(a, b) {
    const interWidth = Math.min(a.xMax, b.xMax) - Math.max(a.xMin, b.xMin);
    const interHeight = Math.min(a.yMax, b.yMax) - Math.max(a.yMin, b.yMin);

    if (interWidth <= 0 || interHeight <= 0) return 0;

    const intersection = interWidth * interHeight;
    const areaA = (a.xMax - a.xMin) * (a.yMax - a.yMin);
    const areaB = (b.xMax - b.xMin) * (b.yMax - b.yMin);

    return intersection / (areaA + areaB - intersection);
}

/**
 * Une detecções de várias passadas (frame inteiro + recortes) e remove
 * duplicatas da mesma classe, mantendo a de maior confiança.
 *
 * @param {Array<Object>} detections - Detecções de todas as passadas.
 * @param {number} iouThreshold - Sobreposição a partir da qual são duplicatas.
 */
function mergeDetections(detections, iouThreshold = 0.5) {
    const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
    const kept = [];

    for (const candidate of sorted) {
        const isDuplicate = kept.some(
            (other) =>
                other.classId === candidate.classId &&
                calculateIoU(other.boundingBox, candidate.boundingBox) > iouThreshold
        );

        if (!isDuplicate) kept.push(candidate);
    }

    return kept;
}

export { processDetections, mergeDetections };
