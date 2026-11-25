
const CLASS_THRESHOLD = 0.5;

/**
 * Transforma as saídas do YOLOv5n em detecções estruturadas.
 *
 * @param {Object} data - Dados extraídos dos tensores do modelo.
 * @param {Array} labels - Classes disponíveis no modelo.
 * @returns {Array<Object>} Detecções válidas.
 */
function processDetections(
    { boxes, scores, classes, numDetections },
    labels
) {
    const detections = [];

    // Limita o processamento à quantidade de detecções retornada pelo modelo.
    const detectionCount = Math.min(
        numDetections,
        scores.length,
        classes.length,
        boxes.length / 4
    );

    for (let index = 0; index < detectionCount; index++) {
        const confidence = scores[index];
        const classId = classes[index];

        // Ignora valores inválidos e detecções com baixa confiança.
        if (
            !Number.isFinite(confidence) ||
            confidence < CLASS_THRESHOLD ||
            !Number.isInteger(classId) ||
            classId < 0
        ) {
            continue;
        }

        // Recupera o nome da classe identificada.
        const className = labels[classId];

        if (!className) continue;

        // Extrai as coordenadas da bounding box.
        const [x1, y1, x2, y2] = boxes.slice(
            index * 4,
            (index + 1) * 4
        );

        // Valida as coordenadas retornadas pelo modelo.
        if (![x1, y1, x2, y2].every(Number.isFinite)) {
            continue;
        }

        if (x2 <= x1 || y2 <= y1) continue;

        detections.push({
            classId,
            className,
            confidence,
            boundingBox: {
                xMin: x1,
                yMin: y1,
                xMax: x2,
                yMax: y2
            }
        });
    }

    return detections;
}

export { processDetections };
