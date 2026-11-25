// ============================================================================
// DetectionProcessor.js — Pós-processamento
//
// O modelo devolve listas soltas de números (caixas, notas, classes). Aqui
// transformamos isso em objetos fáceis de usar, descartando o que for inválido
// ou pouco confiável.
// ============================================================================

// Confiança mínima: detecções com nota abaixo de 0.5 (50%) são ignoradas.
const CLASS_THRESHOLD = 0.5;

/**
 * Transforma as saídas do YOLOv5n em detecções estruturadas.
 *
 * Os parâmetros usam desestruturação: em vez de receber um objeto `data` e
 * escrever data.boxes, já pegamos `boxes`, `scores` etc. direto.
 *
 * @param {Object} data
 * @param {number[]} data.boxes   - Lista plana de coordenadas: [x1,y1,x2,y2, x1,y1,x2,y2, ...]
 * @param {number[]} data.scores  - Confiança de cada detecção.
 * @param {number[]} data.classes - Id da classe de cada detecção.
 * @param {number} data.numDetections - Quantas detecções o modelo considera válidas.
 * @param {string[]} labels - Nomes das classes (o id é a posição na lista).
 * @returns {Array<Object>} Detecções válidas.
 */
function processDetections(
    { boxes, scores, classes, numDetections },
    labels
) {
    const detections = [];

    // O modelo devolve arrays de tamanho fixo, geralmente com "lixo" no final.
    // Usamos o menor entre os valores para nunca ler fora dos limites.
    // (boxes tem 4 números por objeto, por isso dividimos por 4.)
    const detectionCount = Math.min(
        numDetections,
        scores.length,
        classes.length,
        boxes.length / 4
    );

    for (let detectionIndex = 0; detectionIndex < detectionCount; detectionIndex++) {
        const confidence = scores[detectionIndex];
        const classId = classes[detectionIndex];

        // `continue` pula para a próxima detecção. Descartamos se:
        //  - a confiança não for um número válido (NaN, Infinity);
        //  - a confiança for menor que o mínimo;
        //  - o id da classe não for inteiro ou for negativo.
        if (
            !Number.isFinite(confidence) ||
            confidence < CLASS_THRESHOLD ||
            !Number.isInteger(classId) ||
            classId < 0
        ) {
            continue;
        }

        // Traduz o id numérico (ex.: 2) para o nome (ex.: "car").
        const className = labels[classId];

        if (!className) continue; // id sem nome correspondente

        // As coordenadas ficam em sequência no array: a detecção N ocupa as
        // posições de N*4 até N*4+3. `slice` recorta esse pedaço e a
        // desestruturação separa nas 4 variáveis.
        //   (x1, y1) = canto superior esquerdo
        //   (x2, y2) = canto inferior direito
        const [x1, y1, x2, y2] = boxes.slice(
            detectionIndex * 4,
            (detectionIndex + 1) * 4
        );

        // Todas as coordenadas precisam ser números válidos.
        if (![x1, y1, x2, y2].every(Number.isFinite)) {
            continue;
        }

        // Uma caixa válida tem largura e altura positivas.
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
