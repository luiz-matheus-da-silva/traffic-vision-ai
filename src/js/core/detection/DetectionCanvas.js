
/**
 * DetectionCanvas.js — Renderização das detecções.
 *
 * Responsável por desenhar bounding boxes e informações
 * dos objetos identificados pelo modelo sobre o vídeo.
 */

// Uma cor por classe facilita distinguir os veículos de relance.
const CLASS_COLORS = Object.freeze({
    car: '#2DD4BF',
    truck: '#F5A524',
    bus: '#60A5FA',
    motorcycle: '#F472B6'
});

const DEFAULT_COLOR = '#2DD4BF';

class DetectionCanvas {
    constructor(canvas) {
        if (!(canvas instanceof HTMLCanvasElement)) {
            throw new Error('The provided element must be an HTMLCanvasElement.');
        }

        this.canvas = canvas;
        this.context = canvas.getContext('2d');

        if (!this.context) {
            throw new Error('Could not get the Canvas 2D context.');
        }
    }

    /**
     * Ajusta a resolução interna do Canvas.
     */
    resize(width, height) {
        if (width <= 0 || height <= 0) {
            throw new Error('Canvas dimensions must be positive.');
        }

        this.canvas.width = width;
        this.canvas.height = height;
    }

    /**
     * Remove os desenhos anteriores.
     */
    clear() {
        this.context.clearRect(
            0,
            0,
            this.canvas.width,
            this.canvas.height
        );
    }

    /**
     * Desenha todas as detecções recebidas.
     *
     * @param {Array<Object>} detections - Detecções em pixels do vídeo.
     */
    draw(detections) {
        this.clear();

        for (const detection of detections) {
            this.drawBoundingBox(detection);
        }
    }

    /**
     * Desenha uma bounding box e sua identificação.
     */
    drawBoundingBox(detection) {
        const { className, confidence, boundingBox } = detection;

        const { xMin, yMin, xMax, yMax } = boundingBox;

        // Calcula as dimensões da bounding box.
        const boxWidth = xMax - xMin;
        const boxHeight = yMax - yMin;

        // Define o tamanho dos elementos proporcionalmente ao vídeo.
        // O teto evita rótulos gigantes em vídeos Full HD ou 4K.
        const scale = Math.min(this.canvas.width / 640, 2.5);
        const color = CLASS_COLORS[className] ?? DEFAULT_COLOR;

        const lineWidth = Math.max(2, scale * 2);
        const fontSize = Math.max(12, scale * 14);

        // Desenha o contorno da bounding box.
        this.context.strokeStyle = color;
        this.context.lineWidth = lineWidth;

        this.context.strokeRect(
            xMin,
            yMin,
            boxWidth,
            boxHeight
        );

        // Monta o rótulo da detecção.
        const label = `${className} ${(confidence * 100).toFixed(1)}%`;

        this.context.font = `bold ${fontSize}px Arial`;

        const textWidth = this.context.measureText(label).width;
        const textHeight = fontSize + 6;

        // Posiciona o rótulo sem ultrapassar os limites do Canvas.
        const labelX = Math.max(
            0,
            Math.min(xMin, this.canvas.width - textWidth - 8)
        );

        const labelY = Math.max(textHeight, yMin);

        // Fundo do rótulo.
        this.context.fillStyle = color;

        this.context.fillRect(
            labelX,
            labelY - textHeight,
            textWidth + 8,
            textHeight
        );

        // Texto do rótulo.
        this.context.fillStyle = '#0d1520';

        this.context.fillText(
            label,
            labelX + 4,
            labelY - 5
        );
    }
}

export { DetectionCanvas };
