/**
 * detection.worker.js — Pipeline de detecção em segundo plano.
 *
 * Roda em uma thread separada da página, então a inferência (pesada) não
 * trava o vídeo nem a interface. A comunicação com a Main Thread é feita
 * apenas por mensagens (postMessage), sem memória compartilhada.
 *
 * Mensagens enviadas para a Main Thread:
 * - worker-ready       : o Worker foi criado.
 * - model-loaded       : o modelo terminou de carregar.
 * - inference-completed: detecções de um frame + latência em ms.
 * - error              : algo falhou (com texto para exibir).
 *
 * Mensagem recebida:
 * - process-frame      : { frame: ImageBitmap } — um frame para analisar.
 */

import * as tf from '@tensorflow/tfjs';

import { loadModelAndLabels } from '../core/inference/ModelLoader.js';
import { preprocessImage } from '../core/inference/ImagePreprocessor.js';
import { runInference } from '../core/inference/InferenceEngine.js';
import { processDetections, mergeDetections } from '../core/detection/DetectionProcessor.js';
import { MODEL_CONFIG } from '../config/ModelConfig.js';

// Estado do Worker. Como o módulo é carregado uma vez, estas variáveis
// vivem enquanto a página estiver aberta.
// Estado compartilhado durante o ciclo de vida do Worker.
let yoloModel = null;
let classLabels = [];
let classThreshold = MODEL_CONFIG.confidenceThreshold;

// Controle adaptativo de carga.
//
// Por que existe: analisar recortes melhora a detecção de objetos pequenos,
// mas multiplica o custo por frame. Se a máquina não acompanhar, as caixas
// ficam atrasadas em relação ao vídeo. Então medimos o tempo médio e, acima
// do limite, voltamos a analisar só o frame inteiro (uma única passada).
//
// A média móvel exponencial (LATENCY_SMOOTHING) evita que um único frame
// lento desligue os recortes: cada medida novo pesa 30%.
const MAX_INFERENCE_MS_WITH_TILES = 220;
const LATENCY_SMOOTHING = 0.3;

let averageInferenceMs = 0;
let isTilingEnabled = true;

let isInitialized = false;
let isProcessing = false;

/**
 * Inicializa o TensorFlow.js e carrega o modelo YOLOv5n.
 *
 * Essa operação acontece apenas uma vez durante a inicialização
 * do Worker, evitando carregar o modelo a cada frame recebido.
 */
async function initialize() {
    try {
        // Configura o WebGL como backend para executar as operações na GPU.
        const backendReady = await tf.setBackend('webgl');

        if (!backendReady) {
            throw new Error('Could not initialize the WebGL backend.');
        }

        await tf.ready();

        // Carrega o modelo e suas classes.
        const loaded = await loadModelAndLabels();

        yoloModel = loaded.model;
        classLabels = loaded.labels;
        classThreshold = loaded.classThreshold;

        isInitialized = true;

        // Informa à Main Thread que o modelo está pronto.
        self.postMessage({
            type: 'model-loaded',
            message: 'YOLOv5n loaded successfully.'
        });

    } catch (error) {
        self.postMessage({
            type: 'error',
            message: `Failed to initialize the model: ${error.message}`
        });
    }
}

/**
 * Define as regiões analisadas em cada frame.
 *
 * Por que recortar: o modelo recebe 640×640. Um vídeo Full HD é reduzido
 * a ~33% do tamanho, e carros distantes viram poucos pixels (o modelo perde
 * confiança neles). Um recorte de 60% da largura é reduzido bem menos
 * (~59% em vez de ~33%), então os objetos chegam maiores ao modelo.
 *
 * Os dois recortes se sobrepõem em 20% para que um veículo na divisa
 * apareça inteiro em pelo menos um deles. Sem recortes, usa o frame inteiro
 * (`null` = sem recorte).
 *
 * @returns {Array<{x: number, y: number, width: number, height: number}|null>}
 */
function buildRegions(width, height) {
    if (!isTilingEnabled) return [null];

    const regions = [];

    const isLandscape = width > height * 1.2;
    const isPortrait = height > width * 1.2;

    if (isLandscape) {
        const tileWidth = Math.round(width * 0.6);

        regions.push(
            { x: 0, y: 0, width: tileWidth, height },
            { x: width - tileWidth, y: 0, width: tileWidth, height }
        );
    } else if (isPortrait) {
        const tileHeight = Math.round(height * 0.6);

        regions.push(
            { x: 0, y: 0, width, height: tileHeight },
            { x: 0, y: height - tileHeight, width, height: tileHeight }
        );
    }

    // Frames quase quadrados não ganham nada com recortes.
    return regions.length > 0 ? regions : [null];
}

/**
 * Executa o pipeline completo em uma região do frame.
 *
 * Retorna detecções já em coordenadas do frame completo (o deslocamento do
 * recorte é somado no pós-processamento), então os resultados de regiões
 * diferentes podem ser combinados diretamente.
 */
async function detectInRegion(frame, region) {
    // 1. Pré-processamento (letterbox) da região.
    const { tensor: inputTensor, metadata } = preprocessImage(
        frame,
        MODEL_CONFIG.inputWidth,
        MODEL_CONFIG.inputHeight,
        region
    );

    // 2. Inferência.
    const outputTensors = await runInference(yoloModel, inputTensor);

    try {
        // 3. Extração dos resultados.
        // O modelo devolve 4 tensores: caixas [1,100,4], scores [1,100],
        // classes [1,100] e a quantidade de detecções válidas [1].
        // .data() copia da GPU para a CPU (é assíncrono).
        const [boxes, scores, classes, count] = await Promise.all(
            outputTensors.map(async (tensor) => Array.from(await tensor.data()))
        );

        // 4. Pós-processamento: coordenadas voltam ao frame completo.
        return processDetections(
            { boxes, scores, classes, numDetections: count[0] },
            classLabels,
            classThreshold,
            metadata
        );
    } finally {
        // Tensores vivem na memória da GPU e não são coletados pelo GC:
        // é obrigatório chamar dispose(), senão há vazamento de memória.
        outputTensors.forEach((tensor) => tensor.dispose());
    }
}

/**
 * Recebe frames da Main Thread e executa o pipeline de detecção.
 *
 * Fluxo:
 * Frame -> Pré-processamento -> Inferência -> Pós-processamento.
 */
self.onmessage = async ({ data }) => {
    // Ignora mensagens que não representam solicitações de inferência.
    if (data.type !== 'process-frame') return;

    let frame = data.frame;

    // Impede a execução antes do carregamento do modelo.
    if (!isInitialized || !yoloModel) {
        frame?.close();

        self.postMessage({
            type: 'error',
            message: 'The model is not available yet.'
        });

        return;
    }

    // Evita executar duas inferências simultaneamente (a Main Thread já só
    // envia um frame por vez; esta é uma proteção extra).
    if (isProcessing) {
        frame?.close();
        return;
    }

    isProcessing = true;

    try {
        // Cada região (frame inteiro + recortes) passa pelo pipeline completo.
        // As passadas são sequenciais para não saturar a memória da GPU.
        const startTime = performance.now();
        const regions = buildRegions(frame.width, frame.height);
        const allDetections = [];

        for (const region of regions) {
            allDetections.push(...await detectInRegion(frame, region));
        }

        // O frame original pode ser liberado após todas as passadas.
        frame.close();
        frame = null;

        // Une as passadas e remove duplicatas nas áreas sobrepostas.
        const detections = regions.length > 1
            ? mergeDetections(allDetections)
            : allDetections;

        // Mede a duração e liga/desliga os recortes conforme a carga.
        const elapsedMs = performance.now() - startTime;

        averageInferenceMs = averageInferenceMs === 0
            ? elapsedMs
            : averageInferenceMs +
              (elapsedMs - averageInferenceMs) * LATENCY_SMOOTHING;

        if (isTilingEnabled && averageInferenceMs > MAX_INFERENCE_MS_WITH_TILES) {
            isTilingEnabled = false;
            // Estimativa do custo sem recortes, para poder reavaliar depois.
            averageInferenceMs /= regions.length;
        } else if (
            !isTilingEnabled &&
            averageInferenceMs * 2 < MAX_INFERENCE_MS_WITH_TILES
        ) {
            isTilingEnabled = true;
        }

        // Envia as detecções para a Main Thread.
        self.postMessage({
            type: 'inference-completed',
            detections,
            latencyMs: Math.round(elapsedMs)
        });

    } catch (error) {
        // Encaminha erros de processamento para a Main Thread.
        self.postMessage({
            type: 'error',
            message: error.message
        });

    } finally {
        // Garante a liberação do ImageBitmap mesmo quando ocorre uma falha.
        frame?.close();

        // Libera o Worker para receber o próximo frame.
        isProcessing = false;
    }
};

// Informa à Main Thread que o Worker foi criado.
self.postMessage({
    type: 'worker-ready',
    message: 'Detection Worker initialized.'
});

// Inicia o carregamento do modelo.
initialize();
