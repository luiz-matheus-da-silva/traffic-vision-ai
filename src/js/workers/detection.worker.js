// ============================================================================
// detection.worker.js — Roda em segundo plano (Web Worker)
//
// Este arquivo NÃO tem acesso à página (document, janela etc.). Ele só se
// comunica com o main.js por mensagens:
//   - recebe via `self.onmessage`  (self = o próprio Worker)
//   - envia  via `self.postMessage`
//
// Fluxo de cada frame:
//   imagem -> tensor (números) -> modelo YOLOv5n -> saídas -> lista de veículos
// ============================================================================

// TensorFlow.js: biblioteca de IA que roda no navegador.
import * as tf from '@tensorflow/tfjs';

import { loadModelAndLabels } from '../core/inference/ModelLoader.js';
import { preprocessImage } from '../core/inference/ImagePreprocessor.js';
import { runInference } from '../core/inference/InferenceEngine.js';
import { processDetections } from '../core/detection/DetectionProcessor.js';

// Tamanho de imagem que o YOLOv5n espera receber (640 x 640 pixels).
const MODEL_INPUT_SIZE = 640;

// Estado do Worker. Começam vazios e são preenchidos em initialize().
let yoloModel = null;      // o modelo de IA carregado
let classLabels = [];      // nomes das classes: ["person", "car", "bus", ...]
let classThreshold = 0.5;  // confiança mínima (0 a 1) vinda do ModelLoader

/**
 * Prepara o TensorFlow.js e carrega o modelo. Roda uma vez, ao iniciar.
 */
async function initialize() {
    try {
        // "Backend" é o motor que faz as contas. 'webgl' usa a placa de vídeo (GPU),
        // muito mais rápido que a CPU para IA.
        await tf.setBackend('webgl');
        await tf.ready(); // espera o backend estar pronto

        // Baixa e prepara o modelo e a lista de classes.
        const loaded = await loadModelAndLabels();

        yoloModel = loaded.model;
        classLabels = loaded.labels;
        classThreshold = loaded.classThreshold;

        // Avisa o main.js que já pode mandar frames.
        self.postMessage({
            type: 'model-loaded',
            message: 'YOLOv5n carregado com sucesso.'
        });

    } catch (error) {
        // Em caso de falha, avisa o main.js em vez de quebrar silenciosamente.
        self.postMessage({
            type: 'error',
            message: error.message
        });
    }
}

/**
 * Chamada toda vez que o main.js envia uma mensagem para este Worker.
 * `({ data })` pega o conteúdo da mensagem.
 */
self.onmessage = async ({ data }) => {
    // Só nos interessam mensagens do tipo 'process-frame'.
    if (data.type !== 'process-frame') return;

    // Se chegou frame antes do modelo carregar, responde com erro.
    if (!yoloModel) {
        self.postMessage({
            type: 'error',
            message: 'Modelo ainda não está carregado.'
        });

        return;
    }

    // Guardamos o frame numa variável para poder fechá-lo no `finally`.
    let frame = data.frame;

    try {
        // 1) PRÉ-PROCESSAMENTO: transforma a imagem em um tensor
        //    (uma grade de números) no formato que o modelo entende.
        const inputTensor = preprocessImage(frame, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

        // O tensor já guarda os pixels, então a imagem original pode ser descartada.
        frame.close();
        frame = null; // marca como já fechado (o `finally` abaixo não fecha de novo)

        // 2) INFERÊNCIA: o modelo analisa o tensor e devolve suas "respostas".
        const outputTensors = await runInference(yoloModel, inputTensor);

        let detections = [];

        try {
            // 3) LEITURA DOS RESULTADOS: converte os tensores em arrays comuns do JavaScript.
            //    O YOLOv5n (com NMS embutido) devolve 4 saídas, nesta ordem:
            //      - boxes:   coordenadas das caixas (4 números por objeto)
            //      - scores:  confiança de cada detecção (0 a 1)
            //      - classes: id da classe de cada detecção
            //      - count:   quantas detecções válidas existem
            //    `tensor.data()` é assíncrono; Promise.all espera todas terminarem.
            const [boxes, scores, classes, count] = await Promise.all(
                outputTensors.map(async (tensor) => {
                    return Array.from(await tensor.data());
                })
            );

            // 4) PÓS-PROCESSAMENTO: monta objetos legíveis ({className, confidence, boundingBox}).
            detections = processDetections(
                {
                    boxes,
                    scores,
                    classes,
                    numDetections: count[0]
                },
                classLabels
            );
        } finally {
            // Tensores ficam na memória da GPU e o JavaScript NÃO os limpa
            // sozinho. Precisamos chamar dispose() para evitar vazamento de memória.
            outputTensors.forEach((tensor) => tensor.dispose());
        }

        // Devolve o resultado ao main.js.
        self.postMessage({
            type: 'inference-completed',
            detections
        });

    } catch (error) {
        // Qualquer erro acima cai aqui e é reportado ao main.js.
        self.postMessage({
            type: 'error',
            message: error.message
        });
    } finally {
        // `finally` roda SEMPRE (com ou sem erro). Se o frame ainda estiver aberto
        // (erro antes do close), fechamos aqui. O `?.` ignora se `frame` for null.
        frame?.close();
    }
};

// Avisa o main.js que o Worker acabou de ser criado.
self.postMessage({
    type: 'worker-ready',
    message: 'Detection Worker inicializado.'
});

// Começa a carregar o modelo já, sem esperar o primeiro frame.
initialize();
