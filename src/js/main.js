
import { VideoService } from './services/VideoService.js';

const videoInput = document.querySelector('#video-input');
const videoElement = document.querySelector('#source-video');
const statusElement = document.querySelector('#status');

const videoService = new VideoService(videoElement);

// Inicializa o Worker responsável pela inferência.
const worker = new Worker(
    new URL('./workers/detection.worker.js', import.meta.url),
    { type: 'module' }
);

const FRAME_RATE = 5;
const FRAME_INTERVAL = 1000 / FRAME_RATE;

let modelReady = false;
let isProcessing = false;
let animationFrameId = null;
let lastFrameTime = 0;

/**
 * Atualiza o status apresentado na interface.
 */
function updateStatus(message) {
    statusElement.textContent = message;
}

/**
 * Inicia a extração contínua de frames.
 */
function startFrameCapture() {
    if (animationFrameId !== null) return;

    animationFrameId = requestAnimationFrame(processFrame);
}

/**
 * Interrompe o ciclo de extração de frames.
 */
function stopFrameCapture() {
    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

/**
 * Captura e envia frames ao Worker respeitando o limite de processamento.
 *
 * @param {number} timestamp - Timestamp fornecido pelo requestAnimationFrame.
 */
async function processFrame(timestamp) {
    animationFrameId = null;

    if (videoElement.paused || videoElement.ended) return;

    // Aguarda o modelo e evita enviar novos frames durante uma inferência.
    if (
        modelReady &&
        !isProcessing &&
        timestamp - lastFrameTime >= FRAME_INTERVAL
    ) {
        lastFrameTime = timestamp;
        isProcessing = true;

        try {
            // Extrai o frame atual do vídeo.
            const frame = await videoService.captureFrame();

            // Transfere o ImageBitmap para o Worker sem copiar seus dados.
            worker.postMessage(
                {
                    type: 'process-frame',
                    frame
                },
                [frame]
            );
        } catch (error) {
            isProcessing = false;
            console.error('Falha ao capturar frame:', error);
        }
    }

    // Agenda a próxima verificação enquanto o vídeo estiver reproduzindo.
    if (!videoElement.paused && !videoElement.ended) {
        animationFrameId = requestAnimationFrame(processFrame);
    }
}

// Carrega o vídeo selecionado pelo usuário.
videoInput.addEventListener('change', async ({ target }) => {
    const file = target.files[0];

    if (!file) return;

    try {
        const metadata = await videoService.load(file);

        updateStatus(
            `Vídeo carregado: ${metadata.width}x${metadata.height}`
        );

        console.log('Metadados do vídeo:', metadata);
    } catch (error) {
        updateStatus(error.message);
        console.error(error);
    }
});

// Inicia a extração quando o usuário reproduzir o vídeo.
videoElement.addEventListener('play', startFrameCapture);

// Interrompe a extração quando o vídeo for pausado.
videoElement.addEventListener('pause', stopFrameCapture);

// Interrompe a extração ao chegar ao final do vídeo.
videoElement.addEventListener('ended', stopFrameCapture);

// Recebe mensagens enviadas pelo Worker.
worker.onmessage = ({ data }) => {
    switch (data.type) {
        case 'worker-ready':
            console.log('[Worker]', data.message);
            break;

        case 'model-loaded':
            modelReady = true;
            updateStatus('Modelo YOLOv5n pronto para inferência.');

            console.log('[Worker]', data.message);

            // Caso o vídeo já esteja reproduzindo, inicia o processamento.
            if (!videoElement.paused) {
                startFrameCapture();
            }
            break;

        case 'inference-completed':
            isProcessing = false;

            console.log('Detecções do YOLOv5n:', data.detections);

            console.table(data.detections);

            break;

        case 'error':
            isProcessing = false;

            console.error('[Worker Error]', data.message);
            updateStatus(data.message);
            break;
    }
};

// Captura erros inesperados do Worker.
worker.onerror = (error) => {
    isProcessing = false;

    console.error('[Worker Error]', error.message);
    updateStatus('Erro durante o processamento.');
};

// Libera os recursos quando a página for encerrada.
window.addEventListener('beforeunload', () => {
    stopFrameCapture();
    videoService.dispose();
    worker.terminate();
});
