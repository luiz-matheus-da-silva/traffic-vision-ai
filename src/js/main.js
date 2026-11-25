
/**
 * main.js — Ponto de entrada da aplicação.
 *
 * Responsabilidades:
 * 1. Carregar o vídeo selecionado pelo usuário.
 * 2. Capturar frames periodicamente.
 * 3. Enviar os frames para o Web Worker.
 * 4. Receber as detecções produzidas pelo YOLOv5n.
 * 5. Atualizar o Canvas, os contadores e a lista de detecções recentes.
 *
 * Esta thread cuida só da interface. Qualquer trabalho pesado (TensorFlow)
 * fica no Worker.
 *
 * A inferência acontece no Worker para não bloquear a Main Thread.
 */

import { VideoService } from './services/VideoService.js';
import { DetectionCanvas } from './core/detection/DetectionCanvas.js';

// ============================================================================
// ELEMENTOS DA INTERFACE
// ============================================================================

const videoInput = document.querySelector('#video-input');
const videoElement = document.querySelector('#source-video');
const canvasElement = document.querySelector('#detection-canvas');
const statusElement = document.querySelector('#status');
const statusDot = document.querySelector('#status-dot');
const videoEmpty = document.querySelector('#video-empty');
const modelLoading = document.querySelector('#model-loading');
const modelLoadingText = document.querySelector('#model-loading-text');

const confidenceRange = document.querySelector('#confidence-range');
const confidenceValue = document.querySelector('#confidence-value');
const fpsRange = document.querySelector('#fps-range');
const fpsValue = document.querySelector('#fps-value');
const fpsStatus = document.querySelector('#fps-status');
const latencyStatus = document.querySelector('#latency-status');
const classToggles = document.querySelectorAll('.class-toggle');

const statElements = {
    total: document.querySelector('#stat-total'),
    car: document.querySelector('#stat-car'),
    truck: document.querySelector('#stat-truck'),
    bus: document.querySelector('#stat-bus')
};

const recentList = document.querySelector('#recent-list');
const recentEmpty = document.querySelector('#recent-empty');
const recentCount = document.querySelector('#recent-count');

// Valida se os elementos necessários existem no HTML.
if (
    !videoInput || !videoElement || !canvasElement || !statusElement ||
    !confidenceRange || !fpsRange || !recentList
) {
    throw new Error('Could not find all required interface elements.');
}

// ============================================================================
// COMPONENTES
// ============================================================================

// Responsável por carregar o vídeo e capturar frames.
const videoService = new VideoService(videoElement);

// Responsável por desenhar as detecções sobre o vídeo.
const detectionCanvas = new DetectionCanvas(canvasElement);

// ============================================================================
// WEB WORKER
// ============================================================================

const worker = new Worker(
    new URL('./workers/detection.worker.js', import.meta.url),
    { type: 'module' }
);

// ============================================================================
// CONFIGURAÇÕES
// ============================================================================

// Quantidade máxima de frames enviados para análise por segundo.
// O valor pode ser alterado pelo usuário na interface.
let framesPerSecond = Number(fpsRange.value);

// Intervalo mínimo entre capturas.
let minMsBetweenFrames = 1000 / framesPerSecond;

// Filtros aplicados às detecções antes de exibi-las.
// O Worker entrega tudo acima de 10% de confiança; aqui o usuário refina.
// Filtrar na Main Thread permite mudar o slider sem reprocessar nada.
let confidenceThreshold = Number(confidenceRange.value) / 100;
const enabledClasses = new Set(
    [...classToggles].filter((toggle) => toggle.checked).map((toggle) => toggle.dataset.class)
);

// Quantidade máxima de itens mantidos em "Recent detections".
const MAX_RECENT_ITEMS = 30;

// ============================================================================
// ESTADO DA APLICAÇÃO
// ============================================================================

let isModelReady = false;
let isInferenceRunning = false;

let animationFrameRequestId = null;
let lastCaptureTimestamp = 0;

// ============================================================================
// UTILITÁRIOS
// ============================================================================

/**
 * Atualiza a mensagem de status da aplicação.
 *
 * @param {string} message - Mensagem exibida na interface.
 */
function updateStatus(message, state = 'idle') {
    statusElement.textContent = message;

    const colors = { idle: 'bg-secondary', ready: 'bg-success', error: 'bg-danger' };

    statusDot.className = `d-inline-block rounded-circle ${colors[state] ?? colors.idle}`;
}

// Motos são pequenas e muitas vezes parcialmente cobertas pelo piloto, então
// o modelo dá a elas scores mais baixos que a carros. Para essas classes o
// limiar do slider é multiplicado por um fator menor que 1.
const CLASS_THRESHOLD_FACTOR = Object.freeze({ motorcycle: 0.5 });

/**
 * Mantém apenas as detecções que o usuário pediu para ver:
 * classes de veículos habilitadas e confiança mínima (por classe).
 */
function filterDetections(detections) {
    return detections.filter((detection) => {
        const factor = CLASS_THRESHOLD_FACTOR[detection.className] ?? 1;

        return (
            enabledClasses.has(detection.className) &&
            detection.confidence >= confidenceThreshold * factor
        );
    });
}

/**
 * Atualiza os contadores do frame atual.
 */
function updateStats(detections) {
    const count = (className) =>
        detections.filter((detection) => detection.className === className).length;

    statElements.total.textContent = detections.length;
    statElements.car.textContent = count('car');
    statElements.truck.textContent = count('truck');
    statElements.bus.textContent = count('bus');
}

/**
 * Formata o tempo do vídeo como mm:ss.
 */
function formatVideoTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const rest = Math.floor(seconds % 60);

    return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`;
}

/**
 * Adiciona as detecções mais confiáveis do frame à lista recente.
 */
function addRecentDetections(detections) {
    if (detections.length === 0) return;

    const time = formatVideoTime(videoElement.currentTime);

    const topDetections = [...detections]
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3);

    // textContent evita injetar HTML a partir de dados externos.
    for (const detection of topDetections) {
        const item = document.createElement('li');
        item.className = 'list-group-item d-flex justify-content-between align-items-center small';

        const name = document.createElement('span');
        name.className = 'text-capitalize fw-semibold';
        name.textContent = detection.className;

        const meta = document.createElement('span');
        meta.className = 'text-secondary tabular';
        meta.textContent = `${(detection.confidence * 100).toFixed(0)}% · ${time}`;

        item.append(name, meta);
        recentList.prepend(item);
    }

    while (recentList.children.length > MAX_RECENT_ITEMS) {
        recentList.lastElementChild.remove();
    }

    recentEmpty.hidden = true;
    recentCount.textContent = recentList.children.length;
}

/**
 * Limpa contadores, lista e caixas (ao trocar de vídeo).
 */
function resetResults() {
    detectionCanvas.clear();
    updateStats([]);

    recentList.replaceChildren();
    recentEmpty.hidden = false;
    recentCount.textContent = '0';
}

/**
 * Ajusta a resolução interna do Canvas às dimensões originais do vídeo.
 *
 * As bounding boxes retornadas pelo DetectionProcessor utilizam
 * coordenadas em pixels do frame original. Por isso a resolução INTERNA do
 * canvas (width/height) é igual à do vídeo; o tamanho exibido na tela é
 * controlado pelo CSS. Assim as caixas não precisam de conversão de escala.
 */
function resizeDetectionCanvas() {
    if (!videoElement.videoWidth || !videoElement.videoHeight) return;

    detectionCanvas.resize(
        videoElement.videoWidth,
        videoElement.videoHeight
    );
}

// ============================================================================
// CAPTURA DE FRAMES
// ============================================================================

/**
 * Inicia o ciclo de captura de frames.
 *
 * requestAnimationFrame agenda a próxima execução antes da renderização
 * de um novo frame da página.
 */
function startFrameCapture() {
    if (animationFrameRequestId !== null) return;

    animationFrameRequestId = requestAnimationFrame(processFrame);
}

/**
 * Interrompe o ciclo de captura de frames.
 */
function stopFrameCapture() {
    if (animationFrameRequestId !== null) {
        cancelAnimationFrame(animationFrameRequestId);
        animationFrameRequestId = null;
    }
}

/**
 * Verifica se chegou o momento de capturar um novo frame.
 *
 * O requestAnimationFrame executa aproximadamente na frequência
 * de atualização da tela. O intervalo configurado limita a captura
 * à frequência escolhida (Target FPS).
 *
 * @param {number} timestamp - Tempo atual fornecido pelo navegador.
 */
async function processFrame(timestamp) {
    // O agendamento atual já foi executado.
    animationFrameRequestId = null;

    // Não captura frames se o vídeo estiver parado.
    if (videoElement.paused || videoElement.ended) return;

    const canCapture =
        isModelReady &&
        // Contrapressão: só um frame em análise por vez. Se o Worker for mais
        // lento que o Target FPS, os frames intermediários são descartados
        // em vez de formar uma fila (que aumentaria o atraso das caixas).
        !isInferenceRunning &&
        timestamp - lastCaptureTimestamp >= minMsBetweenFrames;

    if (canCapture) {
        lastCaptureTimestamp = timestamp;
        isInferenceRunning = true;

        let frame = null;

        try {
            // Captura o frame atual do vídeo.
            frame = await videoService.captureFrame();

            // O vídeo pode ter sido pausado durante a captura.
            if (videoElement.paused || videoElement.ended) {
                frame.close();
                frame = null;
                isInferenceRunning = false;
                return;
            }

            // Transfere o ImageBitmap para o Worker sem copiar seus dados.
            // O segundo argumento é a lista de "transferíveis": após o envio,
            // este lado perde o acesso ao objeto (propriedade movida).
            worker.postMessage(
                {
                    type: 'process-frame',
                    frame
                },
                [frame]
            );

            // A propriedade do frame foi transferida para o Worker.
            frame = null;

        } catch (error) {
            // Libera recursos caso a captura ou transferência falhe.
            frame?.close();

            isInferenceRunning = false;

            console.error('Failed to capture or send frame:', error);
            updateStatus('Failed to capture a video frame.');
        }
    }

    // Continua verificando enquanto o vídeo estiver sendo reproduzido.
    if (!videoElement.paused && !videoElement.ended) {
        animationFrameRequestId = requestAnimationFrame(processFrame);
    }
}

// ============================================================================
// EVENTOS DO VÍDEO
// ============================================================================

/**
 * Carrega o vídeo selecionado pelo usuário.
 */
videoInput.addEventListener('change', async ({ target }) => {
    const file = target.files[0];

    if (!file) return;

    // Limpa o input para que escolher o MESMO arquivo de novo também dispare
    // o evento "change" (útil para reiniciar a análise).
    target.value = '';

    stopFrameCapture();

    // Remove os resultados do vídeo anterior.
    resetResults();

    try {
        updateStatus('Loading video...');

        const metadata = await videoService.load(file);

        // Sincroniza a resolução do Canvas com o vídeo carregado.
        resizeDetectionCanvas();

        videoEmpty.hidden = true;

        updateStatus(
            `Video loaded: ${metadata.width}x${metadata.height}`,
            isModelReady ? 'ready' : 'idle'
        );

    } catch (error) {
        updateStatus(error.message, 'error');
        console.error('Failed to load video:', error);
    }
});

// Ajusta o Canvas quando os metadados do vídeo estiverem disponíveis.
videoElement.addEventListener('loadedmetadata', resizeDetectionCanvas);

// Inicia a captura quando o vídeo começa a reproduzir.
videoElement.addEventListener('play', startFrameCapture);

// Interrompe a captura quando o vídeo é pausado.
videoElement.addEventListener('pause', stopFrameCapture);

// Interrompe a captura quando o vídeo termina.
videoElement.addEventListener('ended', stopFrameCapture);

// ============================================================================
// CONFIGURAÇÕES DA INTERFACE
// ============================================================================

confidenceRange.addEventListener('input', () => {
    confidenceThreshold = Number(confidenceRange.value) / 100;
    confidenceValue.textContent = `${confidenceRange.value}%`;
});

fpsRange.addEventListener('input', () => {
    framesPerSecond = Number(fpsRange.value);
    minMsBetweenFrames = 1000 / framesPerSecond;

    fpsValue.textContent = framesPerSecond;
    fpsStatus.textContent = framesPerSecond;
});

for (const toggle of classToggles) {
    toggle.addEventListener('change', () => {
        if (toggle.checked) {
            enabledClasses.add(toggle.dataset.class);
        } else {
            enabledClasses.delete(toggle.dataset.class);
        }
    });
}

// ============================================================================
// MENSAGENS DO WEB WORKER
// ============================================================================

worker.onmessage = ({ data }) => {
    switch (data.type) {

        case 'worker-ready':
            console.log('[Worker]', data.message);
            break;

        case 'model-loaded':
            isModelReady = true;

            // Libera a interface: a partir daqui já é possível analisar vídeos.
            modelLoading.hidden = true;

            updateStatus('Model ready. Select a video and press play.', 'ready');

            console.log('[Worker]', data.message);

            // Caso o vídeo já esteja reproduzindo, inicia a captura.
            if (!videoElement.paused && !videoElement.ended) {
                startFrameCapture();
            }

            break;

        case 'inference-completed':
            // O Worker terminou e pode receber outro frame.
            isInferenceRunning = false;

            {
                const visibleDetections = filterDetections(data.detections);

                latencyStatus.textContent = `${data.latencyMs} ms`;

                // Bounding boxes sobre o vídeo.
                detectionCanvas.draw(visibleDetections);

                updateStats(visibleDetections);
                addRecentDetections(visibleDetections);
            }

            break;

        case 'error':
            isInferenceRunning = false;

            console.error('[Worker Error]', data.message);

            updateStatus(data.message, 'error');

            // Se o erro ocorreu durante o carregamento, o overlay continua
            // visível, mas mostra o problema em vez de girar para sempre.
            if (!isModelReady) {
                modelLoadingText.textContent = data.message;
                modelLoading.querySelector('.spinner-border').hidden = true;
            }

            break;

        default:
            console.warn('[Worker] Unrecognized message:', data);
    }
};

// ============================================================================
// TRATAMENTO DE ERROS DO WORKER
// ============================================================================

worker.onerror = (error) => {
    isInferenceRunning = false;

    console.error('[Worker Error]', error.message);

    updateStatus('Error during AI processing.', 'error');

    if (!isModelReady) {
        modelLoadingText.textContent = 'Failed to start the detection worker.';
        modelLoading.querySelector('.spinner-border').hidden = true;
    }
};

worker.onmessageerror = (error) => {
    isInferenceRunning = false;

    console.error('[Worker Message Error]', error);

    updateStatus('Failed to receive data from the Worker.', 'error');
};

// ============================================================================
// LIMPEZA DE RECURSOS
// ============================================================================

window.addEventListener('beforeunload', () => {
    stopFrameCapture();

    videoService.dispose();

    worker.terminate();
});
