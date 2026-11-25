// ============================================================================
// main.js — Arquivo principal (roda na "Main Thread", a thread da página)
//
// Papel deste arquivo:
//   1. Ler o vídeo que o usuário escolheu.
//   2. A cada ~200ms, tirar uma "foto" (frame) do vídeo.
//   3. Mandar essa foto para o Worker, que roda a IA (YOLOv5n) em segundo plano.
//   4. Receber do Worker a lista de veículos detectados.
//
// Por que usar um Worker? A IA é pesada. Se rodasse aqui, a página inteira
// travaria (botões, vídeo etc.). O Worker é como um "funcionário" em outra
// thread: trabalha em paralelo e só conversa conosco por mensagens.
// ============================================================================

import { VideoService } from './services/VideoService.js';

// --- Elementos HTML que vamos manipular (definidos no index.html) ---
const videoInput = document.querySelector('#video-input');     // <input type="file">
const videoElement = document.querySelector('#source-video');   // <video>
const statusElement = document.querySelector('#status');        // texto de status

// Objeto que sabe carregar o vídeo e extrair frames dele.
const videoService = new VideoService(videoElement);

// Cria o Worker. `new URL(..., import.meta.url)` diz: "o arquivo está nesta
// pasta, relativo a este arquivo". `type: 'module'` permite usar import/export dentro dele.
const worker = new Worker(
    new URL('./workers/detection.worker.js', import.meta.url),
    { type: 'module' }
);

// --- Configuração de velocidade ---
// Não precisamos analisar os ~30 frames/s do vídeo; 5 por segundo já basta
// e poupa o computador.
const FRAMES_PER_SECOND = 5;
const MIN_MS_BETWEEN_FRAMES = 1000 / FRAMES_PER_SECOND; // 5 fps => 200ms

// --- Estado da aplicação (variáveis que mudam com o tempo) ---
let isModelReady = false;           // true quando o Worker terminou de carregar a IA
let isInferenceRunning = false;     // true enquanto o Worker está analisando um frame
let animationFrameRequestId = null; // "ticket" do agendamento atual (serve para cancelar)
let lastCaptureTimestamp = 0;       // quando foi a última captura (em ms)

/**
 * Mostra uma mensagem de status na tela.
 */
function updateStatus(message) {
    statusElement.textContent = message;
}

/**
 * Começa o ciclo de captura de frames.
 *
 * requestAnimationFrame pede ao navegador: "chame esta função antes de
 * desenhar o próximo quadro da tela". Retorna um número (id) para cancelarmos depois.
 */
function startFrameCapture() {
    // Se já existe um agendamento ativo, não cria outro.
    if (animationFrameRequestId !== null) return;

    animationFrameRequestId = requestAnimationFrame(processFrame);
}

/**
 * Para o ciclo de captura de frames.
 */
function stopFrameCapture() {
    if (animationFrameRequestId !== null) {
        cancelAnimationFrame(animationFrameRequestId);
        animationFrameRequestId = null;
    }
}

/**
 * Função chamada repetidamente pelo navegador (cerca de 60x por segundo).
 * Ela decide SE já é hora de capturar um novo frame e enviá-lo ao Worker.
 *
 * @param {number} timestamp - Tempo atual em ms, fornecido pelo requestAnimationFrame.
 */
async function processFrame(timestamp) {
    // Este agendamento acabou de ser executado, então não há mais "ticket" pendente.
    animationFrameRequestId = null;

    // Se o vídeo parou, encerra o ciclo.
    if (videoElement.paused || videoElement.ended) return;

    // Só envia um frame se TODAS as condições forem verdadeiras:
    //  - o modelo já carregou;
    //  - o Worker não está ocupado (evita fila de frames acumulando);
    //  - já passou tempo suficiente desde a última captura (respeita os 5 fps).
    if (
        isModelReady &&
        !isInferenceRunning &&
        timestamp - lastCaptureTimestamp >= MIN_MS_BETWEEN_FRAMES
    ) {
        lastCaptureTimestamp = timestamp;
        isInferenceRunning = true;

        try {
            // Tira a "foto" do vídeo neste instante (um ImageBitmap).
            const frame = await videoService.captureFrame();

            // Envia a foto ao Worker.
            // O segundo argumento [frame] é a lista de "transferíveis": em vez de
            // COPIAR a imagem (lento), o dado é MOVIDO para o Worker (instantâneo).
            // Depois disso, `frame` não pode mais ser usado aqui.
            worker.postMessage(
                {
                    type: 'process-frame',
                    frame
                },
                [frame]
            );
        } catch (error) {
            // Se falhou, libera o Worker para a próxima tentativa.
            isInferenceRunning = false;
            console.error('Falha ao capturar frame:', error);
        }
    }

    // Agenda a próxima verificação, enquanto o vídeo continuar tocando.
    if (!videoElement.paused && !videoElement.ended) {
        animationFrameRequestId = requestAnimationFrame(processFrame);
    }
}

// ----------------------------------------------------------------------------
// EVENTOS DA INTERFACE
// ----------------------------------------------------------------------------

// Quando o usuário escolhe um arquivo de vídeo no <input>.
// `({ target })` é desestruturação: pega só a propriedade `target` do evento.
videoInput.addEventListener('change', async ({ target }) => {
    const file = target.files[0]; // primeiro arquivo selecionado

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

// Eventos do <video>: ligamos/desligamos a captura conforme o vídeo toca ou para.
videoElement.addEventListener('play', startFrameCapture);
videoElement.addEventListener('pause', stopFrameCapture);
videoElement.addEventListener('ended', stopFrameCapture);

// ----------------------------------------------------------------------------
// MENSAGENS RECEBIDAS DO WORKER
// ----------------------------------------------------------------------------
// O Worker nos envia objetos com um campo `type` que diz o que aconteceu.
worker.onmessage = ({ data }) => {
    switch (data.type) {
        case 'worker-ready':
            // O Worker foi criado (mas o modelo ainda pode estar carregando).
            console.log('[Worker]', data.message);
            break;

        case 'model-loaded':
            // A IA está pronta para analisar frames.
            isModelReady = true;
            updateStatus('Modelo YOLOv5n pronto para inferência.');

            console.log('[Worker]', data.message);

            // Se o usuário já deu play antes do modelo ficar pronto, começa agora.
            if (!videoElement.paused) {
                startFrameCapture();
            }
            break;

        case 'inference-completed':
            // O Worker terminou de analisar um frame: já pode receber outro.
            isInferenceRunning = false;

            console.log('Detecções do YOLOv5n:', data.detections);

            // console.table mostra a lista como uma tabela no console do navegador.
            console.table(data.detections);

            break;

        case 'error':
            isInferenceRunning = false;

            console.error('[Worker Error]', data.message);
            updateStatus(data.message);
            break;
    }
};

// Erros inesperados dentro do Worker (que não foram tratados por ele).
worker.onerror = (error) => {
    isInferenceRunning = false;

    console.error('[Worker Error]', error.message);
    updateStatus('Erro durante o processamento.');
};

// Antes de fechar a aba, limpa tudo para não deixar recursos "vazando".
window.addEventListener('beforeunload', () => {
    stopFrameCapture();
    videoService.dispose();
    worker.terminate();
});
