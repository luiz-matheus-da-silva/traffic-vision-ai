
import * as tf from '@tensorflow/tfjs';

/**
 * Inicializa a aplicação e configura a comunicação com o Worker.
 */
async function initialize() {
    // Inicializa o TensorFlow.js na Main Thread.
    await tf.ready();

    console.log('TrafficVision AI iniciado.');
    console.log('TensorFlow.js:', tf.version.tfjs);

    // Cria o Worker responsável pelo processamento dos frames.
    const worker = new Worker(
        new URL('./workers/detection.worker.js', import.meta.url),
        { type: 'module' }
    );

    // Recebe mensagens e resultados enviados pelo Worker.
    worker.onmessage = ({ data }) => {
        switch (data.type) {
            case 'worker-ready':
                console.log('[Worker]', data.message);
                break;

            case 'model-loaded':
                console.log('[Worker]', data.message);
                break;

            case 'inference-completed':
                console.log('Saídas do YOLOv5n:', data.outputs);
                break;

            case 'error':
                console.error('[Worker Error]', data.message);
                break;
        }
    };

    // Captura erros de execução do Worker.
    worker.onerror = (error) => {
        console.error('[Worker Error]', error.message);
    };

    // Disponibiliza o Worker para os componentes da aplicação.
    window.detectionWorker = worker;
}

// Trata possíveis erros durante a inicialização.
initialize().catch(console.error);
