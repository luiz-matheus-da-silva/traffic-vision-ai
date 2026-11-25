
/**
 * Gerencia o carregamento e a extração de frames de vídeos.
 */
class VideoService {
    constructor(videoElement) {
        this.video = videoElement;
        this.currentVideoUrl = null;
    }

    /**
     * Carrega um arquivo de vídeo no elemento HTML.
     *
     * @param {File} file - Arquivo selecionado pelo usuário.
     */
    async load(file) {
        if (!file || !file.type.startsWith('video/')) {
            throw new Error('Arquivo de vídeo inválido.');
        }

        // Libera a URL do vídeo anteriormente carregado.
        this.releaseVideo();

        // Cria uma URL temporária para acessar o arquivo local.
        this.currentVideoUrl = URL.createObjectURL(file);

        this.video.src = this.currentVideoUrl;

        // Aguarda os metadados para confirmar o carregamento.
        await new Promise((resolve, reject) => {
            const onLoaded = () => {
                cleanup();
                resolve();
            };

            const onError = () => {
                cleanup();
                reject(new Error('Não foi possível carregar o vídeo.'));
            };

            const cleanup = () => {
                this.video.removeEventListener('loadedmetadata', onLoaded);
                this.video.removeEventListener('error', onError);
            };

            this.video.addEventListener('loadedmetadata', onLoaded, { once: true });
            this.video.addEventListener('error', onError, { once: true });

            this.video.load();
        });

        return {
            duration: this.video.duration,
            width: this.video.videoWidth,
            height: this.video.videoHeight
        };
    }

    /**
     * Extrai o frame atual do vídeo como ImageBitmap.
     *
     * @returns {Promise<ImageBitmap>}
     */
    async captureFrame() {
        if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            throw new Error('Vídeo ainda não possui um frame disponível.');
        }

        return createImageBitmap(this.video);
    }

    /**
     * Libera a URL temporária do vídeo.
     */
    releaseVideo() {
        if (this.currentVideoUrl) {
            URL.revokeObjectURL(this.currentVideoUrl);
            this.currentVideoUrl = null;
        }
    }

    /**
     * Libera os recursos utilizados pelo serviço.
     */
    dispose() {
        this.video.pause();
        this.video.removeAttribute('src');
        this.video.load();

        this.releaseVideo();
    }
}

export { VideoService };
