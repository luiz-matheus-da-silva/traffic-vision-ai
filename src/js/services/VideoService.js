// ============================================================================
// VideoService.js — Cuida de tudo relacionado ao vídeo
//   - carregar um arquivo de vídeo local dentro do <video>;
//   - extrair um frame (imagem) do vídeo no momento atual.
// ============================================================================

/**
 * Gerencia o carregamento e a extração de frames de vídeos.
 *
 * Uma `class` agrupa dados (this.video...) e funções (load, captureFrame...)
 * que trabalham juntos. Criamos uma instância com `new VideoService(elemento)`.
 */
class VideoService {
    /**
     * O constructor roda uma vez, quando fazemos `new VideoService(...)`.
     *
     * @param {HTMLVideoElement} videoElement - A tag <video> da página.
     */
    constructor(videoElement) {
        this.video = videoElement;
        // Endereço temporário do vídeo na memória do navegador (blob:...).
        this.currentVideoUrl = null;
    }

    /**
     * Carrega um arquivo de vídeo no elemento <video>.
     *
     * `async` indica que a função usa `await` e retorna uma Promise
     * (uma "promessa" de resultado futuro).
     *
     * @param {File} file - Arquivo selecionado pelo usuário.
     * @returns {Promise<{duration: number, width: number, height: number}>}
     */
    async load(file) {
        // Valida: tem que existir e o tipo MIME precisa começar com "video/".
        if (!file || !file.type.startsWith('video/')) {
            throw new Error('Arquivo de vídeo inválido.');
        }

        // Descarta a URL do vídeo anterior (se houver) para não gastar memória.
        this.releaseVideo();

        // O navegador não lê o arquivo do disco direto: criamos uma URL temporária
        // que aponta para ele (algo como "blob:http://localhost/abc-123").
        this.currentVideoUrl = URL.createObjectURL(file);

        this.video.src = this.currentVideoUrl;

        // Carregar vídeo é assíncrono: precisamos ESPERAR o navegador terminar.
        // Criamos uma Promise que só se resolve quando o evento certo acontecer.
        await new Promise((resolve, reject) => {
            // Sucesso: os metadados (tamanho, duração) foram lidos.
            const onMetadataLoaded = () => {
                removeListeners();
                resolve();
            };

            // Falha: o navegador não conseguiu abrir o vídeo.
            const onLoadError = () => {
                removeListeners();
                reject(new Error('Não foi possível carregar o vídeo.'));
            };

            // Remove os "ouvintes" de evento para não deixá-los pendurados.
            const removeListeners = () => {
                this.video.removeEventListener('loadedmetadata', onMetadataLoaded);
                this.video.removeEventListener('error', onLoadError);
            };

            // `{ once: true }` faz o ouvinte rodar no máximo uma vez.
            this.video.addEventListener('loadedmetadata', onMetadataLoaded, { once: true });
            this.video.addEventListener('error', onLoadError, { once: true });

            // Manda o navegador começar a carregar.
            this.video.load();
        });

        return {
            duration: this.video.duration,
            width: this.video.videoWidth,
            height: this.video.videoHeight
        };
    }

    /**
     * Tira uma "foto" do vídeo no instante atual.
     *
     * ImageBitmap é uma imagem já decodificada e pronta para uso rápido,
     * e pode ser enviada a um Worker sem cópia.
     *
     * @returns {Promise<ImageBitmap>}
     */
    async captureFrame() {
        // readyState diz quanto o vídeo já carregou. Precisamos de pelo menos
        // o frame atual disponível; senão a "foto" sairia vazia.
        if (this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            throw new Error('Vídeo ainda não possui um frame disponível.');
        }

        return createImageBitmap(this.video);
    }

    /**
     * Libera a URL temporária (revoke = "revogar") para liberar memória.
     */
    releaseVideo() {
        if (this.currentVideoUrl) {
            URL.revokeObjectURL(this.currentVideoUrl);
            this.currentVideoUrl = null;
        }
    }

    /**
     * Limpeza geral: para o vídeo e libera os recursos.
     */
    dispose() {
        this.video.pause();
        this.video.removeAttribute('src'); // desvincula o arquivo
        this.video.load();                 // faz o <video> "esquecer" o conteúdo

        this.releaseVideo();
    }
}

export { VideoService };
