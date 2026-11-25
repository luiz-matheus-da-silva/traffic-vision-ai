
/**
 * Configuration of the YOLOv5n model.
 *
 * These dimensions must match the model's input signature
 * ([1, 640, 640, 3] in model.json).
 */
const MODEL_CONFIG = Object.freeze({
    inputWidth: 640,
    inputHeight: 640,
    channels: 3,
    // Piso aplicado no Worker; o slider da interface filtra acima dele.
    confidenceThreshold: 0.1
});

export { MODEL_CONFIG };
