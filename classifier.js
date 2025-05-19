"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MobileNetClassifier = void 0;
const tf = __importStar(require("@tensorflow/tfjs"));
const imagenet_classes_1 = require("./imagenet-classes");
// Main class for the MobileNetV2 classifier
class MobileNetClassifier {
    constructor() {
        this.model = null;
        this.isModelLoading = false;
        this.modelLoadingPromise = null;
        // Use the imported ImageNet classes
        this.IMAGENET_CLASSES = imagenet_classes_1.IMAGENET_CLASSES;
        // Initialize the model loading process
        this.loadModel();
    }
    /**
     * Loads the MobileNetV2 model
     */ loadModel() {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.model)
                return Promise.resolve();
            if (this.isModelLoading && this.modelLoadingPromise) {
                return this.modelLoadingPromise;
            }
            this.isModelLoading = true;
            this.modelLoadingPromise = new Promise((resolve) => __awaiter(this, void 0, void 0, function* () {
                try {
                    // Load MobileNetV2 model from TensorFlow Hub
                    this.model = yield tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1', { fromTFHub: true });
                    console.log('MobileNetV2 model loaded successfully');
                }
                catch (error) {
                    console.error('Failed to load MobileNetV2 model:', error);
                }
                finally {
                    this.isModelLoading = false;
                    resolve();
                }
            }));
            return this.modelLoadingPromise;
        });
    }
    /**
     * Checks if the model is loaded
     */
    isModelLoaded() {
        return this.model !== null;
    }
    /**
     * Preprocesses an image for the model
     * @param image The image to preprocess
     * @returns A tensor representation of the image
     */
    preprocessImage(image) {
        return tf.tidy(() => {
            // Read the image data
            let tensor = tf.browser.fromPixels(image)
                // Resize to 224x224 (required input size for MobileNetV2)
                .resizeBilinear([224, 224])
                // Expand dimensions to add batch size
                .expandDims(0)
                // Normalize from [0,255] to [-1,1]
                .toFloat()
                .div(tf.scalar(127.5))
                .sub(tf.scalar(1));
            return tensor;
        });
    }
    /**
     * Classifies an image
     * @param image The image to classify
     * @param topK Number of top predictions to return
     * @returns Array of predictions
     */ classifyImage(image_1) {
        return __awaiter(this, arguments, void 0, function* (image, topK = 5) {
            if (!this.model) {
                throw new Error('Model is not loaded yet');
            }
            // Preprocess and predict inside tf.tidy, but do post-processing outside
            const { values, indices } = tf.tidy(() => {
                // Preprocess the image
                const tensor = this.preprocessImage(image);
                // Run the model prediction
                const predictions = this.model.predict(tensor);
                // Get top K predictions
                return tf.topk(predictions, topK);
            });
            // Convert to arrays
            const valuesArray = yield values.data();
            const indicesArray = yield indices.data();
            // Dispose tensors
            values.dispose();
            indices.dispose();
            // Apply softmax to get proper probability distribution
            // First, find the maximum value for numerical stability
            let maxValue = -Infinity;
            for (let i = 0; i < topK; i++) {
                if (valuesArray[i] > maxValue) {
                    maxValue = valuesArray[i];
                }
            }
            // Apply exp and sum
            let sum = 0;
            const expValues = new Array(topK);
            for (let i = 0; i < topK; i++) {
                expValues[i] = Math.exp(valuesArray[i] - maxValue);
                sum += expValues[i];
            }
            // Create and return prediction objects with normalized probabilities
            const results = [];
            for (let i = 0; i < topK; i++) {
                const index = indicesArray[i];
                const className = this.IMAGENET_CLASSES[index] || `Class ${index}`;
                // Ensure probability is correctly normalized between 0 and 1
                const probability = expValues[i] / sum;
                results.push({
                    className: className,
                    probability: probability // This is a value between 0 and 1
                });
            }
            return results;
        });
    }
}
exports.MobileNetClassifier = MobileNetClassifier;
