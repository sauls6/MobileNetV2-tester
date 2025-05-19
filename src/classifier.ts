import * as tf from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './imagenet-classes';

// Define the interface for prediction results
export interface Prediction {
  className: string;
  probability: number;
}

// Main class for the MobileNetV2 classifier
export class MobileNetClassifier {
  private model: tf.GraphModel | null = null;
  private isModelLoading: boolean = false;
  private modelLoadingPromise: Promise<void> | null = null;
  
  // Use the imported ImageNet classes
  private IMAGENET_CLASSES = IMAGENET_CLASSES;

  constructor() {
    // Initialize the model loading process
    this.loadModel();
  }

  /**
   * Loads the MobileNetV2 model
   */  public async loadModel(): Promise<void> {
    if (this.model) return Promise.resolve();
    
    if (this.isModelLoading && this.modelLoadingPromise) {
      return this.modelLoadingPromise;
    }
    
    this.isModelLoading = true;
    this.modelLoadingPromise = new Promise<void>(async (resolve) => {
      try {
        // Load MobileNetV2 model from TensorFlow Hub
        this.model = await tf.loadGraphModel(
          'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1', 
          { fromTFHub: true }
        );
        console.log('MobileNetV2 model loaded successfully');
      } catch (error) {
        console.error('Failed to load MobileNetV2 model:', error);
      } finally {
        this.isModelLoading = false;
        resolve();
      }
    });
    
    return this.modelLoadingPromise;
  }

  /**
   * Checks if the model is loaded
   */
  public isModelLoaded(): boolean {
    return this.model !== null;
  }

  /**
   * Preprocesses an image for the model
   * @param image The image to preprocess
   * @returns A tensor representation of the image
   */
  private preprocessImage(image: HTMLImageElement): tf.Tensor {
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
   */  public async classifyImage(image: HTMLImageElement, topK: number = 5): Promise<Prediction[]> {
    if (!this.model) {
      throw new Error('Model is not loaded yet');
    }

    // Preprocess and predict inside tf.tidy, but do post-processing outside
    const { values, indices } = tf.tidy(() => {
      // Preprocess the image
      const tensor = this.preprocessImage(image);

      // Run the model prediction
      const predictions = this.model!.predict(tensor) as tf.Tensor;

      // Get top K predictions
      return tf.topk(predictions, topK);
    });
    
    // Convert to arrays
    const valuesArray = await values.data();
    const indicesArray = await indices.data();

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
    const results: Prediction[] = [];

    for (let i = 0; i < topK; i++) {
      const index = indicesArray[i];
      const className = this.IMAGENET_CLASSES[index as keyof typeof this.IMAGENET_CLASSES] || `Class ${index}`;
      // Ensure probability is correctly normalized between 0 and 1
      const probability = expValues[i] / sum;
      results.push({
        className: className,
        probability: probability // This is a value between 0 and 1
      });
    }

    return results;
  }
}
