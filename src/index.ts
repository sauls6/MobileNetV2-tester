import { MobileNetClassifier } from './classifier';
import './styles.css';

// DOM Elements
const imageUploadInput = document.getElementById('imageUpload') as HTMLInputElement;
const classifyButton = document.getElementById('classifyBtn') as HTMLButtonElement;
const resultContainer = document.getElementById('resultContainer') as HTMLDivElement;
const outputImage = document.getElementById('outputImage') as HTMLImageElement;
const predictionResults = document.getElementById('predictionResults') as HTMLTableSectionElement;
const spinnerElement = document.getElementById('spinner') as HTMLDivElement;
const loadingText = document.getElementById('loadingText') as HTMLParagraphElement;
const errorContainer = document.getElementById('errorContainer') as HTMLDivElement;
const errorMessage = document.getElementById('errorMessage') as HTMLDivElement;

// Create an instance of our classifier
const classifier = new MobileNetClassifier();

// Helper function to show error messages
function showError(message: string): void {
  errorMessage.textContent = message;
  errorContainer.classList.remove('d-none');
}

// Helper function to hide error messages
function hideError(): void {
  errorContainer.classList.add('d-none');
}

// Helper function to show loading spinner
function showSpinner(message: string): void {
  loadingText.textContent = message;
  spinnerElement.classList.remove('d-none');
}

// Helper function to hide loading spinner
function hideSpinner(): void {
  spinnerElement.classList.add('d-none');
}

// Initialize the application
async function init() {
  try {
    // Show loading spinner while the model loads
    showSpinner('Loading MobileNetV2 model...');
    await classifier.loadModel();
    hideSpinner();
    
    // Enable the classify button once the model is loaded
    if (classifier.isModelLoaded()) {
      classifyButton.disabled = false;
    } else {
      showError('Failed to load the MobileNetV2 model. Please refresh the page and try again.');
    }
  } catch (error) {
    hideSpinner();
    showError(`Error initializing the application: ${error}`);
  }
}

// Handle file input changes
imageUploadInput.addEventListener('change', () => {
  if (imageUploadInput.files && imageUploadInput.files[0]) {
    classifyButton.disabled = false;
    hideError();
    resultContainer.classList.add('d-none');
  }
});

// Handle classify button click
classifyButton.addEventListener('click', async () => {
  if (!imageUploadInput.files || !imageUploadInput.files[0]) {
    showError('Please select an image first');
    return;
  }
  
  try {
    hideError();
    showSpinner('Classifying image...');
    resultContainer.classList.add('d-none');
    
    // Create an image element from the file
    const file = imageUploadInput.files[0];
    const img = new Image();
    img.src = URL.createObjectURL(file);
    
    // Wait for the image to load
    await new Promise((resolve) => {
      img.onload = resolve;
    });
    
    // Classify the image
    const predictions = await classifier.classifyImage(img);
    
    // Display the results
    outputImage.src = img.src;
    
    // Clear previous results
    predictionResults.innerHTML = '';
    
    // Add new results to the table
    predictions.forEach(prediction => {
      const row = document.createElement('tr');
      
      const classCell = document.createElement('td');
      classCell.textContent = prediction.className;
      row.appendChild(classCell);
      
      const probCell = document.createElement('td');
      probCell.textContent = `${(prediction.probability * 100).toFixed(2)}%`;
      row.appendChild(probCell);
      
      predictionResults.appendChild(row);
    });
    
    // Show the results container
    hideSpinner();
    resultContainer.classList.remove('d-none');
    
  } catch (error) {
    hideSpinner();
    showError(`Error classifying image: ${error}`);
  }
});

// Initialize the application when the page loads
window.addEventListener('load', init);
