"use strict";
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
const classifier_1 = require("./classifier");
require("./styles.css");
// DOM Elements
const imageUploadInput = document.getElementById('imageUpload');
const classifyButton = document.getElementById('classifyBtn');
const resultContainer = document.getElementById('resultContainer');
const outputImage = document.getElementById('outputImage');
const predictionResults = document.getElementById('predictionResults');
const spinnerElement = document.getElementById('spinner');
const loadingText = document.getElementById('loadingText');
const errorContainer = document.getElementById('errorContainer');
const errorMessage = document.getElementById('errorMessage');
// Create an instance of our classifier
const classifier = new classifier_1.MobileNetClassifier();
// Helper function to show error messages
function showError(message) {
    errorMessage.textContent = message;
    errorContainer.classList.remove('d-none');
}
// Helper function to hide error messages
function hideError() {
    errorContainer.classList.add('d-none');
}
// Helper function to show loading spinner
function showSpinner(message) {
    loadingText.textContent = message;
    spinnerElement.classList.remove('d-none');
}
// Helper function to hide loading spinner
function hideSpinner() {
    spinnerElement.classList.add('d-none');
}
// Initialize the application
function init() {
    return __awaiter(this, void 0, void 0, function* () {
        try {
            // Show loading spinner while the model loads
            showSpinner('Loading MobileNetV2 model...');
            yield classifier.loadModel();
            hideSpinner();
            // Enable the classify button once the model is loaded
            if (classifier.isModelLoaded()) {
                classifyButton.disabled = false;
            }
            else {
                showError('Failed to load the MobileNetV2 model. Please refresh the page and try again.');
            }
        }
        catch (error) {
            hideSpinner();
            showError(`Error initializing the application: ${error}`);
        }
    });
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
classifyButton.addEventListener('click', () => __awaiter(void 0, void 0, void 0, function* () {
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
        yield new Promise((resolve) => {
            img.onload = resolve;
        });
        // Classify the image
        const predictions = yield classifier.classifyImage(img);
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
    }
    catch (error) {
        hideSpinner();
        showError(`Error classifying image: ${error}`);
    }
}));
// Initialize the application when the page loads
window.addEventListener('load', init);
