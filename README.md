# MobileNetV2 Image Classifier

A web application that uses TensorFlow.js and the pre-trained MobileNetV2 model to classify images into 1000 categories from the ImageNet dataset.

## Overview

This application demonstrates the use of a pre-trained neural network model (MobileNetV2) for image classification directly in the browser. MobileNetV2 is an efficient CNN architecture developed by Google that has been pre-trained on the ImageNet dataset, containing over 14 million images across 1000 categories.

## Features

- Loads the pre-trained MobileNetV2 model through TensorFlow.js
- Allows users to upload images for classification
- Displays top 5 predictions with their corresponding probabilities
- Responsive UI that works on both desktop and mobile devices

## Technical Implementation

- **Frontend**: HTML, CSS, TypeScript
- **Model**: MobileNetV2 pre-trained on ImageNet
- **Libraries**: TensorFlow.js
- **Bundling**: Webpack

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm (v6+)

### Installation

1. Clone this repository
```bash
git clone https://github.com/your-username/mobilenetv2-tester.git
cd mobilenetv2-tester
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npm run serve
```

4. Open your browser and navigate to `http://localhost:8080`

### Build for production

```bash
npm run build
```

The built files will be available in the `dist` directory.

## Model Performance

MobileNetV2 achieves:
- 71.8% Top-1 accuracy on ImageNet
- 90.6% Top-5 accuracy on ImageNet

## Limitations

- The model is limited to the 1000 classes from the ImageNet dataset
- Performance may drop when dealing with images very different from the training set
- Requires an internet connection to download the model on first use

## Author

- Saúl Sánchez Rangel

## License

This project is licensed under the MIT License - see the LICENSE file for details.
