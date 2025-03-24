# Visual Reference Resolution Using Transfer Learning

## Project Overview
This project implements a visual reference resolution system that identifies objects in images using natural language referring expressions. The system utilizes transfer learning on the MS COCO dataset, specifically working with the "RefCOCO" subset that contains images with bounding box annotations and associated referring expressions.

## Key Features
- Word-by-image classification using logistic regression models
- Transfer learning approach using pre-trained CLIP embeddings
- Position-based feature enhancement for improved spatial reasoning
- Word-level probability integration for handling complex multi-word expressions

## Dataset
The project uses the RefCOCO dataset, which contains:
- 142,210 expressions for 50,000 reference objects across 19,994 images
- Train/validation/test splits for systematic evaluation
- Multiple referring expressions per object (e.g., "white brown sheep right", "black sheep on right")

## Implementation Details

### Data Processing
- Image preprocessing using the CLIP ViT-B/32 model
- Extraction of bounding box subimages for focused analysis
- Computation of positional features (relative coordinates, area, etc.)
- Natural language processing to extract unique non-stopwords

### Machine Learning Approach
- Word-as-classifier paradigm, creating individual classifiers for each word
- Positive and negative example collection for training word-specific models
- Logistic regression models with tuned hyperparameters for classification
- Feature vector concatenation of visual and positional data

### Training Pipeline
The training pipeline consists of three main stages:
1. **Positive example collection**: For each word, collecting feature vectors from images where the word appears in referring expressions
2. **Negative example collection**: For each word, collecting feature vectors from images where the word does not appear
3. **Model training**: Fitting logistic regression classifiers for each word with balanced positive and negative examples

### Validation and Testing
- Implemented validation on a hold-out set to measure model performance
- Test set evaluation for final performance reporting
- Comparison against random selection baseline
- Systematic error analysis to identify model limitations

## Results
- Test set accuracy: ~50%
- Random baseline: ~15%
- First-position baseline: ~18%

## Model Assumptions and Limitations
- Word independence assumption (treats each word separately without considering dependencies)
- Limited context modeling between words in multi-word expressions
- Potential issues with complex spatial relationships
- Sensitivity to word order variations in referring expressions

## Potential Improvements
- Exploring alternative ConvNet architectures for feature extraction
- Investigating different classifier models (e.g., LogisticRegressionCV)
- Hyperparameter tuning for improved generalization
- Exploring alternative methods for combining word probabilities
- Incorporating word dependencies and phrase structures

## Usage
To run the model:
1. Ensure the MS COCO dataset is downloaded and properly structured
2. Run the training pipeline to collect examples and train word classifiers
3. Evaluate on validation set during development
4. Perform final evaluation on test set

## Dependencies
- Python 3.x
- PyTorch
- CLIP
- scikit-learn
- NumPy
- NLTK
- Pandas
- PIL (Pillow)
- tqdm

## Acknowledgments
This project uses the REFER API and MSCOCO dataset with referring expressions.
