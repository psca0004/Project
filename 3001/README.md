# Apple Detection and Counting Using Machine Learning

## 1. Project Overview

This project investigates the application of machine learning to apple detection and counting in orchard images. Developed as part of MMA3001 at Monash University, it evaluates the performance of object-detection models trained using different quantities of data.

The project uses YOLO11n to identify apples in RGB images. Three models are trained using progressively larger datasets to investigate the relationship between training dataset size and detection performance. A separate image-processing method also investigates the use of RGB and depth data to identify potential apples.

## 2. Project Objectives

The objectives of this project are to:

- Develop machine learning models for apple detection in orchard images.
- Investigate the effect of training dataset size on detection performance.
- Compare predicted apple counts across three trained models.
- Quantify counting errors using appropriate evaluation metrics.
- Investigate the application of depth information to apple localisation.
- Assess the limitations of the implemented methods and their potential for practical agricultural applications.

## 3. Computational Methods

### 3.1 YOLO11n Object Detection

Three YOLO11n models were trained using different quantities of training data.

| Model | Dataset | Training images |
|---|---|---:|
| Model 1 | AL | 11 |
| Model 2 | AL2 | 26 |
| Model 3 | AL3 | 56 |

Each model was trained for 100 epochs, using an image size of 640 pixels and a batch size of 4.

The models were evaluated using a common set of 16 Summer RGB images. Their predictions were recorded in CSV files, enabling comparison of the predicted apple counts against the corresponding actual counts.

### 3.2 RGB and Depth Image Processing

An alternative image-processing method was developed using OpenCV to investigate the identification of potential apples through colour and depth information.

The method applies a threshold to grayscale depth images to isolate closer objects. Red and orange regions are then identified in HSV colour space, followed by morphological operations to reduce noise. Contour detection is used to identify potential apple regions.

This approach provides an alternative to trained object detection; however, colour similarity, overlapping objects and inaccurate contour separation can affect counting accuracy.

## 4. Repository Structure

The repository contains Python scripts for model training, model testing, architecture inspection and RGB/depth image processing. The training scripts generate the trained models, while the testing scripts evaluate predictions and record apple counts for subsequent analysis.

The resulting CSV files support quantitative comparison between the models. Additional documentation and testing resources describe the implementation and software verification process.

## 5. Software and Dependencies

The project is implemented in Python and uses the following libraries:

- **Ultralytics:** YOLO11n model training and inference.
- **OpenCV:** Image loading, colour-space conversion, thresholding and contour detection.
- **NumPy:** Numerical operations on image arrays.
- **pandas:** CSV data handling, where used by the scripts.

## 6. Execution

The project consists of separate training and testing scripts. Model training produces the weights required for subsequent inference, while testing scripts apply the trained models to the selected evaluation images and record their predictions.

Execution requires the dataset, model weights and Python dependencies to be available at the paths specified in the relevant scripts. File paths may differ between environments.

The generated CSV files contain the predicted apple counts and provide the data used for model comparison.

## 7. Evaluation and Validation

The three models are evaluated using the same 16 images, providing a consistent basis for comparison. Detection performance metrics and apple-counting errors are used to assess the models' effectiveness.

Mean absolute error (MAE) provides a measure of the average absolute difference between predicted and actual apple counts. Additional detection metrics help distinguish object-detection performance from counting accuracy.

The evaluation results are limited by the size and composition of the evaluation dataset. Performance on these images alone does not establish generalisation to unseen trees, different seasons, lighting conditions or other orchard environments.

## 8. Limitations

The performance of the implemented methods may be affected by several factors:

- **Training data:** Limited training data may not adequately represent the variation in orchard images.
- **Occlusion:** Overlapping or partially obscured apples may be missed or incorrectly counted.
- **Image conditions:** Lighting, background objects and colour variation can affect detection.
- **Generalisation:** Performance on a small evaluation set may not represent performance in unseen environments.
- **Depth processing:** Threshold selection and depth-image quality influence which objects are retained.
- **Contour detection:** Colour-based regions do not necessarily correspond to individual apples, potentially resulting in false detections or merged objects.

These limitations are considered when interpreting the evaluation results and assessing the suitability of the methods for practical applications.

## 9. Documentation and Software Testing

The project documentation describes the source code, computational methods, inputs, outputs and relevant assumptions. Software tests assess the behaviour of individual components, while the test report records the tests performed and their outcomes.

Software testing and model validation serve different purposes: software tests assess whether the implementation behaves as intended, whereas model validation assesses predictive performance against reference data.

## 10. Use of AI

AI tools were used to assist with aspects of code development, debugging and documentation. AI-generated suggestions were reviewed against the implementation and observed outputs.

The project evaluation and interpretation of results are based on the experimental evidence obtained. The role of AI, including its benefits and limitations during development, is addressed in the accompanying report.

## 11. Licence

The repository's `LICENSE` file specifies the licence applicable to the project source code. Third-party libraries, datasets and pretrained model components may be subject to separate licence conditions.
