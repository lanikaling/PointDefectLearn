# Point Defect Classification in Metal Structures Using Neural Networks

This project tackles the classification of point defect subtypes in metal structures based on pair distribution function (PDF) data using a neural network. 
More details are provided in Chapter 6 of Ling Lan’s PhD thesis (Columbia University).

## Background

Point defects are zero-dimensional imperfections in a crystal lattice. The four main types of point defects considered are:

- **Vacancies**: Missing atoms from lattice sites.
- **Self-Interstitials**: Extra atoms of the same type in interstitial sites.
- **Substitutional Impurities**: Lattice atoms replaced by different atoms.
- **Interstitial Impurities**: Different atoms occupying interstitial spaces.

The PDF data of metal crystals with these four point defects were used as input data to train a convolutional neural network (CNN) model. Results indicate strong classification capability for these defect types.

## Dataset

### Pure Metal Structures

The dataset consists of pure metal structures extracted from the ICSD database. We filtered structures to retain only those in the six most frequent space groups, resulting in a subset of 1,172 out of 1,438 structures, covering 81.5%. For each structure, a 5x5x5 supercell was generated using symmetry operations and periodic boundary conditions.

| Space Group | Label     | Lattice Parameters                        | Crystal System | Count |
|-------------|-----------|-------------------------------------------|----------------|-------|
| 225         | Fm-3m     | $a = b = c$, $\alpha = \beta = \gamma = 90^\circ$ | Cubic          | 391   |
| 194         | P6₃/mmc   | $a = b \neq c$, $\alpha = \beta = 90^\circ$, $\gamma = 120^\circ$ | Hexagonal     | 341   |
| 229         | Im-3m     | $a = b = c$, $\alpha = \beta = \gamma = 90^\circ$ | Cubic          | 273   |
| 139         | I4/mmm    | $a = b \neq c$, $\alpha = \beta = \gamma = 90^\circ$ | Tetragonal    | 83    |
| 141         | I4₁/amd   | $a = b \neq c$, $\alpha = \beta = \gamma = 90^\circ$ | Tetragonal    | 52    |
| 140         | I4/mcm    | $a = b \neq c$, $\alpha = \beta = \gamma = 90^\circ$ | Tetragonal    | 32    |

### Point Defect Simulations

Point defects were simulated in the supercells, with three variations per defect type:
- **Vacancy**: Randomly chosen vacancy percentages.
- **Self-Interstitial**: Random self-interstitial percentages with defined interstitial sites.
- **Substitutional Impurities**: Random substitutional impurity percentages with metal atoms.
- **Interstitial Impurities**: Similar to self-interstitial using defined interstitial sites.

## Data Processing

PDFs were truncated between $r = 1.5$ Å and $r = 30$ Å, interpolated on a grid of 300 points, and normalized using:

$x = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)}$

Each normalized PDF was formatted for model input by concatenating original and distorted PDFs into a shape of (2, 300), allowing simultaneous access to both unaltered and defect-affected data.

### Experimental Parameters for PDF Calculation

| Parameter       | Value |
|-----------------|-------|
| $r_{\text{max}}$ ($Å$)  | 30.0  |
| $r_{\text{step}}$ ($Å$) | 0.01  |
| $q_{\text{min}}$ ($Å^{-1}$) | 0.6   |
| $q_{\text{max}}$ ($Å^{-1}$) | 23.6  |
| $q_{\text{damp}}$ ($Å^{-1}$) | 0.029 |
| $q_{\text{broad}}$ ($Å^{-1}$) | 0.010 |

## Model Architecture

The CNN model begins with a 2D convolutional layer on the concatenated PDFs, followed by batch normalization and dropout (0.5 rate). Two residual blocks increase feature depth from 64 to 256 channels. A transition layer reduces feature depth, followed by flattening and two fully connected layers, ending with a four-class output for defect classification.

## Optimization

The model uses CrossEntropy loss with class weights (inverse of class frequency) to manage class imbalance. The Adam optimizer with a learning rate of 0.0001 is used, along with a learning rate scheduler (`ReduceLROnPlateau`) that reduces the rate by 0.85 if no validation improvement is observed within five epochs, with a threshold of 0.5% and minimum learning rate of $1 \times 10^{-7}$.

## Results

- **Classification Accuracy**: Achieved strong top-two accuracy scores approaching 100%, with an overall test accuracy of 80.10%.

For more detailed analysis and results, please refer to Chapter 6 of Ling Lan’s thesis.
