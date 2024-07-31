Neuron Tracking Comparison in VR and BMI Experiments

This project compares two methods of tracking neurons across different sessions of VR and BMI experiments in different mice. The two methods are CellReg and bmi_tools. The outputs of these methods are global masks mapping each cell position and contour based on a number of different experiments.
Methods
1. Overlap Calculation

Overlaps between cells from each session in their original position with the global mask cells are calculated. Overlap is defined as the Jaccard index of the respective cell from the global mask with the corresponding session cell within a certain distance. The set of all pixels for each cell are the elements of the sets.
2. Shape Similarity Quantification

Quantification measures for scoring the similarity of cell shapes are performed after aligning their centers. For this, dynamic time warping is used.
Scoring

A weighted combination of the overlap calculation and shape similarity quantification gives a score. This score indicates the likelihood of cell X from session Z being the same cell as cell Y from the global mask. The score values have no independent meaning but serve as a relative measure.
Usage

    Install the required dependencies for both CellReg and bmi_tools as mentioned in their respective repositories.
    Run the comparison script which uses the overlap and shape similarity measures to generate scores.
    Analyze the scores to determine the likelihood of cell correspondences across sessions.

References

    CellReg
    bmi_tools
    Jaccard Index
    Dynamic Time Warping
