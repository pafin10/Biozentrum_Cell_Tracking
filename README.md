This code compares two methods of tracking neurons across different sessions of VR and BMI experiments in different mice. The two methods are [CellReg](https://github.com/zivlab/CellReg) and [bmi_tools](https://github.com/catubc/bmi_tools). The outputs of these are global masks mapping each cell position and contour on the basis of a number of different experiments.

Here, I use

1. overlaps between cells from each session in their original position with the global mask cells. 
Overlap is simply defined as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) of the respective cell from the global mask with the 
corresponding session cell within a certain distance. The set of all pixels for each cell are the elements of the sets.
2. Quantification measures for scoring the similarity of cell shapes after aligning their centers. 
For this, I use [dynamic time warping](https://en.wikipedia.org/wiki/Dynamic_time_warping).

A weighted combination of these factors gives a score, indicating the likelihood of cell X from session Z to be the same cell as cell Y from the global mask. The score values have no meaning independently but just as a relative measure.
