This code compares two methods of tracking neurons across different sessions of VR and BMI experiments in different mice. The two methods are [CellReg](https://github.com/zivlab/CellReg) and [bmi_tools](https://github.com/catubc/bmi_tools). The output of the latter is a global mask mapping each cell position and contour on the basis of a number of different experiments. For the former, the output includes average centroids and cell contours for each session. On the basis of this, I created a global mask for the Cell Reg output in order to sensibly compare the outputs of the methods. 

Here, I use

1. overlaps between cells from each session in their original position with the global mask cells. 
Overlap is simply defined as the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index) of the respective cell from the global mask with the 
corresponding session cell within a certain distance. The set of all pixels for each cell are the elements of the sets.
This should incorporate mostly information about original position.
2. Quantification measures for scoring the similarity of cell shapes after aligning their centers. 
For this, I use [dynamic time warping](https://en.wikipedia.org/wiki/Dynamic_time_warping).
This should incorporate shape and geometric nuances. 
3. The overlap after aligining the centers to account for cases with few matches and to include additional shape information. 
This should account for shape and size information. 
Partly, it also acts as a control for cases where DTW distances are close and the most similar match in terms of DTW is ambiguous. 

A weighted combination of these factors gives a score, indicating the likelihood of cell X from session Z to be the same cell as cell Y from the global mask. The score values have no meaning independently but just as a relative measure.
