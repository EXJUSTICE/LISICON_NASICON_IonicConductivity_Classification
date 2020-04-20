# Machine-learning assisted cross-domain prediction of ionic conductivity in sodium and lithium-based superionic conductors 
Repository for the publication submitted to J Phys Commun by Xu et. al

Abstract
In this work, we present a novel machine-learning based approach to predict the ionic conductivity of sodium and lithium-based SICON compounds. Using primarily theoretical elemental feature desciptors derivable from tabulated information on the unit cell and the atomic properties of the components of a target compound on a limited dataset of 70 NASICON-examples, we have designed a logistic regression-based model capable of distinguishing between poor and good superionic conductors with a cross-validation accuracy of over 82%. Moreover, we demonstrate how such a system is capable of cross-domain classification on lithium-based examples at the same accuracy, despite being introduced to zero lithium-based compounds during training. Through a systematic permutation-based evaluation process, we reduced the number of considered features from 47 to 7,  reduction of over 83%, while simultaneously improving model performance. The contributions of different electronic and structural features to overall ionic conductivity is also discussed, and contrasted with accepted theories in literature. Our results demonstrate the utility of such a facile tool in providing opportunities for initial screening of potential candidates as solid-state electrolytes through the use of existing data examples and simple tabulated or calculated features, reducing the time-to-market of such materials by helping to focus efforts on promising candidates. Given enough data utilizing suitable descriptors, high accurate cross-domain classifiers could be created for experimentalists, improving laboratory and computational efficiency.

The legend of the features is as follows:


* d2occu - absolute occupancy of 2nd dopant position
* S - ionic conductivity (S /cm2)
* Slog - logarithmic value of ionic conductivity
* S -class 
* Ea - activation energy (eV)
* a - lattice parameter (angstroms)
* c - lattice parameter (angstroms)
* a/c - ratio of parameters
* Vcalc - calculated volume
* D1ionicr - absolute radius of dopant in position 1 (leftmost)
* D2ionicr - absolute radius of dopant in position 2
* D3ionicr - absolute radius of dopant in position 3 (rightmost)
* D1eff - effective radius of all dopants in position 1 in formula
* D2eff - effective radius of all dopants in position 2 in formula
* D3eff - effective radius of all dopants in position 3 in formula
* D1volperatom - absolute volume of dopant in position 1
* D2volperatom - absolute volume of dopant in position 2
* D3volperatom - absolute volume of dopant in position 3
* D1vol - effective volume of all dopants in position 1 in formula
* D2vol - effective volume of all dopants in position 2 in formula
* D3vol - effective volume of all dopants in position 3 in formula
* SiestimVol - estimated volume of all silicate groups in formula
* PestimVol- estimated volume of all phosphate groups in formula
* TotalVol - total volume of all dopants and groups in formula
* RelD1vol- proportion of total volume occupied by D1
* RelD2vol - proportion of total volume occupied by D2
* RelD3vol - proportion of total volume occupied by D3
* RelSivol - proportion of total volume occupied by silicate
* RelPvol - proportion of total volume occupied by phosphate
* D1eneg - absolute electronegativity of dopant in position 1 
* D2neg - absolute electronegativity of dopant in position 2
* D3neg - absolute electronegativity of dopant in position 3 
* D1eneff - effective electronegative of all dopants in position 1 in formula
* D2eneff - effective electronegative of all dopants in position 2 in formula
* D3eneff- effective electronegative of all dopants in position 3 in formula
* Zr, Na, Cr..... - number of ions of species in formula
