# Cortical Morphological Congruence as a Biomarker of Brain Development Assessed with Magnetic Resonance Imaging

Authors: Jacob Levman $^{\text{{1,2,3}}}$, Derek Berger$^{\text{2}}$, (others maybe Bruce Fischl?$^{\text{4}}$)

1. Martinos Center for Biomedical Imaging, Massachusetts General Hospital, Harvard Medical School, Charlestown, MA, USA.
1. Department of Computer Science, St. Francis Xavier University, Antigonish, NS, Canada.
1. Nova Scotia Health Authority, Halifax, NS, Canada.
1. Harvard Medical School, Department of Radiology, Boston, MA, USA.


# Abstract


A set of novel biomarkers are proposed for the extraction of regional
cortical morphological congruence (CMC) measurements from neurological
magnetic resonance imaging examinations. These proposed novel biomarkers
assess a cortical region’s deviation from expectation for a perfectly regular
structure. The proposed CMC biomarkers have been applied to a large sample
(n=1113) of neurological MRI examinations of neurotypical patients from the
Human Connectome Project. Results demonstrate consistent patterns of regional
brain differences in CMC across the cortex, implying differential development
regionally in the healthy brain characterizable by CMC. Results demonstrate
relatively small standard deviations of CMC values across a large population,
implying potential for these biomarkers as a reliable and easily reproducible
method to characterize brain development. <perhaps a sentence here on
similarities and differences between males and females as part of this
baseline analysis. Perhaps Sentence on some other stuff, like our ability to
predict IQ or something from CMC measurement>. Future work will investigate
CMC’s potential to further characterize healthy brain development, as well as
to characterize a variety of different pathological conditions.


**Keywords**: morphological, congruence, cortex, neurodevelopment, magnetic resonance imaging, healthy


# Introduction

Characterization of human cortical development in vivo requires medical
imaging technology that provides tissue contrast between gray and white
matter. Magnetic resonance imaging (MRI) is sensitive to hydrogen proton
concentration, which is variable across tissues, thus MRI provides excellent
soft tissue contrast, including between the gray and white matter in the
brain (Dubois et al., 2021). Automated methods for extracting biomarkers of
potential interest, such as a regional cortical tissue’s volume (mm3),
surface area (mm2) or thickness (mm), have long been relied upon for the
study of the human brain (Fischl, 2012; Levman et al., 2017, 2019; McCann et
al., 2021). However, the variability of those volume, surface area, and
thickness measurements across a given population is known to be quite large
(Levman et al., 2017, 2019), potentially contributing to known
reproducibility challenges in modern neuroscience studies (Martinez et al.,
2015; Marek et al., 2022), and may be part of the reason that these
techniques are generally not yet relied upon for clinical characterization.

Very broadly, congruence is analogous to agreement between two or more
objects, studies, shapes, individual measurements, etc. For instance, the
results of one study may be congruent with those already in the literature,
or two objects are deemed congruent if they have the same shape and size.
Congruence can be applied in many ways, and has been the subject of limited
and diverse studies focused on the human brain. Research has suggested that
the development of visual cortical properties is dependent on
visuo-proprioceptive congruence (Buisseret, 1993). More recently, a model has
been specifically developed for congruence of binocular vision (how
information from the left and right eye are incorporated) in the primary
visual cortex (Somaratna and Freeman, 2022). Congruence has also been
assessed between interoceptive predictions and hippocampal-related memory
(Edwards-Duric et al., 2020). Congruence between the development of the
circulatory and nervous systems, or neurovascular congruence, has been the
subject of a study focused on cortical development (Stubbs et al., 2009).
Additionally, it has been reported that congruence based contextual
plausibility modulates cortical activity during vibrotactile perception (Kang
et al. 2022). Neuronal congruency has also been assessed in the macaque
prefrontal cortex (Yao and Vanduffel, 2022). This manuscript presents a novel
set of biomarkers for characterization of regional cortical morphological
congruence (CMC), which can be referred to more simply as cortical congruence
(CC). The proposed methods assess the degree of congruence between multiple
cortical measurements, thus providing novel biomarkers which we hypothesize
may help characterize neurodevelopment.

# Methods

## Patient Populations and Imaging

Data were provided [in part] by the Human Connectome Project, WU-Minn
Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil;
1U54MH091657) funded by the 16 NIH Institutes and Centers that support the
NIH Blueprint for Neuroscience Research; and by the McDonnell Center for
Systems Neuroscience at Washington University. This cohort included 1,113
healthy patients imaged with MRI. Detailed information on the magnetic
resonance imaging (MRI) scanners and protocols used in the Human Connectome
Project dataset are available in the literature (Elam et al., 2021).

## Postprocessing

The Human Connectome Project’s WU-Minn HCP cohort (n=1,113 with MRI
examinations) was processed by FreeSurfer (Fischl, 2012) and the results were
made publicly available through the Human Connectome Project’s website
(https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release).

### Congruence Features

For each cortical region of interest (ROI) $r$ (e.g. $r$ = "precuneus")
supported by the publicly available FreeSurfer results (see Appendix
@Tbl:lateral_fs for the base FreeSurfer features), Cortical Morphological
Congruence (CMC) measurements were computed in each patient.

Define $V(r)$ to be the volume of region (ROI) $r$, and $A(r)$ and $d(r)$ to
be the surface area and average thickness, respectively, of the ROI. Denote
the left and right ROIs for $r$ as $r_l$ and $r_r$ respectively (e.g. if $r$
= "precuneus", then $r_l$ and $r_r$ are the left and right precuneus). Then
define the *lateral* CMC metrics to be:

$$
\text{CMC}(r) = \text{CMC}_{\text{Lateral}}(r) = \frac{ V(r) }{ A(r) \cdot d(r) }
$$

Note that $A(r) \cdot d(r)$ is the product of an area and thickness: we thus
denote the *pseudo-volume*:

$$ \hat{V}(r) = \text{CMC}_{pV} = V(r) \cdot d(r) $$

Then:

$$
\begin{align}
\text{CMC}_l &= \text{CMC}_{\text{Lateral}}(r_l) \\
\text{CMC}_r &= \text{CMC}_{\text{Lateral}}(r_r) \\
\end{align}
$$ {#eq:cmc_laterals}

Also define "bilateral" CMC as:

$$
\text{CMC}_{lr}
= \frac{ V(r_l) + V(r_r)}{ \hat{V}(r_l) + \hat{V}(r_r)}
= \frac{\text{ROI total volume}}{\text{ROI total pseudo-volume}}
$$ {#eq:cmc_bilateral}


Define the *asymmetric* CMC measures:

$$ \text{CMC}_{l - r} = \text{CMC}_{l} - \text{CMC}_{r} $$ {#eq:asym_signed_diff}

and

$$ \text{CMC}_{|l - r|} = \lvert\text{CMC}_{l - r} \rvert $$ {#eq:asym_unsigned_diff}

which are the *signed* and *unsigned* CMC differences, respectively. Also define

$$ \text{CMC}_{l / r} = \text{CMC}_{l} / \text{CMC}_{r} $$ {#eq:asym_ratio}

to be the asymmetric *ratio* CMC measure.

Then, for each ROI $r$, this defines seven $\text{CMC}$ measures:
$\text{CMC}_{pV}$, $\text{CMC}_l$, $\text{CMC}_r$, $\text{CMC}_{lr}$,
$\text{CMC}_{l-r}$, $\text{CMC}_{|l-r|}$, and $\text{CMC}_{l/r}$.  The CMC
features defined in @eq:cmc_laterals and @eq:cmc_bilateral are unitless
ratios assessing a cortical region’s deviation from uniform thickness, where
the region pseudo-volume would be expected to be equal to the surface area
times the average cortical thickness. In the case of equality, this produces
a lateral CMC biomarker value of 1. Deviations from 1 in either direction
have major implications for structural cortical presentation, imply
differential neurological development has occurred, and is addressed in
detail in the Discussion.

@Eq:asym_signed_diff defines a biomarker that preserves directionality of
asymmetry, with left/right dominant regions taking on positive/negative
values of $\text{CMC}_{l - r}$, respectively. Likewise, @eq:asym_ratio defines a
quantity which is greater/less than $1$ when left/right lateral $\text{CMC}$ is
larger.



## Statistical Analyses and Machine Learning

For each of the 34 cortical regions $r$ available from the FreeSurfer analysis (see
@tbl:lateral_cmc below), each of the seven CMC features was computed for each
subject.

### Descriptive Statistics

#### CMC Features by Sex

For each CMC feature, we compute a measure of group separation by sex using
Cohen's d and the Mann-Whitney U test. While all pseudo-volume features are
significant (@fig:cmc_sex_diffs), otherwise, only some lateral and bilateral
CMC metrics differ significantly based on sex (@tbl:sig_cmc_sex).

| ROI                         |      d |          U |   U (p) |
|:----------------------------|-------:|-----------:|--------:|
| bh-temporalpole             | -0.294 | 127593.000 |   0.000 |
| lh-temporalpole             | -0.283 | 129332.000 |   0.000 |
| bh-insula                   |  0.233 | 174258.000 |   0.004 |
| rh-temporalpole             | -0.229 | 134217.000 |   0.010 |
| bh-paracentral              | -0.187 | 136173.000 |   0.035 |
| rh-paracentral              | -0.177 | 136298.000 |   0.039 |

: CMC features with significant separation by sex.
d = Cohen's d, U = Mann-Whitney U, U (p) = p-value for Mann-Whitney U.
bh = both hemisphere, lh/rh = left/right hemisphere.
Note: p-values are adjusted for multiple comparisons using the
Holm-Bonferroni stepdown method
{#tbl:sig_cmc_sex}


![CMC Feature differences by sex](../../figures/HCP_d_p_plots.png){#fig:cmc_sex_diffs}


#### Lateral CMC Features

For each ROI, it is reasonable to ask whether $\text{CMC}_l$ and $\text{CMC}_r$ differ.
We find this to be the case for all but the superior frontal and parsopercularis
regions (@Tbl:lateral_cmc).


| ROI                      |      d |           U |   U (p) |          W |   W (p) |
|:-------------------------|-------:|------------:|--------:|-----------:|--------:|
| inferiortemporal         | -1.890 |      17.000 |   0.000 |      0.000 |   0.000 |
| insula                   | -1.778 |    5616.000 |   0.000 |     21.000 |   0.000 |
| pericalcarine            | -1.883 |     249.000 |   0.000 |      0.000 |   0.000 |
| precuneus                | -1.772 |    4138.000 |   0.000 |      0.000 |   0.000 |
| precentral               | -1.661 |   19754.000 |   0.000 |     75.000 |   0.000 |
| rostralanteriorcingulate |  1.673 | 1221463.000 |   0.000 |    143.000 |   0.000 |
| fusiform                 |  1.706 | 1225671.000 |   0.000 |    260.000 |   0.000 |
| superiorparietal         | -1.598 |   28954.000 |   0.000 |    376.000 |   0.000 |
| postcentral              |  1.625 | 1210184.000 |   0.000 |    534.000 |   0.000 |
| caudalanteriorcingulate  |  1.630 | 1209423.000 |   0.000 |   1092.000 |   0.000 |
| isthmuscingulate         | -1.498 |   47158.000 |   0.000 |   1886.000 |   0.000 |
| cuneus                   | -1.648 |   41310.000 |   0.000 |   2429.000 |   0.000 |
| transversetemporal       | -1.510 |   64112.000 |   0.000 |   2995.000 |   0.000 |
| lingual                  |  1.531 | 1170556.000 |   0.000 |   5488.000 |   0.000 |
| paracentral              | -1.406 |   92294.500 |   0.000 |    593.000 |   0.000 |
| lateralorbitofrontal     | -1.337 |   92478.000 |   0.000 |   1710.000 |   0.000 |
| posteriorcingulate       |  1.375 | 1134397.500 |   0.000 |   8640.000 |   0.000 |
| superiortemporal         |  1.176 | 1054705.000 |   0.000 |  18322.000 |   0.000 |
| supramarginal            |  1.019 | 1003090.000 |   0.000 |  59895.000 |   0.000 |
| middletemporal           |  0.940 |  960696.000 |   0.000 |  44794.000 |   0.000 |
| parahippocampal          | -0.934 |  281663.000 |   0.000 |  39694.000 |   0.000 |
| parsorbitalis            | -0.939 |  288320.500 |   0.000 |  76597.000 |   0.000 |
| frontalpole              |  0.880 |  939993.500 |   0.000 |  70533.000 |   0.000 |
| lateraloccipital         |  0.894 |  935000.000 |   0.000 |  79894.000 |   0.000 |
| parstriangularis         |  0.611 |  891834.000 |   0.000 |  90160.000 |   0.000 |
| caudalmiddlefrontal      |  0.589 |  833330.000 |   0.000 | 127030.000 |   0.000 |
| inferiorparietal         |  0.482 |  791186.000 |   0.000 | 171122.000 |   0.000 |
| bankssts                 | -0.417 |  475058.000 |   0.000 | 187242.000 |   0.000 |
| medialorbitofrontal      | -0.357 |  487862.000 |   0.000 | 180895.000 |   0.000 |
| entorhinal               | -0.364 |  499072.500 |   0.000 | 192057.000 |   0.000 |
| rostralmiddlefrontal     |  0.275 |  706419.000 |   0.000 | 224623.000 |   0.000 |
| temporalpole             | -0.216 |  543206.000 |   0.000 | 235096.000 |   0.000 |
| parsopercularis          | -0.055 |  606403.000 |   0.784 | 298155.000 |   0.541 |
| superiorfrontal          |  0.030 |  618573.000 |   0.957 | 306615.000 |   0.754 |

: Measures of separation of lateral CMC features (left vs. right
hemisphere). d = Cohen's d, U = Mann-Whitney U, U (p) = p-value for
Mann-Whitney U, W = Wilcoxon signed rank test, W (p) = p-value for W.
Note: p-values are adjusted for multiple comparisons using the
Holm-Bonferroni stepdown method
{#tbl:lateral_cmc}

### Predictive Analyses


The HCP data provides a wealth of [openly-accessible behavioural
data](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)
on participating subjects. Full details of the hundreds of features are [available
online](https://wiki.humanconnectome.org/docs/HCP-YA%20Data%20Dictionary-%20Updated%20for%20the%201200%20Subject%20Release.html).

#### Behavioural Data Cleaning {#sec:cleaning}

To investigate the predictive potential of CMC features, a number of *ad hoc*
regression targets were constructed from the HCP behavioural data. First,
behavioural data were cleaned of irrelevant, missing, constant, or redundant
features (see @sec:cleaning). (`src/munging/hcp.py::load_hcp_phenotypic`):


```{#lst:cleaning .txt}
Behavioural Data initial shape
						⬇︎
				  (1206, 582)
						⬇︎
Remove obvious junk, book-keeping (IDs, scan counts) and
subscale items, or features with 200 or more NaN values
						⬇︎
				  (1206, 400)
						⬇︎
Remove highly-correlated features (e.g. various dexterity_unadj
and dexterity_ageadj pairs), with correlations above 0.95, and
other features with correlations of 1.0.
						⬇︎
				  (1206, 375)
						⬇︎
Remove constant and string columns
						⬇︎
				  (1206, 369)
						⬇︎
Remove subjects with all missing values for all remaining features
						⬇︎
				  (1113, 369)
						⬇︎
Remove sex and age class as potential target features
						⬇︎
				  (1113, 367)
```

Then, the remaining behavioural features were reduced in an exploratory
fashion.

First, clustering was performed on the features using HDBSCAN [cite
paper, cite sklearn algorithm] with the absolute value of the Pearson
correlation as the feature distance metric. HDBSCAN does not require
specifying the number of clusters, and allows for assignment to "noise" or
"background" clusters, and thus acts as a natural method to notice patterns
of correlation in the data.

Second, feature clusters were reduced to a single dimension via factor analysis.


The resulting feature clusters are shown in [Appendix B](#appendix-b).




A <get Derek to provide a little description of the psychological explained
variance test with citation(s)> was performed. Anything interesting from the
machine learning analysis? Can you run this again with current df-analyze? If
so is there anything interesting in there worth including in this manuscript?
If so describe the methods here and adapt the text immediately below. If not,
cut the text below. Machine learning was completed with df-analyze, publicly
available machine learning software (github.com/stfxecutables/df-analyze),
which was developed in house, and has previously been applied to a brain MRI
predictive application focused on schizophrenia diagnostics (Levman et al.,
2022). In this application, df-analyze is tasked with predicting a set of
phenotypic features from the patients in our dataset from the proposed CMC
biomarkers. The targeted phenotypic features include <insert list and basic
descriptions and justification for their selection – input from Derek). The
software selects the best combination of CMC biomarkers, the best combination
of traditional FreeSurfer biomarkers, and the best combination of both, while
being restricted by the user to limit the analysis to a fixed number of
biomarkers <double check this with Derek>. The number of fixed biomarkers to
provide to machine learning is varied, and the resultant predictive capacity
of the models, alongside which biomarkers were selected for, is reported.
This was just a blurb of what we could/maybe do, any df-analyze methods
providing something interesting to say is fine, please adapt the text. Making
df-analyze publicly available (github.com/stfxecutables/df-analyze)
facilitates reproducibility of our study findings, alongside our use of large
public datasets of MRI examinations. In addition, we have provided our
analytic software that helps demonstrate to the user how to use df-analyze
and how to perform the statistical analyses employed in this study (cite
another git hub thing – get link from Derek).


# Results

Results from the Human Connectome Project, involving computing CMCLateral
(equation 1) for each cortical region in each patient, and presenting the
results as box plots (or violin plots might be slightly better) dividing
between the left and right hemispheres, as well as between males and females,
is provided in Figure 1. Note the relatively small standard deviations
relative to the varying average CMC values across cortical regions.

***Please create this big ass montage for me****

Figure 1. Boxplots of each brain region (with left and right hemispheres
paired together) across all patients (n=1113) in the neurotypical Human
Connectome Project cohort, divided by gender.

<update this section with that explained variance test that Derek did instead>

<update this section with the results of any df-analyze experiments Derek completed>

A machine learning predictive analysis was performed and the leading models,
their predictive accuracy, and the biomarkers on which they based their
predictions are provided in Table 2.


# Discussion

The human brain’s regional cortical development proceeds with a variety of
underlying factors maturing in tandem with one another. Cortical volume,
surface area and thickness are excellent examples of measurable biomarkers
that clearly exhibit interdependencies with one another. However, it should
be noted that the relative maturation of each of these biomarkers may proceed
at varying rates in any particular combination of brain region, patient or
pathology. Gray matter (GM) volume is known to increase with age, as does the
surface area, while cortical thickness thins with long-term development.
These three biomarkers are inter-related, and existing studies focused on
these measurements typically do not consider the inherent interdependence
between their respective development, even though underlying
interdependencies are inevitable. This paper presents a novel set of cortical
morphological congruence (CMC) biomarkers that are based on cortical volume,
surface area and thickness, and produces measurements with relatively small
standard deviations (see Figure 1), implying potential reliability and
reproducibility from the proposed methods.

## Interpretation of CMC and its potential relationship with macro-structural cortical development

The proposed CMC biomarkers, defined in equations 1 through 4, rely on
underlying measurements of gray matter volume (measured in mm3), surface area
(measured in mm2) and average cortical thickness (measured in mm). The nature
of the proposed equations are such that they each produce a unitless index of
CMC, different values of which imply potentially major differences in the
conformation of the local cortical region, potentially implying major
differences in the tissue’s historical neurodevelopment. When a cortical
region exhibits a CMC (see equations 1 & 2) measurement equal to 1, which is
expected for very regularly shaped structures, that region exhibits a
relatively simple cortical morphological presentation, and can be said to
have high underlying cortical morphological congruence. Examples of this can
be found consistently across patients in brain structures such as the banks
of the superior temporal sulcus (n=1,113, mean CMCLateral=1.0*, std dev
CMCLateral=0.025*), and the insula (mean CMCLateral=1.0?, std dev
CMCLateral=???). When CMC values deviate from 1, this implies varying degrees
of incongruence between the region’s volumetric biomarkers and its surface
area and cortical thickness biomarkers combined. The directionality of that
incongruence (i.e. whether CMC is above or below 1) has major implications
for the presentation of the conformation of that tissue, and implies
differential cortical development has occurred.

When a cortical region exhibits CMC above 1, the GM volume has developed to
be larger than the surface area times the mean thickness. This can occur when
the overall growth of regional cortical tissue proceeds more quickly than
increases in the surface area. Broadly speaking, the morphological structure
that maximizes volume relative to surface area is the sphere. Thus, it is
expected that convex (and thus partly spherical) presentation on the surface
of the cortical region (well-rounded boundaries between the cortex and the
pia mater which surrounds the cortex) will contribute to CMC measurements
above 1. Regions such as the entorhinal cortex, which plays a role in working
memory and thus is a highly relied upon region of cortical tissue, exhibits
high CMC values (mean = 1.3**, standard deviation = 0.?). This implies that
the entorhinal region’s development may have involved rapid increases in
volume relative to its respective increases in surface area. These high CMC
values may also implicate a distribution of pruning locations that supports
sulcal formation, leading to more convex (partly spherical) local surface
areas within the entorhinal cortex adjacent to locations of sulcal formation.
Thus, we hypothesize that pruning in the entorhinal cortex has been more
extensive (and possibly proceeded faster) than pruning in regions exhibiting
cortical morphological congruence (CMC = 1) such as the banks of the superior
temporal sulcus, or the insula. Results demonstrate that in addition to the
entorhinal cortex, multiple regions exhibit consistently high CMC values,
including the temporal pole, frontal pole, and pars orbitalis (see Figure 1).

When a cortical region exhibits CMC values below 1, the combination of the
surface area times the mean thickness has developed to be larger than the
gray matter volume. This can occur when the surface area, which is expected
to be affected by several underlying factors, including regional brain
growth, cortical folding and pruning, develops more rapidly than the growth
in overall regional gray matter volume alone. Additionally, the distribution
of locations of pruning within the cortex can result in the emergence of
comparatively complex surfaces relative to the more spherical/convex surfaces
already discussed, potentially resulting in comparatively large surface areas
yielding reduced values for our CMC biomarkers. Regions such as the
pericalcarine cortex exhibit low CMC values (mean = 0.95*, standard deviation
= ??), which could imply that surface area growth has outpaced corresponding
volumetric growth in this region’s development relative to other cortical
regions.

## Potential for CMC to characterize important aspects of brain development

The combination of regional cortical volume, surface area and mean thickness,
biomarkers with relatively high variability across patients, into a single
CMC biomarker with relatively small variability in regional cortical
measurements is noteworthy. Reliability and reproducibility are a major
ongoing challenge in neuroscience research (Martinez et al., 2015; Marek et
al., 2022), so any biomarkers that present consistently across a large
population, and reliably demonstrate differential presentation across
cortical regions, has considerable potential to assist in reliable and
reproducible characterization of the human brain.

For most of the CMC biomarkers evaluated, males and females exhibit highly
overlapping distributions, implying negligible differences in most cortical
regions, which could imply the proposed biomarkers provide standardization
benefits towards reproducible studies, and is consistent with largely
overlapping functional abilities between the genders in most capacities.
However, some male-female differences were observed in the temporal pole, the
frontal pole and the pars opercularis, with females exhibiting higher CMC
biomarkers on average than their male counterparts, though the distributions
are still overlapping (see Figure 1). The temporal pole has been implicated
in many functions, including emotional processing (Corcoles-Parada et al.,
2019), and the frontal pole has been reported to contribute to control over
emotional approach-avoidance actions (Bramson et al., 2020). Thus, gender
differences in the presentation of the temporal and frontal poles, as
assessed by CMC, may assist in characterization of known gender differences
in emotional expression (Chaplin, 2015). The pars opercularis is involved in
language processing (Grewe et al., 2005), and sex differences in the pars
opercularis, as assessed with CMC, may be indicative of underlying known
differences in language development between males and females (Sato, 2020).
Indeed, it is encouraging that group-wise differences are observed in
overlapping distributions as although we know sex effects exist in emotional
expression and language development, there is a wide amount of variability in
function across both genders, which is reflected in our CMC biomarker results
exhibiting partially overlapping distributions between the sexes.


***Hopefully there is something related to phenotypic data that Derek’s analysis will uncover?

***Discuss variation explained psychological statistical test results here

***Discuss machine learning predictive capacity and features underlying those predictions here

Although most cortical regions exhibited consistent CMC values in the left
and right hemispheres, we did observe asymmetries in the transverse temporal,
entorhinal, caudal anterior cingulate and pericalcarine regions. Asymmetries
have previously been observed in the entorhinal (Simic et al., 2005) cortex,
with larger surface areas being reported in the left hemisphere, which is
consistent with our findings of decreased CMC in the left hemisphere
(increased surface area results in decreased CMC). The transverse temporal
cortex is known to exhibit leftward asymmetries that are detectable by 31
weeks gestation (Chi et al., 1977), which is consistent with our findings of
decreased CMC in the left hemisphere. Asymmetries have also been previously
reported in the anterior cingulate (Yan et al. 2009). The pericalcarine
cortex has also been reported to exhibit asymmetries (Chiarello et al., 2016;
Koelkebeck et al., 2014), which our analysis was also able to detect with CMC
biomarkers. Our identification of asymmetries of CMC biomarkers implies that
our analyses have considerable consistency with known asymmetric properties
of the human brain.

## Potential for CMC to characterize pathologies

A wide variety of pathological conditions have been demonstrated to exhibit
abnormal phenotypic presentation of regions of the brain, including Down
Syndrome (Lee et al., 2016; Levman et al., 2019a), attention deficit
hyperactivity disorder (ADHD) (Liston et al., 2011; Stanley et al., 2008),
schizophrenia (Innocenti et al., 2003; Keshavan et al., 1994; Feinberg, 1990;
Hoffman and Dobscha, 1989; Rimol et al., 2010; Narr et al., 2005;
Venkatasubramanian et al., 2008; van Haren et al., 2011; Schultz et al.,
2010; Nesvag et al., 2008; Seitz et al., 2018; Qiu et al., 2010; Johnson et
al., 2014; MacKinley et al., 2020), psychotic disorders (Bakker et al.,
2016), autism (Khundrakpam et al., 2017; Pereira et al., 2018; Zielinski et
al., 2014; Levman et al., 2019b; Levman et al., 2021b), and multiple
sclerosis (Brex et al., 2002; Losseff et al., 1996; Chen et al., 2004; Sailer
et al., 2003; Levman et al., 2021c).

Thus, future work will entail the characterization of the development of the
pathological brain with CMC. As an additional novel biomarker not previously
available, CMC may characterize regional abnormal development of the cortex
in a manner not previously characterized, and the feature measurements
generated by the approach outlined in this manuscript may also be a useful
addition to future machine learning / artificial intelligence technologies
that perform predictions for diagnostics, prognostics and treatment planning.
Future work will investigate the potential for a variety of pathologies to be
associated with macro-structural developmental abnormalities, such as
aberrant folding and sulcal formation, and thus CMC may assist in the
characterization of the macro-level phenotypic presentation of the brain. It
is hoped that the CMC technique presented in this manuscript will be helpful
in characterizing and understanding the developmental processes and
etiological factors associated with healthy brain development, as well as a
variety of neurodevelopmental disorders. It is also hoped that congruence
based biomarkers will assist in characterizing important aspects of healthy
and abnormal brain, reliably and reproducibly.

***

Talk to Bruce Fischl with Emi about possibility of it being included in future versions of FreeSurfer and whether he has any feedback that might be relevant and may want to join on this project as a co-author.

# Appendix A {#sec:appendix_a}

| ROI                                |      d |          W |   W (p) |
|:-----------------------------------|-------:|-----------:|--------:|
| ThickAvg__inferiortemporal         |  1.913 | 168552.000 |   0.000 |
| ThickAvg__rostralanteriorcingulate | -1.898 |   4993.500 |   0.000 |
| ThickAvg__fusiform                 | -1.843 |      0.000 |   0.000 |
| ThickAvg__pericalcarine            |  1.842 |   7625.500 |   0.000 |
| ThickAvg__postcentral              | -1.825 |      0.000 |   0.000 |
| ThickAvg__precentral               |  1.824 |      0.000 |   0.000 |
| ThickAvg__isthmuscingulate         |  1.780 |      0.000 |   0.000 |
| ThickAvg__insula                   |  1.774 |      0.000 |   0.000 |
| ThickAvg__paracentral              |  1.751 |  30331.500 |   0.000 |
| ThickAvg__precuneus                |  1.716 |      0.000 |   0.000 |
| ThickAvg__caudalanteriorcingulate  | -1.651 |      0.000 |   0.000 |
| ThickAvg__cuneus                   |  1.607 |      0.000 |   0.000 |
| ThickAvg__lateralorbitofrontal     |  1.599 |      0.000 |   0.000 |
| ThickAvg__lingual                  | -1.563 |      0.000 |   0.000 |
| ThickAvg__transversetemporal       |  1.538 |      0.000 |   0.000 |
| ThickAvg__superiortemporal         | -1.472 | 116444.500 |   0.000 |
| ThickAvg__posteriorcingulate       | -1.470 |      0.000 |   0.000 |
| ThickAvg__parahippocampal          |  1.466 |      0.000 |   0.000 |
| ThickAvg__superiorparietal         |  1.433 |      0.000 |   0.000 |
| Area__frontalpole                  | -1.360 |   3510.500 |   0.000 |
| SurfArea__frontalpole              | -1.360 |  47046.000 |   0.000 |
| GrayVol__frontalpole               | -1.330 |   3527.500 |   0.000 |
| SurfArea__transversetemporal       |  1.319 |   7695.000 |   0.000 |
| Area__transversetemporal           |  1.319 |  12451.000 |   0.000 |
| Area__parsorbitalis                | -1.266 |   4209.000 |   0.000 |
| SurfArea__parsorbitalis            | -1.266 |  88337.000 |   0.000 |
| GrayVol__parsorbitalis             | -1.265 |   2215.000 |   0.000 |
| ThickAvg__middletemporal           | -1.218 |      0.000 |   0.000 |
| GrayVol__inferiorparietal          | -1.163 |   2039.500 |   0.000 |
| ThickAvg__parstriangularis         | -1.159 |      0.000 |   0.000 |
| ThickAvg__supramarginal            | -1.123 |      0.000 |   0.000 |
| GrayVol__transversetemporal        |  1.104 |   1464.500 |   0.000 |
| SurfArea__rostralanteriorcingulate |  1.084 |      0.000 |   0.000 |
| Area__rostralanteriorcingulate     |  1.084 |  34500.000 |   0.000 |
| Area__inferiorparietal             | -1.078 |    833.000 |   0.000 |
| SurfArea__inferiorparietal         | -1.078 | 120187.500 |   0.000 |
| ThickAvg__frontalpole              | -1.037 |      0.000 |   0.000 |
| Thick___medialorbitofrontal        | -1.031 |      0.000 |   0.000 |
| GrayVol__rostralanteriorcingulate  |  1.024 |  21433.500 |   0.000 |
| ThickAvg__lateraloccipital         | -1.017 |     49.000 |   0.000 |
| ThickAvg__caudalmiddlefrontal      | -0.973 |  11104.000 |   0.000 |
| GrayVol__parstriangularis          | -0.966 |  41362.000 |   0.000 |
| ThickAvg__medialorbitofrontal      | -0.955 |      0.000 |   0.000 |
| SurfArea__temporalpole             |  0.929 |  91259.500 |   0.000 |
| Area__temporalpole                 |  0.929 | 187447.000 |   0.000 |
| ThickAvg__parsorbitalis            |  0.915 |      0.000 |   0.000 |
| Area__parsopercularis              |  0.911 |  41545.000 |   0.000 |
| SurfArea__parsopercularis          |  0.911 | 297040.500 |   0.744 |
| GrayVol__parsopercularis           |  0.896 |  38251.500 |   0.000 |
| SurfArea__parstriangularis         | -0.867 |  16618.000 |   0.000 |
| Area__parstriangularis             | -0.867 |  29132.500 |   0.000 |
| SurfArea__paracentral              | -0.841 |      0.000 |   0.000 |
| Area__paracentral                  | -0.841 |  62069.500 |   0.000 |
| GrayVol__middletemporal            | -0.803 |  18687.000 |   0.000 |
| Thick___bankssts                   | -0.777 |      0.000 |   0.000 |
| ThickAvg__temporalpole             | -0.770 |      0.000 |   0.000 |
| GrayVol__paracentral               | -0.766 |  43831.000 |   0.000 |
| Thick___caudalanteriorcingulate    |  0.753 |      0.000 |   0.000 |
| Thick___temporalpole               | -0.741 |      0.000 |   0.000 |
| Area__entorhinal                   |  0.738 | 126581.000 |   0.000 |
| SurfArea__entorhinal               |  0.738 | 150876.000 |   0.000 |
| Area__middletemporal               | -0.718 |  11374.500 |   0.000 |
| SurfArea__middletemporal           | -0.718 |  12275.000 |   0.000 |
| SurfArea__caudalanteriorcingulate  | -0.707 |    469.500 |   0.000 |
| Area__caudalanteriorcingulate      | -0.707 | 151094.000 |   0.000 |
| ThickAvg__inferiorparietal         | -0.706 | 226159.000 |   0.000 |
| Thick___inferiorparietal           | -0.635 |      0.000 |   0.000 |
| GrayVol__pericalcarine             | -0.593 |  57421.000 |   0.000 |
| GrayVol__caudalanteriorcingulate   | -0.589 | 106594.500 |   0.000 |
| Thick___middletemporal             | -0.555 |      0.000 |   0.000 |
| SurfArea__pericalcarine            | -0.530 |      0.000 |   0.000 |
| Area__pericalcarine                | -0.530 |  46557.500 |   0.000 |
| SurfArea__medialorbitofrontal      |  0.496 |  22949.000 |   0.000 |
| Area__medialorbitofrontal          |  0.496 | 285946.500 |   0.161 |
| Thick___parstriangularis           | -0.486 |   8126.500 |   0.000 |
| GrayVol__entorhinal                |  0.485 |  62042.000 |   0.000 |
| Thick___entorhinal                 | -0.484 |      0.000 |   0.000 |
| ThickAvg__entorhinal               | -0.471 |  16634.500 |   0.000 |
| Area__bankssts                     |  0.471 | 225056.000 |   0.000 |
| SurfArea__bankssts                 |  0.471 | 268972.000 |   0.002 |
| SurfArea__insula                   | -0.463 |     16.000 |   0.000 |
| Area__insula                       | -0.463 | 106012.500 |   0.000 |
| Thick___lateraloccipital           | -0.462 |      0.000 |   0.000 |
| GrayVol__parahippocampal           |  0.458 | 198655.000 |   0.000 |
| SurfArea__isthmuscingulate         |  0.446 |      0.000 |   0.000 |
| Area__isthmuscingulate             |  0.446 | 184355.500 |   0.000 |
| Thick___transversetemporal         | -0.443 |      0.000 |   0.000 |
| SurfArea__caudalmiddlefrontal      |  0.433 |  66604.000 |   0.000 |
| Area__caudalmiddlefrontal          |  0.433 | 179733.500 |   0.000 |
| ThickAvg__rostralmiddlefrontal     | -0.431 |      0.000 |   0.000 |
| GrayVol__precuneus                 | -0.410 |  96547.500 |   0.000 |
| Thick___superiortemporal           | -0.403 |      0.000 |   0.000 |
| SurfArea__superiortemporal         |  0.400 |   5106.000 |   0.000 |
| Area__superiortemporal             |  0.400 | 204111.000 |   0.000 |
| SurfArea__supramarginal            |  0.398 |  42292.500 |   0.000 |
| Area__supramarginal                |  0.398 | 170684.500 |   0.000 |
| Thick___supramarginal              | -0.396 |      0.000 |   0.000 |
| SurfArea__precuneus                | -0.393 |      0.000 |   0.000 |
| Area__precuneus                    | -0.393 |  85970.000 |   0.000 |
| Thick___superiorparietal           | -0.376 | 106136.500 |   0.000 |
| GrayVol__temporalpole              |  0.374 |  52650.000 |   0.000 |
| SurfArea__postcentral              |  0.355 |      0.000 |   0.000 |
| Area__postcentral                  |  0.355 | 182161.500 |   0.000 |
| GrayVol__rostralmiddlefrontal      | -0.349 | 144353.000 |   0.000 |
| GrayVol__insula                    | -0.343 |  95660.500 |   0.000 |
| Thick___parsopercularis            | -0.342 |      0.000 |   0.000 |
| Thick___parsorbitalis              | -0.339 |      0.000 |   0.000 |
| GrayVol__isthmuscingulate          |  0.334 | 154757.000 |   0.000 |
| GrayVol__supramarginal             |  0.333 | 144106.500 |   0.000 |
| GrayVol__caudalmiddlefrontal       |  0.313 | 137658.000 |   0.000 |
| Thick___inferiortemporal           | -0.305 | 122975.000 |   0.000 |
| Thick___insula                     |  0.302 |      0.000 |   0.000 |
| Thick___precuneus                  | -0.302 |      0.000 |   0.000 |
| SurfArea__inferiortemporal         |  0.286 |      0.000 |   0.000 |
| Area__inferiortemporal             |  0.286 | 245206.000 |   0.000 |
| Area__rostralmiddlefrontal         | -0.286 | 115219.500 |   0.000 |
| SurfArea__rostralmiddlefrontal     | -0.286 | 188723.500 |   0.000 |
| SurfArea__parahippocampal          |  0.275 |   3113.500 |   0.000 |
| Area__parahippocampal              |  0.275 | 140248.500 |   0.000 |
| GrayVol__bankssts                  |  0.266 | 152069.500 |   0.000 |
| GrayVol__postcentral               |  0.265 | 140871.500 |   0.000 |
| Thick___superiorfrontal            | -0.248 |   2378.500 |   0.000 |
| Thick___posteriorcingulate         |  0.242 |   9243.000 |   0.000 |
| Thick___paracentral                | -0.221 |   2957.500 |   0.000 |
| Thick___postcentral                | -0.218 |      0.000 |   0.000 |
| GrayVol__cuneus                    | -0.217 | 208953.500 |   0.000 |
| GrayVol__fusiform                  |  0.216 | 212240.500 |   0.000 |
| SurfArea__lateralorbitofrontal     |  0.211 |    110.000 |   0.000 |
| Area__lateralorbitofrontal         |  0.211 | 194772.500 |   0.000 |
| SurfArea__fusiform                 |  0.206 |      0.000 |   0.000 |
| Area__fusiform                     |  0.206 | 208310.000 |   0.000 |
| ThickAvg__superiorfrontal          | -0.195 |      0.000 |   0.000 |
| GrayVol__superiortemporal          |  0.190 | 106677.000 |   0.000 |
| SurfArea__posteriorcingulate       | -0.187 |   6120.500 |   0.000 |
| Area__posteriorcingulate           | -0.187 | 275233.000 |   0.010 |
| Thick___isthmuscingulate           | -0.183 |      0.000 |   0.000 |
| SurfArea__cuneus                   | -0.179 |   1519.000 |   0.000 |
| Area__cuneus                       | -0.179 | 191317.500 |   0.000 |
| Area__superiorfrontal              |  0.175 | 250741.000 |   0.000 |
| SurfArea__superiorfrontal          |  0.175 | 254495.000 |   0.000 |
| GrayVol__superiorparietal          | -0.174 | 307024.500 |   0.784 |
| SurfArea__lateraloccipital         |  0.164 |  66088.000 |   0.000 |
| Area__lateraloccipital             |  0.164 | 297578.500 |   0.744 |
| Thick___fusiform                   | -0.158 |      0.000 |   0.000 |
| GrayVol__lateralorbitofrontal      |  0.157 | 170688.000 |   0.000 |
| Thick___caudalmiddlefrontal        | -0.153 |      0.000 |   0.000 |
| Thick___lateralorbitofrontal       | -0.149 |      0.000 |   0.000 |
| SurfArea__precentral               | -0.140 |     10.000 |   0.000 |
| Area__precentral                   | -0.140 | 262601.500 |   0.000 |
| Thick___lingual                    | -0.136 |  22381.500 |   0.000 |
| GrayVol__inferiortemporal          |  0.135 | 177113.000 |   0.000 |
| Thick___rostralmiddlefrontal       | -0.129 |      0.000 |   0.000 |
| ThickAvg__bankssts                 |  0.127 |      0.000 |   0.000 |
| Thick___frontalpole                | -0.119 |      0.000 |   0.000 |
| GrayVol__posteriorcingulate        | -0.099 | 246876.500 |   0.000 |
| GrayVol__superiorfrontal           |  0.094 | 195883.000 |   0.000 |
| GrayVol__precentral                | -0.088 | 232292.500 |   0.000 |
| Thick___parahippocampal            |  0.087 |      0.000 |   0.000 |
| GrayVol__lingual                   | -0.063 | 290324.500 |   0.268 |
| Thick___rostralanteriorcingulate   |  0.063 |      0.000 |   0.000 |
| Thick___cuneus                     | -0.063 | 132102.000 |   0.000 |
| SurfArea__lingual                  | -0.044 |   3167.500 |   0.000 |
| Area__lingual                      | -0.044 | 284216.000 |   0.112 |
| GrayVol__medialorbitofrontal       | -0.044 | 128854.000 |   0.000 |
| ThickAvg__parsopercularis          |  0.040 |      0.000 |   0.000 |
| Thick___pericalcarine              | -0.031 |   4999.500 |   0.000 |
| Thick___precentral                 |  0.029 |   4019.500 |   0.000 |
| GrayVol__lateraloccipital          |  0.024 | 232025.500 |   0.000 |
| SurfArea__superiorparietal         | -0.009 |    592.500 |   0.000 |
| Area__superiorparietal             | -0.009 | 215625.000 |   0.000 |

: Measures of Separation of base FreeSurfer Features (left vs. right
hemisphere). d = Cohen's d, W = Wilcoxon signed rank test, W (p) = p-value
for W. Note: p-values were adjusted for multiple comparisons using the
Holm-Bonferroni stepdown method {#tbl:lateral_fs}

# Appendix B

In the tables below, names follow HCP data naming conventions, unless otherwise
indicated. The right column label indicates the name given to the single-factor
synthetic target.

_perf = test performance, i.e. test score
_rt = reaction time, i.e. reaction times on a test
wm = working memory

|                                  |   gambling_perf |
|:---------------------------------|----------------:|
| gambling_task_perc_larger        |         -1.0000 |
| gambling_task_reward_perc_larger |         -0.8926 |
| gambling_task_punish_perc_larger |         -0.8337 |


: Synthetic target `gambling_perf` factor loadings.
{#tbl:gambling_perf}


|                        |   emotion_perf |
|:-----------------------|---------------:|
| emotion_task_acc       |        -1.0000 |
| emotion_task_shape_acc |        -0.8952 |
| emotion_task_face_acc  |        -0.8298 |


: Synthetic target `emotion_perf` factor loadings.
{#tbl:emotion_perf}


|                               |   language_rt |
|:------------------------------|--------------:|
| language_task_median_rt       |        1.0000 |
| language_task_story_median_rt |        0.8261 |
| language_task_math_median_rt  |        0.8138 |


: Synthetic target `language_rt` factor loadings.
{#tbl:language_rt}


|                                 |   relational_rt |
|:--------------------------------|----------------:|
| relational_task_median_rt       |         -1.0000 |
| relational_task_rel_median_rt   |         -0.9503 |
| relational_task_match_median_rt |         -0.8610 |


: Synthetic target `relational_rt` factor loadings.
{#tbl:relational_rt}


|                              |   emotion_rt |
|:-----------------------------|-------------:|
| emotion_task_median_rt       |       1.0000 |
| emotion_task_face_median_rt  |       0.9528 |
| emotion_task_shape_median_rt |       0.9373 |


: Synthetic target `emotion_rt` factor loadings.
{#tbl:emotion_rt}


|                                          |   language_perf |
|:-----------------------------------------|----------------:|
| language_task_math_acc                   |         -0.9975 |
| language_task_acc                        |         -0.8386 |
| language_task_story_avg_difficulty_level |         -0.7606 |


: Synthetic target `language_perf` factor loadings.
{#tbl:language_perf}


|             |   p_matrices |
|:------------|-------------:|
| p_matrices  |       1.0000 |
| pmat24_a_cr |       0.7158 |
| pmat24_a_si |      -0.7005 |


: Synthetic target `p_matrices` factor loadings.
{#tbl:p_matrices}


|                                     |   social_rt |
|:------------------------------------|------------:|
| social_task_median_rt_random        |      0.9838 |
| social_task_random_median_rt_random |      0.9826 |
| social_task_tom_median_rt_tom       |      0.4105 |
| social_task_median_rt_tom           |      0.4017 |


: Synthetic target `social_rt` factor loadings.
{#tbl:social_rt}


|            |   psqi_latent |
|:-----------|--------------:|
| psqi_score |        1.0000 |
| psqi_comp1 |        0.6853 |
| psqi_comp4 |        0.6379 |
| psqi_comp3 |        0.6266 |
| psqi_comp7 |        0.4649 |


: Synthetic target `psqi_latent` factor loadings.
{#tbl:psqi_latent}


|                                        |   gambling_rt |
|:---------------------------------------|--------------:|
| gambling_task_median_rt_smaller        |        1.0000 |
| gambling_task_reward_median_rt_smaller |        0.9512 |
| gambling_task_punish_median_rt_smaller |        0.9504 |
| gambling_task_median_rt_larger         |        0.8836 |
| gambling_task_reward_median_rt_larger  |        0.8451 |
| gambling_task_punish_median_rt_larger  |        0.8357 |


: Synthetic target `gambling_rt` factor loadings.
{#tbl:gambling_rt}


|                                |   social_random_perf |
|:-------------------------------|---------------------:|
| social_task_random_perc_random |              -1.0000 |
| social_task_perc_random        |              -0.9096 |
| social_task_random_perc_unsure |               0.8121 |
| social_task_perc_unsure        |               0.7238 |
| social_task_random_perc_tom    |               0.5595 |
| social_task_perc_tom           |               0.3081 |
| social_task_tom_perc_tom       |              -0.1047 |
| social_task_tom_perc_unsure    |               0.0870 |
| social_task_tom_perc_random    |               0.0638 |


: Synthetic target `social_random_perf` factor loadings.
{#tbl:social_random_perf}


|                      |   int_g_like |
|:---------------------|-------------:|
| cogtotalcomp_unadj   |      -1.0000 |
| cogearlycomp_unadj   |      -0.8750 |
| cogfluidcomp_unadj   |      -0.8501 |
| cogcrystalcomp_unadj |      -0.7735 |
| readeng_unadj        |      -0.7230 |
| picvocab_unadj       |      -0.6767 |
| listsort_unadj       |      -0.5698 |
| cardsort_unadj       |      -0.5668 |
| procspeed_unadj      |      -0.5640 |
| flanker_unadj        |      -0.5116 |
| picseq_unadj         |      -0.4938 |


: Synthetic target `int_g_like` factor loadings.
{#tbl:int_g_like}


|                  |   neg_emotionality |
|:-----------------|-------------------:|
| percstress_unadj |             0.8250 |
| sadness_unadj    |             0.8165 |
| loneliness_unadj |             0.7812 |
| neofac_n         |             0.7596 |
| angaffect_unadj  |             0.7127 |
| fearaffect_unadj |             0.7000 |
| percreject_unadj |             0.6994 |
| lifesatisf_unadj |            -0.6571 |
| posaffect_unadj  |            -0.6541 |
| anghostil_unadj  |             0.6524 |
| emotsupp_unadj   |            -0.6347 |
| meanpurp_unadj   |            -0.6078 |
| friendship_unadj |            -0.5947 |
| selfeff_unadj    |            -0.5589 |
| perchostil_unadj |             0.5512 |
| neofac_e         |            -0.4649 |
| instrusupp_unadj |            -0.4578 |
| neofac_a         |            -0.3779 |
| fearsomat_unadj  |             0.3752 |


: Synthetic target `neg_emotionality` factor loadings.
{#tbl:neg_emotionality}


|                                       |   wm_rt |
|:--------------------------------------|--------:|
| wm_task_median_rt                     |  1.0000 |
| wm_task_0bk_median_rt                 |  0.9059 |
| wm_task_2bk_median_rt                 |  0.8999 |
| wm_task_0bk_face_median_rt            |  0.7928 |
| wm_task_0bk_tool_median_rt            |  0.7924 |
| wm_task_0bk_body_median_rt            |  0.7914 |
| wm_task_0bk_place_median_rt           |  0.7913 |
| wm_task_0bk_tool_median_rt_nontarget  |  0.7911 |
| wm_task_2bk_face_median_rt            |  0.7886 |
| wm_task_0bk_face_median_rt_nontarget  |  0.7847 |
| wm_task_0bk_place_median_rt_nontarget |  0.7826 |
| wm_task_0bk_body_median_rt_nontarget  |  0.7816 |
| wm_task_2bk_place_median_rt           |  0.7638 |
| wm_task_2bk_face_median_rt_nontarget  |  0.7573 |
| wm_task_2bk_tool_median_rt            |  0.7409 |
| wm_task_2bk_place_median_rt_nontarget |  0.7372 |
| wm_task_2bk_tool_median_rt_nontarget  |  0.7083 |
| wm_task_2bk_body_median_rt            |  0.6629 |
| wm_task_2bk_tool_median_rt_target     |  0.6306 |
| wm_task_2bk_body_median_rt_nontarget  |  0.6159 |
| wm_task_0bk_face_median_rt_target     |  0.5825 |
| wm_task_2bk_face_median_rt_target     |  0.5699 |
| wm_task_0bk_place_median_rt_target    |  0.5302 |


: Synthetic target `wm_rt` factor loadings.
{#tbl:wm_rt}


|                                 |   wm_perf |
|:--------------------------------|----------:|
| wm_task_acc                     |   -0.9931 |
| wm_task_0bk_acc                 |   -0.8816 |
| wm_task_2bk_acc                 |   -0.7975 |
| wm_task_0bk_body_acc            |   -0.7361 |
| wm_task_0bk_tool_acc            |   -0.7269 |
| wm_task_0bk_place_acc           |   -0.6881 |
| wm_task_0bk_tool_acc_nontarget  |   -0.6827 |
| wm_task_0bk_body_acc_nontarget  |   -0.6796 |
| wm_task_0bk_face_acc            |   -0.6730 |
| wm_task_2bk_body_acc            |   -0.6715 |
| wm_task_0bk_face_acc_nontarget  |   -0.6543 |
| wm_task_0bk_place_acc_nontarget |   -0.6469 |
| wm_task_2bk_face_acc            |   -0.6418 |
| wm_task_2bk_tool_acc            |   -0.6395 |
| wm_task_2bk_place_acc           |   -0.6326 |
| wm_task_0bk_body_acc_target     |   -0.6263 |
| wm_task_0bk_tool_acc_target     |   -0.5900 |
| wm_task_2bk_body_acc_nontarget  |   -0.5782 |
| wm_task_0bk_face_acc_target     |   -0.5502 |
| wm_task_0bk_place_acc_target    |   -0.5501 |
| wm_task_2bk_body_acc_target     |   -0.5481 |
| wm_task_2bk_tool_acc_nontarget  |   -0.5383 |
| wm_task_2bk_tool_acc_target     |   -0.4792 |
| wm_task_2bk_face_acc_target     |   -0.4539 |
| wm_task_2bk_place_acc_target    |   -0.4516 |


: Synthetic target `wm_perf` factor loadings.
{#tbl:wm_perf}




# References