- [Preprocessing](#preprocessing)
  - [Target Variable Selection](#target-variable-selection)
  - [Target Variable Reduction](#target-variable-reduction)
    - [Factor Analysis](#factor-analysis)
- [Appendix](#appendix)
  - [Raw Feature Table](#raw-feature-table)
  - [Cluster Components](#cluster-components)
  - [Latent Factor Loadings](#latent-factor-loadings)
  - [Latent Factor Correlations](#latent-factor-correlations)
- [References](#references)


# Preprocessing

## Target Variable Selection

The phenotypic data available with the HCP is comprehensive, consisting of
hundreds of different features. Many of these features (83) are essentially HCP
"book-keeping" features (such as the number of fMRI scans a subject completed,
whether a subject completed a particular measurement or not, or identification
numbers of various kinds), and were excluded as potential prediction targets.

A number of other features (91) record the responses to single items from
larger measurement measurement instruments. This includes 60 single-items
comprising the NEO-FFI [@mccraeContemplatedRevisionNEO2004], 19 items
comprising the Pittsburgh Sleep Quality Index [PSQI\;
@buyssePittsburghSleepQuality1989], and 12 items measuring delay discounting
[@doyleSurveyTimePreference2013] at different combinations of compensation and
delays. The phenotypic data already includes total scale summary measures for
these, so these single-item measures were also excluded.

Of the remaining features, 7 features were excluded which were over 50% missing
values. The phenotypic features also include a number of features that have
both an "age-adjusted" and unadjusted form. However, a number (16) of these
feature pairs are extremely highly correlated (r >= 0.97), and thus only the
unadjusted versions were retained from these pairs. Likewise, a small number of
features (7) were perfectly correlated by virtue of being directly computable
from other features, and were also dropped from consideration. This left a total
of 373 potential targets.

Of these remaining potential targets, 199 are FreeSurfer measures (predictors),
and two are demographic variables (sex, age), leaving 172 potential target
variables comprising a wide variety of cognitive, psychological, emotional, and
social variables (see [Appendix: Feature Table](#raw-feature-table) below for
details). All these remaining targets are continuous variables suitable for
prediction with regression models.

## Target Variable Reduction

Two methods were employed to reduce the large number of potential target
features to a more interpretable and computationally tractable set of
regression targets.

First, feature similarities were computed by computing $\lvert\rho_{ij}\rvert$
for where $\rho_{ij}$ is the Pearson correlation between features $i$ and $j$.
Then, these were converted to distances $1 - \lvert\rho_{ij}\rvert$ and
clustered using the scikit-learn [@JMLR:v12:pedregosa11a] implementation of
HDBSCAN [@campelloDensityBasedClusteringBased2013;
@campelloHierarchicalDensityEstimates2015]. HDBSCAN has the advantage of
requiring less tuning than some other clustering algorithms, and also allows
for elements to be unclustered (assigned to a noise cluster). This is in
contrast to methods like PCA, Factor Analysis, and Independent Components
Analysis, which require pre-specifying the number of components, and which
forcefully assigns all features to one of these components, even if the
contribution is weak.

HDBSCAN was performed with a minimum cluster size of three, and cluster
selection epsilon [@malzerHybridApproachHierarchical2020] of 0.2. Feature clusters
extracted with different larger minimum cluster sizes and epsilon values were
highly similar. Smaller minimum cluster sizes and epsilons resulted in a much
larger number of feature clusters, which, for the purpose of data reduction and
interpretability, seemed far less desirable. Final cluster components (with the
component correlations) can be found in [Appendix: Cluster
Components](#cluster-components).


### Factor Analysis

After identifying these feature clusters, missing values were filled with the
feature mean^[Less than about 5% of features in all cases.], and each cluster
was reduced to a single latent factor via factor analysis. This resulted in the
factor loadings (expressed as Pearson correlations between the original and
latent features) in [Appendix: Latent Factor Loadings](#latent-factor-loadings).

Factor analysis was chosen for dimension reduction of the clustered target features for
a number of reasons:

1. Factor analytic reduction explicitly models and attempts to exclude error
   variance for each factor, such that the reduction better reflects the common
   variance of the factors [@mulaikFoundationsFactorAnalysis2013]. PCA
   reduction would include this error variance in the reduction
   [@fruchterIntroductionFactorAnalysis1954; @cattellScientificUseFactor1978;
   @childEssentialsFactorAnalysis2006a; @gorsuchFactorAnalysis1983;
   @mcdonaldFactorAnalysisRelated2014].
1. FA-reduced latent variables can be more reasonably interpreted as underlying
   variables that determine the correlation structure of the unreduced
   variables, whereas this interpretation for PCA-reduced factors is less sound
   [@fabrigarEvaluatingUseExploratory1999].
1. In this case the correlated features are reduced to a single dimension,
   which makes it especially important that we are not including variance which
   would be better modeled as error variance
1. Because of the one-dimensional reduction, each reduced feature is a simple
   linear combination of the components, and excluding noise variance in each
   component. This makes the factor loadings and signs in the [Appendix: Latent
   Factor Loadings](#latent-factor-loadings) directly interpretable as the
   signed contribution to each latent factor.


This yielded a total of 15 latent factors suitable to be used as interpretable
regression targets.

# Appendix

## Raw Feature Table

| feature                                  | kind      | fulldisplayname                                                                                                           |
|:-----------------------------------------|:----------|:--------------------------------------------------------------------------------------------------------------------------|
| angaffect_unadj                          | emotion   | NIH Toolbox Anger-Affect Survey: Unadjusted Scale Score                                                                   |
| angaggr_unadj                            | emotion   | NIH Toolbox Anger-Physical Aggression Survey: Unadjusted Scale Score                                                      |
| anghostil_unadj                          | emotion   | NIH Toolbox Anger-Hostility Survey: Unadjusted Scale Score                                                                |
| cardsort_unadj                           | cognition | NIH Toolbox Dimensional Change Card Sort Test: Unadjusted Scale Score                                                     |
| cogcrystalcomp_unadj                     | cognition | NIH Toolbox Cognition Crystallized Composite: Unadjusted Scale Score                                                      |
| cogearlycomp_unadj                       | cognition | NIH Toolbox Cognition Early Childhood Composite: Unadjusted Scale Score                                                   |
| cogfluidcomp_unadj                       | cognition | NIH Toolbox Cognition Fluid Composite: Unadjusted Scale Score                                                             |
| cogtotalcomp_unadj                       | cognition | NIH Toolbox Cognition Total Composite Score: Unadjusted Scale Score                                                       |
| ddisc_auc_200                            | psych     | Delay Discounting: Area Under the Curve for Discounting of $200 (DDisc_AUC_200)                                           |
| ddisc_auc_40k                            | psych     | Delay Discounting: Area Under the Curve for Discounting of $40,000 (DDisc_AUC_40K)                                        |
| dexterity_unadj                          | physio    | NIH Toolbox 9-hole Pegboard Dexterity Test : Unadjusted Scale Score                                                       |
| emotion_task_acc                         | emotion   | OVERALL Emotion Task accuracy                                                                                             |
| emotion_task_face_acc                    | emotion   | Emotion Task FACE accuracy                                                                                                |
| emotion_task_face_median_rt              | timing    | Emotion Task FACE median Reaction Time                                                                                    |
| emotion_task_median_rt                   | timing    | OVERALL Emotion Task Reaction Time                                                                                        |
| emotion_task_shape_acc                   | emotion   | Emotion Task SHAPE accuracy                                                                                               |
| emotion_task_shape_median_rt             | timing    | Emotion Task SHAPE median Reaction Time                                                                                   |
| emotsupp_unadj                           | emotion   | NIH Toolbox Emotional Support Survey: Unadjusted Scale Score                                                              |
| endurance_unadj                          | physio    | NIH Toolbox 2-minute Walk Endurance Test : Unadjusted Scale Score                                                         |
| er40_cr                                  | emotion   | Penn Emotion Recognition Test: Number of Correct Responses (ER40_CR)                                                      |
| er40_crt                                 | timing    | Penn Emotion Recognition Test: Correct Responses Median Response Time (ms) (ER40_CRT)                                     |
| er40ang                                  | emotion   | Penn Emotion Recognition Test: Number of Correct Anger Identifications (ER40ANG)                                          |
| er40fear                                 | emotion   | Penn Emotion Recognition Test: Number of Correct Fear Identifications (ER40FEAR)                                          |
| er40hap                                  | emotion   | Penn Emotion Recognition Test: Number of Correct Happy Identifications (ER40HAP)                                          |
| er40noe                                  | emotion   | Penn Emotion Recognition Test: Number of Correct Neutral Identifications (ER40NOE)                                        |
| er40sad                                  | emotion   | Penn Emotion Recognition Test: Number of Correct Sad Identifications (ER40SAD)                                            |
| fearaffect_unadj                         | emotion   | NIH Toolbox Fear-Affect Survey: Unadjusted Scale Score                                                                    |
| fearsomat_unadj                          | emotion   | NIH Toolbox Fear-Somatic Arousal Survey: Unadjusted Scale Score                                                           |
| flanker_unadj                            | cognition | NIH Toolbox Flanker Inhibitory Control and Attention Test: Unadjusted Scale Score                                         |
| friendship_unadj                         | social    | NIH Toolbox Friendship Survey: Unadjusted Scale Score                                                                     |
| gaitspeed_comp                           | physio    | NIH Toolbox 4-Meter Walk Gait Speed Test: Computed Score                                                                  |
| gambling_task_median_rt_larger           | timing    | Gambling Task Overall Reaction Time 'Larger'                                                                              |
| gambling_task_median_rt_smaller          | timing    | Gambling Task Overall Reaction Time 'Smaller'                                                                             |
| gambling_task_perc_larger                | psych     | Gambling Task Overall Percentage 'Larger'                                                                                 |
| gambling_task_perc_nlr                   | psych     | Gambling Task Overall Percentage No Logged Response                                                                       |
| gambling_task_punish_median_rt_larger    | timing    | Gambling Task Median Reaction Time 'Larger' in Punish                                                                     |
| gambling_task_punish_median_rt_smaller   | timing    | Gambling Task Median Reaction Time 'Smaller' in Punish                                                                    |
| gambling_task_punish_perc_larger         | psych     | Gambling Task Percentage 'Larger' in Punish                                                                               |
| gambling_task_punish_perc_nlr            | psych     | Gambling Task Percentage No Logged Response in Punish                                                                     |
| gambling_task_reward_median_rt_larger    | timing    | Gambling Task Median Reaction Time 'Larger' in Reward                                                                     |
| gambling_task_reward_median_rt_smaller   | timing    | Gambling Task Median Reaction Time 'Smaller' in Reward                                                                    |
| gambling_task_reward_perc_larger         | psych     | Gambling Task Percentage 'Larger' in Reward                                                                               |
| gambling_task_reward_perc_nlr            | psych     | Gambling Task Percentage No Logged response in Reward                                                                     |
| instrusupp_unadj                         | social    | NIH Toolbox Instrumental Support Survey: Unadjusted Scale Score                                                           |
| iwrd_rtc                                 | timing    | Penn Word Memory Test: Median Reaction Time for Correct Responses (IWRD_RTC)                                              |
| iwrd_tot                                 | cognition | Penn Word Memory Test: Total Number of Correct Responses (IWRD_TOT)                                                       |
| language_task_acc                        | cognition | Language Task OVERALL accuracy                                                                                            |
| language_task_math_acc                   | cognition | Language Task MATH accuracy                                                                                               |
| language_task_math_avg_difficulty_level  | cognition | Language Task MATH difficulty level                                                                                       |
| language_task_math_median_rt             | timing    | Language Task MATH median Reaction Time                                                                                   |
| language_task_median_rt                  | timing    | Language Task OVERALL median Reaction Time                                                                                |
| language_task_story_acc                  | cognition | Language Task STORY accuracy                                                                                              |
| language_task_story_avg_difficulty_level | cognition | Language Task STORY difficulty level                                                                                      |
| language_task_story_median_rt            | timing    | Language Task STORY median Reaction Time                                                                                  |
| lifesatisf_unadj                         | psych     | NIH Toolbox General Life Satisfaction Survey: Unadjusted Scale Score                                                      |
| listsort_unadj                           | cognition | NIH Toolbox List Sorting Working Memory Test: Unadjusted Scale Score                                                      |
| loneliness_unadj                         | social    | NIH Toolbox Loneliness Survey: Unadjusted Scale Score                                                                     |
| mars_errs                                | physio    | Errors on Mars                                                                                                            |
| mars_final                               | physio    | Mars Final Contrast Sensitivity Score                                                                                     |
| meanpurp_unadj                           | psych     | NIH Toolbox Meaning and Purpose Survey: Unadjusted Scale Score                                                            |
| mmse_score                               | psych     | Mini Mental Status Exam Total Score                                                                                       |
| neofac_a                                 | psych     | NEO-FFI Agreeableness (NEOFAC_A)                                                                                          |
| neofac_c                                 | psych     | NEO-FFI Conscientiousness (NEOFAC_C)                                                                                      |
| neofac_e                                 | psych     | NEO-FFI Extraversion (NEOFAC_E)                                                                                           |
| neofac_n                                 | psych     | NEO-FFI Neuroticism (NEOFAC_N)                                                                                            |
| neofac_o                                 | psych     | NEO-FFI Openness to Experience (NEOFAC_O)                                                                                 |
| odor_unadj                               | physio    | NIH Toolbox Odor Identification Age 3+ Unadjusted Scale Score                                                             |
| painintens_rawscore                      | physio    | NIH Toolbox Pain Intensity Survey Age 18+: Raw score                                                                      |
| paininterf_tscore                        | physio    | NIH Toolbox Pain Interference Survey Age 18+: T-score                                                                     |
| perchostil_unadj                         | social    | NIH Toolbox Perceived Hostility Survey: Unadjusted Scale Score                                                            |
| percreject_unadj                         | social    | NIH Toolbox Perceived Rejection Survey: Unadjusted Scale Score                                                            |
| percstress_unadj                         | psych     | NIH Toolbox Perceived Stress Survey: Unadjusted Scale Score                                                               |
| picseq_unadj                             | cognition | NIH Toolbox Picture Sequence Memory Test: Unadjusted Scale Score                                                          |
| picvocab_unadj                           | cognition | NIH Toolbox Picture Vocabulary Test: Unadjusted Scale Score                                                               |
| pmat24_a_cr                              | cognition | Penn Progressive Matrices: Number of Correct Responses (PMAT24_A_CR)                                                      |
| pmat24_a_rtcr                            | timing    | Penn Progressive Matrices: Median Reaction Time for Correct Responses (PMAT24_A_RTCR)                                     |
| pmat24_a_si                              | cognition | Penn Progressive Matrices: Total Skipped Items (PMAT24_A_SI)                                                              |
| posaffect_unadj                          | emotion   | NIH Toolbox Positive Affect Survey: Unadjusted Scale Score                                                                |
| procspeed_unadj                          | cognition | NIH Toolbox Pattern Comparison Processing Speed Test: Unadjusted Scale Score                                              |
| psqi_comp1                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 1 Score                                                                  |
| psqi_comp2                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 2 Score                                                                  |
| psqi_comp3                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 3 Score                                                                  |
| psqi_comp4                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 4 Score                                                                  |
| psqi_comp5                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 5 Score                                                                  |
| psqi_comp6                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 6 Score                                                                  |
| psqi_comp7                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Component 7 Score                                                                  |
| psqi_score                               | physio    | Sleep (Pittsburgh Sleep Questionnaire) Total Score                                                                        |
| readeng_unadj                            | cognition | NIH Toolbox Oral Reading Recognition Test: Unadjusted Scale Score                                                         |
| relational_task_acc                      | cognition | Relational Task OVERALL accuracy                                                                                          |
| relational_task_match_acc                | cognition | Relational Task MATCH accuracy                                                                                            |
| relational_task_match_median_rt          | timing    | Relational Task MATCH median Reaction Time                                                                                |
| relational_task_median_rt                | timing    | Relational Task OVERALL Reaction Time                                                                                     |
| relational_task_rel_acc                  | cognition | Relational Task RELATIONAL block (REL) accuracy                                                                           |
| relational_task_rel_median_rt            | timing    | Relational Task RELATIONAL block (REL) median Reaction Time                                                               |
| sadness_unadj                            | emotion   | NIH Toolbox Sadness Survey: Unadjusted Scale Score                                                                        |
| scpt_lrnr                                | cognition | Short Penn Continuous Performance Test: Longest Run of Non-Responses (SCPT_LRNR)                                          |
| scpt_sen                                 | cognition | Short Penn Continuous Performance Test: Sensitivity = SCPT_TP/(SCPT_TP + SCPT_FN) (SCPT_SEN)                              |
| scpt_spec                                | cognition | Short Penn Continuous Performance Test: Specificity = SCPT_TN/(SCPT_TN + SCPT_FP) (SCPT_SPEC)                             |
| selfeff_unadj                            | psych     | NIH Toolbox Self-Efficacy Survey: Unadjusted Scale Score                                                                  |
| social_task_median_rt_random             | timing    | Social Task Overall Reaction Time 'Random'                                                                                |
| social_task_median_rt_tom                | timing    | Social Task Overall Reaction Time 'TOM'                                                                                   |
| social_task_perc_nlr                     | cognition | Social Task Overall Percentage No Logged Response                                                                         |
| social_task_perc_random                  | cognition | Social Task Overall Percentage 'Random'                                                                                   |
| social_task_perc_tom                     | cognition | Social Task Overall Percentage 'TOM'                                                                                      |
| social_task_perc_unsure                  | cognition | Social Task Overall Percentage 'Unsure'                                                                                   |
| social_task_random_median_rt_random      | timing    | Social Task Median Reaction Time 'Random' in Random condition                                                             |
| social_task_random_perc_nlr              | cognition | Social Task Percentage No Logged Response in Random condition                                                             |
| social_task_random_perc_random           | cognition | Social Task Percentage 'Random' in Random condition                                                                       |
| social_task_random_perc_tom              | cognition | Social Task Percentage 'TOM' in Random condition                                                                          |
| social_task_random_perc_unsure           | cognition | Social Task Percentage 'Unsure' in Random condition                                                                       |
| social_task_tom_median_rt_tom            | timing    | Social Task Median Reaction Time 'TOM' in Social (TOM) condition                                                          |
| social_task_tom_perc_nlr                 | cognition | Social Task Percentage No Logged Response in Social (TOM) condition                                                       |
| social_task_tom_perc_random              | cognition | Social Task Percentage 'Random' in Social (TOM) condition                                                                 |
| social_task_tom_perc_tom                 | cognition | Social Task Percentage 'TOM' in Social (TOM) condition                                                                    |
| social_task_tom_perc_unsure              | cognition | Social Task Percentage 'Unsure' in Social (TOM) condition                                                                 |
| strength_unadj                           | physio    | NIH Toolbox Grip Strength Test: Unadjusted Scale Score                                                                    |
| taste_unadj                              | physio    | NIH Toolbox Regional Taste Intensity Age 12+ Unadjusted Scale Score                                                       |
| vsplot_crte                              | timing    | Variable Short Penn Line Orientation: Median Reaction Time Divided by Expected Number of Clicks for Correct (VSPLOT_CRTE) |
| vsplot_off                               | cognition | Variable Short Penn Line Orientation: Total Positions Off for All Trials (VSPLOT_OFF)                                     |
| vsplot_tc                                | cognition | Variable Short Penn Line Orientation: Total Number Correct (VSPLOT_TC)                                                    |
| wm_task_0bk_acc                          | cognition | Working Memory Task Accuracy for 0-back                                                                                   |
| wm_task_0bk_body_acc                     | cognition | Working Memory Task Accuracy for 0-back Body                                                                              |
| wm_task_0bk_body_acc_nontarget           | cognition | Working Memory Task Accuracy for 0-back Body Nontargets                                                                   |
| wm_task_0bk_body_acc_target              | cognition | Working Memory Task Accuracy for 0-back Body Targets                                                                      |
| wm_task_0bk_body_median_rt               | timing    | Working Memory Task Median Reaction Time for 0-back Body                                                                  |
| wm_task_0bk_body_median_rt_nontarget     | timing    | Working Memory Task Median Reaction Time for 0-back Body Nontargets                                                       |
| wm_task_0bk_body_median_rt_target        | timing    | Working Memory Task Median Reaction Time for 0-back Body Targets                                                          |
| wm_task_0bk_face_acc                     | cognition | Working Memory Task Accuracy for 0-back Face                                                                              |
| wm_task_0bk_face_acc_nontarget           | cognition | Working Memory Task Accuracy for 0-back Face Nontargets                                                                   |
| wm_task_0bk_face_acc_target              | cognition | Working Memory Task Accuracy for 0-back Face Targets                                                                      |
| wm_task_0bk_face_median_rt               | timing    | Working Memory Task Median Reaction Time for 0-back Face                                                                  |
| wm_task_0bk_face_median_rt_nontarget     | timing    | Working Memory Task Median Reaction Time for 0-back Face Nontargets                                                       |
| wm_task_0bk_face_median_rt_target        | timing    | Working Memory Task Median Reaction Time for 0-back Face Targets                                                          |
| wm_task_0bk_median_rt                    | timing    | Working Memory Task Median Reaction Time for 0-back                                                                       |
| wm_task_0bk_place_acc                    | cognition | Working Memory Task Accuracy for 0-back Place                                                                             |
| wm_task_0bk_place_acc_nontarget          | cognition | Working Memory Task Accuracy for 0-back Place Nontargets                                                                  |
| wm_task_0bk_place_acc_target             | cognition | Working Memory Task Accuracy for 0-back Place Targets                                                                     |
| wm_task_0bk_place_median_rt              | timing    | Working Memory Task Median Reaction Time for 0-back Place                                                                 |
| wm_task_0bk_place_median_rt_nontarget    | timing    | Working Memory Task Median Reaction Time for 0-back Place Nontargets                                                      |
| wm_task_0bk_place_median_rt_target       | timing    | Working Memory Task Median Reaction Time for 0-back Place Targets                                                         |
| wm_task_0bk_tool_acc                     | cognition | Working Memory Task Accuracy for 0-back Tool                                                                              |
| wm_task_0bk_tool_acc_nontarget           | cognition | Working Memory Task Accuracy for 0-back Tool Nontargets                                                                   |
| wm_task_0bk_tool_acc_target              | cognition | Working Memory Task Accuracy for 0-back Tool Targets                                                                      |
| wm_task_0bk_tool_median_rt               | timing    | Working Memory Task Median Reaction Time for 0-back Tool                                                                  |
| wm_task_0bk_tool_median_rt_nontarget     | timing    | Working Memory Task Median Reaction Time for 0-back Tool Nontargets                                                       |
| wm_task_0bk_tool_median_rt_target        | timing    | Working Memory Task Median Reaction Time for 0-back Tool Targets                                                          |
| wm_task_2bk_acc                          | cognition | Working Memory Task Accuracy for 2-back                                                                                   |
| wm_task_2bk_body_acc                     | cognition | Working Memory Task Accuracy for 2-back Body                                                                              |
| wm_task_2bk_body_acc_nontarget           | cognition | Working Memory Task Accuracy for 2-back Body Nontargets                                                                   |
| wm_task_2bk_body_acc_target              | cognition | Working Memory Task Accuracy for 2-back Body Targets                                                                      |
| wm_task_2bk_body_median_rt               | timing    | Working Memory Task Median Reaction Time for 2-back Body                                                                  |
| wm_task_2bk_body_median_rt_nontarget     | timing    | Working Memory Task Median Reaction Time for 2-back Body Nontargets                                                       |
| wm_task_2bk_body_median_rt_target        | timing    | Working Memory Task Median Reaction Time for 2-back Body Targets                                                          |
| wm_task_2bk_face_acc                     | cognition | Working Memory Task Accuracy for 2-back Face                                                                              |
| wm_task_2bk_face_acc_nontarget           | cognition | Working Memory Task Accuracy for 2-back Face Nontargets                                                                   |
| wm_task_2bk_face_acc_target              | cognition | Working Memory Task Accuracy for 2-back Face Targets                                                                      |
| wm_task_2bk_face_median_rt               | timing    | Working Memory Task Median Reaction Time for 2-back Face                                                                  |
| wm_task_2bk_face_median_rt_nontarget     | timing    | Working Memory Task Median Reaction Time for 2-back Face Nontargets                                                       |
| wm_task_2bk_face_median_rt_target        | timing    | Working Memory Task Median Reaction Time for 2-back Face Targets                                                          |
| wm_task_2bk_median_rt                    | timing    | Working Memory Task Median Reaction Time for 2-back                                                                       |
| wm_task_2bk_place_acc                    | cognition | Working Memory Task Accuracy for 2-back Place                                                                             |
| wm_task_2bk_place_acc_nontarget          | cognition | Working Memory Task Accuracy for 2-back Place Nontargets                                                                  |
| wm_task_2bk_place_acc_target             | cognition | Working Memory Task Accuracy for 2-back Place Targets                                                                     |
| wm_task_2bk_place_median_rt              | timing    | Working Memory Task Median Reaction Time for 2-back Place                                                                 |
| wm_task_2bk_place_median_rt_nontarget    | timing    | Working Memory Task Median Reaction Time for 2-back Place Nontargets                                                      |
| wm_task_2bk_place_median_rt_target       | timing    | Working Memory Task Median Reaction Time for 2-back Place Targets                                                         |
| wm_task_2bk_tool_acc                     | cognition | Working Memory Task Accuracy for 2-back Tool                                                                              |
| wm_task_2bk_tool_acc_nontarget           | cognition | Working Memory Task Accuracy for 2-back Tool Nontargets                                                                   |
| wm_task_2bk_tool_acc_target              | cognition | Working Memory Task Accuracy for 2-back Tool Targets                                                                      |
| wm_task_2bk_tool_median_rt               | timing    | Working Memory Task Median Reaction Time for 2-back Tool                                                                  |
| wm_task_2bk_tool_median_rt_nontarget     | timing    | Working Memory Task Median Reaction Time for 2-back Tool Nontargets                                                       |
| wm_task_2bk_tool_median_rt_target        | timing    | Working Memory Task Median Reaction Time for 2-back Tool Targets                                                          |
| wm_task_acc                              | cognition | Working Memory Task Overall Accuracy                                                                                      |
| wm_task_median_rt                        | timing    | Working Memory Task Overall Reaction Time                                                                                 |

## Cluster Components

gambling_perf

| x                                | y                                |        r |
|:---------------------------------|:---------------------------------|---------:|
| gambling_task_reward_perc_larger | gambling_task_perc_larger        | 0.892458 |
| gambling_task_perc_larger        | gambling_task_punish_perc_larger | 0.83386  |
| gambling_task_reward_perc_larger | gambling_task_punish_perc_larger | 0.495172 |

emotion_perf

| x                      | y                      |        r |
|:-----------------------|:-----------------------|---------:|
| emotion_task_acc       | emotion_task_shape_acc | 0.895004 |
| emotion_task_acc       | emotion_task_face_acc  | 0.829981 |
| emotion_task_shape_acc | emotion_task_face_acc  | 0.494029 |

language_rt

| x                             | y                             |        r |
|:------------------------------|:------------------------------|---------:|
| language_task_median_rt       | language_task_story_median_rt | 0.826075 |
| language_task_median_rt       | language_task_math_median_rt  | 0.813792 |
| language_task_story_median_rt | language_task_math_median_rt  | 0.344736 |

relational_rt

| x                             | y                               |        r |
|:------------------------------|:--------------------------------|---------:|
| relational_task_median_rt     | relational_task_rel_median_rt   | 0.950155 |
| relational_task_median_rt     | relational_task_match_median_rt | 0.861279 |
| relational_task_rel_median_rt | relational_task_match_median_rt | 0.659925 |

emotion_rt

| x                            | y                           |       r |
|:-----------------------------|:----------------------------|--------:|
| emotion_task_face_median_rt  | emotion_task_median_rt      | 0.95276 |
| emotion_task_shape_median_rt | emotion_task_median_rt      | 0.93737 |
| emotion_task_shape_median_rt | emotion_task_face_median_rt | 0.78729 |

language_perf

| x                                        | y                                        |        r |
|:-----------------------------------------|:-----------------------------------------|---------:|
| language_task_math_acc                   | language_task_acc                        | 0.807506 |
| language_task_math_acc                   | language_task_story_avg_difficulty_level | 0.732899 |
| language_task_story_avg_difficulty_level | language_task_acc                        | 0.603686 |

p_matrices

| x             | y           |         r |
|:--------------|:------------|----------:|
| pmat24_a_cr   | pmat24_a_si | -0.970141 |
| pmat24_a_rtcr | pmat24_a_cr |  0.7158   |
| pmat24_a_rtcr | pmat24_a_si | -0.700523 |

social_rt

| x                                   | y                             |        r |
|:------------------------------------|:------------------------------|---------:|
| social_task_random_median_rt_random | social_task_median_rt_random  | 0.937064 |
| social_task_median_rt_tom           | social_task_tom_median_rt_tom | 0.925799 |
| social_task_random_median_rt_random | social_task_tom_median_rt_tom | 0.382999 |
| social_task_tom_median_rt_tom       | social_task_median_rt_random  | 0.379269 |
| social_task_random_median_rt_random | social_task_median_rt_tom     | 0.371255 |
| social_task_median_rt_tom           | social_task_median_rt_random  | 0.370109 |

psqi_latent

| x          | y          |        r |
|:-----------|:-----------|---------:|
| psqi_score | psqi_comp1 | 0.682749 |
| psqi_score | psqi_comp4 | 0.635743 |
| psqi_score | psqi_comp3 | 0.62361  |
| psqi_score | psqi_comp7 | 0.464359 |
| psqi_comp3 | psqi_comp4 | 0.439622 |
| psqi_comp3 | psqi_comp1 | 0.395577 |
| psqi_comp1 | psqi_comp4 | 0.301622 |
| psqi_comp7 | psqi_comp1 | 0.276915 |
| psqi_comp3 | psqi_comp7 | 0.179938 |
| psqi_comp7 | psqi_comp4 | 0.11352  |

gambling_rt

| x                                      | y                                      |        r |
|:---------------------------------------|:---------------------------------------|---------:|
| gambling_task_median_rt_larger         | gambling_task_punish_median_rt_larger  | 0.95122  |
| gambling_task_reward_median_rt_smaller | gambling_task_median_rt_smaller        | 0.951192 |
| gambling_task_punish_median_rt_smaller | gambling_task_median_rt_smaller        | 0.950419 |
| gambling_task_median_rt_larger         | gambling_task_reward_median_rt_larger  | 0.949323 |
| gambling_task_median_rt_larger         | gambling_task_median_rt_smaller        | 0.884308 |
| gambling_task_median_rt_smaller        | gambling_task_reward_median_rt_larger  | 0.846051 |
| gambling_task_median_rt_larger         | gambling_task_reward_median_rt_smaller | 0.845245 |
| gambling_task_punish_median_rt_smaller | gambling_task_median_rt_larger         | 0.836065 |
| gambling_task_median_rt_smaller        | gambling_task_punish_median_rt_larger  | 0.835891 |
| gambling_task_reward_median_rt_smaller | gambling_task_reward_median_rt_larger  | 0.823178 |
| gambling_task_punish_median_rt_smaller | gambling_task_reward_median_rt_smaller | 0.807898 |
| gambling_task_punish_median_rt_larger  | gambling_task_reward_median_rt_larger  | 0.804724 |
| gambling_task_punish_median_rt_smaller | gambling_task_punish_median_rt_larger  | 0.804458 |
| gambling_task_reward_median_rt_smaller | gambling_task_punish_median_rt_larger  | 0.785076 |
| gambling_task_punish_median_rt_smaller | gambling_task_reward_median_rt_larger  | 0.78495  |

social_random_perf

| x                              | y                              |          r |
|:-------------------------------|:-------------------------------|-----------:|
| social_task_random_perc_random | social_task_perc_random        |  0.906435  |
| social_task_perc_unsure        | social_task_random_perc_unsure |  0.885781  |
| social_task_random_perc_unsure | social_task_random_perc_random | -0.80942   |
| social_task_tom_perc_tom       | social_task_tom_perc_unsure    | -0.745297  |
| social_task_random_perc_unsure | social_task_perc_random        | -0.74115   |
| social_task_tom_perc_tom       | social_task_perc_tom           |  0.721554  |
| social_task_perc_unsure        | social_task_random_perc_random | -0.720447  |
| social_task_tom_perc_tom       | social_task_tom_perc_random    | -0.700455  |
| social_task_perc_unsure        | social_task_perc_random        | -0.652429  |
| social_task_perc_tom           | social_task_random_perc_tom    |  0.646252  |
| social_task_random_perc_random | social_task_random_perc_tom    | -0.563319  |
| social_task_perc_tom           | social_task_tom_perc_unsure    | -0.557247  |
| social_task_perc_unsure        | social_task_tom_perc_unsure    |  0.546735  |
| social_task_perc_tom           | social_task_perc_random        | -0.502173  |
| social_task_random_perc_tom    | social_task_perc_random        | -0.500112  |
| social_task_perc_tom           | social_task_tom_perc_random    | -0.48457   |
| social_task_perc_unsure        | social_task_tom_perc_tom       | -0.426946  |
| social_task_tom_perc_random    | social_task_perc_random        |  0.337166  |
| social_task_perc_unsure        | social_task_perc_tom           | -0.327731  |
| social_task_random_perc_random | social_task_perc_tom           | -0.307918  |
| social_task_tom_perc_tom       | social_task_perc_random        | -0.18122   |
| social_task_random_perc_random | social_task_tom_perc_tom       |  0.107879  |
| social_task_random_perc_unsure | social_task_tom_perc_unsure    |  0.0955874 |
| social_task_random_perc_unsure | social_task_tom_perc_tom       | -0.094383  |
| social_task_random_perc_unsure | social_task_perc_tom           | -0.0867945 |
| social_task_random_perc_random | social_task_tom_perc_unsure    | -0.0847639 |
| social_task_random_perc_random | social_task_tom_perc_random    | -0.0708954 |
| social_task_random_perc_tom    | social_task_tom_perc_random    |  0.0656612 |
| social_task_tom_perc_unsure    | social_task_perc_random        | -0.061332  |
| social_task_perc_unsure        | social_task_tom_perc_random    |  0.0544257 |
| social_task_tom_perc_tom       | social_task_random_perc_tom    | -0.0508125 |
| social_task_tom_perc_unsure    | social_task_tom_perc_random    |  0.0462029 |
| social_task_random_perc_unsure | social_task_tom_perc_random    |  0.0390888 |
| social_task_random_perc_unsure | social_task_random_perc_tom    | -0.0292317 |
| social_task_perc_unsure        | social_task_random_perc_tom    | -0.0199713 |
| social_task_tom_perc_unsure    | social_task_random_perc_tom    |  0.0097798 |

int_g_like

| x                    | y                    |        r |
|:---------------------|:---------------------|---------:|
| readeng_unadj        | cogcrystalcomp_unadj | 0.92767  |
| cogcrystalcomp_unadj | picvocab_unadj       | 0.90933  |
| cogearlycomp_unadj   | cogtotalcomp_unadj   | 0.875269 |
| cogearlycomp_unadj   | cogfluidcomp_unadj   | 0.870489 |
| cogfluidcomp_unadj   | cogtotalcomp_unadj   | 0.850966 |
| cogcrystalcomp_unadj | cogtotalcomp_unadj   | 0.775179 |
| readeng_unadj        | cogtotalcomp_unadj   | 0.726282 |
| procspeed_unadj      | cogfluidcomp_unadj   | 0.723044 |
| cogearlycomp_unadj   | cardsort_unadj       | 0.699752 |
| picvocab_unadj       | cogtotalcomp_unadj   | 0.699515 |
| cogfluidcomp_unadj   | cardsort_unadj       | 0.684551 |
| readeng_unadj        | picvocab_unadj       | 0.677498 |
| cogearlycomp_unadj   | flanker_unadj        | 0.67435  |
| cogearlycomp_unadj   | picseq_unadj         | 0.64883  |
| flanker_unadj        | cogfluidcomp_unadj   | 0.629628 |
| cogfluidcomp_unadj   | picseq_unadj         | 0.591585 |
| cogtotalcomp_unadj   | listsort_unadj       | 0.574653 |
| procspeed_unadj      | cogtotalcomp_unadj   | 0.569278 |
| cardsort_unadj       | cogtotalcomp_unadj   | 0.569108 |
| cogfluidcomp_unadj   | listsort_unadj       | 0.554159 |
| cogcrystalcomp_unadj | cogearlycomp_unadj   | 0.544717 |
| picvocab_unadj       | cogearlycomp_unadj   | 0.540716 |
| flanker_unadj        | cogtotalcomp_unadj   | 0.519353 |
| flanker_unadj        | cardsort_unadj       | 0.511351 |
| cogtotalcomp_unadj   | picseq_unadj         | 0.496299 |
| readeng_unadj        | cogearlycomp_unadj   | 0.468685 |
| procspeed_unadj      | cogearlycomp_unadj   | 0.437023 |
| procspeed_unadj      | cardsort_unadj       | 0.420937 |
| cogearlycomp_unadj   | listsort_unadj       | 0.396311 |
| cogcrystalcomp_unadj | listsort_unadj       | 0.378531 |
| procspeed_unadj      | flanker_unadj        | 0.376212 |
| readeng_unadj        | listsort_unadj       | 0.346378 |
| cogcrystalcomp_unadj | cogfluidcomp_unadj   | 0.343073 |
| picvocab_unadj       | listsort_unadj       | 0.332631 |
| listsort_unadj       | picseq_unadj         | 0.327054 |
| readeng_unadj        | cogfluidcomp_unadj   | 0.325001 |
| picvocab_unadj       | cogfluidcomp_unadj   | 0.302961 |
| readeng_unadj        | cardsort_unadj       | 0.235433 |
| cogcrystalcomp_unadj | cardsort_unadj       | 0.219957 |
| cogcrystalcomp_unadj | flanker_unadj        | 0.191728 |
| cogcrystalcomp_unadj | picseq_unadj         | 0.191612 |
| cardsort_unadj       | picseq_unadj         | 0.188415 |
| readeng_unadj        | picseq_unadj         | 0.186397 |
| cardsort_unadj       | listsort_unadj       | 0.183183 |
| picvocab_unadj       | flanker_unadj        | 0.180242 |
| procspeed_unadj      | picseq_unadj         | 0.178994 |
| readeng_unadj        | flanker_unadj        | 0.168539 |
| picvocab_unadj       | picseq_unadj         | 0.16706  |
| procspeed_unadj      | cogcrystalcomp_unadj | 0.164599 |
| picvocab_unadj       | cardsort_unadj       | 0.164516 |
| procspeed_unadj      | listsort_unadj       | 0.16286  |
| readeng_unadj        | procspeed_unadj      | 0.144168 |
| procspeed_unadj      | picvocab_unadj       | 0.140717 |
| flanker_unadj        | listsort_unadj       | 0.131421 |
| flanker_unadj        | picseq_unadj         | 0.131359 |

neg_emotionality

| x                | y                |          r |
|:-----------------|:-----------------|-----------:|
| sadness_unadj    | fearaffect_unadj |  0.708565  |
| percstress_unadj | neofac_n         |  0.685489  |
| angaffect_unadj  | sadness_unadj    |  0.681294  |
| sadness_unadj    | percstress_unadj |  0.659242  |
| angaffect_unadj  | fearaffect_unadj |  0.638173  |
| fearaffect_unadj | percstress_unadj |  0.618427  |
| perchostil_unadj | percreject_unadj |  0.610784  |
| loneliness_unadj | percreject_unadj |  0.604949  |
| sadness_unadj    | neofac_n         |  0.603914  |
| loneliness_unadj | sadness_unadj    |  0.597592  |
| angaffect_unadj  | percstress_unadj |  0.583546  |
| lifesatisf_unadj | meanpurp_unadj   |  0.578924  |
| friendship_unadj | emotsupp_unadj   |  0.575482  |
| friendship_unadj | loneliness_unadj | -0.57328   |
| loneliness_unadj | neofac_n         |  0.560684  |
| fearaffect_unadj | neofac_n         |  0.554284  |
| lifesatisf_unadj | posaffect_unadj  |  0.549968  |
| anghostil_unadj  | neofac_n         |  0.547017  |
| loneliness_unadj | percstress_unadj |  0.541738  |
| emotsupp_unadj   | percreject_unadj | -0.53142   |
| anghostil_unadj  | percstress_unadj |  0.526901  |
| posaffect_unadj  | sadness_unadj    | -0.524555  |
| lifesatisf_unadj | percstress_unadj | -0.5229    |
| emotsupp_unadj   | loneliness_unadj | -0.520342  |
| posaffect_unadj  | meanpurp_unadj   |  0.519302  |
| emotsupp_unadj   | instrusupp_unadj |  0.516501  |
| percreject_unadj | percstress_unadj |  0.513986  |
| selfeff_unadj    | percstress_unadj | -0.508231  |
| loneliness_unadj | anghostil_unadj  |  0.494197  |
| lifesatisf_unadj | sadness_unadj    | -0.494128  |
| angaffect_unadj  | neofac_n         |  0.487259  |
| angaffect_unadj  | perchostil_unadj |  0.480146  |
| sadness_unadj    | percreject_unadj |  0.478808  |
| fearsomat_unadj  | fearaffect_unadj |  0.478754  |
| neofac_e         | friendship_unadj |  0.475446  |
| loneliness_unadj | posaffect_unadj  | -0.475239  |
| posaffect_unadj  | percstress_unadj | -0.468279  |
| selfeff_unadj    | neofac_n         | -0.466442  |
| friendship_unadj | percreject_unadj | -0.454428  |
| angaffect_unadj  | percreject_unadj |  0.451631  |
| loneliness_unadj | fearaffect_unadj |  0.450443  |
| lifesatisf_unadj | loneliness_unadj | -0.449908  |
| emotsupp_unadj   | posaffect_unadj  |  0.448497  |
| lifesatisf_unadj | neofac_n         | -0.447326  |
| angaffect_unadj  | loneliness_unadj |  0.444101  |
| loneliness_unadj | meanpurp_unadj   | -0.44268   |
| anghostil_unadj  | percreject_unadj |  0.44217   |
| emotsupp_unadj   | meanpurp_unadj   |  0.439628  |
| anghostil_unadj  | sadness_unadj    |  0.439269  |
| lifesatisf_unadj | emotsupp_unadj   |  0.437417  |
| perchostil_unadj | percstress_unadj |  0.433119  |
| angaffect_unadj  | anghostil_unadj  |  0.427924  |
| sadness_unadj    | meanpurp_unadj   | -0.417197  |
| meanpurp_unadj   | neofac_n         | -0.416712  |
| meanpurp_unadj   | percstress_unadj | -0.416103  |
| emotsupp_unadj   | percstress_unadj | -0.411955  |
| angaffect_unadj  | posaffect_unadj  | -0.40801   |
| percreject_unadj | neofac_n         |  0.407856  |
| friendship_unadj | posaffect_unadj  |  0.405275  |
| angaffect_unadj  | fearsomat_unadj  |  0.403185  |
| neofac_e         | loneliness_unadj | -0.399605  |
| lifesatisf_unadj | anghostil_unadj  | -0.398991  |
| neofac_e         | posaffect_unadj  |  0.397293  |
| selfeff_unadj    | meanpurp_unadj   |  0.396957  |
| fearaffect_unadj | percreject_unadj |  0.394938  |
| lifesatisf_unadj | percreject_unadj | -0.391956  |
| perchostil_unadj | loneliness_unadj |  0.39133   |
| neofac_e         | meanpurp_unadj   |  0.389913  |
| posaffect_unadj  | neofac_n         | -0.38937   |
| neofac_e         | emotsupp_unadj   |  0.389105  |
| selfeff_unadj    | posaffect_unadj  |  0.38605   |
| loneliness_unadj | instrusupp_unadj | -0.385451  |
| emotsupp_unadj   | anghostil_unadj  | -0.378884  |
| emotsupp_unadj   | sadness_unadj    | -0.378002  |
| anghostil_unadj  | meanpurp_unadj   | -0.375628  |
| perchostil_unadj | sadness_unadj    |  0.374764  |
| posaffect_unadj  | percreject_unadj | -0.374719  |
| anghostil_unadj  | neofac_a         | -0.374049  |
| posaffect_unadj  | fearaffect_unadj | -0.370354  |
| selfeff_unadj    | loneliness_unadj | -0.367448  |
| emotsupp_unadj   | perchostil_unadj | -0.36721   |
| friendship_unadj | percstress_unadj | -0.366222  |
| selfeff_unadj    | anghostil_unadj  | -0.365318  |
| perchostil_unadj | anghostil_unadj  |  0.365002  |
| percreject_unadj | instrusupp_unadj | -0.363389  |
| friendship_unadj | sadness_unadj    | -0.362656  |
| friendship_unadj | instrusupp_unadj |  0.360227  |
| lifesatisf_unadj | instrusupp_unadj |  0.359044  |
| friendship_unadj | selfeff_unadj    |  0.35786   |
| anghostil_unadj  | fearaffect_unadj |  0.356734  |
| friendship_unadj | anghostil_unadj  | -0.356719  |
| friendship_unadj | meanpurp_unadj   |  0.354782  |
| emotsupp_unadj   | neofac_n         | -0.35406   |
| lifesatisf_unadj | selfeff_unadj    |  0.353613  |
| selfeff_unadj    | sadness_unadj    | -0.353088  |
| lifesatisf_unadj | friendship_unadj |  0.352371  |
| neofac_e         | neofac_n         | -0.343228  |
| lifesatisf_unadj | fearaffect_unadj | -0.341952  |
| lifesatisf_unadj | angaffect_unadj  | -0.337325  |
| fearsomat_unadj  | sadness_unadj    |  0.335902  |
| friendship_unadj | neofac_n         | -0.334731  |
| meanpurp_unadj   | percreject_unadj | -0.334185  |
| angaffect_unadj  | emotsupp_unadj   | -0.333834  |
| selfeff_unadj    | fearaffect_unadj | -0.333277  |
| perchostil_unadj | fearaffect_unadj |  0.333177  |
| neofac_e         | selfeff_unadj    |  0.328754  |
| angaffect_unadj  | neofac_a         | -0.328094  |
| emotsupp_unadj   | selfeff_unadj    |  0.3266    |
| friendship_unadj | perchostil_unadj | -0.325287  |
| perchostil_unadj | neofac_a         | -0.314309  |
| selfeff_unadj    | percreject_unadj | -0.312679  |
| instrusupp_unadj | percstress_unadj | -0.310979  |
| posaffect_unadj  | instrusupp_unadj |  0.310135  |
| anghostil_unadj  | posaffect_unadj  | -0.307916  |
| angaffect_unadj  | meanpurp_unadj   | -0.306668  |
| emotsupp_unadj   | neofac_a         |  0.305513  |
| meanpurp_unadj   | instrusupp_unadj |  0.302079  |
| neofac_e         | lifesatisf_unadj |  0.301878  |
| fearsomat_unadj  | neofac_n         |  0.298749  |
| meanpurp_unadj   | fearaffect_unadj | -0.291336  |
| angaffect_unadj  | friendship_unadj | -0.290738  |
| perchostil_unadj | neofac_n         |  0.286448  |
| sadness_unadj    | instrusupp_unadj | -0.28551   |
| neofac_a         | neofac_n         | -0.284254  |
| percreject_unadj | neofac_a         | -0.283322  |
| neofac_e         | neofac_a         |  0.280772  |
| angaffect_unadj  | selfeff_unadj    | -0.280714  |
| neofac_e         | sadness_unadj    | -0.279706  |
| neofac_e         | percreject_unadj | -0.272049  |
| friendship_unadj | fearaffect_unadj | -0.271354  |
| fearsomat_unadj  | percstress_unadj |  0.268261  |
| anghostil_unadj  | instrusupp_unadj | -0.266061  |
| selfeff_unadj    | instrusupp_unadj |  0.257706  |
| emotsupp_unadj   | fearaffect_unadj | -0.255713  |
| perchostil_unadj | meanpurp_unadj   | -0.253672  |
| instrusupp_unadj | neofac_n         | -0.248324  |
| neofac_e         | anghostil_unadj  | -0.248009  |
| lifesatisf_unadj | perchostil_unadj | -0.247755  |
| lifesatisf_unadj | neofac_a         |  0.24715   |
| loneliness_unadj | neofac_a         | -0.245279  |
| neofac_a         | percstress_unadj | -0.243523  |
| fearsomat_unadj  | percreject_unadj |  0.24348   |
| selfeff_unadj    | perchostil_unadj | -0.243452  |
| meanpurp_unadj   | neofac_a         |  0.242587  |
| neofac_e         | percstress_unadj | -0.241596  |
| perchostil_unadj | fearsomat_unadj  |  0.240989  |
| perchostil_unadj | posaffect_unadj  | -0.239109  |
| posaffect_unadj  | neofac_a         |  0.223879  |
| friendship_unadj | neofac_a         |  0.223646  |
| loneliness_unadj | fearsomat_unadj  |  0.222309  |
| neofac_e         | angaffect_unadj  | -0.216987  |
| neofac_e         | instrusupp_unadj |  0.207232  |
| anghostil_unadj  | fearsomat_unadj  |  0.206054  |
| neofac_e         | fearaffect_unadj | -0.203228  |
| fearaffect_unadj | instrusupp_unadj | -0.201982  |
| perchostil_unadj | instrusupp_unadj | -0.200362  |
| neofac_a         | instrusupp_unadj |  0.184196  |
| fearsomat_unadj  | neofac_a         | -0.179395  |
| sadness_unadj    | neofac_a         | -0.176168  |
| angaffect_unadj  | instrusupp_unadj | -0.175094  |
| neofac_e         | perchostil_unadj | -0.17484   |
| selfeff_unadj    | fearsomat_unadj  | -0.171978  |
| posaffect_unadj  | fearsomat_unadj  | -0.150147  |
| friendship_unadj | fearsomat_unadj  | -0.133227  |
| fearaffect_unadj | neofac_a         | -0.123203  |
| emotsupp_unadj   | fearsomat_unadj  | -0.121994  |
| fearsomat_unadj  | meanpurp_unadj   | -0.115666  |
| lifesatisf_unadj | fearsomat_unadj  | -0.112415  |
| fearsomat_unadj  | instrusupp_unadj | -0.0983412 |
| neofac_e         | fearsomat_unadj  | -0.0911218 |
| selfeff_unadj    | neofac_a         |  0.0672093 |

wm_rt

| x                                     | y                                     |        r |
|:--------------------------------------|:--------------------------------------|---------:|
| wm_task_0bk_body_median_rt            | wm_task_0bk_body_median_rt_nontarget  | 0.985877 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_tool_median_rt            | 0.982711 |
| wm_task_0bk_place_median_rt           | wm_task_0bk_place_median_rt_nontarget | 0.977094 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_face_median_rt            | 0.976941 |
| wm_task_2bk_place_median_rt           | wm_task_2bk_place_median_rt_nontarget | 0.968949 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_face_median_rt            | 0.961339 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_tool_median_rt_nontarget  | 0.949634 |
| wm_task_2bk_body_median_rt            | wm_task_2bk_body_median_rt_nontarget  | 0.935166 |
| wm_task_0bk_median_rt                 | wm_task_median_rt                     | 0.90569  |
| wm_task_2bk_median_rt                 | wm_task_median_rt                     | 0.904318 |
| wm_task_0bk_median_rt                 | wm_task_0bk_place_median_rt           | 0.872332 |
| wm_task_0bk_median_rt                 | wm_task_0bk_tool_median_rt            | 0.866976 |
| wm_task_0bk_median_rt                 | wm_task_0bk_face_median_rt            | 0.865382 |
| wm_task_0bk_median_rt                 | wm_task_0bk_body_median_rt            | 0.863372 |
| wm_task_0bk_median_rt                 | wm_task_0bk_place_median_rt_nontarget | 0.857278 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_median_rt                 | 0.856945 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_median_rt                 | 0.854162 |
| wm_task_2bk_median_rt                 | wm_task_2bk_face_median_rt            | 0.849126 |
| wm_task_0bk_median_rt                 | wm_task_0bk_body_median_rt_nontarget  | 0.846949 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_median_rt                 | 0.845958 |
| wm_task_2bk_median_rt                 | wm_task_2bk_place_median_rt           | 0.829926 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_median_rt                 | 0.807834 |
| wm_task_2bk_median_rt                 | wm_task_2bk_tool_median_rt_nontarget  | 0.806152 |
| wm_task_0bk_body_median_rt            | wm_task_median_rt                     | 0.79702  |
| wm_task_2bk_median_rt                 | wm_task_2bk_place_median_rt_nontarget | 0.796814 |
| wm_task_median_rt                     | wm_task_0bk_face_median_rt            | 0.793896 |
| wm_task_0bk_tool_median_rt            | wm_task_median_rt                     | 0.793734 |
| wm_task_0bk_place_median_rt           | wm_task_median_rt                     | 0.793662 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_median_rt                     | 0.792741 |
| wm_task_median_rt                     | wm_task_2bk_face_median_rt            | 0.790935 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_median_rt                     | 0.788111 |
| wm_task_median_rt                     | wm_task_0bk_body_median_rt_nontarget  | 0.786406 |
| wm_task_median_rt                     | wm_task_0bk_place_median_rt_nontarget | 0.784593 |
| wm_task_2bk_median_rt                 | wm_task_2bk_body_median_rt            | 0.777963 |
| wm_task_2bk_place_median_rt           | wm_task_median_rt                     | 0.766887 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_median_rt                     | 0.759261 |
| wm_task_2bk_tool_median_rt            | wm_task_median_rt                     | 0.752196 |
| wm_task_median_rt                     | wm_task_2bk_place_median_rt_nontarget | 0.740206 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_place_median_rt           | 0.736747 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_place_median_rt_nontarget | 0.736686 |
| wm_task_0bk_place_median_rt           | wm_task_0bk_tool_median_rt            | 0.734751 |
| wm_task_0bk_tool_median_rt            | wm_task_0bk_place_median_rt_nontarget | 0.730481 |
| wm_task_2bk_median_rt                 | wm_task_2bk_body_median_rt_nontarget  | 0.728846 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_median_rt                     | 0.716622 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_body_median_rt            | 0.707392 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_place_median_rt_nontarget | 0.704679 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_place_median_rt           | 0.704345 |
| wm_task_0bk_body_median_rt            | wm_task_0bk_place_median_rt           | 0.70422  |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_body_median_rt_nontarget  | 0.702536 |
| wm_task_0bk_body_median_rt            | wm_task_0bk_place_median_rt_nontarget | 0.702114 |
| wm_task_0bk_body_median_rt_nontarget  | wm_task_0bk_place_median_rt_nontarget | 0.700963 |
| wm_task_0bk_body_median_rt            | wm_task_0bk_face_median_rt            | 0.700246 |
| wm_task_0bk_place_median_rt           | wm_task_0bk_face_median_rt            | 0.699645 |
| wm_task_0bk_place_median_rt           | wm_task_0bk_body_median_rt_nontarget  | 0.697357 |
| wm_task_0bk_face_median_rt            | wm_task_0bk_place_median_rt_nontarget | 0.695154 |
| wm_task_0bk_body_median_rt_nontarget  | wm_task_0bk_face_median_rt            | 0.693473 |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_face_median_rt            | 0.692446 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_body_median_rt_nontarget  | 0.692161 |
| wm_task_2bk_median_rt                 | wm_task_2bk_tool_median_rt_target     | 0.69207  |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_face_median_rt_nontarget  | 0.690726 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_tool_median_rt_target     | 0.690258 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_tool_median_rt            | 0.690189 |
| wm_task_0bk_tool_median_rt            | wm_task_0bk_face_median_rt            | 0.689796 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_body_median_rt            | 0.68894  |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_face_median_rt            | 0.688688 |
| wm_task_0bk_tool_median_rt            | wm_task_0bk_body_median_rt_nontarget  | 0.68832  |
| wm_task_0bk_body_median_rt            | wm_task_0bk_tool_median_rt            | 0.688285 |
| wm_task_2bk_place_median_rt           | wm_task_2bk_face_median_rt            | 0.675986 |
| wm_task_2bk_face_median_rt            | wm_task_2bk_place_median_rt_nontarget | 0.669606 |
| wm_task_2bk_body_median_rt            | wm_task_median_rt                     | 0.667503 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_place_median_rt           | 0.66639  |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_place_median_rt_nontarget | 0.665996 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_face_median_rt            | 0.663366 |
| wm_task_0bk_median_rt                 | wm_task_0bk_face_median_rt_target     | 0.66143  |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_2bk_face_median_rt            | 0.655786 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_tool_median_rt_nontarget  | 0.649625 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_place_median_rt           | 0.645847 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_2bk_place_median_rt           | 0.643667 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_face_median_rt_nontarget  | 0.643375 |
| wm_task_0bk_median_rt                 | wm_task_2bk_median_rt                 | 0.638441 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_2bk_place_median_rt_nontarget | 0.636491 |
| wm_task_2bk_tool_median_rt_target     | wm_task_median_rt                     | 0.635629 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_place_median_rt_nontarget | 0.633317 |
| wm_task_2bk_median_rt                 | wm_task_2bk_face_median_rt_target     | 0.626354 |
| wm_task_0bk_place_median_rt           | wm_task_0bk_place_median_rt_target    | 0.620007 |
| wm_task_median_rt                     | wm_task_2bk_body_median_rt_nontarget  | 0.61955  |
| wm_task_0bk_median_rt                 | wm_task_0bk_place_median_rt_target    | 0.604949 |
| wm_task_2bk_face_median_rt_target     | wm_task_2bk_face_median_rt            | 0.602709 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_face_median_rt_target     | 0.597219 |
| wm_task_0bk_face_median_rt_target     | wm_task_median_rt                     | 0.58389  |
| wm_task_0bk_median_rt                 | wm_task_2bk_face_median_rt            | 0.583214 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_body_median_rt            | 0.581075 |
| wm_task_2bk_median_rt                 | wm_task_0bk_body_median_rt            | 0.578766 |
| wm_task_2bk_median_rt                 | wm_task_0bk_body_median_rt_nontarget  | 0.576196 |
| wm_task_median_rt                     | wm_task_2bk_face_median_rt_target     | 0.572537 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_median_rt                 | 0.571827 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_median_rt                 | 0.570741 |
| wm_task_2bk_median_rt                 | wm_task_0bk_face_median_rt            | 0.570097 |
| wm_task_2bk_body_median_rt            | wm_task_2bk_face_median_rt            | 0.569772 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_median_rt                 | 0.567127 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_2bk_body_median_rt            | 0.565117 |
| wm_task_2bk_median_rt                 | wm_task_0bk_tool_median_rt            | 0.564255 |
| wm_task_2bk_place_median_rt           | wm_task_2bk_body_median_rt            | 0.562809 |
| wm_task_0bk_median_rt                 | wm_task_2bk_place_median_rt           | 0.560417 |
| wm_task_2bk_median_rt                 | wm_task_0bk_place_median_rt           | 0.559924 |
| wm_task_2bk_median_rt                 | wm_task_0bk_place_median_rt_nontarget | 0.558181 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_body_median_rt_nontarget  | 0.553216 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_body_median_rt            | 0.552521 |
| wm_task_2bk_face_median_rt            | wm_task_2bk_body_median_rt_nontarget  | 0.55122  |
| wm_task_2bk_place_median_rt           | wm_task_2bk_body_median_rt_nontarget  | 0.54936  |
| wm_task_2bk_body_median_rt            | wm_task_2bk_place_median_rt_nontarget | 0.54935  |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_2bk_body_median_rt_nontarget  | 0.54739  |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_body_median_rt_nontarget  | 0.54519  |
| wm_task_0bk_median_rt                 | wm_task_2bk_place_median_rt_nontarget | 0.544917 |
| wm_task_2bk_place_median_rt_nontarget | wm_task_2bk_body_median_rt_nontarget  | 0.540482 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_face_median_rt            | 0.538262 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_face_median_rt            | 0.535588 |
| wm_task_2bk_place_median_rt           | wm_task_0bk_place_median_rt_nontarget | 0.531252 |
| wm_task_median_rt                     | wm_task_0bk_place_median_rt_target    | 0.531149 |
| wm_task_2bk_face_median_rt            | wm_task_0bk_body_median_rt_nontarget  | 0.530963 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_place_median_rt           | 0.530319 |
| wm_task_0bk_place_median_rt           | wm_task_2bk_face_median_rt            | 0.52998  |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_tool_median_rt_nontarget  | 0.529417 |
| wm_task_2bk_place_median_rt_nontarget | wm_task_0bk_place_median_rt_nontarget | 0.529351 |
| wm_task_2bk_face_median_rt            | wm_task_0bk_place_median_rt_nontarget | 0.527851 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_body_median_rt            | 0.527846 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_place_median_rt_nontarget | 0.527048 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_body_median_rt_nontarget  | 0.525368 |
| wm_task_2bk_place_median_rt           | wm_task_0bk_place_median_rt           | 0.524735 |
| wm_task_0bk_place_median_rt_target    | wm_task_0bk_place_median_rt_nontarget | 0.524468 |
| wm_task_0bk_tool_median_rt            | wm_task_2bk_face_median_rt            | 0.524466 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_place_median_rt_nontarget | 0.52061  |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_face_median_rt            | 0.520603 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_face_median_rt_nontarget  | 0.519164 |
| wm_task_0bk_place_median_rt           | wm_task_2bk_place_median_rt_nontarget | 0.517943 |
| wm_task_2bk_place_median_rt           | wm_task_0bk_tool_median_rt            | 0.517859 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_face_median_rt_nontarget  | 0.517552 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_place_median_rt           | 0.517128 |
| wm_task_0bk_tool_median_rt            | wm_task_2bk_place_median_rt_nontarget | 0.513405 |
| wm_task_2bk_face_median_rt            | wm_task_0bk_face_median_rt            | 0.512551 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_median_rt                 | 0.51242  |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_face_median_rt            | 0.506377 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_tool_median_rt            | 0.504786 |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_face_median_rt            | 0.503118 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_place_median_rt           | 0.499307 |
| wm_task_2bk_place_median_rt           | wm_task_0bk_face_median_rt            | 0.49849  |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_body_median_rt_nontarget  | 0.496647 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_body_median_rt            | 0.496079 |
| wm_task_2bk_place_median_rt           | wm_task_0bk_body_median_rt_nontarget  | 0.493891 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_place_median_rt           | 0.491375 |
| wm_task_0bk_median_rt                 | wm_task_2bk_tool_median_rt_nontarget  | 0.48824  |
| wm_task_2bk_place_median_rt_nontarget | wm_task_0bk_body_median_rt_nontarget  | 0.48734  |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_body_median_rt            | 0.487139 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_place_median_rt_nontarget | 0.486116 |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_tool_median_rt            | 0.485086 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_place_median_rt_nontarget | 0.484227 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_0bk_body_median_rt_nontarget  | 0.48207  |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_face_median_rt_target     | 0.48198  |
| wm_task_2bk_place_median_rt_nontarget | wm_task_0bk_face_median_rt            | 0.480945 |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_place_median_rt           | 0.479686 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_tool_median_rt_nontarget  | 0.478404 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_face_median_rt_nontarget  | 0.473024 |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_body_median_rt_nontarget  | 0.472383 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_tool_median_rt_nontarget  | 0.466756 |
| wm_task_0bk_median_rt                 | wm_task_2bk_tool_median_rt_target     | 0.465092 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_face_median_rt            | 0.465057 |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_place_median_rt_nontarget | 0.464792 |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_face_median_rt_target     | 0.462538 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_tool_median_rt_target     | 0.460041 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_0bk_face_median_rt            | 0.454239 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_tool_median_rt_nontarget  | 0.44971  |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_2bk_face_median_rt_target     | 0.449521 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_place_median_rt           | 0.448684 |
| wm_task_0bk_tool_median_rt            | wm_task_0bk_place_median_rt_target    | 0.447682 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_place_median_rt_nontarget | 0.446424 |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_place_median_rt           | 0.444817 |
| wm_task_0bk_place_median_rt_target    | wm_task_0bk_face_median_rt            | 0.441223 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_0bk_place_median_rt_nontarget | 0.440999 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_0bk_place_median_rt_target    | 0.440715 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_tool_median_rt            | 0.440464 |
| wm_task_0bk_body_median_rt            | wm_task_0bk_place_median_rt_target    | 0.440071 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_0bk_place_median_rt           | 0.437429 |
| wm_task_0bk_median_rt                 | wm_task_2bk_body_median_rt            | 0.432205 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_0bk_place_median_rt_target    | 0.430244 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_tool_median_rt_nontarget  | 0.427735 |
| wm_task_0bk_body_median_rt_nontarget  | wm_task_0bk_place_median_rt_target    | 0.420869 |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_body_median_rt            | 0.41879  |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_place_median_rt_nontarget | 0.416397 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_0bk_tool_median_rt            | 0.415229 |
| wm_task_2bk_tool_median_rt_target     | wm_task_0bk_face_median_rt            | 0.411785 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_body_median_rt            | 0.411389 |
| wm_task_2bk_place_median_rt           | wm_task_2bk_face_median_rt_target     | 0.411243 |
| wm_task_0bk_median_rt                 | wm_task_2bk_face_median_rt_target     | 0.410135 |
| wm_task_2bk_body_median_rt            | wm_task_0bk_body_median_rt_nontarget  | 0.409491 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_body_median_rt            | 0.408283 |
| wm_task_2bk_tool_median_rt            | wm_task_2bk_face_median_rt_target     | 0.408178 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_tool_median_rt_target     | 0.40583  |
| wm_task_2bk_tool_median_rt_target     | wm_task_0bk_tool_median_rt            | 0.405067 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_tool_median_rt_target     | 0.401798 |
| wm_task_2bk_tool_median_rt_target     | wm_task_0bk_body_median_rt_nontarget  | 0.399711 |
| wm_task_2bk_body_median_rt            | wm_task_0bk_face_median_rt            | 0.399216 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_tool_median_rt_target     | 0.396825 |
| wm_task_0bk_median_rt                 | wm_task_2bk_body_median_rt_nontarget  | 0.394002 |
| wm_task_2bk_median_rt                 | wm_task_0bk_face_median_rt_target     | 0.392936 |
| wm_task_0bk_tool_median_rt            | wm_task_2bk_face_median_rt_target     | 0.381468 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_body_median_rt_nontarget  | 0.380874 |
| wm_task_2bk_body_median_rt            | wm_task_0bk_place_median_rt_nontarget | 0.377875 |
| wm_task_0bk_body_median_rt_nontarget  | wm_task_2bk_body_median_rt_nontarget  | 0.376937 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_body_median_rt            | 0.376781 |
| wm_task_2bk_tool_median_rt_target     | wm_task_0bk_place_median_rt           | 0.376202 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_face_median_rt_target     | 0.37614  |
| wm_task_2bk_tool_median_rt_target     | wm_task_2bk_body_median_rt_nontarget  | 0.375414 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_body_median_rt_nontarget  | 0.374369 |
| wm_task_0bk_place_median_rt           | wm_task_2bk_body_median_rt            | 0.374293 |
| wm_task_2bk_face_median_rt_target     | wm_task_2bk_place_median_rt_nontarget | 0.371707 |
| wm_task_2bk_body_median_rt            | wm_task_2bk_face_median_rt_target     | 0.368943 |
| wm_task_2bk_tool_median_rt_target     | wm_task_0bk_place_median_rt_nontarget | 0.368369 |
| wm_task_2bk_body_median_rt            | wm_task_0bk_tool_median_rt            | 0.365755 |
| wm_task_2bk_body_median_rt_nontarget  | wm_task_0bk_face_median_rt            | 0.364892 |
| wm_task_2bk_body_median_rt_nontarget  | wm_task_0bk_place_median_rt_nontarget | 0.356235 |
| wm_task_2bk_median_rt                 | wm_task_0bk_place_median_rt_target    | 0.356187 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_2bk_face_median_rt_target     | 0.352725 |
| wm_task_0bk_place_median_rt           | wm_task_2bk_face_median_rt_target     | 0.349403 |
| wm_task_0bk_place_median_rt           | wm_task_2bk_body_median_rt_nontarget  | 0.348571 |
| wm_task_2bk_face_median_rt_target     | wm_task_0bk_face_median_rt            | 0.344304 |
| wm_task_0bk_body_median_rt            | wm_task_2bk_face_median_rt_target     | 0.343368 |
| wm_task_0bk_tool_median_rt_nontarget  | wm_task_2bk_body_median_rt_nontarget  | 0.340268 |
| wm_task_2bk_face_median_rt_target     | wm_task_0bk_place_median_rt_nontarget | 0.337231 |
| wm_task_2bk_face_median_rt_target     | wm_task_0bk_body_median_rt_nontarget  | 0.33632  |
| wm_task_0bk_tool_median_rt            | wm_task_2bk_body_median_rt_nontarget  | 0.327823 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_place_median_rt           | 0.325951 |
| wm_task_2bk_face_median_rt_target     | wm_task_2bk_body_median_rt_nontarget  | 0.323823 |
| wm_task_0bk_face_median_rt_nontarget  | wm_task_2bk_face_median_rt_target     | 0.322434 |
| wm_task_2bk_face_median_rt            | wm_task_0bk_place_median_rt_target    | 0.314692 |
| wm_task_0bk_face_median_rt_target     | wm_task_0bk_place_median_rt_target    | 0.31438  |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_tool_median_rt_target     | 0.314267 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_face_median_rt            | 0.313705 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_face_median_rt_target     | 0.307534 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_face_median_rt_target     | 0.303629 |
| wm_task_2bk_tool_median_rt_target     | wm_task_0bk_place_median_rt_target    | 0.300392 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_place_median_rt_nontarget | 0.300246 |
| wm_task_2bk_tool_median_rt            | wm_task_0bk_place_median_rt_target    | 0.300067 |
| wm_task_2bk_place_median_rt           | wm_task_0bk_place_median_rt_target    | 0.295983 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_tool_median_rt_nontarget  | 0.290537 |
| wm_task_2bk_face_median_rt_nontarget  | wm_task_0bk_place_median_rt_target    | 0.290067 |
| wm_task_2bk_tool_median_rt_nontarget  | wm_task_0bk_place_median_rt_target    | 0.279534 |
| wm_task_2bk_place_median_rt_nontarget | wm_task_0bk_place_median_rt_target    | 0.277418 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_face_median_rt_target     | 0.276564 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_body_median_rt            | 0.244432 |
| wm_task_2bk_face_median_rt_target     | wm_task_0bk_place_median_rt_target    | 0.242406 |
| wm_task_2bk_body_median_rt            | wm_task_0bk_place_median_rt_target    | 0.227798 |
| wm_task_2bk_body_median_rt_nontarget  | wm_task_0bk_place_median_rt_target    | 0.213385 |
| wm_task_0bk_face_median_rt_target     | wm_task_2bk_body_median_rt_nontarget  | 0.208709 |

wm_perf

| x                               | y                               |         r |
|:--------------------------------|:--------------------------------|----------:|
| wm_task_0bk_face_acc_nontarget  | wm_task_0bk_face_acc            | 0.948258  |
| wm_task_0bk_body_acc            | wm_task_0bk_body_acc_nontarget  | 0.930513  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_0bk_tool_acc            | 0.929481  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_body_acc            | 0.914567  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_place_acc           | 0.894479  |
| wm_task_0bk_acc                 | wm_task_acc                     | 0.864271  |
| wm_task_2bk_tool_acc            | wm_task_2bk_tool_acc_nontarget  | 0.863909  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_face_acc            | 0.856447  |
| wm_task_0bk_place_acc           | wm_task_0bk_place_acc_target    | 0.852658  |
| wm_task_0bk_body_acc            | wm_task_0bk_body_acc_target     | 0.845626  |
| wm_task_2bk_acc                 | wm_task_acc                     | 0.828401  |
| wm_task_0bk_tool_acc            | wm_task_0bk_tool_acc_target     | 0.826341  |
| wm_task_2bk_acc                 | wm_task_2bk_tool_acc            | 0.795211  |
| wm_task_0bk_body_acc            | wm_task_0bk_acc                 | 0.787802  |
| wm_task_0bk_acc                 | wm_task_0bk_tool_acc            | 0.787571  |
| wm_task_2bk_acc                 | wm_task_2bk_body_acc            | 0.783427  |
| wm_task_2bk_place_acc           | wm_task_2bk_place_acc_target    | 0.769975  |
| wm_task_2bk_acc                 | wm_task_2bk_place_acc           | 0.765456  |
| wm_task_0bk_acc                 | wm_task_0bk_face_acc            | 0.764315  |
| wm_task_0bk_place_acc           | wm_task_0bk_acc                 | 0.76145   |
| wm_task_0bk_body_acc_target     | wm_task_0bk_acc                 | 0.740688  |
| wm_task_2bk_acc                 | wm_task_2bk_face_acc            | 0.735461  |
| wm_task_2bk_body_acc            | wm_task_2bk_body_acc_target     | 0.725113  |
| wm_task_2bk_tool_acc            | wm_task_2bk_tool_acc_target     | 0.71896   |
| wm_task_0bk_acc                 | wm_task_0bk_tool_acc_nontarget  | 0.712917  |
| wm_task_2bk_acc                 | wm_task_2bk_body_acc_target     | 0.70755   |
| wm_task_2bk_acc                 | wm_task_2bk_tool_acc_target     | 0.706035  |
| wm_task_0bk_body_acc            | wm_task_acc                     | 0.705185  |
| wm_task_acc                     | wm_task_0bk_tool_acc            | 0.702382  |
| wm_task_0bk_acc                 | wm_task_0bk_face_acc_nontarget  | 0.69863   |
| wm_task_0bk_face_acc_target     | wm_task_0bk_acc                 | 0.697092  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_acc                 | 0.683914  |
| wm_task_2bk_face_acc            | wm_task_2bk_face_acc_target     | 0.683256  |
| wm_task_acc                     | wm_task_2bk_body_acc            | 0.682528  |
| wm_task_0bk_acc                 | wm_task_0bk_place_acc_target    | 0.681577  |
| wm_task_0bk_acc                 | wm_task_0bk_tool_acc_target     | 0.679985  |
| wm_task_0bk_place_acc           | wm_task_acc                     | 0.666065  |
| wm_task_acc                     | wm_task_2bk_tool_acc            | 0.658645  |
| wm_task_acc                     | wm_task_2bk_place_acc           | 0.655638  |
| wm_task_2bk_face_acc            | wm_task_acc                     | 0.654023  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_acc                 | 0.653442  |
| wm_task_acc                     | wm_task_0bk_tool_acc_nontarget  | 0.64992   |
| wm_task_2bk_acc                 | wm_task_2bk_place_acc_target    | 0.649576  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_face_acc_nontarget  | 0.648227  |
| wm_task_0bk_body_acc_nontarget  | wm_task_acc                     | 0.647719  |
| wm_task_acc                     | wm_task_0bk_face_acc            | 0.646774  |
| wm_task_2bk_acc                 | wm_task_2bk_face_acc_target     | 0.639085  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_acc                 | 0.634581  |
| wm_task_acc                     | wm_task_0bk_face_acc_nontarget  | 0.624719  |
| wm_task_0bk_place_acc_nontarget | wm_task_acc                     | 0.61624   |
| wm_task_0bk_body_acc_target     | wm_task_acc                     | 0.610792  |
| wm_task_2bk_acc                 | wm_task_2bk_tool_acc_nontarget  | 0.589665  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_body_acc_nontarget  | 0.589383  |
| wm_task_acc                     | wm_task_0bk_tool_acc_target     | 0.584881  |
| wm_task_2bk_body_acc_nontarget  | wm_task_acc                     | 0.576763  |
| wm_task_acc                     | wm_task_2bk_body_acc_target     | 0.57571   |
| wm_task_2bk_tool_acc            | wm_task_2bk_body_acc            | 0.569161  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_0bk_tool_acc_target     | 0.560331  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_tool_acc_nontarget  | 0.545248  |
| wm_task_acc                     | wm_task_0bk_place_acc_target    | 0.544018  |
| wm_task_acc                     | wm_task_2bk_tool_acc_nontarget  | 0.536737  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_body_acc_nontarget  | 0.536077  |
| wm_task_0bk_face_acc_target     | wm_task_acc                     | 0.535374  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_place_acc_target    | 0.529083  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_tool_acc_nontarget  | 0.52646   |
| wm_task_0bk_body_acc            | wm_task_0bk_tool_acc_nontarget  | 0.522813  |
| wm_task_0bk_body_acc            | wm_task_0bk_tool_acc            | 0.518763  |
| wm_task_acc                     | wm_task_2bk_tool_acc_target     | 0.518075  |
| wm_task_0bk_body_acc            | wm_task_0bk_face_acc            | 0.51803   |
| wm_task_2bk_body_acc            | wm_task_2bk_tool_acc_nontarget  | 0.51783   |
| wm_task_2bk_face_acc            | wm_task_2bk_place_acc           | 0.515834  |
| wm_task_0bk_face_acc_nontarget  | wm_task_0bk_tool_acc_nontarget  | 0.514906  |
| wm_task_0bk_tool_acc            | wm_task_0bk_face_acc            | 0.513414  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_tool_acc_nontarget  | 0.511148  |
| wm_task_2bk_face_acc            | wm_task_2bk_tool_acc            | 0.508964  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_0bk_face_acc            | 0.507157  |
| wm_task_0bk_place_acc           | wm_task_0bk_tool_acc_nontarget  | 0.505132  |
| wm_task_0bk_body_acc            | wm_task_0bk_face_acc_nontarget  | 0.503393  |
| wm_task_0bk_body_acc            | wm_task_0bk_place_acc_nontarget | 0.503351  |
| wm_task_0bk_place_acc           | wm_task_0bk_tool_acc            | 0.503028  |
| wm_task_0bk_body_acc            | wm_task_0bk_place_acc           | 0.500711  |
| wm_task_0bk_face_acc_nontarget  | wm_task_0bk_tool_acc            | 0.497695  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_tool_acc            | 0.496108  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_face_acc_nontarget  | 0.495113  |
| wm_task_acc                     | wm_task_2bk_place_acc_target    | 0.489311  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_tool_acc            | 0.486675  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_tool_acc            | 0.486241  |
| wm_task_acc                     | wm_task_2bk_face_acc_target     | 0.485738  |
| wm_task_2bk_tool_acc            | wm_task_2bk_place_acc           | 0.485001  |
| wm_task_0bk_place_acc           | wm_task_0bk_face_acc            | 0.484033  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_face_acc_nontarget  | 0.483675  |
| wm_task_2bk_body_acc            | wm_task_2bk_place_acc           | 0.481836  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_place_acc           | 0.48048   |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_face_acc            | 0.479232  |
| wm_task_0bk_place_acc           | wm_task_0bk_face_acc_nontarget  | 0.476473  |
| wm_task_2bk_face_acc            | wm_task_2bk_body_acc            | 0.47544   |
| wm_task_2bk_tool_acc            | wm_task_2bk_body_acc_target     | 0.470362  |
| wm_task_2bk_tool_acc_target     | wm_task_2bk_body_acc_target     | 0.469015  |
| wm_task_2bk_tool_acc_target     | wm_task_2bk_place_acc_target    | 0.465108  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_face_acc            | 0.453944  |
| wm_task_2bk_tool_acc_target     | wm_task_2bk_face_acc_target     | 0.445793  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_face_acc            | 0.441476  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_face_acc            | 0.435537  |
| wm_task_2bk_tool_acc_target     | wm_task_2bk_place_acc           | 0.433069  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_tool_acc            | 0.432839  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_face_acc_target     | 0.43123   |
| wm_task_2bk_face_acc            | wm_task_2bk_tool_acc_nontarget  | 0.426376  |
| wm_task_0bk_body_acc            | wm_task_0bk_face_acc_target     | 0.423882  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_tool_acc            | 0.422076  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_place_acc           | 0.420631  |
| wm_task_2bk_acc                 | wm_task_0bk_acc                 | 0.410474  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_place_acc           | 0.401452  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_face_acc            | 0.397419  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_acc                 | 0.394982  |
| wm_task_2bk_face_acc_target     | wm_task_2bk_place_acc_target    | 0.393731  |
| wm_task_2bk_face_acc_target     | wm_task_2bk_body_acc_target     | 0.390065  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_face_acc            | 0.389744  |
| wm_task_2bk_tool_acc            | wm_task_2bk_face_acc_target     | 0.3887    |
| wm_task_0bk_place_acc_target    | wm_task_0bk_face_acc            | 0.388191  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_tool_acc_target     | 0.38727   |
| wm_task_0bk_face_acc_target     | wm_task_0bk_place_acc           | 0.38612   |
| wm_task_0bk_body_acc_target     | wm_task_0bk_face_acc_nontarget  | 0.385958  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_body_acc_target     | 0.384659  |
| wm_task_2bk_face_acc            | wm_task_2bk_tool_acc_target     | 0.384308  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_body_acc            | 0.383805  |
| wm_task_2bk_place_acc           | wm_task_2bk_body_acc_target     | 0.383508  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_tool_acc_nontarget  | 0.38293   |
| wm_task_0bk_body_acc_target     | wm_task_0bk_place_acc_target    | 0.381048  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_place_acc_target    | 0.379625  |
| wm_task_0bk_body_acc            | wm_task_2bk_acc                 | 0.379608  |
| wm_task_2bk_acc                 | wm_task_0bk_tool_acc            | 0.379345  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_tool_acc_nontarget  | 0.379086  |
| wm_task_0bk_face_acc            | wm_task_0bk_tool_acc_target     | 0.378532  |
| wm_task_2bk_face_acc            | wm_task_0bk_tool_acc_nontarget  | 0.376964  |
| wm_task_0bk_acc                 | wm_task_2bk_body_acc            | 0.376172  |
| wm_task_0bk_tool_acc            | wm_task_0bk_place_acc_target    | 0.374971  |
| wm_task_0bk_acc                 | wm_task_2bk_face_acc            | 0.374765  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_body_acc_nontarget  | 0.37449   |
| wm_task_2bk_body_acc            | wm_task_2bk_tool_acc_target     | 0.373146  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_acc                 | 0.372836  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_tool_acc_target     | 0.368971  |
| wm_task_2bk_acc                 | wm_task_0bk_tool_acc_nontarget  | 0.367885  |
| wm_task_2bk_tool_acc            | wm_task_2bk_place_acc_target    | 0.367553  |
| wm_task_0bk_body_acc            | wm_task_0bk_tool_acc_target     | 0.366639  |
| wm_task_2bk_face_acc            | wm_task_0bk_face_acc_nontarget  | 0.365279  |
| wm_task_0bk_body_acc            | wm_task_2bk_face_acc            | 0.364279  |
| wm_task_0bk_body_acc            | wm_task_0bk_place_acc_target    | 0.36211   |
| wm_task_0bk_body_acc            | wm_task_2bk_body_acc            | 0.359854  |
| wm_task_0bk_place_acc           | wm_task_0bk_tool_acc_target     | 0.358302  |
| wm_task_2bk_tool_acc_nontarget  | wm_task_2bk_place_acc           | 0.357859  |
| wm_task_2bk_face_acc            | wm_task_0bk_tool_acc            | 0.357669  |
| wm_task_2bk_face_acc_target     | wm_task_2bk_place_acc           | 0.35596   |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_body_acc            | 0.352973  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_place_acc           | 0.351436  |
| wm_task_0bk_place_acc_target    | wm_task_0bk_tool_acc_target     | 0.351055  |
| wm_task_0bk_acc                 | wm_task_2bk_place_acc           | 0.349253  |
| wm_task_2bk_body_acc_target     | wm_task_2bk_place_acc_target    | 0.346041  |
| wm_task_0bk_face_acc_target     | wm_task_0bk_body_acc_nontarget  | 0.344299  |
| wm_task_2bk_acc                 | wm_task_0bk_place_acc           | 0.344165  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_acc                 | 0.343628  |
| wm_task_2bk_face_acc            | wm_task_2bk_body_acc_target     | 0.343529  |
| wm_task_0bk_body_acc            | wm_task_2bk_body_acc_nontarget  | 0.34325   |
| wm_task_0bk_tool_acc            | wm_task_2bk_body_acc            | 0.341171  |
| wm_task_2bk_acc                 | wm_task_0bk_face_acc_nontarget  | 0.339382  |
| wm_task_0bk_face_acc_nontarget  | wm_task_0bk_place_acc_target    | 0.3391    |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_body_acc            | 0.337206  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_place_acc_nontarget | 0.336625  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_place_acc           | 0.336319  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_tool_acc_nontarget  | 0.335544  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_place_acc           | 0.331667  |
| wm_task_0bk_face_acc_nontarget  | wm_task_0bk_tool_acc_target     | 0.331406  |
| wm_task_0bk_tool_acc            | wm_task_2bk_tool_acc_nontarget  | 0.329126  |
| wm_task_0bk_body_acc            | wm_task_2bk_place_acc           | 0.328102  |
| wm_task_0bk_tool_acc            | wm_task_2bk_tool_acc            | 0.327703  |
| wm_task_2bk_body_acc            | wm_task_2bk_face_acc_target     | 0.326295  |
| wm_task_0bk_acc                 | wm_task_2bk_tool_acc            | 0.326236  |
| wm_task_0bk_body_acc_target     | wm_task_0bk_place_acc_nontarget | 0.325932  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_body_acc            | 0.325193  |
| wm_task_2bk_face_acc            | wm_task_2bk_place_acc_target    | 0.323828  |
| wm_task_0bk_place_acc           | wm_task_2bk_face_acc            | 0.323196  |
| wm_task_0bk_tool_acc            | wm_task_2bk_place_acc           | 0.321943  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_0bk_place_acc_target    | 0.321542  |
| wm_task_0bk_place_acc           | wm_task_2bk_body_acc            | 0.321076  |
| wm_task_0bk_acc                 | wm_task_2bk_tool_acc_nontarget  | 0.320602  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_tool_acc_nontarget  | 0.319851  |
| wm_task_2bk_face_acc            | wm_task_0bk_face_acc            | 0.318236  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_tool_acc            | 0.316792  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_tool_acc            | 0.314453  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_face_acc_nontarget  | 0.313667  |
| wm_task_0bk_place_acc           | wm_task_2bk_place_acc           | 0.31215   |
| wm_task_2bk_tool_acc_nontarget  | wm_task_2bk_body_acc_target     | 0.311537  |
| wm_task_2bk_acc                 | wm_task_0bk_face_acc            | 0.307638  |
| wm_task_2bk_body_acc            | wm_task_2bk_place_acc_target    | 0.306664  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_tool_acc_nontarget  | 0.303216  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_face_acc_target     | 0.302248  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_place_acc           | 0.294932  |
| wm_task_2bk_body_acc            | wm_task_0bk_face_acc            | 0.290464  |
| wm_task_2bk_acc                 | wm_task_0bk_tool_acc_target     | 0.290124  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_tool_acc            | 0.289171  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_tool_acc_target     | 0.288045  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_tool_acc_nontarget  | 0.285701  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_tool_acc            | 0.285591  |
| wm_task_0bk_body_acc_nontarget  | wm_task_0bk_place_acc_target    | 0.28547   |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_tool_acc_nontarget  | 0.283677  |
| wm_task_0bk_body_acc            | wm_task_2bk_tool_acc            | 0.282758  |
| wm_task_0bk_place_acc_nontarget | wm_task_0bk_tool_acc_target     | 0.281517  |
| wm_task_0bk_body_acc            | wm_task_2bk_tool_acc_nontarget  | 0.276089  |
| wm_task_0bk_acc                 | wm_task_2bk_body_acc_target     | 0.27346   |
| wm_task_0bk_place_acc           | wm_task_2bk_tool_acc_nontarget  | 0.273318  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_face_acc            | 0.273158  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_place_acc           | 0.272379  |
| wm_task_2bk_tool_acc_target     | wm_task_2bk_tool_acc_nontarget  | 0.271053  |
| wm_task_0bk_place_acc           | wm_task_2bk_tool_acc            | 0.268321  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_tool_acc            | 0.26538   |
| wm_task_0bk_body_acc_target     | wm_task_2bk_acc                 | 0.259422  |
| wm_task_2bk_place_acc           | wm_task_0bk_face_acc            | 0.254314  |
| wm_task_2bk_tool_acc            | wm_task_0bk_tool_acc_target     | 0.25217   |
| wm_task_2bk_body_acc            | wm_task_0bk_tool_acc_target     | 0.251245  |
| wm_task_2bk_tool_acc_nontarget  | wm_task_0bk_face_acc            | 0.247402  |
| wm_task_0bk_tool_acc            | wm_task_2bk_body_acc_target     | 0.24325   |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_body_acc_target     | 0.238329  |
| wm_task_0bk_body_acc            | wm_task_2bk_body_acc_target     | 0.236859  |
| wm_task_2bk_tool_acc            | wm_task_0bk_face_acc            | 0.236255  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_body_acc_target     | 0.232434  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_body_acc            | 0.232205  |
| wm_task_0bk_place_acc           | wm_task_2bk_body_acc_target     | 0.230623  |
| wm_task_2bk_face_acc            | wm_task_0bk_tool_acc_target     | 0.227589  |
| wm_task_2bk_tool_acc_nontarget  | wm_task_0bk_tool_acc_target     | 0.226737  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_body_acc_target     | 0.225011  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_tool_acc_target     | 0.224723  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_face_acc            | 0.222047  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_tool_acc_target     | 0.21775   |
| wm_task_2bk_acc                 | wm_task_0bk_place_acc_target    | 0.217515  |
| wm_task_2bk_place_acc           | wm_task_0bk_tool_acc_target     | 0.216526  |
| wm_task_2bk_tool_acc_nontarget  | wm_task_2bk_face_acc_target     | 0.215273  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_place_acc           | 0.209527  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_body_acc_nontarget  | 0.209208  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_face_acc_target     | 0.208289  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_body_acc_target     | 0.208121  |
| wm_task_2bk_body_acc_nontarget  | wm_task_2bk_place_acc_target    | 0.207832  |
| wm_task_2bk_body_acc_target     | wm_task_0bk_tool_acc_target     | 0.202664  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_tool_acc            | 0.200606  |
| wm_task_0bk_place_acc_target    | wm_task_2bk_place_acc           | 0.199425  |
| wm_task_2bk_body_acc_target     | wm_task_0bk_face_acc            | 0.197832  |
| wm_task_0bk_place_acc_target    | wm_task_2bk_body_acc            | 0.196903  |
| wm_task_0bk_acc                 | wm_task_2bk_face_acc_target     | 0.191218  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_tool_acc_nontarget  | 0.190973  |
| wm_task_0bk_acc                 | wm_task_2bk_place_acc_target    | 0.18719   |
| wm_task_0bk_face_acc_target     | wm_task_2bk_acc                 | 0.18598   |
| wm_task_0bk_acc                 | wm_task_2bk_tool_acc_target     | 0.181055  |
| wm_task_0bk_body_acc            | wm_task_2bk_place_acc_target    | 0.18026   |
| wm_task_0bk_place_acc_target    | wm_task_2bk_tool_acc            | 0.175519  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_body_acc_target     | 0.173753  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_face_acc_target     | 0.173157  |
| wm_task_0bk_tool_acc            | wm_task_2bk_face_acc_target     | 0.173027  |
| wm_task_0bk_tool_acc            | wm_task_2bk_tool_acc_target     | 0.172093  |
| wm_task_2bk_tool_acc_nontarget  | wm_task_2bk_place_acc_target    | 0.171991  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_face_acc            | 0.16929   |
| wm_task_2bk_tool_acc_target     | wm_task_0bk_tool_acc_target     | 0.169039  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_body_acc            | 0.167862  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_place_acc           | 0.167038  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_place_acc_target    | 0.166389  |
| wm_task_0bk_place_acc_target    | wm_task_2bk_body_acc_target     | 0.166089  |
| wm_task_2bk_face_acc_target     | wm_task_0bk_tool_acc_target     | 0.165121  |
| wm_task_0bk_body_acc            | wm_task_2bk_face_acc_target     | 0.164883  |
| wm_task_0bk_place_acc_target    | wm_task_2bk_tool_acc_nontarget  | 0.164409  |
| wm_task_0bk_place_acc           | wm_task_2bk_face_acc_target     | 0.162714  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_place_acc_target    | 0.162637  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_place_acc_target    | 0.159653  |
| wm_task_0bk_body_acc            | wm_task_2bk_tool_acc_target     | 0.159389  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_face_acc_target     | 0.159114  |
| wm_task_0bk_body_acc_nontarget  | wm_task_2bk_tool_acc_target     | 0.15838   |
| wm_task_2bk_face_acc            | wm_task_0bk_place_acc_target    | 0.157958  |
| wm_task_2bk_place_acc_target    | wm_task_0bk_tool_acc_target     | 0.149466  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_face_acc_target     | 0.148193  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_face_acc_target     | 0.146322  |
| wm_task_2bk_body_acc_nontarget  | wm_task_0bk_face_acc_target     | 0.145099  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_tool_acc_target     | 0.142383  |
| wm_task_0bk_tool_acc            | wm_task_2bk_place_acc_target    | 0.137587  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_body_acc_target     | 0.136059  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_place_acc_target    | 0.13572   |
| wm_task_0bk_place_acc           | wm_task_2bk_tool_acc_target     | 0.135621  |
| wm_task_2bk_place_acc_target    | wm_task_0bk_face_acc            | 0.135252  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_tool_acc            | 0.13511   |
| wm_task_0bk_face_acc_target     | wm_task_2bk_tool_acc_nontarget  | 0.132093  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_face_acc_target     | 0.13089   |
| wm_task_2bk_face_acc_target     | wm_task_0bk_face_acc            | 0.130693  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_tool_acc_target     | 0.127368  |
| wm_task_0bk_place_acc           | wm_task_2bk_place_acc_target    | 0.126729  |
| wm_task_0bk_body_acc_target     | wm_task_2bk_tool_acc_target     | 0.119846  |
| wm_task_0bk_place_acc_nontarget | wm_task_2bk_place_acc_target    | 0.119292  |
| wm_task_0bk_face_acc_nontarget  | wm_task_2bk_tool_acc_target     | 0.115705  |
| wm_task_2bk_tool_acc_target     | wm_task_0bk_face_acc            | 0.110103  |
| wm_task_0bk_place_acc_target    | wm_task_2bk_tool_acc_target     | 0.10856   |
| wm_task_0bk_place_acc_target    | wm_task_2bk_face_acc_target     | 0.106473  |
| wm_task_0bk_tool_acc_nontarget  | wm_task_2bk_place_acc_target    | 0.104454  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_place_acc_target    | 0.103697  |
| wm_task_0bk_place_acc_target    | wm_task_2bk_place_acc_target    | 0.101121  |
| wm_task_0bk_face_acc_target     | wm_task_2bk_tool_acc_target     | 0.0759269 |
| wm_task_0bk_face_acc_target     | wm_task_2bk_face_acc_target     | 0.0724839 |

## Latent Factor Loadings


gambling_perf

|                                  |   gambling_perf |
|:---------------------------------|----------------:|
| gambling_perf                    |           1.000 |
| gambling_task_perc_larger        |          -1.000 |
| gambling_task_reward_perc_larger |          -0.893 |
| gambling_task_punish_perc_larger |          -0.834 |

emotion_perf

|                        |   emotion_perf |
|:-----------------------|---------------:|
| emotion_perf           |          1.000 |
| emotion_task_acc       |         -1.000 |
| emotion_task_shape_acc |         -0.895 |
| emotion_task_face_acc  |         -0.830 |

language_rt

|                               |   language_rt |
|:------------------------------|--------------:|
| language_rt                   |         1.000 |
| language_task_median_rt       |         1.000 |
| language_task_story_median_rt |         0.826 |
| language_task_math_median_rt  |         0.814 |

relational_rt

|                                 |   relational_rt |
|:--------------------------------|----------------:|
| relational_rt                   |           1.000 |
| relational_task_median_rt       |          -1.000 |
| relational_task_rel_median_rt   |          -0.950 |
| relational_task_match_median_rt |          -0.861 |

emotion_rt

|                              |   emotion_rt |
|:-----------------------------|-------------:|
| emotion_rt                   |        1.000 |
| emotion_task_median_rt       |        1.000 |
| emotion_task_face_median_rt  |        0.953 |
| emotion_task_shape_median_rt |        0.937 |

language_perf

|                                          |   language_perf |
|:-----------------------------------------|----------------:|
| language_perf                            |           1.000 |
| language_task_math_acc                   |          -0.998 |
| language_task_acc                        |          -0.839 |
| language_task_story_avg_difficulty_level |          -0.761 |

p_matrices

|               |   p_matrices |
|:--------------|-------------:|
| pmat24_a_rtcr |        1.000 |
| p_matrices    |        1.000 |
| pmat24_a_cr   |        0.716 |
| pmat24_a_si   |       -0.701 |

social_rt

|                                     |   social_rt |
|:------------------------------------|------------:|
| social_rt                           |       1.000 |
| social_task_median_rt_random        |       0.984 |
| social_task_random_median_rt_random |       0.983 |
| social_task_tom_median_rt_tom       |       0.410 |
| social_task_median_rt_tom           |       0.402 |

psqi_latent

|             |   psqi_latent |
|:------------|--------------:|
| psqi_latent |         1.000 |
| psqi_score  |         1.000 |
| psqi_comp1  |         0.685 |
| psqi_comp4  |         0.638 |
| psqi_comp3  |         0.627 |
| psqi_comp7  |         0.465 |

gambling_rt

|                                        |   gambling_rt |
|:---------------------------------------|--------------:|
| gambling_rt                            |         1.000 |
| gambling_task_median_rt_smaller        |         1.000 |
| gambling_task_reward_median_rt_smaller |         0.951 |
| gambling_task_punish_median_rt_smaller |         0.950 |
| gambling_task_median_rt_larger         |         0.884 |
| gambling_task_reward_median_rt_larger  |         0.845 |
| gambling_task_punish_median_rt_larger  |         0.836 |

social_random_perf

|                                |   social_random_perf |
|:-------------------------------|---------------------:|
| social_random_perf             |                1.000 |
| social_task_random_perc_random |               -1.000 |
| social_task_perc_random        |               -0.910 |
| social_task_random_perc_unsure |                0.812 |
| social_task_perc_unsure        |                0.724 |
| social_task_random_perc_tom    |                0.559 |
| social_task_perc_tom           |                0.308 |
| social_task_tom_perc_tom       |               -0.105 |
| social_task_tom_perc_unsure    |                0.087 |
| social_task_tom_perc_random    |                0.064 |

int_g_like

|                      |   int_g_like |
|:---------------------|-------------:|
| int_g_like           |        1.000 |
| cogtotalcomp_unadj   |       -1.000 |
| cogearlycomp_unadj   |       -0.875 |
| cogfluidcomp_unadj   |       -0.850 |
| cogcrystalcomp_unadj |       -0.773 |
| readeng_unadj        |       -0.723 |
| picvocab_unadj       |       -0.677 |
| listsort_unadj       |       -0.570 |
| cardsort_unadj       |       -0.567 |
| procspeed_unadj      |       -0.564 |
| flanker_unadj        |       -0.512 |
| picseq_unadj         |       -0.494 |

neg_emotionality

|                  |   neg_emotionality |
|:-----------------|-------------------:|
| neg_emotionality |              1.000 |
| percstress_unadj |              0.825 |
| sadness_unadj    |              0.816 |
| loneliness_unadj |              0.781 |
| neofac_n         |              0.760 |
| angaffect_unadj  |              0.713 |
| fearaffect_unadj |              0.700 |
| percreject_unadj |              0.699 |
| lifesatisf_unadj |             -0.657 |
| posaffect_unadj  |             -0.654 |
| anghostil_unadj  |              0.652 |
| emotsupp_unadj   |             -0.635 |
| meanpurp_unadj   |             -0.608 |
| friendship_unadj |             -0.595 |
| selfeff_unadj    |             -0.559 |
| perchostil_unadj |              0.551 |
| neofac_e         |             -0.465 |
| instrusupp_unadj |             -0.458 |
| neofac_a         |             -0.378 |
| fearsomat_unadj  |              0.375 |

wm_rt

|                                       |   wm_rt |
|:--------------------------------------|--------:|
| wm_rt                                 |   1.000 |
| wm_task_median_rt                     |   1.000 |
| wm_task_0bk_median_rt                 |   0.906 |
| wm_task_2bk_median_rt                 |   0.900 |
| wm_task_0bk_face_median_rt            |   0.793 |
| wm_task_0bk_tool_median_rt            |   0.792 |
| wm_task_0bk_body_median_rt            |   0.791 |
| wm_task_0bk_place_median_rt           |   0.791 |
| wm_task_0bk_tool_median_rt_nontarget  |   0.791 |
| wm_task_2bk_face_median_rt            |   0.789 |
| wm_task_0bk_face_median_rt_nontarget  |   0.785 |
| wm_task_0bk_place_median_rt_nontarget |   0.783 |
| wm_task_0bk_body_median_rt_nontarget  |   0.782 |
| wm_task_2bk_place_median_rt           |   0.764 |
| wm_task_2bk_face_median_rt_nontarget  |   0.757 |
| wm_task_2bk_tool_median_rt            |   0.741 |
| wm_task_2bk_place_median_rt_nontarget |   0.737 |
| wm_task_2bk_tool_median_rt_nontarget  |   0.708 |
| wm_task_2bk_body_median_rt            |   0.663 |
| wm_task_2bk_tool_median_rt_target     |   0.631 |
| wm_task_2bk_body_median_rt_nontarget  |   0.616 |
| wm_task_0bk_face_median_rt_target     |   0.583 |
| wm_task_2bk_face_median_rt_target     |   0.570 |
| wm_task_0bk_place_median_rt_target    |   0.530 |

wm_perf

|                                 |   wm_perf |
|:--------------------------------|----------:|
| wm_perf                         |     1.000 |
| wm_task_acc                     |    -0.993 |
| wm_task_0bk_acc                 |    -0.882 |
| wm_task_2bk_acc                 |    -0.798 |
| wm_task_0bk_body_acc            |    -0.736 |
| wm_task_0bk_tool_acc            |    -0.727 |
| wm_task_0bk_place_acc           |    -0.688 |
| wm_task_0bk_tool_acc_nontarget  |    -0.683 |
| wm_task_0bk_body_acc_nontarget  |    -0.680 |
| wm_task_0bk_face_acc            |    -0.673 |
| wm_task_2bk_body_acc            |    -0.671 |
| wm_task_0bk_face_acc_nontarget  |    -0.654 |
| wm_task_0bk_place_acc_nontarget |    -0.647 |
| wm_task_2bk_face_acc            |    -0.642 |
| wm_task_2bk_tool_acc            |    -0.639 |
| wm_task_2bk_place_acc           |    -0.633 |
| wm_task_0bk_body_acc_target     |    -0.626 |
| wm_task_0bk_tool_acc_target     |    -0.590 |
| wm_task_2bk_body_acc_nontarget  |    -0.578 |
| wm_task_0bk_face_acc_target     |    -0.550 |
| wm_task_0bk_place_acc_target    |    -0.550 |
| wm_task_2bk_body_acc_target     |    -0.548 |
| wm_task_2bk_tool_acc_nontarget  |    -0.538 |
| wm_task_2bk_tool_acc_target     |    -0.479 |
| wm_task_2bk_face_acc_target     |    -0.454 |
| wm_task_2bk_place_acc_target    |    -0.452 |


## Latent Factor Correlations

| x                  | y                  |      r |
|:-------------------|:-------------------|-------:|
| wm_perf            | int_g_like         |  0.530 |
| int_g_like         | language_perf      |  0.468 |
| wm_perf            | wm_rt              |  0.450 |
| wm_rt              | emotion_rt         |  0.429 |
| wm_perf            | language_perf      |  0.398 |
| wm_rt              | gambling_rt        |  0.360 |
| int_g_like         | emotion_rt         |  0.349 |
| neg_emotionality   | psqi_score         |  0.321 |
| emotion_rt         | relational_rt      | -0.315 |
| wm_rt              | int_g_like         |  0.303 |
| wm_rt              | language_rt        |  0.300 |
| wm_rt              | relational_rt      | -0.294 |
| social_rt          | emotion_rt         |  0.278 |
| wm_perf            | emotion_rt         |  0.269 |
| social_rt          | language_rt        |  0.257 |
| gambling_rt        | relational_rt      | -0.244 |
| wm_rt              | social_rt          |  0.240 |
| social_random_perf | social_rt          |  0.240 |
| gambling_rt        | emotion_rt         |  0.239 |
| emotion_rt         | language_rt        |  0.236 |
| language_perf      | emotion_rt         |  0.210 |
| language_perf      | language_rt        |  0.197 |
| relational_rt      | language_rt        | -0.197 |
| gambling_rt        | language_rt        |  0.193 |
| int_g_like         | language_rt        |  0.191 |
| wm_perf            | social_random_perf |  0.186 |
| int_g_like         | p_matrices         | -0.186 |
| wm_perf            | language_rt        |  0.181 |
| wm_rt              | language_perf      |  0.172 |
| gambling_rt        | social_rt          |  0.172 |
| p_matrices         | language_perf      | -0.170 |
| social_random_perf | emotion_perf       |  0.166 |
| int_g_like         | relational_rt      | -0.157 |
| social_rt          | relational_rt      | -0.151 |
| int_g_like         | gambling_rt        |  0.150 |
| int_g_like         | social_random_perf |  0.149 |
| int_g_like         | emotion_perf       |  0.145 |
| wm_perf            | p_matrices         | -0.140 |
| wm_perf            | gambling_rt        |  0.137 |
| wm_perf            | emotion_perf       |  0.137 |
| wm_perf            | relational_rt      | -0.133 |
| int_g_like         | social_rt          |  0.131 |
| language_perf      | emotion_perf       |  0.131 |
| wm_perf            | social_rt          |  0.122 |
| social_random_perf | emotion_rt         |  0.120 |
| wm_rt              | neg_emotionality   |  0.118 |
| p_matrices         | relational_rt      | -0.117 |
| wm_perf            | psqi_score         |  0.110 |
| int_g_like         | psqi_score         |  0.110 |
| social_rt          | language_perf      |  0.107 |
| wm_perf            | neg_emotionality   |  0.103 |
| gambling_rt        | language_perf      |  0.101 |
| wm_rt              | social_random_perf |  0.099 |
| neg_emotionality   | language_perf      |  0.097 |
| psqi_score         | emotion_perf       |  0.088 |
| psqi_score         | language_perf      |  0.088 |
| neg_emotionality   | social_random_perf |  0.084 |
| neg_emotionality   | int_g_like         |  0.083 |
| psqi_score         | p_matrices         | -0.082 |
| neg_emotionality   | emotion_rt         |  0.079 |
| psqi_score         | language_rt        |  0.077 |
| neg_emotionality   | gambling_rt        |  0.077 |
| wm_rt              | p_matrices         |  0.072 |
| language_perf      | relational_rt      | -0.071 |
| social_random_perf | language_perf      |  0.069 |
| p_matrices         | emotion_perf       | -0.069 |
| gambling_rt        | gambling_perf      | -0.066 |
| emotion_rt         | emotion_perf       |  0.066 |
| neg_emotionality   | language_rt        |  0.065 |
| social_random_perf | language_rt        |  0.063 |
| wm_rt              | psqi_score         |  0.061 |
| p_matrices         | gambling_perf      |  0.058 |
| social_rt          | p_matrices         |  0.055 |
| psqi_score         | relational_rt      |  0.050 |
| gambling_rt        | emotion_perf       | -0.050 |
| social_random_perf | psqi_score         |  0.049 |
| social_rt          | gambling_perf      |  0.048 |
| language_perf      | gambling_perf      | -0.041 |
| relational_rt      | emotion_perf       |  0.037 |
| neg_emotionality   | gambling_perf      | -0.035 |
| psqi_score         | gambling_perf      | -0.034 |
| emotion_perf       | gambling_perf      | -0.032 |
| language_rt        | gambling_perf      | -0.031 |
| social_rt          | emotion_perf       |  0.030 |
| wm_rt              | gambling_perf      |  0.030 |
| gambling_rt        | psqi_score         |  0.026 |
| neg_emotionality   | relational_rt      | -0.023 |
| int_g_like         | gambling_perf      | -0.021 |
| social_random_perf | gambling_rt        |  0.021 |
| social_random_perf | relational_rt      |  0.020 |
| social_random_perf | p_matrices         | -0.018 |
| wm_perf            | gambling_perf      |  0.016 |
| psqi_score         | emotion_rt         |  0.016 |
| social_random_perf | gambling_perf      |  0.013 |
| gambling_rt        | p_matrices         |  0.012 |
| neg_emotionality   | emotion_perf       | -0.011 |
| p_matrices         | emotion_rt         |  0.009 |
| p_matrices         | language_rt        |  0.009 |
| language_rt        | emotion_perf       |  0.008 |
| emotion_rt         | gambling_perf      |  0.008 |
| relational_rt      | gambling_perf      | -0.007 |
| neg_emotionality   | p_matrices         | -0.007 |
| psqi_score         | social_rt          |  0.006 |
| neg_emotionality   | social_rt          |  0.003 |
| wm_rt              | emotion_perf       |  0.000 |


# References
