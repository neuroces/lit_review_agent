# Health AI Model Benchmarking with Wearable Data: A Structured Literature Review

---

## 1. Introduction

Wearable devices—ranging from consumer smartwatches and fitness bands to clinical-grade patches and inertial measurement units (IMUs)—generate continuous, high-resolution physiological and kinematic data streams that hold significant promise for health AI applications. Tasks including atrial fibrillation (AFib) detection, sleep staging, human activity recognition (HAR), stress detection, gait analysis, and chronic disease monitoring have all attracted sustained research interest. The quality of AI model benchmarking in this space directly determines whether deployed systems can be trusted in clinical and consumer contexts.

Yet the field suffers from well-documented methodological fragmentation: heterogeneous devices, inconsistent evaluation protocols, opaque split strategies, small and demographically narrow cohorts, and a persistent gap between controlled-laboratory validation and real-world deployment. This review synthesizes the current state of AI model benchmarking with wearable data, assessing methodological rigor, cross-cutting themes, and open gaps to inform future research priorities.

---

## 2. Methods

This is an AI-assisted literature review conducted using an automated multi-agent pipeline. Papers were identified through systematic searches across **PubMed** and **Semantic Scholar**, using queries targeting health AI, wearable sensors, digital health benchmarking, and related clinical tasks. A critic loop was applied to iteratively assess corpus coverage, resolving ambiguities in metadata extraction and flagging papers with incomplete or inconsistent reporting.

- **Total papers included:** 63  
- **Year range:** 2019–2026  
- **Primary sources:** PubMed (majority), Semantic Scholar  
- **Paper types:** Original empirical studies, device validation studies, protocol papers, systematic reviews, and narrative reviews  

Metadata extracted per paper included: task, device(s), sensor modalities, cohort size and demographics, reference standard, split strategy, evaluation metrics, key findings, and methodological quality notes. This review is explicitly AI-assisted; human expert oversight has been applied to synthesis and interpretation. All findings are drawn exclusively from the provided corpus metadata.

---

## 3. Corpus Summary

| Attribute | Detail |
|---|---|
| Total papers | 63 |
| Year range | 2019–2026 (majority 2024–2026) |
| Sources | PubMed (dominant), Semantic Scholar |
| Primary empirical studies | ~38 |
| Review articles (narrative/systematic) | ~25 |
| **Task distribution** | |
| Human Activity Recognition (HAR) | 17 |
| AFib detection | 8 |
| Sleep staging | 6 |
| Stress/anxiety detection | 3 |
| Gait / balance / motor function | 6 |
| Cardiovascular disease / risk assessment | 4 |
| Blood glucose / hyperglycemia | 2 |
| Injury prediction | 2 |
| Signal quality / vital sign monitoring | 5 |
| Other (QTc, potassium, VO₂, seizure, etc.) | 10 |

---

## 4. Methodology Comparison Table

| Paper | Year | Task | Device(s) | Sensor | Cohort N | Reference Standard | Split Strategy | Metrics | Key Finding |
|---|---|---|---|---|---|---|---|---|---|
| Tsenos et al. [1] | 2026 | Injury prediction | GPS, IMU, wearable sensors | GPS, accelerometer, gyroscope | Unknown | Not specified | External validation (rare in reviewed studies) | Not specified | AI injury prediction limited by single-team datasets and rare external validation |
| Li & Kang [2] | 2026 | HAR | Wearable IMU | Accelerometer, gyroscope | 8 | Not specified | Segment 5-fold CV + LOSO | Accuracy, F1 | 99.89% segment-level vs. 89.34% LOSO — 10.55% gap highlights segment-level inflation |
| Orfanos et al. [3] | 2026 | Gait abnormality classification | Wearable gyroscope | Gyroscope | Unknown | Not specified | LOSO | Accuracy, AUC | RF/SVM AUC = 0.98 for post-stroke vs. healthy gait |
| Rajashekaraiah et al. [4] | 2026 | VO₂ estimation | Calf-mounted IMU | Accelerometer, gyroscope | 24 | Laboratory metabolic testing | LOSO | F1, R² | Median R² = 0.687; LOSO + SHAP; lightweight ensemble viable for wearable deployment |
| Marciniak & Zubert [5] | 2026 | Gait impairment detection | Smartphone | Accelerometer, gyroscope | 16 | Impairment simulation goggles | Within-subject 80/20 split | Accuracy, F1 | 71.4% accuracy; within-subject protocol inflates estimates; generalisation unestablished |
| Wojcik et al. [6] | 2026 | Obesity risk stratification | Not specified | Not specified | Unknown | Not specified | External validation (noted as needed) | Not specified | AI obesity tools require external validation and bias auditing before deployment |
| Palermi et al. [7] | 2026 | Sports medicine AI (multi-task) | Wearable sensors | Not specified | Unknown | Not specified | Not specified | Not specified | AI shows promise but limited by bias, poor generalisability, and low interpretability |
| Thavasi et al. [8] | 2026 | Breath diagnostics | Wearable condensate/humidity sensors | Mass spectrometry, chemical sensors | Unknown | Not specified | Not specified | Not specified | Real-time breath diagnostics promising but vulnerable to drift and lack of standardisation |
| Oloko-Oba et al. [9] | 2026 | Multi-task wearable health monitoring | Smartwatch, wearables | ECG, PPG, accelerometer, EDA | Unknown | Not specified | Not specified | Latency, energy/inference | End-to-end edge ML deployment pathway for wearables synthesised; clinical adoption uneven |
| Luca et al. [10] | 2026 | Rehabilitation monitoring | Wearable monitoring/robotic devices | Not specified | Unknown | Not specified | Not specified | Not specified | AI technically mature at session level; longitudinal orchestration at emerging stage |
| Ha et al. [11] | 2026 | Hyperglycemia detection | ECG Holter, wearable ECG | ECG | 30 | Interstitial glucose monitoring | Temporal split | Accuracy, AUROC | ~89% accuracy / ~0.89 AUROC; individual delay correction improves HRV-glucose correlation |
| Ullah et al. [12] | 2026 | Keystroke recognition (sEMG) | Wearable sEMG, edge device | sEMG | 19 | Labeled keystroke annotations | Unclear (participant-level not confirmed) | Accuracy, AUC | 96.53% accuracy, AUC >0.994; split strategy unclear — data leakage risk |
| Feng & Ma [13] | 2026 | Sports injury prediction/warning | 9-axis IMU, strain, pressure sensors | Accelerometer, gyroscope, magnetometer, strain, pressure | Unknown | Not specified | Not specified | False alarm rate, sensitivity, specificity | Multimodal system detects knee valgus risk 400 ms pre-peak; no split strategy reported |
| Wang et al. [14] | 2026 | Blood glucose prediction | Not specified | CGM, PPG | Unknown | CGM data | Not specified | MAE, RMSE, R² | U-Net CNN: MAE 8.48 mg/dL (30 min), 14.02 mg/dL (60 min) on OhioT1DM |
| Jernsletten et al. [15] | 2026 | Presymptomatic illness detection | Consumer wearables | PPG, accelerometer | Unknown | Not specified | Not specified | Sensitivity, false-positive rate | KDE model detects sick users with high sensitivity but high false-positive rate |
| Choi et al. [16] | 2026 | Alcohol detection | CMOS camera, LED | IPPG | Unknown (mice) | Not specified | Not specified | Frequency/magnitude ratios | Non-contact IPPG detects alcohol in mice; not applicable to human AI benchmarking |
| Zhou & Wang [17] | 2026 | Cardiotoxicity detection | Wearable devices | Not specified | Unknown | Not specified | Not specified | Accuracy | AI predicts LVEF decline with moderate accuracy; limited by small cohorts and no external validation |
| Zuern et al. [18] | 2026 | HRV estimation | Bora band (PPG wristband) | PPG, ECG | 66 | 12-lead ECG | Not specified | Biweight mid-correlation, Cliff's delta, TOST, Bland-Altman | Strong agreement for mean HR/SDNN; weaker for entropy metrics under controlled conditions |
| Zhang & Yuan [19] | 2026 | AFib detection | Wearable devices | PPG | 91 (30,773 segments) | Not specified | k-fold CV + external (MIMIC-III) | AUROC, accuracy | Transformer-VAE: AUC 0.9097, 88.4% accuracy; external validation on MIMIC-III is strength |
| Al Fahoum [20] | 2026 | CVD classification | Consumer wearables | PPG | 612 | Annotated PPG segments | Not specified | Accuracy, F1, Cohen's Kappa | Deep CNN: 93.48% accuracy; split strategy absent — data leakage risk |
| Lee & Kim [21] | 2026 | Neurodevelopmental disorder monitoring | IMU, EEG, smart glasses, smartwatch | Accelerometer, gyroscope, PPG, EDA, EEG | Unknown | Not specified | Not specified | Accuracy, sensitivity | ADHD detection 80–90%, ASD 96%, seizure 89–94% sensitivity; standardised validation lacking |
| Wang et al. [22] | 2026 | CVD risk assessment | Not specified | PPG | 144 | Clinical CVD diagnosis | Not specified | Proportions, p-values | PPG-derived SI, SWR, HRV-TP differentiate CVD vs. healthy; no classifier/AUC reported |
| Weber et al. [23] | 2026 | AFib detection | Smartphone apps, smartwatches, Apple Watch | PPG, ECG | Unknown | Not specified | External validation across studies | Sensitivity, specificity, PPV, AUC | Systematic review: top tools sensitivity/specificity ≥94%, AUC ≥0.95; consumer devices show lower specificity |
| Lee et al. [24] | 2026 | HR and energy expenditure estimation | Apple Watch, Galaxy, Fitbit, Garmin | PPG | 62 | ECG (HR), indirect calorimetry (EE) | Not specified | ICC, Pearson r, Bland-Altman, ANOVA | All smartwatches accurate for HR; EE consistently inaccurate, especially resistance exercise |
| Gonzalez et al. [25] | 2026 | Hemorrhage detection / CRM monitoring | Epicore EPIC patch, finger pulse oximeter | PPG | 20 | Clinical-grade finger oximeter | Not specified | Signal quality, HR accuracy, CRM accuracy | Triceps placement adequate for HR; CRM prediction magnitudes inaccurate |
| Nugroho et al. [26] | 2026 | Chronic stress detection | Wearable multimodal platforms | EEG, PPG, EDA, HRV | Unknown | Not specified | Not specified | Not specified | Classical ML (SVM, ensembles) dominate; small datasets limit DL adoption; stress labeling ambiguous |
| Ha et al. [27] | 2026 | Nocturnal BP monitoring in OSA | PPG ring device | PPG, SpO₂ | 62 | Polysomnography | Not specified | Not specified | Consistent BP dip/rebound pattern during OSA events; characterisation study, not ML classifier |
| Han et al. [28] | 2026 | HRV/PRV estimation reliability | Not specified | PPG | 914 | Not specified | External validation (3 independent cohorts) | ICC | 72.1% of PRV metrics excellent reliability (ICC ≥0.75) across sessions and lifespan |
| Van Slambrouck et al. [29] | 2026 | Vital sign monitoring (ICU) | Biobeat chest patch | PPG | 32 | Standard ICU bedside monitoring | Not specified | Bias, LoA, Bland-Altman, Pearson r, Clarke Error Grid | Variable agreement; 61.4% of RR measurements exceeded 10% threshold |
| Woelk et al. [30] | 2026 | Heart rate estimation | Webcam (rPPG) | rPPG, ECG | 77 | ECG | Not specified | Signal quality, cardiac variability | rPPG reliable for mean HR; individual-level HRV accuracy limited |
| Herberger et al. [31] | 2025 | Sleep staging | Oura, SleepOn, Circul rings | Not specified | Unknown | PSG | Not specified | Accuracy, sensitivity, epoch agreement, TST difference | Sleep stage accuracy 35–53%; individual-level errors large; not suitable for clinical use |
| Kawasaki et al. [32] | 2022 | Sleep staging | Fitbit Alta HR | Accelerometer, PPG | 40 | EEG | Not specified | Pearson r | TST r=0.83; N3 r=0.68–0.73; no epoch-level agreement metrics |
| Ghorbani et al. [33] | 2022 | Sleep staging | Oura Ring Gen2/Gen3 | Accelerometer, PPG, temperature | 58 | PSG | External (Gen3 trained on independent dataset) | Accuracy, sensitivity, specificity, Cohen's d | Gen3: 92.6% accuracy, 94.9% sensitivity, 78.5% specificity vs. PSG |
| Markun & Sampat [34] | 2020 | Sleep staging | Consumer wearables, HSAT | ECG, EMG, EOG | Unknown | PSG | Not specified | Not specified | PSG remains gold standard; consumer wearables lack appropriate validation |
| Lee et al. [35] | 2019 | Sleep staging | Fitbit Alta HR, Actiwatch 2 | Accelerometer, PPG | 58 | PSG | Not specified | Sensitivity, specificity, TST, WASO | Fitbit comparable for duration; inaccurate sleep staging in adolescents, especially N3 |
| Nazar et al. [36] | 2026 | HAR | Not specified | Accelerometer, gyroscope | Unknown | Not specified | Not specified | Accuracy | Ensemble MLP >95% accuracy on PAMAP2, MHealth, HHAR benchmarks |
| Feng et al. [37] | 2026 | HAR dataset | 17-sensor IMU suit | Accelerometer, gyroscope | 30 | Structured activity annotations | Not specified | Not specified | 17-IMU full-body dataset for HAR benchmarking introduced; no performance results reported |
| Tokgoz & Kocaoglu [38] | 2026 | Fall detection / HAR | Wearable IMU | Accelerometer, gyroscope, magnetometer | 17 | Manual labeling | Not specified | Not specified | AybuFall dataset introduced with prayer movements to reduce false positives; no ML evaluation |
| Chaudhary et al. [39] | 2026 | HAR (domain adaptation) | Wearable sensors | Accelerometer | Unknown | Not specified | External validation (4 benchmark datasets) | Macro-F1, accuracy | MSAFAN: +8.4% macro-F1, +10.3% accuracy, -26% compute vs. SOTA across 4 benchmarks |
| Maylor et al. [40] | 2025 | HAR, sleep, HR, sedentary behaviour | Multi-site wearables, PSG, ECG patch, camera | Accelerometer, PPG, ECG | 160 (target; 15 enrolled) | PSG, annotated video, ECG, ankle accelerometer | Not specified (protocol) | Precision, recall, F1, kappa | OxWEARS protocol: open-access free-living benchmark dataset in progress |
| Zhang et al. [41] | 2025 | HAR | MPU6050 sensor | Accelerometer | Unknown | Not specified | Not specified | Accuracy | Dual-INABA: 96.95–98.89% on self-collected + PAMAP2/UCI-HAR/WISDM; no split strategy |
| Meini et al. [42] | 2025 | Stress/attention/HAR in education | Wearable devices | Not specified | Unknown | Not specified | Not specified | Not specified | Systematic review: 43 studies; absence of standardised reporting limits benchmarking |
| Bassani et al. [43] | 2025 | HAR (manual material handling) | Wearable sensors | Not specified | 14 | Not specified | 70-30 + LOSO | F1, accuracy | BiLSTM: 95.7% (70-30) vs. 90.3% (LOSO); small cohort but dual split strategy is strength |
| Mamiya & Fuller [44] | 2025 | HAR | Wearable devices | Accelerometer | 37 | Not specified | LOSO | Accuracy, F1, confusion matrix | BART provides uncertainty quantification comparable to RF; labeling method not described |
| Sheng & Huber [45] | 2025 | HAR | Wearable sensors | Not specified | Unknown | Not specified | Not specified | Accuracy | Weakly self-supervised learning with 10% labels achieves comparable performance to full supervision |
| Al Farid et al. [46] | 2025 | HAR (multi-view, AAL) | Not specified | Not specified | Unknown | Not specified | Not specified | Accuracy | Multi-view DL improves HAR robustness for AAL; narrative review only |
| Airaksinen et al. [47] | 2025 | HAR (infant movement) | Multi-sensor IMU suit | Accelerometer, gyroscope | 41 | Human video annotations | Not specified | Cohen's kappa | Minimum config: 1 upper + 1 lower extremity sensor, ≥13 Hz, accelerometer (+ gyroscope preferred); posture kappa 0.89–0.91 |
| Fu et al. [48] | 2025 | HAR (early-exit, meta-learning) | Wearable devices | Accelerometer | Unknown | Not specified | Not specified | Accuracy | Meta-learning sample reweighting improves accuracy-efficiency on UCI-HAR, WISDM, UniMiB-SHAR |
| Yu & Al-Qaness [49] | 2025 | HAR | Wearable sensors | Accelerometer | Unknown | Benchmark labels | Not specified | Accuracy | DKInception: 87.5–95.7% on UCI-HAR/Opportunity/Daphnet/PAMAP2; no split strategy disclosed |
| Mondal et al. [50] | 2026 | AFib detection (cardiology AI review) | Not specified | ECG, echocardiography, CMR | Unknown | Not specified | Not specified | Not specified | AI in cardiology hindered by limited external validation and subgroup fairness gaps |
| Stachteas et al. [51] | 2025 | AFib detection | Smartwatch, Holter, ILR | PPG, ECG | Unknown | Not specified | Not specified | Not specified | Smartwatch AF monitoring improves detection but faces motion artifacts and regulatory challenges |
| Narodova [52] | 2025 | Seizure detection | Smartwatch, wearable detectors | Not specified | Unknown | Not specified | Not specified | Sensitivity, specificity, false-alarm rates | Validated detectors: sensitivity 80–95%; most tools lack adaptive personalisation |
| Choi et al. [53] | 2025 | Anxiety/stress detection | Samsung Galaxy Watch6 | PPG, ECG | Unknown (very small) | GAD-7 | External validation (cross-dataset) | AUC, p-value | AUC 0.859 (CI 0.664–0.992) for anxiety in post-stroke patients; very wide CI suggests tiny cohort |
| Cay et al. [54] | 2025 | HAR / adherence detection | SmartBoot, Sensoria, smartwatch | Accelerometer, gyroscope, IMU | 93 (12 healthy + 81 DFU) | Gold-standard wearable, staff logs | Not specified | Accuracy, bias | 96–97% adherence detection accuracy; validation in healthy adults, not target DFU population |
| Cossu et al. [55] | 2024 | Signal processing / feature extraction (MS, ALS) | Garmin Vivoactive 4 | Not specified | Unknown | Not specified | Not specified | Not specified | Preprocessing pipeline for MS/ALS wearable data; no ML evaluation performed |
| Chiu et al. [56] | 2024 | Serum potassium prediction | Smartwatch | ECG | 1463 (train) + 40 (prospective) | Serum laboratory potassium | External validation (2 independent sites) | AUROC, MAE | Kardio-Net AUC 0.831 for hyperkalemia from smartwatch ECG; prospective cohort n=40 |
| Antiperovitch et al. [57] | 2024 | AFib detection | Smartwatch | PPG, ECG | 204 | ECG | Participant-level holdout + external | AUROC, sensitivity, specificity | CNN AUC ~0.973 ≈ heuristic (0.972) but classifies 95% vs. 62% of continuous data |
| Presley et al. [58] | 2023 | Balance assessment | Smartwatch | Accelerometer, gyroscope, IMU | Unknown | Strictly placed IMU | Not specified | Pearson r, p-value | Smartwatch IMU balance metrics r = 0.861–0.970 vs. reference IMU |
| Mannhart et al. [59] | 2023 | AFib detection | Apple Watch, Fitbit, AliveCor, Samsung, Withings | ECG | 117 | 12-lead ECG | Not specified | Sensitivity, specificity, conclusive rate | DNN reduces inconclusive tracings across 5 devices; cross-platform evaluation is strength |
| van der Zande et al. [60] | 2023 | ECG lead validation | Apple Watch | ECG | 200 | 12-lead ECG | Not specified | Bias, offset, 95% LoA | Apple Watch precordial ECGs comparable to 12-lead; positive R-wave amplitude bias in V1/V3/V6 |
| Mannhart et al. [61] | 2023 | AFib detection | Apple Watch 6, Samsung Galaxy Watch 3, Withings Scanwatch, Fitbit Sense, AliveCor KardiaMobile | ECG, PPG | 201 | 12-lead ECG | Not specified | Sensitivity, specificity | Sensitivity 58–85%, specificity 69–79%; 17–26% inconclusive rate across 5 devices |
| Guo et al. [62] | 2022 | Motor function assessment | Smartphones, smartwatches, Kinect, wearable sensors | Accelerometer, motion sensors | Unknown | Clinical measures | Not specified | Sensitivity, specificity, accuracy, Pearson r | Mean sensitivity 0.83, specificity 0.84, accuracy 0.90; most studies proof-of-concept |
| Mannhart et al. [63] | 2022 | QTc interval measurement | Withings Scanwatch | ECG | 317 | 12-lead ECG (dual blinded cardiologists) | Not specified | AUROC, bias, LoA | AUC 0.91/0.89 for prolonged QTc; algorithm failed in 44% of patients |

---

## 5. Themes

### 5.1 Pervasive Evaluation Metric Heterogeneity

A striking feature of the corpus is the absence of a shared evaluation lingua franca. HAR studies predominantly report accuracy alone (Papers 36, 41, 48, 49), often without sensitivity, specificity, F1, or AUROC. Device validation studies (Papers 18, 24, 29, 60, 63) favor Bland-Altman limits of agreement and ICC, which are appropriate for agreement studies but cannot be compared to classifier benchmarks. Cardiac monitoring studies (Papers 57, 59, 61) report sensitivity/specificity but vary in whether they compute AUC or PPV. This fragmentation directly impedes cross-study benchmarking and meta-analysis.

### 5.2 Data Leakage and Split Strategy Opacity

Perhaps the most consequential finding of this review is the systematic failure to report—or adopt—participant-level data splitting. The majority of HAR papers (Papers 36, 41, 48, 49) either do not describe their split strategy at all or employ segment- or observation-level splits, which generate inflated performance estimates because signal windows from the same participant appear in both training and test sets. Paper 2 explicitly demonstrates this: segment-level cross-validation yielded 99.89% accuracy on the same dataset where LOSO yielded 89.34%—a 10.55% gap. Papers 5 and 12 similarly employ within-subject or ambiguous splits. Only a minority of studies (Papers 4, 43, 44) consistently apply LOSO or participant-level holdouts that approximate real-world deployment conditions.

### 5.3 Cohort Size and Demographic Narrowness

The median cohort size across primary empirical studies with reported N is modest: most fall in the range of 14–117 participants, with only a handful exceeding 200 (Papers 20, 56, 60, 61, 63). Several studies report no cohort size at all (Papers 3, 13, 36, 41, 48, 49). Demographic reporting is similarly sparse: the majority of papers either omit age, sex, race/ethnicity, and BMI, or restrict their populations to healthy adults, athletes, or single-condition patient cohorts. Notable exceptions include the OxWEARS protocol (Paper 40), which explicitly stratifies by age, sex, and BMI, and the PRV reliability study (Paper 28), which includes lifespan subgroup analyses. The implication is that many reported performance figures cannot be assumed to generalise across sex, age, or ethnicity.

### 5.4 The Gap Between Laboratory and Free-Living Validation

A recurring theme is the performance degradation that occurs when models trained or validated under controlled conditions are applied to free-living data. Sleep ring trackers (Paper 31) show acceptable group-level TST agreement but individual-level staging accuracy of only 35–53%—far below controlled-setting figures. Smartwatch HR monitoring degrades notably during resistance exercise (Paper 24). The rPPG webcam study (Paper 30) finds reliable mean HR but limited individual HRV accuracy. The OxWEARS protocol (Paper 40) is explicitly designed to address this gap, but its dataset is not yet complete. Only Papers 33, 40, and 57 employ truly free-living or multi-night home validation paradigms.

### 5.5 Device Heterogeneity as a Benchmarking Barrier

The corpus spans an extremely broad device landscape: consumer smartwatches (Apple Watch, Galaxy Watch, Fitbit, Garmin, Oura), clinical patches (Biobeat), custom IMU suits, ring trackers, and smartphone cameras. Even within single-task domains, direct comparison is complicated by differences in sensor placement, sampling frequency, proprietary signal conditioning, and algorithm opacity. The BASEL Wearable Study (Paper 61) is the most systematic head-to-head comparison, evaluating five devices under identical conditions, and finds meaningful sensitivity/specificity variation (58–85% / 69–79%) across devices. Domain adaptation work (Paper 39) addresses cross-device generalisation algorithmically, but evaluation remains confined to public benchmark datasets.

### 5.6 Reference Standard Inconsistency

Reference standards vary markedly even within the same clinical task. For sleep staging, PSG is near-universally accepted (Papers 31–35) but rarely achieved in free-living contexts. For AFib, some studies use physician-interpreted 12-lead ECG (Papers 59, 61), others use Holter-derived annotations, and some do not specify (Paper 19). For HAR, most studies rely on implicit benchmark dataset labels with no description of how ground truth was assigned. For physiological parameters such as VO₂ (Paper 4), HRV (Paper 18), and blood glucose (Paper 14), reference standards range from laboratory metabolic testing to interstitial CGM to self-reported surveys (Paper 53). This variability renders cross-study performance comparison unreliable.

### 5.7 External Validation Remains the Exception

The overwhelming majority of empirical studies in this corpus evaluate their models only on internal data. External validation on independent cohorts—arguably the most important indicator of clinical readiness—is performed in only a small minority of papers: Papers 28, 33, 39, 53, 56, 57 report some form of cross-cohort or cross-device evaluation. The systematic reviews (Papers 1, 6, 7) identify the absence of external validation as a field-wide limitation. This is particularly concerning given that many studies report near-saturated performance on small or non-representative internal datasets.

### 5.8 Review Inflation in the Literature

Approximately 40% of the corpus (25 of 63 papers) consists of narrative or systematic reviews rather than original empirical studies. While several of these are valuable (Papers 23, 42, 62), many are non-systematic narratives that synthesise findings across heterogeneous studies without formal quality assessment, bias auditing, or quantitative pooling. This creates a risk of compounding the methodological weaknesses of primary studies through uncritical repetition of reported metrics. The representation of review articles in benchmarking-focused literature searches is a systematic issue that inflates the apparent evidence base.

### 5.9 Edge Deployment and Computational Efficiency

Several papers explicitly address the constraints of on-device inference, a requirement for real-world wearable deployment (Papers 4, 9, 43, 48). The shift toward lightweight architectures—ELM+RF ensembles (Paper 4), BiLSTM (Paper 43), early-exit networks (Paper 48), and TCN/attention hybrids (Paper 9)—reflects growing awareness that benchmark accuracy on desktop hardware does not translate to wearable viability. However, standardised reporting of inference latency, energy consumption, and memory footprint remains rare.

### 5.10 Uncertainty Quantification and Explainability

Only two papers in the corpus directly address uncertainty quantification (Paper 44, via Bayesian BART) and explainability (Paper 4, via SHAP analysis). The remaining 61 papers provide point estimates of performance without confidence intervals, calibration metrics, or feature attribution. This represents a significant gap for clinical deployment, where decision support systems must communicate their confidence to users and clinicians. Narrative reviews (Papers 7, 17, 50) note interpretability as a recurring barrier to adoption.

---

## 6. Gaps and Open Questions

1. **Participant-level benchmarking standards.** No consensus framework exists for defining and reporting the minimum acceptable cross-validation strategy for wearable AI studies. The field urgently needs community standards (analogous to TRIPOD for prediction models) that mandate participant-level holdout or LOSO evaluation, alongside segment-level results for methodological completeness.

2. **Demographic representativeness and subgroup reporting.** The near-universal absence of subgroup analyses by sex, age, race/ethnicity, and BMI means that reported performance figures cannot be assumed to apply to real-world clinical populations. Studies evaluating fairness or differential performance across protected attributes are almost entirely absent from this corpus.

3. **Free-living and longitudinal validation.** Controlled-laboratory studies dominate the corpus. Long-duration, free-living, multi-night, or multiday evaluations are rare. The OxWEARS protocol (Paper 40) represents the most ambitious attempt to address this gap, but is still in data collection. Longitudinal drift in wearable signal quality, user behavior, and physiological state are not adequately modelled in current benchmarks.

4. **Cross-device and cross-platform generalisation.** Most models are trained and tested on a single device type. With the exception of a few head-to-head studies (Papers 59, 61) and domain adaptation work (Paper 39), the question of whether a model trained on one wearable ecosystem generalises to another remains largely unaddressed.

5. **Standardised reference standards.** The field lacks consensus on appropriate reference standards even for well-established tasks. For sleep staging, PSG is accepted but operationalised inconsistently across studies. For HRV, ECG-based calculation is standard but rarely accompanied by lead-placement or pre-processing protocol details. This prevents meaningful meta-analysis.

6. **Open benchmark datasets with clinical populations.** The majority of HAR benchmarks are collected from healthy young adults under controlled conditions. Datasets representing elderly, chronically ill, or post-surgical populations—the primary beneficiaries of health AI wearable applications—are severely underrepresented. AybuFall (Paper 38) and OxWEARS (Paper 40) are positive steps, but clinical