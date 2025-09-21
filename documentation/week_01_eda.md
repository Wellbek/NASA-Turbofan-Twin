# Week 1 – Exploratory Data Analysis (EDA)

## 1. Dataset Overview
**Dataset Name:** FD001 – NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset    

**Key Facts:**
- **Number of Engines (Trajectories):** 100 (both training and test sets)  
- **Operating Conditions:** ONE (Sea Level) → all engines run under the same standard environment (normal atmospheric pressure, temperature, and throttle).  
- **Fault Mode:** ONE (HPC Degradation) → only the High Pressure Compressor (HPC) degrades over time.  

FD001 is the **simplest dataset** in the C-MAPSS family. Here’s what it really means:

1. **One Operating Condition (Sea Level):**  
   - All engines are running under the same environmental settings.  
   - Sensor readings mostly reflect **engine health**, not changes in environment.  

2. **One Fault (HPC Degradation):**  
   - Only the High Pressure Compressor of each engine is degrading over time.  
   - “Degradation” means the component slowly loses efficiency or performance.  
   - This affects specific sensors more than others (e.g., sensors measuring pressure, temperature in HPC).  
   - No overlapping faults

3. **Sensor Noise:**  
   - Measurements are not perfect; some sensors may fluctuate slightly even if the engine is healthy.  
   - Noise is normal and part of the challenge in modeling.

---

## 3. Data Structure
26 columns:  
* 1.Engine ID  
* 2.Time (cycle)  
* 3–5.Operational settings (3 total)  
* 6–26.Sensor measurements (21 total)

Each row = snapshot of one engine cycle.

---

## 4. Exploratory Analysis Findings

### Missing Values

**Why check for missing values:**
* Missing data can distort analysis and modeling.
* Some sensors might fail to record measurements, leading to NaNs or blanks.
* Detecting missing values early helps decide whether to **impute**, or **ignore** columns.

**Results for FD001:**

| Column                  | Missing Values | Interpretation                                  |
| ----------------------- | -------------- | ----------------------------------------------- |
| engine\_id              | 0              | All engine IDs are present.                     |
| time\_cycles            | 0              | Cycle numbers recorded for all engines.         |
| operational\_setting\_1 | 0              | No missing operational settings.                |
| operational\_setting\_2 | 0              | No missing operational settings.                |
| operational\_setting\_3 | 0              | No missing operational settings.                |
| sensor\_1               | 0              | Complete data.                                  |
| sensor\_2               | 0              | Complete data.                                  |
| ...                     | ...            | ...                                             |
| sensor\_21              | 0              | Complete data.                                  |
| RUL                     | 0              | Remaining Useful Life provided for all engines. |

* **FD001 has no missing values**, which is ideal for initial exploration.
* All sensors, settings, and RUL values are complete, so no imputation is required.

---

### Outlier Detection

**Why check for outliers:**

* Outliers are unusually high or low values that differ significantly from the majority of data.
* They can **distort statistical analysis**, **bias models**, or indicate **sensor malfunctions**.
* Identifying outliers early helps decide whether to **remove, correct, or keep** them.

**How we checked for outliers:**

* Boxplots were created for all sensor measurements.
* Boxplots visualize the **median, quartiles, and potential extreme values** for each sensor.

**Results for FD001:**

| Sensor     | Outlier Presence | Interpretation                     |
| ---------- | ---------------- | ---------------------------------- |
| sensor\_1  | None             | Values fall within expected range. |
| sensor\_2  | None             | Values fall within expected range. |
| ...        | ...              | ...                                |
| sensor\_21 | None             | Values fall within expected range. |

<img width="866" height="360" alt="image" src="https://github.com/user-attachments/assets/6e74c041-8112-409f-8aeb-36968e68c230" />

**Summary / Interpretation:**

* **No significant outliers detected** across all sensors in FD001.
* Sensor readings are consistent and within expected ranges.
* This implies **no sensor malfunctions or extreme deviations**, so we do **not need outlier removal** at this stage.

Absolutely! Let’s **turn this into a professional, beginner-friendly section** for your README, highlighting **sensor distributions** and explaining why **skew is important**. I’ll format it consistently with your previous sections.

---

### Sensor Distributions

**Why check sensor distributions:**

* Understanding **how sensor values are distributed** helps identify trends, patterns, and potential issues.
* Distributions reveal **central tendency (mean/median)**, **spread (standard deviation, min/max)**, and **skewness**.
* Skewness indicates whether the data is **symmetrically distributed** or has a **long tail**, which can affect modeling; models often perform better if distributions are roughly normal.

**Findings for FD001 (Training Set):**

| Sensor     | Mean    | Std   | Min     | Max     | Skew  | Interpretation / Key Fact                                                |
| ---------- | ------- | ----- | ------- | ------- | ----- | ------------------------------------------------------------------------ |
| sensor\_1  | 518.67  | 0.00  | 518.67  | 518.67  | 0.00  | Constant; not informative.                                            |
| sensor\_2  | 642.68  | 0.50  | 641.21  | 644.53  | 0.32  | Slight right skew; generally stable.                                     |
| sensor\_3  | 1590.52 | 6.13  | 1571.04 | 1616.91 | 0.31  | Mild right skew; gradual degradation visible over cycles.                |
| sensor\_4  | 1408.93 | 9.00  | 1382.25 | 1441.49 | 0.44  | Mild right skew; some variability, may be informative for RUL.           |
| sensor\_5  | 14.62   | 0.00  | 14.62   | 14.62   | 0.00  | Constant; not informative.                                        |
| sensor\_6  | 21.61   | 0.001 | 21.60   | 21.61   | -6.92 | Strong left skew; very little variation.                                 |
| sensor\_7  | 553.37  | 0.89  | 549.85  | 556.06  | -0.39 | Slight left skew; moderate variability.                                  |
| sensor\_8  | 2388.10 | 0.07  | 2387.90 | 2388.56 | 0.48  | Slight right skew; stable.                                               |
| sensor\_9  | 9065.24 | 22.08 | 9021.73 | 9244.59 | 2.56  | Significant right skew; long tail; potentially sensitive to degradation. |
| sensor\_10 | 1.30    | 0.00  | 1.30    | 1.30    | 0.00  | Constant; not informative.                                               |
| sensor\_11 | 47.54   | 0.27  | 46.85   | 48.53   | 0.47  | Slight right skew; minor variability.                                    |
| sensor\_12 | 521.41  | 0.74  | 518.69  | 523.38  | -0.44 | Slight left skew; relatively stable.                                     |
| sensor\_13 | 2388.10 | 0.07  | 2387.88 | 2388.56 | 0.47  | Slight right skew; stable.                                               |
| sensor\_14 | 8143.75 | 19.08 | 8099.94 | 8293.72 | 2.37  | High right skew; potentially sensitive to degradation.                   |
| sensor\_15 | 8.44    | 0.04  | 8.32    | 8.58    | 0.39  | Minor right skew; small variations.                                      |
| sensor\_16 | 0.03    | 0.00  | 0.03    | 0.03    | 0.00  | Constant; not informative.                                               |
| sensor\_17 | 393.21  | 1.55  | 388.00  | 400.00  | 0.35  | Slight right skew; may indicate gradual changes.                         |
| sensor\_18 | 2388.00 | 0.00  | 2388.00 | 2388.00 | 0.00  | Constant; not informative.                                               |
| sensor\_19 | 100.00  | 0.00  | 100.00  | 100.00  | 0.00  | Constant; not informative.                                               |
| sensor\_20 | 38.82   | 0.18  | 38.14   | 39.43   | -0.36 | Slight left skew; small variation.                                       |
| sensor\_21 | 23.29   | 0.11  | 22.89   | 23.62   | -0.35 | Slight left skew; small variation.                                       |

<img width="867" height="579" alt="image" src="https://github.com/user-attachments/assets/95e89bf2-98c8-4b18-889f-ad327cc62730" />

**Key Highlights / Insights:**

1. **Skew matters:**

   * Sensors with **high positive skew** (e.g., sensor\_9, sensor\_14) have a long tail of higher values — these are often **good indicators of degradation** as the readings increase before failure.
   * Sensors with **zero or near-zero skew** and constant values are **not informative** and can be dropped from RUL modeling.

---

### Operational Settings Distributions

**Why analyze operational settings:**

* Operational settings (ops) control engine behavior and can **directly affect sensor readings**.
* Understanding their distribution helps determine whether sensors vary due to **engine degradation** or **changes in settings**.
* Some models need operational settings as **features** to correctly predict Remaining Useful Life (RUL).

**Observations from FD001 scatterplots:**

| Operational Setting     | Observed Distribution               | Interpretation / Key Fact                                                                                                                     |
| ----------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| operational\_setting\_1 | Full range of values across engines | Setting 1 varies normally; likely simulates realistic engine operation. Sensor readings may depend on this.                                   |
| operational\_setting\_2 | Only certain discrete values appear | Setting 2 is **discrete or limited**, not continuously varied. Models may treat it as categorical or discrete numeric.                        |
| operational\_setting\_3 | Single constant line across engines | Setting 3 is **constant** in FD001; it does **not vary**, so it likely **does not influence sensor readings**. |

<img width="896" height="701" alt="image" src="https://github.com/user-attachments/assets/046e41f7-19cc-4efd-b1e6-ba131190a99a" />

**Interpretation:**

1. **Operational Setting 1 (full range):**

   * Sensors vary along a wide range of engine states; this setting is informative and should be included in analysis.

2. **Operational Setting 2 (discrete values):**

   * Sensors only yield specific values instead of continuous.
   * It’s important to treat as **categorical data**.

3. **Operational Setting 3 (constant):**

   * No variation → this setting does **not provide information** about degradation.

---

### RUL Data Distribution

**Why analyze RUL data distribution:**

* **Assess data coverage:** Helps determine how many data points are available at each RUL (Remaining Useful Life) level.
* **Evaluate balance:** Shows whether the training data is evenly distributed across the degradation spectrum, which impacts model training and performance.

**Results for FD001:**

* The distribution shows an **exponential decrease** in the number of entries as RUL increases.
* This pattern is expected: more data is available in low RULs at the start of their lifes, but the number of entries decreases as engines approach failure, dropping off one by one.

<img width="822" height="493" alt="image" src="https://github.com/user-attachments/assets/69a24074-a81d-4519-825c-c4ce5b1f59c5" />

---

### Engine Lifetime Distribution

**Why analyze engine lifetime distribution:**

* **Addresses the core engineering problem:** Understanding how long each engine operates before failure.
* Provides insight into the **range of lifetimes** in the dataset, which is essential for modeling RUL.
* Helps identify **typical vs. extreme cases**, guiding how models might handle engines with unusually long or short lifetimes.

**Results for FD001:**

* Most engines fail between **150–220 cycles**, indicating the typical operational lifespan in this dataset.
* Only a few engines exceed **220 cycles**, and very few go beyond **350 cycles**, representing extreme cases.
* **Implications for modeling:** Models may be trained mostly on engines in the 150–220 cycle range; rare long-lived engines could require special handling or weighting to predict accurately.

<img width="791" height="491" alt="image" src="https://github.com/user-attachments/assets/a66c620e-4fa2-4863-8583-189251c4f08c" />

---

### Correlation Heatmap of Sensors

**What it shows:**

* Pairwise correlation between all sensors.
* **Red = strong positive correlation**, **Blue = strong negative correlation**.

**Why valuable:**

* CMAPSS datasets contain many **redundant sensors**. Highly correlated sensors add little new information. Dropping them can reduce dimensionality and simplify modeling.
* Completely uncorrelated sensors may be **uninformative or noisy**.
* Helps **avoid overfitting** by reducing the number of features.

**Results / Interpretation for FD001:**

* Sensors identified for potential removal:

  * **Constant:** `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`, `sensor_14`
  * **Duplicate / highly correlated:** `sensor_6`, `sensor_14`
* **Implication:** Fewer sensors may be used without losing predictive power, simplifying feature engineering and reducing computation.

<img width="882" height="772" alt="image" src="https://github.com/user-attachments/assets/ad95b8e9-70e8-4690-b3ad-0dd0e8b6ed7e" />


---

### Engine Sensor Time-Series

**What it shows:**

* Evolution of selected sensor values over operational cycles for individual engines (e.g., `engine_id=1`).

**Why valuable:**

* Reveals **degradation patterns**: some sensors drift steadily with wear; others fluctuate randomly.
* Identifies potential **“health indicators”** - sensors that reflect engine deterioration.
* Provides the first clues for **feature engineering**, such as trends, slopes, or rolling statistics.

**Interpretation for FD001:**

| Sensor Pattern      | Sensors                           | What it tells us / Implications for future weeks                                                                                                   |
| ------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Constant**        | 1, 5, 6, 10, 16, 18, 19           | These sensors do **not reflect degradation**. They can likely be ignored in modeling; including them adds noise but no predictive value.           |
| **Upwards Trend**   | 2, 3, 4, 8, 9, 11, 12, 13, 15, 17 | These sensors **increase as the engine wears**. They are strong candidates for **RUL prediction features** and can indicate early signs of faults. |
| **Downwards Trend** | 7, 9, 12, 14, 20, 21              | These sensors **decrease with degradation**. Like upward-trending sensors, they are informative and useful for monitoring engine health.           |

<img width="876" height="566" alt="image" src="https://github.com/user-attachments/assets/74acfbe1-862c-464a-913c-6888cb1af9e3" />

**Forward-looking implications for Week 2+:**

1. Focus feature engineering on sensors with **clear trends** (upwards or downwards) — e.g., slopes, rolling means, or normalized values.
2. Consider dropping **constant sensors** to simplify the dataset and reduce overfitting risk.
3. These patterns give an early **sensor selection guide** for predictive modeling; future weeks can prioritize these trends in regression or machine learning approaches.

---

### Operational Settings Distribution

**Why we check operational settings distribution:**

* Engines operate under different conditions; operational settings **influence sensor readings and degradation rate**.
* Understanding their distribution helps **control for confounding factors** in modeling.

**Interpretation for FD001:**

* **Operational Setting 1:** Full range of values → informative for modeling.
* **Operational Setting 2:** Discrete values only → may be treated as categorical.
* **Operational Setting 3:** Constant → not informative, can be ignored.

<img width="860" height="284" alt="image" src="https://github.com/user-attachments/assets/9db69397-bebe-493c-96ec-96b9f6c4fcfd" />

---

### Sensor vs. RUL Scatter

**What it shows:**

* Relationship between individual sensor readings and Remaining Useful Life.

<img width="863" height="566" alt="image" src="https://github.com/user-attachments/assets/7df7a4ea-bb8c-486a-a49a-36648a43c39a" />

**Why valuable:**

* Sensors with **monotonic trends** (increase or decrease steadily as RUL decreases) are **strong predictors** of remaining life.
* Scatter plots help distinguish **predictive sensors** from noisy or irrelevant ones.

**Interpretation for FD001:**

* `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21` show **clear monotonic trends** → strong candidates for predictive modeling.
* Constant or highly noisy sensors (`sensor_1`, `sensor_5`,  `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) show **no trend**, likely uninformative.

---

### Normalized Lifecycle Plots

**What it shows:**

* Anaologue to before, sensor trajectories averaged across engines, aligned by **normalized cycle** (`0=start`, `1=end-of-life`).

<img width="863" height="555" alt="image" src="https://github.com/user-attachments/assets/fce8a48e-4d8a-4cdb-9320-cda9914e9f34" />

**Why we check eventhough already examined Sensor vs. RUL via scatter:**

* Engines have different lifespans; raw cycle counts are **not directly comparable**.
* Normalization allows comparison of **trends across engines**, highlighting consistent degradation patterns.

---



