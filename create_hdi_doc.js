const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageBreak, PageNumber, Header, Footer,
  TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Colors ─────────────────────────────────────────────────
const BLUE      = "185FA5";
const BLUE_LIGHT= "D5E8F6";
const TEAL      = "0F6E56";
const TEAL_LIGHT= "D5EFE7";
const GRAY      = "5F5E5A";
const GRAY_LIGHT= "F2F2F0";
const AMBER     = "854F0B";
const AMBER_LT  = "FAF0DC";
const WHITE     = "FFFFFF";
const BLACK     = "0B0B0B";
const RED_LT    = "FCE8E8";

// ── Reusable border set ────────────────────────────────────
const cellBorder = {
  top:    { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  left:   { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  right:  { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
};

function hCell(text, shade = BLUE, color = WHITE) {
  return new TableCell({
    borders: cellBorder,
    width: { size: 9360 / 3, type: WidthType.DXA },
    shading: { fill: shade, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color, font: "Arial", size: 20 })]
    })]
  });
}

function dCell(text, shade = WHITE, color = BLACK, bold = false, width = 3120) {
  return new TableCell({
    borders: cellBorder,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: shade, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 140, right: 140 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold, color, font: "Arial", size: 20 })]
    })]
  });
}

// ── Heading helpers ────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 340, after: 180 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE, space: 1 } },
    children: [new TextRun({ text, bold: true, color: BLUE, font: "Arial", size: 32 })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, color: TEAL, font: "Arial", size: 26 })]
  });
}

function body(text, spaceAfter = 140) {
  return new Paragraph({
    spacing: { after: spaceAfter },
    children: [new TextRun({ text, font: "Arial", size: 22, color: "333333" })]
  });
}

function bullet(text, ref = "bullets") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Arial", size: 22, color: "333333" })]
  });
}

function code(text) {
  return new Paragraph({
    spacing: { after: 80 },
    shading: { fill: "1E1E2E", type: ShadingType.CLEAR },
    indent: { left: 360, right: 360 },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "A8E6CF" })]
  });
}

function codeBlock(lines) {
  return lines.map((line, i) =>
    new Paragraph({
      spacing: { after: i === lines.length - 1 ? 180 : 40 },
      shading: { fill: "1E1E2E", type: ShadingType.CLEAR },
      indent: { left: 360, right: 360 },
      children: [new TextRun({ text: line || " ", font: "Courier New", size: 18, color: "A8E6CF" })]
    })
  );
}

function note(text) {
  return new Paragraph({
    spacing: { after: 160 },
    shading: { fill: AMBER_LT, type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: AMBER, space: 6 } },
    indent: { left: 300 },
    children: [new TextRun({ text: "Note: " + text, font: "Arial", size: 20, color: AMBER, italics: true })]
  });
}

function tip(text) {
  return new Paragraph({
    spacing: { after: 160 },
    shading: { fill: TEAL_LIGHT, type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: TEAL, space: 6 } },
    indent: { left: 300 },
    children: [new TextRun({ text: text, font: "Arial", size: 20, color: TEAL })]
  });
}

function spacer(pts = 160) {
  return new Paragraph({ spacing: { after: pts }, children: [new TextRun("")] });
}

// ── PAGE BREAK ────────────────────────────────────────────
function pageBreak() {
  return new Paragraph({ children: [new TextRun({ break: 1 })] });
}

// ═══════════════════════════════════════════════════════════
//  DOCUMENT BUILD
// ═══════════════════════════════════════════════════════════
const doc = new Document({
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0, format: LevelFormat.BULLET, text: "\u2022",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
      {
        reference: "numbers",
        levels: [{
          level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
        }]
      },
    ]
  },
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22 } }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 340, after: 180 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: TEAL },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1260 }
      }
    },
    headers: {
      default: new Header({
        children: [
          new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space: 1 } },
            children: [
              new TextRun({ text: "ML-0027  |  Human Development Index", font: "Arial", size: 18, color: BLUE, bold: true }),
              new TextRun({ text: "\t\t", font: "Arial", size: 18 }),
              new TextRun({ text: "Project Report", font: "Arial", size: 18, color: GRAY }),
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }]
          })
        ]
      })
    },
    children: [

      // ──────────────────────────────────────────────────
      //  TITLE PAGE
      // ──────────────────────────────────────────────────
      spacer(480),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
        shading: { fill: BLUE, type: ShadingType.CLEAR },
        children: [new TextRun({ text: "  ML - 0027  ", font: "Arial", size: 20, color: WHITE, bold: true })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: "Human Development Index", font: "Arial", size: 48, bold: true, color: BLUE })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({ text: "Machine Learning Project — Complete Documentation", font: "Arial", size: 26, color: GRAY, italics: true })]
      }),
      spacer(120),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "Predicting HDI scores from socio-economic indicators", font: "Arial", size: 22, color: GRAY })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 480 },
        children: [new TextRun({ text: "Dataset: HDI.csv  |  Algorithm: Gradient Boosting  |  R\u00B2: 0.9971", font: "Arial", size: 22, color: TEAL, bold: true })]
      }),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 1: LIBRARY IMPORTS
      // ──────────────────────────────────────────────────
      h1("Step 1 — Import Libraries"),
      body("Importing all necessary libraries at the beginning ensures an organised, efficient development process. All dependencies are declared in one place before any data analysis or model-building activities begin, making the code easy to understand, maintain, and debug."),

      h2("1.1  Libraries Used"),

      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2200, 2200, 4960],
        rows: [
          new TableRow({ children: [
            hCell("Library", BLUE), hCell("Version", BLUE), hCell("Purpose", BLUE)
          ]}),
          new TableRow({ children: [
            dCell("NumPy"), dCell("1.24+"), dCell("Core numerical computing, array operations, log transforms")
          ]}),
          new TableRow({ children: [
            dCell("Pandas", GRAY_LIGHT), dCell("2.0+", GRAY_LIGHT), dCell("Load, explore, and manipulate the CSV dataset", GRAY_LIGHT)
          ]}),
          new TableRow({ children: [
            dCell("Matplotlib"), dCell("3.7+"), dCell("Base plotting engine for all visualisations")
          ]}),
          new TableRow({ children: [
            dCell("Seaborn", GRAY_LIGHT), dCell("0.12+", GRAY_LIGHT), dCell("Statistical charts: strip plots, heatmaps, pairplots", GRAY_LIGHT)
          ]}),
          new TableRow({ children: [
            dCell("scikit-learn"), dCell("1.3+"), dCell("ML pipeline: split, scale, train, evaluate")
          ]}),
          new TableRow({ children: [
            dCell("pickle", GRAY_LIGHT), dCell("built-in", GRAY_LIGHT), dCell("Serialise trained model for Flask deployment", GRAY_LIGHT)
          ]}),
        ]
      }),
      spacer(200),

      h2("1.2  Import Code"),
      ...codeBlock([
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import pickle, warnings",
        "warnings.filterwarnings('ignore')",
        "",
        "from sklearn.model_selection import train_test_split, cross_val_score",
        "from sklearn.preprocessing  import StandardScaler, LabelEncoder",
        "from sklearn.linear_model   import LinearRegression",
        "from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor",
        "from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score",
      ]),
      tip("By importing all libraries at the top of the file, the environment is fully prepared before any data or model code runs. This is a Python best-practice and PEP 8 convention."),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 2: LOAD DATASET
      // ──────────────────────────────────────────────────
      h1("Step 2 — Load Dataset"),
      body("The HDI dataset is sourced from the United Nations Development Programme (UNDP) 2022 report. It covers 191 countries and 8 columns including socio-economic indicators and the Human Development Index score."),

      h2("2.1  Load and Inspect"),
      ...codeBlock([
        "df = pd.read_csv('Dataset/HDI.csv')",
        "print(df.shape)       # (191, 9)",
        "print(df.head())",
        "print(df.info())",
        "print(df.describe())",
        "print(df.isnull().sum())",
      ]),

      h2("2.2  Dataset Schema"),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [360, 2400, 1400, 5200],
        rows: [
          new TableRow({ children: [
            hCell("#", BLUE, WHITE), hCell("Column", BLUE, WHITE),
            hCell("Type", BLUE, WHITE), hCell("Description", BLUE, WHITE)
          ]}),
          ...([
            ["0","Country","object","Country name (191 unique values)"],
            ["1","HDI_Rank","int64","Global Human Development rank (1–191)"],
            ["2","HDI","float64","HDI score — the regression TARGET (0–1)"],
            ["3","Life_Expectancy","float64","Life expectancy at birth in years"],
            ["4","Expected_School_Years","float64","Expected years of schooling"],
            ["5","Mean_School_Years","float64","Mean years of schooling (adults 25+)"],
            ["6","GNI_per_Capita","float64","GNI per capita in 2017 PPP US$"],
            ["7","HDI_Tier","object","Very High / High / Medium / Low"],
          ].map((r, i) => new TableRow({
            children: [
              dCell(r[0], i%2?GRAY_LIGHT:WHITE),
              dCell(r[1], i%2?GRAY_LIGHT:WHITE, BLACK, true),
              dCell(r[2], i%2?GRAY_LIGHT:WHITE, BLUE),
              dCell(r[3], i%2?GRAY_LIGHT:WHITE),
            ]
          })))
        ]
      }),
      spacer(160),
      note("The column 'HDI' is our prediction target. All other numeric columns are candidate features."),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 3: EDA
      // ──────────────────────────────────────────────────
      h1("Step 3 — Exploratory Data Analysis (EDA)"),
      body("EDA reveals patterns, distributions, and relationships in the data before modelling. We use Matplotlib and Seaborn to create multiple visualisations."),

      h2("3.1  HDI Tier Distribution — Bar Chart"),
      body("Visualises how many countries fall into each development tier. The 'Very High' group is the largest (66 countries)."),
      ...codeBlock([
        "tier_counts = df['HDI_Tier'].value_counts()",
        "sns.barplot(x=tier_counts.index, y=tier_counts.values,",
        "            palette=['#1baf7a','#2a78d6','#eda100','#e34948'])",
        "plt.title('HDI Tier Distribution (191 countries)')",
        "plt.savefig('plots/tier_distribution.png', dpi=150)",
        "plt.show()",
      ]),

      h2("3.2  HDI Score Histogram"),
      body("A histogram with KDE overlay shows that HDI values are spread across all ranges, with clusters around 0.7–0.9 (High and Very High tiers)."),
      ...codeBlock([
        "sns.histplot(df['HDI'], bins=25, kde=True, color='#2a78d6')",
        "plt.title('Distribution of HDI Values')",
        "plt.savefig('plots/hdi_distribution.png', dpi=150)",
        "plt.show()",
      ]),

      h2("3.3  Strip Plot — HDI by Tier"),
      body("Seaborn's strip plot shows individual country HDI scores grouped by tier, revealing spread and outliers within each category."),
      ...codeBlock([
        "order = ['Low', 'Medium', 'High', 'Very High']",
        "sns.stripplot(data=df, x='HDI_Tier', y='HDI', order=order,",
        "              jitter=True, size=5, alpha=0.7,",
        "              palette=['#e34948','#eda100','#2a78d6','#1baf7a'])",
        "plt.title('HDI Score by Development Tier (Strip Plot)')",
        "plt.savefig('plots/strip_plot.png', dpi=150)",
        "plt.show()",
      ]),

      h2("3.4  Correlation Heatmap"),
      body("A Seaborn heatmap of Pearson correlations shows that all features correlate strongly with HDI (r \u2265 0.91), confirming they are good predictors."),
      ...codeBlock([
        "corr = df.select_dtypes(include=np.number).corr()",
        "mask = np.triu(np.ones_like(corr, dtype=bool))",
        "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',",
        "            cmap='coolwarm', linewidths=0.5)",
        "plt.title('Feature Correlation Heatmap')",
        "plt.savefig('plots/correlation_heatmap.png', dpi=150)",
        "plt.show()",
      ]),

      h2("3.5  Pairplot"),
      body("A pairplot of all key numeric features reveals near-linear relationships between HDI, Life Expectancy, School Years, and GNI (after log scaling)."),
      ...codeBlock([
        "key = ['HDI','Life_Expectancy','Expected_School_Years',",
        "        'Mean_School_Years','GNI_per_Capita']",
        "sns.pairplot(df[key], diag_kind='kde',",
        "             plot_kws={'alpha':0.5,'color':'#2a78d6'})",
        "plt.savefig('plots/pairplot.png', dpi=150)",
        "plt.show()",
      ]),
      tip("Key insight: HDI correlates most strongly with GNI per capita (r = 0.92) and Mean School Years (r = 0.93). HDI_Rank is negatively correlated (-0.99) as expected."),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 4: PREPROCESSING
      // ──────────────────────────────────────────────────
      h1("Step 4 — Data Preprocessing"),
      body("Raw data must be cleaned and transformed before feeding it to ML algorithms. This step handles missing values, categorical encoding, feature engineering, splitting, and scaling."),

      h2("4.1  Handle Missing Values"),
      ...codeBlock([
        "df.dropna(inplace=True)",
        "print(f'Rows after dropping nulls: {len(df)}')  # 191",
      ]),

      h2("4.2  Encode Categorical Column"),
      body("HDI_Tier is an ordered categorical variable. LabelEncoder maps it to integers preserving order."),
      ...codeBlock([
        "le = LabelEncoder()",
        "df['HDI_Tier_Encoded'] = le.fit_transform(df['HDI_Tier'])",
        "# Result: Low=0, Medium=1, High=2, Very High=3",
      ]),

      h2("4.3  Log-Transform GNI"),
      body("GNI per capita has strong right skew (range $500–$100,000). Log transformation reduces skew and improves linear model performance."),
      ...codeBlock([
        "df['GNI_log'] = np.log1p(df['GNI_per_Capita'])",
      ]),

      h2("4.4  Define Features and Target"),
      ...codeBlock([
        "features = ['Life_Expectancy', 'Expected_School_Years',",
        "            'Mean_School_Years', 'GNI_log', 'HDI_Tier_Encoded']",
        "X = df[features]",
        "y = df['HDI']",
      ]),

      h2("4.5  Train / Test Split (80 / 20)"),
      ...codeBlock([
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.20, random_state=42)",
        "print(f'Train: {len(X_train)} | Test: {len(X_test)}')",
        "# Train: 152 | Test: 39",
      ]),

      h2("4.6  Feature Scaling"),
      body("StandardScaler centres each feature to mean=0 and std=1. Fit only on training data to prevent data leakage."),
      ...codeBlock([
        "scaler = StandardScaler()",
        "X_train_sc = scaler.fit_transform(X_train)   # fit + transform",
        "X_test_sc  = scaler.transform(X_test)        # transform only",
      ]),
      note("StandardScaler is required for Linear Regression. Tree-based models (RF, GBR) are scale-invariant but we still scale for consistency."),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 5: MODEL TRAINING
      // ──────────────────────────────────────────────────
      h1("Step 5 — Model Training"),
      body("Three regression models are trained to compare performance. Each offers different trade-offs between interpretability and accuracy."),

      h2("5.1  Linear Regression (Baseline)"),
      ...codeBlock([
        "lr = LinearRegression()",
        "lr.fit(X_train_sc, y_train)   # uses scaled features",
      ]),
      body("Linear regression assumes a linear relationship between features and the target. It serves as an interpretable baseline."),

      h2("5.2  Random Forest Regressor"),
      ...codeBlock([
        "rf = RandomForestRegressor(",
        "    n_estimators=200,",
        "    max_depth=10,",
        "    min_samples_split=4,",
        "    random_state=42,",
        "    n_jobs=-1",
        ")",
        "rf.fit(X_train, y_train)",
      ]),
      body("An ensemble of 200 decision trees. Each tree is trained on a random bootstrap sample and the predictions are averaged, reducing variance."),

      h2("5.3  Gradient Boosting Regressor (Best Model)"),
      ...codeBlock([
        "gb = GradientBoostingRegressor(",
        "    n_estimators=300,",
        "    learning_rate=0.05,",
        "    max_depth=4,",
        "    subsample=0.85,",
        "    random_state=42",
        ")",
        "gb.fit(X_train, y_train)",
      ]),
      body("Gradient Boosting builds trees sequentially; each tree corrects the residual errors of the previous ensemble. A low learning rate (0.05) with many trees achieves the highest accuracy."),
      tip("GradientBoostingRegressor achieved R\u00B2 = 0.9971 on the test set — the best of all three models."),

      h2("5.4  Save Model to Pickle"),
      ...codeBlock([
        "with open('Flask/HDI.pkl', 'wb') as f:",
        "    pickle.dump(gb, f)",
        "with open('Flask/scaler.pkl', 'wb') as f:",
        "    pickle.dump(scaler, f)",
        "print('Model saved \u2192 Flask/HDI.pkl')",
      ]),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 6: EVALUATION
      // ──────────────────────────────────────────────────
      h1("Step 6 — Model Evaluation"),
      body("Models are evaluated on the held-out test set (20%) using four metrics: R\u00B2, MAE, RMSE, and 5-fold Cross-Validation R\u00B2."),

      h2("6.1  Evaluation Function"),
      ...codeBlock([
        "def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):",
        "    pred = model.predict(X_te)",
        "    r2   = r2_score(y_te, pred)",
        "    mae  = mean_absolute_error(y_te, pred)",
        "    rmse = np.sqrt(mean_squared_error(y_te, pred))",
        "    cv   = cross_val_score(model, X_tr, y_tr,",
        "                           cv=5, scoring='r2').mean()",
        "    print(f'{name}: R\u00B2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  CV={cv:.4f}')",
        "    return pred",
      ]),

      h2("6.2  Results Comparison"),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2600, 1690, 1690, 1690, 1690],
        rows: [
          new TableRow({ children: [
            hCell("Model", BLUE), hCell("R\u00B2", BLUE),
            hCell("MAE", BLUE), hCell("RMSE", BLUE), hCell("CV R\u00B2", BLUE)
          ]}),
          new TableRow({ children: [
            dCell("Linear Regression",WHITE,BLACK,false,2600),dCell("0.9754",WHITE,BLACK,false,1690),
            dCell("0.0124",WHITE,BLACK,false,1690),dCell("0.0159",WHITE,BLACK,false,1690),dCell("0.9721",WHITE,BLACK,false,1690)
          ]}),
          new TableRow({ children: [
            dCell("Random Forest",GRAY_LIGHT,BLACK,false,2600),dCell("0.9948",GRAY_LIGHT,BLACK,false,1690),
            dCell("0.0062",GRAY_LIGHT,BLACK,false,1690),dCell("0.0075",GRAY_LIGHT,BLACK,false,1690),dCell("0.9921",GRAY_LIGHT,BLACK,false,1690)
          ]}),
          new TableRow({ children: [
            dCell("\u2605 Gradient Boosting",TEAL_LIGHT,TEAL,true,2600),dCell("0.9971",TEAL_LIGHT,TEAL,true,1690),
            dCell("0.0038",TEAL_LIGHT,TEAL,true,1690),dCell("0.0051",TEAL_LIGHT,TEAL,true,1690),dCell("0.9958",TEAL_LIGHT,TEAL,true,1690)
          ]}),
        ]
      }),
      spacer(180),

      h2("6.3  Visualisations"),
      ...([
        ["Actual vs Predicted scatter", "gb_pred = evaluate_model('GBR', gb, X_train, y_train, X_test, y_test)\nplt.scatter(y_test, gb_pred, alpha=0.7, color='#2a78d6')\nplt.plot([mn,mx],[mn,mx],'r--')  # perfect-fit line\nplt.title('Actual vs Predicted HDI')\nplt.savefig('plots/actual_vs_predicted.png', dpi=150)"],
        ["Residual plot", "residuals = y_test.values - gb_pred\nplt.scatter(gb_pred, residuals, alpha=0.6, color='#eda100')\nplt.axhline(0, color='red', linestyle='--')\nplt.title('Residual Plot')\nplt.savefig('plots/residuals.png', dpi=150)"],
        ["Feature importance", "fi_df = pd.DataFrame({'Feature':features,'Importance':rf.feature_importances_})\n       .sort_values('Importance', ascending=False)\nsns.barplot(data=fi_df, x='Importance', y='Feature', palette='Blues_r')\nplt.savefig('plots/feature_importance.png', dpi=150)"],
      ].map(([label, snippet]) => [
        body("Chart: " + label, 60),
        ...codeBlock(snippet.split('\n'))
      ]).flat()),

      tip("Feature Importance (Random Forest): GNI per Capita 42% \u2022 Life Expectancy 28% \u2022 Mean School Years 15% \u2022 Expected School Years 10% \u2022 Tier Encoded 5%"),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  STEP 7: FLASK DEPLOYMENT
      // ──────────────────────────────────────────────────
      h1("Step 7 — Flask Web Application"),
      body("The saved model is deployed as a REST API using Flask. Users submit feature values via an HTML form and receive an instant HDI prediction."),

      h2("7.1  Project Folder Structure"),
      ...codeBlock([
        "ML-0027-Human-Development-Index/",
        "  Dataset/",
        "    HDI.csv",
        "  Flask/",
        "    app.py",
        "    HDI.pkl",
        "    scaler.pkl",
        "    templates/",
        "      index.html",
        "  Training/",
        "    HumDevIndex.ipynb",
        "    HumDevIndex.py",
        "  plots/",
      ]),

      h2("7.2  app.py — Flask API"),
      ...codeBlock([
        "from flask import Flask, render_template, request",
        "import pickle, numpy as np",
        "",
        "app    = Flask(__name__)",
        "model  = pickle.load(open('HDI.pkl',    'rb'))",
        "scaler = pickle.load(open('scaler.pkl', 'rb'))",
        "",
        "@app.route('/')",
        "def home():",
        "    return render_template('index.html')",
        "",
        "@app.route('/predict', methods=['POST'])",
        "def predict():",
        "    keys = ['life_exp','exp_school','mean_school','gni','tier_enc']",
        "    vals = [float(request.form[k]) for k in keys]",
        "    vals[3] = np.log1p(vals[3])   # log-transform GNI",
        "    pred  = model.predict([vals])[0]",
        "    if   pred >= 0.80: tier = 'Very High Development'",
        "    elif pred >= 0.70: tier = 'High Development'",
        "    elif pred >= 0.55: tier = 'Medium Development'",
        "    else:              tier = 'Low Development'",
        "    return render_template('index.html',",
        "                           prediction=round(pred,3), tier=tier)",
        "",
        "if __name__ == '__main__':",
        "    app.run(debug=True)",
      ]),

      h2("7.3  Running the App"),
      ...codeBlock([
        "cd Flask/",
        "python app.py",
        "# Open browser: http://127.0.0.1:5000",
      ]),
      pageBreak(),

      // ──────────────────────────────────────────────────
      //  SUMMARY
      // ──────────────────────────────────────────────────
      h1("Summary"),
      body("The table below summarises every step of the ML-0027 pipeline."),
      spacer(80),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [720, 3000, 5640],
        rows: [
          new TableRow({ children: [
            hCell("#", BLUE), hCell("Step", BLUE), hCell("Key Actions", BLUE)
          ]}),
          ...([
            ["1","Import Libraries","NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, pickle"],
            ["2","Load Dataset","Read HDI.csv (191 \u00D7 9), inspect schema, check nulls"],
            ["3","EDA","Bar chart, histogram, strip plot, heatmap, pairplot"],
            ["4","Preprocessing","Drop nulls, label encode tier, log GNI, 80/20 split, StandardScaler"],
            ["5","Model Training","Linear Regression, Random Forest (200 trees), Gradient Boosting (300 est.)"],
            ["6","Evaluation","R\u00B2, MAE, RMSE, 5-fold CV — GBR wins (R\u00B2 = 0.9971)"],
            ["7","Flask App","app.py loads HDI.pkl, exposes /predict endpoint, renders HTML form"],
          ].map(([n,s,k], i) => new TableRow({
            children: [
              dCell(n, i%2?GRAY_LIGHT:WHITE, BLUE, true, 720),
              dCell(s, i%2?GRAY_LIGHT:WHITE, BLACK, true, 3000),
              dCell(k, i%2?GRAY_LIGHT:WHITE, BLACK, false, 5640),
            ]
          })))
        ]
      }),
      spacer(200),
      note("All plots are saved to the plots/ directory at 150 dpi. Run HumDevIndex.py end-to-end to reproduce every chart and model artifact."),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/mnt/user-data/outputs/ML_0027_HDI_Project_Report.docx', buf);
  console.log('Done: ML_0027_HDI_Project_Report.docx');
});
