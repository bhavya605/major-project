# ICU Sepsis Research demo (synthetic only)

This isolated Streamlit Cloud package runs the research dashboard using six prebuilt demo models and three generated patient records. **Every included model and patient record is synthetic demonstration material. Nothing here reports clinical or PhysioNet performance, and this app is not for patient care.**

## Deploy

Upload this directory (`deployment/streamlit-cloud`) to a GitHub repository or copy its contents to the repository root configured in Streamlit Community Cloud. Select Python 3.11 in Streamlit Community Cloud and set the app entry point to `app.py`. The package includes its own root-level `sepsis/` copy so imports work without the source project or an editable install.

Streamlit Cloud installs `requirements.txt`; it pins the local serialization/runtime versions, including CPU PyTorch from the official PyTorch wheel index. First build can take several minutes because it includes PyTorch, XGBoost, and SHAP.

## Data and target

The three `data/demo/demo_00*.psv` records were generated from invented signals and labels. `demo_manifest.json` records their synthetic provenance. They are software fixtures only. PhysioNet 2019 SepsisLabel already begins six hours before recorded clinical onset; any additional prediction horizon in the dashboard is an additional shift beyond that published label and must not be mistaken for onset time.

The dashboard warns that scores are unvalidated and labels the displayed replay as synthetic. Do not use outputs for diagnosis, treatment, or clinical decisions.
