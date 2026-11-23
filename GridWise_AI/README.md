# GridWise AI

Intelligent Energy Optimization and Smart Building Management.

## Features
- Streamlit energy dashboard
- Random Forest Regression energy prediction
- Isolation Forest anomaly detection
- HVAC and lighting what-if simulator
- Rule-based optimization recommendations
- Local AI Energy Assistant fallback
- CSV upload and download

## Recommended setup (Windows)
Use Python 3.10 or 3.11 for a predictable data-science environment.

```bat
cd path\to\GridWise_AI
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run app.py
```

If Python 3.11 is not installed, use `python -m venv venv` instead of `py -3.11 -m venv venv`.

Open the local URL printed by Streamlit, usually http://localhost:8501.

## CSV schema
The uploaded CSV must contain:
`timestamp,building,room,temperature_c,humidity_pct,occupancy,hvac_kw,lighting_kw,equipment_kw,energy_kwh`

- `timestamp`: parseable date/time
- Numeric columns must contain numbers
- `energy_kwh`: energy consumed during the row's interval
- Keep the interval and units consistent

## GitHub
Create an empty repository on GitHub, then run these commands in the project folder:

```bat
git init
git add .
git commit -m "Initial commit - GridWise AI"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the URL with your own repository URL. If `origin` already exists, use:
`git remote set-url origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git`

Do not commit virtual environments, API keys, or private building data.
