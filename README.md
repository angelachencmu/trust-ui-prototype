# Trust UI Prototype

This is a prototype of the Trust UI project.
It is a simple web application that aims to collect telemetry data from users to understand the trust between humans and machines.

This repository is developed using Python 3.11.8, but it _should_ work with any Python 3.8+ version.

## How to run

This web application uses [Streamlit](https://streamlit.io/) to run.

First, create a virtual environment and install the dependencies:
```bash
python -m venv ./env
source env/bin/activate
pip install -r requirements.txt
```

Then, run the application:
```bash
streamlit run deployment/app.py
```
