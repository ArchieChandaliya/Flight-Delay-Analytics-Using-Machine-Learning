# Flight Delay Analytics Dashboard

This project is a web-based dashboard that provides insights into flight delay data and offers an interactive tool to predict potential flight delays. The backend is built with **Flask**, and the frontend uses standard **HTML, CSS, and JavaScript** with the **Chart.js** library for data visualization.

## Features ✨
- **Key Insights**: Displays pre-calculated metrics like the best and worst times to fly based on historical data.
- **Interactive Prediction**: Allows users to input flight details (origin, destination, aircraft type, and scheduled hour) to get a real-time delay prediction.
- **Visualizations**: Includes charts and heatmaps to visualize flight delay patterns.

## Project Structure 📁
- `app.py`: The main Flask application file. It handles data processing, machine learning model training, and API endpoints.
- `honeywell_data.xlsx`: The raw dataset containing flight information.
- `static/style.css`: The CSS file for styling the web dashboard.
- `templates/index.html`: The main HTML template for the dashboard frontend.

## Setup and Installation 🚀
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd flight-delay-analytics
    ```

2.  **Install dependencies**:
    This project requires `Flask`, `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.
    ```bash
    pip install Flask pandas numpy matplotlib seaborn scikit-learn openpyxl
    ```
    *(Note: `openpyxl` is needed to read the `.xlsx` file)*

3.  **Run the application**:
    ```bash
    python app.py
    ```

4.  **Access the dashboard**:
    Open your web browser and navigate to `http://127.0.0.1:5000`.

## How It Works 🤖
The `app.py` script performs the following key functions:
- **Data Ingestion**: Reads flight data from `honeywell_data.xlsx`.
- **Preprocessing**: Cleans the data, parses date/time columns, and calculates departure and arrival delays.
- **Analytics**: Computes key metrics like average delay by hour and identifies the busiest flights and those with the largest cascading impact.
- **Machine Learning**: Trains a `RandomForestRegressor` model to predict departure delays based on `From` airport, `To` airport, `Aircraft` type, and `Scheduled Hour`. An `OneHotEncoder` is used within a `Pipeline` to handle categorical features.
- **API Endpoints**:
    - `/`: Serves the main dashboard HTML page.
    - `/api/dashboard_data`: Provides JSON data for the key insights and the line chart.
    - `/api/predict_delay`: Accepts a POST request with flight details and returns a predicted delay.
    - `/api/heatmap_image`: Generates and serves the heatmap visualization.

## Model Performance 📊
The machine learning model is trained to predict `dep_delay_min` (departure delay in minutes) and clips extreme values to improve performance. The trained model's performance on the test set can be evaluated by checking the Mean Absolute Error (`MAE`) and R-squared (`R²`) scores within the `app.py` script.

## Contributing 🤝
Feel free to open issues or submit pull requests to improve the project.