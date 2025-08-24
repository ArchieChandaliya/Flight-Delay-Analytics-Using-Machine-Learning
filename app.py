import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, jsonify, render_template, request # Import 'request' for handling POST data

# --- Data Loading and Preprocessing (Your original code) ---
df = pd.read_excel("honeywell_data.xlsx")

def try_parse_datetime(x):
    try:
        return pd.to_datetime(x)
    except:
        return pd.NaT

df['flight_dt'] = df['Flight time'].apply(try_parse_datetime)

if df['flight_dt'].isna().any():
    def extract_date_from_string(s):
        try:
            s = str(s)
            tokens = s.split()
            for t in tokens:
                try:
                    return pd.to_datetime(t, dayfirst=True)
                except:
                    try:
                        return pd.to_datetime(t)
                    except:
                        continue
            return pd.NaT
        except:
            return pd.NaT
    df['flight_dt'] = df['flight_dt'].fillna(df['Flight time'].apply(extract_date_from_string))

time_cols = ['STD', 'ATD', 'STA', 'ATA']
for c in time_cols:
    if c in df.columns:
        parsed = pd.to_datetime(df[c], errors='coerce')
        df[c + '_parsed'] = parsed

for c in time_cols:
    parsed_col = c + '_parsed'
    if parsed_col in df.columns:
        mask_time_strings = df[parsed_col].isna() & df[c].notna()
        def combine_date_time(row, col=c):
            dt_base = row['flight_dt']
            t_str = str(row[col]).strip()
            try:
                t = pd.to_datetime(t_str, format='%H:%M').time()
            except:
                try:
                    t = pd.to_datetime(t_str).time()
                except:
                    return pd.NaT
            if pd.isna(dt_base):
                base = pd.to_datetime('today').normalize()
            else:
                base = pd.to_datetime(dt_base).normalize()
            return pd.to_datetime(datetime.combine(base, t))
        df.loc[mask_time_strings, parsed_col] = df[mask_time_strings].apply(combine_date_time, axis=1)

for c in time_cols:
    p = c + '_parsed'
    if p in df.columns:
        df[p] = pd.to_datetime(df[p], errors='coerce')

if 'STD_parsed' in df.columns and 'ATD_parsed' in df.columns:
    df['dep_delay_min'] = (df['ATD_parsed'] - df['STD_parsed']).dt.total_seconds() / 60.0
else:
    df['dep_delay_min'] = np.nan

if 'STA_parsed' in df.columns and 'ATA_parsed' in df.columns:
    df['arr_delay_min'] = (df['ATA_parsed'] - df['STA_parsed']).dt.total_seconds() / 60.0
else:
    df['arr_delay_min'] = np.nan

df['dep_delay_min'] = df['dep_delay_min'].replace([np.inf, -np.inf], np.nan)
df['arr_delay_min'] = df['arr_delay_min'].replace([np.inf, -np.inf], np.nan)

df['sched_hour'] = df['STD_parsed'].dt.hour
df['sched_date'] = df['STD_parsed'].dt.date

flights_by_hour = df.groupby('sched_hour').size().rename('num_flights').reset_index()
avg_delay_hour = df.groupby('sched_hour')['dep_delay_min'].mean().reset_index().rename(columns={'dep_delay_min':'avg_dep_delay'})
hour_stats = flights_by_hour.merge(avg_delay_hour, on='sched_hour', how='outer').sort_values('sched_hour').fillna(0)

best_hour = hour_stats.loc[hour_stats['avg_dep_delay'].idxmin()]
worst_hour = hour_stats.loc[hour_stats['avg_dep_delay'].idxmax()]
top_busiest = hour_stats.sort_values('num_flights', ascending=False).head(5)

model_df = df.dropna(subset=['dep_delay_min']).copy()

cat_features = []
for c in ['From','To','Aircraft']:
    if c in model_df.columns:
        cat_features.append(c)
num_features = ['sched_hour']
features = num_features + cat_features # Combine features for the model

X = model_df[features]
y = model_df['dep_delay_min'].clip(lower=-120, upper=720) # Clip extreme delays

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

pipe = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# --- Cascade Analysis ---
df_sorted = df.sort_values(['Aircraft','STD_parsed'])
cascade_scores = []
for aircraft, group in df_sorted.groupby('Aircraft'):
    group = group.sort_values('STD_parsed').reset_index(drop=True)
    for i in range(len(group)):
        curr_fid = group.loc[i, 'Flight Number']
        downstream = group.loc[i+1:]
        cascade_impact = downstream['dep_delay_min'].fillna(0).sum()
        cascade_scores.append({
            'Aircraft': aircraft,
            'FlightNumber': curr_fid,
            'STD': group.loc[i,'STD_parsed'],
            'dep_delay_min': group.loc[i,'dep_delay_min'],
            'cascade_impact_sum_min': cascade_impact
        })
cascade_df = pd.DataFrame(cascade_scores).sort_values('cascade_impact_sum_min', ascending=False)
top_cascade = cascade_df.head(5)

# --- Flask Application Setup ---
app = Flask(__name__)

# Route to serve the main HTML page
@app.route('/')
def home():
    # Get unique values for dropdowns to pass to the frontend
    unique_from = df['From'].dropna().unique().tolist() if 'From' in df.columns else []
    unique_to = df['To'].dropna().unique().tolist() if 'To' in df.columns else []
    unique_aircraft = df['Aircraft'].dropna().unique().tolist() if 'Aircraft' in df.columns else []

    return render_template('index.html', 
                           unique_from=unique_from, 
                           unique_to=unique_to, 
                           unique_aircraft=unique_aircraft)

# API endpoint to get the dashboard data
@app.route('/api/dashboard_data')
def dashboard_data():
    best = f"Best scheduled hour: {int(best_hour['sched_hour'])}:00"
    worst = f"Worst scheduled hour: {int(worst_hour['sched_hour'])}:00"
    busiest = ", ".join([f"{int(x)}:00" for x in top_busiest['sched_hour'].tolist()])
    cascade_list = "; ".join([f"{r['FlightNumber']} (impact {int(r['cascade_impact_sum_min'])} min)" for _, r in top_cascade.iterrows()])

    data = {
        'best_hour': best,
        'worst_hour': worst,
        'busiest_hours': busiest,
        'cascading_flights': cascade_list,
        'hour_stats': hour_stats.to_dict('records')
    }
    return jsonify(data)

# API endpoint for flight delay prediction
@app.route('/api/predict_delay', methods=['POST'])
def predict_delay():
    try:
        data = request.get_json()
        from_airport = data.get('from_airport')
        to_airport = data.get('to_airport')
        aircraft_type = data.get('aircraft_type')
        scheduled_hour = int(data.get('scheduled_hour'))

        # Create a DataFrame for prediction, ensuring column names match training features
        # Provide default/dummy values for any missing categorical features if needed
        # (Though handle_unknown='ignore' in OneHotEncoder should handle unseen categories)
        
        input_data = {
            'From': from_airport,
            'To': to_airport,
            'Aircraft': aircraft_type,
            'sched_hour': scheduled_hour
        }
        
        # Ensure all features expected by the model are present in the input_df
        # Create a single row DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        predicted_delay = pipe.predict(input_df)[0]
        
        return jsonify({
            'from_airport': from_airport,
            'to_airport': to_airport,
            'aircraft_type': aircraft_type,
            'scheduled_hour': scheduled_hour,
            'predicted_delay_min': round(predicted_delay, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to generate and serve the heatmap image
@app.route('/api/heatmap_image')
def heatmap_image():
    if 'sched_date' in df.columns:
        pivot = df.pivot_table(index=df['sched_date'], columns='sched_hour', values='dep_delay_min', aggfunc='mean')
        
        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
        plt.title("Average Departure Delay (min): Date vs. Hour")
        
        img_path = 'static/heatmap.png'
        plt.savefig(img_path)
        plt.close() # Close the figure to free up memory
        
        return jsonify({'image_url': img_path})
    return jsonify({'error': 'Heatmap data not available'}), 404

if __name__ == '__main__':
    app.run(debug=True)