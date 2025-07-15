# Forest Fire Spread Simulator – ISRO Hackathon (Real-Time Data + Weather)
# ----------------------------------------------------------------------
# ENHANCEMENTS:
#   • Real-time Fire Data Integration (Mocked ISRO API)
#   • Weather API (OpenWeatherMap placeholder logic)
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from io import BytesIO
import imageio.v2 as imageio
import requests
import json

try:
    import rasterio
except ImportError:
    rasterio = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="ISRO Forest Fire Simulator", layout="wide")
st.title("🚀 ISRO Hackathon • Forest Fire Simulation + Real-Time Data")

# ---------------- Sidebar ------------------------
with st.sidebar:
    st.header("Simulation Parameters")
    GRID_SIZE = st.slider("Grid Size", 50, 400, 200, 10)
    IGNITIONS = st.slider("Synthetic Ignition Points", 1, 20, 6)
    TIME_STEPS = st.slider("Time Steps", 20, 400, 150, 10)
    RAIN_INT = st.slider("Rain Interval", 5, 120, 30, 5)
    URBAN_RATIO = st.slider("Urban/Firebreak %", 0, 30, 6)

    st.markdown("---")
    st.subheader("📡 Real-Time Fire Data")
    use_live_fire = st.checkbox("Enable Live Fire Points (Mocked ISRO API)")

    st.subheader("☁️ Weather Settings")
    use_weather = st.checkbox("Enable Real-Time Weather")
    weather_city = st.text_input("City (for weather API)", "Bangalore")

    st.markdown("---")
    gif_opt = st.checkbox("Generate GIF")

    st.markdown("---")
    st.header("Data Inputs (optional)")
    ndvi_file = st.file_uploader("NDVI GeoTIFF", ["tif","tiff"])
    dem_file  = st.file_uploader("DEM GeoTIFF",  ["tif","tiff"])
    fire_csv  = st.file_uploader("Historical Fire CSV (x,y,label)")
    run_btn = st.button("🔥 Run Simulation")

np.random.seed(42)

# ---------------- Constants -----------------------
EMPTY, TREE, BURNING, URBAN = 0, 1, 2, 3
CMAP = plt.cm.get_cmap("RdYlGn", 4)

# ---------------- Helper Functions ----------------
def load_geotiff(upload):
    if not rasterio:
        st.warning("rasterio not installed; using synthetic data")
        return None
    try:
        with rasterio.open(upload) as src:
            arr = src.read(1).astype(np.float32)
        arr = arr[:GRID_SIZE, :GRID_SIZE]
        return (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-6)
    except Exception as e:
        st.error(f"GeoTIFF error: {e}")
        return None

def synthetic_ndvi():
    base = np.zeros((GRID_SIZE, GRID_SIZE))
    for _ in range(8):
        cx, cy = np.random.randint(0, GRID_SIZE, 2)
        xv, yv = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
        base += np.exp(-((xv-cx)**2 + (yv-cy)**2)/(2*20**2))
    return (base - base.min())/(base.max()-base.min())

def synthetic_slope():
    elev = np.zeros((GRID_SIZE, GRID_SIZE))
    for _ in range(5):
        hx, hy = np.random.randint(0, GRID_SIZE, 2)
        h = np.random.uniform(40, 120)
        xv, yv = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing="ij")
        elev += h*np.exp(-((xv-hx)**2 + (yv-hy)**2)/(2*30**2))
    gy, gx = np.gradient(elev)
    slope = np.sqrt(gx**2 + gy**2)
    slope = (slope - slope.min())/(slope.max()-slope.min())
    return slope, gx, gy

def fetch_weather(city):
    try:
        api_key = "demo"  # Replace with real API key
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            return data['wind']['speed'], data['main']['humidity'] / 100
    except:
        pass
    return 1.5, 0.4  # Defaults

def wind_field(speed=1.5):
    return speed + 0.5*(np.random.rand(GRID_SIZE,GRID_SIZE)-0.5), 0.4*(np.random.rand(GRID_SIZE,GRID_SIZE)-0.5)

def humidity_field(base=0.4):
    return np.clip(np.random.normal(base, 0.1, (GRID_SIZE, GRID_SIZE)), 0.1, 0.9)

def urban_mask():
    mask = np.zeros((GRID_SIZE, GRID_SIZE), bool)
    count = int((URBAN_RATIO/100)*GRID_SIZE**2)
    mask[np.random.randint(0,GRID_SIZE,count), np.random.randint(0,GRID_SIZE,count)] = True
    return mask

def neighbors(arr,x,y):
    return arr[max(0,x-1):x+2, max(0,y-1):y+2]

def dir_bonus(u,v,dx,dy):
    wm, sm = np.hypot(u,v), np.hypot(dx,dy)
    return 1+0.5*max((u*dx+v*dy)/(wm*sm),0) if wm and sm else 1

def spread_prob(cell, neigh, u,v, s,h, dx,dy):
    if cell!=TREE or np.sum(neigh==BURNING)==0:
        return 0.
    base=0.25+0.07*np.sum(neigh==BURNING)
    return np.clip((base + 0.05*np.hypot(u,v) + 0.3*s - 0.5*h)*dir_bonus(u,v,dx,dy),0,1)

def fetch_live_fires():
    # Replace with actual ISRO API later
    if not use_live_fire:
        return []
    return [(np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)) for _ in range(IGNITIONS)]

def train_risk(ndvi,slope,fire_df=None):
    X = np.column_stack([ndvi.flatten(), slope.flatten()])
    if fire_df is not None:
        y = np.zeros(ndvi.size, int)
        for _,r in fire_df.iterrows():
            if 0<=r['x']<GRID_SIZE and 0<=r['y']<GRID_SIZE:
                y[int(r['x'])*GRID_SIZE+int(r['y'])] = int(r['label'])
    else:
        y = ((ndvi>0.6)&(slope>0.4)).astype(int).flatten()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
    rf = RandomForestClassifier(n_estimators=120, random_state=42).fit(Xtr,ytr)
    rep = classification_report(yte,rf.predict(Xte),output_dict=True)
    cm  = confusion_matrix(yte,rf.predict(Xte)).tolist()
    prob = rf.predict_proba(X)[:,1].reshape(GRID_SIZE,GRID_SIZE)
    return prob, rep, cm

# ---------------- Simulation ---------------------
def run_sim():
    ndvi = load_geotiff(ndvi_file) if ndvi_file else synthetic_ndvi()
    if dem_file:
        dem = load_geotiff(dem_file)
        if dem is not None:
            gy,gx=np.gradient(dem); slope=np.sqrt(gx**2+gy**2); slope=(slope-slope.min())/(slope.max()-slope.min())
        else:
            slope,gx,gy = synthetic_slope()
    else:
        slope,gx,gy = synthetic_slope()

    fire_df = None
    if fire_csv:
        try:
            fire_df = pd.read_csv(fire_csv)
            st.success("CSV loaded")
        except:
            st.error("Failed to read fire CSV")

    risk_map, rep, cm = train_risk(ndvi, slope, fire_df)

    urb = urban_mask()
    forest = np.where(ndvi>0.35, TREE, EMPTY); forest[urb] = URBAN

    ign_pts = fetch_live_fires() if use_live_fire else []
    if fire_df is not None:
        ign_pts += [(int(x), int(y)) for x, y, l in fire_df.values if l == 1 and not urb[int(x), int(y)]]
    if len(ign_pts) < IGNITIONS:
        veg = np.argwhere((ndvi > 0.6) & (~urb))
        extra = veg[np.random.choice(len(veg), IGNITIONS - len(ign_pts), replace=False)]
        ign_pts += list(map(tuple, extra))
    for x, y in ign_pts:
        forest[x, y] = BURNING

    wind_speed, base_humidity = fetch_weather(weather_city) if use_weather else (1.5, 0.4)
    wind_u, wind_v = wind_field(wind_speed)
    humidity = humidity_field(base_humidity)

    frames, burn_hist = [], []
    placeholder = st.empty()

    for t in range(TIME_STEPS):
        humidity = np.clip(humidity + np.random.uniform(-0.02, 0.02, humidity.shape), 0.1, 0.95)
        if t % RAIN_INT == 0 and t != 0:
            band = slice((t*3)%GRID_SIZE, ((t*3)%GRID_SIZE)+25)
            humidity[:, band] = np.clip(humidity[:, band]+0.35, 0, 1)

        new_forest = forest.copy()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if forest[i,j] == TREE:
                    p = spread_prob(forest[i,j], neighbors(forest,i,j), wind_u[i,j], wind_v[i,j], slope[i,j], humidity[i,j], gx[i,j], gy[i,j])
                    if np.random.rand() < p:
                        new_forest[i,j] = BURNING
                elif forest[i,j] == BURNING:
                    new_forest[i,j] = EMPTY

        forest = new_forest
        burned = np.mean(forest == EMPTY)
        burn_hist.append(burned)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(forest, cmap=CMAP, vmin=0, vmax=3)
        ax1.axis('off'); ax1.set_title(f"t={t} Burned {burned*100:.1f}%")
        im = ax2.imshow(risk_map, cmap='hot', vmin=0, vmax=1)
        ax2.axis('off'); ax2.set_title('Risk Map')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        placeholder.pyplot(fig); plt.close(fig)

        if gif_opt:
            buf = BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
            frames.append(imageio.imread(buf)); buf.close()

    st.subheader("🔥 Burned Fraction Over Time")
    st.line_chart(pd.Series(burn_hist))

    st.subheader("📊 Classifier Report")
    st.json(rep)
    st.write("Confusion Matrix", cm)

    csv_buf = BytesIO(); pd.DataFrame({'t': range(TIME_STEPS), 'burned': burn_hist}).to_csv(csv_buf, index=False); csv_buf.seek(0)
    st.download_button("📄 Download Burn CSV", csv_buf, "burned.csv", "text/csv")

    if gif_opt and frames:
        gif_buf = BytesIO()
        imageio.mimsave(gif_buf, frames, fps=4, format='GIF')
        gif_buf.seek(0)
        st.download_button("📽 Download GIF", gif_buf, "simulation.gif", "image/gif")

# ---------------- Run Button ---------------------
if run_btn:
    run_sim()
else:
    st.info("Set parameters and click Run Simulation")
