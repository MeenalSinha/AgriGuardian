# 🌾 AgriGuardian – AI-Powered Agricultural Intelligence System

**AgriGuardian** is an AI-powered agricultural assistant designed to empower farmers with early crop disease detection, real-time health monitoring, and multilingual voice-based recommendations.  
Built using **Streamlit, OpenAI, Firebase, TensorFlow, and gTTS**, it bridges the gap between cutting-edge AI and accessible farming technology for sustainable agriculture.

---

## 🚀 Overview

Modern farming faces unpredictable challenges — from climate variability to late disease detection.  
AgriGuardian helps farmers make **data-driven decisions** by analyzing crop images, predicting disease risks, and providing **personalized treatment guidance** in their **local language** through an intuitive web dashboard.

---

## 🧠 Core Features

- 🖼️ **AI-Based Crop Disease Detection**  
  Upload crop images to detect potential diseases using a trained CNN model (PlantVillage dataset + custom field data).

- 🔍 **Severity Analysis**  
  Calculates disease severity scores using environmental and image-based metrics.

- 🌦️ **Weather-Aware Insights**  
  Integrates real-time weather data for contextual recommendations.

- 🔊 **Multilingual Voice Output**  
  Uses gTTS to deliver spoken recommendations in multiple regional languages.

- 📊 **Interactive Dashboard**  
  Real-time visualization of health metrics, risk heatmaps, and crop condition analytics via Streamlit.

- ☁️ **Firebase Integration**  
  Stores prediction data, logs user inputs, and enables potential future multi-user access.

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend / UI** | Streamlit |
| **Backend / Logic** | Python, TensorFlow, OpenAI |
| **Database** | Firebase Firestore |
| **Voice Interface** | Google Text-to-Speech (gTTS) |
| **Visualization** | Plotly, Folium |
| **Data Handling** | Pandas, NumPy |

---

## 🏗️ System Architecture

```
Farmer → Streamlit Interface → Preprocessing → CNN Model → Prediction & Severity Scoring
                                     ↘ Weather API + Firebase ↙
                          → Voice Output (gTTS) → Dashboard Visualization
```

### Architecture Layers:
1. **Input Layer** – Handles image and data input with validation  
2. **AI Engine** – Performs disease prediction and severity analysis  
3. **Recommendation Engine** – Generates treatment and prevention guidance  
4. **Visualization Layer** – Displays insights and risk maps  
5. **Accessibility Layer** – Converts insights into audio for inclusive access  

---

## ⚙️ Installation & Setup

### **1. Clone Repository**
```bash
git clone https://github.com/username/AgriGuardian.git
cd AgriGuardian
```

### **2. Create Virtual Environment (optional)**
```bash
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run Application**
```bash
streamlit run app.py
```

---

## 🧮 Key Functions

```python
def predict(image):
    img = preprocess(image)
    result = model.predict(img)
    return np.argmax(result)

def _calculate_severity(prob):
    return "High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"

def multilingual_voice_player(text, lang='hi'):
    from gtts import gTTS
    tts = gTTS(text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"
```

---

## 🌦️ Example Weather API Integration

```python
import requests

def get_weather(lat, lon):
    api_key = "YOUR_API_KEY"
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    return response.json()
```

---

## ☁️ Firebase Example

```python
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def save_result(crop, disease, risk):
    db.collection("predictions").add({
        "crop": crop,
        "disease": disease,
        "risk": risk
    })
```

---

## 🧪 Model Details

- **Dataset**: PlantVillage + field data (Punjab Agricultural Stations)
- **Architecture**: CNN (ResNet50 pretrained)
- **Accuracy**: 92% on validation set
- **Average Inference Time**: ~1.2 seconds per image

---

## 💡 Future Enhancements

- 🌱 Integration of soil sensor & satellite data
- 📱 Mobile companion app with offline voice mode
- 🧬 Transfer learning for region-specific crop types
- 🌍 Expanded language support
- ✅ AI verification layer for treatment authenticity

---

## 👩‍💻 Contributors

- **Meenal Sinha** – AI Integration & Interface Design

---

## 🪴 License

This project is licensed under the MIT License.

---

## 🌍 Vision

> "AgriGuardian bridges the gap between intelligence and accessibility — bringing the power of AI to every farm, in every language."
