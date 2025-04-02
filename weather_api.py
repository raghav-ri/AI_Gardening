import requests

# ✅ Correct API Key
API_KEY = "d11c290e48035730e9aedb8687f1f66a"

def get_weather(city="Patna"):
    """Fetch weather data for a given city"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    try:
        data = response.json()
        if response.status_code == 200:
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            rainfall = data.get("rain", {}).get("1h", 0)  # Default to 0 if no rain data
            print(f"✅ Weather Fetched: {temperature}°C, {humidity}%, {rainfall}mm in {city}")
            return temperature, humidity, rainfall  # ✅ Returns 3 values
        else:
            print(f"❌ API Error: {data.get('message', 'Unknown error')}")
            return None, None, None  # ✅ Always return 3 values
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None, None, None  # ✅ Always return 3 values

# Test API manually
if __name__ == "__main__":
    city_name = input("Enter city name: ")
    print(get_weather(city_name))
