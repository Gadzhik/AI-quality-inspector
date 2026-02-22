from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/telemetry/defect', methods=['POST'])
def receive_defect():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        defect_type = data.get("type", "Unknown")
        if data.get("defect") and defect_type == "Unknown":
             defect_type = "General Defect"
            
        confidence = data.get("confidence", "N/A")
        inference_time = data.get("inference_time", "N/A")
        
        print(f"[TELEMETRY RECEIVE] Обнаружен дефект: {defect_type} | Уверенность: {confidence}% | Время: {inference_time}s")
        
        return jsonify({"status": "received"}), 200
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Mock Telemetry Server on port 8080...")
    app.run(host='0.0.0.0', port=8080)
