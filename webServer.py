from flask import Flask, request, jsonify

app = Flask(__name__)

received_data = {'requestedObject': None, 'distance': None}

@app.route('/send_data', methods=['POST'])
def send_data():
    if request.is_json:
        data = request.get_json()

        if data.get('requestedObject') is not None:
            received_data['requestedObject'] = data['requestedObject']
        elif data.get('distance') is not None:
            received_data['distance'] = data['distance']
        else:
            # If neither requestedObject nor distance is provided, return an error
            return jsonify({"status": "error", "message": "requestedObject or distance is required"}), 400
        
        print("Data received successfully.")
        print("Received data:", received_data)
        return jsonify({"status": "success", "data": received_data}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400
    
@app.route('/get_requested_object', methods=['GET'])
def get_requested_object():
    if received_data['requestedObject'] is not None:
        return jsonify({"requestedObject": received_data['requestedObject']}), 200
    else:
        return jsonify({"status": "error", "message": "No requested object found"}), 404
    
@app.route('/get_distance', methods=['GET'])
def get_distance():
    if received_data['distance'] is not None:
        return jsonify({"distance": received_data['distance']}), 200
    else:
        return jsonify({"status": "error", "message": "No distance found"}), 404
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)