from flask import Flask, request, jsonify

app = Flask(__name__)

received_data = {'requestedObject': None, 'distance': None}

@app.route('/')
def index():
    return "Welcome to the Object Tracking API!"

@app.route('/update_distance', methods=['POST'])
def update_distance():
    if request.is_json:
        data = request.get_json()
        if 'distance' in data:
            received_data['distance'] = data['distance']
            print(f"Distance updated: {received_data['distance']}")
            return jsonify({"status": "success", "message": "Distance updated successfully"}), 200
        else:
            return jsonify({"status": "error", "message": "Distance not provided"}), 400
    else:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400
    
@app.route('/update_requested_object', methods=['POST'])
def update_requested_object():
    if request.is_json:
        data = request.get_json()
        if 'requestedObject' in data:
            received_data['requestedObject'] = data['requestedObject']
            print(f"Requested object updated: {received_data['requestedObject']}")
            return jsonify({"status": "success", "message": "Requested object updated successfully"}), 200
        else:
            return jsonify({"status": "error", "message": "Requested object not provided"}), 400
    else:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400
    
@app.route('/get_requested_object', methods=['GET'])
def get_requested_object():
    print(f"Requested object: {received_data['requestedObject']}")
    if received_data['requestedObject'] is not None:
        return jsonify({"requestedObject": received_data['requestedObject']}), 200
    else:
        return jsonify({"status": "error", "message": "No requested object found"}), 404
    
@app.route('/get_distance', methods=['GET'])
def get_distance():
    print(f"Distance: {received_data['distance']}")
    if received_data['distance'] is not None:
        return jsonify({"distance": received_data['distance']}), 200
    else:
        return jsonify({"status": "error", "message": "No distance found"}), 404
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)