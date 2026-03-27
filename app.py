from flask import Flask, jsonify,request
from crowd import crowdaa
from face import facea

app = Flask(__name__)  # Correct: __name__ (double underscores)

@app.route('/api/hello', methods=['GET', 'POST'])

def crowd():
    result = crowdaa()
    return jsonify(result)
def face():
    resultt = facea()
    return jsonify(resultt)


if __name__ == '__main__':
    app.run(debug=True)
