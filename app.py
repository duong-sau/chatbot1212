from flask import Flask, request

from Morintor.Facebook.FacebookDeploy import FacebookDeploy

app = Flask(__name__)
fb = FacebookDeploy()


@app.route('/')
def home():
    return "HELLO TO Phạm Văn Sậu"


@app.route('/facebook', methods=['GET'])
def verify_facebook():
    return fb.get_(request)


@app.route('/facebook', methods=['POST'])
def send_message_facebook():
    return fb.post(request)


@app.route('/iqTree', methods=['GET'])
def iqTree_reply():
    return "a"


if __name__ == '__main__':
    app.run()
