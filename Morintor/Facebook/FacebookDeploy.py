from pymessenger.bot import Bot


class FacebookDeploy:
    ACCESS_TOKEN = 'EAAReQKqnKGYBAJVVZCB7xT3ec77PTGQLlZB41qesLOHhEsGh5fHfZAIL01sHhbVTXqs2S3PLldF02yYCKERldqkRJt4knu2ZAto9T9qC3ZCViIeuVCCMjk80EZA94fN3M9vjUBZAaXcmnpSjUBog81LAYthXJT0t8p7pPu783Xy4E8woXwe9L8w '
    VERIFY_TOKEN = 'DS1211'
    bot = Bot(ACCESS_TOKEN)

    def __init__(self) -> None:
        super().__init__()

    def get_(self, request) -> str:
        token_sent = request.args.get("hub.verify_token")
        if token_sent == self.VERIFY_TOKEN:
            return request.args.get("hub.challenge")
        return 'Invalid verification token'

    def post(self, request) -> str:
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    # Facebook Messenger ID for user so we know where to send response back to
                    recipient_id = message['sender']['id']
                    content = message['message'].get('text')
                    print(content)
                    self.bot.send_text_message(recipient_id, "reply(content, db)")
                    pass
                pass
            pass
        return 'success'

