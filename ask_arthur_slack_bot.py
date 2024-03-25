import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_version import run
app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.message(".*")
def message_handler(message, say, logger):
    print(message)
    say(run(message["text"]))

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
