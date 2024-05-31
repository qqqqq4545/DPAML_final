from telegram import Bot

# Initialize your bot with your token
TOKEN = '7013365797:AAEr7bNQbmAw7J8c0vNVfDdJQL-hktWRtAk'
bot = Bot(TOKEN)

chat_id = 5918028021
def send_telegram_Wakeup(chat_id):
    """Send a wakeup notification."""
    message = "Wake-up alarm! Time to start the day."
    bot.send_message(chat_id=chat_id, text=message)

def send_telegram_Outside(chat_id):
    """Send a notification when going outside."""
    message = "I am going outside now."
    bot.send_message(chat_id=chat_id, text=message)

def send_telegram_Moving(chat_id):
    """Send a notification when moving."""
    message = "I am moving around."
    bot.send_message(chat_id=chat_id, text=message)


#token 7013365797:AAEr7bNQbmAw7J8c0vNVfDdJQL-hktWRtAk
#id 5918028021