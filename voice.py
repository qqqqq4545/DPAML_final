from gtts import gTTS
import os
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

def play_voice_alert(message):
    """Generate and play a voice alert from the given message."""
    tts = gTTS(message, lang='en')
    filename = 'temp.mp3'
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for audio to finish playing
        pygame.time.Clock().tick(10)
    os.remove(filename)  # Remove the temporary file after playing

def voice_Alert_Wakeup():
    """Voice alert for wakeup."""
    play_voice_alert("Time to wake up!")

def voice_Alert_Outside():
    """Voice alert for going outside."""
    play_voice_alert("Heading outside now.")

def voice_Alert_Moving():
    """Voice alert for moving."""
    play_voice_alert("Starting to move.")
