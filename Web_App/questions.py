question1 = ["Alright, what brings you here today? What would you like to discuss?", "Okay, what made you decide to see me today? What would you like to talk about?"]
question2 = ["Has this been going on for some time? How long has this been present?", "Fow how long have you been going through this?"]
question3 = ["Would you like to tell me a bit more about what's going on? In a little more detail?", "Can you elaborate a bit more on your symptoms?"]
question4 = ["Alright, now why don't we talk about your mood? How is it generally?", "Okay, now do you mind telling me a bit about your mood? How do you most commonly feel these days?"]
question5 = ["Remind me again, what is it you do as a job?", "And remind me again, what do you do for work?"]
question6 = ["And do you have many responsibilities in your life? Demanding work, a family to provide for, or anything really that's starting to feel like a burden?", "And would you consider yourself to carry many responsibilities in your day-to-day life, that are starting to feel heavier than usual?"]
question7 = ["Alright, do you ever feel like you're overwhelmed, or like that everything is just too much?", "Okay, are you noticing yourself getting overwhelmed frequently?"]
question8 = ["Do you tend to feel alone at times, more so than you would typically?", "How often do you find yourself lonely or just feel like you're going through everything alone?"]
question9 = ["What about crying? Do you feel the need to cry more often than normal?", "How about being tearful? Are you feeling the need to cry often?"]
question10 = ["During your hard times, do you have someone you feel you can trust, someone who is there for you for support?", "Now, what is your support system like? Do you have friends/family whom you can lean on, that are there for help?"]
question11 = ["What about other things, like, are you able to keep up with everything? How're your energy levels?", "How's your energy like throughout the day?"]
question12 = ["What are some things you live for, things you feel good about?", "What about things you are passionate about? Do you still find you are able to enjoy them?"]
question13 = ["So, when you're relaxing or working, how would you rate your concentration and memory?", "And would you say your memory and concentration levels have gone up and down in the past few days?"]
question14 = ["Would you say you are taking proper care of yourself, and your health?", "Are you doing any wholesome activities to take care of your physical or emotional health?"]
question15 = ["Alright, now do you by any chance have a problem with substance abuse?", "Okay, now do you have any addictions to drugs or alcohol?"]
question16 = ["What about your eating habits? Has your appetite changed at all?", "What's your appetite like? Are you eating okay?"]
question17 = ["Do you think you get enough sleep most days?", "How are you sleeping these days?"]
question18 = ["And once you do get to sleep, would you say you're getting quality sleep?", "And do you find yourself waking up often during the night?"]
question19 = ["When you wake up in the morning, do you sometimes feel tired or drained?", "Do you sometimes feel like you don't want to wake up in the morning?"]
question20 = ["I understand this could be an uncomfortable question, however, have you ever had moments where you just didn't want to exist or maybe have thoughts of possible ways to end your life (and maybe even have tried)?", "I understand this could be an uncomfortable question, however, have you ever had moments where you just didn't want to exist or maybe have thoughts of possible ways to end your life (and maybe even have tried)?"]
question21 = ["And this is very difficult for me to ask, but something I would ask anybody in your situation: have you ever felt so low, so miserable, that you wanted to lash out, and physically harm someone?", "And this is very difficult for me to ask, but something I would ask anybody in your situation: have you ever felt so low, so miserable, that you wanted to lash out, and physically harm someone?"]
question22 = ["Now, have you ever had similar symptoms in your past?", "Have you gone through some variation of what you're currently experiencing, in the past?"]
question23 = ["Lastly, I have to ask, do you have any family history of clinical mental health issues?", "Lastly, are you aware of any family members who have had mental health problems in the past?"]

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('key')
text_to_speech = TextToSpeechV1(authenticator=authenticator)
text_to_speech.set_service_url('instance')
text_to_speech.set_disable_ssl_verification(True)

textual_data = [question1, question2, question3, question4, question5, question6, question7, question8, question9, question10, question11, question12, question13, question14, question15, question16, question17, question18, question19, question20, question21, question22, question23]

for text in textual_data:
    for i in range(len(text)):
        with open('questions/' + [ k for k, v in locals().items() if v == text ][0] + '_' + str(i) + '.wav', 'wb') as audio_file: audio_file.write(text_to_speech.synthesize(text[i], voice='en-US_MichaelV3Voice', accept='audio/wav').get_result().content)
