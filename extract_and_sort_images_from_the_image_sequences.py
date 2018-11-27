import glob
from shutil import copyfile

# Define emotion order from CK+ readme
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
# Returns a list of all folders with participant numbers
participants = glob.glob("source_emotion\\*")

for participant in participants:
    participant_number = "%s" % participant[-4:]  # (eg. S005)
    # Store list of sessions for current participant
    for sessions in glob.glob("%s\\*" % participant):
        for files in glob.glob("%s\\*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')
            # emotions are encoded as a float, read line as float, then convert to integer.
            emotion = int(float(file.readline()))
            # get path for last image in sequence, which contains the emotion
            sourcefile_emotion = glob.glob("source_images\\%s\\%s\\*" % (participant_number, current_session))[-1]
            # do same for neutral image
            sourcefile_neutral = glob.glob("source_images\\%s\\%s\\*" % (participant_number, current_session))[0]
            # Generate path to put neutral image
            destination_neutral = "sorted_set\\neutral\\%s" % sourcefile_neutral[25:]
            # Do same for emotion containing image
            destination_emotion = "sorted_set\\%s\\%s" % (emotions[emotion], sourcefile_emotion[25:])
            # Copy files
            copyfile(sourcefile_neutral, destination_neutral)
            copyfile(sourcefile_emotion, destination_emotion)
