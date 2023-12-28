import tkinter
from tkinter import filedialog

import mediapipe as mp
import cv2
import numpy as np
from cnn import Model, DataGatherer
from Auto_Correct_SpellChecker import Auto_Correct
from GUI import GUI
from tkinter import *
from PIL import ImageTk, Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = None

classifier = Model.load_classifier('my_model1.h5')

frames_since_last_Change = 0

def draw_region(image, center):
    cropped_image = cv2.rectangle(image, (center[0] - 130, center[1] - 130),
                                  (center[0] + 130, center[1] + 130), (0, 0, 255), 2)
    return cropped_image[center[1] - 130:center[1] + 130, center[0] - 130:center[0] + 130], cropped_image


def start_gui(title, size):
    gui = GUI(title, size)
    gui_frame = gui.create_frame(600, 600, 'ne', 1, 0, 'green')
    vid_label = Label(gui_frame)
    vid_label.grid()

    return gui, vid_label


def exit_app(gui, cap):
    gui.root.destroy()
    cap.release()


def update_frame(image, vid_label):
    image_fromarray = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image_fromarray)

    vid_label.imgtk = imgtk
    vid_label.config(image=imgtk)


def get_threshold(label_entrybox):
    value = label_entrybox.get('1.0', END)
    try:
        return float(value)
    except:
        return 0.95


def get_char(gesture):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    return Model.predict(classes, classifier, gesture)


def AddCharToWord(word, curr_char):
    temp_word = word
    if curr_char == 'space':

        temp_word = ""
    elif curr_char == 'del':
        temp_word = temp_word[0:-1]
        print('character has been deleted')
    elif curr_char != 'nothing':
        temp_word += curr_char.lower()
        print('character has been added: ', curr_char.lower())

    return [temp_word, curr_char]


def frame_video_stream(names, curr_char, prev_char, word, sentence, *args):
    global frames_since_last_Change
    kwargs = dict(zip(names, args))

    threshold = get_threshold(kwargs['th_box'])
    curr_char = curr_char
    prev_char = prev_char

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    update_frame(image, kwargs['vid_label'])

    image.flags.writeable = False
    results = kwargs['hands'].process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]
            center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped_img, full_img = draw_region(image, center)

            landmarks = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in hand_landmarks.landmark]
            # Example: Check the x-coordinate of the thumb and pinky
            thumb_x = landmarks[mp_hands.HandLandmark.THUMB_TIP.value][0]
            pinky_x = landmarks[mp_hands.HandLandmark.PINKY_TIP.value][0]
            # Determine left or right based on thumb and pinky positions
            if thumb_x < pinky_x:
                cropped_img = cv2.flip(cropped_img, 1)
            else:
                pass

            update_frame(full_img, kwargs['vid_label'])

            try:
                # print('from try')
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                gray = DataGatherer().edge_detection(gray)

                curr_char, pred = get_char(gray)
                print(curr_char)
                char = cv2.putText(full_img, curr_char, (center[0] - 135, center[1] - 135), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                   (0, 0, 255), 2, cv2.LINE_AA)
                char_prob = cv2.putText(full_img, '{0:.2f}'.format(np.max(pred)), (center[0] + 60, center[1] - 135),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                update_frame(full_img, kwargs['vid_label'])

                kwargs['cc_box'].delete('1.0', 'end')
                kwargs['cc_box'].insert('end', curr_char)
                if ((curr_char != prev_char) or frames_since_last_Change > 30) and (np.max(pred) >= threshold):
                    frames_since_last_Change = 0
                    temp = AddCharToWord(word, curr_char)

                    if (curr_char == 'space'):
                        print("---------------------space")
                        correct_word = Auto_Correct(word)
                        if correct_word!=None:
                            print(correct_word)
                        else:
                            print("None")
                            print("word: ", word, ", sentence: ", sentence)
                            correct_word = word
                        kwargs['sent_box'].insert('end', correct_word  + " ")
                        kwargs['ow_box'].delete('1.0', 'end')
                        kwargs['cw_box'].delete('1.0', 'end')
                        kwargs['cw_box'].insert('end', correct_word)
                    elif (curr_char == 'del'):
                        text_content = kwargs['ow_box'].get("1.0", "end-2c")  # Get all text except the last character
                        kwargs['ow_box'].delete("1.0", "end")  # Clear the entire content
                        kwargs['ow_box'].insert("1.0", text_content)
                    elif(curr_char== 'nothing'):
                        pass
                    else:
                        kwargs['ow_box'].insert('end', curr_char)

                    word = temp[0]

                    prev_char = curr_char
                elif prev_char == curr_char and (np.max(pred) > threshold):
                    print("frames_since_last_Change: ", frames_since_last_Change)
                    frames_since_last_Change += 1
            except Exception as e:
                print(e)
                pass

    kwargs['vid_label'].after(1, frame_video_stream, names, curr_char, prev_char, word, sentence, *args)


def pipe_cam(gui, vid_label):
    curr_char = None
    prev_char = None
    word = ""
    sentence = ""
    threshold = float(0.95)
    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    global cap
    cap = cv2.VideoCapture(0)

    labels_num = 5
    labels = ['threshold', 'current char', 'original word', 'corrected word', 'sentence']

    Labels, entryboxes = gui.create_labels(labels_num, labels, 'nw', 0, 0, y_spacing=0.06, create_entrybox_per_label=1)

    entryboxes['original word_entrybox'].config(width=18)
    entryboxes['corrected word_entrybox'].config(width=18)
    entryboxes['sentence_entrybox'].config(width=18, height=8)

    entryboxes['threshold_entrybox'].insert('end', threshold)
    th_entrybox = entryboxes['threshold_entrybox']

    cc_entrybox = entryboxes['current char_entrybox']

    ow_entrybox = entryboxes['original word_entrybox']

    cw_entrybox = entryboxes['corrected word_entrybox']

    sent_entrybox = entryboxes['sentence_entrybox']

    Exit_program_btn = gui.create_buttons(1, ['Exit'], 'center', 0.5, 0.9, command=lambda: exit_app(gui, cap))
    # Image_Load = gui.create_buttons(1, ['PIC'], 'center', 0.4, 0.8, command=lambda: open_image(gui))

    names = ['vid_label', 'hands', 'th_box', 'cc_box', 'ow_box', 'cw_box', 'sent_box']
    with mp_hands.Hands(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5,
            max_num_hands=2) as hands:
        frame_video_stream(names, curr_char, prev_char, word, sentence, vid_label,
                           hands, th_entrybox, cc_entrybox, ow_entrybox, cw_entrybox, sent_entrybox)
        gui.root.mainloop()


title = "Sign Language Recognition GUI"
size = "1100x1100"

gui, vid_label = start_gui(title, size)

pipe_cam(gui, vid_label)
