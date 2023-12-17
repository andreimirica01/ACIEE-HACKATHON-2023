# import the necessary packages
import argparse
import json
import time
import cv2
import dlib
import requests
from imutils import face_utils
from imutils.video import VideoStream
from twilio.rest import Client

# Set de puncte pentru ochi
EYE_LANDMARKS = set(range(0, 12))
# Set de puncte pentru gură
MOUTH_LANDMARKS = set(range(12, 24))

# Variabile pentru cronometre
closed_eyes_timer = 0
yawn_timer = 0

# Temporizator api call
last_closed_eyes_api_call_time = 0
last_yawn_api_call_time = 0

FRAMES_BEFORE_API_CALL = 500


def twillio_sms():
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_='+18582155345',
        body='Buna, pari obosit! Pentru a evita situatii neplacute, te rugam sa faci o pauza!',
        to='+40757214117'
    )

    print(message.sid)


# Funcție pentru a trimite datele către API
def send_data_to_api(wwarning, description, event_type):
    global last_closed_eyes_api_call_time, last_yawn_api_call_time

    # Verifică timpul ultimului apel pentru evenimentul dat
    current_time = time.time()
    last_call_time = (
        last_closed_eyes_api_call_time if event_type == "closed_eyes" else last_yawn_api_call_time
    )

    # Verifică dacă a trecut suficient timp de la ultimul apel
    if current_time - last_call_time >= 3:  # Schimbă 1 cu intervalul dorit în secunde
        api_url = "https://driver-data.onrender.com/log"  # înlocuiește cu URL-ul tău
        api_headers = {'Content-Type': 'application/json'}

        # Convert the payload to JSON
        # json_data_to_send = json.dumps(data_to_send)
        json_data_to_send = {
            "warning": wwarning,
            "description": description
        }
        data = json.dumps(json_data_to_send)
        # Make the POST request to the API
        response = requests.post(api_url, headers=api_headers, data=data)

        # Print the response from the API
        print(f"API Response: {response.status_code}")
        print(response.text)

        # Actualizează timpul ultimului apel pentru evenimentul dat
        if event_type == "closed_eyes":
            last_closed_eyes_api_call_time = current_time
        elif event_type == "yawn":
            last_yawn_api_call_time = current_time



# Funcție pentru a calcula aspectul ochilor
def calculate_eye_aspect_ratio(eye_points):
    return (
            ((eye_points[1][0] - eye_points[5][0]) ** 2 + (eye_points[1][1] - eye_points[5][1]) ** 2) +
            ((eye_points[2][0] - eye_points[4][0]) ** 2 + (eye_points[2][1] - eye_points[4][1]) ** 2)
    ) / (2 * (
            ((eye_points[0][0] - eye_points[3][0]) ** 2 + (eye_points[0][1] - eye_points[3][1]) ** 2) +
            ((eye_points[1][0] - eye_points[5][0]) ** 2 + (eye_points[1][1] - eye_points[5][1]) ** 2)
    ))


# Funcție pentru verificarea condiției de închidere a ochilor
def check_closed_eyes(eye_aspect_ratio):
    eye_aspect_ratio_limit = 0.05  # Ajustează acest parametru în funcție de preferințe
    return eye_aspect_ratio < eye_aspect_ratio_limit


# Funcție pentru verificarea condiției de cască
def check_yawn(normalized_lip_distance, lip_distance_limit_ratio):
    return normalized_lip_distance > lip_distance_limit_ratio


# Funcție pentru a randa textul pe frame
def draw_text(frame, text, y_position):
    cv2.putText(frame, text, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# Funcție principală
def main():
    global closed_eyes_timer
    global yawn_timer

    frames_since_last_api_call = 0


    yawn_counter = 0
    # initialize the last event times
    last_yawn_time = 0
    last_closed_eyes_time = 0

    # set the lip distance limit as a ratio of face size
    lip_distance_limit_ratio = 0.23  # Ajustează acest parametru în funcție de preferințe

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")
    args = vars(ap.parse_args())

    # initialize dlib's face detector (HOG-based) and then load our
    # trained shape predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] camera sensor warming up...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the video stream, resize it to have a
        # maximum width of 400 pixels, and convert it to grayscale
        frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # convert the dlib rectangle into an OpenCV bounding box and
            # draw a bounding box surrounding the face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # use our custom dlib shape predictor to predict the location
            # of our landmark coordinates, then convert the prediction to
            # an easily parsable NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # select coordinates for eyes
            eye_points = shape[list(EYE_LANDMARKS)]

            # verifica dacă există cel puțin 6 puncte pentru a forma două triunghiuri (2 ochi)
            if len(eye_points) >= 6:
                # calculate the aspect ratio of eyes
                eye_aspect_ratio = calculate_eye_aspect_ratio(eye_points)

                # check if eyes are closed
                if check_closed_eyes(eye_aspect_ratio):
                    # update the closed eyes timer
                    closed_eyes_timer += 1
                    # reset the yawn timer
                    yawn_timer = 0

                    # send data to API
                    # send_data_to_api("closed_eyes")

                    # draw text on frame
                    draw_text(frame, "Ochii sunt inchisi!", 30)
                else:
                    # reset the closed eyes timer
                    closed_eyes_timer = 0

                # select coordinates for mouth
                mouth_points = shape[list(MOUTH_LANDMARKS)]

                # verifica dacă există cel puțin 3 puncte pentru a forma un triunghi
                if len(mouth_points) >= 3:
                    # calculate the distance between the upper and lower lips
                    upper_lip = shape[12]  # punctul 12 pentru buza superioară
                    lower_lip = shape[21]  # punctul 21 pentru buza inferioară
                    lip_distance = ((lower_lip[0] - upper_lip[0]) ** 2 + (lower_lip[1] - upper_lip[1]) ** 2) ** 0.5

                    # calculate the size of the face
                    face_size = w  # poti ajusta aceasta valoare in functie de preferinte

                    # normalize lip distance based on face size
                    normalized_lip_distance = lip_distance / face_size

                    # check if user has yawned
                    if check_yawn(normalized_lip_distance, lip_distance_limit_ratio):
                        # update the yawn timer
                        yawn_timer += 1
                        # reset the closed eyes timer
                        closed_eyes_timer = 0

                        # draw text on frame
                        draw_text(frame, "Ai cascat!", 60)
                    else:
                        # reset the yawn timer
                        yawn_timer = 0
                else:
                    print("Nu există suficiente puncte pentru a calcula aria gurii.")
            else:
                print("Nu există suficiente puncte pentru a calcula aspectul ochilor și gura.")

            # loop over the (x, y)-coordinates from our dlib shape
            # predictor model draw them on the image
            for (sX, sY) in shape:
                cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # check if eyes have been closed for 2 seconds consecutively
        if closed_eyes_timer >= 5:
            closed_eyes_timer = 0
            send_data_to_api("Danger", "Ai adormit",

                             "closed_eyes")
            print("Ochii sunt inchisi de mult timp!")
            frames_since_last_api_call = 0
            # aici poți adăuga orice acțiune suplimentară dorită

        # check if yawn has been detected for 2 seconds consecutively
        if yawn_timer >= 5:
            closed_eyes_timer = 0
            yawn_counter += 1
            yawn_timer = 0
            if(yawn_counter % 4 == 0 ):
                send_data_to_api("Warning", f"Casti de multe ori: {yawn_counter}",
                                 "yawn")
                twillio_sms()
            print("Ai cascat consecutiv mai mult timp!")
            frames_since_last_api_call = 0
            # aici poți adăuga orice acțiune suplimentară dorită

            # Increment the frame counter since the last API call
        frames_since_last_api_call += 1

        # Check if it's time to send additional data to the API
        if frames_since_last_api_call == FRAMES_BEFORE_API_CALL:
            # Trimite date suplimentare către API
            send_data_to_api("Normal", "Toto bene", "additional_info")
            # frames_since_last_api_call = 0  # Resetează numărul de cadre pentru următorul apel


    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    main()
