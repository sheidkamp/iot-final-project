from gpiozero import MotionSensor
from gpiozero import Button
import threading
from threading import Timer
import time
import RPi.GPIO as GPIO
import sys
import pigpio

# CV
import cv2
import sys
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils


pir = MotionSensor(17)
button = Button(2)


############## CV #######################
def look(model: str, camera_id: int, width: int, height: int, num_threads: int,
		enable_edgetpu: bool) -> None:
	"""Continuously run inference on images acquired from the camera.

	Args:
	model: Name of the TFLite object detection model.
	camera_id: The camera id to be passed to OpenCV.
	width: The width of the frame captured from the camera.
	height: The height of the frame captured from the camera.
	num_threads: The number of CPU threads to run the model.
	enable_edgetpu: True/False whether the model is a EdgeTPU model.
	"""
	global person_detected, person_detected_at, animal_detected_at

	# Variables to calculate FPS
	counter, fps = 0, 0
	start_time = time.time()

	# Start capturing video input from the camera
	cap = cv2.VideoCapture(camera_id)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	# Visualization parameters
	row_size = 20	# pixels
	left_margin = 24	# pixels
	text_color = (0, 0, 255)	# red
	font_size = 1
	font_thickness = 1
	fps_avg_frame_count = 10

	# Initialize the object detection model
	base_options = core.BaseOptions(
		file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
	detection_options = processor.DetectionOptions(
		max_results=3, score_threshold=0.3, category_name_allowlist=["teddy bear", "person"])
	options = vision.ObjectDetectorOptions(
		base_options=base_options, detection_options=detection_options)
	detector = vision.ObjectDetector.create_from_options(options)
	print("Starting Webcam loop")

	# Continuously capture images from the camera and run inference
	while cap.isOpened():
		#print("webcam loop")
		if camera_stop.is_set():
			break
		success, image = cap.read()
		if not success:
			sys.exit(
				'ERROR: Unable to read from webcam. Please verify your webcam settings.'
			)

		counter += 1
		image = cv2.flip(image, 1)

		# Convert the image from BGR to RGB as required by the TFLite model.
		rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Create a TensorImage object from the RGB image.
		input_tensor = vision.TensorImage.create_from_array(rgb_image)

		# Run object detection estimation using the model.
		detection_result = detector.detect(input_tensor)

		#print("Detection!", detection_result)
		person_detected = False

		for detection in detection_result.detections:
			category = detection.categories[0]
			category_name = category.category_name
			print("Found:", category_name)
			if category_name == "person":
				person_detected = True
				person_detected_at = time.time()

			if category_name == "teddy bear":
				animal_detected_at = time.time()

		#if person_detected: print("Found Person")

		# image = utils.visualize(image, detection_result)
		# # Calculate the FPS
		# if counter % fps_avg_frame_count == 0:
		# 	end_time = time.time()
		# 	fps = fps_avg_frame_count / (end_time - start_time)
		# 	start_time = time.time()

		# # Show the FPS
		# fps_text = 'FPS = {:.1f}'.format(fps)
		# text_location = (left_margin, row_size)
		# cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
		# 			font_size, text_color, font_thickness)

		# # Stop the program if the ESC key is pressed.
		# if cv2.waitKey(1) == 27:
		# 	break
		# cv2.imshow('object_detector', image)

	cap.release()
	# cv2.destroyAllWindows()
	print("Stopping webcam")

###
button_pressed = False
motion_detected = False
person_detected = False
person_detected_at = 0
button_at = 0
animal_detected_at = 0

cool_off = 2
servo_pin = 13

camera_stop = threading.Event()
done = threading.Event()

def set_red():
	GPIO.output(RED_PIN, GPIO.HIGH)
	GPIO.output(GREEN_PIN, GPIO.LOW)

def set_green():
	GPIO.output(RED_PIN, GPIO.LOW)
	GPIO.output(GREEN_PIN, GPIO.HIGH)


def watch_button():
	global button_pressed, button_at
	while True:
		button.wait_for_press()
		button_pressed = True
		print("Button was pressed")
		button.wait_for_release()
		print("Button was released")
		button_at = time.time()
		button_pressed = False
    

def watch_motion_detector():
	global motion_detected, camera_stop, person_detected, person_detected_at

	model = 'efficientdet_lite0.tflite'
	cameraId = 0
	frameWidth = 320
	frameHeight = 240
	numThreads = 2
	enableEdgeTPU = False

	while True:
		pir.wait_for_motion()
		print("motion!")
		camera_stop = threading.Event()
		look_thread = threading.Thread(target = look, args=(model, int(cameraId), frameWidth, frameHeight,
			int(numThreads), bool(enableEdgeTPU)))
		look_thread.start()
		motion_detected = True
		
		pir.wait_for_no_motion()
		until_at_least = time.time() + 3
		print("Still!")

		while person_detected or (until_at_least > time.time()):
			time.sleep(0.2)

		camera_stop.set()
		
		motion_detected = False
		look_thread.join()


RED_PIN = 27
GREEN_PIN = 22
def setup_leds():
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(RED_PIN, GPIO.OUT)
	GPIO.setup(GREEN_PIN, GPIO.OUT)


def recent_person():
	global person_detected_at
	return recently(person_detected_at)

def recent_button():
	global button_at
	return recently(button_at)

def recent_animal():
	global animal_detected_at
	return recently(animal_detected_at, 1)


def recently(value, delta=3):
	if value + delta > time.time():
		return True
	return False

def run():
	global person_detected
	GPIO.setmode(GPIO.BCM)
	setup_leds()
	pi = pigpio.pi()
	OPEN = 600 
	CLOSED = 1150

	if not pi.connected:
		exit()

	was_open = False
	while True:	
		# print("bp", button_pressed, "md", motion_detected, "pd", person_detected)
		if button_pressed or recent_button() or (recent_person() and not recent_animal()):
			#print("button pressed: ", button_pressed)
			#print("recent_person", recent_person(), "recent_button", recent_button())
			if not was_open:
				print("Opening")
			set_green()
			pi.set_servo_pulsewidth(servo_pin, OPEN)
			was_open = True
		else:
			if was_open:
				print("Closing")
			set_red()
			pi.set_servo_pulsewidth(servo_pin, CLOSED)
			was_open = False
		time.sleep(0.1)



def main():
	motion_thread = threading.Thread(target=watch_motion_detector)
	button_thread = threading.Thread(target=watch_button)
	run_thread = threading.Thread(target=run)

	try:
		motion_thread.start()
		button_thread.start()
		run_thread.start()
	except KeyboardInterrupt:
		print("keyboard interrupt in main")
		GPIO.output(RED_PIN, GPIO.LOW)
		GPIO.output(GREEN_PIN, GPIO.LOW)

	motion_thread.join()
	button_thread.join()
	run_thread.join()

if __name__ == "__main__":
	try: 
		main()
	finally: 
		print("cleaning up")
		GPIO.output(RED_PIN, GPIO.LOW)
		GPIO.output(GREEN_PIN, GPIO.LOW)
		GPIO.cleanup()
