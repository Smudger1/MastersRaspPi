import os
from reactNativeCommSingle import main as react_native_comms_main


option = ""
while option != "q":
    print("Masters Raspberry Pi Script Menu:")
    print("1. Capture Images for Calibration")
    print("2. Calibrate Cameras")
    print("3. Start communication with React Native App")
    print("q. Quit")

    option = input("Enter your choice: ")
    if option == "q":
        print("Exiting the program.")
    elif option == "1":
        print("Starting image capture for both cameras...")
        os.system("python getImagesIdentical.py")
    elif option == "2":
        print("Starting camera calibration process...")
        os.system("python calibScript.py")
    elif option == "3":
        print("Starting communication with React Native App...")
        react_native_comms_main()
    else:
        print("Invalid option. Please enter 'q' to quit.")