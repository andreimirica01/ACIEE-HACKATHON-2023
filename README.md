# ACIEE – HACKATHON 2023 

## Challenge 2023:

Creation of a monitoring, detection and alerting system in the event of sleep/microsomnia at the wheel

- **Installation and configuration of all applications and software tools required to implement the theme/solution using an operating system compatible with the Raspberry Pi board;**
- **Detection of sleep, yawning, blinking often, etc. (dynamic detection);**
- **Generate alerts;**
- **Realization of graphic interface (screen with live/zoom image + various alerts);**
- **Equitable/correct division of tasks between team members and involvement of all team members in the realization of the system;**
- **Realization of the final presentation (5-6 slides) and your support in front of the participants and jury members in the most attractive format and for everyone;**
- **Real-time solution performance demonstration of the developed application.**


## What we did
- **We used the 300W Dataset that is an image dataset that focuses on face images. It contains 300 indoor images and 300 outdoor images with human faces, captured in the wild. It covers a variety of identities, facial expressions, lighting conditions, poses, occlusions, and face sizes.**
- **We used it to train a custom model that uses points arround eyes and mouth to recognize blinks and yawns.**
- **We created a mobile app using React Native where we send alarms,calls and notification when the system recognizes that the driver is tired.**
