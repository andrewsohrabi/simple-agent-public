# VVPR-P01-236 Rev B: MX1 Software System v3.4.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-236
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.4.0
- Source filename: VVPR-P01-236 - MX1 Software System v3.4.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-236 - MX1 Software System v3.4.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Updates to Body Part Examined list
WPA3 to ensure data confidentiality, integrity, and origin authenticity
Motion interlock for serial radiographic acquisitions (DDR)
Mender updates with signed artifacts
2 second serial radiography (DDR) preview display delay
Modification to perpendicularity indicator behavior (+/- 3 degrees)
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System (SS) and MedAI Device App (ODA) as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.4.0 release.
Note that the tests for Mender signed artifact (Table 4) and the 2 second serial radiography (DDR) preview display (Table 5) are re-verified in this protocol to provide support for clarifications made during the FDA Interactive Request Responses for K241567. No changes or modifications have been made to these features as part of MX1 SS v3.4.0.
Testing remote upgrades with signed and unsigned Mender artifacts is intended to show that the MX1 Software System will NOT allow an upgrade with an unsigned artifact
Testing for DDR preview delay is intended to show that the 2-second delay remains for all images shown through the capture.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. H
IFU-MX1 - Instructions for Use, Rev. J
MATERIALS
E1 Emitter BOM Rev. H
C1 Cassette BOM Rev. I
M50133 Rev. A, Galaxy Tablet S8+
MX1 Software System v3.4.0
APP MedAI Device App v3.4.0
T-129 End of Line Test Fixture, Rev. C (or equivalent)
USB-C hub with ethernet
(Optional) Positioning Wedge (to adjust angles, Fanwar B0925GXGZX)
Additional tools/equipment:
EQP-139 (or equivalent) Control Company Stopwatch 4YMT7
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure - Tables 1 through 5
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed. If steps require x-ray emission, use appropriate radiation protective equipment.
Table 1. Updated Body Part Examined - Requirements, Verification Steps, and Expected Results
Table 2. WPA3 - Requirements, Verification Steps, and Expected Results
Table 3. Serial Radiography Motion Interlock - Requirements, Verification Steps, and Expected Results
Table 4. Mender Updates with Signed Artifacts - Requirements, Verification Steps, and Expected Results
Table 5. 2 second Serial Radiography Preview Display Delay - Requirements, Verification Steps, and Expected Results
Experimental Procedure - Table 6 (Perpendicularity Indicator Verification)
The following test is intended to ensure that the MX1 Software System meets the following requirement in MEMO-P01-630 Rev. H:
SRS-14.4 - The SS shall use the computed beam angle to indicate perpendicularity if the computed beam angle is less than or equal to 3 degrees
Data Definitions:
Target Angle - Target angle of the Cassette as defined by the procedure
Calculated Angle - Angle of the cassette as reported by MX1
Perpendicularity indicator - Indicates Beam Perpendicularity Status
Rotational Direction - 1 is Positive Rotational Direction, 2 is the Negative Rotational Direction
Calculated Angle Coordinate to Record - Indicates which XR-controller-reported beam angle coordinate (x or y) to record
Test Setup
Place the emitter and cassette into T-129 - End of Line Test Fixture (or equivalent) at an SID of 30 cm
Note that the defined SID value is for ease of testing. No specific SID is required to conduct this test.
Connect the emitter to a known network using the USB-C dongle
In a terminal, SSH into the emitter with the following command:
ssh imager@emitter-<hostname>.local
Disable xr-controller service with the following command:
systemctl --user stop xr-controller.service
Navigate to /opt/medai/data/config
In xr-config.json, set ‘show_collimator_debug’ to 'true’
When run in a terminal, xr-controller will output the x and y coordinates of the calculated beam angle. Use the following command to run xr-controller in a terminal:
xr-controller
Test Procedure
Figure 1 - Rotational Axes and Directions
*i.e. Lifting from Direction 1 along the X Axis results in the Y Axis Rotation - Direction 1 used in this protocol
Figure 2: Perpendicularity Indicator on Emitter Display
For each line in Table 6, lift the appropriate edge of the cassette while observing the calculated angle readout on terminal. If needed, use the angle positioning wedge to stabilize the cassette for each angle
Use Figure 1 to determine the appropriate cassette edge for each test. For example, for the first line of Table 6, lift the bottom edge of the cassette to rotate around the x-axis in the positive direction (direction 1).
Once the terminal readout reports that the cassette is at the target angle, record the defined coordinate of the calculated beam angle reported in terminal - the calculated beam angle should be within 1 degree of the specified target angle
Using Figure 2 as a reference, record the state of the perpendicularity indicator observed on the emitter display
Repeat steps 1 through 3 for the remaining target angles, rotational axes, and rotational directions.
Table 6. Perpendicularity Indicator Accuracy
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
For Tables 1 and 2:
The acceptance criteria for Tables 1 and 2 is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
For Table 3:
The acceptance criteria for Table 3 is 100% PASS for each target angle. PASS is determined at each target angle according to the following conditions:
for Calculated Angles resulting in 3° or less, the Perpendicularity Indicator shall show the beam is NORMAL as depicted in Figure 2 above;
and for Calculated Angles greater than 3°, the Perpendicularity Indicator shall show the beam is TILTED as depicted in Figure 2 above.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220 and 1222
C1 Cassette Rev. I, SN: 1079 and 1223
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System v3.4.0
T-129 EOL Test Fixture Rev. B
EQP-275 Control Company Stopwatch 4YMT7
RESULTS
Table 1. Updated Body Part Examined - Requirements, Verification Steps, and Expected Results
Table 2. WPA3 - Requirements, Verification Steps, and Expected Results
Table 3. Serial Radiography Motion Interlock - Requirements, Verification Steps, and Expected Results
Table 4. Mender Updates with Signed Artifacts - Requirements, Verification Steps, and Expected Results
Table 5. 2 second Serial Radiography Preview Display Delay - Requirements, Verification Steps, and Expected Results
Table 6. Perpendicularity Indicator Accuracy
Evidence Recorded By: GCDate: 04DEC24
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies found during the course of testing
LIST OF APPENDICES
Appendix 1 through 19 - Verification Evidence as Specified in Results Table 1 - 6.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1: ODA Body Parts Examined List
Appendix 2: Cassette Network from Emitter Network Manager - WPA3
imager@emitter-dv26:~$ nmcli dev wifi list
IN-USE  BSSID              SSID                         MODE   CHAN  RATE        SIGNAL  BARS  SECURITY
*       9C:B6:D0:35:6F:86  cassette-dv26                Infra  48    130 Mbit/s  100     ▂▄▆█  WPA3
Appendix 3: DDR Motion Interlock - Cassette and Emitter MI LEDs Red
Appendix 4: DDR Motion Interlock - ODA Interlock Status Message
Appendix 5: Emitter Successful Artifact Signature Verification
Dec  5 02:12:03 emitter-dv25 mender[1047]: time="2024-12-05T02:12:03Z" level=info msg="Validating the Update Info: example.com/artifacts-us/65a6a192e7bc805e8474131c/9c9c2bdb-1cd0-417f-a6ec-71afac699285?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=a848c3308d7d52663bbd129dff6a18b7%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T021203Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22emitter-image-v3.4.0-rb3.mender%22&response-content-type=application%2Fvnd.mender-artifact&x-id=GetObject&X-Amz-Signature=5399d5efff7408ac4e02dcb747c1ce7e41a6f366deedd40322b99a4715e66760 [name: emitter-image-v3.4.0-rb3; devices: [MX1E]]"
Dec  5 02:14:35 emitter-dv25 mender[1047]: time="2024-12-05T02:14:35Z" level=info msg="Installer: authenticated digital signature of artifact"
Appendix 6: Cassette Successful Artifact Signature Verification
Dec  5 02:11:36 cassette-dv25 mender[1017]: time="2024-12-05T02:11:36Z" level=info msg="Validating the Update Info: example.com/artifacts-us/65a6a192e7bc805e8474131c/b1520962-3812-4a7e-9bfd-5a342a3bff06?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=a848c3308d7d52663bbd129dff6a18b7%2F20241205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241205T021136Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22cassette-image-v3.4.0-rb3.mender%22&response-content-type=application%2Fvnd.mender-artifact&x-id=GetObject&X-Amz-Signature=febb265f135601e2ec770c4d8f17583373390459f5ed7e08ab42f3dbe74ff734 [name: cassette-image-v3.4.0-rb3; devices: [MX1C]]"
Dec  5 02:14:20 cassette-dv25 mender[1017]: time="2024-12-05T02:14:20Z" level=info msg="Installer: authenticated digital signature of artifact"
Appendix 7: Mender - Failed Update Indication
Appendix 8: Mender Deployment Log - Failed Update
Appendix 9: Perpendicularity Indicator 2.75 Degrees X Axis Direction 1
Appendix 10: Perpendicularity Indicator 3.44 Degrees X Axis Direction 1
Appendix 11: Perpendicularity Indicator 2.90 Degrees X Axis Direction 2
Appendix 12: Perpendicularity Indicator 3.11 Degrees X Axis Direction 2
Appendix 13: Perpendicularity Indicator 2.60 Degrees Y Axis Direction 1
Appendix 14: Perpendicularity Indicator 3.14 Degrees Y Axis Direction 1
Appendix 15: Perpendicularity Indicator 2.93 Degrees Y Axis Direction 2
Appendix 16: Perpendicularity Indicator 3.11 Degrees Y Axis Direction 2
Appendix 17: Example of Debug Output
Appendix 18: Cassette Orchestrator Log - Annotated Output
Appendix 19: Timestamp Comparison
* The timestamp of image retrieval is subtracted from the timestamp of sending to websocket to determine the time difference value

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Body Parts Examined List |  |  |  |  |
| SRS-38.10 | The SS shall provide fields to manually input the following patient/exam information: 1. Patient first name 2. Patient last name 3. Patient ID 4. Patient Date of Birth (DoB) 5. Laterality 6. Beam Indicator 7. Body Part Examined 8. Description | 1. On the MedAI Device App, navigate to the Exam Screen 2. Tap the "Body Part Examined" field to review the options in the dropdown menu 3. Verify that the options for "Body Part Examined" match the list defined in the Pass Criteria column | The following options are available in the Body Part Examined dropdown menu: 1. Ankle joint 2. Calcaneus 3. Elbow joint 4. Finger 5. Foot 6. Forearm 7. Hand 8. Humerus 9. Radius and ulna 10. Tarsal joint 11. Thumb 12. Tibia and fibula 13. Toe 14. Wrist joint |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Cassette Network - WPA3 |  |  |  |  |
| SRS-7.1 | The SS shall use the WPA3/WPA2/RSN to ensure data confidentiality, integrity, and origin authenticity | 1. SSH into the emitter 2. List the WiFi networks available to the emitter with nmcli device wifi list 3. Search for the cassette WiFi network in the list 4. Verify that the cassette's network is listed as WPA3 | In the list of WiFi networks, the cassette's network is listed as WPA3 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in serial radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Move emitter before attempting serial radiography |  |  |  |  |
| SRS-12.38 | The SS shall disallow serial radiography if the emitter IMU detects movement one (1) second prior to acquisition | 1. Set the emitter in a stand and take a serial radiographic acquisition without moving the emitter 2. Start timer. In less than 1 second, move the emitter and attempt to acquire serial radiographic images. Ensure movement does not break tracking or positioning interlocks. 3. Verify that the device indicates that the system is NOT in ready state and does not allow x-ray emission | Emitter MI LEDs are red |  |  |
|  |  |  | Cassette MI LEDs are red |  |  |
|  |  |  | ODA interlock status message displays "Motion detected - stabilize emitter" |  |  |
|  |  |  | No x-ray emission occurs |  |  |
|  |  | 1. Start timer. Leave emitter undisturbed for 1 second 2. After one second, take a serial radiographic acquisition 3. Verify that the device allows x-ray emission | X-ray emission occurs |  |  |
|  | Test Case: Move emitter while acquiring serial radiograph |  |  |  |  |
| SRS-12.39 | The SS shall not terminate x-ray emission if the emitter IMU detects movement during serial radiographic acquisition | 1. Set the emitter in T-129 EOL Test Fixture (or equivalent) and start a serial radiographic acquisition 2. During the acquisition, move the emitter while making sure to not break any positioning or tracking interlocks 3. Verify that x-ray emission continues despite emitter movement | X-ray emission continues while moving emitter during serial radiographic acquisition |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
| Test Case: Failed Mender update when using an unsigned artifact Note: The following tests can be run on either an emitter or cassette; the behavior will be the same for either device component. For ease of testing, only the cassette is specified. |  |  |  |  |  |
| SRS-45.18 | The SS shall check Over the Air (OTA) update artifact signatures before installation. OTA update artifacts shall be signed at build time | 1. Create a deployment in Mender for the cassette under test 2. For this specific deployment, create a different artifact key 3. Initiate the deployment in an attempt to update the cassette 4. Verify that the device fails to update. In Mender, wait for a message indicating a failed update 5. Review the deployment log for the cassette 6. Verify that the deployment log indicates that the update failed as a result of an improperly signed artifact | Mender displays indication of a failed update |  |  |
|  |  |  | Deployment log contains indication that update failed as a result of improperly signed artifact |  |  |
|  |  | Test Case: Successful Mender update when using a signed artifact Note: The following tests can be run on either an emitter or cassette; the behavior will be the same for either device component. For ease of testing, only the cassette is specified. |  |  |  |
|  |  | 1. Create and initiate a deployment in Mender for the cassette under test 2. When update modals appear on the MedAI Device App, tap “Install” for the device under test 3. Wait for the update to complete. The update progress bars in the MedAI Device App will indicate a completed update. 4. Power cycle the cassette under test 5. SSH into the cassette with ssh imager@<cassette-hostname> 6. Enter the following command: cat /var/log/syslog | grep mender 2. Using the log output, verify that the Mender client on the MX1 device performed appropriate checks of the artifact signature prior to installation | MedAI Device App indicates a successful Mender update via complete progress bar |  |  |
|  |  |  | Log output confirms that the MX1 system performed checks for the artifact signature prior to installation |  |  |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in serial radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
| Test Case: 2 Second DDR Preview Display Note: Cassette Orchestrator (CO) logs include information about images retrieved from the detector after an x-ray is acquired and when the processed images are sent to be displayed on the MedAI Device App. The following are a list of terms or phrases that are searched for while reviewing logs for the proceeding tests: "Got an IPC image”  - Log lines including the term "IPC" denote when an image acquired by the detector is received by CO. "Sending an image down the websocket" - These log lines indicate when CO sends an image for display on the tablet. “Adding ddr delay images” - This log line indicates that the next 2 seconds sent for display will be the “DDR Starting” placeholder images During the DDR preview, the display is 1 frame per second (FPS)[SRS-39.10]. The log will document each image received with an “adding image index” or “skipping image index” to indicate images that will be shown as a preview image or skipped, this will occur at a rate of 1:4 to maintain the 2-second delay through the image preview. Every 5th image will be added to the preview (MX1 uses 5 FPS for DDR capture). |  |  |  |  |  |
| SRS-39.9 | For serial radiography, the SS shall impose a 2 second delay between acquisition and display of images | 1. SSH into the cassette with ssh imager@<cassette-hostname> 2. Take a serial radiographic acquisition 3. After completing the acquisition, run the following command in terminal: cat /var/log/medai/cassette-orchestrator.log | grep -a -e "Got an IPC image"  -e "Adding ddr delay images" -e "ing image index" -e "Sending an image down the websocket" 4. Confirm that there is one instance of "Adding ddr delay images" in the log for this DDR capture. (This is evidence that the system added preview frames with the text "DDR starting."). 5. Confirm that the log output includes lines containing "Adding image index " for all indexes that are multiples of 5 (0, 5, 10, 15, etc.). These are images that will be sent to the web socket, meaning these are the only DDR frames that will be shown in the DDR preview display on the tablet. 6. Confirm that the log output includes lines containing "Skipping image index " for all indexes that are NOT multiples of 5 (1, 2, 3, 4, 6, 7, 8, 9, 11, etc.). These are images that will NOT be sent to the web socket, meaning they will NOT be shown during the DDR preview. 7. Record the timestamp of the first line containing "Got an IPC image". This corresponds to the first DDR frame retrieved from the detector. Label this timestamp "Image retrieved from detector - Frame 0". 8. Record the timestamp of the 3rd line containing "Sending an image down the websocket". This corresponds to the first real DDR frame sent to be displayed on the tablet. Label this timestamp "Image sent to tablet - Frame 0". (Note that the first two lines containing "Sending an image down the websocket" refer to the "DDR Starting" placeholder image, which is sent twice to start the 2 second preview display delay). 9. Continue to review the log. Find the sixth instance of "Got an IPC image". Label this timestamp accordingly (e.g. "Image retrieved from detector - Frame 5"). 10. Find the next instance of  "Sending an image down a websocket". Label this timestamp accordingly (e.g. "Image sent to tablet - Frame 5"). 11. For each paired instance of "Got an IPC image" and "Sending an image down a websocket", compare the timestamps. Verify that each image is sent to the display("Sending an image down a websocket") at least 2 seconds after being retrieved from the detector ("Got an IPC image"). | Each DDR frame sent to the tablet during DDR preview display is sent at least 2 seconds after being retrieved from the detector |  |  |

### Table 7
| Target Angle (°) | Rotational Axis | Rotational Direction | Calculated Angle Coordinate to Record | Calculated Angle (°) | Expected Perpendicularity Indicator (NORMAL/TILT) | Observed Perpendicularity Indicator (NORMAL/TILT) | PASS/FAIL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | x | 1 | x |  | NORMAL |  |  |
|  |  | 2 | x |  | NORMAL |  |  |
|  | y | 1 | y |  | NORMAL |  |  |
|  |  | 2 | y |  | NORMAL |  |  |
| 4 | x | 1 | x |  | TILT |  |  |
|  |  | 2 | x |  | TILT |  |  |
|  | y | 1 | y |  | TILT |  |  |
|  |  | 2 | y |  | TILT |  |  |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 04 Dec 2024 | 24-701 |

### Table 9
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-275 | 05/01/2024 | 05/01/2026 |

### Table 10
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Body Parts Examined List |  |  |  |  |
| SRS-38.10 | The SS shall provide fields to manually input the following patient/exam information: 1. Patient first name 2. Patient last name 3. Patient ID 4. Patient Date of Birth (DoB) 5. Laterality 6. Beam Indicator 7. Body Part Examined 8. Description | 1. On the MedAI Device App, navigate to the Exam Screen 2. Tap the "Body Part Examined" field to review the options in the dropdown menu 3. Verify that the options for "Body Part Examined" match the list defined in the Pass Criteria column | The following options are available in the Body Part Examined dropdown menu: 1. Ankle joint 2. Calcaneus 3. Elbow joint 4. Finger 5. Foot 6. Forearm 7. Hand 8. Humerus 9. Radius and ulna 10. Tarsal joint 11. Thumb 12. Tibia and fibula 13. Toe 14. Wrist joint | Expected outcome verified. See Appendix 1. Verified by AM 04DEC24 | PASS |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Cassette Network - WPA3 |  |  |  |  |
| SRS-7.1 | The SS shall use the WPA3/WPA2/RSN to ensure data confidentiality, integrity, and origin authenticity | 1. SSH into the emitter 2. List the WiFi networks available to the emitter with nmcli device wifi list 3. Search for the cassette WiFi network in the list 4. Verify that the cassette's network is listed as WPA3 | In the list of WiFi networks, the cassette's network is listed as WPA3 | Expected outcome verified. See Appendix 2. Verified by AM 04DEC24 | PASS |

### Table 12
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in serial radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Move emitter before attempting serial radiography |  |  |  |  |
| SRS-12.38 | The SS shall disallow serial radiography if the emitter IMU detects movement one (1) second prior to acquisition | 1. Set the emitter in a stand and take a serial radiographic acquisition without moving the emitter 2. Start timer. In less than 1 second, move the emitter and attempt to acquire serial radiographic images. Ensure movement does not break tracking or positioning interlocks. 3. Verify that the device indicates that the system is NOT in ready state and does not allow x-ray emission | Emitter MI LEDs are red | Expected outcome verified. See Appendix 3. Verified by AM 04DEC24 | PASS |
|  |  |  | Cassette MI LEDs are red | Expected outcome verified. See Appendix 3. Verified by AM 04DEC24 | PASS |
|  |  |  | ODA interlock status message displays "Motion detected - stabilize emitter" | Expected outcome verified. See Appendix 4. Verified by AM 04DEC24 | PASS |
|  |  |  | No x-ray emission occurs | Expected outcome verified. MI LEDs remained red. ODA interlock message continued to display “Motion detected - stabilize emitter.” No x-ray emission occurred upon trigger pull. Verified by AM 04DEC24 | PASS |
|  |  | 1. Start timer. Leave emitter undisturbed for 1 second 2. After one second, take a serial radiographic acquisition 3. Verify that the device allows x-ray emission | X-ray emission occurs | Expected outcome verified. Trigger pulled after waiting for 1.74 seconds. X-ray emission occurred after trigger pull. Verified by AM 04DEC24 | PASS |
|  | Test Case: Move emitter while acquiring serial radiograph |  |  |  |  |
| SRS-12.39 | The SS shall not terminate x-ray emission if the emitter IMU detects movement during serial radiographic acquisition | 1. Set the emitter in T-129 EOL Test Fixture (or equivalent) and start a serial radiographic acquisition 2. During the acquisition, move the emitter while making sure to not break any positioning or tracking interlocks 3. Verify that x-ray emission continues despite emitter movement | X-ray emission continues while moving emitter during serial radiographic acquisition | Expected outcome verified. Verified by AM 04DEC24 | PASS |

### Table 13
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
| Test Case: Failed Mender update when using an unsigned artifact Note: The following tests can be run on either an emitter or cassette; the behavior will be the same for either device component. For ease of testing, only the cassette is specified. |  |  |  |  |  |
| SRS-45.18 | The SS shall check Over the Air (OTA) update artifact signatures before installation. OTA update artifacts shall be signed at build time | 1. Create a deployment in Mender for the cassette under test 2. For this specific deployment, create a different artifact key 3. Initiate the deployment in an attempt to update the cassette 4. Verify that the device fails to update. In Mender, wait for a message indicating a failed update 5. Review the deployment log for the cassette 6. Verify that the deployment log indicates that the update failed as a result of an improperly signed artifact | Mender displays indication of a failed update | Expected outcome verified. See Appendix 7. Verified by AM 04DEC24 | PASS |
|  |  |  | Deployment log contains indication that update failed as a result of improperly signed artifact | Expected outcome verified. See Appendix 8. Deployment log indicated failed update cites “invalid signature. Verified by AM 04DEC24 | PASS |
|  |  | Test Case: Successful Mender update when using a signed artifact Note: The following tests can be run on either an emitter or cassette; the behavior will be the same for either device component. For ease of testing, only the cassette is specified. |  |  |  |
|  |  | 1. Create and initiate a deployment in Mender for the cassette under test 2. When update modals appear on the MedAI Device App, tap “Install” for the device under test 3. Wait for the update to complete. The update progress bars in the MedAI Device App will indicate a completed update. 4. Power cycle the cassette under test 5. SSH into the cassette with ssh imager@<cassette-hostname> 6. Enter the following command: cat /var/log/syslog | grep mender 2. Using the log output, verify that the Mender client on the MX1 device performed appropriate checks of the artifact signature prior to installation | MedAI Device App indicates a successful Mender update via complete progress bar | Expected outcome verified. Verified by GC 04DEC24 | PASS |
|  |  |  | Log output confirms that the MX1 system performed checks for the artifact signature prior to installation | Expected outcome verified. Both emitter and cassette were upgraded successfully. See Appendix 4 and 5. Verified by GC 04DEC24 | PASS |

### Table 14
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in serial radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
| Test Case: 2 Second DDR Preview Display Note: Cassette Orchestrator (CO) logs include information about images retrieved from the detector after an x-ray is acquired and when the processed images are sent to be displayed on the MedAI Device App. The following are a list of terms or phrases that are searched for while reviewing logs for the proceeding tests: "Got an IPC image”  - Log lines including the term "IPC" denote when an image acquired by the detector is received by CO. "Sending an image down the websocket" - These log lines indicate when CO sends an image for display on the tablet. “Adding ddr delay images” - This log line indicates that the next 2 seconds sent for display will be the “DDR Starting” placeholder images During the DDR preview, the display is 1 frame per second (FPS)[SRS-39.10]. The log will document each image received with an “adding image index” or “skipping image index” to indicate images that will be shown as a preview image or skipped, this will occur at a rate of 1:4 to maintain the 2-second delay through the image preview. Every 5th image will be added to the preview (MX1 uses 5 FPS for DDR capture). |  |  |  |  |  |
| SRS-39.9 | For serial radiography, the SS shall impose a 2 second delay between acquisition and display of images | 1. SSH into the cassette with ssh imager@<cassette-hostname> 2. Take a serial radiographic acquisition 3. After completing the acquisition, run the following command in terminal: cat /var/log/medai/cassette-orchestrator.log | grep -a -e "Got an IPC image"  -e "Adding ddr delay images" -e "ing image index" -e "Sending an image down the websocket" 4. Confirm that there is one instance of "Adding ddr delay images" in the log for this DDR capture. (This is evidence that the system added preview frames with the text "DDR starting."). 5. Confirm that the log output includes lines containing "Adding image index " for all indexes that are multiples of 5 (0, 5, 10, 15, etc.). These are images that will be sent to the web socket, meaning these are the only DDR frames that will be shown in the DDR preview display on the tablet. 6. Confirm that the log output includes lines containing "Skipping image index " for all indexes that are NOT multiples of 5 (1, 2, 3, 4, 6, 7, 8, 9, 11, etc.). These are images that will NOT be sent to the web socket, meaning they will NOT be shown during the DDR preview. 7. Record the timestamp of the first line containing "Got an IPC image". This corresponds to the first DDR frame retrieved from the detector. Label this timestamp "Image retrieved from detector - Frame 0". 8. Record the timestamp of the 3rd line containing "Sending an image down the websocket". This corresponds to the first real DDR frame sent to be displayed on the tablet. Label this timestamp "Image sent to tablet - Frame 0". (Note that the first two lines containing "Sending an image down the websocket" refer to the "DDR Starting" placeholder image, which is sent twice to start the 2 second preview display delay). 9. Continue to review the log. Find the sixth instance of "Got an IPC image". Label this timestamp accordingly (e.g. "Image retrieved from detector - Frame 5"). 10. Find the next instance of  "Sending an image down a websocket". Label this timestamp accordingly (e.g. "Image sent to tablet - Frame 5"). 11. For each paired instance of "Got an IPC image" and "Sending an image down a websocket", compare the timestamps. Verify that each image is sent to the display("Sending an image down a websocket") at least 2 seconds after being retrieved from the detector ("Got an IPC image"). | Each DDR frame sent to the tablet during DDR preview display is sent at least 2 seconds after being retrieved from the detector | Expected outcome verified. The log output of a 20 frame DDR was acquired and reviewed. Appendix 18 shows the annotated log output. The lines that are bolded and marked with an asterisk at the end correspond to the DDR frames that were displayed on the tablet during DDR preview, which are frames 0, 5, 10, and 15. The timestamp for image retrieval from detector and the timestamp for image send for display on the tablet were compared. As shown in Appendix 19, there is at least 2 seconds between the timestamps for all frames of interest. Verified by AM 04DEC24 | PASS |

### Table 15
| Target Angle (°) | Rotational Axis | Rotational Direction | Calculated Angle Coordinate to Record | Calculated Angle (°) | Expected Perpendicularity Indicator (NORMAL/TILT) | Observed Perpendicularity Indicator (NORMAL/TILT) | PASS/FAIL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | x | 1 | x | 2.75 | NORMAL | NORMAL (Appendix 9) | PASS |
|  |  | 2 | x | 2.90 | NORMAL | NORMAL (Appendix 11) | PASS |
|  | y | 1 | y | 2.60 | NORMAL | NORMAL (Appendix 13) | PASS |
|  |  | 2 | y | 2.93 | NORMAL | NORMAL (Appendix 15) | PASS |
| 4 | x | 1 | x | 3.44 | TILT | TILT (Appendix 10) | PASS |
|  |  | 2 | x | 3.11 | TILT | TILT (Appendix 12) | PASS |
|  | y | 1 | y | 3.14 | TILT | TILT (Appendix 14) | PASS |
|  |  | 2 | y | 3.20 | TILT | TILT (Appendix 16) | PASS |

### Table 16
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-643 |  |

### Table 17
| Cassette Orchestrator Log from Cassette-DV25 | Log Line Description |
| --- | --- |
| 2024-12-05 INFO 03:28:18.057145 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image* | Image retrieved from detector - Frame 0 |
| 2024-12-05 INFO 03:28:18.305516 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 1 |
| 2024-12-05 INFO 03:28:18.444229 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 2 |
| 2024-12-05 INFO 03:28:18.450497 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> Adding ddr delay images, mode: -1 | Queued delay images with "DDR Starting" text |
| 2024-12-05 INFO 03:28:18.636228 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 3 |
| 2024-12-05 INFO 03:28:18.673419 cassette-dv25 (websktSessTh1) Sent an image down the websocket | Display of "DDR Starting" image |
| 2024-12-05 INFO 03:28:18.773623 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> Adding image index 0 to VCR | Addition of Frame 0 image to display queue |
| 2024-12-05 INFO 03:28:18.825923 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 4 |
| 2024-12-05 INFO 03:28:18.987886 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 1 in VCR |  |
| 2024-12-05 INFO 03:28:19.027746 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image* | Image retrieved from detector - Frame 5 |
| 2024-12-05 INFO 03:28:19.185062 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 2 in VCR |  |
| 2024-12-05 INFO 03:28:19.271217 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 6 |
| 2024-12-05 INFO 03:28:19.346718 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 3 in VCR |  |
| 2024-12-05 INFO 03:28:19.472956 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 7 |
| 2024-12-05 INFO 03:28:19.530730 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 4 in VCR |  |
| 2024-12-05 INFO 03:28:19.614972 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 8 |
| 2024-12-05 INFO 03:28:19.673550 cassette-dv25 (websktSessTh1) Sent an image down the websocket | Display of "DDR Starting" image |
| 2024-12-05 INFO 03:28:19.686332 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> Adding image index 5 to VCR | Addition of Frame 5 image to display queue |
| 2024-12-05 INFO 03:28:19.829154 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 9 |
| 2024-12-05 INFO 03:28:19.853358 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 6 in VCR |  |
| 2024-12-05 INFO 03:28:20.019044 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image* | Image retrieved from detector - Frame 10 |
| 2024-12-05 INFO 03:28:20.037944 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 7 in VCR |  |
| 2024-12-05 INFO 03:28:20.206532 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 11 |
| 2024-12-05 INFO 03:28:20.224254 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 8 in VCR |  |
| 2024-12-05 INFO 03:28:20.422593 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 9 in VCR |  |
| 2024-12-05 INFO 03:28:20.458814 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 12 |
| 2024-12-05 INFO 03:28:20.601433 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 13 |
| 2024-12-05 INFO 03:28:20.606965 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> Adding image index 10 to VCR | Addition of Frame 10 image to display queue |
| 2024-12-05 INFO 03:28:20.673917 cassette-dv25 (websktSessTh1) Sent an image down the websocket* | Image sent to display - Frame 0 |
| 2024-12-05 INFO 03:28:20.778443 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 11 in VCR |  |
| 2024-12-05 INFO 03:28:20.817283 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 14 |
| 2024-12-05 INFO 03:28:20.965100 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 12 in VCR |  |
| 2024-12-05 INFO 03:28:21.002497 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image* | Image retrieved from detector - Frame 15 |
| 2024-12-05 INFO 03:28:21.180185 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 13 in VCR |  |
| 2024-12-05 INFO 03:28:21.194779 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 16 |
| 2024-12-05 INFO 03:28:21.365641 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 14 in VCR |  |
| 2024-12-05 INFO 03:28:21.440249 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 17 |
| 2024-12-05 INFO 03:28:21.538123 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> Adding image index 15 to VCR | Addition of Frame 15 image to display queue |
| 2024-12-05 INFO 03:28:21.651135 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 18 |
| 2024-12-05 INFO 03:28:21.677241 cassette-dv25 (websktSessTh1) Sent an image down the websocket* | Image sent to display - Frame 5 |
| 2024-12-05 INFO 03:28:21.769233 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 16 in VCR |  |
| 2024-12-05 INFO 03:28:21.788897 cassette-dv25 (ImageConsumerFl) <IPC> Got an IPC image | Image retrieved from detector - Frame 19 |
| 2024-12-05 INFO 03:28:21.929557 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 17 in VCR |  |
| 2024-12-05 INFO 03:28:22.095342 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 18 in VCR |  |
| 2024-12-05 INFO 03:28:22.282598 cassette-dv25 (sf-112) <XRAY> <DDR_DELAY> skipping image index 19 in VCR |  |
| 2024-12-05 INFO 03:28:22.678503 cassette-dv25 (websktSessTh1) Sent an image down the websocket* | Image sent to display - Frame 10 |
| 2024-12-05 INFO 03:28:23.679165 cassette-dv25 (websktSessTh1) Sent an image down the websocket* | Image sent to display - Frame 15 |

### Table 18
| Frame Number | Timestamp of Image Retrieval from Detector | Timestamp of Sending to Websocket for Display | Time Difference (in seconds)* |
| --- | --- | --- | --- |
| Frame 0 | 03:28:18.057145 | 03:28:20.673917 | 2.616772 |
| Frame 5 | 03:28:19.027746 | 03:28:21.677241 | 2.649495 |
| Frame 10 | 03:28:20.019044 | 03:28:22.678503 | 2.659459 |
| Frame 15 | 03:28:21.002497 | 03:28:23.679165 | 2.676668 |
