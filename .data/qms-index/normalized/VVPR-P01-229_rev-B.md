# VVPR-P01-229 Rev B: MX1 Software System v3.3.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-229
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.3.0
- Source filename: VVPR-P01-229 - MX1 Software System v3.3.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-229 - MX1 Software System v3.3.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
<30 cm SSD interlock
2 second DDR preview delay
Features for PHI clearing
Power off updates
Resolved anomaly testing:
Network status update fix
ODA-CP connection interlock fix
Loading Time + Total Exposure Time
Resending images to PACS
Fix for maximum air kerma buzzer
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.3.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. G
IFU-MX1 - Instructions for Use, Rev. G
MATERIALS
E1 Emitter BOM Rev. H
C1 Cassette BOM Rev. I
M50133 Rev. A, Galaxy Tablet  S8+
MX1 Software System v3.3.0
APP MedAI Device App v3.3.0
S10045 MX1 Debug Window HTML, v1.2.0
Additional tools/equipment:
EQP-275 (or equivalent) Control Company Stopwatch 4YMT7
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed. If steps require x-ray emission, use appropriate radiation protective equipment.
If Verification Steps require viewing information stored in the capture-presenter database, use the following steps to access the database:
SSH into the cassette via ssh imager@<cassette hostname>.local
Using the terminal, enter as root using the following command:
su
Connect to the database using the following command:
mariadb mx1
Once connected to the database, use SQL queries defined in the Verification Steps below. Ensure a colon (;) is added to the end of all queries before running.
Table 1. <30 cm SSD Interlock - Requirements, Verification Steps, and Expected Results
Table 2. 2 Second DDR Preview Display Delay - Requirements, Verification Steps, and Expected Results
Table 3. “Clear PHI” Button - Requirements, Verification Steps, and Expected Results
Table 4. Power Off - Requirements, Verification Steps, and Expected Results
Table 5. Resolved Anomalies - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
Tab;e 3, SRS-37.2 - Original SQL query had incorrect syntax, leading to an error message. The query was updated to correctly display the STUDY table, allowing for the test engineer to record the number of entries.
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220
C1 Cassette Rev. I, SN: 1079
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System v3.3.0
EQP-275 Control Company Stopwatch 4YMT7
RESULTS
Table 1. <30 cm SSD Interlock - Requirements, Verification Steps, and Expected Results
Table 2. 2 Second DDR Preview Display Delay - Requirements, Verification Steps, and Expected Results
Table 3. “Clear PHI” Button - Requirements, Verification Steps, and Expected Results
Table 4. Power Off - Requirements, Verification Steps, and Expected Results
Table 5. Resolved Anomalies - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Table 5, SRS-12.26 - If the device is in ready state before the tablet is disconnected from the cassette WiFi, the MX1 device will remain in ready state after disconnection.
LIST OF APPENDICES
Appendix 1 and 16 - Verification Evidence as Specified in Results Table 1 - 5.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1: Single Mode Above 30 cm SSD No Interlock
Appendix 2: Serial Radiographic Above 30 cm SSD No Interlock
Appendix 3: Radioscopic Mode Above 30 cm SSD No Interlock
Appendix 4: Single Mode At 30 cm SSD No Interlock
Appendix 5: Serial Radiographic At 30 cm SSD No Interlock
Appendix 6: Radioscopic At 30 cm SSD No Interlock
Appendix 7: Single Mode Below 30 cm SSD with Interlock
Appendix 8: Serial Radiographic Below 30 cm SSD with Interlock
Appendix 9: Radioscopic Below 30 cm SSD with Interlock
Appendix 10: Time of DDR Preview
Appendix 11: Database Output After Clearing
3 rows in set (0.001 sec)
MariaDB [mx1]> select * from study;
Empty set (0.001 sec)
MariaDB [mx1]>
Appendix 12: Interlock After Boot without Tablet
Appendix 13: Interlock After Closing App
Appendix 14: ODA Screen After Exceeding Time Limit
Appendix 15: ODA Screen After Tapping Reset
Appendix 16: ODA Screen After Tapping Reset

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: SSD Interlock - Above 30 cm Bound - Single Radiographic Mode |  |  |  |  |
| SRS-12.25 | The SS shall allow x-ray emission if the calculated SSD is equal to or greater than 30 cm | 1. Ensure the system is in single radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is above 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is green when SSD is greater than 30 cm, indicating system is in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state |  |  |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state |  |  |
|  |  | Test Case: SSD Interlock - At 30 cm SSD - Single Radiographic Mode |  |  |  |
|  |  | 1. Place the system in single radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is green when SSD is equal to 30 cm, indicating system is in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state |  |  |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state |  |  |
|  |  | Test Case: SSD Interlock - Below 30 cm Bound - Single Radiographic Mode |  |  |  |
|  |  | 1. Place the system in single radiographic mode 2. Place an object on the cassette and position the emitter such that SSD is below 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is NOT in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is red when SSD is below 30 cm, indicating system is NOT in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are red, indicating system is NOT in ready state |  |  |
|  |  |  | Interlock status bar in ODA is red, indicating system is NOT in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Low SSD - Emitter too close to patient", indicating system is NOT in ready state |  |  |
|  |  | Test Case: SSD Interlock - Above 30 cm Bound - Serial Radiographic Mode |  |  |  |
|  |  | 1. Place the system in serial radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is above 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is green when SSD is greater than 30 cm, indicating system is in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state |  |  |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state |  |  |
|  |  | Test Case: SSD Interlock - At 30 cm SSD - Serial Radiographic Mode |  |  |  |
|  |  | 1. Place the system in serial radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is green when SSD is equal to 30 cm, indicating system is in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state |  |  |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state |  |  |
|  |  | Test Case: SSD Interlock - Below 30 cm Bound - Serial Radiographic Mode |  |  |  |
|  |  | 1. Place the system in serial radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is below 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is NOT in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is red when SSD is below 30 cm, indicating system is NOT in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are red, indicating system is NOT in ready state |  |  |
|  |  |  | Interlock status bar in ODA is red, indicating system is NOT in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Low SSD - Emitter too close to patient", indicating system is NOT in ready state |  |  |
|  |  | Test Case: SSD Interlock - Above 30 cm Bound - Radioscopic Mode |  |  |  |
|  |  | 1. Place the system in radioscopic mode 2. Place an object on the cassette and position the emitter such that the SSD is above 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is green when SSD is greater than 30 cm, indicating system is in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state |  |  |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state |  |  |
|  |  | Test Case: SSD Interlock - At 30 cm SSD - Radioscopic Mode |  |  |  |
|  |  | 1. Place the system in radioscopic mode 2. Place an object on the cassette and position the emitter such that the SSD is 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is green when SSD is equal to 30 cm, indicating system is in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state |  |  |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state |  |  |
|  |  | Test Case: SSD Interlock - Below 30 cm Bound - Radioscopic Mode |  |  |  |
|  |  | 1. Place the system in radioscopic mode 2. Place an object on the cassette and position the emitter such that the SSD is below 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is NOT in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML |  |  |
|  |  |  | ViewFinder SSD display box is red when SSD is below 30 cm, indicating system is NOT in ready state |  |  |
|  |  |  | Cassette and emitter MI LEDs are red, indicating system is NOT in ready state |  |  |
|  |  |  | Interlock status bar in ODA is red, indicating system is NOT in ready state |  |  |
|  |  |  | Interlock status message in ODA is "Low SSD - Emitter too close to patient", indicating system is NOT in ready state |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Serial Radiographic Preview Display Delay |  |  |  |  |
| SRS-39.9 | For serial radiography, the SS shall impose a 2 second delay between acquisition and display of images | 1. Press and hold the emitter trigger. At the same time, start a timer. 2. Stop the timer upon display of the first serial radiographic frame. | “DDR starting” message is displayed in ODA prior to display of first serial radiographic frame |  |  |
|  |  |  | First serial radiographic frame is displayed at least 2.3s after trigger pull (Note* 300ms is for the first x-ray acquisition to initiate and complete, and 2 seconds is for the imposed image display delay) |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Clear PHI Stored in CP Database |  |  |  |  |
| SRS-37.2 | The SS shall provide a UI element to clear all PHI stored in CP's database via the MedAI Device App | 1. Open and conduct multiple exams, with multiple images of all capture types 2. Record the number of completed exams 3. Use the steps listed under Experimental Procedure to connect to the database. 4. Run the following query in terminal: select count (*) from study; 5. Record the number of entries in the study table 6. Navigate to the Device Settings page in the MedAI Device App 7. Click the “All Types” button. Tap "Yes" in the confirmation popup. 8. Using the database query from above, verify that all entries are deleted from the study table | Recorded number of entries in the study table before clearing |  |  |
|  |  |  | All entries in the study table are deleted after clearing |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Emitter Power Off |  |  |  |  |
| SRS-5.1 | The SS shall initiate the emitter power down sequence when the emitter center HMI button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. | 1. Ensure the emitter is powered on 2. Start timer. At the same time, press and hold emitter center HMI button for 3 seconds 3. Stop timer when emitter has powered down | Emitter powers down within 5 seconds of the initial press after the center HMI button has been held down for 3 seconds |  |  |
|  |  |  | Emitter MI LEDs turn off after the center HMI button has been held down for 5 seconds |  |  |
|  | Test Case: Cassette Power Off |  |  |  |  |
| SRS-5.2 | The SS shall initiate the cassette power down sequence when the cassette power button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. | 1. Ensure the cassette is powered on 2. Start timer. At the same time, press and hold cassette power button for 3 seconds 3. Stop timer when cassette has powered down | Cassette power down sequence completes in 5 seconds |  |  |
|  |  |  | Cassette MI LEDs turn off after the power button has been held down for 5 seconds |  |  |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App, H1 Wired Charger |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: WiFi Network Status Update Fix |  |  |  |  |
| SRS-34.2 | The SS should update the wireless network connection status in the MedAI Device App in under 90 seconds | 1. In the Network Settings page, connect the MX1 system to an external network (e.g. mobile hotspot) 2. Disable the external network. Additionally, start a timer. 3. Stop timer when the Network Status message changes from "Connected" to "Not Connected" or "Not Configured" 4. Repeat above steps to verify the behavior for the Network Status message in MedAI Cloud and Network Preferences menu | Network Status message in Network Settings page changes to "NOT CONNECTED" within 90 seconds |  |  |
|  |  |  | Network Status message in MedAI Cloud and Network Preferences menu changes to "NOT CONNECTED" within 90 seconds |  |  |
|  |  | 1. Reenable the external network. Addtionally, start a timer. 2. Stop timer when the Network Status message in the Network Settings page changes from "Not Connected" to "Connected" 3. Repeat above steps to verify the behavior for the Network Status message in MedAI Cloud and Network Preferences menu | Network Status message in Network Settings page changes to "CONNECTED" within 90 seconds |  |  |
|  |  |  | Network Status message in MedAI Cloud and Network Preferences menu changes to "CONNECTED" within 90 seconds |  |  |
|  | Test Case: ODA-CP Connection Interlock Fix - |  |  |  |  |
| SRS-12.26 | The SS shall disallow captures when ODA disconnects from CP | 1. Power the cassette and emitter on. Do not connect the tablet to the cassette WiFi. 2. After successful start up, verify that the system is NOT in ready state | Emitter MI LEDs are steady red |  |  |
|  |  |  | Cassette MI LEDs are steady red |  |  |
|  |  | 1. With all MX1 system components powered and connected, navigate to the Acquisition Screen in ODA 2. Ensure that the system is in ready state 3. Disconnect the tablet from the cassette WiFi 4. Verify that the system exits ready state | Emitter MI LEDs turn steady red upon tablet disconnection |  |  |
|  |  |  | Cassette MI LEDs turn steady red upon tablet disconnection |  |  |
|  |  | 1. Power the cassette and emitter on. 2. Connect the tablet to the cassette WiFi. 3. Start the MedAI Device App and navigate to the Acquisition Screen. 4. Verify that the system enters ready state | Emitter MI LEDs turn steady green |  |  |
|  |  |  | Cassette MI LEDs turn steady green |  |  |
|  |  | 1. With all MX1 system components powered and connected, navigate to the Acquisition Screen in ODA 2. Ensure that the system is in ready state 3. Close the MedAI Device App 4. Verify that the system exits ready state | Emitter MI LEDs turn steady red upon closing ODA |  |  |
|  |  |  | Cassette MI LEDs turn steady red upon closing ODA |  |  |
|  | Test Case: Loading and Total Exposure Time Reset Updates |  |  |  |  |
| SRS-32.5 | ODA shall contain the ability to reset the loading time limit and the set time limit shall persist between resets | 1. Navigate to the Device Settings page 2. Set the loading time limit to 3 seconds 3. Start an exam. Acquire a 5 second serial radiographic capture 4. Verify that the loading time display in the Acquisition Screen increments for the duration of the acquisition 5. Verify that the displayed loading time text turns red at 4 seconds 6. Verify that the loading time limit buzzer begins when the timer displays 4 seconds and continues for the rest of the acquisition 7. Verify that total exposure time displays 5 seconds at the end of the acquisition | Loading time display in Acquisition Screen increases as the acquisition continues |  |  |
|  |  |  | Displayed loading time text turns red at 4 seconds |  |  |
|  |  |  | Loading time limit buzzer begins at 4 seconds |  |  |
| SRS-32.30 | ODA shall display cumulative x-ray acquisition time during an exam in ODA |  | Total exposure time is 5 seconds |  |  |
|  |  | 1. Ensure the loading time limit display continues to display 5 seconds 2. Acquire another 5 second serial radiographic capture 3. Verify that the loading time display in the Acquisition Screen continues to increment for the duration of the second acquisition 4. Verify that the displayed loading time text remains red 5. Verify that the loading time limit buzzer continues for the entire duration of the second acquisition 6. Verify that total exposure time is 10 seconds | Loading time display in Acquisition Screen increases as the acquisition continues |  |  |
|  |  |  | Displayed loading time text remains red |  |  |
|  |  |  | Loading time limit buzzer continues for the entire duration of the acquisition |  |  |
|  |  |  | Total exposure time is 10 seconds |  |  |
|  |  | 1. Tap the "Reset Timer" 2. Verify the loading time display resets to 0 s 3. Verify that the total exposure time display does NOT reset | Loading time display resets to 0 s |  |  |
|  |  |  | Total exposure time does NOT reset or change |  |  |
|  |  | 1. Ensure the loading time display has been reset 2. Acquire a 5 second serial radiographic capture 3. Verify that the loading time display in the Acquisition Screen increments for the duration of the acquisition, staring at 0 seconds 5. Verify that the displayed loading time text turns red at 4 seconds 6. Verify that the loading time limit buzzer begins when the timer displays 4 seconds and continues for the rest of the acquisition 7. Verify that total exposure time displays 15 seconds by the end of acquisition | Loading time display in Acquisition Screen increases from 0 seconds as the acquisition continues |  |  |
|  |  |  | Displayed loading time text turns red at 4 seconds |  |  |
|  |  |  | Loading time limit buzzer begins at 4 seconds |  |  |
|  |  |  | Total exposure time is 15 seconds |  |  |
|  | Test Case: Resending Images to PACS Fix |  |  |  |  |
| SRS-40.3 | The SS shall allow for the export of all images to PACS servers in conformance with the DICOM standard via the MedAI Device App | 1. Navigate to the Library Screen in ODA 2. Select any image and send to a test PACS server 3. After the first DICOM study is sent, return to the Library Screen 4. Reselect and resend the same image to the same test PACS server 5. Verify that the PACS server has two different studies with the same image 6. Verify that ODA returns to the Library Screen and remains functional | Test PACS server has two DICOM studies for the same resent image |  |  |
|  |  |  | ODA returns to the Library Screen and remains functional |  |  |
|  | Test Case: Maximum Air Kerma Buzzer Fix |  |  |  |  |
| SRS-16.21 | The SS shall emit a constant audible warning at 1861 Hz during loading when the maximum air kerma rate at the patient entrance reference point is exceeded | Take a shot that makes the set dose rate buzzer go off Ensure emitter is wired or wirelessly charging Power cycle emitter Take another shot, this time with an expected dose rate lower than the set limit Verify that the normal audible signal goes off 1. Begin acquriring a serial radiographic capture that is past the set dose rate limit. Note that the limit may be set to a low value for ease of testing. 2. Verify that the dose limit audible warning is enabled when the dose limit is exceeded 3. After the acquisition is completed, begin charging the emitter with an H1 charger 4. While charging, power cycle the emitter 5. Position the emitter at an SID such that another serial radiographic capture should NOT exceed the set dose limit 6. Begin a serial radiographic acquisition 7. Verify that the audible buzzer indicating a serial radiographic capture is enabled | Audible warning indicating exceeded dose limit is enabled before power cycling the emitter |  |  |
|  |  |  | Audible buzzer indicating serial radiographic acquisition is enabled after power cycling the emitter |  |  |

### Table 7
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 01 Nov 2024 | 24-631 |

### Table 8
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-275 | 05/01/2024 | 05/01/2026 |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: SSD Interlock - Above 30 cm Bound - Single Radiographic Mode |  |  |  |  |
| SRS-12.25 | The SS shall allow x-ray emission if the calculated SSD is equal to or greater than 30 cm | 1. Ensure the system is in single radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is above 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 44.1 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is green when SSD is greater than 30 cm, indicating system is in ready state | Expected outcome verified. See Appendix 1. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state | Expected outcome verified. See Appendix 1. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - At 30 cm SSD - Single Radiographic Mode |  |  |  |
|  |  | 1. Place the system in single radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 30.0 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is green when SSD is equal to 30 cm, indicating system is in ready state | Expected outcome verified. See Appendix 3. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state | Expected outcome verified. See Appendix 3. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - Below 30 cm Bound - Single Radiographic Mode |  |  |  |
|  |  | 1. Place the system in single radiographic mode 2. Place an object on the cassette and position the emitter such that SSD is below 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is NOT in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 25.3 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is red when SSD is below 30 cm, indicating system is NOT in ready state | Expected outcome verified. See Appendix 7. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are red, indicating system is NOT in ready state | Expected outcome verified. See Appendix 7. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is red, indicating system is NOT in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Low SSD - Emitter too close to patient", indicating system is NOT in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - Above 30 cm Bound - Serial Radiographic Mode |  |  |  |
|  |  | 1. Place the system in serial radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is above 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 44.1 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is green when SSD is greater than 30 cm, indicating system is in ready state | Expected outcome verified. See Appendix 2. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state | Expected outcome verified. See Appendix 2. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - At 30 cm SSD - Serial Radiographic Mode |  |  |  |
|  |  | 1. Place the system in serial radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 30.0 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is green when SSD is equal to 30 cm, indicating system is in ready state | Expected outcome verified. See Appendix 4. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state | Expected outcome verified. See Appendix 4. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - Below 30 cm Bound - Serial Radiographic Mode |  |  |  |
|  |  | 1. Place the system in serial radiographic mode 2. Place an object on the cassette and position the emitter such that the SSD is below 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is NOT in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 25.3 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is red when SSD is below 30 cm, indicating system is NOT in ready state | Expected outcome verified. See Appendix 8. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are red, indicating system is NOT in ready state | Expected outcome verified. See Appendix 8. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is red, indicating system is NOT in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Low SSD - Emitter too close to patient", indicating system is NOT in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - Above 30 cm Bound - Radioscopic Mode |  |  |  |
|  |  | 1. Place the system in radioscopic mode 2. Place an object on the cassette and position the emitter such that the SSD is above 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 44.1 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is green when SSD is greater than 30 cm, indicating system is in ready state | Expected outcome verified. See Appendix 2. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state | Expected outcome verified. See Appendix 2. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - At 30 cm SSD - Radioscopic Mode |  |  |  |
|  |  | 1. Place the system in radioscopic mode 2. Place an object on the cassette and position the emitter such that the SSD is 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 30.0 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is green when SSD is equal to 30 cm, indicating system is in ready state | Expected outcome verified. See Appendix 5. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are green, indicating system is in ready state | Expected outcome verified. See Appendix 5. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is green, indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Ready", indicating system is in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  | Test Case: SSD Interlock - Below 30 cm Bound - Radioscopic Mode |  |  |  |
|  |  | 1. Place the system in radioscopic mode 2. Place an object on the cassette and position the emitter such that the SSD is below 30 cm. Ensure all other safety and positioning interlocks are met. 3. Record the calculated SSD displayed in S10045 MX1 Debug Window HTML 4. Verify that the system indicates that it is NOT in ready state | Recorded SSD value displayed on S10045 MX1 Debug Window HTML | Record SSD 25.3 cm Verified by GC 1NOV24 | N/A |
|  |  |  | ViewFinder SSD display box is red when SSD is below 30 cm, indicating system is NOT in ready state | Expected outcome verified. See Appendix 9. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette and emitter MI LEDs are red, indicating system is NOT in ready state | Expected outcome verified. See Appendix 9. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status bar in ODA is red, indicating system is NOT in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Interlock status message in ODA is "Low SSD - Emitter too close to patient", indicating system is NOT in ready state | Expected outcome verified. Verified by GC 1NOV24 | PASS |

### Table 10
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Serial Radiographic Preview Display Delay |  |  |  |  |
| SRS-39.9 | For serial radiography, the SS shall impose a 2 second delay between acquisition and display of images | 1. Press and hold the emitter trigger. At the same time, start a timer. 2. Stop the timer upon display of the first serial radiographic frame. | “DDR starting” message is displayed in ODA prior to display of first serial radiographic frame | Expected outcome verified. Verified by GC 1NOV24 | PASS |
|  |  |  | First serial radiographic frame is displayed at least 2.3s after trigger pull (Note* 300ms is for the first x-ray acquisition to initiate and complete, and 2 seconds is for the imposed image display delay) | Expected outcome verified, 2.75 seconds of delay. See Appendix 10. Verified by GC 1NOV24 | PASS |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Clear PHI Stored in CP Database |  |  |  |  |
| SRS-37.2 | The SS shall provide a UI element to clear all PHI stored in CP's database via the MedAI Device App | 1. Open and conduct multiple exams, with multiple images of all capture types 2. Record the number of completed exams 3. Use the steps listed under Experimental Procedure to connect to the database. 4. Run the following query in terminal:  select * from study; 5. Record the number of entries in the study table 6. Navigate to the Device Settings page in the MedAI Device App 7. Click the “All Types” button. Tap "Yes" in the confirmation popup. 8. Using the database query from above, verify that all entries are deleted from the study table Deviation*: Query updated to “select * from study;” Refer to Protocol Deviations above. | Recorded number of entries in the study table before clearing | Number of rows: 3 Verified by GC 1NOV24 | N/A |
|  |  |  | All entries in the study table are deleted after clearing | Expected outcome verified. See Appendix 11. Verified by GC 1NOV24 | PASS |

### Table 12
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Emitter Power Off |  |  |  |  |
| SRS-5.1 | The SS shall initiate the emitter power down sequence when the emitter center HMI button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. | 1. Ensure the emitter is powered on 2. Start timer. At the same time, press and hold emitter center HMI button for 3 seconds 3. Stop timer when emitter has powered down | Emitter powers down within 5 seconds of the initial press after the center HMI button has been held down for 3 seconds | Expected result verified Emitter powered in 3.69 seconds. Verified by GC 1NOV24 | PASS |
|  |  |  | Emitter MI LEDs turn off after the center HMI button has been held down for 5 seconds | Expected result verified Emitter powered in 3.69 seconds. Verified by GC 1NOV24 | PASS |
|  | Test Case: Cassette Power Off |  |  |  |  |
| SRS-5.2 | The SS shall initiate the cassette power down sequence when the cassette power button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. | 1. Ensure the cassette is powered on 2. Start timer. At the same time, press and hold cassette power button for 3 seconds 3. Stop timer when cassette has powered down | Cassette power down sequence completes in 5 seconds | Expected result verified Emitter powered in 3.84 seconds. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette MI LEDs turn off after the power button has been held down for 5 seconds | Expected result verified Emitter powered in 3.84 seconds. Verified by GC 1NOV24 | PASS |

### Table 13
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App, H1 Wired Charger |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: WiFi Network Status Update Fix |  |  |  |  |
| SRS-34.2 | The SS should update the wireless network connection status in the MedAI Device App in under 90 seconds | 1. In the Network Settings page, connect the MX1 system to an external network (e.g. mobile hotspot) 2. Disable the external network. Additionally, start a timer. 3. Stop timer when the Network Status message changes from "Connected" to "Not Connected" or "Not Configured" 4. Repeat above steps to verify the behavior for the Network Status message in MedAI Cloud and Network Preferences menu | Network Status message in Network Settings page changes to "NOT CONNECTED" within 90 seconds | Expected result verified “NOT CONNECTED” appeared in 18.34 seconds. Verified by GC 1NOV24 | PASS |
|  |  |  | Network Status message in MedAI Cloud and Network Preferences menu changes to "NOT CONNECTED" within 90 seconds | Expected result verified “NOT CONNECTED” appeared in 38.53 seconds. Verified by GC 1NOV24 | PASS |
|  |  | 1. Reenable the external network. Addtionally, start a timer. 2. Stop timer when the Network Status message in the Network Settings page changes from "Not Connected" to "Connected" 3. Repeat above steps to verify the behavior for the Network Status message in MedAI Cloud and Network Preferences menu | Network Status message in Network Settings page changes to "CONNECTED" within 90 seconds | Expected result verified “NOT CONNECTED” appeared in 71.79 seconds. Verified by GC 1NOV24 | PASS |
|  |  |  | Network Status message in MedAI Cloud and Network Preferences menu changes to "CONNECTED" within 90 seconds | Expected result verified “NOT CONNECTED” appeared in 48.00 seconds. Verified by GC 1NOV24 | PASS |
|  | Test Case: ODA-CP Connection Interlock Fix - |  |  |  |  |
| SRS-12.26 | The SS shall disallow captures when ODA disconnects from CP | 1. Power the cassette and emitter on. Do not connect the tablet to the cassette WiFi. 2. After successful start up, verify that the system is NOT in ready state | Emitter MI LEDs are steady red | Expected outcome verified. See Appendix 12. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette MI LEDs are steady red | Expected outcome verified. See Appendix 12. Verified by GC 1NOV24 | PASS |
|  |  | 1. With all MX1 system components powered and connected, navigate to the Acquisition Screen in ODA 2. Ensure that the system is in ready state 3. Disconnect the tablet from the cassette WiFi 4. Verify that the system exits ready state | Emitter MI LEDs turn steady red upon tablet disconnection | Failure, MI LEDs stay green. Verified by GC 1NOV24 | FAIL |
|  |  |  | Cassette MI LEDs turn steady red upon tablet disconnection | Failure, MI LEDs stay green. Verified by GC 1NOV24 | FAIL |
|  |  | 1. Power the cassette and emitter on. 2. Connect the tablet to the cassette WiFi. 3. Start the MedAI Device App and navigate to the Acquisition Screen. 4. Verify that the system enters ready state | Emitter MI LEDs turn steady green | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette MI LEDs turn steady green | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  | 1. With all MX1 system components powered and connected, navigate to the Acquisition Screen in ODA 2. Ensure that the system is in ready state 3. Close the MedAI Device App 4. Verify that the system exits ready state | Emitter MI LEDs turn steady red upon closing ODA | Expected outcome verified. See Appendix 13. Verified by GC 1NOV24 | PASS |
|  |  |  | Cassette MI LEDs turn steady red upon closing ODA | Expected outcome verified. See Appendix 13. Verified by GC 1NOV24 | PASS |
|  | Test Case: Loading and Total Exposure Time Reset Updates |  |  |  |  |
| SRS-32.5 | ODA shall contain the ability to reset the loading time limit and the set time limit shall persist between resets | 1. Navigate to the Device Settings page 2. Set the loading time limit to 3 seconds 3. Start an exam. Acquire a 5 second serial radiographic capture 4. Verify that the loading time display in the Acquisition Screen increments for the duration of the acquisition 5. Verify that the displayed loading time text turns red at 4 seconds 6. Verify that the loading time limit buzzer begins when the timer displays 4 seconds and continues for the rest of the acquisition 7. Verify that total exposure time displays 5 seconds at the end of the acquisition | Loading time display in Acquisition Screen increases as the acquisition continues | Expected outcome verified. See Appendix 14. Verified by GC 1NOV24 | PASS |
|  |  |  | Displayed loading time text turns red at 4 seconds | Expected outcome verified. See Appendix 14. Verified by GC 1NOV24 | PASS |
|  |  |  | Loading time limit buzzer begins at 4 seconds | Expected result verified. Verified by GC 1NOV24 | PASS |
| SRS-32.30 | ODA shall display cumulative x-ray acquisition time during an exam in ODA |  | Total exposure time is 5 seconds | Expected outcome verified. See Appendix 14. Verified by GC 1NOV24 | PASS |
|  |  | 1. Ensure the loading time limit display continues to display 5 seconds 2. Acquire another 5 second serial radiographic capture 3. Verify that the loading time display in the Acquisition Screen continues to increment for the duration of the second acquisition 4. Verify that the displayed loading time text remains red 5. Verify that the loading time limit buzzer continues for the entire duration of the second acquisition 6. Verify that total exposure time is 10 seconds | Loading time display in Acquisition Screen increases as the acquisition continues | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Displayed loading time text remains red | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Loading time limit buzzer continues for the entire duration of the acquisition | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Total exposure time is 10 seconds | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  | 1. Tap the "Reset Timer" 2. Verify the loading time display resets to 0 s 3. Verify that the total exposure time display does NOT reset | Loading time display resets to 0 s | Expected outcome verified. See Appendix 15. Verified by GC 1NOV24 | PASS |
|  |  |  | Total exposure time does NOT reset or change | Expected outcome verified. See Appendix 15. Verified by GC 1NOV24 | PASS |
|  |  | 1. Ensure the loading time display has been reset 2. Acquire a 5 second serial radiographic capture 3. Verify that the loading time display in the Acquisition Screen increments for the duration of the acquisition, staring at 0 seconds 5. Verify that the displayed loading time text turns red at 4 seconds 6. Verify that the loading time limit buzzer begins when the timer displays 4 seconds and continues for the rest of the acquisition 7. Verify that total exposure time displays 15 seconds by the end of acquisition | Loading time display in Acquisition Screen increases from 0 seconds as the acquisition continues | Expected outcome verified. See Appendix 16. Verified by GC 1NOV24 | PASS |
|  |  |  | Displayed loading time text turns red at 4 seconds | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Loading time limit buzzer begins at 4 seconds | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Total exposure time is 15 seconds | Expected outcome verified. See Appendix 16. Verified by GC 1NOV24 | PASS |
|  | Test Case: Resending Images to PACS Fix |  |  |  |  |
| SRS-40.3 | The SS shall allow for the export of all images to PACS servers in conformance with the DICOM standard via the MedAI Device App | 1. Navigate to the Library Screen in ODA 2. Select any image and send to a test PACS server 3. After the first DICOM study is sent, return to the Library Screen 4. Reselect and resend the same image to the same test PACS server 5. Verify that the PACS server has two different studies with the same image 6. Verify that ODA returns to the Library Screen and remains functional | Test PACS server has two DICOM studies for the same resent image | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | ODA returns to the Library Screen and remains functional | Expected result verified. Verified by GC 1NOV24 | PASS |
|  | Test Case: Maximum Air Kerma Buzzer Fix |  |  |  |  |
| SRS-16.21 | The SS shall emit a constant audible warning at 1861 Hz during loading when the maximum air kerma rate at the patient entrance reference point is exceeded | Take a shot that makes the set dose rate buzzer go off Ensure emitter is wired or wirelessly charging Power cycle emitter Take another shot, this time with an expected dose rate lower than the set limit Verify that the normal audible signal goes off 1. Begin acquriring a serial radiographic capture that is past the set dose rate limit. Note that the limit may be set to a low value for ease of testing. 2. Verify that the dose limit audible warning is enabled when the dose limit is exceeded 3. After the acquisition is completed, begin charging the emitter with an H1 charger 4. While charging, power cycle the emitter 5. Position the emitter at an SID such that another serial radiographic capture should NOT exceed the set dose limit 6. Begin a serial radiographic acquisition 7. Verify that the audible buzzer indicating a serial radiographic capture is enabled | Audible warning indicating exceeded dose limit is enabled before power cycling the emitter | Expected result verified. Verified by GC 1NOV24 | PASS |
|  |  |  | Audible buzzer indicating serial radiographic acquisition is enabled after power cycling the emitter | Expected result verified. Verified by GC 1NOV24 | PASS |

### Table 14
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-602 |  |
