# VVPR-P01-176 Rev B: MX1 Software System Power OnOff and Power States v3.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-176
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.0.0
- Source filename: VVPR-P01-176 - MX1 Software System Power OnOff and Power States v3.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-176 - MX1 Software System Power OnOff and Power States v3.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Power on indications
Power off
Emitter-cassette connectivity MI LED indications
Enter and exit criteria for idle states
System behavior in lite idle and idle states
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.0.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. B
IFU-MX1 - Instructions for Use, Rev. D
MATERIALS
E1 Emitter Rev. H
C1 Cassette Rev. I
F1 Foot Pedal Rev. B
M50133 Rev. A, Galaxy Tablet  S8+
APP MedAI Device App
Additional tools/equipment:
S10046 MX1 MedAI Rest Server Rev. A
EQP-139 (or equivalent) Control Company Stopwatch 4YMT7
A smartphone without an IR filter
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 1. Power On - Requirements, Verification Steps, and Expected Results
Table 2. Power States - Lite Idle - Requirements, Verification Steps, and Expected Results
Table 3. Power States - Idle - Requirements, Verification Steps, and Expected Results
Table 4. Power Off - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
Table 2, SRS-8.29 - When the MX1 System is in lite idle state, the original Pass Criteria incorrectly stated “Cassette display shows "Idle" message” due to a clerical error. The corrected Pass Criteria is “Cassette display shows “Not Ready” message.”
Table 4, SRS-5.5 - For testing that the ODA shutdown button and the “Yes” button on the ensuing confirmation prompt does initiate system shutdown, the original Pass Criteria incorrectly stated “System remains powered on” due to a clerical error. The corrected Pass Criteria is “MX1 System shuts down.”
Table 4, SRS-5.5 - Upon successful shutdown via ODA shutdown button, the original Pass Criteria incorrectly stated that the message displayed on ODA was “Shutdown complete” due to clerical error. The corrected Pass Criteria is that the message is “Cassette was shutdown.”
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220
C1 Cassette Rev. I, SN: 1221
F1 Foot Pedal Rev. B, Lot #: 10010
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52X101FM1N
MX1 Software System v3.0.0
EQP-139 Control Company Stopwatch 4YMT7
RESULTS
Table 1. Power On - Requirements, Verification Steps, and Expected Results
Table 2. Power States - Lite Idle - Requirements, Verification Steps, and Expected Results
Table 3. Power States - Idle - Requirements, Verification Steps, and Expected Results
Table 4. Power Off - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: Pass with deviation
Anomalies - Refer to MEMO-P01-636 - MX1 Software System, v3.0.0, Unresolved Anomalies for resolution of the following anomalies found during the course of testing:
Table 3, SRS-8.13 - The MX1 System exits idle state 27.22 seconds after meeting an idle exit condition instead of the 20 seconds as defined in SRS-8.13.
LIST OF APPENDICES
Appendix 1 through Appendix 22 - Verification Evidence as Specified in Results Tables 1 to 4.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1. Emitter Before Power On
Appendix 2. Emitter MI LEDs Blue
Appendix 3. Emitter Splash Screen Animation
Appendix 4. Emitter MI LEDs Blink Cyan
Appendix 5. Cassette Before Power On
Appendix 6. Cassette MI LEDs Blue
Appendix 7. Cassette MI LEDs Blink Cyan
Appendix 8. Cassette display shows “Not Ready”
Appendix 9: Emitter in Active Ready State
Appendix 10. ICD Command Successful Response
Appendix 11. Ping of Detector IP
Appendix 12. IR LEDs On
Appendix 13: Cassette Lite Idle State
Appendix 14. Emitter Lite Idle State
Appendix 15. IR LEDs Off
Appendix 16. Image Acquired After Exiting Lite Idle
Appendix 17. MX1 IdleState
Appendix 18. Ping Detector Unsuccessful
Appendix 19. MedAI Device App Confirmation
Appendix 20. Confirmation Prompt Disappears
Appendix 21. ICD Command To Monoblock Timeout
Appendix 22: Single Radiograph after Exiting Lite Idle

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | The emitter and cassette are powered off. |  |  |  |  |
|  | Test Case: Emitter Power On Only |  |  |  |  |
| SRS-4.4 | The SS shall indicate power on and system initialization by setting the emitter MI (Mode Indicator) LEDs to solid blue within 3 seconds of powering on | 1. Press emitter power button and, at the same time, start a timer. 2. Stop timer when emitter is in an active state, indicated by blinking MI LEDs | Emitter MI LEDs set to blue within 3 seconds of power on |  |  |
| SRS-4.1 | The SS shall ensure the emitter is in active state within 180 seconds after receiving a power on signal |  | Emitter MI LEDs blink cyan within 180 seconds |  |  |
| SRS-4.7 | The SS shall indicate awaiting emitter-cassette WiFi connection by blinking the emitter MI LEDs cyan |  |  |  |  |
|  | Test Case: Emitter Splash Screen |  |  |  |  |
| SRS-4.10 | The SS shall display a splash screen on the emitter touchscreen display within 15 seconds of powering on | 1. Press emitter power button and, at the same time, start a timer. 2. Stop timer when the splash screen shows on the emitter display | Splash screen animation begins within 15 seconds of powering on the emitter |  |  |
|  | Test Case: Cassette Power On Only |  |  |  |  |
| SRS-4.5 | The SS shall indicate power on and system initialization by setting the cassette MI LEDs to solid blue within 3 seconds of powering on | 1. Press cassette power button and, at the same time, start a timer. 2. Stop timer when cassette is in an active state, indicated by blinking MI LEDs | Cassette MI LEDs set to blue within 3 seconds of power on |  |  |
| SRS-4.2 | The SS shall ensure the cassette is in active state within 180 seconds after receiving a power on signal |  | Cassette MI LEDs blink cyan within 180 seconds |  |  |
| SRS-4.8 | The SS shall indicate awaiting emitter-cassette WiFi connection by blinking the cassette MI LEDs cyan |  |  |  |  |
| SRS-4.6 | When the cassette is in a processing state, the SS shall indicate via notification on the cassette display | 1. Ensure the cassette is powered off 2. Restart cassette 3. While the cassette is in a processing state, record evidence of the cassette display showing a “Not Ready” message | Cassette display shows “Not Ready” |  |  |
|  | Test Case: Paired Emitter and Cassette Power On Ensure the emitter and cassette used during this test have been paired. Ensure both device components are powered off prior to conducting the following tests. |  |  |  |  |
| SRS-4.1 | The SS shall ensure the emitter is in active state within 180 seconds after receiving a power on signal | 1. Power the cassette and emitter on and, at the same time, start a timer 2. Stop timer when the emitter and cassette MI LEDs indicate tracking/positioning interlock status (green or red) | Emitter MI LEDs turn green or red within 180 seconds |  |  |
| SRS-4.2 | The SS shall ensure the cassette is in active state within 180 seconds after receiving a power on signal |  | Cassette MI LEDs turn green or red within 180 seconds |  |  |
| SRS-4.3 | When the cassette is ready, the SS shall indicate via notification on the cassette display |  | Cassette display shows "Ready" message |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Active State |  |  |  |  |
| SRS-8.2 | The SS shall have an active state | 1. Set the tube voltage and current-time product to 60 kV and 0.25 mAs, respectively. 2. Ensure the emitter is positioned such that all tracking and positioning interlocks are met. 3. With the emitter in active state, verify the following: Note: use the following REST ICD command to confirm the Monoblock LV PCBA is powered on: http://<cassette-hostname>:8081/remote_api?command=p&pid=1&op=0&reg=0 | Emitter display backlight is on |  |  |
|  |  |  | Emitter MI LEDs are green |  |  |
|  |  |  | Lasers are projecting |  |  |
|  |  |  | ICD read command to Monoblock returns with a successful response |  |  |
|  |  |  | Technique factors are 60 kV and 0.25 mAs |  |  |
|  |  | With the cassette in active state, verify the following: | Cassette MI LEDs are green |  |  |
|  |  |  | Cassette display shows "Ready" message |  |  |
|  |  |  | Ping of detector IP address (192.168.8.8) is successful |  |  |
|  |  |  | A smartphone without an IR filter on its camera can see the IR LEDs are on |  |  |
|  | Test Case: Enter Lite Idle State Ensure the device meets the active state criteria prior to conducting the following tests. |  |  |  |  |
| SRS-8.25 | The SS shall have a lite idle state | 1. Start timer 2. Leave the emitter undisturbed for 50 seconds 3. Stop timer when system enters lite idle | Using the time, verify both emitter and cassette enter lite idle at 50 second (+/-5%) |  |  |
| SRS-8.26 | The SS shall enter lite idle state after no idle exit criteria is met for 50 seconds (+/- 5%) |  |  |  |  |
| SRS-8.12 | The SS shall place the cassette in idle state as the connected emitter enters idle state |  |  |  |  |
| SRS-8.27 | When in lite idle state, the SS shall enable the MI LEDs on the emitter to pulse blue | With the emitter in lite idle state, verify the following: Note: use the following REST ICD command to confirm the Monoblock LV PCBA is powered on: http://<cassette-hostname>:8081/remote_api?command=p&pid=1&op=0&reg=0 | All emitter MI LEDs pulse blue |  |  |
| SRS-8.31 | Upon entering lite idle state, the SS shall disable lasers |  | Lasers turn off |  |  |
| SRS-8.30 | Upon entering lite idle state, the SS shall kill power to the Monoblock |  | ICD read command to Monoblock times out |  |  |
| SRS-8.32 | Upon entering lite idle state, the SS shall command the emitter touchscreen display to enter a sleep state |  | Emitter display backlight is off |  |  |
| SRS-8.33 | When in lite idle state, the SS shall enable the MI LEDs on the cassette to pulse blue | With the cassette in lite idle state, verify the following: | All cassette MI LEDs pulse blue |  |  |
| SRS-8.29 | Upon entering lite idle state, the SS shall indicate via notification on the cassette display |  | Cassette display shows "Idle" message |  |  |
| SRS-8.28 | Upon entering lite idle state, the SS shall set the IR LED brightness to 0 |  | A smartphone without an IR filter on its camera can see the IR LEDs are off |  |  |
|  | Test Case: Exit Lite Idle with IMU |  |  |  |  |
| SRS-8.35 | The SS shall exit lite idle state upon meeting the same exit conditions as idle state | 1. Ensure the emitter is in lite idle state 2. Start timer and move the emitter at the same time 3. Stop timer when system fully exits lite idle 4. Acquire a single radiographic image | Emitter display backlight turns on |  |  |
| SRS-8.14 | If connected, the SS shall wake a device component (cassette or emitter) when the connected component exits idle state |  | Emitter MI LEDs turn green |  |  |
| SRS-8.15 | The SS shall exit idle state upon detection of non-zero readings from the emitter IMU (Internal Measurement Unit) |  | Lasers are projecting |  |  |
|  |  |  | Technique factors shown on emitter display are 60 kV and 0.25 mAs |  |  |
| SRS-8.21 | After exiting from idle state, the SS shall restore power to all unpowered hardware components |  | Cassette MI LEDs are green |  |  |
|  |  |  | Cassette display shows "Ready" message |  |  |
|  |  |  | A smartphone without an IR filter on its camera can see the IR LEDs are on |  |  |
| SRS-8.34 | If in lite idle state, the SS shall exit within 5 seconds of meeting an idle exit condition |  | Using the timer, verify the above occurs within 5 seconds of moving the emitter |  |  |
| SRS-8.23 | If entering radiographic mode after exiting idle state, the SS shall set the technique factors to the values selected prior to the system entering idle state |  | Verify single radiographic image is acquired at 60 kV and 0.25 mAs |  |  |
|  | Test Case: Exit Lite Idle into Photo Mode |  |  |  |  |
| SRS-8.22 | After exiting from idle state, the SS shall enter the imaging mode it was in prior to entering idle state | 1. Switch to photographic mode 2. Allow system to enter lite idle state 3. Move the emitter for the system to exit lite idle state | Upon exiting lite idle state, verify MX1 is in photographic mode |  |  |
|  | Test Case: Exit Lite Idle with Triggers and Buttons |  |  |  |  |
| SRS-8.16 | The SS shall exit idle state upon detection of any three emitter HMI button presses | 1. Allow system to enter lite idle state. 2. Press the emitter right HMI button once the device is in idle state. Verify system exits lite idle. 3. Repeat steps 1-2 for the center HMI button, left HMI button, trigger, foot pedal button and foot pedal trigger. Note that pressing the emitter center HMI button may also result in switching imaging modes. This is expected behavior. | MX1 exits lite idle state when the right emitter HMI button is pressed |  |  |
|  |  |  | MX1 exits lite idle state when the center emitter HMI button is pressed |  |  |
|  |  |  | MX1 exits lite idle state when the left emitter HMI button is pressed |  |  |
| SRS-8.17 | The SS shall exit idle state upon detection of an emitter trigger press |  | MX1 exits lite idle state when an emitter trigger is pulled |  |  |
| SRS-8.18 | The SS shall exit idle state upon detection of a cassette power button press |  | MX1 exits lite idle state when the cassette power button is pressed |  |  |
| SRS-8.19 | The SS shall exit idle state upon detection of foot pedal button event |  | MX1 exits lite idle state when the left foot pedal button is pressed |  |  |
|  |  |  | MX1 exits lite idle state when the right foot pedal button is pressed |  |  |
| SRS-8.20 | The SS shall exit idle state upon detection of foot pedal trigger event |  | MX1 exits lite idle state when the left foot pedal is pressed |  |  |
|  |  |  | MX1 exits lite idle state when the right foot pedal is pressed |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, T1 Tablet, APP MedAI Device App, F1 Foot Pedal |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Enter Idle State Ensure the device meets the active state criteria prior to conducting the following tests. |  |  |  |  |
| SRS-8.1 | The SS shall have an idle state | 1. Start timer 2. Leave the emitter undisturbed for 100 seconds 3. Verify that the MX1 system enters lite idle after 50 seconds 4. Stop timer when system enters idle | Using the time, verify both emitter and cassette enter lite idle at 50 seconds (+/-5%) |  |  |
| SRS-8.11 | The SS shall enter idle state after the emitter IMU reports no detected movement and no idle exit criteria is met for 100 seconds (+/- 5%) |  |  |  |  |
| SRS-8.12 | The SS shall place the cassette in idle state as the connected emitter enters idle state |  | Using the time, verify both emitter and cassette enter idle at 100 seconds (+/- 5%) |  |  |
| SRS-8.10 | When in idle state, the SS shall enable the MI LEDs on the emitter to pulse blue | With the emitter in idle state, verify the following: Note: use the following REST ICD command to confirm the Monoblock LV PCB is powered on: http://<cassette-hostname>:8081/remote_api?command=p&pid=1&op=0&reg=0 | All emitter MI LEDs pulse blue |  |  |
| SRS-8.9 | Upon entering idle state, the SS shall command the emitter touchscreen display to enter a sleep state |  | Emitter display backlight is off |  |  |
| SRS-8.8 | Upon entering idle state, the SS shall disable lasers |  | Lasers turn off |  |  |
| SRS-8.7 | Upon entering idle state, the SS shall kill power to the Monoblock |  | ICD read command to Monoblock times out |  |  |
| SRS-8.4 | Upon entering idle state, the SS shall enable the MI LEDs on the cassette to pulse blue | With the cassette in idle state, verify the following: | All cassette MI LEDs pulse blue |  |  |
| SRS-8.6 | Upon entering idle state, the SS shall indicate via notification on the cassette display |  | Cassette display shows "Idle" message |  |  |
| SRS-8.5 | Upon entering idle state, the SS shall set the IR LED brightness to 0 |  | A smartphone without an IR filter on its camera can see the IR LEDs are off |  |  |
| SRS-8.3 | Upon entering idle state, the SS shall kill power to the detector |  | Ping of detector IP address (192.168.8.8) is unsuccessful |  |  |
|  | Test Case: Exit Idle with IMU |  |  |  |  |
| SRS-8.14 | If connected, the SS shall wake a device component (cassette or emitter) when the connected component exits idle state | 1. Set the tube voltage and current-time product to 60 kV and 0.25 mAs, respectively. 2. Ensure the emitter is positioned such that all tracking and positioning interlocks are met. 1. Start timer 2. Move the emitter 3. Stop timer when system fully exits idle 4. Acquire a single radiographic image | Emitter display backlight turns on |  |  |
| SRS-8.15 | The SS shall exit idle state upon detection of non-zero readings from the emitter IMU |  | Emitter MI LEDs turn green |  |  |
| SRS-8.21 | After exiting from idle state, the SS shall restore power to all unpowered hardware components |  | Lasers are projecting |  |  |
|  |  |  | ICD read command to Monoblock returns with a successful response |  |  |
|  |  |  | Technique factors shown on emitter display are 60 kV and 0.25 mAs |  |  |
|  |  |  | Cassette MI LEDs are green |  |  |
|  |  |  | Cassette display shows "Ready" message |  |  |
|  |  |  | Ping of detector IP address (192.168.8.8) is successful |  |  |
|  |  |  | A smartphone without an IR filter on its camera can see the IR LEDs are on |  |  |
| SRS-8.13 | If in idle state, the SS shall exit idle state within 20 seconds of meeting an idle exit condition |  | Using the time, verify the above occurs within 20 seconds of moving the emitter |  |  |
| SRS-8.23 | If entering radiographic mode after exiting idle state, the SS shall set the technique factors to the values selected prior to the system entering idle state |  | Verify single radiographic image is acquired at 60 kV and 0.25 mAs |  |  |
|  | Test Case: Exit Idle State into Photo Mode |  |  |  |  |
| SRS-8.22 | After exiting from idle state, the SS shall enter the imaging mode it was in prior to entering idle state | 1. Switch to photographic mode 2. Allow system to enter idle state 3. Move the emitter for the system to exit idle state | Upon exiting idle state, verify MX1 is in photographic mode |  |  |
|  | Test Case: Exit Idle State with Triggers and Buttons |  |  |  |  |
| SRS-8.16 | The SS shall exit idle state upon detection of any three emitter HMI button presses | 1. Pair a foot pedal to the emitter by entering the foot pedal ID in the Settings page of the MedAI Device App 2. Allow system to enter idle state. 3. Press the emitter left HMI button once the device is in idle mode. Verify system exits idle state. 4. Repeat steps 1-2 for the right HMI button, center HMI button, trigger, foot pedal button and foot pedal trigger. Note that pressing the emitter center HMI button may also result in switching imaging modes. This is expected behavior. | MX1 exits idle state when the right emitter HMI button is pressed |  |  |
|  |  |  | MX1 exits idle state when the center emitter HMI button is pressed |  |  |
|  |  |  | MX1 exits idle state when the left emitter HMI button is pressed |  |  |
| SRS-8.17 | The SS shall exit idle state upon detection of an emitter trigger press |  | MX1 exits idle state when an emitter trigger is pulled |  |  |
| SRS-8.18 | The SS shall exit idle state upon detection of a cassette power button press |  | MX1 exits idle state when the cassette power button is pressed |  |  |
| SRS-8.19 | The SS shall exit idle state upon detection of foot pedal button event |  | MX1 exits idle state when the left foot pedal button is pressed |  |  |
|  |  |  | MX1 exits idle state when the right foot pedal button is pressed |  |  |
| SRS-8.20 | The SS shall exit idle state upon detection of foot pedal trigger event |  | MX1 exits idle state when the left foot pedal is pressed |  |  |
|  |  |  | MX1 exits idle state when the right foot pedal is pressed |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, T1 Tablet, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: MX1 System Shutdown |  |  |  |  |
| SRS-5.3 | The SS shall indicate the emitter power down by enabling the MI LEDs to blink blue for the duration of power down | 1. Start timer 2. Press and hold emitter center HMI button for 3 seconds 3. Stop timer when emitter has powered down | Emitter MI LEDs blink blue after the center HMI button has been held down for 3 seconds |  |  |
| SRS-5.1 | The SS shall initiate the emitter power down sequence when the emitter center HMI button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. |  | Emitter powers down within 5 seconds of the initial press after the center HMI button has been held down for 3 seconds |  |  |
| SRS-5.4 | The SS shall indicate the cassette power down by enabling the MI LEDs to blink blue for the duration of power down | 1. Start timer 2. Press and hold cassette power button for 3 seconds 3. Stop timer when cassette has powered down | Cassette MI LEDs blink blue after the power button has been held down for 3 seconds |  |  |
| SRS-5.2 | The SS shall initiate the cassette power down sequence when the cassette power button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. |  | Cassette power down sequence completes in 5 seconds |  |  |
|  | Test Case: MX1 System Shutdown - MedAI Device App Initiated |  |  |  |  |
| SRS-5.5 | The SS shall provide a single UI element in the MedAI Device App to initiate graceful shut down of the cassette and connected emitter | 1. Ensure the cassette and emitter are powered on and in radiographic mode. Ensure a tablet with the MedAI Device App is powered on and connected to the cassette. 2. Tap the shutdown button on the App. 3. Verify a confirmation prompt is displayed in the App. | Confirmation prompt displays with the following message: “Are you sure you want to shutdown the cassette?” |  |  |
|  |  |  | Confirmation prompt displays with “Yes” and “No” buttons |  |  |
|  |  |  | Confirmation prompt displays with a close (“X”) button |  |  |
|  |  | Tap the “X” button on the confirmation prompt | Prompt disappears |  |  |
|  |  |  | System remains powered on |  |  |
|  |  | 1. Tap the shutdown button on the App 2. Tap the “Yes” button on the confirmation prompt | Prompt disappears |  |  |
|  |  |  | System remains powered on |  |  |
|  |  |  | Emitter MI LEDs turn off after App shutdown button is tapped |  |  |
|  |  |  | Emitter powers down within 5 seconds of the initial tap of the App shutdown button |  |  |
|  |  |  | Cassette MI LEDs turn off after App shutdown button is tapped |  |  |
|  |  |  | Cassette powers down within 5 seconds of the initial tap of the App shutdown button |  |  |
|  |  |  | MedAI Device App displays a modal with the following message: “Shutdown complete” |  |  |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 14 May 2024 | 24-235 |

### Table 7
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-139 | 9/12/2022 | 9/12/2024 |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | The emitter and cassette are powered off. |  |  |  |  |
|  | Test Case: Emitter Power On Only |  |  |  |  |
| SRS-4.4 | The SS shall indicate power on and system initialization by setting the emitter MI (Mode Indicator) LEDs to solid blue within 3 seconds of powering on | 1. Press emitter power button and, at the same time, start a timer. 2. Stop timer when emitter is in an active state, indicated by blinking MI LEDs | Emitter MI LEDs set to blue within 3 seconds of power on | Expected outcome verified. See Appendix 1 and Appendix 2. Appendix 1 shows emitter MI LED are off before power on. Appendix 2 shows emitter MI LEDs are blue at 0.5 seconds after power on. Verified by RM 14MAY24 | P |
| SRS-4.1 | The SS shall ensure the emitter is in active state within 180 seconds after receiving a power on signal |  | Emitter MI LEDs blink cyan within 180 seconds | Expected outcome verified. See Appendix 2. MI LEDs begin blinking cyan at 65 sec. Verified by WP 14MAY24 | P |
| SRS-4.7 | The SS shall indicate awaiting emitter-cassette WiFi connection by blinking the emitter MI LEDs cyan |  |  |  |  |
|  | Test Case: Emitter Splash Screen |  |  |  |  |
| SRS-4.10 | The SS shall display a splash screen on the emitter touchscreen display within 15 seconds of powering on | 1. Press emitter power button and, at the same time, start a timer. 2. Stop timer when the splash screen shows on the emitter display | Splash screen animation begins within 15 seconds of powering on the emitter | Expected outcome verified. See Appendix 3. Splash screen visible at 14.94 sec. Verified by RM 14MAY24 | P |
|  | Test Case: Cassette Power On Only |  |  |  |  |
| SRS-4.5 | The SS shall indicate power on and system initialization by setting the cassette MI LEDs to solid blue within 3 seconds of powering on | 1. Press cassette power button and, at the same time, start a timer. 2. Stop timer when cassette is in an active state, indicated by blinking MI LEDs | Cassette MI LEDs set to blue within 3 seconds of power on | Expected outcome verified. See Appendix 5 and Appendix 6. Appendix 5 shows the cassette before power on. Appendix 6 shows the cassette after power on. MI LEDs are blue at 1 second after power on. Verified by WP 14MAY24 | P |
| SRS-4.2 | The SS shall ensure the cassette is in active state within 180 seconds after receiving a power on signal |  | Cassette MI LEDs blink cyan within 180 seconds | Expected outcome Verified. See Appendix 7. LEDs blinking at 61 sec Verified by WP 14MAY24 | P |
| SRS-4.8 | The SS shall indicate awaiting emitter-cassette WiFi connection by blinking the cassette MI LEDs cyan |  |  |  |  |
| SRS-4.6 | When the cassette is in a processing state, the SS shall indicate via notification on the cassette display | 1. Ensure the cassette is powered off 2. Restart cassette 3. While the cassette is in a processing state, record evidence of the cassette display showing a “Not Ready” message | Cassette display shows “Not Ready” | Expected outcome verified. See Appendix 8. Verified by RM 14MAY24 | P |
|  | Test Case: Paired Emitter and Cassette Power On Ensure the emitter and cassette used during this test have been paired. Ensure both device components are powered off prior to conducting the following tests. |  |  |  |  |
| SRS-4.1 | The SS shall ensure the emitter is in active state within 180 seconds after receiving a power on signal | 1. Power the cassette and emitter on and, at the same time, start a timer 2. Stop timer when the emitter and cassette MI LEDs indicate tracking/positioning interlock status (green or red) | Emitter MI LEDs turn green or red within 180 seconds | Expected outcome verified. LEDs red at 1 min 57 sec Verified by RM 14MAY24 | P |
| SRS-4.2 | The SS shall ensure the cassette is in active state within 180 seconds after receiving a power on signal |  | Cassette MI LEDs turn green or red within 180 seconds | Expected outcome verified. LEDs red at 1 min 57 sec Verified by RM 14MAY24 | P |
| SRS-4.3 | When the cassette is ready, the SS shall indicate via notification on the cassette display |  | Cassette display shows "Ready" message | Expected outcome verified. Verified by RM 14MAY24 | P |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Active State |  |  |  |  |
| SRS-8.2 | The SS shall have an active state | 1. Set the tube voltage and current-time product to 60 kV and 0.25 mAs, respectively. 2. Ensure the emitter is positioned such that all tracking and positioning interlocks are met. 3. With the emitter in active state, verify the following: Note: use the following REST ICD command to confirm the Monoblock LV PCBA is powered on: http://<cassette-hostname>:8081/remote_api?command=p&pid=1&op=0&reg=0 | Emitter display backlight is on | Expected outcome verified. See Appendix 9. Verified by WP 14MAY24 | P |
|  |  |  | Emitter MI LEDs are green | Expected outcome verified. See Appendix 9. Verified by WP 14MAY24 | P |
|  |  |  | Lasers are projecting | Expected outcome verified. See Appendix 9. Verified by WP 14MAY24 | P |
|  |  |  | ICD read command to Monoblock returns with a successful response | Expected outcome verified. See Appendix 10. Payload returned 0x01 0x00, indicating a powered on monoblock. Verified by WP 14MAY24 | P |
|  |  |  | Technique factors are 60 kV and 0.25 mAs | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  | With the cassette in active state, verify the following: | Cassette MI LEDs are green | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | Cassette display shows "Ready" message | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | Ping of detector IP address (192.168.8.8) is successful | Expected outcome verified. See Appendix 11. Verified by WP 14MAY24 | P |
|  |  |  | A smartphone without an IR filter on its camera can see the IR LEDs are on | Expected outcome verified. See Appendix 12. Verified by WP 14MAY24 | P |
|  | Test Case: Enter Lite Idle State Ensure the device meets the active state criteria prior to conducting the following tests. |  |  |  |  |
| SRS-8.25 | The SS shall have a lite idle state | 1. Start timer 2. Leave the emitter undisturbed for 50 seconds 3. Stop timer when system enters lite idle | Using the time, verify both emitter and cassette enter lite idle at 50 second (+/-5%) | Expected outcome verified. See Appendix 13. Entered lite idle state at 49.54 sec. Verified by RM 14MAY24 | P |
| SRS-8.26 | The SS shall enter lite idle state after no idle exit criteria is met for 50 seconds (+/- 5%) |  |  |  |  |
| SRS-8.12 | The SS shall place the cassette in idle state as the connected emitter enters idle state |  |  |  |  |
| SRS-8.27 | When in lite idle state, the SS shall enable the MI LEDs on the emitter to pulse blue | With the emitter in lite idle state, verify the following: Note: use the following REST ICD command to confirm the Monoblock LV PCBA is powered on: http://<cassette-hostname>:8081/remote_api?command=p&pid=1&op=0&reg=0 | All emitter MI LEDs pulse blue | Expected outcome verified. See Appendix 14 Verified by RM 14MAY24 | P |
| SRS-8.31 | Upon entering lite idle state, the SS shall disable lasers |  | Lasers turn off | Expected outcome verified. See Appendix 14 Verified by RM 14MAY24 | P |
| SRS-8.30 | Upon entering lite idle state, the SS shall kill power to the Monoblock |  | ICD read command to Monoblock times out | Expected outcome verified. See Appendix 21. Payload returned 0x00 0x00, indicating an unpowered monoblock. Verified by AM 14MAY24 | P |
| SRS-8.32 | Upon entering lite idle state, the SS shall command the emitter touchscreen display to enter a sleep state |  | Emitter display backlight is off | Expected outcome verified. Verified by RM 14MAY24 See Appendix 14 | P |
| SRS-8.33 | When in lite idle state, the SS shall enable the MI LEDs on the cassette to pulse blue | With the cassette in lite idle state, verify the following: | All cassette MI LEDs pulse blue | Expected outcome verified. See Appendix 13 Verified by RM 14MAY24 | P |
| SRS-8.29 | Upon entering lite idle state, the SS shall indicate via notification on the cassette display |  | Cassette display shows "Not Ready" message * Deviation: Cassette display shows “Not Ready” message, not “Idle”, when in lite idle state. See Protocol Deviations 1.a | Expected outcome verified. See Appendix 13. Cassette display shows “Not Ready” message. Verified by AM 14MAY24 | P |
| SRS-8.28 | Upon entering lite idle state, the SS shall set the IR LED brightness to 0 |  | A smartphone without an IR filter on its camera can see the IR LEDs are off | Expected outcome verified. See Appendix 15. Verified by RM 14MAY24 | P |
|  | Test Case: Exit Lite Idle with IMU |  |  |  |  |
| SRS-8.35 | The SS shall exit lite idle state upon meeting the same exit conditions as idle state | 1. Ensure the emitter is in lite idle state 2. Start timer and move the emitter at the same time 3. Stop timer when system fully exits lite idle 4. Acquire a single radiographic image | Emitter display backlight turns on | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-8.14 | If connected, the SS shall wake a device component (cassette or emitter) when the connected component exits idle state |  | Emitter MI LEDs turn green | Expected outcome verified. Occurred in 3.22 sec after exiting lite idle Verified by RM 14MAY24 | P |
| SRS-8.15 | The SS shall exit idle state upon detection of non-zero readings from the emitter IMU (Internal Measurement Unit) |  | Lasers are projecting | Expected outcome verified. Occurred in 3.22 sec after exiting lite idle. Verified by RM 14MAY24 | P |
|  |  |  | Technique factors shown on emitter display are 60 kV and 0.25 mAs | Expected outcome verified. Occurred in 3.22 sec after exiting lite idle. Verified by RM 14MAY24 | P |
| SRS-8.21 | After exiting from idle state, the SS shall restore power to all unpowered hardware components |  | Cassette MI LEDs are green | Expected outcome verified. Occurred in 3.22 sec after exiting lite idle. Verified by RM 14MAY24 | P |
|  |  |  | Cassette display shows "Ready" message | Expected outcome verified. Occurred in 3.22 sec after exiting lite idle. Verified by RM 14MAY24 | P |
|  |  |  | A smartphone without an IR filter on its camera can see the IR LEDs are on | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-8.34 | If in lite idle state, the SS shall exit within 5 seconds of meeting an idle exit condition |  | Using the timer, verify the above occurs within 5 seconds of moving the emitter | Expected outcome verified. Occurred in 3.22 sec after exiting lite idle. Verified by RM 14MAY24 | P |
| SRS-8.23 | If entering radiographic mode after exiting idle state, the SS shall set the technique factors to the values selected prior to the system entering idle state |  | Verify single radiographic image is acquired at 60 kV and 0.25 mAs | Expected outcome verified. See Appendix 16. Verified by RM 14MAY24. | P |
|  | Test Case: Exit Lite Idle into Photo Mode |  |  |  |  |
| SRS-8.22 | After exiting from idle state, the SS shall enter the imaging mode it was in prior to entering idle state | 1. Switch to photographic mode 2. Allow system to enter lite idle state 3. Move the emitter for the system to exit lite idle state | Upon exiting lite idle state, verify MX1 is in photographic mode | Expected outcome verified. MI LEDs turn white and emitter display shows icon indicating photo mode. Verified by AM 14MAY24 | P |
|  | Test Case: Exit Lite Idle with Triggers and Buttons |  |  |  |  |
| SRS-8.16 | The SS shall exit idle state upon detection of any three emitter HMI button presses | 1. Allow system to enter lite idle state. 2. Press the emitter right HMI button once the device is in idle state. Verify system exits lite idle. 3. Repeat steps 1-2 for the center HMI button, left HMI button, trigger, foot pedal button and foot pedal trigger. Note that pressing the emitter center HMI button may also result in switching imaging modes. This is expected behavior. | MX1 exits lite idle state when the right emitter HMI button is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by WP 14MAY24 | P |
|  |  |  | MX1 exits lite idle state when the center emitter HMI button is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by WP 14MAY24 | P |
|  |  |  | MX1 exits lite idle state when the left emitter HMI button is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by WP 14MAY24 | P |
| SRS-8.17 | The SS shall exit idle state upon detection of an emitter trigger press |  | MX1 exits lite idle state when an emitter trigger is pulled | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by WP 14MAY24 | P |
| SRS-8.18 | The SS shall exit idle state upon detection of a cassette power button press |  | MX1 exits lite idle state when the cassette power button is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by WP 14MAY24 | P |
| SRS-8.19 | The SS shall exit idle state upon detection of foot pedal button event |  | MX1 exits lite idle state when the left foot pedal button is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by RM 14MAY24 | P |
|  |  |  | MX1 exits lite idle state when the right foot pedal button is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by RM 14MAY24 | P |
| SRS-8.20 | The SS shall exit idle state upon detection of foot pedal trigger event |  | MX1 exits lite idle state when the left foot pedal is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by RM 14MAY24 | P |
|  |  |  | MX1 exits lite idle state when the right foot pedal is pressed | Expected outcome verified. MI LEDs turn green, lasers project crosshair pattern, and emitter display backlight is on. Verified by RM 14MAY24 | P |

### Table 10
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App, F1 Foot Pedal |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Enter Idle State Ensure the device meets the active state criteria prior to conducting the following tests. |  |  |  |  |
| SRS-8.1 | The SS shall have an idle state | 1. Start timer 2. Leave the emitter undisturbed for 100 seconds 3. Verify that the MX1 system enters lite idle after 50 seconds 4. Stop timer when system enters idle | Using the time, verify both emitter and cassette enter lite idle at 50 seconds (+/-5%) | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-8.11 | The SS shall enter idle state after the emitter IMU reports no detected movement and no idle exit criteria is met for 100 seconds (+/- 5%) |  |  |  |  |
| SRS-8.12 | The SS shall place the cassette in idle state as the connected emitter enters idle state |  | Using the time, verify both emitter and cassette enter idle at 100 seconds (+/- 5%) | Expected outcome verified. Entered idle at 95.28 secs Verified by RM 14MAY24 | P |
| SRS-8.10 | When in idle state, the SS shall enable the MI LEDs on the emitter to pulse blue | With the emitter in idle state, verify the following: Note: use the following REST ICD command to confirm the Monoblock LV PCB is powered on: http://<cassette-hostname>:8081/remote_api?command=p&pid=1&op=0&reg=0 | All emitter MI LEDs pulse blue | Expected outcome verified. See Appendix 17. Verified by WP 14MAY24 | P |
| SRS-8.9 | Upon entering idle state, the SS shall command the emitter touchscreen display to enter a sleep state |  | Emitter display backlight is off | Expected outcome verified. See Appendix 17. Verified by WP 14MAY24 | P |
| SRS-8.8 | Upon entering idle state, the SS shall disable lasers |  | Lasers turn off | Expected outcome verified. See Appendix 17. Verified by WP 14MAY24 | P |
| SRS-8.7 | Upon entering idle state, the SS shall kill power to the Monoblock |  | ICD read command to Monoblock times out | Expected outcome verified. Payload returned is 0x00 0x00, indicating unpowered monoblock. Verified by WP 14MAY24 | P |
| SRS-8.4 | Upon entering idle state, the SS shall enable the MI LEDs on the cassette to pulse blue | With the cassette in idle state, verify the following: | All cassette MI LEDs pulse blue | Expected outcome verified. See Appendix 17. Verified by WP 14MAY24 | P |
| SRS-8.6 | Upon entering idle state, the SS shall indicate via notification on the cassette display |  | Cassette display shows "Idle" message | Expected outcome verified. See Appendix 17. Verified by WP 14MAY24 | P |
| SRS-8.5 | Upon entering idle state, the SS shall set the IR LED brightness to 0 |  | A smartphone without an IR filter on its camera can see the IR LEDs are off | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-8.3 | Upon entering idle state, the SS shall kill power to the detector |  | Ping of detector IP address (192.168.8.8) is unsuccessful | Expected outcome verified. See Appendix 18. Verified by WP 14MAY24 | P |
|  | Test Case: Exit Idle with IMU |  |  |  |  |
| SRS-8.14 | If connected, the SS shall wake a device component (cassette or emitter) when the connected component exits idle state | 1. Set the tube voltage and current-time product to 60 kV and 0.25 mAs, respectively. 2. Ensure the emitter is positioned such that all tracking and positioning interlocks are met. 1. Start timer 2. Move the emitter 3. Stop timer when system fully exits idle 4. Acquire a single radiographic image | Emitter display backlight turns on | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-8.15 | The SS shall exit idle state upon detection of non-zero readings from the emitter IMU |  | Emitter MI LEDs turn green | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-8.21 | After exiting from idle state, the SS shall restore power to all unpowered hardware components |  | Lasers are projecting | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | ICD read command to Monoblock returns with a successful response | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | Technique factors shown on emitter display are 60 kV and 0.25 mAs | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | Cassette MI LEDs are green | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | Cassette display shows "Ready" message | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | Ping of detector IP address (192.168.8.8) is successful | Expected outcome verified. Verified by WP 14MAY24 | P |
|  |  |  | A smartphone without an IR filter on its camera can see the IR LEDs are on | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-8.13 | If in idle state, the SS shall exit idle state within 20 seconds of meeting an idle exit condition |  | Using the time, verify the above occurs within 20 seconds of moving the emitter | Failure. System exits idle state at 27.22 seconds. Verified by WP14MAY24 | F |
| SRS-8.23 | If entering radiographic mode after exiting idle state, the SS shall set the technique factors to the values selected prior to the system entering idle state |  | Verify single radiographic image is acquired at 60 kV and 0.25 mAs | Expected outcome verified. Verified by WP 14MAY24 | P |
|  | Test Case: Exit Idle State into Photo Mode |  |  |  |  |
| SRS-8.22 | After exiting from idle state, the SS shall enter the imaging mode it was in prior to entering idle state | 1. Switch to photographic mode 2. Allow system to enter idle state 3. Move the emitter for the system to exit idle state | Upon exiting idle state, verify MX1 is in photographic mode | Expected outcome verified. Verified by RM 14MAY24 | P |
|  | Test Case: Exit Idle State with Triggers and Buttons |  |  |  |  |
| SRS-8.16 | The SS shall exit idle state upon detection of any three emitter HMI button presses | 1. Pair a foot pedal to the emitter by entering the foot pedal ID in the Settings page of the MedAI Device App 2. Allow system to enter idle state. 3. Press the emitter left HMI button once the device is in idle mode. Verify system exits idle state. 4. Repeat steps 1-2 for the right HMI button, center HMI button, trigger, foot pedal button and foot pedal trigger. Note that pressing the emitter center HMI button may also result in switching imaging modes. This is expected behavior. | MX1 exits idle state when the right emitter HMI button is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | MX1 exits idle state when the center emitter HMI button is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | MX1 exits idle state when the left emitter HMI button is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-8.17 | The SS shall exit idle state upon detection of an emitter trigger press |  | MX1 exits idle state when an emitter trigger is pulled | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-8.18 | The SS shall exit idle state upon detection of a cassette power button press |  | MX1 exits idle state when the cassette power button is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-8.19 | The SS shall exit idle state upon detection of foot pedal button event |  | MX1 exits idle state when the left foot pedal button is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | MX1 exits idle state when the right foot pedal button is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-8.20 | The SS shall exit idle state upon detection of foot pedal trigger event |  | MX1 exits idle state when the left foot pedal is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | MX1 exits idle state when the right foot pedal is pressed | Expected outcome verified. Verified by RM 14MAY24 | P |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, T1 Tablet, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: MX1 System Shutdown |  |  |  |  |
| SRS-5.3 | The SS shall indicate the emitter power down by enabling the MI LEDs to blink blue for the duration of power down | 1. Start timer 2. Press and hold emitter center HMI button for 3 seconds 3. Stop timer when emitter has powered down | Emitter MI LEDs blink blue after the center HMI button has been held down for 3 seconds | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-5.1 | The SS shall initiate the emitter power down sequence when the emitter center HMI button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. |  | Emitter powers down within 5 seconds of the initial press after the center HMI button has been held down for 3 seconds | Expected outcome verified. 4.9 seconds observed. Verified by WP 14MAY24 | P |
| SRS-5.4 | The SS shall indicate the cassette power down by enabling the MI LEDs to blink blue for the duration of power down | 1. Start timer 2. Press and hold cassette power button for 3 seconds 3. Stop timer when cassette has powered down | Cassette MI LEDs blink blue after the power button has been held down for 3 seconds | Expected outcome verified. Verified by WP 14MAY24 | P |
| SRS-5.2 | The SS shall initiate the cassette power down sequence when the cassette power button is pressed and held for 3 seconds. The power down sequence shall complete within 5 seconds of the initial button press. |  | Cassette power down sequence completes in 5 seconds | Expected outcome verified. 4.3 Seconds Verified by WP 14MAY24 | P |
|  | Test Case: MX1 System Shutdown - MedAI Device App Initiated |  |  |  |  |
| SRS-5.5 | The SS shall provide a single UI element in the MedAI Device App to initiate graceful shut down of the cassette and connected emitter | 1. Ensure the cassette and emitter are powered on and in radiographic mode. Ensure a tablet with the MedAI Device App is powered on and connected to the cassette. 2. Tap the shutdown button on the App. 3. Verify a confirmation prompt is displayed in the App. | Confirmation prompt displays with the following message: “Are you sure you want to shutdown the cassette?” | Expected outcome verified. See Appendix 19. Verified by RM 14MAY24 | P |
|  |  |  | Confirmation prompt displays with “Yes” and “No” buttons | Expected outcome verified. See Appendix 19. Verified by RM 14MAY24 | P |
|  |  |  | Confirmation prompt displays with a close (“X”) button | Expected outcome verified. See Appendix 19. Verified by RM 14MAY24 | P |
|  |  | Tap the “X” button on the confirmation prompt | Prompt disappears | Expected outcome verified. See Appendix 20. Verified by RM 14MAY24 | P |
|  |  |  | System remains powered on | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  | 1. Tap the shutdown button on the App 2. Tap the “Yes” button on the confirmation prompt | Prompt disappears | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | MX1 system shuts down * Deviation: System should power off after tapping the ODA shutdown button.. See Protocol Deviations 1.b | Expected outcome verified. Verified by AM 14MAY24 | p |
|  |  |  | Emitter MI LEDs turn off after App shutdown button is tapped | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | Emitter powers down within 5 seconds of the initial tap of the App shutdown button | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | Cassette MI LEDs turn off after App shutdown button is tapped | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | Cassette powers down within 5 seconds of the initial tap of the App shutdown button | Expected outcome verified. Verified by RM 14MAY24 | P |
|  |  |  | MedAI Device App displays a modal with the following message: “Cassette was shutdown” * Deviation: ODA message is “Cassette was shutdown”, not “Shutdown complete”. See Protocol Deviations 1.c | Expected outcome verified. Verified by WP 14MAY24 | P |

### Table 12
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-440 |  |
