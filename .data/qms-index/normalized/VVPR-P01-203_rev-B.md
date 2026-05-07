# VVPR-P01-203 Rev B: MX1 Software System Motion Interlock v3.2.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-203
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.2.0
- Source filename: VVPR-P01-203 - MX1 Software System Motion Interlock v3.2.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-203 - MX1 Software System Motion Interlock v3.2.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Updates to trigger timing for single-frame serial radiographic and radioscopic acquisition
Motion interlock
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.2.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. D
IFU-MX1 - Instructions for Use, Rev. D
MATERIALS
E1 Emitter BOM Rev. H
C1 Cassette BOM Rev. I
M50133 Rev. A, Galaxy Tablet  S8+
MX1 Software System v3.2.0
APP MedAI Device App v3.2.0
Additional tools/equipment:
EQP-139 (or equivalent) Control Company Stopwatch 4YMT7
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed. If steps require x-ray emission, use appropriate radiation protective equipment.
Table 1. Motion Interlock - Requirements, Verification Steps, and Expected Results
Table 2. Single Frame Serial Radiographic/Radioscopic - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None.
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1204
C1 Cassette Rev. I, SN: 1205
M50133 Galaxy Tablet S8+, Rev. A, MPN: R52X101FM1N
MX1 Software System v3.2.0
APP MedAI Device App v3.2.0
Additional tools/equipment:
EQP-275 - Control Company Stopwatch 4YMT7
RESULTS
Table 1. Motion Interlock - Requirements, Verification Steps, and Expected Results
Table 2. Single Frame Serial Radiographic/Radioscopic - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 and 10 - Verification Evidence as Specified in Results Tables 1 through 2.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1
Appendix 2
Appendix 3
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.868380 [ec] (triggerPq) <TRIGGER> Trigger pressed with mode == -2
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.868952 [ec] (triggerPq) <TRIGGER> trigger1 is true
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.869196 [ec] (triggerPq) ExternalTrigger1 trigger pressed
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.869427 [ec] (triggerPq) <externalt> Type: 1 Code: 264 Value: 0
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.869702 [ec] (triggerPq) <PAL_EM> Button pressed on emitter: 264; 0
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.969189 [ec] (triggerPq) <TRIGGER> Trigger released with mode == -2
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.969754 [ec] (triggerPq) <TRIGGER> trigger1 is false
Sep 16 23:02:30 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:02:30.969982 [ec] (triggerPq) ExternalTrigger1 trigger released
Appendix 4
Sep 16 23:13:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:13:41.657124 [ec] (triggerPq) <TRIGGER> Trigger pressed with mode == -1
Sep 16 23:13:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:13:41.658071 [ec] (triggerPq) <TRIGGER> trigger1 is true
Sep 16 23:13:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:13:41.658460 [ec] (triggerPq) ExternalTrigger1 trigger pressed
Sep 16 23:13:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:13:41.658665 [ec] (triggerPq) <externalt> Type: 1 Code: 264 Value: 0
Sep 16 23:13:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:13:41.658918 [ec] (triggerPq) <PAL_EM> Button pressed on emitter: 264; 0
Sep 16 23:13:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:13:41.777953 [ec] (triggerPq) <TRIGGER> Trigger released with mode == -1
Appendix 5
Appendix 6
Appendix 7
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.429510 [ec] (triggerPq) <TRIGGER> Trigger pressed with mode == -1
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.429984 [ec] (triggerPq) <TRIGGER> trigger1 is true
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.430289 [ec] (triggerPq) ExternalTrigger1 trigger pressed
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.430520 [ec] (triggerPq) <externalt> Type: 1 Code: 264 Value: 0
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.430828 [ec] (triggerPq) <PAL_EM> Button pressed on emitter: 264; 0
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.735872 [ec] (trigger-eval) <XRAY> <TECHNIQUE> Using ddr techniques with emitter movie mode of ddr
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.736342 [ec] (trigger-eval) <TRIGGER> <XRAY> About to send prexray args, but first we're going to try moving the collimator into position
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.751947 [ec] (trigger-eval) <PAL_COL> Reading collimator move register
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.736561 [ec] (trigger-eval) <COLLIMATION> Requestion collimator move to size 6.25 and rotation 0
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:  DEBUG 23:29:41.787085 [ec] (trigger-eval) <PAL_BASE> CollimatorImpl flags read as
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.787405 [ec] (trigger-eval) <COLLIMATION> Collimator newSize is 6.11, newRotation is 360
Sep 16 23:29:41 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.987902 [ec] (trigger-eval) <PAL_COL> Reading collimator move register
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:  DEBUG 23:29:42.022663 [ec] (trigger-eval) <PAL_BASE> CollimatorImpl flags read as
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.023119 [ec] (trigger-eval) <COLLIMATION> After 1 tries, Collimator newSize is 6.11, newRotation is 360
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.023956 [ec] (trigger-eval) <TRIGGER> <XRAY> Sending pre xray args to cassette, will wait for cassette-ready id: 106
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.024670 [ec] (trigger-eval) <PAL_EM> <XRAY> <UNDERWAY> Underway signal: true
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.024958 [ec] (trigger-eval) Set the waitingForWrx to true
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.751947 [ec] (trigger-eval) <PAL_COL> Reading collimator move register
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.787405 [ec] (trigger-eval) <COLLIMATION> Collimator newSize is 6.11, newRotation is 360
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:41.987902 [ec] (trigger-eval) <PAL_COL> Reading collimator move register
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.023119 [ec] (trigger-eval) <COLLIMATION> After 1 tries, Collimator newSize is 6.11, newRotation is 360
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.023956 [ec] (trigger-eval) <TRIGGER> <XRAY> Sending pre xray args to cassette, will wait for cassette-ready id: 106
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.024670 [ec] (trigger-eval) <PAL_EM> <XRAY> <UNDERWAY> Underway signal: true
Sep 16 23:29:42 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:42.024958 [ec] (trigger-eval) Set the waitingForWrx to true
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.693581 [ec] (triggerPq) <PAL_EM> xray firing is false; stored value true
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.706566 [ec] (triggerPq) <PAL_MB> <XRAY> Turned off xray
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.707054 [ec] (triggerPq) <TRIGGER> <XRAY> Firing state false
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.707443 [ec] (triggerPq) <TRIGGER> Trigger released with mode == -1
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.707713 [ec] (triggerPq) <TRIGGER> trigger1 is false
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.707938 [ec] (triggerPq) ExternalTrigger1 trigger released
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.708160 [ec] (triggerPq) <externalt> Type: 1 Code: 264 Value: 1
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.708402 [ec] (triggerPq) <PAL_EM> Button pressed on emitter: 264; 1
Sep 16 23:29:47 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.711012 [ec] (trigger-eval) <TRIGGER> Trigger debounce lambda start
Sep 16 23:29:48 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.693581 [ec] (triggerPq) <PAL_EM> xray firing is false; stored value true
Sep 16 23:29:48 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.706566 [ec] (triggerPq) <PAL_MB> <XRAY> Turned off xray
Sep 16 23:29:48 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.707054 [ec] (triggerPq) <TRIGGER> <XRAY> Firing state false
Sep 16 23:29:48 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:29:47.707443 [ec] (triggerPq) <TRIGGER> Trigger released with mode == -1
Appendix 8
Appendix 9
Sep 16 23:40:06 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.699388 [ec] (triggerPq) <TRIGGER> Trigger pressed with mode == -2
Sep 16 23:40:06 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.699791 [ec] (triggerPq) <TRIGGER> trigger1 is true
Sep 16 23:40:06 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.700037 [ec] (triggerPq) ExternalTrigger1 trigger pressed
Sep 16 23:40:06 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.700242 [ec] (triggerPq) <externalt> Type: 1 Code: 264 Value: 0
Sep 16 23:40:06 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.700583 [ec] (triggerPq) <PAL_EM> Button pressed on emitter: 264; 0
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.010105 [ec] (trigger-eval) <XRAY> <TECHNIQUE> Using fluoro techniques with emitter movie mode of fluoro
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.010643 [ec] (trigger-eval) <TRIGGER> <XRAY> About to send prexray args, but first we're going to try moving the collimator into position
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.010881 [ec] (trigger-eval) <COLLIMATION> Requestion collimator move to size 6.25 and rotation 0
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.026136 [ec] (trigger-eval) <PAL_COL> Reading collimator move register
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:  DEBUG 23:40:07.068420 [ec] (trigger-eval) <PAL_BASE> CollimatorImpl flags read as
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.068910 [ec] (trigger-eval) <COLLIMATION> Collimator newSize is 6.11, newRotation is 360
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.069210 [ec] (trigger-eval) <COLLIMATION> After 0 tries, Collimator newSize is 6.11, newRotation is 360
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.069919 [ec] (trigger-eval) <TRIGGER> <XRAY> Sending pre xray args to cassette, will wait for cassette-ready id: 109
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.071398 [ec] (trigger-eval) <PAL_EM> <XRAY> <UNDERWAY> Underway signal: true
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.078954 [ec] (trigger-eval) Set the waitingForWrx to true
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.699388 [ec] (triggerPq) <TRIGGER> Trigger pressed with mode == -2
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.699791 [ec] (triggerPq) <TRIGGER> trigger1 is true
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.700037 [ec] (triggerPq) ExternalTrigger1 trigger pressed
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.700242 [ec] (triggerPq) <externalt> Type: 1 Code: 264 Value: 0
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:06.700583 [ec] (triggerPq) <PAL_EM> Button pressed on emitter: 264; 0
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.010105 [ec] (trigger-eval) <XRAY> <TECHNIQUE> Using fluoro techniques with emitter movie mode of fluoro
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.010643 [ec] (trigger-eval) <TRIGGER> <XRAY> About to send prexray args, but first we're going to try moving the collimator into position
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.010881 [ec] (trigger-eval) <COLLIMATION> Requestion collimator move to size 6.25 and rotation 0
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.026136 [ec] (trigger-eval) <PAL_COL> Reading collimator move register
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.068910 [ec] (trigger-eval) <COLLIMATION> Collimator newSize is 6.11, newRotation is 360
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.069210 [ec] (trigger-eval) <COLLIMATION> After 0 tries, Collimator newSize is 6.11, newRotation is 360
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.069919 [ec] (trigger-eval) <TRIGGER> <XRAY> Sending pre xray args to cassette, will wait for cassette-ready id: 109
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.071398 [ec] (trigger-eval) <PAL_EM> <XRAY> <UNDERWAY> Underway signal: true
Sep 16 23:40:07 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:07.078954 [ec] (trigger-eval) Set the waitingForWrx to true
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.066713 [ec] (triggerPq) <PAL_EM> xray firing is false; stored value true
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.080919 [ec] (triggerPq) <PAL_MB> <XRAY> Turned off xray
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.081328 [ec] (triggerPq) <TRIGGER> <XRAY> Firing state false
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.082947 [ec] (triggerPq) <TRIGGER> Trigger released with mode == -2
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.083363 [ec] (triggerPq) <TRIGGER> trigger1 is false
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.085137 [ec] (triggerPq) ExternalTrigger1 trigger released
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.066713 [ec] (triggerPq) <PAL_EM> xray firing is false; stored value true
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.080919 [ec] (triggerPq) <PAL_MB> <XRAY> Turned off xray
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.081328 [ec] (triggerPq) <TRIGGER> <XRAY> Firing state false
Sep 16 23:40:13 emitter-dv25 emitter-orchestrator[1695]:   INFO 23:40:13.082947 [ec] (triggerPq) <TRIGGER> Trigger released with mode == -2
Appendix 10

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in radioscopic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Move emitter before attempting radioscopy |  |  |  |  |
| SRS-12.23 | The SS shall disallow radioscopy if the emitter IMU detects movement one (1) second prior to acquisition | 1. Set the emitter in a stand and take a radioscopic acquisition without moving the emitter 2. Start timer. In less than 1 second, move the emitter and attempt to acquire radioscopic images. Ensure movement does not break tracking or positioning interlocks. 3. Verify the device does not allow x-ray emission | No x-ray emission occurs |  |  |
|  |  | 1. Start timer. Leave emitter undisturbed for 1 second 2. After one second, take a radioscopic acquisition | X-ray emission occurs |  |  |
|  | Test Case: Move emitter while taking a radioscopic acquisition |  |  |  |  |
| SRS-12.29 | The SS shall not terminate x-ray emission if the emitter IMU detects movement during radioscopic acquisition | 1. Set the emitter in a stand and start a radioscopic acquisition 2. During the acquisition, move the emitter while making sure to not break any positioning or tracking interlocks 3. Verify that x-ray emission continues despite emitter movement | X-ray emission continues while moving emitter during radioscopic acquisition |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in radioscopic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Single Frame Serial Radiographic Acquisition |  |  |  |  |
| SRS-16.2 | The SS shall acquire a single radiographic or radioscopic image if the trigger is pressed and released in less than 300 ms | 1. Ensure the system is in serial radiographic mode 2. Briefly press and release the trigger 3. Verify that x-ray emission ends upon trigger release | Check log for indication that single frame serial radiographic acquisition occurred before 300 ms of trigger press |  |  |
|  |  |  | Serial radiographic capture displayed on MedAI Device App |  |  |
|  |  |  | Serial radiographic capture consists of 1 frame |  |  |
|  |  | Test Case: Single Frame Radioscopic Acquisition |  |  |  |
|  |  | 1. Ensure the system is in radioscopic mode 2. Briefly press and release the trigger 3. Verify that x-ray emission ends upon trigger release | Check log for indication that single frame radioscopic acquisition occurred before 300 ms of trigger press |  |  |
|  |  |  | Radioscopic capture displayed on MedAI Device App |  |  |
|  |  |  | Radioscopic capture consists of 1 frame |  |  |
|  | Test Case: Multi-frame Serial Radiographic Acquisition |  |  |  |  |
| SRS-16.3 | The SS shall begin acquisition of a series of radiographic or radioscopic images if the trigger is held for longer than 300 ms | 1. Ensure the system is in serial radiographic mode 2. Press and hold the trigger for approximately 5 seconds. Use a timer to check trigger hold time 3. Verify that x-ray emission ends upon trigger release | Check log for indication that serial radiographic acquisition started after 300 ms of trigger press |  |  |
|  |  |  | Series of radiographic images displayed on MedAI Device App |  |  |
|  |  |  | Serial radiographic capture consists of 24 or 25 frames |  |  |
|  |  | Test Case: Multi-frame Radioscopic Acquisition |  |  |  |
|  |  | 1. Ensure the system is in radioscopic mode 2. Press and hold the trigger for approximately 5 seconds. Use a timer to check trigger hold time 3. Verify that x-ray emission ends upon trigger release | Check log for indication that radioscopic acquisition started after 300 ms of trigger press |  |  |
|  |  |  | Series of radioscopic images displayed on MedAI Device App |  |  |
|  |  |  | Radioscopic capture consists of 24 or 25 frames |  |  |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 18 Sep 2024 | 24-538 |

### Table 5
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-275 | 05/01/2024 | 05/01/2026 |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in radioscopic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Move emitter before attempting radioscopy |  |  |  |  |
| SRS-12.23 | The SS shall disallow radioscopy if the emitter IMU detects movement one (1) second prior to acquisition | 1. Set the emitter in a stand and take a radioscopic acquisition without moving the emitter 2. Start timer. In less than 1 second, move the emitter and attempt to acquire radioscopic images. Ensure movement does not break tracking or positioning interlocks. 3. Verify the device does not allow x-ray emission | No x-ray emission occurs | Expected outcome verified. Appendix 1 Verified by RN 18SEP24 | PASS |
|  |  | 1. Start timer. Leave emitter undisturbed for 1 second 2. After one second, take a radioscopic acquisition | X-ray emission occurs | Expected outcome verified. Appendix 2 Verified by RN 18SEP24 | PASS |
|  | Test Case: Move emitter while taking a radioscopic acquisition |  |  |  |  |
| SRS-12.29 | The SS shall not terminate x-ray emission if the emitter IMU detects movement during radioscopic acquisition | 1. Set the emitter in a stand and start a radioscopic acquisition 2. During the acquisition, move the emitter while making sure to not break any positioning or tracking interlocks 3. Verify that x-ray emission continues despite emitter movement | X-ray emission continues while moving emitter during radioscopic acquisition | Expected outcome verified. Emission continued after moving the emitter. Verified by GC 18SEP24 | PASS |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in radioscopic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Single Frame Serial Radiographic Acquisition |  |  |  |  |
| SRS-16.2 | The SS shall acquire a single radiographic or radioscopic image if the trigger is pressed and released in less than 300 ms | 1. Ensure the system is in serial radiographic mode 2. Briefly press and release the trigger 3. Verify that x-ray emission ends upon trigger release | Check log for indication that single frame serial radiographic acquisition occurred before 300 ms of trigger press | Expected outcome verified. 102 ms trigger pull. Appendix 4 Verified by GC 18SEP24 | PASS |
|  |  |  | Serial radiographic capture displayed on MedAI Device App | Expected outcome verified. Appendix 6 Verified by GC 18SEP24 | PASS |
|  |  |  | Serial radiographic capture consists of 1 frame | Expected outcome verified. Appendix 6 Verified by GC 18SEP24 | PASS |
|  |  | Test Case: Single Frame Radioscopic Acquisition |  |  |  |
|  |  | 1. Ensure the system is in radioscopic mode 2. Briefly press and release the trigger 3. Verify that x-ray emission ends upon trigger release | Check log for indication that single frame radioscopic acquisition occurred before 300 ms of trigger press | Expected outcome verified. 102 ms trigger pull. Appendix 3 Verified by GC 18SEP24 | PASS |
|  |  |  | Radioscopic capture displayed on MedAI Device App | Expected outcome verified. Appendix 5 Verified by GC 18SEP24 | PASS |
|  |  |  | Radioscopic capture consists of 1 frame | Expected outcome verified. Appendix 5 Verified by GC 18SEP24 | PASS |
|  | Test Case: Multi-frame Serial Radiographic Acquisition |  |  |  |  |
| SRS-16.3 | The SS shall begin acquisition of a series of radiographic or radioscopic images if the trigger is held for longer than 300 ms | 1. Ensure the system is in serial radiographic mode 2. Press and hold the trigger for approximately 5 seconds. Use a timer to check trigger hold time 3. Verify that x-ray emission ends upon trigger release | Check log for indication that serial radiographic acquisition started after 300 ms of trigger press | Expected outcome verified. Appendix 7 Verified by GC 18SEP24 | PASS |
|  |  |  | Series of radiographic images displayed on MedAI Device App | Expected outcome verified. Appendix 8 Verified by GC 18SEP24 | PASS |
|  |  |  | Serial radiographic capture consists of 24 or 25 frames | Expected outcome verified. 26 Frames captured in 5.31 seconds Verified by GC 18SEP24 | PASS |
|  |  | Test Case: Multi-frame Radioscopic Acquisition |  |  |  |
|  |  | 1. Ensure the system is in radioscopic mode 2. Press and hold the trigger for approximately 5 seconds. Use a timer to check trigger hold time 3. Verify that x-ray emission ends upon trigger release | Check log for indication that radioscopic acquisition started after 300 ms of trigger press | Expected outcome verified. Appendix 9 Verified by GC 18SEP24 | PASS |
|  |  |  | Series of radioscopic images displayed on MedAI Device App | Expected outcome verified. Appendix 10 Verified by GC 18SEP24 | PASS |
|  |  |  | Radioscopic capture consists of 24 or 25 frames | Expected outcome verified. 27 Frames captured in 5.29 seconds Verified by GC 18SEP24 | PASS |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-470 |  |
