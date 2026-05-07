# VVPR-P01-181 Rev B: MX1 Software System Critical Faults v3.1.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-181
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.1.0
- Source filename: VVPR-P01-181 - MX1 Software System Critical Faults v3.1.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-181 - MX1 Software System Critical Faults v3.1.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Temperature faults for emitter main PCB, monoblock LV PCB, collimator PCB, and cassette main PCB
Voltage monitor faults for emitter main PCB, monoblock LV PCB, collimator PCB, and cassette main PCB
Monoblock technique factor faults
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification.
The scope of this study is limited to the verification of the propagation of critical faults from peripheral MCU boards to a proper response by the jetson.
A fault is reported by peripheral MCU’s to the Jetson when the nominal value of monitored voltage rails and temperature sensors exceeds a specific threshold specified by firmware. There are two possible methods of tripping said faults
Manipulation of hardware and environmental conditions to exceed threshold values set in firmware
Manipulation of thresholds so that faults will be reported under normal operating conditions without manipulation of hardware or environment
Faults related to beam current and timing during radiographic exposure will use the first method while all other faults utilize the second.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev B
IFU-MX1 - Instructions for Use, Rev D
MATERIALS
MX1 System Rev. E Components:
E1 Emitter Rev. H
C1 Cassette Rev. I
MX1 Software System v3.1.0-beta
Additional tools/equipment:
Rigol DP712 or DP711 Calibrated Benchtop Power Supply - EQP-111 or EQP-112
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI, Inc office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI, Inc engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Test Setup Background
This test involves the manipulation of existing voltage/temperature monitoring thresholds to trigger faults and their appropriate response. Table 1 depicts voltage thresholds during normal operating conditions, the nominal values of these lines during normal operation conditions, and the adjusted thresholds used to trigger faults. Table 2 depicts temperature fault thresholds during normal operating conditions, the nominal values of these lines during normal operation conditions, and the adjusted thresholds used to trigger faults.
Each threshold can be changed via a specific byte sequence written to register 0xE8(for voltage) or 0xD1(for temperature). The exact byte sequence is [sensor #][Threshold Value][high or low threshold?]. For ease of conducting this test, the ICD payload byte sequence has been pre-calculated and inserted into the “ICD Payload” Column of table 1 and table 2. In order to reset all threshold values to normal operating values, the payload 00000000 needs to be sent to the corresponding voltage threshold register (0xE8) or temperature threshold register (0xD1)
Table 1. Voltage Faults and Thresholds
Table 2. Temperature Faults and Thresholds
EXPERIMENTAL PROCEDURE
Follow the steps outlined in Tables 3 through 6 below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 3. Safe State - Requirements, Verification Steps, and Expected Results
Table 4. Temperature Fault Testing - Requirements, Verification Steps, and Expected Results
Table 5. Voltage Monitoring Faults - Requirements, Verification Steps, and Expected Results
Table 6. Technique Faults - Requirements, Verification Steps, and Expected Results
Table 7. Collimator Timeout Fault - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests in Tables 3 through 6 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 3 through 6 per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
NOTE: Some steps required additional commits to the firmware in order to allow the ability for critical faults to be tripped safely. Specifically, code was added to achieve the following testing functionality with software v3.1.0. The specific changes and reasoning are listed below.
Deviations in Table 6 - Technique Faults Testing
The entirety of Protocol table 6 was replaced by Table 6A below. The changes and reasoning behind them are as follows:
SRS-20.12 - The purpose of this test is to ensure that the Monoblock LV detects when the exposure time exceeds a certain amount and causes the software system to go into a safe state. The fault was unable to be tested with the original protocol as the software system prevents the tester from setting an exposure time of >200ms while the hardware-based threshold for tripping this fault is 210 ms. In order to properly test this fault, register 0x20 was added to allow the setting of a 220 ms exposure shot.
Changes made to test SRS-20.12 in Table 6:
Addition of register 0x04 that enables/disables the ability to modify the exposure time set limit. This register requires a specific bit sequence “password” to be enabled during testing
Addition of register 0x20 which allows the tester to change the settable exposure time limit. The over-exposure-time fault is triggered by a hardware circuit when the X_RAY_ON net/pin on the monoblock exceeds a period of 210 ms. In the Monoblock Firmware, a limit exists for the settable exposure time (200 ms). This register now allows us to adjust the settable exposure threshold, allowing us to trigger X-ray exposures of >210 ms in order to trigger the hardware based exposure-time fault. Testers must still manually set the exposure time window; this register is purely a write to the limit that can be written to the exposure time window register
SRS-20.7 - The purpose of this test is to ensure that the software system detects and triggers safe state when the tube voltage exceeds a specified threshing during x-ray exposure. The original protocol instructs testers to set an invalid tube voltage. However, the software system prevents this. Instead, the protocol was modified to use register 0x11 to adjust the voltage threshold that will cause a tube HV fault.
SRS-20.8 - The purpose of this test is to ensure that the software system detects and triggers safe state when the tube voltage exceeds a specified threshold during x-ray-non-exposure. The original protocol instructs the tester to inject a voltage to a testpoint on the Monoblock LV PCBA. The protocol was modified to instead use register 0x11 to adjust the voltage threshold that will cause a tube HV fault. This prevents the opening of a device and prevents potential damage to the device.
Changes made to test SRS-20.7 and SRS-20.8 in Table 6:
Addition of register 0x11 which allows the tester to change the threshold at which the tube HV fault is triggered. This register was added in order to: 1) provide uniformity in the testing of critical faults by adjusting thresholds, 2) prevents the potential damaging of the device via the injection of a voltage, 3) prevents unsafe radiation exposure to the tester
SRS-20.17 - The purpose of this test is to ensure that when the filament current exceeds a certain threshold during x-ray exposure, the software system enters safe state. In the original protocol, the endpoint in step 2 has been updated and the software system prevents users from setting the beam current above 2.0 mA. However, the threshold to trigger a fault is 2.4 mA. In order to circumnavigate this, a different strategy was employed. Much like the temperature and voltage faults, the filament current threshold that is considered a critical fault was modified in order to trip an out-of-bounds filament beam current fault even when firing with mA values well within normal operating values.
SRS-20.10 - The purpose of this test is to ensure that the software system detects and triggers safety state when the tube beam current exceeds a specified current threshold during x-ray non-exposure. The original protocol instructs the tester to inject a voltage to a testpoint on the Monoblock LV PCBA. The protocol was modified to instead, use register 0x44 to adjust the current threshold that will trip a current fault. This prevents the opening of a device and prevents potential damage to the device.
Changes made to test SRS-20.17 and SRS-20.10 in Table 6
Addition of register 0x44 which allows the tester to change the threshold at which the beam current fault is triggered. This register was added in order to: 1) provide uniformity in the testing of critical faults by adjusting thresholds, 2) prevents the potential damaging of the device via the injection of a voltage, 3) prevents unsafe radiation exposure to the tester
Addition of SRS-20.6 and SRS-20.9
During the drafting of the original procedure, SRS-20.6 and SRS-20.9 were accidentally omitted from the steps that tested them. However, the tests exist and have been conducted.
SRS-20.6 - The SS shall detect and report an out-of-bounds power supply (PWS) voltage event as a critical fault
Tested in Table 5, Monoblock Voltage Faults
Results are a pass and evidence shown in appendix 47 and appendix 48
SRS-20.9 - The SS shall detect and report an out-of-bounds beam current event during exposure as a critical fault
Tested in Table 6A, Out-of-Bounds Beam Current, Tube VOltage, and Exposure Time Faults
Results are a pass and evidence shown in appendix 86
Table 6A. Technique Faults -  Requirements, Verification Steps, and Expected Results - DEVIATIONS
Deviations in Table 4 - Temperature Fault Testing
SRS-20.14 - The purpose of this test is to ensure that the software system detects and enters safe state when any temperatures on the cassette exceed a lower or upper threshold. In table 2, the first byte in the payloads for CAS_BMS_TS2, CAS_BMS_TS3, and CAS_BMS_TSINT are incorrect. This was due to a mistranslation from a decimal value to a hexadecimal value. The corrected payloads can be found below in table 2A
Table 2A. Temperature Faults and Thresholds - DEVIATIONS
DEVICES, COMPONENTS, OR EQUIPMENT USED
MX1 Emitter E1 SN: 1220
MX1 Cassette C1 SN: 1079
MX1 Wired chargers H1 Lot: 10001 (qty 2)
MX1 Software System 3.1.0-beta
Lead Shielding
RESULT
Result Table 1 Safe State - Requirements, Verification Steps, and Expected Results
Result Table 2 Temperature Fault Testing - Requirements, Verification Steps, and Expected Results
Result Table 3 - Temperature Faults and Thresholds
Result Table 4 - Voltage Monitoring Faults - Requirements, Verification Steps, and Expected Results
Results Table 5 - Voltage Faults and Thresholds
Result Table 6 - Technique Faults -  Requirements, Verification Steps, and Expected Results
Result Table 7 - Collimator Timeout Fault - Requirements, Verification Steps, and Expected Results
DISCUSSION
This report has tested that the software system successfully enters safe-sate upon the detection of all critical fault conditions including:
Over and under voltage conditions on the Collimator PCBA, Monoblock LV PCBA, Collimator PCBA, Emitter main PCBA, and Cassette main PCBA
Over and under Temperature conditions on the Monoblock LV PCBA, Collimator PCBA, Emitter main PCBA, Cassette main PCBA, Cassette battery, and Emitter battery
Collimation Timeout
Technique faults:
Out of bounds tube voltage during exposure
Out of bounds tube voltage during non-exposure
Out of bounds beam current during exposure
Out of bounds beam current during non-exposure
Over exposure of radiation with regards to time
Deviations are listed above in the deviations sections before the results tables. Procedural deviations were associated with alternative methods to trip the technique faults with a safe and non-destructive methodology. There were two deviations associated with ICD payloads in the Cassette main temperature faults section due to arithmetic errors when converting decimal numbers to hexadecimal values.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
APPENDICES
Appendix 1:
Appendix 2:
Appendix 3:
Appendix 4:
Appendix 5:
Appendix 6:
Appendix 7:
Appendix 8:
Appendix 9:
Appendix 10:
Appendix 11:
Appendix 12:
Appendix 13:
Appendix 14:
Appendix 15:
Appendix 16:
Appendix 17:
Appendix 18:
Appendix 19:
Appendix 20:
Appendix 21:
Appendix 22:
Appendix 23:
Appendix 24:
Appendix 25:
Appendix 26:
Appendix 27:
Appendix 28:
Appendix 29:
Appendix 30:
Appendix 31:
Appendix 32:
Appendix 33:
Appendix 34:
Appendix 35:
Appendix 36:
Appendix 37:
Appendix 38:
Appendix 39:
Appendix 40:
Appendix 41:
Appendix 42:
Appendix 43:
Appendix 44:
Appendix 45:
Appendix 46:
Appendix 47:
Appendix 48:
Appendix 49:
Appendix 50:
Appendix 51:
Appendix 52:
Appendix 53:
Appendix 54:
Appendix 55:
Appendix 56:
Appendix 57:
Appendix 58:
Appendix 59:
Appendix 60:
Appendix 61:
Appendix 62:
Appendix 63:
Appendix 64:
Appendix 65:
Appendix 66:
Appendix 67:
Appendix 68:
Appendix 69:
Appendix 70:
Appendix 71:
Appendix 72:
Appendix 73:
Appendix 74:
Appendix 75:
Appendix 76:
Appendix 77:
Appendix 78:
Appendix 79:
Appendix 80:
Appendix 81:
Appendix 82:
Appendix 83:
Appendix 84:
Appendix 85:
Appendix 86:
Appendix 87:
Appendix 88:
Appendix 89:
Appendix 90:
Appendix 91:
Appendix 92:
Appendix 93:
Appendix 94:
Appendix 95:
Appendix 96:
Appendix 97:
Appendix 98:
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Board | Signal | Nominal Value | Fault Threshold | Modified Threshold | ICD Payload |
| --- | --- | --- | --- | --- | --- |
| Monoblock | VBAT_MON | 24 V | > 40 V | > 18 V | 08080701 |
|  |  |  | < 18 V | < 40 V | 08A00F00 |
|  | XRAY_PWS | 81 V | > 86 V | > 9 V | 04840301 |
|  |  |  | < 9 V | < 86 V | 04982100 |
|  | 5V0 | 5 V | > 5.3 V | > 4.7 V | 00D60101 |
|  |  |  | < 4.7 V | < 5.3 V | 00120200 |
|  | 3V3 | 3.3 V | > 3.5 V | > 3 V | 012C0101 |
|  |  |  | < 3 V | < 3.5 V | 015E0100 |
|  | 15V0_FIL | 15 V | > 15.5 V | > 14.5 V | 02AA0501 |
|  |  |  | < 14.5 V | < 15.5 V | 020E0600 |
|  | M15V0 | 15 V | > 15.5 V | > 14.5 V | 06AA0501 |
|  |  |  | < 14.5 V | < 15.5 V | 060E0600 |
|  | P15V0 | 15 V | > 15.5 V | > 14.5 V | 05AA0501 |
|  |  |  | < 14.5 V | < 15.5 V | 050E0600 |
|  | P3V3_ANA | 3.3 V | > 3.5 V | > 3.1 V | 07360101 |
|  |  |  | < 3.1 V | < 3.3 V | 074A0100 |
| Emitter Main | BAT_V_MON | 24 V | > 39 V | > 18 V | 00080701 |
|  |  |  | < 18 V | < 39 V | 003C0F00 |
|  | DISPLAY_V_MON | 24 V | > 39 V | > 18 V | 02080701 |
|  |  |  | < 18 V | < 39 V | 023C0F00 |
|  | COL_V_MON | 24 V | > 39 V | > 18 V | 03080701 |
|  |  |  | < 18 V | < 39 V | 033C0F00 |
|  | JTSN_5V0 | 5 V | > 5.25 V | > 4.8 V | 04E00101 |
|  |  |  | < 4.8 V | < 5.25 V | 040D0200 |
|  | CONTROL_5V0 | 5 V | > 5.5 V | > 4.5 V | 06C20101 |
|  |  |  | < 4.5 V | < 5.5 V | 06260200 |
| Collimator | COL_V_MON | 28 V | > 29 V | > 19 V | 096C0701 |
|  |  |  | < 19 V | < 29 V | 09540B00 |
|  | 5V0 | 5 V | > 5.5 V | > 4.5 V | 08C20101 |
|  |  |  | < 4.5 V | < 5.5 V | 08260200 |
| Cassette Main | BAT_V_MON | 14 V | > 21 V | > 9.8 V | 01D40301 |
|  |  |  | < 9.8 V | < 21 V | 01340800 |
|  | CONTROL_5V0 | 5 V | > 5.15 V | > 4.78 V | 03DE0101 |
|  |  |  | < 4.78 V | < 5.15 V | 03030200 |
|  | JET_3V3 | 3.3 V | > 3.4 V | > 3.15 V | 063B0101 |
|  |  |  | < 3.15 V | < 3.4 V | 06540100 |
|  | DET_22V0 | 22 V | > 23 V | > 19 V | 076C0701 |
|  |  |  | < 19 V | < 23 V | 07FC0800 |
|  | JET_5V0 | 5 V | > 5.15 V | > 4.8 V | 02E00101 |
|  |  |  | < 4.8 V | < 5.15 V | 02030200 |

### Table 3
| Board | Signal | Nominal Value | Fault Threshold | Modified Threshold | ICD Payload |
| --- | --- | --- | --- | --- | --- |
| Monoblock | MB_HEATSINK | 22 C | > 80 C | > 0 C | 00000001 |
|  |  |  | < 0 C | < 80 C | 00401F00 |
|  | MB_SIDE | 22 C | > 80 C | > 0C | 01000001 |
|  |  |  | < 0 C | < 80 C | 01401F00 |
| Emitter Main | MB_HEAT_PIPE | 22 C | > 70 C | > 0 C | 00000001 |
|  |  |  | < 0 C | < 80 C | 00401F00 |
|  | EM_HANDLE | 22 C | > 48 C | > 0 C | 01000001 |
|  |  |  | < 0 C | < 80 C | 01401F00 |
|  | EM_PCB | 22 C | > 80 C | > 0 C | 02000001 |
|  |  |  | < 0 C | < 80 C | 02401F00 |
|  | EM_PMUX | 22 C | > 95 C | > 0 C | 03000001 |
|  |  |  | < 0 C | < 95 C | 031C2500 |
|  | EM_BMS_TS1 | 22 C | > 60 C | > 0 C | 04000001 |
|  |  |  | < 0 C | < 80 C | 04401F00 |
|  | EM_BMS_TS2 | 22 C | > 60 C | > 0 C | 05000001 |
|  |  |  | < 0 C | < 80 C | 05401F00 |
|  | EM_BMS_TS3 | 22 C | > 60 C | > 0 C | 06000001 |
|  |  |  | < 0 C | < 80 C | 06401F00 |
|  | EM_BMS_TSINT | 22 C | > 80 C | > 0 C | 07000001 |
|  |  |  | < 0 C | < 80 C | 07401F00 |
| Cassette Main | CAS_BAT_CON | 22 C | > 91 C | > 0 C | 00000001 |
|  |  |  | < 0 C | < 80 C | 00401F00 |
|  | CAS_OP_AMP | 22 C | > 80 C | > 0 C | 01000001 |
|  |  |  | < 0 C | < 80 C | 01401F00 |
|  | CAS_MCU | 22 C | > 80 C | > 0 C | 02000001 |
|  |  |  | < 0 C | < 80 C | 02401F00 |
|  | CAS_CHGR | 22 C | >95 C | > 0 C | 03000001 |
|  |  |  | < 0 C | < 95 C | 031C2500 |
|  | CAS_JTSN_PWR | 22 C | > 70 C | > 0 C | 04000001 |
|  |  |  | < 0 C | < 80 C | 04401F00 |
|  | CAS_JTSN | 22 C | > 80 C | > 0 C | 05000001 |
|  |  |  | < 0 C | < 80 C | 05401F00 |
|  | CAS_LTE | 22 C | > 80 C | > 0 C | 06000001 |
|  |  |  | < 0 C | < 80 C | 06401F00 |
|  | CAS_WIFI | 22 C | > 80 C | > 0 C | 07000001 |
|  |  |  | < 0 C | < 80 C | 07401F00 |
|  | CAS_J13 | 22 C | > 65 C | > 0 C | 08000001 |
|  |  |  | < 0 C | < 80 C | 08401F00 |
|  | CAS_BMS_TS1 | 22 C | > 60 C | > 0 C | 09000001 |
|  |  |  | < 0 C | < 80 C | 09401F00 |
|  | CAS_BMS_TS2 | 22 C | > 60 C | > 0 C | 10000001 |
|  |  |  | < 0 C | < 80 C | 10401F00 |
|  | CAS_BMS_TS3 | 22 C | > 60 C | > 0 C | 11000001 |
|  |  |  | < 0 C | < 80 C | 11401F00 |
|  | CAS_BMS_TSINT | 22 C | > 80 C | > 0 C | 12000001 |
|  |  |  | < 0 C | < 80 C | 12401F00 |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: E1 Emitter, C1 Cassette, M50133 Galaxy Tablet  S8+ Rev. A, APP MedAI Device App Emitter and cassette are placed in T-063 MX1 Positioning Jig. Emitter is fitted with a modified side shell that makes the Monoblock power light visible. Precondition: All device components are powered on and in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |  |
| Test Case: System behavior when in safe state |  |  |  |  |  |
| SRS-19.2 | When in safe state, the SS shall set all MI (Mode Indicator) LEDs on the cassette to magenta | To force the system into safe state, use the following steps to stop the emitter-frontend service: 1. SSH into the cassette via ssh imager@<cassette hostname> 2. Using the cassette session, ssh into the emitter via imager@<emitter hostname> 3. Navigate to /opt/medai/bin 4. End the emitter frontend service with the systemctl --user stop emitter-frontend 5. Run python3.8 -m mx1.services status to get a list of running services. Ensure emitter-frontend is not listed. | All cassette MI LEDs set to magenta |  |  |
| SRS-19.4 | Upon entering safe state, the SS shall set the IR LED brightness to 0 |  | A smartphone without an IR filter on its camera can see the IR LEDs are off |  |  |
| SRS-19.3 | Upon entering safe state, the SS shall kill power to the detector |  | Ping of detector IP address (192.168.8.8) is unsuccessful |  |  |
| SRS-19.5 | When in safe state, the SS shall set all MI LEDs on the emitter to magenta |  | All emitter MI LEDs set to magenta |  |  |
| SRS-19.7 | Upon entering safe state, the SS shall disable lasers |  | Lasers turn off |  |  |
| SRS-19.6 | Upon entering safe state, the SS shall disable power to the Monoblock |  | Monoblock power light is off |  |  |
| SRS-19.8 | When in safe state, the SS shall display an error message via notification on the MedAI Device App |  | Error message displayed on MedAI Device App |  |  |
| SRS-19.9 | When in safe state, the SS shall indicate via notification on the cassette display |  | Cassette Display shows “Error” |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: E1 Emitter, C1 cassette Precondition: All device components are powered on and in radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |  |
| Test Case: Emitter Main Temperature Fault Testing |  |  |  |  |  |
| SRS-20.1 | The SS shall detect and report any out-of-bounds emitter temperature event as a critical fault | 1. Ensure that ethernet is plugged in to the emitter via USB-C 2. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081/remote_api?command=p&reg=D1&pid=3&op=1&payload=%22{payload}%22 3. Ensure that the Emitter device enters magenta safe state 4. In the cassette-orchestrator debug window, ensure that the Emitter Device has entered safety state and that the FLAG_EVT_TEMPERATURE_FAULT is displayed in the ememittermainFlags.  Take a screenshot for proof 5. In a web browser, send the ICD command: example.com/}.local:8081//remote_api?command=p&reg=D2&pid=3&op=0&payload=%22{payload}%22 6. Ensure that the bit corresponding to the bit position for the threshold you are testing is set to 1 (Table 2) 7. Reset all thresholds by sending the command: example.com/}.local:8081/remote_api?command=p&reg=D1&pid=3&op=1&payload=%2200000000%22 8. Repeat steps 1-7 for all Temperature thresholds in table 2 | FLAG_EVT_TEMPERATURE_FAULT is displayed in the ememittermainFlags row of the cassette-orchestrator debug window |  |  |
|  |  |  | Corresponding bit is set as 1 in the read response to register 0xD2 via ICD |  |  |
|  |  |  | Cassette and Emitter MI LEDs turn magenta upon detecting a critical fault |  |  |
|  | The SS shall enter safe state upon detection of a critical fault |  |  |  |  |
| SRS-19.1 |  |  |  |  |  |
| Test Case: Monoblock Temperature Fault Testing |  |  |  |  |  |
| SRS-20.11 | The SS shall detect and report any out-of-bounds Monoblock temperature event as a critical fault | 1. Ensure that ethernet is plugged in to the emitter via USB-C 2. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081/remote_api?command=p&reg=D1&pid=1&op=1&payload=%22{payload}%22 3. Ensure that the Emitter device enters safe state 4. In the cassette-orchestrator debug window, ensure that the Emitter Device has entered safety state and that the FLAG_EVT_TEMPERATURE_FAULT is displayed in the monoblockFlags.  Take a screenshot for proof 5. In a web browser, send the ICD command: example.com/}.local:8081//remote_api?command=p&reg=D2&pid=1&op=0&payload=%22{payload}%22 6. Ensure that the bit corresponding to the bit position for the threshold you are testing is set to 1 (Table 1) 7. Reset all thresholds by sending the command: example.com/}.local:8081/remote_api?command=p&reg=D1&pid=1&op=1&payload=%2200000000%22 8. Repeat steps 1-7 for all Temperature thresholds in table 1 | FLAG_EVT_TEMPERATURE_FAULT is displayed in the ememittermainFlags row of the cassette-orchestrator debug window |  |  |
|  |  |  | Bits 1 and 2 are set as 1 in the read response to register 0xD2 via rest_icd |  |  |
|  |  |  | Cassette and Emitter MI LEDs turn magenta upon detecting a critical fault |  |  |
| SRS-19.1 | The SS shall enter safe state upon detection of a critical fault |  |  |  |  |
| Test Case: Cassette Temperature Fault Testing |  |  |  |  |  |
| SRS-20.14 | The SS shall detect and report any out-of-bounds cassette temperature event as a critical fault | 1. Ensure that ethernet is plugged in to the Cassette via USB-C 2. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081/remote_api?command=p&reg=D1&pid=5&op=1&payload=%22{payload}%22 3. Ensure that the Cassette device enters magenta safe state 4. In the cassette-orchestrator debug window, ensure that the Emitter Device has entered safety state and that the FLAG_EVT_TEMPERATURE_FAULT is displayed in the cscassettemainFlags.  Take a screenshot for proof 5. In a web browser, send the ICD command: example.com/}.local:8081//remote_api?command=p&reg=D2&pid=5&op=0&payload=%22{payload}%22 6. Ensure that the bit corresponding to the bit position for the threshold you are testing is set to 1 (Table 1) 7. Reset all thresholds by sending the command: example.com/}.local:8081/remote_api?command=p&reg=D1&pid=5&op=1&payload=%2200000000%22 8. Repeat steps 1-7 for all Temperature thresholds in table 1 | FLAG_EVT_TEMPERATURE_FAULT is displayed in the cassettemainFlags row of the cassette-orchestrator debug window |  |  |
|  |  |  | Bits 0-12 are set as 1 in the read response to register 0xD2 via rest_icd |  |  |
| SRS-19.1 | The SS shall enter safe state upon detection of a critical fault |  |  |  |  |
|  |  |  | Cassette and Emitter MI LEDs turn magenta upon detecting a critical fault |  |  |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: E1 Emitter, C1 cassette |  |  |  |  |  |
| Test Case: Emitter Voltage Faults |  |  |  |  |  |
| SRS-20.3 | The SS shall detect and report any out-of-bounds emitter voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the emittermainFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the emittermainFlags row |  |  |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1). example.com/}.local:8081//remote_api?command=p&reg=E8&pid=3&op=1&payload=%22{payload}%22 2. Ensure that the Emitter device enters magenta safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the emittermainFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values: example.com/}.local:8081//remote_api?command=p&reg=E8&pid=3&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all emitter main PCB voltage thresholds specified in Table 1 Note: Voltage event flags clear upon system restart | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the emittermainFlags row for the corresponding threshold under test |  |  |
| Test Case: Cassette Voltage Faults |  |  |  |  |  |
| SRS-20.16 | The SS shall detect and report any out-of-bounds cassette voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the cscassetteFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the Flags row |  |  |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081//remote_api?command=p&reg=E8&pid=5&op=1&payload=%22{payload}%22 2. Ensure that the Cassette device enters magenta safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the cscassettemainFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values: example.com/}.local:8081//remote_api?command=p&reg=E8&pid=5&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all cassette main PCB voltage thresholds specified in Table 1 Note: Voltage event flags clear upon system restart | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the cscassetteFlags row for the corresponding threshold under test |  |  |
| Test Case: Monoblock Voltage Faults |  |  |  |  |  |
| SRS-20.13 | The SS shall detect and report any out-of-bounds monoblock voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the monoblockFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the monoblockFlags row |  |  |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081//remote_api?command=p&reg=E8&pid=1&op=1&payload=%22{payload}%22 2. Ensure that the Emitter device enters magenta safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the monoblockFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values example.com/}.local:8081//remote_api?command=p&reg=E8&pid=1&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all monoblock PCB voltage thresholds specified in Table 1 Note: Voltage event flags clear upon system restart | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the monoblockFlags row for the corresponding threshold under test |  |  |
| Test Case: Collimator Voltage Faults |  |  |  |  |  |
| SRS-20.5 | The SS shall detect and report any out-of-bounds Collimator voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the collimatorFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the collimatorFlags row |  |  |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1): example.com/}.local:8081//remote_api?command=p&reg=E8&pid=4&op=1&payload=%22{payload}%22 2. Ensure that the Emitter device enters safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the collimatorFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values: example.com/}.local:8081//remote_api?command=p&reg=E8&pid=4&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all collimator PCB voltage thresholds specified in Table 1 | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the collimatorFlags row for the corresponding threshold under test |  |  |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter, C1 Cassette Emitter and cassette are placed in T-063 MX1 Positioning Jig. Ensure emitter is positioned within the acceptable SID range. |  |  |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode. |  |  |  |  |
| Test Case: Out-of-Bounds Beam Current, Tube Voltage, and Exposure Time Faults |  |  |  |  |  |
| SRS-20.17 | The SS shall detect and report an out-of-bounds filament current event as a critical fault | 1. Set the mode to radiographic mode with http://<ip address>:8081/remote_api?command=mode_set&mode=xray_manual 2. Set an invalid beam current fault with http://<ip address>:8081/remote_api?command=set_techniques&kv=40&exposure=40&beam_current=2.5 3. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 4. Wait for FLAG_EVT_BEAM_CURR_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 5. Take an image for evidence | FLAG_EVT_BEAM_CURR_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| SRS-20.7 | The SS shall detect and report an out-of-bounds tube voltage event during exposure as a critical fault | 1. Set an invalid tube voltage fault with example.com/}:8081/remote_api?command=p&reg=11&pid=1&op=1&payload=%2210000000%22 2. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 3. Wait for FLAG_EVT_HV_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 4. Take an image for evidence | FLAG_EVT_HV_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| SRS-20.12 | The SS shall detect and report an out-of-bounds exposure time event as a critical fault | 1. Set an invalid exposure time fault with http://<ip address>:8081/remote_api?command=set_techniques&kv=40&exposure=220&beam_current=1 2. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 3. Wait for FLAG_EVT_HV_ON_TIME_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 4. Take an image for evidence Note: Restart the system to clear the out-of-bounds exposure time fault prior to running the following tests | FLAG_EVT_HV_ON_TIME_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode. Emitter Sideshell off in order to access Monoblock test points |  |  |  |  |
| Test Case: Non-Zero Beam Current and Tube Voltage Faults |  |  |  |  |  |
| SRS-20.10 | The SS shall detect and report a non-zero beam current event during non-exposure as a critical fault | 1. Using a calibrated bench power supply, apply 1.5mV between TP30 and TP31, the Isense circuit input equivalent to a false x-ray beam current of 1.5mA. 2. Wait for FLAG_EVT_BEAM_NON_EXPOSURE_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 3. Take an image for evidence | FLAG_EVT_BEAM_NON_EXPOSURE_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| SRS-20.8 | The SS shall detect and report a non-zero tube voltage event during non-exposure as a critical fault | 1. Using a calibrated bench power supply, apply 5V between Pin9 and Pin10 of J2 on the MB LV PCB, the Vsense circuit input equivalent to a false x-ray beam Voltage of 45kV. 2. Wait for FLAG_EVT_HV_NON_EXPOSURE_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 3. Take an image for evidence | FLAG_EVT_HV_NON_EXPOSURE_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter, C1 Cassette Emitter and cassette are placed in T-063 MX1 Positioning Jig. Ensure emitter is positioned within the acceptable SID range. |  |  |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode, front face is removed |  |  |  |  |
| Test Case: Collimator Timeout |  |  |  |  |  |
| SRS-20.4 | The SS shall detect and report a collimator timeout event as a critical fault | 1. With the front face removed, unplug both mother cables from the collimator. 2. In a browser window, send the following command: example.com/}.local:8081//remote_api?command=p&reg=81&pid=4&op=1&payload=%22D0403500A086010000%22 3. Ensure that the collimator debug LED is blinking red 4. Ensure that the FLAG_COL_TIMEOUT_FAULT appears in the collimatorFlags row of the cassette-orchestrator debug window | FLAG_COL_TIMEOUT_FAULT appears in the collimatorFlags row of the cassette-orchestrator debug window |  |  |

### Table 9
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 17 May 2024 | 24-251 |

### Table 10
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter, C1 Cassette |  |  |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode. |  |  |  |  |
| Test Case: Out-of-Bounds Beam Current, Tube Voltage, and Exposure Time Faults |  |  |  |  |  |
| SRS-20.17 | The SS shall detect and report an out-of-bounds filament current event as a critical fault | 1. Set the acceptable beam current threshold to be 5 mA with: example.com/}.local:8081//remote_api?command=p&reg=44&pid=1&op=1&payload=%220005%22 3. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 4. Wait for FLAG_EVT_BEAM_CURR_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 5. Take an image for evidence | FLAG_EVT_BEAM_CURR_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| SRS-20.7 | The SS shall detect and report an out-of-bounds tube voltage event during exposure as a critical fault | 1. Set the acceptable tube voltage threshold to be 60 kV with: example.com/}.local:8081//remote_api?command=p&reg=11&pid=1&op=1&payload=%2260EA0000%22 2. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 3. Wait for FLAG_EVT_HV_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 4. Take an image for evidence | FLAG_EVT_HV_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| SRS-20.12 | The SS shall detect and report an out-of-bounds exposure time event as a critical fault | 1. Unlock exposure time limit changing capabilities with: example.com/}.local:8081//remote_api?command=p&reg=04&pid=1&op=1&payload=%2201%22 2. Set the exposure time window limit to be 255 ms with: example.com/}.local:8081//remote_api?command=p&reg=20&pid=1&op=1&payload=%22FF%22 3. Set the exposure time window to be 254 ms with: example.com/}.local:8081//remote_api?command=p&reg=13&pid=1&op=1&payload=%22FE%22 4.  Wait for FLAG_EVT_HV_ON_TIME_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 5. Take an image for evidence Note: Restart the system to clear the out-of-bounds exposure time fault prior to running the following tests | FLAG_EVT_HV_ON_TIME_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode. Emitter Sideshell off in order to access Monoblock test points |  |  |  |  |
| Test Case: Non-Zero Beam Current and Tube Voltage Faults |  |  |  |  |  |
| SRS-20.10 | The SS shall detect and report a non-zero beam current event during non-exposure as a critical fault | 1. Set the acceptable beam current threshold to be 1 mA with:example.com/}.local:8081//remote_api?command=p&reg=44&pid=1&op=1&payload=%220001%22 2. Wait for FLAG_EVT_BEAM_NON_EXPOSURE_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 3. Take an image for evidence | FLAG_EVT_BEAM_NON_EXPOSURE_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |
| SRS-20.8 | The SS shall detect and report a non-zero tube voltage event during non-exposure as a critical fault | 1. Set the acceptable tube voltage to be 1V with: example.com/}.local:8081//remote_api?command=p&reg=11&pid=1&op=1&payload=%2200000001%22 2. Wait for FLAG_EVT_HV_NON_EXPOSURE_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 3. Take an image for evidence | FLAG_EVT_HV_NON_EXPOSURE_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window |  |  |

### Table 12
| Board | Signal | Nominal Value | Fault Threshold | Modified Threshold | ICD Payload |
| --- | --- | --- | --- | --- | --- |
| CAS | CAS_BMS_TS2 | 22 C | > 60 C | > 0 C | 0A000001 |
|  |  |  | < 0 C | < 80 C | 0A401F00 |
| CAS | CAS_BMS_TS3 | 22 C | > 60 C | > 0 C | 0B000001 |
|  |  |  | < 0 C | < 80 C | 0B401F00 |
| CAS | CAS_BMS_TSINT | 22 C | > 80 C | > 0 C | 0C000001 |
|  |  |  | < 0 C | < 80 C | 0C401F00 |

### Table 13
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: E1 Emitter, C1 Cassette, M50133 Galaxy Tablet  S8+ Rev. A, APP MedAI Device App Emitter and cassette are placed in T-063 MX1 Positioning Jig. Emitter is fitted with a modified side shell that makes the Monoblock power light visible. Precondition: All device components are powered on and in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |  |
| Test Case: System behavior when in safe state |  |  |  |  |  |
| SRS-19.2 | When in safe state, the SS shall set all MI (Mode Indicator) LEDs on the cassette to magenta | To force the system into safe state, use the following steps to stop the emitter-frontend service: 1. SSH into the cassette via ssh imager@<cassette hostname> 2. Using the cassette session, ssh into the emitter via imager@<emitter hostname> 3. Navigate to /opt/medai/bin 4. End the emitter frontend service with the systemctl --user stop emitter-frontend 5. Run python3.8 -m mx1.services status to get a list of running services. Ensure emitter-frontend is not listed. | All cassette MI LEDs set to magenta | See Appendix 91 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.4 | Upon entering safe state, the SS shall set the IR LED brightness to 0 |  | A smartphone without an IR filter on its camera can see the IR LEDs are off | See Appendix 92 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.3 | Upon entering safe state, the SS shall kill power to the detector |  | Ping of detector IP address (192.168.8.8) is unsuccessful | See Appendix 93 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.5 | When in safe state, the SS shall set all MI LEDs on the emitter to magenta |  | All emitter MI LEDs set to magenta | See Appendix 91 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.7 | Upon entering safe state, the SS shall disable lasers |  | Lasers turn off | See Appendix 94 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.6 | Upon entering safe state, the SS shall disable power to the Monoblock |  | Monoblock power light is off | See Appendix 95 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.8 | When in safe state, the SS shall display an error message via notification on the MedAI Device App |  | Error message displayed on MedAI Device App | See Appendix 96 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-19.9 | When in safe state, the SS shall indicate via notification on the cassette display |  | Cassette Display shows “Error” | See Appendix 97 Verified by: S. Park Date: 05/24/24 | PASS |

### Table 14
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: E1 Emitter, C1 cassette Precondition: All device components are powered on and in radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |  |
| Test Case: Emitter Main Temperature Fault Testing |  |  |  |  |  |
| SRS-20.1 | The SS shall detect and report any out-of-bounds emitter temperature event as a critical fault | 1. Ensure that ethernet is plugged in to the emitter via USB-C 2. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081/remote_api?command=p&reg=D1&pid=3&op=1&payload=%22{payload}%22 3. Ensure that the Emitter device enters magenta safe state 4. In the cassette-orchestrator debug window, ensure that the Emitter Device has entered safety state and that the FLAG_EVT_TEMPERATURE_FAULT is displayed in the ememittermainFlags.  Take a screenshot for proof 5. In a web browser, send the ICD command: example.com/}.local:8081//remote_api?command=p&reg=D2&pid=3&op=0&payload=%22{payload}%22 6. Ensure that the bit corresponding to the bit position for the threshold you are testing is set to 1 (Table 2) 7. Reset all thresholds by sending the command: example.com/}.local:8081/remote_api?command=p&reg=D1&pid=3&op=1&payload=%2200000000%22 8. Repeat steps 1-7 for all Temperature thresholds in table 2 | FLAG_EVT_TEMPERATURE_FAULT is displayed in the ememittermainFlags row of the cassette-orchestrator debug window | Refer to result table 3: Appendix 5 -  Appendix 20 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  |  | Corresponding bit is set as 1 in the read response to register 0xD2 via ICD |  | PASS |
|  |  |  | Cassette and Emitter MI LEDs turn magenta upon detecting a critical fault | Refer to Appendix 91 Verified by: S. Park Date: 05/24/24 | PASS |
|  | The SS shall enter safe state upon detection of a critical fault |  |  |  |  |
| SRS-19.1 |  |  |  |  |  |
| Test Case: Monoblock Temperature Fault Testing |  |  |  |  |  |
| SRS-20.11 | The SS shall detect and report any out-of-bounds Monoblock temperature event as a critical fault | 1. Ensure that ethernet is plugged in to the emitter via USB-C 2. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081/remote_api?command=p&reg=D1&pid=1&op=1&payload=%22{payload}%22 3. Ensure that the Emitter device enters safe state 4. In the cassette-orchestrator debug window, ensure that the Emitter Device has entered safety state and that the FLAG_EVT_TEMPERATURE_FAULT is displayed in the monoblockFlags.  Take a screenshot for proof 5. In a web browser, send the ICD command: example.com/}.local:8081//remote_api?command=p&reg=D2&pid=1&op=0&payload=%22{payload}%22 6. Ensure that the bit corresponding to the bit position for the threshold you are testing is set to 1 (Table 1) 7. Reset all thresholds by sending the command: example.com/}.local:8081/remote_api?command=p&reg=D1&pid=1&op=1&payload=%2200000000%22 8. Repeat steps 1-7 for all Temperature thresholds in table 1 | FLAG_EVT_TEMPERATURE_FAULT is displayed in the ememittermainFlags row of the cassette-orchestrator debug window | Refer to result table 3: Appendix 1 - Appendix 4 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  |  | Bits 1 and 2 are set as 1 in the read response to register 0xD2 via rest_icd |  |  |
|  |  |  | Cassette and Emitter MI LEDs turn magenta upon detecting a critical fault | Refer to Appendix 91 Verified by: S. Park Date: 05/24/24 |  |
| SRS-19.1 | The SS shall enter safe state upon detection of a critical fault |  |  |  |  |
| Test Case: Cassette Temperature Fault Testing |  |  |  |  |  |
| SRS-20.14 | The SS shall detect and report any out-of-bounds cassette temperature event as a critical fault | 1. Ensure that ethernet is plugged in to the Cassette via USB-C 2. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081/remote_api?command=p&reg=D1&pid=5&op=1&payload=%22{payload}%22 3. Ensure that the Cassette device enters magenta safe state 4. In the cassette-orchestrator debug window, ensure that the Emitter Device has entered safety state and that the FLAG_EVT_TEMPERATURE_FAULT is displayed in the cscassettemainFlags.  Take a screenshot for proof 5. In a web browser, send the ICD command: example.com/}.local:8081//remote_api?command=p&reg=D2&pid=5&op=0&payload=%22{payload}%22 6. Ensure that the bit corresponding to the bit position for the threshold you are testing is set to 1 (Table 1) 7. Reset all thresholds by sending the command: example.com/}.local:8081/remote_api?command=p&reg=D1&pid=5&op=1&payload=%2200000000%22 8. Repeat steps 1-7 for all Temperature thresholds in table 1 | FLAG_EVT_TEMPERATURE_FAULT is displayed in the cassettemainFlags row of the cassette-orchestrator debug window | Refer to result table 3:Appendix 21 - Appendix 44 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  |  | Bits 0-12 are set as 1 in the read response to register 0xD2 via rest_icd |  |  |
| SRS-19.1 | The SS shall enter safe state upon detection of a critical fault |  |  |  |  |
|  |  |  | Cassette and Emitter MI LEDs turn magenta upon detecting a critical fault | Refer to Appendix 91 Verified by: S. Park Date: 05/24/24 |  |

### Table 15
| Board | Signal | Nominal Value | Fault Threshold | Modified Threshold | ICD Payload | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Monoblock | MB_HEATSINK | 22 C | > 80 C | > 0 C | 00000001 | Refer Appendix 1 |
|  |  |  | < 0 C | < 80 C | 00401F00 | Refer Appendix 2 |
|  | MB_SIDE | 22 C | > 80 C | > 0C | 01000001 | Refer Appendix 3 |
|  |  |  | < 0 C | < 80 C | 01401F00 | Refer Appendix 4 |
| Emitter Main | MB_HEAT_PIPE | 22 C | > 70 C | > 0 C | 00000001 | Refer Appendix 5 |
|  |  |  | < 0 C | < 80 C | 00401F00 | Refer Appendix 6 |
|  | EM_HANDLE | 22 C | > 48 C | > 0 C | 01000001 | Refer Appendix 7 |
|  |  |  | < 0 C | < 80 C | 01401F00 | Refer Appendix 8 |
|  | EM_PCB | 22 C | > 80 C | > 0 C | 02000001 | Refer Appendix 9 |
|  |  |  | < 0 C | < 80 C | 02401F00 | Refer Appendix 10 |
|  | EM_PMUX | 22 C | > 95 C | > 0 C | 03000001 | Refer Appendix 11 |
|  |  |  | < 0 C | < 95 C | 031C2500 | Refer Appendix 12 |
|  | EM_BMS_TS1 | 22 C | > 60 C | > 0 C | 04000001 | Refer Appendix 13 |
|  |  |  | < 0 C | < 80 C | 04401F00 | Refer Appendix 14 |
|  | EM_BMS_TS2 | 22 C | > 60 C | > 0 C | 05000001 | Refer Appendix 15 |
|  |  |  | < 0 C | < 80 C | 05401F00 | Refer Appendix 16 |
|  | EM_BMS_TS3 | 22 C | > 60 C | > 0 C | 06000001 | Refer Appendix 17 |
|  |  |  | < 0 C | < 80 C | 06401F00 | Refer Appendix 18 |
|  | EM_BMS_TSINT | 22 C | > 80 C | > 0 C | 07000001 | Refer Appendix 19 |
|  |  |  | < 0 C | < 80 C | 07401F00 | Refer Appendix 20 |
| Cassette Main | CAS_BAT_CON | 22 C | > 91 C | > 0 C | 00000001 | Refer Appendix 21 |
|  |  |  | < 0 C | < 80 C | 00401F00 | Refer Appendix 22 |
|  | CAS_OP_AMP | 22 C | > 80 C | > 0 C | 01000001 | Refer Appendix 23 |
|  |  |  | < 0 C | < 80 C | 01401F00 | Refer Appendix 24 |
|  | CAS_MCU | 22 C | > 80 C | > 0 C | 02000001 | Refer Appendix 25 |
|  |  |  | < 0 C | < 80 C | 02401F00 | Refer Appendix 26 |
|  | CAS_CHGR | 22 C | >95 C | > 0 C | 03000001 | Refer Appendix 27 |
|  |  |  | < 0 C | < 95 C | 031C2500 | Refer Appendix 28 |
|  | CAS_JTSN_PWR | 22 C | > 70 C | > 0 C | 04000001 | Refer Appendix 29 |
|  |  |  | < 0 C | < 80 C | 04401F00 | Refer Appendix 30 |
|  | CAS_JTSN | 22 C | > 80 C | > 0 C | 05000001 | Refer Appendix 31 |
|  |  |  | < 0 C | < 80 C | 05401F00 | Refer Appendix 32 |
|  | CAS_LTE | 22 C | > 80 C | > 0 C | 06000001 | Refer Appendix 33 |
|  |  |  | < 0 C | < 80 C | 06401F00 | Refer Appendix 34 |
|  | CAS_WIFI | 22 C | > 80 C | > 0 C | 07000001 | Refer Appendix 35 |
|  |  |  | < 0 C | < 80 C | 07401F00 | Refer Appendix 36 |
|  | CAS_BMS_TS1 | 22 C | > 60 C | > 0 C | 09000001 | Refer Appendix 37 |
|  |  |  | < 0 C | < 80 C | 09401F00 | Refer Appendix 38 |
|  | CAS_BMS_TS2 | 22 C | > 60 C | > 0 C | 0A000001 | Refer Appendix 39 |
|  |  |  | < 0 C | < 80 C | 0A401F00 | Refer Appendix 40 |
|  | CAS_BMS_TS3 | 22 C | > 60 C | > 0 C | 0B000001 | Refer Appendix 41 |
|  |  |  | < 0 C | < 80 C | 0B401F00 | Refer Appendix 42 |
|  | CAS_BMS_TSINT | 22 C | > 80 C | > 0 C | 0C000001 | Refer Appendix 43 |
|  |  |  | < 0 C | < 80 C | 0C401F00 | Refer Appendix 44 |

### Table 16
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: E1 Emitter, C1 cassette |  |  |  |  |  |
| Test Case: Emitter Voltage Faults |  |  |  |  |  |
| SRS-20.3 | The SS shall detect and report any out-of-bounds emitter voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the emittermainFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the emittermainFlags row | Refer to Appendix 98 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1). example.com/}.local:8081//remote_api?command=p&reg=E8&pid=3&op=1&payload=%22{payload}%22 2. Ensure that the Emitter device enters magenta safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the emittermainFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values: example.com/}.local:8081//remote_api?command=p&reg=E8&pid=3&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all emitter main PCB voltage thresholds specified in Table 1 Note: Voltage event flags clear upon system restart | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the emittermainFlags row for the corresponding threshold under test | Refer to result table 5: Appendix 61 - Appendix 70 Verified by: S. Park Date: 05/24/24 | PASS |
| Test Case: Cassette Voltage Faults |  |  |  |  |  |
| SRS-20.16 | The SS shall detect and report any out-of-bounds cassette voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the cscassetteFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the Flags row | Refer to Appendix 98 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081//remote_api?command=p&reg=E8&pid=5&op=1&payload=%22{payload}%22 2. Ensure that the Cassette device enters magenta safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the cscassettemainFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values: example.com/}.local:8081//remote_api?command=p&reg=E8&pid=5&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all cassette main PCB voltage thresholds specified in Table 1 Note: Voltage event flags clear upon system restart | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the cscassetteFlags row for the corresponding threshold under test | Refer to result table 5: Appendix 75 - Appendix 84 Verified by: S. Park Date: 05/24/24 |  |
| Test Case: Monoblock Voltage Faults |  |  |  |  |  |
| SRS-20.13 | The SS shall detect and report any out-of-bounds monoblock voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the monoblockFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the monoblockFlags row | Refer to result table 5: Appendix 45 - Appendix 60 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1) example.com/}.local:8081//remote_api?command=p&reg=E8&pid=1&op=1&payload=%22{payload}%22 2. Ensure that the Emitter device enters magenta safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the monoblockFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values example.com/}.local:8081//remote_api?command=p&reg=E8&pid=1&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all monoblock PCB voltage thresholds specified in Table 1 Note: Voltage event flags clear upon system restart | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the monoblockFlags row for the corresponding threshold under test |  |  |
| Test Case: Collimator Voltage Faults |  |  |  |  |  |
| SRS-20.5 | The SS shall detect and report any out-of-bounds Collimator voltage events as a critical fault | 1. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the collimatorFlags row | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is NOT present in the collimatorFlags row | Refer to Appendix 98 Verified by: S. Park Date: 05/24/24 | PASS |
|  |  | 1. In a web browser, send the ICD payload that corresponds to the threshold being changed (Table 1): example.com/}.local:8081//remote_api?command=p&reg=E8&pid=4&op=1&payload=%22{payload}%22 2. Ensure that the Emitter device enters safe state 3. Using the cassette debug window, confirm FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the collimatorFlags row. Take an image of the debug window for evidence 4. In a web browser, send the ICD payload to reset the threshold to normal operational values: example.com/}.local:8081//remote_api?command=p&reg=E8&pid=4&op=1&payload=%2200000000%22 5. Ensure that the device exits safe state 6. Repeat steps 1 through 5 for all collimator PCB voltage thresholds specified in Table 1 | FLAG_EVT_VOLTAGE_REGULATOR_FAULT is present in the collimatorFlags row for the corresponding threshold under test | Refer to result table 5: Appendix 71 - Appendix 74 Verified by: S. Park Date: 05/24/24 | PASS |

### Table 17
| Board | Signal | Nominal Value | Fault Threshold | Modified Threshold | ICD Payload | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Monoblock | VBAT_MON | 24 V | > 40 V | > 18 V | 08080701 | Refer Appendix 45 |
|  |  |  | < 18 V | < 40 V | 08A00F00 | Refer Appendix 46 |
|  | XRAY_PWS | 81 V | > 86 V | > 9 V | 04840301 | Refer Appendix 47 |
|  |  |  | < 9 V | < 86 V | 04982100 | Refer Appendix 48 |
|  | 5V0 | 5 V | > 5.3 V | > 4.7 V | 00D60101 | Refer Appendix 49 |
|  |  |  | < 4.7 V | < 5.3 V | 00120200 | Refer Appendix 50 |
|  | 3V3 | 3.3 V | > 3.5 V | > 3 V | 012C0101 | Refer Appendix 51 |
|  |  |  | < 3 V | < 3.5 V | 015E0100 | Refer Appendix 52 |
|  | 15V0_FIL | 15 V | > 15.5 V | > 14.5 V | 02AA0501 | Refer Appendix 53 |
|  |  |  | < 14.5 V | < 15.5 V | 020E0600 | Refer Appendix 54 |
|  | M15V0 | 15 V | > 15.5 V | > 14.5 V | 06AA0501 | Refer Appendix 55 |
|  |  |  | < 14.5 V | < 15.5 V | 060E0600 | Refer Appendix 56 |
|  | P15V0 | 15 V | > 15.5 V | > 14.5 V | 05AA0501 | Refer Appendix 57 |
|  |  |  | < 14.5 V | < 15.5 V | 050E0600 | Refer Appendix 58 |
|  | P3V3_ANA | 3.3 V | > 3.5 V | > 3.1 V | 07360101 | Refer Appendix 59 |
|  |  |  | < 3.1 V | < 3.3 V | 074A0100 | Refer Appendix 60 |
| Emitter Main | BAT_V_MON | 24 V | > 39 V | > 18 V | 00080701 | Refer Appendix 61 |
|  |  |  | < 18 V | < 39 V | 003C0F00 | Refer Appendix 62 |
|  | DISPLAY_V_MON | 24 V | > 39 V | > 18 V | 02080701 | Refer Appendix 63 |
|  |  |  | < 18 V | < 39 V | 023C0F00 | Refer Appendix 64 |
|  | COL_V_MON | 24 V | > 39 V | > 18 V | 03080701 | Refer Appendix 65 |
|  |  |  | < 18 V | < 39 V | 033C0F00 | Refer Appendix 66 |
|  | JTSN_5V0 | 5 V | > 5.25 V | > 4.8 V | 04E00101 | Refer Appendix 67 |
|  |  |  | < 4.8 V | < 5.25 V | 040D0200 | Refer Appendix 68 |
|  | CONTROL_5V0 | 5 V | > 5.5 V | > 4.5 V | 06C20101 | Refer Appendix 69 |
|  |  |  | < 4.5 V | < 5.5 V | 06260200 | Refer Appendix 70 |
| Collimator | COL_V_MON | 28 V | > 29 V | > 19 V | 096C0701 | Refer Appendix 71 |
|  |  |  | < 19 V | < 29 V | 09540B00 | Refer Appendix 72 |
|  | 5V0 | 5 V | > 5.5 V | > 4.5 V | 08C20101 | Refer Appendix 73 |
|  |  |  | < 4.5 V | < 5.5 V | 08260200 | Refer Appendix 74 |
| Cassette Main | BAT_V_MON | 14 V | > 21 V | > 9.8 V | 01D40301 | Refer Appendix 75 |
|  |  |  | < 9.8 V | < 21 V | 01340800 | Refer Appendix 76 |
|  | CONTROL_5V0 | 5 V | > 5.15 V | > 4.78 V | 03DE0101 | Refer Appendix 77 |
|  |  |  | < 4.78 V | < 5.15 V | 03030200 | Refer Appendix 78 |
|  | JET_3V3 | 3.3 V | > 3.4 V | > 3.15 V | 063B0101 | Refer Appendix 79 |
|  |  |  | < 3.15 V | < 3.4 V | 06540100 | Refer Appendix 80 |
|  | DET_22V0 | 22 V | > 23 V | > 19 V | 076C0701 | Refer Appendix 81 |
|  |  |  | < 19 V | < 23 V | 07FC0800 | Refer Appendix 82 |
|  | JET_5V0 | 5 V | > 5.15 V | > 4.8 V | 02E00101 | Refer Appendix 83 |
|  |  |  | < 4.8 V | < 5.15 V | 02030200 | Refer Appendix 84 |

### Table 18
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter, C1 Cassette |  |  |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode. |  |  |  |  |
| Test Case: Out-of-Bounds Beam Current, Tube Voltage, and Exposure Time Faults |  |  |  |  |  |
| SRS-20.17 | The SS shall detect and report an out-of-bounds filament current event as a critical fault | 1. Set the acceptable beam current threshold to be 5 mA with: example.com/}.local:8081//remote_api?command=p&reg=44&pid=1&op=1&payload=%220005%22 3. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 4. Wait for FLAG_EVT_BEAM_CURR_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 5. Take an image for evidence | FLAG_EVT_BEAM_CURR_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window | See Appendix 86 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-20.7 | The SS shall detect and report an out-of-bounds tube voltage event during exposure as a critical fault | 1. Set the acceptable tube voltage threshold to be 60 kV with: example.com/}.local:8081//remote_api?command=p&reg=11&pid=1&op=1&payload=%2260EA0000%22 2. Force the fault by pulling the trigger with http://<ip address>:8081/remote_api?command=trigger_press&press_time=100 3. Wait for FLAG_EVT_HV_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 4. Take an image for evidence | FLAG_EVT_HV_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window | See Appendix 87 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-20.12 | The SS shall detect and report an out-of-bounds exposure time event as a critical fault | 1. Unlock exposure time limit changing capabilities with: example.com/}.local:8081//remote_api?command=p&reg=04&pid=1&op=1&payload=%2201%22 2. Set the exposure time window limit to be 255 ms with: example.com/}.local:8081//remote_api?command=p&reg=20&pid=1&op=1&payload=%22FF%22 3. Set the exposure time window to be 254 ms with: example.com/}.local:8081//remote_api?command=p&reg=13&pid=1&op=1&payload=%22FE%22 4.  Wait for FLAG_EVT_HV_ON_TIME_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 5. Take an image for evidence Note: Restart the system to clear the out-of-bounds exposure time fault prior to running the following tests | FLAG_EVT_HV_ON_TIME_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window | See Appendix 88 Verified by: S. Park Date: 05/24/24 | PASS |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode. Emitter Sideshell off in order to access Monoblock test points |  |  |  |  |
| Test Case: Non-Zero Beam Current and Tube Voltage Faults |  |  |  |  |  |
| SRS-20.10 | The SS shall detect and report a non-zero beam current event during non-exposure as a critical fault | 1. Set the acceptable beam current threshold to be 1 mA with:example.com/}.local:8081//remote_api?command=p&reg=44&pid=1&op=1&payload=%220001%22 2. Wait for FLAG_EVT_BEAM_NON_EXPOSURE_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 3. Take an image for evidence | FLAG_EVT_BEAM_NON_EXPOSURE_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window | See Appendix 89 Verified by: S. Park Date: 05/24/24 | PASS |
| SRS-20.8 | The SS shall detect and report a non-zero tube voltage event during non-exposure as a critical fault | 1. Set the acceptable tube voltage to be 1V with: example.com/}.local:8081//remote_api?command=p&reg=11&pid=1&op=1&payload=%2200000001%22 2. Wait for FLAG_EVT_HV_NON_EXPOSURE_FAULT to appear in the emmonoblockFlags row of the cassette-orchestrator debug window 3. Take an image for evidence | FLAG_EVT_HV_NON_EXPOSURE_FAULT is displayed in the emmonoblockFlags row of the cassette-orchestrator debug window | See Appendix 90 Verified by: S. Park Date: 05/24/24 | PASS |

### Table 19
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter, C1 Cassette Emitter and cassette are placed in T-063 MX1 Positioning Jig. Ensure emitter is positioned within the acceptable SID range. |  |  |  |  |
| Precondition: | The emitter and cassette are powered on, paired, and in radiographic mode, front face is removed |  |  |  |  |
| Test Case: Collimator Timeout |  |  |  |  |  |
| SRS-20.4 | The SS shall detect and report a collimator timeout event as a critical fault | 1. With the front face removed, unplug both mother cables from the collimator. 2. In a browser window, send the following command: example.com/}.local:8081//remote_api?command=p&reg=81&pid=4&op=1&payload=%22D0403500A086010000%22 3. Ensure that the collimator debug LED is blinking red 4. Ensure that the FLAG_COL_TIMEOUT_FAULT appears in the collimatorFlags row of the cassette-orchestrator debug window | FLAG_COL_TIMEOUT_FAULT appears in the collimatorFlags row of the cassette-orchestrator debug window | Refer Appendix 85 Verified by: S. Park Date: 05/24/24 | PASS |

### Table 20
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-449 |  |  |
