# VVPR-P01-182 Rev B: MX1 Software System Voltage and Temperature Monitoring Accuracy v3.1.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-182
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.1.0
- Source filename: VVPR-P01-182 - MX1 Software System Voltage and Temperature Monitoring Accuracy v3.1.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-182 - MX1 Software System Voltage and Temperature Monitoring Accuracy v3.1.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification, specifically in regards to the accuracy in monitoring of:
Temperature readings for the emitter main PCB, Monoblock LV PCBA, Collimator PCBA, and Cassette Main PCBA
Voltage readings for the Emitter Main PCBA, Monoblock LV PCBA, Collimator PCBA, and Cassette Main PCBA
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI, Inc for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification.
The scope of this study is limited to the verification of the accuracy of temperature and voltage readings of the MX1 Software System
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev B
IFU-MX1 - Instructions for Use, Rev D
Emitter Main PCBA Layout and Schematic
Cassette Main PCBA Layout and Schematic
Collimator PCBA Layout and Schematic
Monoblock LV PCBA Layout and Schematic
MATERIALS
For Test Configuration 1 - Emitter:
ES-10003 Rev B.2 - Emitter Main PCBA
ES-10008 Rev C.1 - Collimator PCBA
ES-10015 Rev C.1 - PMUX PCBA
ES-10019 Rev A.3 - Monoblock LV PCBA
M50101 Rev A - Jetson Xavier
M50095 Rev A - NVME Module
M10372 Rev A - FFC, 0.5mm pitch, 28 CKT, 103mm, Shielded
M50045 Rev A - FFC, 0.5mm pitch, 14 Ckt, 76mm
MS-10010 BOM Rev G - Emitter Battery Pack
Fluke Multimeter EQP-093 (or equivalent)
Keysight DAQ  EQP-165 (or equivalent)
AE Temp/Humidity Test Chamber - EQP-199
S10088 - MX1 Monitoring Script- 1.0.0
Type T Beaded Thermocouple 30 Gage PFA Insulated 80 Inch Long Wire Leads with Stripped Ends (Evosensors, MPN t1x-wbwx-30g-ex-0-25-pfxx-80-stwl)
For Test Configuration 2 - Cassette:
ES-10004 Rev B.2 - Cassette Main PCBA
ES-10036 Rev B.1 - Tracking PCBA
M50101 Rev A - Jetson Xavier
M50095 Rev A - NVME Module
MS-10083 BOM Rev G - Cassette Battery Pack
Fluke Multimeter EQP-093 (or equivalent)
Keysight DAQ  EQP-165 (or equivalent)
AE Temp/Humidity Test Chamber - EQP-199
S10088 MX1 Monitoring Script - 1.0.0
Type T Beaded Thermocouple 30 Gage PFA Insulated 80 Inch Long Wire Leads with Stripped Ends (Evosensors, MPN t1x-wbwx-30g-ex-0-25-pfxx-80-stwl)
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI Medica, Inc. office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Test Setup Background
This test involves the validating the accuracy of the MX1 Software System’s voltage and temperature measurements compared to external measurements
For voltage monitoring, each test involves the comparison between an external measurement and an internal MCU measurement of the target voltage line. The external measurement will be acquired from a EQP-093 and the on-board ADC measurement will be read from the MX1 Software System via an ICD register call . The resulting readings from this test will be filled into the “External Multimeter Reading”, “Internal ADC Reading, and “Error” columns in table 1
For temperature monitoring, each test involves the comparison between an external measurement and an internal MCU measurement of a specific temperature sensor. The external measurement will be acquired from a temperature probe at the appropriate location with EQP-165 and the internal sensor/ADC measurement will be read from the MX1 Software System via an ICD register call. “The resulting readings from this test will be filled into the “External DAQ Temp”, “Internal ADC Temp”, and “Error“ columns in table 2.
A “PASS” criteria for all tests requires the error to be <10%
All “Error” calculations will be derived using the following formula:
Table 1. Voltage Signals to be Measured
Table 2. Temperatures to be Measured
EXPERIMENTAL PROCEDURE
Follow the steps outlined in Tables 3 through 5 below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 3. Temperature Fault Testing - Requirements, Verification Steps, and Expected Results
Table 4. Voltage Monitoring Faults Testing- Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests in Tables 3 and 4 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 3 and 4  which equates to <10% accuracy in the “Error” column in tables 1 and 2
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
An SRS was accidentally omitted from the test procedure that is meant to test it:
SRS-20.6 - The SS shall detect and report an out-of-bounds power supply (PWS) voltage event as a critical fault.
This requirement is tested and passed without any procedural changes. The results can be found in result table 1, Monoblock Voltages, X-RAY_PWS
DEVICES, COMPONENTS, OR EQUIPMENT USED
Test Configuration 1
ES-10003 Rev B.2 - Emitter Main PCBA - SN-0008 - LOT-10130 (SW Version v3.1.0-beta)
ES-10008 Rev C.1 - Collimator PCBA - SN-0006 - LOT-P022614 (SW Version v3.1.0-beta)
ES-10015 Rev C.1 - PMUX PCBA - SN-0013 - LOT-10106 (SW Version v3.1.0-beta)
ES-10019 Rev A.3 - Monoblock LV PCBA - SN-0006 - LOT-10067 (SW Version v3.1.0-beta)
M50101 Rev A - Jetson Xavier (SW Version v3.1.0-beta)
M50095 Rev A - NVME Module (SW Version v3.1.0-beta)
M10372 Rev A - FFC, 0.5mm pitch, 28 CKT, 103mm, Shielded
M50045 Rev A - FFC, 0.5mm pitch, 14 Ckt, 76mm
MS-10010 BOM Rev G - Emitter Battery Pack
Fluke Multimeter EQP-093 (or equivalent)
Keysight DAQ  EQP-165 (or equivalent)
AE Temp/Humidity Test Chamber - EQP-199
S10088 - MX1 Monitoring Script- 1.0.0
Type T Beaded Thermocouple 30 Gage PFA Insulated 80 Inch Long Wire Leads with Stripped Ends (Evosensors, MPN t1x-wbwx-30g-ex-0-25-pfxx-80-stwl)
Image 1 - Test Configuration 1
Test Configuration 2
ES-10004 Rev B.2 - Cassette Main PCBA - SN-0003 - LOT-10167
ES-10036 Rev B.1 - Tracking PCBA - SN-0006 - LOT-10163
M50101 Rev A - Jetson Xavier - SN
M50095 Rev A - NVME Module - SN
MS-10083 BOM Rev G - Cassette Battery Pack
Fluke Multimeter EQP-093 (or equivalent)
Keysight DAQ  EQP-165 (or equivalent)
AE Temp/Humidity Test Chamber - EQP-199
S10088 MX1 Monitoring Script - 1.0.0
Type T Beaded Thermocouple 30 Gage PFA Insulated 80 Inch Long Wire Leads with Stripped Ends (Evosensors, MPN t1x-wbwx-30g-ex-0-25-pfxx-80-stwl)
Image 2 - Test Configuration 2
RESULTS
Result Table 1. Voltage Signals to be Measured
Result Table 2. Temperatures to be Measured
Result Table 3. Temperature Fault Testing - Requirements, Verification Steps, and Expected Results
Table 4. Voltage Monitoring Faults Testing- Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
All voltage and temperature monitoring tests passed with accuracies within 10%
LIST OF APPENDICES
No appendices
REPORT APPROVAL
Digital Key: example.com/

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Board | Signal | External Probe Point | ICD Payload | External Multimeter Voltage | Internal ADC Voltage | Error |
| --- | --- | --- | --- | --- | --- | --- |
| Monoblock | VBAT_MON | R170. VBAT side | 08 |  |  |  |
|  | XRAY_PWS | R169, V_XRAY side | 04 |  |  |  |
|  | 5V0 | R102, 5V side | 00 |  |  |  |
|  | 3V3 | R109, 3V3 side | 01 |  |  |  |
|  | 15V0_FIL | R121, 15V FIL side | 02 |  |  |  |
|  | M15V0 | R100, M15V0 side | 06 |  |  |  |
|  | P15V0 | R112, P15V0 side | 05 |  |  |  |
|  | P3V3_ANA | R118, P3V3_ANA side | 07 |  |  |  |
| Emitter Main | BAT_V_MON | R34, HW_SW_MON side | 00 |  |  |  |
|  | DISPLAY_V_MON | R44, DISP_28V side | 02 |  |  |  |
|  | COL_V_MON | R52, COL_28V side | 03 |  |  |  |
|  | 5V0 | R31, JET_5V0 side | 04 |  |  |  |
|  | LED_5V0 | R41, LED_5V0 side | 06 |  |  |  |
| Collimator | VMOTOR_24V | TP17 | 09 |  |  |  |
|  | 5V0 | TP20 | 08 |  |  |  |
| Cassette Main | BAT_V_MON | R50, CM_PWR side | 01 |  |  |  |
|  | CONTROL_5V0 | R37, MCU_5V0 side | 02 |  |  |  |
|  | JET_3V3 | R53, JET_3V3 side | 06 |  |  |  |
|  | DET_22V0 | R44, DETECTOR_22V side | 07 |  |  |  |
|  | JET_5V0 | R41, JET_5V0 | 03 |  |  |  |

### Table 3
| Board | Signal | Probe Point | External DAQ Temp (Temp 1) | Internal ADC Temp (Temp 1) | Error | External DAQ Temp (Temp 2) | Internal ADC Temp (Temp 2) | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Monoblock | MB_HEATSINK | J1 Thermistor |  |  |  |  |  |  |
|  | MB_SIDE | J10 Thermistor |  |  |  |  |  |  |
| Emitter Main | MB_HEAT_PIPE | J4 Thermistor |  |  |  |  |  |  |
|  | EM_HANDLE | J5 Thermistor |  |  |  |  |  |  |
|  | EM_PCB | U6 |  |  |  |  |  |  |
|  | EM_PMUX | U12 (PMUX) |  |  |  |  |  |  |
|  | EM_BMS_TS1 | RT2 |  |  |  |  |  |  |
|  | EM_BMS_TS2 | RT3 |  |  |  |  |  |  |
|  | EM_BMS_TS3 | RT4 |  |  |  |  |  |  |
| Cassette Main | CAS_BAT_CON | RT1 |  |  |  |  |  |  |
|  | CAS_OP_AMP | RT2 |  |  |  |  |  |  |
|  | CAS_MCU | U6 |  |  |  |  |  |  |
|  | CAS_CHGR | RT3 |  |  |  |  |  |  |
|  | CAS_JTSN_PWR | RT4 |  |  |  |  |  |  |
|  | CAS_JTSN | RT5 |  |  |  |  |  |  |
|  | CAS_LTE | RT6 |  |  |  |  |  |  |
|  | CAS_WIFI | RT7 |  |  |  |  |  |  |
|  | CAS_BMS_T1 | RT2(bms) |  |  |  |  |  |  |
|  | CAS_BMS_T2 | RT3(bms) |  |  |  |  |  |  |
|  | CAS_BMS_TS3 | RT4(bms) |  |  |  |  |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Test Configuration 1 Assembled as Shown in Figure 1. Ensure DAQ has a USB where data can be logged Place the entire test-setup into the temperature and humidity chamber. Precondition: Thermocouples attached to all temperature signal source |  |  |  |  |  |
| Test Case: Monoblock LV and Emitter Main PCB Temperature Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.1 | The SS shall detect and report any out-of-bounds emitter temperature event as a critical fault | 1. Ensure that the Emitter main PCBA has an ethernet connection via USB-C port 2. Power on the DAQ device with thermocouples 3. Power on Emitter Main and SSH into the Emitter Main Jetson in a terminal window with the following command: ssh imager@emitter-{serial-number} 4. Set Temperature and Humidity chamber to 30 degrees C and wait 15 minutes for the chamber to acclimate 5. Start data logging on the DAQ for all channels 6. In the terminal window where you are ssh’d into the emitter jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device emitter 7. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 8. Stop data logging on the DAQ for all channels 9. Set Temperature and Humidity chamber to 60 degrees C and wait 15 minutes for the chamber to acclimate 10. Start data logging on the DAQ for all channels 11. In the terminal window where you are ssh’d into the emitter jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device emitter 12. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 13. Stop data logging on the DAQ for all channels 14. Ensure that you have acquired data from both devices at both time points. Compile data into table 2. | The internal ADC measurements from the software system are within 10% of the externally measured DAQ values | Place all Data and Results into Table 2 |  |
| SRS-20.11 | The SS shall detect if the monoblock temperature is out of acceptable bounds and report as a critical fault |  |  |  |  |
| Test Setup: Test Configuration 2 Assembled as Shown in Figure 2. Flash modified firmware on all PCBA MCU’s (Cassette) Ensure DAQ has a USB where data can be logged Place the entire test-setup into the temperature and humidity chamber. Precondition: Thermocouples attached to all temperature signal source Modified firmware is flashed |  |  |  |  |  |
| Test Case: Cassette Main PCB Temperature Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.14 | The SS shall detect and report any out-of-bounds cassette temperature event as a critical fault | 1. Ensure that the Cassette main PCBA has an ethernet connection via USB-C port 2. Power on the DAQ device with thermocouples 3. Power on Cassette Main and SSH into the Cassette Main Jetson in a terminal window with the following command: ssh imager@Cassette-{serial-number} 4. Set Temperature and Humidity chamber to 30 degrees C and wait 15 minutes for the chamber to acclimate 5. Start data logging on the DAQ for all channels 6. In the terminal window where you are ssh’d into the Cassette jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device cassette 7. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 8. Stop data logging on the DAQ for all channels 9. Set Temperature and Humidity chamber to 60 degrees C and wait 15 minutes for the chamber to acclimate 10. Start data logging on the DAQ for all channels 11. In the terminal window where you are ssh’d into the emitter jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device cassette 12. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 13. Stop data logging on the DAQ for all channels 14. Ensure that you have acquired data from both devices at both time points. Compile data into table 2. | The internal ADC measurements from the software system are within 10% of the externally measured values | Place all Data and Results into Table 2 |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Test Configuration 1 Assembled as shown in figure 1 Modified firmware is flashed Start the rest server on Emitter Main Jetson |  |  |  |  |  |
| Test Case: Emitter Main PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.3 | The SS shall detect and report any out-of-bounds emitter voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Emitter Main PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=3&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all emitter main voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Place all Data and Results into Table 1 |  |
| Test Setup: Test Configuration 2 Assembled as shown in figure 2 Modified firmware is flashed Start the rest server on Cassette Main Jetson |  |  |  |  |  |
| Test Case: Cassette Main PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-22.16 | The SS shall detect and report any out-of-bounds cassette voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Cassette Main PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=5&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all Cassette main voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Place all Data and Results into Table 1 |  |
| Test Setup: Test Configuration 1 Assembled as shown in figure 1 Modified firmware is flashed Start the rest server on Emitter Main Jetson |  |  |  |  |  |
| Test Case: Monoblock LV PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-22.13 | The SS shall detect and report any out-of-bounds monoblock voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Monoblock LV PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=1&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all Monoblock LV voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Place all Data and Results into Table 1 |  |
| Test Setup: Test Configuration 1 Assembled as shown in figure 1 Modified firmware is flashed Start the rest server on Emitter Main Jetson |  |  |  |  |  |
| Test Case: Collimator PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.5 | The SS shall detect and report any out-of-bounds collimator voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Collimator PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=4&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all Collimator voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Place all Data and Results into Table 1 |  |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | SW Engineering Quality Engineering Regulatory Affairs | 17 May 2024 | 24-241 |

### Table 7
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 8
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Fluke 87V Industrial Multimeter | EQP-147 | 06/29/2023 | 06/29/2024 |
| Keysight DAQ970A | EQP-165 | 07/14/2023 | 07/14/2024 |

### Table 9
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Fluke 87V Industrial Multimeter | EQP-147 | 06/29/2023 | 06/29/2024 |
| Keysight DAQ970A | EQP-165 | 07/14/2023 | 07/14/2024 |

### Table 10
| Board | Signal | External Probe Point | ICD Payload | External Multimeter Voltage | Internal ADC Voltage | Error |
| --- | --- | --- | --- | --- | --- | --- |
| Monoblock | VBAT_MON | R170. VBAT side | 08 | 33.15 V | 33.27 V | 0.361 % |
|  | XRAY_PWS | R169, V_XRAY side | 04 | 9.89 V | 10.7 V | 8.190 % |
|  | 5V0 | R102, 5V side | 00 | 5.03 V | 5.02 V | 0.198 % |
|  | 3V3 | R109, 3V3 side | 01 | 3.31 V | 3.32 V | 0.302 % |
|  | 15V0_FIL | R121, 15V FIL side | 02 | 15.05 V | 15.15 V | 0.664 % |
|  | M15V0 | R100, M15V0 side | 06 | 15 V | 15.03 V | 0.200 % |
|  | P15V0 | R112, P15V0 side | 05 | 15.04 V | 15.22 V | 1.196 % |
|  | P3V3_ANA | R118, P3V3_ANA side | 07 | 3.29 V | 3.28 V | 0.303 % |
| Emitter Main | BAT_V_MON | R34, HW_SW_MON side | 00 | 33.05 V | 33.19 V | 0.423 % |
|  | DISPLAY_V_MON | R44, DISP_28V side | 02 | 33.09 V | 33.08 V | 0.030 % |
|  | COL_V_MON | R52, COL_28V side | 03 | 33.08 V | 33 V | 0.241 % |
|  | 5V0 | R31, JET_5V0 side | 04 | 5.08 V | 5.09 V | 0.196 % |
|  | LED_5V0 | R41, LED_5V0 side | 06 | 4.925 V | 5 V | 1.522 % |
| Collimator | VMOTOR_24V | TP17 | 09 | 24.18 V | 24.06 V | 0.496 % |
|  | 5V0 | TP20 | 08 | 5.015 V | 5 V | 0.279 % |
| Cassette Main | BAT_V_MON | R50, CM_PWR side | 01 | 20.05 V | 19.66 V | 1.945 % |
|  | CONTROL_5V0 | R37, MCU_5V0 side | 02 | 4.998 V | 4.950 V | 0.960 % |
|  | JET_3V3 | R53, JET_3V3 side | 06 | 3.327 V | 3.310 V | 0.510 % |
|  | DET_22V0 | R44, DETECTOR_22V side | 07 | 22.08 V | 22.10 V | 0.090 % |
|  | JET_5V0 | R41, JET_5V0 | 03 | 4.973 V | 4.980 V | 0.140 % |

### Table 11
| Board | Signal | Probe Point | External DAQ Temp (Temp 1) | Internal ADC Temp (Temp 1) | Error | External DAQ Temp (Temp 2) | Internal ADC Temp (Temp 2) | Error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Monoblock | MB_HEATSINK | J1 Thermistor | 25.95 C | 24.44 C | 5.818 % | 57.12 C | 59.69 C | 4.499 % |
|  | MB_SIDE | J10 Thermistor | 26.1 C | 24.44 C | 6.360 % | 57.6 C | 60.57 C | 5.156 % |
| Emitter Main | MB_HEAT_PIPE | J4 Thermistor | 25.53 C | 25.67 C | 0.548 % | 56.67 C | 56.7 C | 0.052 % |
|  | EM_HANDLE | J5 Thermistor | 25.9 C | 26.08 C | 0.694 % | 57 C | 56.7 C | 0.526 % |
|  | EM_PCB | U6 | 27.63 C | 28.44 C | 2.931 % | 58.95 C | 59.95 C | 1.696 % |
|  | EM_PMUX | U12 (PMUX) | 25.6 C | 26.53 C | 3.632 % | 57.23 C | 57.42 C | 0.331 % |
|  | EM_BMS_TS1 | RT2 | 25.16 C | 26.75 C | 6.319 % | 56.7 C | 57.35 C | 1.146 % |
|  | EM_BMS_TS2 | RT3 | 25.4 C | 26.95 C | 6.102 % | 55.77 C | 57.55 C | 3.191 % |
|  | EM_BMS_TS3 | RT4 | 25.67 C | 26.65 C | 3.817 % | 54.57 C | 56.75 C | 3.994 % |
| Cassette Main | CAS_BAT_CON | RT1 | 26.97 C | 26.81 C | 0.593 % | 59.95 C | 60.48 C | 0.884 % |
|  | CAS_OP_AMP | RT2 | 26.35 C | 26.5 C | 0.569 % | 58.77 C | 57.84 C | 1.582 % |
|  | CAS_MCU | U6 | 26.37 C | 26.49 C | 0.455 % | 58.85 C | 56.93 C | 3.262 % |
|  | CAS_CHGR | RT3 | 26.77 C | 26.45 C | 1.193 % | 59.75 C | 59.9 C | 0.251 % |
|  | CAS_JTSN_PWR | RT4 | 28.68 C | 30.27 C | 5.543 % | 61.46 C | 62.38 C | 1.496 % |
|  | CAS_JTSN | RT5 | 29.54 C | 29.41 C | 0.440% | 61.21 C | 60.03 C | 1.927 % |
|  | CAS_LTE | RT6 | 27.85 C | 27.93 C | 0.287 % | 61.17 C | 62.15 C | 1.602 % |
|  | CAS_WIFI | RT7 | 26.73 C | 26.63 C | 0.374 % | 59.67 C | 60.26 C | 0.988 % |
|  | CAS_BMS_T1 | RT2(bms) | 25.9 C | 26.35 C | 1.737 % | 56.36 C | 55.45 C | 1.64 % |
|  | CAS_BMS_T2 | RT3(bms) | 25.85 C | 26.25 C | 1.547 % | 56.68 C | 55.65 C | 1.817 % |
|  | CAS_BMS_TS3 | RT4(bms) | 25.95 C | 26.45 C | 1.907 % | 54.68 C | 55.65 C | 1.773  % |

### Table 12
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Test Configuration 1 Assembled as Shown in Figure 1. Ensure DAQ has a USB where data can be logged Place the entire test-setup into the temperature and humidity chamber. Precondition: Thermocouples attached to all temperature signal source |  |  |  |  |  |
| Test Case: Monoblock LV and Emitter Main PCB Temperature Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.1 | The SS shall detect and report any out-of-bounds emitter temperature event as a critical fault | 1. Ensure that the Emitter main PCBA has an ethernet connection via USB-C port 2. Power on the DAQ device with thermocouples 3. Power on Emitter Main and SSH into the Emitter Main Jetson in a terminal window with the following command: ssh imager@emitter-{serial-number} 4. Set Temperature and Humidity chamber to 30 degrees C and wait 15 minutes for the chamber to acclimate 5. Start data logging on the DAQ for all channels 6. In the terminal window where you are ssh’d into the emitter jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device emitter 7. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 8. Stop data logging on the DAQ for all channels 9. Set Temperature and Humidity chamber to 60 degrees C and wait 15 minutes for the chamber to acclimate 10. Start data logging on the DAQ for all channels 11. In the terminal window where you are ssh’d into the emitter jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device emitter 12. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 13. Stop data logging on the DAQ for all channels 14. Ensure that you have acquired data from both devices at both time points. Compile data into table 2. | The internal ADC measurements from the software system are within 10% of the externally measured DAQ values | Refer to Result Table 2 Verified by: S. Park Date: 05/23/24 | PASS |
| SRS-20.11 | The SS shall detect if the monoblock temperature is out of acceptable bounds and report as a critical fault |  |  |  | PASS |
|  |  |  |  |  | PASS |
| Test Setup: Test Configuration 2 Assembled as Shown in Figure 2. Flash modified firmware on all PCBA MCU’s (Cassette) Ensure DAQ has a USB where data can be logged Place the entire test-setup into the temperature and humidity chamber. Precondition: Thermocouples attached to all temperature signal source Modified firmware is flashed |  |  |  |  |  |
| Test Case: Cassette Main PCB Temperature Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.14 | The SS shall detect and report any out-of-bounds cassette temperature event as a critical fault | 1. Ensure that the Cassette main PCBA has an ethernet connection via USB-C port 2. Power on the DAQ device with thermocouples 3. Power on Cassette Main and SSH into the Cassette Main Jetson in a terminal window with the following command: ssh imager@Cassette-{serial-number} 4. Set Temperature and Humidity chamber to 30 degrees C and wait 15 minutes for the chamber to acclimate 5. Start data logging on the DAQ for all channels 6. In the terminal window where you are ssh’d into the Cassette jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device cassette 7. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 8. Stop data logging on the DAQ for all channels 9. Set Temperature and Humidity chamber to 60 degrees C and wait 15 minutes for the chamber to acclimate 10. Start data logging on the DAQ for all channels 11. In the terminal window where you are ssh’d into the emitter jetson, enter the command: python monitor_vvpr.py --serial_number {serial-number} --device cassette 12. Wait 30 seconds for the script to complete. Verify that a logfile has been created with the current time that contains data. 13. Stop data logging on the DAQ for all channels 14. Ensure that you have acquired data from both devices at both time points. Compile data into table 2. | The internal ADC measurements from the software system are within 10% of the externally measured values | Refer to Result Table 2 Verified by: S. Park Date: 05/23/24 | PASS |

### Table 13
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Test Configuration 1 Assembled as shown in figure 1 Modified firmware is flashed Start the rest server on Emitter Main Jetson |  |  |  |  |  |
| Test Case: Emitter Main PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.3 | The SS shall detect and report any out-of-bounds emitter voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Emitter Main PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=3&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all emitter main voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Refer to Result Table 1 Verified by: S. Park Date: 05/23/24 | PASS |
| Test Setup: Test Configuration 2 Assembled as shown in figure 2 Modified firmware is flashed Start the rest server on Cassette Main Jetson |  |  |  |  |  |
| Test Case: Cassette Main PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-22.16 | The SS shall detect and report any out-of-bounds cassette voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Cassette Main PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=5&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all Cassette main voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Refer to Result Table 1 Verified by: S. Park Date: 05/23/24 | PASS |
| Test Setup: Test Configuration 1 Assembled as shown in figure 1 Modified firmware is flashed Start the rest server on Emitter Main Jetson |  |  |  |  |  |
| Test Case: Monoblock LV PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-22.13 | The SS shall detect and report any out-of-bounds monoblock voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Monoblock LV PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=1&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all Monoblock LV voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Refer to Result Table 1 Verified by: S. Park Date: 05/23/24 | PASS |
| SRS-20.6 | The SS shall detect and report an out-of-bounds power supply (PWS) voltage event as a critical fault |  |  |  |  |
| Test Setup: Test Configuration 1 Assembled as shown in figure 1 Modified firmware is flashed Start the rest server on Emitter Main Jetson |  |  |  |  |  |
| Test Case: Collimator PCBA Voltage Monitoring Accuracy Test |  |  |  |  |  |
| SRS-20.5 | The SS shall detect and report any out-of-bounds collimator voltage events as a critical fault | 1. Select the signal under test from table 1 2. Using the Collimator PCB and Schematic, locate a probe test point for the signal and list it in table 1 3. Using EQP-093, probe the selected test point and record the measured voltage into table 14. In a web browser enter the following command for the selected signal: example.com/}.local:8090//remote_api?command=p&reg=E5&pid=4&op=0&payload=%22{ICD-PAYLOAD}%22 5. Record the measured value into table 1. 6. Repeat steps 1-5 for all Collimator voltage signals | The internal ADC measurements from the software system are within 10% of the externally measured values | Refer to Result Table 1 Verified by: S. Park Date: 05/23/24 | PASS |

### Table 14
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-449 |  |
