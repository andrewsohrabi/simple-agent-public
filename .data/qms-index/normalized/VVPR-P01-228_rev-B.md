# VVPR-P01-228 Rev B: MX1 Software System X-Ray Emission and Trigger Timing v3.3.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-228
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.3.0
- Source filename: VVPR-P01-228 - MX1 Software System X-Ray Emission and Trigger Timing v3.3.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-228 - MX1 Software System X-Ray Emission and Trigger Timing v3.3.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the following requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification
SRS-16.14 -  In radioscopic mode, the SS shall terminate acquisition within 100 ms of foot pedal trigger release if the loading time is longer than 500 ms
SRS-16.15 - In radioscopic mode, the SS shall terminate acquisition within 100 ms of trigger release if the loading time is longer than 500 ms
SRS-16.16 - In radioscopic mode, the SS shall terminate acquisition within 100 ms of foot pedal trigger release if the loading time is shorter than 500 ms
SRS-16.17 - In radioscopic mode, the SS shall terminate acquisition within 100 ms of trigger release if the loading time is shorter than 500 ms
SRS-16.27 - In serial radiography mode, the SS shall terminate acquisition within 100 ms of foot pedal trigger release if the loading time is longer than 500 ms
SRS-16.28 - In serial radiography mode, the SS shall terminate acquisition within 100 ms of trigger release if the loading time is longer than 500 ms
SRS-16.29 - In serial radiography mode, the SS shall terminate acquisition within 100 ms of foot pedal trigger release if the loading time is shorter than 500 ms
SRS-16.30 - In serial radiography mode, the SS shall terminate acquisition within 100 ms of trigger release if the loading time is longer than 500 ms
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification.
The scope of this study is limited to the verification of acceptable timing between the release of the trigger mechanism and the end  of X-Ray-Emission
The scope of this study is limited to the two user-facing methods of triggering X-ray emission on the MX1 Portable X-ray system:
Trigger Mechanism on the Emitter
Trigger Mechanism on the Foot Pedal
Faults related to beam current and timing during radiographic exposure will use the first method while all other faults utilize the second.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev G
IFU-MX1 - Instructions for Use, Rev G
ES-10019-SCH-REVA.3 (Monoblock LV PCB Schematic)
ES-10007-SCH-REVB (Footpedal PCB Schematic)
ES-10005-SCH-REVA.1 (Emitter Display PCB Schematic)
MATERIALS
MX1 Software System v3.3.0
MX1 System Rev. E Components:
E1 Emitter Rev. H
C1 Cassette Rev. I
F1 Foot Pedal Rev. B
Additional tools/equipment:
SALAE 16 Pro Logic Analyzer - EQP-257 or equivalent
RIGOL DG812 WaveForm Generator - EQP-077 or equivalent
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
BACKGROUND
Locations and Personnel Responsibilities
Verification - To be performed at MedAI, Inc office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI, Inc engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Test Setup Background
This test involves measuring the voltage signals on both the triggering mechanisms and the X-ray radiation emitting circuit to properly analyze the timing between trigger release and X-ray emission termination. Exact X-ray emission timing will be measured via a voltage probe on a test point in the X-Ray emission circuitry. Exact trigger timing will be measured via voltage probe on trigger/pedal testpoints. A waveform generator will be used for consistent trigger timings. In order to achieve this, wires were soldered to the following points of interest:
XRAY_ON - TP3A in ES-10019-SCH-REVA: This testpoint is set HIGH when radiation is actively being emitted and will be used to measure exposure time
Figure 1. XRAY_ON - TP3A as shown in Schematic
Figure 2. XRAY_ON - TP3A as shown in PCBA Diagram
Figure 3. Installation of wire on XRAY_ON TP3A of opened E1 Emitter
Figure 4. XRAY_ON TP3A signal wire on reassembled E1 Emitter
PEDAL_RIGHT_2 and GND - S2 Connector in ES10007-SCH-REVB: When PEDAL_RIGHT_2 is shorted to ground, the footpedal will emit a signal to trigger an X-ray.
Figure 5. PEDAL_RIGHT_2  and GND Test point as shown in Schematic
Figure 6. PEDAL_RIGHT_2 and GND Test point as shown in PCBA Diagram
Figure 7. Installation of wire onto PEDAL_RIGHT_2 and GND Test points
DISP_TRIGGER_2 on J4 Connector - When DISP_TRIGGER_2 is shorted to ground, the software system will be notified to trigger an X-ray
Figure 8. DISP_TRIGGER_2 as shown in PCB Diagram
Figure 9. Replacement J4 Connector with exposed leads
Figure 10. Installation of new J4 Connector with exposed leads
Figure 11. Exposed J4 leads after Emitter E1 reassembly
EXPERIMENTAL PROCEDURE
Attach function generator probes to exposed trigger leads on E1 Emitter (GND to BLACK and Signal to RED)
Attach Digital Logic Analyzer Signal 1 probe to RED on exposed E1 Emitter trigger leads
Attach Digital Logic Analyzer Signal 2 probe to XRAY_ON signal wire
Configure function generator to baseline at 3.3V and pulse LOW for 1.5 seconds
Set MX1 Device to “Fluoro” mode
Start Capture on Logic Analyzer
Trigger output signal to fire an X-ray
Measure X-Ray Emission Length from the beginning of the first X-ray pulse to the end of the last X-ray pulse and record in table 1.
Measure termination time between trigger release edge and X-ray emission termination edge and record in table 1.
Repeat steps 4-9 for all Fluoro and DDR Trigger Press Length values. Ensure to switch to DDR mode when testing DDR
Disconnect function generator probes from exposed E1 Emitter trigger leads and attach them to exposed Footpedal trigger leads (GND to BLACK and Signal to RED)
Repeat steps 4-10 for Table 2
Table 1. X-Ray Emission and E1 Trigger Timing Table
Note - Termination time is the time between trigger release and X-ray emission termination
Table 2. X-Ray Emission and Foot Pedal Trigger Timing Table
Note - Termination time is the time between trigger release and X-ray emission termination
Data Analysis
All of the verification tests in Tables 1 through 2 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all data presented in Tables 1 and 2 per the results in the “PASS/FAIL” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS - NONE
DEVICES, COMPONENTS, OR EQUIPMENT USED
MX1 Portable X-ray System Rev E
E1 Emitter SN: 1222
C1 Cassette SN: 1223
F1 Footpedal UUID: 4718671
RESULTS
Table 1. X-Ray Emission and E1 Trigger Timing Table
Note - Termination time is the time between trigger release and X-ray emission termination
Table 2. X-Ray Emission and Foot Pedal Trigger Timing Table
Note - Termination time is the time between trigger release and X-ray emission termination
DISCUSSION
This report has tested that the software system terminates X-Ray emission within an acceptable time window of both the E1 Emitter trigger and F1 Footpedal Trigger release.
The reader may observe negative termination times within results table 1 and table 2. A negative timing window occurs when the trigger release occurs after the last X-Ray pulse. An example of this scenario is shown below:
Figure 12. Depicts a termination time of -18.53 ms.
Blue = Trigger Time, Green = X-Ray Emission Time, Red = Termination Time
CONCLUSION
Overall Result:
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 through Appendix 24  - Verification Evidence as Specified in Results Table 1 - 2.
REPORT APPROVAL
Digital Key:
example.com/
APPENDICES
Legend:
Blue = Trigger Time, Green = X-Ray Emission Time, Red = Termination Time
Appendix 1 - E1 Emitter Trigger, 1.5 Seconds, Fluoro
Appendix 2 - E1 Emitter Trigger, 2 Seconds, Fluoro
Appendix 3 - E1 Emitter Trigger, 4 Seconds, Fluoro
Appendix 4 - E1 Emitter Trigger, 6 Seconds, Fluoro
Appendix 5 - E1 Emitter Trigger, 8 Seconds, Fluoro
Appendix 6 - E1 Emitter Trigger, 10 Seconds, Fluoro
Appendix 7 - E1 Emitter Trigger, 1.5 Seconds, DDR
Appendix 8 - E1 Emitter Trigger, 2 Seconds, DDR
Appendix 9 - E1 Emitter Trigger, 4 Seconds, DDR
Appendix 10 - E1 Emitter Trigger, 6 Seconds, DDR
Appendix 11 - E1 Emitter Trigger, 8 Seconds, DDR
Appendix 12 - E1 Emitter Trigger, 10 Seconds, DDR
Appendix 13 - F1 Footpedal Trigger, 1.5 Seconds, Fluoro
Appendix 14 - F1 Footpedal Trigger, 2 Seconds, Fluoro
Appendix 15 - F1 Footpedal Trigger, 4 Seconds, Fluoro
Appendix 16 - F1 Footpedal Trigger, 6 Seconds, Fluoro
Appendix 17 - F1 Footpedal Trigger, 8 Seconds, Fluoro
Appendix 18 - F1 Footpedal Trigger, 10 Seconds, Fluoro
Appendix 19 - F1 Footpedal Trigger, 1.5 Seconds, DDR
Appendix 20 - F1 Footpedal Trigger, 2 Seconds, DDR
Appendix 21 - F1 Footpedal Trigger, 4 Seconds, DDR
Appendix 22 - F1 Footpedal Trigger, 6 Seconds, DDR
Appendix 23 - F1 Footpedal Trigger, 8 Seconds, DDR
Appendix 24 - F1 Footpedal Trigger, 10 Seconds, DDR

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Mode | Trigger Press Length (s) | X-Ray Emission Length (ms) | Termination Time (ms) | Acceptance Criteria (ms) | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Fluoro | 1.5 |  |  | < 100 |  |
|  | 2 |  |  | < 100 |  |
|  | 4 |  |  | < 100 |  |
|  | 6 |  |  | < 100 |  |
|  | 8 |  |  | < 100 |  |
|  | 10 |  |  | < 100 |  |
| DDR | 1.5 |  |  | < 100 |  |
|  | 2 |  |  | < 100 |  |
|  | 4 |  |  | < 100 |  |
|  | 6 |  |  | < 100 |  |
|  | 8 |  |  | < 100 |  |
|  | 10 |  |  | < 100 |  |

### Table 3
| Mode | Trigger Press Length (s) | X-Ray Emission Length (ms) | Termination Time (ms) | Acceptance Criteria (ms) | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Fluoro | 1.5 |  |  | < 100 |  |
|  | 2 |  |  | < 100 |  |
|  | 4 |  |  | < 100 |  |
|  | 6 |  |  | < 100 |  |
|  | 8 |  |  | < 100 |  |
|  | 10 |  |  | < 100 |  |
| DDR | 1.5 |  |  | < 100 |  |
|  | 2 |  |  | < 100 |  |
|  | 4 |  |  | < 100 |  |
|  | 6 |  |  | < 100 |  |
|  | 8 |  |  | < 100 |  |
|  | 10 |  |  | < 100 |  |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 30 Oct 2024 | 24-627 |

### Table 5
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| RIGOL DG812 | EQP-077 | 28 Jun 2024 | 30 Jun 2025 |
| Saleae Logic Pro 16 | EQP-257 | N/A | N/A |

### Table 6
| Mode | Trigger Press Length (s) | X-Ray Emission Length (ms) | Termination Time (ms) | Acceptance Criteria (ms) | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Fluoro | 1.5 | 431.85 | -18.53 | < 100 | PASS |
|  | 2 | 819.42 | 31.42 | < 100 | PASS |
|  | 4 | 2788.96 | -123.48 | < 100 | PASS |
|  | 6 | 4950.30 | -21.43 | < 100 | PASS |
|  | 8 | 6910.57 | 10.48 | < 100 | PASS |
|  | 10 | 8880.62 | -29.54 | < 100 | PASS |
| DDR | 1.5 | 425.13 | 38.90 | < 100 | PASS |
|  | 2 | 825.45 | -74.69 | < 100 | PASS |
|  | 4 | 2790.64 | -114.80 | < 100 | PASS |
|  | 6 | 4755.11 | -107.70 | < 100 | PASS |
|  | 8 | 6911.07 | 1.17 | < 100 | PASS |
|  | 10 | 8680.85 | -123.80 | < 100 | PASS |

### Table 7
| Mode | Trigger Press Length (s) | X-Ray Emission Length (ms) | Termination Time (ms) | Acceptance Criteria (ms) | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Fluoro | 1.5 | 431.95 | 26.98 | < 100 | PASS |
|  | 2 | 824.67 | -72.22 | < 100 | PASS |
|  | 4 | 2984.67 | 25.44 | < 100 | PASS |
|  | 6 | 4750.11 | -16.50 | < 100 | PASS |
|  | 8 | 6327.10 | -109.16 | < 100 | PASS |
|  | 10 | 8271.06 | 43.63 | < 100 | PASS |
| DDR | 1.5 | 432.36 | 46.10 | < 100 | PASS |
|  | 2 | 824.50 | 33.73 | < 100 | PASS |
|  | 4 | 2593.79 | -98.25 | < 100 | PASS |
|  | 6 | 4949.64 | 61.19 | < 100 | PASS |
|  | 8 | 6711.98 | 35.81 | < 100 | PASS |
|  | 10 | 8878.03 | 31.19 | < 100 | PASS |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-602 |  |
