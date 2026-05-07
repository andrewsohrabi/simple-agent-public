# VVPR-P01-164 Rev B: Serial Radiography and Radioscopy Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-164
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-164 - Serial Radiography and Radioscopy Verification Protocol and Report_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-164 - Serial Radiography and Radioscopy Verification Protocol and Report_B-signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
This study verifies requirements for the serial radiography and radioscopy imaging modes of the MX1 Portable X-ray System. Requirements To be verified include:
X-ray pulse rate (as verified via pulse count at the maximum serial exposure duration) during serial radiography and radioscopy
X-ray pulse width (milliseconds) during serial radiography and radioscopy
Tube assembly potential (kV) during serial radiography and radioscopy
Current (mA) during serial radiography and radioscopy
OBJECTIVE AND SCOPE
The primary purpose of this study is to verify the requirements set by MedAI, Inc for the MX1 Portable X-ray System serial radiography and radioscopy.
Testing will be conducted at both mAs settings available for serial radiography and radioscopy (0.04 mAs and 0.08 mAs), and at the minimum, nominal, and maximum tube assembly voltage potentials (40 kV, 60 kV, 80 kV for serial radiography, 40 kV, 50 kV, 64 kV for radioscopy). The maximum serial radiography and radioscopy exposure time (20 seconds) will be used as a worst-case parameter in order to verify accuracy of outputs throughout the entire exposure time allowed by the system. The parameter ranges tested were selected as representative to encompass indicated serial radiography and radioscopy parameters.
REFERENCES
IFU-MX1 - Rev. D Instructions for Use
RSK-P01-010 Rev. A - Risk Assessment
IEC 60601-2-54:2022, Medical electrical equipment - Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy
IEC 60601-2-43:2022, Medical electrical equipment - Part 2-43: Particular requirements for the basic safety and essential performance of X-ray equipment for interventional procedures
MEMO-P01-662 Rev. A - Essential Performance Determination
MATERIALS
MX1 Portable X-ray System with SW v3.0.0 or higher (E1 emitter, C1 cassette, H1 wired charger)
RadCal AGDM+ Accu-gold Digitizer
RadCal AGMS-DM+ Accu-Gold Multi-Sensor
RadCal 90M9-AG Accu-Gold mAs Sensor
Accu-Gold Radiation Measurement Software
MX1 Positioning Jig (T-063)
Radiation PPE
SAMPLE SIZE
For this test, one (1) MX1 Portable X-ray System will be used. Accuracy of x-ray tube voltage, current, and loading time, which are being measured during serial and radioscopic imaging as part of this protocol, are required elements of essential performance per IEC 60601-2-54 (also reference MEMO-P01-662 Rev. A). Per IEC 60601-2-54 clause 201.5, clause 5 of IEC 60601-1:2005 is applicable to these tests; clause 5 of IEC 60601-1:2005 states that tests in scope of the standard are considered Type Tests. Type Tests are performed on a single representative sample of the item being tested (n=1).
Serial radiography and radioscopy pulse rates (as verified via pulse count during a single 20 second acquisition) is an internal MedAI requirement and is a software-dependent feature. Per QSP-026, Statistical Methods, characteristics driven by software can be tested with a minimum of n=1, based on the consistent nature of software and the inherent lack of variability in outputs given the same inputs.
METHODS
Locations and Personnel Responsibilities
This study shall take place in the MedAI office.
This study shall be conducted by members of the MedAI engineering team. This study does not require any special training.
Experimental Procedure
Set up the MX1 system in the T-063 positioning jig
Power the system on
Place the RadCal AGMS-DM+ Accu-Gold Multi-Sensor in the center of the beam with the sensor side of the sensor facing the Emitter
Connect the RadCal AGMS-DM+ Accu-Gold Multi-Sensor to the RadCal AGDM+ Accu-gold Digitizer and connect the Digitizer to a computer through the included USB cable
Connect the RadCal 90M9-AG mAs Sensor to the test point in the Emitter and connect the included cable to the RadCal AGDM+ Accu-gold Digitizer
Open the Accu-Gold Radiation Measurement Software and connect to the Digitizer via USB
In the software, ensure that the columns: “Ave. kV AGMS”, “Duration”, “Pulse Frequency”, “Pulse Count”, and “Current / Pulse mA” are present
Enclose the test setup in lead shielding of 0.5mm Pb equivalent or higher
Set the loading factors to 40kV and 0.04mAs
Trigger 1 serial radiograph of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Trigger 1 radioscopy x-ray of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 50kV and 0.04mAs
Trigger 1 radioscopy x-ray of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 60kV and 0.04mAs
Trigger 1 serial radiograph of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 64kV and 0.04mAs
Trigger 1 radioscopy x-ray of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 80kV and 0.04mAs
Trigger 1 serial radiograph of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 40kV and 0.08mAs
Trigger 1 serial radiograph of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Trigger 1 radioscopy x-ray of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 50kV and 0.08mAs
Trigger 1 radioscopy x-ray of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 60kV and 0.08mAs
Trigger 1 serial radiograph of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 64kV and 0.08mAs
Trigger 1 radioscopy x-ray of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Set the loading factors to 80kV and 0.08mAs
Trigger 1 serial radiograph of maximum duration allowed by the system
After the exposure, save screenshots from the waveform view where a single pulse is set as the area of interest. Gather screenshots of the 1st, 50th, and last pulses.
Save the Accu-Gold file to save all graphs and data.
Export the measurement log as an Excel file
Data Collection and Analysis
Open the saved Excel sheet and recorded pulse width screenshots for the 1st, 50th, and last pulses
X-ray pulse count:
Locate the column labeled “Pulse Count” and verify that each value listed has a value of 100 +/- 1 pulses
Record the results of this verification in Table 1.
X-ray pulse width (ms)
Open each saved screenshot of the pulse width data (1st, 50th, last pulses), and record the results (pulse duration time in ms)
Tube potential (kV)
From each pulse screenshot (1st, 50th, last) record the Average kV over the single pulse duration
Current (mA)
From each pulse screenshot (1st, 50th, last) record the Average mA over the single pulse duration
ACCEPTANCE CRITERIA
Acceptance criteria is listed in Table 1 below. Variable data will be recorded and included in the test report for reference, however data will be treated as attribute (pass/fail) for analysis purposes. Reference Appendix A for a data sheet template that includes acceptance criteria values for all test points.
Table 1. Acceptance Criteria - Pulse Number, Width, Voltage & Current
APPENDICES
Appendix A - Data Sheet Template with Acceptance Criteria
Table 1: Data Sheet Template with  Acceptance Criteria
Tested By: ______________________________________________Date: ______________
PROTOCOL APPROVAL
Digital Key: example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
Recorded By: _Alex Hartman___Date: __26 May 2024____
Recorded By: _Alex Hartman___Date: __26 May 2024____
RESULTS
Tested By: ________________Lu Dong_______________________Date: ___05/24/2024__
Conclusion:
The serial radiography and radioscopy verification tests for the MX1 Portable X-ray System were conducted successfully, and the results confirm that the system meets the established performance requirements. The key findings are as follows:
X-ray Beam Current:
The beam current measurements across all tested voltage and current settings consistently fell within the specified acceptance criteria. This indicates robust control over the X-ray tube's current output, ensuring reliable and accurate imaging performance.
Tube Voltage Accuracy:
The accuracy of the tube voltage was maintained within the required tolerance of ±8% for all test points. This demonstrates the system's capability to deliver precise voltage levels, which is critical for achieving consistent image quality and patient safety.
Pulse Duration:
The pulse durations were measured accurately and remained within the set time limits of 35-45 milliseconds across various test conditions. This confirms that the system can produce stable and predictable X-ray pulses, essential for high-quality radiographic imaging.
Reproducibility and Stability:
The number of pulses per 20-second acquisition was consistently within the acceptable range of 100 ± 1 pulse, verifying the system's ability to maintain a stable pulse rate over extended exposure times. This reproducibility is crucial for serial radiography applications, where consistent image acquisition is required.
Overall Results:
Pass
Fail
Other: Pass with Deviations
REPORT APPROVAL
Digital Key:example.com/

### Table 1
| Test Attribute | Data Type | Acceptance Criteria |
| --- | --- | --- |
| Number of pulses in a single 20 second acquisition (pulse rate of 5 frames per second) | Variable | 100 + 1 pulse (5 fps + 0.05 fps) |
| X-ray pulse width (time) | Variable | Set ms + (10% + 1ms) |
| Tube Potential (pulse voltage) | Variable | Set kV + 8% |
| Current mA (pulse current) | Variable | Set mA + 20% |

### Table 2
| Test | Result | Acceptance Criteria | Pass/ Fail |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Number of pulses in a single 20 second serial acquisition |  | 100 +/- 1 pulse |  |  |  |  |
| Test Point | Pulse | Voltage (kV) | Current (mA) | Time (ms) | Acceptance Criteria | Pass/ Fail |
| SERIAL Radiographic Imaging: 40kV, 1mA/40ms (0.04mAs) | 1st pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| Radioscopic 40 kV, 1mA/40ms (0.04 mAs) | 1st pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| SERIAL Radiographic Imaging: 60kV, 1mA/40ms (0.04mAs) | 1st pulse |  |  |  | Voltage: 55.2-64.8 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| Radioscopic 50 kV, 1mA/40ms (0.04 mAs) | 1st pulse |  |  |  | Voltage: 46-54 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| SERIAL Radiographic Imaging: 80kV, 1mA/40ms (0.04mAs) | 1st pulse |  |  |  | Voltage: 73.6-86.4 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| Radioscopic 64 kV, 1mA/40ms (0.04 mAs) | 1st pulse |  |  |  | Voltage: 58.9 - 69.1 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| SERIAL Radiographic Imaging: 40kV, 2mA/40ms (0.08mAs) | 1st pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| Radioscopic 40 kV, 2mA/40ms (0.08 mAs) | 1st pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| SERIAL Radiographic Imaging: 60kV, 2mA/40ms (0.08mAs) | 1st pulse |  |  |  | Voltage: 55.2-64.8 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| Radioscopic 50 kV, 2mA/40ms (0.08 mAs) | 1st pulse |  |  |  | Voltage: 46-54 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| SERIAL Radiographic Imaging: 80kV, 2mA/40ms (0.08mAs) | 1st pulse |  |  |  | Voltage: 73.6-86.4 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |
| Radioscopic 64 kV, 2mA/40ms (0.08 mAs) | 1st pulse |  |  |  | Voltage: 58.9 - 69.1 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
|  | 50th pulse |  |  |  |  |  |
|  | Last pulse |  |  |  |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 23 May 2024 | 24-262 |

### Table 4
| Equipment | Description | Last Calibration Date | Calibration Due Date |
| --- | --- | --- | --- |
| EQP-169 | 90M9-AG mAs Sensor | 9/28/2023 | 9/28/2024 |
| EQP-154 | AGDM+ Advanced Digitizer Module | 7/13/2023 | 7/13/2024 |

### Table 5
| Device ID | MX1 / DV26 |
| --- | --- |
| Emitter SN / Cassette SN | E1: SN 1222 C1: SN 1223 |
| Software Version | v3.1.0-beta |

### Table 6
| Test | Result | Acceptance Criteria | Pass/ Fail |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Number of pulses in a single 20 second serial acquisition | 99 | 100 +/- 1 pulse | Pass |  |  |  |
| Test Point | Pulse | Voltage (kV) | Current (mA) | Time (ms) | Acceptance Criteria | Pass/ Fail |
| SERIAL Radiographic Imaging: 40kV, 1mA/40ms (0.04mAs) | 1st pulse | 40.9 | 0.9481 | 39 | Voltage: 36.8-43.2 kV Current: 0.8-1.2 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 40.9 | 0.9401 | 39 |  | Pass |
|  | Last pulse | 41 | 0.9294 | 39 |  | Pass |
| Radioscopic 40 kV, 1mA/40ms (0.04 mAs) | 1st pulse | 40.88 | 0.941 | 38.8 | Voltage: 36.8-43.2 kV Current: 0.8-1.2 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 40.87 | 0.9307 | 38.2 |  | Pass |
|  | Last pulse | 40.93 | 0.9339 | 39 |  | Pass |
| SERIAL Radiographic Imaging: 60kV, 1mA/40ms (0.04mAs) | 1st pulse | 60.83 | 0.9654 | 39.2 | Voltage: 55.2-64.8 kV Current: 0.8-1.2 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 60.9 | 0.9562 | 39 |  | Pass |
|  | Last pulse | 60.83 | 0.9527 | 39 |  | Pass |
| Radioscopic 50 kV, 1mA/40ms (0.04 mAs) | 1st pulse | 50.51 | 0.9555 | 39 | Voltage: 46-54 kV Current: 0.8-1.2 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 50.45 | 0.9548 | 39 |  | Pass |
|  | Last pulse | 50.51 | 0.9542 | 39 |  | Pass |
| SERIAL Radiographic Imaging: 80kV, 1mA/40ms (0.04mAs) | 1st pulse | 81.1 | 0.999 | 39.2 | Voltage: 73.6-86.4 kV Current: 0.8-1.2 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 81.19 | 0.9825 | 39 |  | Pass |
|  | Last pulse | 81.08 | 0.982 | 39 |  | Pass |
| Radioscopic 64 kV, 1mA/40ms (0.04 mAs) | 1st pulse | 64.73 | 0.966 | 39 | Voltage: 58.9 - 69.1 kV Current: 0.8-1.2 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 64.55 | 0.9707 | 39 |  | Pass |
|  | Last pulse | 64.38 | 0.9605 | 39 |  | Pass |
| SERIAL Radiographic Imaging: 40kV, 2mA/40ms (0.08mAs) | 1st pulse | 40.68 | 1.856 | 39 | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 40.69 | 1.841 | 39 |  | Pass |
|  | Last pulse | 40.58 | 1.826 | 39 |  | Pass |
| Radioscopic 40 kV, 2mA/40ms (0.08 mAs) | 1st pulse | 40.56 | 1.873 | 38.9 | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 40.65 | 1.85 | 39 |  | Pass |
|  | Last pulse | 40.66 | 1.848 | 39 |  | Pass |
| SERIAL Radiographic Imaging: 60kV, 2mA/40ms (0.08mAs) | 1st pulse | 60.52 | 1.958 | 38.9 | Voltage: 55.2-64.8 kV Current: 1.6-2.4 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 60.43 | 1.944 | 39 |  | Pass |
|  | Last pulse | 60.35 | 1.942 | 39 |  | Pass |
| Radioscopic 50 kV, 2mA/40ms (0.08 mAs) | 1st pulse | 50.05 | 1.929 | 39.1 | Voltage: 46-54 kV Current: 1.6-2.4 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 50.1 | 1.901 | 39 |  | Pass |
|  | Last pulse | 50.15 | 1.893 | 38.5 |  | Pass |
| SERIAL Radiographic Imaging: 80kV, 2mA/40ms (0.08mAs) | 1st pulse | 80.76 | 1.967 | 38.8 | Voltage: 73.6-86.4 kV Current: 1.6-2.4 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 80.74 | 1.954 | 38 |  | Pass |
|  | Last pulse | 80.85 | 1.951 | 38 |  | Pass |
| Radioscopic 64 kV, 2mA/40ms (0.08 mAs) | 1st pulse | 64.56 | 1.975 | 38.9 | Voltage: 58.9 - 69.1 kV Current: 1.6-2.4 mA Time: 35-45 ms | Pass |
|  | 50th pulse | 64.47 | 1.935 | 38.5 |  | Pass |
|  | Last pulse | 64.38 | 1.929 | 39 |  | Pass |

### Table 7
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 27 May 2024 | 23-290 |
