# VVPR-SWV-037 Rev B: Emitter Firmware WS-013 Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-037
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-037 - Emitter Firmware WS-013 Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-037 - Emitter Firmware WS-013 Verification and Validation Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the Emitter Firmware WS-013 meets usability and functional requirements as stated in MEMO-P01-830 - Emitter Firmware WS-013 Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify the Non-Product Software Tool: Emitter Firmware WS-013
REFERENCES
MEMO-P01-830 - Emitter Firmware WS-013 Software Requirements Specification Rev. A
MATERIALS
S10112 - Emitter Firmware WS-013 - v1.0.0-alpha (74281a7d7cae0ba1191949a997595e8a9d14ecfa)
F1 Footpedal BOM Rev D
T-100 Rev A with MS-10928 Rev A
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification will be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001 by trained MedAI engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in the table(s) below.
Table 1: Foot Pedal UUID Reader Requirement Verification
Data Analysis
All of the verification tests in Table 1 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Table 1 per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
Recorded By: SUNGWON PARKDate: 1/30/25
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
F1 Footpedal BOM Rev D (UUID: 4587586)
T-100 Rev A with MS-10928 Rev A
S10112 - Emitter Firmware WS-013 - v1.0.0-alpha (74281a7d7cae0ba1191949a997595e8a9d14ecfa)
RESULTS
Table 1: Foot Pedal UUID Reader Requirement Verification
CONCLUSIONS
The above procedure was meant to test functionality of the Emitter Firmware WS-013 to print out the reception of Left Pedal, Right pedal, Left Button, and Right Button presses to serial. All tests successfully passed with no anomalies
Overall Result:.
Pass
Fail
Other: _______
LIST OF APPENDICES
Appendix 1
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: Materials as listed Above |  |  |  |  |
| Precondition: | Power on T-100 and MS-10928, Flash emitter with S10112, pair F1 Foot Pedal with WS-013 Emitter, Plug in USB-mini B from emitter mcu serial port to NUC, open serial terminal baude rate 115200 |  |  |  |  |
| SRS-1.1 | The Emitter Firmware WS-013 shall print to serial "Left Button Pressed" and "Left Button Released" when the left Foot Pedal button is pressed and released | 1. Press and Release Left Button on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Left Button Pressed” and “Left Button Released” are present | “Left Button Pressed” and “Left Button Released” are present in serial terminal |  |  |
| SRS-1.2 | The Emitter Firmware WS-013 shall print to serial "Right Button Pressed" and "Right Button Released" when the right Foot Pedal button is pressed and released | 1. Press and Release Right Button on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Right Button Pressed” and “Right Button Released” are present | “Right Button Pressed” and “Right Button Released” are present in serial terminal |  |  |
| SRS-1.3 | The Emitter Firmware WS-013 shall print to serial "Left Pedal Pressed" and "Left Pedal Released" when the left Foot Pedal pedal is pressed and released | 1. Press and Release Left Pedal on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Left Pedal Pressed” and “Left Pedal Released” are present | “Left Pedal Pressed” and “Left Pedal Released” are present in serial terminal |  |  |
| SRS-1.4 | The Emitter Firmware WS-013 shall print to serial "Right Pedal Pressed" and "Right Pedal Released" when the right Foot Pedal pedal is pressed and released | 1. Press and Release Right Pedal on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Right Pedal Pressed” and “Right Pedal Released” are present | “Right Pedal Pressed” and “Right Pedal Released” are present in serial terminal |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-692 |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: Materials as listed Above |  |  |  |  |
| Precondition: | Power on T-100 and MS-10928, Flash emitter with S10112, pair F1 Foot Pedal with WS-013 Emitter, Plug in USB-mini B from emitter mcu serial port to NUC, open serial terminal baude rate 115200 |  |  |  |  |
| SRS-1.1 | The Emitter Firmware WS-013 shall print to serial "Left Button Pressed" and "Left Button Released" when the left Foot Pedal button is pressed and released | 1. Press and Release Left Button on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Left Button Pressed” and “Left Button Released” are present | “Left Button Pressed” and “Left Button Released” are present in serial terminal | See Appendix 1 | PASS Verified By: SP 1/30/25 |
| SRS-1.2 | The Emitter Firmware WS-013 shall print to serial "Right Button Pressed" and "Right Button Released" when the right Foot Pedal button is pressed and released | 1. Press and Release Right Button on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Right Button Pressed” and “Right Button Released” are present | “Right Button Pressed” and “Right Button Released” are present in serial terminal | See Appendix 1 | PASS Verified By: SP 1/30/25 |
| SRS-1.3 | The Emitter Firmware WS-013 shall print to serial "Left Pedal Pressed" and "Left Pedal Released" when the left Foot Pedal pedal is pressed and released | 1. Press and Release Left Pedal on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Left Pedal Pressed” and “Left Pedal Released” are present | “Left Pedal Pressed” and “Left Pedal Released” are present in serial terminal | See Appendix 1 | PASS Verified By: SP 1/30/25 |
| SRS-1.4 | The Emitter Firmware WS-013 shall print to serial "Right Pedal Pressed" and "Right Pedal Released" when the right Foot Pedal pedal is pressed and released | 1. Press and Release Right Pedal on F1 Foot Pedal 2. In the serial terminal, verify that the messages “Right Pedal Pressed” and “Right Pedal Released” are present | “Right Pedal Pressed” and “Right Pedal Released” are present in serial terminal | See Appendix 1 | PASS Verified By: SP 1/30/25 |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release | Refer to ECR-696 |  |  |
