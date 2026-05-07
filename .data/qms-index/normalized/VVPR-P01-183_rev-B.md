# VVPR-P01-183 Rev B: MX1 Software System Photographic Acquisition v3.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-183
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.0.0
- Source filename: VVPR-P01-183 - MX1 Software System Photographic Acquisition v3.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-183 - MX1 Software System Photographic Acquisition v3.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Photographic mode indications
Photographic acquisition
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
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 1. Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220
C1 Cassette Rev. I, SN: 1221
F1 Foot Pedal Rev. B, Lot #: 10010
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System v3.0.0
EQP-139 Control Company Stopwatch 4YMT7
RESULTS
Table 1. Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 and 3 - Verification Evidence as Specified in Results Table 1.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1: Emitter Display in Photo Mode
Appendix 2: MX1 Device in Photo mode
Appendix 3: Photograph in ODA

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App, F1 Foot Pedal |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Acquisition Mode Indications - Photographic Mode |  |  |  |  |
| SRS-21.1 | The SS shall support a user-selectable mode for taking photos | 1. Ensure a foot pedal is paired to the emitter 2. Using either the emitter keypad or foot pedal, switch into photographic mode 3. Verify only the UI elements listed in the Pass Criteria column remain on the emitter display | Mode indication icon, indicating photo mode |  |  |
| SRS-21.5 | In photographic mode, the SS shall display only the following elements on the emitter touchscreen display: 1. Mode indication icon 2. Dewarped camera feed 3. Cassette connection icon 4. Foot pedal connection icon (if applicable) 5. Battery charge indicator |  |  |  |  |
|  |  |  | Flat lens / non-fisheye lens camera feed |  |  |
|  |  |  | Cassette connection icon |  |  |
|  |  |  | Foot pedal connection icon |  |  |
|  |  |  | Battery charge indicator |  |  |
| SRS-21.3 | The SS shall indicate photographic mode by setting emitter and cassette MI LEDs to white |  | Emitter MI (Mode Indicator) LEDs set to steady white |  |  |
|  |  |  | Cassette MI LEDs set to steady white |  |  |
| SRS-21.4 | During acquisition of a photo, the SS shall set the cassette MI LEDs to 0 for the duration of photographic capture | 1. Navigate to the Acquisition Screen of the MedAI Device App 2. Use emitter trigger or foot pedal to capture photo | Verify cassette MI LEDs turn off for the duration of capture |  |  |
| SRS-21.2 | The SS shall take a photo when the emitter trigger is pressed when system is in photo mode |  | Photograph appears in the Acquisition Screen of ODA |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 14 May 2024 | 24-239 |

### Table 3
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-139 | 9/12/2022 | 9/12/2024 |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App, F1 Foot Pedal |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Acquisition Mode Indications - Photographic Mode |  |  |  |  |
| SRS-21.1 | The SS shall support a user-selectable mode for taking photos | 1. Ensure a foot pedal is paired to the emitter 2. Using either the emitter keypad or foot pedal, switch into photographic mode 3. Verify only the UI elements listed in the Pass Criteria column remain on the emitter display | Mode indication icon, indicating photo mode | Expected outcome verified. See Appendix 1. Verified by AM 21MAY24 | P |
| SRS-21.5 | In photographic mode, the SS shall display only the following elements on the emitter touchscreen display: 1. Mode indication icon 2. Dewarped camera feed 3. Cassette connection icon 4. Foot pedal connection icon (if applicable) 5. Battery charge indicator |  |  |  |  |
|  |  |  | Flat lens / non-fisheye lens camera feed | Expected outcome verified. See Appendix 1. Verified by AM 21MAY24 | P |
|  |  |  | Cassette connection icon | Expected outcome verified. See Appendix 1. Verified by AM 21MAY24 | P |
|  |  |  | Foot pedal connection icon | Expected outcome verified. See Appendix 1. Verified by AM 21MAY24 | P |
|  |  |  | Battery charge indicator | Expected outcome verified. See Appendix 1. Verified by AM 21MAY24 | P |
| SRS-21.3 | The SS shall indicate photographic mode by setting emitter and cassette MI LEDs to white |  | Emitter MI (Mode Indicator) LEDs set to steady white | Expected outcome verified. See Appendix 2. Verified by RM 14MAY24 | P |
|  |  |  | Cassette MI LEDs set to steady white | Expected outcome verified. See Appendix 2. Verified by RM 14MAY24 | P |
| SRS-21.4 | During acquisition of a photo, the SS shall set the cassette MI LEDs to 0 for the duration of photographic capture | 1. Navigate to the Acquisition Screen of the MedAI Device App 2. Use emitter trigger or foot pedal to capture photo | Verify cassette MI LEDs turn off for the duration of capture | Expected outcome verified. Verified by RM 14MAY24 | P |
| SRS-21.2 | The SS shall take a photo when the emitter trigger is pressed when system is in photo mode |  | Photograph appears in the Acquisition Screen of ODA | Expected outcome verified. See Appendix 3. Verified by AM 21MAY24 | P |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-440 |  |
