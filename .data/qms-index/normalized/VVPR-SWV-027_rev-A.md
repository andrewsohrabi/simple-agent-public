# VVPR-SWV-027 Rev A: Galden Verification Firmware Verification and Validation Protocol and Report B(1)

## Metadata
- Document ID: VVPR-SWV-027
- Revision: A
- Prefix: VVPR
- Latest revision: False
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-027 - Galden Verification Firmware Verification and Validation Protocol and Report_B(1).docx
- Source path: Example QMS - MedAI/VVPR-SWV-027 - Galden Verification Firmware Verification and Validation Protocol and Report_B(1).docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the galden-verification-firmware meets usability and functional requirements as stated in MEMO-P01-738 galden-verification-firmware Software Requirements Specification Rev. A.
OBJECTIVE
The primary objective of this study is to verify the Production Firmware Tool: galden-verification-firmware for use in T-199 as part of WS-017 for production of the monoblock used in the MX1 emitter.
REFERENCES
MEMO-P01-738 - galden-verification-firmware Software Requirements Specification Rev A
MWI-276 - MS-11235 Monoblock, Power Assembly Rev A
MWI-275 - WS-017 Workstation Installation Rev B
MATERIALS
MS-11225 WS-017 Electronics Box Rev B
S10103 galden-verification-firmware v1.0.0-alpha
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121 or equivalent)
M50564 Dell Computer Rev A (or equivalent)
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in each table below.
Table 1: System Interfacing.
Data Analysis
All of the verification tests in Tables 1 through 5 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 through 5 per the expected results documented in the “Expected Result/Pass Criteria” column
DOCUMENT REVISION HISTORY
Digital Key:
example.com/
Report Section
Deviations
None
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.0-alpha
MS-11235 Monoblock Power Assembly Rev. A
SPD3303X Siglent Benchtop power supply (EQP-246 or equivalent)
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121 or equivalent)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238 or equivalent)
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Table 1: System Interfacing.
Discussion
No anomalies were found during the course of testing, and no changes or deviations were recorded.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Appendix 1:
1.1 Frequencies
391.1kHz (396kHz)
520.2kHz (525kHz)
647.5kHz (654kHz)
720kHz (728kHz)
2.2 Pulse width 200ms (199ms)
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | MS-11225 PC Oscilloscope |  |  |  |  |
| Test Setup: | 1. Connect the USB cables coming from MS-11225 to the PC 2. Establish a connection between the PC and the device such that they can communicate 3. Connect one of the pins of the secondary harness extruding from MS-11225 to the oscilloscope |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The galden-verification-firmware shall be capable of adjusting H-Bridge frequency | 1. Set the exposure time to H-Bridge duration to 200 ms 2. Set the frequency to 391.3 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope 3. Set the frequency to 520.2 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope 4. Set the frequency to 647.5 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope 5. Set the frequency to 720 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope | The measured frequencies are within 5% of the input frequency |  |  |
| SRS-1.2 | The galden-verification-firmware shall be capable of setting the duration during which the H-Bridge is active to 200 ms | 1. Set the frequency to 500 kHz 2. Set the H-Bridge duration to 200 ms, activate the H-Bridge, and measure the duration using the oscilloscope | The measured duration is within 5% of 200 ms |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-542 |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | MS-11225 PC Oscilloscope |  |  |  |  |
| Test Setup: | 1. Connect the USB cables coming from MS-11225 to the PC 2. Establish a connection between the PC and the device such that they can communicate 3. Connect one of the pins of the secondary harness extruding from MS-11225 to the oscilloscope |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The galden-verification-firmware shall be capable of adjusting H-Bridge frequency | 1. Set the exposure time to H-Bridge duration to 200 ms 2. Set the frequency to 391.3 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope 3. Set the frequency to 520.2 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope 4. Set the frequency to 647.5 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope 5. Set the frequency to 720 kHz, activate the H-Bridge, and measure the frequency using the oscilloscope | The measured frequencies are within 5% of the input frequency | Expected operation verified See Appendix 1.1 Verified by EM 05SEP2024 | Pass |
| SRS-1.2 | The galden-verification-firmware shall be capable of setting the duration during which the H-Bridge is active to 200 ms | 1. Set the frequency to 500 kHz 2. Set the H-Bridge duration to 200 ms, activate the H-Bridge, and measure the duration using the oscilloscope | The measured duration is within 5% of 200 ms | Expected operation verified See Appendix 1.2 Verified by EM 05SEP2024 | Pass |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-553 |  |  |
