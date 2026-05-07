# VVPR-SWV-013 Rev B: Camera Self Test Utility Verification and Validation v1.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-SWV-013
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.0.0
- Source filename: VVPR-SWV-013 - Camera Self Test Utility Verification and Validation v1.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-013 - Camera Self Test Utility Verification and Validation v1.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Camera Test Utility meets usability and functional requirements as stated in MEMO-P01-704 - MX1 Camera Self Test Utility Software Requirements Specification.
OBJECTIVE
The objective of this study is to verify that the v1.0.0 release of the MedAI Camera Self Test Utility meets system-level requirements set by MedAI as documented in MEMO-P01-704.
REFERENCES
MEMO-P01-704 - MX1 Camera Self Test Utility Software Requirements Specification Rev. A
MATERIALS
MX1 E1 Emitter Rev. H
MX1 SS Version v3.1.0
S10057 MX1 Camera Self Test Utility v1.0.0
A computer capable of running a web browser
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Table 1: Requirements, Verification Steps, and Expected Response.
Data Analysis
All of the verification tests in Table 1 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Table 1 per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: DV-26
MX1 SS Version v3.1.0
S10046 MX1 MedAI Rest Server v1.0.0.
RESULTS
Table 1: Requirements, Verification Steps, and Expected Response
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
APPENDIX/ATTACHMENTS
Appendix 1: CTU Correctly Checks IMX577 Existence
Appendix 2: CTU Opens IMX577
Appendix 3: CTU Signals IMX577 to Start Capture
Appendix 4: CTU Expects IMX577 Test Patterns, Returns Whether Test Patterns Match
Appendix 5: CTU Correctly Checks IMX715 Existence
Appendix 6: CTU Opens IMX715
Appendix 7: CTU Signals IMX715 To Start Capture
Appendix 8: CTU Expects IMX715 Test Patterns, Returns Whether Test Patterns Match
Appendix 9: CTU Correctly Checks IMX335 Existence
Appendix 10: CTU Opens IMX335
Appendix 11:  CTU Signals IMX335 to Start Capture
Appendix 12: CTU Expects IMX335 Test Patterns, Returns Whether Test Patterns Match
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The Emitter is powered on and a remote connection has been established via SSH. |  |  |  |  |
|  |  | Test Case: Verification of camera imx577 |  |  |  |
| SRS-2.1 | The CTU shall correctly check the existence of camera imx577 | 1. Run the command /opt/medai/bin/camera-test-util -o cam-test-output.json >> camera-self-test.log 2. Run the commandcat camera-self-test.log and locate the lines "INFO <TIME_STAMP> [ec] (CameraFlow) Argus- init device" "INFO <TIME_STAMP> [ec] CameraFlow) Searching for imx577 | Verify the lines below exist in the log: INFO <TIME_STAMP> [ec] (CameraFlow) Searching for imx577 INFO <TIME_STAMP> [ec] (CameraFlow) Camera info: |  |  |
| SRS-2.2 | The CTU shall open camera imx577 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx577" -> "camera_opened" | Verify the value is true |  |  |
| SRS-2.3 | The CTU shall signal camera imx577 to start capture | 1. Run the command cat camera-self-test.log | grep -a "Camera flow for imx577 got" | Locate the message "Camera flow for imx577 got start capture signal" in the log file |  |  |
| SRS-2.4 | The CTU shall expect specific test patterns from camera imx577 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx577" -> "test_patterns" | For each sub object, verify that the key "captured" has value true |  |  |
| SRS-2.5 | The CTU shall verify that returned test patterns match expected values for camera imx577 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx577" -> "test_patterns" | For each sub object, verify that the key "pass" has value true |  |  |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The emitter is powered on. |  |  |  |  |
|  |  | Test Case: Verification of camera imx715 |  |  |  |
| SRS-2.6 | The CTU shall correctly check the existence of camera imx715 | 1. Run the command /opt/medai/bin/camera-test-util -o cam-test-output.json >> camera-self-test.log 2. Run the commandcat camera-self-test.log and locate the lines "INFO <TIME_STAMP> [ec] (CameraFlow) Argus- init device" "INFO <TIME_STAMP> [ec] CameraFlow) Searching for imx715 | Verify the lines below exist in the log INFO <TIME_STAMP> [ec] (CameraFlow) Searching for imx715 INFO <TIME_STAMP> [ec] (CameraFlow) Camera info: |  |  |
| SRS-2.7 | The CTU shall open camera imx715 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx715" -> "camera_opened" | Verify the value is true |  |  |
| SRS-2.8 | The CTU shall signal camera imx715 to start capture | Run the command cat camera-self-test.log | grep -a "Camera flow for imx715 got" | Locate the message "Camera flow for imx715 got start capture signal" in the log file |  |  |
| SRS-2.9 | The CTU shall expect specific test patterns from camera imx715 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx715" -> "test_patterns" | For each sub object, verify that the key "captured" has value true |  |  |
| SRS-2.10 | The CTU shall verify that returned test patterns match expected values for camera imx715 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx715" -> "test_patterns" | For each sub object, verify that the key "pass" has value true |  |  |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The emitter is powered on. |  |  |  |  |
|  |  | Test Case: Verification of camera imx335 |  |  |  |
| SRS-2.11 | The CTU shall correctly check the existence of camera imx335 | 1. Run the command /opt/medai/bin/camera-test-util -o cam-test-output.json >> camera-self-test.log 2. Run the commandcat camera-self-test.log and locate the lines "INFO <TIME_STAMP> [ec] (CameraFlow) Argus- init device" "INFO <TIME_STAMP> [ec] CameraFlow) Searching for imx335 | Verify the lines below exist in the log INFO <TIME_STAMP> [ec] (CameraFlow) Searching for imx335 INFO <TIME_STAMP> [ec] (CameraFlow) Camera info: |  |  |
| SRS-2.12 | The CTU shall open camera imx335 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx335" -> "camera_opened" | Verify the value is true |  |  |
| SRS-2.13 | The CTU shall signal camera imx335 to start capture | Run the command cat camera-self-test.log | grep -a "Camera flow for imx335 got" | Locate the message "Camera flow for imx335 got start capture signal" in the log file |  |  |
| SRS-2.14 | The CTU shall expect specific test patterns from camera imx335 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx335" -> "test_patterns" | For each sub object, verify that the key "captured" has value true |  |  |
| SRS-2.15 | The CTU shall verify that returned test patterns match expected values for camera imx335 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx335" -> "test_patterns" | For each sub object, verify that the key "pass" has value true |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 02 Jul 2024 | 24-399 |

### Table 3
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The Emitter is powered on and a remote connection has been established via SSH. |  |  |  |  |
|  |  | Test Case: Verification of camera imx577 |  |  |  |
| SRS-2.1 | The CTU shall correctly check the existence of camera imx577 | 1. Run the command /opt/medai/bin/camera-test-util -o cam-test-output.json >> camera-self-test.log 2. Run the commandcat camera-self-test.log and locate the lines "INFO <TIME_STAMP> [ec] (CameraFlow) Argus- init device" "INFO <TIME_STAMP> [ec] CameraFlow) Searching for imx577 | Verify the lines below exist in the log: INFO <TIME_STAMP> [ec] (CameraFlow) Searching for imx577 INFO <TIME_STAMP> [ec] (CameraFlow) Camera info: | Expected outcome verified. See Appendix 1. Verified by MS 02JULY24 | P |
| SRS-2.2 | The CTU shall open camera imx577 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx577" -> "camera_opened" | Verify the value is true | Expected outcome verified. See Appendix 2. Verified by MS 02JULY24 | P |
| SRS-2.3 | The CTU shall signal camera imx577 to start capture | 1. Run the command cat camera-self-test.log | grep -a "Camera flow for imx577 got" | Locate the message "Camera flow for imx577 got start capture signal" in the log file | Expected outcome verified. See Appendix 3. Verified by MS 02JULY24 | P |
| SRS-2.4 | The CTU shall expect specific test patterns from camera imx577 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx577" -> "test_patterns" | For each sub object, verify that the key "captured" has value true | Expected outcome verified. See Appendix 4. Verified by MS 02JULY24 | P |
| SRS-2.5 | The CTU shall verify that returned test patterns match expected values for camera imx577 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx577" -> "test_patterns" | For each sub object, verify that the key "pass" has value true | Expected outcome verified. See Appendix 4. Verified by MS 02JULY24 | P |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The emitter is powered on. |  |  |  |  |
|  |  | Test Case: Verification of camera imx715 |  |  |  |
| SRS-2.6 | The CTU shall correctly check the existence of camera imx715 | 1. Run the command /opt/medai/bin/camera-test-util -o cam-test-output.json >> camera-self-test.log 2. Run the commandcat camera-self-test.log and locate the lines "INFO <TIME_STAMP> [ec] (CameraFlow) Argus- init device" "INFO <TIME_STAMP> [ec] CameraFlow) Searching for imx715 | Verify the lines below exist in the log INFO <TIME_STAMP> [ec] (CameraFlow) Searching for imx715 INFO <TIME_STAMP> [ec] (CameraFlow) Camera info: | Expected outcome verified. See Appendix 5. Verified by MS 02JULY24 | P |
| SRS-2.7 | The CTU shall open camera imx715 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx715" -> "camera_opened" | Verify the value is true | Expected outcome verified. See Appendix 6. Verified by MS 02JULY24 | P |
| SRS-2.8 | The CTU shall signal camera imx715 to start capture | Run the command cat camera-self-test.log | grep -a "Camera flow for imx715 got" | Locate the message "Camera flow for imx715 got start capture signal" in the log file | Expected outcome verified. See Appendix 7. Verified by MS 02JULY24 | P |
| SRS-2.9 | The CTU shall expect specific test patterns from camera imx715 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx715" -> "test_patterns" | For each sub object, verify that the key "captured" has value true | Expected outcome verified. See Appendix 8. Verified by MS 02JULY24 | P |
| SRS-2.10 | The CTU shall verify that returned test patterns match expected values for camera imx715 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx715" -> "test_patterns" | For each sub object, verify that the key "pass" has value true | Expected outcome verified. See Appendix 8. Verified by MS 02JULY24 | P |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The emitter is powered on. |  |  |  |  |
|  |  | Test Case: Verification of camera imx335 |  |  |  |
| SRS-2.11 | The CTU shall correctly check the existence of camera imx335 | 1. Run the command /opt/medai/bin/camera-test-util -o cam-test-output.json >> camera-self-test.log 2. Run the commandcat camera-self-test.log and locate the lines "INFO <TIME_STAMP> [ec] (CameraFlow) Argus- init device" "INFO <TIME_STAMP> [ec] CameraFlow) Searching for imx577 | Verify the lines below exist in the log INFO <TIME_STAMP> [ec] (CameraFlow) Searching for imx335 INFO <TIME_STAMP> [ec] (CameraFlow) Camera info: | Expected outcome verified. See Appendix 9. Verified by MS 02JULY24 | P |
| SRS-2.12 | The CTU shall open camera imx335 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx335" -> "camera_opened" | Verify the value is true | Expected outcome verified. See Appendix 10. Verified by MS 02JULY24 | P |
| SRS-2.13 | The CTU shall signal camera imx335 to start capture | Run the command cat camera-self-test.log | grep -a "Camera flow for imx335 got" | Locate the message "Camera flow for imx335 got start capture signal" in the log file | Expected outcome verified. See Appendix 11. Verified by MS 02JULY24 | P |
| SRS-2.14 | The CTU shall expect specific test patterns from camera imx335 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx335" -> "test_patterns" | For each sub object, verify that the key "captured" has value true | Expected outcome verified. See Appendix 12. Verified by MS 02JULY24 | P |
| SRS-2.15 | The CTU shall verify that returned test patterns match expected values for camera imx335 | 1. Run the command cat cam-test-output.json 2. Locate the key "imx335" -> "test_patterns" | For each sub object, verify that the key "pass" has value true | Expected outcome verified. See Appendix 12. Verified by MS 02JULY24 | P |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-484 |  |  |
