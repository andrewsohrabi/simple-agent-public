# VVPR-SWV-011 Rev B: MX1 MedAI Rest Server Verification and Validation v1.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-SWV-011
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.0.0
- Source filename: VVPR-SWV-011 - MX1 MedAI Rest Server Verification and Validation v1.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-011 - MX1 MedAI Rest Server Verification and Validation v1.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify that the software medai-rest-icd allows the user to send commands to the device firmware.
OBJECTIVE
The objective of this study is to verify that the v1.0.0 release of the MedAI Rest Server meets system-level requirements set by MedAI as documented in MEMO-P01-702.
REFERENCES
MEMO-P01-702 - MedAI Rest Server (ORS) Software Requirements Specification Rev. A
MATERIALS
MX1 E1 Emitter Rev. E SW Version v3.1.0
S10046 MX1 MedAI Rest Server v1.0.0
A computer capable of running a web browser
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in Table 1 below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
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
S10046 MX1 MedAI Rest Server v1.0.0
RESULT
Table 1: Requirements, Verification Steps, and Expected Response
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
APPENDIX/ATTACHMENT
Appendix 1: Successful Firmware Read
Appendix 2: Successful Firmware Write
Appendix 3: ICD Error (Wrong Register) Reported
Appendix 4: ICD Error (Wrong Length) Reported
Appendix 5: Time Of Flight Data Reported
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The Emitter is powered on and a remote connection has been established via SSH. |  |  |  |  |
|  |  | Test Case: MedAI Rest Server |  |  |  |
| SRS-1.1 | The MedAI Rest Server (ORS) shall allow read commands to firmware | 1. Open a terminal and stop services with python3.11 -m mx1.services stop 2. Start Rest ICD service with the command systemctl --user start rest-icd.service 3. Open a web browser and read register zero: {device_name}:8090/remote_api?command=p&pid=3&op=0&reg=0 | 1. Json response should contain: { "icd_error": "none", "op": 0, "payload": "0x03", "reg": 0, "response": "0xc0 0x08 [SQN] 0x03 0x00 0x00 0x03 [CS] 0x00 0xc0" } |  |  |
| SRS-1.2 | The ORS shall allow write commands to firmware | 4. Open a web browser and write {device_name}:8090/remote_api?command=p&pid=3&op=1&reg=02&payload=00000000 | 2. Json response should contain: { "flags": "", "icd_error": "none", "op": 1, "payload": "0x00 0x00 0x00 0x00", "reg": 2, "response": "0xc0 0x0b [SQN] 0x03 0x01 0x02 0x00 0x00 0x00 0x00 [CS] 0x00 0xc0" } |  |  |
| SRS-1.3 | The ORS shall return the firmware's error for incorrect commands to firmware | 1. Register error. Send an incorrect command to a register. {device_name}:8090/remote_api?command=p&pid=3&op=1&reg=00&payload=0 | 3. Json response should contain: { "icd_error": "ICD_ERROR_REGISTER", "op": 1, "payload": "", "reg": 0, "response": "0xc0 [SQN] 0x03 0x03 0x10 0x0a [CS] 0x00 0xc0" } |  |  |
|  |  | 2. Length Error. Send a payload of incorrect length to a register {device_name}:8090/remote_api?command=p&pid=3&op=1&reg=00&payload=000000 | 4. Json response should contain: { "icd_error": "ICD_ERROR_LENGTH", "op": 1, "payload": "", "reg": 0, "response": "0xc0 [SQN] 0x04 0x03 0x10 0x02 [CS] 0x00 0xc0" } |  |  |
| SRS-1.4 | The ORS shall stream time of flight data from the firmware | 1. Start ToF Rest ICD service with the command systemctl --user start rest-icd-tof.service 2. Send request to read time of flight data with the following command: {device_name}:8099/remote_api?command=f&pid=4&quantity=4&clear=ba | 1. Json response should contain: { "errors": "none | none | none | none", "packets": [ "0xc0 ... 0xc0", "0xc0 ... 0xc0", "0xc0 ... 0xc0", "0xc0 ... 0xc0" ], "payloads": [ [payload one], [payload two], [payload three], [payload four] ] }, where the ellipsis represent a series of bytes. |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Assurance Regulatory Affairs | 28 Jun 2024 | 24-395 |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The Emitter is powered on and a remote connection has been established via SSH. |  |  |  |  |
|  |  | Test Case: MedAI Rest Server |  |  |  |
| SRS-1.1 | The MedAI Rest Server (ORS) shall allow read commands to firmware | 1. Open a terminal and stop services with python3.11 -m mx1.services stop 2. Start Rest ICD service with the command systemctl --user start rest-icd.service 3. Open a web browser and read register zero: {device_name}:8090/remote_api?command=p&pid=3&op=0&reg=0 | 1. Json response should contain: { "icd_error": "none", "op": 0, "payload": "0x03", "reg": 0, "response": "0xc0 0x08 [SQN] 0x03 0x00 0x00 0x03 [CS] 0x00 0xc0" } | Expected outcome verified. See Appendix 1. Verified by MS 28JUNE24 | P |
| SRS-1.2 | The ORS shall allow write commands to firmware | 4. Open a web browser and write {device_name}:8090/remote_api?command=p&pid=3&op=1&reg=02&payload=00000000 | 2. Json response should contain: { "flags": "", "icd_error": "none", "op": 1, "payload": "0x00 0x00 0x00 0x00", "reg": 2, "response": "0xc0 0x0b [SQN] 0x03 0x01 0x02 0x00 0x00 0x00 0x00 [CS] 0x00 0xc0" } | Expected outcome verified. See Appendix 2. Verified by MS 28JUNE24 | P |
| SRS-1.3 | The ORS shall return the firmware's error for incorrect commands to firmware | 1. Register error. Send an incorrect command to a register. {device_name}:8090/remote_api?command=p&pid=3&op=1&reg=00&payload=0 | 3. Json response should contain: { "icd_error": "ICD_ERROR_REGISTER", "op": 1, "payload": "", "reg": 0, "response": "0xc0 [SQN] 0x03 0x03 0x10 0x0a [CS] 0x00 0xc0" } | Expected outcome verified. See Appendix 3. Verified by MS 28JUNE24 | P |
|  |  | 2. Length Error. Send a payload of incorrect length to a register {device_name}:8090/remote_api?command=p&pid=3&op=1&reg=00&payload=000000 | 4. Json response should contain: { "icd_error": "ICD_ERROR_LENGTH", "op": 1, "payload": "", "reg": 0, "response": "0xc0 [SQN] 0x04 0x03 0x10 0x02 [CS] 0x00 0xc0" } | Expected outcome verified. See Appendix 4. Verified by MS 28JUNE24 | P |
| SRS-1.4 | The ORS shall stream time of flight data from the firmware | 1. Start ToF Rest ICD service with the command systemctl --user start rest-icd-tof.service 2. Send request to read time of flight data with the following command: {device_name}:8099/remote_api?command=f&pid=4&quantity=4&clear=ba | 1. Json response should contain: { "errors": "none | none | none | none", "packets": [ "0xc0 ... 0xc0", "0xc0 ... 0xc0", "0xc0 ... 0xc0", "0xc0 ... 0xc0" ], "payloads": [ [payload one], [payload two], [payload three], [payload four] ] }, where the ellipsis represent a series of bytes. | Expected outcome verified. See Appendix 5. Verified by MS 28JUNE24 | P |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-484 |  |  |
