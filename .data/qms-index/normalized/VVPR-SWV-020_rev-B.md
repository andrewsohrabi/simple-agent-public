# VVPR-SWV-020 Rev B: IRay License Checker Verification and Validation v1.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-SWV-020
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.0.0
- Source filename: VVPR-SWV-020 - IRay License Checker Verification and Validation v1.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-020 - IRay License Checker Verification and Validation v1.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 IRay License Checker meets usability and functional requirements as stated in MEMO-P01-707 - IRay License Checker Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify the Non-Product Software Tool: MX1 Iray License Checker for use in production of the MX1 C1 Cassette.
REFERENCES
MEMO-P01-707 - MX1 Iray License Checker Software Requirements Specification Rev. A
MATERIALS
MX1 C1 Cassette Rev. I
MX1 SS Version v3.1.0
S10093 Iray License Checker v1.0.0
M50004 Detector Rev A (Licensed)
M50004 Detector Rev A (Unlicensed)
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
C1 Cassette Rev. I, SN: EV-22
MX1 SS Version v3.1.0
S10093 Iray License Checker v1.0.0
M50004 Detector Rev A, SN: MK590010K0422220006 (Licensed)
M50004 Detector Rev A, SN: MK591101T0811229005 (Unlicensed)
RESULTS
Table 1: Requirements, Verification Steps, and Expected Response
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
APPENDIX/ATTACHMENTS
Appendix 1: Licensed Detector’s Connection Status Reported Successfully
Appendix 2: Licensed Detector’s License Status Reported Successfully
Appendix 3: Unlicensed Detector’s Connection Status Reported Successfully
Appendix 4: Unlicensed Detector’s License Status Reported Successfully
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: C1 Cassette |  |  |  |  |
| Precondition: | The Cassette is powered on and a remote connection has been established via SSH. |  |  |  |  |
| N/A | N/A | 1. Run the command systemctl --user stop cassette-orchestrator.service 2. Run the command systemctl --user start rest-icd.service 3. Run the command curl 'localhost:8090/remote_api?command=p&pid=5&op=1&reg=e0&payload=0601' 4. Wait at least 15 seconds before proceeding | Verify the detector has been powered on by verifying that "payload":"0x06 0x01" is in the output of step 3 | N/A - This step must be performed to conduct this protocol | N/A |
| SRS-1.1 | The Iray License Checker shall indicate the connection status of the detector to the user via HTTP request | 1. Run the command systemctl --user start iray-license-check.service and wait at least 15 seconds 2. Run the command curl 'localhost:8040/status' | Verify the output of step 2 is 'Detector Connected' |  |  |
| SRS-1.2 | The Iray License Checker shall indicate the license status of the detector to the user via HTTP request | 1. Run the command curl 'localhost:8040/check_license' | Verify the output of step 1 contains the field: "licensed":"true" |  |  |
| SRS-1.2 | The Iray License Checker shall indicate the license status of the detector to the user via HTTP request | 2. Power off the Cassette and swap the currently installed Detector for the unlicensed one from inventory. Power on, connect via SSH, and repeat all previous steps | Verify the output of step 2 contains the field “licensed”:”false” |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 28 Jun 2024 | 24-400 |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: C1 Cassette |  |  |  |  |
| Precondition: | The Cassette is powered on and a remote connection has been established via SSH. |  |  |  |  |
| N/A | N/A | 1. Run the command systemctl --user stop cassette-orchestrator.service 2. Run the command systemctl --user start rest-icd.service 3. Run the command curl 'localhost:8090/remote_api?command=p&pid=5&op=1&reg=e0&payload=0601' 4. Wait at least 15 seconds before proceeding | Verify the detector has been powered on by verifying that "payload":"0x06 0x01" is in the output of step 3 | N/A - This step must be performed to conduct this protocol | N/A |
| SRS-1.1 | The Iray License Checker shall indicate the connection status of the detector to the user via HTTP request | 1. Run the command systemctl --user start iray-license-check.service and wait at least 15 seconds 2. Run the command curl 'localhost:8040/status' | Verify the output of step 2 is 'Detector Connected' | Expected outcome verified. See Appendix 1. Verified by MS 01JULY24 | P |
| SRS-1.2 | The Iray License Checker shall indicate the license status of the detector to the user via HTTP request | 1. Run the command curl 'localhost:8040/check_license' | Verify the output of step 1 contains the field: "licensed":"true" | Expected outcome verified. See Appendix 2. Verified by MS 01JULY24 | P |
| SRS-1.2 | The Iray License Checker shall indicate the license status of the detector to the user via HTTP request | 2. Power off the Cassette and swap the currently installed Detector for the unlicensed one from inventory. Power on, connect via SSH, and repeat all previous steps | Verify the output of step 2 contains the field “licensed”:”false” | Expected outcome verified. See Appendices 3 - 4. Verified by MS 01JULY24 | P |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-484 |  |  |
