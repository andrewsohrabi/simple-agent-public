# VVPR-SWV-019 Rev B: Cassette OLED Tester Verification and Validation v1.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-SWV-019
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.0.0
- Source filename: VVPR-SWV-019 - Cassette OLED Tester Verification and Validation v1.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-019 - Cassette OLED Tester Verification and Validation v1.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify that the software oled-tester allows the user to check the functionality of the Cassette OLED display.
OBJECTIVE
The objective of this study is to verify that the v1.0.0 release of the OLED Tester meets system-level requirements set by MedAI as documented in MEMO-P01-705.
REFERENCES
MEMO-P01-705 - MX1 OLED Tester Software Requirements Specification Rev. A
MATERIALS
MX1 C1 Cassette Rev. I
MX1 SS Version v3.1.0
S10094 OLED Tester v1.0.0
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
MX1 C1 Cassette Rev. I, SN: DV-26
MX1 SS Version v3.1.0
S10094 OLED Tester v1.0.0
RESULTS
Table 1: Requirements, Verification Steps, and Expected Response.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
APPENDIX/ATTACHMENTS
Appendix 1: OLED Display Fully On
Appendix 2: OLED Display Fully Off
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: C1 Cassette |  |  |  |  |  |
| Precondition: The Cassette is powered on and a remote connection has been established via SSH. |  |  |  |  |  |
|  | Test Case: OLED Tester |  |  |  |  |
| SRS-10.1 | The OLED tester shall enable all pixels of the MX1 Cassette OLED display | 1. Run the command systemctl –user stop oled-overseer.service 2. Run the OLED tester program with the command: /opt/medai/bin/oled-tester | Cassette OLED display lights up fully |  |  |
| SRS-10.2 | The OLED tester shall disable all pixels of the MX1 Cassette OLED display | 2. Press any key to end the program | Cassette OLED display turns off fully |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 28 Jun 2024 | 24-396 |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: Device Components Needed: C1 Cassette |  |  |  |  |  |
| Precondition: The Cassette is powered on and a remote connection has been established via SSH. |  |  |  |  |  |
|  | Test Case: OLED Tester |  |  |  |  |
| SRS-10.1 | The OLED tester shall enable all pixels of the MX1 Cassette OLED display | 1. Run the command systemctl –user stop oled-overseer.service 2. Run the OLED tester program with the command: /opt/medai/bin/oled-tester | Cassette OLED display lights up fully | Expected outcome verified. See Appendix 1. Verified by MS 28JUNE24 | P |
| SRS-10.2 | The OLED tester shall disable all pixels of the MX1 Cassette OLED display | 2. Press any key to end the program | Cassette OLED display turns off fully | Expected outcome verified. See Appendix 2. Verified by MS 28JUNE24 | P |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-484 |  |  |
