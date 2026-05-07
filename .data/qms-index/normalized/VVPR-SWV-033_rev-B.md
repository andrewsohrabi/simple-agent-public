# VVPR-SWV-033 Rev B: Foot Pedal UUID Reader Firmware Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-033
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-033 - Foot Pedal UUID Reader Firmware Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-033 - Foot Pedal UUID Reader Firmware Verification and Validation Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the Foot Pedal UUID Reader meets usability and functional requirements as stated in MEMO-P01-804 - Foot Pedal UUID Reader Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify the Non-Product Software Tool: Foot Pedal UUID Reader
REFERENCES
MEMO-P01-804 - Foot Pedal UUID Reader Software Requirements Specification Rev. A
MATERIALS
S10108 - Footpedal UUID Reader Firmware - v1.0.0-alpha (a0a1b76fe4d229a995c9de23df1205b96c8cafc5)
ES-10007 Rev B - Footpedal PCBA with v4.1.0-alpha firmware
T-200 Rev A -  Foot Pedal UUID Reader
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
Report Section
Recorded By:SUNGWON PARKDate: 12/16/24
PROTOCOL DEVIATIONS
None.
DEVICES, COMPONENTS, OR EQUIPMENT USED
T-200 - Rev A
ES-10007 Rev B - Footpedal PCBA with S10006 v4.1.0-alpha & UUID: 4784185 (previously identified)
S10108 - Footpedal UUID Reader Firmware - v1.0.0-alpha (a0a1b76fe4d229a995c9de23df1205b96c8cafc5)
RESULT
DISCUSSION
This protocol successfully verified a method of obtaining the UUID of a Foot Pedal PCBA that will be used during F1 production. The Foot Pedal Unit Under Test’s UUID was determined prior to executing this protocol using an alternative method of reading the UUID. A debugger was used to read the memory location that contains the UUID: this method is the “gold standard” for obtaining a UUID with no errors. This protocol demonstrated that the production method is equivalent to the”gold standard” method.
CONCLUSION
Overall Result:
Pass
Fail
Other: _______
The production method to obtain a Foot Pedal’s UUID was successfully verified.
LIST OF APPENDICES
Appendix 1: T-200 LCD display after power switch turned to ON position
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: Materials as listed Above |  |  |  |  |
| Precondition: | Power on T-200 and place Foot Pedal PCBA on bed of nails |  |  |  |  |
| SRS-1.1 | The Foot Pedal UUID Reader shall read the UUID from a Foot Pedal PCBA and display it on a screen | 1. Flip switch on the side of T-200 to the ON position 2. Observe the screen for resulting behavior | An integer UUID is displayed on the screen |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering | Refer to ECR-649 |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: Materials as listed Above |  |  |  |  |
| Precondition: | Power on T-200 and place Footpedal PCBA on bed of nails |  |  |  |  |
| SRS-1.1 | The Foot Pedal UUID Reader shall read the UUID from a Foot Pedal PCBA and display it on a screen | 1. Flip power switch on the side of T-200 to the ON position. 2. Observe the screen for resulting behavior. | An integer UUID is displayed on the screen that matches the known UUID of the Footpedal PCBA. | Expected results observed. See Appendix 1. Verified by SP 16DEC24 | PASS |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-666 |  |  |
