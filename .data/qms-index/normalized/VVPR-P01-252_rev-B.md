# VVPR-P01-252 Rev B: MX1 MedAI Diagnostic Tool WS-002 WS-005 and WS-006 v3.1.1 Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-252
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.1.1
- Source filename: VVPR-P01-252- MX1 MedAI Diagnostic Tool WS-002 WS-005 and WS-006 v3.1.1 Verification Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-252- MX1 MedAI Diagnostic Tool WS-002 WS-005 and WS-006 v3.1.1 Verification Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Tab 1
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 MedAI Diagnostic Tool (ODT) v3.1.1-alpha meets the requirements as stated in MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification Rev. K for WorkStations WS-002, WS-005 and WS-006.
OBJECTIVE AND SCOPE
Verify the software requirements established by MedAI for ODT v3.1.1-alpha release installed on the following WorkStations:
WS-002 - MS-10008 Collimator Verification
WS-005 - MS-10301 HMI Display Verification
WS-006 - MS-10141 PMUX and MS-10568 Wireless Transmitter PCBA Verification
Verify the software meets the requirements in ODT Software Requirements Specification, MEMO-P01-604 Rev K.
REFERENCES
MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification, Rev K
MEMO-P01-695 - ODT SW System Architecture Diagram, Rev C
MWI-219 Rev E- WS-002 Workstation Installation
MWI-220 Rev D- MS-10200 Collimator Verification
MWI-227 Rev E - WS-006 Workstation Installation
MWI-228 Rev D - MS-10141 Power Cleat Verification
MWI-271 Rev C - MS-10568 Wireless Charger Verification
MATERIALS
Power Cleat Assembly, MS-10141 Rev D
WTX_PCBA with Heat Sink MS-10568 Rev B
Collimator Bracket-Camera Assembly MS-10149 REV C
WS-002, MWI-219 - WS-002 Workstation Installation Rev D
WS-006, MWI-227 - WS-006 Workstation Installation Rev D
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification shall be performed at MedAI office building: 100 Main Street, Suite 700 Springfield, IL 60001 by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Setup
Set ODT to verbose mode
Experimental Procedure
Verify ODT v3.1.1-alpha changes do not impact requirements not included in Tables 1, 2 and 3. Include reference to code review in the report.
Execute remaining verification test steps defined in Tables 1, 2, and 3. Create and save screen captures as evidence for ODT behavior.
Record results in Evidence columns in Tables 1, 2, and 3, and reference applicable screen captures demonstrating ODT behavior. Include “Verified by <initials> <date>”.
Record Pass or Fail based on results and evidence.
Table 1: Collimator Verification WorkStation WS-002 - Requirements, Verification Steps, and Expected Results
The Manufacturing Work Instructions for WS-002, MWI-220 Rev D, MS-10200 Collimator Verification, should be used to guide operation of the WorkStation as needed to verifying ODT and MX1-Collimator-Test-Plugin.
Table 2. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-228, MS-10141 Power Cleat Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-PMUX-Test-Plugin.
Table 3. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-271, MS-10568 Wireless Charger Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-WTX-Test_Plugin.
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key: example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
Clerical errors were addressed as encountered.
Removed references to WS-005 related materials as WS-005 had no test performed.
MATERIALS
Power Cleat Assembly, MS-10141 Rev D
WTX_PCB with Heat Sink MS-10568 Rev B
Collimator Bracket-Camera Assembly MS-10149 Rev C
REFERENCES
WS-002, MWI-219 - WS-002 Workstation Installation Rev F
WS-002, MWI-220 - MS-10200 Collimator-Line Laser Verification Rev E
WS-006, MWI-227 - WS-006 Workstation Installation Rev E
WS-006, MWI-228 - MS-10141 Power Cleat Verification Rev D
WS-006, MWI-271 - MS-10568 Wireless Charger Verification Rev C
RESULTS
Unaffected requirements were confirmed in code review. See MEMO-P01-824.
Table 1. Collimator Verification WorkStation WS-002 - Requirements, Verification Steps, and Expected Results
The Manufacturing Work Instructions for WS-002, MWI-220 Rev E, MS-10200 Collimator Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-Collimator-Test-Plugin.
Table 2. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-228 Rev D, MS-10141 Power Cleat Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-PMUX-Test-Plugin.
Table 3. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-271 Rev C, MS-10568 Wireless Charger Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-WTX-Test_Plugin.
Figure 1. Max Wireless Power Test passing
Figure 2. Max Wireless Power Test failing with 5 attempts
Figure 3. WTX Load Test passing
Figure 4. WTX Load Test failing with 5 attempts
Figure 5. Collimator Aperture Position Test Passing
CONCLUSION
Overall Result:
Pass
Fail
Other:
ODT v3.1.1-alpha has been verified for use on WS-002, WS-005, and WS-006.
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-002, Collimator PCBA |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT4.16 | ODT shall PASS the aperture and timestamp test when 98.1% of the collimator orientation, size, and timestamp values are within the specified tolerance range, which corresponds to a 99% confidence that the aperture and timestamp behavior is within specification. | Run Collimator Aperture Position Test Verify ODT passes test when 16 or less values for each variable of the 800 measured values (Orientation, Size and Timestamp) are beyond specification. | ODT shall pass when 16 or less values for each of the parameters Orientation, Size and Timestamp of the 800 measured values are beyond specification. |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT8.25 | ODT shall command electronic load to step up power until wireless transmission stops and PASS when power measured ≥ 75 W while tolerating up to 5 dropouts | Run Max Wireless Power Test Verify ODT passes when power measured is ≥ 75 W | ODT Max Wireless Power Test PASS when power measured is ≥ 75 W |  |  |
|  |  | Disconnect the PMUX from the electronic load. Run Max Wireless Power Test Confirm 5 retries occur Verify ODT fails when power measured is < 75 W | ODT Max Wireless Power Test FAILS when power measured is less than 75 W |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT10.9 | ODT shall command electronic load to step up power until wireless transmission stops and PASS when power measured ≥ 75 W while tolerating up to 5 dropouts | Run WTX Load Test Verify ODT passes when power measured is ≥ 75 W | ODT WTX Load Test Test PASS when power measured is ≥ 75 W |  |  |
|  |  | Disconnect the PMUX from the electronic load. Run WTX Load Test Test Confirm 5 retries occur Verify ODT fails when power measured is < 75 W | ODT WTX Load Test Test FAILS when power measured is less than 75 W |  |  |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-671 |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-002, Collimator PCBA |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT4.16 | ODT shall PASS the aperture and timestamp test when 98.1% of the collimator orientation, size, and timestamp values are within the specified tolerance range, which corresponds to a 99% confidence that the aperture and timestamp behavior is within specification. | Run Collimator Aperture Position Test Verify ODT passes test when 16 or less values for each variable of the 800 measured values (Orientation, Size and Timestamp) are beyond specification. | ODT shall pass when 16 or less values for each of the parameters Orientation, Size and Timestamp of the 800 measured values are beyond specification. | See Figure 5. Verified by KJ 21JAN25 | PASS |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT8.25 | ODT shall command electronic load to step up power until wireless transmission stops and PASS when power measured ≥ 75 W while tolerating up to 5 dropouts | Run Max Wireless Power Test Verify ODT passes when power measured is ≥ 75 W | ODT Max Wireless Power Test PASS when power measured is ≥ 75 W | Expected outcome verified. See Figure 1. Verified by KJ 14JAN25 | PASS |
|  |  | Disconnect the PMUX from the electronic load. Run Max Wireless Power Test Confirm 5 retries occur Verify ODT fails when power measured is < 75 W | ODT Max Wireless Power Test FAILS when power measured is less than 75 W | Expected outcome verified. See Figure 2. Verified by KJ 14JAN25 | PASS |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT10.9 | ODT shall command electronic load to step up power until wireless transmission stops and PASS when power measured ≥ 75 W while tolerating up to 5 dropouts | Run WTX Load Test Verify ODT passes when power measured is ≥ 75 W | ODT WTX Load Test Test PASS when power measured is ≥ 75 W | Expected outcome verified. See Figure 3. Verified by KJ 14JAN25 | PASS |
|  |  | Disconnect the PMUX from the electronic load. Run WTX Load Test Test Confirm 5 retries occur Verify ODT fails when power measured is < 75 W | ODT WTX Load Test Test FAILS when power measured is less than 75 W | Expected outcome verified. See Figure 4. Verified by KJ 14JAN25 | PASS |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release, Corrections to Protocol listed in Report Section 1. | Refer to ECR-681 |  |  |
