# VVPR-P01-271 Rev B: MX1 MedAI Diagnostic Tool WS-002 and WS-006 v3.1.2-alpha Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-271
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.1.2
- Source filename: VVPR-P01-271- MX1 MedAI Diagnostic Tool WS-002 and WS-006 v3.1.2-alpha Verification Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-271- MX1 MedAI Diagnostic Tool WS-002 and WS-006 v3.1.2-alpha Verification Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Tab 1
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 MedAI Diagnostic Tool (ODT) v3.1.2-alpha meets the requirements as stated in MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification Rev. M for WorkStations WS-002 and WS-006.
OBJECTIVE AND SCOPE
Verify the software requirements established by MedAI for ODT v3.1.2-alpha release installed on the following WorkStations:
WS-002 - MS-10008 Collimator Verification
WS-006 - MS-10141 PMUX and MS-10568 Wireless Transmitter PCBA Verification
Verify the software meets the requirements in ODT Software Requirements Specification, MEMO-P01-604 Rev M.
REFERENCES
MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification, Rev M
MEMO-P01-695 - ODT SW System Architecture Diagram, Rev C
MWI-219 Rev F- WS-002 Workstation Installation
MWI-220 Rev E- MS-10200 Collimator Verification
MWI-227 Rev E - WS-006 Workstation Installation
MWI-228 Rev E - MS-10141 Power Cleat Verification
MWI-271 Rev D - MS-10568 Wireless Charger Verification
MATERIALS
MS-10200 Rev C - Collimator - Line Laser Assy
Power Cleat Assembly, MS-10141 Rev C
WTX_PCBA with Heat Sink MS-10568 Rev B
WS-002, MWI-219 - WS-002 Workstation Installation Rev F
WS-006, MWI-227 - WS-006 Workstation Installation Rev E
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification shall be performed at MedAI office building: 100 Main Street, Suite 700 Springfield, IL 60001 by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Setup
Set ODT to verbose mode
Experimental Procedure
Verify ODT v3.1.2-alpha changes do not impact requirements not included in Tables 1, 2 and 3. Include reference to code review in the report.
Execute remaining verification test steps defined in Tables 1, 2, and 3. Create and save screen captures as evidence for ODT behavior.
Record results in Evidence columns in Tables 1, 2, and 3, and reference applicable screen captures demonstrating ODT behavior. Include “Verified by <initials> <date>”.
Record Pass or Fail based on results and evidence.
Table 1. Collimator Verification WorkStation WS-002 - Requirements, Verification Steps, and Expected Results
The Manufacturing Work Instructions for WS-002, MWI-220 Rev E, MS-10200 Collimator Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-Collimator-Test-Plugin.
Table 2. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-228, MS-10141 Power Cleat Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-PMUX-Test-Plugin.
Table 3. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-271, MS-10568 Wireless Charger Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-WTX-Test_Plugin.
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
The Wireless RX test was originally referred to as WRX test; this was corrected.
The wrong revision for the WS-002 MWI was referenced as D, not E
The wrong collimator assembly was referenced, this was corrected
REFERENCES
MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification, Rev M
MEMO-P01-695 - ODT SW System Architecture Diagram, Rev C
MWI-219 Rev F- WS-002 Workstation Installation
MWI-220 Rev E- MS-10200 Collimator Verification
MWI-227 Rev E - WS-006 Workstation Installation
MWI-228 Rev E - MS-10141 Power Cleat Verification
MWI-271 Rev D- MS-10568 Wireless Charger Verification
MATERIALS
Collimator-Line Laser Assy, MS-10200 Rev C
Power Cleat Assembly, MS-10141 Rev C
WTX_PCBA with Heat Sink MS-10568 Rev B
WS-002, MWI-219 - WS-002 Workstation Installation Rev F
WS-006, MWI-227 - WS-006 Workstation Installation Rev E
RESULTS
Unaffected requirements were confirmed in code review. See MEMO-P01-832.
Affected requirements that are still met were confirmed in code review. See MEMO-P01-832.
Three existing requirements had limits adjusted and were confirmed in code review. See MEMO-P01-832.
One new requirement was confirmed in code review. See MEMO-P01-832.
Table 1. Collimator Verification WorkStation WS-002 - Requirements, Verification Steps, and Expected Results
The Manufacturing Work Instructions for WS-002, MWI-220 Rev E, MS-10200 Collimator Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-Collimator-Test-Plugin.
Table 2. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-228, MS-10141 Power Cleat Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-PMUX-Test-Plugin.
Table 3. Power Cleat and Wireless Charger Verification WS-006 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-006, MWI-271 Rev D , MS-10568 Wireless Charger Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-WTX-Test_Plugin.
Figure 1. ODT passing Collimator Aperture Position Test
Figure 2. ODT failing Collimator Aperture Position Test
Figure 3. ODT passing Collimator Aperture Position Test
Figure 4. ODT failing Collimator Aperture Position Test
Figure 5. ODT passing Collimator Aperture Position Test after failing Collimator Aperture Position Test
Figure 6. ODT Passing Wireless RX Test after 4 failed transmission attempts
Figure 7. ODT failing Wireless RX Test after 5 failed transmission attempts
Figure 8. ODT Passing WTX Resonance Test - Transmitting after 4 failed transmission attempts
Figure 9. ODT failing  WTX Resonance Test - Transmitting after 5 failed transmission attempts
CONCLUSION
Overall Result:
Pass
Fail
Other:
ODT v3.1.2-alpha has been verified for use on WS-002 and WS-006.
Attachments
Attachment 1 - Collimator Log 1 (Collimator Log for 4.11.1 pass)
Attachment 2 - Collimator Log 2 (Collimator Log for 4.11.2 failure)
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-002, Collimator PCBA, faulty Collimator PCBA |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT4.11 | ODT shall issue 800 write commands with randomly generated aperture settings (orientation and size) to a connected Collimator and will read back orientation and size values until the Collimator Status bit equals 0x20 at which point ODT shall record the actual orientation and size values, and the MCU timestamp which indicates how long it took the Collimator to reach both positions | 1. Run Collimator Aperture Position Test 2. Observe the debug window3. Verify Production records number of failures accurately | Confirm the collimator size and orientation change Confirm ODT passes if there are less than 16 failures Confirm an accurate count of failures in the product log. |  |  |
|  |  | 1. Run Collimator Aperture Position Test 2. Observe the debug window 3. Verify Production records number of failures accurately | Confirm ODT fails if there are more than 16 failures |  |  |
| ODT4.16 | ODT shall PASS the aperture and timestamp test when 98.1% of the collimator orientation, size, and timestamp values are within the specified tolerance range, which corresponds to a 99% confidence that the aperture and timestamp behavior is within specification. | 1. Run Collimator Aperture Position Test 2. Verify ODT passes test when 16 or less values for each variable of the 800 measured values (Orientation, Size and Timestamp) are beyond specification. | Confirm ODT passes if there are less than 16 failures |  |  |
|  |  | 1. Run Collimator Aperture Position Test on a defective MS-10200 assembly 2. Verify ODT fails test when more than16 values for the variables Orientation, Size and Timestamp of the 800 measured values are beyond specification. | Confirm ODT fails if there are more than 16 failures |  |  |
| ODT4.17 | ODT shall reset error count on the aperture and timestamp test between runs | After verifying ODT4.16 and observing a failure, run the test again on a valid collimator | ODT shall pass the collimator aperture position test after a failure |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT8.31 | ODT shall check for stability of WTX transmission at start of transmission and retry up to 5 times if not successful | Run the Wireless RXTest without placing the WTX in proximity of the PMUX. After the debug window says "Transmission never started: Attempt 4”, slide the WTX in proximity of the PMUX and observe the result. | Confirm ODT prints “Transmission never started: Attempt 1/2/3/4” Confirm ODT passes the test after sliding the WTX in proximity of the PMUX. |  |  |
|  |  | Run the Wireless RXTest without placing the WTX in proximity of the PMUX. | ODT shall abort the test after 5 failed attempts and prints “Transmission never started: Attempt 1/2/3/4/5” |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT10.14 | ODT shall check for stability of WTX transmission at start of transmission and retry up to 5 times if not successful | Run the WTXResonance Test - Transmitting without placing the WTX in proximity of the PMUX. After the debug window says "Transmission never started: Attempt 4”, slide the WTX in proximity of the PMUX and observe the result. | Confirm ODT prints “Transmission never started: Attempt 1/2/3/4” Confirm ODT passes the test after sliding the WTX in proximity of the PMUX. |  |  |
|  |  | Run the WTXResonance Test - Transmitting without placing the WTX in proximity of the PMUX. | ODT shall abort the test after 5 failed attempts and prints “Transmission never started: Attempt 1/2/3/4/5” |  |  |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Engineering Regulatory Affairs | 03 Feb 2025 | 25-067 |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-002, Collimator PCBA, faulty Collimator PCBA |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT4.11 | ODT shall issue 800 write commands with randomly generated aperture settings (orientation and size) to a connected Collimator and will read back orientation and size values until the Collimator Status bit equals 0x20 at which point ODT shall record the actual orientation and size values, and the MCU timestamp which indicates how long it took the Collimator to reach both positions | 1. Run Collimator Aperture Position Test 2. Observe the debug window3. Verify Production records number of failures accurately | Confirm the collimator size and orientation change Confirm ODT passes if there are less than 16 failures Confirm an accurate count of failures in the product log. | Collimator size and orientation change was observed. Less than 16 failures of each type occurred. See Attachment 1. See Figure 1. Verified by KJ 03FEB25 | PASS |
|  |  | 1. Run Collimator Aperture Position Test 2. Observe the debug window 3. Verify Production records number of failures accurately | Confirm ODT fails if there are more than 16 failures | More than 16 failures occurred. See Attachment 2. See Figure 2. Verified by KJ 03FEB25 | PASS |
| ODT4.16 | ODT shall PASS the aperture and timestamp test when 98.1% of the collimator orientation, size, and timestamp values are within the specified tolerance range, which corresponds to a 99% confidence that the aperture and timestamp behavior is within specification. | 1. Run Collimator Aperture Position Test 2. Verify ODT passes test when 16 or less values for each variable of the 800 measured values (Orientation, Size and Timestamp) are beyond specification. | Confirm ODT passes if there are less than 16 failures | Orientation, Size and Timestamp failures are each less than 16. See Figure 3. Verified by KJ 03FEB25 | PASS |
|  |  | 1. Run Collimator Aperture Position Test on a defective MS-10200 assembly 2. Verify ODT fails test when more than16 values for the variables Orientation, Size and Timestamp of the 800 measured values are beyond specification. | Confirm ODT fails if there are more than 16 failures | Orientation, Size and Timestamp failures are each greater than 16. See Figure 4. Verified by KJ 03FEB25 | PASS |
| ODT4.17 | ODT shall reset error count on the aperture and timestamp test between runs | After verifying ODT4.16 and observing a failure, run the test again on a valid collimator | ODT shall pass the collimator aperture position test after a failure | Expected behavior observed. See Figure 5. Verified by KJ 03FEB25 | PASS |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT8.31 | ODT shall check for stability of WTX transmission at start of transmission and retry up to 5 times if not successful | Run the WRXTest without placing the WTX in proximity of the PMUX. After the debug window says "Transmission never started: Attempt 4”, slide the WTX in proximity of the PMUX and observe the result. | Confirm ODT prints “Transmission never started: Attempt 1/2/3/4” Confirm ODT passes the test after sliding the WTX in proximity of the PMUX. | Expected behavior observed. See Figure 6. Verified by KJ 03FEB25 | PASS |
|  |  | Run the WRXTest without placing the WTX in proximity of the PMUX. | ODT shall abort the test after 5 failed attempts and prints “Transmission never started: Attempt 1/2/3/4/5” | Expected behavior observed. See Figure 7. Verified by KJ 03FEB25 | PASS |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-006 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT10.14 | ODT shall check for stability of WTX transmission at start of transmission and retry up to 5 times if not successful | Run the WTXResonance Test - Transmitting without placing the WTX in proximity of the PMUX. After the debug window says "Transmission never started: Attempt 4”, slide the WTX in proximity of the PMUX and observe the result. | Confirm ODT prints “Transmission never started: Attempt 1/2/3/4” Confirm ODT passes the test after sliding the WTX in proximity of the PMUX. | Expected behavior observed. See Figure 8. Verified by KJ 03FEB25 | PASS |
|  |  | Run the WTXResonance Test - Transmitting without placing the WTX in proximity of the PMUX. | ODT shall abort the test after 5 failed attempts and prints “Transmission never started: Attempt 1/2/3/4/5” | Expected behavior observed. See Figure 9. Verified by KJ 03FEB25 | PASS |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release, Corrections to Protocol listed in Report Section 1. | Refer to ECR-685 |  |  |
