# VVPR-SWV-036 Rev B: MB Burn In Test Fixture Script Verification and Validation Protocol v1.2.0-beta

## Metadata
- Document ID: VVPR-SWV-036
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.2.0
- Source filename: VVPR-SWV-036 - MB Burn In Test Fixture Script Verification and Validation Protocol v1.2.0-beta_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-036 - MB Burn In Test Fixture Script Verification and Validation Protocol v1.2.0-beta_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the MB-burnin-test-fixture-script meets usability and functional requirements as stated in MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify MB-burnin-test-fixture-script version 1.2.0 addresses the updated requirements in MEMO-P01-735.
The MB-burnin-test-fixture-script script will be evaluated to confirm all new requirements in both the For Human Use (FHU) and Industrial (IND) Monoblock specifications documented in MEMO-P01-735 and MEMO-P01-790, respectively, as these scripts are identical with the exception of the four values modified for IND use.
REFERENCES
MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification Rev H
MEMO-P01-790 MB-burnin-test-fixture-IND Software Requirements Specification Rev C
MEMO-P01-780 - WS-015 Script Updates For IND Monoblock Rev A
MWI-260 - MS-10579 Monoblock Encapsulated Assembly Verification Rev C
MWI-259 - WS-015 Workstation Installation Rev C
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.2.0-beta
MS-10579 Monoblock Encapsulated Assembly Rev D or equivalent
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-259 to install WS-015.
Follow the steps outlined in each table below. MWI-260 should be used to guide operation of the workstation as needed. Note, all new requirements will be evaluated.
Table 1: Flagging Requirements.
Data Analysis
All of the verification tests in Tables 1 and 2 shall be treated as attributes and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 and 2 per the expected results documented in the “Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
DEVIATIONS
None
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.2.0-beta
MS-10007 Rev B, LN PO22063
RESULTS
Table 1: Flagging Requirements.
DISCUSSION
No issues were discovered.
CONCLUSION
Overall Result:
Pass
Fail
Other:
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Appendix 1
Test continued until number of shots was reached and then failed the monoblock:
Monoblock failed successfully:
No more exposures were taken and the monoblock failed:

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The MB-burnin-test-fixture-script shall fail a monoblock if 10 arcs are detected | 1. Open the script folder with visual studio code and manually change self.burnIn_shots on line 108 so that only 3 shots are taken. 2. Place a breakpoint on line 446 of characterizeMB.py 3. Run main.py in debug mode and continue the test from the last milestone if applicable. 4. Once the breakpoint is reached, place another breakpoint at line 845 of characterizeMB.py and continue the program. 5. Once the breakpoint on line 845 is reached, manually set self.numArcs equal to any number from 10 to 49 6. Continue the program and wait until the test completes. | The test continues after self.numArcs has been set. When the test completes, the log outputs "Monoblock has arced <insert value here> times. FAIL MONOBLOCK." |  |  |
| SRS-5.8 | The MB-burnin-test-fixture-script shall fail a monoblock if the dose after stressing the monoblock has changed by more than 10% | 1. Open the script folder with visual studio code and manually change self.burnIn_shots on line 108 so that only 3 shots are taken. 2. Place a breakpoint on line 510 of characterizeMB.py 3. Run main.py in debug mode and continue the test from the last milestone if applicable. 5. Once the breakpoint is reached, manually set dose_end such that the percent difference between it and dose_init is greater than or equal to 10 6. Continue the program and wait until the test completes. | Log outputs "Final dose ({final dose value} mGy) has drifted more than 10% from initial ({initial dose value} mGy). FAIL MONOBLOCK." |  |  |
| SRS-5.9 | The MB-burnin-test-fixture-script shall abort the test if 50 or more arcs are detected | 1. Open the script folder with visual studio code and place a breakpoint on line 446 of characterizeMB.py 3. Run main.py in debug mode and continue the test from the last milestone if applicable. 4. Once the breakpoint is reached, place another breakpoint at line 845 of characterizeMB.py and continue the program. 5. Once the breakpoint on line 845 is reached, manually set self.numArcs equal to 50 6. Continue the program and wait until the test completes. | No more exposures are taken The test completes and the log outputs "Monoblock has arced 50 times. FAIL MONOBLOCK." |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-686 |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The MB-burnin-test-fixture-script shall fail a monoblock if 10 arcs are detected | 1. Open the script folder with visual studio code and manually change self.burnIn_shots on line 108 so that only 3 shots are taken. 2. Place a breakpoint on line 446 of characterizeMB.py 3. Run main.py in debug mode and continue the test from the last milestone if applicable. 4. Once the breakpoint is reached, place another breakpoint at line 845 of characterizeMB.py and continue the program. 5. Once the breakpoint on line 845 is reached, manually set self.numArcs equal to any number from 10 to 49 6. Continue the program and wait until the test completes. | The test continues after self.numArcs has been set. When the test completes, the log outputs "Monoblock has arced <insert value here> times. FAIL MONOBLOCK." | Expected outcome verified See Appendix 1.1 Verified by MI 28JAN2025 | P |
| SRS-5.8 | The MB-burnin-test-fixture-script shall fail a monoblock if the dose after stressing the monoblock has changed by more than 10% | 1. Open the script folder with visual studio code and manually change self.burnIn_shots on line 108 so that only 3 shots are taken. 2. Place a breakpoint on line 510 of characterizeMB.py 3. Run main.py in debug mode and continue the test from the last milestone if applicable. 5. Once the breakpoint is reached, manually set dose_end such that the percent difference between it and dose_init is greater than or equal to 10 6. Continue the program and wait until the test completes. | Log outputs "Final dose ({final dose value} mGy) has drifted more than 10% from initial ({initial dose value} mGy). FAIL MONOBLOCK." | Expected outcome verified See Appendix 1.2 Verified by MI 28JAN2025 | P |
| SRS-5.9 | The MB-burnin-test-fixture-script shall abort the test if 50 or more arcs are detected | 1. Open the script folder with visual studio code and place a breakpoint on line 446 of characterizeMB.py 3. Run main.py in debug mode and continue the test from the last milestone if applicable. 4. Once the breakpoint is reached, place another breakpoint at line 845 of characterizeMB.py and continue the program. 5. Once the breakpoint on line 845 is reached, manually set self.numArcs equal to 50 6. Continue the program and wait until the test completes. | No more exposures are taken The test completes and the log outputs "Monoblock has arced 50 times. FAIL MONOBLOCK." | Expected outcome verified See Appendix 1.2 Verified by MI 28JAN2025 | P |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report Release | See ECR-693 |  |  |
