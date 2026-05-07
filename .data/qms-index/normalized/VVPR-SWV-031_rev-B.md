# VVPR-SWV-031 Rev B: MB Burn In Test Fixture Script Verification and Validation Protocol

## Metadata
- Document ID: VVPR-SWV-031
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-031 - MB Burn In Test Fixture Script Verification and Validation Protocol_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-031 - MB Burn In Test Fixture Script Verification and Validation Protocol_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the MB-burnin-test-fixture-script meets usability and functional requirements as stated in MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify that the beta version of the verification script addresses the two flag requirements that failed in VVPR-SWV-025.
REFERENCES
MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification Rev A
MWI-260 - MS-10579 Monoblock Encapsulated Assembly Verification Rev A
MWI-259 - WS-015 Workstation Installation Rev A
VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report_B1
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.0.1-alpha
MS-10579 Monoblock Encapsulated Assembly Rev A
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-259 to install WS-015.
Follow the steps outlined in each table below. MWI-260 should be used to guide operation of the workstation as needed.
Items Retested
A partial verification of the requirements outlined in VVPR-SWV-025 is required as modifications incorporated into the v1.0.1-alpha version code are only called by the specified requirements.
Requirement SRS-5.4 requires verification as changes were made to the script to correct a logic statement in the error checking block for this requirement. The function was modified to add an additional condition to the failure state. The failure for a 10% difference between positive and negative tube voltages now only triggers if the total tube voltage is about 40kV. This was modified to prevent failing good monoblocks for exhibiting asymmetric waveforms at low kV.
The code modifications only impacted the logic block where this requirement is checked. No other requirements are impacted.
Table 2: Operation Requirements.
Table 5: Flagging Requirements.
Data Analysis
All of the verification tests in Tables 1 through 5 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 through 5 per the expected results documented in the “Expected Result/Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
Deviations
Clerical update to materials: Benchtop power supply, Oscilloscope, and Arbitrary Function Generator were not required for the listed verification tests.
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.0.1-alpha
MS-10579 Monoblock Encapsulated Assembly Rev A
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-259 to install WS-015.
Follow the steps outlined in each table below. MWI-260 should be used to guide operation of the workstation as needed.
Table 2: Operation Requirements.
Table 5: Flagging Requirements.
Discussion
No issues were discovered.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
DOCUMENT REVISION HISTORY
Appendix 1
vIn increased after 3 shots in a row below example burn in kV
vIn decreased by .1 V after 3 shots in a row above example burn in kV
Appendix 2

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.8 | The MB-burnin-test-fixture-script shall generate a model to predict duty cycle and H-bridge frequency to achieve 83 kV, 2.1 mA | Wait until the log outputs "----- Beginning Burn In -----" | 1. Log outputs "++++ Richardson's Curve for 83 has been found ++++" 2. Log outputs "++++ Predicted Duty Cycle and FreqN ++++" 3. Log outputs "filDuty = {value1}" and "freqN = {value2}" |  |  |
| SRS-2.9 | The MB-burnin-test-fixture-script shall approach 83 kV (±1%), 2.1 mA (±5%) without overshooting kV by more than 2 kV | Wait until the log outputs "++++ Initial Inputs Used For Burn In ++++" | 1. In the exposure preceding "++++ Initial Inputs Used For Burn In ++++": 1.1. kV is within 1% of 83 1.2. mA is within 5% of 2.1 2. The kV for all of the other exposures for that duty cycle do not exceed 85 kV |  |  |
| SRS-2.10 | The MB-burnin-test-fixture-script shall repeatedly stress the monoblock for 3600 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 88 kV, in which case vIn will drop by .2 V |  |  |
| SRS-2.11 | When stressing the monoblock, the MB-burnin-test-fixture-script shall increase vIn when the kV for 3 exposures in a row fall below 1.25 kV lower than the desired stress kV | Force three exposures in a row to fall below 1.25 less than the desired stress kV | vIn increases by .1 V |  |  |
| SRS-2.12 | When stressing the monoblock, the MB-burnin-test-fixture-script shall decrease vIn when the kV for 3 exposures in a row fall below 1.25 kV lower than the desired stress kV | Force three exposures in a row to frise above 1.25 more than the desired stress kV | vIn decreases by .1 V |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.4 | The MB-burnin-test-fixture-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% above 40k | Run the script and generate a Vsn and Vsp that differ by more than 10% at 40k or above | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-571 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.8 | The MB-burnin-test-fixture-script shall generate a model to predict duty cycle and H-bridge frequency to achieve 83 kV, 2.1 mA | Wait until the log outputs "----- Beginning Burn In -----" | 1. Log outputs "++++ Richardson's Curve for 83 has been found ++++" 2. Log outputs "++++ Predicted Duty Cycle and FreqN ++++" 3. Log outputs "filDuty = {value1}" and "freqN = {value2}" | Expected outcome verified See Appendix 1.1 Verified by MI 23SEP2024 | P |
| SRS-2.9 | The MB-burnin-test-fixture-script shall approach 83 kV (±1%), 2.1 mA (±5%) without overshooting kV by more than 2 kV | Wait until the log outputs "++++ Initial Inputs Used For Burn In ++++" | 1. In the exposure preceding "++++ Initial Inputs Used For Burn In ++++": 1.1. kV is within 1% of 83 1.2. mA is within 5% of 2.1 2. The kV for all of the other exposures for that duty cycle do not exceed 85 kV | Expected outcome verified See Appendix 1.2 Verified by MI 23SEP2024 | P |
| SRS-2.10 | The MB-burnin-test-fixture-script shall repeatedly stress the monoblock for 3600 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 88 kV, in which case vIn will drop by .2 V | Expected outcome verified See Appendix 1.3 Verified by MI 23SEP2024 | P |
| SRS-2.11 | When stressing the monoblock, the MB-burnin-test-fixture-script shall increase vIn when the kV for 3 exposures in a row fall below 1.25 kV lower than the desired stress kV | Force three exposures in a row to fall below 1.25 less than the desired stress kV | vIn increases by .1 V | Expected outcome verified See Appendix 1.4 Verified by MI 23SEP2024 | P |
| SRS-2.12 | When stressing the monoblock, the MB-burnin-test-fixture-script shall decrease vIn when the kV for 3 exposures in a row fall below 1.25 kV lower than the desired stress kV | Force three exposures in a row to frise above 1.25 more than the desired stress kV | vIn decreases by .1 V | Expected outcome verified See Appendix 1.4 Verified by MI 23SEP2024 | P |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.4 | The MB-burnin-test-fixture-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% above 40k | Run the script and generate a Vsn and Vsp that differ by more than 10% at 40k or above | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." | Expected outcome verified See Appendix 2 Verified by MI 23SEP2024 | P |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report Release; removal of equipment not used in this test report | Refer to ECR-576 |  |  |
