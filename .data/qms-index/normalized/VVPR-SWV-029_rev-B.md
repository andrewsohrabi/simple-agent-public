# VVPR-SWV-029 Rev B: MB Burn In Test Fixture Script Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-029
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-029 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-029 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report_B.docx
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
S10102 MB-burnin-test-fixture-script v1.0.0-beta
MS-10579 Monoblock Encapsulated Assembly Rev A
SPD3303X Siglent Benchtop power supply (EQP-246 or equivalent)
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121 or equivalent)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238 or equivalent)
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
A partial verification of the requirements outlined in VVPR-SWV-025 are required as modifications incorporated into the -beta version code are only called by the specified requirements.
Requirement SRS-5.1 requires verification as changes were made to the script to correct a logic statement in the error checking block for this requirement. The function was modified to check when the number of detected arcs is greater than or equal to 3 rather than just greater than 3. In the latter logic, the script would fail once 4 arcs are detected rather than the desired 3.
The code modifications only impacted the logic block where this requirement is checked. No other requirements are impacted.
Requirement SRS-5.5 requires verification as changes were made to the script to correct how frequency peaks are found. The function was modified to only consider a kV as having passed its peak if it is less than 99.7% of the previous kV. This prevents the script from falsely identifying a peak due to noise differences between exposures.
The code modifications impact how the frequency peak is found which also affects SRS-2.5. Therefore, SRS-2.5 also needs to be retested.
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
SRS-2.5 pass criteria says the error text is “Resonant frequency not within bounds.” when it is actually “Script failed to find peak” The text “Resonant frequency not within bounds” is the error text that WS-017 displays for the same type of error. This can be found in the SRS for WS-017 (MEMO-P01-738).
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.0.0-beta
MS-10579 Monoblock Encapsulated Assembly Rev A
Tulsa mb5
Tulsa mb13
SPD3303X Siglent Benchtop power supply (EQP-246, calibration exp 7/29/25)
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121, calibration exp 10/31/24)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238, calibration exp 6/11/25)
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Table 2: Operation Requirements.
Table 5: Flagging Requirements.
DISCUSSION
SRS-2.2 passed in VVPR-SWV-025, and was tested again to ensure changes made in the beta release did not impact peak finding functionality.
Overvoltage failures  in SRS-5.2 and resonant frequency failures in SRS-5.5 were both able to be triggered when forced into error conditions.
As noted in the deviations, SRS-5.5 contains the wrong error text message verbiage for pass criteria, and was copied over from a similar error used on WS-017. However, as both texts describe and convey the same error information this is deemed non-critical and passing.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Appendix
SRS-2.5 Peak Found
SRS-5.1 - Monoblock has arced 3 times
SRS-5.5 Script failed to find peak
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.5 | The MB-burnin-test-fixture-script shall locate the operating frequency for each filament duty cycle tested | Wait until the log outputs "++++ Peak Found ++++" twice | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle for both duty cycles tested |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The MB-burnin-test-fixture-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." |  |  |
| SRS-5.5 | The MB-burnin-test-fixture-script shall flag a monoblock if the operating frequency is below 400 kHz or above 650 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-553 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.5 | The MB-burnin-test-fixture-script shall locate the operating frequency for each filament duty cycle tested | Wait until the log outputs "++++ Peak Found ++++" twice | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle for both duty cycles tested | Expected outcome verified See Appendix Verified by EM 12SEP2024 | P |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The MB-burnin-test-fixture-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 12SEP2024 | P |
| SRS-5.5 | The MB-burnin-test-fixture-script shall flag a monoblock if the operating frequency is below 400 kHz or above 650 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." | Minor difference in error text phrasing, see deviations and discussion. See Appendix Verified by EM 12SEP2024 | P |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-563 |  |  |
