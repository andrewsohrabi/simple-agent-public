# VVPR-SWV-028 Rev B: Galden Verification Script Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-028
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-028 - Galden Verification Script Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-028 - Galden Verification Script Verification and Validation Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the galden-verification-script meets operational and functional requirements as stated in MEMO-P01-737 - Galden-Verification-Script Software Requirements Specification Rev. A.
OBJECTIVE
The primary objective of this study is to verify that the beta version of the verification script address the two flag requirements that failed in VVPR-SWV-024
REFERENCES
MEMO-P01-737 - Galden-Verification-Script Software Requirements Specification Rev A
MWI-276 - MS-11235 Monoblock, Power Assembly Rev A
MWI-275 - WS-017 Workstation Installation Rev B
VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report Rev A
VVPR-SWV-024 - Galden Verification Script Verification and Validation Protocol and Report Rev B
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.0-beta
MS-11235 Monoblock Power Assembly Rev. A
SPD3303X Siglent Benchtop power supply (EQP-246 or equivalent)
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121 or equivalent)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238 or equivalent)
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter the test result with their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-275 to install WS-017.
Follow the steps outlined in each table below. MWI-276 should be used to guide operation of the workstation as needed.
Follow the steps outlined in Tables 2 and 5 Test Case column below. Note, Tables 1, and 3 were not included as Tables 2 and 5 hve all tests that require repeating.
Requirements Retested
A partial verification of the requirements outlined in VVPR-SWV-024 are required as modifications incorporated into the -beta version code are only called by the specified requirements.
Requirement SRS-5.2 requires verification as changes were made to the script to correct a logic statement in the error checking block for this requirement. The function was modified to compare the desired input variable (kVnew) to the error threshold.
The logic block for 5.2 checks only its respective requirement to determine if an error is present. Since the code modifications to the logic block only impact SRS-5.2, testing this requirement only is acceptable..
Requirement SRS-5.5 requires verification as the threshold constant was modified from 500kHz to 460kHz as defined by SRS-5.5. The code and logic functioned as intended but was set to trigger an error at a frequency higher than intended.
The logic block for SRS-5.5 checks only its respective requirement to determine if an error is present. Since the code modifications only impact SRS-5.5, testing this requirement only is acceptable.
Requirement SRS-2.2 shall be retested to verify that the changes regarding SRS-5.5 do not impact the functionality of peak-finding success.
Both of these have been deemed minor enough to justify a partial retest. The only code impacted is the logical block where these errors are checked. In both cases it is only the two lines of code that are comparing input variables against a threshold constant. There is no impact to any other code.
Table 2: Operation Requirements.
Table 5: Flagging Requirements.
Data Analysis
All of the verification tests in Tables 2 and 5 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 2 and 5 per the results documented in the “Evidence/Test Result” column and compared to the “Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
Deviations
No Deviations
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.0-beta
MS-11235 Monoblock Power Assembly Rev. A
SPD3303X Siglent Benchtop power supply (EQP-250, calibration exp 6/11/25)
SDS2104X Siglent Digital Oscilloscope (EQP-248, calibration exp 3/3/25)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238, calibration exp 4/20/25)
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-275 to install WS-017.
Follow the steps outlined in each table below. MWI-276 should be used to guide operation of the workstation as needed.
Table 2: Operation Requirements.
Table 5: Flagging Requirements.
Discussion
SRS-2.2 passed in VVPR-SWV-024, and was tested again to ensure changes made in the beta release did not impact peak finding functionality. In both instances of testing, the peak was found to be 692.3kHz.
Overvoltage failures  in SRS-5.2 and resonant frequency failures in SRS-5.5 were both able to be triggered when forced into error conditions.
No anomalies or other issues were found during testing.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Appendix
1. SRS-2.2 - Peak Found
2. SRS-5.2 - Overvoltage in peak detector
3. SRS-5.5 - Resonant frequency not within bounds
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.2 | The galden-verification-script shall locate the operating frequeny for the tested filament duty cycle | Wait until the log outputs "++++ Peak Found ++++" | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.2 | The galden-verification-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." |  |  |
| SRS-5.5 | The galden-verification-script shall flag a monoblock if the operating frequency is below 460 kHz or above 800 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-553 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.2 | The galden-verification-script shall locate the operating frequeny for the tested filament duty cycle | Wait until the log outputs "++++ Peak Found ++++" | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle | Expected outcome verified See Appendix Peak found at 692.3kHz Verified by EM 11SEP2024 | P |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.2 | The galden-verification-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 11SEP2024 | P |
| SRS-5.5 | The galden-verification-script shall flag a monoblock if the operating frequency is below 460 kHz or above 800 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 11SEP2024 | P |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-560 |  |  |
