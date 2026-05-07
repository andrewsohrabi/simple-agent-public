# VVPR-SWV-032 Rev B: MB Burn In Test Fixture Script Verification and Validation Protocol v1.1.0-alpha

## Metadata
- Document ID: VVPR-SWV-032
- Revision: B
- Prefix: VVPR
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: v1.1.0
- Source filename: VVPR-SWV-032 - MB Burn In Test Fixture Script Verification and Validation Protocol v1.1.0-alpha_B-Obsolete.docx
- Source path: Example QMS - MedAI/VVPR-SWV-032 - MB Burn In Test Fixture Script Verification and Validation Protocol v1.1.0-alpha_B-Obsolete.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the MB-burnin-test-fixture-script-IND meets usability and functional requirements as stated in MEMO-P01-790 MB-burnin-test-fixture-script-IND Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify MB-burnin-test-fixture-script-IND version 1.1.0 addresses the updated requirements in MEMO-P01-735 and MEMO-P01-790.
The MB-burnin-test-fixture-script-IND script will be evaluated to confirm all new requirements in both the For Human Use (FHU) and Industrial (IND) Monoblock specifications documented in MEMO-P01-735 and MEMO-P01-790, respectively, as these scripts are identical with the exception of the four values modified for IND use.
REFERENCES
MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification Rev E
MEMO-P01-790 MB-burnin-test-fixture-IND Software Requirements Specification Rev A
MEMO-P01-780 - WS-015 Script Updates For IND Monoblock Rev A
MWI-260 - MS-10579 Monoblock Encapsulated Assembly Verification Rev A
MWI-259 - WS-015 Workstation Installation Rev B
VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report Rev B
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10106 MB-burnin-test-fixture-script-IND v1.1.0-alpha
MS-10579 Monoblock Encapsulated Assembly Rev A equivalent
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-259 to install WS-015.
Follow the steps outlined in each table below. MWI-260 should be used to guide operation of the workstation as needed. Note, all new requirements will be evaluated.
Items Retested
A partial verification of the requirements outlined in VVPR-SWV-025 is required as modifications incorporated into the v1.1.0-alpha version code are only called by the specified requirements.
Requirement SRS-3.6 requires verification as changes were made to the arc detection function to improve the script’s ability to detect arcs while removing a python function which was causing intermittent issues. A portion of the algorithm was improved but the same basic method is used.
The code modification only impacts arc detection. No other requirements are impacted.
Table 1: Math Requirements.
Table 2: Plotting, Logging, and Saving Requirements.
Data Analysis
All of the verification tests in Tables 1 and 2 shall be treated as attributes and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 and 2 per the expected results documented in the “Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
Deviations
During testing, an arc occurred which caused the H-Bridge to drop out. This resulted in the algorithm failing. To account for this, 2 lines of code were added, as shown in Appendix 1.1c, which use the input frequency in the case that the percent difference between the measured and input frequencies is greater than 10%. These changes are released in S10106 v1.1.0-beta.
During testing, a rounding error occurred which had not been encountered previously, preventing the verification of the milestones feature from completing successfully. To account for this error, an edit was made to one line of code to round the number correctly, as shown in Appendix 1.2e. This change is released in S10106 v1.1.0-beta.
A protocol generation error occurred which resulted in known non-arcs not being tested in the SRS-3.6 test case since more than 100 of these non-arc wave forms were evaluated while testing SRS-4.10.
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
ES-10035 A.1 was used on T-183
S10106 MB-burnin-test-fixture-script-IND v1.1.0-alpha
MS-10579 Monoblock Encapsulated Assembly Rev A equivalent
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
EXPERIMENTAL PROCEDURE
Follow the steps outlined in MWI-259 to install WS-015.
Follow the steps outlined in each table below. MWI-260 should be used to guide operation of the workstation as needed.
Table 1: Math Requirements.
Table 2: Plotting, Logging, and Saving Requirements.
Appendix 1
Arc detection test
Only the 3 arc shots are shown here because hundreds of shots were performed during validation of SRS-4.10 and none resulted in false arc detection.
All known arcs were correctly identified.
When running the Monoblock at 80 kV, an event occurred which should be classified as an arc. The arc resulted in the H-bridge dropping out making the measured H-bridge frequency drop to 0. This resulted in the filtering within the arc detection algorithm to fail which caused the waveform to not be counted as an arc. The following lines were added as lines 36 and 37 in mathHelper.py so that if the percent difference between the measured H-bridge frequency and the input frequency is greater than 10%, use the input frequency. Ten (10) data points were looked at to verify that, in normal operating conditions, the percent difference between the measured H-bridge frequency and input frequency does not exceed 10%, so this code addition does not affect any previous testing.
After implementing these changes into v1.1.0-beta, the waveform was passed through the arc detection algorithm and was correctly identified as containing an arc.
Continuing test feature verification evidence
The script was terminated after the lower filament duty cycle was found and then continued by loading the saved file
The program continued by starting the search for the filament duty cycle for max mA, which is the next test that occurs:
Script was terminated after finding the upper filament duty cycle, which is the second milestone and then continued by loading the saved file
Program continued by locating peaks between min and max duty cycles, which is the next test in the sequence
Script was terminated after peaks between min and max duty cycles were found (third milestone), and then continued by loading the saved file.
Program continue by modeling mA vs kV at each duty cycle which is the next step in the sequence
Script was terminated after modeling mA vs kV for each duty cycle (fourth milestone) and then continued by loading the saved file.
Program continued to burn in, which is the next test in the sequence. However, before building up to the burn in technique, the script warmed up the monoblock.
Script was terminated an arbitrary number of shots into burn in (fifth milestone) and then continued by loading the saved file.
Terminated 5 shots into stress test
Program started by warming up the monoblock and then continued with next exposure before termination.
*NOTE: In order for this test to pass, the script required a very minor, almost trivial edit which addressed an unexpected rounding issue. The script was changed from the line in the top image to the line in the bottom image.
This change only affects continuing the test from burn in, so all prior testing is not affected.
Script was terminated an arbitrary number of shots into burn in (fifth milestone) for a second time and then continued by loading the saved file.
Terminated 28 shots into burn in
Program started by warming up mb and then continued to next exposure before termination
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.6 | The MB-burnin-test-fixture-script shall detect arcs on the positive and negative Vsense lines which deviate more than a desired threshold from the average voltage | Run at least 3 different known arcs and 3 different waveforms that are known not arcs through the arc detection function Run the script and force an arc to occur | The function can correctly identify an arc and does not trigger on the non-arcing waveforms The script detects that an arc occurred, throws a warning, and pauses the program |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a functional monoblock has been correctly installed in T-183 Run the script using from a debugger |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.10 | The MB-burnin-test-fixture-script shall save the most recently completed test as a milestone from which the script can start | 1. Stop the program after each test corresponding to a milestone and rerun the script using the saved test file. 2. Stop the program at two different exposures during burn-in | The script prompts the user whether he/she would like to continue the test using a test file saved in the same directory as the test script. The script continues from the most recently completed test. When continuing from after burn-in has started, the script starts first by warming up the monoblock and then continuing stressing the MB from the last completed exposure. |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-628 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.6 | The MB-burnin-test-fixture-script shall detect arcs on the positive and negative Vsense lines which deviate more than a desired threshold from the average voltage | Run at least 3 different known arcs and 3 different waveforms that are known not arcs through the arc detection function Run the script and force an arc to occur | The function can correctly identify an arc and does not trigger on the non-arcing waveforms The script detects that an arc occurred, throws a warning, and pauses the program | See Appendix 1.1 Verified by MI 20NOV2024 | Pass with deviations (see Appendix 1.1c) |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a functional monoblock has been correctly installed in T-183 Run the script using from a debugger |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.10 | The MB-burnin-test-fixture-script shall save the most recently completed test as a milestone from which the script can start | 1. Stop the program after each test corresponding to a milestone and rerun the script using the saved test file. 2. Stop the program at two different exposures during burn-in | The script prompts the user whether he/she would like to continue the test using a test file saved in the same directory as the test script. The script continues from the most recently completed test. When continuing from after burn-in has started, the script starts first by warming up the monoblock and then continuing stressing the MB from the last completed exposure. | See Appendix 1.2 Verified by MI 19NOV2024 | Pass with deviation (see Appendix 1.2e) |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release | See ECR-636 |  |  |
