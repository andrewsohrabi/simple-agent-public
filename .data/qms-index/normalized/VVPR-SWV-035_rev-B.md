# VVPR-SWV-035 Rev B: MB Burn In Test Fixture Script Verification and Validation Protocol v1.2.0-alpha

## Metadata
- Document ID: VVPR-SWV-035
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.2.0
- Source filename: VVPR-SWV-035 - MB Burn In Test Fixture Script Verification and Validation Protocol v1.2.0-alpha_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-035 - MB Burn In Test Fixture Script Verification and Validation Protocol v1.2.0-alpha_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the MB-burnin-test-fixture-script meets usability and functional requirements as stated in MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify MB-burnin-test-fixture-script version 1.2.0 addresses the updated requirements in MEMO-P01-735.
The MB-burnin-test-fixture-script script will be evaluated to confirm all new requirements in both the For Human Use (FHU) and Industrial (IND) Monoblock specifications documented in MEMO-P01-735 and MEMO-P01-790, respectively, as these scripts are identical with the exception of the four values modified for IND use.
REFERENCES
MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification Rev G
MEMO-P01-790 MB-burnin-test-fixture-IND Software Requirements Specification Rev B
MEMO-P01-780 - WS-015 Script Updates For IND Monoblock Rev A
MWI-260 - MS-10579 Monoblock Encapsulated Assembly Verification Rev C
MWI-259 - WS-015 Workstation Installation Rev C
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.2.0-alpha
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
Table 1: 42Q CSV Requirements.
Table 2: Google Cloud Upload Requirements.
Data Analysis
All of the verification tests in Tables 1 and 2 shall be treated as attributes and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 and 2 per the expected results documented in the “Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
Deviations
Minor deviation in SRS-6.13 to account for a change in the directory that Sanmina wants the 42Q csvs saved to.
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.2.0-alpha
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
Table 1: 42Q CSV Requirements.
Table 2: Google Cloud Upload Requirements.
Data Analysis
All of the verification tests in Tables 1 and 2 shall be treated as attributes and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 and 2 per the expected results documented in the “Pass Criteria” column.
DISCUSSION
No issues were discovered
CONCLUSION
Overall Result:
Pass
Fail
Other: Pass with deviation. See report section 1.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Appendix 1
Csv for mb3 exists and the name is in the correct format
The header exists and matches the correct format
Outputs of Calibration Point 1 printed to csv in correct format
Outputs of Calibration Point 2 printed to csv in correct format
Outputs of Find Minimum Duty Cycle printed to csv in correct format
Outputs of Find Maximum Duty Cycle printed to csv in correct format
Outputs of Distribute Peaks printed to csv in correct format
Outputs of Model mA vs kV printed to csv in correct format
Outputs of Achieve Burn In Technique printed to csv in correct format
Outputs of Stress Test printed to csv in correct format
3 failures: maximum duty cycle too high, over voltage in model mA vs kV, too many arcs
Header updated in both the pass and fail conditions
Csv was copied to the C:\In folder
Deviation: The location that Sanmina wants the csvs copied to is actually O:\In. This was changed in the script and verified operation. A fake O: drive was used for verification at MedAI.
Appendix 2
The .log, .hdf, and .csv files from the script directory were uploaded to the google cloud bucket:
Files were deleted:

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-6.1 | The MB-burnin-test-fixture-script shall create a csv with a filename following the 42Q format | Run the script and wait for the first exposure | A csv exists in the script directory with a name matching the format required by 42Q documentation |  |  |
| SRS-6.2 | The MB-burnin-test-fixture-script shall print a header in the csv with the correct 42Q format | Run the script and wait for the first exposure | The csv header matches the format required by 42Q documentation |  |  |
| SRS-6.3 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, and kV of Calibration Point 1 to the csv if the monoblock passes that test | Run the script and wait for Calibration Point 1 test to complete The log will output "********** Finding Peak for Filament Duty of 24% **********" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, and tube potential are present and match the format required by 42Q |  |  |
| SRS-6.4 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, and kV of Calibration Point 2 to the csv if the monoblock passes that test | Run the script and wait for Calibration Point 2 test to complete The log will output "----------------- Locating Filament Duty Cycle for Min mA -----------------" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, and tube potential are present and match the format required by 42Q |  |  |
| SRS-6.5 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, duty cycle, kV and mA of Find Minimum Duty Cycle to the csv if the monoblock passes that test | Run the script and wait for Find Minimum Duty Cycle test to complete The log will output "+++ LOWER FILAMENT DUTY CYCLE FOUND +++" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, duty cycle, beam current, and tube potential are present and match the format required by 42Q |  |  |
| SRS-6.6 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, duty cycle, kV and mA of Find Maximum Duty Cycle to the csv if the monoblock passes that test | Run the script and wait for Find Maximum Duty Cycle test to complete The log will output "+++ UPPER FILAMENT DUTY CYCLE FOUND +++" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, duty cycle, beam current, and tube potential are present and match the format required by 42Q |  |  |
| SRS-6.7 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, and kV of each peak found during Distribute Peaks to the csv if the monoblock passes that test | Run the script and wait for Distribute Peaks test to complete The log will output "----------------- Raising to 70 kV and Modeling mA vs kV At Each Duty Cycle -----------------" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, and tube potential for each peak are present and match the format required by 42Q |  |  |
| SRS-6.8 | The MB-burnin-test-fixture-script shall print true for each duty cycle of Model mA vs kV to the csv if the monoblock passes that test | Run the script and wait for Model mA vs kV test to complete The log will output "+++++ !CHARACTERIZATION COMPLETE! +++++" | The final results of the test are in the csv with "PASSED" and a boolean "1" for each duty cycle is present and matches the format required by 42Q |  |  |
| SRS-6.9 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, duty cycle, kV, and mA of Achieve Burn-In Technique to the csv if the monoblock passes that test | Run the script and wait for Stress Test to complete The log will output "!!!!! TEST COMPLETE !!!!!" | The final results of the Achieve Burn-In Technique test are in the csv with "PASSED" and values for freqN, input voltage, duty cycle, tube potential, and beam current are present and match the format required by 42Q |  |  |
| SRS-6.10 | The MB-burnin-test-fixture-script shall print final dose drift of Stress Test and number of arcs found during Stress Test to the csv if the monoblock passes that test | Run the script and wait for Stress Test to complete The log will output "!!!!! TEST COMPLETE !!!!!" | The final results of the test are in the csv with "PASSED" and values for dose drift and number of arcs are present and match the format required by 42Q |  |  |
| SRS-6.11 | The MB-burnin-test-fixture-script shall print the failure and corresponding value to the csv | Run the script and force 3 different monoblock failures to occur at different points in the script | The test that failed is printed to the csv with "FAILED" and the value corresponding to the failure |  |  |
| SRS-6.12 | The MB-burnin-test-fixture-script shall update the csv header with the overall pass/fail and time of completion when the test completes | Run the script until the mb passes the full test Run the script and force a mb to fail | The header is updated with the overall PASSED result and time of completion The header is updated with the overall FAILED result and time of completion |  |  |
| SRS-6.13 | The MB-burnin-test-fixture-script shall move the csv to the correct directory for 42Q when the test completes | Run the script until mb either passes or fails | Check that the csv has been copied to the directory required by 42Q |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-7.1 | The MB-burnin-test-fixture-script shall upload all .txt, .hdf5, and .csv files within the script directory to a google cloud bucket | Run the upload_data.py script | The terminal window prints "All files have been uploaded" The workstation 15 google cloud bucket should contain the .txt, .csv, and .hdf5 files generated by WS-015 |  |  |
| SRS-7.2 | The MB-burnin-test-fixture-script shall delete all .txt, .hdf5, and .csv files within the script directory after upload if desired by the user | Run the upload_data.py script type "y" when prompted and press enter | The terminal window prints "Files have been deleted." The files are no longer present inside script directory |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-677 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-6.1 | The MB-burnin-test-fixture-script shall create a csv with a filename following the 42Q format | Run the script and wait for the first exposure | A csv exists in the script directory with a name matching the format required by 42Q documentation | Expected outcome verified See Appendix 1.1 Verified by MI 14JAN2025 | P |
| SRS-6.2 | The MB-burnin-test-fixture-script shall print a header in the csv with the correct 42Q format | Run the script and wait for the first exposure | The csv header matches the format required by 42Q documentation | Expected outcome verified See Appendix 1.2 Verified by MI 14JAN2025 | P |
| SRS-6.3 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, and kV of Calibration Point 1 to the csv if the monoblock passes that test | Run the script and wait for Calibration Point 1 test to complete The log will output "********** Finding Peak for Filament Duty of 24% **********" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, and tube potential are present and match the format required by 42Q | Expected outcome verified See Appendix 1.3 Verified by MI 14JAN2025 | P |
| SRS-6.4 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, and kV of Calibration Point 2 to the csv if the monoblock passes that test | Run the script and wait for Calibration Point 2 test to complete The log will output "----------------- Locating Filament Duty Cycle for Min mA -----------------" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, and tube potential are present and match the format required by 42Q | Expected outcome verified See Appendix 1.4 Verified by MI 14JAN2025 | P |
| SRS-6.5 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, duty cycle, kV and mA of Find Minimum Duty Cycle to the csv if the monoblock passes that test | Run the script and wait for Find Minimum Duty Cycle test to complete The log will output "+++ LOWER FILAMENT DUTY CYCLE FOUND +++" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, duty cycle, beam current, and tube potential are present and match the format required by 42Q | Expected outcome verified See Appendix 1.5 Verified by MI 14JAN2025 | P |
| SRS-6.6 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, duty cycle, kV and mA of Find Maximum Duty Cycle to the csv if the monoblock passes that test | Run the script and wait for Find Maximum Duty Cycle test to complete The log will output "+++ UPPER FILAMENT DUTY CYCLE FOUND +++" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, duty cycle, beam current, and tube potential are present and match the format required by 42Q | Expected outcome verified See Appendix 1.6 Verified by MI 14JAN2025 | P |
| SRS-6.7 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, and kV of each peak found during Distribute Peaks to the csv if the monoblock passes that test | Run the script and wait for Distribute Peaks test to complete The log will output "----------------- Raising to 70 kV and Modeling mA vs kV At Each Duty Cycle -----------------" | The final results of the test are in the csv with "PASSED" and values for freqN, input voltage, and tube potential for each peak are present and match the format required by 42Q | Expected outcome verified See Appendix 1.7 Verified by MI 14JAN2025 | P |
| SRS-6.8 | The MB-burnin-test-fixture-script shall print true for each duty cycle of Model mA vs kV to the csv if the monoblock passes that test | Run the script and wait for Model mA vs kV test to complete The log will output "+++++ !CHARACTERIZATION COMPLETE! +++++" | The final results of the test are in the csv with "PASSED" and a boolean "1" for each duty cycle is present and matches the format required by 42Q | Expected outcome verified See Appendix 1.8 Verified by MI 14JAN2025 | P |
| SRS-6.9 | The MB-burnin-test-fixture-script shall print final freqN, input voltage, duty cycle, kV, and mA of Achieve Burn-In Technique to the csv if the monoblock passes that test | Run the script and wait for Stress Test to complete The log will output "!!!!! TEST COMPLETE !!!!!" | The final results of the Achieve Burn-In Technique test are in the csv with "PASSED" and values for freqN, input voltage, duty cycle, tube potential, and beam current are present and match the format required by 42Q | Expected outcome verified See Appendix 1.9 Verified by MI 14JAN2025 | P |
| SRS-6.10 | The MB-burnin-test-fixture-script shall print final dose drift of Stress Test and number of arcs found during Stress Test to the csv if the monoblock passes that test | Run the script and wait for Stress Test to complete The log will output "!!!!! TEST COMPLETE !!!!!" | The final results of the test are in the csv with "PASSED" and values for dose drift and number of arcs are present and match the format required by 42Q | Expected outcome verified See Appendix 1.10 Verified by MI 14JAN2025 | P |
| SRS-6.11 | The MB-burnin-test-fixture-script shall print the failure and corresponding value to the csv | Run the script and force 3 different monoblock failures to occur at different points in the script | The test that failed is printed to the csv with "FAILED" and the value corresponding to the failure | Expected outcome verified See Appendix 1.11 Verified by MI 14JAN2025 | P |
| SRS-6.12 | The MB-burnin-test-fixture-script shall update the csv header with the overall pass/fail and time of completion when the test completes | Run the script until the mb passes the full test Run the script and force a mb to fail | The header is updated with the overall PASSED result and time of completion The header is updated with the overall FAILED result and time of completion | Expected outcome verified See Appendix 1.12 Verified by MI 14JAN2025 | P |
| SRS-6.13 | The MB-burnin-test-fixture-script shall move the csv to the correct directory for 42Q when the test completes | Run the script until mb either passes or fails | Check that the csv has been copied to the directory required by 42Q | Expected outcome verified See Appendix 1.13 Verified by MI 14JAN2025 | Pass with deviation |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-7.1 | The MB-burnin-test-fixture-script shall upload all .txt, .hdf5, and .csv files within the script directory to a google cloud bucket | Run the upload_data.py script | The terminal window prints "All files have been uploaded" The workstation 15 google cloud bucket should contain the .txt, .csv, and .hdf5 files generated by WS-015 | Expected outcome verified See Appendix 2.1 Verified by MI 14JAN2025 | P with deviation |
| SRS-7.2 | The MB-burnin-test-fixture-script shall delete all .txt, .hdf5, and .csv files within the script directory after upload if desired by the user | Run the upload_data.py script type "y" when prompted and press enter | The terminal window prints "Files have been deleted." The files are no longer present inside script directory | Expected outcome verified See Appendix 2.2 Verified by MI 14JAN2025 | P |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | See ECR-686 |  |  |
