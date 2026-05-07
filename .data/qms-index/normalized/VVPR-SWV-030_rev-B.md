# VVPR-SWV-030 Rev B: Galden Verification Script Verification and Validation Protocol

## Metadata
- Document ID: VVPR-SWV-030
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-030 - Galden Verification Script Verification and Validation Protocol_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-030 - Galden Verification Script Verification and Validation Protocol_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the galden-verification-script meets usability and functional requirements as stated in MEMO-P01-737 - galden-verification-script Software Requirements Specification Rev. A.
OBJECTIVE
The primary objective of this study is to verify the Production Software Tool: galden-verification-script for use in WS-017 for production of the monoblock used in the MX1 emitter.
REFERENCES
MEMO-P01-737 - galden-verification-script Software Requirements Specification Rev A
MWI-276 - MS-11235 Monoblock, Power Assembly Rev A
MWI-275 - WS-017 Workstation Installation Rev B
VVPR-SWV-024 - Galden Verification Script Verification and Validation Protocol and Report Rev B
VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report Rev B
S10102 MB-burnin-test-fixture-script v1.0.0
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.1-alpha
MS-11235 Monoblock Power Assembly Rev. A
SPD3303X Siglent Benchtop power supply (EQP-246 or equivalent)
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121 or equivalent)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238 or equivalent)
SAMPLE SIZE
This is a software verification test, and therefore will use a sample size of 1.
REQUIREMENTS RETESTED
A partial verification of the requirements outlined in VVPR-SWV-024 is required as the patched code in v1.0.1 does not impact functionality.
Section 3 “Math Requirements” of Software Requirements Specification (MEMO-P01-737) is a code base pulled from WS-015 (S10102 MB-burnin-test-fixture-script v1.0.0) and has not been modified. The results for this testing can be found directly in VVPR-SWV-025, and are also referenced in VVPR-SWV-024 appendix 3.
All other sections will be retested
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-275 to install WS-017.
Follow the steps outlined in each table below. MWI-276 should be used to guide operation of the workstation as needed.
Table 1: System Interfacing.
Table 2: Operation Requirements.
Table 4: Plotting, Logging, and Saving Requirements.
Table 5: Flagging Requirements.
Data Analysis
All of the verification tests in Tables 1 through 5 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 through 5 per the expected results documented in the “Expected Result/Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
Deviations
No Deviations
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.1-alpha
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
Table 1: System Interfacing.
Table 2: Operation Requirements.
Table 4: Plotting, Logging, and Saving Requirements.
Table 5: Flagging Requirements.
DISCUSSION
No anomalies or issues were discovered during testing
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
APPENDIX
Table 1 Results
Connected to Scope
Connected to 3 Channel Power Supply
Connected to 1 Channel Power Supply
Connected to Nucleo Board
Table 2 Results
Peak Found
Changing Peak to 70kV
kV Does not exceed 72kV
Stress Test Complete
Table 4 Results
Plot
Frequency Values
Input Voltage
Waveform Measurements
Arc Detected
HDF
Table 5 Results
Arc Failure
Overvoltage Failure
Input Voltage Error
Difference between Vsn and Vsp Failure
Resonant frequency Failure
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The galden-verification-script shall interface with an oscilliscope | Confirm that the log outputs "Connected to Scope" | Confirm that the log outputs "Connected to Scope" |  |  |
| SRS-1.2 | The galden-verification-script shall interface with a multi channel power supply | Confirm that the log outputs "Connected to 3 channel power supply: SPD3303X-E" | Confirm that the log outputs "Connected to 3 channel power supply: SPD3303X-E" |  |  |
| SRS-1.3 | The galden-verification-script shall interface with a single channel power supply | Confirm that the log outputs "Connected to single channel power supply: SPD1305X" | Confirm that the log outputs "Connected to single channel power supply: SPD1305X" |  |  |
| SRS-1.4 | The galden-verification-script shall interface with a Nucleo board | Confirm that the log outputs "Connected to Nucleo Board" | Confirm that the log outputs "Connected to Nucleo Board" |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The galden-verification-script shall locate the operating frequeny for the tested filament duty cycle | Wait until the log outputs "++++ Peak Found ++++" | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle |  |  |
| SRS-2.2 | The galden-verification-script shall dictate the excitation voltage to generate high voltage potential on the monoblock outputs | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the program takes 4 more exposures | 1. For each of the 4 exposures, the value in the log output reading "vIn = {value} V" is equal to the first number in the TX vector in the log output divided by 10 2. vIn increases with each subsequent exposure |  |  |
| SRS-2.3 | The galden-verification-script shall safely approach 70 kV (±1%) without overshooting kV by more tha 2 kV | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the log outputs "****** Commencing Stress Test ******" | For each exposure between the log outputs listed in verification steps 1 and 2, the kV does not exceed 72 kV |  |  |
| SRS-2.4 | The galden-verification-script shall repeatedly stress the monoblock components for 100 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 86 kV, in which case vIn will drop by .2 V |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.1 | The galden-verification-script shall plot the processed tube potential and average steady state tube potential | Wait until the program excites the unit under test | 1. Figure appears with plot titled "Tube Potential" and contains a legend and 2 waveforms: one for tube voltage and one for average tube voltage, as indicated by the legend |  |  |
| SRS-4.2 | The galden-verification-script shall log the input frequency sent to the Nucleo board | Wait until the program excites the unit under test | log outputs a line containing "FreqN = {value1}, Freq = {value2} kHz" where the values are numbers greater than 0. |  |  |
| SRS-4.3 | The galden-verification-script shall log the input voltage sent to the single channel power supply | Wait until the program excites the unit under test | 1. log outputs a line containing "vIn = {value}" where value is a number greater than 0. |  |  |
| SRS-4.4 | The galden-verification-script shall log measurements made on the analyzed waveforms | Wait until the program excites the unit under test | 1. Log outputs a line which reads "Measurements: Vsp: {value1} V, Vsn: {value2} V" where the values are numbers 2. Log outputs a line which reads "Rise time = {value} ms where value is a number 3. Log outputs a line which reads "Driving Frequency = {value} kHz" where value is a number 4. Log outputs a line which reads "H bridge voltage amplitude = {value} V" where value is a number 5. Log outputs a line which reads "Average beam current = {value} mA" where value is a number 6. Log outputs a line which reads "Extrapolated Average Tube Voltage: {value} kV" where value is a number |  |  |
| SRS-4.5 | The galden-verification-script shall log a warning if an arc is detected | Run the script and force and arc to occur | Log outputs a warning which reads "!!! Potential Arc Detected !!!" |  |  |
| SRS-4.6 | The galden-verification-script shall save the waveforms and measurement results for every exposure taken | 1. Wait until the program excites the unit under test 3 times. 2. Open the hdf file using an hdf viewer | 1. hdf file contains 1 folder labeled "DutyCycle1 2. The DutyCycle1 folder contains 3 folders named "Exposure1," "Exposure2," and "Exposure3" 3. Each Exposure folder contains a dataset named "HV," "Vsn," "Vsp," "beamI," "time", and "SettingsAndMsrmts" |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The galden-verification-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." |  |  |
| SRS-5.2 | The galden-verification-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." |  |  |
| SRS-5.3 | The galden-verification-script shall flag a monoblock if the rate of change in tube potential is not positive with the rate of change in input voltage, if the input voltage has increased at least 15%. | Run the script and prevent kV from rising when vIn rises | Log outputs "Changing vIn has minimal effect on kV. FAIL MONOBLOCK." |  |  |
| SRS-5.4 | The galden-verification-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% | Run the script and generate a Vsn and Vsp that differ by more than 10% | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." |  |  |
| SRS-5.5 | The galden-verification-script shall flag a monoblock if the operating frequency is below 460 kHz or above 800 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." |  |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-566 |  |  |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The galden-verification-script shall interface with an oscilliscope | Confirm that the log outputs "Connected to Scope" | Confirm that the log outputs "Connected to Scope" | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-1.2 | The galden-verification-script shall interface with a multi channel power supply | Confirm that the log outputs "Connected to 3 channel power supply: SPD3303X-E" | Confirm that the log outputs "Connected to 3 channel power supply: SPD3303X-E" | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-1.3 | The galden-verification-script shall interface with a single channel power supply | Confirm that the log outputs "Connected to single channel power supply: SPD1305X" | Confirm that the log outputs "Connected to single channel power supply: SPD1305X" | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-1.4 | The galden-verification-script shall interface with a Nucleo board | Confirm that the log outputs "Connected to Nucleo Board" | Confirm that the log outputs "Connected to Nucleo Board" | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The galden-verification-script shall locate the operating frequeny for the tested filament duty cycle | Wait until the log outputs "++++ Peak Found ++++" | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-2.2 | The galden-verification-script shall dictate the excitation voltage to generate high voltage potential on the monoblock outputs | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the program takes 4 more exposures | 1. For each of the 4 exposures, the value in the log output reading "vIn = {value} V" is equal to the first number in the TX vector in the log output divided by 10 2. vIn increases with each subsequent exposure | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-2.3 | The galden-verification-script shall safely approach 70 kV (±1%) without overshooting kV by more tha 2 kV | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the log outputs "****** Commencing Stress Test ******" | For each exposure between the log outputs listed in verification steps 1 and 2, the kV does not exceed 72 kV | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-2.4 | The galden-verification-script shall repeatedly stress the monoblock components for 100 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 86 kV, in which case vIn will drop by .2 V | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.1 | The galden-verification-script shall plot the processed tube potential and average steady state tube potential | Wait until the program excites the unit under test | 1. Figure appears with plot titled "Tube Potential" and contains a legend and 2 waveforms: one for tube voltage and one for average tube voltage, as indicated by the legend | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-4.2 | The galden-verification-script shall log the input frequency sent to the Nucleo board | Wait until the program excites the unit under test | log outputs a line containing "FreqN = {value1}, Freq = {value2} kHz" where the values are numbers greater than 0. | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-4.3 | The galden-verification-script shall log the input voltage sent to the single channel power supply | Wait until the program excites the unit under test | 1. log outputs a line containing "vIn = {value}" where value is a number greater than 0. | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-4.4 | The galden-verification-script shall log measurements made on the analyzed waveforms | Wait until the program excites the unit under test | 1. Log outputs a line which reads "Measurements: Vsp: {value1} V, Vsn: {value2} V" where the values are numbers 2. Log outputs a line which reads "Rise time = {value} ms where value is a number 3. Log outputs a line which reads "Driving Frequency = {value} kHz" where value is a number 4. Log outputs a line which reads "H bridge voltage amplitude = {value} V" where value is a number 5. Log outputs a line which reads "Average beam current = {value} mA" where value is a number 6. Log outputs a line which reads "Extrapolated Average Tube Voltage: {value} kV" where value is a number | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-4.5 | The galden-verification-script shall log a warning if an arc is detected | Run the script and force and arc to occur | Log outputs a warning which reads "!!! Potential Arc Detected !!!" | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-4.6 | The galden-verification-script shall save the waveforms and measurement results for every exposure taken | 1. Wait until the program excites the unit under test 3 times. 2. Open the hdf file using an hdf viewer | 1. hdf file contains 1 folder labeled "DutyCycle1 2. The DutyCycle1 folder contains 3 folders named "Exposure1," "Exposure2," and "Exposure3" 3. Each Exposure folder contains a dataset named "HV," "Vsn," "Vsp," "beamI," "time", and "SettingsAndMsrmts" | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The galden-verification-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-5.2 | The galden-verification-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-5.3 | The galden-verification-script shall flag a monoblock if the rate of change in tube potential is not positive with the rate of change in input voltage, if the input voltage has increased at least 15%. | Run the script and prevent kV from rising when vIn rises | Log outputs "Changing vIn has minimal effect on kV. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-5.4 | The galden-verification-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% | Run the script and generate a Vsn and Vsp that differ by more than 10% | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |
| SRS-5.5 | The galden-verification-script shall flag a monoblock if the operating frequency is below 460 kHz or above 800 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." | Expected outcome verified See Appendix Verified by EM 18SEP2024 | P |

### Table 10
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report Release | Refer to ECR-568 |  |  |
