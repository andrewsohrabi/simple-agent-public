# VVPR-SWV-024 Rev B: Galden Verification Script Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-024
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-024 - Galden Verification Script Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-024 - Galden Verification Script Verification and Validation Protocol and Report_B.docx
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
VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report Rev A
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.0-alpha
MS-11235 Monoblock Power Assembly Rev. A
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
Follow the steps outlined in MWI-275 to install WS-017.
Follow the steps outlined in each table below. MWI-276 should be used to guide operation of the workstation as needed.
Table 1: System Interfacing.
Table 2: Operation Requirements.
Table 3: Math Requirements.
Table 4: Plotting, Logging, and Saving Requirements.
Table 5: Flagging Requirements.
Data Analysis
All of the verification tests in Tables 1 through 5 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 through 5 per the expected results documented in the “Expected Result/Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key:
example.com/
Report Section
Deviations
SRS-2.1 is copied from WS-015 SRS but does not apply for this workstation. It has been listed as “Not Evaluated” and will be removed in the next revision of the SRS.
MATERIALS
WS-017 - MS-11235 Monoblock Power Assembly Verification (Ref MWI-275)
S10103 galden-verification-script v1.0.0-alpha
MS-11235 Monoblock Power Assembly Rev. A
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
Follow the steps outlined in MWI-275 to install WS-017.
Follow the steps outlined in each table below. MWI-276 should be used to guide operation of the workstation as needed.
Table 1: System Interfacing.
Table 2: Operation Requirements.
Table 3: Math Requirements.
Table 4: Plotting, Logging, and Saving Requirements.
Table 5: Flagging Requirements.
Discussion
SRS-5.2 was marked as a failure. The logical statement references the wrong variable and does not trigger. Once updated to the correct variable, this error detection works as intended and will be included in the beta release of this script.
SRS-5.5 was marked as a failure. The threshold constant was at 500kHz instead of 460kHz outlined by the SRS, the code and logic functioned as intended but triggered at the wrong frequency. Updating this constant results in correctly functioning code. This update will be included in the beta release of the script.
As listed in the deviations. SRS-2.1 is not relevant to this workstation and will be removed in the next revision of the SRS
Outside of these minor issues no other anomalies were discovered.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Appendix 1: System Interfacing Requirements Evidence
1.1 - 1.4
Appendix 2: Operation Requirements Evidence
2.1 Not evaluated.. There is no filament for this workstation
2.2 Peak found at FreqN = 60
2.3 Increasing vIN to target 70kV
2.4 Highest kV was 69.72 during Exposure 40, matches FreqN = 60 found in SRS-2.2
2.5 Completed 100 Exposures
Appendix 3: Math Requirements
3.1 - Evidence from SRS-3.6 on VVPR-SWV-025 Report
3.2 and 3.3 - Evidence from SRS-3.3 on VVPR-SWV-025 Report
Oscilliscope H Bridge measurements
3.3.1 Frequency measurement: 528.74 Hz
3.3.2 Average max amplitude: 1.45 V (note the amplitude will need to be multiplied by 10.5932 to scale to actually amplitude of H bridge)
3.3.3 Average min amplitude: 58.6 mV
3.4 - Evidence from SRS-3.4 on VVPR-SWV-025 Report
mA measurement evidence
Appendix 4: Plotting, Logging, and Saving Requirements Evidence
4.1 Plotting
4.2 Log Outputs for input frequency
4.3 Log Outputs for input voltage
4.4 Log Outputs for analyzed waveforms
4.5 Log outputs a potential arc warning
4.6 HDF File
Appendix 5: Flagging Requirements Evidence
5.1 Monoblock has arced 3 times
5.2 Overvoltage in peak detector could not be triggered
5.3 Changing vIN has minimal effect
5.4 Difference between Vsp and Vsn larger than 10%
5.5 Resonant Frequency not within bounds could not be triggered
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
| SRS-2.1 | The galden-verification-script shall dictate which filament duty cycle the test board will use | 1. Wait until the log outputs "Finding Peak for Filament Duty of 22%" | The value in the log output reading "Filament Duty Cycle = {value}%" should be equal to the last number in the TX vector in the log output divided by 10 |  |  |
| SRS-2.2 | The galden-verification-script shall locate the operating frequeny for the tested filament duty cycle | Wait until the log outputs "++++ Peak Found ++++" | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle |  |  |
| SRS-2.3 | The galden-verification-script shall dictate the excitation voltage to generate high voltage potential on the monoblock outputs | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the program takes 4 more exposures | 1. For each of the 4 exposures, the value in the log output reading "vIn = {value} V" is equal to the first number in the TX vector in the log output divided by 10 2. vIn increases with each subsequent exposure |  |  |
| SRS-2.4 | The galden-verification-script shall safely approach 70 kV (±1%) without overshooting kV by more tha 2 kV | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the log outputs "****** Commencing Stress Test ******" | For each exposure between the log outputs listed in verification steps 1 and 2, the kV does not exceed 72 kV |  |  |
| SRS-2.5 | The galden-verification-script shall repeatedly stress the monoblock components for 100 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 86 kV, in which case vIn will drop by .2 V |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Refer to Table 3 of VVPR-SWV-025 |  |  |  |  |
| Test Setup: | Refer to Table 3 of VVPR-SWV-025 |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.1 | The galden-verification-script shall detect arcs on the positive and negative Vsense lines which deviate more than 6% from the average voltage | Verify SRS 3.6 for MB-burnin-test-fixture script | SRS 3.6 for MB-burnin-test-fixture script passes |  |  |
| SRS-3.2 | The galden-verification-script shall analyze H Bridge waveform and calculate H bridge amplitude | Verify SRS 3.3 for MB-burnin-test-fixture script | SRS 3.3 for MB-burnin-test-fixture script passes |  |  |
| SRS-3.3 | The galden-verification-script shall calculate H bridge driving frequency and ensure it is greater than 300 kHz | Verify SRS 3.3 for MB-burnin-test-fixture script | SRS 3.3 for MB-burnin-test-fixture script passes |  |  |
| SRS-3.4 | The galden-verification-script shall analyze beam current and calculate steady state average | Verify SRS 3.4 for MB-burnin-test-fixture script | SRS 3.4 for MB-burnin-test-fixture script passes |  |  |
| SRS-3.5 | The galden-verification-script shall analyze Vsn and Vsp to calculate tube voltage and steady state average | Verify SRS 3.5 for MB-burnin-test-fixture script | SRS 3.5 for MB-burnin-test-fixture script passes |  |  |

### Table 4
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

### Table 5
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

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-542 |  |  |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The galden-verification-script shall interface with an oscilliscope | Confirm that the log outputs "Connected to Scope" | Confirm that the log outputs "Connected to Scope" | Expected outcome verified See Appendix 1 Verified by EM 04SEP2024 | P |
| SRS-1.2 | The galden-verification-script shall interface with a multi channel power supply | Confirm that the log outputs "Connected to 3 channel power supply: SPD3303X-E" | Confirm that the log outputs "Connected to 3 channel power supply: SPD3303X-E" | Expected outcome verified See Appendix 1 Verified by EM 04SEP2024 | P |
| SRS-1.3 | The galden-verification-script shall interface with a single channel power supply | Confirm that the log outputs "Connected to single channel power supply: SPD1305X" | Confirm that the log outputs "Connected to single channel power supply: SPD1305X" | Expected outcome verified See Appendix 1 Verified by EM 04SEP2024 | P |
| SRS-1.4 | The galden-verification-script shall interface with a Nucleo board | Confirm that the log outputs "Connected to Nucleo Board" | Confirm that the log outputs "Connected to Nucleo Board" | Expected outcome verified See Appendix 1 Verified by EM 04SEP2024 | P |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The galden-verification-script shall dictate which filament duty cycle the test board will use | 1. Wait until the log outputs "Finding Peak for Filament Duty of 22%" | The value in the log output reading "Filament Duty Cycle = {value}%" should be equal to the last number in the TX vector in the log output divided by 10 | Not evaluated, there is no filament for this workstation | NE |
| SRS-2.2 | The galden-verification-script shall locate the operating frequeny for the tested filament duty cycle | Wait until the log outputs "++++ Peak Found ++++" | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle | Expected operation verified See Appendix 2.2 Verified by EM 04SEP2024 | P |
| SRS-2.3 | The galden-verification-script shall dictate the excitation voltage to generate high voltage potential on the monoblock outputs | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the program takes 4 more exposures | 1. For each of the 4 exposures, the value in the log output reading "vIn = {value} V" is equal to the first number in the TX vector in the log output divided by 10 2. vIn increases with each subsequent exposure | Expected operation verified See Appendix 2.3 Verified by EM 04SEP2024 | P |
| SRS-2.4 | The galden-verification-script shall safely approach 70 kV (±1%) without overshooting kV by more tha 2 kV | 1. Wait until the log outputs "####### Changing Peak V to 70 kV #######" 2. Wait until the log outputs "****** Commencing Stress Test ******" | For each exposure between the log outputs listed in verification steps 1 and 2, the kV does not exceed 72 kV | Expected operation verified See Appendix 2.4 Verified by EM 04SEP2024 | P |
| SRS-2.5 | The galden-verification-script shall repeatedly stress the monoblock components for 100 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 86 kV, in which case vIn will drop by .2 V | Expected operation verified See Appendix 2.5 Verified by EM 04SEP2024 | P |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Refer to Table 3 of VVPR-SWV-025 |  |  |  |  |
| Test Setup: | Refer to Table 3 of VVPR-SWV-025 |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.1 | The galden-verification-script shall detect arcs on the positive and negative Vsense lines which deviate more than 6% from the average voltage | Verify SRS 3.6 for MB-burnin-test-fixture script | SRS 3.6 for MB-burnin-test-fixture script passes | Expected outcome verified See VVPR-SWV-025 Rev B Table 3 Report and Appendix 3 Verified by MI 04SEP2024 | P |
| SRS-3.2 | The galden-verification-script shall analyze H Bridge waveform and calculate H bridge amplitude | Verify SRS 3.3 for MB-burnin-test-fixture script | SRS 3.3 for MB-burnin-test-fixture script passes | Expected outcome verified See VVPR-SWV-025 Rev B Table 3 Report and Appendix 3 Verified by MI 04SEP2024 | P |
| SRS-3.3 | The galden-verification-script shall calculate H bridge driving frequency and ensure it is greater than 300 kHz | Verify SRS 3.3 for MB-burnin-test-fixture script | SRS 3.3 for MB-burnin-test-fixture script passes | Expected outcome verified See VVPR-SWV-025 Rev B Table 3 Report and Appendix 3 Verified by MI 04SEP2024 | P |
| SRS-3.4 | The galden-verification-script shall analyze beam current and calculate steady state average | Verify SRS 3.4 for MB-burnin-test-fixture script | SRS 3.4 for MB-burnin-test-fixture script passes | Expected outcome verified See VVPR-SWV-025 Rev B Table 3 Report and Appendix 3 Verified by MI 04SEP2024 | P |
| SRS-3.5 | The galden-verification-script shall analyze Vsn and Vsp to calculate tube voltage and steady state average | Verify SRS 3.5 for MB-burnin-test-fixture script | SRS 3.5 for MB-burnin-test-fixture script passes | Expected outcome verified See VVPR-SWV-025 Rev B Table 3 Report and Appendix 3 Verified by MI 04SEP2024 | P |

### Table 10
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | WS-017 |  |  |  |  |
| Test Setup: | WS-017 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\WS-017\Documents\GitHub\galden-verification-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.1 | The galden-verification-script shall plot the processed tube potential and average steady state tube potential | Wait until the program excites the unit under test | 1. Figure appears with plot titled "Tube Potential" and contains a legend and 2 waveforms: one for tube voltage and one for average tube voltage, as indicated by the legend | Expected operation verified See Appendix 4.1 Verified by EM 06SEP2024 | P |
| SRS-4.2 | The galden-verification-script shall log the input frequency sent to the Nucleo board | Wait until the program excites the unit under test | log outputs a line containing "FreqN = {value1}, Freq = {value2} kHz" where the values are numbers greater than 0. | Expected operation verified See Appendix 4.2 Verified by EM 04SEP2024 | P |
| SRS-4.3 | The galden-verification-script shall log the input voltage sent to the single channel power supply | Wait until the program excites the unit under test | 1. log outputs a line containing "vIn = {value}" where value is a number greater than 0. | Expected operation verified See Appendix 4.3 Verified by EM 04SEP2024 | P |
| SRS-4.4 | The galden-verification-script shall log measurements made on the analyzed waveforms | Wait until the program excites the unit under test | 1. Log outputs a line which reads "Measurements: Vsp: {value1} V, Vsn: {value2} V" where the values are numbers 2. Log outputs a line which reads "Rise time = {value} ms where value is a number 3. Log outputs a line which reads "Driving Frequency = {value} kHz" where value is a number 4. Log outputs a line which reads "H bridge voltage amplitude = {value} V" where value is a number 5. Log outputs a line which reads "Average beam current = {value} mA" where value is a number 6. Log outputs a line which reads "Extrapolated Average Tube Voltage: {value} kV" where value is a number | Expected operation verified See Appendix 4.4 Verified by EM 04SEP2024 | P |
| SRS-4.5 | The galden-verification-script shall log a warning if an arc is detected | Run the script and force and arc to occur | Log outputs a warning which reads "!!! Potential Arc Detected !!!" | Expected operation verified See Appendix 4.5 Verified by EM 04SEP2024 | P |
| SRS-4.6 | The galden-verification-script shall save the waveforms and measurement results for every exposure taken | 1. Wait until the program excites the unit under test 3 times. 2. Open the hdf file using an hdf viewer | 1. hdf file contains 1 folder labeled "DutyCycle1 2. The DutyCycle1 folder contains 3 folders named "Exposure1," "Exposure2," and "Exposure3" 3. Each Exposure folder contains a dataset named "HV," "Vsn," "Vsp," "beamI," "time", and "SettingsAndMsrmts" | Expected operation verified See Appendix 4.4 Verified by EM 04SEP2024 | P |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The galden-verification-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." | Expected operation verified See Appendix 5.1 Verified by EM 06SEP2024 | P |
| SRS-5.2 | The galden-verification-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." | Expected operation verified See Appendix 5.2 Verified by EM 06SEP2024 | F |
| SRS-5.3 | The galden-verification-script shall flag a monoblock if the rate of change in tube potential is not positive with the rate of change in input voltage, if the input voltage has increased at least 15%. | Run the script and prevent kV from rising when vIn rises | Log outputs "Changing vIn has minimal effect on kV. FAIL MONOBLOCK." | Expected operation verified See Appendix 5.3 Verified by EM 06SEP2024 | P |
| SRS-5.4 | The galden-verification-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% | Run the script and generate a Vsn and Vsp that differ by more than 10% | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." | Expected operation verified See Appendix 5.4 Verified by EM 06SEP2024 | P |
| SRS-5.5 | The galden-verification-script shall flag a monoblock if the operating frequency is below 460 kHz or above 800 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." | Expected operation verified See Appendix 5.5 Verified by EM 06SEP2024 Verified by EM 06SEP2024 | F |

### Table 12
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-553 |  |  |
