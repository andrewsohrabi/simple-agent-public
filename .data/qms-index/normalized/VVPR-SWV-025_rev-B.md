# VVPR-SWV-025 Rev B: MB Burn In Test Fixture Script Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-025
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-025 - MB Burn In Test Fixture Script Verification and Validation Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the MB-burnin-test-fixture-script meets usability and functional requirements as stated in MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify the Production Software Tool: MB-burnin-test-fixture-script for use in WS-015 for production of the monoblock used in the MX1 emitter.
REFERENCES
MEMO-P01-735 MB-burnin-test-fixture-script Software Requirements Specification Rev A
MWI-260 - MS-10579 Monoblock Encapsulated Assembly Verification Rev A
MWI-259 - WS-015 Workstation Installation Rev A
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.0.0-alpha
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
None
MATERIALS
WS-015 - MS-10579 Monoblock Encapsulated Assembly Verification (Ref MWI-259)
S10102 MB-burnin-test-fixture-script v1.0.0-alpha
MS-10579 Monoblock Encapsulated Assembly Rev A
Tulsa mb5
Tulsa mb13
SPD3303X Siglent Benchtop power supply (EQP-246 or equivalent)
RTB2004 Rhode and Schwarz Oscilloscope (EQP-121 or equivalent)
AFG31000 Rhode and Schwarz Arbitrary Function Generator (EQP-238 or equivalent)
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in MWI-259 to install WS-015.
Follow the steps outlined in each table below. MWI-260 should be used to guide operation of the workstation as needed.
Table 1: System Interfacing.
Table 2: Operation Requirements.
Table 3: Math Requirements.
Table 4: Plotting, Logging, and Saving Requirements.
Table 5: Flagging Requirements.
Discussion
Three of the requirements could not be tested at this time due to a lack of monoblocks available for testing. These requirements shall be tested prior to Phase 4 closure. Additionally there are two failure states that were not able to be properly detected, due to minor code errors. A beta release of this software is required to address those two software requirements.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Appendix 1: System Interfacing Requirements Evidence
Radcal Connection Evidence
MB test board connection evidence
Appendix 2: Operation Requirements Evidence
Filament Duty Cycle Dictation Evidence
Radcal Dose and kV evidence
Radcal mAs evidence
Radcal measured kV was found to be highest at freqN = 348. The measured kV at 346 and 350 were both higher, so this is confirmed:
Lower and upper duty cycle evidence
Excitation voltage dictation evidence
Not Evaluated
Not Evaluated
Not Evaluated
Appendix 3: Math Requirements Evidence
Oscilliscope H Bridge measurements
Frequency measurement: 528.74 Hz
Average max amplitude: 1.45 V (note the amplitude will need to be multiplied by 10.5932 to scale to actually amplitude of H bridge)
Average min amplitude: 58.6 mV
mA measurement evidence
kV measurement evidence
Arc logging evidence
Appendix 4: Plotting, Logging, and Saving Requirements Evidence
Appendix 5: Flagging Requirements Evidence
Failed. Error could not be generated
Failed. Error could not be generated
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The MB-burnin-test-fixture script shall interface with a DAQ card and detect if is not responding | Confirm that the log outputs "Connected to DAQ" | Log outputs "Connected to DAQ" |  |  |
| SRS-1.2 | The MB-burnin-test-fixture script shall interface with a Radcal and detect if it is not responding before increasing above 40 kV | 1. Confirm that the log outputs "Connected to Radcal" 2. Confirm that the log outputs a dose measurement before 40 kV is achieved or that the program throws an error. 3. If a radcal time out error has not occurred yet, unplug the USB from the Radcal, run the script again, and confirm that the log outputs "Radcal timed out..." | 1. Log outputs "Connected to Radcal" 2. Log outputs dose measurement before reaching 40 kV 3. Log outputs "Radcal timed out..." if not connected |  |  |
| SRS-1.3 | The MB-burnin-test-fixture script shall interface with a test board and detect if it is not responding | 1. Confirm that the log outputs "Connected to Test Board" 2. Confirm that the log outputs "Test board looks good. Continuing..." 3. If a serial port error has not yet occurred, unplug the COM connector USB from the workstation PC, run the script again, and verify the log outputs "Could not open serial port. Exiting". Plug the USB back into the computer 4. If a test board not responding error has not yet occurred, turn off the power supply, run the script again and verify the log outputs "Test board not responding." | 1. Log outputs "Connected to Test Board" 2. Log outputs "Test board looks good." 3. Log outputs "Could not open serial port." 4. Log outputs "Test board not responding." |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The MB-burnin-test-fixture script shall dictate which filament duty cycle the test board will use to power the filament | 1. Wait until the log outputs "Finding Peak for Filament Duty of 24%" 2. Wait until the program completes the first exposure at duty cycle 2 | 1. For the first duty cycle, the value in the log output reading "Filament Duty Cycle = {value}%" should be equal to the last number in the TX vector in the log output divided by 10 2. For the second duty cycle, the value in the log output reading "Filament Duty Cycle = {value}%" should be equal to the last value in the TX vector in the log output divided by 10 |  |  |
| SRS-2.2 | The MB-burnin-test-fixture script shall monitor kV and dose rate from a Radcal for exposures high enough to trigger the Radcal | Allow the program to achieve a technique which outputs at least 47 kV from the monoblock | 1. Log outputs "Radcal Measurement Finished" followed by "Radcal Measured Dose" and "Radcal Measured Average Tube Voltage" 2. Plot titled "Radcal kV and Dose Rate Measurements" contains waveform for both the Dose Rate and kV |  |  |
| SRS-2.3 | The MB-burnin-test-fixture script shall monitor mA from a Radcal for exposures high enough to trigger the Radcal if the mAs sensor is connected | Allow the program to achieve a technique which outputs at least 30 kV from the monoblock | 1. Log outputs "Radcal Measurement Finished" followed by "Radcal Measured Average Beam Current" 2. Plot titled "Radcal mA Measurements" contains the waveform for mA |  |  |
| SRS-2.4 | The MB-burnin-test-fixture script shall monitor dose from a Radcal for exposures high enough to trigger the Radcal | Allow the program to achieve a technique which outputs at least 30 kV from the monoblock | 1. Log outputs "Radcal Measurement Finished" followed by "Radcal Measured Dose" 2. A number is present in the log next to "Radcal Measured Dose" and is a reasonable value (between .001 and 2 mGy) |  |  |
| SRS-2.5 | The MB-burnin-test-fixture script shall locate the operating frequeny for each filament duty cycle tested | Wait until the log outputs "++++ Peak Found ++++" twice | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle for both duty cycles tested |  |  |
| SRS-2.6 | The MB-burnin-test-fixture script shall determine the range of filament duty cycles necessary to to achieve 50 kV (±1%), .5 mA (±2%) and 50 kV (±1%), 2.2 mA (±1%) | Wait until the log outputs "+++ UPPER FILAMENT DUTY CYCLE FOUND +++" | 1. Under "+++ LOWER FILAMENT DUTY CYCLE FOUND +++" the following is true: 1.1. mA is within 2% of 0.5 1.2. kV is within 1% of 50 2. Under "+++ UPPER FILAMENT DUTY CYCLE FOUND +++" the following is true: 2.1. mA is within 1% of 2.2 2.2. kV is within 1% of 50 |  |  |
| SRS-2.7 | The MB-burnin-test-fixture script shall dictate the excitation voltage to generate high voltage potential on the monoblock outputs | 1. Wait until the log outputs "####### Changing Peak V to 50 kV #######" 2. Wait until the program takes 4 more exposures | 1. For each of the 4 exposures, the value in the log output reading "vIn = {value} V" is equal to the first number in the TX vector in the log output divided by 10 2. vIn increases with each subsequent exposure |  |  |
| SRS-2.8 | The MB-burnin-test-fixture script shall generate a model to predict duty cycle and H-bridge frequency to achieve 83 kV, 2.1 mA | Wait until the log outputs "----- Beginning Burn In -----" | 1. Log outputs "++++ Richardson's Curve for 83 has been found ++++" 2. Log outputs "++++ Predicted Duty Cycle and FreqN ++++" 3. Log outputs "filDuty = {value1}" and "freqN = {value2}" |  |  |
| SRS-2.9 | The MB-burnin-test-fixture script shall approach 83 kV (±1%), 2.1 mA (±5%) without overshooting kV by more than 2 kV | Wait until the log outputs "++++ Initial Inputs Used For Burn In ++++" | 1. In the exposure preceding "++++ Initial Inputs Used For Burn In ++++": 1.1. kV is within 1% of 83 1.2. mA is within 5% of 2.1 2. The kV for all of the other expsures for that duty cycle do not exceed 85 kV |  |  |
| SRS-2.10 | The MB-burnin-test-fixture script shall repeatedly stress the monoblock for 3600 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 88 kV, in which case vIn will drop by .2 V |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Oscilloscope, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, power supply is off |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.1 | The MB-burnin-test-fixture script shall calculate average kV from a Radcal for exposures high enough to trigger the Radcal | 1. From the previous tests, determine a duty cycle and exposure number which was high enough to trigger the radcal 2. Inside the .hdf file associated with the test mentioned in step 1, navigate to the duty cycle and exposure directories for the duty cycle and exposure determined in step 1 3. The radcal kV data is located in the radcal_waves directory and is titled voltage_wave 4. The radcal mA data is located in the radcal_waves directory and is titled current_wave | Calculated mean of the second half of the kV data is within 5% of radcal kV measurement outputted to log for corresponding duty cycle and exposure |  |  |
| SRS-3.2 | The MB-burnin-test-fixture script shall calculate average mA from a Radcal for exposures high enough to trigger the Radcal if the mAs sensor is connected |  | Calculated mean measurement of the second half of the mA data is within 5% of radcal mA measurement outputted to log for corresponding duty cycle and exposure |  |  |
| SRS-3.3 | The MB-burnin-test-fixture script shall analyze H Bridge voltage waveform for H bridge amplitude and driving frequency | 1. Connect the SMA-BNC for the H-bridge to the DAQ through a T connector and connect the open end of the T connector to channel 1 of the oscilloscope 2. Connect a monoblock to WS-015 3. Setup the trigger on the oscilloscope 4. Take one shot with the program 5. Use the oscilloscope to measure the frequency and average peak to peak amplitude | The measurements on the oscilloscope are within 10% of the measurements on the program log |  |  |
| SRS-3.4 | The MB-burnin-test-fixture script shall analyze beam current and calculate steady state average | 1. From the previous tests, determine a duty cycle and exposure number which was high enough to trigger the radcal using the .log file 2. Locate duty cycle/exposure combos which resulted in a radcal mA reading below 1 mA and above 1 mA | The percent difference between the Radcal mA measurement and the board measurment shall be: < 5.5% for mA < 1 < 2.5% for mA > 1 |  |  |
| SRS-3.5 | The MB-burnin-test-fixture script shall analyze Vsn and Vsp to calculate tube voltage and steady state average | 1. From the previous tests, determine a duty cycle and exposure number which was high enough to trigger the radcal using the .log file 2. Locate duty cycle/exposure combos which resulted in a Radcal kV reading around 50 and around 70 | The percent difference between the Radcal kV measurement and the board measurement shall be < 5% |  |  |
| SRS-3.6 | The MB-burnin-test-fixture script shall detect arcs on the positive and negative Vsense lines which deviate more than 6% from the average voltage | Run the script and force and arc to occur | The script detects that an arc occurred, throws a warning, and pauses the program |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a functional monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.1 | The MB-burnin-test-fixture script shall plot the processed tube potential, average steady state tube potential, beam current, and average steady state beam current | Wait until the program shoots an x-ray | 1. Figure with 2 subplots appears 1.a. One subplot is titled "Tube Potential" and contains a legend and 2 waveforms: one for tube voltage and one for average tube voltage, as indicated by the legend 1.b. One subplot is titled "Beam Current" and contains a legend and 2 waveforms: one for beam current and one for average average beam current, as indicated by the legend |  |  |
| SRS-4.2 | The MB-burnin-test-fixture script shall plot kV and dose rate from a Radcal for exposures high enough to trigger the Radcal | Allow the program to achieve a technique which outputs at least 47 kV from the monoblock | Plot titled "Radcal kV and Dose Rate Measurements" contains waveform for both the Dose Rate and kV |  |  |
| SRS-4.3 | The MB-burnin-test-fixture script shall plot mA from a Radcal for exposures high enough to trigger the Radcal if the mAs sensor is connected | Allow the program to achieve a technique which outputs at least 30 kV from the monoblock | Plot titled "Radcal mA Measurements" contains the waveform for mA |  |  |
| SRS-4.4 | The MB-burnin-test-fixture script shall log the input voltage, frequency, and filament duty cycle used for the next exposure | Wait until the program shoots an x-ray | 1. log outputs a line containing "vIn = {value1} V, FreqN = {value2}, Freq = {value3} kHz, Filament Duty Cycle = {value4}%" where the values are numbers greater than 0. 2. Log outputs the line in step 2 before outputting "shooting x-ray" |  |  |
| SRS-4.5 | The MB-burnin-test-fixture script shall log commands sent to the MB test board | Wait until the program shoots an x-ray | 1. In the log, there are 4 lines starting with "TX:" 2. In the log, there are 4 lines starting with "RX:" |  |  |
| SRS-4.6 | The MB-burnin-test-fixture script shall log measurements made on the analyzed waveforms | Wait until the program shoots an x-ray | 1. Log outputs a line which reads "Measurements: Vsp: {value1} V, Vsn: {value2} V" where the values are numbers 2. Log outputs a line which reads "Rise time = {value} ms where value is a number 3. Log outputs a line which reads "Driving Frequency = {value} kHz" where value is a number 4. Log outputs a line which reads "H bridge voltage amplitude = {value} V" where value is a number 5. Log outputs a line which reads "Average beam current = {value} mA" where value is a number 6. Log outputs a line which reads "Extrapolated Average Tube Voltage: {value} kV" where value is a number |  |  |
| SRS-4.7 | The MB-burnin-test-fixture script shall log a warning if an arc is detected | Run the script and force and arc to occur | Log outputs a warning which reads "!!! Potential Arc Detected !!!" |  |  |
| SRS-4.8 | The MB-burnin-test-fixture script shall save the waveforms and measurement results for every exposure taken | 1. Wait until the program shoots 3 x-rays. 2. Open the hdf file using an hdf viewer | 1. hdf file contains 1 folder labeled "DutyCycle1 2. The DutyCycle1 folder contains 3 folders named "Exposure1," "Exposure2," and "Exposure3" 3. Each Exposure folder contains a dataset named "HV," "Vsn," "Vsp," "beamI," "time", and "SettingsAndMsrmts" |  |  |
| SRS-4.9 | The MB-burnin-test-fixture script shall save the Radcal waveforms for exposures high enough to trigger the Radcal | 1. Wait until the program shoots 3 x-rays in a row that are above 45 kV 2. Open the hdf file using an hdf viewer 3. Navigate to the appropriate duty cycle number and exposure number | 1. Each Exposure folder for the exposures high enough to trigger the Radcal contain a folder named "radcal_waves" 2. Each "radcal_waves" folder contain datasets named "current_wave," "rate_wave," "voltage_wave," and" "time" |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The MB-burnin-test-fixture-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." |  |  |
| SRS-5.2 | The MB-burnin-test-fixture-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." |  |  |
| SRS-5.3 | The MB-burnin-test-fixture-script shall flag a monoblock if the rate of change in tube potential is not positive with the rate of change in input voltage, if the input voltage has increased at least 15%. | Run the script and prevent kV from rising when vIn rises | Log outputs "Changing vIn has minimal effect on kV. FAIL MONOBLOCK." |  |  |
| SRS-5.4 | The MB-burnin-test-fixture-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% | Run the script and generate a Vsn and Vsp that differ by more than 10% | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." |  |  |
| SRS-5.5 | The MB-burnin-test-fixture-script shall flag a monoblock if the operating frequency is below 400 kHz or above 650 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." |  |  |
| SRS-5.6 | The MB-burnin-test-fixture-script shall flag a monoblock if the duty cycle required to achieve 50 kV, .5 mA is below 10% | Run the script and force mA to peak higher than .7 mA for every duty cycle tested | Log outputs "{suggested duty cycle}% filament duty cycle is too low. Flag monoblock and send MedAI .log and .hdf files." |  |  |
| SRS-5.7 | The MB-burnin-test-fixture-script shall flag a monoblock if the duty cycle required to achieve 50 kV, 2.2 mA is above 45% | Run the script and force mA to peak lower than 2 mA for every duty cycle tested | Log outputs "{suggested duty cycle}% filament duty cycle is too high. Flag monoblock and send MedAI .log and .hdf files." |  |  |
| SRS-5.8 | The MB-burnin-test-fixture-script shall fail a monoblock if the dose after stressing the monoblock has changed by more than 10% | 1. Run the stress test using 80 kV, 2 mA and set the number of shots to 3 2. Change the technique for the last shot to be more than 10% lower dose than the the first 2 shots | Log outputs "Final dose ({final dose value} mGy) has drifted more than 10% from initial ({initial dose value} mGy). FAIL MONOBLOCK." |  |  |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-542 |  |  |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The MB-burnin-test-fixture script shall interface with a DAQ card and detect if is not responding | Confirm that the log outputs "Connected to DAQ" | Log outputs "Connected to DAQ" | Expected outcome verified See Appendix 1.1 Verified by MI 04SEP2024 | P |
| SRS-1.2 | The MB-burnin-test-fixture script shall interface with a Radcal and detect if it is not responding before increasing above 40 kV | 1. Confirm that the log outputs "Connected to Radcal" 2. Confirm that the log outputs a dose measurement before 40 kV is achieved or that the program throws an error. 3. If a radcal time out error has not occurred yet, unplug the USB from the Radcal, run the script again, and confirm that the log outputs "Radcal timed out..." | 1. Log outputs "Connected to Radcal" 2. Log outputs dose measurement before reaching 40 kV 3. Log outputs "Radcal timed out..." if not connected | Expected outcome verified See Appendix 1.2 Verified by MI 04SEP2024 | P |
| SRS-1.3 | The MB-burnin-test-fixture script shall interface with a test board and detect if it is not responding | 1. Confirm that the log outputs "Connected to Test Board" 2. Confirm that the log outputs "Test board looks good. Continuing..." 3. If a serial port error has not yet occurred, unplug the COM connector USB from the workstation PC, run the script again, and verify the log outputs "Could not open serial port. Exiting". Plug the USB back into the computer 4. If a test board not responding error has not yet occurred, turn off the power supply, run the script again and verify the log outputs "Test board not responding." | 1. Log outputs "Connected to Test Board" 2. Log outputs "Test board looks good." 3. Log outputs "Could not open serial port." 4. Log outputs "Test board not responding." | Expected outcome verified See Appendix 1.3 Verified by MI 04SEP2024 | P |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The MB-burnin-test-fixture script shall dictate which filament duty cycle the test board will use to power the filament | 1. Wait until the log outputs "Finding Peak for Filament Duty of 24%" 2. Wait until the program completes the first exposure at duty cycle 2 | 1. For the first duty cycle, the value in the log output reading "Filament Duty Cycle = {value}%" should be equal to the last number in the TX vector in the log output divided by 10 2. For the second duty cycle, the value in the log output reading "Filament Duty Cycle = {value}%" should be equal to the last value in the TX vector in the log output divided by 10 | Expected outcome verified See Appendix 2.1 Verified by MI 04SEP2024 | P |
| SRS-2.2 | The MB-burnin-test-fixture script shall monitor kV and dose rate from a Radcal for exposures high enough to trigger the Radcal | Allow the program to achieve a technique which outputs at least 47 kV from the monoblock | 1. Log outputs "Radcal Measurement Finished" followed by "Radcal Measured Dose" and "Radcal Measured Average Tube Voltage" 2. Plot titled "Radcal kV and Dose Rate Measurements" contains waveform for both the Dose Rate and kV | Expected outcome verified See Appendix 2.2 Verified by MI 04SEP2024 | P |
| SRS-2.3 | The MB-burnin-test-fixture script shall monitor mA from a Radcal for exposures high enough to trigger the Radcal if the mAs sensor is connected | Allow the program to achieve a technique which outputs at least 30 kV from the monoblock | 1. Log outputs "Radcal Measurement Finished" followed by "Radcal Measured Average Beam Current" 2. Plot titled "Radcal mA Measurements" contains the waveform for mA | Expected outcome verified See Appendix 2.3 Verified by MI 04SEP2024 | P |
| SRS-2.4 | The MB-burnin-test-fixture script shall monitor dose from a Radcal for exposures high enough to trigger the Radcal | Allow the program to achieve a technique which outputs at least 30 kV from the monoblock | 1. Log outputs "Radcal Measurement Finished" followed by "Radcal Measured Dose" 2. A number is present in the log next to "Radcal Measured Dose" and is a reasonable value (between .001 and 2 mGy) | Expected outcome verified See Appendix 2.4 Verified by MI 04SEP2024 | P |
| SRS-2.5 | The MB-burnin-test-fixture script shall locate the operating frequeny for each filament duty cycle tested | Wait until the log outputs "++++ Peak Found ++++" twice | The freqN in the log output under "++++ Peak Found ++++" matches the freqN associated with the highest kV in the section of the log that reads "Refinding peak after raising kV" for the corresponding duty cycle for both duty cycles tested | Expected outcome verified See Appendix 2.5 Verified by MI 04SEP2024 | P |
| SRS-2.6 | The MB-burnin-test-fixture script shall determine the range of filament duty cycles necessary to to achieve 50 kV (±1%), .5 mA (±2%) and 50 kV (±1%), 2.2 mA (±1%) | Wait until the log outputs "+++ UPPER FILAMENT DUTY CYCLE FOUND +++" | 1. Under "+++ LOWER FILAMENT DUTY CYCLE FOUND +++" the following is true: 1.1. mA is within 2% of 0.5 1.2. kV is within 1% of 50 2. Under "+++ UPPER FILAMENT DUTY CYCLE FOUND +++" the following is true: 2.1. mA is within 1% of 2.2 2.2. kV is within 1% of 50 | 1. mA is 1.2% off of 0.5 and kV is .14% off of 50 kV. See Appendix 2.6.1 2.  mA is .36% off of 2.2 mA and kV is .72% off of 50 kV. See Appendix 2.6.1 Verified by MI 04SEP2024 | P |
| SRS-2.7 | The MB-burnin-test-fixture script shall dictate the excitation voltage to generate high voltage potential on the monoblock outputs | 1. Wait until the log outputs "####### Changing Peak V to 50 kV #######" 2. Wait until the program takes 4 more exposures | 1. For each of the 4 exposures, the value in the log output reading "vIn = {value} V" is equal to the first number in the TX vector in the log output divided by 10 2. vIn increases with each subsequent exposure | Expected outcome verified See Appendix 2.7 Verified by MI 04SEP2024 | P |
| SRS-2.8 | The MB-burnin-test-fixture script shall generate a model to predict duty cycle and H-bridge frequency to achieve 83 kV, 2.1 mA | Wait until the log outputs "----- Beginning Burn In -----" | 1. Log outputs "++++ Richardson's Curve for 83 has been found ++++" 2. Log outputs "++++ Predicted Duty Cycle and FreqN ++++" 3. Log outputs "filDuty = {value1}" and "freqN = {value2}" | Could not be evaluated without a functional monoblock. Requirement shall be evaluator prior to Phase 4 closure. | NE |
| SRS-2.9 | The MB-burnin-test-fixture script shall approach 83 kV (±1%), 2.1 mA (±5%) without overshooting kV by more than 2 kV | Wait until the log outputs "++++ Initial Inputs Used For Burn In ++++" | 1. In the exposure preceding "++++ Initial Inputs Used For Burn In ++++": 1.1. kV is within 1% of 83 1.2. mA is within 5% of 2.1 2. The kV for all of the other expsures for that duty cycle do not exceed 85 kV | Could not be evaluated without a functional monoblock. Requirement shall be evaluator prior to Phase 4 closure. | NE |
| SRS-2.10 | The MB-burnin-test-fixture script shall repeatedly stress the monoblock for 3600 exposures | 1. Wait until the log outputs "****** Commencing Stress Test ******" 2. Wait until the program takes 100 exposures | 1. freqN and duty cycle for all exposures are the same 2. vIn for all exposures are the same unless kV exceeds 88 kV, in which case vIn will drop by .2 V | Could not be evaluated without a functional monoblock. Requirement shall be evaluator prior to Phase 4 closure. | NE |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Oscilloscope, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, power supply is off |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.1 | The MB-burnin-test-fixture script shall calculate average kV from a Radcal for exposures high enough to trigger the Radcal | 1. From the previous tests, determine a duty cycle and exposure number which was high enough to trigger the radcal 2. Inside the .hdf file associated with the test mentioned in step 1, navigate to the duty cycle and exposure directories for the duty cycle and exposure determined in step 1 3. The radcal kV data is located in the radcal_waves directory and is titled voltage_wave 4. The radcal mA data is located in the radcal_waves directory and is titled current_wave | Calculated mean of the second half of the kV data is within 5% of radcal kV measurement outputted to log for corresponding duty cycle and exposure | The calculated average was 50.348, so the percent difference is less than 0.1%. See Appendix 3.1 Verified by MI 04SEP2024 | P |
| SRS-3.2 | The MB-burnin-test-fixture script shall calculate average mA from a Radcal for exposures high enough to trigger the Radcal if the mAs sensor is connected |  | Calculated mean measurement of the second half of the mA data is within 5% of radcal mA measurement outputted to log for corresponding duty cycle and exposure | The calculated average was 1.354, so the percent difference is 0.15%. See Appendix 3.2 Verified by MI 04SEP2024 | P |
| SRS-3.3 | The MB-burnin-test-fixture script shall analyze H Bridge voltage waveform for H bridge amplitude and driving frequency | 1. Connect the SMA-BNC for the H-bridge to the DAQ through a T connector and connect the open end of the T connector to channel 1 of the oscilloscope 2. Connect a monoblock to WS-015 3. Setup the trigger on the oscilloscope 4. Take one shot with the program 5. Use the oscilloscope to measure the frequency and average peak to peak amplitude | The measurements on the oscilloscope are within 10% of the measurements on the program log | Percent difference is 1.05% for frequency and 5.96% for amplitude See Appendix 3.3 Verified by MI 04SEP2024 | P |
| SRS-3.4 | The MB-burnin-test-fixture script shall analyze beam current and calculate steady state average | 1. From the previous tests, determine a duty cycle and exposure number which was high enough to trigger the radcal using the .log file 2. Locate duty cycle/exposure combos which resulted in a radcal mA reading below 1 mA and above 1 mA | The percent difference between the Radcal mA measurement and the board measurment shall be: < 5.5% for mA < 1 < 2.5% for mA > 1 | Percent difference is 3.3% for less than 1 mA and 2.2% for greater than 1 mA . See Appendix 3.4 Verified by MI 04SEP2024 | P |
| SRS-3.5 | The MB-burnin-test-fixture script shall analyze Vsn and Vsp to calculate tube voltage and steady state average | 1. From the previous tests, determine a duty cycle and exposure number which was high enough to trigger the radcal using the .log file 2. Locate duty cycle/exposure combos which resulted in a Radcal kV reading around 50 and around 70 | The percent difference between the Radcal kV measurement and the board measurement shall be < 5% | Percent difference is 1.88% for 50 kV and 1.75% for 70 kV See Appendix 3.5 Verified by MI 04SEP2024 | P |
| SRS-3.6 | The MB-burnin-test-fixture script shall detect arcs on the positive and negative Vsense lines which deviate more than 6% from the average voltage | Run the script and force and arc to occur | The script detects that an arc occurred, throws a warning, and pauses the program | Expected outcome verified. See Appendix 3.6 Verified by MI 04SEP2024 | P |

### Table 10
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/ Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Functional Monoblock, WS-015 |  |  |  |  |
| Test Setup: | WS-015 is installed, including the Radcal mAs sensor, and a functional monoblock has been correctly installed in T-183 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) 4. Run the script using: python main.py |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-4.1 | The MB-burnin-test-fixture script shall plot the processed tube potential, average steady state tube potential, beam current, and average steady state beam current | Wait until the program shoots an x-ray | 1. Figure with 2 subplots appears 1.a. One subplot is titled "Tube Potential" and contains a legend and 2 waveforms: one for tube voltage and one for average tube voltage, as indicated by the legend 1.b. One subplot is titled "Beam Current" and contains a legend and 2 waveforms: one for beam current and one for average average beam current, as indicated by the legend | Expected outcome verified. See Appendix 4.1 Verified by MI 04SEP2024 | P |
| SRS-4.2 | The MB-burnin-test-fixture script shall plot kV and dose rate from a Radcal for exposures high enough to trigger the Radcal | Allow the program to achieve a technique which outputs at least 47 kV from the monoblock | Plot titled "Radcal kV and Dose Rate Measurements" contains waveform for both the Dose Rate and kV | Expected outcome verified. See Appendix 4.2 Verified by MI 04SEP2024 | P |
| SRS-4.3 | The MB-burnin-test-fixture script shall plot mA from a Radcal for exposures high enough to trigger the Radcal if the mAs sensor is connected | Allow the program to achieve a technique which outputs at least 30 kV from the monoblock | Plot titled "Radcal mA Measurements" contains the waveform for mA | Expected outcome verified. See Appendix 4.3 Verified by MI 04SEP2024 | P |
| SRS-4.4 | The MB-burnin-test-fixture script shall log the input voltage, frequency, and filament duty cycle used for the next exposure | Wait until the program shoots an x-ray | 1. log outputs a line containing "vIn = {value1} V, FreqN = {value2}, Freq = {value3} kHz, Filament Duty Cycle = {value4}%" where the values are numbers greater than 0. 2. Log outputs the line in step 2 before outputting "shooting x-ray" | Expected outcome verified. See Appendix 4.4 Verified by MI 04SEP2024 | P |
| SRS-4.5 | The MB-burnin-test-fixture script shall log commands sent to the MB test board | Wait until the program shoots an x-ray | 1. In the log, there are 4 lines starting with "TX:" 2. In the log, there are 4 lines starting with "RX:" | Expected outcome verified. See Appendix 4.5 Verified by MI 04SEP2024 | P |
| SRS-4.6 | The MB-burnin-test-fixture script shall log measurements made on the analyzed waveforms | Wait until the program shoots an x-ray | 1. Log outputs a line which reads "Measurements: Vsp: {value1} V, Vsn: {value2} V" where the values are numbers 2. Log outputs a line which reads "Rise time = {value} ms where value is a number 3. Log outputs a line which reads "Driving Frequency = {value} kHz" where value is a number 4. Log outputs a line which reads "H bridge voltage amplitude = {value} V" where value is a number 5. Log outputs a line which reads "Average beam current = {value} mA" where value is a number 6. Log outputs a line which reads "Extrapolated Average Tube Voltage: {value} kV" where value is a number | Expected outcome verified. See Appendix 4.6 Verified by MI 04SEP2024 | P |
| SRS-4.7 | The MB-burnin-test-fixture script shall log a warning if an arc is detected | Run the script and force and arc to occur | Log outputs a warning which reads "!!! Potential Arc Detected !!!" | Expected outcome verified. See Appendix 4.7 Verified by MI 04SEP2024 | P |
| SRS-4.8 | The MB-burnin-test-fixture script shall save the waveforms and measurement results for every exposure taken | 1. Wait until the program shoots 3 x-rays. 2. Open the hdf file using an hdf viewer | 1. hdf file contains 1 folder labeled "DutyCycle1 2. The DutyCycle1 folder contains 3 folders named "Exposure1," "Exposure2," and "Exposure3" 3. Each Exposure folder contains a dataset named "HV," "Vsn," "Vsp," "beamI," "time", and "SettingsAndMsrmts" | Expected outcome verified. See Appendix 4.8 Verified by MI 04SEP2024 | P |
| SRS-4.9 | The MB-burnin-test-fixture script shall save the Radcal waveforms for exposures high enough to trigger the Radcal | 1. Wait until the program shoots 3 x-rays in a row that are above 45 kV 2. Open the hdf file using an hdf viewer 3. Navigate to the appropriate duty cycle number and exposure number | 1. Each Exposure folder for the exposures high enough to trigger the Radcal contain a folder named "radcal_waves" 2. Each "radcal_waves" folder contain datasets named "current_wave," "rate_wave," "voltage_wave," and" "time" | Expected outcome verified. See Appendix 4.9 Verified by MI 04SEP2024 | P |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Arbitrary function generator, power supply, oscilloscope, WS-015, functional monoblock |  |  |  |  |
| Test Setup: | WS-015 is installed 1. Open windows powershell (or other terminal window) 2. Navigate to the script folder by typing the following line and pressing ENTER. cd C:\Users\MedAI\Documents\MB-burnin-test-fixture-script 3. Activate the virtual environment: .venv\Scripts\Activate.ps1 (or other command depending on terminal environment) |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-5.1 | The MB-burnin-test-fixture-script shall fail a monoblock if 3 arcs are detected | Run the script and force 3 arcs to occur | Log outputs "Monoblock has arced 3 times. FAIL MONOBLOCK." | Error could not be generated due to an issue in the script. Verified by MI 04SEP2024 | F |
| SRS-5.2 | The MB-burnin-test-fixture-script shall flag a monoblock if the average tube voltage rises above 70 kV when the minimum input voltage for a particular duty cycle is applied | Run the script and generate a kV greater than 70 kV while searching for a resonant peak | Log outputs "Overvoltage in peak detector. FAIL MONOBLOCK." | Expected outcome verified. See Appendix 5.2 Verified by MI 04SEP2024 | P |
| SRS-5.3 | The MB-burnin-test-fixture-script shall flag a monoblock if the rate of change in tube potential is not positive with the rate of change in input voltage, if the input voltage has increased at least 15%. | Run the script and prevent kV from rising when vIn rises | Log outputs "Changing vIn has minimal effect on kV. FAIL MONOBLOCK." | Expected outcome verified. See Appendix 5.3 Verified by MI 04SEP2024 | P |
| SRS-5.4 | The MB-burnin-test-fixture-script shall fail a monoblock if the absolute value of Vsn and Vsp voltages differ by more the 10% | Run the script and generate a Vsn and Vsp that differ by more than 10% | Log outputs "Difference between Vsp and Vsn is larger than 10% {actual percent difference}. FAIL MONOBLOCK." | Expected outcome verified. See Appendix 5.4 Verified by MI 04SEP2024 | P |
| SRS-5.5 | The MB-burnin-test-fixture-script shall flag a monoblock if the operating frequency is below 400 kHz or above 650 kHz | Run the script and prevent kV from rising as frequency changes | Log outputs "Resonant frequency not within bounds. FAIL MONOBLOCK." | Error could not be generated due to an issue in the script. Verified by MI 04SEP2024 | F |
| SRS-5.6 | The MB-burnin-test-fixture-script shall flag a monoblock if the duty cycle required to achieve 50 kV, .5 mA is below 10% | Run the script and force mA to peak higher than .7 mA for every duty cycle tested | Log outputs "{suggested duty cycle}% filament duty cycle is too low. Flag monoblock and send MedAI .log and .hdf files." | Expected outcome verified. See Appendix 5.6 Verified by MI 04SEP2024 | P |
| SRS-5.7 | The MB-burnin-test-fixture-script shall flag a monoblock if the duty cycle required to achieve 50 kV, 2.2 mA is above 45% | Run the script and force mA to peak lower than 2 mA for every duty cycle tested | Log outputs "{suggested duty cycle}% filament duty cycle is too high. Flag monoblock and send MedAI .log and .hdf files." | Expected outcome verified. See Appendix 5.7 Verified by MI 04SEP2024 | P |
| SRS-5.8 | The MB-burnin-test-fixture-script shall fail a monoblock if the dose after stressing the monoblock has changed by more than 10% | 1. Run the stress test using 80 kV, 2 mA and set the number of shots to 3 2. Change the technique for the last shot to be more than 10% lower dose than the the first 2 shots | Log outputs "Final dose ({final dose value} mGy) has drifted more than 10% from initial ({initial dose value} mGy). FAIL MONOBLOCK." | Expected outcome verified. See Appendix 5.8 Verified by MI 04SEP2024 | P |

### Table 12
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-553 |  |  |
