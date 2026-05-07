# VVPR-SWV-026 Rev B: Monoblock Test Fixture Firmware Verification and Validation Protocol and Report

## Metadata
- Document ID: VVPR-SWV-026
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-SWV-026 Monoblock Test Fixture Firmware Verification and Validation Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-026 Monoblock Test Fixture Firmware Verification and Validation Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that the firmware for the Monoblock Test Fixture PCBA (ES-10035) meets usability and functional requirements as outlined in MEMO-P01-734 MB-burnin-test-fixture Firmware Requirements Specification Rev. A.
OBJECTIVE
The primary objective of this study is to verify the ES-10035 Rev A for use in Workstation 015.
REFERENCES
MEMO-P01-734 Rev. A MB-burnin-test-fixture Firmware Requirements Specification
MATERIALS
SAMPLE SIZE
This is a firmware verification test, and therefore will use a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in each table below.
Data Analysis
All of the verification tests in Tables 1 through 5 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 through 3 per the expected results documented in the “Expected Result/Pass Criteria” column.
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Report Section
Deviations
During the course of testing changes to  pySerial_wrapper_verbose.py  were necessary to put the board into an error state. However, no changes were made directly to the firmware. The helper script is not a part of the firmware and is outside of the scope of V&V.
MATERIALS
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified
DISCUSSION
No anomalies were found during the course of testing. All requirements passed with expected results.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Appendix 1 System Interfacing Evidence
1.1 RS-232 Communication
Appendix 2 Operation Requirements Evidence
2.1 X-Ray PWS Control
Fluke DMM measurements, 16.06V
Fluke DMM measurement 79.6V
2.2 Duty Cycle Control
2.3 Frequency Control
2.4 Pulse Length Control
2.5 ADC Measurements
2.6 Software Interlock
Appendix 3 Error Requirements Evidence
3.1 Incompatible Hardware Error
3.2 Incorrect passkey Error
3.3 PWS Undervoltage Error
3.4 Filament Ramping Error
3.5 Filament Open Error
3.6 Input Range Error
3.7 Command Not Found Error
3.8 Temp Error
3.9 ADC Monitor Error
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| MFG P/N | MFG | Description | Notes |
| --- | --- | --- | --- |
| ES-10035 Rev A | MedAI | Monoblock Test Fixture PCBA |  |
| ES-10035 Rev 1 | MedAI | Monoblock Test Fixture PCBA | Used for non-compatible hardware verification |
| ES-10024 Rev A | MedAI | Monoblock LV PWS Rider |  |
| E11046 Rev A | MedAI | Monoblock HV PCB |  |
| DT-6552-1.8M | DTech | RS-232/USB Cable |  |
| RTB2004 | Rhode and Schwartz | Oscilloscope | EQP-121 or equivalent |
| SPD3303X | Siglent | Benchtop Power Supply | EQP-246 or equivalent |
| HSA50R50J | TE Connectivity | 50 Ohm Power Resistor |  |
| 17B+ | Fluke | Digital Multimeter | EQP-239 or equivalent |
| ST-LINK/V2 | STMicroelectronics | J-LINK Programmer |  |
| ‎2688-20 | Milwaukee | Heat Gun |  |
| S10104 | MedAI | MB-burnin-test-fixture Firmware | v1.0.0-alpha |

### Table 2
| Table 1. | System Interfacing |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Device Components Needed: | DC Power Supply ES-10035 Rev A RS-232/USB Cable PC |  |  |  |  |
| Test Setup: | 1. Connect RS-232 Cable to PC 2. Run the python script: pySerial_wrapper_verbose.py without powering on the device 3. Turn on the DC Power Supply set to 24V |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The MB-burnin-test-fixture firmware shall interface with a PC over RS-232 | 1. Confirm the log outputs: DEBUG:COMX - USB Serial Port (COMX) INFO: COMX opened Where COMX is the COM port number of the RS-232 Cable. If the log reads: CRITICAL: Could not open the serial port. Exiting Check to see that the cable is correctly connected to the PC, or potentially swap cables 2. Power on the device, after a second press enter in the terminal | 1. The log outputs text confirming that a COM port has been opened 2. After powering on the device, and pressing enter, the terminal displays the text "BOOT" or "B" '79' '79' '84' 3. If no text appears, press the reset button on the board, and press enter in the terminal again |  |  |

### Table 3
| Table 2. | Operation Requirements |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Device Components Needed: | DC Power Supply ES-10035 Rev A ES-10024 Rev A RS-232/USB Cable, and PC HV board with filament load 50ohm load for H-Bridge SMA to BNC cable Oscilloscope Multimeter |  |  |  |  |
| Test Setup: | 1. Connect RS-232 Cable to PC 2. Connect C1 on the oscilloscope to J25 using an SMA to BNC cable. Set a measurement on the oscilloscope to measure frequency on C1 3. Connect the HV board to the 10 pin connector on the PCBA. Connect C2 on the oscilloscope to the leads going to the first transformer on the filament load. Set a measurement on the oscilloscope to measure frequency on C2 4. Connect ES-10024 to the board on the 24 pin connector 5. Run the python script: pySerial_wrapper_verbose.py 6. Turn on the Power Supply set to 24V |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The MB-burnin-test-fixture firmware shall be capable of adjusting monoblock input voltage from 16V to 80V | 1. Issue a T command with the following parameters: Voltage: 160 Frequency: 3500 Duty: 180 2. Issue I Command 3. Measure X-RAY rail voltage 5. Issue I Command 6. Issue a T command with the following parameters: Voltage: 800 Frequency: 3500 Duty: 180 7. Issue I Command 8. Measure X-RAY rail voltage | 1. First DMM measurement of rail voltage should be 16V +/- 1V 2. Second DMM measurement of rail voltage should be 80V +/- 1V |  |  |
| SRS-2.2 | The MB-burnin-test-fixture firmware shall be capable of adjusting filament duty cycle from 18-40% | 1. Issue a T command with the following parameters: Voltage: 160 Frequency: 3500 Duty: 180 2. Issue I Command 3. Measure Duty cycle 5. Issue I Command 6. Issue a T command with the following parameters: Voltage: 800 Frequency: 3500 Duty: 400 7. Issue I Command 8. Measure Duty cycle | 1. First oscilloscope measurement should have a duty cycle of 18% +/- 1% 2. Second oscilloscope measurement should have a duty cycle of 35% +/- 1% |  |  |
| SRS-2.3 | The MB-burnin-test-fixture firmware shall be capable of adjusting inverter frequency from 350kHz to 700kHz | 1. Issue a T command with the following parameters: Voltage: 200 Frequency: 3500 Duty: 180 2. Issue I Command 3. Issue S Command 4. Measure frequency 5. Reset oscilloscope trigger 6. Issue a T command with the following parameters: Voltage: 200 Frequency: 7000 Duty: 180 7. Issue I Command 8. Issue S Command 9. Measure Frequency | 1. First oscilloscope measurement should have a duty cycle of 350kHz +/- 10kHz 2. Second oscilloscope measurement should have a duty cycle of 750kHz +/- 10kHz |  |  |
| SRS-2.4 | The MB-burnin-test-fixture firmware shall be capable of driving the monoblock for a duration of 10ms-200ms | 1. Issue a P command with the following parameters: Width: 20 Count: 1 Delay: 10 2. Issue I Command 3. Issue S Command 4. Measure pulse width 5. Reset oscilloscope trigger 6. Issue a P command with the following parameters: Width: 200 Count: 1 Delay: 10 7. Issue I Command 8. Issue S Command 9. Measure pulse width | 1. First oscilloscope measurement should have a pulse width of 20ms +/- 2ms 2.Second oscilloscope measurement should have a pulse width of 200ms +/- 5ms |  |  |
| SRS-2.5 | The MB-burnin-test-fixture firmware shall be capable of capturing voltage rail and filament current measurements | 1. Issue a T command with the following parameters: Voltage: 200 Frequency: 3500 Duty: 180 2. Issue I Command 3. Issue A Command, Channel 4. Record Value 4. Issue A Command, Channel 12, Record Value | 1. Channel 4 should be 20V +/- 1V 2. Channel 12 should be between 1.2V and 2.2V |  |  |
| SRS-2.6 | The MB-burnin-test-fixture firmware shall have a digital safety interlock | 1. Issue S Command 2. If no error is reported, issue E command 3. Verify that an interlock error has been reported 4. Issue I command, verify that the blue interlock LED is on 5. Issue S command | The shoot command should not function unless an interlock is engaged |  |  |

### Table 4
| Table 3. | Error Requirements |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Device Components Needed: | DC Power Supply ES-10035 Rev A ES-10035 Rev 1 ES-10024 Rev A RS-232/USB Cable, and PC HV board with filament load Heat-Gun ST-LINK/V2 |  |  |  |  |
| Test Setup: | 1. Connect RS-232 Cable to PC 2. Connect the HV board to the 10 pin connector on the PCBA. 3. Connect ES-10024 on the 24 pin connector 4. Run the python script: pySerial_wrapper_verbose.py 5. Turn on the Power Supply set to 24V |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.1 | The MB-burnin-test-fixture firmware shall flag hardware revision that is incomptable with current code | 1. Flash a Rev 1 unit with the current firmware 2. Connect to the unit over RS-232 3. Issue E Command | Error #1 Revision should be on the error report |  |  |
| SRS-3.2 | The MB-burnin-test-fixture firmware shall flag incorrect interlock or shoot digital keys | 1. Issue K Command 2. Issue E Command | Error #2 Command Key should be on the error report |  |  |
| SRS-3.3 | The MB-burnin-test-fixture firmware shall flag under/over voltage conditions | 1. Power down the board if currently on 2. Unplug ES-10024 board from the unit 3. Restart board 4. Issue I command 5. Issue E command | Error #3 Undervoltage/Over voltage should be on the error report A red fault LED should be lit up |  |  |
| SRS-3.4 | The MB-burnin-test-fixture firmware shall flag a "shoot-xray command" issued while filament duty cycle ramps | 1. Issue T command with the following parameters: Voltage: 820 Frequency: 3500 Duty: 350 2. Issue an I command followed immediately by an S command | Error #5 Filament Ramping should be on the error report A red fault LED should be lit up |  |  |
| SRS-3.5 | The MB-burnin-test-fixture firmware shall flag filament open/under current conditions | 1. Power down the board if currently on 2. Unplug HV board from the unit 3. Restart board 4. Issue I command 5. Issue E command | Error #6 Filament Open / Not warmed up should be on the error report A red fault LED should be lit up |  |  |
| SRS-3.6 | The MB-burnin-test-fixture firmware shall flag CLI command inputs out of designated range | 1. Issue T command with the following parameters: Voltage: 820 Frequency: 3500 Duty: 180 | Error #10 input range should be on the error report |  |  |
| SRS-3.7 | The MB-burnin-test-fixture firmware shall flag CLI command not found errors | 1. Issue Y command 2. Issue E command | Error #11 command not found should be on the error report |  |  |
| SRS-3.8 | The MB-burnin-test-fixture firmware shall flag over temperature errors | 1. Using a heat gun, carefully heat up the area surrounding the 2 pin connector 2. Issue A command for channel 10 3. Issue E command | Error #14 temp error should be on error report |  |  |
| SRS-3.9 | The MB-burnin-test-fixture firmware shall flag supply voltage errors | 1. Reduce input voltage to 22V 2. Issue E command | Error #15 ADC monitor error should be on error report |  |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-542 |  |  |

### Table 6
| MFG P/N | MFG | Description | Notes |
| --- | --- | --- | --- |
| ES-10035 Rev A | MedAI | Monoblock Test Fixture PCBA |  |
| ES-10035 Rev 1 | MedAI | Monoblock Test Fixture PCBA | Used for non-compatible hardware verification |
| ES-10024 Rev A | MedAI | Monoblock LV PWS Rider |  |
| E11046 Rev A | MedAI | Monoblock HV PCB |  |
| DT-6552-1.8M | DTech | RS-232/USB Cable |  |
| RTB2004 | Rhode and Schwartz | Oscilloscope | EQP-121 or equivalent |
| SPD3303X | Siglent | Benchtop Power Supply | EQP-246 or equivalent |
| HSA50R50J | TE Connectivity | 50 Ohm Power Resistor |  |
| 17B+ | Fluke | Digital Multimeter | EQP-239 or equivalent |
| ST-LINK/V2 | STMicroelectronics | J-LINK Programmer |  |
| ‎2688-20 | Milwaukee | Heat Gun |  |
| S10104 | MedAI | MB-burnin-test-fixture Firmware | v1.0.0-alpha |

### Table 7
| Table 1. | System Interfacing |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Device Components Needed: | DC Power Supply ES-10035 Rev A RS-232/USB Cable PC |  |  |  |  |
| Test Setup: | 1. Connect RS-232 Cable to PC 2. Run the python script: pySerial_wrapper_verbose.py without powering on the device 3. Turn on the DC Power Supply set to 24V |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-1.1 | The MB-burnin-test-fixture firmware shall interface with a PC over RS-232 | 1. Confirm the log outputs: DEBUG:COMX - USB Serial Port (COMX) INFO: COMX opened Where COMX is the COM port number of the RS-232 Cable. If the log reads: CRITICAL: Could not open the serial port. Exiting Check to see that the cable is correctly connected to the PC, or potentially swap cables 2. Power on the device, after a second press enter in the terminal | 1. The log outputs text confirming that a COM port has been opened 2. After powering on the device, and pressing enter, the terminal displays the text "BOOT" or "B" '79' '79' '84' 3. If no text appears, press the reset button on the board, and press enter in the terminal again | Expected operation verified See Appendix 1.1 Verified by EM 05SEP2024 | Pass |

### Table 8
| Table 2. | Operation Requirements |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Device Components Needed: | DC Power Supply ES-10035 Rev A ES-10024 Rev A RS-232/USB Cable, and PC HV board with filament load 50ohm load for H-Bridge SMA to BNC cable Oscilloscope Multimeter |  |  |  |  |
| Test Setup: | 1. Connect RS-232 Cable to PC 2. Connect C1 on the oscilloscope to J25 using an SMA to BNC cable. Set a measurement on the oscilloscope to measure frequency on C1 3. Connect the HV board to the 10 pin connector on the PCBA. Connect C2 on the oscilloscope to the leads going to the first transformer on the filament load. Set a measurement on the oscilloscope to measure frequency on C2 4. Connect ES-10024 to the board on the 24 pin connector 5. Run the python script: pySerial_wrapper_verbose.py 6. Turn on the Power Supply set to 24V |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-2.1 | The MB-burnin-test-fixture firmware shall be capable of adjusting monoblock input voltage from 16V to 80V | 1. Issue a T command with the following parameters: Voltage: 160 Frequency: 3500 Duty: 180 2. Issue I Command 3. Measure X-RAY rail voltage 5. Issue I Command 6. Issue a T command with the following parameters: Voltage: 800 Frequency: 3500 Duty: 180 7. Issue I Command 8. Measure X-RAY rail voltage | 1. First DMM measurement of rail voltage should be 16V +/- 1V 2. Second DMM measurement of rail voltage should be 80V +/- 1V | Expected operation verified See Appendix 2.1 Measured 16.06V and 79.6V on DMM Verified by EM 05SEP2024 | Pass |
| SRS-2.2 | The MB-burnin-test-fixture firmware shall be capable of adjusting filament duty cycle from 18-40% | 1. Issue a T command with the following parameters: Voltage: 160 Frequency: 3500 Duty: 180 2. Issue I Command 3. Measure Duty cycle 5. Issue I Command 6. Issue a T command with the following parameters: Voltage: 800 Frequency: 3500 Duty: 400 7. Issue I Command 8. Measure Duty cycle | 1. First oscilloscope measurement should have a duty cycle of 18% +/- 1% 2. Second oscilloscope measurement should have a duty cycle of 35% +/- 1% | Expected operation verified See Appendix 2.2 Verified by EM 05SEP2024 | Pass |
| SRS-2.3 | The MB-burnin-test-fixture firmware shall be capable of adjusting inverter frequency from 350kHz to 700kHz | 1. Issue a T command with the following parameters: Voltage: 200 Frequency: 3500 Duty: 180 2. Issue I Command 3. Issue S Command 4. Measure frequency 5. Reset oscilloscope trigger 6. Issue a T command with the following parameters: Voltage: 200 Frequency: 7000 Duty: 180 7. Issue I Command 8. Issue S Command 9. Measure Frequency | 1. First oscilloscope measurement should have a duty cycle of 350kHz +/- 10kHz 2. Second oscilloscope measurement should have a duty cycle of 750kHz +/- 10kHz | Expected operation verified See Appendix 2.3 Verified by EM 05SEP2024 | Pass |
| SRS-2.4 | The MB-burnin-test-fixture firmware shall be capable of driving the monoblock for a duration of 10ms-200ms | 1. Issue a P command with the following parameters: Width: 20 Count: 1 Delay: 10 2. Issue I Command 3. Issue S Command 4. Measure pulse width 5. Reset oscilloscope trigger 6. Issue a P command with the following parameters: Width: 200 Count: 1 Delay: 10 7. Issue I Command 8. Issue S Command 9. Measure pulse width | 1. First oscilloscope measurement should have a pulse width of 20ms +/- 2ms 2.Second oscilloscope measurement should have a pulse width of 200ms +/- 5ms | Expected operation verified See Appendix 2.4 Verified by EM 05SEP2024 | Pass |
| SRS-2.5 | The MB-burnin-test-fixture firmware shall be capable of capturing voltage rail and filament current measurements | 1. Issue a T command with the following parameters: Voltage: 200 Frequency: 3500 Duty: 180 2. Issue I Command 3. Issue A Command, Channel 4. Record Value 4. Issue A Command, Channel 12, Record Value | 1. Channel 4 should be 20V +/- 1V 2. Channel 12 should be between 1.2V and 2.2V | Expected operation verified See Appendix 2.5 Verified by EM 05SEP2024 | Pass |
| SRS-2.6 | The MB-burnin-test-fixture firmware shall have a digital safety interlock | 1. Issue S Command 2. If no error is reported, issue E command 3. Verify that an interlock error has been reported 4. Issue I command, verify that the blue interlock LED is on 5. Issue S command | The shoot command should not function unless an interlock is engaged | Expected operation verified See Appendix 2.6 Verified by EM 05SEP2024 | Pass |

### Table 9
| Table 3. | Error Requirements |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| Device Components Needed: | DC Power Supply ES-10035 Rev A ES-10035 Rev 1 ES-10024 Rev A RS-232/USB Cable, and PC HV board with filament load Heat-Gun ST-LINK/V2 |  |  |  |  |
| Test Setup: | 1. Connect RS-232 Cable to PC 2. Connect the HV board to the 10 pin connector on the PCBA. 3. Connect ES-10024 on the 24 pin connector 4. Run the python script: pySerial_wrapper_verbose.py 5. Turn on the Power Supply set to 24V |  |  |  |  |
|  | Test Case: |  |  |  |  |
| SRS-3.1 | The MB-burnin-test-fixture firmware shall flag hardware revision that is incomptable with current code | 1. Flash a Rev 1 unit with the current firmware 2. Connect to the unit over RS-232 3. Issue E Command | Error #1 Revision should be on the error report | Expected operation verified See Appendix 3.1 Verified by EM 05SEP2024 | Pass |
| SRS-3.2 | The MB-burnin-test-fixture firmware shall flag incorrect interlock or shoot digital keys | 1. Issue K Command 2. Issue E Command | Error #2 Command Key should be on the error report | Expected operation verified See Appendix 3.3 Verified by EM 05SEP2024 | Pass |
| SRS-3.3 | The MB-burnin-test-fixture firmware shall flag under/over voltage conditions | 1. Power down the board if currently on 2. Unplug ES-10024 board from the unit 3. Restart board 4. Issue I command 5. Issue E command | Error #3 Undervoltage/Over voltage should be on the error report A red fault LED should be lit up | Expected operation verified See Appendix 3.3 Verified by EM 05SEP2024 | Pass |
| SRS-3.4 | The MB-burnin-test-fixture firmware shall flag a "shoot-xray command" issued while filament duty cycle ramps | 1. Issue T command with the following parameters: Voltage: 820 Frequency: 3500 Duty: 350 2. Issue an I command followed immediately by an S command | Error #5 Filament Ramping should be on the error report A red fault LED should be lit up | Expected operation verified See Appendix 3.4 Verified by EM 05SEP2024 | Pass |
| SRS-3.5 | The MB-burnin-test-fixture firmware shall flag filament open/under current conditions | 1. Power down the board if currently on 2. Unplug HV board from the unit 3. Restart board 4. Issue I command 5. Issue E command | Error #6 Filament Open / Not warmed up should be on the error report A red fault LED should be lit up | Expected operation verified See Appendix 3.5 Verified by EM 05SEP2024 | Pass |
| SRS-3.6 | The MB-burnin-test-fixture firmware shall flag CLI command inputs out of designated range | 1. Issue T command with the following parameters: Voltage: 820 Frequency: 3500 Duty: 180 | Error #10 input range should be on the error report | Expected operation verified See Appendix 3.6 Verified by EM 05SEP2024 | Pass |
| SRS-3.7 | The MB-burnin-test-fixture firmware shall flag CLI command not found errors | 1. Issue Y command 2. Issue E command | Error #11 command not found should be on the error report | Expected operation verified See Appendix 3.7 Verified by EM 05SEP2024 | Pass |
| SRS-3.8 | The MB-burnin-test-fixture firmware shall flag over temperature errors | 1. Using a heat gun, carefully heat up the area surrounding the 2 pin connector 2. Issue A command for channel 10 3. Issue E command | Error #14 temp error should be on error report | Expected operation verified See Appendix 3.8 Verified by EM 05SEP2024 | Pass |
| SRS-3.9 | The MB-burnin-test-fixture firmware shall flag supply voltage errors | 1. Reduce input voltage to 22V 2. Issue E command | Error #15 ADC monitor error should be on error report | Expected operation verified See Appendix 1.1 Verified by EM 05SEP2024 | Pass |

### Table 10
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-553 |  |  |
