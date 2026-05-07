# VVPR-P01-211 Rev B: MX1 MedAI Diagnositc Tool v2.3.1 WS-007 Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-211
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v2.3.1
- Source filename: VVPR-P01-211 - MX1 MedAI Diagnositc Tool v2.3.1 WS-007 Verification Protocol and Report _B.docx
- Source path: Example QMS - MedAI/VVPR-P01-211 - MX1 MedAI Diagnositc Tool v2.3.1 WS-007 Verification Protocol and Report _B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to perform required regression testing to demonstrate MX1 MedAI Diagnostic Tool (ODT) v2.3.1 meets the requirements as stated in MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification Rev. E for WorkStation WS-007.
OBJECTIVE AND SCOPE
Verify ODT changes from v2.1.0 to v2.3.1 do not impact WorkStation performance and continue to meet the software requirements set by MedAI installed in MEMO-P01-604 Rev E on the following WorkStation:
WS-007, Cassette Verification which uses the MX1-Cassette-Test-Plugin
Verify changes to the logging information to the logMC file have been implemented as intended.
REFERENCES
MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification, Rev E
MEMO-P01-695 - ODT SW System Architecture Diagram, Rev C
IFU-MX1 - Instructions for Use, Rev. F
MWI-230 Rev F, MS-10511 Cassette Verification
MATERIALS
Cassette without battery installed, MS-10511 (assembled per BOM-008 - C1 Cassette Rev. J)
WS-007, Rev. A
S10099 Firmware ODT Tripper v1.0.2
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
REGRESSION TESTING
A partial verification of the requirements previously verified in VVPR-P01-192 is required as the changes in v2.3.1 do not broadly impact functionality and satisfaction of requirements.  Verification will be completed by executing the Production Run Procedure, analyzing the resulting log, and a code comparison of the previously released version, v2.1.0, and the release in process, v2.3.1, to ensure no parameters or other elements were changed erroneously.
The requirements found in the following subsections of Section 9, “Cassette,” of the Software Requirements Specification (MEMO-P01-604) are affected by changes to the way they log results of execution:
ODT9.6
ODT9.8
ODT9.9
ODT9.10
ODT9.11
ODT9.12
ODT9.13
ODT9.14
ODT9.16
ODT9.15
ODT9.17
ODT9.18
ODT9.19
ODT9.20
ODT9.21
ODT9.22
ODT9.23
ODT9.24
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Setup
Flash UUTs boards (Emitter Main, Collimator and Cassette Main PCBAs) with ODT Tripper firmware v.1.0.2
Connect to each board and PSU in ODT
Set ODT to verbose mode
Hardware Mock Testing
The Concept of mocking hardware via a modified version of firmware is employed during this test. During verification, the ODT-Tripper firmware will “feed” ODT values outside of the range of ODT’s acceptable values. Depending on if ODT sets the power supply to LOW, NOM, or HIGH voltage, the voltage threshold at which ODT considers a failure changes. The firmware has register D3 at which the operator can alter which mocked hardware voltage values that the ODT-Tripper Firmware must return to match the active ODT Thresholds.
Experimental Procedure
Fill out the results table and follow the verification step instructions. For tests that require simulation via the ODT-Tripper Firmware, refer to the simulation and ODT-Tripper configuration steps. For tests that require a corrupted json file, refer to simulation of corrupt json file steps.
Simulated Device Type Test - Used in WS-007
Select Device Type Test
Run Test, expect PASS
Write register 0 with payload =2
example.com/
Run Test, expect FAIL
Set register 0 back to payload == pid
example.com/
Simulated Voltage Tests - Used in WS-001 and WS-007
Simulated Low Voltage Tests
Run D3 payload 0 example.com/
Select only voltage tests in ODT with PSU:LOW
Write register d3 with payload=0 - tells firmware use low battery
Write register d4 with payload =0 - tells firmware report values below threshold
Run test (all should fail for undervolting)
Write register d4 with payload=1
Run test (all should fail for overvolting)
Write register d4 with payload=2
Run test (all should pass)
Simulated Nominal Voltage Tests
Run D3 payload 1 example.com/
Select only voltage tests in ODT with PSU: NOM
Write register d3 with payload=1
Write register d4 with payload =0
Run test (all should fail for undervolting)
Write register d4 with payload=1
Run test (all should fail for overvolting)
Write register d4 with payload=2
Run test (all should pass)
Simulated High Voltage Test
Run D3 payload 2 example.com/
Select only voltage  tests in ODT with PSU: HIGH
Write register d3 with payload=2
Write register d4 with payload =0
Run test (all should fail for undervolting)
Write register d4 with payload=1
Run test (all should fail for overvolting)
Write register d4 with payload=2
Run test (all should pass)
Simulated Corrupt Json File Test - Used in WS-004
ssh into the emitter jetson on WS-004
Ensure that no file named “cam-test-output.json” exists
Create an empty file named “cam-test-output.json”
Complete a Production Run to output complete log file.
Table 3 Cassette Verification WorkStation WS-007 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-007, MWI-230, MS-10511 Cassette Verification, should be used to guide operation of the WorkStation as needed. Verify ODT and MX1-Cassette-Test-Plugin.
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
Code Review
Utilize Git diff tool to compare changes between previously released version and active alpha version to confirm no unwarranted changes have been made to the software.  Relevant changes in Appendix A, summarized as follows:
Updated logging for conformance with manufacturer’s system
Production Run logging of part number
Production Run logging of time/date formatted as DD-MM-YYYY
Production Run logging of operator tied to added Authentication GUI
Production Run logging of test sequence tracks software version
Substitution of commas with semicolons
Populate Expected Value for all tests or add ‘N/A’ where appropriate
Populate Unit of Measure for all tests
Change PASS/FAIL to PASSED/FAILED
Removal of unused commented sections
Removal of attempt to read parameters from Google Drive wherein no parameters had previously been stored
Addition of authentication occurs outside of MX1-Cassette-Test Plugin
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
Appendix A
Code comparison screenshots comparing v2.1.0 vs 2.3.1
example.com/
Figure 1. Substitution of semicolons in place of commas, deletion of unused comments
Figure 2. Addition of Expected Value for MON_JET_5V0 and MON_MCU_5V0, deletion of unused comments
Figure 3. Addition of Expected Value for MON_JET_3V3 and MON_DET_22V
Figure 4. Substitution of ‘PASSED’ in place of ‘PASS’ and ‘FAILED’ in place of ‘FAIL’ in Buzzer test
Figure 5. Addition of PluginNumber for reference, removal of querying Google Drive for parameters
Figure 6. Addition of Unit of Measure for IR LED tests
Figure 7. Substitution of ‘PASSED’ in place of ‘PASS’ and ‘FAILED’ in place of ‘FAIL’  in Indicator LED test
Figure 8.  Substitution of ‘PASSED’ in place of ‘PASS’ and ‘FAILED’ in place of ‘FAIL’ in OLED test
The following figures are all highlighting changes to Production Run, summary follows after last figure
Figures 9 - 20: Production Run changes that include:addition of authentication launcher, button to relaunch authentication, substitution of ‘PASSED’ in place of ‘PASS and ‘FAILED’ in place of ‘FAIL’, switching Date and Time format from MM/dd/yyyy HH:mm:ss to dd/MM/yyyy HH:mm:ss, and typo corrections
Report Section
PROTOCOL DEVIATIONS
There were no protocol deviations.
DEVICES, COMPONENTS, OR EQUIPMENT USED
MS-10511 Cassette Enclosure w/o Battery/Battery Cover/Cover Screws Rev. A (C1 SN1280 Internally identified as DV27)
WS-007 Rev. A Cassette Verification WorkStation
ODT v2.3.1-alpha
RESULTS
Discussion
This MedAI Diagnostic Tool v2.3.1 evaluation demonstrated intended modifications were implemented appropriately and changes occurring to ODT from v2.1.0 to v2.3.1 did not affect WS-007 performance.
Code review and WS-007 Production Run results identified and demonstrated the code changes implemented to comply with CMO logging requirements have been successfully incorporated. These changes include:
Production Run logging of part number
Production Run logging of time/date formatted as DD-MM-YYYY
Production Run logging of operator tied to added Authentication GUI
Production Run logging of test sequence tracks software version
Substitution of commas with semicolons
Populate Expected Value for all tests or add ‘N/A’ where appropriate
Populate Unit of Measure for all tests
Change PASS/FAIL to PASSED/FAILED
Removal of unused commented sections
Removal of attempt to read parameters from Google Drive wherein no parameters had previously been stored
Log changes due to ODT Production Run logging changes including date format to DD/MM/YYYY.
Addition of authentication occurs outside of MX1-Cassette-Test Plugin
During Cassette Log review, the Cassette Analog Monitor Test MON-HS-SW; PSU NOM, LOW and HIGH it was determined the ODT range values were different from the ranges in the acceptance criteria of this protocol. Regardless, the results were within the ODT range and the VVPR acceptance criteria range. Further assessment of range corrections will be completed and implemented.
Conclusions
MedAI Diagnostic Tool v2.3.1 has been verified to work per the ODT requirements, require operator authentication and produce the required CMO logging output.
Attachments
Attachment 1 - Cassette Log File
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Cassette, WS-007 |  |  |  |  |
| Precondition | N/A |  |  |  |  |
| ODT9.6 | ODT shall control the mode of the IR LEDs on a connected Cassette | Run Cassette IR LED OFF test Verify all IR LEDs are turned OFF Run Cassette IR LEDX ON test (Run 36 times) Verify 36 IR LEDs are turned ON Code Review | Cassette IR LEDs ON Test returns a 36 count of functioning IR LEDs Code review shows no changes to LED test parameters |  |  |
| ODT9.8 | ODT shall query the monitor of the High Side (CM_PWR) power rail of a connected Cassette | Run Cassette Analog Monitor Test MON_HS_SW, PSU: NOM Verify ODT passes the Cassette Analog Monitor Test MON_HS_SW, PSU: NOM test if read back values range from 13.16V - 14.84V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_HS_SW, PSU: NOM test if read back values range from 13.16V - 14.84V Code review shows no changes to analog monitor test parameters |  |  |
|  |  | Run Cassette Analog Monitor Test MON_HS_SW, PSU: LOW Verify ODT passes the Cassette Analog Monitor Test MON_HS_SW, PSU: LOW test if read back values range from 8V - 12V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_HS_SW, PSU: LOW test if read back values range from 8V - 12V Code review shows no changes to analog monitor test parameters |  |  |
|  |  | Run Cassette Analog Monitor Test MON_HS_SW, PSU: HIGH Verify ODT passes the Cassette Analog Monitor Test MON_HS_SW, PSU: HIGH test if read back values range from 15.96-17.64V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_HS_SW, PSU: HIGH test if read back values range from 15.96-17.64V Code review shows no changes to analog monitor test parameters |  |  |
| ODT9.9 | ODT shall query the monitor of the 5.0V power rail going to the Jetson of a connected Cassette | Run Cassette Analog Monitor Test MON_JET_5V0, PSU: NOM Verify ODT passes read back value ranging from 4.75-5.25V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_JET_5V0, PSU: NOM/LOW/HIGH if read back value ranges from 4.75V-5.25V Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.10 | ODT shall query the monitor of the 5.0V power rail going to the MCU of a connected Cassette | Run Cassette Analog Monitor Test MON_MCU_5V0, PSU: NOM Verify ODT passes read back value ranging from 4.75-5.25V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_MCU_5V0, PSU: NOM/LOW/HIGH if read back value ranges from 4.75-5.25V Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.11 | ODT shall query the monitor of the 3.3V power rail going to the Jetson of a connected Cassette | Run Cassette Analog Monitor Test MON_JET_3V3, PSU: NOM Verify ODT passes read back value ranging from 3.135-3.465V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_JET_3V3, PSU: NOM/LOW/HIGH if read back value ranges from 3.135V - 3.465V Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.12 | ODT shall query the monitor of the 22V power rail going to the Detector of a connected Cassette | Run Cassette Analog Monitor Test MON_DET_22V, PSU: NOM Verify ODT passess read back value ranging from 20.9V-23.1V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_DET_22V, PSU: NOM/LOW/HIGH if read back value ranges from 20.9V - 23.1V Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.13 | ODT shall turn on the buzzer of a connected Cassette to different tones with buzzer input frequencies 1,2,3 and 4 kHz | Run Cassette Buzzer Test Verify four distinct audible tones are generated. Code Review | Cassette Buzzer Test creates four distinct audible tones using frequencies of 1, 2, 3 and 4 kHz Code review shows no changes to buzzer execution |  |  |
| ODT9.14 | ODT shall set the output of RGB LEDs of a connected Cassette | Run Cassette LED Indicator Test: Verify all LEDS turn to WHITE . Code Review | ALL on-board RGB LED turn white Code review show no changes to RGB LED execution |  |  |
| ODT9.16 | ODt shall query the monitor of the thermistor near the battery connector of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT1 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.17 | ODT shall query the monitor of the thermistor between the Jetson and MCU of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT2 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.18 | ODt shall query the monitor of the thermistor between the IMU/M-LVDS transceiver near the MCU of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT3 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.19 | ODT shall query the monitory of the thermistor near the battery charger IC of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT4 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.20 | ODt shall query the monitor of the thermistor near the 5V0 regulator of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT5 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.21 | ODT shall query the monitor of the thermistor near the Jetson of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT6 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.22 | ODt shall query the monitor of the thermistor near the LTE module of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT7 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.23 | ODT shall query the monitor of the thermistor near the WiFi module of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT8 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |
| ODT9.24 | ODT shall query the monitor of the off-board thermocouple near the Detector of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT9 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-579 |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | Cassette, WS-007 |  |  |  |  |
| Precondition | N/A |  |  |  |  |
| ODT9.6 | ODT shall control the mode of the IR LEDs on a connected Cassette | Run Cassette IR LED OFF test Verify all IR LEDs are turned OFF Run Cassette IR LEDX ON test (Run 36 times) Verify 36 IR LEDs are turned ON Code Review | Cassette IR LEDs ON Test returns a 36 count of functioning IR LEDs Code review shows no changes to LED test parameters | Log file showed 36 counts of IR LEDs ON tests PASSED. Code review shows Unit of Measure set to “Bool,” and resulting log shows “PASSED” instead of “PASS” | P |
| ODT9.8 | ODT shall query the monitor of the High Side (CM_PWR) power rail of a connected Cassette | Run Cassette Analog Monitor Test MON_HS_SW, PSU: NOM Verify ODT passes the Cassette Analog Monitor Test MON_HS_SW, PSU: NOM test if read back values range from 13.16V - 14.84V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_HS_SW, PSU: NOM test if read back values range from 13.16V - 14.84V Code review shows no changes to analog monitor test parameters | Cassette Log file readback value of 13.31V and PASSED. ODT range is different (12.5685 - 13.8915V) than pass criteria range, but readback value passes both. Resulting log shows no illegal commas and shows “PASSED” instead of “PASS | P |
|  |  | Run Cassette Analog Monitor Test MON_HS_SW, PSU: LOW Verify ODT passes the Cassette Analog Monitor Test MON_HS_SW, PSU: LOW test if read back values range from 8V - 12V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_HS_SW, PSU: LOW test if read back values range from 8V - 12V Code review shows no changes to analog monitor test parameters | Cassette Log file readback value of 8.73V and PASSED. ODT range is different (8.3125 -9.1875V) than pass criteria range, but readback value passes both. Resulting log shows no illegal commas and shows “PASSED” instead of “PASS | P |
|  |  | Run Cassette Analog Monitor Test MON_HS_SW, PSU: HIGH Verify ODT passes the Cassette Analog Monitor Test MON_HS_SW, PSU: HIGH test if read back values range from 15.96-17.64V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_HS_SW, PSU: HIGH test if read back values range from 15.96-17.64V Code review shows no changes to analog monitor test parameters | Cassette Log file readback value of 16.22V and PASSED. ODT range is different (15.333-16.947V) than pass criteria range, but readback value passes both. Resulting log shows no illegal commas and shows “PASSED” instead of “PASS | P |
| ODT9.9 | ODT shall query the monitor of the 5.0V power rail going to the Jetson of a connected Cassette | Run Cassette Analog Monitor Test MON_JET_5V0, PSU: NOM Verify ODT passes read back value ranging from 4.75-5.25V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_JET_5V0, PSU: NOM/LOW/HIGH if read back value ranges from 4.75V-5.25V Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_JET_5V0 readback value of 4.92V for NOM, LOW and HIGH cases. All tests PASSED. Code Review shows Expected Value added to print and reflected in the resulting log. Log also shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.10 | ODT shall query the monitor of the 5.0V power rail going to the MCU of a connected Cassette | Run Cassette Analog Monitor Test MON_MCU_5V0, PSU: NOM Verify ODT passes read back value ranging from 4.75-5.25V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_MCU_5V0, PSU: NOM/LOW/HIGH if read back value ranges from 4.75-5.25V Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_MCU_5V0 readback values of 5.02, 5.02 and 5.03V for NOM, LOW and HIGH cases, respectively. All tests PASSED. Code Review shows Expected Value added to print and reflected in the resulting log. Log also shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.11 | ODT shall query the monitor of the 3.3V power rail going to the Jetson of a connected Cassette | Run Cassette Analog Monitor Test MON_JET_3V3, PSU: NOM Verify ODT passes read back value ranging from 3.135-3.465V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_JET_3V3, PSU: NOM/LOW/HIGH if read back value ranges from 3.135V - 3.465V Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_JET_3V3 readback value of 3.31V for NOM, LOW  and HIGH cases. All tests PASSED. Code Review shows Expected Value added to print and reflected in the resulting log. Log also shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.12 | ODT shall query the monitor of the 22V power rail going to the Detector of a connected Cassette | Run Cassette Analog Monitor Test MON_DET_22V, PSU: NOM Verify ODT passess read back value ranging from 20.9V-23.1V Code Review | ODT shall pass the Cassette Analog Monitor Test MON_DET_22V, PSU: NOM/LOW/HIGH if read back value ranges from 20.9V - 23.1V Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_DET_22V readback values of 22.06, 22.02, and 22.08V for NOM, LOW and HIGH cases, respectively. All tests PASSED. Code Review shows Expected Value added to print and reflected in the resulting log. Log also shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.13 | ODT shall turn on the buzzer of a connected Cassette to different tones with buzzer input frequencies 1,2,3 and 4 kHz | Run Cassette Buzzer Test Verify four distinct audible tones are generated. Code Review | Cassette Buzzer Test creates four distinct audible tones using frequencies of 1, 2, 3 and 4 kHz Code review shows no changes to buzzer execution | Cassette Buzzer Test created four distinct audible tones. Cassette Log shows Cassette Buzzer Test PASSED. Resulting log shows “PASSED” instead of “PASS” | P |
| ODT9.14 | ODT shall set the output of RGB LEDs of a connected Cassette | Run Cassette LED Indicator Test: Verify all LEDS turn to WHITE . Code Review | ALL on-board RGB LED turn white Code review show no changes to RGB LED execution | Cassette LED Indicator Test turned all RGB LEDs white. Cassette Log shows all LED ON Tests PASSED. Resulting log shows “PASSED” instead of “PASS” | P |
| ODT9.15 | ODT shall issue a command to turn on all pixels of the display | Run Cassette OLED Test:  Verify all pixels turn on | OLED display illuminates white Code review show no changes to OLED execution | OLED Test PASSED. Resulting log shows “PASSED” instead of “PASS” | P |
| ODT9.16 | ODT shall query the monitor of the thermistor near the battery connector of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT1 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT1 readback value of 44.53 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.17 | ODT shall query the monitor of the thermistor between the Jetson and MCU of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT2 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT2 readback value of 43.61 degC  and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.18 | ODt shall query the monitor of the thermistor between the IMU/M-LVDS transceiver near the MCU of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT3 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT3 readback value of 42.22 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.19 | ODT shall query the monitory of the thermistor near the battery charger IC of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT4 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT4 readback value of 44.13 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.20 | ODT shall query the monitor of the thermistor near the 5V0 regulator of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT5 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT5 readback value of 50.64 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.21 | ODT shall query the monitor of the thermistor near the Jetson of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT6 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT6 readback value of 48.05 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.22 | ODt shall query the monitor of the thermistor near the LTE module of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT7 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT7 readback value of 48.61 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.23 | ODT shall query the monitor of the thermistor near the WiFi module of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT8 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT8 readback value of 44.89 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |
| ODT9.24 | ODT shall query the monitor of the off-board thermocouple near the Detector of a connected Cassette | Run Cassette Analog Monitor Test MON_TEMP, RT9 Code Review | ODT shall read back a temperature value from the designated thermistor Code review shows no changes to analog monitor test parameters, except the expected value | Cassette Log shows MON_TEMP, RT9 readback value of 0.01 degC and PASSED Code Review shows Expected Value prints “N/A.” Log shows no illegal commas and shows “PASSED” instead of “PASS” | P |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-586 |  |  |
