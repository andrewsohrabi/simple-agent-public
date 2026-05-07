# VVPR-P01-196 Rev B: MX1 MedAI Diagnostic Tool Tranch 2 v2.2.1 Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-196
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v2.2.1
- Source filename: VVPR-P01-196 - MX1 MedAI Diagnostic Tool Tranch 2 v2.2.1 Verification Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-196 - MX1 MedAI Diagnostic Tool Tranch 2 v2.2.1 Verification Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 MedAI Diagnostic Tool (ODT) meets the requirements as stated in MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification Rev. A for the WorkStations included in Tranche 2: WS-003, WS-005, and WS-006.
OBJECTIVE AND SCOPE
Verify the software requirements set by MedAI for ODT v2.2.1-alpha release installed on the following WorkStations:
WS-003, MS-10155 X-ray Assembly Calibration
WS-005, MS-10301 HMI Display Verification
Verify the software meets the requirements in ODT Software Requirements Specification, MEMO-P01-604 Rev D.
A test firmware version created and installed on Units Under Test (UUT) sends ODT two types of information:
Nominal, low and high parameter values to demonstrate ODT accurately assesses input data as PASS or FAIL.
True and False data to demonstrate ODT can differentiate between true and false input data
WS-005, HMI Verification WorkStation was shipped as part of Tranche 1, but ODT verification was moved to Tranche 2 due to the issue that needed to be resolved with the Raspberry Pi. Therefore, WS-005 will be considered to be part of Tranche 2.
REFERENCES
MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification, Rev D
MEMO-P01-695 - ODT SW System Architecture Diagram, Rev B
IFU-MX1 - Instructions for Use, Rev. F
MWI-221 Rev B WS-003 Workstation Installation
MWI-222 Rev B  - MS-10155 X-ray Assembly Calibration & Verification
MWI-225 Rev C WS-005 Workstation Installation
MWI-226 Rev C - MS-10401 HMI Display Verification
MATERIALS
X-ray Assembly , MS-10155 Rev  C
MS-10514 Wireless Charger Assembly Rev C (Lot 10250)
HMI and Display PCBA Assembly, MS-10401 Rev A
WS-003, X-ray Assembly Calibration & Verification WorkStation
WS-005, HMI Verification WorkStation
S10099 Firmware ODT Tripper v1.0.1
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Setup
Flash UUTs boards (LV PCBAs) with ODT Tripper firmware v1.0.1
Connect to each board and PSU in ODT
Set ODT to verbose mode
Hardware Mock Testing
The Concept of mocking hardware via a modified version of firmware is employed during this test. During verification, the ODT-Tripper firmware will “feed” ODT values outside of the range of ODT’s acceptable values. Depending on if ODT sets the power supply to LOW, NOM, or HIGH voltage, the voltage threshold at which ODT considers a failure changes. The firmware has register D3 at which the operator can alter which mocked hardware voltage values that the ODT-Tripper Firmware must return to match the active ODT Thresholds.
Experimental Procedure
Fill out the results table and follow the verification step instructions. For tests that require simulation via the ODT-Tripper Firmware, refer to the simulation and ODT-Tripper configuration steps. For tests that require a corrupted json file, refer to simulation of corrupt json file steps.
Simulated Device Type Test - Used in WS-003
Select Device Type Test
Run Test, expect PASS
Write register 0 with payload =2. All commands are rest server calls to the workstation jetson which tell ODT Tripper which values to return for the various tests.
example.com/
Run Test, expect FAIL
Set register 0 back to payload == pid
example.com/
Simulated Voltage Tests - Used in WS-003 and WS-006
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
Complete Production Run for each WorkStation, screen shot all displayed test results and save production log files.
Perform all testing described in Tables 1-3 and record results in Evidence Column.
Select Pass or Fail based on results.
Table 1. X-ray Assembly Calibration WorkStation WS-003 - Requirements, Verification Steps, and Expected Results
The Manufacturing Work Instructions for WS-003, MWI-222 Rev B, MS-10155 X-ray Assembly Calibration & Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-Monoblock-Test-Plugin.
Table 2. HMI Display Verification WorkStation WS-005 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-005, MWI-226, MS-10401 HMI Display Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-HMI-Test-Plugin.
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
Report Section
PROTOCOL DEVIATIONS
Protocol execution was started using ODT-Tripper v1.0.1. The deviations below are clerical modifications only and have been added to ODT-Tripper which will be released as ODT-Tripper v1.0.2.
ODT Tripper was modified to include flash writing and erasing sequences previously established to address timing issues with the flashing process that were inadvertently omitted from ODT Tripper firmware.
ODT Tripper was modified to correct the Low Threshold value from -14.24 V to -15.76 V since failing on the low side of the range -14.25 to -15.75 should be a value less than -15.75.
ODT Tripper was modified to include temperature tripper commands in order to assess ODT5.13, ODT shall query the temperature measured by the thermocouples of a connected Monoblock. During protocol execution it was determined the LV PCBA firmware was not capable of reading back negative temperature values. The negative ODT Tripper value read back 655 C and failed demonstrating ODT successfully identified the temperature value as out of range. In addition, the 81C ODT Tripper value to assess the high temperature limit, 80C, unexpectedly passed. After further investigation it was determined that ODT sets the upper temperature limit to 87.5 C. ODT successfully measured a nominal temperature value and correctly identified the value to be within range. As ODT demonstrated its ability to query the thermocouple temperatures and accurately identify whether the value was within the acceptable range, further testing on this requirement was not deemed necessary.
ODT software was modified to address two HMI Heartbeat bugs, released in ECR-548 as v2.2.1-beta and installed on WS-003 for verification. WorkStation testing that requires communication with the monoblock fixture fans will be repeated as these bugs impact ODT’s ability to communicate with the fixture fans.
There was a protocol generation error in Table 1, X-ray Assembly Verification Workstation, where SRS ODT5.12 through ODT5.32 were inadvertently not included due to an error with linking all necessary cells from the spreadsheet.  These additional sections were previously reviewed by Engineering for accuracy in the linked spreadsheet.
During protocol execution it was determined the EUT was not capable of firing x-rays. Therefore requirements comparing values measured by the monoblock and values measured by the RadCal, ODT5.18, ODT5.19, ODT5.22 and ODT5.25 were not evaluated (NE). These requirements shall be assessed prior to Phase 4 closure when functioning monoblocks are available.
Procedure for testing ODT5.20 was modified to indicate ODT needs divider parameter value to be within a range and not the frequency calculated using the divider.  The divider range stated in the procedure was used as inputs for this test.
During protocol execution WS-005, HMI Verification, did not function as intended. This workstation is not required for Monoblock fabrication and verification and therefore removing it from this VVPR does not impact production efforts. Further investigations will be completed on WS-005 to determine ODT changes required to meet requirements and verification will occur once this effort is complete.
DEVICES, COMPONENTS, OR EQUIPMENT USED
MS-10155, E1 X-ray Assembly
WS-003
MS-10401 Rev B
S10099 Firmware ODT Tripper v1.0.2
ODT v2.2.1-beta
RESULTS
Table 1. X-ray Assembly Calibration WorkStation WS-003 - Requirements, Verification Steps, and Expected Results
The Manufacturing Work Instructions for WS-003, MWI-222 Rev B, MS-10155 X-ray Assembly Calibration & Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-Monoblock-Test-Plugin.
Table 2. HMI Display Verification WorkStation WS-005 - Requirements, Verification Steps, and Expected Results
Follow the steps outlined below. The Manufacturing Work Instructions for WS-005, MWI-226, MS-10401 HMI Display Verification, should be used to guide operation of the WorkStation as needed. Verifying ODT and MX1-HMI-Test-Plugin.
DISCUSSION
The About MedAI Diagnostic Tool button on ODT displays v2.3.0, Build 2.3.0-tags-v2-2-1-beta.1+59 GitHash:17780e6. (See Appendix 1a.)This VVPR is intended to verify v2.2.1-beta. Additional work is required for the About ODT button to display 2.2.1-beta as the Version. Note, testing started with v2.2.1-alpha, however, two bugs with HMI communications were identified, fixed and released in v2.2.1-beta. All testing was performed with v2.2.1-beta.
After VVPR execution, the cause of the versioning issue identified in 4.1 was determined. However, during the effort to address this issue ODT v2.2.1-beta was modified to add code. Then it was determined the code did not address the issue so the code was commented out.  These modifications resulted in a new Githash, 2f4c35d (See Appendix 1b.), which has the same code as the Githash version tested. Therefore, these versions are identical. The issue was resolved by modifying Gitversion tags to remove the lower case v before the software version. This change was made outside of ODT.
During the execution of VVPR-P01-196 several protocol generation errors were identified and are listed in the deviations section above.
During WS-003 verification testing for ODT5.24 the EUT was determined not to be able to fire X-rays. Therefore, the testing described in Deviation 1.4 (ODT5.18, ODT5.19, ODT5.22 and ODT5.25) could not be performed.  This testing will be completed prior to Phase 4 closure when a fully functional monoblocks is available.
During WS-003 verification testing ODT5.13 and ODT5.24 passed with deviations.
To verify ODT5.13, ODT Tripper provided a negative temperature value, a value between 0 and 80C and a value greater than 80C. During testing it was determined the LV PCBA firmware cannot measure negative temperatures because a -1 input yielded 655C.  However, this test failed as expected. In addition, when a value greater than 80 was input (81C) the test passed. After further evaluation it was determined ODT uses 87.5C as the upper temperature bound. ODT successfully demonstrated the ability to appropriately identify passing and failing temperatures and was therefore determined to pass with deviation.
To verify ODT5.24 Monoblock Calibration was initiated and successfully triggered an x-ray on the EUT. However, the energy emitted by the shot was not enough for the RadCal to detect. Therefore, this test passed with deviation since it fired an x-ray, but the x-ray could not be measured and the RadCal measurements could not be compared to the measurements taken by the EUT. The EUT passed the acceptance criteria of firing x-rays, but the remainder of the acceptance criteria could not be assessed. This acceptance criteria can be found in ODT5.18, ODT5.19, ODT5.22 and ODT5.25. Future testing with a functioning monoblock is required to verify this requirement fully.
During WS-005 verification testing the HMI Touch Test (ODT7.2) and HMI Button Test (ODT7.4) yielded results which warranted engineering evaluation of workstation performance. The issues noted were the last three touch locations failing and the button test not operating properly. Specifically, ODT asked the operator to Click the Resume button then press the left button. Once completed, ODT automatically failed the middle button and asked the operator to Click the Resume button then press the right button.
The ODT5.21 SRS should be updated to include Monoblock Calibration Value Test Filament and PWS values. PWS limits should be incorporated into this requirement. For the PWS data for the full temperature range:
R squared value less than 0.98 will fail
Max Over Limit value greater than +40 DAC codes shall fail
Max Under Limit less than -40 DAC codes shall fail (MBCAL_PWS_MAX_UNDERLESS)
Max Over Limit value greater than +60 DAC codes shall fail
Max Under Limit value less than -60 DAC codes shall fail (MBCAL_PWS_MAX_UNDER)
CONCLUSION
Overall Result:.
Pass
Fail
Other: Pass with deviations
LIST OF APPENDICES
Appendix 1 through Appendix 15 - Verification Evidence as Specified in Results Table 1.
REPORT APPROVAL
Digital Key: example.com/
Appendix 1a: ODT Software Information v2.2.1-beta
Appendix 1b: ODT Software Information v2.2.1
Appendix 1c: Device Type Test Results
Appendix 1d: ODT5.2 thru ODT5.4 evidence
Appendix 2a: Low Voltage Power Supply Setting (20V)
Appendix 2b: Nominal Voltage Power Supply Setting (28V)
Appendix 2c: High Voltage Power Supply Setting (36V)
Appendix 3a: Nominal Voltage Nominal Threshold Test Results
Appendix 3b: Nominal Voltage Low Threshold Test Results
Appendix 3c: Nominal Voltage High Threshold Test Results
Appendix 4a: Low Voltage Nominal Threshold Test Results
Appendix 4b: Low Voltage Low Threshold Test Results
Appendix 4c: Low Voltage High Threshold Test Results
Appendix 5a: High Voltage Nominal Threshold Test Results
Appendix 5b: High Voltage Low Threshold Test Results
Appendix 5c: High Voltage High Threshold Test Results
Appendix 6a: Monoblock Temperature Test Nominal Results
Appendix 6b: Monoblock Temperature Test Low Results
Appendix 6c: Monoblock Temperature Test High Results
Appendix 7: Filament Calibration Test Results
Appendix 8a:
Appendix 8b: Filament Gain and Offset Results with values at nominal
Appendix 8c: Filament Gain and Offset Results with values less than expected range
Appendix 8d: Filament Gain and Offset Results with values greater than expected range
Appendix 9: Vsense and Isense Calibration Initiation Results
Appendix 10a: Monoblock Calibration Value Tests: Vsense and Isense Offset and Gain Results
Appendix 10b: Monoblock Calibration Value Tests with values at Nominal Values expected to fail
Appendix 10c: Monoblock Calibration Value Tests with values lower than expected range
Appendix 10d: Monoblock Calibration Value Tests with values higher than expected range
Appendix 11a:
Appendix 11b:
Appendix 11c:
Appendix 12a: Monoblock Calibration Value Test Results for Filament and PWS Coefficients
Appendix 12b: Good Calibration Values Used for Testing
Appendix 12c: Monoblock Calibration Value Test Results for Filament and PWS Coefficients when data table altered to generate an R-squared value less than 0.98
Appendix 13: ODT MBC_TEST Ready for Operator to Run
Appendix 14a: ODT Temperature Control Behavior during Beam Calibration for Lower Limit, 15C
Appendix 14b: ODT Temperature Control Behavior during Beam Calibration for Upper Limit, 70C
Appendix 15: ODT Temperature Control Behavior during Monoblock Conditioning and Frequency Calibration Results
Appendix 16: ODT Temperature Control Behavior during Filament Calibration Results

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-003, X-ray Assembly Calibration WorkStation, MS-10155, MWI-222 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT5.1 | ODT shall query the Device Type of a connected Monoblock | Run Monoblock Register Test: Device Type Verify Device Type returns 0x01 | Monoblock Register Test: DeviceType returns 0x01 |  |  |
| ODT5.2 | ODT shall query the assembly revision of a connected Monoblock | Run Monoblock Register Test: BoardRevision Verify test returns an integer value that is not 0. | Monoblock Register Test: BoardRevision returns an integer value yields an ASCII integer value less that is not = 0 |  |  |
| ODT5.3 | ODT shall query the unique ID of a connected Monoblock's MCU | Run Monoblock Register Test: MUCUniqueID Verify test returns a 12 byte unique MCU ID | Monoblock Register Test: MCUUniqueID returns the 12 byte unique MCU ID |  |  |
| ODT5.4 | ODT shall query the Git Hash of a connected Monoblock | Run Monoblock Register Test: GitHash Verify test returns a GitHash | Monoblock Register Test: Githash returns a four-byte sequence that represent the Git hash of the FW/SW deployed on that Monoblock |  |  |
| ODT5.5 | ODT shall control power supply voltage output for Emitter Battery Voltage when running a voltage-dependent test | Verify Nominal Power Supply setting is 28V | When running a voltage test, the power supply will be configured to test the Monoblock at the Nominal, Low, and High expected battery voltage levels. Those are 28V, 20V, and 36V respectively Confirm ODT sets power supply correctly. |  |  |
|  |  | Verify Low Power Supply Setting is 20V | When running a voltage test, the power supply will be configured to test the Monoblock at Low expected battery voltage levels, 20V. Confirm ODT sets power supply correctly. |  |  |
|  |  | Verify High Power Supply Setting is 36V | When running a voltage test, the power supply will be configured to test the Monoblock at the High expected battery voltage levels, 36V. Confirm ODT sets power supply correctly. |  |  |
| ODT5.6 | ODT shall query the monitor of the High Side (XRAY_BATT) power rail of a connected Monoblock | Run Monoblock Voltage Test XRAY_BATT_NOM Verify ODT passes test when readings of the power supplied range from 26.6V to 29.4V | ODT shall pass the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied range from 26.6 - 29.4V, 19V - 21V, 34.2 - 37.8V when the power supply is set to 28V, 20V and 36V respectively. |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_NOM Verify ODT fails test when readings of the power supplied are less than 26.6 V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied are less than 26.6V when the power supply is set to 28V |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_NOM Verify ODT fails test when readings of the power supplied are greater than 29.4 V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied are greater than 29.4V when the power supply is set to 28V |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_LOW Verify ODT passes test when readings of the power supplied range from 19-21V | ODT shall pass the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied range from 19V - 21 V when the power supply is set to 20V. |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_LOW Verify ODT fails test when readings of the power supplied are less than 19V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_LOW/when readings of the power supplied are less than 19V when the power supply is set to 20V. |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_LOW Verify ODT fails test when readings of the power supplied are greater than 21V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_LOW/when readings of the power supplied are greater than 21 V when the power supply is set to 20V. |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_HIGH Verify ODT passes test when readings of the power supplied range from 34.2-37.8V | ODT shall pass the Monoblock Voltage Test XRAY_BATT_HIGH when readings of the power supplied range from 34.2 - 37.8 V when the power supply is set to 36V. |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_HIGH Verify ODT fails test when readings of the power supplied are less than 34.2V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_HIGH when readings of the power supplied are less than 34.2 V when the power supply is set to 36V. |  |  |
|  |  | Run Monoblock Voltage Test XRAY_BATT_HIGH Verify ODT fails test when readings of the power supplied are greater than 37.8V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_HIGH when readings of the power supplied are greater than 37.8V when the power supply is set to 36V. |  |  |
| ODT5.7 | ODT shall query the monitor of the 3.3V power rail of a connected Monoblock | Run Monoblock Voltage Test P3V3A Verify ODT passes with Nominal read back value ranging between 3.14 -3.46V | ODT shall pass the Monoblock Voltage Test P3V3A: NOM/LOW/HIGH when read back value ranges from 3.14V - 3.46V |  |  |
|  |  | Run Monoblock Voltage Test P3V3A Verify ODT fails with read back value less than 3.14V | ODT shall fail the Monoblock Voltage Test P3V3A: NOM/LOW/HIGH when read back value is less than 3.14V |  |  |
|  |  | Run Monoblock Voltage Test P3V3A Verify ODT fails with read back value greater than 3.46V | ODT shall fail the Monoblock Voltage Test P3V3A: NOM/LOW/HIGH when read back value is greater than 3.46V |  |  |
| ODT5.8 | ODT shall query the monitor of the 5.0V power rail of a connected Monoblock | Run Monoblock Voltage Test P5V0 Verify ODT passes with read back value ranging from 4.750- 5.25V | ODT shall pass the Monoblock Voltage Test P5V0: NOM/LOW/HIGH when read back values range from 4.75V - 5.25V |  |  |
|  |  | Run Monoblock Voltage Test P5V0 Verify ODT fails with read back value less than 4.750V | ODT shall fail the Monoblock Voltage Test P5V0: NOM/LOW/HIGH when read back value is less than 4.75V |  |  |
|  |  | Run Monoblock Voltage Test P5V0 Verify ODT fails with read back value greater than 5.25V | ODT shall fail the Monoblock Voltage Test P5V0: NOM/LOW/HIGH when read back value is greater than 5.25V |  |  |
| ODT5.9 | ODT shall query the monitor of the 15.0V filament power rail of a connected Monoblock | Run Monoblock Voltage Test P15V0FIL Verify ODT passes with nominal read back value ranging from 14.25 - 15.75V | ODT shall pass the Monoblock Voltage Test P15V0FIL: NOM/LOW/HIGH when read back values range from 14.25V - 15.75V. |  |  |
|  |  | Run Monoblock Voltage Test P15V0FIL Verify ODT fails with nominal read back value less than 14.25 | ODT shall fail the Monoblock Voltage Test P15V0FIL: NOM/LOW/HIGH when read back value is less than 14.25V. |  |  |
|  |  | Run Monoblock Voltage Test P15V0FIL Verify ODT fails with nominal read back value greater than 15.75V | ODT shall fail the Monoblock Voltage Test P15V0FIL: NOM/LOW/HIGH when read back value is greater than 15.75V. |  |  |
| ODT5.10 | ODT shall query the monitor of the 15.0V analog power rail of a connected Monoblock | Run Monoblock Voltage Test P15VA Verify ODT passes with nominal read back value ranging from 14.25 - 15.75V | ODT shall pass the Monoblock Voltage Test P15VA: NOM/LOW/HIGH when read back values 14.25V - 15.75V. |  |  |
|  |  | Run Monoblock Voltage Test P15VA Verify ODT fails with read back value less than 14.25 V | ODT shall fail the Monoblock Voltage Test P15VA: NOM/LOW/HIGH when read back value is less than 14.25V. |  |  |
|  |  | Run Monoblock Voltage Test P15VA Verify ODT fails with read back value greater than 15.75V | ODT shall fail the Monoblock Voltage Test P15VA: NOM/LOW/HIGH when read back value is greater than 15.75V. |  |  |
| ODT5.11 | ODT shall query the monitor of the -15.0V analog power rail of a connected Monoblock | Run Monoblock Voltage Test M15VA Verify ODT passes with nominal value ranging from -14.25 to -15.75V | ODT shall pass the Monoblock Voltage Test M15VA: NOM/LOW/HIGH when read back values range from -14.25V to -15.75V. |  |  |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with value less than -15.75V | ODT shall fail the Monoblock Voltage Test M15VA: NOM/LOW/HIGH when read back value is less than -15.75V. |  |  |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with values greater than -14.25V | ODT shall fail the Monoblock Voltage Test M15VA: NOM/LOW/HIGH when read back value is greater than -14.25V. |  |  |
| ODT5.12 | ODT shall query the monitor of the 3.0V reference power rail of a connected Monoblock | Run Monoblock Voltage Test P3V0 Verify ODT passes with nominal range values between 2.9 -3.03 V. | ODT shall pass the Monoblock Voltage Test P3V0: NOM/LOW/HIGH when read back values range from 2.97V - 3.03V. |  |  |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with value less than 2.9 V | ODT shall fail the Monoblock Voltage Test P3V0: NOM/LOW/HIGH when read back value is less than 2.97V. |  |  |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with value greater than 3.03 V | ODT shall fail the Monoblock Voltage Test P3V0: NOM/LOW/HIGH when read back value is greater than 3.03V. |  |  |
| ODT5.13 | ODT shall query the temperature measured by the thermocouples of a connected Monoblock | Run Monoblock Temperature Test Verify read back value is between 0 and 80C | Monoblock Temperature Test: Sensors 1 & 2 return readings between 0 and 80C to confirm functionality. |  |  |
|  |  | Run Monoblock Temperature Test Verify ODT fails when Sensor 1 and Sensor 2 read back values are less than 0 C | Monoblock Temperature Test: ODT shall fail when Sensors 1 & 2 return readings less than 0 C. |  |  |
|  |  | Run Monoblock Temperature Test Verify ODT fails when Sensor 1 and Sensor 2 read back values are greater than 80C | Monoblock Temperature Test: ODT shall fail when Sensors 1 & 2 return readings greater than 80 C. |  |  |
| ODT5.14 | ODT shall initiate filament calibration of a connected Monoblock | Run Filament Calibration Test Verify ODT initiates filament calibration of a connected Monoblock | ODT shall run the Filament Calibration Tests: Monoblock Calibration Value Test: Filament_Gain, Filament_Offset when Monoblock connected and Test QR code scanned |  |  |
| ODT5.15 | ODT shall run a Calibration Value Test to ensure Filament Gain output is 1 ± 0.75 and Offset Output is 0 ± 500 and not equal to nominal values | Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain passes when it is 1 ± 0.75 with nominal value failing (1.0=fail) Verify calculated filament Offset passes when it is 0 ± 500 with nominal value failing (0=fail) | The Filament Calibration Test generates two calibration factors: gain and offset. ODT Calibration Test shall pass when Gain value is 1.0 ±0.75 (0.25- 1.75) and Offset is 0.0 ±500 (-500 to +500), excluding the nominal value. |  |  |
|  |  | Overwrite <MCUUniqueID>.json file to have filament Gain = 1 and Filament Offset = 0 Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain fails when it is equal to 1 Verify calculated filament Offset fails when it is equal to 0 | ODT fails Filament Gain when it is equal to 1 ODT fails Filament Offset when it is equal to 0 |  |  |
|  |  | Overwrite <MCUUnique ID>.json file to have Filament Gain < 0.25 and filament Offset < -500 Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain fails when value is less than 0.25 Verify calculated filament Offset fails when value is less than -500 | ODT fails Filament Gain when it is less than 0.25 ODT fails Filament Offset when it is less than -500 |  |  |
|  |  | Overwrite <MCUUnique ID>.json file to have Filament Gain > 1.75 and filament Offset > 500 Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain fails when value is greater than 1.75 Verify calculated filament Offset fails when value is greater than 500 | ODT fails Filament Gain when it is greater than 1.75 ODT fails Filament Offset when it is greater than 500 |  |  |
| ODT5.16 | ODT shall initiate a calibration routine for calibrating current sense and voltage sense of a connected Monoblock | Run MBC_VSENSE and MBC_ISENSE Calibration Test Verify ODT initiates current and voltage sense calibration when Monoblock is connected | ODT initiates current and voltage sense calibration when Monoblock connected and Test QR code scanned |  |  |
| ODT5.17 | ODT shall run a Calibration Value Test to ensure Gain output for both is 1 ± 0.75, Isense Offset is ± 0.08 and Vsense Offset ± 9000 | Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output passes when value is within 1 ± 0.75 and not 1 Verify Isense Offset passes when value is within ± 0.08 and not 0 Verify Vsense Offset passes when value is within ± 9000 and not 0 | ODT shall pass Monoblock Calibration values: Vsense and Isense Gains between 1.0 ±0.75 (0.25 - 1.75) Isense Offset within 0 ±0.08 (-0.08 to 0.08) Vsense Offset within 0 ±9000 (-9000 to 9000) with exception of nominal values |  |  |
|  |  | Overwrite <MCUUniqueID>.json file to have Vsense Gain = 1 and Vsense Offset = 0, and Isense Gain = 1 and Isense Offset = 0 Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output fails when value is 1 Verify Isense Offset fails when value is 0 Verify Vsense Offset fails when value is 0 | ODT shall fail Monoblock Calibration when Vsense and Isense Gain and Offset equal nominal values |  |  |
|  |  | Overwrite <MCUUniqueID>.json file to have Vsense Gain < 0.25 and Vsenses Offset < -9000, and Isense Gain < 0.25 and Isense Offset < -0.08 Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output fails when value is less than 0.25 Verify Isense Offset fails when value is less than -0.08 Verify Vsense Offset fails when value is less than -9000 | ODT shall fail Monoblock Calibration when Vsense and Isense Gain are less than 0.25, Isense Offset is less than -0.08 and Vsense Offset is less than -9000 |  |  |
|  |  | Overwrite <MCUUniqueID>.json file to have Vsense Gain > 1.75 and Vsenses Offset > 9000, and Isense Gain > 1.75 and Isense Offset > 0.08 Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output fails when value is greater than 1.75 Verify Isense Offset fails when value is greater than 0.08 Verify Vsense Offset fails when value is greater than 9000 | ODT shall fail Monoblock Calibration when Vsense and Isense Gain are greater than 1.75, Isense Offset is greater than 0.08 and Vsense Offset is greater than 9000 |  |  |
| ODT5.18 | ODT shall measure beam current monitoring value within ±5% | Verify ODT passes Isense value within ± 5% of RadCal Measurement Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT passes Isense value is within ± 5% of RadCal Measurement | ODT shall pass Isense value within ±5% of RadCal measurement |  |  |
|  |  | Verify ODT fails Isense value not within ± 5% of RadCal Measurement Install ODT Tripper on LV PCBA Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT fails Isense value greater than ± 5% of RadCal Measurement | ODT shall fail Isense value not within ±5% of RadCal measurement |  |  |
| ODT5.19 | ODT shall measure voltage monitoring value within ±3% | Verify ODT passes Vsense value within ± 3% RadCal Measurement Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT passes Vsense value is within ± 3% of RadCal Measurement | ODT shall pass Vsense value within ±3% of RadCal measurement |  |  |
|  |  | Verify ODT fails Vsense value not within ± 3% RadCal Measurement Install ODT Tripper on LV PCBA Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT fails Vsense value greater than ± 3% of RadCal Measurement | ODT shall fail Vsense value not within ±3% of RadCal measurement |  |  |
| ODT5.20 | ODT shall initiate a check of the frequency tuning of a connected Monoblock | Run MBC_FREQVADJ Frequency Verify and Adjust Test Verify frequency tuning test steps are loaded and ready to start. | ODT shall pass an adjusted frequency with range 452 kHz - 500 kHz after 10 adjustment attempts |  |  |
|  |  | Overwrite <MCUUnique ID>.json file with frequency value less than 180MHz/358 Run MBC_FREQVADJ Frequency Verify and Adjust Test Verify frequency tuning test steps are loaded and ready to start. Verify MBC_FREQVADJ Test fails | ODT shall fail an adjusted frequency less than 452 kHz after 10 adjustment attempts |  |  |
|  |  | Overwrite <MCUUnique ID>.json file with frequency value greater than 180MHz/398 Run MBC_FREQVADJ Frequency Verify and Adjust Test Verify frequency tuning test steps are loaded and ready to start. Verify MBC_FREQVADJ Test fails | ODT shall fail an adjusted frequency greater than 500 kHz after 10 adjustment attempts |  |  |
| ODT5.21 | ODT shall initiate beam current calibration of a connected Monoblock | Run MBC_BEAMCAL Test Verify Beam Current Calibration test steps are loaded and ready to start Verify ODT passes when filCoef and pwsCoef that yield R2 value greater than 0.98, Max Over Limit less than or equal to 20 ADC codes and Min Under Limit greater than or equal to -20 ADC codes | ODT shall pass if calibration data table yields R2 value greater than 0.98, Max Over Limit value less than +20 ADC codes and Max Under Limit value greater than -20 ADC codes |  |  |
|  |  | Run MBC_BEAMCAL Test Modify data in table so have R2 less than 0.98, Max Over Limit greater than +20 ADC codes and Max Under Limit less than -20 ADC codes Verify Beam Current Calibration test steps are loaded and ready to start Verify ODT passes when filCoef and pwsCoef that yield R2 value greater than 0.98, Max Over Limit less than or equal to 20 ADC codes and Min Under Limit greater than or equal to -20 ADC codes | ODT shall fail if calibration data table yields R2 value less than 0.98, Max Over Limit value greater than +20 ADC codes and Max Under Limit less than -20 ADC codes |  |  |
| ODT5.22 | ODT shall measure output X-Ray Tube Beam current within ±8% | MBC_Test Monoblock Test Tab Number 15400 | ODT shall pass if set 1mA current and radCal measures 0.92 - 1.08mA |  |  |
|  |  | MBC_Test Monoblock Test Tab Number 15400 Verify ODT fails test when beam current more that -8% different from RadCal value | ODT shall fail if set 1mA current and radCal measures less than 0.92 mA |  |  |
|  |  | MBC_Test Monoblock Test Tab Number 15400 Verify ODT fails test when beam current more that 8% different from RadCal value | ODT shall fail if set 1mA current and radCal measures greater than 1.08 mA |  |  |
| ODT5.23 | ODT shall measure output X-Ray Tube Beam voltage within ±3% | MBC_Test Monoblock Test Tab Number 15400 | ODT shall pass if set 50 kV and RadCal measures 48.5 - 51.5 V |  |  |
| ODT5.24 | ODT shall initiate firing of x-rays | While running all Monoblock calibration tests with exception of Filament Calibration, Verify ODT initiates firing an x-ray | ODT shall pass when the Monoblock Fire X-ray Test results for a nominal mAs and kV x-ray yields the following values: Voltage measured by RadCal nominal kV +/- 3%) Current measured by RadCal nominal kV +/- 8% X-ray time measured by RadCal XX +/- 5% mAs measured by RadCal nominal +/- 10% Voltage measured by Vsense nominal +/- 3% Current measured by Isense nominal +/- 5% |  |  |
| ODT5.25 | The output Current Time Product shall be within ±10% | MBC_Test Monoblock Test Tab Number 15400 Charge value | ODT Current Time Product (mAs) value shall be within ±10% of the measured RadCal mAs value. |  |  |
| ODT5.27 | ODT shall perform a monoblock conditioning cycle on uncalibrated monoblock | Run MBC_CONDITION Test Verify condition test steps are loaded and ready to start | Need acceptance criteria - loose compared to other measurement accuracies because uncalibrated monoblock |  |  |
| ODT5.28 | MB Test shall be performed to ensure all calibration data is valid. | Run MBC_TEST Verify MBC_TEST test steps are loaded and ready to start | ODT shall pass calibration when all calibration data is determined to be valid: What are parameters and values? |  |  |
| ODT5.29 | ODT shall read and write calibration data for each monoblock to and from Google Drive | Verify calibration data is saved in calibration folder R&D/P01/Calibration <MCU unique ID>.json | ODT shall meet this requirement if a csv file for unique monoblock calibration is in the Calibration Folder on the Google Drive |  |  |
| ODT5.30 | ODT shall control monoblock temperature within the temperature chamber from 15 - 70 ±1 C during beam calibration | Populate a dummy file to say last run was at 15C, then stop Rewrite file to show at 70C, then stop Verify by creating local beam calibration file in ODT folder such that MBC_BEAMCAL will go to the temperature that was last tested | ODT shall meet this requirement if it can control the monoblock temperature within the temperature chamber from 15-70 C ±1 C |  |  |
| ODT5.31 | ODT shall control monoblock temperature within the temperature chamber to 25 ±3 C for conditioning and frequency calibration | Set temperature chamber beyond 25 ±3 C and verify ODT adjusts temperature until monoblock return 25 ±3 C | ODT shall meet this requirement if it can control the monoblock temperature with the temperature chamber to 25 ±3 C for conditioning and frequency calibration |  |  |
| ODT5.32 | ODT shall control monoblock temperature within the temperature chamber to 25 ±1 C for filament calibration | Set temperature chamber beyond 25 ±1 C and verify ODT adjusts temperature until monoblock return 25 ±1 C | ODT shall meet this requirement if it can control the monoblock temperature within the temperature chamber to 25 ±1 C for filament calibration |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-005, HMI WorkStation, MS-10401 HMI Display, MWI-226 |  |  |  |  |
| Precondition: | N/A |  |  |  |  |
| ODT7.1 | ODT shall command the target Raspberry Pi to exercise the HMI Display | Run HMI LCD test NOM Verify display dims and then fully illuminates | Display dims and then fully illuminates |  |  |
|  |  | Run HMI LCD test NOM Verify display dims and then fully illuminates | Display dims and then fully illuminates |  |  |
|  |  | Run HMI LCD test NOM Verify display dims and then fully illuminates | Display dims and then fully illuminates |  |  |
| ODT7.2 | ODT shall query at least 5 X/Y touch locations of a connected HMI Display and ensure touch locations are within 25% of specific values. | Run HMI Touch test Operator should touch X/Y coordinates within +/-25% Verify good touches pass Verify logged values are within specification | HMI Touch Test should pass when operator touches X/Y coordinates within +/- 25% and operator should confirm logged values are within specification. |  |  |
|  |  | Run HMI Touch test on ODT Operator should touch X/Y coordinates beyond +/-25% Verify bad touches fail Verify logged values are beyond specification | HMI Touch Test should fail when operator touches X/Y coordinates beyond +/- 25% and operator should confirm logged values are beyond specification |  |  |
| ODT7.3 | ODT shall allow 5 attempts to complete 5 X/Y touch locations successfully and fail after a fifth unsuccessful attempt | Run HMI Touch test Operator should touch X/Y coordinates beyond +/-25% for each of the 5 points Operator should repeat previous step 4 additional times Verify HMI touch test fails after 5 failing tests | Fail this test 5 times to confirm it fails |  |  |
| ODT7.4 | ODT shall query the HMI GPIO Expander for Button Tests | Run HMI Button 1 test (left) Operator follows prompts and depresses buttons. Verify successful button press advances test | Operator follows prompts and depresses buttons. Buttons tested sequentially. Pass advances and eventually Pass/Fail is logged |  |  |
|  |  | Run HMI Button 2 test (middle) Operator follows prompts and depresses buttons. Verify successful button press advances test |  |  |  |
|  |  | Run HMI Button 3 test (right) Operator follow prompts and depresses buttons, Verify successful button press advances test |  |  |  |
|  |  | Run HMI Button 3 test (no press) Operator starts test and does not depress button. Verify button test fails with NO button press | Button Test Press Fails with NO Button Press |  |  |
| ODT7.5 | ODT shall query the HMI GPIO Expander for Trigger Tests | Run HMI Trigger test: forward ODT reads value to determine if trigger pressed Verify test returns Pass/Fail in Data Log | Forward/Downward button Test returns Pass/Fail in Data Log |  |  |
|  |  | Run HMI Trigger test: downward ODT reads value to determine if trigger pressed Verify test returns Pass/Fail in Data Log |  |  |  |
| ODT7.6 | ODT shall issue a command to the target Raspberry Pi to set the display white | Run Display Pixel test Verify test sets the LCD to white to check for dead pixel | Display Test sets the LCD to white to check for dead pixels |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-541 |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-003, X-ray Assembly Calibration WorkStation, MS-10155, MWI-222 |  |  |  |  |
| Precondition: |  |  |  |  |  |
| ODT5.1 | ODT shall query the Device Type of a connected Monoblock | Run Monoblock Register Test: Device Type Verify Device Type returns 0x01 | Monoblock Register Test: DeviceType returns 0x01 | Expected outcome verified. See Appendix 1c. Verified by SA 03SEPT24 | P |
| ODT5.2 | ODT shall query the assembly revision of a connected Monoblock | Run Monoblock Register Test: BoardRevision Verify test returns an integer value that is not 0. | Monoblock Register Test: BoardRevision returns an integer value yields an ASCII integer value less that is not = 0 | Expected outcome verified. See Appendix 1d. Verified by SA 03SEPT24 | P |
| ODT5.3 | ODT shall query the unique ID of a connected Monoblock's MCU | Run Monoblock Register Test: MUCUniqueID Verify test returns a 12 byte unique MCU ID | Monoblock Register Test: MCUUniqueID returns the 12 byte unique MCU ID | Expected outcome verified. See Appendix 1d. Verified by SA 03SEPT24 | P |
| ODT5.4 | ODT shall query the Git Hash of a connected Monoblock | Run Monoblock Register Test: GitHash Verify test returns a GitHash | Monoblock Register Test: Githash returns a four-byte sequence that represent the Git hash of the FW/SW deployed on that Monoblock | Expected outcome verified. See Appendix 1d. Verified by SA 03SEPT24 | P |
| ODT5.5 | ODT shall control power supply voltage output for Emitter Battery Voltage when running a voltage-dependent test | Verify Nominal Power Supply setting is 28V | When running a voltage test, the power supply will be configured to test the Monoblock at the Nominal, Low, and High expected battery voltage levels. Those are 28V, 20V, and 36V respectively Confirm ODT sets power supply correctly. | Expected outcome verified. See Appendix 2b. Verified by SA 05SEPT24 | P |
|  |  | Verify Low Power Supply Setting is 20V | When running a voltage test, the power supply will be configured to test the Monoblock at Low expected battery voltage levels, 20V. Confirm ODT sets power supply correctly. | Expected outcome verified. See Appendix 2a. Verified by SA 05SEPT24 | P |
|  |  | Verify High Power Supply Setting is 36V | When running a voltage test, the power supply will be configured to test the Monoblock at the High expected battery voltage levels, 36V. Confirm ODT sets power supply correctly. | Expected outcome verified. See Appendix 2c. Verified by SA 05SEPT24 | P |
| ODT5.6 | ODT shall query the monitor of the High Side (XRAY_BATT) power rail of a connected Monoblock | Run Monoblock Voltage Test XRAY_BATT_NOM Verify ODT passes test when readings of the power supplied range from 26.6V to 29.4V | ODT shall pass the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied range from 26.6 - 29.4V, 19V - 21V, 34.2 - 37.8V when the power supply is set to 28V, 20V and 36V respectively. | Expected outcome verified. See Appendix 3a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_NOM Verify ODT fails test when readings of the power supplied are less than 26.6 V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied are less than 26.6V when the power supply is set to 28V | Expected outcome verified. See Appendix 3b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_NOM Verify ODT fails test when readings of the power supplied are greater than 29.4 V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied are greater than 29.4V when the power supply is set to 28V | Expected outcome verified. See Appendix 3c. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_LOW Verify ODT passes test when readings of the power supplied range from 19-21V | ODT shall pass the Monoblock Voltage Test XRAY_BATT_NOM/LOW/HIGH when readings of the power supplied range from 19V - 21 V when the power supply is set to 20V. | Expected outcome verified. See Appendix 4a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_LOW Verify ODT fails test when readings of the power supplied are less than 19V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_LOW/when readings of the power supplied are less than 19V when the power supply is set to 20V. | Expected outcome verified. See Appendix 4b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_LOW Verify ODT fails test when readings of the power supplied are greater than 21V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_LOW/when readings of the power supplied are greater than 21 V when the power supply is set to 20V. | Expected outcome verified. See Appendix 4c. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_HIGH Verify ODT passes test when readings of the power supplied range from 34.2-37.8V | ODT shall pass the Monoblock Voltage Test XRAY_BATT_HIGH when readings of the power supplied range from 34.2 - 37.8 V when the power supply is set to 36V. | Expected outcome verified. See Appendix 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_HIGH Verify ODT fails test when readings of the power supplied are less than 34.2V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_HIGH when readings of the power supplied are less than 34.2 V when the power supply is set to 36V. | Expected outcome verified. See Appendix 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test XRAY_BATT_HIGH Verify ODT fails test when readings of the power supplied are greater than 37.8V | ODT shall fail the Monoblock Voltage Test XRAY_BATT_HIGH when readings of the power supplied are greater than 37.8V when the power supply is set to 36V. | Expected outcome verified. See Appendix 5c. Verified by SA 05SEPT24 | P |
| ODT5.7 | ODT shall query the monitor of the 3.3V power rail of a connected Monoblock | Run Monoblock Voltage Test P3V3A Verify ODT passes with Nominal read back value ranging between 3.14 -3.46V | ODT shall pass the Monoblock Voltage Test P3V3A: NOM/LOW/HIGH when read back value ranges from 3.14V - 3.46V | Expected outcome verified. See Appendix 3a, 4a and 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P3V3A Verify ODT fails with read back value less than 3.14V | ODT shall fail the Monoblock Voltage Test P3V3A: NOM/LOW/HIGH when read back value is less than 3.14V | Expected outcome verified. See Appendix 3b, 4b and 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P3V3A Verify ODT fails with read back value greater than 3.46V | ODT shall fail the Monoblock Voltage Test P3V3A: NOM/LOW/HIGH when read back value is greater than 3.46V | Expected outcome verified. See Appendix 3c, 4c and 5c. Verified by SA 05SEPT24 | P |
| ODT5.8 | ODT shall query the monitor of the 5.0V power rail of a connected Monoblock | Run Monoblock Voltage Test P5V0 Verify ODT passes with read back value ranging from 4.750- 5.25V | ODT shall pass the Monoblock Voltage Test P5V0: NOM/LOW/HIGH when read back values range from 4.75V - 5.25V | Expected outcome verified. See Appendix 3a, 4a and 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P5V0 Verify ODT fails with read back value less than 4.750V | ODT shall fail the Monoblock Voltage Test P5V0: NOM/LOW/HIGH when read back value is less than 4.75V | Expected outcome verified. See Appendix 3b, 4b and 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P5V0 Verify ODT fails with read back value greater than 5.25V | ODT shall fail the Monoblock Voltage Test P5V0: NOM/LOW/HIGH when read back value is greater than 5.25V | Expected outcome verified. See Appendix 3c, 4c and 5c. Verified by SA 05SEPT24 | P |
| ODT5.9 | ODT shall query the monitor of the 15.0V filament power rail of a connected Monoblock | Run Monoblock Voltage Test P15V0FIL Verify ODT passes with nominal read back value ranging from 14.25 - 15.75V | ODT shall pass the Monoblock Voltage Test P15V0FIL: NOM/LOW/HIGH when read back values range from 14.25V - 15.75V. | Expected outcome verified. See Appendix 3a, 4a, 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P15V0FIL Verify ODT fails with nominal read back value less than 14.25 | ODT shall fail the Monoblock Voltage Test P15V0FIL: NOM/LOW/HIGH when read back value is less than 14.25V. | Expected outcome verified. See Appendix 3b, 4b, and 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P15V0FIL Verify ODT fails with nominal read back value greater than 15.75V | ODT shall fail the Monoblock Voltage Test P15V0FIL: NOM/LOW/HIGH when read back value is greater than 15.75V. | Expected outcome verified. See Appendix 3c, 4c, and 5c. Verified by SA 05SEPT24 | P |
| ODT5.10 | ODT shall query the monitor of the 15.0V analog power rail of a connected Monoblock | Run Monoblock Voltage Test P15VA Verify ODT passes with nominal read back value ranging from 14.25 - 15.75V | ODT shall pass the Monoblock Voltage Test P15VA: NOM/LOW/HIGH when read back values 14.25V - 15.75V. | Expected outcome verified. See Appendix 3a, 4a, and 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P15VA Verify ODT fails with read back value less than 14.25 V | ODT shall fail the Monoblock Voltage Test P15VA: NOM/LOW/HIGH when read back value is less than 14.25V. | Expected outcome verified. See Appendix 3b, 4b and 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test P15VA Verify ODT fails with read back value greater than 15.75V | ODT shall fail the Monoblock Voltage Test P15VA: NOM/LOW/HIGH when read back value is greater than 15.75V. | Expected outcome verified. See Appendix 3c, 4c and 5c. Verified by SA 05SEPT24 | P |
| ODT5.11 | ODT shall query the monitor of the -15.0V analog power rail of a connected Monoblock | Run Monoblock Voltage Test M15VA Verify ODT passes with nominal value ranging from -14.25 to -15.75V | ODT shall pass the Monoblock Voltage Test M15VA: NOM/LOW/HIGH when read back values range from -14.25V to -15.75V. | Expected outcome verified. See Appendix 3a, 4a and 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with value less than -15.75V | ODT shall fail the Monoblock Voltage Test M15VA: NOM/LOW/HIGH when read back value is less than -15.75V. | Expected outcome verified. See Appendix 3b, 4b and 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with values greater than -14.25V | ODT shall fail the Monoblock Voltage Test M15VA: NOM/LOW/HIGH when read back value is greater than -14.25V. | Expected outcome verified. See Appendix 3c, 4c and 5c. Verified by SA 05SEPT24 | P |
| ODT5.12 | ODT shall query the monitor of the 3.0V reference power rail of a connected Monoblock | Run Monoblock Voltage Test P3V0 Verify ODT passes with nominal range values between 2.9 -3.03 V. | ODT shall pass the Monoblock Voltage Test P3V0: NOM/LOW/HIGH when read back values range from 2.97V - 3.03V. | Expected outcome verified. See Appendix 3a, 4a and 5a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with value less than 2.9 V | ODT shall fail the Monoblock Voltage Test P3V0: NOM/LOW/HIGH when read back value is less than 2.97V. | Expected outcome verified. See Appendix 3b, 4b and 5b. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Voltage Test M15VA Verify ODT fails with value greater than 3.03 V | ODT shall fail the Monoblock Voltage Test P3V0: NOM/LOW/HIGH when read back value is greater than 3.03V. | Expected outcome verified. See Appendix 3c, 4c and 5c. Verified by SA 05SEPT24 | P |
| ODT5.13 | ODT shall query the temperature measured by the thermocouples of a connected Monoblock | Run Monoblock Temperature Test Verify read back value is between 0 and 80C | Monoblock Temperature Test: Sensors 1 & 2 return readings between 0 and 80C to confirm functionality. | Expected outcome verified. See Appendix 6a. Verified by SA 05SEPT24 | P |
|  |  | Run Monoblock Temperature Test Verify ODT fails when Sensor 1 and Sensor 2 read back values are less than 0 C | Monoblock Temperature Test: ODT shall fail when Sensors 1 & 2 return readings less than 0 C. | ODT Tripper input temperature of -1C yielded a temperature measurement of 655C. ODT successfully identified this as a failure. Determined LV PCBA firmware cannot measure negative temperature values. See Appendix 6b. Verified by SA 05SEPT24 | Pass with deviation |
|  |  | Run Monoblock Temperature Test Verify ODT fails when Sensor 1 and Sensor 2 read back values are greater than 80C | Monoblock Temperature Test: ODT shall fail when Sensors 1 & 2 return readings greater than 80 C. | ODT Tripper input temperature of 81C yielded passing temperature value. Determined ODT upper temperature limit is 87.5 C. ODT successfully identified temperature value within range. See Appendix 6c> Verified by SA 05SEPT24 | Pass with deviation |
| ODT5.14 | ODT shall initiate filament calibration of a connected Monoblock | Run Filament Calibration Test Verify ODT initiates filament calibration of a connected Monoblock | ODT shall run the Filament Calibration Tests: Monoblock Calibration Value Test: Filament_Gain, Filament_Offset when Monoblock connected and Test QR code scanned | Expected outcome verified. Test initiated through ODT Engineering mode. Demonstrated QR code scanned properly. See Appendix 7. Verified by SA 04SEPT24 25±1 | P |
| ODT5.15 | ODT shall run a Calibration Value Test to ensure Filament Gain output is 1 ± 0.75 and Offset Output is 0 ± 500 and not equal to nominal values | Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain passes when it is 1 ± 0.75 with nominal value failing (1.0=fail) Verify calculated filament Offset passes when it is 0 ± 500 with nominal value failing (0=fail) | The Filament Calibration Test generates two calibration factors: gain and offset. ODT Calibration Test shall pass when Gain value is 1.0 ±0.75 (0.25- 1.75) and Offset is 0.0 ±500 (-500 to +500), excluding the nominal value. | Expected outcome verified. Filament Offset = 1.00 Filament Gain = 1.13 data from MB NVM. See Appendix 8a. Verified by SA 04SEPT24 300033000B51323433383532 | P |
|  |  | Overwrite <MCUUniqueID>.json file to have filament Gain = 1 and Filament Offset = 0 Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain fails when it is equal to 1 Verify calculated filament Offset fails when it is equal to 0 | ODT fails Filament Gain when it is equal to 1 ODT fails Filament Offset when it is equal to 0 | Expected outcome verified. See Appendix 8b. Verified by SA 04SEPT24 | P |
|  |  | Overwrite <MCUUnique ID>.json file to have Filament Gain < 0.25 and filament Offset < -500 Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain fails when value is less than 0.25 Verify calculated filament Offset fails when value is less than -500 | ODT fails Filament Gain when it is less than 0.25 ODT fails Filament Offset when it is less than -500 | Expected outcome verified. Read back values 0.24 and -501 Fail. See Appendix 8c. Verified by SA 04SEPT24 | P |
|  |  | Overwrite <MCUUnique ID>.json file to have Filament Gain > 1.75 and filament Offset > 500 Run Monoblock Calibration Value Test: Filament_Gain, Filament_Offset Verify calculated filament Gain fails when value is greater than 1.75 Verify calculated filament Offset fails when value is greater than 500 | ODT fails Filament Gain when it is greater than 1.75 ODT fails Filament Offset when it is greater than 500 | Expected outcome verified. Read back values 1.76 and 501 Fail. See Appendix 8d. Verified by SA 04SEPT24 | P |
| ODT5.16 | ODT shall initiate a calibration routine for calibrating current sense and voltage sense of a connected Monoblock | Run MBC_VSENSE and MBC_ISENSE Calibration Test Verify ODT initiates current and voltage sense calibration when Monoblock is connected | ODT initiates current and voltage sense calibration when Monoblock connected and Test QR code scanned | Re-entered calibrated Gain & Offset values to correct values, Started Cal, Vsense readback 8390V which was too low to trigger RadCal. RadCal timed out which caused test failure. ODT successfully identified failure. Expected Results with faulty monoblock.See Appendix 9. Verified by SA 04SEPT24 | P |
| ODT5.17 | ODT shall run a Calibration Value Test to ensure Gain output for both is 1 ± 0.75, Isense Offset is ± 0.08 and Vsense Offset ± 9000 | Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output passes when value is within 1 ± 0.75 and not 1 Verify Isense Offset passes when value is within ± 0.08 and not 0 Verify Vsense Offset passes when value is within ± 9000 and not 0 | ODT shall pass Monoblock Calibration values: Vsense and Isense Gains between 1.0 ±0.75 (0.25 - 1.75) Isense Offset within 0 ±0.08 (-0.08 to 0.08) Vsense Offset within 0 ±9000 (-9000 to 9000) with exception of nominal values | Expected outcome verified. ODT passed input values: Vsense Gain = 0.26 Isense Gain =1.74 Vsense Offset = -8999 Isense Offset = 0.079 See Appendix 10a. Verified by SA 04SEPT24 Determined-Fix req’t to be less than all values not less than & equal to | P |
|  |  | Overwrite <MCUUniqueID>.json file to have Vsense Gain = 1 and Vsense Offset = 0, and Isense Gain = 1 and Isense Offset = 0 Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output fails when value is 1 Verify Isense Offset fails when value is 0 Verify Vsense Offset fails when value is 0 | ODT shall fail Monoblock Calibration when Vsense and Isense Gain and Offset equal nominal values | Expected outcome verified. ODT failed input values: Vsense Gain = 1 Isense Gain = 1 Vsense Offset = 0 Isense Offset = 0 See Appendix 10b. Verified by SA 04SEPT24 | P |
|  |  | Overwrite <MCUUniqueID>.json file to have Vsense Gain < 0.25 and Vsenses Offset < -9000, and Isense Gain < 0.25 and Isense Offset < -0.08 Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output fails when value is less than 0.25 Verify Isense Offset fails when value is less than -0.08 Verify Vsense Offset fails when value is less than -9000 | ODT shall fail Monoblock Calibration when Vsense and Isense Gain are less than 0.25, Isense Offset is less then -0.08 and Vsense Offset is less than -9000 | Expected outcome verified. ODT failed input values: Vsense Gain= 0.24 Isense Gain = 0.24 Vsense Offset = -9001 Isense Offset = -0.081 See Appendix 10c. Verified by SA 04SEPT24 | P |
|  |  | Overwrite <MCUUniqueID>.json file to have Vsense Gain > 1.75 and Vsenses Offset > 9000, and Isense Gain > 1.75 and Isense Offset > 0.08 Run Monoblock Calibration Value Test: Vsense_Gain, Vsesne_Offset, Isense_Gain, Isense_Offset Verify Isense and Vsense Gain output fails when value is greater than 1.75 Verify Isense Offset fails when value is greater than 0.08 Verify Vsense Offset fails when value is greater than 9000 | ODT shall fail Monoblock Calibration when Vsense and Isense Gain are greater than 1.75, Isense Offset is greater then 0.08 and Vsense Offset is greater than 9000 | Expected outcome verified. ODT failed input values: Vsense Gain = 1.76 Isense Gain = 1.76 Isense Offset = 0.081 Vsense Offset = 9001 See Appendix 10d. Verified by SA 04SEPT24 | P |
| ODT5.18 | ODT shall measure beam current monitoring value within ±5% | Verify ODT passes Isense value within ± 5% of RadCal Measurement Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT passes Isense value is within ± 5% of RadCal Measurement | ODT shall pass Isense value within ±5% of RadCal measurement | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
|  |  | Verify ODT fails Isense value not within ± 5% of RadCal Measurement Install ODT Tripper on LV PCBA Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT fails Isense value greater than ± 5% of RadCal Measurement | ODT shall fail Isense value not within ±5% of RadCal measurement | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
| ODT5.19 | ODT shall measure voltage monitoring value within ±3% | Verify ODT passes Vsense value within ± 3% RadCal Measurement Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT passes Vsense value is within ± 3% of RadCal Measurment | ODT shall pass Vsense value within ±3% of RadCal measurement | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
|  |  | Verify ODT fails Vsense value not within ± 3% RadCal Measurement Install ODT Tripper on LV PCBA Select Monoblock tab in ODT Engineering mode Select Test 15400 Verify ODT fails Vsense value greater than ± 3% of RadCal Measurment | ODT shall fail Vsense value not within ±3% of RadCal measurement | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA SA 04SEPT24 | NE |
| ODT5.20 | ODT shall initiate a check of the frequency tuning of a connected Monoblock | Run MBC_FREQVADJ Frequency Verify and Adjust Test Verify frequency tuning test steps are loaded and ready to start. | ODT shall pass an adjusted frequency with range 452 kHz - 500 kHz after 10 adjustment attempts | Expected outcome verified. ODT passes Divider parameter = 378 ODT assesses divider parameter used to calculate frequency not frequency. See Appendix 11a. Verified by SA 04SEPT24 | P |
|  |  | Overwrite <MCUUnique ID>.json file with divider value less than 358 Run MBC_FREQVADJ Frequency Verify and Adjust Test Verify frequency tuning test steps are loaded and ready to start. Verify MBC_FREQVADJ Test fails | ODT shall fail an adjusted frequency less than 452 kHz after 10 adjustment attempts | Expected outcome verified. MB Cal Test 3400-14 run Freq = 451 kHz ODT assesses divider parameter used to calculate frequency not frequency. Divider value input(357) which is less than minimum limit(358) failed as expected. See Appendix 11b. Verified by SA 04SEPT24 | P |
|  |  | Overwrite <MCUUnique ID>.json file with divider value greater than 398Run MBC_FREQVADJ Frequency Verify and Adjust Test Verify frequency tuning test steps are loaded and ready to start. Verify MBC_FREQVADJ Test fails | ODT shall fail an adjusted frequency greater than 500 (should be 503 - 180/358) kHz after 10 adjustment attempts | Expected outcome verified. Divider parameter input value (399) which is greater than upper limit (398) failed. See Appendix 11c. Verified by SA 04SEPT24 | P |
| ODT5.21 | ODT shall initiate beam current calibration of a connected Monoblock | Run MBC_BEAMCAL Test Verify Beam Current Calibration test steps are loaded and ready to start Verify ODT passes when filCoef yields R2 value greater than or equal to 0.98, Max Over Limit less than 20 ADC codes, Min Under Limit greater than -20 ADC codes and pwsCoef yields R2 value greater than or equal to 0.98, Max Over Limit value is less than +40 DAC codes and Max Under Limit is greater than -40 DAC codes, and PWS Max Over Limit is less than +60 DAC codes and Max Under Limit is greater than -60 DAC codes for the full temperature range. | ODT shall pass if calibration data table yields R2 value greater than 0.98, Max Over Limit value less than +20 ADC codes and Max Under Limit value greater than -20 ADC codes, and pwsCoef yields R2 value greater than or equal to 0.98, Max Over Limit value is less than +40 DAC codes and Max Under Limit is greater than -40 DAC codes, and PWS Max Over Limit is less than +60 DAC codes and Max Under Limit is greater than -60 DAC codes for the full temperature range. | Expected outcome verified. Values used saved in screen shot of Engineering Tools Run 3700-15, 3800-16, 3900-17, 4000-18, 4100-19, 4200-20, 4300-21 Good data values PASS See Appendix 12a and 12b. Verified by SA 04SEPT24 | P |
|  |  | Run MBC_BEAMCAL Test Modify data in table so have R2 less than 0.98, Max Over Limit greater than +20 ADC codes and Max Under Limit less than -20 ADC codes Verify Beam Current Calibration test steps are loaded and ready to start Verify ODT passes when filCoef yields R2 value greater than 0.98, Max Over Limit less than or equal to 20 ADC codes and Min Under Limit greater than or equal to -20 ADC codes, and pwsCoef yields R2 value greater than or equal to 0.98, Max Over Limit value is greater than +40 DAC codes and Max Under Limit is less than -40 DAC codes, and PWS Max Over Limit is greater than +60 DAC codes and Max Under Limit less than -60 DAC codes for the full temperature range. | ODT shall fail if filCoef calibration data table yields R2 value less than 0.98, Max Over Limit value greater than +20 ADC codes and Max Under Limit less than -20 ADC codes, and pwsCoef yields R2 value less than or 0.98, Max Over Limit value is greater than +40 DAC codes and Max Under Limit is less than -40 DAC codes, and PWS Max Over Limit is greater than +60 DAC codes and Max Under Limit less than -60 DAC codes for the full temperature range. | Expected outcome verified. Beam Cal data table with values beyond limits fail. Updating json file only - coefficients - leaving data table the same Changed values are NOT causing failure - not as expected - changing coefficients should change calculated table which is being compared to actual data table. Test only assessing quality of data in table. No comparison occurring See Appendix 12c. Verified by SA 05SEPT24 | P |
| ODT5.22 | ODT shall measure output X-Ray Tube Beam current within ±8% | MBC_Test Monoblock Test Tab Number 15400 | ODT shall pass if set 1mA current and radCal measures 0.92 - 1.08mA | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
|  |  | MBC_Test Monoblock Test Tab Number 15400 Verify ODT fails test when beam current more that -8% different from RadCal value | ODT shall fail if set 1mA current and radCal measures less than 0.92 mA | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
|  |  | MBC_Test Monoblock Test Tab Number 15400 Verify ODT fails test when beam current more that 8% different from RadCal value | ODT shall fail if set 1mA current and radCal measures greater than 1.08 mA | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
| ODT5.23 | ODT shall measure output X-Ray Tube Beam voltage within ±3% | MBC_Test Monoblock Test Tab Number 15400 | ODT shall pass if set 50 kV and RadCal measures 48.5 - 51.5 V | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
| ODT5.24 | ODT shall initiate firing of x-rays | While running all Monoblock calibration tests with exception of Filament Calibration, Verify ODT initiates firing an x-ray | ODT shall pass when the Monoblock Fire X-ray Test results for a nominal mAs and kV x-ray yields the following values: Voltage measured by RadCal nominal kV +/- 3%) Current measured by RadCal nominal kV +/- 8% X-ray time measured by RadCal XX +/- 5% mAs measured by RadCal nominal +/- 10% Voltage measured by Vsense nominal +/- 3% Current measured by Isense nominal +/- 5% | In the execution of ODT5.18, ODT5.19 and ODT5.22 it was determined an x-ray was fired but the device did not emit enough radiation to detect. Verified by SA 04SEPT2024 | Pass with deviation |
| ODT5.25 | The output Current Time Product shall be within ±10% | MBC_Test Monoblock Test Tab Number 15400 Change value | ODT Current Time Product (mAs) value shall be within ±10% of the measured RadCal mAs value. | Faulty Monoblock could not fire x-ray RadCal could detect Verified by SA 04SEPT24 | NE |
| ODT5.27 | ODT shall perform a monoblock conditioning cycle on uncalibrated monoblock | Run MBC_CONDITION Test Verify condition test steps are loaded and ready to start | Need acceptance criteria - loose compared to other measurement accuracies because uncalibrated monoblock | Expected outcome verified. Verified by SA 04SEPT24 | P |
| ODT5.28 | MB Test shall be performed to ensure all calibration data is valid. | Run MBC_TEST Verify MBC_TEST test steps are loaded and ready to start | ODT shall pass calibration when all calibration data is determined to be valid: What are parameters and values? | Expected outcome demonstrated. See Appendix 13. Verified by SA 04SEPT24 | P |
| ODT5.29 | ODT shall read and write calibration data for each monoblock to and from Google Drive | Verify calibration data is saved in calibration folder R&D/P01/Calibration <MCU unique ID>.json | ODT shall meet this requirement if a csv file for unique monoblock calibration is in the Calibration Folder on the Google Drive | Expected outcome demonstrated. Demonstrated during Filament Cal. Verified by SA 05SEPT24 | P |
| ODT5.30 | ODT shall control monoblock temperature within the temperature chamber from 15 - 70 ±1 C during beam calibration | Populate a dummy file to say last run was at 15C, then stop Rewrite file to show at 70C, then stop Verify by creating local beam calibration file in ODT folder such that MBC_BEAMCAL will go to the temperature that was last tested | ODT shall meet this requirement if it can control the monoblock temperature within the temperature chamber from 15-70 C ±1 C | Expected outcome verified. Erased beam cal file Altered local beam cal file. See Appendix 14a and 14b. Verified by SA 04SEPT24 | P |
| ODT5.31 | ODT shall control monoblock temperature within the temperature chamber to 25 ±3 C for conditioning and frequency calibration | Set temperature chamber beyond 25 ±3 C and verify ODT adjusts temperature until monoblock return 25 ±3 C | ODT shall meet this requirement if it can control the monoblock temperature with the temperature chamber to 25 ±3 C for conditioning and frequency calibration | Expected outcome verified. Ran 1600 Stopped at 21.97C See Appendix 15. Verified by SA 04SEPT24 | P |
| ODT5.32 | ODT shall control monoblock temperature within the temperature chamber to 25 ±1 C for filament calibration | Set temperature chamber beyond 25 ±1 C and verify ODT adjusts temperature until monoblock return 25 ±1 C | ODT shall meet this requirement if it can control the monoblock temperature within the temperature chamber to 25 ±1 C for filament calibration | Expected outcome verified. Temp control stopped at 26.02C See Appendix 16. Verified by SA 04SEPT24 | P |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | WS-005, HMI WorkStation, MS-10401 HMI Display, MWI-226 |  |  |  |  |
| Precondition: | N/A |  |  |  |  |
| ODT7.1 | ODT shall command the target Raspberry Pi to exercise the HMI Display | Run HMI LCD test NOM Verify display dims and then fully illuminates | Display dims and then fully illuminates |  | NE |
|  |  | Run HMI LCD test NOM Verify display dims and then fully illuminates | Display dims and then fully illuminates |  | NE |
|  |  | Run HMI LCD test NOM Verify display dims and then fully illuminates | Display dims and then fully illuminates |  | NE |
| ODT7.2 | ODT shall query at least 5 X/Y touch locations of a connected HMI Display and ensure touch locations are within 25% of specific values. | Run HMI Touch test Operator should touch X/Y coordinates within +/-25% Verify good touches pass Verify logged values are within specification | HMI Touch Test should pass when operator touches X/Y coordinates within +/- 25% and operator should confirm logged values are within specification. |  | NE |
|  |  | Run HMI Touch test on ODT Operator should touch X/Y coordinates beyond +/-25% Verify bad touches fail Verify logged values are beyond specification | HMI Touch Test should fail when operator touches X/Y coordinates beyond +/- 25% and operator should confirm logged values are beyond specification |  | NE |
| ODT7.3 | ODT shall allow 5 attempts to complete 5 X/Y touch locations successfully and fail after a fifth unsuccessful attempt | Run HMI Touch test Operator should touch X/Y coordinates beyond +/-25% for each of the 5 points Operator should repeat previous step 4 additional times Verify HMI touch test fails after 5 failing tests | Fail this test 5 times to confirm it fails |  | NE |
| ODT7.4 | ODT shall query the HMI GPIO Expander for Button Tests | Run HMI Button 1 test (left) Operator follows prompts and depresses buttons. Verify successful button press advances test | Operator follows prompts and depresses buttons. Buttons tested sequentially. Pass advances and eventually Pass/Fail is logged |  | NE |
|  |  | Run HMI Button 2 test (middle) Operator follows prompts and depresses buttons. Verify successful button press advances test |  |  | NE |
|  |  | Run HMI Button 3 test (right) Operator follow prompts and depresses buttons, Verify successful button press advances test |  |  | NE |
|  |  | Run HMI Button 3 test (no press) Operator starts test and does not depress button. Verify button test fails with NO button press | Button Test Press Fails with NO Button Press |  | NE |
| ODT7.5 | ODT shall query the HMI GPIO Expander for Trigger Tests | Run HMI Trigger test: forward ODT reads value to determine if trigger pressed Verify test returns Pass/Fail in Data Log | Forward/Downward button Test returns Pass/Fail in Data Log |  | NE |
|  |  | Run HMI Trigger test: downward ODT reads value to determine if trigger pressed Verify test returns Pass/Fail in Data Log |  |  | NE |
| ODT7.6 | ODT shall issue a command to the target Raspberry Pi to set the display white | Run Display Pixel test Verify test sets the LCD to white to check for dead pixel | Display Test sets the LCD to white to check for dead pixels |  | NE |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release; Update Protocol to include ODT5.12- ODT5.32 | Refer to ECR-551 |  |  |
