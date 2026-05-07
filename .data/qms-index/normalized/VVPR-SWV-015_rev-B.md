# VVPR-SWV-015 Rev B: EDID Writer Verification and Validation v1.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-SWV-015
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v1.0.0
- Source filename: VVPR-SWV-015 - EDID Writer Verification and Validation v1.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-SWV-015 - EDID Writer Verification and Validation v1.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 EDID Writer meets usability and functional requirements as stated in MEMO-P01-703 - EDID Writer Software Requirements Specification.
OBJECTIVE
The primary objective of this study is to verify the Non-Product Software Tool: MX1 EDID Writer for use in production of the MX1 E1 Emitter.
REFERENCES
MEMO-P01-703 - EDID Writer Software Requirements Specification Rev. A
MATERIALS
MX1 E1 Emitter Rev. H
MX1 SS Version v3.1.0
S10059 MX1 EDID Writer v1.0.0
A computer capable of running a web browser
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Experimental Procedure
Follow the steps outlined in the table(s) below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Data Analysis
All of the verification tests in Table 1 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Table 1 per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: DV-26
MX1 SS Version v3.1.0
S10059 MX1 EDID Writer v1.0.0
RESULTS
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
APPENDIX/ATTACHMENTS
Appendix 1: Emitter Screen not functioning properly
Appendix 2: Emitter Display Functioning Properly
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The Emitter is powered on and a remote connection has been established via SSH. |  |  |  |  |
| SRS-1.1 | The EW shall write to bus 3 and i2c address 0x50 | 1. Run the command python3.11 -m mx1.services start 2. Run the following command "i2cdetect -y -r 3" | Verify that the output of step 1 matches the below: 0 1 2 3 4 5 6 7 8 9 a b c d e f 00: -- -- -- -- -- -- -- -- -- -- -- -- -- 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 70: -- -- -- -- -- -- -- -- |  |  |
|  |  | 1. Run the following commands to enable the DDC bus $ su $ echo 1 > /sys/kernel/debug/tegra_hdmi/ddc_power_toggle $ exit 2. Run the following command "i2cdetect -y -r 3" to confirm the DDC bus has been enabled | Verify the output of step 3 matches the below 0 1 2 3 4 5 6 7 8 9 a b c d e f 00: -- -- -- -- -- -- -- -- -- -- -- -- -- 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 50: 50 51 52 53 54 55 56 57 -- -- -- -- -- -- -- -- 60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 70: -- -- -- -- -- -- -- -- |  |  |
|  |  | 1. Run the command: /opt/medai/bin/write-edid.sh -f ./blank-edid-eraser.bin and press y 2. Power the Emitter off and back on and observe the HMI display | Verify the HMI display is not functioning correctly (screen image exhibiting de-synced scrolling and offset white bar). |  |  |
|  |  | 1. Repeat steps 3 and 4 to enable the DDC bus 2. Run the command: ./write-edid.sh -f ./emitter-edid.bin and press y 3. Power the Emitter off and back on and observe the HMI display | Verify that the HMI display is now functioning correctly (stable image with no artifacts). |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 28 Jun 2024 | 24-397 |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Setup: | Device Components Needed: E1 Emitter |  |  |  |  |
| Precondition: | The Emitter is powered on and a remote connection has been established via SSH. |  |  |  |  |
| SRS-1.1 | The EW shall write to bus 3 and i2c address 0x50 | 1. Run the command python3.11 -m mx1.services start 2. Run the following command "i2cdetect -y -r 3" | Verify that the output of step 1 matches the below: 0 1 2 3 4 5 6 7 8 9 a b c d e f 00: -- -- -- -- -- -- -- -- -- -- -- -- -- 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 70: -- -- -- -- -- -- -- -- | Expected outcome verified. Verified by MS 28JUNE24 | P |
|  |  | 1. Run the following commands to enable the DDC bus $ su $ echo 1 > /sys/kernel/debug/tegra_hdmi/ddc_power_toggle $ exit 2. Run the following command "i2cdetect -y -r 3" to confirm the DDC bus has been enabled | Verify the output of step 3 matches the below 0 1 2 3 4 5 6 7 8 9 a b c d e f 00: -- -- -- -- -- -- -- -- -- -- -- -- -- 10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 50: 50 51 52 53 54 55 56 57 -- -- -- -- -- -- -- -- 60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 70: -- -- -- -- -- -- -- -- | Expected outcome verified. Verified by MS 28JUNE24 | P |
|  |  | 1. Run the command: /opt/medai/bin/write-edid.sh -f ./blank-edid-eraser.bin and press y 2. Power the Emitter off and back on and observe the HMI display | Verify the HMI display is not functioning correctly (screen image exhibiting de-synced scrolling and offset white bar). | Expected outcome verified. See Appendix 1. Verified by MS 28JUNE24 | P |
|  |  | 1. Repeat steps 3 and 4 to enable the DDC bus 2. Run the command: ./write-edid.sh -f ./emitter-edid.bin and press y 3. Power the Emitter off and back on and observe the HMI display | Verify that the HMI display is now functioning correctly (stable image with no artifacts). | Expected outcome verified. See Appendix 2. Verified by MS 28JUNE24 | P |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-484 |  |  |
