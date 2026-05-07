# VVPR-P01-219 Rev B: MX1 Software System v3.2.1 Protocol and Report

## Metadata
- Document ID: VVPR-P01-219
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.2.1
- Source filename: VVPR-P01-219 - MX1 Software System v3.2.1 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-219 - MX1 Software System v3.2.1 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Automatic software “roll back”
Over-The-Air (OTA) OS updates
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.2.1 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. F
IFU-MX1 - Instructions for Use, Rev. G
MATERIALS
E1 Emitter BOM Rev. H
C1 Cassette BOM Rev. I
M50133 Rev. A, Galaxy Tablet  S8+
MX1 App, v3.2.1
MX1 Software System, v3.2.1
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Suite 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.  If steps require x-ray emission, use appropriate radiation protective equipment.
Table 1. Remote Upgrades - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1222
C1 Cassette Rev. I, SN: 1223
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System v3.2.1
RESULTS
Table 1. Remote Upgrades - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 and 4 - Verification Evidence as Specified in Results Table 1.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1: Emitter Jetson Boot Partition Before Upgrade
Appendix 2: CassetteJetson Boot Partition Before Upgrade
Appendix 3: Emitter Jetson Boot Partition After Upgrade
Appendix 4: Cassette Jetson Boot Partition After Upgrade

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. |  |  |  |  |
|  | Test Case: OTA Updates of MX1 Software System - Automatic Rollback |  |  |  |  |
| SRS-45.17 | The SS shall be architected in a way such that the deployed version of the software automatically "rolls back" to a previous version in less than an hour | 1. SSH into the emitter 2. Enter nvbootctrl get-current-slot 3. Record the reported partition the device is currently utilizing 4. Repeat steps 1 through 3 for the cassette under test 5. Ensure the cassette under test has a network connection 6. Create and initiate deployments in Mender for the test emitter and cassette 7. When update modals appear on the MedAI Device App, tap “Install” for both emitter and cassette installs 8. Power cycle both the emitter and cassette while in the middle of an upgrade. Use the update progress bar to determine when to power cycle (e.g. at 20% of the upgrade) 9. Restart the devices 10. Repeat steps 1 through 4 11. Verify that both emitter and cassette Jetsons remain on the same partition in the case of a failed update | Recorded boot partition on emitter before upgrade attempt |  |  |
|  |  |  | Recorded boot partition on cassette before upgrade attempt |  |  |
|  |  |  | Boot partition on emitter remains the same after failed upgrade |  |  |
|  |  |  | Boot partition on cassette remains the same after failed upgrade |  |  |
|  | Test Case: OTA Updates of MX1 Software System - Operating System Updates |  |  |  |  |
| SRS-45.5 | The SS shall support OTA updates of the emitter and cassette via Operating System Images | 1. SSH into the emitter 2. Enter nvbootctrl get-current-slot 3. Record the reported partition the device is currently utilizing 4. Repeat steps 1 through 3 for the cassette under test 5. Ensure the cassette under test has a network connection 6. Create and initiate deployments in Mender for the test emitter and cassette. 7. When update modals appear on the MedAI Device App, tap “Install” for both emitter and cassette installs 8. Wait for the update to complete. The update progress bars in the MedAI Device App will indicate a completed update. 9. Power cycle the devices under test 10. Repeat steps 1 through 4 11. Verify that both emitter and cassette Jetsons switch to partitions in the case of a successful update | Recorded boot partition on emitter before upgrade attempt |  |  |
|  |  |  | Recorded boot partition on cassette before upgrade attempt |  |  |
|  |  |  | Boot partition on emitter switches after successful upgrade |  |  |
|  |  |  | Boot partition on cassette switches after successful upgrade |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 09 Oct 2024 | 24-587 |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. |  |  |  |  |
|  | Test Case: OTA Updates of MX1 Software System - Automatic Rollback |  |  |  |  |
| SRS-45.17 | The SS shall be architected in a way such that the deployed version of the software automatically "rolls back" to a previous version in less than an hour | 1. SSH into the emitter 2. Enter nvbootctrl get-current-slot 3. Record the reported partition the device is currently utilizing 4. Repeat steps 1 through 3 for the cassette under test 5. Ensure the cassette under test has a network connection 6. Create and initiate deployments in Mender for the test emitter and cassette 7. When update modals appear on the MedAI Device App, tap “Install” for both emitter and cassette installs 8. Power cycle both the emitter and cassette while in the middle of an upgrade. Use the update progress bar to determine when to power cycle (e.g. at 20% of the upgrade) 9. Restart the devices 10. Repeat steps 1 through 4 11. Verify that both emitter and cassette Jetsons remain on the same partition in the case of a failed update | Recorded boot partition on emitter before upgrade attempt | Expected outcome verified. Current emitter Jetson boot partition is 0. See Appendix 1. Verified by AM 09OCT24 | PASS |
|  |  |  | Recorded boot partition on cassette before upgrade attempt | Expected outcome verified. Current cassette Jetson boot partition is 0. See Appendix 2. Verified by AM 09OCT24 | PASS |
|  |  |  | Boot partition on emitter remains the same after failed upgrade | Expected outcome verified. Emitter Jetson boot partition remains at 0. Verified by AM 09OCT24 | PASS |
|  |  |  | Boot partition on cassette remains the same after failed upgrade | Expected outcome verified. Cassette Jetson boot partition remains at 0. Verified by AM 09OCT24 | PASS |
|  | Test Case: OTA Updates of MX1 Software System - Operating System Updates |  |  |  |  |
| SRS-45.5 | The SS shall support OTA updates of the emitter and cassette via Operating System Images | 1. SSH into the emitter 2. Enter nvbootctrl get-current-slot 3. Record the reported partition the device is currently utilizing 4. Repeat steps 1 through 3 for the cassette under test 5. Ensure the cassette under test has a network connection 6. Create and initiate deployments in Mender for the test emitter and cassette. 7. When update modals appear on the MedAI Device App, tap “Install” for both emitter and cassette installs 8. Wait for the update to complete. The update progress bars in the MedAI Device App will indicate a completed update. 9. Power cycle the devices under test 10. Repeat steps 1 through 4 11. Verify that both emitter and cassette Jetsons switch to partitions in the case of a successful update | Recorded boot partition on emitter before upgrade attempt | Expected outcome verified. Current emitter Jetson boot partition is 0. See Appendix 1. Verified by AM 09OCT24 | PASS |
|  |  |  | Recorded boot partition on cassette before upgrade attempt | Expected outcome verified. Current cassette Jetson boot partition is 0. See Appendix 2. Verified by AM 09OCT24 | PASS |
|  |  |  | Boot partition on emitter switches after successful upgrade | Expected outcome verified. Emitter boot partition switches to 1. See Appendix 3. Verified by AM 09OCT24 | PASS |
|  |  |  | Boot partition on cassette switches after successful upgrade | Expected outcome verified. Cassette boot partition switches to 1. See Appendix 4. Verified by AM 09OCT24 | PASS |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-574 |  |
