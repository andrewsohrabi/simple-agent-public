# VVPR-P01-254 Rev B: WS-002 Workstation Operational Qualification

## Metadata
- Document ID: VVPR-P01-254
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-254 - WS-002 Workstation Operational Qualification_B-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-254 - WS-002 Workstation Operational Qualification_B-Signed.docx
- Extraction warnings: none

## Extracted Content
PROTOCOL SECTION
PURPOSE
The purpose of this Operational Qualification is to verify and document that the  WS-002, MS-10200 Collimator Verification Workstation, properly verifies the Collimator and consistently produces acceptable units.
SCOPE
This protocol will qualify WS-002, MS-10200 Collimator Verification Workstation, used for testing MS-10200, Collimator - Line Laser ASSY,  as installed at MedAI in a sectioned off area located at 100 Main St. NE Suite 700, Springfield, IL 60001, operates as intended.
REFERENCES
QSP-025 Rev B, Process Validation
MWI-219 Rev F, WS-002 Installation Verification
MWI-220 Rev E, MS-10200 Collimator Verification
BOM-055 Rev K MX1 (Top-level Assembly)
Code of Federal Regulations Quality System Regulations: Process Validation (21CFR820.75)
ISO 13485 Quality Management Systems-Medical Devices-System Requirements for regulatory purposes
QSF-129 Rev B- Equipment OQ Template
SAMPLE SIZE
Operational Qualification for WS-002, MS-10200 Collimator Verification Workstation, shall utilize three Collimator - Line Laser Assemblies for Units Under Test (UUTs) to qualify workstation operation.
Three samples are sufficient to assess process variability and consistency for verification, and demonstrate workstation operation as the programming processes have low variance and are low risk.
IDENTIFICATION
BACKGROUND
WS-002 is used to verify MS-10200, Collimator-Line Laser Assy.
OPERATIONAL QUALIFICATION DOCUMENTATION REQUIREMENTS
List equipment documentation, associated procedures, and document storage location.
TEST EQUIPMENT
MATERIALS
TRAINING
List the name of the OQ executor and the date that training to the OQ protocol was performed (read only training, no trainee verification required). MWI training shall be documented in QSF-076 Production Training Record per QSP-008 Personnel Training.
METHODS
Identify protocol steps in tables below.
Fault Simulation
System Operational Check
SUMMARY/RECOMMENDATIONS/CONCLUSIONS
FOLLOW-UP
OPERATIONAL QUALIFICATION SUMMARY RESULTS
PROTOCOL APPROVAL
Digital Key: example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
There were no deviations from the protocol.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table 1: Equipment Table
RESULTS
Identification
Materials
Training
Fault Simulation
System Operational Check
SUMMARY/RECOMMENDATIONS/CONCLUSIONS
FOLLOWUP
OPERATIONAL QUALIFICATION SUMMARY RESULTS
DISCUSSION
Although the Aperture Size Accuracy test failed for one of the 3 UUTs, WS-002 appropriately identified a Collimator not meeting specification. This Collimator will be put in quarantine until further engineering investigation into the root cause of the failure is performed.
Based on WS-002 data, two Collimator UUTs are acceptable for production as they meet all ODT functional requirements.
The WS-002 Installation Qualification documented in VVPR-P01-240 Rev A was completed with MWI-219 Rev E, WS-002 Workstation Installation. The Workstation Installation MWI is currently at Rev F. However, changes from Rev E to Rev F include adding a step to map network location for 42Q, updating folder location instructions and removing Appendix A. Therefore, the IQ performed in VVPR-P01-240 represents a complete qualification of WS-002 installation at MedAI.
CONCLUSION
WS-002 Passed Operational Qualification. See section 3.  WS-002 has been verified to accurately assess Collimator performance and validated for use during MS-10200 Collimator Line-Laser Assembly Verification.
ATTACHMENTS
VVPR-P01-254 Attachment 1:  MWI-219 Rev F - WS-002 Workstation Installation
VVPR-P01-254 Attachment 2:  MWI-220 Rev E - MS-10200 Collimator-Line Laser Verification
REPORT APPROVAL
Digital Key:
example.com/
QSF-129 (TEMPLATE) REVISION HISTORY
Digital Key:
example.com/

### Table 1
| Equipment Description: | WS-002, MS-10200 Collimator Verification Workstation MWI-219 Rev F- WS-002 Installation Verification |
| --- | --- |
| Equipment Number: | WS-002 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: |  |
| Completed IQ Document Number | VVPR-P01-240 Rev A- WS-002 Workstation Installation Qualification |

### Table 2
| Document Description: | Document Storage Location: |
| --- | --- |
| MWI-219 Rev F - WS-002 Workstation Installation | MedAI QMS |
| MWI-220 Rev E - MS-10200 Collimator-Line Laser Verification | MedAI QMS |

### Table 3
| EQP | EQUIPMENT/ INSTRUMENT/ DEVICE | MFG./MODEL # | RANGE |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |

### Table 4
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-002 | N/A | MS-10200 Collimator Verification Workstation | 1 | MWI-220 Rev E - MS-10200 Collimator-Line Laser Verification |
| Unit Under Test (UUT) |  |  |  |  |
| MS-10200 | E | Collimator - Line Laser ASSY | 1 | Sample #1SN/Lot: ______ |
| MS-10200 | E | Collimator - Line Laser ASSY | 1 | Sample #2 SN/Lot: ______ |
| MS-10200 | E | Collimator - Line Laser ASSY | 1 | Sample #3SN/Lot: ______ |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 5
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
|  | VVPR-P01-254 Rev A- WS-002 Workstation Operational Qualification |  |
|  | MWI-220 Rev E - MS-10200 Collimator-Line Laser Verification |  |

### Table 6
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test  are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-220, but in Procedure Section step 12 leave power supply OFF or turn OFF manually then perform Verification Section up to step 3 | The STATUS of Power Supply on the ODT interface should be DISCONNECTED |  | Pass Pass with Deviation Fail |
| 2 | Using a collimator (MS-10200) perform all Procedure Section steps in MWI-220. Continue with Verification Section but in Verification Section step 6. Do not click the Emitter PWS On/Off toggle button. Then perform Verification Section step 8. | The STATUS for Emitter Connect on the ODT interface should be DISCONNECTED |  | Pass Pass with Deviation Fail |
| 3 | Using a collimator (MS-10200) perform all Procedure Section steps in MWI-220. Continue with the Verification Section but in Verification Section step 7. Do not press the PWR+ button, S1 on the Emitter of T-093. Then perform Verification Section step 9. | The STATUS for Emitter Connect on the ODT interface should be DISCONNECTED |  | Pass Pass with Deviation Fail |
| 4 | Using a collimator (MS-10200), do not connect the flat flex cable connecting to the collimator. Follow instructions in MWI-220 and perform all Procedure steps and up to Verification Section step 8. | The STATUS for Collimator Connect on the ODT interface should be DISCONNECTED |  | Pass Pass with Deviation Fail |

### Table 7
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on, login per test procedure. | PC Starts windows, login successful, desktop is shown. |  | Pass Pass with Deviation Fail |
| 2 | Launch the MX1 MedAI Diagnostic Tool | The test app opens without any errors. |  | Pass Pass with Deviation Fail |
| 3 | Test the 3 sample collimators following the instructions in MWI-220 | Verify the workstation accurately assesses the Collimators. |  | Sample #1 Pass Pass with Deviation Fail |
|  |  |  |  | Sample #2 Pass Pass with Deviation Fail |
|  |  |  |  | Sample #3 Pass Pass with Deviation Fail |
| Overall Results: Pass Pass with Deviation Fail |  |  |  |  |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 8
| Summarize the OQ and document any conclusions and/or recommendations in the Test Report. Summarize risk assessments and any changes or additions needed to the control plan and/or PFMEA. Summary of acceptance or failure of Equipment Qualification. |
| --- |

### Table 9
| Provide a general summary of the equipment function and intended use: Describe the actions to be taken if the test results are not acceptable. This information must be very specific to what actions, why, who, the date of completion and verification by Quality Assurance. |
| --- |

### Table 10
| System Name: |  |
| --- | --- |
| System Version: |  |
| Known Issues (if any): |  |
| Result: | Pass Pass with Limitations Fail |
| Completed By: |  |
| Date Completed: |  |

### Table 11
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Operations | 23 Jan 2025 | 25-045 |

### Table 12
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A |

### Table 13
| Equipment Description: | WS-002, MS-10200 Collimator Verification Workstation MWI-219 Rev F- WS-002 Installation Verification |
| --- | --- |
| Equipment Number: | WS-002 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: | 1247 |
| Completed IQ Document Number | VVPR-P01-240 Rev A- WS-002 Workstation Installation Qualification |

### Table 14
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-002 | N/A | MS-10200 Collimator Verification Workstation | 1 | MWI-220 Rev E - MS-10200 Collimator-Line Laser Verification |
| Unit Under Test (UUT) |  |  |  |  |
| MS-10200 | E | Collimator - Line Laser ASSY | 1 | Sample #1SN/Lot: LOT10273 |
| MS-10200 | E | Collimator - Line Laser ASSY | 1 | Sample #2 SN/Lot: LOT10273 |
| MS-10200 | E | Collimator - Line Laser ASSY | 1 | Sample #3SN/Lot: LOT10273 |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 15
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
| Iris Holt | VVPR-P01-254 Rev A- WS-002 Workstation Operational Qualification | 1/23/2025 |
| Iris Holt | MWI-220 Rev E - MS-10200 Collimator-Line Laser Verification | 1/23/2025 |

### Table 16
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test  are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-220, but in Procedure Section step 12 leave power supply OFF or turn OFF manually then perform Verification Section up to step 3 | The STATUS of Power Supply on the ODT interface should be DISCONNECTED | Power Supply STATUS was DISCONNECTED | Pass Pass with Deviation Fail |
| 2 | Using a collimator (MS-10200) perform all Procedure Section steps in MWI-220. Continue with Verification Section but in Verification Section step 6. Do not click the Emitter PWS On/Off toggle button. Then perform Verification Section step 8. | The STATUS for Emitter Connect on the ODT interface should be DISCONNECTED | The STATUS for Emitter Connect is DISCONNECTED | Pass Pass with Deviation Fail |
| 3 | Using a collimator (MS-10200) perform all Procedure Section steps in MWI-220. Continue with the Verification Section but in Verification Section step 7. Do not press the PWR+ button, S1 on the Emitter of T-093. Then perform Verification Section step 9. | The STATUS for Emitter Connect on the ODT interface should be DISCONNECTED | The STATUS for Emitter Connect is DISCONNECTED | Pass Pass with Deviation Fail |
| 4 | Using a collimator (MS-10200), do not connect the flat flex cable connecting to the collimator. Follow instructions in MWI-220 and perform all Procedure steps and up to Verification Section step 8. | The STATUS for Collimator Connect on the ODT interface should be DISCONNECTED | The STATUS for Collimator Connect is DISCONNECTED | Pass Pass with Deviation Fail |

### Table 17
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on, login per test procedure. | PC Starts windows, login successful, desktop is shown. | PC Starts windows, login successful, desktop is shown. | Pass Pass with Deviation Fail |
| 2 | Launch the MX1 MedAI Diagnostic Tool | The test app opens without any errors. | The test app opens without any errors. | Pass Pass with Deviation Fail |
| 3 | Test the 3 sample collimators following the instructions in MWI-220 | Verify the workstation accurately assesses the Collimators. | Workstation accurately assessed the Collimator. ODT Passed UUT. | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | Workstation accurately assessed the Collimator. ODT Passed UUT. | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | Workstation accurately assessed the Collimator. UUT failed ODT. UUT failure due to the Aperture Size Accuracy Test failing. | Sample #3 Pass Pass with Deviation Fail |
| Overall Results: Pass Pass with Deviation Fail |  |  |  |  |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 18
| Summarize the OQ and document any conclusions and/or recommendations in the Test Report. Summarize risk assessments and any changes or additions needed to the control plan and/or PFMEA. Summary of acceptance or failure of Equipment Qualification. |
| --- |
| WS-002 properly ran through the required ODT Tests and successfully passed two collimators and failed one collimator for aperture size accuracy test. No changes to risk assessment WS-002 has been validated for verification of MS-10200 Collimator-Line Laser Assy |

### Table 19
| Provide a general summary of the equipment function and intended use: Describe the actions to be taken if the test results are not acceptable. This information must be very specific to what actions, why, who, the date of completion and verification by Quality Assurance. |
| --- |
| N/A |

### Table 20
| System Name: | WS-002 MS-10200 Collimator Verification Workstation MWI-219 Rev F - WS-002 Workstation Installation |
| --- | --- |
| System Version: | ODT v3.1.1 Git Hash: dc40c64 |
| Known Issues (if any): | N/A |
| Result: | Pass Pass with Limitations Fail |
| Completed By: | E. Holt |
| Date Completed: | 1/24/2025 |

### Table 21
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release | Engineering Quality Engineering Operations | 25 Jan 2025 | 25-049 |

### Table 22
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Operations | 15 Jan 2025 | 25-024 |
| B | Update Section 14 to move result options to Results row from Known Issues | Engineering Quality Engineering Operations | 22 Jan 2025 | 25-043 |
