# VVPR-P01-255 Rev B: WS-006 Operational Qualification

## Metadata
- Document ID: VVPR-P01-255
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-255- WS-006 Operational Qualification_B-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-255- WS-006 Operational Qualification_B-Signed.docx
- Extraction warnings: none

## Extracted Content
PROTOCOL SECTION
PURPOSE
The purpose of this Operational Qualification is to verify and document the WS-006, Power Cleat and Wireless Charger Verification Workstation, properly verifies the MS-10141 Power Cleat Assy and consistently produces acceptable units.
SCOPE
This protocol will qualify WS-006, Power Cleat and Wireless Charger Verification Workstation, used for testing MS-10141 Power Cleat Assy as installed at MedAI in a sectioned off area located at 100 Main St. NE Suite 700, Springfield, IL 60001, operates as intended.
REFERENCES
QSP-025 Rev B - Process Validation
QSF-129 Rev B - Equipment OQ Template
MWI-227 Rev E - WS-006 Workstation Installation
MWI-228 Rev D - MS-10141 Power Cleat Verification
BOM-055 Rev K - MX1 (Top-level assembly)
Code of Federal Regulations Quality System Regulations: Process Validation (21CFR820.75)
ISO 13485 Quality Management Systems-Medical Devices-System Requirements for regulatory purposes
SAMPLE SIZE
Operational Qualification for WS-006, Power Cleat and Wireless Charger Verification Workstation, shall utilize three Power Cleats for Units Under Tests (UUTs) to qualify workstation operation.
Three samples are sufficient to assess process variability and consistency for each verification, and demonstrate workstation operation as the programming processes have low variance and are low risk.
IDENTIFICATION
BACKGROUND
WS-006 is used for verification of MS-10141, Power Cleat Verification Workstation which verifies MS-10141 Power Cleat and MS-10568 Wireless Charger
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
SUMMARY/RECOMMENDATIONS/CONCLUSION
FOLLOW-UP
OPERATIONAL/QUALIFICATION SUMMARY RESULTS
DISCUSSION
Although the humidity test failed for 2 of the 3 UUTs, the failure was due to the ambient humidity value being less than ODT’s acceptable lower limit, 20%RH, for the measurement. WS-006 Humidity Test intention is to evaluate communication between SHTC3 chip on the PMUX and Emitter Main over differential i2c, not to evaluate ambient humidity. Presence of a humidity value indicates SHTC3 successfully communicated a value to Emitter Main.
Based on WS-006 data, the 3 PMUX UUTs are acceptable for production as they meet all ODT functional requirements. The allowed humidity range should be adjusted to 0% - 100%RH in order to eliminate the assessment of ambient humidity at WS-006.
The WS-006 Installation Qualification documented in VVPR-P01-243 Rev A was completed with MWI-227 Rev D, WS-006 Workstation Installation. The Workstation Installation MWI is currently at Rev E. However, changes from Rev D to Rev E include adding a step to map network location for 42Q, updating folder location instructions and removing Appendix A. Therefore, the IQ performed in VVPR-P01-243 represents a full qualification of WS-006 installation.
CONCLUSION
WS-006 Passed Operational Qualification. See section 3.  WS-006 has been verified to accurately assess PMUX performance and validated for use during MS-10410 Power Cleat Assembly Verification.
ATTACHMENTS
9.1 VVPR-P01-255 Attachment 1: MWI-228 - MS-10141 Power Cleat Verification_D9.2 VVPR-P01-255 Attachment 2: MWI-227 - WS-006 Workstation Installation_E
REPORT APPROVAL
Digital Key: example.com/
QSF-129 (TEMPLATE) REVISION HISTORY
Digital Key:
example.com/

### Table 1
| Equipment Description: | WS-006, Power Cleat and Wireless Charger Verification Workstation |
| --- | --- |
| Equipment Number: | WS-006 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: |  |
| Completed IQ Document Number | VVPR-P01-243 Rev A - WS-006 Workstation Installation Qualification |

### Table 2
| Document Description: | Document Storage Location: |
| --- | --- |
| MWI-227 Rev E - WS-006 Workstation Installation | MedAI QMS |

### Table 3
| EQP | EQUIPMENT/ INSTRUMENT/ DEVICE | MFG./MODEL # | RANGE |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |

### Table 4
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-006 | N/A | Power Cleat Verification & Wireless Charger Verification Workstation | 1 | MWI-227 Rev E - WS-006 Workstation Installation |
| Unit Under Test (UUT) |  |  |  |  |
| MS-10141 | E | Power Cleat | 1 | Sample #1SN/Lot: |
| MS-10141 | E | Power Cleat | 1 | Sample #2 SN/Lot: |
| MS-10141 | E | Power Cleat | 1 | Sample #3SN/Lot: |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 5
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
|  | VVPR-P01-255 Rev A- WS-006 Operational Qualification |  |
|  | MWI-228 Rev D - MS-10141 Power Cleat Verification |  |

### Table 6
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test  are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-228, but in Procedure Section step 22, change the PMUX fixture hostname to something else. Then, perform Procedure Section step 24. | The STATUS of PMUX on the ODT interface should not be CONNECTED |  | Pass Pass with Deviation Fail |
| 2 | In Procedure Section step 16, do not plug in the wall wart of the AC/DC Wall Mount Adapter connected to the Raspberry Pi back into the AC outlet, then perform Procedure Section step 24. | The STATUS of PMUX on the ODT interface should not be CONNECTED |  | Pass Pass with Deviation Fail |
| 3 | Follow instructions in MWI-228, but in Procedure Section step 8, leave the Power Supply OFF. Then perform Procedure Section step 28. | The STATUS for Power Supply on the ODT interface should be DISCONNECTED |  | Pass Pass with Deviation Fail |

### Table 7
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on, login per test procedure. | PC Starts windows, login successful, desktop is shown. |  | Pass Pass with Deviation Fail |
| 2 | Launch the MX1 MedAI Diagnostic Tool | The test app opens without any errors. |  | Pass Pass with Deviation Fail |
| 3 | Test the 3 sample Power Cleat Assemblies following the instructions in MWI-228 | Verify the workstation accurately assesses the PMUX. |  | Sample #1 Pass Pass with Deviation Fail |
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
| A | Initial Release | Engineering Quality Engineering Operations | 21 Jan 2025 | 25-044 |

### Table 12
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |

### Table 13
| Equipment Description: | WS-006, Power Cleat and Wireless Charger Verification Workstation |
| --- | --- |
| Equipment Number: | WS-006 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: | 1255 |
| Completed IQ Document Number | VVPR-P01-243 Rev A - WS-006 Workstation Installation Qualification |

### Table 14
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-006 | N/A | Power Cleat Verification & Wireless Charger Verification Workstation | 1 | MWI-227 Rev E - WS-006 Workstation Installation |
| Unit Under Test (UUT) |  |  |  |  |
| MS-10141 | E | Power Cleat | 1 | Sample #1SN/Lot: LOT 10272 |
| MS-10141 | E | Power Cleat | 1 | Sample #2 SN/Lot: LOT 10272 |
| MS-10141 | E | Power Cleat | 1 | Sample #3SN/Lot: LOT 10272 |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 15
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
| E. Holt | VVPR-P01-255 Rev A- WS-006 Operational Qualification | 1/22/2025 |
| E. Holt | MWI-228 Rev D - MS-10141 Power Cleat Verification | 1/22/2025 |

### Table 16
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test  are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-228, but in Procedure Section step 22, change the PMUX fixture hostname to something else. Then, perform Procedure Section step 24. | The STATUS of PMUX on the ODT interface should not be CONNECTED | PMUX STATUS was not CONNECTED | Pass Pass with Deviation Fail |
| 2 | In Procedure Section step 16, do not plug in the wall wart of the AC/DC Wall Mount Adapter connected to the Raspberry Pi back into the AC outlet, then perform Procedure Section step 24. | The STATUS of PMUX on the ODT interface should not be CONNECTED | PMUX STATUS was not CONNECTED | Pass Pass with Deviation Fail |
| 3 | Follow instructions in MWI-228, but in Procedure Section step 8, leave the Power Supply OFF. Then perform Procedure Section step 28. | The STATUS for Power Supply on the ODT interface should be DISCONNECTED | PowerSupply STATUS was  DISCONNECTED | Pass Pass with Deviation Fail |

### Table 17
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on, login per test procedure. | PC Starts windows, login successful, desktop is shown. | PC Starts windows, login is successful, desktop is shown. | Pass Pass with Deviation Fail |
| 2 | Launch the MX1 MedAI Diagnostic Tool | The test app opens without any errors. | The test app opens without any errors. | Pass Pass with Deviation Fail |
| 3 | Test the 3 sample Power Cleat Assemblies following the instructions in MWI-228 | Verify the workstation accurately assesses the PMUX. | ODT appropriately Failed PMUX HUMIDITY TEST since UUT reported a humidity value of 17%. (< 20% acceptable minimum on ODT) | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | ODT Passed UUT. | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | UUT appropriately  Failed PMUX HUMIDITY TEST since UUT reported a humidity value of 19%. (< 20% acceptable minimum on ODT) | Sample #3 Pass Pass with Deviation Fail |
| Overall Results: Pass Pass with Deviation Fail |  |  |  |  |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 18
| Summarize the OQ and document any conclusions and/or recommendations in the Test Report. Summarize risk assessments and any changes or additions needed to the control plan and/or PFMEA. Summary of acceptance or failure of Equipment Qualification. |
| --- |
| WS-006 Humidity Test intention is to evaluate communication between SHTC3 chip on the PMUX and Emitter Main over differential i2c. Presence of a humidity value indicates SHTC3 communicated a value to Emitter Main. Recommend changing the humidity test for the workstation to increase the passing humidity range from 20% and 60% to 0 to 100% in order to verify communication rather than ambient humidity value. |

### Table 19
| Provide a general summary of the equipment function and intended use: Describe the actions to be taken if the test results are not acceptable. This information must be very specific to what actions, why, who, the date of completion and verification by Quality Assurance. |
| --- |
| Modify ODT v3.1.1 to adjust passing humidity range values in order to verify i2c communication and not the ambient humidity value. Electrical Engineering plans to complete ODT updates and code review for this change by 2/2/2025. |

### Table 20
| System Name: | WS-006 Power Cleat and Wireless Charger VerificationMWI-227 - WS-006 Workstation Installation (IQ) Rev E |
| --- | --- |
| System Version: | ODT v3.1.1 GITHASH: dc40c64 |
| Known Issues (if any): | N/A |
| Result: | Pass Pass with Limitations Fail |
| Completed By: | E. Holt |
| Date Completed: | 1/24/2025 |

### Table 21
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release | Engineering Quality Engineering Operations | 25 Jan 2025 | 25-048 |

### Table 22
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Operations |  |  |
| B | Update Section 14 to move result options to Results row from Known Issues | Engineering Quality Engineering Operations |  | 25-043 |
