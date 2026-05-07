# VVPR-P01-267 Rev B: WS-010 Workstation Operational Qualification

## Metadata
- Document ID: VVPR-P01-267
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-267 - WS-010 Workstation Operational Qualification_B-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-267 - WS-010 Workstation Operational Qualification_B-Signed.docx
- Extraction warnings: none

## Extracted Content
PROTOCOL SECTION
PURPOSE
The purpose of this Operational Qualification is to verify and document the WS-010, Battery Pack Verification Workstation properly verifies the Cassette Battery Pack (MS-10083) and Emitter Battery Pack (MS-10010), and consistently produces acceptable units.
SCOPE
This protocol will qualify WS-010, Battery Pack Verification Workstation, used for testing MS-10083, Cassette Battery Pack and MS-10010, Emitter Battery Pack as installed at MedAI in a sectioned off area located at 100 Main St. NE Suite 700, Springfield, IL 60001, operates as intended.
REFERENCES
QSP-025 Rev B - Process Validation
QSF-129 Rev B - Equipment OQ Template
MWI-210 Rev D- WS-010 Workstation Installation (IQ)
MWI-214 Rev D- MS-10083 Cassette Battery Pack Verification
MWI-211 Rev D - MS-10010 Emitter Battery Pack Verification
MS-10010 Rev I - Emitter Battery Pack
MS-10083 Rev J - Cassette Battery Pack
Code of Federal Regulations Quality System Regulations: Process Validation (21CFR820.75)
ISO 13485 Quality Management Systems-Medical Devices-System Requirements for regulatory purposes
SAMPLE SIZE
Operational Qualification for WS-010, Battery Pack Verification Workstation, shall utilize three Cassette Battery Packs and three Emitter Battery packs for Units Under Tests (UUTs) to qualify workstation operation.
Three samples are sufficient to assess process variability and consistency for each verification, and demonstrate workstation operation as the programming processes have low variance and are low risk.
IDENTIFICATION
BACKGROUND
WS-010 is used to verify MS-10083, Cassette Battery Pack, and MS-10010, Emitter Battery Pack.
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
Digital Key:example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
All protocol steps in MWI-214 were followed for the first 2 cassette battery packs. Step 19 in MWI-214 and Step 20 in MWI-211 reset the cycle count for the battery packs. These Steps would be performed on batteries prior to cycling. However, all batteries used for this qualification had already been cycled. Therefore, these steps were omitted for the remainder of testing.
An additional step was added to the protocol to verify the battery backs were adequately cycled on equipment at MedAI. The acceptance criteria was cycle count ≥ 2. After confirming the battery pack readback results, “Monitor” was selected and the cycle count was recorded. This was performed for each UUT.
The second deviation shall be performed Cassette Battery packs to capture the number of cycles at a later date.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table 1: Equipment Table
RESULTS
Identification
Materials
Training
Fault Simulation
System Operational Check
SUMMARY/RECOMMENDATIONS/CONCLUSIONS
FOLLOW-UP
OPERATIONAL QUALIFICATION SUMMARY RESULTS
DISCUSSION
WS-010 is used for testing Emitter and Cassette battery pack verifications. The workstation functioned as intended and successfully verified the Emitter and Cassette Battery Packs.
In addition, WS-010 was used to verify the Emitter battery packs were successfully cycled at MedAI, showing cycle counts ≥ 2 for the 3 packs.
The Installation and Operational Qualifications used a workstation with the same MWI-210 revision. See Attachment 1.
During WS-010 verification, Step 19 in MWI-214 Rev D, specifically pressing the Full Reset button, was performed on the first two Cassette Battery Packs. This resulted in the battery pack cycles resetting to zero. Therefore, the packs were no longer reset as part of the operational qualification. This does not impact the operational qualification of the workstation as the batteries passed all testing.
CONCLUSION
WS-010 passed Operational Qualification. See section 3. WS-010 has been verified to accurately assess Battery Pack performance and validated for use for MS-10010 Emitter Battery Pack verification and MS-10083 Cassette Battery Pack verification.
ATTACHMENTS
VVPR-P01-267 Attachment 1: MWI-210 Rev D- WS-010 Workstation Installation (IQ)
VVPR-P01-267 Attachment 2: MWI-214 Rev D- MS-10083 Cassette Battery Pack Verification
VVPR-P01-267 Attachment 3: MWI-211 Rev D- MS-10010 Emitter Battery Pack Verification
REPORT APPROVAL
Digital Key: example.com/
QSF-129 (TEMPLATE) REVISION HISTORY
Digital Key:example.com/

### Table 1
| Equipment Description: | WS-010, Battery Pack Verification Workstation |
| --- | --- |
| Equipment Number: | WS-010 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: |  |
| Completed IQ Document Number | VVPR-P01-265 Rev A- WS-010 Workstation Installation Qualification |

### Table 2
| Document Description: | Document Storage Location: |
| --- | --- |
| MWI-210 Rev D- WS-010 Workstation Installation (IQ) | MedAI QMS |
| MWI-214 Rev D- MS-10083 Cassette Battery Pack Verification | MedAI QMS |
| MWI-211 Rev D- MS-10010 Emitter Battery Pack Verification | MedAI QMS |

### Table 3
| EQP | EQUIPMENT/ INSTRUMENT/ DEVICE | MFG./MODEL # | RANGE |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |

### Table 4
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-010 | N/A | Battery Pack Verification Workstation. | 1 | MWI-210 Rev D - WS-010 Workstation Installation (IQ) |
| Unit Under Test (UUT) |  |  |  |  |
| MS-10083 | J | Cassette Battery Pack | 1 | Sample #1SN/Lot: 10295 |
| MS-10083 | J | Cassette Battery Pack | 1 | Sample #2 SN/Lot: 10295 |
| MS-10083 | J | Cassette Battery Pack | 1 | Sample #3SN/Lot: 10295 |
| MS-10010 | I | Emitter Battery Pack | 1 | Sample #1SN/Lot: |
| MS-10010 | I | Emitter Battery Pack | 1 | Sample #2 SN/Lot: |
| MS-10010 | I | Emitter Battery Pack | 1 | Sample #3SN/Lot: |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 5
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
|  | VVPR-P01-267 Rev A- WS-010 Workstation Operational Qualification |  |
|  | MWI-214 Rev D- MS-10083 Cassette Battery Pack Verification |  |
|  | MWI-211 Rev D- MS-10010 Emitter Battery Pack Verification |  |

### Table 6
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-214, but in procedure step 4, don’t choose a COM port for the MX1 BMS Interface. Then perform Step 15. | Popup will read “Serial Port Not Open” |  | Pass Pass with Deviation Fail |
| 2 | Follow instructions in MWI-214, but in procedure step 3, don’t connect the Cassette Battery Pack to the MX1 BMS Interface and continue through step 15. | The message next to “Device check” will read “BQ76952 FAIL,MAX17205 FAIL,” |  | Pass Pass with Deviation Fail |
| 3 | Follow instructions in MWI-214, but in procedure step 8, click the CassetteBQ file as the MAX file and in step 12, click the CassetteMAX file as the BQ file. Then continue through step 18. | Popup will read “File mismatch” |  | Pass Pass with Deviation Fail |
| 4 | Follow instructions in MWI-214, but skip procedure section steps 5 through 13. Then continue through to step 18. | Popup will read “Choose a valid file path” |  | Pass Pass with Deviation Fail |

### Table 7
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on | PC Starts windows, login successful, desktop is shown. |  | Pass Pass with Deviation Fail |
| 2 | Test and shutdown the 3 sample Cassette Battery Packs per MWI-214 | Verify the workstation accurately assesses and shuts down the Cassette Battery Packs |  | Sample #1 Pass Pass with Deviation Fail |
|  |  |  |  | Sample #2 Pass Pass with Deviation Fail |
|  |  |  |  | Sample #3 Pass Pass with Deviation Fail |
| 3 | Test and shutdown the 3 sample Emitter Battery Packs per MWI-211 | Verify the workstation accurately assesses and shuts down the Emitter Battery Packs |  | Sample #1 Pass Pass with Deviation Fail |
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
| A | Initial Release | Quality Engineering Engineering Operations |  | 25-047 |

### Table 12
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| N/A | N/A | N/A | N/A | N/A |

### Table 13
| Equipment Description: | WS-010, Battery Pack Verification Workstation MWI-210 Rev D - WS-010 Workstation Installation (IQ) |
| --- | --- |
| Equipment Number: | WS-010 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: | 1319 |
| Completed IQ Document Number | VVPR-P01-265 Rev A- WS-010 Workstation Installation Qualification |

### Table 14
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-010 | N/A | Battery Pack Verification Workstation. | 1 | MWI-210 Rev D - WS-010 Workstation Installation (IQ) |
| Unit Under Test (UUT) |  |  |  |  |
| MS-10083 | J | Cassette Battery Pack | 1 | Sample #1SN/Lot: 10295 |
| MS-10083 | J | Cassette Battery Pack | 1 | Sample #2 SN/Lot: 10295 |
| MS-10083 | J | Cassette Battery Pack | 1 | Sample #3SN/Lot: 10295 |
| MS-10010 | I | Emitter Battery Pack | 1 | Sample #1SN/Lot: 10296 |
| MS-10010 | I | Emitter Battery Pack | 1 | Sample #2 SN/Lot: 10296 |
| MS-10010 | I | Emitter Battery Pack | 1 | Sample #3SN/Lot: 10296 |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 15
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
| Matt Beckham | VVPR-P01-267 Rev A- WS-010 Workstation Operational Qualification | 1/23/2025 |
| Matt Beckham | MWI-214 Rev D- MS-10083 Cassette Battery Pack Verification | 12/4/2024 |
| Matt Beckham | MWI-211 Rev D- MS-10010 Emitter Battery Pack Verification | 12/4/2024 |

### Table 16
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-214, but in procedure step 4, don’t choose a COM port for the MX1 BMS Interface. Then perform Step 15. | Popup will read “Serial Port Not Open” | Popup read “Serial Port Not Open” | Pass Pass with Deviation Fail |
| 2 | Follow instructions in MWI-214, but in procedure step 3, don’t connect the Cassette Battery Pack to the MX1 BMS Interface and continue through step 15. | The message next to “Device check” will read “BQ76952 FAIL,MAX17205 FAIL,” | The message next to “Device check” read “BQ76952 FAIL,MAX17205 FAIL,” | Pass Pass with Deviation Fail |
| 3 | Follow instructions in MWI-214, but in procedure step 8, click the CassetteBQ file as the MAX file and in step 12, click the CassetteMAX file as the BQ file. Then continue through step 18. | Popup will read “File mismatch” | Popup read “File mismatch” | Pass Pass with Deviation Fail |
| 4 | Follow instructions in MWI-214, but skip procedure section steps 5 through 13. Then continue through to step 18. | Popup will read “Choose a valid file path” | Popup read “Choose a valid file path” | Pass Pass with Deviation Fail |

### Table 17
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on | PC Starts windows, login successful, desktop is shown. | PC Starts windows, login successful, desktop is shown. | Pass Pass with Deviation Fail |
| 2 | Test and shutdown the 3 sample Cassette Battery Packs per MWI-214 | Verify the workstation accurately assesses and shuts down the Cassette Battery Packs | UUT passed and shut down battery pack | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | UUT passed and shut down battery pack | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | UUT passed and shut down battery pack | Sample #3 Pass Pass with Deviation Fail |
| 3 | Test and shutdown the 3 sample Emitter Battery Packs per MWI-211 | Verify the workstation accurately assesses and shuts down the Emitter Battery Packs | UUT passed and shut down battery packCycle Count: 3.20 | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | UUT passed and shut down battery packCycle Count: 3.20 | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | UUT passed and shut down battery packCycle Count: 2.88 | Sample #3 Pass Pass with Deviation Fail |
| Overall Results: Pass Pass with Deviation Fail |  |  |  |  |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 18
| Summarize the OQ and document any conclusions and/or recommendations in the Test Report. Summarize risk assessments and any changes or additions needed to the control plan and/or PFMEA. Summary of acceptance or failure of Equipment Qualification. |
| --- |
| WS-010 properly ran through the Battery Verification workstation and accurately verified 3 Cassette Battery Packs and 3 Emitter Battery Packs. No changes to risk assessment needed WS-010 has been validated to verify the Cassette and Emitter battery packs. |

### Table 19
| Provide a general summary of the equipment function and intended use: Describe the actions to be taken if the test results are not acceptable. This information must be very specific to what actions, why, who, the date of completion and verification by Quality Assurance. |
| --- |
| N/A |

### Table 20
| System Name: | WS-010 Battery Pack Verification Workstation MWI-210 Rev D- WS-010 Workstation Installation (IQ) |
| --- | --- |
| System Version: | MX1 BMS Programming Utility v1.1.1 |
| Known Issues (if any): | N/A |
| Result: | Pass Pass with Limitations Fail |
| Completed By: | M. Beckham |
| Date Completed: | 24 Jan 2025 |

### Table 21
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release | Engineering Quality Engineering Operations | 25 Jan 2025 | 25-050 |

### Table 22
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Operations | 15 Jan 2025 | 25-024 |
| B | Update Section 14 to move result options to Results row from Known Issues | Engineering Quality Engineering Operations | 22 Jan 2025 | 25-043 |
