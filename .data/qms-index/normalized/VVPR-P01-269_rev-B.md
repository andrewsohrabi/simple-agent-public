# VVPR-P01-269 Rev B: WS-013 Operational Qualification Report

## Metadata
- Document ID: VVPR-P01-269
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-269 - WS-013 Operational Qualification Report_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-269 - WS-013 Operational Qualification Report_B-signed.docx
- Extraction warnings: none

## Extracted Content
PROTOCOL SECTION
PURPOSE
The purpose of this Operational Qualification is to verify and document that WS-013, Foot Pedal Workstation, properly verifies the F1 Foot Pedal, obtains the unique pairing ID of ES-10007 Foot Pedal PCBA, and consistently produces acceptable units.
SCOPE
This protocol will qualify WS-013, Foot Pedal Workstation, used for testing F1 Foot Pedal, as installed at MedAI in a sectioned off area located at 100 Main St. NE Suite 700, Springfield, IL 60001, operates as intended.
REFERENCES
QSP-025 Rev B - Process Validation
QSF-129 Rev B - Equipment OQ Template
MWI-186 Rev C - F1 Foot Pedal
MWI-331 Rev A - WS-013 Workstation Installation
MWI-330 Rev A - F1 Foot Pedal Verification
MWI-329 Rev A - ES-10007 Foot Pedal UUID Reading
BOM-021 Rev C - Foot Pedal
Code of Federal Regulations Quality System Regulations: Process Validation (21CFR820.75)
ISO 13485 Quality Management Systems-Medical Devices-System Requirements for regulatory purposes
SAMPLE SIZE
Operational Qualification for WS-013, Foot Pedal Workstation, shall utilize foot pedals for Units Under Tests (UUTs) to qualify workstation operation.
Three samples are sufficient to assess process variability and consistency for each verification, and demonstrate workstation operation as the programming processes have low variance and are low risk.
IDENTIFICATION
BACKGROUND
WS-013, Foot Pedal Workstation is used for verification of F1 Foot Pedal and obtains the unique pairing ID of ES-10007 Foot Pedal PCBA.
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
F1 Foot Pedal Rev D assemblies were not available for the execution of this protocol.  Instead, a single MS-50006 Rev B was used with the three ES-10007 Rev B PCBAs.
Additional fault simulation testing was performed in Section 11.1 across all three permutations to simulate two fault scenarios and ensure the test equipment would catch these failures.
Swap two of the connectors and confirm that the terminal print-outs and audible tones indicate that such a swap occurred.
Disconnect one of the connectors and confirm that no print-out or tone occurs when the disconnected foot pedal is operated.
DEVICES, COMPONENTS, OR EQUIPMENT USED
IDENTIFICATION
BACKGROUND
WS-013, Foot Pedal Workstation is used to obtain the unique pairing ID of ES-10007 Foot Pedal PCBA and verify F1 Foot Pedal operation.
Prior to running UUTs through WS-013, ES-10007 Rev B was flashed at WS-008 with firmware v4.1.0.
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
DISCUSSION
Slight modifications to MWI-253 - ES-10007 Footpedal PCBA Programming  were noted and resulted in ECR-717 releasing MWI-253 Rev C.
CONCLUSION
WS-013 passed Operational Qualification. See Section 7.  WS-013 has been verified to obtain the Foot Pedal PCBA unique identifier and accurately assess F1 Foot Pedal functionality, and validated for use during F1 Assembly and Verification.
REPORT APPROVAL
Digital Key: example.com/

### Table 1
| Equipment Description: | WS-013, Foot Pedal Workstation |
| --- | --- |
| Equipment Number: | WS-013 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: |  |
| Completed IQ Document Number | VVPR-P01-268 Rev A - WS-013 Installation Qualification |

### Table 2
| Document Description: | Document Storage Location: |
| --- | --- |
| MWI-331 Rev A - WS-013 Workstation Installation | MedAI QMS |
| MWI-330 Rev A - F1 Foot Pedal Verification | MedAI QMS |
| MWI-329 Rev A - ES-10007 Foot Pedal UUID Reading | MedAI QMS |

### Table 3
| EQP | EQUIPMENT/ INSTRUMENT/ DEVICE | MFG./MODEL # | RANGE |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |

### Table 4
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-013 | N/A | Foot Pedal Workstation | 1 | MWI-331 Rev A - WS-013 Workstation Installation |
| Unit Under Test (UUT) |  |  |  |  |
| F1 | D | Foot Pedal |  | Sample #1SN/Lot: ______ |
| F1 | D | Foot Pedal |  | Sample #2 SN/Lot: ______ |
| F1 | D | Foot Pedal |  | Sample #3SN/Lot: ______ |
| ES-10007 | 3 | Foot Pedal PCBA |  | Sample #1SN/Lot: ______ |
| ES-10007 | 3 | Foot Pedal PCBA |  | Sample #2 SN/Lot: ______ |
| ES-10007 | 3 | Foot Pedal PCBA |  | Sample #3SN/Lot: ______ |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 5
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
|  | VVPR-P01-269 Rev A- WS-013 Operational Qualification |  |
|  | MWI-330 Rev A - F1 Foot Pedal Verification |  |
|  | MWI-329 Rev A - ES-10007 Foot Pedal UUID Reading |  |

### Table 6
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test  are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow the instructions in MWI-329 but at step 3, do not place ES-10007 footpedal PCBA onto T-200. Then continue through to step 6. | LCD Display reads “Place UUT” |  | Pass Pass with Deviation Fail |
| 2 | Follow the instructions in MWI-330 but at step 10, do not type in the correct Foot Pedal UUT. Continue through step 14. | Foot pedal does not beep |  | Pass Pass with Deviation Fail |
| 3 | In the first step of MWI-186, swap the connector housings for S1 and S2, left and right foot pedals, respectively.  Complete the remainder of the MWI.  Follow the instructions in MWI-330.  Confirm that steps 17 and 18 are reversed | Pressing the right foot pedal registers both in the terminal and audibly as a left foot pedal press.  Similarly, pressing the left foot pedal registers both in the terminal and audibly as a right foot pedal press. |  | Pass Pass with Deviation Fail |
| 4 | In the first step of MWI-186, do not connect the connector housing for S2 to the PCBA.  Complete the remainder of the MWI.  Follow the instructions in MWI-330.  Confirm that step 18 fails to print any message in the terminal or beep | Pressing the right foot pedal will not register in the terminal or audibly |  | Pass Pass with Deviation Fail |

### Table 7
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on, login per test procedure. | PC Starts windows, login successful, desktop is shown. |  | Pass Pass with Deviation Fail |
| 2 | Test the 3 sample Foot Pedal PCB boards following the instructions in MWI-329 | 7-8 digit integer appears on LCD display on T-200 |  | Sample #1 Pass Pass with Deviation Fail |
|  |  |  |  | Sample #2 Pass Pass with Deviation Fail |
|  |  |  |  | Sample #3 Pass Pass with Deviation Fail |
| 3 | Test the 3 sample Foot Pedals following the instructions in MWI-330 | Verify the workstation accurately assesses the Foot Pedals. |  | Sample #1 Pass Pass with Deviation Fail |
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
| A | Initial release | Quality Engineering Engineering Operations | 14Feb2025 | 25-094 |

### Table 12
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-013 | N/A | Foot Pedal Workstation | 1 | MWI-331 Rev A - WS-013 Workstation Installation |
| Unit Under Test (UUT) |  |  |  |  |
| MS-50006 | B | Wireless Footpedal GP211 (without PCBA) | 1 | SN:  00019 |
| ES-10007 | B | Foot Pedal PCBA | 1 | Sample #1SN/Lot: PO23454-0005 |
| ES-10007 | B | Foot Pedal PCBA | 1 | Sample #2 SN/Lot: PO23454-0008 |
| ES-10007 | B | Foot Pedal PCBA | 1 | Sample #3SN/Lot: PO23454-0013 |

### Table 13
| Equipment Description: | WS-013, Foot Pedal Workstation |
| --- | --- |
| Equipment Number: | WS-013 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: | N/A |
| Completed IQ Document Number | VVPR-P01-268 Rev A - WS-013 Installation Qualification |

### Table 14
| Document Description: | Document Storage Location: |
| --- | --- |
| MWI-331 Rev A - WS-013 Workstation Installation | MedAI QMS |
| MWI-330 Rev A - F1 Foot Pedal Verification | MedAI QMS |
| MWI-329 Rev A - ES-10007 Foot Pedal UUID Reading | MedAI QMS |

### Table 15
| EQP | EQUIPMENT/ INSTRUMENT/ DEVICE | MFG./MODEL # | RANGE |
| --- | --- | --- | --- |
| N/A | N/A | N/A | N/A |

### Table 16
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-013 | N/A | Foot Pedal Workstation | 1 | MWI-331 Rev A - WS-013 Workstation Installation |
| Unit Under Test (UUT) |  |  |  |  |
| F1 | D | Foot Pedal |  | Sample #1SN/Lot: N/A |
| F1 | D | Foot Pedal |  | Sample #2 SN/Lot: N/A |
| F1 | D | Foot Pedal |  | Sample #3SN/Lot: N/A |
| MS-50006 | B | Wireless Footpedal GP211 (without PCBA) | 1 | SN:  00019 |
| ES-10007 | B | Foot Pedal PCBA | 1 | Sample #1SN/Lot: PO23454-0005 |
| ES-10007 | B | Foot Pedal PCBA | 1 | Sample #2 SN/Lot: PO23454-0008 |
| ES-10007 | B | Foot Pedal PCBA | 1 | Sample #3SN/Lot: PO23454-0013 |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 17
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
| Phoenix Mason | VVPR-P01-269 Rev A- WS-013 Operational Qualification | 2/14/2025 |
| Phoenix Mason | MWI-330 Rev A - F1 Foot Pedal Verification | 2/14/2025 |
| Phoenix Mason | MWI-329 Rev A - ES-10007 Foot Pedal UUID Reading | 2/14/2025 |

### Table 18
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test  are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow the instructions in MWI-329 but at step 3, do not place ES-10007 footpedal PCBA onto T-200. Then continue through to step 6. | LCD Display reads “Place UUT” | LCD reads “Place UUT” | Pass Pass with Deviation Fail |
| 2 | Follow the instructions in MWI-330 but at step 10, do not type in the correct Foot Pedal UUT. Continue through step 14. | Foot pedal does not beep | No beep heard | Pass Pass with Deviation Fail |
| 3 | In the first step of MWI-186, swap the connector housings for S1 and S2, left and right foot pedals, respectively.  Complete the remainder of the MWI.  Follow the instructions in MWI-330.  Confirm that steps 17 and 18 are reversed | Pressing the right foot pedal registers both in the terminal and audibly as a left foot pedal press.  Similarly, pressing the left foot pedal registers both in the terminal and audibly as a right foot pedal press. | Physically operating the right foot pedal registered in terminal and audibly as left foot pedal.  Physically operating the left foot pedal registered in terminal and audibly as right foot pedal.  Left and right button behavior unaffected | Pass Pass with Deviation Fail |
| 4 | In the first step of MWI-186, do not connect the connector housing for S2 to the PCBA.  Complete the remainder of the MWI.  Follow the instructions in MWI-330.  Confirm that step 18 fails to print any message in the terminal or beep | Pressing the right foot pedal will not register in the terminal or audibly | No press or release captured on right foot pedal, all other pedals and buttons operational without issue | Pass Pass with Deviation Fail |

### Table 19
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the computer on, login per test procedure. | PC Starts windows, login successful, desktop is shown. | Login successful | Pass Pass with Deviation Fail |
| 2 | Test the 3 sample Foot Pedal PCB boards following the instructions in MWI-329 | 7-8 digit integer appears on LCD display on T-200 | Sample #1:  4718673 | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | Sample #2:  4784187 | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | Sample #3:  4784189 | Sample #3 Pass Pass with Deviation Fail |
| 3 | Test the 3 sample Foot Pedals following the instructions in MWI-330 | Verify the workstation accurately assesses the Foot Pedals. | All button presses and releases registered in terminal and audibly | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | All button presses and releases registered in terminal and audibly | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | All button presses and releases registered in terminal and audibly | Sample #3 Pass Pass with Deviation Fail |
| Overall Results: Pass Pass with Deviation Fail |  |  |  |  |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 20
| Summarize the OQ and document any conclusions and/or recommendations in the Test Report. Summarize risk assessments and any changes or additions needed to the control plan and/or PFMEA. Summary of acceptance or failure of Equipment Qualification. |
| --- |
| WS-013 works as intended; interfacing with Foot Pedal PCBA to get the UUID of the MCU and registering the 4 different button presses on an assembled unit.  The workstation will be used at two different stages of the assembly process: During PCBA intake for an assembly to get the UID of the MCU During testing of a completed assembly |

### Table 21
| Provide a general summary of the equipment function and intended use: Describe the actions to be taken if the test results are not acceptable. This information must be very specific to what actions, why, who, the date of completion and verification by Quality Assurance. |
| --- |
| N/A |

### Table 22
| System Name: | WS-013 |
| --- | --- |
| System Version: | Installed per MWI-331 Rev A |
| Known Issues (if any): | N/A |
| Result: | Pass Pass with Limitations Fail |
| Completed By: | Phoenix Mason |
| Date Completed: | 2/17/25 |

### Table 23
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Report Release | Quality Engineering Engineering Operations | 28 Feb 2025 | 25-110 |
