# VVPR-P01-253 Rev B: WS-008 Workstation PMUX PCBA Programming Operational Qualification

## Metadata
- Document ID: VVPR-P01-253
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-253 - WS-008 Workstation PMUX PCBA Programming Operational Qualification_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-253 - WS-008 Workstation PMUX PCBA Programming Operational Qualification_B-signed.docx
- Extraction warnings: none

## Extracted Content
PROTOCOL SECTION
PURPOSE
The purpose of this Operational Qualification is to verify and document that the WS-008, Programming Workstation, properly configures the PMUX USB-PD port when following MWI-252 Rev C instructions for programming the PMUX USB-PD consistently produces acceptable units.
SCOPE
This protocol qualifies WS-008, Programming Workstation, when used for programming the USB-PD on ES-10015 PMUX PCBA, as installed at MedAI in a sectioned off area located at 100 Main St. NE Suite 700, Springfield, IL 60001.
This protocol will validate the USB-PD flashing process for the PMUX, Emitter Main and WTX.
This operational qualification does not adversely impact other operational qualifications already performed on WS-008.
REFERENCES
QSP-025 Rev B, Process Validation
MWI-231 Rev G- WS-008 Workstation Installation
MWI-252  Rev C- ES-10015 PMUX PCBA Programming
Code of Federal Regulations Quality System Regulations: Process Validation (21CFR820.75)
ISO 13485 Quality Management Systems-Medical Devices-System Requirements for regulatory purposes
SAMPLE SIZE
Operational Qualification for WS-008, Programming Workstation, shall utilize three PMUX PCBAs for Units Under Test (UUTs) to qualify workstation operation.
Three samples are sufficient to assess process variability and consistency for each verification, and demonstrate workstation operation as the programming processes have low variance and are low risk.
IDENTIFICATION
BACKGROUND
WS-008 is used to program custom MedAI software and firmware on off the shelf components and custom PCBAs. The WS-008 at MedAI has currently been validated to program ES-10004 Cassette Main PCBA’s (including flashing main MCU f/w and USB PD chips) and M50004 Detector.
A uniform flashing process at WS-008 is used to flash board firmware on the Cassette Main, the Collimator (ES-10008), the Emitter Main (ES-10003), Monoblock LV (ES-10019) and Foot Pedal (ES-10007) PCBAs. Therefore, the WS-008 Cassette Main qualification also qualifies the flashing process of these other PCBAs.
Two USB-PD flashing processes exist at WS-008; both processes use the same application to flash an off-the-shelf binary onto the USB-PD chips. The USB-PD chips on the Emitter Main, PMUX, and WTX are flashed by creating a new project in the application and selecting a .bin file to flash. The USB-PD chips on the Cassette isolated and non-isolated ports use a pre-created project which contains the .bin file within them.
Although WS-008 IQ was completed using MWI-231 Rev E workstation configuration, the changes between Rev E and Rev G corrected a clerical error with the MX1 Software System BOM, updated the MX1 SS BOM to latest software revision and removed a power cable from the BOM that was part of a tool. These modifications do not impact the Installation Qualification.
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
There was one deviation to the protocol. After completing the system operational check, the PMUX was connected to an H1 and the voltage across C9 was verified to be within 5% of 20V. This assessment was performed on each of the three PMUXs.
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
WS-008 is used to flash a .bin file on the PMUX PCBA (ES-10015) for its USB ports to operate as intended. The workstation functioned as intended and successfully loaded the file onto the boards. To verify WS-008 properly flashed the PMUX, an H1 charger was plugged into each of the three PCBAs, and the PCBAs were probed with EQP-146 across point C9. All units had a voltage within 5% of 20V. Confirming that the USB ports were properly flashed.
CONCLUSION
WS-008 Passed Operational Qualification. See Section 3.
ATTACHMENTS
VVPR-P01-253 Attachment 1: MWI-252 Rev C - ES-10015 PMUX PCBA Programming
VVPR-P01-253 Attachment 2: MWI-231 Rev G - WS-008 Workstation Installation
REPORT APPROVAL
Digital Key: example.com/
VVPR-P01-253 Attachment 1: MWI-252 Rev C - ES-10015 PMUX PCBA Programming
VVPR-P01-253 Attachment 2: MWI-231 Rev G - WS-008 Workstation Installation
QSF-129 (TEMPLATE) REVISION HISTORY
Digital Key:
example.com/

### Table 1
| Equipment Description: | WS-008, Jetson/NVME Programming Workstation |
| --- | --- |
| Equipment Number: | WS-008 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: |  |
| Completed IQ Document Number | VVPR-P01-221 Rev B - WS-008 Equipment Qualification |

### Table 2
| Document Description: | Document Storage Location: |
| --- | --- |
| MWI-252 Rev C - ES-10015 PMUX PCBA Programming | MedAI QMS |
| MWI-231 Rev G - WS-008 Workstation Installation | MedAI QMS |

### Table 3
| EQP | EQUIPMENT/ INSTRUMENT/ DEVICE | MFG./MODEL # | RANGE |
| --- | --- | --- | --- |
| EQP-146 | Multimeter | Fluke 2670150 | Up to 1000V |

### Table 4
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-008 | N/A | Programming Workstation | 1 | MWI-231 WS-008 Workstation Installation Rev G |
| H1 | D | Wired Charger | 1 | N/A |
| Unit Under Test (UUT) |  |  |  |  |
| ES-10015 | D.1 | ES-10015 PMUX PCBA | 1 | Sample #1SN/Lot: |
| ES-10015 | D.1 | ES-10015 PMUX PCBA | 1 | Sample #2 SN/Lot: |
| ES-10015 | D.1 | ES-10015 PMUX PCBA | 1 | Sample #3SN/Lot: |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 5
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
|  | VVPR-P01-253 Rev A - WS-008 Workstation PMUX PCBA Programming Operational Qualification |  |
|  | MWI-252 Rev C- ES-10015 PMUX PCBA Programming |  |

### Table 6
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-252, but in Procedure Section step 5 leave the JTAG connector disconnected then continue Procedure Section through step 18. | The Flash to Device popup will remain on the “Erasing Device” step and not move past 0% |  | Pass Pass with Deviation Fail |
| 2 | Follow instructions in MWI-252, but in Procedure Section step 6, leave the USB-mini B connector disconnected from T-127 and continue Procedure Section through Step 12. | An exception window pops up with the message, “no FTDI I2C channels detected. |  | Pass Pass with Deviation Fail |

### Table 7
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the workstation on, login per test procedure in MWI-252. | PC Starts windows, login successful, desktop is shown. |  | Pass Pass with Deviation Fail |
| 2 | Program the 3 UTT PMUX PCBA USB-PDs following the instructions in MWI-252. | Popup declares flash has been successful |  | Sample #1 Pass Pass with Deviation Fail |
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
| Known Issues (if any): | Pass Pass with Limitations Fail |
| Result: |  |
| Completed By: |  |
| Date Completed: |  |

### Table 11
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Operations | 16 Jan 2025 | 25-028 |

### Table 12
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| EQP-146 | Multimeter | 7/11/2024 | 7/31/2025 | E. Holt 1/17/24 |

### Table 13
| Equipment Description: | WS-008, Jetson/NVME Programming Workstation |
| --- | --- |
| Equipment Number: | WS-008 |
| Manufacturer: | MedAI |
| Model Number: | N/A |
| Serial Number: | 1259 |
| Completed IQ Document Number | VVPR-P01-221 Rev B - WS-008 Equipment Qualification |

### Table 14
| Part Number | Rev | Description | Qty | Notes |
| --- | --- | --- | --- | --- |
| Test Materials List |  |  |  |  |
| WS-008 | N/A | Programming Workstation | 1 | MWI-231 WS-008 Workstation Installation Rev G |
| H1 | D | Wired Charger | 1 | N/A |
| Unit Under Test (UUT) |  |  |  |  |
| ES-10015 | D.1 | ES-10015 PMUX PCBA | 1 | Sample #1SN/Lot: LOT10255-0010 |
| ES-10015 | D.1 | ES-10015 PMUX PCBA | 1 | Sample #2 SN/Lot: LOT10255-0013 |
| ES-10015 | D.1 | ES-10015 PMUX PCBA | 1 | Sample #3SN/Lot: LOT10255-0011 |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 15
| NAME | TRAINING RECEIVED | DATE |
| --- | --- | --- |
| Iris Holt | VVPR-P01-253 Rev A - WS-008 Workstation PMUX PCBA Programming Operational Qualification | 1/16/2025 |
| Iris Holt | MWI-252 Rev C- ES-10015 PMUX PCBA Programming | QSF-076 Production Training Record_B - Iris Holt10/3/2024 |

### Table 16
| Objective: To verify that equipment end limits, warning signals, Quality Inspection and Test are identifying defects properly during normal production mode. Instructions: List all faults introduced to the product along with the serial number(s) of the units affected. Document if and where faults were detected. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Follow instructions in MWI-252, but in Procedure Section step 5 leave the JTAG connector disconnected then continue Procedure Section through step 18. | The Flash to Device popup will remain on the “Erasing Device” step and not move past 0% | The Flash to Device popup remained on the “Erasing Device” step and did not move past 0% | Pass Pass with Deviation Fail |
| 2 | Follow instructions in MWI-252, but in Procedure Section step 6, leave the USB-mini B connector disconnected from T-127 and continue Procedure Section through Step 12. | An exception window pops up with the message, “no FTDI I2C channels detected. | An exception window popped up with the message, “no FTDI I2C channels detected. | Pass Pass with Deviation Fail |

### Table 17
| Objective: To verify that equipment operates as intended during normal production mode. Instructions: List work instruction steps as necessary to demonstrate intended operation. Acceptance Criteria: Confirm by inspection that the executed test results match expected results documented below. Complete the following table: |  |  |  |  |
| --- | --- | --- | --- | --- |
| STEPS | TEST | EXPECTED RESULTS | ACTUAL RESULTS | PASS/ FAIL |
| 1 | Turn the workstation on, login per test procedure in MWI-252. | PC Starts windows, login successful, desktop is shown. | PC Starts windows, login successful, desktop is shown. | Pass Pass with Deviation Fail |
| 2 | Program the 3 UTT PMUX PCBA USB-PDs following the instructions in MWI-252. | Popup declares flash has been successful | Popup declared flash was successful | Sample #1 Pass Pass with Deviation Fail |
|  |  |  | Popup declared flash was successful | Sample #2 Pass Pass with Deviation Fail |
|  |  |  | Popup declared flash was successful | Sample #3 Pass Pass with Deviation Fail |
| Overall Results: Pass Pass with Deviation Fail |  |  |  |  |
| The number of test samples shall be determined by Engineering and Quality and shall be statistically relevant. |  |  |  |  |

### Table 18
| Summarize the OQ and document any conclusions and/or recommendations in the Test Report. Summarize risk assessments and any changes or additions needed to the control plan and/or PFMEA. Summary of acceptance or failure of Equipment Qualification. |
| --- |
| WS-008 properly configures the PMUX USB-PD port when following MWI-252 and consistently produces acceptable units. Actual results matched expected results. Testing demonstrated PMUX USB-PD was successfully programmed. No changes or additions are required to the risk assessment. WS-008 has been validated for use in flashing the USB-PD on the PMUX, Emitter Main and WTX. |

### Table 19
| Provide a general summary of the equipment function and intended use: Describe the actions to be taken if the test results are not acceptable. This information must be very specific to what actions, why, who, the date of completion and verification by Quality Assurance. |
| --- |
| N/A |

### Table 20
| System Name: | WS-008, Programming Workstation MWI-231 Rev G - WS-008 Workstation Installation |
| --- | --- |
| System Version: | N/A |
| Known Issues (if any): | Pass Pass with Limitations Fail |
| Result: | Pass |
| Completed By: | E. Holt |
| Date Completed: | 1/16/2025 |

### Table 21
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report Release | Quality Engineering Ops | 21 Jan 2025 | 25-037 |

### Table 22
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Operations | 15 Jan 2025 | 25-028 |
