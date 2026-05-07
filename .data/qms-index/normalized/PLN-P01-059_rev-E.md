# PLN-P01-059 Rev E: MX1 Assembly Workstations Verification and Validation Plan

## Metadata
- Document ID: PLN-P01-059
- Revision: E
- Prefix: PLN
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: PLN-P01-059 - MX1 Assembly Workstations Verification and Validation Plan_E-signed.docx
- Source path: Example QMS - MedAI/PLN-P01-059 - MX1 Assembly Workstations Verification and Validation Plan_E-signed.docx
- Extraction warnings: none

## Extracted Content
Purpose
Document the verification and validation plan for the MX1 Assembly Workstations developed by MedAI, Inc to perform in-process testing and assess subassembly performance during the manufacturing of the MX1 System or MX1-IND System including the Cassette (C1), Emitter (E1), Foot Pedal (F1) and Wireless Charger (W1) assemblies. The MX1 Assembly WorkStations include:
References
PLN-P01-060 - MX1 Project Plan
PLN-P01-044 - W1 Wireless Charger Project Plan
DR-P01-005 - Design Inputs
VVAM-P01-004 - Verification & Validation Trace Matrix
MEMO-P01-604 - MedAI Diagnostic Tool Software Requirements Specification
MEMO-P01-695 Rev C - ODT SW System Architecture Diagram
VVPR-SWV-011 Rev B - MX1 MedAI Rest Server Verification and Validation
VVPR-SWV-013 Rev B - Camera Self Test Utility Verification and Validation
VVPR-SWV-015 Rev B - EDID Writer Verification and Validation
VVPR-SWV-019 Rev B - Cassette OLED Tester Verification and Validation
VVPR-SWV-020 Rev B - iRay License Checker Verification and Validation
Definitions
WS = “WorkStation”, developed to enable MX1 manufacturing in-process testing for the Emitter (E1), Cassette (C1) and Foot Pedal (F1) and provide manufacturing data that demonstrates subassemblies meet requirements. Work Stations shall be designed to test requirements of each subassembly.
ODT = “MedAI Diagnostic Tool”, software tool developed to provide the means for an operator to interface with each WorkStation and the unit under test (UUT), perform required subassembly testing and document test results for each subassembly.  The ODT has two modes: Engineering Mode and Production Mode.  Engineering Mode allows for individual tests to be selected and performed, facilitates troubleshooting activities, and generates data for each test run.  Production Mode has required tests pre-selected for each WorkStation, informs the operator of test failures and creates a log of the test results which is uploaded to Google Drive at each WorkStation. The mode currently defaults to Engineering Mode when launching ODT. Production Mode is accessible by the operator when a UUT is connected.
UUT = “Unit Under Test”, describes the MX1 subassemblies or assembly being tested at a WorkStation
CMO = “Contract Manufacturer”
Strategy
WorkStations Design Transfer
MedAI shall manufacture required Work Stations at MedAI’s office located at 100 Main Street, Suite 700, Springfield, IL 60001 following Good Manufacturing Practices (GMP).
WorkStations should be released in at least four tranches:
Tranche 1 includes WS-001, WS-004, WS-005 and WS-007
Tranche 2 should include WS-003, WS-006, WS-008, WS-011 and WS-016 but adjustments to tranche may be made as necessary
Tranches 3 should include WS-002, WS-012, WS-015 and WS-017 but adjustments to tranche may be made as necessary
Tranche 4 should include WS-013 and WS-014 (WS-009 and WS-010 may be added to this tranche if updates to these workstations are required due to battery accuracy modifications. Adjustments to tranche may be made as necessary.
WorkStations may be released individually as necessary to resolve workstation issues and prevent delaying release of remaining workstations in a tranche.
Each WorkStation Tranche shall have a new version of ODT installed to allow implementation of additional workstations, e.g. Tranche 1 shall have v2.0.0-alpha, Tranche 2 v2.1.0-alpha, Tranche 3 v2.2.0-alpha, etc. Once each version is verified the fully released version shall be sent to CMO to be installed on the specified tranche.
Prior to completing ODT verification on a WorkStation tranche with a specified software version, the software version installed on a tranche may be updated.
If ODT verification on a WorkStation tranche has occurred, then the Software Architecture Diagram shall be reviewed to determine if installing the new version on the previous tranche will require additional ODT verification.  Note, the new ODT version will only need to be installed on WorkStations if features impacting ODT as a whole are modified.
Process Risks that may impact WorkStation requirements will be assessed once CMO releases Process Failure Modes and Effects Analysis.
WorkStation Verifications
MedAI shall perform WorkStation Installation Verification (IV) on each workstation at MedAI to verify instructions for WorkStation assembly and setup.
WorkStation Installation shall be performed using released work instructions.
A Bill of Materials shall be documented for each workstation and each component shall be verified to be included in the assembled workstation.
For custom tools, if available, packing slips may be provided to indicate components used in a tool assembly. Part drawings for custom parts are included in the Tool Specifications to enable reordering of these components.
WorkStation assembly instructions and schematics (if applicable) shall be documented in the Installation Verification instructions.
WorkStation installation verification shall be successfully completed for a WorkStation prior to sending the WorkStation to the contract manufacturer (CMO).
Once WorkStation Installation Verification is completed, the WorkStation shall be identified as complete to prevent inadvertent tampering with WorkStation components.
Using ODT Production Mode, MedAI shall perform WorkStation Operation Verification on each workstation at MedAI to verify WorkStation operation and operating instructions.  WorkStation Operation Verification shall be evaluated using MX1 Rev E assemblies built with released work instructions.
One unit is required for WorkStation operation and operating instructions verification, but additional units may be used if desired.
WorkStation Operation Verification shall use a preliminary released ODT version (e.g. 2.0.0-alpha). The same version of ODT shall be installed on each WorkStation in the tranche.
ODT may be updated between tranches. In any instance where a new version of ODT is released during WorkStation verification, the changes shall be evaluated for impact to previous operation verification, and regression testing will be performed when applicable.
WorkStation operation verification shall be successfully completed for a WorkStation prior to sending the WorkStation to the CMO.
WorkStation tranches shall be shipped to CMO with the preliminary released ODT version.
MedAI software team shall verify the MedAI Production Tools for ODT operated workstations (mcx-production-bins tools) installed on the Jetson or WorkStation NUC to configure components and assess configurations in order to demonstrate conformance to requirements.
Software Requirement Specifications will be released for each production tool.
Verification of MedAI Rest Server, EDID Writer, iRay License Checker, Camera Self Test and OLED Tester shall be completed prior to ODT verification following MedAI software verification procedures.  Once completed these tools shall be installed on the appropriate WorkStations.
Production tools shall be verified by software engineering per the software verification process. Tools must be verified before performing ODT verification and prior to shipping WorkStations to CMO.
Verification and Validation of MedAI Production Tools operations will be performed using their corresponding MX1 Software System Versions.
Modifications to MedAI Production Tools shall be assessed via change order (ECR) to determine when additional verification testing is required.
MedAI shall perform a verification of other custom mcx-production-bins tools used at non-ODT WorkStations including WS-012, WS-014, WS-015 and WS-017 to demonstrate conformance to requirements. See Section 6 for the complete list of MedAI Production Tools which include the MedAI EOL Tool, the MX1 Detector Calibration Tool, the MX1 Projection Utility Tool and the Camera Calibration Pyramid Tool.
Using ODT Production Mode, released product firmware and released modified firmware versions, MedAI shall perform a Verification of the ODT software by comparing the expected outputs to the actual outputs for each ODT requirement as stated in ODT Software Requirement Specification (SRS) and verifying ODT correctly identifies passing and failing values.
Protocols will be written specifying SRS Sections in MEMO-P01-604 to be tested. Note, sections may be omitted if WorkStations are not ready for validation or changes to a specific WorkStation require revalidation.  A verification of all ODT software requirements will occur prior to shipping WorkStations to CMO.
Code reviews may be sufficient for ODT changes, such as parameter value changes, made after the initial verification that do not affect ODT functionality.
Product firmware installed on properly operating subassemblies will be used to verify ODT operation when expected values within specified limits are present.
Modified product firmware installed on properly operating subassemblies will output register values intended to trigger failures above and below specified limits as well as incorrect queried values.
ODT verification may use an agreed upon, released version of MX1 system software on test and device Jetsons, with rationale.  In other words, using the final production version of MX1 software is not required to complete ODT Verification.
Modifications to ODT shall be assessed and verification of relevant ODT features shall be performed based on a review of the ODT Software System Architecture Diagram.
Single Board Computers (SBC) incorporated into WorkStation designs shall be assessed during ODT Verification to demonstrate their capabilities to perform commands required to meet SRS.
The Contract Manufacturer shall perform Equipment Qualifications (EQ).
Equipment Qualification may include Installation Qualification (IQ), Operation Qualification (OQ), or Test Method Validation (TMV), as applicable.
MedAI will perform Installation Qualifications on all workstations required for MX1-IND builds.
MedAI shall perform EQs on WorkStations required to program PCBAs and Detectors, and verify Emitter, Power Cleat Cap Collimator, and Wireless Charger subassemblies if MedAI intends to produce or ship these programmed components or subassemblies to the CMO for use in Process Qualification or initial production builds.
Equipment Qualifications may include IQ, OQ or TMV as applicable.
MedAI EQs shall be documented using a VVPR to define the installation inspection qualification and operational qualification testing, and record qualification results.
Trained operators shall perform EQ testing following the released EQ VVPR and WorkStation manufacturing work instructions (MWIs). One UUT will be used to evaluate challenge conditions defined in the VVPR unless otherwise instructed in the VVPR.
Three UUTs will be programmed per the MWI and then verified by engineering to demonstrate workstation programming successfully installed firmwares and licenses as required.
Equipment Qualification data collected shall be reviewed and approved by Quality Assurance.
Workstation EQs shall be performed on all MedAI workstations prior to use for servicing MX1 devices.
MedAI shall not perform Performance Qualifications on programming PCBAs or detectors because verifying the programming workstation outputs properly programmed components during EQ demonstrates the programming process defined in the MWIs yields acceptable components. See Section 4.2.8.4.
CMO will include components programmed at MedAI in CMO process validations.
MedAI shall perform Performance Qualifications on processes required to supply Emitter, Collimator, and Power Cleat Cap subassemblies to the CMO If MedAI intends to provide said components.
Released MWIs shall define the process to be qualified.
Trained Operators shall follow released MWIs to produce 3 units, recording required Device History Record (DHR) information.
Quality Assurance shall review and approve DHRs, and note a successful process qualification in the DHR if applicable.
The Contract Manufacturer shall use WorkStations during Process Qualification (PQ).
MedAI will not perform Performance Qualification on processes required to flash and test the Wireless Charger PCBA as workstation EQ demonstrates the defined programming process yields acceptable components and all process outputs will be 100% verified. MedAI may choose to perform Performance Qualification for the Wireless Charger assembly process in the future to remove the 100% verification inspection steps.
MedAI shall perform Performance Qualification on workstations to be used for servicing by qualifying workstations during internal MX1 engineering or pilot production builds if not previously qualified.
WorkStation Changes During or After Testing
Once internal Installation and Operation Verifications are completed for a WorkStation configuration, changes to the WorkStation shall be assessed prior to implementation. If changes are approved by the cross-functional team, an impact assessment shall be completed to determine if additional verification is required.
Once WorkStation configuration is released, changes to the WorkStation identified to be incorporated after CMO EQ testing (e.g equipment and fixture modifications) shall be documented by the CMO, and EQ or other required testing shall be repeated as necessary. If changes are determined to not impact previous EQ, then rationale shall be documented.
Changes to ODT required as a result of CMO EQ testing (e.g. software updates) shall be provided to MedAI for further evaluation.  EQ shall be repeated by the CMO as necessary. If changes are determined to not impact previous EQ, then rationale shall be documented required.
WorkStation and ODT documentation shall be updated to reflect all changes per the Engineering Change Request WI-005.
Methods
Requirements may be verified by the following methods:
Inspection: Verification by BOM, schematics and drawings (where applicable) reviews.
Demonstration: Verification by performing an action with the WorkStation as indicated by the stated requirement. Demonstrations shall not require the use of non-device equipment.
Test: Verification by measurement/data collection using the WorkStation during Installation and Operation Verifications at MedAI and IQ/OQ at CMO.
Analysis: Verification by review of test data and/or software code. Analysis is to be used where the requirement has no obvious input and output.
Traceability
The verification method, results, and pass/fail outcome for all design inputs shall be documented during ODT verification..
The ODT and product software versions installed on workstations shall be released and documented in either the tool documentation or Workstation Installation Instructions.
If an Equipment Qualification is performed, then the ODT version on the workstation shall be recorded in the EQ.
The product firmware version used for each board shall be released and documented in the software BOM (SBOM). Modified product firmware used for ODT verification shall be released as a part, but not included in the SBOM as it is only utilized for ODT verification.
Where modified firmware is used for verification, a description of the firmware function will be documented in the Verification Summary Report.
The verification method, results, and pass/fail outcome for WorkStation operation shall be documented in the EQ report.
Identified traceability requirements shall be included in CMO production documentation.
Third Parties
After completing verifications, WorkStations may be disassembled and packaged for shipment to the Contract Manufacturer (CMO) or new WorkStation components may be shipped to the CMO. Released MedAI Installation and Operation Verification instructions shall be provided to CMO.
The CMO shall release internal EQ protocols for each workstation which MedAI QA shall approve. The CMO shall perform EQ testing and document prior to using a workstation on the manufacturing line.
The CMO shall include WorkStations in the MX1 Process Qualification (PQ).
Sample Sizes
Preliminary Installation and Operation Verifications of WorkStation operation using released work instructions will be completed with one sample (n=1) as the procedures do not have any variability.  Additional samples may be used if desired.
Verifying ODT (the software used to perform testing at each workstation) operation will be completed with one sample (n=1) of each subassembly required by the WorkStations. The modified firmware on each unit will provide lower limit failing values, nominal passing values and upper limit failing values.
Operation and Process Qualifications will be completed with three samples of each subassembly required by the WorkStations since this is representative of quantities built in a three day period.
Quality Controls
WorkStation parts are not required to be purchased from Approved Suppliers.
Custom parts should be received and inspected per released drawings.
All parts shall be appropriately qualified through purchasing controls and process validation.
WorkStation Descriptions
MedAI Production Tools Description
Bench Performance Testing
ODT Verification
MedAI Production Tools Verification
Appendix
WorkStation Design Transfer Process
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| ID No. | Installation MWI | Operation MWI | Name | Description |
| --- | --- | --- | --- | --- |
| WS-001 | MWI-215 | MWI-216 | ES-10003 Emitter Verification | See Section 5 |
| WS-002 | MWI-219 | MWI-220 | MS-10008 Collimator Verification | See Section 5 |
| WS-003 | MWI-221 | MWI-222 | MS-10155 X-ray Assembly Calibration | See Section 5 |
| WS-004 | MWI-223 | MWI-224 | MS-10149 Camera Sensor Board Verification | See Section 5 |
| WS-005 | MWI-225 | MWI-226 | MS-10301 HMI Display Verification | See Section 5 |
| WS-006 | MWI-227 | MWI-228 MWI-271 | MS-10141 PMUX Verification MS-10568  - Wireless Charger Verification | See Section 5 |
| WS-007 | MWI-229 | MWI-230 | MS-10511 Cassette Verification | See Section 5 |
| WS-008 | MWI-231 | MWI-232 MWI-248 MWI-249 MWI-250 MWI-251 MWI-252 MWI-253 MWI-254 | M50101/M50095-Jetson & NVMe Programming ES-10004 Cassette Main PCBA Programming ES-10003 Emitter Main PCBA Programming ES-10008 Collimator PCBA Programming ES-10019 Monoblock LV PCBA Programming ES-10015 PMUX PCBA Programming ES-10007 Foot Pedal PCBA Programming ES-10027 Wireless Charger PCBA Programming | See Section 5 |
| WS-009 | MWI-207 | MWI-208 MWI-209 | MS-10022 Cassette BMS Programming MS-10033 Emitter BMS Programming | See Section 5 |
| WS-010 | MWI-210 | MWI-211 MWI-214 | MS-10010 Emitter Battery Pack Verification MS-10083 Cassette Battery Pack Verification | See Section 5 |
| WS-012 | MWI-234 | MWI-235 MWI-237 MWI-239 | E1 / C1 Software Upgrade E1 / C1 End of Line E1 / C1 Password Setting and Release Mode | See Section 5 |
| WS-013 | MWI-298 | MWI-299 | MWI-299 - ES-10007 Verification & Pairing ID Reading | See Section 5 |
| WS-014 | MWI-240 | MWI-241 | F1 Foot Pedal Verification | See Section 5 |
| WS-015 | MWI-259 | MWI-260 | Monoblock Verification | See Section 5 |
| WS-017 | MWI-275 | MWI-276 | High Voltage Verification | See Section 5 |

### Table 2
| WS ID No. | Name | Description |
| --- | --- | --- |
| WS-001 | ES-10003 Emitter Main EDID Programming & Verification | WorkStation verifies Emitter Main PCBA functions as intended and flashes the EDID to the Emitter Main EEPROM. |
| WS-002 | MS-10008 Collimator Verification | WorkStation verifies Collimator PCBA functions as intended using ODT. |
| WS-003 | MS-10155 X-ray Assembly Calibration | WorkStation enables calibration of X-Ray assembly in a temperature-controlled, shielded environment using an Emitter Main and supplementary electronics (including a Raspberry Pi) by measuring X-Ray emissions and modifying on-board parameters using ODT. |
| WS-004 | MS-10149 Camera Sensor Board Verification | WorkStation verifies Camera Sensor Board functions as intended with Emitter test station using ODT. |
| WS-005 | MS-10301 HMI Display Verification | WorkStation verifies HMI Subassembly functionality (including all buttons) at nominal power settings using ODT to command a Raspberry Pi. |
| WS-006 | MS-10141 PMUX and Wireless Transmitter PCBA Verification | WorkStation verifies PMUX functions as intended using ODT. WorkStation verifies wireless transmitter functionality. Confirms wireless charger turns on, configures electronics settings and verifies functionality (power transmission, foreign object detection and temperature monitoring) |
| WS-007 | MS-10511 Cassette Verification | WorkStation verifies Cassette Main PCBA, Cassette Tracking PCBA, Cassette Angled Tracking and Cassette Display function as intended using ODT. |
| WS-008 | Jetson, NMVe and PCBA Programming | WorkStation provides capability to flash Emitter and Cassette Jetsons’ NVMes, and flash firmware on Emitter PCBAs: Power Input , Emitter Main, Collimator, Monoblock LV, Cassette PCBAs: Cassette Main and Cassette Detector, and Wireless Charger PCBA: Wireless Transmitter PCBA.  WorkStation has NUC partitioned with two OS: Windows (used for firmware flashing) and Linux (used for detector and Jetson flashing) and uses off the shelf software (STM32CubeProgrammer for f/w, Texas Instrument tool for USB-PD), and MedAI supplied programs. |
| WS-009 | Battery Management System (BMS) Programming | WorkStation for programming BMS boards with a Bed of Nails before assembly to battery packs using BMS Programming Utility developed by MedAI. |
| WS-010 | Battery Pack Verification | WorkStation for verification of battery packs after assembly using BMS Programming Utility developed by MedAI. |
| WS-012 | E1 / C1 SW Configuration & Verification E1 / C1 Pairing E1 / C1 Detector Calibration E1 / C1 Camera Calibration Pyramid E1 / C1 Focal Spot Calibration E1 / C1 Time of Flight Calibration E1 / C1 Burn-in E1 / C1 Password Setting | WorkStation for End of Line verification utilizing a shielded chamber to calibrate IR camera inside of an Emitter with an IR LED target and focal spot, calibrate Time of Flight and Detector, run Device Burn-in and complete software configuration & verification using MedAI developed software. WorkStation for aligning the crosshair lasers to the x-ray optic axis |
| WS-013 | F1 Foot Pedal Programming | WorkStation for verifying Foot Pedal PCBA & reading the unique Pairing ID with a Bed of Nails fixture before assembly into the F1 Foot Pedal. |
| WS-014 | F1 Foot Pedal Verification | WorkStation for verification of foot pedal PCBA and connected button interfaces. |
| WS-015 | MS-10579 Monoblock Verification | WorkStation for verification of tube assembly and Monoblock assembly. |
| WS-017 | MS-11235 Monoblock Power Assembly Verification | WorkStation for verification of all High Voltage components working together in order to detect dielectric breakdown or component failures prior to potting. |

### Table 3
| PN | Name | Description | Applicable to WorkStation: |
| --- | --- | --- | --- |
| S10038 | MX1 BMS Programming Utility | A C# program used to interface with the two chips that reside on the BMS PCB: the BQ76952 pack monitor IC and the MAX17205 coulomb counter | WS-009 |
| S10044 | MX1 BMS Programming Utility Firmware | Firmware residing on the NUCLEO-F401RE that is used to help interface between the MX1 BMS Programming Utility and the BMS PCB | WS-009 |
| S10046 | MX1 MedAI Rest Server | A REST-based service that can be used to send ICD commands to the firmwares | WS-001, WS-002, WS-003, WS-004, WS-005. WS-007, WS-012 |
| S10057 | MX1 Camera Self Test | Tool used to conduct IR and ViewFinder camera self-tests | WS-004 |
| S10058 | MX1 Projection Utility | Tool used to perform focal spot calibration | WS-012 |
| S10059 | MX1 EDID Writer | Script used to flash the Emitter Main PCBA EDID EEPROM | WS-001 |
| S10052 | MX1 HMI/Monoblock Rest Server | REST API residing on the Raspberry Pi that is used to help interface between ODT and the HMI display | WS-003, WS-005 |
| SS-10053 | MX1 HMI/Monoblock Test Interface Software Image | Image for the Raspberry Pi used in WS-003 and WS-005. Image contains S10052. | WS-003, WS-005 |
| S10093 | MX1 Iray License Checker | Tool used to check the license of the Iray Detector in the Cassette | WS-007 |
| S10094 | MX1 OLED Tester | Tool used to test the OLED display on the Cassette | WS-007 |
| S10096 | medai-eol-tool | Tool that performs all steps of EOL (configuration setup, burn in, camera calibration, projection calibration, TOF calibration, detector calibration and config verification) in a operator accessible manner | WS-012 |
| S10097 | MX1 Camera Calibration Pyramid | Tool that performs Camera Calibration, the application will be opened by the medai-eol-tool | WS-012 |

### Table 4
| Test Performed | Objective | Sample Size/ Description | Consensus Standard(s) | Testing & Report Completed by |
| --- | --- | --- | --- | --- |
| MX1 WorkStation Installation Verification | The objective of this testing is to verify the MX1 WorkStations can be installed using the associated documentation. | n = 1 | N/A | MedAI |
| MX1 WorkStation Operation Verification | The objective of this testing is to verify the MX1 WorkStations operates as intended using the associated documentation. | n = 1 | N/A | MedAI |

### Table 5
| Test Performed | Objective | Sample Size/ Description | Consensus Standard(s) | Testing & Report Completed by |
| --- | --- | --- | --- | --- |
| MedAI Diagnostic Tool VVPR | The objective of this testing is to validate the MedAI Diagnostic Tool | n = 1 | N/A | MedAI |

### Table 6
| Test Performed | Objective | Sample Size/ Description | Consensus Standard(s) | Testing & Report Completed by |
| --- | --- | --- | --- | --- |
| VVPR-SWV-011 - MX1 MedAI Rest Server Verification and Validation Protocol | The objective of this testing is to validate the MedAI Rest Server | n = 1 | N/A | MedAI |
| VVPR-SWV-012 - MX1 Camera Utility Verification and Validation Protocol | The objective of this testing is to validate the MX1 Camera Utility | n = 1 | N/A | MedAI |
| VVPR-SWV-013 - MX1 Camera Self Test Verification and Validation Protocol | The objective of this testing is to validate the Camera Self Test | n = 1 | N/A | MedAI |
| VVPR-SWV-014 - MX1 Projection Utility Verification and Validation Protocol | The objective of this testing is to validate the Projection Utility | n = 1 | N/A | MedAI |
| VVPR-SWV-015 - MX1 EDID Writer Verification and Validation Protocol | The objective of this testing is to validate the EDID Writer | n = 1 | N/A | MedAI |
| VVPR-SWV-020 - MX1iRay License Checker Verification and Validation Protocol | The objective of this testing is to validate the MX1 iRay License Checker | n=1 | N/A | MedAI |
| VVPR-SWV-019 - MX1 OLED Testing Verification and Validation Protocol | The objective of this testing is to validate the MX1 OLED allows operators to assess the functionality of the Cassette OLED display. | n=1 | N/A | MedAI |
| VVPR-SWV-021- MedAI End of Line Verification and Validation Protocol | The objective of this testing is to validate the MedAI End of Line Tool | n=1 | N/A | MedAI |
| VVPR-SWV-023 - MX1 Camera Calibration Pyramid Verification and Validation Protocol | The objective of this testing is to validate the MX1 Camera Calibration Pyramid | n=1 | N/A | MedAI |

### Table 7
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 26 Apr 2024 | 24-062 |
| B | Added WorkStations -013 and -017. Incorporated WS-016 into WS-006 Removed WS-011 Updated 4.1 to include tranches and updated design transfer process. Updated plan to change MedAI Installation Qualification and Operation Qualification to verifications. Updated CMO IQ/OQ to Equipment Qualification. Updated MedAI Production Tools in Section 9. Added Design Transfer Process Flowchart in Section 10. | Engineering Quality Engineering Regulatory Affairs | 19 Sep 2024 | 24-416 |
| C | Add Sections 4.2.8 - 4.2.10 to include workstation qualification plans | Refer to ECR-604 |  |  |
| D | Add workstation plans for MX1-IND devices Update Section 4 Strategy to add 4.2.5.2, 4.2.8, 4.2.10, and 4.7.3, and update 4.2.11,4.2.12, 4.3.3, 4.3.4, and 4.5 Remove tensioning from WS-002 as this feature no longer exists at workstation Update Table 6 S10046 row to include WS-012 | Engineering Quality Engineering Regulatory Affairs | 17 Dec 2024 | 24-727 |
| E | Added reference to Wireless Charger for clarity where applicable | Engineering Quality Engineering Operations | 14 Mar 2025 | 25-163 |
