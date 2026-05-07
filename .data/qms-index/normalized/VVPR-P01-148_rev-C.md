# VVPR-P01-148 Rev C: WTX Foreign Object Detection and Resonance Lock Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-148
- Revision: C
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-148 - WTX Foreign Object Detection and Resonance Lock Verification Protocol and Report_C-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-148 - WTX Foreign Object Detection and Resonance Lock Verification Protocol and Report_C-signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
Wireless chargers pose the risk of transmitting energy into metal objects unintentionally which may cause a burn hazard to users. This study is being conducted to verify implemented safety features prevent the W1 Wireless Charger from transmitting energy into metal objects.
OBJECTIVE
The primary study objective is to verify foreign object detection (FOD) and the resonance lock features of the W1 wireless charger perform as intended and prevent the W1 Wireless Charger from transmitting energy into metal objects.
REFERENCES
IEC 60601-1 Ed. 3.1 Clause 4.2.3.2
IEC 62368-1 Ed 4.0 Clause 9.6
DR-P01-005 Rev A  - Design Inputs
RSK-P01-010 Rev B
MATERIALS
Materials and Equipment
Representative verification prototype of W1 Wireless Charger, BOM Rev C to match the final design. Traceability to be included in the report.
EQP-219 Steel disc -  per specification in IEC 62368-1 Clause 9.6.2
EQP-220 Aluminum (Al) Ring  -per test object specification in IEC 62638-1 Clause 9.6.2
EQP-221 Aluminum (Al) Foil - per test object specification in IEC 62638-1 Clause 9.6.2
T-113 REV A - PMUX Test Fixture
MS-10141 REV C- Power Cleat Assembly
ES-10027 Rev D - Wireless TX
MS-10476 - Electronic Load to PMUX harness, Male Rev. A
M50683 - USB Type-C and Power Delivery Discovery kit with STM32G071RB MCU Rev. A
EQP-045 or equivalent Temperature Monitor
H1 - 100 Watt USB-C PD Medical Desktop Power Supply
EQP-122 or equivalent - Dual Input DC Electronic Load
EQP-104 or equivalent - Caliper
SAMPLE SIZE
The tests performed in this protocol are based on test methods in IEC 62368-1 ed 4.0. This test will utilize a sample size of 1 due to the nature of the tests in question. Per IEC 60601-1:2005+Amd 1:2012 section 5.2, these tests are considered TYPE TESTS and are performed on a representative sample of the item being tested (n=1).
METHODS
Locations and Personnel Responsibilities
This study shall take place in MedAI facilities.
The Electrical Engineering Department (EE) - EE is responsible for execution of the protocol.
Where screenshot evidence is not attached the personnel who verified the outcome/expected results shall enter the test result, and sign and date the requirement was verified.
Training Requirements - This study does not require any special training.
Equipment Requirements - This study shall use calibrated equipment.
Note - When possible, in order to maintain internal independence of review, MedAI shall assign internal staff members that are not involved in a particular design or its implementation, but who have sufficient knowledge to evaluate the project, to conduct the verification and validation activities.
Experimental Procedure
Tests selected were based on those applicable in IEC 60601-1: Ed 3.1 and IEC 62368-1 Ed 4.0 section 9.6.3, and W1 wireless charger design requirements, DR-P01-004 Rev. C. Applicable clauses from the IEC 62638-1 standard, W1 design requirements and W1 risk requirements are documented in Table 1 below.
Table 1. FOD Tests, Methods and Acceptance Criteria
The W1 wireless charger maximum power is 100W. Transmitter temperature shall be measured by an onboard digital temperature sensor and displayed with an external display board. The onboard temperature measurement of the TX coil will yield the maximum temperature in the W1 assembly. Therefore, the temperature of the wireless charger exposed front face will be inferred to be less than the TX coil.  Foreign object temperature shall be measured using the data logger (EQP-045 or equivalent).
Data Collection Form
*Maximum Allowable temperature per IEC 62368-1:2023 Table 37 - Touch temperature limits for accessible parts - “Surfaces that does not have to be touched to operate the equipment (< 1 s)”
Performed By: ___________________________________________________________________Date: __________________
Data Analysis
Variable data outputs will be recorded, compared to acceptable limits and marked Pass or Fail accordingly.  Attribute data outputs will be recorded, compared to acceptable state and marked Pass or Fail accordingly. For each test case all results will be reviewed to determine overall test case result. All individual tests must pass for the overall test case to pass.
Data analysis is not required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Table 1.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
There were no deviations to the protocol.
DEVICES, COMPONENTS, OR EQUIPMENT USED
3. RESULTS
Performed By: Kit Jensen         Date: 17 May 2024
4. CONCLUSION
Verification testing demonstrated that foreign objects placed on the transmitter are detected by the W1 resulting in minimal temperature rises that are below the maximum allowed limit for all tests.  The Wireless Charger, W1 Rev C, passes the Foreign Object Detection and Resonance Lock Verification testing per IEC 62368-1.
5. APPENDIX/ATTACHMENTS
None
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Risk ID | Requirement ID | Risk Mitigation Text for Reference | Test Item ID | Verification Methods and Acceptance Criteria |
| --- | --- | --- | --- | --- |
| Ac9.6 R.W1.9.4 R.W1.9.5 | W1.PRD.1.3 | Device shall have foreign object detection and resonance to limit transmission to only when in proximity of proper receivers | 1 | The test procedure in IEC 62368 9.6.3 Part A shall be followed. Transmitter temperature will be recorded from the onboard digital temperature sensor reporting maximum temperature on the TX coil. See Data Collection Form in 6.3 for acceptance criteria. |
| Ac9.6 R.W1.9.4 R.W1.9.5 | W1.PRD.1.3 | Device shall have foreign object detection and resonance limit transmission to only when in proximity of proper receivers | 2 | The test procedure in IEC 62368 9.6.3 Part B shall be followed. Transmitter temperature will be recorded from the onboard digital temperature sensor reporting maximum temperature on the TX coil. See Data Collection Form in 6.3 for acceptance criteria. |
| R.W1.9.4 R.W1.9.5 | W1.PRD.1.3 | Device shall have foreign object detection and resonance to limit transmission to only when in proximity of proper receivers | 3 | While performing Test Item 1, LED behavior, and input power to the device shall also be recorded. For all test cases, the LEDs should be RED and input power should be <5 W. |
| R.W1.1.72 | W1.PRD.1.11 | Device shall have foreign object detection and resonance to limit transmission to only when in proximity of proper receivers | 4 | The W1 wireless charger shall be set up to charge the E1. Foreign objects shall be positioned as near to the charging interface as possible for 1 minute or until temperature change < 1℃ over 10 seconds. Foreign object temperatures will be recorded using EQP-045 or equivalent. See Data Collection Form in 6.3 for acceptance criteria. |
| Ac9.6 | RSK_R71 | The IFU shall warn against placing metal or devices in front of the Device | 5 | IFU Section 4 - Using the Wireless Charger,  Wirelessly Charging the Emitter WARNING: Do not place other metals or devices nearby in close proximity to the wireless charger. Doing so may cause the charger to supply power to those metals or devices, potentially damaging them or heating them to harmful temperatures. |

### Table 2
| Test ID | Foreign Object | FO Temp (C) | Max Allowed Temp (C) | P/F | TX Temp (C) | Max Allowed Temp (C) | P/F | LED State | P/F P=RED | Input Power (W) | Max Input Power (W) | P/F | Overall Result P/F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,3 | Steel Disc |  | 85 |  |  | 104* |  |  |  |  | 5 |  |  |
| 1,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 1,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Steel Disc |  | 85 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Ring |  | 120 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 2,3 | Al Foil |  | 155 |  |  | 104 |  |  |  |  | 5 |  |  |
| 4 | Steel Disc |  | 85 |  |  | 104 | N/A | N/A | N/A | N/A | N/A | N/A |  |
| 4 | Al Ring |  | 120 |  |  | 104 | N/A | N/A | N/A | N/A | N/A | N/A |  |
| 4 | Al Foil |  | 155 |  |  | 104 | N/A | N/A | N/A | N/A | N/A | N/A |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 15 Mar 2024 | 24-073 |
| B | Update referenced documentation, equipment list to include IEC specified test instruments and data tables to remove erroneous data collection | Engineering Quality Engineering Regulatory Affairs | 17 May 2024 | 24-252 |

### Table 4
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 5
| Equipment | Description | Last Calibration Date | Calibration Due |
| --- | --- | --- | --- |
| EQP-219 | Steel disc | 2024-03-12 | 2025-03-11 |
| EQP-220 | Aluminum (Al) Ring | 2024-03-12 | 2025-03-11 |
| EQP-221 | Aluminum (Al) Foil | 2024-03-12 | 2025-03-11 |
| EQP-045 | Temperature Monitor | 2023-10-31 | 2024-10-30 |
| EQP-122 | Dual Input DC Electronic Load | 2023-10-31 | 2024-10-30 |
| EQP-104 | Caliper | 2023-10-31 | 2024-10-30 |

### Table 6
| Material or Tool | Description | Lot/Serial Number |
| --- | --- | --- |
| MS-10141 REV C | Power Cleat Assembly | 10048 |
| ES-10027 Rev D | Wireless TX | 10105 |
| MS-10476 Rev. A | Electronic Load to PMUX harness, Male | MS-10476 continuity verified via: EQP-147 Cal: 29 Jun 2023 Cal Due: 28 Jun 2024 |
| M50683 Rev. A | USB Type-C and Power Delivery Discovery kit with STM32G071RB MCU | PO22790 |
| T-113 REV A | PMUX Test Fixture | Lot:10185 SN:01 |
| H1 REV A | 100 Watt USB-C PD Medical Desktop Power Supply | M10173 M10172 |

### Table 7
| Test ID | Foreign Object | FO Temp (C) | Max Allowed Temp (C) | P/F | TX Temp (C) | Max Allowed Temp (C) | P/F | LED State | P/F P=RED | Input Power (W) | Max Input Power (W) | P/F | Overall Result P/F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,3 | Steel Disc | 37.9 | 85 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Steel Disc | 38.0 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Steel Disc | 38.9 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Steel Disc | 39.9 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Ring | 33.3 | 120 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Ring | 33.2 | 120 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Ring | 34.2 | 120 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Ring | 35.2 | 120 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Foil | 39.8 | 155 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Foil | 38.3 | 155 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Foil | 40.9 | 155 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 1,3 | Al Foil | 42.1 | 155 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Steel Disc | 39.0 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Steel Disc | 35.4 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Steel Disc | 36.1 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Steel Disc | 39.8 | 85 | P | 38 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Ring | 33.4 | 120 | P | 36 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Ring | 33.0 | 120 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Ring | 34.4 | 120 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Ring | 35.6 | 120 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Foil | 34.9 | 155 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Foil | 38.9 | 155 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Foil | 41.6 | 155 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 2,3 | Al Foil | 42.5 | 155 | P | 37 | 104 | P | RED | P | 2.5 | 5 | P | P |
| 4 | Steel Disc | 42.0 | 85 | P | 65 | 104 | N/A | N/A | N/A | N/A | N/A | N/A | P |
| 4 | Al Ring | 39.1 | 120 | P | 65 | 104 | N/A | N/A | N/A | N/A | N/A | N/A | P |
| 4 | Al Foil | 41.0 | 155 | P | 65 | 104 | N/A | N/A | N/A | N/A | N/A | N/A | P |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| C | Initial Release Corrected Wireless Charger Revision in Materials section of protocol and equipment required. | Engineering Quality Engineering Regulatory Affairs | 20 May 2024 | 24-258 |
