# PLN-P01-065 Rev C: MX1 Verification and Validation Plan

## Metadata
- Document ID: PLN-P01-065
- Revision: C
- Prefix: PLN
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: PLN-P01-065 - MX1 Verification and Validation Plan_C.docx
- Source path: Example QMS - MedAI/PLN-P01-065 - MX1 Verification and Validation Plan_C.docx
- Extraction warnings: none

## Extracted Content
Purpose
Document the verification and validation plan for MedAI, Inc’s  MX1 Portable X-ray System when used with its optional accessories. The MX1 system consists of the E1 Emitter, C1 Cassette, H1 Wired Charger, P1 Case, and optional accessories K1 Cart (which includes the W1 Wireless Charger), F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP).
References
PLN-P01-061 - Regulatory Plan
DR-P01-005 - MX1 Design Inputs
RSK-P01-010 - Risk Assessment
VVAM-P01-004 - MX1 Verification & Validation Trace Matrix
QSR-028 Finished Goods Identification Log
MEMO-P01-455 - MX1 Software Requirements Specification
MEMO-P01-458 - MX1 Software Traceability Matrix
MEMO-P01-644 - Moxtek Tube Inspection Data
3P-P01-19 - UN38.3 and IEC 62133-2 MX1 Battery Testing
3P-P01-21 - Image Quality Labs DICOM Report for Samsung Galaxy S8+
VVPR-P01-127 K1 Usability Summative Evaluation Protocol and Report
VVPR-P01-140 W1 Usability Summative Evaluation Protocol and Report
3P-P01-24 Nelson Letter Report HCR24078-MAI01 for MX1 Portable X-ray System
3P-P01-26 - MX1 SGS IEC 60601-1-2 EMC Summary Report
3P-P01-27 - F2 Labs Wireless Coexistence and RFID & 5G Immunity Summary
3P-P01-28 Intertek IEC 60601-1, 60601-1-3, 60601-1-6, 60601-2-28, 60601-2-43 and 60601-2-54 MX1 Summary Report
60601-1 Edition 3.2 2020 - General requirements for basic safety and essential performance
60601-1-2 Edition 4.1 2020 Part 1-2: General requirements for basic safety and essential performance - Collateral Standard: Electromagnetic disturbances
60601-1-3 Edition 2.2 2021 - General requirements for basic safety and essential performance - Collateral Standard: Radiation protection in diagnostic X-ray equipment
60601-1-6 Edition 3.2 2020 - General requirements for basic safety and essential performance - Collateral standard: Usability
60601-2-28 Edition 3.0 2017 - Particular requirements for the basic safety and essential performance of X-ray tube assemblies for medical diagnosis
60601-2-43 Edition 2.2 2019 - Particular requirements for the basic safety and essential performance of X-ray equipment for interventional procedures
60601-2-54 Edition 2.0 2022 - Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy
IEEE ANSI C63.27-2021 - American National Standard for Evaluation of Wireless Coexistence
Definitions
DV Units = “Design Verification Units'' used for verification and validation activities. DV Units shall be representative samples of the final design and include all MX1 System Components E1 Emitter, C1 Cassette, H1 Charger (Qty 2), and P1 Case.  Optional accessories F1 Foot Pedal, T1 Tablet with MedAI Device App (APP), K1 cart with Wireless Charger (W1) used in verification and validation activities shall also be representative of the final design.  DV Units are assigned internal reference numbers (DV#) which are correlated to actual serial numbers in the Finished Good Identification Log, QSR-028.
Strategy
Verification and Validation Units
MedAI shall manufacture one engineering verification unit including E1 and C1 to MX1 Rev D.  This unit shall be used for engineering verification testing to identify required design changes to be implemented in Rev E.
The engineering verification unit shall have software version 2.1.0-gamma installed.  If  software or firmware modifications are required during engineering verification these upgrades will be completed by the software team and documented in the DHR.
The engineering verification unit shall be evaluated to determine changes required to incorporate prior to building five Design Verification and Validation units.
The engineering verification unit shall be modified once the engineering evaluation is complete to have the same configuration as the Design Verification units.
MedAI shall build 5 additional Design Verification (DV) Units which include E1 and C1; two units shall be built to Rev D and then updated to Rev E when components are available and three units shall be built to Rev E, at MedAI’s office located at 100 Main Street, Suite 700, Springfield, IL 60001 following Good Manufacturing Practices (GMP).
Emitters shall be built with parts from inventory. Monoblocks with minor arcs may be used as arcing effects the lifetime of the monoblock not the performance.
The DV units used for the imaging study, DV24 and DV26, shall have  monoblocks previously measured during MedAI unofficial incoming inspection to have zero arcs and 0.8 focal spots measured by Moxtek (Ref MEMO-P01-644).
The Cassette shall be built with components from inventory with the exception of the detector.  The detector shall be reprogrammed with the latest version of firmware and iRay license shall be confirmed to be activated per MWI-255.
The most current software version, 2.1.0-gamma and firmware versions shall be installed on the MX1 system and then replaced with system software v3.0.0 when it is available.  A released software version shall be installed prior to verification testing and documented in the DHRs.
Additional mocked units shall be built or rebuilt for internal cleaning for expected service life testing.
The unit for cleaning for expected service life testing shall include shells and other components that need to be evaluated for preventing fluid ingress during cleaning.
The components used shall be documented in a memo including PN, Rev, and Lot/SN as applicable.
MedAI shall build at least two additional Design Verification (DV) Units which include E1 and C1 to Rev F BOM-alpha at MedAI’ office located at 100 Main Street, Suite 700, Springfield, IL 60001 following Good Manufacturing Practices (GMP)
Emitters shall be built with parts from inventory with the exception of Monoblocks. Monoblocks from previous DV units shall be evaluated by engineering to confirm they meet Essential Performance requirements before being built into new verification units. Note, Monoblocks with minor arcs may be used as arcing effects the lifetime of the monoblock not the performance.
The Cassette shall be built with components from inventory with the exception of the detector which may either come from a DV Rev E unit or inventory.  The A0 detector, 2.0 kg M50004 Rev A, manufactured by iRay Imaging (MPN 3180016), shall be used for all units and reprogrammed with the latest version of firmware. The iRay license shall be confirmed to be activated per MWI-255.
The most current software version, 4.0.0-alpha and firmware versions shall be installed on the MX1 system and then replaced with system software v4.0.0 when it is available.  A released software version shall be installed prior to verification testing and documented in the DHRs.
The wireless charger (W1) used for Rev F testing will be electrically representative of W1 Rev D. However, deviations will be made to W1 Rev C configuration to build representative W1 Rev D units. Specifically, the heat sink will be reworked to remove anodized coating in one section so the WTX PCBA may be grounded to it. W1 parts included in Rev D which incorporate mechanical component improvements may not all be included in the W1 tested. All deviations to W1 Rev D-alpha BOM will be noted in the DHR.
MedAI shall build at least two additional Wireless Chargers (W1) to Rev D BOM at MedAI’ office located at 100 Main Street, Suite 700, Springfield, IL 60001 following Good Manufacturing Practices (GMP).
Deviations to W1 Rev D units will be required as three updated parts are not available: M10561 (WTX Handle Base), M10562 (WTX Handle Cover) and M10566 (WTX Main Shaft). These components had minor changes that will not impact verification activities. Specifically, WTX Handle Base has a shortened handle length, WTX Handle Cover has a wider slot width length. And WTX Main Shaft has tolerances added.
The heat sink will be reworked to remove anodized coating in one section so the WTX PCBA may be grounded to it. W1 parts included in Rev D which incorporate mechanical component improvements may not all be included in the W1 tested. All deviations to W1 Rev D-alpha BOM will be noted in the DHR
The wireless charger (W1) used for Rev F testing will be electrically representative of W1 Rev D.
The changes outlined below have been incorporated into the MX1 Rev F build and did not impact standards required for testing. In addition, modifications did not introduce new risks. The identified changes have been assessed to determine impact on verification and validation testing and additional testing has been identified in Tables 1 through 10 based on this assessment.
Successful completion of tests identified shall demonstrate these modifications did not introduce new risks or adversely impact product performance..
A formal risk assessment will be completed prior to Phase 3 Closure Review.
Emitter Changes for MX1 Rev F ( E1 Rev I)
Cassette Changes for MX1 Rev F (C1 Rev J)
Temperature testing will not be repeated as board changes positively impact IR LED components which had the highest temperature in addition to the Detector. The improved IR LED efficiency is expected to reduce relevant temperature and therefore does not warrant additional testing.
Wireless Charger Changes for W1 Rev D
Foot Pedal Changes for MX1 Rev F
The changes outlined below have been incorporated into the MX1 Rev G build and did not impact standards required for testing. In addition, the Rev G modifications did not introduce new risks. The identified changes were assessed in ECR-593 to determine impact on verification and validation testing and no additional testing was required.
The new antennas were demonstrated to function equivalently to the original antennas.
The new hardware did not require additional mechanical testing as the materials and functionality are similar to existing hardware components previously evaluated per IEC 60601-1.
The new adhesive has similar overlap shear strength for ABS plastic as the previously tested adhesive and is also a two part adhesive.
The impact assessment for ES-10019 Rev B and B.1 was included in ECR-510 and ECR-591. Per the assessment no additional verification testing is required.
Emitter Changes for MX1 Rev G ( E1 Rev_K)
Cassette Changes for MX1 Rev G (C1 Rev_L)
A Device History Record (DHR) shall be created for each DV unit, including all components of the system (emitter, cassette, case, foot pedal and wired charger).
Component traceability shall be documented including PN, Rev, and Lot/SNs as applicable.
Manufacturing processes shall be documented. If manufacturing work instructions are not approved at the time of manufacture, units shall be produced under lead engineering supervision and the processes described and documented, including references to any bills of materials and finished assembly drawings.
The engineering verification unit shall be built with work instructions that have been released and then redlined to address issues identified during the previous build.
The DV Units shall be built with released work instructions that incorporate all redlined items from the engineering verification build.
Verification and Validation Unit Testing Priorities, Rev E
The engineering verification unit shall be used to complete the following testing in order of priority:
Collimator Accuracy
SSD Accuracy and Foot Pedal  Distance Connectivity Verification
Essential Performance at High Temperature
Beam Current and Voltage Accuracy
Serial Radiography Maximum Pulse Verification
Design Verification via Demonstration
Requirements for this initial verification shall be identified and reviewed.  All requirements will not be evaluated during engineering verification.
Two Design Verification units, DV22 and DV23, with software version 3.0.0 shall be available for testing at SGS in Duluth, Georgia. SGS testing may begin in parallel with Intertek testing if necessary.  Ideally, testing performed at SGS would be completed prior to initiating Intertek testing. Once SGS testing is complete DV22 shall be used for Wireless Coexistence and FCC testing. Note, DV Units are assigned an internal reference number which correlates to the device serial numbers in QSR-028.
Two Design Verification units, DV24 and DV25, with software version 3.0.0 shall be available for testing at Intertek in Springfield, Illinois. Intertek testing may begin in parallel with SGS testing. Once Intertek testing is complete DV 24 shall be used for the Imaging Study.
One Design Verification unit, DV23, with software version 3.1.0 shall be available for Human Factors testing.
One Design Verification unit, DV21, with software version 3.0.0 shall be available for internal V&V testing.
One Design Verification unit, DV26, with software version 3.0.0 shall be available for internal radiation testing.  When needed this unit will be used in the Imaging Study.
Verification and Validation Unit Testing Priorities, Rev F
The engineering verification unit shall be used to complete the following testing in order of priority:
Collimator Accuracy
SID & SSD Accuracy (IR LED Tracking)
Detector Calibration
Beam Current & Voltage Monitoring
This VVPR will only be performed on Rev F DV units if the Moxtek tube traceability cannot be established and additional Moxtek tubes will be received and built into For Human Use units.
Residual Radiation
As determined in a prior assessment, Residual Radiation testing shall be repeated as a DV unit with an A1 detector (DV23) was used for prior testing. Note, detector version A1 weighs 2.7 kg and detector version A0 weighs 2.0 kg.
One Design Verification unit with software v4.0.0-alpha at a minimum shall be available for testing at Intertek in Duluth, Georgia.  This same unit may be used for testing at SGS in Duluth, Georgia.
Verification and Validation Third Party Testing, Rev E
Intertek will perform basic safety and essential performance testing per 60601-1 2020 at Springfield Intertek facility.  See references for standard edition required for testing.
Intertek will perform x-ray related testing per collateral standard 60601-1-3, and particular standards 60601-2-28, 60601-2-43, 60601-2-54 at Springfield Intertek facility. See references for standard editions required for testing.
Intertek test facility is not accredited to particular standard 60601-2-43.  A note shall be added to the report to indicate that although they are not certified to 60601-2-43 they are certified for similar x-ray testing and have necessary equipment.
MedAI shall develop test protocols and procedures for internal x-ray evaluations and may perform some official testing using released VVPRs as agreed upon with Intertek.
SGS in Duluth, Georgia will perform 60601-1-2 testing.  See references for standard edition required for testing. Ideally this testing would occur prior to all other third party testing to ensure success prior to initiating additional testing, but it may be run in parallel. Two devices should be available for this testing to allow testing in parallel. Anticipated test duration is 2 weeks.
F2 Labs or SGS will perform Wireless Coexistence, RFID Immunity and  testing required for FCC certification.
Draft reports for all third party testing are required for the MedAI regulatory submission. Note, FCC certification is not required for the regulatory submission.
Cybersecurity testing may be performed by a third party test lab if required.
All third party test labs shall be evaluated per the requirements in QSP-007 Rev. G - Supplier Management and approved as suppliers in the Approved Supplier List (QSR-004) prior to testing.
Verification and Validation Third Party Testing, Rev F
Intertek in Duluth, Georgia will perform Dielectric Hipot and Leakage testing per IEC 60601-1 2020 at Springfield Intertek facility to assess changes to the isolation method implemented for Cassette Power and Data ports. See references for standard edition required for testing.
SGS in Duluth, Georgia will perform radiated and conducted emissions testings per IEC 60601-1-2 for the MX1 configuration that performed the worst during MX1 Rev E EMC testing. SGS will be consulted to finalize the selected configuration.
SGS in Duluth, Georgia will perform testing required for FCC and ISED Certification.
FCC Part 18 testing for W1 include radiated emissions for 9 kHz to 1 GHz range, conducted emissions and Maximum Permissible Exposure. Data collected will be evaluated by SGS for both FCC and ISED limits and presented in the same report where possible.
SGS will prepare separate reports for EMC testing and Maximum Permissible Exposure testing.
Permissible Change Testing for the WiFi module is required as the antenna was modified in order to fit into MX1 Emitter and Cassette. The radio module also had an antenna modification, which is a class 2 permissible change. Required testing will be performed at SGS.
Specific Absorption Rate (SAR) testing is required on the WiFi module because the operator and/or patient may be less than 20 cm from the WiFi module and the transmit power is above the SAR exempt threshold.
The radio module is exempt from SAR testing because the transmit power is below the FCC threshold requiring SAR testing. For 900MHz, at 5 mm, the allowed power threshold is 16 mW. The transmitter is 11.6 dBm which is just under 15 mW and therefore the radio meets the SAR exclusion criteria.
Third party test houses will prepare reports to document test results. These reports will be incorporated as attachments in internally released reports for third party testing completed for MX1 Rev F verification.
Verification and Validation Software Testing, Rev E
Software version 3.0.0 will undergo testing defined in Section 6.  Tests performed, and test protocol and report numbers may be modified as required by the Software Requirements Specification.
Software version 3.1.0 will undergo regression testing to evaluate the changes incorporated to 3.0.0. These VVPRs are not included in this document, but will be added when they are available. Additional testing may be required based on test results.
The software changes incorporated into v3.0.0 to create v3.1.0 will include modifications to the MedAI device application and cybersecurity features. We expect these new software features will have no impact on the features implemented in v3.0.0. Current regression testing is based on this expectation. If additional features or software modifications are implemented prior to executing 3.1.0 VVPRs, then the required regression testing shall be reassessed.
Verification and Validation Software Testing, Rev F
Software version 4.0.0 will undergo testing as defined in Section 9. Tests performed, and test protocol and report numbers may be modified as required by the Software Requirements Specification.
Software version 4.1.0 will undergo regression testing to evaluate the changes incorporated into 4.0.0. These VVPRs are not included in this document.  Additional testing may be required based on test results.
Verification and Validation Radiation Testing, Rev E
The following radiation characterization testing shall be performed: Dose Information and Accuracy, Half Value Layer Analysis, Attenuation Equivalent, Isokerma Mapping, Stray Radiation Mapping, Residual Radiation and Radiation Output Reproducibility.
MedAI shall perform radiation testing in-house to characterize MX1 radiation output.
MedAI may perform additional required radiation testing to standards.
Intertek may perform required radiation testing to standards.
Verification and Validation Radiation Testing, Rev F
Radiation characterization testing will not be performed on the 3 Moxtek engineering tubes shipped August 21, 2024 on PO22730 as these tubes may only be incorporated into ‘Not for Human Use’ devices which will not be used for x-ray verification testing or any other qualifications.
Documentation from Moxtek for TUB00999 SNs DEV080, DEV081 and DEV082 is required for component traceability in order to be assembled into ‘Not for Human Use’ Device Verification units.
If additional Moxtek tubes are delivered and will be built into MX1 devices intended for other qualifications or ‘For Human Use’, then radiation testing similar to testing performed for MX1 Rev E will be required if Moxtek cannot provide evidence to demonstrate equivalency to tubes used in Rev E Monoblocks.
Radiation testing that may need to be performed is highlighted in yellow in Tables 2 and 4.
Verification and Validation Radiation Testing, Rev G
Additional verification and validation testing is not required for MX1 Rev G based on impact assessment in Section 4.1.7.
Changes to the Device During or After Testing
Changes to the device during or after verification or validation testing (e.g. software updates, PCB modifications) shall be documented in the Verification and Validation Summary Report, and rationale provided as to whether or not any previous testing needs to be repeated.
Methods
Requirements may be verified by the following methods:
Inspection: Verification by design review of specifications, drawings, schematics, etc.
Demonstration: Verification by performing an action with the device as indicated by the stated requirement. Demonstrations shall not require the use of non-device equipment.
Test: Verification by measurement/data collection using external test equipment (including usability testing).
Analysis: Verification by review of test data and/or software code. Analysis is to be used where the requirement has no obvious input and output.
Traceability
The verification method, results, and pass/fail outcome for all design inputs and user needs shall be documented using the VVAM-P01-004- MX1 Verification & Validation Trace Matrix.
The verification method, results, and pass/fail outcome for all software requirements shall be documented using the MEMO-P01-458 - MX1 Software Traceability Matrix.
Third Parties
Requirements that are verified by third parties shall be documented in official third party reports, reviewed and approved by MedAI, and released into the MedAI quality management system drive.
Sample Sizes
For risks that are considered “Moderate” or “Intolerable” pre-mitigation, mitigations that include testing must use sample sizes in accordance with QSP-026 Statistical Techniques unless the risk is mitigated by one of the following in association with the aforementioned test:
Type Test per an IEC 60601 series standard
Software Test
Usability Test
Engineering Mode
An “Engineering Mode” shall be developed to allow for remote operation of the device during certain Electromagnetic Compatibility and Electrical Safety testing and certain verification tests. Engineering Mode will be verified prior to use in design verification testing.The Engineering Mode shall be incorporated in MX1 System Software releases but only accessible to the MedAI Engineering team. Engineering Mode contains the following unique features:
The device may be remotely operated (via a wifi connection, or via a laptop connected to ethernet) without an operator having to repeatedly actuate the trigger or stand next to the device during testing.
The device may perform a scripted sequence of actions (via a wifi connection, or via a laptop connected to ethernet) which may be terminated at any time remotely via a laptop command or if disconnection between the device and laptop occurs.
NOTE: When performing scripted actions that mimic signals received from the emitter input panel, the device will ignore the physical emitter input panel as this interferes with the script. Control may be switched back to the physical emitter input panel via a scripted command.
Electromagnetic Compatibility, Electrical Safety, and Wireless Testing, MX1 Rev E
Electromagnetic Compatibility, Electrical Safety, and Wireless Testing, MX1 Rev F
Non-Clinical Bench Performance Testing, MX1 Rev E
Non-Clinical Bench Performance Testing, MX1 Rev F
Design Validation, MX1 Rev E
Design Validation, MX1 Rev F
Software Verification and Cybersecurity, Rev E
Software Verification and Cybersecurity, MX1 Rev F
System Level Software Testing, MX1 Rev E
System Level Software Testing, MX1 Rev F
DOCUMENT REVISION HISTORY
Digital Key:
example.com/

### Table 1
| Item | Change | Change Type | Risk Assessment | Repeat V&V |
| --- | --- | --- | --- | --- |
| 1 | Label Change | Regulatory Info Update | None - Impacts Labeling and requires design verification by inspection | ✔ |
| 2 | NVMe Swap | Part Obsolescence | Low - EMC | ✔ |
| 3 | 850 nm Camera Filter | Performance Improvement | Low - Impacts Tracking and Time of Flight Performance need to repeat VVPR | ✔ |
| 4 | HMI PCBA (B.1) | DFM & Reliability Improvements | None | N/A |
| 5 | Collimator PCBA (D) | Performance & Reliability Improvements | Low - EMC | ✔ |
| 6 | Emitter Main PCBA (C) | DFM & Reliability Improvements | Low - EMC | ✔ |
| 7 | Emitter Power Input PCBA (D.1) | DFM & Reliability Improvements | Low - EMC | ✔ |
| 8 | HMI FFC guard | Reliability Improvement | None | N/A |
| 9 | Sub-GHz cowling | Reliability Improvement | None | N/A |
| 10 | WiFi antenna foam spacer | Reliability Improvement | None | N/A |
| 11 | Pediatric puck improvements | Reliability Improvement | None | N/A |
| 12 | Laser Mounts | DFM & Reliability Improvements | None | N/A |
| 13 | Laser Build modifications | DFM & Reliability Improvements | None | N/A |
| 14 | Enclosure pad print change | DFM Improvement | None | N/A |
| 15 | Forward Button modification for tactile feedback | Reliability Improvement | None | N/A |
| 16 | Trigger Harness heat shrink location | DFM Improvement | None | N/A |
| 17 | Coil Motor Shaft Epoxy | Reliability Improvement | None | N/A |

### Table 2
| Item | Change | Change Type | Risk Assessment | Repeat V&V |
| --- | --- | --- | --- | --- |
| 1 | Label Change | Regulatory Info Update | Impacts Labeling and requires design verification by inspection | ✔ |
| 2 | Detector Area Label Change | Label | Impacts Labeling and requires design verification by inspection | ✔ |
| 3 | NVMe Swap | Part Obsolescence | Low - EMC | ✔ |
| 4 | Cassette Main PCBA (C) | Performance & Reliability Improvements | Low - Dielectric and Leakage & EMC | ✔ |
| 5 | Cassette Display PCBA (B) | Reliability Improvements | Low - EMC | ✔ |
| 6 | Tracker Main PCBA (C.1) | Performance & Reliability Improvements | Low | ✔ |
| 7 | Angled Tracking PCBA (B) | Performance & Reliability Improvements | Low | ✔ |
| 8 | FFC Angled Tracking Straps | Reliability Improvement | None | N/A |
| 9 | Heat Pipe geometry updates | Performance Improvement | None | N/A |
| 10 | Silicone light pipes | Reliability Improvement | None - Cleanability already assessed in 3P-P01-24 Attachment 1 | N/A |

### Table 3
| Item | Change | Change Type | Risk Assessment | Repeat V&V |
| --- | --- | --- | --- | --- |
| 1 | Label Change | Regulatory Info Update | Impacts Labeling and requires design verification by inspection | ✔ |
| 2 | Enclosure pad printing | DFM Improvement | None | N/A |
| 3 | Injection Molded Dog Bone part | DFM Improvement | None | N/A |
| 4 | Injection molded IGUS bushings | DFM Improvement | None | N/A |
| 5 | WTX PCBA (D.1) | Performance & Reliability Improvements | Low - EMC | ✔ |
| 6 | Heatsink modification for grounding | Performance Improvement | Low - EMC | ✔ |
| 7 | Part modifications: Plastic cover slot width length widened, Shortened handle length, tolerances added to main shaft diameter | DFM Improvements | None | N/A |

### Table 4
| Item | Change | Change Type | Risk Assessment | Repeat V&V |
| --- | --- | --- | --- | --- |
| 1 | Foot Pedal PCBA (B) | Performance & Reliability Improvements | Low | ✔ |

### Table 5
| Item | Change | Change Type | Risk Assessment | Repeat V&V |
| --- | --- | --- | --- | --- |
| 1 | Sub-GHz antenna | FCC Compliance | None - equivalent functionality | N/A |
| 2 | ES-10019 B.1 | Performance Improvement, DFM | None | N/A |
| 3 | New Acrylic Adhesive (M51187) | Performance Improvement | None | N/A |
| 4 | Update MS-10134 | FCC Compliance | None | N/A |
| 5 | Update MS-10136 | FCC Compliance | None | N/A |

### Table 6
| Item | Change | Change Type | Risk Assessment | Repeat V&V |
| --- | --- | --- | --- | --- |
| 1 | Sub-GHz antenna | FCC Compliance | None - functionally equivalent | N/A |
| 2 | Update M11085 | Inspection | None | N/A |

### Table 7
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Medical electrical equipment - Part 1: General requirements for basic safety and essential performance | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-1: 2020-08 Ed. 3.2 | Intertek Testing Services | x |  | DV24, DV25 |
| Medical electrical equipment - Part 1-2: General requirements for basic safety and essential performance - Collateral Standard: Electromagnetic disturbances - Requirements and tests | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1.Prior to Electromagnetic Compatibility testing, MedAI shall provide completed Intertek test plan input document to the test lab | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-1-2: 2020-09 Ed. 4.1 | SGS | x |  | DV21, DV22 |
| Medical electrical equipment – Part 1-3: General requirements for basic safety and essential performance – Collateral Standard: Radiation protection in diagnostic X-ray equipment | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-1-3: 2021-01 Ed. 2.2 | Intertek Testing Services | x |  | DV24, DV25 |
| Medical electrical equipment - Part 1-6: General requirements for basic safety and essential performance - Collateral standard: Usability | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. MedAI will perform Human Factors summative testing and write a summary report.  Report will be sent to Intertek for their approval. | Document Review Only | IEC 60601-1-6: 2020-07 Ed. 3.2 | Intertek Testing Services |  | x | DV23 |
| Medical Electrical Equipment - Part 2-28: Particular Requirements For The Basic Safety And Essential Performance Of X-Ray Tube Assemblies For Medical Diagnosis | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-2-28: 2017-06 Ed. 3.0 | Intertek Testing Services | x |  | DV24, DV25 |
| Medical Electrical Equipment Part 2-43: Particular requirements for the safety of X-ray equipment for interventional procedures | The objective of this study is to verify the performance and associated documentation of the device battery packs is in compliance with the applicable standard. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | IEC 60601-2-43: 2019-10 Ed. 2.2 | Intertek Testing Services | x |  | DV24, DV25 |
| Medical electrical equipment – Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-2-54: 2022-09 Ed. 2.0 | Intertek Testing Services | x |  | DV24, DV25 |
| American National Standard For Evaluation Of Wireless Coexistence | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standards. Notes: 1. MedAI shall provide a signed F2 labs test plan input document to the test lab prior to testing. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Sample size per ANSI IEEE C63.27-2017 | ANSI IEEE C63.27-2021 | F2 Labs or SGS | x |  | DV22 |
| Medical electrical equipment and system electromagnetic immunity test for exposure to radio frequency identification readers - an aim standard. (General II (ES/EMC)) | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. MedAI shall provide a signed F2 labs test plan input document to the test lab prior to testing. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Sample size per AIM Standard 7351731. | AIM Standard 7351731 | F2 Labs or SGS | x |  | DV22 |

### Table 8
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev F 4.0.0 | Rev F 4.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Medical electrical equipment - Part 1: General requirements for basic safety and essential performance | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation Leakage and Dielectric testing will be performed on the cassette and interfaces with K1 and E1 due to changes to the cassette isolation barrier between isolated and non-isolated sides on Cassette Main PCBA Temperature testing will not be performed as changes are expected to positively impact temperature results. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-1: 2020-08 Ed. 3.2 | Intertek Testing Services | x |  | DV28 |
| Medical electrical equipment - Part 1-2: General requirements for basic safety and essential performance - Collateral Standard: Electromagnetic disturbances - Requirements and tests | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1.Prior to Electromagnetic Compatibility testing, MedAI shall provide completed Intertek test plan input document to the test lab Worst case configurations of MX1 system shall be evaluated for radiated and conducted emissions. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-1-2: 2020-09 Ed. 4.1 | SGS | x |  | DV28 |
| Medical electrical equipment – Part 1-3: General requirements for basic safety and essential performance – Collateral Standard: Radiation protection in diagnostic X-ray equipment | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation Subset of standard tested as required for new tube | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-1-3: 2021-01 Ed. 2.2 | Intertek Testing Services | x |  |  |
| Medical electrical equipment - Part 1-6: General requirements for basic safety and essential performance - Collateral standard: Usability | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. MedAI will perform Human Factors summative testing and write a summary report.  Report will be sent to Intertek for their approval. | Document Review Only | IEC 60601-1-6: 2020-07 Ed. 3.2 | Intertek Testing Services |  | x | TBD |
| Medical Electrical Equipment - Part 2-28: Particular Requirements For The Basic Safety And Essential Performance Of X-Ray Tube Assemblies For Medical Diagnosis | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation Subset of standard tested as required for new tube | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-2-28: 2017-06 Ed. 3.0 | Intertek Testing Services | x |  |  |
| Medical Electrical Equipment Part 2-43: Particular requirements for the safety of X-ray equipment for interventional procedures | The objective of this study is to verify the performance and associated documentation of the device battery packs is in compliance with the applicable standard. Subset of standard tested as required for new tube | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | IEC 60601-2-43: 2019-10 Ed. 2.2 | Intertek Testing Services | x |  |  |
| Medical electrical equipment – Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy | The objective of this study is to verify the performance and associated documentation of the device is in compliance with the applicable standard. Notes: 1. Prior to evaluation, MedAI shall provide required documents to the test lab per MEMO-P01-440 - Required Items for Intertek Safety Evaluation Subset of standard tested as required for new tube | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type tests per IEC 60601-1:2005+Amd 1:2012 section 5.2 | IEC 60601-2-54: 2022-09 Ed. 2.0 | Intertek Testing Services | x |  |  |

### Table 9
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | Rev E 3.2.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UN Manual of Tests and Criteria (UN 38.3) 7th revised edition for a secondary battery pack | The objective of this study is to verify the performance and associated documentation of the device battery packs is in compliance with the applicable standard. Leverage previous testing as Energy Assurance indicated changing firmware and coulomb counter training did not require retest. Give pictures of updated design to Energy Assurance so they can provide updated reports. Original Report 3P-P01-19 + justification | MS-10010 Emitter Battery Pack:  Includes 8 batteries for T1-T5 Testing and 8 batteries for T7 testing. MS-10083 Cassette Battery Pack:  Includes 8 batteries for T1-T5 Testing and 8 batteries for T7 testing. Sample sizes per UN Manual of Tests and Criteria section 38.3.3 | UN 38.3: 7th edition | Energy Assurance | N/A | N/A | N/A | N/A |
| Secondary cells and batteries containing alkaline or other non-acid electrolytes – Safety requirements for portable sealed secondary cells, and for batteries made from them, for use in portable applications –Part 2: Lithium systems | The objective of this study is to verify the performance and associated documentation of the device battery packs is in compliance with the applicable standard. Memo: Leverage previous testing as Energy Assurance  indicated changing firmware and coulomb counter training did not require retest. Give pictures of updated design to Energy Assurance so they can provide updated reports. Original Report 3P-P01-19 + justification | MS-10010 Emitter Battery Pack: Includes 17 closed and 5 open batteries each for the following tests: 3 vibration 3 shock 3 drop 5 overcharge 3 case stress 5 short circuit (open) MS-10083 Cassette Battery Pack: Includes 17 closed and 5 open batteries each for the following tests: 3 vibration 3 shock 3 drop 5 overcharge 3 case stress 5 short circuit (open) Sample sizes per IEC 62133-2:2017 Table 1 – Sample size for type tests | IEC 62133-2:2017 | Energy Assurance | N/A | N/A | N/A | N/A |
| Battery Accuracy Test | The objective of this study is to verify the emitter and cassette battery packs are able to estimate the battery capacity to within 5% per PRD5.28 | n=10 | N/A | MedAI | N/A | N/A | N/A | N/A |
| Tablet DICOM Calibration Testing | The objective of this study is to verify the use of the Samsung Galaxy S8+ tablet with accompanying MedAI Mobile Device App software for DICOM PS3.14 compliant diagnostic viewing of images. Memo: Reference existing report: 3P-P01-21 in an updated justification and include new Samsung results. Perform internal VVPR to demonstrate inputs/outputs for 1.0.0 are the same as 3.0.0. *Memo was not written at the time of submission. Reprioritized. | n=1 T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Software dependent test | IEC 62563-1: 2021 DICOM PS3.14 | Image Quality Labs may be utilized to perform testing Justification/ Internal VVPR MedAI |  | x* | x | N/A |
| Packaged-Products for Parcel Delivery System Shipment 70 kg (150 lb) or Less | The objective of this study is to perform a visual assessment and essential performance testing before and after ISTA 3A conditioning to ensure the device is not negatively impacted by normal transit. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher n = 1 F1 Foot Pedal SW 3.0.0 or higher One sample (each) is required for this test procedure per ISTA 3A 2018. Additional Sample Size Justification: All units in production are inspected to meet expected performance specifications which are sufficiently rigorous to highlight or detect changes in performance due to application of stresses which challenge the integrity of the device. All units perform a start-up check every time the unit turns ON to verify performance. | ISTA 3A 2018: Packaged Products for Parcel Delivery System Shipments 70 kg (150 lbs) or less (standard, small, flat or elongated) | MedAI ATS (Applied Technical Services) Shall perform and provide the ISTA 3A section of report to be included as an appendix | x |  | N/A | DV22 |
| Beam Current and Voltage Monitoring Accuracy at Room Temperature | The objective of this study is to verify the ability of the device to accurately measure the x-ray beam current and voltage, which is used for faulting monitoring at room temperature.   Essential Performance will be performed at room temperature. | n = 1 DV Unit SW 3.0.0 or higher Software dependent test. | Not applicable VVPR-P01-160 | MedAI | x |  | N/A | DV23 |
| Serial Radiography Max Pulses Verification | The objective of this study is to verify the max pulse count and accuracy of loading factors during the maximum duration of exposure (20 second) in Serial Radiography Mode and Fluoroscopy Mode of the device. | n = 1 DV Unit and T1 Tablet with MedAI Device App (A  PP) SW 3.0.0  or higher Software dependent test | Not applicable VVPR-P01-164 | MedAI | x |  | N/A | DV23 |
| Focal Spot Size Measurement | The objective of this study is to verify the size of the x-ray tube anode focal spot size for disclosure in the IFU and image quality reference. | n=1 | 603368 VVPR-P01-150 | MedAI | x |  | N/A | Various |
| Dosimetric Indications | The objective of this study is to verify the radiation output and dose accuracy at various device configurations including SID, loading factors, pucks and pediatric filters. In addition, this study will recalculate and re-validate the dose equation. Reference MEMO-P01-503. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.5.2.4.5.102 60601-2-43 203.5.2.4.5.102 MEMO-P01-672 | MedAI | x |  | N/A | DV26 |
| Half Value Layer | The objective of this study is to determine the half value layer of the beam in millimeters of aluminum. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-1-3 7.4, 7.5 60601-2-54 203.7.1 VVPR-P01-151 | MedAI | x |  | N/A | DV26 |
| Attenuation Equivalent | The objective of this study is to determine the attenuation factor of parts between the patient and the detector in millimeters of aluminum. Note: narrow beam setup equipment shall be brought by MedAI to Intertek testing | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.10.101 VVPR-P01-172 | MedAI | x |  | N/A | DV26 |
| Isokerma Map | The objective of this study is to verify the Scatter and Leakage Radiation of the device as described in IEC 60601-1-3 and 60601-2-43. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-43 203.13.4 203.13.6 Annex BB Protocol in VVPR-P01-105 | West Physics | x |  | N/A | DV26 |
| Stray Radiation Map | The objective of this study is to verify the Scatter and Leakage Radiation of the device as described in IEC 60601-1-3 and 60601-2-43. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-1-3 13.6 VVPR-P01-15405 | West Physics | x |  | N/A | DV26 |
| Residual Radiation | The objective of this study is to ensure residual radiation does not occur above allowable limits. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.11.101 203.11.102 VVPR-P01-155 | MedAI | x |  | N/A | DV26 |
| Radiation Output Reproducibility | The objective of this study is to verify that the x-ray output air kerma remains consistent through successive emissions, for all loading factor combinations | n=1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.6.3.2.103.1 203.6.3.2.103.2 VVPR-P01-156 | MedAI & Intertek | x |  | N/A | DV26 |
| Radiography Linearity and Constancy | The objective of this study is to verify that the x-ray output air kerma remains linear and constant through successive emissions, for all loading factor combinations | n=1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.6.3.2.103.2 VVPR-P01-157 | MedAI & Intertek | x |  | N/A | DV26 |
| Battery Life Verification | The objective of this study is to verify the battery life of the device Emitter and Cassette from fully charged. Testing will use battery only with correct electronic load | n = 3 DV Unit, and T1 Tablet with MedAI Device App (APP) SW 3.0.0  or higher This is a controlled variable-measurement test. Variability is expected to be low due to well-established tolerances and performance characteristics of the battery cells. Only 3 DV Unit samples will be available for this test. These are representative samples of the item being tested. AVED Battery pack required | Not applicable Protocol in VVPR-P01-162 | MedAI | x |  | N/A | N/A |
| Cleaning For Expected Service Life | The objective of this study is to verify that the cleaning protocol specified in the instructions for use is appropriate for the expected service life of the device, and does not degrade the device safety nor essential performance features. | n = 1 DV Unit, and F1 Foot Pedal SW 3.0.0 or higher | FDA Guidance: Reprocessing Medical Devices in Health Care Settings: Validation Methods and Labeling, June 2017. VVPR-P01-173 | MedAI | x |  | N/A | DV21 |
| Cleaning & Disinfection | The objective of this study is to verify the MX1 Portable X-ray System may adopt the validated cleaning & disinfection protocol used for Imager Medical Imaging System, P00. Notes: 1. MedAI shall provide enclosures of the MX1 System (all externally exposed components), as well as material datasheets to perform a desktop assessment. | n = 1 E1 Emitter, C1 Cassette, W1 and F1 (relies on pre-cert from mfr) Enclosures (shells with all externally exposed components), H1 wired charger | ISO 17664-2:2021 ANSI/AAMI ST98:2022 AAMI TIR 12:2020 | Nelson Labs | N/A | N/A | N/A | N/A |
| SSD & SID Accuracy Under Challenge Condition Verification | The purpose of this test protocol is to verify the MX1 Portable X-ray System meets the Source to Skin Distance (SSD) accuracy specifications and foot pedal operating distance specifications established by MedAI. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | Not applicable VVPR-P01-168 | MedAI | x |  | N/A | DV23 |
| Foot Pedal Operating Distance | The purpose of this test protocol is to verify the MX1 Portable X-ray System meets foot pedal operating distance specifications established by MedAI. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | VVPR-P01-158 | MedAI | x |  | x* | DV23 *DV |
| Device Weight Verification | The objective of this study is to verify the MX1 Portable X-ray System and accessories meet the weight specifications established by MedAI. | n = 1 DV Unit (including P1 case), F1 Foot Pedal | Not applicable VVPR-P01-159 | MedAI | x |  | N/A | DV23 |
| Design Verification Via Demonstration | The objective of this study is to verify select MX1 product specifications via demonstration. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 2.1.0 or higher | Not applicable VVPR-P01-166 | MedAI |  | x | N/A | DV23 |
| Design Verification Via Inspection | The objective of this study is to verify select MX1 product specifications via inspection (documentation review). | Documentation review only. No product needed. | Not applicable VVPR-P01-165 | MedAI |  | x | N/A | DV23 |
| Foreign Object Testing | Demonstrate wireless charger does not transfer heat to foreign objects in proximity | n =1 | IEC 62368 VVPR-P01-148 | MedAI | x |  | N/A | DV23 |

### Table 10
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev F 4.0.0 | Rev F 4.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tablet DICOM Calibration Testing | The objective of this study is to verify the use of the Samsung Galaxy S8+ tablet with accompanying MedAI Mobile Device App software for DICOM PS3.14 compliant diagnostic viewing of images. Memo: Reference existing report: 3P-P01-21 in an updated justification and include new Samsung results. Perform internal VVPR to demonstrate inputs/outputs for 1.0.0 are the same as 3.0.0. | n=1 T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Software dependent test | IEC 62563-1: 2021 DICOM PS3.14 | Image Quality Labs Justification/ Internal VVPR MedAI | N/A | N/A | N/A |
| Beam Current and Voltage Monitoring Accuracy at Room Temperature | The objective of this study is to verify the ability of the device to accurately measure the x-ray beam current and voltage, which is used for faulting monitoring at room temperature.   Essential Performance will be performed at room temperature. | n = 1 DV Unit SW 3.0.0 or higher Software dependent test. | Not applicable Update VVPR-P01-160 | MedAI | x |  | DV |
| Serial Radiography Max Pulses Verification | The objective of this study is to verify the max pulse count and accuracy of loading factors during the maximum duration of exposure (20 second) in Serial Radiography Mode and Fluoroscopy Mode of the device. | n = 1 DV Unit and T1 Tablet with MedAI Device App (A  PP) SW 3.0.0  or higher Software dependent test | Not applicable Update VVPR-P01-164 | MedAI | x |  | DV |
| Focal Spot Size Measurement | The objective of this study is to verify the size of the x-ray tube anode focal spot size for disclosure in the IFU and image quality reference. | n=1 | 603368 Update VVPR-P01-150 | MedAI | x |  | Various |
| Dosimetric Indications | The objective of this study is to verify the radiation output and dose accuracy at various device configurations including SID, loading factors, pucks and pediatric filters. In addition, this study will recalculate and re-validate the dose equation. Reference MEMO-P01-503. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.5.2.4.5.102 60601-2-43 203.5.2.4.5.102 Update MEMO-P01-672 | MedAI | x |  |  |
| Half Value Layer | The objective of this study is to determine the half value layer of the beam in millimeters of aluminum. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-1-3 7.4, 7.5 60601-2-54 203.7.1 Update VVPR-P01-151 | MedAI | x |  |  |
| Attenuation Equivalent | The objective of this study is to determine the attenuation factor of parts between the patient and the detector in millimeters of aluminum. Note: narrow beam setup equipment shall be brought by MedAI to Intertek testing | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.10.101 Update VVPR-P01-172 | MedAI | x |  |  |
| Isokerma Map | The objective of this study is to verify the Scatter and Leakage Radiation of the device as described in IEC 60601-1-3 and 60601-2-43. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-43 203.13.4 203.13.6 Annex BB Update VVPR-P01-154 or include procedure in WP report | West Physics | x |  |  |
| Stray Radiation Map | The objective of this study is to verify the Scatter and Leakage Radiation of the device as described in IEC 60601-1-3 and 60601-2-43. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-1-3 13.6 Update VVPR-P01-154 or include procedure in WP report | West Physics | x |  |  |
| Residual Radiation | The objective of this study is to ensure residual radiation does not occur above allowable limits. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.11.101 203.11.102 Update VVPR-P01-155 | MedAI | x |  | TBD |
| Radiation Output Reproducibility | The objective of this study is to verify that the x-ray output air kerma remains consistent through successive emissions, for all loading factor combinations | n=1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.6.3.2.103.1 203.6.3.2.103.2 Update VVPR-P01-156 | MedAI & Intertek | x |  |  |
| Radiography Linearity and Constancy | The objective of this study is to verify that the x-ray output air kerma remains linear and constant through successive emissions, for all loading factor combinations | n=1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.6.3.2.103.2 Update VVPR-P01-157 | MedAI & Intertek | x |  |  |
| SSD & SID Accuracy Under Challenge Conditions Verification | The purpose of this test protocol is to verify the MX1 Portable X-ray System meets the Source to Skin Distance (SSD) accuracy specifications and foot pedal operating distance specifications established by MedAI. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | Not applicable Update VVPR-P01-168 | MedAI | x |  | TBD |
| Foot Pedal Operating Distance | The purpose of this test protocol is to verify the MX1 Portable X-ray System meets foot pedal operating distance specifications established by MedAI. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | VVPR-P01-158 | MedAI | x |  | TBD |
| Device Weight Verification | The objective of this study is to verify the MX1 Portable X-ray System and accessories meet the weight specifications established by MedAI. | n = 1 DV Unit (including P1 case), F1 Foot Pedal | Not applicable Update VVPR-P01-159 | MedAI | N/A | N/A | TBD |
| Design Verification Via Demonstration | The objective of this study is to verify select MX1 product specifications via demonstration. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 2.1.0 or higher | Not applicable Update VVPR-P01-166 | MedAI | N/A | N/A | N/A |
| Design Verification Via Inspection | The objective of this study is to verify select MX1 product specifications via inspection (documentation review). *A subset of testing will be performed to assess labeling changes only. | Documentation review only. No product needed. | Not applicable Update VVPR-P01-165 | MedAI | N/A | N/A | x* |

### Table 11
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | Rev E 3.2.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Technique Factor Clinical Input Study | The objective of this study is to determine loading factor recommendation for the device by having x-ray images compared by practitioners at various loading factor combinations. Defines technique factors per anatomy GUI NOT REQUIRED | n = 1 SW 3.0.0 or higher At least one board-certified practitioner with an active medical license experienced in imaging related to the device's intended use will be participating in this image comparison study. MedAI will collect radiographs for phantom anatomies/orientations at every available technique. Note: 1. Images shall be viewed on a DICOM compliant screen. | Not applicable | MedAI | x |  | N/A | DV24 |
| Image Quality Study | The objective of this study is to test the image quality of x-ray captures produced by the MX1 Portable X-ray System by comparing them to equivalent captures taken by the MinXray TR90BH Diagnostic X-Ray System (Reference Device: K182207). Uses Technique Factors determined by above test | At least three board-certified practitioners with active medical licenses experienced in imaging related to the device's intended use, at least two of whom will be orthopedic surgeons and at least one of whom will be a radiologist, will be participating in the image comparison study. MedAI will collect 10 radiographs of phantom extremities and hips with both the MX1 and TR90BH devices. | Not applicable Update VVPR-P01-080 VVPR-P01-120 | MedAI @ cadaver lab | x |  | N/A | DV24 |
| Summative Usability Testing | The objective of this study is to identify if the operator can interpret the Instructions for Use (IFU) and operate the device with minimum use-related hazards as well as to observe the operator interactions with the MX1 Portable X-ray System. The summative evaluation shall demonstrate and provide evidence that the MX1 Portable X-ray System can be used safely and effectively. Note: Emitter, Cassette, Table, UI and Foot Pedal will be repeated, including repeating ergonomics and new cassette handle.  Interoperability between K1 & MX1 shall also be evaluated with K1 including W1. | n = 1 SW 3.2.0  or higher K1, W1 n = 15 physicians, 15 non-physicians Use environments include hospital intensive care unit/Hospital Emergency Department/clinic/urgent care facility | ● IEC 60601-1-6: 2013 ● IEC 62366-1: 2015 ● IEC 62366-2: 2020 Update VVPR-P01-081 | MedAI *IDE Group |  | x | x* | DV21 *DV25 *DV26 |
| Summative Usability Testing K1 | VVPR-P01-127 - K1 Usability Summative Evaluation Protocol and Report | N/A | IEC 60601-1-6: 2013 IEC 62366-1: 2015 | MedAI | N/A | N/A | N/A | N/A |
| Summative Usability Testing W1 | VVPR-P01-140 - W1 Usability Summative Evaluation Protocol and Report | N/A | IEC 60601-1-6: 2013 IEC 62366-1: 2015 | MedAI | N/A | N/A | N/A | N/A |

### Table 12
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev F 4.0.0 | Rev F 4.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Summative Usability Testing | The objective of this study is to identify if the operator can interpret the Instructions for Use (IFU) and operate the device with minimum use-related hazards as well as to observe the operator interactions with the MX1 Portable X-ray System. The summative evaluation shall demonstrate and provide evidence that the MX1 Portable X-ray System can be used safely and effectively. Note: Emitter, Cassette, Table, UI and Foot Pedal will be repeated, including repeating ergonomics and new cassette handle.  Interoperability between K1 & MX1 shall also be evaluated with K1 including W1. | n = 1 SW 4.1.0  or higher K1, W1 n = 15 physicians, 15 non-physicians Use environments include hospital intensive care unit/Hospital Emergency Department/clinic/urgent care facility | ● IEC 60601-1-6: 2013 ● IEC 62366-1: 2015 ● IEC 62366-2: 2020 VVPR needed | IDE Group MedAI |  | x | TBD |

### Table 13
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) and Test Method Summary | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | Rev E 3.2.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Software System Cybersecurity Penetration Testing | The objective of this study is to evaluate the software system security and exploit flaws while reporting the findings back to MedAI. Note: Test focus is to perform attacks similar to those of a malicious remote attacker and to attempt to infiltrate the system, alter or remove data, or cause degradation of system performance. | n = 1 DV Unit SW 3.1.0 or higher Software dependent test | Not applicable | MedAI *Contractor: Max Sterling |  | x | x* | DV23 *DV |

### Table 14
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) and Test Method Summary | Testing and Report Completed by | Rev F 4.0.0 | Rev F 4.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Software System Cybersecurity Penetration Testing | The objective of this study is to evaluate the software system security and exploit flaws while reporting the findings back to MedAI. Note: Test focus is to perform attacks similar to those of a malicious remote attacker and to attempt to infiltrate the system, alter or remove data, or cause degradation of system performance. | n = 1 DV Unit SW 4.1.0 or higher Software dependent test | Not applicable | MedAI | N/A | N/A | N/A |

### Table 15
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) and Test Method Summary | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | Rev E 3.2.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Debug and Release Modes & System Configuration | Evaluate SS development modes - Debug and Release, System logging and System Configuration | n = 1 DV Unit | VVPR-P01-175 | MedAI | x |  | x* | DV23 *DV |
| Power On/Off & Power States | Evaluate Power On/Startup, Power Off and Power States including Idle *Subset of testing to be performed since only change was to the power handler (Power On & Off and Charging) | n = 1 DV Unit | VVPR-P01-176 | MedAI | x |  | x* | DV23 *DV |
| Technique Selection and Display | Evaluate Imaging Modes & Technique Factor Selection/Display | n = 1 DV Unit | VVPR-P01-177 | MedAI | x |  | N/A | DV23 |
| Photographic Acquisition | Evaluate Photographic Acquisition | n = 1 DV Unit | VVPR-P01-183 | MedAI | x |  | N/A | DV23 |
| Foot Pedal Integration | Evaluate Foot Ped Integration | n = 1 DV Unit | VVPR-P01-184 | MedAI | x |  | x* | DV23 *DV |
| Batteries and Wired/Wireless Charging | Evaluate Batteries and BMS Integration & Wired charging (not accuracy of SoC or wired charging interlock check) | n = 1 DV Unit | VVPR-P01-185 | MedAI | x |  | N/A | DV23 |
| Safety Interlocks | Evaluate Safe State & Safety Interlocks (Point to Faults VVPR for the fault interlock) *Subset of testing to be performed since IMU changed. IMU (fluoroscopy) related interlocks testing only. | n = 1 DV Unit | VVPR-P01-178 | MedAI | x | x | x* | DV23 *DV |
| Critical Faults | Evaluate Faults at system level (Point to Safe State VVPR in description) | n = 1 DV Unit Requires final hardware and firmware - multidisciplinary effort | VVPR-P01-181 | MedAI | x |  | N/A | DV23 |
| Collimator Accuracy | Evaluate Collimation Accuracy | n = 1 DV Unit | IEC 60601-2-54: 2022 Section 203.8.5.3 “Correspondence between X-RAY FIELD and EFFECTIVE IMAGE RECEPTION AREA” VVPR-P01-189 | MedAI | x |  | x | DV23 |
| Radiographic and Radioscopic Acquisition | Evaluate Tracking/Positioning Display including VF, MI LEDs and lasers, Collimation Display and Radiographic Acquisition including trigger behavior, Single, DDR and Fluoroscopy | n = 1 DV Unit | VVPR-P01-179 | MedAI | x |  | N/A | DV23 |
| MedAI Device App | Evaluate MedAI Device App | n = 1 DV Unit Retest Interlocks only for 3.1.0 | VVPR-P01-186 | MedAI | x | x | x* | DV23 *DV |
| X-ray Timing | Evaluate x-ray pulse and detector synchronization | n=1 DV Unit | VVPR-P01-180 | MedAI |  |  | N/A |  |
| Voltage and Temperature Monitoring Accuracy | Evaluate accuracy of temperature and voltage readings | n=1 DV Unit | VVPR-P01-182 | MedAI |  |  | N/A |  |
| Viewfinder Assessment | Evaluate Tracking/Positioning, Technique Factor Settings, and Acquisition Mode Indications on the ViewFinder. | n=1 DV Unit | VVPR Needed | MedAI |  |  | x | TBD |
| Detector Calibration | Verify detector calibration method | N=1 DV Unit | VVPR Needed | MedAI |  |  | N/A |  |

### Table 16
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) and Test Method Summary | Testing and Report Completed by | Rev F 4.0.0 | Rev F 4.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Foot Pedal Integration | Evaluate Foot Ped Integration | n = 1 DV Unit | VVPR-P01-184 | MedAI |  | x |  |
| Collimator Accuracy | Evaluate Collimation Accuracy | n = 1 DV Unit | IEC 60601-2-54: 2022 Section 203.8.5.3 “Correspondence between X-RAY FIELD and EFFECTIVE IMAGE RECEPTION AREA” VVPR-P01-189 | MedAI | x |  | TBD |
| Detector Calibration | Verify Detector Calibration Method | n=1 DV Unit | VVPR Needed | MedAI |  | x | TBD |
| Radiographic and Radioscopic Acquisition | Evaluate Tracking/Positioning Display including VF, MI LEDs and lasers, Collimation Display and Radiographic Acquisition including trigger behavior, Single, DDR and Fluoroscopy | n = 1 DV Unit | Update VVPR-P01-107 | MedAI | x |  | TBD |
| MedAI Device App | Evaluate MedAI Device App | n = 1 DV Unit Retest Interlocks only for 3.1.0 | Update VVPR-P01-114 | MedAI | N/A | x | TBD |

### Table 17
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 09 Apr 2024 | 24-134 |
| B | Update plan to include MX1 Rev E s/w v3.2.0 additional testing. Add VVPRs for Rev E testing completed. Update plan to include MX1 Rev F device verification unit build information, change assessment, and additional testing required for s/w v4.0.0 and v4.1.0. | Refer to ECR-543 |  |  |
| C | Update plan to include MX1 Rev G changes and impact assessment. No additional V&V required. | Refer to ECR-723 |  |  |
