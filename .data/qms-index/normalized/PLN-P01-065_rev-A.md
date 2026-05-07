# PLN-P01-065 Rev A: MX1 Verification and Validation Plan

## Metadata
- Document ID: PLN-P01-065
- Revision: A
- Prefix: PLN
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: unknown
- Source filename: PLN-P01-065 - MX1 Verification and Validation Plan_A-Obsolete.docx
- Source path: Example QMS - MedAI/PLN-P01-065 - MX1 Verification and Validation Plan_A-Obsolete.docx
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
A Device History Record (DHR) shall be created for each DV unit, including all components of the system (emitter, cassette, case, foot pedal and wired charger).
Component traceability shall be documented including PN, Rev, and Lot/SNs as applicable.
Manufacturing processes shall be documented. If manufacturing work instructions are not approved at the time of manufacture, units shall be produced under lead engineering supervision and the processes described and documented, including references to any bills of materials and finished assembly drawings.
The engineering verification unit shall be built with work instructions that have been released and then redlined to address issues identified during the previous build.
The DV Units shall be built with released work instructions that incorporate all redlined items from the engineering verification build.
Verification and Validation Unit Testing Priorities
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
Verification and Validation Third Party Testing
Intertek will perform basic safety and essential performance testing per 60601-1 2020 at Springfield Intertek facility.  See references for standard edition required for testing.
Intertek will perform x-ray related testing per collateral standard 60601-1-3, and particular standards 60601-2-28, 60601-2-43, 60601-2-54 at Springfield Intertek facility. See references for standard editions required for testing.
Intertek test facility is not accredited to particular standard 60601-2-43.  A note shall be added to the report to indicate that although they are not certified to 60601-2-43 they are certified for similar x-ray testing and have necessary equipment.
MedAI shall develop test protocols and procedures for internal x-ray evaluations and may perform some official testing using released VVPRs as agreed upon with Intertek.
SGS in Duluth, Georgia will perform 60601-1-2 testing.  See references for standard edition required for testing. Ideally this testing would occur prior to all other third party testing to ensure success prior to initiating additional testing, but it may be run in parallel. Two devices should be available for this testing to allow testing in parallel. Anticipated test duration is 2 weeks.
F2 Labs or SGS will perform Wireless Coexistence, RFID Immunity and  testing required for FCC certification.
Draft reports for all third party testing are required for the MedAI regulatory submission. Note, FCC certification is not required for the regulatory submission.
Cybersecurity testing may be performed by a third party test lab if required.
All third party test labs shall be evaluated per the requirements in QSP-007 Rev. G - Supplier Management and approved as suppliers in the Approved Supplier List (QSR-004) prior to testing.
Verification and Validation Software Testing
Software version 3.0.0 will undergo testing defined in Section 6.  Tests performed, and test protocol and report numbers may be modified as required by the Software Requirements Specification.
Software version 3.1.0 will undergo regression testing to evaluate the changes incorporated to 3.0.0. These VVPRs are not included in this document, but will be added when they are available. Additional testing may be required based on test results.
The software changes incorporated into v3.0.0 to create v3.1.0 will include modifications to the MedAI device application and cybersecurity features. We expect these new software features will have no impact on the features implemented in v3.0.0. Current regression testing is based on this expectation. If additional features or software modifications are implemented prior to executing 3.1.0 VVPRs, then the required regression testing shall be reassessed.
Verification and Validation Radiation Testing
The following radiation characterization testing shall be performed: Dose Information and Accuracy, Half Value Layer Analysis, Attenuation Equivalent, Isokerma Mapping, Stray Radiation Mapping, Residual Radiation and Radiation Output Reproducibility.
MedAI shall perform radiation testing in-house to characterize MX1 radiation output.
MedAI may perform additional required radiation testing to standards.
Intertek may perform required radiation testing to standards.
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
Electromagnetic Compatibility, Electrical Safety, and Wireless Testing
Non-Clinical Bench Performance Testing
Design Validation
Software Verification and Cybersecurity
System Level Software Testing
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
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

### Table 2
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UN Manual of Tests and Criteria (UN 38.3) 7th revised edition for a secondary battery pack | The objective of this study is to verify the performance and associated documentation of the device battery packs is in compliance with the applicable standard. Leverage previous testing as Energy Assurance indicated changing firmware and coulomb counter training did not require retest. Give pictures of updated design to Energy Assurance so they can provide updated reports. Original Report 3P-P01-19 + justification | MS-10010 Emitter Battery Pack:  Includes 8 batteries for T1-T5 Testing and 8 batteries for T7 testing. MS-10083 Cassette Battery Pack:  Includes 8 batteries for T1-T5 Testing and 8 batteries for T7 testing. Sample sizes per UN Manual of Tests and Criteria section 38.3.3 | UN 38.3: 7th edition | Energy Assurance | N/A | N/A | N/A |
| Secondary cells and batteries containing alkaline or other non-acid electrolytes – Safety requirements for portable sealed secondary cells, and for batteries made from them, for use in portable applications –Part 2: Lithium systems | The objective of this study is to verify the performance and associated documentation of the device battery packs is in compliance with the applicable standard. Memo: Leverage previous testing as Energy Assurance  indicated changing firmware and coulomb counter training did not require retest. Give pictures of updated design to Energy Assurance so they can provide updated reports. Original Report 3P-P01-19 + justification | MS-10010 Emitter Battery Pack: Includes 17 closed and 5 open batteries each for the following tests: 3 vibration 3 shock 3 drop 5 overcharge 3 case stress 5 short circuit (open) MS-10083 Cassette Battery Pack: Includes 17 closed and 5 open batteries each for the following tests: 3 vibration 3 shock 3 drop 5 overcharge 3 case stress 5 short circuit (open) Sample sizes per IEC 62133-2:2017 Table 1 – Sample size for type tests | IEC 62133-2:2017 | Energy Assurance | N/A | N/A | N/A |
| Battery Accuracy Test | The objective of this study is to verify the emitter and cassette battery packs are able to estimate the battery capacity to within 5% per PRD5.28 | n=10 | N/A | MedAI | N/A | N/A | N/A |
| Tablet DICOM Calibration Testing | The objective of this study is to verify the use of the Samsung Galaxy S8+ tablet with accompanying MedAI Mobile Device App software for DICOM PS3.14 compliant diagnostic viewing of images. Memo: Reference existing report: 3P-P01-21 in an updated justification and include new Samsung results. Perform internal VVPR to demonstrate inputs/outputs for 1.0.0 are the same as 3.0.0. | n=1 T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Software dependent test | IEC 62563-1: 2021 DICOM PS3.14 | Image Quality Labs Justification/ Internal VVPR MedAI |  | x | N/A |
| Packaged-Products for Parcel Delivery System Shipment 70 kg (150 lb) or Less | The objective of this study is to perform a visual assessment and essential performance testing before and after ISTA 3A conditioning to ensure the device is not negatively impacted by normal transit. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher n = 1 F1 Foot Pedal SW 3.0.0 or higher One sample (each) is required for this test procedure per ISTA 3A 2018. Additional Sample Size Justification: All units in production are inspected to meet expected performance specifications which are sufficiently rigorous to highlight or detect changes in performance due to application of stresses which challenge the integrity of the device. All units perform a start-up check every time the unit turns ON to verify performance. | ISTA 3A 2018: Packaged Products for Parcel Delivery System Shipments 70 kg (150 lbs) or less (standard, small, flat or elongated) | MedAI ATS (Applied Technical Services) Shall perform and provide the ISTA 3A section of report to be included as an appendix | x |  | DV22 |
| Beam Current and Voltage Monitoring Accuracy at Room Temperature | The objective of this study is to verify the ability of the device to accurately measure the x-ray beam current and voltage, which is used for faulting monitoring at room temperature.   Essential Performance will be performed at room temperature. | n = 1 DV Unit SW 3.0.0 or higher Software dependent test. | Not applicable Update VVPR-P01-082 | MedAI | x |  | DV23 |
| Serial Radiography Max Pulses Verification | The objective of this study is to verify the max pulse count and accuracy of loading factors during the maximum duration of exposure (20 second) in Serial Radiography Mode and Fluoroscopy Mode of the device. | n = 1 DV Unit and T1 Tablet with MedAI Device App (A  PP) SW 3.0.0  or higher Software dependent test | Not applicable Update VVPR-P01-096 | MedAI | x |  | DV23 |
| Focal Spot Size Measurement | The objective of this study is to verify the size of the x-ray tube anode focal spot size for disclosure in the IFU and image quality reference. | n=1 | 603368 VVPR needed | MedAI | x |  | Various |
| Dosimetric Indications | The objective of this study is to verify the radiation output and dose accuracy at various device configurations including SID, loading factors, pucks and pediatric filters. In addition, this study will recalculate and re-validate the dose equation. Reference MEMO-P01-503. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.5.2.4.5.102 60601-2-43 203.5.2.4.5.102 Update VVPR-P01-111 | MedAI | x |  | DV26 |
| Half Value Layer | The objective of this study is to determine the half value layer of the beam in millimeters of aluminum. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-1-3 7.4, 7.5 60601-2-54 203.7.1 VVPR needed | MedAI | x |  | DV26 |
| Attenuation Equivalent | The objective of this study is to determine the attenuation factor of parts between the patient and the detector in millimeters of aluminum. Note: narrow beam setup equipment shall be brought by MedAI to Intertek testing | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.10.101 VVPR needed | MedAI | x |  | DV26 |
| Isokerma Map | The objective of this study is to verify the Scatter and Leakage Radiation of the device as described in IEC 60601-1-3 and 60601-2-43. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-43 203.13.4 203.13.6 Annex BB Protocol in VVPR-P01-105 | West Physics | x |  | DV26 |
| Stray Radiation Map | The objective of this study is to verify the Scatter and Leakage Radiation of the device as described in IEC 60601-1-3 and 60601-2-43. | n = 1 DV Unit and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-1-3 13.6 Protocol in VVPR-P01-105 | West Physics | x |  | DV26 |
| Residual Radiation | The objective of this study is to ensure residual radiation does not occur above allowable limits. | n = 1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.11.101 203.11.102 VVPR needed | MedAI | x |  | DV26 |
| Radiation Output Reproducibility | The objective of this study is to verify that the x-ray output air kerma remains consistent through successive emissions, for all loading factor combinations | n=1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.6.3.2.103.1 203.6.3.2.103.2 VVPR needed | MedAI & Intertek | x |  | DV26 |
| Radiography Linearity and Constancy | The objective of this study is to verify that the x-ray output air kerma remains linear and constant through successive emissions, for all loading factor combinations | n=1 DV Unit SW 3.0.0 or higher Type test per IEC 60601-1:2005+Amd 1:2012 section 5.2 | 60601-2-54 203.6.3.2.103.2 VVPR needed | MedAI & Intertek | x |  | DV26 |
| Battery Life Verification | The objective of this study is to verify the battery life of the device Emitter and Cassette from fully charged. Testing will use battery only with correct electronic load | n = 3 DV Unit, and T1 Tablet with MedAI Device App (APP) SW 3.0.0  or higher This is a controlled variable-measurement test. Variability is expected to be low due to well-established tolerances and performance characteristics of the battery cells. Only 3 DV Unit samples will be available for this test. These are representative samples of the item being tested. AVED Battery pack required | Not applicable Protocol in VVPR-P01-097 | MedAI | x |  | N/A |
| Cleaning For Expected Service Life | The objective of this study is to verify that the cleaning protocol specified in the instructions for use is appropriate for the expected service life of the device, and does not degrade the device safety nor essential performance features. | n = 1 DV Unit, and F1 Foot Pedal SW 3.0.0 or higher | FDA Guidance: Reprocessing Medical Devices in Health Care Settings: Validation Methods and Labeling, June 2017. Update VVPR-P01-136 | MedAI | x |  | DV21 |
| Cleaning & Disinfection | The objective of this study is to verify the MX1 Portable X-ray System may adopt the validated cleaning & disinfection protocol used for Imager Medical Imaging System, P00. Notes: 1. MedAI shall provide enclosures of the MX1 System (all externally exposed components), as well as material datasheets to perform a desktop assessment. | n = 1 E1 Emitter, C1 Cassette, W1 and F1 (relies on pre-cert from mfr) Enclosures (shells with all externally exposed components), H1 wired charger | ISO 17664-2:2021 ANSI/AAMI ST98:2022 AAMI TIR 12:2020 | Nelson Labs | N/A | N/A | N/A |
| SSD Accuracy Verification | The purpose of this test protocol is to verify the MX1 Portable X-ray System meets the Source to Skin Distance (SSD) accuracy specifications and foot pedal operating distance specifications established by MedAI. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | Not applicable Update VVPR-P01-101 | MedAI | x |  | DV23 |
| Foot Pedal Operating Distance | The purpose of this test protocol is to verify the MX1 Portable X-ray System meets foot pedal operating distance specifications established by MedAI. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 3.0.0 or higher | VVPR-P01-158 | MedAI | x |  | DV23 |
| Device Weight Verification | The objective of this study is to verify the MX1 Portable X-ray System and accessories meet the weight specifications established by MedAI. | n = 1 DV Unit (including P1 case), F1 Foot Pedal | Not applicable Update VVPR-P01-098 | MedAI | x |  | DV23 |
| Design Verification Via Demonstration | The objective of this study is to verify select MX1 product specifications via demonstration. | n = 1 DV Unit, F1 Foot Pedal, and T1 Tablet with MedAI Device App (APP) SW 2.1.0 or higher | Not applicable Update VVPR-P01-100 | MedAI |  | x | DV23 |
| Design Verification Via Inspection | The objective of this study is to verify select MX1 product specifications via inspection (documentation review). | Documentation review only. No product needed. | Not applicable Update VVPR-P01-099 | MedAI |  | x | DV23 |
| Foreign Object Testing | Demonstrate wireless charger does not transfer heat to foreign objects in proximity | n =1 | IEC 62368 VVPR-P01-148 | MedAI | x |  | DV23 |

### Table 3
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Technique Factor Clinical Input Study | The objective of this study is to determine loading factor recommendation for the device by having x-ray images compared by practitioners at various loading factor combinations. Defines technique factors per anatomy GUI NOT REQUIRED | n = 1 SW 3.0.0 or higher At least one board-certified practitioner with an active medical license experienced in imaging related to the device's intended use will be participating in this image comparison study. MedAI will collect radiographs for phantom anatomies/orientations at every available technique. Note: 1. Images shall be viewed on a DICOM compliant screen. | Not applicable | MedAI | x |  | DV24 |
| Image Quality Study | The objective of this study is to test the image quality of x-ray captures produced by the MX1 Portable X-ray System by comparing them to equivalent captures taken by the MinXray TR90BH Diagnostic X-Ray System (Reference Device: K182207). Uses Technique Factors determined by above test | At least three board-certified practitioners with active medical licenses experienced in imaging related to the device's intended use, at least two of whom will be orthopedic surgeons and at least one of whom will be a radiologist, will be participating in the image comparison study. MedAI will collect 10 radiographs of phantom extremities and hips with both the MX1 and TR90BH devices. | Not applicable Update VVPR-P01-080 VVPR-P01-120 | MedAI @ cadaver lab | x |  | DV24 |
| Summative Usability Testing | The objective of this study is to identify if the operator can interpret the Instructions for Use (IFU) and operate the device with minimum use-related hazards as well as to observe the operator interactions with the MX1 Portable X-ray System. The summative evaluation shall demonstrate and provide evidence that the MX1 Portable X-ray System can be used safely and effectively. Note: Emitter, Cassette, Table, UI and Foot Pedal will be repeated, including repeating ergonomics and new cassette handle.  Interoperability between K1 & MX1 shall also be evaluated with K1 including W1. | n = 1 SW 3.1.0  or higher K1, W1 n = 15 physicians, 15 non-physicians 20 uses in clinic/home use 5 uses outdoors 5 uses in surgical environments For purposes of use environments, the clinic and home use settings shall be grouped since there are limited differences in device users, setup, or use. | ● IEC 60601-1-6: 2013 ● IEC 62366-1: 2015 ● IEC 62366-2: 2020 Update VVPR-P01-081 | MedAI |  | x | DV21 |
| Summative Usability Testing K1 | VVPR-P01-127 - K1 Usability Summative Evaluation Protocol and Report | N/A | IEC 60601-1-6: 2013 IEC 62366-1: 2015 | MedAI | N/A | N/A | N/A |
| Summative Usability Testing W1 | VVPR-P01-140 - W1 Usability Summative Evaluation Protocol and Report | N/A | IEC 60601-1-6: 2013 IEC 62366-1: 2015 | MedAI | N/A | N/A | N/A |

### Table 4
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) and Test Method Summary | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Software System Requirements | The objective of this study is to verify the software system requirements from the SRS. Notes: 1. It is recommended to complete this testing prior to Electromagnetic Compatibility, Electrical Safety, Wireless Coexistence, and Secondary Battery Pack testing, since rationale must be provided as to whether any previous testing needs to be repeated in case of software changes. | n = 1 DV Unit SW 3.0.0 or higher Regression testing will be performed on 3.1.0 Software dependent test | Not applicable See Table in Section 6. | MedAI | x | x | DV23 |
| Collimator Light Field Accuracy Verification | The objective of this study is to demonstrate that the effective image reception area, as indicated by the Viewfinder UI, corresponds to the actual x-ray field at the x-ray image detector and any alignment error is within acceptable limits. | n = 1 SW 3.0.0 Software dependent test | IEC 60601-2-54: 2022 Section 203.8.5.3 “Correspondence between X-RAY FIELD and EFFECTIVE IMAGE RECEPTION AREA” Update VVPR-P01-087 | MedAI | x |  | DV23 |
| Software System Cybersecurity Penetration Testing | The objective of this study is to evaluate the software system security and exploit flaws while reporting the findings back to MedAI. Note: Test focus is to perform attacks similar to those of a malicious remote attacker and to attempt to infiltrate the system, alter or remove data, or cause degradation of system performance. | n = 1 DV Unit SW 3.1.0 or higher Software dependent test | Not applicable | MedAI |  | x | DV23 |

### Table 5
| Test Performed | Objective | Sample Size / Description | Consensus standard(s) and Test Method Summary | Testing and Report Completed by | Rev E 3.0.0 | Rev E 3.1.0 | DV Unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Engineering Mode | Evaluate SS development modes - Engineering mode requirements only | n = 1 DV Unit | Update VVPR-P01-088 | MedAI | x |  | DV23 |
| Debug and Release Modes & System Configuration | Evaluate SS development modes - Debug and Release, System logging and System Configuration | n = 1 DV Unit | Update VVPR-P01-109 | MedAI | x |  | DV23 |
| Power On/Off & Power States | Evaluate Power On/Startup, Power Off and Power States including Idle | n = 1 DV Unit | Update VVPR-P01-106 | MedAI | x |  | DV23 |
| Device Component Communications | Evaluate Device Component Communications | n = 1 DV Unit | Update VVPR-P01-115 | MedAI | x |  | DV23 |
| Emitter HMI | Evaluate Imaging Modes & Technique Factor Selection/Display | n = 1 DV Unit | Update VVPR-P01-110 | MedAI | x |  | DV23 |
| Photographic Acquisition | Evaluate Photographic Acquisition | n = 1 DV Unit | Update VVPR-P01-103 | MedAI | x |  | DV23 |
| Foot Pedal Integration | Evaluate Foot Ped Integration | n = 1 DV Unit | Update VVPR-P01-104 | MedAI | x |  | DV23 |
| Batteries and Wired Charging | Evaluate Batteries and BMS Integration & Wired charging (not accuracy of SoC or wired charging interlock check) | n = 1 DV Unit | Update VVPR-P01-108 | MedAI | x |  | DV23 |
| Safe State and Safety Interlocks | Evaluate Safe State & Safety Interlocks (Point to Faults VVPR for the fault interlock) | n = 1 DV Unit | Update VVPR-P01-102 | MedAI | x | x | DV23 |
| System Faults | Evaluate Faults at system level (Point to Safe State VVPR in description) | n = 1 DV Unit Requires final hardware and firmware - multidisciplinary effort | Update VVPR-P01-112 | MedAI | x |  | DV23 |
| Collimator Accuracy | Evaluate Collimation Accuracy | n = 1 DV Unit | Update VVPR-P01-078 | MedAI | x |  | DV23 |
| Radiographic Acquisition | Evaluate Tracking/Positioning Display including VF, MI LEDs and lasers, Collimation Display and Radiographic Acquisition including trigger behavior, Single, DDR and Fluoroscopy | n = 1 DV Unit | Update VVPR-P01-107 | MedAI | x |  | DV23 |
| MedAI Device App | Evaluate MedAI Device App | n = 1 DV Unit Retest Interlocks only for 3.1.0 | Update VVPR-P01-114 | MedAI | x | x | DV23 |
| Penetration Test | Evaluation at MedAI | n=1 DV Unit |  | MedAI |  | x | DV23 |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 09 Apr 2024 | 24-134 |
