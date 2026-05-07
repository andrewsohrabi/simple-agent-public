# MEMO-P01-699 Rev A: Regulatory Assessment for MX1 Rev F Design Changes

## Metadata
- Document ID: MEMO-P01-699
- Revision: A
- Prefix: MEMO
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: unknown
- Source filename: MEMO-P01-699 - Regulatory Assessment for MX1 Rev F Design Changes_A-Obsolete.docx
- Source path: Example QMS - MedAI/MEMO-P01-699 - Regulatory Assessment for MX1 Rev F Design Changes_A-Obsolete.docx
- Extraction warnings: none

## Extracted Content
Purpose
The purpose of this document is to evaluate the regulatory impact of planned design changes to the MX1 Portable X-Ray System. Revision E of the MX1 system was submitted to the FDA for review in 510(k) submission K241567 on May 31, 2024.
Scope
This document includes the regulatory assessment for upcoming design changes to the MX1 system to determine the regulatory pathway and any additional verifications requirements/documentation needed as part of the regulatory strategy.
References
Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Change to an Existing Device, October 25, 2017
Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Software Change to an Existing Device, October 25, 2017
Guidance for Industry and Food and Drug Administration Staff: The Special 510(k) Program, September 13, 2019
3P-P01-23 Rev A - T69454 MedAI HALT Test Report
IFU-MX1 Rev F - MX1 Instructions for Use
K241567 MX1 Portable X-ray System (MAI)
MEMO-P01-674 Rev C - MX1 Biocompatibility Assessment
PLN-P01-065 Rev B -  MX1 Verification and Validation Plan
QSP-018 Rev G - Design Control
BOM-055 Rev F-alpha - MX1 (Top-level assembly)
Background
MedAI is implementing design changes to the current MX1 system in order to circumvent supply chain issues, improve manufacturability, address engineering issues, and improve device reliability and performance.
There were several primary issues observed during the previously completed V&V and engineering testing that are being addressed through the changes described in Section 5 below. The development and testing cycles of the MX1 product has revealed slight image quality concerns in regard to X-ray images where EMI caused visible aberrations in radiographs, which posed a potential risk of masking hairline fractures in large anatomies. Lab tests with modified circuits has led to the aforementioned changes, and has shown improved results. Additionally, Highly Accelerated Life Testing (HALT) was performed and identified multiple performance and reliability issues. These issues were primarily related to thermal, vibration, and drop testing where components either sustained damage or failed during testing. Multiple changes were made to the mechanical design to improve the longevity of the device, as the device exhibited no such failures during V&V.
Summary of Changes
E1 Emitter
The changes described below are incorporated into BOM-004 Rev I - E1 Emitter.
Label: Emitter Ra Left (M10159) RA label artwork updated to include compliance statement: “Review Accompanying Documents for Compliance Information.”
Label: Emitter Ra Right (M10160)RA label artwork updated with new FCC ID number.
Label: Standardized GTIN (M10178)Label width dimension was modified and a note was added to the drawing to specify the stock label PN.
Emitter Battery Pack (MS-10010)S10042 Emitter BQ76952 Configuration File updated from v1.2 to v1.3 and S10043 Emitter MAX17205 Configuration File updated from v1.0 to v1.1. This change increased the voltage at which the emitter battery pack BMS and coulomb counter circuitry would shut down due to a safety undervoltage condition. This improves the long-term life of the battery cells and ensures that the cells are operating in their proper ranges.
HMI and Display PCBA Assembly (MS-10401)The P01 Emitter Display PCBA (ES-10005) design was updated to remove backlight control functionality and ambient light sensing functionality as well as move the backlight circuit to the bottom of the PCBA. These changes incorporate the Rev A.1 rework changes and removed hardware that was not utilized. ES-10005 Rev A.1 was the PCBA revision used in MX1 Rev E.
Emitter Power Input PCBA (ES-10015)Circuit changes were made to improve the responsiveness of the charger detection and remove the ambiguity of power states when a charger is used, Pogo Pins were added to the back face of the PCBA to ensure that the coil bracket is properly grounded in order to reduce conducted emissions, and the PCBA design was updated to incorporate the C.1 reworks. Additionally, epoxy was added to the capacitors and inductors to provide structural and vibrational support for the large component on the board. This epoxy was chosen specifically due to its thermal conductivity to ensure heat will not be trapped under the epoxy.
Emitter Main PCBA (ES-10003)
PCBA design was updated to incorporate the B.1 reworks, improvements to the 5V power rail to fix an issue where the Jetson module was receiving too low of a voltage, a BOM change to coordinate new charge detect from PMUX, as well as minor mechanical changes to the PCBA to receive the Sub-GHz cowling to improve the connection of the antenna to the module.
NVMe Viking - M.2 2230 256GB (M51021)This component is replacing NVMe Samsung (M50095) to alleviate supply chain issues. The components are identical in functionality.
Forward Button Slide, IM (M10253)The circular post within the component was extended 0.5mm in order to make the forward slide work with more button options as well as to make the button press more reliable.
E1 Lower Internal ASSY (MS-10136)Polyimide Tape 1/2" (M50293) This part was removed from the assembly. This component was not used in BOM-004 Rev H and was included by mistake. Emitter Sub-GHz Coax Cowling (M10755) and Emitter WIFI Coax Cowling (M10908) were added to the assembly in order to increase the reliability by ensuring that the imageroax antenna connection to the modules does not come loose over time.
Lit Cleat Cap ASSY (MS-10369)The pad print artwork wording was updated on the Power Cleat Cap, IM (M10250). There was no change to the assembly mechanical design.
Collimator PCBA (ES-10008)
The major change to the PCBA was the MCU was upgraded to PN (STM32F303RET6) which is faster and has more internal flash space, than the previous mictro-controller. This was done to improve performance of the collimator with faster UART/i2c Comms and to increase the collimator boot-up time upon startup. The PCBA design includes several HALT-related enhancements, such as modifying the board shape to prevent contact with the enclosure during extreme vibrations, and adding epoxy and anti-vibration caps to secure components under these same conditions. Additionally, fuses on the lasers were integrated to protect the device from short circuits in case of displacement. This revision also incorporates the C.1 reworks.
Framos Sensor, FSM-IMX335M- 02O-V1A (M50008)
The Framos tracking camera lens filter was changed from 940nm to 850nm to match the IR LED change on the cassette. The LED change was done to shift the tracking system IR system  away from the ToF IR system (940nm) in order to  limit interference between the two systems.
RH Laser ASSY (MS-10222) and LH Laser ASSY (MS-10221)
The line lasers (M10061) and their respective housing were both changed. The line lasers were changed so that the vendor now adds epoxy to the joint where the leads attach to the PCBA, this was done to improve the reliability of that joint.
The laser mounts were also changed to improve angle adjustment and to interface with an adjustment fixture. These changes were implemented to align the intersection of the crosshair lasers with the Viewfinder focal spot projection.
Drive ASSY (MS-10030)
Retention method of pulley to motor shaft is changed from set screws to epoxy (M50897). Pulleys retained by set screws were loosening after some period of operation, causing the collimator aperture to be inoperable. Epoxy replaces the set screw to strengthen the joint.
Pucks (M10101 - M10118)No change to part material or geometry. Drawing note updated to add additional detail for coating thickness specification.
Pediatric Filter (MS-10939)The pediatric filter sticker was added to create the new pediatric filter assembly in order to improve the durability of the pediatric filter.
C1 Cassette
The changes described below are incorporated into BOM-008 Rev J - C1 Cassette.
Label: Standardized GTIN (M10178)Label width dimension was modified and a note was added to the drawing to specify the stock label PN.
SAFETY RA LABEL (M10162)RA label artwork updated with new FCC ID number and corrected typo.
Cassette Battery Pack (MS-10083)Updated CT200/2.00 to CT200/1.50 per Aved Electronics request (2" is longer than needed). S10040 Cassette BQ76952 Configuration File updated from v1.2 to v1.3 and Cassette MAX17205 Configuration File updated from v1.0 to v1.1. This change increased the voltage at which the emitter battery pack BMS and coulomb counter circuitry would shut down due to a safety undervoltage condition. This improves the long-term life of the battery cells and ensures that the cells are operating in their proper ranges.
Cassette Battery Mounting Foam (M10407)Correction to drawing note. No change to design.
Cassette Handle Assembly (MS-11139)Minor updates to the Cassette Handle Connector Screw Base (M11135) and Cassette Handle Connector Screw Cap (M11136) including finish updates and a thread depth change to improve strength and reliability.
Cassette Top Populated (MS-11089)The in-house Heat Pipe assembly was replaced with vendor-manufactured components to improve manufacturability, the metal ribbon cable strap replaced with plastic injection molded component, screw pre-applied Loctite material was updated with an equivalent material to alleviate vendor sourcing issues, and the  pre-applied loctite was removed from tracking board mounting screws and replaced with liquid formulation to improve serviceability.
Cassette Enclosure Bottom with Inserts (MS-11092)Drawing revision changed due to note update. No change to design.
Cassette Bottom Bumper (M11040)Part color update. No change to geometry or material.
MX1 Cassette Display PCBA (ES-10037)The updated design incorporates design modifications present on Revision A.1, which was the revision used in MX1 Rev E. This change adds additional diodes for ESD immunity improvements. Additional capacitors have been added to the design for power integrity improvements. Additional diodes provide improved ESD performance and additional capacitance improves power integrity.
DISPLAY ASSEMBLY ESD ADHESIVE (M10450)Adhesive material updated to improve water ingress protection.
CASSETTE MOLDED BUTTON (M11078)Material updated from ELASTOSIL 3003/30 A/B to ELASTOSIL R 401 20A to improve ingress protection.
MX1 Cassette Angled Tracking PCBA (ES-10038)The PCBA design was modified to allow for power delivery network improvements via additional capacitance, EMI improvements to reduce impact on Detector, and stackup modifications to allow for these changes within the same board shape. The updated revision of this design incorporates these modifications in a controlled manner and will provide improved performance for the MX1 product. New IR LEDs are also incorporated to shift the tracking system IR system away from the ToF IR system (940nm) in order to limit interference between the two systems.
NVMe Viking - M.2 2230 256GB (M51021)This component is replacing NVMe Samsung (M50095) to alleviate supply chain issues. The components are identical in functionality.
Cassette Main PCBA (ES-10004)The Cassette Main REV C design updated the isolated DC/DC converter used for generating a 5V power rail on the isolated USB port region of the design. The previous isolated DC/DC converter radiated EMI such that visible aberrations were present in X-ray images, and posed a potential risk of masking hairline fractures. The regulator circuit which provides power to the system's detector has also been updated with new passive component values to modify the compensation of the regulator circuit. This modification aids in reducing voltage ripple, and aids in improving detector image quality.
Additional PEM nut locations have been added to the board layout, at the request of the MedAI mechanical engineering team for future use with additional thermal management solutions. The physical layout of the circuit board has been updated to accommodate the size and pin configuration of the new isolated regulator, and additional PEM nuts.
USB FERRITE (MS-10402)Assembly removed, components moved to Cassette Main - Modules Populated (MS-11086).
Label: Active Area Decal (M11104)The artwork on the Active Area Decal was updated to specify that only that area is rated for 300 lbf.
Cassette silicone LED light pipes (M11165)New component added to Cassette + Inserts and Light Pipes Subassembly (MS-11113). The change in geometry of the light pipe was to increase retention of the light pipe into the housing as previous reliability tests showed that in certain edge cases the part can become detached. The change in material of the light pipe was to increase the sealing properties of the light pipe interface to the enclosure and improve ingress protection.
Tracking PCBA (ES-10036)The layout of this design has been modified to allow for power delivery network improvements, EMI improvements to reduce interference with Detector (image quality), and stackup modifications to allow for these changes within the same board shape. New IR LEDs are also incorporated to shift the tracking system IR system away from the ToF IR system (940nm) in order to limit interference between the two systems.
Software System
The changes described below are incorporated into SS v4.0.0 and are mostly firmware changes to allow the Software System to work with the new Rev F Hardware changes.
Collimator MCU change
The collimator imagerontroller was changed to to improve performance of the collimator with faster. This hardware change caused minimal updates to the collimator firmware, as the new imagerontroller was compatible with the previous one. This change resulted in faster capabilities for certain communication protocols.
IR Tracking change
IR LEDs were updated from 940 nm to 850 nm, and ToF sensors remained 940 nm. This change prevents interference between the LIDAR sensor and the tracking LEDs. Cassette firmware was updated to accommodate this hardware change.
EM/PMUX change
Modified firmware to work with new PBCA revision that improves charge detection.
Foot Pedal Improvements
Updates to Foot Pedal firmware to introduce 2-way communication with the foot pedal, allowing for remote upgrade of foot pedal firmware and improved reliability/responsiveness.
Auto provisioning remote access (Tailscale)
Adding capability for remote access for device troubleshooting in-field to provide better customer support.
Regulatory Assessment
United States
The changes were evaluated per Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Change to an Existing Device (2017) and Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Software Change to an Existing Device (2017). Note that this assessment was performed prior to completion of testing and final risk analysis updates, therefore it may be updated in the future based on final results and analysis. A formal letter to file and complete assessment per QSF-077 is planned at the completion of Phase 3 deliverables.
Assessment of non-software changes per Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Change to an Existing Device (2017):
Main Flowchart Questions
Change made with intent to significantly improve the safety or effectiveness of the device?
No, the changes described in Section 5 were not made with intent to significantly improve the safety or effectiveness of the device. Rather, the changes were made to improve the device reliability and rectify issues that would otherwise prevent the device from performing as intended.
Labeling Change?
No, there are no labeling changes resulting from the design changes.
Technology, engineering, or performance change?
Yes, changes were made to improve the device reliability and performance. Multiple issues were identified and rectified in order to allow the device to perform as intended and without failures during use.
Materials Change?
Yes, several component materials were updated to improve ingress protection.
Chart B Questions
B1 - Is the device an IVD?
No, the device is not an IVD.
B2 - Is it a control mechanism, operating principle, or energy type change?
No, there are no changes to the control mechanism, operating principle, or energy type. The changes do not affect the power input/output, operational algorithms, or design mechanisms.
B3 - Is it a change in sterilization, cleaning, or disinfection?
No, none of the changes described are related to cleaning or disinfection. MX1 is not a sterile device.
B4 - Is there a change in packaging or expiration dating?
No, there are no changes to packaging. MX1 does not have an expiration date.
B5 - Is there a change in design (e.g., dimensions, performance specifications, wireless communications, components or accessories, patient/user interface)?
Yes, there are design changes including dimensions, component material changes, and various improvements to device performance and reliability where performance/reliability issues were identified.
B5.1 - Does the change significantly affect the use of the device?
No, the changes described do not affect device usability or device function. The changes do not impact the device indications for use.
B5.2 - Does a risk assessment identify any new or significantly modified risks?
No, a risk assessment was performed in PLN-P01-065 Rev B -  MX1 Verification and Validation Plan (see Section 4). The design changes have low or no impact on previously identified risks, and do not introduce any new risks. There were no changes made to features that are critical to the device’s safe or effective operation.
B5.3 - Is clinical data necessary?
No, clinical data is not necessary. All affected bench testing will be repeated as outlined in PLN-P01-065 Rev B -  MX1 Verification and Validation Plan.
B5.4 - Any unexpected issues from V&V activities?
No issues are expected to be produced during routine V&V. Should any unexpected issues arise and/or the modified design cannot be verified/validated, the testing results will be reviewed and analyzed and this assessment will be reviewed and updated as necessary. The design changes do not necessitate new test methods or acceptance criteria.
Decision: Documentation. A new 510(k) is not required.
Chart C Questions
C1 - Is the device an IVD?
No, the device is not an IVD.
C2 - Change in material type, formulation, chemical composition, or the material’s processing?
Material type changes were made to several parts (M11165, ​​M10450, M11078) to improve ingress protection.
C3 - Will the changed material directly or indirectly contact body tissues or fluids?
M11165 is expected to have limited contact with intact skin of both the patient and operator. M11078 is expected to have limited contact with intact skin of the operator.
C4 - Does a risk assessment identify any new or increased biocompatibility concerns?
Materials were assessed in MEMO-P01-674 Rev C - MX1 Biocompatibility Assessment. It is unlikely that any adverse reactions would occur when in contact with intact skin.
C5 - Could the change affect performance specifications?
No, the changes do not affect performance specifications.
Decision: Documentation. A new 510(k) is not required.
Assessment of software changes per Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Software Change to an Existing Device (2017):
Flowchart Questions
1 - Is the change made solely to strengthen cybersecurity and does not have any other impact on the software or device?
No, the changes described were made to ensure the compatibility between the MX1 software system and the updated hardware.
2 - Is the change made solely to return the system into specification of the most recently cleared device?
No, the changes were not made to return the system into specification of the most recently cleared device. The updated software system includes changes to ensure the compatibility between the MX1 software system and the updated hardware. The hardware was updated to improve device performance/reliability.
3a - Does the change introduce a new risk or modify an existing risk that could result in significant harm and that is not effectively mitigated in the most recently cleared device?
-OR-
3b - Does the change create or necessitate a new risk control measure or a modification of an existing risk control for a hazardous situation that could result in significant harm?
The updated software system does not introduce new risks or modify existing risks that could result in significant harm and that are not effectively mitigated. Should any software or firmware failure occur, the system heartbeat/watchdog monitoring will trigger a device safe state, as described in RSK-P01-010 - MX1 Risk Assessment.
4 - Could the change significantly affect clinical functionality or performance specifications that are directly associated with the intended use of the device?
No, the change does not significantly affect clinical functionality; there is no impact to clinical decision-making. Changes do not impact the user interface, user groups, or use environments. The software modifications allow for successful integration of the updated device hardware.
The software modifications would not be described as “code maintenance” or “infrastructure” modifications per Section VI of  Guidance for Industry and Food and Drug Administration Staff: Deciding When to Submit a 510(k) for a Software Change to an Existing Device (2017), Additional Factors to Consider When Determining When to Submit a New 510(k) for a Software Change to an Existing Device.
Decision: Documentation. A new 510(k) is not required.
Quality Engineering
Below are the DHF-008 deliverables that will require review and/or updating as a result of the BOM-055 Rev F and software system v4.0.0 changes:
Conclusion
The memo evaluates the regulatory pathway to implement the design changes of  the MX1 Portable X-Ray System documented in BOM-055 Rev F. The design changes were primarily made to address supply chain, engineering, and reliability issues. Notable design changes include updates to the PCBAs to reduce EMI impact to the detector as well as mechanical improvements to address component failures due to vibration/impact. The system also underwent software updates to ensure compatibility with new hardware.
The regulatory assessment, following FDA guidance, concludes that a new 510(k) submission is not required since the changes improve reliability but do not significantly alter device performance and safety, introduce new risks, or modify existing risk mitigations. Verification and validation testing will be repeated as necessary, and relevant DHF deliverables will be reviewed and updated to reflect the design changes.
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Device Hazard Analysis | RSK-P01-010 - MX1 Risk Assessment |
| --- | --- |
| DFMEA | RSK-P01-012 - MX1 DFMEA |
| Software Hazard Analysis | RSK-P01-010 - MX1 Risk Assessment |
| Initial Cybersecurity Risk Analysis | RSK-P01-011 - MX1 Security Risk Assessment |
| Software Requirements Specifications (SRS) | MEMO-P01-630 - MX1 Software Requirement Specifications |
| Software Architectural Design (SAD) | MEMO-P01-658 - MX1 System Architecture Diagram |
|  | MEMO-P01-633 - MX1 System and Software Architecture Design |
| Software Design Specifications (SDS) | MEMO-P01-631 - MX1 Software Design Specifications |
| Verification and Validation Plan | PLN-P01-065 Rev B -  MX1 Verification and Validation Plan |
| Design Verification Protocol(s) and Report(s) | Refer to PLN-P01-065 Rev B - MX1 Verification and Validation Plan |
| Design Verification Summary Report | Document will be revised following initial release of V&V summary for BOM-055 Rev E |
| Software Unresolved Anomaly (Bugs or Defects) | Document to be created if software unresolved anomalies remain following v4.0.0 verification and validation |
| Biocompatibility Assessment | MEMO-P01-674 - MX1 Biocompatibility Assessment |
| Risk Management Report | Document will be revised, if necessary, following initial release of Risk Management Summary for BOM-055 Rev E |
| Component Inspection Plans | QSR-026 - Lot Acceptance Sampling |
| Verification & Validation Trace Matrix | VVAM-P01-004 - MX1 Verification & Validation Trace Matrix |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | See ECR-543 |  |  |
