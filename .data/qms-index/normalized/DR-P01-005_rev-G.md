# DR-P01-005 Rev G: MX1 Design Inputs

## Metadata
- Document ID: DR-P01-005
- Revision: G
- Prefix: DR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: DR-P01-005 - MX1 Design Inputs_G.docx
- Source path: Example QMS - MedAI/DR-P01-005 - MX1 Design Inputs_G.docx
- Extraction warnings: none

## Extracted Content
DR-P01-005 - MX1 Design Inputs_G
Sheet: Reqs Deleted from rev D1
Sheet: Signoff
Sheet: Introduction
Sheet: User Needs (UN)
Sheet: Product Req (PRD)
Sheet: IFU Req (IFU)
Sheet: (OLD) IFU Req (IFU)
Sheet: RSK Reqs
Sheet: Deleted RSK Reqs
Sheet: RSK Req
Sheet: SRS gap check
Sheet: Business Needs
Sheet: VVAM - TEMP
Sheet: Specs to Review
Sheet: Medtronic Req Gap
Sheet: UNPRD Lookup
Sheet: Template Revision History
Sheet: Benchtop Test Sequence
Sheet: Intertek Pre-DV Recommendations
Sheet: Copy of User Needs_BB

### Table 1
| Requirement |  |
| --- | --- |
| RSK_R038 The device shall contain firmware timers with a timeout limit to mitigate Grid Signal Timing Errors. |  |
| RSK_R048 The device enclosure(s) shall be made with a material that minimizes cracking and wear. |  |
| RSK_R097 The device shall utilize proper application matching to mitigate vacuum loss. |  |
| RSK_R152 The device shall be sealed after assembly with tamper proof stickers. |  |
| RSK_R159 The device may be provided with a wrist strap. |  |
| RSK_R173 The device UI shall display the x-ray parameters before an image taken. |  |
| RSK_R187 The device shall use fasteners for tie mounts. |  |
| RSK_R301 The device manufacturer shall record the essential performance. |  |
| RSK_R336 The device shall utilize a Non-stackable puck design. |  |
| RSK_R342 The space between the detector and patient contacting surface shall be no less than 5mm. |  |
| RSK_R349 The manufacturer shall inform the end user to eIFU update(s). |  |
| RSK_R353 The device shells be constructed via molding. |  |
| USE_R005 Patient cable shall be designed to provide free range of motion from point of contact with the device for a minimum of 3 feet |  |
| USE_R008 The device shall display imaging modes upon switching modes. |  |
| USE_R017 The device shall inform the operator once the emitter has been removed from the active area of the detector. |  |
| USE_R021 The device may inform the operator to incompatible devices. |  |
| USE_R025 The device should provide feedback when device components are not functioning. |  |
| USE_R035 Information should be included to communicate the end of the warm-up process. |  |
| USE_R036 The device should communicate to the operator if warm-up procedure is required. |  |
| USE_R043 The connecting cables should be long enough to allow the operator to position the cassette around the intended anatomy. |  |
| USE_R107 The device shall have all ports positioned such that it may be attached during set up of the device. |  |
| USE_R112 The Control Unit Power Switch shall be switched on and off |  |
| USE_R145 The device may produce an error message when operators attempt to improperly transfer images to PACS. |  |
| USE_R176 The device UI should allow the Operator to reset the image. |  |
| USE_R212 The lasers should be disabled when emitter is not in use. |  |
| USE_R247 The device shall have no more than 6 pucks. |  |
| USE_R256 The device should be able to send more than one image at a time to PACS. |  |
| USE_R281 The primary packaging (case) shall be made of lightweight material(s). |  |
| USE_R308 The screens used to convey information to the operator shall be at an illumination level at least 10% greater than the expected ambient luminance level |  |
| USE_R322 The device should allow the Operator to switch between modes while Emitter is held |  |
| USE_R330 Touchpoints shall provide tactile, audible and or visual feedback to the operator once an action has been initiated. |  |
| USE_R336 The trigger shall require X" of movement from the resting position to initiate. |  |
| USE_R373 The image should allow for labeling of the appropriate extremity laterality (left/right). |  |
| USE_R376 The device shall be able to provide feedback to the operator if they attempt to set a parameter to a value that is not allowed by the device |  |
| USE_R394 The device shall allow accessibility to critical information. |  |
| USE_R400 The device shall provide labeling to mitigate Improper placement of device and accessories. |  |
| USE_R412 The device shall allow the user to manipulate image(s). |  |
| USE_R413 The device shall provide UI feedback during use. |  |
| USE_R426 The device shall provide a means to mitigate incorrectly positioning the image (incorrect view). |  |
| USE_R449 The device shall provide a means to mitigate the pucks from falling on the patient |  |
| USE_R477 The operator may use the emitter from the side where the patient cables connect. |  |
| USE_R478 The drapes should not interfere with an image. |  |

### Table 2
| MedAI MEDICAL, INC |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Document: | DR-P01-005 - Design Inputs & Specifications |  |  |  |  |
| Project: | P01 |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |
| Revision | Description | DCO # | Approved By | Eff. Date | Digital Key |
| A | Initial Release | 24-190 | EngineeringQuality EngineeringRegulatory Affairs | 2024-04-26 00:00:00 | example.com/ |
| B | Updated Product Update, Indications for Use, & ContraindicationsSummative Usability Results Update; Requirement updates per Phase 3 testing.Adjusted SSD lockout and added angle req for SID and SSD.Removed req for inversion in any mode | Refer to ECR-444 |  |  | example.com/ |
| C | Updated Contraindications | Refer to ECR-461 |  |  | example.com/ |
| D | Updates per 3.2.0 MX1 App UI changes and summative usability | Refer to ECR-570 |  |  | example.com/ |
| E | Removal of RSK_R365 and several IFU requirements per RSK-P01-010 Rev D, Update PRD to reflect Indications for Use | Refer to ECR-601 |  |  | example.com/ |
| F | Update to PRD 3.11 to include prevention of handheld serial radiography | Refer to ECR-643 |  |  | example.com/ |
| G | -Added RSK_R384 through RSK_R392-Corrected information in RSK_R329 specification as previous information was inaccurate-Removed RSK_R279. Item is a component supplier requirement-Removed RSK_R267. Dose verification was performed as part of software V&V-Corrected PN listed in PRD7.9 Specification | Refer to ECR-741 |  |  | example.com/ |

### Table 3
| Product Overview |  |
| --- | --- |
| The MX1 Portable X-ray System is designed to aid clinicians with point-of-care visualization and guidance during X-rays of extremities and shoulders. It is intended for use in clinical environments and is not intended for surgical applications.        The MX1 device consists of five major components: The Emitter, Cassette, Foot Pedal, Wired Charger, and MedAI Application. The system is intended to be used with an external display such as tablet or touchscreen, clinical cart, and wireless charger. The device is also intended to integrate with the MedAI Platform. |  |
| Product Description |  |
| Product Architecture Diagram | Block diagram of the MX1 system and all its major subsystems |
| Product Isolation Diagram | Isolation diagram of the MX1 system and power isolation schemes |
| Emitter | This component is a battery powered device that contains the operator interface, Viewfinder, x-ray tube with power supply, and camera system. The operator interface allows the operator to control the major functions of the device, including technique factors. The Viewfinder shows the operator the overlay of the detector Active Area, x-ray field, and anatomy of interest to aid in imaging the anatomy of interest. This component is controlled and held in the operator's hand. |
| Cassette | This component in a battery powered device that contains the x-ray detector and captures the x-ray energy from the emitter and develops the x-ray image. This component also contains status and IR LEDs to assist in x-ray field positioning. The patient anatomy of interest is placed on top of this component. |
| Wired Charger | This component is a 60601-1 compliant, AC to DC power supply that will isolate the device from MAINS. It has USB-C connector and operate using the USB-PD protocol. |
| Foot Pedal | This component is a wireless footswitch that allows the operator to trigger a single shot x-ray captures. |
| MedAI App | This component is a software application that will be preloaded on a manufacturer supplied tablet. |
| Indications for Use |  |
| The MX1 Portable X-ray System is indicated for use by qualified/trained medical professionals on adult patients for orthopedic radiographic, orthopedic serial radiographic, orthopedic fluoroscopic, and orthopedic interventional procedures of only shoulders to fingers and knees to toes. The device is not intended for use during surgery. The device is not intended to replace a stationary radiographic system. The device is to be used in healthcare facilities where qualified operators are present (e.g., outpatient clinics, urgent cares, imaging centers, sports medicine facilities, occupational medicine clinics)The device is not intended to be used in environments with the following characteristics:     Aseptic or sterile fields, such as in surgery     Home or residential settings or other settings where qualified operators are not present     Vehicular and moving environments     Environments under direct sunlight     Oxygen-rich environments, such as near an operating oxygenation concentrator |  |
| Contraindications |  |
| The MX1 System is NOT intended for bariatric patientsThe MX1 System is NOT intended for mammography.The MX1 System is NOT intended for dental applications.The MX1 System is NOT intended to come in contact with non-intact skin.The MX1 System is NOT intended for cardiac applications.The MX1 System is not intended for use in proximity to pacemakers or implantable cardioverter-defibrillators (ICDs) |  |

### Table 4
| MX1 System Design Record - Operator Needs |  |  |  |
| --- | --- | --- | --- |
| User Need ID | PRD Cnt | Summary | User Need |
| UN1. | #REF! | Servicing Device UN | Manufacturer shall be able to service the device. |
| UN2. | #REF! | Patient Population UN | Operator shall use the device with adult and pediatric patients. |
| UN3. | #REF! | Radiograph UN | Operator shall capture diagnostic radiographic images of extremities and shoulders. |
| UN4. | #REF! | DDR & Radioscopy UN | Operator shall capture diagnostic serial radiography, radioscopy of extremities and shoulders. |
| UN5. | #REF! | Photography UN | Operator shall capture photographic images of anatomies and objects. |
| UN6. | #REF! | Minimal PPE UN | Operator should use the device without surpassing their yearly occupational dose limits, per Code of Federal Regulations, Title 10, Part 20.1201. |
| UN7. | #REF! | Layperson UN | Operator should be a medical professional and be able to transport, set up, use, and pack up the system alone without tools and with Accompanying Documents. |
| UN8. | #REF! | No Lead Lined Room UN | Operator shall be able to use the device without lead-lined rooms for radiation protection, if local regulations allow. |
| UN9. | #REF! | Battery Powered UN | Operator shall use the device while battery-powered. |
| UN10. | #REF! | Environment UN | Operator shall use the device in the following environments: office, clinical. |
| UN11. | #REF! | Packaging UN | Operator or Manufacturer shall be able to transport the packaged device safely in an automobile and airplane cargo. |
| UN12. | #REF! | Disconnected Cassette+Emitter UN | Operator shall use the device without mechanically or electrically tethering the cassette and emitter together. |
| UN13. | #REF! | Ergonomic Shooting UN | Operator should comfortably use the emitter and cassette in use positions where the emitter is pointing down and use positions where the emitter is pointing forward. |
| UN16. | #REF! | PACS et al UN | Operator shall send images and data to PACS and peripheral storage drives. |
| UN17. | #REF! | View and Post-Processing UN | Operator shall view an image and conduct post-processing (e.g. rotate, zoom, etc.) with and without internet connection. |
| UN19. | #REF! | Import Patient Info UN | Operator may import patient information from an external source. |
| UN20. | #REF! | Viewfinder UN | Operator shall view the anticipated x-ray beam illumination in order to distance, angle, align, and collimate the emitter, anatomy, and cassette for an intended radiograph. |
| UN21. | #REF! | Tracking UN | Operator shall only be able to emit x-ray radiation while pointing the emitter at the cassette within allowable SID ranges |
| UN23. | #REF! | X-ray Technique UN | Operator should adjust loading factors (kVp, mAs) and acquisition type (Radiography, Photography) using HMI on the Emitter and Tablet. |
| UN25. | #REF! | Facility Metrics UN | Operator may have access to the device's images taken, image study dosage information for facility quality reviews. |
| UN26. | #REF! | Regional Markets UN | Manufacturer should market the device in the United States, Canada, Mexico, and European Union. |
| UN27. | #REF! | Foot Pedal UN | Operator shall be able to trigger acquisition wirelessly through a foot pedal. |
| UN28. | #REF! | Weight Bearing x-ray UN | 75th Percentile American male patient shall stand on cassette for weight-bearing images of the foot and ankle. |

### Table 5
| ID | Requirement | Specification |
| --- | --- | --- |
| 1. General |  |  |
| PRD1.1 | The device shall be able to be packed, setup, and repacked without the use of a tool. | Specification identical to requirement |
| PRD1.2 | The device shall include Accompanying Documents (IFU) | Reference IFU Requirements |
| 2. X-ray Imaging |  |  |
| PRD2.1 | The x-ray tube assembly shall be self-shielded. | -No additional shielding outside the monoblock-Complies with IEC 60601-1-3 |
| PRD2.2 | The device shall monitor and log beam current, filament current, and monoblock temperature with each acquisition. | Reference SRS |
| PRD2.3 | The x-ray tube focal spot size shall meet engineering specification for indicated anatomies and use enviroments | Specification identical to requirement (component specification - not to be verified at device top level) |
| PRD2.4 | The x-ray tube shall operate between 40 kV to 80 kV in 10kV increments. | Reference SRS |
| PRD2.5 | The x-ray tube beam current shall operate between 1mA to 2mA. | Reference SRS |
| PRD2.6 | The x-ray tube shall operate between 0.04 - 0.40 mAs in 5 steps; the options shall be 0.04, 0.08, 0.16, 0.25, 0.40 mAs. | Reference SRS |
| PRD2.7 | The x-ray exposure in serial radiographic mode shall be 40ms per frame, 5 frames per second, for a maximum of 20 seconds. | Test Points:@ 40kV, 0.04mAs 20 second DDR@ 60kV, 0.04mAs 20 second DDR@ 80kV, 0.04mAs 20 second DDR@ 40kV, 0.08mAs 20 second DDR@ 60kV, 0.08mAs 20 second DDR@ 80kV, 0.08mAs 20 second DDRAcceptance Criteria:Results in 100 + 1 pulses (frames)Sample 1st, 50th, and Last pulseMeasured Voltage ± 8% errorMeasured Current + 20% errorMeasured Time ± (10 % + 1ms) errorReference SRS |
| PRD2.8 | The device shall be able to perform DDR up to 80 kV and 2mA. | Specification identical to requirement |
| PRD2.9 | The device shall be able to perform single exposure x-rays up to 80 kV and 2mA. | Specification identical to requirement |
| PRD2.10 | The primary fixed collimation shall collimate to 43 deg. | The primary fixed collimation shall collimate to 43 deg +/- 0.5 degReference M10076 drawing |
| PRD2.11 | The device shall have a minimum Aluminum equivalent total x-ray beam filtration of 2.5 mm. | HVL > 2.5mm AL @70kVHVL > 2.9mm AL @80kV |
| PRD2.12 | The device shall utilize a digital flat field x-ray detector. | Reference cassette drawings for C1 |
| PRD2.13 | The device shall have a detector with an active area of 213.5mm x 213.5mm. | Reference cassette drawings for C1 |
| PRD2.14 | The detector shall contain shielding or have shielding behind the detector. | Reference cassette drawings for C1 |
| PRD2.19 | The nominal x-ray exposure in Fluoroscopy Mode shall be 40ms per frame, 5 frames per second, for a maximum of 20 seconds. | Test Points:@ 40kV, 0.04mAs 20 second Fluro@ 50kV, 0.04mAs 20 second Fluro@ 60kV, 0.04mAs 20 second Fluro@ 64kV, 0.04mAs 20 second Fluro@ 40kV, 0.08mAs 20 second Fluro@ 50kV, 0.08mAs 20 second Fluro@ 60kV, 0.08mAs 20 second Fluro@ 64kV, 0.08mAs 20 second FluroAcceptance Criteria:Results in 100 + 1 pulses (frames)Sample 1st, 50th, and Last pulseMeasured Voltage ± 8% errorMeasured Current + 20% errorMeasured Time ± (10 % + 1ms) errorReference SRS |
| PRD2.20 | The nominal voltage for Radioscopy shall be 80% or less than that of Radiography | Nominal voltage in fluoroscopy shall be 64kV |
| PRD2.21 | The device shall provide a Low Dose Fluoroscopy Mode utilizing loading factors of 40 ms exposure time per frame, two point five (2.5) frames per second, for a maximum of 20 seconds duration | Test Points:@ 40kV, 0.04mAs 20 second Fluro@ 50kV, 0.04mAs 20 second Fluro@ 60kV, 0.04mAs 20 second Fluro@ 64kV, 0.04mAs 20 second Fluro@ 40kV, 0.08mAs 20 second Fluro@ 50kV, 0.08mAs 20 second Fluro@ 60kV, 0.08mAs 20 second Fluro@ 64kV, 0.08mAs 20 second FluroAcceptance Criteria:Results in 50 + 1 pulses (frames)Sample 1st, 50th, and Last pulseMeasured Voltage ± 8% errorMeasured Current + 20% errorMeasured Time ± (10 % + 1ms) errorReference SRS |
| 3. Positioning and Alignment |  |  |
| PRD3.1 | The positioning system shall compute the Source to Detector distance (SID) within 5% error, through the full SID range and x-ray beam angles up to 30 deg realative to the normal axis of the cassette | The positioning system shall compute the Source to Detector distance (SID) with <5% error@ 25cm, 40cm, 60cm, 80cmComplies with IEC 60601-1-3 |
| PRD3.2 | The positioning system shall compute the Source to Skin distance (SSD) such that calculated values are a) within error of 8% or 15mm (whichever is larger) and b) offset so that the calculated value is always less than the physical measured value. Tested at the SSD specified in PRD3.10. | The device shall compute the Source to Skin distance (SSD) with <  15mm error@ 25cm, 40cm, 60cm, 80cm |
| PRD3.3 | The device shall only allow x-ray emissions within a source-to-detector (SID) distance between 25cm and 80cm. | Reference SRSX-ray emissions disabled <25cm (LEDs red and will not emit)X-ray emissions allowed 25-80cm (LEDs green and able to emit)X-ray emissions Disabled > 80cm (LEDs red and will not emit) |
| PRD3.4 | The tracking system should function when 50% of the LEDs are not visible. | Tracking system functions (LEDs are green and device will emit x-rays) when 50% of the IR LEDs are covered |
| PRD3.5 | The tracking system should function accurately with 1 standard medical drape over the cassette. | Tracking system functions (LEDs are green and device will emit x-rays) when 100% of the IR LEDs are covered by a single layer medical drape |
| PRD3.6 | The tracking system shall operate under high ambient light conditions. | Tracking system functions (LEDs are green and device will emit x-rays)@2,000 lux (per IEC 60601-1 Subclause 7.1.2) |
| PRD3.7 | The device shall display to the operator the status of the system via indicator LEDs. | Reference SRS |
| PRD3.8 | The automatic collimator to confine the x-ray field shall be able to adjust aperture size and rotation to line up with the detector at any specified SID. | @ 25cm, 40cm, 60cm, 80cm. Rotate the emitter 360 degrees in 45 degree increments and make sure the major & minor axes of the collimated field remain parallel to the walls of the active area at each increment. |
| PRD3.9 | The automatic collimator shall be able to adjust the aperture size at any given SID; selectable steps shall not exceed 0.8 cm in the length and width when in a plane orthogonal to the reference at a distance of 80 cm from the focal spot. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm. |
| PRD3.10 | The device shall prevent x-ray emission when the calculated Source-to-Skin Distance is less than 130cm | Reference SRS |
| PRD3.11 | The device shall prevent hand-held Radioscopy and DDR | Reference SRS |
| 4. Viewfinder UI |  |  |
| PRD4.1 | The device shall contain a viewfinder UI that allows the operator to view the detector active area while the cassette is draped, and provides a means to to align the anatomy and detector. | Reference SRS |
| PRD4.2 | The viewfinder shall display the optical image transformed into an image as seen from the Cassette. | Reference SRS |
| PRD4.3 | The viewfinder shall calculate and display the collimated x-ray field. | Reference SRS |
| PRD4.4 | The viewfinder shall calculate and display the active area of the detector. | Reference SRS |
| PRD4.5 | The viewfinder shall include a reference point to indicate the center of the x-ray field. | Reference SRS |
| PRD4.6 | The viewfinder shall overlay the x-ray field and active area on the optical image. | Reference SRS |
| PRD4.7 | The viewfinder shall display loading factors before taking an image. | Reference SRS |
| PRD4.8 | The viewfinder shall provide positioning guidance in the form of angle and SID. | Reference SRS |
| PRD4.9 | The viewfinder shall provide guidance on the UI to aid in aligning x-ray axis to cassette axis. | Reference SRS |
| PRD4.10 | The viewfinder shall display the non-active area uniquely from the active area. | Reference SRS |
| PRD4.11 | The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm.Reference SRS |
| PRD4.12 | The viewfinder shall include a reference gauge so that the operator understands where the emitter is positioned in reference to the detector. | Reference SRS |
| PRD4.13 | The viewfinder shall show the x-ray field projection for the puck that is selected. | Reference SRS |
| PRD4.14 | The viewfinder shall display the imaging mode (Radiography, Radioscopy, or Photography). | Reference SRS |
| 5. Batteries and Charging |  |  |
| PRD5.1 | The emitter shall contain a rechargeable internal battery pack with integrated BMS. | Reference E1 emitter drawings |
| PRD5.2 | The cassette shall contain a rechargeable internal battery pack with integrated BMS. | Reference C1 cassette drawings |
| PRD5.3 | The emitter shall display the status of the charging system. | Reference SRS |
| PRD5.4 | The cassette shall display the status of the charging system. | Reference SRS |
| PRD5.5 | The emitter and cassette battery packs shall have a rated capacity less than or equal to 100 WHr in order to allow for air transit. | Specification Identical to requirementReference MS-10010 and MS-10083 battery pack drawings |
| PRD5.6 | The emitter fully charged battery shall support 90 minutes of operation without intermittent charging for worst use case. | When subjected to the following use conditions, the Emitter battery shall last >90 minutes:80kV, 0.08mAs; 5s DDR, 15s wait, 5s DDR, 15s wait, 5s DDR, 15s wait, 15min wait. Repeat same DDR sequence every 15 minutes until device powers off. |
| PRD5.7 | The cassette fully charged battery shall support 90 minutes of operation without intermittent charging for worst use case. | When subjected to the following use conditions, the cassette battery shall last >90 minutes:80kV, 0.08mAs; 5s DDR, 15s wait, 5s DDR, 15s wait, 5s DDR, 15s wait, 15min wait. Repeat same DDR sequence every 15 minutes until device powers off. |
| PRD5.8 | The emitter shall be chargeable via wired power connection. | Specification Identical to requirement |
| PRD5.9 | The charging power supplies shall contain ISO 60320 female plug to adapt to US and international plugs/outlets. | Specification Identical to requirementReference H1 Wired Charger drawings |
| PRD5.10 | The charging power supplies shall be compatible with input voltage and frequency ranges 100-240 V and 50-60 Hz. | Specification Identical to requirementReference H1 Wired Charger drawings |
| PRD5.11 | The cassette shall support x-ray emissions while wired charging. | Reference SRS |
| PRD5.12 | The cassette shall be chargeable via wired power connection. | Specification Identical to requirementReference C1 cassette drawings |
| PRD5.13 | The emitter and cassette shall include a coin cell battery that is able to maintain a Real Time Clock (RTC) for a minimum of 3 months. | The minimum capacity of the coin cell battery shall be at least 4.3 mAh. (Reference: MEMO-P01-491 - Jetson RTC Battery Calculation, Rev A) |
| PRD5.14 | The emitter and cassette shall indicate when charging | The emitter and cassette UI shows a lighting bolt when charging |
| 6. Critical Fault Monitoring |  |  |
| PRD6.1 | The device shall perform a startup procedure to check wireless comms and calibration. | Reference SRS |
| 7. Hardware System |  |  |
| PRD7.1 | The device shall work with wireless viewing hardware (wireless tablets and wireless monitors) | Reference SRS |
| PRD7.2 | The device shall work with MedAI supplied and customer supplied Android tablets (with Android 10 or higher) over WiFi. | Reference SRS |
| PRD7.3 | The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. The loading state status shall have a yellow indicator. | Reference SRS |
| PRD7.4 | The emitter shall have an optical camera for the Viewfinder display. | Specification identical to requirementReference E1 emitter drawings |
| PRD7.5 | The emitter shall have an IR optimized camera/sensor for IR Tracking System. | Specification identical to requirementReference E1 emitter drawings |
| PRD7.6 | The emitter shall contain Class 1 lasers to indicate the center of the x-ray field. | Specification identical to requirementReference E1 emitter drawings |
| PRD7.7 | The lasers output power on the emitter shall be between 0.4 and 1.0 mW. | Specification identical to requirement |
| PRD7.8 | The device shall automatically reconnect to a known WiFi network after inputting password the first time. | Reference SRS |
| PRD7.9 | The emitter and cassette shall allow for WiFi connectivity using 2.4GHz and 5GHz bandwidths. | WiFi Module (M50817 ) shall be rated for 2.4GHz and 5GHz bandwidths |
| PRD7.10 | The device shall serve as a private WiFi Access Point. | Reference SRS |
| Device Connections |  |  |
| PRD7.21 | The cassette shall have service port(s), that is covered with a plug and requires a tool to access. | Specification Identical to requirementReference C1 cassette drawings |
| PRD7.22 | The cassette shall have 2 usb-c ports for power input and to connect accessories. | Specification Identical to requirementReference C1 cassette drawings |
| PRD7.23 | The emitter shall have service port(s), that is covered with a plug and requires a tool to access. | Specification Identical to requirementReference E1 emitter drawings |
| PRD7.24 | The emitter shall have a usb-c for power input. | Specification Identical to requirementReference E1 emitter drawings |
| PRD7.25 | The cassette shall have a usb-c port that can support connection to HDMI, Ethernet, and usb-a via a connector adapter. | Specification Identical to requirementReference C1 cassette drawings |
| 8. Software System |  |  |
| PRD8.1 | The device shall allow the operator to switch between different imaging modes. | Reference SRS |
| PRD8.2 | The device shall contain different indicators for each mode. | Reference SRS |
| PRD8.3 | The system should save images upon end of acquisition | Reference SRS |
| PRD8.4 | The device idle state shall be distinguished from active state. | Reference SRS |
| PRD8.5 | The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). | Reference SRS |
| PRD8.6 | The device shall allow sending files to PACS in the DICOM format. | Reference SRS |
| PRD8.7 | The device should provide confirmation that the image study has been successfully submitted to PACS or local storage (USB Drive). | Reference SRS |
| PRD8.8 | The cassette shall be able to send/stream a image(s) to display hardware within 1 second from trigger release. | Reference SRS |
| PRD8.9 | The emitter and tablet shall be able to pair to the cassette, and the foot pedal shall pair to the emitter. | Reference SRS |
| PRD8.10 | The device shall contain debug and release modes for service operators. | Reference SRS |
| PRD8.11 | Removed |  |
| PRD8.12 | Removed |  |
| PRD8.14 | The device shall only initiate x-rays when the computed x-ray field is contained within the image reception area. The device shall terminate x-rays if any part of the projected x-ray field is moved outside the image reception area. | Reference SRS |
| PRD8.15 | The device shall limit the duty-cycle of single radiographs to a maximum of 200ms of exposure and 1800ms minimum of cooldown. | Reference SRS |
| PRD8.16 | The device shall accept hyphens and spaces as part of name inputs. | Reference SRS |
| PRD8.17 | The device shall limit the duty-cycle of serial radiographic and radioscopy mode to a maximum of 20 seconds of duration and proportional cooldown with a maximum of 40 seconds of cooldown. | Reference SRS |
| PRD8.18 | The system shall be display the loading factors (kV, mAs) used for capturing the image. | Reference SRS |
| PRD8.19 | The device shall support image queuing for use off-network and network submission when connected. | Reference SRS |
| PRD8.20 | The system may allow viewing two images at a time for surgical comparison on large monitor(s), and pinning images for comparison. | Reference SRS |
| PRD8.21 | The Mobile Device App shall be compatible with Android devices. | Reference SRS |
| PRD8.22 | The device shall support at least the WPA2 protocol. | Reference SRS |
| PRD8.23 | The device shall enter an idle state when the device is not utilized for 100 seconds. | Reference SRS |
| PRD8.24 | The device shall exit an idle state within 30 seconds upon detection of emitter or foot pedal activity. | Reference SRS |
| PRD8.25 | The device shall disallow x-ray acquisition when the device is in idle state | Reference SRS |
| PRD8.26 | The device shall allow the operator to take and view images without external internet connectivity. | Reference SRS |
| PRD8.27 | The device shall normally prevent or stop x-ray acquisition if there is zero storage space for a full-length capture. Captures confirmed as sent to external storage (e.g. PACS), or images chosen by the user to be deleted, may be deleted before preventing x-rays. | Reference SRS |
| PRD8.28 | The system shall provide a means to document the image orientation on both displayed and stored images | Reference SRS |
| PRD8.29 | The system shall provide a means to document the patient orientation for each image, when appropriate. | Reference SRS |
| PRD8.30 | The live image displayed in fluoroscopy mode shall be displayed with less than a 0.350 second delay from irradiation to image appearance | Reference SRS |
| PRD8.31 | The system shall be able to perform exams during network communication activities (e.g. Sending to PACS) | Reference SRS |
| PRD8.32 | The device shall be able to enter Emergency Radioscopy Mode within 2 minutes of user initiation after a recoverable failure. | The MX1 shall be able to be powered ON and operator hit "Emergency Exam" button in 2 minutesReference SRS |
| PRD8.33 | The device shall be able to recover all functions within 10 minutes. | The MX1 shall be able to be powered ON and operator shall be able to place patient info into to exam, take an x-ray, and view within 10 minutes |
| PRD8.34 | The Exam Screen shall include an "Irradiation Disabling Switch" which, when activated, will disable x-ray emissions until switched off. The Irradiation Disabling Switch may be activated at any time, including in the middle of an imaging sequence. | Reference SRS |
| PRD8.35 | The System shall store all frames of a capture made in either DDR mode and Fluoroscopy Mode | Reference SRS |
| PRD8.36 | The SW system shall include a user-adjustable Timing Device that emits an audible warning after the limit has been exceeded. | Characteristics:- The operator shall be able to set the Timing Device to allow total emission times in an exam of up to 5 minutes without warning.- Any ray tube emission made without the Timing Device having been set shall cause a continuous audible warning signal during the loading.- Any x-ray tube emission made subsequent to the expirary of a previous set period shall cause a continuous audible warning signal during the loading- Resetting the Timing Device shall be possible, even during loading,- Means to control or reset the Timing Device cannot be the triggering switch or button.Reference SRS |
| PRD8.37 | X-ray tube emission shall stop after the control is released and before more than one additional radiation pulse has been emitted. | "Loading Time" here is defined as the time between the start of the first pulse and the end of the last pulse. X-ray tube emission shall stop within 0.1 seconds of releasing any trigger, except when the loading time is less than 0.5 seconds. In that case, the emission may terminate within 0.5 seconds after the control is released. |
| PRD8.38 | The system shall provide means to set a limit, in normal use and no higher than 176 mGy/min, the maximum air kerma rate at the patient entrance reference point. Choosing to emit over this limit, when allowed, is Referred to as High Level Control. | Reference SRS |
| PRD8.41 | RDSR (Radiation Dose Structured Reports) shall be created and exported for each exam, and have the capability to be sent to one or more destinations. | IEC 61910-1 Clauses 5.1.2 and 5.1.3, only "SHALL" features. Ignore Gantry angulations data requirements.Reference SRS |
| PRD8.42 | The system shall limit DDR capture preview to 1 fps | Reference SRS |
| PRD8.43 | The system shall reduce DDR capture preview resolution to 25% of the image | Reference SRS |
| PRD8.44 | The system shall delay the first frame of a DDR capture preview by 2s | Reference SRS |
| PRD8.45 | The system shall intentionally delay the display of frames after the first frame of a DDR capture such that the frame rate is reduced to 1:5 frames. | Reference SRS |
| PRD8.46 | The DDR preview shall display a warning text overlay to signify it as a preview | Reference SRS |
| 9. Software UI |  |  |
| PRD9.1 | All data presented on the software UI shall have a unit of measure or label. | Reference SRS |
| PRD9.2 | The MedAI Device App shall display the manufacturer contact information, a unique UDI, a message to refer to the MX1 IFU, and a warning that primary image interpretation shoul occur on DICOM displays. | Reference SRS |
| PRD9.5 | The software UI should display the image or replay sequence after exposure without the operator interacting with the UI. | Reference SRS |
| PRD9.7 | The software UI shall allow the operator to select and view acquired images. | Reference SRS |
| PRD9.8 | The software UI shall allow the operator to independently manipulate the images. | Reference SRS |
| PRD9.9 | The software UI shall allow the operator to "pinch to zoom" images. | Reference SRS |
| PRD9.10 | The software UI shall allow the operator to rotate images; 360 degrees of rotation in 90 degree increments. | Reference SRS |
| PRD9.11 | The software UI should persist rotation adjustments. | Reference SRS |
| PRD9.16 | The software UI should display the network, device connection status, and signal strength, updating in under 90 seconds. | Reference SRS |
| PRD9.17 | The software UI should display the PACS connection status. | Reference SRS |
| PRD9.21 | The software UI shall display the SID during use. | Reference SRS |
| PRD9.22 | The software UI shall display the dose after each image acquisition. | Reference SRS |
| PRD9.26 | The software UI should inform the operator if any fault occurs. | Reference SRS |
| PRD9.32 | The software UI should indicate the state of the device (e.g. Powered on, Charging, Available for imaging, Emitting radiation, and Error State) | Reference SRS |
| PRD9.35 | The software UI shall allow acquisition workflows while simultaneously uploading studies to PACS | MX1 system is able to take an x-ray image while loading to PACS. |
| PRD9.37 | The software UI shall allow the ability to select a puck before use. | Reference SRS |
| PRD9.38 | The software UI shall display the source-to-skin distance (SSD). | Reference SRS |
| PRD9.39 | The software UI shall allow operator to select puck collimation size from a series of preselected options. | Reference SRS |
| PRD9.40 | The software UI shall allow the user to adjust brightness, contrast, and sharpness of an image. | Reference SRS |
| PRD9.41 | The software UI in non-emergency Radioscopy Mode shall display the Patients name and date of birth as well as the exam start date and time. | Reference SRS |
| PRD9.42 | The software UI shall indicate the available image storage capacity at the beginning of an exam. | Reference SRS |
| PRD9.43 | The software UI shall indiciate to the operator whether there is sufficient storage space to store a complete acquisition after selecting the mode and loading factors but prior to taking an image. | Reference SRS |
| PRD9.44 | All displayed captured on the MedAI App Exam Screen shall be labeled with either "Live" or "Stored", as applicable. | Reference SRS |
| PRD9.45 | Cumulative Air Kerma and Cumulative Dose Area Product during an exam shall be continuously displayed on the Device App and resets between Exams. | - Updated at least every 5 seconds- Accuracy of ±35% when greater than 100mGy, 5µGy*m2, and 6 mGy/min, respectively.Reference SRS |
| PRD9.46 | The live capture shall always appear and be displayed in the same location on the monitor display. | Reference SRS |
| PRD9.47 | The system shall provide an indication for when the x-ray beam axis is normal to the Active Area plane | Reference SRS |
| PRD9.48 | The software UI shall display "Emergency Mode" when the device is being used in Emergency Mode | Reference SRS |
| PRD9.49 | Choosing between Radioscopy and Radiography shall be available on the Device App or Emitter UI | Reference SRS |
| PRD9.50 | The software UI shall indicate when in LOW dose Radioscopy mode | Reference SRS |
| PRD9.51 | If a DDR or Radioscopy Capture is terminated for any reason other than releasing the trigger, the Device App shall notify them that a "Safety Feature" has ended the capture. | Feature to be released in Phase 4 |
| PRD9.52 | When the device is set and positioned to exceed the air kerma maximum chosen for the High Level Control, the system shall emit an audible signal, unique to this warning, continuously. | Reference SRS |
| PRD9.53 | The system shall provide the ability to inactivate any audible signals from the device, except for the High Level Control signal. | Reference SRS |
| PRD9.54 | The Device shall sound an audible signal for initiation of x-ray tube emission. This sound in Fluoroscopy Mode shall be different than that of DDR Mode. | Reference SRS |
| PRD9.55 | The Cumulative Reference Air Kerma and Reference Air Kerma Rate shall be clearly legible 2.5m from the display | Reference SRS |
| PRD9.56 | The Device shall by default display captures in Radiography Mode as light bones on dark background, and in Radioscopy Mode as dark bones on light background. | Reference SRS |
| PRD9.57 | Removed |  |
| PRD9.58 | The Device App shall display the Reference Air Kerma Rate in mGy/min continuously, updated every second, during Radioscopy emission. | - Updated at least every 1 seconds- Accuracy of ±35% when greater than 100mGy, 5µGy*m2, and 6 mGy/min, respectively.Reference SRS |
| 10. HMI |  |  |
| Emitter |  |  |
| PRD10.1 | The emitter keypad shall contain 3 tactile buttons | Specification Identical to requirementReference E1 emitter drawings |
| PRD10.2 | The emitter center button shall allow the operator to select between radiography and photography modes. | Reference SRS |
| PRD10.3 | The emitter left button shall allow the operator to cycle between kV and right button shall allow the operator to cycle between mAs when in manual mode. | Reference SRS |
| PRD10.4 | The emitter keypad buttons shall be controlled by the operator's thumb while holding the emitter with the same hand. | Specification identical to requirement |
| PRD10.5 | The emitter should provide haptic feedback when buttons and trigger are pressed. | The following buttons shall provide haptic feedback when pressed: Trigger 1 (inner handle), Trigger 2 (outer handle), UI buttons for OLED display (qty 3). |
| PRD10.6 | The emitter shall contain 2 trigger(s) for forward and downward x-ray emissions. | Specification Identical to requirementReference E1 emitter drawings |
| PRD10.7 | The emitter trigger(s) shall allow the operator to trigger an x-ray or photograph. Further x-ray or photograph capture shall not be allowed until the trigger is released. | Reference SRS |
| PRD10.8 | The emiter trigger(s) shall be able to actuated with one finger. | Specification Identical to requirement |
| PRD10.9 | The emitter shall contain an LDC display, that is a minimum size of 3.8" diagonally and has a minimum resolution of 720 x 720. | Specification Identical to requirementReference E1 emitter drawings |
| PRD10.10 | The emitter display shall display the viewfinder. | Reference SRS |
| PRD10.11 | The emitter display shall display the remaining battery life in the form of a bar or percent. | Reference SRS |
| PRD10.12 | The emitter display shall display the pairing status of the Foot Pedal. | Reference SRS |
| PRD10.13 | The emitter keypad center button should gracefully shut off the emitter when press/hold for 3 seconds. | Reference SRS |
| PRD10.14 | The emitter keypad center button should hard shut-off the emitter when press/hold for 10 seconds. | Specification Identical to requirement |
| PRD10.15 | The emitter keypad center button should wake the emitter from idle with a single press of button. | Reference SRS |
| PRD10.16 | The emitter should not take longer than 3 seconds to display a response once the power button is pushed. | Reference SRS |
| PRD10.17 | The emitter shall be available for use in less than 180 seconds of initiating power on. | Reference SRS |
| PRD10.18 | The emitter shall indicate when it is ON via the screen or an indicator light. | Reference SRS |
| Cassette |  |  |
| PRD10.19 | The cassette shall contain a Monochrome OLED graphic display, that is at least 55mm x 13mm in size. | Specification Identical to requirementReference C1 cassette drawings |
| PRD10.20 | The cassette shall contain 2 buttons - 'Power Button' and 'Multi-Function Button'. | Specification Identical to requirementReference C1 cassette drawings |
| PRD10.21 | The cassette power button shall power the cassette ON with at least 2 presses. | Specification Identical to requirement |
| PRD10.22 | The casssette should not take longer than 3 seconds to display a response once the power button is pushed. | Reference SRS |
| PRD10.23 | The cassette shall be available for use in less than 180 seconds of initiating power on. | Reference SRS |
| PRD10.24 | The cassette shall indicate when it is ON via the screen or an indicator light. | Reference SRS |
| PRD10.25 | The cassette power button shall power the cassette OFF when press/hold for 3 seconds. | Reference SRS |
| 11. Ergonomics |  |  |
| PRD11.1 | The emitter should be comfortable to hold and move the emitter with all degrees of freedom in usable range during use. | Specification Identical to requirement |
| PRD11.2 | The emitter shall be usable as intended with a left or right hand. | Emitter design shall be symmetrical |
| PRD11.3 | The emitter shall be useable with one or two hands (primary + support hand). | The emitter shall be equal to or less than 8.0 lb |
| PRD11.5 | The emitter shall allow the operator to simultaneously hold the emitter in a downward position and interact with the keypad buttons (via thumb) with one hand. | Specification Identical to requirementReference C1 cassette drawings |
| PRD11.6 | The emitter should be usable in the forward and downward directions. | Specification Identical to requirement |
| PRD11.7 | The cassette active area shall support a minimum static load of 300 lbs with a 2X safety factor (test to 600 lb) | When 600 lb is applied to the cassette over an area of 0.1 m2 for 1 min, the cassette shall not:- Show any damage or permanent deflection greater than 5°. - BASIC SAFETY andESSENTIAL PERFORMANCE shall be maintained as defined by: MEMO-P01-441 |
| PRD11.8 | The weight of the cassette shall be such that it can be easily moved/trasported by a single person. | The cassette shall weigh equal to or less than 15.5 lbs. |
| PRD11.9 | The weight of the packaging case shall be such that it can be easily moved/trasported by a single person. | The packaging case shall weigh equal to or less than 19 lbs. |
| PRD11.10 | The weight of the device and packaging (when combined) shall be such that it can be moved/transported by a single person. | The device and packaging shall weigh equal to or less than 47 lbs (when combined) |
| PRD11.11 | The operator shall be able to view the emitter and cassette indicator LEDs while the device is in use. | Specification Identical to requirementReference C1 cassette and E1 emitter drawings for detailed LED locations |
| PRD11.12 | The weight of the foot pedal shall be such that it is easily moved/transported by a single person. | The foot pedal shall weigh equal to or less than 4 lbs. |
| 12. Packaging and Transportation |  |  |
| PRD12.1 | The device shall be able to be packaged in a reusable hard shell case. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |
| PRD12.2 | The case shall have a handle and wheels to be transported by a single operator. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |
| PRD12.3 | The case shall have a telescoping handle. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |
| PRD12.4 | The case shall incorporate foam or other shock absorbing padding. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |
| PRD12.5 | The case should provide compartments to hold all the fixed and detachable components of the device. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |
| 13. Cleaning, Disinfection, and Sterile Bagging |  |  |
| PRD13.1 | Removed |  |
| PRD13.2 | The operator shall be able to clean all commonly touched surfaces without disassembling the device. | Specification Identical to requirement |
| PRD13.3 | The cleaning procedure shall include standard materials and techniques. | The device shall be able to be cleaned using isopropyl alcohol and Cavicide. |
| PRD13.4 | The device shall be able to be cleaned in less than 5 minutes. | -The MX1 System enclosures materials and geometries must be deemed similar, or easier, to clean and disinfect than the P00 system, as determined by a third party lab. |
| PRD13.5 | The device disinfection time shall be 5 minutes or less with Cavicide. | -The MX1 System enclosures materials and geometries must be deemed similar, or easier, to clean and disinfect than the P00 system, as determined by a third party lab. |
| PRD13.6 | The device enclosures shall have sufficiently smooth outer shell to enable cleaning. | -The MX1 System enclosures materials and geometries must be deemed similar, or easier, to clean and disinfect than the P00 system, as determined by a third party lab. |
| PRD13.7 | The device shall be durable enough to withstand cleaning & disinfection for expected service life of the product. | - The E1 emitter and C1 cassette shall not show any major degradation such as rips, tears, or wear after 1,095 wipes with both Cavicide and 70% Isopropyl alcohol.- The F1 foot pedal and H1 charger shall not show any major degradation such as rips, tears, or wear after 548 wipes with both Cavicide and 70% Isopropyl alcohol. |
| 15. Operating Environment |  |  |
| PRD15.1 | The device shall allow for transportation by air freight (cargo of plane). | Battery packs shall have a rated capacity less than 100 WHr |
| PRD15.2 | The device shall be stored in an ambient temperature of -10C +55C. | Components sensitive to temperature shall be rated for storage within -10C to +55C. |
| PRD15.3 | The device shall be stored in a relative humidity of (non-condensing) 20-90%. | Components sensitive to humidity shall be rated for storage within 20-90% RH. |
| PRD15.4 | The device shall operate within an ambient temperature of 0C to +29.9C. | Components sensitive to temperature shall be rated for use within 0.0 C to +29.9C. |
| PRD15.5 | The device shall operate within a relative humidity of (non-condensing) 20-90%. | Components sensitive to humidity shall be rated for use within 20-90% RH. |
| PRD15.6 | The device shall operate at a pressure of 70 kpa to 106 kpa. | Components sensitive to pressure shall be rated for use within 70 to 106 kpa. |
| 16. Wireless Charger |  |  |
| PRD16.1 | Emitter shall stop wireless charging for the duration of the x-ray acquisition | Reference SRS |
| PRD16.2 | Emitter shall allow x-ray emission when connected to wireless charger | Reference SRS |
| PRD16.3 | Removed |  |
| PRD16.4 | Emitter shall only charge from one source when both wired and wireless chargers are present | Specification Identical to requirementCan be checked by plugging an inline current monitor into both power sources and confirming that both sources don't have current over one amp. |
| PRD16.5 | The wireless charger shall monitor internal temperatures and fail safe upon overtemp. | Overtemp limit set to 70CCan be checked by either setting the temp thershold below ambient or heating up the wireless charger in temp chamber and confirming that the W1 fails safe. |
| 17. Foot Pedal |  |  |
| PRD17.1 | The device shall support the use of a foot pedal with 2 triggers and 2 buttons. | Reference SRS |
| PRD17.2 | The foot pedal right pedal (B) shall initiate a single x-ray exposure upon pressing and releasing when in radiographic mode. | Reference SRS |
| PRD17.3 | The foot pedal right pedal (B) shall initiate DDR on the downpress and shall stop the exposure upon release when in radiographic mode. | Reference SRS |
| PRD17.4 | The foot pedal should be rated to a liquid ingress rating of IPX8 per IEC 60529:1989/AMD2:2013/COR1:2019 | Specification identical to requirementComplies per Third Party Test Lab |
| PRD17.5 | The left button (A) shall switch between Radiography and Photography modes. | Reference SRS |
| PRD17.6 | The foot pedal right button (B) shall rotate the image 90 degrees. | Reference SRS |
| PRD17.7 | The foot pedal left pedal (A) shall "Favorite" or Save the current image. | Reference SRS |
| PRD17.8 | The foot pedal shall be wireless and work at a range up to 12 feet or more away from the emitter. | The foot pedal shall be able to trigger static and dynamic x-rays and change modes at a distance of 12 feet or more from the emitter. |
| 18. Collimation Pucks |  |  |
| PRD18.1 | The device shall come with Collimation Pucks to collimate the x-ray field to smaller fields sizes than the automated collimator;  selectable steps shall be set no more than 0.8 cm apart (nominally) in the length and width when in a plane orthogonal to the reference at a distance of 80cm from the focal spot; the device's minimum selectable size shall not exceed 4 cm in length and width when in a plane orthogonal to the x-ray beam axis at a distance of 80 cm from the focal spot. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm. |
| PRD18.2 | The emitter shall incorporate an attachment mechanism that allows an operator to hold the emitter in one hand and the attach or detach a puck with the other hand. | Specification Identical to requirementReference E1 emitter drawings |
| PRD18.3 | The puck attachment mechanism should incorporate operator positive feedback when a puck is attached. | An audible click shall be heard when a puck is attached to an emitter. |
| PRD18.4 | The pucks shall not obstruct the ToF Sensors or Cameras. | With a puck attached to the emitter (any puck may be used), verify:- Puck does not appear in photograph when a photograph is taken- Puck does appear on viewfinder screen- Displayed SSD measurement is not < 5 cm |
| PRD18.5 | The pucks shall be uniquely identified so the operator can choose the appropriate puck for the desired collimation size. | Specification Identical to requirement |
| PRD18.6 | The pucks shall be packed in their own box and be able to be placed within the case | Specification Identical to requirement |
| 20. Applicable Standards |  |  |
| PRD20.1 | The device design shall include processes defined in ISO 14971 Edition 3 2019, Application of risk management to medical devices | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.2 | The device shall conform to 21 CFR 1020.30:2018, PERFORMANCE STANDARDS FOR IONIZING RADIATION EMITTING PRODUCTS; Diagnostic x-ray systems and their major components. 21 CFR 1020.30(c), (h), (k), (l), (m), (n), and (o) shall be met by conforming to IEC 60601-1-3 and  60601-2-54. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.3 | The device shall conform to 21 CFR 1020.31:2015, PERFORMANCE STANDARDS FOR IONIZING RADIATION EMITTING PRODUCTS; Radiographic equipment by conforming to 60601-1-3 and 60601-2-54. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.4 | The device shall conform to 21 CFR 1020.32:2015, PERFORMANCE STANDARDS FOR IONIZING RADIATION EMITTING PRODUCTS; Fluoroscopic equipment. 21 CFR 1020.32(a), (b), (c), (d)(1), (d)(2), (d)(3)(i) – (iv), (d)(4), (f), (h), (i), (j), and (k) shall be met by conforming to IEC 60601-1-3, IEC 60601-2-54, and IEC 60601-2-43. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.5 | The device shall comply with IEC 60601-1 Edition 3.2 2020 Requirements for Medical Electrical Equipment. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.6 | The device  shall comply with IEC 60601-1-2 Edition 4.1 2020 Requirements for Medical Electrical Equipment. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.7 | The device shall comply with IEC 60601-1-3 Edition 2.2 2021 Requirements for Medical Electrical Equipment. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.8 | The device shall comply with IEC 60601-1-6 Edition 3.2 2020 Requirements for Medical Electrical Equipment, and IEC 62366-1 Edition 1.0 2015 Application of usability engineering to medical devices | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.9 | The device shall comply with IEC 60601-2-28 Edition 3.0 2017 Requirements for x-ray Tube Assemblies. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.10 | The device shall comply with IEC 60601-2-43 Edition 2.2 2019 Particular requirements for the basic safety and essential performance of X-ray equipment for interventional procedures | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.11 | The device shall comply with IEC 60601-2-54 Edition 2.0 2022 Requirements for Medical electrical equipment for radiography. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.12 | The x-ray Tube Assembly shall comply with IEC 60336:2005 for x-ray Tube Assemblies. | Complies with clause 201.7.2.102 of IEC 60601-2-28:2017 |
| PRD20.13 | The device shall comply with IEC 60601-2-43 Edition 2.2 2019 Particular requirements for the basic safety and essential performance of X-ray equipment for interventional procedures | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.14 | The device shall comply with IEC 60522 Edition 2.0 1999 for x-ray Tube Assemblies. | Specification identical to requirement |
| PRD20.15 | The device shall comply with IEC 62304 Edition 1.1 2015 for all software product lifecycle development. | Specification identical to requirement |
| PRD20.16 | The device shall comply with ISO 10993 Edition 5 2018 | Specification identical to requirement |
| PRD20.17 | The device labeling shall comply with 21 CFR 801: Labeling. | Specification identical to requirement |
| PRD20.18 | The device shall comply with IEC 62133-2 Edition 1.0 2017-02 Secondary cells and batteries containing alkaline or other non-acid electrolytes - Safety requirements for portable sealed secondary cells, and for batteries made from them, for use in portable applications - Part 2: Lithium systems. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.19 | The device shall comply with section 38.3 of the UN Manual of Tests and Criteria (UN Transportation Testing) | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.20 | The device shall employ reasonable safeguards to prevent disclosure of any data classified as protected health information by and in accordance with the Health Insurance Portability and Accountability Act of 1996 (HIPAA). | Specification identical to requirement |
| PRD20.21 | The device labeling shall comply with IEC 60825-1 Edition 2.0 2007-03 Safety of laser products - Part 1: Equipment classification, and requirements [Including: Technical Corrigendum 1 (2008), Interpretation Sheet 1 (2007), Interpretation Sheet 2 (2007)]. | Specification identical to requirement |
| PRD20.22 | The device shall comply with Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions, September 27, 2023. | Specification identical to requirement |
| PRD20.23 | The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2022) Digital Imaging and Communications in Medicine (DICOM) Set). | Reference SRS |
| PRD20.24 | The device shall be compliant to ANSI IEEE C63.27-2017 American National Standard For Evaluation Of Wireless Coexistence, and AAMI TIR69:2017/(R2020) Technical Information Report Risk management of radio-frequency wireless coexistence for medical devices and systems. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.25 | The device shall maintain essential performance after exposure to comply with ISTA 3A 2018 conditioning for Packaged-Products for Standard Parcel Delivery System Shipment 70 kg (150 lb) or Less. | After exposure to ISTA 3A conditioning for Standard Packaged Product, the device shall:-Maintain essential performance (per MEMO-P01-441)-Have no visible damage that affects safety or performance of the device |
| PRD20.28 | The device shall comply with CISPR 11:2015 Industrial, scientific and medical equipment - Radio-frequency disturbance characteristics - Limits and methods of measurement. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.29 | The device shall comply with FCC 47 CFR Part 15 RADIO FREQUENCY DEVICES. | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.30 | The optional device tablet display shall comply with DICOM PS3.14 and IEC 62563-1 Edition 1.2 2021 for diagnostic image quality | Specification identical to requirementComplies per Third Party Test Lab |
| PRD20.31 | The device shall comply with EPRC requirements that are not met via conformity to equivalent voluntary consensus standards: 21 CFR 1002 Subparts A, C, D, E, F; 21 CFR 1010.3; 21 CFR 1010.4; 21 CFR 1020.30 (a), (b), (d), (e), (g), (j), and (q); and 21 CFR 1020.31 (i), (d)(3)(v), and (g) | Specification identical to requirement |

### Table 6
|  | 561 |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IFU Order (KEEP) | # | ID | High Level Section | Subsection | Requirement | SOURCE | CLAUSE | Design Output Text [TO BE HIDDEN BEFORE DCO] |
|  | 1.0 | IFU.1 |  |  | DELETED |  |  |  |
|  | 10.0 | IFU.10 |  |  | DELETED |  |  |  |
| 225 | 100.0 | IFU.100 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | Tech Desc shall provide technical specifications of the Tablet. | RSK |  |  |
|  | 101.0 | IFU.101 |  |  | DELETED |  |  |  |
|  | 102.0 | IFU.102 |  |  | DELETED |  |  |  |
|  | 103.0 | IFU.103 |  |  | DELETED |  |  |  |
|  | 104.0 | IFU.104 |  |  | DELETED |  |  |  |
| 154 | 105.0 | IFU.105 | 9 - System Info and Alerts | Troubleshooting | IFU shall include instructions for troubleshooting for use | RSK |  | table |
| 104 | 106.0 | IFU.106 | 7 - Device App | 7 - Device App | IFU shall include instructions for using the Device App UI. | RSK |  | 7 - Device App |
|  | 107.0 | IFU.107 |  |  | DELETED |  |  |  |
| 41 | 108.0 | IFU.108 | 4 - Setting Up the System | Charging | IFU shall include a warning to connect and use only approved devices, components, batteries, and accessories that have been specified as part of or compatible with the System. | 60601-160601-1RSK | 16.216.9.1 | WARNING: Only use MedAI-supplied components and approved accessories. Use or connection of incompatible components or accessories, such as off-the-shelf USB-C chargers, may lead to major shock, burn, or injury. See Section 3 - System Overview, for a list of approved system components and accessories. |
| 86 | 109.0 | IFU.109 | 5 - Using the System | Sterile Coverings | IFU shall recommend using a drape over the Cassette during use. | RSK |  | Coverings are recommended to mitigate equipment damage from liquid ingress and cross-contamination. If performing procedures in which covering is necessary, replace drapes or bags after each use. |
|  | 11.0 | IFU.11 |  |  | DELETED |  |  |  |
|  | 110.0 | IFU.110 |  |  | DELETED |  |  |  |
|  | 111.0 | IFU.111 |  |  | DELETED |  |  |  |
|  | 112.0 | IFU.112 |  |  | DELETED |  |  |  |
|  | 113.0 | IFU.113 |  |  | DELETED |  |  |  |
| 10 | 114.0 | IFU.114 | 2 - General Safety | Electrical Safety | IFU shall warn against disassembling, opening, or modifying the device without authorization of MedAI. | RSK |  | WARNING: Never modify or disassemble any of the system components. Only personnel authorized by MedAI may modify or repair the MX1 System. |
| 232 | 115.0 | IFU.115 | 12 - Tech Specs | Electromagnetic Disturbances | IFU shall caution that the system generates and uses energy that may cause electromagnetic disturbances. | 60601-160601-1RSK | 7.2.137.9.2.2 | CAUTION: This equipment generates, uses, and can radiate radio frequency energy. The system may cause or be subject to radio frequency interference with other medical and non–medical devices and radio communications. There may be risks of reciprocal interference posed by ME Equipment. |
| 126 | 116.0 | IFU.116 | 8 - Radiation Exposure | Overview of Radiation Safety | IFU shall Warn the user to wear PPE, including Radiation-protective PPE. | RSK |  | WARNING: Operators should always wear PPE while using the MX1 System. Both an apron and a thyroid collar are recommended. Follow any additional state and/or hospital-specific safety procedures and PPE requirements. Failure to wear PPE may result in increased exposure to backscatter radiation and overexposure hazards. |
| 12 | 117.0 | IFU.117 | 2 - General Safety | Electrical Safety | IFU shall Caution against spilled liquids or excessive fluids. | 60601-1RSK | 11.6.5 | WARNING: The MX1 System is not waterproof and is only designed to defend against accidental spillage. If you suspect liquids entered the system, do not operate the system, disconnect any chargers from the wall outlet, and contact MedAI for assistance. |
|  | 118.0 | IFU.118 |  |  | DELETED |  |  |  |
|  | 119.0 | IFU.119 |  |  | DELETED |  |  |  |
|  | 12.0 | IFU.12 |  |  | DELETED |  |  |  |
|  | 120.0 | IFU.120 |  |  | DELETED |  |  |  |
|  | 121.0 | IFU.121 |  |  | DELETED |  |  |  |
|  | 122.0 | IFU.122 |  |  | DELETED |  |  |  |
|  | 123.0 | IFU.123 |  |  | DELETED |  |  |  |
| 222 | 124.0 | IFU.124 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | IFU shall specify the input power requirements for the Emitter, Cassette, and Wired Charger. | RSK |  | The MX1 Emitter and Cassette must be charged with the Wired Charger BrickInput Rated Voltage / Frequency: 120 VAC / 50-60 Hz |
| 22 | 125.0 | IFU.125 | 2 - General Safety | Environmental Safety | IFU shall caution against using the System in environments with excessive lighting conditions. | RSK |  | CAUTION: Using the device in direct light may interfere with MX1 Tracking System’s ability to allow X-rays, and may make it difficult to see the Emitter Viewfinder screen. Always ensure settings and controls are visible prior to emitting X-rays.Using the device in excessive lighting conditions (especially infrared lighting, such as from direct sunlight) may interfere with MX1 Tracking System’s ability to allow X-rays, and may make it difficult to see the Emitter Viewfinder screen. Always ensure settings and controls are visible prior to emitting X-rays. |
| 112 | 126.0 | IFU.126 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall include instructions for tagging the orientation of an image. | RSK |  | You may also add annotations such as Left/Right orientation indicators by dragging the element from the Post-Processing Tools onto the image, dropping it in place. |
|  | 127.0 | IFU.127 |  |  | DELETED |  |  |  |
|  | 128.0 | IFU.128 |  |  | DELETED |  |  |  |
| 63 | 129.0 | IFU.129 | 5 - Using the System | Positioning the System - Positioning the Cassette | IFU shall instruct the user to place the Cassette on a stable surface. | RSK |  | The Cassette may be positioned for use on a flat dry surface, or positioned in other configurations by aid from MedAI-supplied accessories, such as the optional Clinical Cart. |
|  | 13.0 | IFU.13 |  |  | DELETED |  |  |  |
| 127 | 130.0 | IFU.130 | 8 - Radiation Exposure | Overview of Radiation Safety | IFU shall include a clause to follow local rules/regulations for radiation safety. | 60601-1-3RSK | 5.2.4.6 | Note: The owner must ensure that all personnel follow radiation safety protocol(s) as dictated by the site in which the MX1 System is used, including personal protective equipment (PPE) and radiation monitoring devices. |
| 173 | 131.0 | IFU.131 | 10 - System Upkeep | Periodic Maintenance Schedule | IFU shall instruct the user to send the system to MedAI for any servicing, calibration, and repairs. | RSK |  | WARNING: The MX1 System is factory calibrated and tested prior to release. There are no field calibration processes required. Should maintenance or repairs be needed, contact MedAI for assistance. |
|  | 132.0 | IFU.132 |  |  | DELETED |  |  |  |
|  | 133.0 | IFU.133 |  |  | DELETED |  |  |  |
| 71 | 134.0 | IFU.134 | 5 - Using the System | Aiming and Collimation - Collimation Pucks | IFU shall instruct the user on how to use the collimating pucks. | RSK |  | Follow these steps to attach and detach a puck to the Emitter:1. Hold the puck with the identifying marking facing away from the Emitter body.2. Place the Puck on the front face of the Emitter, pressing it inside the guides until it snaps into place, held by the Emitter’s Magnets.3. Confirm that the Puck is flush against the Emitter body and the identifying marking is facing away from the Emitter.4. Detach the puck after use by gently pulling the puck from the Emitter’s front face until it detaches. |
|  | 135.0 | IFU.135 |  |  | DELETED |  |  |  |
|  | 136.0 | IFU.136 |  |  | DELETED |  |  |  |
|  | 137.0 | IFU.137 |  |  | DELETED |  |  |  |
|  | 138.0 | IFU.138 |  |  | DELETED |  |  |  |
|  | 139.0 | IFU.139 |  |  | DELETED |  |  |  |
|  | 14.0 | IFU.14 |  |  | DELETED |  |  |  |
|  | 140.0 | IFU.140 |  |  | DELETED |  |  |  |
| 72 | 141.0 | IFU.141 | 5 - Using the System | Aiming and Collimation - Tracking System | IFU shall instruct the user on how to arm and emit radiation from the Emitter. | RSK |  | While in any radiation mode, the Emitter and Cassette will light with green or red colors to indicate whether the tracking system will allow X-ray emission or not. Pressing a trigger will emit X-rays only when the tracking system allows them. There may be a few different reasons for Tracking to not allow X-rays or ‘turn red’:- Emitter Pointed Off-Target (X-ray field would land outside the usable Detector Active Area)- Emitter Too Far from Cassette (SID is larger than 80cm)- Emitter Too Close to Cassette (SID is smaller than 20cm)- The Emitter’s Front Face is covered (Emitter camera cannot see the Cassette IR LEDs)- Cassette LEDs covered with thick material (Emitter camera cannot see the Cassette IR LEDs) |
|  | 142.0 | IFU.142 |  |  | DELETED |  |  |  |
| 116 | 143.0 | IFU.143 | 7 - Device App | Reviewing and Exporting Past Exams - Library Page | IFU shall specify means and limitations of data saves on the device. | RSK |  | The MX1 System is not intended for long-term image storage or archival storage. The total Image Storage Capacity of the MX1 System is XX frames, and the initiation of an exam is not allowed when device storage is full. Failure to properly manage device storage may lead to procedure delay. |
|  | 144.0 | IFU.144 |  |  | DELETED |  |  |  |
| 62 | 145.0 | IFU.145 | 5 - Using the System | Positioning the System - Positioning the CassettePositioning the System - Positioning the Emitter | IFU shall instruct the user on how to position or place the Cassette and Emitter for use. | RSK |  | The Cassette may be positioned for use on a flat, dry surface or positioned in other configurations by aid from MedAI-supplied accessories, such as the optional Clinical Cart. Gently position the patient's anatomy of interest against the Cassette’s Active Area to image. Anatomy positioned outside the Active Area will not be captured in an X-ray image.Note: Do not block more than 50% of the Cassette infrared LEDs, which are required for the tracking system to function.After positioning the Cassette and anatomy, position the Emitter at the desired distance from the Cassette and align the Emitter to the Cassette Active Area. In all radiation modes, the Emitter aiming lasers will turn on when the front face is pointed towards the Cassette. Use the Viewfinder screen to aim, view the projected collimation X-ray field size, and capture the image. |
|  | 146.0 | IFU.146 |  |  | DELETED |  |  |  |
|  | 147.0 | IFU.147 |  |  | DELETED |  |  |  |
|  | 148.0 | IFU.148 |  |  | DELETED |  |  |  |
|  | 149.0 | IFU.149 |  |  | DELETED |  |  |  |
|  | 15.0 | IFU.15 |  |  | DELETED |  |  |  |
| 258 | 151.0 | IFU.151 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | IFU shall Note that PACS configuration is the reponsibility of the operator's organization. | RSK |  | The system is also intended to interface with hospital-specific software such as PACS and hospital networks; configuration of PACS systems and other networks is the user’s responsibility. |
|  | 152.0 | IFU.152 |  |  | DELETED |  |  |  |
|  | 153.0 | IFU.153 |  |  | DELETED |  |  |  |
|  | 154.0 | IFU.154 |  |  | DELETED |  |  |  |
|  | 155.0 | IFU.155 |  |  | DELETED |  |  |  |
|  | 156.0 | IFU.156 |  |  | DELETED |  |  |  |
|  | 157.0 | IFU.157 |  |  | DELETED |  |  |  |
|  | 158.0 | IFU.158 |  |  | DELETED |  |  |  |
|  | 159.0 | IFU.159 |  |  | DELETED |  |  |  |
|  | 16.0 | IFU.16 |  |  | DELETED |  |  |  |
|  | 160.0 | IFU.160 |  |  | DELETED |  |  |  |
|  | 161.0 | IFU.161 |  |  | DELETED |  |  |  |
|  | 163.0 | IFU.163 |  |  | DELETED |  |  |  |
|  | 164.0 | IFU.164 |  |  | DELETED |  |  |  |
|  | 165.0 | IFU.165 |  |  | DELETED |  |  |  |
|  | 166.0 | IFU.166 |  |  | DELETED |  |  |  |
|  | 167.0 | IFU.167 |  |  | DELETED |  |  |  |
|  | 168.0 | IFU.168 |  |  | DELETED |  |  |  |
| 163 | 169.0 | IFU.169 | 10 - System Upkeep | Overview of Cleaning | IFU shall specify cleaning solutions that should not be used to clean the system components. | 60601-160601-1 | 11.6.815.3.7 | CAUTION: Only the cleaning and disinfecting agents listed in these Instructions for Use have been tested for compatibility and effectiveness by MedAI. Do not use other cleaning solutions, as certain chemical combinations may deteriorate the MX1 System plastics prematurely. |
|  | 17.0 | IFU.17 |  |  | DELETED |  |  |  |
| 213 | 171.0 | IFU.171 | 12 - Tech Specs | X-ray Flat Panel Detector Specification and Imaging Performance | IFU shall describe the specifications of the Detector. | 60601-1-3 | 6.7.4 | The MX1 System provides diagnostic-quality images of single radiographic, serial radiographic, and radioscopic exposures according to the Intended Use: |
| 6 | 172.0 | IFU.172 | 1 - Introduction | Owner’s Responsibility | IFU shall instruct that the device must be operated in accordance with local/state/federal laws and regulations. | 60601-2-28 | 201.7.9.1 | In addition to complying with federal guidelines, the owner is responsible for complying with applicable state and local guidelines, which may include:- X-ray device registration and licensing- Operator training program or a radiation worker safety program- Ensuring only qualified personnel are authorized to operate the system |
| 153 | 173.0 | IFU.173 | 9 - System Info and Alerts | Troubleshooting | IFU shall Warn against unauthorized modification or disassembly of the system and that doing so will void the customer warranty and render the system unservicable. | 60601-1 | 7.57.9.3.28.4.4 | Do Not Disassemble: Unauthorized modification or disassembly of the MX1 System will void the customer warranty, resulting in a non-serviceable unit by MedAI. |
| 69 | 174.0 | IFU.174 | 5 - Using the System | Positioning the System - Positioning the Emitter | IFU shall Caution to not place unintended objects not specified to be present as part of normal use in the path of the X-ray beam, including accessories, due to the potential for adverse affects to the image. | 60601-160601-160601-160601-1-360601-2-54 | 7.2.137.512.4.210.2203.10.2 | CAUTION: Do not place objects in the path of the X-ray beam. Doing so may adversely affect the image quality and result in a non-diagnostic exposure. |
|  | 175.0 | IFU.175 |  |  | DELETED |  |  |  |
| 66 | 176.0 | IFU.176 | 5 - Using the System | Positioning the System - Positioning the Cassette | IFU shall Caution to not place more weight or load on the Cassette than the rated weight. | 60601-1 | 9.8.1 | CAUTION: Take care when positioning the patient on the Cassette for weight-bearing images. Do not jump on the Cassette, or allow patients who weigh more than 300 lb or are at risk of tripping, slipping, or falling to stand on the Cassette. |
|  | 177.0 | IFU.177 |  |  | DELETED |  |  |  |
| 35 | 178.0 | IFU.178 | 4 - Setting Up the System | Unpacking | IFU shall warn against setting up or subsequently using the system if any damage is observed or suspected. | 60601-1 | 7.5 | WARNING: DO NOT USE IF DAMAGED. If any part of the device is known (or suspected) to be damaged or defective, do not use the system and contact MedAI for assistance. Operation of the equipment with defective components could expose the operator or the patient to radiation or other safety hazards. This could lead to fatal or other serious personal injury, or to clinical misdiagnosis/mistreatment. |
|  | 18.0 | IFU.18 |  |  | DELETED |  |  |  |
| 64 | 180.0 | IFU.180 | 5 - Using the System | Positioning the System - Positioning the Cassette | IFU shall include instructions for positioning the Cassette and Patient in the Patient Environment for standing or weight-bearing on the Cassette. | RSK |  | If desired, patients up to 300 lbs may stand on the Cassette for weight-bearing images. Place the Cassette on a hard, dry floor with balancing supports nearby if required. |
|  | 181.0 | IFU.181 |  |  | DELETED |  |  |  |
|  | 182.0 | IFU.182 |  |  | DELETED |  |  |  |
|  |  | IFU.183 | 5 - Using the System | Aiming and Collimation - Collimation Pucks | IFU shall explain how to choose a puck for collimation. | RSK |  |  |
|  | 184.0 | IFU.184 |  |  | DELETED |  |  |  |
|  | 185.0 | IFU.185 |  |  | DELETED |  |  |  |
|  | 186.0 | IFU.186 |  |  | DELETED |  |  |  |
|  | 187.0 | IFU.187 |  |  | DELETED |  |  |  |
|  | 188.0 | IFU.188 |  |  | DELETED |  |  |  |
|  | 189.0 | IFU.189 |  |  | DELETED |  |  |  |
|  | 19.0 | IFU.19 |  |  | DELETED |  |  |  |
|  | 191.0 | IFU.191 |  |  | DELETED |  |  |  |
|  | 192.0 | IFU.192 |  |  | DELETED |  |  |  |
|  | 193.0 | IFU.193 |  |  | DELETED |  |  |  |
| 132 | 194.0 | IFU.194 | 8 - Radiation Exposure | Dose Outputs | IFU shall specify the dose output associated with all variations of user-controllable parameters and suggest parameters for anatomy to imaged. | 60601-1-3 | 6.7.2 | tablesAlso 6 - Capturing Radiographs and Photographs, Radiograpoh Mode, Table of Thickness Dose |
|  | 195.0 | IFU.195 |  |  | DELETED |  |  |  |
| 228 | 196.0 | IFU.196 | 12 - Tech Specs | Externally Connected Peripherals | IFU shall state that in the Patient Environment, connected devices and equipment should be 60601-1 certified or should be IEC 60950 or IEC 62368 certified. | 60601-1-260601-1-260601-1-2 | 4.28.18.8 | When operating within the patient environment, equipment that meets the requirements of IEC 60601-1 or an equivalent standard and contain all necessary markings and certificates of conformance should be used. If IEC 60601-1 rated components are not available, operators should use equipment certified in conformance IEC 60950 or IEC 62368. |
|  | 197.0 | IFU.197 |  |  | DELETED |  |  |  |
| 54 | 198.0 | IFU.198 | 5 - Using the System | 5 - Using the System | IFU shall warn the user to check equipment for damage before each use. | RSK |  | WARNING: Inspect equipment for damage before each use. If any damage to the packaging or device is observed, do not proceed with set-up and contact MedAI for assistance. Set up and use of a damaged device may result in minor shock or injury. |
| 198 | 199.0 | IFU.199 | 12 - Tech Specs | X-ray Tube Assembly | IFU shall provide available range of loading factors and subsequent loading factors. | 60601-1-3 | 6.3.26.4.3 | X-ray Tube Loading Factors Range and Accuracy |
|  | 2.0 | IFU.2 |  |  | DELETED |  |  |  |
|  | 20.0 | IFU.20 |  |  | DELETED |  |  |  |
| 109 | 200.0 | IFU.200 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall call out locations in the UI of loading factors. | 60601-1-3 | 6.4.3 | On the bottom left of the Active Capture, the kV, current-time product, dose value, dose area product (DAP), and local date-time of capture are overlaid onto the image. All images are labeled with this information for quality purposes. |
|  | 201.0 | IFU.201 |  |  | DELETED |  |  |  |
|  | 202.0 | IFU.202 |  |  | DELETED |  |  |  |
|  | 203.0 | IFU.203 |  |  | DELETED |  |  |  |
|  | 205.0 | IFU.205 |  |  | DELETED |  |  |  |
|  | 206.0 | IFU.206 |  |  | DELETED |  |  |  |
|  | 207.0 | IFU.207 |  |  | DELETED |  |  |  |
|  | 208.0 | IFU.208 |  |  | DELETED |  |  |  |
| 58 | 209.0 | IFU.209 | 5 - Using the System | Positioning the System | IFU shall include a warning to inspect device after being dropped | 60601-160601-160601-160601-160601-1 | 7.515.3.215.3.315.3.4.215.3.5 | WARNING: If any part of the MX1 System is dropped: Ensure that the patient has not sustained injuryInspect the MX1 System for any damagesWipe down the MX1 System before re-use as described in Section 9 - Routine cleaning Information |
|  | 21.0 | IFU.21 |  |  | DELETED |  |  |  |
|  | 210.0 | IFU.210 |  |  | DELETED |  |  |  |
| 230 | 211.0 | IFU.211 | 12 - Tech Specs | Externally Connected Peripherals | Tech Desc shall warn against connecting the ME System to an external Multiple-Socket Outlet or Extension Cord. | 60601-1 | 7.516.2 | WARNING: Multi-socket outlets or power strips are strictly prohibited for connection unless they are rated to IEC 60601-1 and are provided with all necessary markings and certificates of conformance. Connecting the MX1 System to multi-socket outlets that are not rated to IEC 60601-1 may result in fire. |
|  | 212.0 | IFU.212 |  |  | DELETED |  |  |  |
|  | 213.0 | IFU.213 |  |  | DELETED |  |  |  |
| 123 | 214.0 | IFU.214 | 8 - Radiation Exposure | Overview of Radiation Safety | IFU shall Caution to not move or reposition the Cassette or Emitter during emission of radiation. | RSK |  | CAUTION: Movement of the patient, Cassette, or Emitter during imaging may increase risk of non-diagnostic exposure due to motion blur and/or patient injury. Avoid abrupt shaking or movement of the patient, Cassette, and Emitter. |
|  | 216.0 | IFU.216 |  |  | DELETED |  |  |  |
|  | 217.0 | IFU.217 |  |  | DELETED |  |  |  |
|  | 218.0 | IFU.218 |  |  | DELETED |  |  |  |
|  | 219.0 | IFU.219 |  |  | DELETED |  |  |  |
|  | 22.0 | IFU.22 |  |  | DELETED |  |  |  |
|  | 220.0 | IFU.220 |  |  | DELETED |  |  |  |
|  | 221.0 | IFU.221 |  |  | DELETED |  |  |  |
|  | 222.0 | IFU.222 |  |  | DELETED |  |  |  |
| 17 | 223.0 | IFU.223 | 2 - General Safety | Radiation Safety | IFU shall Warn that only qualified medical personnel who have been trained in the use of medical imaging equipment and who have read this IFU and Accompanying Documents may operate this equipment. | RSK |  | WARNING: This equipment is intended for use by qualified medical personnel who have been trained in the use of medical imaging equipment and who have read the MX1 System IFU and Accompanying Documents. |
|  | 224.0 | IFU.224 |  |  | DELETED |  |  |  |
|  | 225.0 | IFU.225 |  |  | DELETED |  |  |  |
|  | 226.0 | IFU.226 |  |  | DELETED |  |  |  |
| 13 | 227.0 | IFU.227 | 2 - General Safety | Electrical Safety | IFU shall include Caution not to operate device if condensation is suspected within the equipment housing. | 60601-1 | 7.5 | WARNING: If you suspect condensation presence within equipment housing, do not operate the system, disconnect any chargers from the wall outlet, and contact MedAI for assistance. |
|  | 228.0 | IFU.228 |  |  | DELETED |  |  |  |
|  | 229.0 | IFU.229 |  |  | DELETED |  |  |  |
|  | 230.0 | IFU.230 |  |  | DELETED |  |  |  |
| 16 | 233.0 | IFU.233 | 2 - General Safety | Radiation Safety | IFU shall contain the warning: "WARNING: This equipment either produces or is used in the vicinity of ionizing radiation. Observe proper safety procedures according to radiation guidelines laid out by your facility." | 60601-160601-1 | 7.57.6.1 | WARNING: This equipment produces or is used in the vicinity of ionizing radiation. Observe proper safety procedures according to radiation guidelines laid out by your facility. |
|  | 235.0 | IFU.235 |  |  | DELETED |  |  |  |
|  | 236.0 | IFU.236 |  |  | DELETED |  |  |  |
|  | 237.0 | IFU.237 |  |  | DELETED |  |  |  |
|  | 238.0 | IFU.238 |  |  | DELETED |  |  |  |
|  | 239.0 | IFU.239 |  |  | DELETED |  |  |  |
| 68 | 24.0 | IFU.24 | 5 - Using the System | Positioning the System - Positioning the Emitter | IFU shall instruct users to align the Emitter to Cassette Active Area during use. | RSK |  | After positioning the Cassette and anatomy, position the Emitter at the desired distance from the Cassette and align the Emitter to the Cassette Active Area. |
|  | 241.0 | IFU.241 |  |  | DELETED |  |  |  |
|  | 242.0 | IFU.242 |  |  | DELETED |  |  |  |
|  | 243.0 | IFU.243 |  |  | DELETED |  |  |  |
|  | 245.0 | IFU.245 |  |  | DELETED |  |  |  |
|  | 246.0 | IFU.246 |  |  | DELETED |  |  |  |
|  | 247.0 | IFU.247 |  |  | DELETED |  |  |  |
|  | 248.0 | IFU.248 |  |  | DELETED |  |  |  |
| 185 | 249.0 | IFU.249 | 11 - Symbols and Labels | Symbols | IFU shall describe the WEEE Symbol marked on the System component(s). | 60601-1 | 7.57.6.1 | WEEE Symbol |
| 79 | 25.0 | IFU.25 | 5 - Using the System | Capturing an Image | IFU shall Caution to not move or reposition the Cassette or Emitter while in use with a Patient. | RSK |  | CAUTION: Movement of the patient, Cassette, or Emitter during imaging may increase risk of motion blur leading to non-diagnostic exposure and/or patient injury. Avoid abrupt shaking or movement of the patient, Cassette, and Emitter. |
|  | 250.0 | IFU.250 |  |  | DELETED |  |  |  |
|  | 251.0 | IFU.251 |  |  | DELETED |  |  |  |
|  | 252.0 | IFU.252 |  |  | DELETED |  |  |  |
| 184 | 254.0 | IFU.254 | 11 - Symbols and Labels | Symbols | Included symbols shall be official IEC or ISO symbols | 60601-1 | 7.6 | table |
| 255 | 255.0 | IFU.255 | General | General | IFU and Accompanying Documents shall be provided, either hard copy or electronically. Risk assessment needs to assess risk with electronic copy. | 60601-1 | 7.9.1 | General |
| 1 | 256.0 | IFU.256 | 1 - Introduction | 1 - Introduction | IFU shall identify the ME System with its Model Reference. | 60601-160601-160601-1 | 7.9.17.9.2.116.2 | This manual describes operation for the MX1 Portable X-ray System (also referred to as the MX1 System). |
| 2 | 257.0 | IFU.257 | 1 - Introduction | 1 - Introduction | IFU shall specify skills or training required for operation. | 60601-1 | 7.9.1 | The device is intended for qualified medical personnel who have been trained in the use of medical imaging equipment and who have read this Instructions for Use and Accompanying Documents. It is not designed to replace or be a substitute for certified training in the radiological or medical field. |
| 253 | 258.0 | IFU.258 | General | General | IFU needs to be written consistent with the education, training and any special needs of the operator. | 60601-1 | 7.9.1 | General |
| 3 | 259.0 | IFU.259 | 1 - Introduction | Intended UseIndications for Use | IFU shall include the following information on the ME System as intended by the Manufacturer:- Intended Use- Indications for Use- Intended Use Environment. | 60601-160601-160601-1-360601-1 | 7.9.2.116.26.7.2203.6.3.2.102 | The MX1 Portable X-ray System is designed to aid clinicians with point-of-care visualization and guidance during X-rays of extremities and shoulders. It is intended for use in clinical environments and is not intended for surgical applications.The MX1 Portable X-ray System is intended for use by qualified/trained medical professionals on both adult and pediatric patients for diagnostic radiographic, serial radiographic, and interventional fluoroscopic procedures. The device is to be used in healthcare facilities both inside and outside the hospital in a variety of procedures of the extremities and shoulders. |
|  | 26.0 | IFU.26 |  |  | DELETED |  |  |  |
| 53 | 260.0 | IFU.260 | 5 - Using the System | 5 - Using the System | IFU shall include the frequently used functions. | 60601-1 | 7.9.2.1 | 5 - Using the System |
| 4 | 261.0 | IFU.261 | 1 - Introduction | Contraindications | IFU shall include any known contraindication(s) to the use of the ME System, including:- Mammography- Dental applications- Contact with non-intact skin- Cardiac Applications | 60601-1RSK | 7.9.2.1 | The MX1 System is NOT intended for:- Mammography- Dental applications- Contact with non-intact skin- Cardiac Applications |
| 162 | 262.0 | IFU.262 | 10 - System Upkeep | Overview of Cleaning | IFU shall include those parts of the ME System that shall not be serviced or maintained while in use with a Patient. | 60601-1 | 7.9.2.1 | CAUTION: Do not clean any part of the MX1 System or MX1 System accessories while in use with a patient. Opening or cleaning the MX1 system while in use with a patient may result in electrical shock. |
| 5 | 263.0 | IFU.263 | 1 - Introduction | MedAI | IFU shall include the name and address of the Manufacturer. | 60601-1 | 7.9.2.116.2 | MedAI, Inc.1230 Main Street, Suite 300Springfield, IL 60001info@medai.com |
| 221 | 265.0 | IFU.265 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | IFU shall include information about all classifications from Clause 6. | 60601-160601-1 | 7.9.2.17.9.2.5 | The MX1 System components are Internally Powered while not charging and Class II ME Equipment while wired charging (according to IEC 60601-1). The MX1 Cassette is the only Applied Part: Type B. |
| 189 | 266.0 | IFU.266 | 11 - Symbols and Labels | Equipment Labels | IFU shall include all outside markings and their locations on equipment with explanation. | 60601-1 | 7.6.17.9.2.1 | table |
| 186 | 267.0 | IFU.267 | 11 - Symbols and Labels | Symbols | IFU shall include all safety signs/symbols on equipment with explanation. | 60601-1 | 7.6.17.9.2.1 | table |
| 252 | 268.0 | IFU.268 | General | General | IFU shall be written in languages acceptable to the operator. | 60601-1 | 7.9.2.1 | General |
| 190 | 269.0 | IFU.269 | 11 - Symbols and Labels | Equipment Labels | IFU shall include all warning and safety notices | 60601-1 | 7.9.2.2 | table |
|  | 27.0 | IFU.27 |  |  | DELETED |  |  |  |
| 235 | 271.0 | IFU.271 | 12 - Tech Specs | Electromagnetic Disturbances | The IFU shall include information regarding potential electromagnetic or other interference between the ME Equipment and other devices together with advice on ways to avoid or minimize such interference in order to prevent adverse events to the Patient and operator. | 60601-160601-1 | 5.2.2.17.9.2.2 | If this equipment is found to cause interference (which may be determined by switching the equipment on and off), the operator should attempt to correct the problem by one or more of the following measure(s):Reorienting the MX1 System or the affected device;Increasing the distance between the MX1 System or the affected device; orChanging the power supply for either device so they do not share the same power source. |
| 45 | 272.0 | IFU.272 | 4 - Setting Up the System | Charging | The IFU shall state any additional power supplies that are intended to be used with the System and how to connect them. | 60601-1RSK | 7.9.2.3 | Follow these steps to charge the Emitter or Cassette:1. Connect the Wired Charger to a power outlet. The LED light on the Wired Charger will illuminate.2. Connect the Wired Charger to the RIGHT USB-C PORT (indicated with the lightning bolt symbol) on the Cassette, or the USB-C port on the Emitter.3. When the device is on, battery charging indicators for the Emitter and Cassette can be found on their respective screens. Alternatively, the LED lights on the Emitter and Cassette also show their charging status, indicated by a slow pulsing cyan light. This light will be shown at the charging port of the Emitter at all times and on a single LED of the Cassette when off. If not charging, check cable connections.4. Upon charging completion, gently disconnect the Wired Charger from the Emitter and Cassette, pulling directly away from the port to reduce the likelihood of damaging the device or cable connector. |
| 174 | 273.0 | IFU.273 | 10 - System Upkeep | Internal Battery Health | For mains-operated ME Equipment with an additional power source not automatically maintained in a fully usable condition, the IFU shall include a warning statement referring to the necessity for periodic checking or replacement of such an additional power source. | 60601-1 | 7.9.2.4 | Users should monitor the health of the batteries by monitoring the duration to deplete and duration to charge. If you suspect there is something wrong with the battery health of either the Emitter or Cassette, discontinue use and contact MedAI to assess and potentially replace the battery.The optional Foot Pedal contains 3 replaceable C cell batteries that must be replaced upon depletion. Users should monitor the battery life of the Foot Pedal via the battery life indicator.Upon battery depletion, unscrew the black cap by rotating counterclockwise; it should pop off. Replace the depleted C cell batteries with 3 new ones, ensuring that the positive side of all batteries are facing outward. The Foot Pedal will not power on if batteries are not inserted in the correct orientation. Replace the cap by aligning the solid notch of the cap with the solid notch of the Foot Pedal and then pushing inward and rotating the cap fully clockwise as shown.image |
|  | 274.0 | IFU.274 |  |  | DELETED |  |  |  |
|  | 275.0 | IFU.275 | N/A | N/A | If an INTERNAL ELECTRICAL POWER SOURCE is replaceable, the IFU shall state its specification. | 60601-1 | 7.9.2.4 | Not replaceable by user |
|  | 276.0 | IFU.276 |  |  | DELETED |  |  |  |
| 28 | 277.0 | IFU.277 | 3 - System Overview | Major Components | The IFU shall include a brief description of the ME Equipment, how the ME Equipment functions; and the significant physical and performance characteristics of the ME Equipment. | 60601-1 | 7.9.2.5 | System Component and Accessory List |
| 59 | 278.0 | IFU.278 | 5 - Using the System | Positioning the System | If applicable, this description shall include the expected positions of the operator, Patient and other persons near the ME Equipment in normal use. | 60601-1 | 7.9.2.5 | image of a scene showing the operator, patient, and all device components laid out as they would be in normal use |
|  | 279.0 | IFU.279 | N/A | N/A | The IFU shall include information on the materials or ingredients to which the Patient or operator is exposed if such exposure can constitute an unacceptable RISK (see 11.7). | 60601-1 | 7.9.2.5 | There are no materials or ingredients that constitute an unacceptable RISK in this product. |
| 171 | 28.0 | IFU.28 | 10 - System Upkeep | Periodic Maintenance Schedule | IFU shall include directions for how to routinely check for damage. | RSK |  | Always inspect the radiographic captures for image quality issues (spots, blurriness, resolution) during each use. At least once monthly, inspect the external surfaces of all components for damage, loose or missing parts, and frayed or damaged cords. Do not use the device if it displays one or more of the above conditions until the problem is corrected and has been verified as operating correctly and safely.WARNING: If it appears the MX1 System has been damaged, modified, or tampered with in any way, do not use the device and contact MedAI. Use of devices that have been modified or tampered with may result in serious injury. |
| 227 | 280.0 | IFU.280 | 12 - Tech Specs | Externally Connected Peripherals | The IFU shall specify any restrictions on other equipment or network/data couplings, other than those forming part of an ME System, to which a signal input/output part may be connected. | 60601-1 | 7.9.2.5 | All connections to the Cassette must be USB-C compliant. |
|  | 282.0 | IFU.282 | N/A | N/A | If installation of the ME Equipment or its parts is required, the IFU shall contain a reference to where the installation instructions are to be found, or contact information for persons designated by the manufacturer as qualified to perform the installation. | 60601-1 | 7.9.2.6 | N/A |
| 43 | 283.0 | IFU.283 | 4 - Setting Up the System | Charging | If an appliance coupler, mains plug, or other separable plug is used as the isolation means to satisfy 8.11.1 a), the IFU shall contain an instruction not to position the ME Equipment so that it is difficult to operate the disconnection device. | 60601-1 | 7.9.2.7 | CAUTION: Ensure that all system components are posiitioned in such a way that would allow disconnection of power cables in case of emergency. |
| 57 | 284.0 | IFU.284 | 5 - Using the System | Positioning the System | IFU shall contain the necessary information for the operator to bring the ME Equipment into operation including such items as any initial control settings, connection to or positioning of the Patient, etc. | 60601-1 | 7.9.2.8 | 5 - Using the System - Positioning the System |
| 34 | 285.0 | IFU.285 | 4 - Setting Up the System | Unpacking | IFU shall detail any treatment or handling needed before the ME Equipment, its parts, or accessories can be used. | 60601-1 | 7.9.2.8 | Remove the MX1 components from packaging and place them all on a flat surface. Inspect all items for the following:- Obvious signs of damage- Cracked, chipped, or broken components- Sounds of loose internal components- Loose or faulty seals- Missing covers, labels, or windows- Broken or non-functioning buttons |
| 187 | 287.0 | IFU.287 | 11 - Symbols and Labels | Symbols | The meanings of figures, symbols, warning statements, abbreviations and indicator lights on ME Equipment shall be explained in the IFU. | 60601-1 | 7.9.2.9 | table |
| 151 | 288.0 | IFU.288 | 9 - System Info and Alerts | Emitter User Interface MessagesCassette User Interface MessagesDevice App User Interface Messages | The IFU shall list all system messages, error messages and fault messages that are generated, unless these messages are self-explanatory. | 60601-1 | 7.9.2.10 | tables |
| 152 | 289.0 | IFU.289 | 9 - System Info and Alerts | Emitter User Interface MessagesCassette User Interface MessagesDevice App User Interface Messages | The list shall include an explanation of messages including important causes, and possible action(s) by the operator, if any, that are necessary to resolve the situation indicated by the message. | 60601-1 | 7.9.2.10 | tables |
|  | 29.0 | IFU.29 |  |  | DELETED |  |  |  |
| 38 | 290.0 | IFU.290 | 4 - Setting Up the System | Powering Off | The IFU shall contain the necessary information for the operator to safely terminate the operation of the ME Equipment. | 60601-1 | 7.9.2.11 | Safely power down the Emitter and Cassette by pressing and holding their power buttons for two three seconds for graceful shutdown or ten seconds for hard shutdown. Alternatively, navigate to the Component Drawer in a connected Device App and select the power button to power down or sleep connected components. |
| 165 | 291.0 | IFU.291 | 10 - System Upkeep | Routine CleaningDisinfection | IFU shall include instructions for cleaning and disinfecting each item of equipment or equipment part forming part of the ME System (see 11.6.6 and 11.6.7) that can become contaminated during use; list applicable parameters such as temperature, pressure, humidity, time limits and number of cycles that such parts or accessories can tolerate. | 60601-160601-160601-160601-1RSK | 7.9.2.1211.6.511.6.616.2 | Cleaning Instructions (Routine Cleaning, Disinfection, and Emitter Dust Removal) |
| 167 | 292.0 | IFU.292 | 10 - System Upkeep | Periodic Maintenance Schedule | IFU shall instruct on preventive inspection to be performed, including the frequency. | 60601-1 | 7.9.2.1316.2 | Always inspect the radiographic captures for image quality issues (spots, blurriness, resolution) during each use. At least once monthly, inspect the external surfaces of all components for damage, loose or missing parts, and frayed or damaged cords. Do not use the device if it displays one or more of the above conditions until the problem is corrected and has been verified as operating correctly and safely. |
| 169 | 294.0 | IFU.294 | 10 - System Upkeep | Periodic Maintenance Schedule | IFU shall identify the parts on which preventive inspection, maintenance, and calibration shall be performed by service personnel, including the periods, but not including details about the performance. | 60601-1 | 7.9.2.13 | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 176 | 295.0 | IFU.295 | 10 - System Upkeep | Internal Battery Health | For ME Equipment containing rechargeable batteries that are intended to be maintained by anyone other than service personnel, IFU shall contain instructions to ensure adequate maintenance. | 60601-160601-1 | 7.9.2.1315.4.3.2 | WARNING: Batteries in the Emitter and Cassette are not intended to be replaced by users. Battery replacement by inadequately trained personnel could result in excessive temperatures, fire, or explosion. Contact MedAI if you suspect a battery needs replacement.Upon battery depletion, unscrew the black cap by rotating counterclockwise; it should pop off. Replace the depleted C cell batteries with 3 new ones, ensuring that the positive side of all batteries are facing outward. The Foot Pedal will not power on if batteries are not inserted in the correct orientation. Replace the cap by aligning the solid notch of the cap with the solid notch of the Foot Pedal and then pushing inward and rotating the cap fully clockwise as shown. |
| 25 | 296.0 | IFU.296 | 3 - System Overview | Components and Accessories List | The IFU shall include a list of all ME and non-ME Equipment, including accessories, detachable parts and materials, that form and are intended for use with the ME System. | 60601-160601-1 | 7.9.2.1416.2 | System Component and Accessory List |
| 223 | 297.0 | IFU.297 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | If ME Equipment is intended to receive its power from other equipment in an ME System, the IFU shall sufficiently specify such other equipment, including actual transient current level. | 60601-160601-160601-160601-1 | 7.9.2.1416.116.316.9.1 | Nominal Output Power: 100W (typical efficiency 86%)Output Nominal Voltage: 20 VDCInput Rated Voltage / Frequency: 90 - 264 VAC / 50-60 Hz |
| 254 | 299.0 | IFU.299 | General | General | IFU shall contain the Technical Description (See 7.9.3) or reference to where to find it. | 60601-1 | 7.9.2.16 | 12 - Technical Specifications |
|  | 3.0 | IFU.3 |  |  | DELETED |  |  |  |
| 135 | 300.0 | IFU.300 | 8 - Radiation Exposure | Dose Outputs | IFU shall indicate the nature, type, intensity and distribution of emitted radiation. | 60601-1 | 7.9.2.17 | tables |
| 250 | 301.0 | IFU.301 | General | Cover Page | IFU shall contain a unique version identifier such as its date of issue. | 60601-1 | 7.9.2.19 | Rev A |
| 194 | 302.0 | IFU.302 | 12 - Tech Specs | General Specifications | Tech Desc shall provide all data essential for safe operation, transport, and storage, and measures or conditions necessary for installing the ME Equipment, and preparing it for use. | 60601-1 | 7.9.3.1 | Environment Operating Conditions |
| 195 | 303.0 | IFU.303 | 12 - Tech Specs | General Specifications | Tech Desc shall include the permissible environmental conditions of use including conditions for transport and storage (see also 7.2.17). | 60601-160601-160601-160601-1RSK | 5.4 a)7.9.3.115.3.716.2 | Environment Operating Conditions |
| 200 | 304.0 | IFU.304 | 12 - Tech Specs | X-ray Tube Assembly | Tech Desc shall include all characteristics of the ME Equipment, including range(s), accuracy, and precision of the displayed values or an indication where they can be found. | 60601-1 | 7.9.3.1 | Other values presented by the system are accurate to a certain degree: |
|  | 305.0 | IFU.305 | N/A | N/A | Tech Desc shall include any special installation requirements such as the maximum permissible apparent impedance (Distribution network impedance + Power Source impedance) of SUPPLY MAINS. | 60601-1 | 7.9.3.1 | No special installation requirements |
|  | 306.0 | IFU.306 | N/A | N/A | Tech Desc shall include permissible range of values of inlet pressure and flow, and the chemical composition of the cooling liquid if liquid is used for cooling | 60601-1 | 7.9.3.1 | no cooling specifications required |
| 224 | 307.0 | IFU.307 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | Tech Desc shall include a description of the means of isolating the ME Equipment from the Supply Mains, if such means is not incorporated in the ME Equipment | 60601-160601-160601-1 | 7.9.3.17.9.3.48.11.1 | Disconnect the charger from the wall outlet by unplugging the AC Cord from the wall outlet. |
|  | 308.0 | IFU.308 | N/A | N/A | Tech Desc shall include a description of the means for checking the oil level in partially sealed oilfilled ME Equipment or its parts. | 60601-1 | 7.9.3.1 | No oil-filled containers in system |
| 164 | 309.0 | IFU.309 | 10 - System Upkeep | Overview of Cleaning | Tech Desc shall include a warning statement that addresses the Hazards that can result from unauthorized modification of the ME Equipment. | 60601-1 | 7.9.3.1201.7.9.1 | Do Not Disassemble: Unauthorized modification or disassembly of the MX1 System will void the customer warranty, resulting in a non-serviceable unit by MedAI. Only follow cleaning procedures or battery health procedures as described in this Instructions for Use. |
|  | 31.0 | IFU.31 |  |  | DELETED |  |  |  |
| 147 | 310.0 | IFU.310 | 8 - Radiation Exposure | Radiation Reporting Methods | Tech Desc shall include information pertaining to Essential Performance and any necessary recurrent Essential Performance and Basic Safety testing including details of the means, methods and recommended frequency. | 60601-1 | 7.9.3.1 | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
|  | 311.0 | IFU.311 | N/A | N/A | If Tech Desc is separable from the IFU, it shall contain classifications, safety info, description of product, functions, and others | 60601-1 | 7.9.3.1 | N/A |
|  | 312.0 | IFU.312 | N/A | N/A | Tech Desc shall document minimum qualifications, if present, for service personnel | 60601-1 | 7.9.3.2 | No Service Personnel, no maintenence for service personnel |
|  | 313.0 | IFU.313 | N/A | N/A | Tech Desc shall include, as applicable, the required type and full rating of fuses used in the SUPPLY MAINS external to PERMANENTLY INSTALLED ME Equipment. | 60601-1 | 7.9.3.2 | No PERMANENTLY INSTALLED ME Equipment |
|  | 315.0 | IFU.315 | N/A | N/A | Tech Desc shall include, as applicable, instructions for correct replacement of interchangeable or detachable parts that the manufacturer specifies as replaceable by service personnel | 60601-1 | 7.9.3.2 | No Service Personnel, no maintenence for service personnel |
| 177 | 316.0 | IFU.316 | 10 - System Upkeep | Internal Battery Health | Tech Desc shall include, as applicable, where replacement of a component could result in an unacceptable RISK, appropriate warnings that identify the nature of the HAZARD and, if the manufacturer specifies the component as replaceable by service personnel, all information necessary to safely replace the component. | 60601-1 | 7.9.3.2 | CAUTION: If you suspect something is wrong with the battery in any battery-powered component, discontinue use and contact MedAI for assistance. Degraded batteries may lead to procedure delays. Unauthorized disassembly may cause explosions, burns, and electrical hazards to the user. |
|  | 317.0 | IFU.317 | N/A | N/A | Tech Desc shall contain a statement that the manufacturer will make available on request information that will assist service personnel to repair parts designated repairable by service personnel. | 60601-1 | 7.9.3.3 | No Service Personnel, no maintenence for service personnel |
| 14 | 319.0 | IFU.319 | 2 - General Safety | Electrical Safety | IFU shall instruct the operator not to simultaneously touch the Patient and accessible parts that fail leakage test limits, even if unlikely to come into contact. | 60601-1 | 8.4.2 c) | WARNING: Do not touch the patient and any exposed metal components, including device ports and connector pins, simultaneously as electrical discharge may occur. |
|  | 32.0 | IFU.32 |  |  | DELETED |  |  |  |
|  | 320.0 | IFU.320 | N/A | N/A | IFU shall instruct the situations for when to open access covers. | 60601-1 | 8.4.2 c) | N/A |
|  | 321.0 | IFU.321 | N/A | N/A | Means of electrical isolation external to the ME System shall be described in the IFU Technical Description | 60601-1 | 8.11.1 b) |  |
| 30 | 322.0 | IFU.322 | 3 - System Overview | Major Components - Wired Charger | The requirements for the isolation device shall be specified in the Accompanying Documents. | 60601-1 | 8.11.1 | The MX1 Wired Charger isolates from and connects to power outlets to charge the Emitter or Cassette. |
|  | 323.0 | IFU.323 | N/A | N/A | IFU shall describe the use and warnings associated with any moving parts | 60601-1 | 9.2.1 | N/A |
|  | 324.0 | IFU.324 | N/A | N/A | IFU shall specify normal use, including the placement/arrangement of doors, drawers, shelves, and the like | 60601-1 | 9.4.2.2 e) | N/A |
|  | 325.0 | IFU.325 | N/A | N/A | IFU shall describe ME Equipment's transport position and safe working load in that position, if applicable. | 60601-1 | 9.4.2.4.3 | N/A |
|  | 327.0 | IFU.327 | N/A | N/A | IFU shall instruct on how to pass over low obstructions in ME Equipment's transport position, if applicable | 60601-1 | 9.8.3.1 | N/A |
| 65 | 328.0 | IFU.328 | 5 - Using the System | Positioning the System - Positioning the Cassette | The IFU shall describe the allowable Patient mass of any Patient support structure. | 60601-1 | 9.8.3.1 | If desired, patients up to 300 lbs may stand on the Cassette for weight-bearing images. Place the Cassette on a hard, dry floor with balancing supports nearby if required. |
| 197 | 329.0 | IFU.329 | 12 - Tech Specs | General Specifications | The IFU shall disclose the mass of all equipment and accessories. | 60601-1 | 9.8.3.1 | Production Weight and Dimensions |
| 81 | 33.0 | IFU.33 | 5 - Using the System | Foot Pedal | IFU shall instruct the user on how to perform each of the Foot Pedal's intended functions: capturing images, cycling imaging modes, rotate images by 90°, and favorite images. | RSK |  | Use the Foot Pedal to capture images, change imaging modes, rotate images by 90°, and mark or “favorite” images for sending using the buttons described below. You may hold the capture button in either DDR Mode or Fluoro Mode to capture a series of exposures (See Section 6 - Capturing Radiographs and Photographs). |
| 23 | 330.0 | IFU.330 | 2 - General Safety | Hot Surfaces and Temperatures | The IFU shall disclose the maximum temperature reached by an applied part surface and conditions for safe contact (duration, etc); only necessary if 41°C is exceeded. | 60601-1RSK | 11.1.2.2 | CAUTION: The MX1 System device surfaces may reach temperatures up to 43°C (109.4°F) under extended use at the max operating temperature. |
| 161 | 331.0 | IFU.331 | 10 - System Upkeep | Routine CleaningDisinfection | IFU shall specify the effects of multiple cleanings and assure that the specified cleaning procedure would not result in loss of Basic Safety or Essential Performance. | 60601-1 | 11.6.6 | Degradation may have detrimental effects on the Basic Safety and Essential Performance of the system. |
| 259 | 332.0 | IFU.332 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | IFU shall provide instructions for connecting the equipment to an IT-Network, including the purpose, required characteristics, required configurations, technical specifications including security specifications, intended information flow between networking devices and device, and a list of Hazardous Situations resulting from the IT-Network's failure to provide specified characteristics. | 60601-1 | 14.13 |  |
| 242 | 333.0 | IFU.333 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | IFU shall instruct that connection of device to IT-Network that includes other equipment could result in previously unidentified Risks to Patients, operators, or third parties. | 60601-1 | 14.13 | Connection to IT-networks, including other equipment not provided with the MX1 System, could result in previously unidentified risks to patients, operators, or third parties. |
| 243 | 334.0 | IFU.334 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | IFU shall instruct that the User/Reponsible Organization should identify, analyze, evaluate and control Risks resulting from connection of device to IT-Network that includes other equipment. | 60601-1 | 14.13 | The Responsible Organization should identify, analyze, evaluate, and control these risks. |
| 244 | 335.0 | IFU.335 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | IFU shall state subsequent changes to the IT-Network could introduce new risks and require additional analysis. | 60601-1 | 14.13 | Changes to the IT-network could introduce new risks that require additional analysis. |
| 245 | 336.0 | IFU.336 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | IFU shall state that changes to the IT-Network include changes in configuration, connection of additional items, disconnecting items, updating equipment, and upgrading equipment. | 60601-1 | 14.13 | Changes may include:Changes in Network ConfigurationConnection of additional itemsDisconnection of itemsEquipment updates or upgrades |
| 251 | 338.0 | IFU.338 | General | General | IFU shall include the Accompanying Documents for each item of ME Equipment and each item of Non-ME Equipment that is provided as part of the ME System by the Manufacturer (see 7.9). | 60601-1 | 16.2 | General |
| 179 | 34.0 | IFU.34 | 10 - System Upkeep | End of Life Procedure | IFU shall clearly state the device's expected service life, including battery life expectations. | RSK |  | The System has an expected service life of 5 years. |
| 33 | 341.0 | IFU.341 | 4 - Setting Up the System | 4 - Setting Up the System | IFU shall include instructions for the installation, assembly and modification of the ME System to ensure continued compliance with this standard. | 60601-1 | 16.2 | 4 - Setting Up the System |
|  | 343.0 | IFU.343 | N/A | N/A | IFU shall include additional safety measures that should be applied, during installation of the ME System | 60601-1 | 16.2 | N/A |
| 60 | 344.0 | IFU.344 | 5 - Using the System | Positioning the System | IFU shall indicate which parts of the ME System are suitable for use within the Patient Environment. | 60601-1 | 16.2 | image of a scene showing the operator, patient, and all device components laid out as they would be in normal use |
| 40 | 346.0 | IFU.346 | 4 - Setting Up the System | Charging | IFU shall warn against connecting the ME System to an external Multiple-Socket Outlet or Extension Cord. | 60601-1 | 7.516.2 | WARNING: Multi-socket outlets or power strips are strictly prohibited for connection unless they are rated to IEC 60601-1 and are provided with all necessary markings and certificates of conformance. Connecting the MX1 System to multi-socket outlets that are not rated to IEC 60601-1 may result in fire. |
| 15 | 349.0 | IFU.349 | 2 - General Safety | Electrical Safety | IFU shall include instructions to the operator not to touch parts referred to in 16.4 and the Patient simultaneously. | 60601-1 | 16.2 | WARNING: Do not touch the patient and any exposed metal components, including device ports and connector pins, simultaneously as electrical discharge may occur. |
|  | 35.0 | IFU.35 |  |  | DELETED |  |  |  |
| 182 | 351.0 | IFU.351 | 10 - System Upkeep | End of Life Procedure | IFU shall include that the assembly of the ME System and modifications during the actual service life require evaluation to the requirements of this standard. | 60601-1 | 16.2 | Note: The assembly of the MX1 System and modifications during the actual service life require evaluation to requirements of IEC 60601-1 and other applicable safety standards. |
|  | 353.0 | IFU.353 | N/A | N/A | The IFU shall disclose the actual transient current in the technical instruction and installation IFU. | 60601-1 | 16.3 | N/A |
| 146 | 354.0 | IFU.354 | 8 - Radiation Exposure | Radiation Reporting Methods | When dosimetric indications are provided on the equipment (DAP, Dose, etc.), the IFU shall contain information and instructions on how to check and maintain the accuracy. | 60601-1-3 | 5.2.2 |  |
| 191 | 355.0 | IFU.355 | 11 - Symbols and Labels | Equipment Labels | IFU shall include replication of all inaccessible label information marked on items. | 60601-1-3 | 5.2.3 | table |
| 130 | 357.0 | IFU.357 | 8 - Radiation Exposure | Dose Outputs | For each intended use of the equipment, the IFU shall provide the radiation quanitity (like entrance surface dose or DAP) used for describing the radiation dose to the Patient (must be useful for assessing the radiation Risk); the description of a specified test object representative of an average Patient; the procedure for measuring the quantity for the specified test object; the value of the radiation quanitity when the specified test object is used; and the influence of the main selections available (mode, loading factors, etc) to the operator on the value of the specified radiation quanitity. | 60601-1-360601-1-3 | 5.2.4.15.2.4.2 | Air Kerma (Kinetic Energy Released per unit Mass), measured in the units of Gray (Gy), is an indication of the radiation delivered to the patient entrance reference point. The MX1 System determines the patient entrance reference point using the Light Detection and Ranging (LIDAR) array in the MX1 Emitter. This is an accurate representation of Source-to-Skin Distance (SSD) as a point along the central X-ray beam axis. |
| 143 | 358.0 | IFU.358 | 8 - Radiation Exposure | Radiation Reporting Methods | The IFU shall describe (directly or by reference to publication) the method used to provide an indication of radiation dose delivered to the Patient during normal use | 60601-1-360601-1-360601-1-3 | 5.2.4.15.2.4.36.4.5 | Dose measurements provided by the MX1 System are calculated based on the tables in Dose Output. Dose per Air Kerma is normalized to the measured SID, resulting in the most appropriate representation of the dose applied to the patient. This value is reported as µGy, and is accurate to within 30%. |
|  | 359.0 | IFU.359 | N/A | N/A | When clinical protocols are proposed by the MFG and preloaded on the Equipment, the IFU shall state if they constitute recommendations to be applied directly so as to allow optimized operation or if they are only examples/starting points, to be replaced by more specific protocols developed by the user | 60601-1-3 | 5.2.4.4 | AiLARA uses a trained algorithm to automatically change the loading factors (kV and mAs) based on the measured anatomy thickness, SID, and SSD. AiLARA sets the lowest loading factors required to acquire a diagnostic image which reduces overexposure and ensures a clinically relevant radiograph. |
|  | 36.0 | IFU.36 |  |  | DELETED |  |  |  |
| 122 | 360.0 | IFU.360 | 8 - Radiation Exposure | Overview of Radiation Safety | If there is a possibility in normal use that the Patient can be exposed to deterministic radiation dose levels, the IFU shall warn of this fact, and draw attention to the need to manage high radiation doses. | 60601-1-360601-1-360601-2-54 | 5.2.4.15.2.4.5203.5.2.4.5.101 | WARNING: Device settings, protective measures, and imaging techniques during use have a considerable effect on the radiation quality, delivered dose rate, and image quality. In prolonged uses, skin dose levels may be high enough to cause deterministic effects such as skin erythema, skin damage, or hair loss.WARNING: To prevent radiation overexposure hazards, always use the shortest exposure time and lowest voltage settings that produce a clinically relevant image for a given anatomy thickness.  Operators should keep as far as possible away from the X-rayx-ray source to avoid overexposure hazards. |
| 128 | 361.0 | IFU.361 | 8 - Radiation Exposure | Overview of Radiation Safety | IFU shall draw the attention to the need to restrict access to the equipment in accordance with local regulations for radiation protection. | 60601-1-360601-1-3 | 5.2.4.15.2.4.6 | Note: Compliance and caution to federal, state, and local regulations should always be applied, including restricting access to radiation-emitting equipment. |
| 124 | 362.0 | IFU.362 | 8 - Radiation Exposure | Overview of Radiation Safety | All information necessary to minimize the irradiation of the operators in normal use shall be provided. | 60601-1-360601-1-3 | 5.2.4.15.2.4.6 | Precautions should also be taken to minimize the potential for and effects of operator irradiation while using the MX1 System. It is recommended that operators and other participants do the following:- Be aware of levels of dose and stray radiation produced by the system in all configurations of use (See Dose Outputs and Stray Radiation for measured values).- Wear more than one form of radiation personal protective equipment (PPE), including a lead apron, thyroid collar, gloves, or glasses.- Utilize a distanced X-ray acquisition method, such as the optional Foot Pedal, whenever possible. |
| 138 | 363.0 | IFU.363 | 8 - Radiation Exposure | Stray Radiation | For each procedure where operators have to stay in significant zones of occupancy (SZO), the IFU shall provide the radiation dose resulting, means to reduce the dose (modes, loading factors, PPE, use precautions), and a list of PPE for radiation protection including those that may not be included in the equipment. | 60601-1-360601-1-360601-1-3 | 5.2.4.15.2.4.613.1 | Scatter charts/tablesTo minimize risk without adversely affecting the clinical objectives, the ALARA standard should be applied. A general guideline is to apply the lowest X-ray tube voltage (kV) and current-time product (mAs) required to provide acceptable image contrast and exposure.WARNING: Operators should always wear PPE while using the MX1 System. Both an apron (with 0.5 mm lead equivalent) and a thyroid collar are recommended. Follow any additional state and/or hospital-specific safety procedures and PPE requirements. Failure to wear PPE may result in increased exposure to backscatter radiation and overexposure hazards. |
| 144 | 364.0 | IFU.364 | 8 - Radiation Exposure | Radiation Reporting Methods | The IFU shall state the accuracy of radiation output. | 60601-1-3 | 6.3.2 | The below table describes dose outputs for a given tube voltage (kV), current-time product (mAs), and Source-to-Image Distance (SID). |
| 199 | 366.0 | IFU.366 | 12 - Tech Specs | X-ray Tube Assembly | Tech Desc shall provide adequate information available to be referenced by the operator before, during, and after loading of an x-ray tube, regarding loading factors or modes of operation enabling the operator to determine and preselect optimal conditions for irradiation, and subsequently obtain data necessary for estimation of radiation dose received by Patient. | 60601-1-3 | 6.4.3 | X-ray Tube Loading Factors Range and Accuracy |
| 212 | 367.0 | IFU.367 | 12 - Tech Specs | X-ray Flat Panel Detector Specification and Imaging Performance | IFU shall specify and describe intended use with metrics describing imaging performance. | 60601-1-3 | 6.7.2 | The MX1 System provides diagnostic-quality images of single radiographic, serial radiographic, and radioscopic exposures according to the Intended Use: |
| 203 | 368.0 | IFU.368 | 12 - Tech Specs | X-ray Tube Assembly | The Nominal focal spot values of the X-ray tube focal spots in the equipment shall be stated according to IEC 60336:1993 or later versions of IEC 60336 and shall be compatible with each application within the intended use. | 60601-1-3 | 6.7.3 | Focal Spot Size: |
| 214 | 369.0 | IFU.369 | 12 - Tech Specs | X-ray Flat Panel Detector Specification and Imaging Performance | The Detector's contribution to the metrics of imaging performance shall be specified; this contribution should ensure the efficient use of radiation. | 60601-1-3 | 6.7.4 | The MX1 System provides diagnostic-quality images of single radiographic, serial radiographic, and radioscopic exposures according to the Intended Use: |
| 37 | 37.0 | IFU.37 | 4 - Setting Up the System | Powering On | IFU shall caution the user to charge periodically and confirm the battery life of the Emitter and Cassette before use | RSK |  | CAUTION: Failure to sufficiently charge batteries prior to use may result in procedure delay. Check battery charge status indicators prior to use to confirm batteries are charged, and charge the system periodically to prevent unexpected loss of internal power during use. The Emitter will NOT allow x-ray emissions while charging. The Cassette, however, will accept x-rays emissions while charging. |
| 75 | 370.0 | IFU.370 | 5 - Using the System | Aiming and Collimation - Tracking System | IFU shall contain particulars of the values or ranges of the focal spot to image receptor distance specified for normal use. | 60601-1-3 | 8.5.2 | The Emitter will not allow X-ray emission when the tracking system measures the SID as lower than 30 cm or higher than 80 cm. |
| 119 | 371.0 | IFU.371 | 8 - Radiation Exposure | Overview of Radiation Safety | IFU shall include information describe the effects of changes in the SSD on the radiation dose to the Patient | 60601-1-3 | 9.2 | In particular anatomy and system configurations required for an examination, patient skin could be significantly closer to the X-ray source, with dose rate increasing as the inverse square of the Source-to-Skin Distance. |
| 219 | 372.0 | IFU.372 | 12 - Tech Specs | System Radiation Filtration | IFU shall state the maximum value of the Attenuation Equivalent of each item interposed between the Patient and the X-ray image receptor and forming part of the X-ray Equipment, including those parts listed in IEC 60601-2-54 Table 203.104 and are concerned for the measurement conditions specified in IEC 60601-2-54 203.10.101 (values of attenuation equivalent, half-value layer, and quality equivalent filtration are expressed as thicknesses of aluminium of 99,9 % purity or higher). | 60601-1-360601-2-54 | 10.2203.10.2 | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 204 | 374.0 | IFU.374 | 12 - Tech Specs | X-ray Tube Assembly | The Accompanying Documents for all X-ray tube asssemblies and X-ray source assemblies shall state the values of Loading Factors that would, if applied at the Nominal X-ray tube voltage, correspond to the maximum specified energy input to the Anode in one hour. Maximum specified energy input in one hour could be as the value permitted by loading in Radiography at the applicable X-ray tube voltage, according to the Radiographic ratings, corresponding to a total current time product during one hour; or as the value corresponding to the specified Continuoous Anode Input Power. | 60601-1-3 | 12.3 | Maximum (Nominal) Loading Factors for Modes of Operation |
| 140 | 376.0 | IFU.376 | 8 - Radiation Exposure | Stray Radiation | IFU shall designate at least one profile of Stray Radiation in the SZO with respect to height from the floor, with one profile containting the point with the highest dose level. | 60601-1-360601-1-3RSK | 13.113.4 |  |
|  | 377.0 | IFU.377 | N/A | N/A | The Accompanying Documents may be provided with the X-ray TUBE ASSEMBLY, or they may be integrated into the Accompanying Documents of any ME System for which the X-ray TUBE ASSEMBLY is compatible. | 60601-2-28 | 201.7.9.1 | Included |
|  | 378.0 | IFU.378 | N/A | N/A | If an X-ray TUBE ASSEMBLY is intended to receive its power from other equipment in an ME System, or otherwise puts special requirements on the supporting ME System, the Accompanying Documents shall sufficiently specify such other equipment to ensure compliance with the requirements of this document. | 60601-2-28 | 201.7.9.1 | Included |
|  | 379.0 | IFU.379 |  |  | DELETED |  |  | N/A |
| 150 | 38.0 | IFU.38 | 9 - System Info and Alerts | Visual and Audible Indicators | IFU shall describe all visual and audible indications and alerts from the system; and device power and mode states with associated indications | RSK |  | table |
|  | 380.0 | IFU.380 | N/A | N/A | Subclause 7.9.2.3 of the general standard does not apply to the Monoblock | 60601-2-28 | 201.7.9.2.3 | Recorded elsewhere |
|  | 381.0 | IFU.381 | N/A | N/A | The second paragraph and Note of General Standard's 7.9.2.14 do NOT apply | 60601-2-28 | 201.7.9.2.14 | Recorded elsewhere |
|  | 382.0 | IFU.382 | N/A | N/A | Subclause 7.9.2.17 of the general standard does not apply to the Monoblock | 60601-2-28 | 201.7.9.2.17 | Recorded elsewhere |
| 211 | 383.0 | IFU.383 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The IFU shall state the following X-ray tube assembly data as appropriate to the intended use:Single Load Rating;Serial Load Rating;Nominal Radiographic Anode Input Power according to IEC 60613:2010;Nominal CT Anode Input Power according to IEC 60613:2010;Nominal CT Scan Power Index according to IEC 60613:2010. | 60601-2-28 | 201.7.9.2.101 | Maximum (Nominal) Loading Factors for Modes of Operation |
| 201 | 384.0 | IFU.384 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall describe a big list of radiation properties of the tube (See clause for full list) | 60601-2-28 | 201.7.9.3.101 | The X-ray Tube Assembly has the following specifications: |
| 183 | 385.0 | IFU.385 | 10 - System Upkeep | Periodic Maintenance Schedule | IFU shall contain quality control procedures to be performed on the X-ray Equipment by the Responsible Organization for ensuring the quality of X-ray delivery delivery and sensitivity, including acceptance criteria and frequency for the tests. | 60601-2-54 | 201.7.9.1 |  |
| 114 | 386.0 | IFU.386 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall contain a description of image processing applied to original data including the revision number or how to determine it and identification of the version if applicable. | 60601-2-54 | 201.7.9.1 | Captures submitted to a PACS system will have post-processing and annotations saved to the image archive. Post-processing and annotations added from the MX1 system may be reverted while stored in the MX1 system and are not permanent. |
| 117 | 387.0 | IFU.387 | 7 - Device App | Reviewing and Exporting Past Exams - Library Page | IFU shall contain a description of the file transfer format of the images acquired with this unit and of any data associated with these images. | 60601-2-54 | 201.7.9.1 | The MX1 System can transfer images to external systems, such as PACS. Images can be transferred in the DICOM image format along with relevant metadata like patient information, user information, etc. The MX1 System meets conformance requirements for interfacing via DICOM. Images may also be transferred to a local storage device like a USB drive. Images will be stored as JPEG, JPEG2000, and DICOM file formats, with the DICOM file containing associated metadata. |
| 205 | 388.0 | IFU.388 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall state the highest x-ray tube current obtainable when operated at the Nominal x-ray tube voltage for both radioscopy and radiography. | 60601-2-54 | 201.7.9.2.1.101 | Maximum (Nominal) Loading Factors for Modes of Operation |
| 206 | 389.0 | IFU.389 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall state the highest x-ray tube voltage obtainable when operated at the highest x-ray tube current for both radioscopy and radiography. | 60601-2-54 | 201.7.9.2.1.101 | Maximum (Nominal) Loading Factors for Modes of Operation |
|  | 39.0 | IFU.39 |  |  | DELETED |  |  |  |
| 207 | 390.0 | IFU.390 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall state which combination of loading factors results in the highest electric power in the high-voltage circuit for both radioscopy and radiography. | 60601-2-54 | 201.7.9.2.1.101 | Maximum (Nominal) Loading Factors for Modes of Operation |
| 208 | 391.0 | IFU.391 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall state the Nominal electric power with the combiination of loading factors used to calculate the value (80 kV, 0.1 s required per standard). | 60601-2-54 | 201.7.9.2.1.101 | Maximum (Nominal) Loading Factors for Modes of Operation |
| 209 | 392.0 | IFU.392 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall state the lowest mAs achievable by the system. | 60601-2-54 | 201.7.9.2.1.101 | Maximum (Nominal) Loading Factors for Modes of Operation |
|  | 393.0 | IFU.393 |  |  | DELETED |  |  |  |
| 210 | 394.0 | IFU.394 | 12 - Tech Specs | X-ray Tube Assembly | The IFU shall state the maximum symmetrical radiation field of the integrated X-ray source assembly determined according to IEC 60806. | 60601-2-54 | 201.7.9.2.1.102 | Maximum symmetrical radiation field: 21.3 x 21.35 cm “squircle” |
| 215 | 395.0 | IFU.395 | 12 - Tech Specs | X-ray Flat Panel Detector Specification and Imaging Performance | IFU shall contain a description of the particular handling and maintenance of the X-ray image receptor. | 60601-2-54 | 201.7.9.2.1.103 | CAUTION: Users should inspect the radiographic images for image quality issues (spots, blurriness, resolution) during each use. If image quality issues occur, discontinue use of the equipment until the problem is corrected and has been verified to be operating correctly and safely. |
| 131 | 396.0 | IFU.396 | 8 - Radiation Exposure | Dose Outputs | For X-ray Equipment the IFU shall provide information as required in 203.5. | 60601-2-54 | 201.7.9.2.17 |  |
|  | 4.0 | IFU.4 |  |  | DELETED |  |  |  |
|  | 40.0 | IFU.40 |  |  | DELETED |  |  |  |
| 226 | 401.0 | IFU.401 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | IFU shall state the max value of either the apparent resistance of supply mains or other appropriate supply mains specifications used in a facility. | 60601-2-54 | 201.4.10.2 |  |
| 196 | 402.0 | IFU.402 | 12 - Tech Specs | Environment Safety | Where certain unguarded ACCESSIBLE SURFACES of X-ray TUBE ASSEMBLIES can attain high temperatures, means shall be provided to make it impossible to contact such surfaces for any purposes connected with normal use.Measures should be taken to avoid all unintentional contact. In such cases the IFU shall state information about temperatures of ACCESSIBLE SURFACES to be expected in normal use; see Tables 22 to 24 of the general standard. | 60601-2-54 | 201.11.101 | CAUTION: The MX1 System device surfaces may reach temperatures up to 43°C (109.4°F) under extended use at the max operating temperature. This temperature limit is appropriate for the healthy skin of adults but may cause discomfort or minor injury when large areas of the skin (10 % of total body surface or more) are in contact with the hot surface, or if unhealthy skin is in contact with the hot surface. |
|  | 403.0 | IFU.403 | N/A | N/A | The IFU shall draw attention to the RISK of local skin dose levels that cause tissue reactions under the intended use in case of repetitive or prolonged exposure. The effect of the various selectable settings available in both RADIOSCOPY and RADIOGRAPHY on the radiation QUALITY, the delivered REFERENCE AIR KERMA or REFERENCE AIR KERMA RATE shall be described | 60601-2-54 | 203.5.2.4.5.101 | See IFU.403 through IFU.408 |
|  | 404.0 | IFU.404 | N/A | N/A | In the IFU, information shall be provided on the available configurations delivered by the manufacturer such as MODES OF OPERATION, settings of LOADING FACTORS and other operating parameters that affect the radiation QUALITY or the prevailing value of REFERENCE AIR KERMA (RATE) in the intended use. If applicable this information shall include:the MODES OF OPERATION in RADIOSCOPY designated e.g. as normal, low or high resolution, or normal, low or high dose mode;the settings in a typical MODE OF OPERATION, as described in 1), giving the default values, and the available ranges of factors that can be varied after the MODE OF OPERATION has been selected;the settings of LOADING FACTORS and other operating parameters in RADIOSCOPY delivering the highest available REFERENCE AIR KERMA RATE;the settings of LOADING FACTORS and other operating parameters in RADIOGRAPHY delivering the highest available REFERENCE AIR KERMA per frame;the settings of the Focal Spot TO image receptor DISTANCE, corresponding to minimal and typical values of REFERENCE AIR KERMA or REFERENCE AIR KERMA RATE. | 60601-2-54 | 203.5.2.4.5.101 | No Radioscopy |
|  | 405.0 | IFU.405 | N/A | N/A | In the IFU, for the MODES OF OPERATION and sets of values described in accordance with the settings of b) above, representative values of REFERENCE AIR KERMA (RATE) shall be given, based on measurement by the method described in 203.5.2.4.5.102. | 60601-2-54 | 203.5.2.4.5.101 | No Radioscopy |
|  | 406.0 | IFU.406 | N/A | N/A | In addition, representative values of REFERENCE AIR KERMA (RATE) based on measurement by the method described in 203.5.2.4.5.102 shall be given in the IFU, for respectively the MODES OF OPERATION and sets of values described in accordance with the settings of b) 1) and b) 2) of this clause, and if they are adjustable by the operator in the MODE OF OPERATION concerned, for two settings of the following factors:selectable ADDED FILTERS;ENTRANCE FIELD SIZE;X-radiation pulse repetition frequency. | 60601-2-54 | 203.5.2.4.5.101 | No Radioscopy |
|  | 407.0 | IFU.407 | N/A | N/A | The IFU shall include:- test geometries and configurations that can be used to verify the values provided for this subclause (IEC 60601-2-54 Subclause 203.5.2.4.5.101) using the measurement method described in IEC 60601-2-54 Subclause 203.5.2.4.5.102 | 60601-2-54 | 203.5.2.4.5.101 | No Radioscopy |
|  | 408.0 | IFU.408 | N/A | N/A | In the IFU, the location of the Patient ENTRANCE REFERENCE POINT shall be described as specified for the type of RADIOSCOPY Equipment. (SEE CLAUSE FOR DETAILS) | 60601-2-54 | 203.5.2.4.5.101 | No Radioscopy |
|  | 409.0 | IFU.409 | N/A | N/A | X-ray Equipment, except MOBILE X-ray Equipment, shall be provided with connections for external electrical devices separate from the ME Equipment that either can prevent the X-ray GENERATOR from starting to emit X-radiation, can cause the X-ray GENERATOR to stop emitting X-radiation; or both.If the state of the signals from these external electrical devices is not displayed on the CONTROL PANEL, the Accompanying Documents shall contain information for the Responsible Organization that this state should be indicated by visual means in the installation. | 60601-2-54 | 203.6.2.1.102 | table |
| 39 | 41.0 | IFU.41 | 4 - Setting Up the System | Powering Off | IFU shall advise operators to monitor device use prior to powering the device down or controlling the device from a tablet or mobile device. | RSK |  | CAUTION: Prior to powering off the MX1 System device remotely via a mobile device, verify the device is not already in use or being controlled by another mobile device. The MX1 System allows for pairing to multiple tablets/mobile devices at the same time. |
| 172 | 411.0 | IFU.411 | 10 - System Upkeep | Periodic Maintenance Schedule | The IFU shall provide information on the operations required to maintain performance of dosimetric indications within specification. | 60601-2-54 | 203.6.4.5 | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
|  | 412.0 | IFU.412 | N/A | N/A | The IFU shall describe means to achieve an ADDED FILTER, whether placed or permanent, of not less than 0.1 mm Cu or 3.5 mm Al for pediatric applications. | 60601-2-54 | 203.7.1 | N/A |
|  | 413.0 | IFU.413 | N/A | N/A | the Accompanying Documents shall include, in the ASSEMBLING INSTRUCTIONS given for particular applications, instructions for attaining the TOTAL FILTRATION required to comply with subclause 7.1 of IEC 60601-1-3 in respect of the items of X-ray Equipment concerned. | 60601-2-54 | 203.7.1.101 | N/A |
| 149 | 414.0 | IFU.414 | 8 - Radiation Exposure | Collimation Sizing | IFU shall describe means to limit the beam to within the image reception area, per the test in 203.8.5.3 | 60601-2-54 | 203.8.5.3 | The MX1 is equipped with different x-ray beam collimation options. Access the Collimation Menu with a connected Device App through the Acquisition Page or top right Menu button.  There are three methods of collimating: Automatic, Manual Collimator, and Pucks. |
| 73 | 415.0 | IFU.415 | 8 - Radiation Exposure | Aiming and Collimation - Tracking System | The IFU must state If the X-ray beam axis does not coincide with the Reference Axis, according to 203.8.104. | 60601-2-54 | 203.8.101 | Point the Emitter towards the Cassette’s Active Area and Viewfinder will display the X-ray field indicator. If the X-ray field indicator is visible, the tracking system is able to calculate the Emitter’s position, but may not necessarily allow X-rays. The center of the X-ray beam and the beam angle is indicated by the Center Marks. When perpendicular to the detector, the center bubble will snap into the center.The automatic collimator, active by default, will grow or shrink the X-ray field to fit within the bounds of the Active Area. Pointing the Emitter perpendicular to and onto the center of the Cassette Active Area will grow the size of the automatic collimator’s Field Size. The Center Mark will turn red and not allow X-ray emissions when outside the bounds of the Active Area. |
| 78 | 417.0 | IFU.417 | 5 - Using the System | Aiming and Collimation - Tracking System | The IFU shall contain the information necessary to enable the operator to determine, prior to loading, the extent of all X-ray fields for the intended use, in terms of their dimensions at appropriate Focal Spot to image receptor distance for the available selections, combinations and settings of the beam limiting devices. | 60601-1-360601-2-54RSK | 8.5.3203.8.102.3 | The automatic collimator, active by default, will grow or shrink the X-ray field to fit within the bounds of the Active Area. Pointing the Emitter perpendicular to and onto the center of the Cassette Active Area will grow the size of the automatic collimator’s Field Size. The Center Mark will turn red and not allow X-ray emissions when outside the bounds of the Active Area. |
|  | 418.0 | IFU.418 | N/A | N/A | The description of a method to check the dimensions of the LIGHT FIELD at the appropriate distance from the Focal Spot shall be included in the Accompanying Documents. | 60601-2-54 | 203.8.102.5 | N/A |
| 76 | 419.0 | IFU.419 | 5 - Using the System | Aiming and Collimation - Tracking System | The IFU shall describe the positions of the X-ray beam available in normal use, in terms of its locations with respect to relevant image reception areas and its angles with respect to relevant image receptor planes. | 60601-2-54 | 203.8.104 | Point the Emitter towards the Cassette’s Active Area and Viewfinder will display the X-ray field indicator. If the X-ray field indicator is visible, the tracking system is able to calculate the Emitter’s position, but may not necessarily allow X-rays. The center of the X-ray beam and the beam angle is indicated by the Center Marks. When perpendicular to the detector, the center bubble will snap into the center. |
| 24 | 42.0 | IFU.42 | 2 - General Safety | Hot Surfaces and Temperatures | IFU shall Caution to not block or obstruct airflow around fans, or risk overheating. | RSK |  | CAUTION:  Avoid setting the device in positions that may restrict airflow to the MX1 System, including excessive covering with materials such as lead aprons and drapes. Blocking the vents of any component may result in the MX1 reaching its rated heat capacity and a delay in procedure. |
| 74 | 420.0 | IFU.420 | 5 - Using the System | Aiming and Collimation - Tracking System | If the X-ray beam axis is not coinciding with the Reference Axis, the position and the angle of the X-ray FIELD and the plane of interest relative to the Reference Axis shall be described in the IFU. | 60601-2-54 | 203.8.104 | Point the Emitter towards the Cassette’s Active Area and Viewfinder will display the X-ray field indicator. If the X-ray field indicator is visible, the tracking system is able to calculate the Emitter’s position, but may not necessarily allow X-rays. The center of the X-ray beam and the beam angle is indicated by the Center Marks. When perpendicular to the detector, the center bubble will snap into the center. |
|  | 423.0 | IFU.423 |  |  | DELETED |  |  |  |
|  | 424.0 | IFU.424 |  |  | DELETED |  |  |  |
|  | 425.0 | IFU.425 |  |  | DELETED |  |  |  |
|  | 426.0 | IFU.426 |  |  | DELETED |  |  |  |
|  | 427.0 | IFU.427 |  |  | DELETED |  |  |  |
|  | 428.0 | IFU.428 |  |  | DELETED |  |  |  |
|  | 429.0 | IFU.429 |  |  | DELETED |  |  |  |
| 42 | 43.0 | IFU.43 | 4 - Setting Up the System | Charging | IFU shall warn against charging or powering devices outside the specified operation range. | RSK |  | WARNING: Only operate the MX1 System in proper operating environments, including when charging the MX1 System. Not doing so may result in battery damage, electrical hazards, or other safety hazards.  See Section 12 - Technical Specifications. |
|  | 430.0 | IFU.430 |  |  | DELETED |  |  |  |
|  | 431.0 | IFU.431 |  |  | DELETED |  |  |  |
|  | 432.0 | IFU.432 |  |  | DELETED |  |  |  |
|  | 433.0 | IFU.433 |  |  | DELETED |  |  |  |
|  | 434.0 | IFU.434 |  |  | DELETED |  |  |  |
|  | 435.0 | IFU.435 |  |  | DELETED |  |  |  |
|  | 436.0 | IFU.436 |  |  | DELETED |  |  |  |
|  | 437.0 | IFU.437 |  |  | DELETED |  |  |  |
| 188 | 438.0 | IFU.438 | 11 - Symbols and Labels | Symbols | The name and publication date of the standard to which the product was classified shall be included on the explanatory label, on the labels shown in 7.2 to 7.7 or elsewhere in close proximity on the product. For Class 1 and Class 1M, instead of the labels on the product, the information may be contained in the IFU. | 60825-1 | 7.9 | Note: The lasers on the Emitter are a CLASS 1 LASER PRODUCT, per IEC 60825 / Edition 3.0, 2014 |
|  | 439.0 | IFU.439 |  |  | DELETED |  |  |  |
|  | 44.0 | IFU.44 |  |  | DELETED |  |  |  |
|  | 441.0 | IFU.441 |  |  | DELETED |  |  |  |
|  | 442.0 | IFU.442 |  |  | DELETED |  |  |  |
| 238 | 443.0 | IFU.443 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | IFU shall include a summary of the operating characteristics of the wireless technology, effective RF radiated power output and operating range, modulation, and bandwidth of receiving section. | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices |  |
| 248 | 444.0 | IFU.444 | 12 - Tech Specs | Wireless Specifications | IFU shall include a brief description of the wireless QoS needed for safe and effective operation. | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | Quality of Service (QoS) |
| 249 | 445.0 | IFU.445 | 12 - Tech Specs | Wireless Specifications | IFU shall include a brief description of the recommended wireless security measures such as the WPA2 wireless encryption for IEEE 802.11 technology. | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | I see in the table but do we need to add brief description to cybersecurity controls section? |
| 156 | 446.0 | IFU.446 | 9 - System Info and Alerts | Troubleshooting | IFU shall include information addressing wireless issues and what to do if problems occur. | FDA GuidanceWCR | Radio Frequency WirelessTechnology in Medical DevicesWCR.1.5WCR2.3WCR2.4WCR2.5WCR4.1WCR4.2WCR4.3WCR4.4WCR4.5WCR4.6WCR5.1WCR5.2WCR6.1WCR7.1WCR7.2WCR7.3WCR7.4WCR7.5WCR7.6 | In wireless info section( took this from an IFU I had from Philips): If this equipment does cause harmful interference to radio or television reception, which can be determined by moving the equipment away and back, the user is encouraged to try to correct the interference by one or more of the following measures: • Reorient or relocate the receiving antenna• Increase the separation between the equipment and receiver • Consult the dealer or an experienced radio/TV technician for helpAlso this caution:CAUTION: MX1 System has been tested in wireless environments consisting of different wireless technologies (Bluetooth, WiFi 802.11 b and cellular communications) with multiple transmitters used simultaneously. If using the MX1 System in environments where other wireless technologies are being used, the user should evaluate the potential risk of interference. It may be necessary to take mitigation measures such as re-orienting or relocating the MX1 System or shielding the location. |
| 247 | 447.0 | IFU.447 | 12 - Tech Specs | FCC Compliance | IFU shall include information about any wireless coexistence issues and mitigations; this can include precautions for proximity to other wireless products, and specific recommendations for separation distances from such products. | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | This equipment has been tested and found to comply with the limits for a class B digital device, pursuant to part 15 of the FCC Rules. These limits are designed to provide reasonable protection against harmful interference in a residential installation. This equipment generates, uses and can radiate radio frequency energy and if not installed and used in accordance with the instructions, may cause harmful interference to radio communications. However, there is no guarantee that interference will not occur in a particular installation. If this equipment does cause harmful interference to radio or television reception, which can be determined by moving the equipment away and back, the user is encouraged to try to correct the interference by one or more of the following measures: • Reorient or relocate the receiving antenna• Increase the separation between the equipment and receiver • Consult the dealer or an experienced radio/TV technician for help |
| 239 | 448.0 | IFU.448 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | IFU shall include appropriate EMC and telecommunications standards compliance and test results summary. | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | EMC section |
| 246 | 449.0 | IFU.449 | 12 - Tech Specs | FCC Compliance | IFU shall include appropriate RF wireless communications information such as those required by FCC rules. | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | The WiFi internet adapter has been tested and complies with the specifications for a Class B digital device, pursuant to Part 15 of the FCC Rules.  Operation is subject to the following two conditions:(1) This device may not cause harmful interference, and (2) this device must accept any interference received, including interference that may cause undesired operation. |
|  | 45.0 | IFU.45 |  |  | DELETED |  |  |  |
| 233 | 450.0 | IFU.450 | 12 - Tech Specs | Electromagnetic Disturbances | IFU shall include warnings about possible effects from RF sources in the vicinity of the device (e.g., electromagnetic security systems, cellular telephones, RFID or other inband transmitters). | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | CAUTION: MX1 System has been tested in wireless environments consisting of different wireless technologies (Bluetooth, WiFi 802.11 b and cellular communications) with multiple transmitters used simultaneously. If using the MX1 System in environments where other wireless technologies are being used, the user should evaluate the potential risk of interference. It may be necessary to take mitigation measures such as re-orienting or relocating the MX1 System or shielding the location. |
|  | 451.0 | IFU.451 |  |  | DELETED |  |  |  |
|  | 453.0 | IFU.453 |  |  | DELETED |  |  |  |
| 80 | 455.0 | IFU.455 | 5 - Using the System | Foot Pedal | IFU shall clearly indicate each pedal's intended function for the Foot Pedal. | RSK |  | picture |
| 26 | 456.0 | IFU.456 | 3 - System Overview | Components and Accessories List | IFU shall include correlations to the markings and identifiers of all removable sub-assemblies, components, and accessories of the X-ray System to so that they may be readily distinguishable. | 60601-1-3 | 5.1.1 | System Component and Accessory List |
|  | 457.0 | IFU.457 | N/A | N/A | Accompanying Documents include the required statements per sub-clauses in Table 2 | 60601-1-3 | 5.2.1 | See IFU.354 through IFU.376, IFU.470 through IFU.475 |
| 220 | 458.0 | IFU.458 | 12 - Tech Specs | System Radiation Filtration | IFU shall state the Quality Equivalent Filtration (QEF) in thickness of Al or other suitable reference material, the radiation quality used for its determination, and the material (including chemical symbol) for all added filters. | 60601-1-3 | 7.3 | The MX1 System has the following added filter available: |
| 217 | 459.0 | IFU.459 | 12 - Tech Specs | System Radiation Filtration | IFU shall provide the permanent filtration of the X-ray tube assembly or the thicknesses of the materials concerned along with their chemical symbols. | 60601-1-3 | 7.3 | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 160 | 46.0 | IFU.46 | 10 - System Upkeep | Overview of Cleaning | IFU shall caution the user against performing maintenance or servicing beyond routine cleaning. | RSK |  | CAUTION: Do not attempt to perform maintenance or perform component replacement. Opening the MX1 System or MX1 System accessories beyond what is specified in these Instructions for Use may result in electrical shock. Always send MX1 System components to MedAI for service, inspection, and corrective maintenance. |
| 236 | 460.0 | IFU.460 | 12 - Tech Specs | Environmental Statement | A statement of the environments the ME System will be used; relevant exclusions, as determined by Risk Analysis, shall also be listed. | 60601-1-2 | 5.2.1.18.9 | The MX1 System is intended to be used in Professional Healthcare Facility environments. The purchaser or operator of the MX1 System should ensure that it is only used in the appropriate environment. |
| 241 | 461.0 | IFU.461 | 12 - Tech Specs | EMC & Essential Performance | IFU shall include the Essential Performance of ME Equipment and a description of what the operator can expect if the Essential Performance is lost or degraded due to EM disturbances. | 60601-1-2 | 5.2.1.1 | The operator may notice minor monitor flickering or communication issues on the MX1 System during strong electromagnetic events. If these issues persist, it is recommended that the MX1 System be repositioned away from potential sources of noise. It may also be necessary to move the system to a different A/C outlet that is on a different circuit. If the system does not recover after a power reset or shows other signs of malfunction, discontinue use of the equipment immediately.  Remove power to the system by placing the power switch in the off position and unplugging the power cord from the AC receptacle. Notify a qualified technician at MedAI.  Do not operate the system until the service technician advises that it is operating properly. |
| 234 | 462.0 | IFU.462 | 12 - Tech Specs | Electromagnetic Disturbances | IFU shall include a warning regarding stacking and location close to other equipment. | 60601-1-2 | 5.2.1.17.5 | CAUTION: Use of the MX1 System adjacent to or stacked with other equipment could result in device failure and should be avoided. If such use is necessary, observe and verify normal operation of the MX1 System in the configuration in which it will be used prior to use. |
| 27 | 463.0 | IFU.463 | 3 - System Overview | Components and Accessories List | IFU shall include a list of cables, transducers and accessories. | 60601-1-2 | 5.2.1.1 | System Component and Accessory List |
| 229 | 464.0 | IFU.464 | 12 - Tech Specs | Externally Connected Peripherals | IFU shall include a warning that other cables and accessories may negatively affect EMC performance | 60601-1-2 | 5.2.1.1 | WARNING: Other equipment could interfere with the medical device or device system, even if the other equipment complies with CISPR8 emission requirements. |
| 231 | 465.0 | IFU.465 | 12 - Tech Specs | Electromagnetic Disturbances | IFU shall include a statement about portable RF communications Equipment, including antennas, can affect ME Equipment; the warning should include a use distance such as “…be used no closer than 30 cm (12 inches) to any part of the [ME Equipment or ME System], including cables specified by the manufacturer.” | 60601-1-2 | 5.2.1.1 | WARNING: Portable RF communications equipment (including peripherals such as antenna cables and external antennas) should be used no closer than 30 cm (12 inches) to any part of the MX1 System, including cables specified by the manufacturer. Otherwise, performance degradation of the equipment could result. |
| 237 | 467.0 | IFU.467 | 12 - Tech Specs | General Electromagentic Immunity | Tech Desc shall include the compliance for each Emissions and Immunity standard or test specified by this collateral standard, e.g. Emissions class and group and Immunity Test Level. | 60601-1-2 | 5.2.2.1 | General EMC Immunity table |
|  | 468.0 | IFU.468 |  |  | DELETED |  |  |  |
| 240 | 469.0 | IFU.469 | 12 - Tech Specs | EMC & Essential Performance | Tech Desc shall include all necessary instructions for maintaining Basic Safety and Essential Performance with regard to EM disturbances for the expected service life. | 60601-1-2 | 5.2.2.1 | EMC events will not cause unacceptable risks due to degraded essential performance.  Some strong EMC events may require the device to be restarted to exit safety mode and return the device to Nominal functioning. |
| 178 | 47.0 | IFU.47 | 10 - System Upkeep | Storing the System After Use | IFU shall Caution users against placing or storing the system under direct sunlight, hot surfaces, or locations that may get hot. | RSK |  | store in a cool, dry location away from direct sunlight, following the environmental conditions in 12 - Technical Specifications |
| 145 | 470.0 | IFU.470 | 8 - Radiation Exposure | Radiation Reporting Methods | IFU shall provide means to allow the user to estimate the radiation dose delivered to the Patient. This requirement may be satisfied by providing information in the Accompanying Documents, by the indication of dosimetric values or by a combination thereof. The resulting accuracy shall also be specified in the Accompanying Documents. | 60601-1-3 | 6.4.5 |  |
|  | 471.0 | IFU.471 | N/A | N/A | The Accompanying Documents shall state the accuracy of AUTOMATIC CONTROL SYSTEMS. | 60601-1-3 | 6.5 | No AUTOMATIC CONTROL SYSTEM |
|  | 472.0 | IFU.472 | N/A | N/A | Means shall be provided to reduce the influence of radiation scattered in the Patient to the X-ray image receptor in case of significant influence on the image quality. If such means are removable by the operator, their presence or absence shall be clearly visible or indicated to the operator. The proper use of such means shall be described in the IFU. | 60601-1-3 | 6.6 |  |
| 218 | 473.0 | IFU.473 | 12 - Tech Specs | System Radiation Filtration | IFU shall state the Quality Equivalent Filtration (QEF) of fixed layers of material in the X-ray beam incident on the Patient in thickness of Al and the radiation quality used for its determination unless they add altogether to a QEF of no more thant 0.2 mm Al and are not intended to be taken into account as part of Total Filtration required in IEC 60601-1-3 subclause 7.1. | 60601-1-3 | 7.3 | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
|  | 474.0 | IFU.474 | N/A | N/A | IFU shall describe means to adjust controls from a distance when equipment is specified exclusively for not being near the Patient. | 60601-1-3 | 13.2 | Do not have X-ray Equipment specified exclusively for examinations that do not need the operator or staff to be close to the patient |
|  | 475.0 | IFU.475 | N/A | N/A | IFU shall describe means to control radiation when equipment is specified exclusively for not being near the Patient. | 60601-1-3 | 13.3 | Do not have X-ray Equipment specified exclusively for examinations that do not need the operator or staff to be close to the patient |
| 157 | 477.0 | IFU.477 | 9 - System Info and Alerts | Troubleshooting | IFU shall include instructions for NFC troubleshooting. | WCR | WCR2.1WCR2.2 |  |
| 46 | 478.0 | IFU.478 | 4 - Setting Up the System | Pairing the Emitter, Cassette, and Foot Pedal | IFU shall instruct to check for visual indication of Emitter, Cassette, and Foot Pedal pairing. | WCR | WCR2.7WCR2.8WCR2.9WCR2.10 | The Emitter, Cassette, and Foot Pedal will be delivered paired by MedAI. Confirm that all components are powered on. If properly connected, the Emitter Viewfinder screen will show icons for the Cassette and Foot Pedal in the top left corner:image |
| 82 | 479.0 | IFU.479 | 5 - Using the System | Foot Pedal | IFU shall specify max allowable distance for operation of the Foot Pedal to Emitter. | WCR | WCR3.2WCR3.6WCR3.7WCR6.2 | Note: The Foot Pedal may be used to trigger X-rays from a safe distance via wireless communication with the Emitter. The maximum distance the Foot Pedal can be from the Emitter is 3.5 meters (12 feet). Before beginning an Exam, ensure that the Foot Pedal is positioned within this operational range to guarantee proper functionality. |
| 29 | 48.0 | IFU.48 | 3 - System Overview | Major Components - Emitter | IFU shall describe the Emitter's laser class and Caution the user to not point the Emitter's lasers near eyes. | 60601-160601-1RSK | 7.2.1312.4.2 | WARNING: While the system is actively tracking or capturing, the Emitter is emitting potentially hazardous energy such as radiation, Class 1 visible lasers, and Class 1 non-visible infrared light. Never look directly into these lasers/lights or point them at others; eye exposure to hazardous energy sources may result in serious eye injury. |
| 158 | 481.0 | IFU.481 | 9 - System Info and Alerts | Troubleshooting | IFU shall instruct the user to contact MedAI when experiencing frequent Emitter overheating or when something is caught in the Emitter fan. | RSK |  | The Cassette or Emitter overheats often - The cooling means inside the component is not functioning correctly - Discontinue use of the device and contact MedAI.An item gets jammed or stuck inside or under the Emitter’s fan cover - Discontinue use of the device and contact MedAI. |
| 44 | 482.0 | IFU.482 | 4 - Setting Up the System | Charging | IFU shall have prominent images and indicators for Emitter and Cassette charging ports. | RSK |  | images |
|  | 483.0 | IFU.483 | 4 - Setting Up the System | N/A | IFU shall have prominent images and indicators for proper Cassette orientation. | RSK | R6.16 | Images indicating proper Cassette orientation to be placed in the IFU |
|  | 484.0 | IFU.484 |  |  | DELETED |  |  |  |
| 166 | 485.0 | IFU.485 | 10 - System Upkeep | Routine Cleaning | The IFU shall specify a procedure for the removal of dust from the Emitter | 60601-2-43 | 201.11.6.5.102 | 1. Shut down the Emitter before cleaning and disconnect any cords, connections, or other accessories.2. To remove the Emitter’s fan cover, first remove the four screw covers and then, using a #1 Phillips Head screwdriver, unscrew all four screws; set aside all components.3. Clean around the fan and between the fan blades using a can of compressed air.4. Once all visible dust has been removed, put the fan cover back on and replace all four screws using the screwdriver, ensuring that the screws are appropriately tightened.5. Replace all four screw covers. |
| 97 | 486.0 | IFU.486 | 6 - Imaging Modes | Dynamic Digital Radiography (DDR) | The IFU shall state that if Radiography Mode(s) are intentionally misued for real-time imaging, the Image Display Delay may be longer than in Radioscopy Mode. | 60601-2-43 | 201.12.4.102 | DDR Mode is not intended to be used for real-time imaging or real-time guidance. To mitigate this: The frames of a DDR shown on the MedAI Device App mid-capture are displayed with Frame rate limited to 2.5 frames per second.75% reduction in resolution.A delay in image preview during capture.A warning text overlay to signify it as a preview and not for real-time guidance. |
| 90 | 487.0 | IFU.487 | 5 - Using the System | Emergency Instructions | The IFU shall indicate: The time necessary to initiate emergency radioscopy mode after a recoverable failure, the time to restore all functions of the system after a recoverable failure, and the required procedure(s) for recovering recoverable failures. | 60601-2-43 | 201.4.101 | Time to Boot into Emergency Radioscopy: 2 minutesTime to Recover Normal Operation: 2 minutesIn the case of a recoverable failure requiring operator intervention, follow these manual recovery steps: |
| 257 | 488.0 | IFU.488 | General | General | The IFU shall include how to check the software version and describe the file format of the images | 60601-2-54 | 201.7.9.1 |  |
| 103 | 489.0 | IFU.489 | 6 - Imaging Modes | Radiation Controls and Audible Signals - Maximum Allowable Air Kerma Rate | The IFU shall describe means to adjust or inactivate the signals for a termination of Radioscopy or Radiography Loadings, NOT including the Timing Device or High Level Control audible signal described in IEC 60601-2-54 203.6.3.102 | 60601-2-43 | 203.6.4.2 | These audible signals may be enabled or disabled in the MedAI Device App settings. See Section 7 - Device App for the steps. In addition to normal emission signals, the MX1 System provides configurable audible radiation warnings to help control radiation dose delivered to the patient: Loading Time Limiter and Radiation Rate Limiter.The Loading Time Limiter and the Radiation Rate Limiter can be configured but not disabled. |
|  | 49.0 | IFU.49 |  |  | DELETED |  |  |  |
| 125 | 491.0 | IFU.491 | 8 - Radiation Exposure | Overview of Radiation Safety | IFU shall include a list of recommended radiation protective devices or accessories to be used during Radioscopically guided interventional procedures; there may be different lists for different kinds of procedures. | 60601-2-43 | 201.7.9.2.101 | Wear more than one form of radiation personal protective equipment (PPE), including a lead apron, thyroid collar, gloves, or glasses. |
| 89 | 492.0 | IFU.492 | 5 - Using the System | Emergency Instructions | The IFU shall include a reproduction of the Emergency Instructions. | 60601-2-43 | 201.7.9.2.103 | emergency instructions |
|  | 493.0 | IFU.493 |  |  | DELETED |  |  |  |
|  | 494.0 | IFU.494 |  |  | DELETED |  |  |  |
|  | 495.0 | IFU.495 |  |  | DELETED |  |  |  |
|  | 496.0 | IFU.496 |  |  | DELETED |  |  |  |
|  | 497.0 | IFU.497 |  |  | DELETED |  |  |  |
|  | 498.0 | IFU.498 |  |  | DELETED |  |  |  |
| 202 | 499.0 | IFU.499 | 12 - Tech Specs | X-ray Tube Assembly | Tech Desc shall contain detail on Reference Axis, Target Angle(s), position and tolerance of the Focal Spot(s), and the Focal Spot Size(s), including if they are considered Nominal Focal Spot Value(s) according to IEC 60336. | 60601-2-54 | 201.7.9.3.101 | Target Angle: 26 Degrees |
|  | 5.0 | IFU.5 |  |  | DELETED |  |  |  |
| 9 | 50.0 | IFU.50 | 2 - General Safety | Electrical Safety | IFU shall Caution against using the device if the user suspects damage or tampering. | RSK |  | WARNING: DO NOT USE IF DAMAGED If any part of the device is known (or suspected) to be damaged or defective, do not use the system and contact MedAI for assistance. Operation of the equipment with defective components could expose the operator or the patient to radiation or other safety hazards. This could lead to fatal or other serious personal injury, or to clinical misdiagnosis/mistreatment. |
| 139 | 501.0 | IFU.501 | 8 - Radiation Exposure | Stray Radiation | A Significant Zone of Occupancy (SZO) (60x60x200cm minimum) shall be designated and described in the IFU, along with details on the types of examinations for which it is used, its location relative to the System, and the effectiveness and application of protective devices specified to be used with the System. | 60601-1-360601-1-3 | 13.113.4 | The Operator Zone or Significant Zone of Occupancy was established for the handheld use and the hands-free use:Handheld: A 60 cm x 60 cm square with a height of 200 cm, with an additional 20 cm x 50 cm x 50 cm volume connecting the Emitter handle to represent an operator’s arm.Hands-free: A 60 cm x 60 cm square with a height of 200 cm, distanced 3.7m from the focal spot, without any connecting volume to the Emitter handle. |
| 142 | 502.0 | IFU.502 | 8 - Radiation Exposure | Stray Radiation | The IFU shall describe the test arrangement used to measure the Scatter Radiation map | 60601-1-3 | 13.4 |  |
| 137 | 504.0 | IFU.504 | 8 - Radiation Exposure | Dose Outputs | The IFU shall provide Test Geometries and configurations that can be used to verify Reference Air Kerma and Air Kerma Rates for user-adjustable settings using the procedure specified in 203.5.2.4.5.102. | 60601-2-43 | 203.5.2.4.5.101 |  |
| 133 | 505.0 | IFU.505 | 8 - Radiation Exposure | Dose Outputs | The IFU shall provide Reference Air Kerma and Reference Air Kerma Rate Values of each user-adjustable setting or mode, with added filters or field sizes. | 60601-2-43 | 203.5.2.4.5.101 |  |
| 134 | 506.0 | IFU.506 | 8 - Radiation Exposure | Dose Outputs | The IFU shall provide one set of Reference Air Kerma and Reference Air Kerma Rate values typical of Radiography for distinctive types of procedure for which the system is intended for use | 60601-2-43 | 203.5.2.4.5.101 |  |
| 129 | 507.0 | IFU.507 | 8 - Radiation Exposure | Dose Outputs | The IFU shall describe the Patient entrance reference point location. | 60601-2-4360601-2-54 | 203.5.2.4.5.101201.7.9.2.17 | Dose values listed below and reported by the MX1 System are at the Patient Entrance Reference Point (PERP) of 15 cm above the surface of the flat panel detector. This translates to XX cm above the outer surface of the Cassette. |
| 136 | 508.0 | IFU.508 | 8 - Radiation Exposure | Dose Outputs | The IFU shall state the user-adjustable settings that affect radiation output and how, including:- The low and normal modes of operation in Radioscopy- The Settings in Radioscopy that can be varied and impact dose or image quality- The Settings that would generate the highest Reference Air Kerma Rate, in Radioscopy- The Settings that would generate the highest Reference Air Kerma per image, in Radiography- The SID corresponding to minimal and typical values of Reference Air Kerma or Reference Air Kerma Rate | 60601-1-360601-2-54 | 12.3203.5.2.4.5.101 |  |
| 21 | 51.0 | IFU.51 | 2 - General Safety | Environmental Safety | IFU shall warn against using the System in a volatile atmospheric environment, including O2-rich environments. | RSK |  | WARNING: Do not use this equipment in environments rich with oxygen, nitrous oxide, or flammable anesthetics. Use in potentially flammable environments may lead to fire. |
| 96 | 510.0 | IFU.510 | 6 - Imaging Modes | Dynamic Digital Radiography (DDR)Fluoroscopy (Fluoro) Mode | The IFU shall describe the minimum and maximum loading times and the corresponding controls in (at least) Radioscopy, if available. | 60601-2-54 | 203.6.2.1 | A maximum of 20 seconds of exposure at a time is allowed on this system, with a cooldown time of 40 seconds. |
| 70 | 512.0 | IFU.512 | 5 - Using the System | Aiming and Collimation - Automatic Collimator | The IFU shall describe methods to check the operation of the automatic collimator and reduce the size to a selectable size, including via the application of collimation pucks. | 60601-2-54 | 203.8.102.1 | Note: Confirm that the automatic collimator is functioning by switching to a radiation mode and aiming the Emitter around the Cassette Active Area. The Viewfinder should show the automatically updating X-ray field indicator. |
| 256 | 513.0 | IFU.513 | General | General | The Emergency Instructions shall be provided in a non-electronic form that is resistant to damage. | 60601-2-43 | 201.7.9.2.103 |  |
| 91 | 514.0 | IFU.514 | 5 - Using the System | Emergency Instructions | The Emergency Instructions shall include instruction for restart or recovery in case of recoverable failure or failure of SUPPLY MAINS,  instruction for location, function, and operation of the IRRADIATION disabling switch, and a list of emergency functions. | 60601-2-43 | 201.7.9.2.103 | emergency instructions |
| 193 | 515.0 | IFU.515 | 12 - Tech Specs | General Specifications | The IFU shall explain any IPXY marking on MX1 components | 60601-2-43 | 201.7.9.2.105 | Degree of protection against ingress of solid foreign objects and/or water: |
| 92 | 516.0 | IFU.516 | 5 - Using the System | Emergency Instructions | The IFU shall include instructions to configure the system to permit CPR. | 60601-2-43 | 201.7.9.2.102 | The MX1 System is not indicated for use in cardiac applications and is not rated to support the forces incurred during CPR. Follow these steps to configure the MX1 System for CPR:1. Remove the Emitter and any MedAI-supplied accessory from the patient space.3. Gently lifting the patient anatomy, remove the Cassette from the patient space.4. Ensure there is sufficient space available around the patient for the unimpeded conduct of CPR. |
| 192 | 517.0 | IFU.517 | 12 - Tech Specs | 12 - Tech Specs | The IFU shall include technical descriptions with information necessary to maintain compliance with IEC 60601-1-3 standard within the relevant main assemblies for items supplied separately from the main assembly. | 60601-1-3 | 5.2.3 | 12 - Tech Specs |
| 110 | 518.0 | IFU.518 | 7 - Device App | Performing an Exam - Acquisition Page | The IFU shall explain the function of the irradiation disabling switch and recommend not using the irradiation disabling switch during an exam. | 60601-2-43 | 203.5.2.4.101 | The red STOP X-RAYS button is located next to the COMPLETE button on the Acquisition Page. Press this button to disable and prevent the emission of X-rays.The STOP X-RAYS button is intended to be used to disallow the emission of radiation at any time to prevent unintended triggering of X-rays. Using this switch during an in-progress exam is not recommended and may cause procedure delay. |
| 85 | 52.0 | IFU.52 | 5 - Using the System | Sterile Coverings | IFU shall provide instructions on how to drape the Cassette. | RSK |  | Lay the drape over the entire Cassette top face before use. Use a single-layer drape that lies flat on the MX1 Cassette such that the operator can see the visible-light LEDs through the drape. If the drape obscures or distorts the infrared (non-visible) LEDs, the system may not work as intended or may prevent X-ray emissions; if this occurs reposition the drape. |
| 141 | 522.0 | IFU.522 | 8 - Radiation Exposure | Stray Radiation | Stray (Scatter) radiation shall be tested and reported in the IFU, using the test setup in the standard; review IEC 60601-2-54 203.13.6 (for stray radiation map test procedure) and IEC 60601-2-43 203.13.6 (for isokerma test procedure). | 60601-1-360601-1-360601-2-4360601-2-4360601-2-54 | 13.113.6203.13.4203.13.6203.13.6 |  |
| 7 | 523.0 | IFU.523 | 1 - Introduction | Regulatory Requirements | Compliance statements for IEC Standards shall include the MX1 Model or Type Reference, Standard Number (e.g. "60601-2-54"), Version Number (e.g. 3.2), and Year of Standard publication. | 60601-1-3 | 4.1 | Compliance Statements |
| 170 | 524.0 | IFU.524 | 10 - System Upkeep | Periodic Maintenance Schedule | IFU shall state the importance of regularly checking storage capacity and securing or archiving important records. | 60601-2-43 | 201.12.4.101.2 | Always be mindful of the storage capacity of the system, especially before starting an exam. Past Image or study deletion must be manually authorized, and the MX1 system is not intended for long-term storage. It is important to periodically secure or archive any important records in a secure location. |
| 120 | 525.0 | IFU.525 | 8 - Radiation Exposure | Overview of Radiation Safety | If there is a possibility in normal use that the Patient can be exposed to deterministic radiation dose levels, the IFU shall identify the number of exposures or duration of loading necessary to reach deterministic effects on the specified average patient and obese patient. | 60601-1-360601-1-360601-2-54 | 5.2.4.15.2.4.5203.5.2.4.5.101 | The following table indicates the predicted number of exposures and duration of loading at the minimum loading factors necessary to reach deterministic levels:table |
| 121 | 526.0 | IFU.526 | 8 - Radiation Exposure | Overview of Radiation Safety | If there is a possibility in normal use that the Patient can be exposed to deterministic radiation dose levels, the IFU shall provide information on the means to manage high radiation doses, concerning available settings (loading factors, technique factors, operating parameters, etc.) that affect radiation quality or prevailing radiation dose and dose rate. | 60601-1-360601-1-360601-2-54 | 5.2.4.15.2.4.5203.5.2.4.5.101 | To minimize risk without adversely affecting the clinical objectives, the As Low As Reasonably Achievable (ALARA) standard should be applied. A general guideline is to apply the lowest X-ray tube voltage (kV) and current-time product (mAs) required to provide acceptable image contrast and exposure. Entrance skin dose, scatter radiation, and effective dose increase as the user increases any loading factors. |
| 102 | 527.0 | IFU.527 | 6 - Imaging Modes | Radiation Controls and Audible Signals - Maximum Allowable Air Kerma Rate | IFU shall describe the High Level Control described in 60601-2-54 subclause 203.6.3.102 (what its function are and how to set). | 60601-2-54 | 203.6.3.102 | A maximum radiation air kerma or energy to the patient may be configured before beginning an exam, using the directions in Section 7 - Device App. This Radiation Rate Limiter is only available in Fluoroscopy and DDR imaging modes, and is measured at the Patient Entrance Reference Point, described in Section 10 - Radiation Exposure. When emitting with settings and positions that would result in an air kerma rate (mGy/min) exceeding the set Radiation Limit, the Emitter’s buzzer will produce a warning tone, different from that of the Loading Time Limiter, continuously throughout X-ray emission. When positioned and set at a rate greater than 176 mGy/min, the Emitter will cease immediately. By default, the Radiation Rate Limiter is set to 88 mGy/min and may be configured according to your facility’s radiation guidelines. |
| 32 | 528.0 | IFU.528 | 3 - System Overview | Major Components - Collimation Pucks | The IFU shall include information on the materials to which the Patient or Operator is exposed if such exposure can constitute an unacceptable Risk. | 60601-1RSK | 7.9.2.5 | CAUTION: The Collimation Pucks may contain nickel and other metals. If you experience metal sensitivity, wear protective gloves when handling the pucks. |
| 101 | 529.0 | IFU.529 | 6 - Imaging Modes | Radiation Controls and Audible Signals - Exam Timer | IFU shall describe the Timing Device described in 60601-2-54 subclause 203.6.2.1 (what its function are and how to set). | 60601-2-54 | 203.6.2.1 | A maximum loading time, or time (in seconds) the tube is actively emitting x-rays in an exam, may be set before beginning an exam, using the directions in Section 7 - Device App. When the Loading Time Limiter maximum has been reached during an exam, the Emitter will produce a warning tone continuously throughout any subsequent X-ray emissions unless the Loading Time Limiter is reset. By default, the Loading Time Limiter is set to 300 seconds and may be configured for facility policies, up to a maximum of 300 seconds. |
|  | 53.0 | IFU.53 |  |  | DELETED |  |  |  |
| 118 | 531.0 | IFU.531 | 8 - Radiation Exposure | Overview of Radiation Safety | If there is a possibility in normal use that the Patient can be exposed to deterministic radiation dose levels, the IFU shall list configurations in which this may occur. | 60601-1-360601-1-360601-2-54 | 5.2.4.15.2.4.5203.5.2.4.5.101 | Skin dose levels in prolonged use may be high enough to cause deterministic effects, but during most procedures the deterministic and stochastic effects of patient irradiation are low. Precautions should be taken when using continuous radiation modes over longer procedures, especially where skin dose in a single location could exceed 1 Gy, as these are the situations in which the chance for deterministic and stochastic effects are higher. In particular anatomy and system configurations required for an examination, patient skin could be significantly closer to the X-ray source, with dose rate increasing as the inverse square of the Source-to-Skin Distance. |
| 105 | 532.0 | IFU.532 | 7 - Device App | Navigating the MedAI Imaging App | IFU shall describe the steps to change the units for displayed Cumulative Dose Area Product through the Device App. | 60601-2-43 | 203.6.4.5 | TBD |
| 20 | 534.0 | IFU.534 | 2 - General Safety | Radiation Safety | IFU shall warn against using the device in use environments, environmental conditions, and useful life as specified. | RSK |  | WARNING: The MX1 System and accessories should always be used within the specified use environments, environmental conditions and useful life of the equipment as specified in these Instructions for Use. Not doing so may result in serious injury or equipment damage. |
| 36 | 535.0 | IFU.535 | 4 - Setting Up the System | Powering On | The IFU shall contain the necessary information for the operator to initiate operation of the ME Equipment. | RSK |  | Power on the Emitter and Cassette by pressing and briefly holding the Emitter’s Middle Button and Cassette’s Power Button: |
| 47 | 536.0 | IFU.536 | 4 - Setting Up the System | Connecting External Devices | IFU shall instruct on how to connect the Tablet to the System. | RSK |  | Connect the MX1 System to a tablet using the following steps:1. If using a non-MedAI-supplied device, install the MedAI Imaging App onto your device from the Google Play Store.2. Open the tablet’s network settings and select the WiFi network found on the Cassette screen and enter the password, also found on the Cassette Screen.3. Open the MedAI Imaging App. You may need to force quit the MedAI Imaging App to reset it.4. Navigate to the Cassette Configuration Screen with Menu > Cassette. If the Cassette is properly connected, its system information will be shown. See Section 7 - MedAI Imaging App for information on using the app. |
| 49 | 537.0 | IFU.537 | 4 - Setting Up the System | Initial MedAI Imaging App Setup - Internet and Wi-Fi Setup | IFU shall instruct on how to connect the System to a wireless network through the device app. | RSK |  | Connect the MX1 System to a wireless network using the following steps:Be sure your Tablet is connected to a Cassette. See Connecting External Devices for directions.On the MedAI Imaging App, open the Network Settings with Menu > Network Settings.Select and enter the Wi-Fi network information and password to connect. |
| 55 | 538.0 | IFU.538 | 5 - Using the System | Power and Idling States | IFU shall describe the purpose of System idle and its indications | RSK |  | table |
| 56 | 539.0 | IFU.539 | 5 - Using the System | Power and Idling States | IFU shall Note that the System can take up to 20 seconds to exit idle. | RSK |  | The MX1 System may take up to 20 seconds to come up from Idle State once woken. |
| 216 | 54.0 | IFU.54 | 12 - Tech Specs | X-ray Flat Panel Detector Specification and Imaging Performance | IFU shall CAUTION to inspect for image quality issues and contact MedAI if they persist. | RSK |  | Note: The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 61 | 540.0 | IFU.540 | 5 - Using the System | Positioning the System | IFU shall call out the locations of the System cooling vents and instruct the user to not cover them | RSK |  | When positioning the MX1 System for use, be aware of the locations of cooling vents on the Emitter and Cassette:[image of intakes and exhausts on cassette and emitter]Avoid setting the device in positions that restrict airflow through these vents. Doing so may result in overheating of the component. |
| 83 | 541.0 | IFU.541 | 5 - Using the System | Foot Pedal | IFU shall instruct operator to maintain communication with Patient when using a distance activation method | RSK |  | Note: It is important to establish effective audio and visual communication with an awake patient if using the wireless Foot Pedal to trigger X-rays from an area away from the patient. Any necessary adjustments of Patient anatomy should be completed prior to distancing. |
| 31 | 544.0 | IFU.544 | 3 - System Overview | Major Components - Wired Charger | IFU shall Caution the user against tripping hazards posed by the Wired Charger when charging. | RSK |  | CAUTION: The Wired Charger cables pose a trip hazard. When charging the device, set up cords in areas of low foot traffic and adequately manage excess cable length to reduce the risk of tripping. |
| 51 | 546.0 | IFU.546 | 4 - Setting Up the System | Initial MedAI Imaging App Setup - DICOM Fields SetupInitial MedAI Imaging App Setup - Users Setup | IFU shall instruct users on how to adjust the required DICOM fields and add or remove Doctors during initial device app setup. | RSK |  | DICOM Fields SetupThe MedAI Imaging App can display different options for Exam information. To adjust the items that appear:1. Open the DICOM Fields menu with Top Menu > DICOM Fields.2. Check or uncheck DICOM fields that you’d like to show in the Exam Setup Page. Checked fields will be required to begin an Exam.3. Close the DICOM Fields menu by tapping the back arrow in the top left.Users SetupThe MX1 system sends the doctor’s or operator’s name with each study to organize and document the image captures and exams. To add, remove, or edit the list of available practitioners:1. Open the Doctors menu with Top Menu > Doctors.2. Add new doctors with the Add doctor button, edit doctors with the edit button, or delete doctors with the trash can.3. When adding or editing a user, select the user type and add their information. Tap Save to confirm changes. |
| 87 | 547.0 | IFU.547 | 5 - Using the System | Sterile Coverings | IFU shall caution the user to always drape the Cassette when using potentially corrosive chemicals used during medical procedures, such as ethyl chloride, as it may damage the enclosure. | RSK |  | CAUTION: It is highly recommended to cover the Cassette when performing procedures that require the use of particular chemicals or substances, such as ethyl chloride (numbing spray). Contact with these substances may cause damage to the Cassette surface. |
| 95 | 548.0 | IFU.548 | 6 - Imaging Modes | Dynamic Digital Radiography (DDR)Fluoroscopy (Fluoro) Mode | IFU shall describe the capture frame rate in both Radiography and Radioscopy Modes. | RSK |  | Images are normally captured at 5 frames per second until the user releases the trigger or the Fuel Gauge fills. |
| 106 | 549.0 | IFU.549 | 7 - Device App | Navigating the MedAI Imaging App | IFU shall include instructions on how to update the app and device, and how to troubleshoot a software version mismatch. | RSK |  |  |
|  | 55.0 | IFU.55 |  |  | DELETED |  |  |  |
| 100 | 550.0 | IFU.550 | 7 - Device App | Loading Factor Selection | IFU shall describe the limitation of mAs in DDR and Fluoro Mode and its indication on the Viewfinder. | RSK |  | MX1 only supports mAs up to 0.08 for DDR and Fluoro captures. Selecting a higher mAs number then holding the trigger will begin a DDR sequence with 0.08 mAs, indicated by the blue color below. The ability to select a higher mAs number in Fluoro Mode is not available.images |
| 67 | 551.0 | IFU.551 | 5 - Using the System | Positioning the System - Positioning the Cassette | IFU shall recommend the use of physical L/R laterality markers to mitigate against error. | RSK |  | Note: Optionally, use lead laterality markers (L/R) to mitigate error. |
| 107 | 552.0 | IFU.552 | 7 - Device App | Navigating the MedAI Imaging App | IFU shall include instructions on how to clear images, PHI, other facility information from the device. | RSK |  |  |
| 108 | 553.0 | IFU.553 | 7 - Device App | Performing an Exam - Exam Setup Page | IFU shall include instructions on how to set up an Exam. | RSK |  | Performing an Exam - Exam Setup Page |
| 104.2 | 554.0 | IFU.554 | 7 - Device App | Navigating the MedAI Imaging App | The IFU shall instruct on how to pair the Foot Pedal to the MX1 System. |  |  |  |
| 111.1 | 555.0 | IFU.555 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall indicate that in-progress captures will be labeled "Live" and past captures will be labeled "Stored" on the Acquisiiton Page. |  |  |  |
| 104.1 | 556.0 | IFU.556 | 7 - Device App | Navigating the MedAI Imaging App | IFU shall describe the steps to enable/disable specific radiation modes in the Device App. |  |  |  |
| 97.1 | 557.0 | IFU.557 | 6 - Imaging Modes | Fluoroscopy (Fluoro) Mode | The IFU shall warn against attempting to picking up and hand-holding the Emitter after initiating a Fluoro capture. |  |  |  |
| 155.1 | 559.0 | IFU.559 | 9 - System Info and Alerts | Troubleshooting | IFU Shall include troubleshooting information on how to resolve errors associated with DDR stopping early |  |  |  |
|  | 56.0 | IFU.56 |  |  | DELETED |  |  |  |
| 97.2 | 560.0 | IFU.560 | 6 - Imaging Modes | Fluoroscopy (Fluoro) Mode | The IFU shall note a momentary delay before x-ray emission starts and shall advise holding the trigger until the image appears. |  |  |  |
| 23.1 | 561.0 | IFU.561 | 2 - General Safety | Hot Surfaces and Temperatures | The IFU shall disclose the maximum temperature reached by the wired chargers (non-applied parts) during extended use. | 60601-1 | 11.1.2.2 | CAUTION: Use caution when handling the H1 Wired Chargers after extended use as the external surfaces may reach high temperatures. Direct skin contact with these surfaces for more than 10 seconds during extended use could result in burns. |
|  |  | IFU.562 | 5 - Using the System | Foot Pedal | IFU shall describe Foot Pedal device states. | RSK |  |  |
|  |  | IFU.563 | 5 - Using the System | Foot Pedal | IFU shall describe Foot Pedal button layout. | RSK |  |  |
|  |  | IFU.564 | 7 - Device App | Send Captures | IFU shall describe how to send images. | RSK |  |  |
|  |  | IFU.565 | 7 - Device App | Device Settings Page | IFU shall describe selectable DAP units. | RSK |  |  |
|  |  | IFU.566 | 7 - Device App | Performing an Exam - Exam Setup Page | IFU shall note that RIS and PACS server may not be available in areas with poor network connection. | RSK |  |  |
|  |  | IFU.567 | 8 - Radiation Exposure | Collimation Sizing | IFU shall explain the different collimation modes. | RSK |  |  |
|  |  | IFU.568 | 1 - Introduction | MedAI | IFU shall contain contact information for MX1 support. | RSK |  | For Customer Support with MedAI products, please contact 855-733-9729 or support@medai.com. |
|  |  | IFU.569 | 5 - Using the System | Emergency Instructions | Emergency Instructions insists that patient info is entered once emergency exam is finished. | RSK |  |  |
| 175 | 57.0 | IFU.57 | 10 - System Upkeep | Internal Battery Health | IFU shall specify the estimated battery capacity, in terms of images in a full charge or similar. | RSK |  | Charging the Emitter for XXX hours will provide it enough capacity for XXX hours of continuous use.  A normal, complete Emitter charge cycle may take XXX hours. Charging the Cassette for between XXX to XXX hours charges the Cassette enough to allow up to XXX hours of continuous use.  A normal, complete charge cycle takes more than XXX hours. If you suspect something is wrong with the battery, discontinue use and contact MedAI for assistance. |
|  |  | IFU.570 | 4 - Setting Up the System | Unpacking | IFU shall describe how to unpack and pack the MX1 System. | RSK |  | The MX1 System and all of its components will arrive packed in the case inside of a cardboard overshipper for protection. Unpack the MX1 System using the following steps:On first-time unboxing, open the overshipper and remove the MX1 Case, setting it flat on the ground, label-side up.Open the case using the four latches surrounding the lid (two on the front and two on the sides). Pressing the middle button and flipping up each latch will allow the case to be opened.Open the inner lid pouch and remove all accompanying documentation for the MX1 System, including the MX1 Instructions for Use, the MX1 Quick User Guide, and the MX1 Emergency Instructions.Remove the Cassette and place on a flat surface. Remove the top layer of foam and set aside.Remove the Emitter, Tablet, Puck Box, and Wired Chargers and place on a flat surface. Place the top layer of foam back into the case and close the case.Ensure any component not currently in use is safely packed back into its corresponding place in the MX1 case so that they are not misplaced. |
|  |  | IFU.571 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall instruct on how to complete an exam. | RSK |  | Complete the exam by tapping the Complete Exam button, which also displays the accumulated dose to the patient as well as the DAP for the current study. Completing the exam will take you to athe Library Page containing only the images taken during the exam. |
|  |  | IFU.572 | 4 - Setting Up the System | Connecting External Devices | IFU shall describe how to access the MX1 App. | RSK |  |  |
|  |  | IFU.573 | 4 - Setting Up the System | Initial MedAI Imaging App Setup - DICOM (PACS) and RIS Setup | IFU shall note to register MX1 during server integration. | RSK |  | Some facilities may require the MX1 System to be registered by the facility server in order to receive worklist orders or send image studies. Ensure that the MX1 System has been registered before attempting to import a worklist or submit image studies if the facility PACS/RIS network requires it. Not doing so may result in rejected images studies and missing image studies in the PACS server. |
|  |  | IFU.574 | 5 - Using the System | Positioning the System | IFU shall describe transport procedure. | RSK |  |  |
|  |  | IFU.575 | 3 - System Overview | Major Components - Wireless Tablet | IFU shall describe how to use the tablet. | RSK |  |  |
|  |  | IFU.576 | 13 - Cybersecurity | Software Updates | IFU shall include a statement about plugging in all MX1 components, including T1 Tablet, prior to updating the MX1 App. | RSK |  |  |
|  | 58.0 | IFU.58 |  |  | DELETED |  |  |  |
|  | 59.0 | IFU.59 |  |  | DELETED |  |  |  |
|  | 6.0 | IFU.6 |  |  | DELETED |  |  |  |
|  | 61.0 | IFU.61 |  |  | DELETED |  |  |  |
| 99 | 63.0 | IFU.63 | 6 - Imaging Modes | Single Radiography (Single) ModeDynamic Digital Radiography (DDR)Fluoroscopy (Fluoro) Mode | IFU shall Caution the user against uneccessary movement during X-ray capture in order to minimize the effects of motion blur. | RSK |  | CAUTION: Movement of the patient, Cassette, or Emitter during imaging may increase risk of non-diagnostic exposure and/or patient injury. Avoid abrupt shaking or movement of the patient, Cassette, and Emitter. |
|  | 64.0 | IFU.64 |  |  | DELETED |  |  |  |
| 94 | 65.0 | IFU.65 | 6 - Imaging Modes | Single Radiography (Single) ModeDynamic Digital Radiography (DDR)Fluoroscopy (Fluoro) Mode | IFU shall include instructions to capture single and serial radiographs. | RSK |  | A DDR can be taken after setting the proper loading factors by pointing the Emitter towards the Cassette and anatomy to be imaged, and pulling and holding the trigger Active Area while the Tracking System allows x-rays. |
|  | 66.0 | IFU.66 |  |  | DELETED |  |  |  |
|  | 67.0 | IFU.67 |  |  | DELETED |  |  |  |
|  | 68.0 | IFU.68 |  |  | DELETED |  |  |  |
| 52 | 69.0 | IFU.69 | 5 - Using the System | 5 - Using the System | IFU shall contain all information necessary to operate the System in accordance with its specification, including an explanation of the following:- the function of controls, displays, and signals in each of the provided modes of operation (Photography, Single Radiography, Serial Radiography/DDR, and Fluoroscopy)- the sequence of operation- the connection and disconnection of detachable parts and accessories- any necessary replacement of materials consumed during operation | 60601-1RSK | 7.9.2.9 | 5 - Using the System |
|  | 7.0 | IFU.7 |  |  | DELETED |  |  |  |
| 113 | 70.0 | IFU.70 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall include instructions for adjusting or manipulating an image | RSK |  | To enhance any other image found in the “B”, “C”, or “D” columns, select the chosen image by single clicking. |
| 111 | 71.0 | IFU.71 | 7 - Device App | Performing an Exam - Acquisition Page | IFU shall include instructions for viewing an image. | RSK |  |  |
| 50 | 72.0 | IFU.72 | 4 - Setting Up the System | Initial MedAI Imaging App Setup - DICOM (PACS) and RIS Setup | IFU shall instruct users on how to set up proper PACS and RIS server system connections during initial device app setup. | RSK |  | The MX1 System allows for the transfer of radiographic images to a PACS. Additionally, the MX1 System can retrieve modality worklist (MWL) orders from a facility’s RIS servers. For further detail on DICOM communications, contact MedAI for assistance. Connect the Cassette to a facility wireless PACS/RIS network using the following steps:1. On the MedAI Imaging App, open the DICOM Servers menu accessed via the Top Menu.2. Add new servers with the Add Server button, delete servers with the Trash Can buttons, or edit server information using the Edit buttons.3. When editing or adding a new server, enter the field information, select the Test button to check for connection after information has been entered, and tap Save to confirm changes.4. Close the DICOM Servers menu by tapping the back arrow in the top left. |
|  | 75.0 | IFU.75 |  |  | DELETED |  |  |  |
| 155 | 76.0 | IFU.76 | 9 - System Info and Alerts | Troubleshooting | IFU shall include troubleshooting for imaging interlock problems | RSK |  | table - Tracking System Interlocks not met (Red) |
| 8 | 77.0 | IFU.77 | 2 - General Safety | 2 - General Safety | IFU shall warn of potential hazards arising from the operation of ME X-ray equipment. | RSK |  | Potential hazards exist in the use of any medical electronic devices and x-ray systems. Operators using the MX1 System should understand the safety issues, emergency procedures, and the operating instructions provided. The following section describes hazardous and potentially hazardous conditions and how to adequately protect device operators and others from possible injury. |
| 93 | 78.0 | IFU.78 | 6 - Imaging Modes | Selecting Imaging Modes | IFU shall instruct how to change capture modes. | RSK |  | Cycle through modes from the Emitter Viewfinder screen by pressing the Mode Indicator or Middle Button: |
| 88 | 79.0 | IFU.79 | 5 - Using the System | Sterile Coverings | IFU shall note to uncover Cassette LEDs to allow X-ray emission and that covering them will prevent tracking system from allowing emission. | RSK |  | If the drape covering the Cassette obscures or distorts the infrared (non-visible) LEDs, the system may not work as intended or may prevent X-ray emissions. If this occurs, reposition the drape. |
|  | 8.0 | IFU.8 |  |  | DELETED |  |  |  |
| 115 | 80.0 | IFU.80 | 7 - Device App | Reviewing and Exporting Past Exams - Library Page | IFU shall Warn that previous Patient images will appear on screen when sending images to PACS | RSK |  | WARNING: Previous patient images may appear on the History Page before sending images to PACS. Verify the image(s) being sent is from the correct patient prior to sending. |
|  | 81.0 | IFU.81 |  |  | DELETED |  |  |  |
|  | 82.0 | IFU.82 |  |  | DELETED |  |  |  |
|  | 85.0 | IFU.85 |  |  | DELETED |  |  |  |
|  | 87.0 | IFU.87 |  |  | DELETED |  |  |  |
|  | 88.0 | IFU.88 |  |  | DELETED |  |  |  |
|  | 89.0 | IFU.89 |  |  | DELETED |  |  |  |
|  | 9.0 | IFU.9 |  |  | DELETED |  |  |  |
|  | 90.0 | IFU.90 |  |  | DELETED |  |  |  |
|  | 91.0 | IFU.91 |  |  | DELETED |  |  |  |
| 98 | 92.0 | IFU.92 | 6 - Imaging Modes | Single Radiography (Single) ModeDynamic Digital Radiography (DDR)Fluoroscopy (Fluoro) ModeLoading Factor Selection | IFU shall include all available options for loading factors in each radiation mode along with recommended imaging parameters | 60601-1RSK |  | Available Loading Factors in Single ModeAvailable Loading Factors in DDR ModeAvailable Loading Factors in Fluoro ModeRecommended Loading Factors Table |
| 181 | 93.0 | IFU.93 | 10 - System Upkeep | End of Life Procedure | IFU shall state the effects of heavy use with higher loading factors with regard to service life of the X-ray tube and the system, including ways to mitigate those effects. | RSK |  | Note: Heavy use with higher loading factors (kV and mAs) may cause the X-ray tube and other system components to deteriorate and require servicing quicker than expected. Reducing use, charging adequately and often, and storing properly can extend the device's service life. |
|  | 94.0 | IFU.94 |  |  | DELETED |  |  |  |
|  | 95.0 | IFU.95 |  |  | DELETED |  |  |  |
|  | 97.0 | IFU.97 |  |  | DELETED |  |  |  |
|  | 99.0 | IFU.99 |  |  | DELETED |  |  |  |
|  |  | The below requirements are only applicable when the MX1 System is indicated for use with pediatric patients |  |  |  |  |  |  |
|  |  | IFU.86 | 6 - Imaging Modes | Pediatric Patients | IFU shall describe risks, information, and user-available controls to mitigate radiation harm in the context of use with pediatric patients |  |  |  |

### Table 7
| 524 |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| # | ID | Requirement | SOURCE | CLAUSE | High Level Section | Subsection | Design Output Text [TO BE HIDDEN/DELETED BEFORE DCO] |
| 254.0 | IFU.254 | Symbols need to be official IEC or ISO symbols | 60601-1 | 7.6 | 11 - Symbols and Labels | Symbols | table |
| 332.0 | IFU.332 | IFU shall provide instructions for connecting the equipment to an IT-NETWORK, including the purpose, required characteristics, required configurations, technical specifications including security specifications, intended information flow between networking devices and device, and a list of HAZARDOUS SITUATIONS resulting from the IT-NETWORK's failure to provide specified characteristics. | 60601-1 | 14.13 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications |  |
| 333.0 | IFU.333 | IFU shall instruct that connection of device to IT-NETWORK that includes other equipment could result in previously unidentified RISKS to patients, operators, or third parties. | 60601-1 | 14.13 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | Connection to IT-networks, including other equipment not provided with the MX1 System, could result in previously unidentified risks to patients, operators, or third parties. The Responsible Organization should identify, analyze, evaluate, and control these risks. |
| 334.0 | IFU.334 | IFU shall instruct that the user/reponsible organization should dentify, analyze, evaluate and control these RISKS | 60601-1 | 14.13 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | Connection to IT-networks, including other equipment not provided with the MX1 System, could result in previously unidentified risks to patients, operators, or third parties. The Responsible Organization should identify, analyze, evaluate, and control these risks. |
| 335.0 | IFU.335 | IFU shall state subsequent changes to the IT-NETWORK could introduce new risks and require additional analysis | 60601-1 | 14.13 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | Changes to the IT-network could introduce new risks that require additional analysis. Changes may include:Changes in Network ConfigurationConnection of additional itemsDisconnection of itemsEquipment updates or upgrades |
| 336.0 | IFU.336 | IFU shall state that changes to the IT-NETWORK include changes in configuration, connection of additional items,  disconnecting items, updating equipment,  and upgrading equipment. | 60601-1 | 14.13 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | Changes to the IT-network could introduce new risks that require additional analysis. Changes may include:Changes in Network ConfigurationConnection of additional itemsDisconnection of itemsEquipment updates or upgrades |
| 337.0 | IFU.337 | IFU shall contain the info necessary for the ME SYSTEM to be used as intended, and MFG address | 60601-1 | 16.2 | 1 - Introduction | MedAI | MedAI, Inc.1230 Main StreetSuite 300Springfield, IL 60001info@medai.com |
| 338.0 | IFU.338 | IFU shall include the ACCOMPANYING DOCUMENTS for each item of ME EQUIPMENT that is provided by the MANUFACTURER (see 7.8.2); | 60601-1 | 16.2 | General | General | General |
| 339.0 | IFU.339 | IFU shall include the ACCOMPANYING DOCUMENTS for each item of non-ME EQUIPMENT that is provided by the MANUFACTURER; | 60601-1 | 16.2 | General | General | General |
| 340.0 | IFU.340 | IFU shall include the specification of the ME SYSTEM, including the use as intended by the MANUFACTURER and a listing of all of the items forming the ME SYSTEM; | 60601-1 | 16.2 | 3 - System Overview | System Component and Accessory List | System Component and Accessory List |
| 341.0 | IFU.341 | IFU shall include instructions for the installation, assembly and modification of the ME SYSTEM to ensure continued compliance with this standard; | 60601-1 | 16.2 | 4 - Setting Up the System | 4 - Setting Up the System | 4 - Setting Up the System |
| 342.0 | IFU.342 | IFU shall include instructions for cleaning and, when applicable, disinfecting and sterilizing each item of equipment or equipment part forming part of the ME SYSTEM (see 11.6.6 and 11.6.7); | 60601-1 | 16.2 | 9 - System Upkeep | Routine CleaningDisinfection | Cleaning Instructions (multiple) |
| 343.0 | IFU.343 | IFU shall include additional safety measures that should be applied, during installation of the ME SYSTEM; | 60601-1 | 16.2 | N/A | N/A | N/A |
| 344.0 | IFU.344 | IFU shall include which parts of the ME SYSTEM are suitable for use within the PATIENT ENVIRONMENT; | 60601-1 | 16.2 | 5 - Using the System | Positioning the System and Patient |  |
| 345.0 | IFU.345 | IFU shall include additional measures that should be applied during preventive maintenance; | 60601-1 | 16.2 | 9 - System Upkeep | Periodic Maintenance Schedule | Always inspect the radiographic captures for image quality issues (spots, blurriness, resolution) during each use. At least once monthly, inspect the external surfaces of all components for damage, loose or missing parts, and frayed or damaged cords. Do not use the device if it displays one or more of the above conditions until the problem is corrected and has been verified as operating correctly and safely. |
| 346.0 | IFU.346 | IFU shall include a warning that an additional MULTIPLE SOCKET-OUTLET or extension cord shall not be connected to the ME SYSTEM; | 60601-1 | 16.2 | 4 - Setting Up the System | Charging | WARNING: Multi-socket outlets or power strips are strictly prohibited for connection unless they are rated to IEC 60601-1 and are provided with all necessary markings and certificates of conformance. Connecting the MX1 System to multi-socket outlets that are not rated to IEC 60601-1 may result in fire. |
| 347.0 | IFU.347 | IFU shall include a warning to connect only items that have been specified as part of the ME SYSTEM or that have been specified as being compatible with the ME SYSTEM; | 60601-1 | 16.2 | 4 - Setting Up the System | Connecting External Devices | WARNING: Use only MedAI-supplied battery chargers and approved accessories. Use or connection of incompatible chargers and accessories may lead to major shock, burn, or injury. See Section 3 - System Overview, for a list of approved system components and accessories. |
| 348.0 | IFU.348 | IFU shall include the permissible environmental conditions of use of the ME SYSTEM including conditions for transport and storage; and | 60601-1 | 16.2 | 12 - Tech Specs | General Specifications | Conditions for Use, Travel, and Storage |
| 349.0 | IFU.349 | IFU shall include instructions to the OPERATOR not to touch parts referred to in 16.4 and the PATIENT simultaneously. | 60601-1 | 16.2 | 2 - General Safety | Electrical Safety | WARNING: Do not touch the patient and any exposed metal components, including ports, buttons, and connector pins, simultaneously as electrical discharge may occur. |
| 350.0 | IFU.350 | IFU shall include advice to the RESPONSIBLE ORGANIZATION to carry out all adjustment cleaning, sterilization and disinfection PROCEDURES specified therein; and | 60601-1 | 16.2 | 9 - System Upkeep | Routine CleaningDisinfection | Cleaning Instructions (multiple) |
| 351.0 | IFU.351 | IFU shall include that the assembly of ME SYSTEMS and modifications during the actual service life require evaluation to the requirements of this standard. | 60601-1 | 16.2 | 9 - System Upkeep | End of Life Procedure | The assembly of the MX1 System and modifications during the actual service life require evaluation to requirements of IEC 60601-1 and other applicable safety standards. |
| 352.0 | IFU.352 | The IFU shall specify equipment outside of the ME SYSTEM intended to provide power to the system, including actual transient current level. | 60601-1 | 16.3 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | Input Rated Voltage / Frequency: 90 - 264 VAC / 50-60 HzOutput Nominal Voltage: 20 VDC |
| 353.0 | IFU.353 | The IFU shall disclose the actual transiet current in the technical instruction and installation instructions for use. | 60601-1 | 16.3 | N/A | N/A | N/A |
| 330.0 | IFU.330 | The IFU shall disclose the maximum temperature reached by an applied part surface and conditions for safe contact (duration, etc). Don't need if 41°C is not exceeded. | 60601-1 | 11.1.2.2 | 2 - General Safety | Environment Safety | CAUTION: The MX1 System device surfaces may reach temperatures up to 43°C (109.4°F) under extended use at the max operating temperature. This temperature limit is appropriate for the healthy skin of adults but may cause discomfort or minor injury when large areas of the skin (10 % of total body surface or more) are in contact with the hot surface, or if unhealthy skin is in contact with the hot surface. |
| 331.0 | IFU.331 | IFU shall specify the cleaning procedure and the effects of multiple cleanings, if deteriorating | 60601-1 | 11.6.6 | 9 - System Upkeep | Routine CleaningDisinfection | Cleaning Instructions (multiple) |
| 253.0 | IFU.253 | The working conditions are specified in the ACCOMPANYING DOCUMENTS. | 60601-1 | 5.4 a) | 12 - Tech Specs | General Specifications | Conditions for Use, Travel, and Storage |
| 255.0 | IFU.255 | IFU and Accompanying Documents shall be provided, either hard copy or electronically. Risk assessment needs to assess risk with electronic copy. | 60601-1 | 7.9.1 | General | General | General |
| 256.0 | IFU.256 | IFU shall identify the system with MFG Name, MFG Address, and Model Reference. | 60601-1 | 7.9.1 | 1 - Introduction | 1 - IntroductionMedAI | This manual describes operation for the MX1 Portable X-ray System (also referred to as the MX1 System).MedAI® and Imager® are registered trademarks within the United States and other countries. |
| 257.0 | IFU.257 | IFU needs to specify skills or training required for operation. | 60601-1 | 7.9.1 | 1 - Introduction | 1 - Introduction | The device is intended for qualified medical personnel who have been trained in the use of medical imaging equipment and who have read this Instructions for Use and Accompanying Documents. |
| 258.0 | IFU.258 | IFU needs to be written consistent with the education, training and any special needs of the operator | 60601-1 | 7.9.1 | General | General | General |
| 259.0 | IFU.259 | IFU shall include the use of the ME EQUIPMENT as intended by the MANUFACTURER, | 60601-1 | 7.9.2.1 | 1 - Introduction | Intended Use | The MX1 System is a hyper portable X-ray system designed to aid clinicians with point of care visualization through diagnostic X-rays of extremities and hips. The device is intended for use in clinical, surgical, home, and ambulatory environments by trained clinicians. |
| 260.0 | IFU.260 | IFU shall include the frequently used functions; | 60601-1 | 7.9.2.1 | 5 - Using the System | 5 - Using the System | 5 - Using the System |
| 261.0 | IFU.261 | IFU shall include any known contraindication(s) to the use of the ME EQUIPMENT; and | 60601-1 | 7.9.2.1 | 1 - Introduction | Contraindications | The MX1 System is NOT intended for:MammographyDental applicationsContact with non-intact skin |
| 262.0 | IFU.262 | IFU shall include those parts of the ME EQUIPMENT that shall not be serviced or maintained while in use with a PATIENT. | 60601-1 | 7.9.2.1 | 9 - System Upkeep | Overview | CAUTION: Do not service or maintain any part of the MX1 System while in use with a PATIENT. Opening or cleaning the MX1 system while in use with a pateint may result in electrical shock. |
| 263.0 | IFU.263 | IFU shall include the name and address of the MANUFACTURER | 60601-1 | 7.9.2.1 | 1 - Introduction | MedAI | MedAI, Inc.1230 Main Street, Suite 300Springfield, IL 60001info@medai.com |
| 264.0 | IFU.264 | IFU shall include the model or type reference of the equipment | 60601-1 | 7.9.2.1 | 1 - Introduction | 1 - Introduction | This manual describes operation for the MX1 Portable X-ray System (also referred to as the MX1 System). |
| 265.0 | IFU.265 | IFU shall include information about all classifications from Clause 6 | 60601-1 | 7.9.2.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | The MX1 System components are Internally Powered while not charging and Class II ME Equipment while wired charging (according to IEC 60601-1). The MX1 Cassette is the only Applied Part: Type B. |
| 266.0 | IFU.266 | IFU shall include all outside markings and their locations on equipment with explanation | 60601-1 | 7.9.2.1 | 11 - Symbols and Labels | Equipment Labels | table |
| 267.0 | IFU.267 | IFU shall include all safety signs/symbols on equipment with explanation | 60601-1 | 7.9.2.1 | 11 - Symbols and Labels | Symbols | table |
| 268.0 | IFU.268 | IFU shall be written in languages acceptable to the OPERATOR | 60601-1 | 7.9.2.1 | General | General | General |
| 288.0 | IFU.288 | The instructions for use shall list all system messages, error messages and fault messages that are generated, unless these messages are self-explanatory. | 60601-1 | 7.9.2.10 | 8 - System Info and Alerts | Emitter User Interface MessagesCassette User Interface MessagesDevice App User Interface Messages | tables |
| 289.0 | IFU.289 | The list shall include an explanation of messages including important causes, and possible action(s) by the OPERATOR, if any, that are necessary to resolve the situation indicated by the message. | 60601-1 | 7.9.2.10 | 8 - System Info and Alerts | Emitter User Interface MessagesCassette User Interface MessagesDevice App User Interface Messages | tables |
| 290.0 | IFU.290 | The instructions for use shall contain the necessary information for the OPERATOR to safely terminate the operation of the ME EQUIPMENT. | 60601-1 | 7.9.2.11 | 5 - Using the System | Power Off Procedure | Safely power down the Emitter and Cassette by pressing and holding their power buttons for two seconds for graceful shutdown or ten seconds for hard shutdown. Alternatively, navigate to the Component Drawer in a connected Device App and select the power button to power down or sleep connected components. |
| 291.0 | IFU.291 | IFU shall contain details about cleaning and disinfection that may be used; and list applicable parameters such as temperature, pressure, humidity, time limits and number of cycles that such ME EQUIPMENT parts or ACCESSORIES can tolerate. | 60601-1 | 7.9.2.12 | 9 - System Upkeep | Routine CleaningDisinfection | Cleaning Instructions (multiple) |
| 292.0 | IFU.292 | IFU shall instruct on preventive inspection, maintenance and calibration to be performed, including the frequency. | 60601-1 | 7.9.2.13 | 9 - System Upkeep | Periodic Maintenance Schedule | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 293.0 | IFU.293 | IFU shall provide information for the safe performance of routine maintenance necessary to ensure the continued safe use of the ME EQUIPMENT. | 60601-1 | 7.9.2.13 | 9 - System Upkeep | Periodic Maintenance Schedule | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 294.0 | IFU.294 | IFU shall identify the parts on which preventive inspection and maintenance shall be performed by SERVICE PERSONNEL, including the periods, but not including details about the performance. | 60601-1 | 7.9.2.13 | 9 - System Upkeep | Periodic Maintenance Schedule | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 295.0 | IFU.295 | For ME EQUIPMENT containing rechargeable batteries that are intended to be maintained by anyone other than SERVICE PERSONNEL, IFU shall contain instructions to ensure adequate maintenance. | 60601-1 | 7.9.2.13 | 9 - System Upkeep | Internal Battery Health | Users should monitor the health of the batteries by monitoring the duration to deplete and duration to charge. If you suspect there is something wrong with the battery health of either the Emitter or Cassette, discontinue use and contact MedAI to assess and potentially replace the battery. |
| 296.0 | IFU.296 | The instructions for use shall include a list of ACCESSORIES, detachable parts and materials that the MANUFACTURER has determined are intended for use with the ME EQUIPMENT. | 60601-1 | 7.9.2.14 | 3 - System Overview | System Component and Accessory List | Accessories Available from MedAI |
| 297.0 | IFU.297 | If ME EQUIPMENT is intended to receive its power from other equipment in an ME SYSTEM, the instructions for use shall sufficiently specify such other equipment. | 60601-1 | 7.9.2.14 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | Nominal Output Power: 100W (typical efficiency 86%)Input Rated Voltage / Frequency: 90 - 264 VAC / 50-60 Hz |
| 298.0 | IFU.298 | IFU shall provide advice on the proper disposal of waste products, ME EQUIPMENT, and ACCESSORIES at the end of their EXPECTED SERVICE LIFE. | 60601-1 | 7.9.2.15 | 9 - System Upkeep | End of Life Procedure | At device or accessory end of life, ship products to MedAI to minimize environmental risks associated with disposal. Disposal should always be performed in accordance with local, state, and federal regulations. Disposal of accessories and consumables associated with this equipment should also be performed in compliance with local, state, and federal regulations. All materials and components that could present risks to the environment must be removed from the end-of-life system before disposal. |
| 299.0 | IFU.299 | IFU shall contain the Technical Description (See 7.9.3) or reference to where to find it. | 60601-1 | 7.9.2.16 | 12 - Tech Specs | 12 - Technical Specifications | 12 - Technical Specifications |
| 300.0 | IFU.300 | IFU shall indicate the nature, type, intensity and distribution of emitted radiation. | 60601-1 | 7.9.2.17 | 10 - Radiation Exposure | Dose Outputs | tables |
| 301.0 | IFU.301 | IFU shall contain a unique version identifier such as its date of issue. | 60601-1 | 7.9.2.19 | General | Cover Page | Rev A |
| 269.0 | IFU.269 | IFU shall include all warning and safety notices | 60601-1 | 7.9.2.2 | 11 - Symbols and Labels | Equipment Labels | table |
| 270.0 | IFU.270 | The instructions for use shall provide the OPERATOR or RESPONSIBLE ORGANIZATION with warnings regarding any significant RISKS of reciprocal interference posed by the presence of the ME EQUIPMENT during specific investigations or treatments. | 60601-1 | 7.9.2.2 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | CAUTION: This equipment generates, uses, and can radiate radio frequency energy. The system may cause or be subject to radio frequency interference with other medical and non–medical devices and radio communications. There may be risks of reciprocal interference posed by ME EQUIPMENT. |
| 271.0 | IFU.271 | The instructions for use shall include information regarding potential electromagnetic or other interference between the ME EQUIPMENT and other devices together with advice on ways to avoid or minimize such interference. | 60601-1 | 7.9.2.2 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | If this equipment is found to cause interference (which may be determined by switching the equipment on and off), the operator should attempt to correct the problem by one or more of the following measure(s):Reorienting the MX1 System or the affected device;Increasing the distance between the MX1 System or the affected device; orChanging the power supply for either device so they do not share the same power source. |
| 272.0 | IFU.272 | If ME EQUIPMENT is intended for connection to a separate power supply, either the power supply shall be specified as part of the ME EQUIPMENT or the combination shall be specified as an ME SYSTEM. The instructions for use shall state this specification. | 60601-1 | 7.9.2.3 | 4 - Setting Up the System | Charging | Follow these steps to charge the Emitter or Cassette:steps |
| 273.0 | IFU.273 | For mains-operated ME EQUIPMENT with an additional power source not automatically maintained in a fully usable condition, the instructions for use shall include a warning statement referring to the necessity for periodic checking or replacement of such an additional power source. | 60601-1 | 7.9.2.4 | 9 - System Upkeep | Periodic Maintenance Schedule | Users should monitor the health of the batteries by monitoring the duration to deplete and duration to charge. If you suspect there is something wrong with the battery health of either the Emitter or Cassette, discontinue use and contact MedAI to assess and potentially replace the battery. |
| 275.0 | IFU.275 | If an INTERNAL ELECTRICAL POWER SOURCE is replaceable, the instructions for use shall state its specification. | 60601-1 | 7.9.2.4 | N/A | N/A | Not replaceable by user |
| 278.0 | IFU.278 | If applicable, this description shall include the expected positions of the OPERATOR, PATIENT and other persons near the ME EQUIPMENT in NORMAL USE. | 60601-1 | 7.9.2.5 | 3 - System Overview | Stray Radiation | The Operator Zone or Significant Zone of Occupancy was established for the handheld use and the hands-free use:Handheld: A 60 cm x 60 cm square with a height of 200 cm, with an additional 20 cm x 50 cm x 50 cm volume connecting the Emitter handle to represent an operator’s arm.Hands-free: A 60 cm x 60 cm square with a height of 200 cm, distanced 3.7m from the focal spot, without any connecting volume to the Emitter handle. |
| 277.0 | IFU.277 | The instructions for use shall include a brief description of the ME EQUIPMENT, how the ME EQUIPMENT functions; and the significant physical and performance characteristics of the ME EQUIPMENT. | 60601-1 | 7.9.2.5 | 3 - System Overview | System Component and Accessory List | System Component and Accessory List |
| 279.0 | IFU.279 | The instructions for use shall include information on the materials or ingredients to which the PATIENT or OPERATOR is exposed if such exposure can constitute an unacceptable RISK (see 11.7). | 60601-1 | 7.9.2.5 | N/A | N/A | There are no materials or ingredients that constitute an unacceptable RISK in this product. |
| 280.0 | IFU.280 | The instructions for use shall specify any restrictions on other equipment or NETWORK/DATA COUPLINGS, other than those forming part of an ME SYSTEM, to which a SIGNAL INPUT/OUTPUT PART may be connected. | 60601-1 | 7.9.2.5 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | All connections to the Cassette must be USB-C compliant. |
| 281.0 | IFU.281 | The instructions for use shall indicate any APPLIED PART. | 60601-1 | 7.9.2.5 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | The MX1 System components are Internally Powered while not charging and Class II ME Equipment while wired charging (according to IEC 60601-1). The MX1 Cassette is the only Applied Part: Type B. |
| 282.0 | IFU.282 | If installation of the ME EQUIPMENT or its parts is required, the instructions for use shall contain a reference to where the installation instructions are to be found, or contact information for persons designated by the MANUFACTURER as qualified to perform the installation. | 60601-1 | 7.9.2.6 | N/A | N/A | N/A |
| 283.0 | IFU.283 | If an APPLIANCE COUPLER or MAINS PLUG or other separable plug is used as the isolation means to satisfy 8.11.1 a), the instructions for use shall contain an instruction not to position the ME EQUIPMENT so that it is difficult to operate the disconnection device. | 60601-1 | 7.9.2.7 | 4 - Setting Up the System | Unpacking | CAUTION: Do not position any system components in a way that would block cooling ports and power ports. Blocking ports may lead to device overheating and increased risk of patient or user injury. |
| 284.0 | IFU.284 | IFU shall contain the necessary information for the OPERATOR to bring the ME EQUIPMENT into operation including such items as any initial control settings, connection to or positioning of the PATIENT, etc. | 60601-1 | 7.9.2.8 | 5 - Using the System | Positioning the System and Patient | Positioning the System and Patient |
| 285.0 | IFU.285 | IFU shall detail any treatment or handling needed before the ME EQUIPMENT, its parts, or ACCESSORIES can be used. | 60601-1 | 7.9.2.8 | 4 - Setting Up the System | 4 - Setting Up the System | 4 - Setting Up the System |
| 286.0 | IFU.286 | The instructions for use shall contain all information necessary to operate the ME EQUIPMENT in accordance with its specification. This shall include explanation of the functions of controls, displays and signals, the sequence of operation, and connection and disconnection of detachable parts and ACCESSORIES, and replacement of material that is consumed during operation. | 60601-1 | 7.9.2.9 | 5 - Using the System | 5 - Using the System | 5 - Using the System |
| 287.0 | IFU.287 | The meanings of figures, symbols, warning statements, abbreviations and indicator lights on ME EQUIPMENT shall be explained in the instructions for use. | 60601-1 | 7.9.2.9 | 11 - Symbols and Labels | Symbols | table |
| 302.0 | IFU.302 | Tech Desc shall provide all data essential for safe operation, transport, and storage, and measures or conditions necessary for installing the ME EQUIPMENT, and preparing it for use. | 60601-1 | 7.9.3.1 | 12 - Tech Specs | General Specifications | Conditions for Use, Travel, and Storage |
| 303.0 | IFU.303 | Tech Desc shall include the permissible environmental conditions of use including conditions for transport and storage. See also 7.2.17; | 60601-1 | 7.9.3.1 | 12 - Tech Specs | General Specifications | Conditions for Use, Travel, and Storage |
| 304.0 | IFU.304 | Tech Desc shall include all characteristics of the ME EQUIPMENT, including range(s), accuracy, and precision of the displayed values or an indication where they can be found; | 60601-1 | 7.9.3.1 | 12 - Tech Specs | General Specifications | Values presented by the system are accurate to a certain degree, depending on the value: |
| 305.0 | IFU.305 | Tech Desc shall include any special installation requirements such as the maximum permissible apparent impedance (Distribution network impedance + Power Source impedance) of SUPPLY MAINS | 60601-1 | 7.9.3.1 | N/A | N/A | No special installation requirements |
| 306.0 | IFU.306 | Tech Desc shall include permissible range of values of inlet pressure and flow, and the chemical composition of the cooling liquid if liquid is used for cooling | 60601-1 | 7.9.3.1 | N/A | N/A | no cooling specifications required |
| 307.0 | IFU.307 | Tech Desc shall include a description of the means of isolating the ME EQUIPMENT from the SUPPLY MAINS, if such means is not incorporated in the ME EQUIPMENT | 60601-1 | 7.9.3.1 | 5 - Using the System | Power Off Procedure | Disconnect the charger from the wall outlet by unplugging the AC Cord from the wall outlet. |
| 308.0 | IFU.308 | Tech Desc shall include a description of the means for checking the oil level in partially sealed oilfilled ME EQUIPMENT or its parts. | 60601-1 | 7.9.3.1 | N/A | N/A | No oil-filled containers in system |
| 310.0 | IFU.310 | Tech Desc shall include information pertaining to ESSENTIAL PERFORMANCE and any necessary recurrent ESSENTIAL PERFORMANCE and BASIC SAFETY testing including details of the means, methods and recommended frequency. | 60601-1 | 7.9.3.1 | 9 - System Upkeep | Periodic Maintenance Schedule | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 311.0 | IFU.311 | If Tech Desc is separable from the instructions for use, it shall contain classifications, safety info, description of product, functions, and others | 60601-1 | 7.9.3.1 | N/A | N/A | N/A |
| 309.0 | IFU.309 | Tech Desc shall include a warning statement that addresses the HAZARDS that can result from unauthorized modification of the ME EQUIPMENT. | 60601-1 | 7.9.3.1RMF4.2 | 9 - System Upkeep | Overview | Do Not Disassemble: Unauthorized modification or disassembly of the MX1 System will void the customer warranty, resulting in a non-serviceable unit by MedAI. |
| 312.0 | IFU.312 | Tech Desc shall document minimum qualifications, if present, for SERVICE PERSONNEL | 60601-1 | 7.9.3.2 | N/A | N/A | No Service Personnel, no maintenence for service personnel |
| 313.0 | IFU.313 | Tech Desc shall include, as applicable, the required type and full rating of fuses used in the SUPPLY MAINS external to PERMANENTLY INSTALLED ME EQUIPMENT. | 60601-1 | 7.9.3.2 | N/A | N/A | No PERMANENTLY INSTALLED ME EQUIPMENT |
| 315.0 | IFU.315 | Tech Desc shall include, as applicable, instructions for correct replacement of interchangeable or detachable parts that the MANUFACTURER specifies as replaceable by SERVICE PERSONNEL | 60601-1 | 7.9.3.2 | N/A | N/A | No Service Personnel, no maintenence for service personnel |
| 316.0 | IFU.316 | Tech Desc shall include, as applicable, where replacement of a component could result in an unacceptable RISK, appropriate warnings that identify the nature of the HAZARD and, if the MANUFACTURER specifies the component as replaceable by SERVICE PERSONNEL, all information necessary to safely replace the component. | 60601-1 | 7.9.3.2 | 9 - System Upkeep | Internal Battery Health | CAUTION: If you suspect something is wrong with the battery in a battery-powered component, do not use or disassemble the component, and contact MedAI for servicing. Doing so may cause explosions, burns, and electrical hazards to the user. |
| 317.0 | IFU.317 | Tech Desc shall contain a statement that the MANUFACTURER will make available on request information that will assist SERVICE PERSONNEL to repair parts designated repairable by SERVICE PERSONNEL. | 60601-1 | 7.9.3.3 | N/A | N/A | No Service Personnel, no maintenence for service personnel |
| 318.0 | IFU.318 | The technical description shall clearly identify means to achieve isolation from MAINS | 60601-1 | 7.9.3.4 | 5 - Using the System | Power Off Procedure | Disconnect the charger from the wall outlet by unplugging the AC Cord from the wall outlet. |
| 322.0 | IFU.322 | The requirements for the isolation device shall be specified in the ACCOMPANYING DOCUMENTS. | 60601-1 | 8.11.1 | 3 - System Overview | System Component and Accessory List | The MX1 Wired Charger isolates from and connects to power outlets to charge the Emitter or Cassette. |
| 321.0 | IFU.321 | Means of electrical isolation external to the ME SYSTEM shall be described in the IFU Technical Description | 60601-1 | 8.11.1 b) | N/A | N/A |  |
| 319.0 | IFU.319 | Instructions for use shall instruct the operator not to simultaneously touch the patient and accessible parts that fail leakage test limits, even if unlikely to come into contact. | 60601-1 | 8.4.2 c) | 2 - General Safety | Electrical Safety | WARNING: Do not touch the patient and any exposed metal components, including ports and connector pins, simultaneously as electrical discharge may occur. |
| 320.0 | IFU.320 | IFU shall instruct the situations for when to open access covers. | 60601-1 | 8.4.2 c) | N/A | N/A | N/A |
| 323.0 | IFU.323 | IFU shall describe the use and warnings associated with any moving parts | 60601-1 | 9.2.1 | N/A | N/A | N/A |
| 324.0 | IFU.324 | IFU shall specify NORMAL USE, including the placement/arrangement of doors, drawers, shelves, and the like | 60601-1 | 9.4.2.2 e) | N/A | N/A | N/A |
| 325.0 | IFU.325 | IFU shall describe ME EQUIPMENT's transport position and safe working load in that position, if applicable. | 60601-1 | 9.4.2.4.3 | N/A | N/A | N/A |
| 327.0 | IFU.327 | IFU shall instruct on how to pass over low obstructions in ME EQUIPMENT's transport position, if applicable | 60601-1 | 9.8.3.1 | N/A | N/A | N/A |
| 328.0 | IFU.328 | The IFU shall describe the allowable patient mass of any patient support structure | 60601-1 | 9.8.3.1 | 5 - Using the System | Positioning the System and Patient | CAUTION: Do not load the Cassette with more than 300lbs of total force. Doing so may result in damage to the Cassette and other equipment, or harm to the patient and nearby persons. |
| 329.0 | IFU.329 | The IFU shall disclose the mass of accessories | 60601-1 | 9.8.3.1 | 12 - Tech Specs | General Specifications | The MX1 System major components consist of the following physical properties. |
| 460.0 | IFU.460 | A statement of the environments the ME equipment will be used. Relevant exclusions, as determined by Risk Analysis, shall also be listed. | 60601-1-2 | 5.2.1.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | The MX1 System is intended to be used in both Professional Healthcare Facility and Home Healthcare environments. The purchaser or operator of the MX1 System should ensure that it is only used in the appropriate environment. |
| 461.0 | IFU.461 | The essential performance of ME equipment and a description of what the operator can expect if the Essential Performance is lost or degraded due to EM disturbances. | 60601-1-2 | 5.2.1.1 |  |  |  |
| 462.0 | IFU.462 | A warning regarding stacking and location close to other equipment | 60601-1-2 | 5.2.1.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | CAUTION: Use of the MX1 System adjacent to or stacked with other equipment could result in device failure and should be avoided. If such use is necessary, observe and verify normal operation of the MX1 System in the configuration in which it will be used prior to use. |
| 463.0 | IFU.463 | List of cables, transducers and accessories | 60601-1-2 | 5.2.1.1 | 3 - System Overview | System Component and Accessory List | table |
| 464.0 | IFU.464 | A warning that other cables and accessories may negatively affect EMC performance | 60601-1-2 | 5.2.1.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | WARNING: Other equipment could interfere with the medical device or device system, even if the other equipment complies with CISPR8 emission requirements. |
| 465.0 | IFU.465 | A statement about portable RF communications equipment. Including antennas, can affect medical electrical equipment. The warning should include a use distance such as “…be used no closer than 30 cm (12 inches) to any part of the [ME EQUIPMENT or ME SYSTEM], including cables specified by the manufacturer” | 60601-1-2 | 5.2.1.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | WARNING: Portable RF communications equipment (including peripherals such as antenna cables and external antennas) should be used no closer than 30 cm (12 inches) to any part of the MX1 System, including cables specified by the manufacturer. Otherwise, performance degradation of the equipment could result. |
| 466.0 | IFU.466 | technical description shall describe precautions to be taken to prevent adverse events to the PATIENT and OPERATOR due to ELECTROMAGNETIC DISTURBANCES | 60601-1-2 | 5.2.2.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | If this equipment is found to cause interference (which may be determined by switching the equipment on and off), the operator should attempt to correct the problem by one or more of the following measure(s):Reorienting the MX1 System or the affected device;Increasing the distance between the MX1 System or the affected device; orChanging the power supply for either device so they do not share the same power source. |
| 467.0 | IFU.467 | technical description shall include the compliance for each EMISSIONS and IMMUNITY standard or test specified by this collateral standard, e.g. EMISSIONS class and group and IMMUNITY TEST LEVEL; | 60601-1-2 | 5.2.2.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | General EMC Immunity table |
| 468.0 | IFU.468 | technical description shall include any deviations from this collateral standard and allowances used; | 60601-1-2 | 5.2.2.1 |  |  |  |
| 469.0 | IFU.469 | technical description shall include all necessary instructions for maintaining BASIC SAFETY and ESSENTIAL PERFORMANCE with regard to ELECTROMAGNETIC DISTURBANCES for the EXPECTED SERVICE LIFE. | 60601-1-2 | 5.2.2.1 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | EMC events will not cause unacceptable risks due to degraded essential performance.  Some strong EMC events may require the device to be restarted to exit safety mode and return the device to nominal functioning. |
| 371.0 | IFU.371 | IFU shall include information describe the effects of changes in the SSD on the RADIATION dose to the PATIENT | 60601-1-3 | 9.2 | 10 - Radiation Exposure | Radiation Safety | Precautions should be taken when skin dose in a single location can exceed 1 Gy in dose. In particular anatomy and system configurations required for an examination, patient skin could be significantly closer to the X-ray source, with dose rate increasing as the inverse square of the source-to-skin distance. |
| 372.0 | IFU.372 | IFU shall state the maximum value of the ATTENUATION EQUIVALENT of each item interposed between the PATIENT and the X-RAY IMAGE RECEPTOR and forming part of the X-RAY EQUIPMENT.Values of attenuation equivalent, half-value layer, and quality equivalent filtration are expressed as thicknesses of aluminium of 99,9 % purity or higher | 60601-1-3 | 10.2 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 373.0 | IFU.373 | For diagnostic X-RAY EQUIPMENT specified to be used in combination with ACCESSORIES or other items not forming part of the same or other diagnostic X-RAY EQUIPMENT, the instructions for use shall include a statement drawing attention to the possible adverse effects arising from materials located in the X-RAY BEAM (e.g. parts of an operating table). | 60601-1-3 | 10.2 | 5 - Using the System | Positioning the System and Patient | CAUTION: Do not place objects in the path of the X-ray beam. Doing so may adversely affect the image quality and result in a non-diagnostic exposure. |
| 374.0 | IFU.374 | The ACCOMPANYING DOCUMENTS for all X-RAY TUBE ASSEMBLIES and X-RAY SOURCE ASSEMBLIES shall state the values of LOADING FACTORS that would, if applied at the NOMINAL X-RAY TUBE VOLTAGE, correspond to the maximum specified energy input to the ANODE in one hour. Maximum specified energy input in one hour could be as the value permitted by LOADING in RADIOGRAPHY at the applicable X-RAY TUBE VOLTAGE, according to the RADIOGRAPHIC RATINGS, corresponding to a total CURRENT TIME PRODUCT during one hour; or as the value corresponding to the specified CONTINUOUS ANODE INPUT POWER. | 60601-1-3 | 12.3 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 375.0 | IFU.375 | Appropriate measures to protect the operator and staff against stray radiation, as required in 13.2 through 13.5, shall be called out in the IFU | 60601-1-3 | 13.1 | 10 - Radiation Exposure | Radiation Safety | WARNING: Operators should always wear PPE while using the MX1 System. Both an apron (with 0.5 mm lead equivalent) and a thyroid collar are recommended. Follow any additional state and/or hospital-specific safety procedures and PPE requirements. Failure to wear PPE may result in increased exposure to backscatter radiation and overexposure hazards. |
| 376.0 | IFU.376 | IFU shall designate the significant zone of occupancy (SZO) including the types of examinations for which it is used, the location of the SZO, one profile of STRAY RADIATION in the SZO with respect to height from the floor, one profile containting the point with the highest dose level, information about the effectiveness and application of PROTECTIVE DEVICES specified with the equipment, and instructions for obtaining loading factors if controlled by an automatic control system. | 60601-1-3 | 13.4 | 10 - Radiation Exposure | Stray Radiation | The Operator Zone or Significant Zone of Occupancy was established for the handheld use and the hands-free use:Handheld: A 60 cm x 60 cm square with a height of 200 cm, with an additional 20 cm x 50 cm x 50 cm volume connecting the Emitter handle to represent an operator’s arm.Hands-free: A 60 cm x 60 cm square with a height of 200 cm, distanced 3.7m from the focal spot, without any connecting volume to the Emitter handle. |
| 474.0 | IFU.474 | IFU shall describe means to adjust controls from a distance when equipment is specified exclusively for not being near the patient. | 60601-1-3 | 13.2 | N/A | N/A | Do not have X-RAY EQUIPMENT specified exclusively for examinations that do not need the OPERATOR or staff to be close to the PATIENT |
| 475.0 | IFU.475 | IFU shall describe means to control radiation when equipment is specified exclusively for not being near the patient. | 60601-1-3 | 13.3 | N/A | N/A | Do not have X-RAY EQUIPMENT specified exclusively for examinations that do not need the OPERATOR or staff to be close to the PATIENT |
| 523.0 | IFU.523 | Compliance statements for IEC Standards shall include the MX1 Model or Type Reference, Standard Number (e.g. "60601-2-54"), Version Number (e.g. 3.2), and Year of Standard publication. | 60601-1-3 | 4.1 |  |  |  |
| 456.0 | IFU.456 | All removable sub-assemblies, components, and ACCESSORIES of X-RAY EQUIPMENT are marked to ensure: they can be identified readily and correlated with ACCOMPANYING DOCUMENTS and - interchangeable devices are individually distinguishable to the OPERATOR in NORMAL USE and for the purpose of replacement | 60601-1-3 | 5.1.1 | 3 - System Overview | System Component and Accessory List | Table, including model numbers |
| 457.0 | IFU.457 | ACCOMPANYING DOCUMENTS include the required statements per sub-clauses in Table 2 | 60601-1-3 | 5.2.1 | N/A | N/A | See IFU.354 through IFU.376, IFU.470 through IFU.475 |
| 354.0 | IFU.354 | When dosimetric indications are provided on the EQUIPMENT, the IFU shall contain information and instructions on how to check and maintain the accuracy of dosimetric indications | 60601-1-3 | 5.2.2 | 9 - System Upkeep |  |  |
| 355.0 | IFU.355 | IFU shall include replication of all inaccessible label information marked on items. | 60601-1-3 | 5.2.3 | 11 - Symbols and Labels | Equipment Labels | table |
| 356.0 | IFU.356 | IFU shall contain all information allowing the user to minimize the possibility of exposing PATIENTS to RADIATION dose levels where deterministic effects may occur during the NORMAL USE, to optimise the RADIATION dose delivered to the PATIENTS and to minimize the IRRADIATION of the OPERATORS | 60601-1-3 | 5.2.4.1 | 10 - Radiation Exposure | Radiation Safety | To minimize risk without adversely affecting the clinical objectives, the ALARA standard should be applied. A general guideline is to apply the lowest X-ray tube voltage (kV) and current-time product (mAs) required to provide acceptable image contrast and exposure. |
| 357.0 | IFU.357 | For each INTENDED USE of the EQUIPMENT, the IFU shall provide the RADIATION QUANTITY (like ENTRANCE SURFACE dose or DAP) used for describing the RADIATION dose to the PATIENT (must be useful for assessing the RADIATION RISK); the description of a specified test object representative of an average PATIENT; the procedure for measuring the quantity for the specified test object; the value of the RADIATION QUANTITY when the specified test object is used; and the influence of the main selections available (Mode, loading factors, etc) to the OPERATOR on the value of the specified RADIATION QUANTITY. | 60601-1-3 | 5.2.4.2 | 10 - Radiation Exposure | Skin Entrance Dose | Air Kerma (Kinetic Energy Released per unit Mass), measured in the units of Gray (Gy), is an indication of the radiation delivered to the patient entrance reference point. The MX1 System determines the patient entrance reference point using the Light Detection and Ranging (LIDAR) array in the MX1 Emitter. This is an accurate representation of Source-to-Skin Distance (SSD) as a point along the central X-ray beam axis. |
| 358.0 | IFU.358 | The IFU shall describe (directly or by reference to publication) the method used to provide RADIATION dose indication during NORMAL USE | 60601-1-3 | 5.2.4.3 | 10 - Radiation Exposure | Radiation Reporting Methods | Dose measurements provided by the MX1 System are calculated based on the tables in Dose Output. Dose per Air Kerma is normalized to the measured SID, resulting in the most appropriate representation of the dose applied to the patient. This value is reported as µGy, and is accurate to within 30%. |
| 359.0 | IFU.359 | When clinical protocols are proposed by the MFG and preloaded on the EQUIPMENT, the IFU shall state if they constitute recommendations to be applied directly so as to allow optimized operation or if they are only examples/starting points, to be replaced by more specific protocols developed by the user | 60601-1-3 | 5.2.4.4 | 6 - Capturing Photos and Radios | Radiograph Mode | AiLARA uses a trained algorithm to automatically change the loading factors (kV and mAs) based on the measured anatomy thickness, SID, and SSD. AiLARA sets the lowest loading factors required to acquire a diagnostic image which reduces overexposure and ensures a clinically relevant radiograph. |
| 360.0 | IFU.360 | If there is a possibility in NORMAL USE that the PATIENT can be exposed to deterministic RADIATION dose levels , the IFU shall address this fact and list configurations, etc in which this may occur. Then the IFU shall draw attention to the need and means (loading factors, mode, parameters, etc.) to manage high RADIATION doses; and identify the number of exposures or duration of loading necessary to reach deterministic effects. | 60601-1-3 | 5.2.4.5 | 10 - Radiation Exposure | Radiation Safety | WARNING: In prolonged or abnormal use, this equipment can produce skin dose levels high enough to cause deterministic effects such as skin erythema, skin damage, or hair loss. It is vital that you strictly follow all radiation safety precautions. |
| 361.0 | IFU.361 | IFU shall draw the attention to the need to restrict access to the EQUIPMENT in accordance with local regulations for RADIATION PROTECTION. | 60601-1-3 | 5.2.4.6 | 10 - Radiation Exposure | Radiation Safety | Compliance and caution to federal, state, and local regulations should always be applied, including restricting access to radiation-emitting equipment. |
| 362.0 | IFU.362 | All information necessary to minimize the IRRADIATION of the OPERATORS in NORMAL USE shall be provided. | 60601-1-3 | 5.2.4.6 | 10 - Radiation Exposure | Radiation Safety | To minimize risk without adversely affecting the clinical objectives, the ALARA standard should be applied. A general guideline is to apply the lowest X-ray tube voltage (kV) and current-time product (mAs) required to provide acceptable image contrast and exposure. |
| 363.0 | IFU.363 | For each procedure where OPERATORS have to stay in SIGNIFICANT ZONES OF OCCUPANCY, the IFU shall provide the radiation dose resulting, means to reduce the dose (modes, loading factors, PPE, use precautions), and a list of PPE for radiation protection including those that may not be included in the equipment. | 60601-1-3 | 5.2.4.6 | 10 - Radiation Exposure | Radiation SafetyStray Radiation | Scatter charts/tablesTo minimize risk without adversely affecting the clinical objectives, the ALARA standard should be applied. A general guideline is to apply the lowest X-ray tube voltage (kV) and current-time product (mAs) required to provide acceptable image contrast and exposure.WARNING: Operators should always wear PPE while using the MX1 System. Both an apron (with 0.5 mm lead equivalent) and a thyroid collar are recommended. Follow any additional state and/or hospital-specific safety procedures and PPE requirements. Failure to wear PPE may result in increased exposure to backscatter radiation and overexposure hazards. |
| 364.0 | IFU.364 | The IFU shall state the accuracy of RADIATION output. | 60601-1-3 | 6.3.2 | 10 - Radiation Exposure | Dose Outputs | The below table describes dose outputs for a given tube voltage (kV), current-time product (mAs), and Source-to-Image Distance (SID). |
| 365.0 | IFU.365 | The ACCOMPANYING DOCUMENTS shall state the accuracy of RADIATION output. | 60601-1-3 | 6.3.2 | 10 - Radiation Exposure | Dose Outputs | The below table describes dose outputs for a given tube voltage (kV), current-time product (mAs), and Source-to-Image Distance (SID). |
| 366.0 | IFU.366 | Adequate information is available to operator before, during, and after loading of an x-ray tube, regarding loading factors or modes of operation enabling the operator to determine and preselect optimal conditions for irradiation, and subsequently obtain data necessary for estimation of radiation dose received by patient | 60601-1-3 | 6.4.3 | 12 - Tech Specs | X-ray Generation and Detection Specifications | X-ray Tube Loading Factors Range and Accuracy |
| 470.0 | IFU.470 | Means shall be provided to allow the user to estimate the RADIATION dose delivered to the PATIENT. This requirement may be satisfied by providing information in the ACCOMPANYING DOCUMENTS, by the indication of dosimetric values or by a combination thereof. The resulting accuracy shall also be specified in the ACCOMPANYING DOCUMENTS. | 60601-1-3 | 6.4.5 |  |  |  |
| 471.0 | IFU.471 | The ACCOMPANYING DOCUMENTS shall state the accuracy of AUTOMATIC CONTROL SYSTEMS. | 60601-1-3 | 6.5 | N/A | N/A | No AUTOMATIC CONTROL SYSTEM |
| 472.0 | IFU.472 | Means shall be provided to reduce the influence of RADIATION scattered in the PATIENT to the X-RAY IMAGE RECEPTOR in case of significant influence on the image quality. If such means are removable by the OPERATOR, their presence or absence shall be clearly visible or indicated to the OPERATOR. The proper use of such means shall be described in the instructions for use. | 60601-1-3 | 6.6 |  |  |  |
| 367.0 | IFU.367 | INTENDED USE with metrics describing imaging performance shall be specified and described in the IFU. | 60601-1-3 | 6.7.2 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The MX1 System provides diagnostic-quality images of static and serial radiographic exposures according to the Intended Use: |
| 369.0 | IFU.369 | If a RADIATION DETECTOR or X-RAY IMAGE RECEPTOR is integrated in the X-RAY EQUIPMENT, its contribution to the metrics of imaging performance shall be specified. This contribution should ensure the efficient use of RADIATION. | 60601-1-3 | 6.7.4 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The MX1 System provides diagnostic-quality images of static and serial radiographic exposures according to the Intended Use: |
| 458.0 | IFU.458 | accompanying documents state the quality equivalent filtration in thickness of Al or other suitable reference material and the radiation quality used for its determination for all added filters. The marking is, optionally, provided in the form of a reference to a statement of these particulars in the accompanying documents | 60601-1-3 | 7.3 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 459.0 | IFU.459 | X-ray tube assemblies, filtering materials, and added filters inspected; and accompanying documents reviewed for verification | 60601-1-3 | 7.3 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 473.0 | IFU.473 | Means shall be provided to indicate:the PERMANENT FILTRATION in the X-RAY BEAM;thickness and chemical composition of each ADDED FILTER. | 60601-1-3 | 7.3 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 370.0 | IFU.370 | the ACCOMPANYING DOCUMENTS shall contain particulars of the values or ranges of the FOCAL SPOT TO IMAGE RECEPTOR DISTANCE specified for NORMAL USE; | 60601-1-3 | 8.5.2 | 6 - Capturing Photos and Radios | Radiograph Mode | Emitter Too Far from Cassette (SID is larger than 80cm) |
| 368.0 | IFU.368 | The nominal focal spot values of the X-RAY TUBE(s) FOCAL SPOTS in the EQUIPMENT shall be stated according to IEC 60336:1993 or later versions of IEC 60336 and shall be compatible with each application within the INTENDED USE. | 60601-1-3RMF | 6.7.3RMF3.9 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Focal Spot Size: |
| 377.0 | IFU.377 | The ACCOMPANYING DOCUMENTS may be provided with the X-RAY TUBE ASSEMBLY, or they may be integrated into the ACCOMPANYING DOCUMENTS of any ME SYSTEM for which the X-RAY TUBE ASSEMBLY is compatible. | 60601-2-28 | 201.7.9.1 | N/A | N/A | Included |
| 378.0 | IFU.378 | If an X-RAY TUBE ASSEMBLY is intended to receive its power from other equipment in an ME SYSTEM, or otherwise puts special requirements on the supporting ME SYSTEM, the ACCOMPANYING DOCUMENTS shall sufficiently specify such other equipment to ensure compliance with the requirements of this document. | 60601-2-28 | 201.7.9.1 | N/A | N/A | Included |
| 383.0 | IFU.383 | The instructions for use of an X-RAY TUBE ASSEMBLY shall state the following data as appropriate to the INTENDED USE:SINGLE LOAD RATING;SERIAL LOAD RATING;NOMINAL RADIOGRAPHIC ANODE INPUT POWER according to IEC 60613:2010;NOMINAL CT ANODE INPUT POWER according to IEC 60613:2010;NOMINAL CT SCAN POWER INDEX according to IEC 60613:2010. | 60601-2-28 | 201.7.9.2.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 381.0 | IFU.381 | The second paragraph and Note of General Standard's 7.9.2.14 do NOT apply | 60601-2-28 | 201.7.9.2.14 | N/A | N/A | Recorded elsewhere |
| 382.0 | IFU.382 | Subclause 7.9.2.17 of the general standard does not apply to the Monoblock | 60601-2-28 | 201.7.9.2.17 | N/A | N/A | Recorded elsewhere |
| 379.0 | IFU.379 | For X-RAY TUBE ASSEMBLIES, the ACCOMPANYING DOCUMENTS shall include a warning statement to the effect: “WARNING: To avoid the risk of electric shock, this equipment must only be connected to a supply with protective earth.” | 60601-2-28 | 201.7.9.2.2 | N/A | N/A | N/A |
| 380.0 | IFU.380 | Subclause 7.9.2.3 of the general standard does not apply to the Monoblock | 60601-2-28 | 201.7.9.2.3 | N/A | N/A | Recorded elsewhere |
| 384.0 | IFU.384 | The IFU shall describe a big list of radiation properties of the tube (See clause for full list) | 60601-2-28 | 201.7.9.3.101RMF4.15 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The X-ray Tube Assembly, known as the Monoblock, has a few characteristics: |
| 402.0 | IFU.402 | Where certain unguarded ACCESSIBLE SURFACES of X-RAY TUBE ASSEMBLIES can attain high temperatures, means shall be provided to make it impossible to contact such surfaces for any purposes connected with NORMAL USE.Measures should be taken to avoid all unintentional contact. In such cases the instructions for use shall state information about temperatures of ACCESSIBLE SURFACES to be expected in NORMAL USE; see Tables 22 to 24 of the general standard. | 60601-2-54 | 201.11.101 | 12 - Tech Specs | Environment Safety | CAUTION: The MX1 System device surfaces may reach temperatures up to 43°C (109.4°F) under extended use at the max operating temperature. This temperature limit is appropriate for the healthy skin of adults but may cause discomfort or minor injury when large areas of the skin (10 % of total body surface or more) are in contact with the hot surface, or if unhealthy skin is in contact with the hot surface. |
| 401.0 | IFU.401 | The internal impedance of a supply mains is to be considered sufficiently low for the operation of X-ray equipment for radiography and radioscopy if the value of the apparent resistance of supply mains does not exceed the value specified in the accompanying documentsEither the apparent resistance of supply mains or other appropriate supply mains specifications used in a facility is specified in the accompanying documents | 60601-2-54 | 201.4.10.2 | 12 - Tech Specs |  |  |
| 385.0 | IFU.385 | The ACCOMPANYING DOCUMENTS shall contain quality control procedures to be performed on the X-RAY EQUIPMENT by the RESPONSIBLE ORGANISATION. These shall include acceptance criteria and frequency for the tests. | 60601-2-54 | 201.7.9.1 | 9 - System Upkeep |  |  |
| 386.0 | IFU.386 | For X-RAY EQUIPMENT provided with an integrated digital X-RAY IMAGE RECEPTOR, the IFU shall contain a description of image processing applied to ORIGINAL DATA including the revision number or how to determine it and identification of the version if applicable; | 60601-2-54 | 201.7.9.1 | 7 - Device App |  |  |
| 387.0 | IFU.387 | For X-RAY EQUIPMENT provided with an integrated digital X-RAY IMAGE RECEPTOR, the IFU shall contain a description of the file transfer format of the images acquired with this unit and of any data associated with these images; | 60601-2-54 | 201.7.9.1 | 7 - Device App |  |  |
| 388.0 | IFU.388 | The IFU shall state the highest x-ray tube current obtainable when operated at the nominal x-ray tube voltage for both radioscopy and radiography. | 60601-2-54 | 201.7.9.2.1.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 389.0 | IFU.389 | The IFU shall state the highest x-ray tube voltage obtainable when operated at the highest x-ray tube current for both radioscopy and radiography. | 60601-2-54 | 201.7.9.2.1.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 390.0 | IFU.390 | The IFU shall state which combination of loading factors results in the highest electric power in the high-voltage circuit for both radioscopy and radiography. | 60601-2-54 | 201.7.9.2.1.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 391.0 | IFU.391 | The IFU shall state the nominal electric power with the combiination of loading factors used to calculate the value (80 kV, 0.1 s required per standard). | 60601-2-54 | 201.7.9.2.1.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 392.0 | IFU.392 | The IFU shall state the lowest mAs achievable by the system. | 60601-2-54 | 201.7.9.2.1.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum (Nominal) Loading Factors for Modes of Operation |
| 394.0 | IFU.394 | The instructions for use shall state the maximum symmetrical RADIATION FIELD of the integrated X-RAY SOURCE ASSEMBLY determined according to IEC 60806. | 60601-2-54 | 201.7.9.2.1.102 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Maximum symmetrical radiation field: 22cm diameter circle |
| 395.0 | IFU.395 | For X-RAY EQUIPMENT provided with an integrated X-RAY IMAGE RECEPTOR, the instructions for use shall contain a description of the particular handling and maintenance of the X-RAY IMAGE RECEPTOR. | 60601-2-54 | 201.7.9.2.1.103 | 12 - Tech Specs | X-ray Generation and Detection Specifications | CAUTION: Users should inspect the radiographic images for image quality issues (spots, blurriness, resolution) during each use. If image quality issues occur, discontinue use of the equipment until the problem is corrected and has been verified to be operating correctly and safely.The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 396.0 | IFU.396 | For X-RAY EQUIPMENT the instructions for use shall provide information as required in 203.5. | 60601-2-54 | 201.7.9.2.17 | 10 - Radiation Exposure |  |  |
| 397.0 | IFU.397 | The IFU for the X-RAY SOURCE ASSEMBLY shall describe the specification of the REFERENCE AXIS to which the TARGET ANGLE(s) and the FOCAL SPOT characteristics of the X-RAY SOURCE ASSEMBLY refer | 60601-2-54 | 201.7.9.3.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The X-ray Tube Assembly, known as the Monoblock, has a few characteristics: |
| 398.0 | IFU.398 | The IFU for the X-RAY SOURCE ASSEMBLY shall describe the TARGET ANGLE(s) with respect to the specified REFERENCE AXIS; | 60601-2-54 | 201.7.9.3.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The X-ray Tube Assembly, known as the Monoblock, has a few characteristics: |
| 399.0 | IFU.399 | The IFU for the X-RAY SOURCE ASSEMBLY shall describe the position of the FOCAL SPOT and its tolerances on the REFERENCE AXIS; | 60601-2-54 | 201.7.9.3.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The X-ray Tube Assembly, known as the Monoblock, has a few characteristics: |
| 400.0 | IFU.400 | The IFU for the X-RAY SOURCE ASSEMBLY shall describe the NOMINAL FOCAL SPOT VALUE(s) determined according to IEC 60336 for the specified REFERENCE AXIS. | 60601-2-54 | 201.7.9.3.101 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The X-ray Tube Assembly, known as the Monoblock, has a few characteristics: |
| 421.0 | IFU.421 | The ACCOMPANYING DOCUMENTS shall state the maximum value of the ATTENUATION EQUIVALENT for each of the items listed in Table 203.104 and forming part of the X-RAY EQUIPMENT concerned for the measurement conditions specified in 203.10.101. | 60601-2-54 | 203.10.2 | 12 - Tech Specs | X-ray Generation and Detection Specifications | Parts of the Emitter, Cassette, and Optional Accessories contribute to the filtration of radiation between its generation and absorption for imaging. The below are part of the permanent filtration: |
| 422.0 | IFU.422 | For diagnostic X-RAY EQUIPMENT specified to be used in combination with ACCESSORIES or other items not forming part of the same or another diagnostic X-RAY EQUIPMENT, the instructions for use shall include a statement drawing attention to the possible adverse effects arising from materials located in the X-RAY BEAM (e.g., parts of an operating table). | 60601-2-54 | 203.10.2 | 5 - Using the System | Positioning the System and Patient | CAUTION: Do not place objects in the path of the X-ray beam. Doing so may adversely affect the image quality and result in a non-diagnostic exposure. |
| 403.0 | IFU.403 | The instructions for use shall draw attention to the RISK of local skin dose levels that cause tissue reactions under the INTENDED USE in case of repetitive or prolonged exposure. The effect of the various selectable settings available in both RADIOSCOPY and RADIOGRAPHY on the RADIATION QUALITY, the delivered REFERENCE AIR KERMA or REFERENCE AIR KERMA RATE shall be described | 60601-2-54 | 203.5.2.4.5.101 | N/A | N/A | See IFU.403 through IFU.408 |
| 404.0 | IFU.404 | In the instructions for use, information shall be provided on the available configurations delivered by the MANUFACTURER such as MODES OF OPERATION, settings of LOADING FACTORS and other operating parameters that affect the RADIATION QUALITY or the prevailing value of REFERENCE AIR KERMA (RATE) in the INTENDED USE. If applicable this information shall include:the MODES OF OPERATION in RADIOSCOPY designated e.g. as normal, low or high resolution, or normal, low or high dose mode;the settings in a typical MODE OF OPERATION, as described in 1), giving the default values, and the available ranges of factors that can be varied after the MODE OF OPERATION has been selected;the settings of LOADING FACTORS and other operating parameters in RADIOSCOPY delivering the highest available REFERENCE AIR KERMA RATE;the settings of LOADING FACTORS and other operating parameters in RADIOGRAPHY delivering the highest available REFERENCE AIR KERMA per frame;the settings of the FOCAL SPOT TO IMAGE RECEPTOR DISTANCE, corresponding to minimal and typical values of REFERENCE AIR KERMA or REFERENCE AIR KERMA RATE. | 60601-2-54 | 203.5.2.4.5.101 | N/A | N/A | No Radioscopy |
| 405.0 | IFU.405 | In the instructions for use, for the MODES OF OPERATION and sets of values described in accordance with the settings of b) above, representative values of REFERENCE AIR KERMA (RATE) shall be given, based on measurement by the method described in 203.5.2.4.5.102. | 60601-2-54 | 203.5.2.4.5.101 | N/A | N/A | No Radioscopy |
| 406.0 | IFU.406 | In addition, representative values of REFERENCE AIR KERMA (RATE) based on measurement by the method described in 203.5.2.4.5.102 shall be given in the instructions for use, for respectively the MODES OF OPERATION and sets of values described in accordance with the settings of b) 1) and b) 2) of this clause, and if they are adjustable by the OPERATOR in the MODE OF OPERATION concerned, for two settings of the following factors:selectable ADDED FILTERS;ENTRANCE FIELD SIZE;X-RADIATION pulse repetition frequency. | 60601-2-54 | 203.5.2.4.5.101 | N/A | N/A | No Radioscopy |
| 407.0 | IFU.407 | The IFU shall include:- test geometries and configurations that can be used to verify the values provided for this subclause (IEC 60601-2-54 Subclause 203.5.2.4.5.101) using the measurement method described in IEC 60601-2-54 Subclause 203.5.2.4.5.102 | 60601-2-54 | 203.5.2.4.5.101 | N/A | N/A | No Radioscopy |
| 408.0 | IFU.408 | In the instructions for use, the location of the PATIENT ENTRANCE REFERENCE POINT shall be described as specified for the type of RADIOSCOPY EQUIPMENT. (SEE CLAUSE FOR DETAILS) | 60601-2-54 | 203.5.2.4.5.101 | N/A | N/A | No Radioscopy |
| 409.0 | IFU.409 | X-RAY EQUIPMENT, except MOBILE X-RAY EQUIPMENT, shall be provided with connections for external electrical devices separate from the ME EQUIPMENT that either can prevent the X-RAY GENERATOR from starting to emit X-RADIATION, can cause the X-RAY GENERATOR to stop emitting X-RADIATION; or both.If the state of the signals from these external electrical devices is not displayed on the CONTROL PANEL, the ACCOMPANYING DOCUMENTS shall contain information for the RESPONSIBLE ORGANISATION that this state should be indicated by visual means in the installation. | 60601-2-54 | 203.6.2.1.102 | 8 - System Info and Alerts | Visual and Audible Indicators | table |
| 410.0 | IFU.410 | IFU shall specify the intended use, such that a tube voltage and patient-replacing phantom thickness may be chosen for testing | 60601-2-54 | 203.6.3.2.102 | 1 - Introduction | Intended Use | The MX1 System is a hyper portable X-ray system designed to aid clinicians with point of care visualization through diagnostic X-rays of extremities and hips. The device is intended for use in clinical, surgical, home, and ambulatory environments by trained clinicians. |
| 411.0 | IFU.411 | The ACCOMPANYING DOCUMENTS shall provide information on the performance of the dosimetric indications and describe the operations required to maintain this performance within specification. | 60601-2-54 | 203.6.4.5 | 12 - Tech Specs | Periodic Maintenance Schedule | The user is not required to calibrate or maintain the system other than routine cleaning procedures as described above. MedAI will perform all servicing that may include disassembly or calibration. |
| 412.0 | IFU.412 | The IFU shall describe means to achieve an ADDED FILTER, whether placed or permanent, of not less than 0.1 mm Cu or 3.5 mm Al for pediatric applications. | 60601-2-54 | 203.7.1 | N/A | N/A | N/A |
| 413.0 | IFU.413 | the ACCOMPANYING DOCUMENTS shall include, in the ASSEMBLING INSTRUCTIONS given for particular applications, instructions for attaining the TOTAL FILTRATION required to comply with subclause 7.1 of IEC 60601-1-3 in respect of the items of X-RAY EQUIPMENT concerned. | 60601-2-54 | 203.7.1.101 | N/A | N/A | N/A |
| 415.0 | IFU.415 | The IFU must state If the X-RAY BEAM AXIS does not coincide with the REFERENCE AXIS, according to 203.8.104. | 60601-2-54 | 203.8.101 | 10 - Radiation Exposure | Collimation Sizing | The Automatic Collimator, active by default, will grow or shrink the X-ray field to fit within the bounds of the Active Area. Pointing the Emitter perpendicular from and onto the center of the Cassette Active Area will grow the size of the Automatic Collimator’s Field Size. The field indicator will turn red when a valid, safe field size cannot be obtained. |
| 417.0 | IFU.417 | The instructions for use shall contain the information necessary to enable the OPERATOR to determine, prior to LOADING, the extent of all X-RAY FIELDS for the INTENDED USE, in terms of their dimensions at appropriate FOCAL SPOT TO IMAGE RECEPTOR DISTANCES for the available selections, combinations and settings of the BEAM LIMITING DEVICES. | 60601-2-54 | 203.8.102.3 | 6 - Capturing Photos and Radios | Viewfinder and Emitter Touchscreen | Indicates the expected x-ray field before shooting, either with the automatic collimator or when a collimating puck is selected. |
| 418.0 | IFU.418 | The description of a method to check the dimensions of the LIGHT FIELD at the appropriate distance from the FOCAL SPOT shall be included in the ACCOMPANYING DOCUMENTS. | 60601-2-54 | 203.8.102.5 | N/A | N/A | N/A |
| 419.0 | IFU.419 | The ACCOMPANYING DOCUMENTS shall describe the positions of the X-RAY BEAM available in NORMAL USE, in terms of its locations with respect to relevant IMAGE RECEPTION AREAS and its angles with respect to relevant IMAGE RECEPTOR PLANES. | 60601-2-54 | 203.8.104 | 6 - Capturing Photos and Radios | Tracking System | While in Radiograph Mode, the Emitter and Cassette will light with green or red colors to indicate whether the Tracking System will allow X-ray emission (Green) or not (Red). Pulling the trigger will emit X-rays ONLY when the Tracking System allows them. There may be a few different reasons for Tracking to not allow X-rays or ‘turn red’:Emitter Pointed Off-Target (X-ray field would land outside the usable detector active area)Emitter Too Far from Cassette (SID is larger than 80cm)Emitter Too Close to Cassette (SID is smaller than 20cm)The Emitter’s Front Face is covered (Emitter camera cannot see the Cassette IR LEDs)Cassette LEDs covered with thick material (Emitter camera cannot see the Cassette IR LEDs) |
| 420.0 | IFU.420 | If the X-RAY BEAM AXIS is not coinciding with the REFERENCE AXIS, the position and the angle of the X-RAY FIELD and the plane of interest relative to the REFERENCE AXIS shall be described in the instructions for use. | 60601-2-54 | 203.8.104 | 6 - Capturing Photos and Radios | Aim and Collimate | The Automatic Collimator, active by default, will grow or shrink the X-ray field to fit within the bounds of the Active Area. Pointing the Emitter perpendicular from and onto the center of the Cassette Active Area will grow the size of the Automatic Collimator’s Field Size. The field indicator will turn red when a valid, safe field size cannot be obtained. |
| 414.0 | IFU.414 | IFU shall describe means to limit the beam to within the IMAGE RECEPTION AREA, per the test in 203.8.5.3 | 60601-2-54 | 203.8.5.3 | 6 - Capturing Photos and Radios | Radiograph Mode | The MX1 is equipped with different x-ray beam collimation options. Access the Collimation Menu with a connected Device App through the Acquisition Page or top right Menu button.  There are three methods of collimating: Automatic, Manual Collimator, and Pucks. |
| 438.0 | IFU.438 | The name and publication date of the standard to which the product was classified shall be included on the explanatory label, on the labels shown in 7.2 to 7.7 or elsewhere in close proximity on the product. For Class 1 and Class 1M, instead of the labels on the product, the information may be contained in the information for the user. | 60825-1 | 7.9 | 11 - Symbols and Labels | Symbols | Note: The lasers on the Emitter are a CLASS 1 LASER PRODUCT, per IEC 60825 / Edition 3.0, 2014 |
| 442.0 | IFU.442 | IFU shall include a summary of the medical device wireless functions and specific wireless technology incorporated into the medical device or device system, including equipment or system specifications (e.g., the standard IEEE 802.11 b/g, IEEE 802.15.4 BluetoothTM class II) | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications |  |
| 443.0 | IFU.443 | IFU shall include a summary of the operating characteristics of the wireless technology, effective RF radiated power output and operating range, modulation, and bandwidth of receiving section; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications |  |
| 444.0 | IFU.444 | IFU shall include a brief description of the wireless QoS needed for safe and effective operation; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications | Quality of Service (QoS) |
| 445.0 | IFU.445 | IFU shall include a brief description of the recommended wireless security measures such as the WPA2 wireless encryption for IEEE 802.11 technology; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications | I see in the table but do we need to add brief description to cybersecurity controls section? |
| 446.0 | IFU.446 | IFU shall include information addressing wireless issues and what to do if problems occur; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 8 - System Info and Alerts | Device Troubleshooting | In wireless info section( took this from an IFU I had from Philips): If this equipment does cause harmful interference to radio or television reception, which can be determined by moving the equipment away and back, the user is encouraged to try to correct the interference by one or more of the following measures: • Reorient or relocate the receiving antenna• Increase the separation between the equipment and receiver • Consult the dealer or an experienced radio/TV technician for helpAlso this caution:CAUTION: MX1 System has been tested in wireless environments consisting of different wireless technologies (Bluetooth, WiFi 802.11 b and cellular communications) with multiple transmitters used simultaneously. If using the MX1 System in environments where other wireless technologies are being used, the user should evaluate the potential risk of interference. It may be necessary to take mitigation measures such as re-orienting or relocating the MX1 System or shielding the location. |
| 447.0 | IFU.447 | IFU shall include information about any wireless coexistence issues and mitigations. This can include precautions for proximity to other wireless products, and specific recommendations for separation distances from such products; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications | This equipment has been tested and found to comply with the limits for a class B digital device, pursuant to part 15 of the FCC Rules. These limits are designed to provide reasonable protection against harmful interference in a residential installation. This equipment generates, uses and can radiate radio frequency energy and if not installed and used in accordance with the instructions, may cause harmful interference to radio communications. However, there is no guarantee that interference will not occur in a particular installation. If this equipment does cause harmful interference to radio or television reception, which can be determined by moving the equipment away and back, the user is encouraged to try to correct the interference by one or more of the following measures: • Reorient or relocate the receiving antenna• Increase the separation between the equipment and receiver • Consult the dealer or an experienced radio/TV technician for help |
| 448.0 | IFU.448 | IFU shall include appropriate EMC and telecommunications standards compliance and test results summary; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications | EMC section |
| 449.0 | IFU.449 | IFU shall include appropriate RF wireless communications information such as those required by FCC rules; | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications | The WiFi internet adapter has been tested and complies with the specifications for a Class B digital device, pursuant to Part 15 of the FCC Rules.  Operation is subject to the following two conditions:(1) This device may not cause harmful interference, and (2) this device must accept any interference received, including interference that may cause undesired operation. |
| 450.0 | IFU.450 | IFU shall include warnings about possible effects from RF sources in the vicinity of the device (e.g., electromagnetic security systems, cellular telephones, RFID or other inband transmitters). | FDA Guidance | Radio Frequency WirelessTechnology in Medical Devices | 12 - Tech Specs | Electrical and Electromagnetic Specifications | CAUTION: MX1 System has been tested in wireless environments consisting of different wireless technologies (Bluetooth, WiFi 802.11 b and cellular communications) with multiple transmitters used simultaneously. If using the MX1 System in environments where other wireless technologies are being used, the user should evaluate the potential risk of interference. It may be necessary to take mitigation measures such as re-orienting or relocating the MX1 System or shielding the location. |
| 511.0 | IFU.511 | The IFU shall desribe the Emitter's permanent total filtration | 60601-1-3 | 7.3 |  |  |  |
| 501.0 | IFU.501 | A Significant Zone of Occupation (60x60x200cm minimum) shall be designated and described in the IFU, along with details on its scatter radiation profile, test methods used to create it, and location relative to the System | 60601-1-3 | 13.4 |  |  |  |
| 502.0 | IFU.502 | The IFU shall describe the test arrangement used to measure the Scatter Radiation map | 60601-1-3 | 13.4 |  |  |  |
| 503.0 | IFU.503 | The IFU shall provide information on how to check and maintain the accuracy of dosimetric indications provided by the device (DAP, Dose, etc.) | 60601-1-3 | 5.2.2 |  |  |  |
| 517.0 | IFU.517 | The IFU shall include technical descriptions with information necessary to maintain compliance with IEC 60601-1-3 standard within the relevant main assemblies for items supplied separately from the main assembly | 60601-1-3 | 5.2.3 |  |  |  |
| 519.0 | IFU.519 | The IFU shall include information to minimize patient and operator dose levels where deterministic effects may occur. | 60601-1-3 | 5.2.4.1 |  |  |  |
| 520.0 | IFU.520 | The IFU shall include the radiation dose quantity, considering all paramaters that would effect it, to the patient and test methods to determine it. | 60601-1-3 | 5.2.4.2 |  |  |  |
| 521.0 | IFU.521 | The IFU shall contain the method used to provide radiation dose indication | 60601-1-3 | 5.2.4.3 |  |  |  |
| 485.0 | IFU.485 | The IFU shall specify a procedure for the removal of dust from the emitter | 60601-2-43 | 201.11.6.5.102 |  |  |  |
| 524.0 | IFU.524 | IFU shall state the importance of regularly checking storge capacity and securing or archiving important records. | 60601-2-43 | 201.12.4.101.2 |  |  |  |
| 486.0 | IFU.486 | The IFU shall state that if Radiography Mode(s) are intentionally misued for real-time imaging, the Image Display Delay may be longer than in Radioscopy Mode. | 60601-2-43 | 201.12.4.102 |  |  |  |
| 487.0 | IFU.487 | The IFU shall indicate: The time necessary to initiate emergency radioscopy mode after a recoverable failure, the time to restore all functions of the system after a recoverable failure, and the required procedure(s) for recovering recoverable failures. | 60601-2-43 | 201.4.101 |  |  |  |
| 491.0 | IFU.491 | IFU shall include a list of recommended radiation protective devices or accessories | 60601-2-43 | 201.7.9.2.101 |  |  |  |
| 516.0 | IFU.516 | The IFU shall include instructions to configure the system to permit CPR. | 60601-2-43 | 201.7.9.2.102 |  |  |  |
| 492.0 | IFU.492 | The IFU shall include a reproduction of the Emergency Instructions. | 60601-2-43 | 201.7.9.2.103 |  |  |  |
| 513.0 | IFU.513 | The Emergency Instructions shall be provided in a non-electronic form that is resistant to damage. | 60601-2-43 | 201.7.9.2.103 |  |  |  |
| 514.0 | IFU.514 | The Emergency Instructions shall include instruction for restart or recovery in case of recoverable failure or failure of SUPPLY MAINS,  instruction for location, function, and operation of the IRRATDIATION disabling switch, and a list of emergency functions. | 60601-2-43 | 201.7.9.2.103 |  |  |  |
| 515.0 | IFU.515 | The IFU shall explain any IPXY marking on MX1 components | 60601-2-43 | 201.7.9.2.105 |  |  |  |
| 518.0 | IFU.518 | The IFU shall recommend not using the IRRADIATION disabling switch during an exam. | 60601-2-43 | 203.5.2.4.101 |  |  |  |
| 504.0 | IFU.504 | The IFU shall provide Test Geometries and configurations that can be used to verify Reference Air Kerma and Air Kerma Rates for user-adjustable settings as required in #60 | 60601-2-43 | 203.5.2.4.5.101 |  |  |  |
| 505.0 | IFU.505 | The IFU shall provide Reference Air Kerma and Reference Air Kerma Rate Values of each user-adjustable setting or mode, including added filters or field sizes | 60601-2-43 | 203.5.2.4.5.101 |  |  |  |
| 506.0 | IFU.506 | The IFU shall provide one set of REFERENCE AIR KERMA values typical of Radiography for distinctive types of procedure for which the system is intended for use | 60601-2-43 | 203.5.2.4.5.101 |  |  |  |
| 507.0 | IFU.507 | The IFU shall describe the PATIENT ENTRANCE REFERENCE POINT location | 60601-2-43 | 203.5.2.4.5.101 |  |  |  |
| 488.0 | IFU.488 | The IFU shall include how to check the software version and describe the file format of the images | 60601-2-54 | 201.7.9.1 |  |  |  |
| 490.0 | IFU.490 | IFU shall include instructions for procedures the User or Facility should perform, including criteria and frequency, for ensuring the quality of x-ray delivery and sensitivity | 60601-2-54 | 201.7.9.1 |  |  |  |
| 499.0 | IFU.499 | The IFU shall contain a technical description that includes detail on reference axis, target angles, position of the focal spot, etc. | 60601-2-54 | 201.7.9.3.101 |  |  |  |
| 500.0 | IFU.500 | The IFU shall state the maximum attenuation value of parts between the patient and detector. | 60601-2-54 | 203.10.2 |  |  |  |
| 522.0 | IFU.522 | Stray (Scatter) radiation shall be tested and reported in the IFU, using the test setup in the standard. Review 60601-2-54 203.13.6 and 60601-2-43 203.13.6 for Test Setup | 60601-2-4360601-1-360601-2-4360601-2-5460601-2-54 | 203.13.413.6203.13.6203.13.6203.13.6 |  |  |  |
| 508.0 | IFU.508 | The IFU shall state the user-adjustable settings that affect radiation output and how, including:- The LOW and NORMAL modes of operation in Radioscopy- The Settings in Radioscopy that can be varied and impact dose or image quality- The Settings that would generate the highest Reference Air Kerma Rate, in Radioscopy- The Settings that would generate the highest Reference Air Kerma per image, in Radiography- The SID corresponding to minimal and typical values of Reference Air Kerma or Reference Air Kerma Rate | 60601-2-5460601-1-3 | 203.5.2.4.5.10112.3 |  |  |  |
| 509.0 | IFU.509 | The IFU shall draw attention to the risk of local skin levels that could cause deterministic effects, with the effects of selectable settings on the radiation quality, the reference air kerma or reference air kerma rate. | 60601-1-360601-2-54 | 5.2.4.5203.5.2.4.5.101 |  |  |  |
| 510.0 | IFU.510 | The IFU shall describe the minimum and maximum loading times and the corresponding controls, if available | 60601-2-54 | 203.6.2.1 |  |  |  |
| 512.0 | IFU.512 | The IFU shall describe methods to check the operation of the automatic collimator and reduce the size to a selectable size | 60601-2-54 | 203.8.102.1 |  |  |  |
| 190.0 | IFU.190 | IFU shall indicate the expected scatter radiation about the Emitter, including the Zone of Occupancy | RMF | R6.47RMF3.26RMF3.21 | 10 - Radiation Exposure | Stray Radiation | charts and graphs |
| 169.0 | IFU.169 | IFU shall specify cleaning solutions that should not be used to clean the system components. | RMF | RMF1.11.42 | 9 - System Upkeep | Overview | CAUTION: Only the cleaning and disinfecting agents listed in these Instructions for Use have been tested for compatibility and effectiveness by MedAI. Do not use other cleaning solutions, since certain chemical combinations may deteriorate the MX1 System plastics prematurely. |
| 244.0 | IFU.244 |  | RMF | RMF1.11.68RMF1.11.69 | 9 - System Upkeep | Internal Battery Health | WARNING: Batteries in the Emitter and Cassette are not intended to be replaced by users. Battery replacement by inadequately trained personnel could result in excessive temperatures, fire, or explosion. Contact MedAI if you suspect a battery needs replacement. |
| 204.0 | IFU.204 | IFU shall Warn to only use the supplied charger to charge the Emitter and Cassette. | RMF | RMF1.12.2RMF1.3.24RMF1.3.25 | 4 - Setting Up the System | Charging | WARNING: Use only MedAI-supplied battery chargers and approved accessories. Use or connection of incompatible chargers and accessories may lead to major shock, burn, or injury. See Section 3 - System Overview, for a list of approved system components and accessories. |
| 227.0 | IFU.227 | IFU shall include Caution not to operate device if condensation is suspected within the equipment housing | RMF | RMF1.3.28 | 2 - General Safety | Water Ingress | CAUTION: If you suspect visible condensation presence within equipment housing, do not operate the system, disconnect any chargers from the wall outlet, and contact MedAI for assistance |
| 178.0 | IFU.178 | IFU shall Caution against setting up or using the system if any damage is observed or suspected. | RMF | RMF1.3.30 | 4 - Setting Up the System | 4 - Setting Up the System | CAUTION: Inspect equipment for damage before each use. If any damage to the packaging or device is observed, do not proceed with set-up and contact MedAI for assistance. Set up and use of a damaged device may result in minor shock or injury. |
| 249.0 | IFU.249 | IFU shall describe the WEEE Symbol marked on the System component(s). | RMF | RMF1.3.35 | 11 - Symbols and Labels | Symbols | WEEE Symbol |
| 211.0 | IFU.211 | IFU shall warn against connecting the system to a multiple socket-outlet such as a power strip. | RMF | RMF1.3.37 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | WARNING: Multi-socket outlets or power strips are strictly prohibited for connection unless they are rated to IEC 60601-1 and are provided with all necessary markings and certificates of conformance. Connecting the MX1 System to multi-socket outlets that are not rated to IEC 60601-1 may result in fire. |
| 231.0 | IFU.231 | IFU shall Warn against using the wrong type of fire extinguisher, and specify the correct type must be available where the equipment is being used. | RMF | RMF1.3.38 | 2 - General Safety | Electrical Safety | WARNING: A class C fire extinguisher, which meets applicable regulations and standards, must be available wherever the MX1 System is being used. Using the wrong type of fire extinguisher presents electrical shock and burn hazards. |
| 233.0 | IFU.233 | IFU shall contain the warning: "WARNING: This equipment either produces or is used in the vicinity of ionizing radiation. Observe proper safety procedures according to radiation guidelines laid out by the hospital." | RMF | RMF1.3.40 | 10 - Radiation Exposure | Radiation Safety | WARNING: This equipment either produces or is used in the vicinity of ionizing radiation. Observe proper safety procedures according to radiation guidelines laid out by the hospital. |
| 215.0 | IFU.215 | IFU shall include a caution not place objects in the path of the x-ray beam. Doing so may cause adverse effects to the image | RMF | RMF1.3.41 | 5 - Using the System | Positioning the System and Patient | CAUTION: Do not place objects in the path of the X-ray beam. Doing so may adversely affect the image quality and result in a non-diagnostic exposure. |
| 214.0 | IFU.214 | IFU shall Note to not move or reposition the Cassette while in use with a patient. | RMF | RMF1.3.42 | 10 - Radiation Exposure | Radiation Safety | CAUTION: Movement of the patient, cassette, or emitter during imaging may increase risk of non-diagnostic exposure and/or patient injury. Avoid abrupt shaking or movement of the patient, cassette, and emitter. |
| 173.0 | IFU.173 | IFU shall Warn against unauthorized modification or disassembly of the system and that doing so will void the customer warranty and render the system unservicable. | RMF | RMF1.3.48RMF1.3.65RMF1.4.16RMF1.4.17RMF1.4.18 | 9 - System Upkeep | OverviewPeriodic Maintenance Schedule | WARNING: Do not disassemble the MX1 System or MX1 System accessories. Unauthorized modification or disassembly of the MX1 System may result in electric shock and will void the customer warranty, resulting in a non-serviceable unit by MedAI. |
| 240.0 | IFU.240 | IFU shall Caution against stacking the System or other equipment to avoid device failure. | RMF | RMF1.3.50 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | CAUTION: Use of the MX1 System adjacent to or stacked with other equipment could result in device failure and should be avoided. If such use is necessary, observe and verify normal operation of the MX1 System in the configuration in which it will be used prior to use. |
| 198.0 | IFU.198 | IFU shall instruct the user to check equipment for damage before each use | RMF | RMF1.3.52 | 5 - Using the System | Unpacking | WARNING: DO NOT USE IF DAMAGED. If any part of the device is known (or suspected) to be damaged or defective, do not use the system and contact MedAI for assistance. Operation of the equipment with defective components could expose the operator or the patient to radiation or other safety hazards. This could lead to fatal or other serious personal injury, or to clinical misdiagnosis/mistreatment. |
| 209.0 | IFU.209 | IFU shall include a warning to inspect device after being dropped | RMF | RMF1.3.52RMF1.11.14RMF1.11.22RMF1.11.30RMF1.11.38 | 5 - Using the System | Positioning the System and Patient | WARNING: If any part of the MX1 System is dropped: Ensure that the patient has not sustained injuryInspect the MX1 System for any damagesWipe down the MX1 System before re-use as described in Section 9 - Routine cleaning Information |
| 180.0 | IFU.180 | IFU shall include instructions for positioning the cassette, patient, and patient environment for standing or weight-bearing on the Cassette. | RMF | RMF1.3.53 | 5 - Using the System | Positioning the System and Patient | If desired, patients may stand on the Cassette for some weight-bearing images. Place the Cassette on a hard, dry floor with balancing supports nearby. Do not allow patients who weigh more than 300lbs or are at risk of tripping, slipping, or falling to stand on the Cassette. |
| 234.0 | IFU.234 | IFU shall caution the user to manage risks of radio interferance and disruption and include mitigation measures. | RMF | RMF1.3.55 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | CAUTION: MX1 System has been tested in wireless environments consisting of different wireless technologies (Bluetooth, WiFi 802.11 b and cellular communications) with multiple transmitters used simultaneously. If using the MX1 System in environments where other wireless technologies are being used, the user should evaluate the potential risk of interference. It may be necessary to take mitigation measures such as re-orienting or relocating the system or shielding the location. |
| 176.0 | IFU.176 | IFU shall Caution to not place more weight or load on the Cassette than the rated weight. | RMF | RMF1.5.28RMF1.5.29RMF1.5.30 | 5 - Using the System | Positioning the System and Patient | CAUTION: Do not jump on the Cassette, or allow patients who weigh more than 300 lb or are at risk of tripping, slipping, or falling to stand on the Cassette. |
| 170.0 | IFU.170 | IFU shall describe physiological hazards of energy outputs from the Emitter, including ionizing radiation, laser radiation, infrared light, and LIDA R laser energy. | RMF | RMF1.8.7RMF1.8.9RMF1.8.11 | 3 - System Overview | Description of Components | WARNING: While the system is actively tracking or capturing, the Emitter is emitting potentially hazardous energy such as radiation, Class 1 lasers, and Class 1 invisible infrared light. Never look directly into these lasers/lights or point them at others; eye exposure to hazardous energy sources may result in serious eye injury. |
| 174.0 | IFU.174 | IFU shall Note to not place unintended objects in the path of the x-ray beam, including accessories. | RMF | RMF1.8.8 | 5 - Using the System | Positioning the System and Patient | CAUTION: Do not place objects in the path of the X-ray beam. Doing so may adversely affect the image quality and result in a non-diagnostic exposure. |
| 183.0 | IFU.183 | IFU shall describe the System's Intended Use Environment | RMF | RMF2.16RMF2.17RMF2.18 | 1 - Introduction | Intended Use | The MX1 System is a hyper portable X-ray system designed to aid clinicians with point of care visualization through diagnostic X-rays of extremities and hips. The device is intended for use in clinical, surgical, home, and ambulatory environments by trained clinicians. |
| 196.0 | IFU.196 | IFU shall state that in the patient environment, connected devices and equipment should be 60601-1 certified or should be IEC 60950 or IEC 62368 certifiedMaybe something like:CAUTION: Connected devices and equipment used with the MX1 System should be IEC 60601-1-2 certified. Failure to use IEC 60601-1-2 certified devices and comply with recommended separation distances may result in electromagnetic interference. | RMF | RMF2.2RMF2.10RMF2.11RMF2.14RMF2.15 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | When operating within the patient environment, equipment that meets the requirements of IEC 60601-1 or an equivalent standard and contain all necessary markings and certificates of conformance should be used. If IEC 60601-1 rated components are not available, operators should use equipment certified in conformance IEC 60950 or IEC 62368. |
| 199.0 | IFU.199 | IFU shall provide available range of loading factors and subsequent loading factors. | RMF | RMF3.1RMF3.2RMF3.3RMF3.4 | 12 - Tech Specs | X-ray Generation and Detection Specifications | X-ray Tube Loading Factors Range and Accuracy |
| 171.0 | IFU.171 | IFU shall describe the specifications of the Detector. | RMF | RMF3.10 | 12 - Tech Specs | X-ray Generation and Detection Specifications | The MX1 System provides diagnostic-quality images of static and serial radiographic exposures for extremities: |
| 200.0 | IFU.200 | IFU shall call out locations in the UI of loading factors. | RMF | RMF3.5 | 7 - Device App | Acquisition Page | On the bottom left of the Active Capture, the kV, current-time product, dose value, dose area product (DAP), and local date-time of capture are overlaid onto the image. All images are labeled with this information for quality purposes. |
| 194.0 | IFU.194 | IFU shall specify the dose output associated with all variations of user-controllable parameters and suggest parameters for anatomy to imaged | RMF | RMF3.7 | 10 - Radiation Exposure | Dose Outputs | tablesAlso 6 - Capturing Radiographs and Photographs, Radiograpoh Mode, Table of Thickness Dose |
| 172.0 | IFU.172 | IFU shall instruct that the device must be operated in accordance with local/state/federal laws and regulations | RMF | RMF4.2 | 1 - Introduction | Owner’s Responsibility | The MX1 System is sold with the understanding that the operator assumes sole responsibility for radiation safety (as well as any state, provincial, or local regulatory compliance) |
| 115.0 | IFU.115 | IFU shall caution that the system generates and uses energy that may cause electromagnetic disturbances. | RSK | R1.158RMF1.3.12 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | CAUTION: This equipment generates, uses, and can radiate radio frequency energy. The system may cause or be subject to radio frequency interference with other medical and non–medical devices and radio communications. There may be risks of reciprocal interference posed by ME EQUIPMENT. |
| 60.0 | IFU.60 | IFU shall include instructions for determining the field size. | RSK | R1.106R1.116R1.124 | 6 - Capturing Photos and Radios | Radiograph Mode | Point the Emitter towards the Cassette’s active area and Viewfinder will display the field indicator. |
| 179.0 | IFU.179 | IFU shall describe how the System indicates the X-ray field size before imaging | RSK | R1.106R1.116R1.124RMF3.15 | 10 - Radiation Exposure | Collimation Sizing | table |
| 54.0 | IFU.54 | IFU shall CAUTION to inspect for image quality issues and contact MedAI if they persist. | RSK | R1.122 | 12 - Tech Specs | X-ray Generation and Detection Specifications | CAUTION: Users should inspect the radiographic images for image quality issues (spots, blurriness, resolution) during each use. If image quality issues occur, discontinue use of the equipment until the problem is corrected and has been verified to be operating correctly and safely. |
| 25.0 | IFU.25 | IFU shall Caution against moving the Emitter during capture | RSK | R1.127R6.56 | 5 - Using the System | Positioning the System and Patient | Position the Emitter still during any capture to reduce the effect of motion blur and the chances of a non-diagnostic radiation exposure. |
| 48.0 | IFU.48 | IFU shall describe the Emitter's laser class and Caution the user to not point the Emitter's lasers near eyes. | RSK | R1.132R1.134R1.136R1.138RMF1.3.32 | 3 - System Overview | Description of Components | WARNING: While the system is actively tracking or capturing, the Emitter is emitting potentially hazardous energy such as radiation, CLASS 1 visible laser energy, and CLASS 1 infrared (non-visible) light. Never look directly into these lasers/lights or point them at others; eye exposure to hazardous energy sources may result in serious eye injury. |
| 34.0 | IFU.34 | IFU shall clearly state the device's expected service life, including battery life expectations. | RSK | R1.150R1.151R1.152R8.6R8.59R8.60 | 9 - System Upkeep | End of Life Procedure | The System has an expected service life of 5 years. |
| 481.0 | IFU.481 | IFU shall instruct the user to contact MedAI when experiencing frequent Emitter overheating or when something is caught in the Emitter fan | RSK | R1.165 | 8 - System Info and Alerts | Device Troubleshooting | The Cassette or Emitter overheats often - The cooling means inside the component is not functioning correctly - Discontinue use of the device and contact MedAI.An item gets jammed or stuck inside or under the Emitter’s fan cover - Discontinue use of the device and contact MedAI. |
| 108.0 | IFU.108 | IFU shall warn to connect and use only approved devices, components, batteries, and accessories.IFU to only use provided charger / batteries | RSK | R1.20R1.69R6.19R6.20R6.22R6.24R6.26R6.27R6.28R6.45R16.9R16.28 | 3 - System Overview | 3 - System Overview | WARNING: Use only MedAI-supplied battery chargers and approved accessories. Use or connection of incompatible chargers and accessories may lead to major shock, burn, or injury. See Section 3 - System Overview, for a list of approved system components and accessories. |
| 114.0 | IFU.114 | IFU shall warn against disassembling, opening, or modifying the device without authorization of MedAI | RSK | R1.5R1.7R1.9R1.11R1.19R1.22RMF1.3.20RMF1.3.19 | 2 - General Safety | Electrical Safety | WARNING: Never modify or disassemble any of the system components. Only personnel authorized by MedAI may modify or repair the MX1 System. |
| 50.0 | IFU.50 | IFU shall Caution against using the device if the user suspects damage or tampering. | RSK | R1.68R1.72R1.76R1.84 | 1 - Introduction | Owner’s Responsibility | MedAI, and its agents or representatives, do not accept responsibility for the following: equipment which has been damaged, modified, or tampered with in any way.New (add, not replace):WARNING: If it appears the MX1 System has been damaged, modified, or tampered with in any way, do not use the device and contact MedAI. Use of devices that have been modified or tampered with may result in serious injury. |
| 51.0 | IFU.51 | IFU shall warn against using the system in a volatile atmospheric environment, including O2-rich environments. | RSK | R1.81R1.74R1.82R3.16RMF1.3.36 | 2 - General Safety | Environment Safety | WARNING: Do not use this equipment in envionments rich with oxygen, nitrous oxide, or flammable anesthetics. Device use in potentially flammable environements may lead to fire. |
| 65.0 | IFU.65 | IFU shall include instructions to capture single and serial radiographs. | RSK | R11.13R11.14 | 6 - Capturing Photos and Radios | Dynamic Digital Radiography (DDR) | A DDR can be taken after setting the proper loading factors by pointing the Emitter towards the Cassette and anatomy to be imaged, and pulling and holding the trigger active area while the Tracking System allows x-rays. |
| 78.0 | IFU.78 | IFU shall instruct how to change capture modes | RSK | R11.16R11.17R11.18R17.6R17.7 | 5 - Using the System | Imaging Modes | Cycle through modes from the Emitter Viewfinder screen by pressing the Mode Indicator or Middle Button: |
| 71.0 | IFU.71 | IFU shall includie instructions for viewing an image. | RSK | R11.19R11.20R11.21 | 7 - Device App | Acquisition Page |  |
| 72.0 | IFU.72 | IFU shall instruct users on how to set up a proper PACS system. | RSK | R11.26R11.27R11.28R11.30R11.31 | 4 - Setting Up the System | DICOM (PACS) and RIS Setup | On the Device UI, Open the Cloud Drawer by pressing the Cloud Icon.Select the Settings Gear near the PACS connections to open the PACS/RIS Configuration Page.Select or enter the server(s) network information to connect and test connection. You may connect to multiple PACS and RIS servers.Save the entered information, closing the window. |
| 109.0 | IFU.109 | IFU shall recommend using a drape over the Cassette during use. | RSK | R11.55R11.56R11.57R11.59 | 5 - Using the System | Sterile Drapes | Drapes are recommended by MedAI to mitigate equipment damage from liquid ingress and patient cross-contamination. |
| 62.0 | IFU.62 | IFU shall Caution about high touch temperatures that may result on Cassette and Emitter surfaces. | RSK | R11.61 | 2 - General Safety | Environment Safety | CAUTION: The MX1 System device surfaces may reach temperatures up to 43°C (109.4°F) under extended use at the max operating temperature. This temperature limit is appropriate for the healthy skin of adults but may cause discomfort or minor injury when large areas of the skin (10 % of total body surface or more) are in contact with the hot surface, or if unhealthy skin is in contact with the hot surface. |
| 105.0 | IFU.105 | IFU shall include general troubleshooting for use | RSK | R11.64R7.26 | 8 - System Info and Alerts | Device Troubleshooting | table |
| 79.0 | IFU.79 | IFU shall note to uncover Cassette LEDs to allow X-ray emission and that covering them will prevent tracking system from allowing emission. | RSK | R11.65 | 6 - Capturing Photos and Radios | Tracking System | Troubleshooting section - Remove excess draping or bagging as well as other items from the Cassette’s top face to allow the Emitter camera to see the IR LEDs. |
| 116.0 | IFU.116 | IFU shall warn the user to wear PPE, including Radiation-protective PPE. | RSK | R12.1RMF3.26 | 10 - Radiation Exposure | Radiation Safety | WARNING: Operators should always wear PPE while using the MX1 System. Both an apron and a thyroid collar are recommended. Follow any additional state and/or hospital-specific safety procedures and PPE requirements. Failure to wear PPE may result in increased exposure to backscatter radiation and overexposure hazards. |
| 86.0 | IFU.86 | IFU shall describe risks, information, and available controls for mitigations in context of Pediatric Use cases | RSK | R14.1R14.3 | 10 - Radiation Exposure | Pediatric and Small Patients | Children are more sensitive to radiation damage than adults and have a longer post-exam life expectancy, increasing the risk of stochastic effects from radiation such as cancer and genetic defects. Special care should be taken when imaging patients outside of the typical adult size range, such as pediatric patients whose size does not overlap the adult size range. |
| 92.0 | IFU.92 | IFU shall include recommended imaging parameters, including designation for Pediatric use cases. | RSK | R15.1R15.7R6.30R6.54R7.3R7.7R7.8R7.16R7.19R7.20R11.15R14.1RMF1.8.17 | 6 - Capturing Photos and Radios | Radiograph Mode | Recommended Loading Factors Table |
| 43.0 | IFU.43 | IFU shall warn against charging or powering devices outside the specified operation range. | RSK | R16.2R16.3R16.4R16.5R16.21R16.22R16.23R16.24 | 4 - Setting Up the System | Charging | WARNING: Only operate the MX1 System in proper operating environments, including when charging the MX1 System. Not doing so may result in battery damage, electrical hazards, or other safety hazards.  See Section 12 - Technical Specifications. |
| 37.0 | IFU.37 | IFU shall caution the user to confirm the battery life of the Emitter and Cassette before use | RSK | R16.39R16.40 | 5 - Using the System | Powering On | CAUTION: Failure to sufficiently charge batteries prior to use may result in procedure delay. Check battery charge status indicators prior to use to confirm batteries are charged, and charge the system periodically to prevent unexpected loss of internal power during use. The Emitter will NOT allow x-ray emissions while charging. The Cassette, however, will accept x-rays emissions while charging. |
| 57.0 | IFU.57 | IFU shall specify the estimated battery capacity, in terms of images in a full charge or similar. | RSK | R16.39R16.40 | 5 - Using the System | Powering On | Charging the wireless Emitter for between XXX to XXX hours charges the Emitter enough to allow up to XXX hours of continuous use.  A normal, complete charge cycle takes more than XXX hours. Charging the wireless Cassette for between XXX to XXX hours charges the Cassette enough to allow up to XXX hours of continuous use.  A normal, complete charge cycle takes more than XXX hours. The wireless Footpedal uses replaceable XXX batteries. |
| 47.0 | IFU.47 | IFU shall Caution users against placing or storing the system under direct sunlight, hot surfaces, or locations that may get hot. | RSK | R16.7R16.26 | 9 - System Upkeep | Storing the System After Use | store in a cool, dry location, away from direct sunlight, following the environmental conditions in 12 - Technical Specifications |
| 455.0 | IFU.455 | IFU shall clearly indicate each pedal's intended function | RSK | R17.4 | 5 - Using the System | Foot Pedal | picture |
| 454.0 | IFU.454 | IFU shall instruct the user how to change imaging modes with the foot pedal. | RSK | R17.5 | 5 - Using the System | Foot Pedal | Use the foot pedal to capture images, change modes, rotate images, and mark images for sending using the buttons below. You may hold the capture button in Radiography mode to capture a DDR. |
| 33.0 | IFU.33 | IFU shall clearly indicate each foot pedal's and button's intended function | RSK | R17.8 | 5 - Using the System | Foot Pedal | Use the foot pedal to capture images, change modes, rotate images, and mark images for sending using the buttons below. You may hold the capture button in Radiography mode to capture a DDR. |
| 452.0 | IFU.452 | IFU shall instruct the user how to capture images with the foot pedal. | RSK | R17.8 | 5 - Using the System | Foot Pedal | Use the foot pedal to capture images, change modes, rotate images, and mark images for sending using the buttons below. You may hold the capture button in Radiography mode to capture a DDR. |
| 30.0 | IFU.30 | IFU shall instruct the user to periodically clean the device and include methods with materials. | RSK | R2.5R2.6R2.7R2.8R3.4R3.9R2.19R2.20R6.39R8.30R8.31R8.48R8.49R8.50R8.51R8.52R8.57R1.148R11.9R11.58RMF1.1.8RMF1.1.9RMF1.3.33RMF1.7.30RMF1.7.31RMF1.7.32RMF1.7.33RMF1.7.16RMF1.7.17RMF1.7.18RMF1.7.19RMF1.7.20RMF1.7.21RMF1.7.22RMF1.7.23RMF1.7.24RMF1.7.26RMF1.7.27RMF1.7.28 | 9 - System Upkeep | Routine CleaningDisinfection | Cleaning Instructions (multiple) |
| 52.0 | IFU.52 | IFU shall provide instructions on how to drape the Cassette. | RSK | R3.1R3.2R3.3R3.6R3.7R3.8R6.64 | 5 - Using the System | Sterile Drapes | Lay the drape over the entire cassette top face before use. Use a single-layer drape that lies flat on the MX1 Cassette such that the operator can see the visible-light LEDs through the drape. If the drape obscures or distorts the infrared (non-visible) LEDs, the system may not work as intended or may prevent X-ray emissions; if this occurs reposition the drape. |
| 76.0 | IFU.76 | IFU shall include troubleshooting for imaging interlock problems | RSK | R3.14R7.42R7.51 | 8 - System Info and Alerts | Device Troubleshooting | Tracking System Interlocks not met (Red) |
| 42.0 | IFU.42 | IFU shall Caution to not block or obstruct airflow around fans, or risk overheating. | RSK | R3.18R11.60 | 2 - General Safety | Cooling | CAUTION:  Avoid excessive covering of the system, including excessive drape layers, that may restrict airflow to the MX1 system. Blocking the vents of any component may result in the MX1 reaching its rated heat capacity and a delay in procedure. |
| 117.0 | IFU.117 | IFU shall Caution against spilled liquids or excessive fluids. | RSK | R4.42RMF1.3.47 | 2 - General Safety | Water Ingress | WARNING: The MX1 System is not waterproof and is only designed to defend against accidental spillage. If you suspect liquids entered the system, do not operate the system, disconnect any chargers from the wall outlet, and contact MedAI for assistance. |
| 98.0 | IFU.98 | IFU shall specify the Operating, Transport, and Storage Environments. | RSK | R4.7R4.27R4.36R4.98R4.44R4.45R4.46R4.63R4.98R8.21R8.32R8.36R8.37RMF1.3.58RMF1.3.59RMF1.3.60RMF1.11.41 | 12 - Tech Specs | General Specifications | Conditions for Use, Travel, and Storage |
| 124.0 | IFU.124 | IFU shall specify the input power requirements for the Emitter, Cassette, and Wired Charger. | RSK | R4.85R4.86R4.87R4.88R4.95RMF1.3.26RMF1.12.2RMF1.12.3RMF1.12.4 | 12 - Tech Specs | Electrical and Electromagnetic Specifications | The MX1 Emitter and Cassette must be charged with the Wired Charger BrickNominal Output Power: 100W (typical efficiency 86%)Input Rated Voltage / Frequency: 90 - 264 VAC / 50-60 Hz |
| 125.0 | IFU.125 | IFU shall caution to use the system in a sufficiently lit environment. | RSK | R4.96 | 2 - General Safety | Environment Safety | CAUTION: Using the device in bright sunlight may make it difficult to see the emitter screen and image viewing screen. Ensure settings and controls are visible prior to emitting x-rays. |
| 24.0 | IFU.24 | IFU shall instruct users to align the Emitter to Cassette active area during use | RSK | R5.11 | 5 - Using the System | Positioning the System and Patient | After positioning the Cassette and anatomy, hold the Emitter at a comfortable distance from the Cassette and align the emitter to the Cassette Active Area. |
| 482.0 | IFU.482 | IFU shall have prominent images and indicators for Emitter and Cassette charging ports. | RSK | R6.10R6.18 | 4 - Setting Up the System | Charging | Images indicating Cassette and Emitter charging ports to be placed in the IFU |
| 73.0 | IFU.73 | IFU shall describe proper USB-C connection instructions. | RSK | R6.13R6.17 | 4 - Setting Up the System | Charging | Connect the end of the USB-C cord to either port of the Cassette or the USB-C port of the Emitter by pushing the connectors firmly into the plugs and confirm they are properly placed. |
| 483.0 | IFU.483 | IFU shall have prominent images and indicators for proper Cassette orientation. | RSK | R6.16 | 4 - Setting Up the System | N/A | Images indicating proper Cassette orientation to be placed in the IFU |
| 162.0 | IFU.162 | IFU shall describe potential deterministic and stochastic effects of radiation exposure along with measures to mitigate them. | RSK | R6.2RMF1.3.5RMF1.3.6RMF1.3.7RMF1.3.8RMF1.3.9RMF1.3.10RMF1.3.11 | 10 - Radiation Exposure | Radiation Safety | WARNING: In prolonged or abnormal use, this equipment can produce skin dose levels high enough to cause deterministic effects such as skin erythema, skin damage, or hair loss. It is vital that you strictly follow all safety precautions. |
| 38.0 | IFU.38 | IFU shall describe all visual and audible indications and alerts from the system; and device power and mode states with associated inidcations | RSK | R6.32R6.35R7.30R7.31R7.32R7.37R7.43R15.2RMF1.8.12RMF1.8.13 | 8 - System Info and Alerts | Visual and Audible Indicators | table |
| 143.0 | IFU.143 | IFU shall specify means and limitations of data saves on the device. | RSK | R6.33 | 7 - Device App | Library Page | The MX1 System is not intended for long-term image storage or archival storage. |
| 145.0 | IFU.145 | IFU shall instruct the user on how to position or place the Cassette, Emitter, | RSK | R6.36R6.37R6.38 | 5 - Using the System | Positioning the System and Patient | The Cassette may be placed for imaging on a flat, dry, supporting surface with all of its non-slip feet contacting the surface or it may be positioned for use in other configurations by aid from MedAI-supplied accessories. Gently place or position the patient's anatomy of interest against the Cassette’s Active Area to image. Anything positioned outside the Active Area indicator will not be captured in a Radiograph. Make sure to not block all the non-visible, Infrared LEDs on the Cassette or the Emitter will not be able to track its position nor emit x-rays. |
| 150.0 | IFU.150 | IFU shall specify contact with Non-intact Skin is contraindicated | RSK | R6.39R11.55R11.56 | 1 - Introduction | Indications for Use | Contraindication: The MX1 System is NOT intended for: Non-intact Skin |
| 130.0 | IFU.130 | Labeling - IFU (Clause to Follow Local Rules/Regulations) | RSK | R6.43 | 10 - Radiation Exposure | Radiation Safety | The owner must ensure that all personnel follow radiation safety protocol(s) as dictated by the site in which the MX1 System is used, including personal protective equipment (PPE) and radiation monitoring devices. |
| 129.0 | IFU.129 | IFU shall instruct the user to place the Cassette on a stable surface. | RSK | R6.46 | 5 - Using the System | Positioning the System and Patient | The Cassette may be placed for imaging on a flat, dry, supporting surface with all of its non-slip feet contacting the surface or it may be positioned for use in other configurations by aid from MedAI-supplied accessories. |
| 63.0 | IFU.63 | IFU - hold still in use | RSK | R6.49R1.155 | 6 - Capturing Photos and Radios | Radiograph Mode | CAUTION: Movement of the patient, cassette, or emitter during imaging may increase risk of non-diagnostic exposure and/or patient injury. Avoid abrupt shaking or movement of the patient, cassette, and emitter. |
| 134.0 | IFU.134 | IFU shall instruct the user on how to use the collimating pucks | RSK | R6.50 | 5 - Using the System | Collimating Pucks | Choose a puck based on the desired collimation size that isn’t covered by the Emitter’s automatic collimation. Each puck is sized to result in a field size diameter between 5 and 22 cm, in 1cm steps, at an SID of 1 meter. |
| 141.0 | IFU.141 | IFU shall instruct the user on how to arm and emit radiation from the Emitter. | RSK | R6.55 | 5 - Using the System | Positioning the System and Patient | After positioning the Cassette and anatomy, hold the Emitter at a comfortable distance from the Cassette and align the emitter to the Cassette Active Area. In Radiography Mode, aiming lasers from the Emitter should turn on when the front face is generally pointed towards the Cassette. Use the Viewfinder screen to aim, adjust the patient, and capture the image. |
| 28.0 | IFU.28 | IFU shall instruct the user to check for damage routinely, including directions for how to do so. | RSK | R6.7R6.8R6.9R6.59R12.4 | 9 - System Upkeep | Periodic Maintenance Schedule | Always inspect the radiographic captures for image quality issues (spots, blurriness, resolution) during each use. At least once monthly, inspect the external surfaces of all components for damage, loose or missing parts, and frayed or damaged cords. Do not use the device if it displays one or more of the above conditions until the problem is corrected and has been verified as operating correctly and safely.WARNING: If it appears the MX1 System has been damaged, modified, or tampered with in any way, do not use the device and contact MedAI. Use of devices that have been modified or tampered with may result in serious injury. |
| 131.0 | IFU.131 | IFU shall instruct the user to send the system to MedAI for any servicing, calibration, and repairs. | RSK | R6.75R6.76RMF1.3.22 | 9 - System Upkeep | Periodic Maintenance Schedule | CAUTION: Do not attempt to open the MX1 system, perform maintenance, or perform component replacement. Opening the MX1 system may result in electrical shock. Always send the MX1 System to MedAI for service, inspection, and corrective or preventive maintenance |
| 69.0 | IFU.69 | IFU shall instruct the user on how to use the components. | RSK | R7.15R7.47R6.5 | 5 - Using the System | 5 - Using the System | 5 - Using the System |
| 77.0 | IFU.77 | IFU shall instruct how to change capture modes | RSK | R7.21 | 2 - General Safety | 2 - General Safety | Potential hazards exist in the use of medical electronic devices and X-ray systems. Operators using the MX1 System should understand the safety issues, emergency procedures, and the operating instructions provided. |
| 70.0 | IFU.70 | IFU shall include instructions for adjusting or manipulating an image | RSK | R7.27R11.32R11.33R11.34R11.35R11.36R11.37 | 7 - Device App | Acquisition Page | To enhance any other image found in the “B”, “C”, or “D” columns, select the chosen image by single clicking. |
| 106.0 | IFU.106 | IFU shall include instructions for using the Device App UI. | RSK | R7.37R7.38R7.44R7.46R15.4 | 7 - Device App | 7 - Device App | 7 - Device App |
| 126.0 | IFU.126 | IFU shall include instructions for tagging the orientation of an image. | RSK | R7.53R7.54R7.58 | 7 - Device App | Acquisition Page | You may also add annotations such as Left/Right orientation indicators by dragging the element from the Post-Processing Tools onto the image, dropping it in place. |
| 80.0 | IFU.80 | IFU shall Note that previous patient images will appear on screen when sending images to PACS | RSK | R7.57R9.32 | 7 - Device App | History Page | WARNING: Previous patient images may appear on the History Page before sending images to PACS. Verify the image(s) being sent is from the correct patient prior to sending. |
| 93.0 | IFU.93 | IFU shall state the effects of heavy use with higher loading factors with regard to service life of the X-ray tube and the system, including ways to mitigate those effects. | RSK | R8.14R8.15R8.24R8.25 | 9 - System Upkeep | Periodic Maintenance Schedule | Heavy use with higher loading factors (kV and mAs) may cause the X-ray tube and other system components to deteriorate and require servicing quicker than expected. Reducing use, charging adequately and often, and storing properly can extend the device's service life. |
| 46.0 | IFU.46 | IFU shall caution the user against performing maintenance or servicing beyond routine cleaning. | RSK | R8.55R8.56 | 9 - System Upkeep | Overview | CAUTION: Do not attempt to open the MX1 system, perform maintenance, or perform component replacement. Opening the MX1 system may result in electrical shock. Always send the MX1 System to MedAI for service, inspection, and corrective or preventive maintenance |
| 96.0 | IFU.96 | IFU shall instruct the operator to return the system to MedAI for End of Life Procedures. | RSK | R8.58 | 9 - System Upkeep | End of Life Procedure | At device or accessory end of life, ship products to MedAI to minimize environmenal risks associated with disposal. |
| 440.0 | IFU.440 | IFU shall instruct the user on how to set up the PACS and RIS server connections. | RSK | R9.1 | 4 - Setting Up the System | Connecting to DICOM (PACS) and RIS Servers | Connecting to DICOM (PACS) and RIS Servers |
| 151.0 | IFU.151 | IFU shall Note that PACS configuration is the reponsibility of the operator's organization. | RSK | R9.1R9.6 | 12 - Tech Specs | Network Integration & Cybersecurity Specifications | The system is also intended to interface with hospital-specific software such as PACS and hospital networks; configuration of PACS systems and other networks is the user’s responsibility. |
| 100.0 | IFU.100 | IFU - Specify Tablet Requirements | RSK | R9.21 | 12 - Tech Specs | Electrical and Electromagnetic Specifications |  |
| 41.0 | IFU.41 | IFU - operators should monitor device use prior to powering the device down or controlling the device from a tablet or mobile device | RSK | R9.31 | 7 - Device App | History Page | CAUTION: Prior to powering off the MX1 System device remotely via a mobile device, verify the device is not already in use or being controlled by another mobile device. The MX1 System allows for pairing to multiple tablets/mobile devices at the same time. |
| 74.0 | IFU.74 | IFU shall describe the System's Intended Use | RSK | RMF3.7 | 1 - Introduction | Intended Use | The MX1 Portable X-ray System is indicated for use by qualified/trained clinicians on adult and pediatric patients for taking diagnostic static and serial radiographic exposures of extremities, hips, pelvis, and cervical spine. The device is not to be used on bariatric patients, unless imaging body extremities.The device is not intended to replace a stationary radiographic system, which may be required for full optimization of image quality and radiation exposure for different exam types. |
| 484.0 | IFU.484 |  | RSK |  |  |  |  |
| 223.0 | IFU.223 | IFU shall Warn that only qualified medical personnel who have been trained in the use of medical imaging equipment and who have read this Instructions for Use and Accompanying Documents may operate this equipment. | RSK/RMF | R6.68R7.55R7.56R7.57RMF1.3.45 | 1 - Introduction | 1 - Introduction | WARNING: The MX1 System is intended for use by qualified medical personnel who have been trained in the use of medical imaging equipment and who have read the MX1 System Instructions for Use and Accompanying Documents. |
| 476.0 | IFU.476 | IFU - Wifi Troubleshooting | WCR | WCR.1.5WCR2.3WCR2.4WCR2.5 |  |  |  |
| 477.0 | IFU.477 | IFU - NFC Troubleshooting | WCR | WCR2.1WCR2.2 |  |  |  |
| 478.0 | IFU.478 | IFU - check for visual indication of pairing | WCR | WCR2.7WCR2.8WCR2.9WCR2.10 |  |  |  |
| 479.0 | IFU.479 | IFU - Specify max distance for foot pedal to emitter | WCR | WCR3.2WCR3.6WCR3.7WCR6.2 |  |  |  |
| 480.0 | IFU.480 | IFU - troubleshooting | WCR | WCR4.1WCR4.2WCR4.3WCR4.4WCR4.5WCR4.6WCR5.1WCR5.2WCR6.1WCR7.1WCR7.2WCR7.3WCR7.4WCR7.5WCR7.6 |  |  |  |
| 119.0 | IFU.119 | DELETED |  |  |  |  |  |
| 22.0 | IFU.22 | DELETED |  |  |  |  |  |
| 29.0 | IFU.29 | DELETED |  |  |  |  |  |
| 44.0 | IFU.44 | DELETED |  |  |  |  |  |
| 53.0 | IFU.53 | DELETED |  |  |  |  |  |
| 85.0 | IFU.85 | DELETED |  |  |  |  |  |
| 89.0 | IFU.89 | DELETED |  |  |  |  |  |
| 137.0 | IFU.137 | DELETED |  |  |  |  |  |
| 184.0 | IFU.184 | DELETED |  |  |  |  |  |
| 203.0 | IFU.203 | DELETED |  |  |  |  |  |
| 210.0 | IFU.210 | DELETED |  |  |  |  |  |
| 224.0 | IFU.224 | DELETED |  |  |  |  |  |
| 230.0 | IFU.230 | DELETED |  |  |  |  |  |
| 238.0 | IFU.238 | DELETED |  |  |  |  |  |
| 251.0 | IFU.251 | DELETED |  |  |  |  |  |
| 1.0 | IFU.1 | DELETED |  |  |  |  |  |
| 2.0 | IFU.2 | DELETED |  |  |  |  |  |
| 3.0 | IFU.3 | DELETED |  |  |  |  |  |
| 4.0 | IFU.4 | DELETED |  |  |  |  |  |
| 5.0 | IFU.5 | DELETED |  |  |  |  |  |
| 6.0 | IFU.6 | DELETED |  |  |  |  |  |
| 7.0 | IFU.7 | DELETED |  |  |  |  |  |
| 8.0 | IFU.8 | DELETED |  |  |  |  |  |
| 9.0 | IFU.9 | DELETED |  |  |  |  |  |
| 10.0 | IFU.10 | DELETED |  |  |  |  |  |
| 11.0 | IFU.11 | DELETED |  |  |  |  |  |
| 12.0 | IFU.12 | DELETED |  |  |  |  |  |
| 13.0 | IFU.13 | DELETED |  |  |  |  |  |
| 14.0 | IFU.14 | DELETED |  |  |  |  |  |
| 15.0 | IFU.15 | DELETED |  |  |  |  |  |
| 16.0 | IFU.16 | DELETED |  |  |  |  |  |
| 17.0 | IFU.17 | DELETED |  |  |  |  |  |
| 18.0 | IFU.18 | DELETED |  |  |  |  |  |
| 19.0 | IFU.19 | DELETED |  |  |  |  |  |
| 20.0 | IFU.20 | DELETED |  |  |  |  |  |
| 21.0 | IFU.21 | DELETED |  |  |  |  |  |
| 26.0 | IFU.26 | DELETED |  |  |  |  |  |
| 27.0 | IFU.27 | DELETED |  |  |  |  |  |
| 31.0 | IFU.31 | DELETED |  |  |  |  |  |
| 32.0 | IFU.32 | DELETED |  |  |  |  |  |
| 35.0 | IFU.35 | DELETED |  |  |  |  |  |
| 36.0 | IFU.36 | DELETED |  |  |  |  |  |
| 39.0 | IFU.39 | DELETED |  |  |  |  |  |
| 40.0 | IFU.40 |  |  |  |  |  |  |
| 45.0 | IFU.45 | DELETED |  |  |  |  |  |
| 49.0 | IFU.49 | DELETED |  |  |  |  |  |
| 55.0 | IFU.55 |  |  |  |  |  |  |
| 56.0 | IFU.56 | DELETED |  |  |  |  |  |
| 58.0 | IFU.58 | DELETED |  |  |  |  |  |
| 59.0 | IFU.59 | DELETED |  |  |  |  |  |
| 61.0 | IFU.61 |  |  |  |  |  |  |
| 64.0 | IFU.64 | DELETED |  |  |  |  |  |
| 66.0 | IFU.66 | DELETED |  |  |  |  |  |
| 67.0 | IFU.67 | DELETED |  |  |  |  |  |
| 68.0 | IFU.68 | DELETED |  |  |  |  |  |
| 75.0 | IFU.75 | DELETED |  |  |  |  |  |
| 81.0 | IFU.81 | DELETED |  |  |  |  |  |
| 82.0 | IFU.82 | DELETED |  |  |  |  |  |
| 87.0 | IFU.87 | DELETED |  |  |  |  |  |
| 88.0 | IFU.88 | DELETED |  |  |  |  |  |
| 90.0 | IFU.90 | DELETED |  |  |  |  |  |
| 91.0 | IFU.91 | DELETED |  |  |  |  |  |
| 94.0 | IFU.94 | DELETED |  |  |  |  |  |
| 95.0 | IFU.95 | DELETED |  |  |  |  |  |
| 97.0 | IFU.97 | DELETED |  |  |  |  |  |
| 99.0 | IFU.99 | DELETED |  |  |  |  |  |
| 101.0 | IFU.101 | DELETED |  |  |  |  |  |
| 102.0 | IFU.102 | DELETED |  |  |  |  |  |
| 103.0 | IFU.103 | DELETED |  |  |  |  |  |
| 104.0 | IFU.104 | DELETED |  |  |  |  |  |
| 107.0 | IFU.107 | DELETED |  |  |  |  |  |
| 110.0 | IFU.110 | DELETED |  |  |  |  |  |
| 111.0 | IFU.111 | DELETED |  |  |  |  |  |
| 112.0 | IFU.112 |  |  |  |  |  |  |
| 113.0 | IFU.113 | DELETED |  |  |  |  |  |
| 118.0 | IFU.118 | DELETED |  |  |  |  |  |
| 120.0 | IFU.120 | DELETED |  |  |  |  |  |
| 121.0 | IFU.121 | DELETED |  |  |  |  |  |
| 122.0 | IFU.122 | DELETED |  |  |  |  |  |
| 123.0 | IFU.123 | DELETED |  |  |  |  |  |
| 127.0 | IFU.127 | DELETED |  |  |  |  |  |
| 128.0 | IFU.128 | DELETED |  |  |  |  |  |
| 132.0 | IFU.132 | DELETED |  |  |  |  |  |
| 133.0 | IFU.133 | DELETED |  |  |  |  |  |
| 135.0 | IFU.135 | DELETED |  |  |  |  |  |
| 136.0 | IFU.136 | DELETED |  |  |  |  |  |
| 138.0 | IFU.138 | DELETED |  |  |  |  |  |
| 139.0 | IFU.139 | DELETED |  |  |  |  |  |
| 140.0 | IFU.140 | DELETED |  |  |  |  |  |
| 142.0 | IFU.142 | DELETED |  |  |  |  |  |
| 144.0 | IFU.144 | DELETED |  |  |  |  |  |
| 146.0 | IFU.146 | DELETED |  |  |  |  |  |
| 147.0 | IFU.147 | DELETED |  |  |  |  |  |
| 148.0 | IFU.148 | DELETED |  |  |  |  |  |
| 149.0 | IFU.149 | DELETED |  |  |  |  |  |
| 152.0 | IFU.152 | DELETED |  |  |  |  |  |
| 153.0 | IFU.153 | DELETED |  |  |  |  |  |
| 154.0 | IFU.154 | DELETED |  |  |  |  |  |
| 155.0 | IFU.155 | DELETED |  |  |  |  |  |
| 156.0 | IFU.156 | DELETED |  |  |  |  |  |
| 157.0 | IFU.157 | DELETED |  |  |  |  |  |
| 158.0 | IFU.158 | DELETED |  |  |  |  |  |
| 159.0 | IFU.159 | DELETED |  |  |  |  |  |
| 160.0 | IFU.160 | DELETED |  |  |  |  |  |
| 161.0 | IFU.161 | DELETED |  |  |  |  |  |
| 163.0 | IFU.163 |  |  |  |  |  |  |
| 164.0 | IFU.164 | DELETED |  |  |  |  |  |
| 165.0 | IFU.165 | DELETED |  |  |  |  |  |
| 166.0 | IFU.166 |  |  |  |  |  |  |
| 167.0 | IFU.167 | DELETED |  |  |  |  |  |
| 168.0 | IFU.168 | DELETED |  |  |  |  |  |
| 175.0 | IFU.175 | DELETED |  |  |  |  |  |
| 177.0 | IFU.177 |  |  |  |  |  |  |
| 181.0 | IFU.181 |  |  |  |  |  |  |
| 182.0 | IFU.182 | DELETED |  |  |  |  |  |
| 185.0 | IFU.185 | DELETED |  |  |  |  |  |
| 186.0 | IFU.186 | DELETED |  |  |  |  |  |
| 187.0 | IFU.187 | DELETED |  |  |  |  |  |
| 188.0 | IFU.188 | DELETED |  |  |  |  |  |
| 189.0 | IFU.189 | DELETED |  |  |  |  |  |
| 191.0 | IFU.191 | DELETED |  |  |  |  |  |
| 192.0 | IFU.192 | DELETED |  |  |  |  |  |
| 193.0 | IFU.193 |  |  |  |  |  |  |
| 195.0 | IFU.195 | DELETED |  |  |  |  |  |
| 197.0 | IFU.197 | DELETED |  |  |  |  |  |
| 201.0 | IFU.201 | DELETED |  |  |  |  |  |
| 202.0 | IFU.202 | DELETED |  |  |  |  |  |
| 205.0 | IFU.205 | DELETED |  |  |  |  |  |
| 206.0 | IFU.206 | DELETED |  |  |  |  |  |
| 207.0 | IFU.207 | DELETED |  |  |  |  |  |
| 208.0 | IFU.208 | DELETED |  |  |  |  |  |
| 212.0 | IFU.212 | DELETED |  |  |  |  |  |
| 213.0 | IFU.213 | DELETED |  |  |  |  |  |
| 216.0 | IFU.216 | DELETED |  |  |  |  |  |
| 217.0 | IFU.217 |  |  |  |  |  |  |
| 218.0 | IFU.218 | DELETED |  |  |  |  |  |
| 219.0 | IFU.219 | DELETED |  |  |  |  |  |
| 220.0 | IFU.220 | DELETED |  |  |  |  |  |
| 221.0 | IFU.221 | DELETED |  |  |  |  |  |
| 222.0 | IFU.222 | DELETED |  |  |  |  |  |
| 225.0 | IFU.225 | DELETED |  |  |  |  |  |
| 226.0 | IFU.226 | DELETED |  |  |  |  |  |
| 228.0 | IFU.228 | DELETED |  |  |  |  |  |
| 229.0 | IFU.229 | DELETED |  |  |  |  |  |
| 235.0 | IFU.235 | DELETED |  |  |  |  |  |
| 236.0 | IFU.236 | DELETED |  |  |  |  |  |
| 237.0 | IFU.237 | DELETED |  |  |  |  |  |
| 239.0 | IFU.239 | DELETED |  |  |  |  |  |
| 241.0 | IFU.241 | DELETED |  |  |  |  |  |
| 242.0 | IFU.242 | DELETED |  |  |  |  |  |
| 243.0 | IFU.243 | DELETED |  |  |  |  |  |
| 245.0 | IFU.245 | DELETED |  |  |  |  |  |
| 246.0 | IFU.246 | DELETED |  |  |  |  |  |
| 247.0 | IFU.247 | DELETED |  |  |  |  |  |
| 248.0 | IFU.248 | DELETED |  |  |  |  |  |
| 250.0 | IFU.250 |  |  |  |  |  |  |
| 252.0 | IFU.252 | DELETED |  |  |  |  |  |
| 274.0 | IFU.274 | DELETED |  |  |  |  |  |
| 276.0 | IFU.276 | DELETED |  |  |  |  |  |
| 393.0 | IFU.393 | DELETED |  |  |  |  |  |
| 423.0 | IFU.423 | DELETED |  |  |  |  |  |
| 424.0 | IFU.424 | DELETED |  |  |  |  |  |
| 425.0 | IFU.425 | DELETED |  |  |  |  |  |
| 426.0 | IFU.426 | DELETED |  |  |  |  |  |
| 427.0 | IFU.427 | DELETED |  |  |  |  |  |
| 428.0 | IFU.428 | DELETED |  |  |  |  |  |
| 429.0 | IFU.429 | DELETED |  |  |  |  |  |
| 430.0 | IFU.430 | DELETED |  |  |  |  |  |
| 431.0 | IFU.431 | DELETED |  |  |  |  |  |
| 432.0 | IFU.432 | DELETED |  |  |  |  |  |
| 433.0 | IFU.433 | DELETED |  |  |  |  |  |
| 434.0 | IFU.434 | DELETED |  |  |  |  |  |
| 435.0 | IFU.435 | DELETED |  |  |  |  |  |
| 436.0 | IFU.436 | DELETED |  |  |  |  |  |
| 437.0 | IFU.437 | DELETED |  |  |  |  |  |
| 439.0 | IFU.439 | DELETED |  |  |  |  |  |
| 441.0 | IFU.441 | DELETED |  |  |  |  |  |
| 451.0 | IFU.451 | DELETED |  |  |  |  |  |
| 453.0 | IFU.453 | DELETED |  |  |  |  |  |
| 489.0 | IFU.489 | The IFU shall describe means to adjust or inactivate the signals for a termination of Radioscopy or Radiography Loadings, NOT including the High Level Control audible signal desicribed in IEC 60601-2-54 203.6.3.102 | IEC 60601-2-43 | 203.6.4.2 |  |  |  |
| 493.0 | IFU.493 | The H1 Brick shall be labeled with its rated input voltage, current, number of phases, and frequency. | IEC 60601-2-54 | 201.7.2.7 |  |  |  |
| 494.0 | IFU.494 | Collimating pucks shall be labeled with individual distinguisable identification | IEC 60601-2-54 | 201.7.2.101 |  |  |  |
| 495.0 | IFU.495 |  |  |  |  |  |  |
| 496.0 | IFU.496 |  |  |  |  |  |  |
| 497.0 | IFU.497 |  |  |  |  |  |  |
| 498.0 | IFU.498 |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |
|  | IFU. |  |  |  |  |  |  |

### Table 8
| ID | Mitigation | Requirement | Specification | NOTES / QUESTIONS / REDLINE COMMENTS |
| --- | --- | --- | --- | --- |
| RSK_R001 | Audible alert during X-ray emission | The device SW shall alert the operator when irradiating. | Reference SRS | 60601-1-3 Section 6.4.2 |
| RSK_R005 | Image Queuing | The device SW shall queue up studies when network is not present. | Reference SRS |  |
| RSK_R007 | Packet Validation | The device SW communication protocol shall perform a Packet Validation. | Reference SRS | Packet Val = Length, seq, checksum |
| RSK_R010 | Use of Deterministic Image Processing Algorithm | The image processing algorithm shall be deterministic. | Reference SRS |  |
| RSK_R012 | COMS Heartbeat check | The device SW shall ensure valid communication using a Heartbeat. | Reference SRS | Do we need a robustness specification? (ex. <1% handshake failure)  - Mo |
| RSK_R013 | Watchdog | The device SW shall utilize a hardware Watchdog. | Reference SRS |  |
| RSK_R014 | Message Checksum | The device SW shall contain a Message Checksum or Cyclic Redunacy Check (CRC) | Reference SRS | Dhruv says yes, we will have this 6SEP22-Mo |
| RSK_R028 | Firmware interlock that disables laser if laser is not pointed within Active Area | The emitter shall contain a software interlock that disables the lasers when they are not pointed at the cassette. | Reference SRS |  |
| RSK_R033 | HV gen requires active FW initiative | The device FW shall initiate HV generation. | Reference SRS |  |
| RSK_R041 |  | The device shall use of proprietary protocols to connect with MedAI accessories only. | Reference SRS | Example. Only able to pair to MedAI Footpedals, and other MedAI Accessories. Can't use random wired USB footpedal |
| RSK_R048 | Exposure Data Tracking | The device shall implement exposure data tracking. | Reference SRS |  |
| RSK_R053 | Debounce circuit on trigger signal | The device shall contain a debounce on the trigger. | While in single-shot mode, the MX1 system shall fire only one x-ray in a one second interval. |  |
| RSK_R056 | Start-up Check | The device shall contain a Start-up Check. | Reference SRS |  |
| RSK_R063 | Images automatically includes imaging parameters | The device UI shall display technique factors post-imaging. | Reference SRS |  |
| RSK_R065 | UI Button to Reset Image Adjustments | The device shall contain a UI Button to Reset Image Adjustments. | Reference SRS |  |
| RSK_R066 | Display Error Message | The Software UI shall display Error Messages. | Reference SRS | In RSK-P01, Rev A, this is used for a lot of things.  In SRS, should split up into multiple error messages with more specificity (i.e overheating, failure to send to PACS server, etc...).Clarified "Software UI" which includes viewfinder & display |
| RSK_R094 | Remove/reduce Concentrated E-Fields | The device shall remove/reduce Concentrated E-Fields. | The device shall be compliant to IEC 60601-1-2 |  |
| RSK_R097 |  | The device shall contain thermistors that cause the device to fail safe in the event of over or under temp. | Reference SRS |  |
| RSK_R098 | Bleed-off circuit (for capacitive energies) | The device shall contain a Bleed-off circuit (for capacitive energies). | Specification identical to requirementReference E1 emitter drawings |  |
| RSK_R099 | HV Transient Suppression on critical circuits | The device shall contain an HV Transient Suppression on critical circuits. | The device shall contain an HV transient suppresssion on the following circuits: HMIs, power input, debug ports | esd protection on crit circuits |
| RSK_R102 | Device disallows x-ray emission when charging cable is detected | The emitter shall contain a FW interlock to prevent x-ray emissions when physically plugged into an external power source. | Reference SRS | I think it requires SW as well, but there is an interlock. There is hardware that allows SW to detected if plugged in. -(Mo per David 31AUG22) |
| RSK_R109 | Tube designed with stationary anode | The device shall not contain a rotating anode. | Specification identical to requirementReference x-ray tube drawing |  |
| RSK_R111 | Cassette UI displays the charging statusEmitter UI displays the charging status | The device shall contain a battery charge status indicator on the emitter and cassette UI. | Reference SRS |  |
| RSK_R117 | BMS includes over-current protection | The device BMS shall contain Over-current protection. | The battery shall comply with IEC 62133-2 (complies per third party report) | Short circuit detection |
| RSK_R118 | BMS includes over-voltage protection | The device BMS shall contain Over-voltage protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R125 | Use of IEC 60601-1 compliant Class 2 power supply to protect against electrical hazards | The device shall contain IEC 60601-1 rated battery chargers. | The H1 Wired Charger shall comply with IEC 60601-1 |  |
| RSK_R139 | HW controlled unexpected tube current interlock fault | The device SW shall open the HV interlock in the event of an unexpected tube current fault. | Reference SRS |  |
| RSK_R142 | Beam Current Monitoring | The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low | Reference SRS |  |
| RSK_R143 | Device includes temperature monitoring features; device enters safe state if overheated | The device SW shall monitor the internal temperature(s) and enter a safe state in the event of under or over temp. | Reference SRS |  |
| RSK_R146 | Monitor For Loss of Communication And Fail Safe When Detected | The device shall verify the integrity of the safety-critical wireless systems within 1 second of every X ray exposure | Reference SRS | this is used in the context of ionizing radiation; mitigation to require regular (every 1 second, or within 1 sec of every exposure) heartbeat checks on all safety-critical wireless systems |
| RSK_R148 | Humidity Sensor triggers safe state in humid environments above safe operating range | The device shall enter safe state (or warn the operator) in the event of under or over humidity | Reference SRS |  |
| RSK_R153 | Implement Design Features to Mitigate Effects of Vibration | The device shall be primarily packaged in a hard shell case with closed cell internal foam or similar style case. | Specification identical to requirementReference P1 case drawing |  |
| RSK_R159 | Internally caged, shrouded, slow fan(s) | The device fans shall be internally caged and shrouded. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R169 | Use of heat and impact resistant materials | The device enclosures shall be composed of an impact and heat resistant plastic. | The enclosures shall be composed of PolycarbonateV2 or similar material with equivalent or better heat and impact resistanceThe device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R172 | Device internals are non accessible without use of tool | The device internals shall not be accessible without the use of a tool. | Specification identical to requirementReference E1 emitter and C1 cassette drawings | Corrected to match RSK assessment. Added new RSK_ |
| RSK_R177 | No accessible conductive components in charger | The device enclosures shall prevent operator access to conductive components / contacts. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R179 | Enclosure window pieces do not contain holes | The device shall utilize covers and guards when necessary to prevent access to electrical, thermal, and moving components. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R182 | Rubber feet | The device enclosures shall reduce falling and tipping risk by providing a surface with a high friction material. | The emitter and cassette shall contain Rubber feet |  |
| RSK_R189 | Enclosure window pieces do not contain holes | The device enclosures window pieces shall not contain holes. | Emitter and Cassette shall not contain holes |  |
| RSK_R190 | Wire routing | The device shall integrate wire routing to prevent mechanical damage to the wires. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R196 |  | The device shall contain locking or friction locking connectors on all internal harnesses and external cables | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R197 | Enclosed collimator | The device collimator shall be encased within the enclosure. | Specification identical to requirementReference E1 emitter drawing |  |
| RSK_R202 | Magnet Redundancies for Puck attachment | The pucks and emitter front face (puck attachment area) shall contain a magnet redundancies (more than one magnet per puck and emitter attachment area). | Specification identical to requirementReference pucks and E1 emitter drawings |  |
| RSK_R230 | Emitter designed to be floating system when High voltage is present | The HV insulation shall integrate redundant safety methods. | Floating loop - Battery shall not connect directly to MAINS. Reference E1 emitter drawing and C1 cassette drawingThe device shall disable operation while plugged in - Reference RSK_R102Dielectric strength & Enclosure as a layer of insulation - The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) | Removed arc monitoring. |
| RSK_R232 | The device shall display a warning in the event of poor wireless communication (COMS) quality. | The device shall display a warning in the event of poor wireless communication (COMS) quality. | Reference SRS | Example. Faulty data packets |
| RSK_R239 |  | The device shall contain DICOM ping to PACS upon data upload/export. | Reference SRS |  |
| RSK_R240 |  | The device sub GHz radios have built-in cyclic redundancy checks (CRCs) | Specification identical to requirementReference ES-10004 & ES-10003 BOM |  |
| RSK_R250 | Fuel Gage indicates time remaining in a single DDR/Fluoro capture | The device UI shall display a countdown of the duration time on DDR and fluoroscopy studies. | Reference SRS |  |
| RSK_R329 | X-ray Trigger on Emitter contains a sliding trigger button with a trigger button cover | The device trigger shall contain features that prevent jamming. | Triggers shall include a seal Triggers shall have high cycle tested buttons |  |
| RSK_R330 | Ping PACS server before sending packet | The device software shall ping PACS server before sending packet(s). | Reference SRS |  |
| RSK_R335 | Batteries shall be keyed connectors to prevent incorrect installation by MedAI service Operator or Patient | Batteries shall be keyed connectors to prevent incorrect installation by MedAI sevice personnel. | Specification Identical to RequirementReference battery drawings |  |
| RSK_R350 | Emitter handleEmitter Handle included for carrying and grip | The emitter and cassette shall have a handle. | Specification identical to requirementReference E1 Emitter and  C1 cassette drawings |  |
| RSK_R351 | Cassette design includes port labels for data export | The device shall contain markings to differentiate cable ports. | Specification Identical to RequirementReference label drawings |  |
| RSK_R352 | USB-C Data Port Rated for normal 20V voltage | Data port(s) shall be rated for normal 20V voltage. | Specification Identical to RequirementReference M50423 |  |
| RSK_R353 | 2 chargers provided | The device shall be provided with multiple chargers. | The emitter and cassette shall be provided with 2 chargers |  |
| RSK_R355 | Enclosure is not pressurized, design includes vents | The device shall not include a pressurized enclosure. | Specification identical to requirementReference E1 Emitter and  C1 cassette drawings |  |
| RSK_R356 | Emitter screen is LED-backlit, visible in indoor environments | Screens shall be backlit so they are visibile in indoor environments. | Specification identical to requirementReference UI drawings |  |
| RSK_R357 | Stability lockout prevents handheld use during fluoroscopy | The device shall incldue a stability lockout to prevent movement during capture. | Reference SRS |  |
| RSK_R358 | Foot Pedal indicator lights will remain off when not powered | The foot pedal shall have power and connectivity indicators. | Specification identical to requirementReference SRS (But not in SRS) |  |
| RSK_R359 | App shows tablet connectivity to the cassette | The device app will show connectivity status of the Cassette to the tablet. | Reference SRS |  |
| RSK_R361 | TVS Diodes, MOVs on USB LinesUSB-PD protocol controls voltage authorizationMOVs present on USB Lines | The device shall contain TVS Diodes and MOVs on all USB inputs. | Specification identical to requirementReference schematics for USB lines |  |
| RSK_R363 | Puck made of solid stainless steel | The puck material will be made out of coated steel. | Specification identical to requirementReference puck drawings |  |
| RSK_R364 | Detachable parts (Pucks/Pediatric Filter) are labeled | All pucks shall be engraved with PN and MedAI logo. | Specification identical to requirementReference Puck drawings |  |
| RSK_R366 | Banner presented on UI to let operator know that the display is non-DICOM | The device software shall display a banner when the connected display is non-MedAI. | Reference SRS |  |
| RSK_R367 | MedAI App not available to download in Apple App Store | The MedAI App shall only be available on the Google Play store. | Reference SRS |  |
| RSK_R368 | Device detects collimator movement failure | The device shall detect collimator movement failures and display an error. | Reference SRS |  |
| RSK_R369 | Backed up app partition (A/B file system) | The device shall contain a backed up app partition (A/B file system). | Reference SRS |  |
| RSK_R341 |  | The stability lockout shall be disabled once a fluoroscopic image capture is initiated. | Reference SRS |  |
| RSK_R342 |  | The device software shall include Integrity checks. | Reference SRS |  |
| RSK_R343 |  | Device app indicates progress of export to USB-C drive | Reference SRS |  |
| RSK_R344 |  | The device app shall display the interlock status of the device and inform the user of steps to take if the interlock is not met. | Reference SRS |  |
| RSK_R345 |  | The device software shall have footpedal signal forward error correction. | Reference SRS |  |
| RSK_R346 |  | The device software shall present a confirmation screen when deleting/clearing images (photographic, single radiographic, radioscopic, and DDR). | Reference SRS |  |
| RSK_R347 |  | Viewfinder UI elements shall stand out from the background and from each other. | Reference SRS |  |
| RSK_R372 |  | The device software shall contain the option to require patient information fields before starting an exam, and shall not allow the exam to continue if the required information is cleared in-process. | Reference SRS |  |
| RSK_R373 |  | The device app shall display a confirmation screen when initiating device shut-down via the device app. | Reference SRS |  |
| RSK_R375 |  | If the user selects to delay a system update, the software shall present a pop-up with the option to update every hour following initial refusal. | Reference SRS |  |
| RSK_R376 |  | Device update cannot start if Emitter or Cassette battery are below threshold | Reference SRS |  |
| RSK_R378 |  | All update pop-ups shall remain on screen until accepted or dismissed. | Reference SRS |  |
| RSK_R381 |  | The SSD indication on the viewfinder shall turn red when SSD is too close | Reference SRS |  |
| RSK_R383 |  | The device software shall present a confirmation screen before deleting users. | Reference SRS |  |
| RSK_R384 |  | The device app shall automatically detect DICOM display device type | Reference SRS |  |
| RSK_R385 |  | The device software shall disables charging if cassette overheats | Reference SRS |  |
| RSK_R386 |  | The device shall contain a hardware control to conduct a timeout in the event of an x-ray exposure time of greater than 210 ms | Reference SRS |  |
| RSK_R387 |  | The device app shall provide the user with a button to navigate directly to the tablet network settings screen | Reference SRS |  |
| RSK_R388 |  | The device software will disallow starting remote updates unless both the emitter and cassette are charging | Reference SRS |  |
| RSK_R389 |  | If a remote update is delayed, the device software shall provide a means to initiate remote update notification after the initial delay | Reference SRS |  |
| RSK_R390 |  | The device software shall include a software and firmware version interlock | Reference SRS |  |
| RSK_R391 |  | The device software shall provide a means to clear all images from local storage | Reference SRS |  |
| RSK_R392 |  | The device software shall disallow x-ray emission if a tablet and device app connection is not detected | Reference SRS |  |

### Table 9
|  | Mitigation | Requirement | Specification | NOTES / QUESTIONS / REDLINE COMMENTS |
| --- | --- | --- | --- | --- |
| RSK_R004 |  | The device SW communication protocol shall contain Sequence Number checks. | Reference SRS |  |
| RSK_R008 |  | The device SW shall contain an interlock if any part of the projected collimated x-ray field is outside the active area. | Reference SRS | PRD20.11 |
| RSK_R022 |  | The device FW shall contain Filament Current Monitoring. | To be added before Phase 4 | "In the risk assessment, this is linked to "low filament current."R026 below covers low beam current Changed to "filament current monitoring"R142 below seems to cover beam current monitoring already."-AkeafaRevisit in Phase 4 |
| RSK_R025 |  | The device FW shall limit kV, exposure time, and tube current to values allowable in normal use. | Reference SRS | "Is this to say, the firmware sets maximum limits for these loading factors (e.g. sw can't request settings beyond those limits)? What does "limited" mean here?" - Akeafa |
| RSK_R026 |  | The device FW shall contain a low Beam Current Monitoring. | Reference SRS |  |
| RSK_R029 |  | The device FW shall limit maximum exposure times for single and serial radiographic images. | Reference SRS |  |
| RSK_R032 |  | The device shall contain an HV control and monitor. | Reference SRS |  |
| RSK_R044 |  | The device shall contain a response verification (i.e message acknowledgement) between components. | Reference SRS |  |
| RSK_R055 |  | The radiation sequence shall terminate within 1 frame post trigger release. | Reference SRS | DDR always checking the trigger state. |
| RSK_R058 |  | The device shall contain a Filament preheat. | Reference SRS | "There is a preheat for the filament" - David/Hartman 1/9/23 |
| RSK_R064 |  | The device shall only allow discrete technique choices when in manual mode. | Reference SRS |  |
| RSK_R079 |  | The device shall contain an AC/DC brick with surge protection specifications. | -- | "Safe state = sw, and fw reports critical faults to swNot sure why this one's in the EE section"-AkeafaRequirement covered by other PRD & RSK items.Example.  PRD6.2, PRD6.3, PRD6.4-Mo |
| RSK_R081 |  | The device shall contain isolation from Mains - Overvoltage Cat 2 Power Supply. | -- |  |
| RSK_R082 |  | The device shall maintain high voltage ONLY during x-ray emission. | Specification identical to requirement | Confirm Cat 2 Power Supply |
| RSK_R083 |  | The device shall fail safe when unexpected arcing occurs. | The x-ray tube does not contain a grid. |  |
| RSK_R084 |  | The device shall fail safely in the event of undercurrent. (e.g. battery dies) | Maintains BASIC SAFETY per IEC 60601-1 |  |
| RSK_R085 |  | The device shall contain current limiting devices. | Maintains BASIC SAFETY per IEC 60601-1 | Reworded to clarify requirement. "Safe state" is used in software/firmware. This mitigation is to verify the device does not act unsafely when the battery dies or provides undercurrent. |
| RSK_R090 |  | The IR LEDs intensity shall be below the injury threshold | -- | Covered by PRD7.18 and RSK_R028 |
| RSK_R095 |  | The device shall restrict HV to within the monoblock. | The device shall be compliant to IEC 60601-1-2 |  |
| RSK_R096 |  | The device shall contain a redundant thermostat. | The device shall maintain BASIC SAFETY per IEC 60601-1 |  |
| RSK_R100 |  | The device charger (H1) shall comply to USB-PD specifications set by USB-IF. | Specification identical to requirementReference H1 emitter drawings |  |
| RSK_R101 |  | The cassette shall isolate power input to patient accessible ports/parts. | Specification identical to requirementReference ES-10014 - Cassette Isolation PCBA |  |
| RSK_R103 |  | The device shall mechanically protect the tube. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab)After exposure to ISTA 3A conditioning for Standard Packaged Product, the device shall:-Maintain essential performance (per MEMO-P01-441)-Have no visible damage that affects safety or performance of the device |  |
| RSK_R108 |  | The x-ray tube assy shall contain a temperature monitor on the anode. | Specification identical to requirementReference x-ray tube drawing |  |
| RSK_R112 |  | The device shall contain a Low battery warning on the Display UI. | Reference SRS |  |
| RSK_R114 |  | The device BMS shall contain under-voltage protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R115 |  | The device BMS shall contain Cell temp monitoring for under and over-temp protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R119 |  | The device BMS shall contain an Overcharge protection circuit. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R120 |  | The device BMS shall contain an Overcurrent draw protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R123 |  | The device BMS shall contain cell-balancing. | Reference E50947 BMS Chip datasheet |  |
| RSK_R124 |  | The battery charger shall be integrated into the device. | Specification Identical to RequirementReference E1 emitter drawing and C1 cassette drawing |  |
| RSK_R131 |  | The device battery shall contain a Thermal fuse or CID. | Specification Identical to RequirementReference battery drawings | overcurrent protection is covered by IEC62133; sufficient to cite 62133-2 |
| RSK_R132 |  | Battery charging circuits shall be designed to accept 5V - 20V charging circuit power input. | Specification Identical to RequirementReference battery drawings |  |
| RSK_R134 |  | The device FW shall open the HV interlock in the event of a filament current fault. | Reference SRS | Revisit in Phase 4 |
| RSK_R135 |  | The device FW shall disallow x-ray emission in the event of a high kV fault. | Reference SRS |  |
| RSK_R136 | Hardware Timing on X-ray exposure | The device HW shall open the HV interlock in the event of a x-ray exposure time (210 ms) interlock fault. | Specification identical to requirementReference ES-10019 P01 Monoblock LV PCBA | Updated 200 to 210 ms. |
| RSK_R138 |  | The device FW shall disallow x-ray emission in the event of a tube current fault. | Reference SRS |  |
| RSK_R140 |  | The Emitter and Cassette battery packs shall contain a temperature controlled power cut out in the event of overheat. | Specification identical to requirementReference BMS register values | All cut outs are FW driven. There is no HW cut offs, except in the battery packs. - Z |
| RSK_R144 |  | The device SW shall contain bounds checking on all sensor data. | Reference SRS |  |
| RSK_R151 |  | The device shall contain a cooling system to maintain internal temperature below standard limits while within operating environmental conditions. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R155 |  | Components and materials used in the device shall be rated to meet safety limits for the lifetime of the device per device requirements. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R156 |  | Electrical connectors shall be fixed or strain relieved to prevent breakage or stress on connectors or other components. | Specification identical to requirementReference PCB assembly drawings |  |
| RSK_R160 |  | The device shall have redundancies in grounding. | Specification identical to requirementReference: ES-10004 CASSETTE MAIN BOARD, ES-10014 Cassette Isolation PCBA, ES-10015 Emitter Power Input PCBA, ES-10019 P01 Monoblock LV PCBA, and ES-10003 P01 Emitter Main PCBA |  |
| RSK_R163 |  | The device shall not implement parts that could contribute to sudden expulsion of parts (ie. vacuum display, mechanical spring, gas pressure cylinder). | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R178 |  | Combined to RSK_R179 |  |  |
| RSK_R181 |  | The x-ray tube assembly shall be EMI shielded. | Specification identical to requirementReference x-ray tube drawing |  |
| RSK_R183 |  | The device shall contain flat resting surfaces. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R185 |  | The device enclosures shall have a min thickness of 1.4mm. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R191 |  | The external cables shall be strain relieved. | Specification identical to requirementReference H1 wired charger drawing |  |
| RSK_R193 |  | The device shall contain internal anchoring/tie-wraps of wires. | Specification identical to requirementReference PCB drawings |  |
| RSK_R194 |  | The device shall contain ties around conductor bundles at the connector lead in. | Specification identical to requirementReference PCB drawings |  |
| RSK_R195 |  | The device shall contain redundant tie points for wires. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R224 |  | The emitter high voltage components shall contain a marking - Dangerous Voltage. | Specification identical to requirementReference label drawingsHigh voltage components are:-Monoblock LV PCB (ES-10019)-Monoblock LV PWS Rider (ES-10024) |  |
| RSK_R229 |  | The device internals shall incorporate thermal dissipation features (ie heat sink. heat pipes, fans vents, etc.). | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R237 |  | The device shall contain Addressing to prevent connection to incorrect network(s) while pairing. | Reference SRS |  |
| RSK_R243 |  | The device emitter, cassette, and footpedal shall be factory-paired prior to shipment. | To be added before Phase 4 |  |
| RSK_R249 |  | The device shall perform RSSI checks on the foot pedal wireless connection. | Reference SRS |  |
| RSK_R254 |  | The device shall perform a trigger interlock check sequence (w/wifi) on the detector. | Reference SRS |  |
| RSK_R258 |  | The device shall disallow the use of non USB-PD inputs. | The device (both emitter and cassette) shall not charge when connected to a <20V USB-C. |  |
| RSK_R260 |  | The manufacturing work instructions shall include Torque Spec And Thread Locking Compound. | To be added before Phase 4 |  |
| RSK_R263 |  | The manufacturing work instructions shall include Calibration of Tracking to Correct Mechanical Misalignment. | To be added before Phase 4 |  |
| RSK_R266 |  | The manufacturer shall maintain Documentation of the System (DHR). | To be added before Phase 4 |  |
| RSK_R269 |  | The manufacturing work instructions shall include steps for OEM Specified Process For Surface Preparation And Bonding Procedures. | To be added before Phase 4 |  |
| RSK_R270 |  | The manufacturing work instructions shall include Dielectric testing. | To be added before Phase 4 |  |
| RSK_R274 |  | The manufactur shall measure and provide focal spot size data. | To be added before Phase 4 |  |
| RSK_R277 |  | The manufacturing work instructions shall include Post-assembly calibration. | To be added before Phase 4 |  |
| RSK_R281 |  | The manufacturing work instructions shall include the use of Crimping Tools, Sized Crimps, And Connectors. | To be added before Phase 4 |  |
| RSK_R285 |  | The manufacturing work instructions shall include a step to check leakage. | To be added before Phase 4 |  |
| RSK_R300 |  | The device software shall include Integrity checks. | Reference SRS |  |
| RSK_R302 |  | The device software shall contain a Firewall. | Reference SRS |  |
| RSK_R317 |  | The device software shall encrypt studies stored on the device. | Reference SRS | Updated by Akeafa |
| RSK_R320 |  | The device software shall require credentials to access AP mode. | Reference SRS | Updated by Akeafa |
| RSK_R336 |  | The device shall use a 2-pole charging connector (i.e. no ground), so it cannot connect directly to PE. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R338 |  | The device software shall utilize AP client isolation to manage communications between cassette and emitter and cassette and tablet | Reference SRS |  |
| RSK_R340 |  | Adhesive as recommended by the manufacturer shall be placed between E50925 and ES-10004 to adhere the two parts together | Specification identical to requirementReference C1 cassette drawings | Added per VVPR-P01-095 and resulting R4.98 |
| RSK_R360 |  | Power supplies for the device shall be IEC 60601-1 compliant |  | Redundant to RSK_R125 |

### Table 10
|  | Requirement | Specification | NOTES / QUESTIONS / REDLINE COMMENTS |
| --- | --- | --- | --- |
| Software Reqs |  |  |  |
| Software/Firmware - General |  |  |  |
| RSK_R001 | The device SW shall alert the operator when irradiating. | Reference SRS | 60601-1-3 Section 6.4.2 |
| RSK_R002 | removed | -- |  |
| RSK_R232 | The device shall display a warning in the event of poor wireless communication (COMS) quality. | Reference SRS | Example. Faulty data packets |
| RSK_R003 | removed | -- |  |
| RSK_R233 | removed | -- | Not in RSK-P01. |
| RSK_R004 | The device SW communication protocol shall contain Sequence Number checks. | Reference SRS |  |
| RSK_R005 | The device SW shall queue up studies when network is not present. | Reference SRS |  |
| RSK_R006 | removed | -- |  |
| RSK_R007 | The device SW communication protocol shall perform a Packet Validation. | Reference SRS | Packet Val = Length, seq, checksum |
| RSK_R008 | The device SW shall contain an interlock if any part of the projected collimated x-ray field is outside the active area. | Reference SRS |  |
| RSK_R009 | removed | -- |  |
| RSK_R010 | The image processing algorithm shall be deterministic. | Reference SRS |  |
| RSK_R011 | removed | -- |  |
| RSK_R012 | The device SW shall ensure valid communication using a Heartbeat. | Reference SRS | Do we need a robustness specification? (ex. <1% handshake failure)  - Mo |
| RSK_R013 | The device SW shall utilize a hardware Watchdog. | Reference SRS |  |
| RSK_R014 | The device SW shall contain a Message Checksum or Cyclic Redunacy Check (CRC) | Reference SRS | Dhruv says yes, we will have this 6SEP22-Mo |
| RSK_R015 | removed | -- | Revisit after 510k"Can remove since we're pre-pairing for 510k" - Akeafa- Is this specific to footpedal, or should be make this req more broad?- Is this there another risk mitigation that already covers this?example.com/ |
| RSK_R016 | removed | -- |  |
| RSK_R017 | removed | -- |  |
| RSK_R019 | removed |  |  |
| RSK_R020 | removed | -- | Revisit after 510k"This is a feature for 1.2.0" - Akeafa |
| RSK_R021 | removed | -- | Revisit after 510k |
| RSK_R022 | The device FW shall contain Filament Current Monitoring. | To be added before Phase 4 | "In the risk assessment, this is linked to "low filament current."R026 below covers low beam current Changed to "filament current monitoring"R142 below seems to cover beam current monitoring already."-AkeafaRevisit in Phase 4 |
| RSK_R023 | removed | -- |  |
| RSK_R025 | The device FW shall limit kV, exposure time, and tube current to values allowable in normal use. | Reference SRS | "Is this to say, the firmware sets maximum limits for these loading factors (e.g. sw can't request settings beyond those limits)? What does "limited" mean here?" - Akeafa |
| RSK_R026 | The device FW shall contain a low Beam Current Monitoring. | Reference SRS |  |
| RSK_R028 | The emitter shall contain a software interlock that disables the lasers when they are not pointed at the cassette. | Reference SRS |  |
| RSK_R029 | The device FW shall limit maximum exposure times for single and serial radiographic images. | Reference SRS |  |
| RSK_R032 | The device shall contain an HV control and monitor. | Reference SRS |  |
| RSK_R033 | The device FW shall initiate HV generation. | Reference SRS |  |
| RSK_R034 | removed | -- | Repeat of PRD8.2 |
| RSK_R036 | removed | -- | Revisit after 510k"The device shall indicate to the operator to replace the battery.""Are there any concerns with using the word alert everywhere, wrt to 60601-1-8 and how we don't conform to it? We were somewhat wary for P00. I don't care too much, but in sw docs, we'll stick to "indicate."  - Akeafa |
| RSK_R037 | removed | -- | Revisit after 510k |
| RSK_R038 | removed | -- | Revisit after 510k |
| RSK_R039 | removed | -- | Revisit after 510k1.2.0 |
| RSK_R040 | removed | -- | Moved to SW v1.2.0 |
| RSK_R041 | The device shall use of proprietary protocols. | Reference SRS | Example. Only able to pair to MedAI Footpedals, and other MedAI Accessories. Can't use random wired USB footpedal |
| RSK_R042 | removed | -- |  |
| RSK_R043 | removed | -- |  |
| RSK_R044 | The device shall contain a response verification (i.e message acknowledgement) between components. | Reference SRS |  |
| RSK_R045 | removed | -- | Not in risk assessment. Combined with PRD7.45 |
| RSK_R046 | removed | -- | Repetitive of RSK_R008 and PRD3.3 |
| RSK_R047 | removed | -- | Repetitive of RSK_R008 and PRD3.3 |
| RSK_R048 | The device shall implement exposure data tracking. | Reference SRS |  |
| RSK_R049 | removed | -- | Revisit after 510k"This is a feature for 1.2.0What's the difference between this and the orientation tagging req above? If possible, it would be ideal for L/R to be referred to as laterality markers and anatomy/patient orientation tags to be referred to in that way. Or, at least clarify which indicates are grouped under orientation."" - Akeafa |
| RSK_R050 | removed | -- | Revisit after 510k"Not sure why a Platform feature is being used as a risk mitigation. Imo this seems like it's more likely to increase risk for both products than to minimize risk for either."- Akeafa |
| RSK_R051 | removed | -- | Repeat of PRD7.13 |
| RSK_R052 | removed | -- | "Where is this 20% coming from? Is that for the hw timer only? SW/FW controls entrance into safe state, so if this is just for the hw timer, would want to remove safe state reference." - Akeafa"This is repeat of RSK_R136. Removed"-Mo |
| RSK_R054 | removed | -- | Risk is mitgated by RSK_R053.This is a software debounce -Akeafa |
| RSK_R055 | The radiation sequence shall terminate within 1 frame post trigger release. | Reference SRS | DDR always checking the trigger state. |
| RSK_R330 | The device software shall ping PACS server before sending packet(s). | Reference SRS |  |
| RSK_R331 | removed | -- | Repeat of RSK_R012 (heartbeat) |
| RSK_R332 | removed | -- |  |
| RSK_R337 | removed | -- | Revisit after 510k"The AiLARA algorithm shall be validated to select appropriate output settings." |
| Start-up Procedure |  |  |  |
| RSK_R056 | The device shall contain a Start-up Check. | Reference SRS |  |
| RSK_R057 | removed | -- |  |
| RSK_R058 | The device shall contain a Filament preheat. | Reference SRS | "There is a preheat for the filament" - David/Hartman 1/9/23 |
| RSK_R059 | removed | -- | Hartman "This makes no sense... I think this should be deleted." |
| RSK_R060 | removed | -- | "There is no warmup for the HV."- David/Hartman 1/9/23 |
| RSK_R062 | removed | -- | "I believe this RSK item was misinterrepted. RMF1.11.78, and RMF1.11.81 refer to having pre-set selectable mA values. Updated the mitigation to PRD2.9"-Mo |
| UI/Display |  |  |  |
| RSK_R063 | The device UI shall display technique factors post-imaging. | Reference SRS |  |
| RSK_R064 | The device shall only allow discrete technique choices when in manual mode. | Reference SRS |  |
| RSK_R065 | The device shall contain a UI Button to Reset Image Adjustments. | Reference SRS |  |
| RSK_R066 | The Software UI shall display Error Messages. | Reference SRS | In RSK-P01, Rev A, this is used for a lot of things.  In SRS, should split up into multiple error messages with more specificity (i.e overheating, failure to send to PACS server, etc...).Clarified "Software UI" which includes viewfinder & display |
| RSK_R067 | removed | -- | Repeat of PRD4.20 |
| RSK_R068 | removed | -- | "It will. In the risk assessment, RSK_R068 looks to be linked to the following mitigation:Internal verification of collimator function upon startup"-Akeafa |
| RSK_R069 | removed | -- | The device shall restrict the operator from shooting multiple x-rays in <1 sec intervals while in a single shot mode setting. |
| RSK_R070 | removed | -- | Consider moving to PRD or SRS?This mitigation not specific to the RSK. If we want to keep it as a requirement, we should add to the PRD.-Mo |
| RSK_R071 | removed | -- | Repeat of PRD4.10, and PRD8.31 |
| RSK_R072 | removed | -- | Repeat of PRD4.10 |
| RSK_R073 | removed | -- | Repeat of PRD9.21 |
| RSK_R074 | removed | -- |  |
| RSK_R075 | removed | -- | Repeat of PRD7.13 |
| RSK_R076 | removed | -- | Repeat of PRD10.12 |
| SW Security |  |  |  |
| RSK_R287 | removed | -- | Updated by Akeafa |
| RSK_R288 | removed | -- |  |
| RSK_R289 | removed | -- |  |
| RSK_R290 | removed | -- |  |
| RSK_R291 | removed | -- |  |
| RSK_R292 | removed | -- |  |
| RSK_R293 | removed | -- |  |
| RSK_R294 | removed | -- | Updated by Akeafa |
| RSK_R295 | removed | -- | Updated by Akeafa |
| RSK_R296 | removed | -- | Re-added by Akeafa |
| RSK_R297 | removed | -- |  |
| RSK_R298 | removed | -- |  |
| RSK_R299 | removed | -- |  |
| RSK_R300 | The device software shall include Integrity checks. | Reference SRS | "When P00 is powered on, there's a script that runs and checks to make sure there's been no unauthorized modification to the sw since the last power-on time. I think the check also runs hourly, but my memory is a bit blurry on that one. There's a step or two during production to handle changes resulting from upgrades. If the check determines there's been an unexpected change, it shuts the system down. There are no other user-facing changes.If you're familiar with people demanding that you "run inithashes", that's related to the integrity checks."-Akeafa |
| RSK_R301 | removed | Reference SRS |  |
| RSK_R302 | The device software shall contain a Firewall. | Reference SRS |  |
| RSK_R303 | removed | Reference SRS | Updated by Akeafa |
| RSK_R304 | removed | Reference SRS | Updated by Akeafa |
| RSK_R305 | removed | Reference SRS |  |
| RSK_R306 | removed | Reference SRS |  |
| RSK_R307 | removed | Reference SRS |  |
| RSK_R308 | removed | Reference SRS |  |
| RSK_R309 | removed | Reference SRS |  |
| RSK_R310 | removed | Reference SRS | Updated by Akeafa |
| RSK_R311 | removed | Reference SRS | Updated by Akeafa |
| RSK_R312 | removed | Reference SRS | Updated by Akeafa |
| RSK_R313 | removed | Reference SRS |  |
| RSK_R315 | removed | Reference SRS | Updated by Akeafa |
| RSK_R316 | removed | Reference SRS | Updated by Akeafa |
| RSK_R317 | The device software shall encrypt studies stored on the device. | Reference SRS |  |
| RSK_R318 | removed | Reference SRS | Updated by Akeafa |
| RSK_R319 | removed | Reference SRS | Updated by Akeafa |
| RSK_R320 | The device software shall require credentials to access AP mode. | Reference SRS |  |
| RSK_R338 | The device software shall utilize AP client isolation to manage communications between cassette and emitter and cassette and tablet | Reference SRS |  |
| RSK_R321 | removed | -- | Updated by Akeafa |
| RSK_R322 | removed | -- | Updated by Akeafa |
| RSK_R325 | removed | -- | Updated by Akeafa |
| RSK_R326 | removed | -- | Updated by Akeafa |
| RSK_R333 | removed | -- | Updated by Akeafa |
| RSK_R334 | removed | -- | Updated by Akeafa |
| Electrical Reqs |  |  |  |
| EE Inputs - General |  |  |  |
| RSK_R053 | The device shall contain a debounce on the trigger. | While in single-shot mode, the MX1 system shall fire only one x-ray in a one second interval. |  |
| RSK_R077 | removed | -- | No longer using single shot mode |
| RSK_R078 | removed | -- | "Safe state = sw, and fw reports critical faults to swNot sure why this one's in the EE section"-AkeafaRequirement covered by other PRD & RSK items.Example.  PRD6.2, PRD6.3, PRD6.4-Mo |
| RSK_R079 | The device shall contain an AC/DC brick with surge protection specifications. | H1 charger complies with IEC 60601-1-2 |  |
| RSK_R080 | removed | -- |  |
| RSK_R081 | The device shall contain isolation from Mains - Overvoltage Cat 2 Power Supply. | Specification identical to requirement | Confirm Cat 2 Power Supply |
| RSK_R082 | The device shall maintain high voltage ONLY during x-ray emission. | The x-ray tube does not contain a grid. |  |
| RSK_R083 | The device shall fail safe when unexpected arcing occurs. | Maintains BASIC SAFETY per IEC 60601-1 |  |
| RSK_R084 | The device shall fail safely in the event of undercurrent. (e.g. battery dies) | Maintains BASIC SAFETY per IEC 60601-1 | Reworded to clarify requirement. "Safe state" is used in software/firmware. This mitigation is to verify the device does not act unsafely when the battery dies or provides undercurrent. |
| RSK_R085 | The device shall contain current limiting devices. | The following PCBAs shall contain current limiting devices:Reference ES-10003 P01 Emitter Main PCBA, ES-10004 CASSETTE MAIN BOARD, ES-10023 Battery Management System (BMS), ES-10022 Cassette BMS PCBA, ES-10015 PMUX (Emitter Power Input) PCB, and ES-10014 Cassette Isolation PCBA |  |
| RSK_R086 | removed | -- |  |
| RSK_R087 | removed | -- |  |
| RSK_R088 | removed | -- |  |
| RSK_R089 | removed | -- | Covered by PRD7.18 and RSK_R028 |
| RSK_R090 | The IR LEDs intensity shall be below the injury threshold | The IR LEDs shall comply with IEC 62471 per component specifications | Refer to:MEMO-P00-159 - Cassette LED Regulatory Analysis |
| RSK_R091 | removed | -- |  |
| RSK_R092 | removed | -- | Internal temperature monitoring is covered under RSK_R143.Foose and Z confirmed - we do not peform external temperatuer monitoring, example if cassette touch temperature threshold is exceeded |
| RSK_R093 | removed | -- | "Is this finalized? Why does the device need to automatically shut down?"-AkeafaRepeat of RSK_R140.  Mitigation for R11.61 (Overheating), which is mitigated by other HW/SW monitors, cut-offs, and warnings.-Mo |
| RSK_R094 | The device shall remove/reduce Concentrated E-Fields. | The device shall be compliant to IEC 60601-1-2 |  |
| RSK_R095 | The device shall restrict HV to within the monoblock. | The device shall maintain BASIC SAFETY per IEC 60601-1 |  |
| RSK_R096 | The device shall contain a redundant thermostat. | Specification identical to requirementReference E1 emitter drawings |  |
| RSK_R097 | The device shall contain redundant thermistor(s) | Specification identical to requirementReference E1 emitter drawings |  |
| RSK_R098 | The device shall contain a Bleed-off circuit (for capacitive energies). | Specification identical to requirementReference E1 emitter drawings |  |
| RSK_R099 | The device shall contain an HV Transient Suppression on critical circuits. | The device shall contain an HV transient suppresssion on the following circuits: HMIs, power input, debug ports | esd protection on crit circuits |
| RSK_R100 | The device charger (H1) shall comply to USB-PD specifications set by USB-IF. | Specification identical to requirementReference H1 emitter drawings |  |
| RSK_R101 | The cassette shall isolate power input to patient accessible ports/parts. | Specification identical to requirementReference ES-10014 - Cassette Isolation PCBA |  |
| RSK_R102 | The emitter shall contain a FW interlock to prevent x-ray emissions when physically plugged into an external power source. | Reference SRS | "I think it requires SW as well, but there is an interlock. There is hardware that allows SW to detected if plugged in." -(Mo per David 31AUG22) |
| RSK_R336 | The device shall use a 2-pole charging connector (i.e. no ground), so it cannot connect directly to PE. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R339 | removed |  |  |
| RSK_R340 | Adhesive as recommended by the manufacturer shall be placed between E50925 and ES-10004 to adhere the two parts together | Specification identical to requirementReference C1 cassette drawings | Added per VVPR-P01-095 and resulting R4.98 |
| X-ray Assembly |  |  |  |
| RSK_R103 | The device shall mechanically protect the tube. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab)After exposure to ISTA 3A conditioning for Standard Packaged Product, the device shall:-Maintain essential performance (per MEMO-P01-441)-Have no visible damage that affects safety or performance of the device |  |
| RSK_R104 | removed | -- |  |
| RSK_R105 | removed | -- | Repeat of PRD2.2 |
| RSK_R235 | removed | -- |  |
| RSK_R230 | The HV insulation shall integrate redundant safety methods. | Floating loop - Battery shall not connect directly to MAINS. Reference E1 emitter drawing and C1 cassette drawingThe device shall disable operation while plugged in - Reference RSK_R102Dielectric strength & Enclosure as a layer of insulation - The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) | Removed arc monitoring. |
| RSK_R107 | removed | -- |  |
| RSK_R108 | The x-ray tube assy shall contain a temperature monitor on the anode. | Specification identical to requirementReference x-ray tube drawing |  |
| RSK_R109 | The device shall not contain a rotating anode. | Specification identical to requirementReference x-ray tube drawing |  |
| RSK_R110 | removed | -- |  |
| Wireless Coexistence |  |  |  |
| RSK_R237 | The device shall contain Addressing to prevent connection to incorrect network(s) while pairing. | Reference SRS |  |
| RSK_R238 | removed | -- | "Need to remove/update this requirement.Channel hopping requires back and forth communication according to Banks. The foot pedal-emitter communication has always been one way (from FP to emitter). We've never had an iteration with two-way communication. No impacts to Wireless Coexistence testing to remove/change this one."- Akeafa |
| RSK_R239 | The device shall contain DICOM ping to PACS upon data upload/export. | Reference SRS |  |
| RSK_R240 | The device sub GHz radios have built-in cyclic redundancy checks (CRCs) | Specification identical to requirementReference E50524 GHz Radio datasheet | Palumbo says not required/super risky, but good practice. Improves performanace and range of footpedal in case a message is obstructed. |
| RSK_R241 | removed | -- | Repeat of RSK_R012 |
| RSK_R242 | removed | -- |  |
| RSK_R244 | removed | -- | Revisit after 510kNFC related. |
| RSK_R245 | removed | -- | Revisit after 510kNFC related. |
| RSK_R246 | removed | -- | Revisit after 510kNFC related. |
| RSK_R247 | removed | -- | Revisit after 510k"Really don't think it's ideal using Platform features as a risk mitigation, just seems to increase burden on both products. Think Platform integration is best handled under PRDs." - Akeafa |
| RSK_R248 | removed | -- | Repeat of RSK_R012 (heartbeat) |
| RSK_R249 | The device shall perform RSSI checks on the foot pedal wireless connection. | Reference SRS |  |
| RSK_R250 | The device shall limit the maximum duration time on DDR studies. | Reference SRS | This is true for remote trigger and device trigger (jammed). The device can only emit up to 20s.Note* The way DDR currently works, it only looks for "start" and "stop".   There is no checking that the trigger is constantly being held, which could be a future mitigation if required. |
| RSK_R252 | removed | -- |  |
| RSK_R253 | removed | -- |  |
| RSK_R254 | The device shall perform a trigger interlock check sequence (w/wifi) on the detector. | Reference SRS |  |
| RSK_R256 | removed | -- | Repeat of PRD9.16 |
| RSK_R257 | removed | -- | Not an apropriate mitigation for "Noticeable DDR delay" |
| Battery |  |  |  |
| RSK_R111 | The device shall contain a battery charge status indicator on the emitter and cassette UI. | Reference SRS |  |
| RSK_R112 | The device shall contain a Low battery warning on the Display UI. | Reference SRS |  |
| RSK_R113 | removed | -- |  |
| RSK_R114 | The device BMS shall contain under-voltage protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R115 | The device BMS shall contain Cell temp monitoring for under and over-temp protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R116 | removed | -- | "Neither David nor I have any idea what this is. It should be deleted from here and from the risk assessment." -Z |
| RSK_R117 | The device BMS shall contain Over-current protection. | The battery shall comply with IEC 62133-2 (complies per third party report) | Short circuit detection |
| RSK_R118 | The device BMS shall contain Over-voltage protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R119 | The device BMS shall contain an Overcharge protection circuit. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R120 | The device BMS shall contain an Overcurrent draw protection. | The battery shall comply with IEC 62133-2 (complies per third party report) |  |
| RSK_R121 | removed | -- |  |
| RSK_R123 | The device BMS shall contain cell-balancing. | Reference E50947 BMS Chip datasheet |  |
| RSK_R124 | The battery charger shall be integrated into the device. | Specification Identical to RequirementReference E1 emitter drawing and C1 cassette drawing |  |
| RSK_R125 | The device shall contain IEC 60601-1 rated battery chargers. | The H1 Wired Charger shall comply with IEC 60601-1 |  |
| RSK_R127 | removed | -- | Revisit after 510k |
| RSK_R128 | removed | -- |  |
| RSK_R129 | removed | -- |  |
| RSK_R131 | The device battery shall contain a Thermal fuse or CID. | Specification Identical to RequirementReference battery drawings | overcurrent protection is covered by IEC62133; sufficient to cite 62133-2 |
| RSK_R132 | Battery charging circuits shall be designed to accept 5V - 20V charging circuit power input. | Specification Identical to RequirementReference battery drawings |  |
| RSK_R133 | removed | -- | Device by requires use of a tool to open. And there is IFU warning not to service device. |
| RSK_R258 | The device shall disallow the use of non USB-PD inputs. | The device (both emitter and cassette) shall not charge when connected to a <20V USB-C. |  |
| RSK_R335 | Batteries shall be keyed connectors to prevent incorrect installation by MedAI sevice personnel. | Specification Identical to RequirementReference battery drawings |  |
| HW & FW Interlock Controls |  |  |  |
| RSK_R134 | The device FW shall open the HV interlock in the event of a filament current fault. | Reference SRS | Revisit in Phase 4 |
| RSK_R135 | The device FW shall disallow x-ray emission in the event of a high kV fault. | Reference SRS |  |
| RSK_R136 | The device HW shall open the HV interlock in the event of a x-ray exposure time (210 ms) interlock fault. | Specification identical to requirementReference ES-10019 P01 Monoblock LV PCBA | Updated 200 to 210 ms. |
| RSK_R137 | removed | -- |  |
| RSK_R138 | The device FW shall disallow x-ray emission in the event of a tube current fault. | Reference SRS |  |
| RSK_R139 | The device HW shall open the HV interlock in the event of an unexpected tube current fault. | Specification identical to requirementReference ES-10019 P01 Monoblock LV PCBA |  |
| RSK_R140 | The Emitter and Cassette battery packs shall contain a temperature controlled power cut out in the event of overheat. | Specification identical to requirementReference BMS register values | All cut outs are FW driven. There is no HW cut offs, except in the battery packs. - Z |
| Monitoring and Protection (HW & SW) |  |  |  |
| RSK_R142 | The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low | Reference SRS |  |
| RSK_R143 | The device SW shall monitor the internal temperature(s) and enter a safe state in the event of under or over temp. | Reference SRS |  |
| RSK_R144 | The device SW shall contain bounds checking on all sensor data. | Reference SRS |  |
| RSK_R146 | The device shall verify the integrity of the safety-critical wireless systems within 1 second of every X ray exposure | Reference SRS | this is used in the context of ionizing radiation; mitigation to require regular (every 1 second, or within 1 sec of every exposure) heartbeat checks on all safety-critical wireless systems |
| RSK_R147 | removed | -- |  |
| RSK_R148 | The device shall enter safe state (or warn the operator) in the event of under or over humidity | Reference SRS |  |
| RSK_R161 | removed | -- | Repeat of PRD2.3. (It seems this requirement is referring to beam current and voltage monitoring) |
| Mechanical Reqs |  |  |  |
| ME Inputs (HW) - General |  |  |  |
| RSK_R149 | removed | -- | Revisit after 510kThe device contains gasket but won't be totally sealed anymore (i.e. no IP rating). Therefore, no pressure valves are required or included in the design. (Mo consulted with Andrew) |
| RSK_R150 | removed | -- |  |
| RSK_R151 | The device shall contain a cooling system to maintain internal temperature below standard limits while within operating environmental conditions. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R152 | The emitter and cassette enclosures shall protect the monoblock and detector using vibration and shock dampening materials. | After exposure to ISTA 3A conditioning for Standard Packaged Product, the device shall:-Maintain essential performance (per MEMO-P01-441)-Have no visible damage that affects safety or performance of the device Passing 60601-1 Report |  |
| RSK_R153 | The device shall be primarily packaged in a hard shell case with closed cell internal foam or similar style case. | Specification identical to requirementReference P1 case drawing |  |
| RSK_R154 | removed | -- | Repeat of PRD4.1 |
| RSK_R155 | Components and materials used in the device shall be rated to meet safety limits for the lifetime of the device per device requirements. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R156 | Electrical connectors shall be fixed or strain relieved to prevent breakage or stress on connectors or other components. | Specification identical to requirementReference PCB assembly drawings |  |
| RSK_R157 | removed | -- | Repeat of PRD20.9 |
| RSK_R158 | removed | -- | RSK-P01 hazard mitigated by other means. |
| RSK_R159 | The device fans shall be internally caged and shrouded. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R160 | The device shall have redundancies in grounding. | Specification identical to requirementReference: ES-10004 CASSETTE MAIN BOARD, ES-10014 Cassette Isolation PCBA, ES-10015 Emitter Power Input PCBA, ES-10019 P01 Monoblock LV PCBA, and ES-10003 P01 Emitter Main PCBA |  |
| RSK_R162 | removed | -- |  |
| RSK_R163 | The device shall not implement parts that could contribute to sudden expulsion of parts (ie. vacuum display, mechanical spring, gas pressure cylinder). | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R229 | The device internals shall incorporate thermal dissipation features (ie heat sink. heat pipes, fans vents, etc.). | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| Accessories |  |  |  |
| RSK_R164 | removed | -- | Not in RSK-P01 |
| RSK_R165 | removed | -- | Not in RSK-P01 |
| RSK_R166 | removed | -- | Move to K1 Cart DR"The device cart shall have locking wheels." |
| RSK_R167 | removed | -- | Move to K1 Cart DR"The cart stand shall contain a gas spring that has a force balanced to the cassette platform weight." |
| RSK_R168 | removed | -- | Move to K1 Cart DR"The cart stand gas spring maximum retraction velocity shall be less than 5 m/s." |
| Enclosure |  |  |  |
| RSK_R169 | The device enclosures shall be composed of an impact and heat resistant plastic. | The enclosures shall be composed of PolycarbonateV2 or similar material with equivalent or better heat and impact resistanceThe device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R234 | removed | -- | Repeat of RSK_R169 |
| RSK_R171 | removed | -- |  |
| RSK_R172 | The device internals shall not be accessible without the use of a tool. | Specification identical to requirementReference E1 emitter and C1 cassette drawings | Corrected to match RSK assessment. Added new RSK_ |
| RSK_R174 | removed | -- |  |
| RSK_R175 | removed | -- |  |
| RSK_R176 | removed | -- |  |
| RSK_R177 | The device enclosures shall prevent operator access to conductive components / contacts. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R178 | Combined to RSK_R179 |  |  |
| RSK_R179 | The device shall utilize covers and guards when necessary to prevent access to electrical, thermal, and moving components. | The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. (complies per third party test lab) |  |
| RSK_R180 | removed | -- | Not required if passing IEC 60601-1-2 |
| RSK_R181 | The x-ray tube assembly shall be EMI shielded. | Specification identical to requirementReference x-ray tube drawing |  |
| RSK_R182 | The device enclosures shall reduce falling and tipping risk by providing a surface with a high friction material. | The emitter and cassette shall contain Rubber feet |  |
| RSK_R183 | The device shall contain flat resting surfaces. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R184 | removed | -- | Repeat of PRD20.4. Comply with 60601-1 tip test |
| RSK_R185 | The device enclosures shall have a min thickness of 1.4mm. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R186 | The Cassette shall be sealed using a gasket or elastomeric seal. | Specification identical to requirementReference C1 cassette drawings | Emitter isn't sealed. Cassette (applied part) is. |
| RSK_R187 | removed | -- |  |
| RSK_R188 | removed | -- |  |
| RSK_R189 | The device enclosures window pieces shall not contain holes. | MS-10270 and MS-10271 shall not contain holes |  |
| RSK_R328 | removed | -- | Covered in usability testing. |
| RSK_R329 | The device trigger shall contain features that prevent jamming. | Device must contain a sliding trigger button with a trigger button cover. |  |
| Cables & Connectors |  |  |  |
| RSK_R190 | The device shall integrate wire routing to prevent mechanical damage to the wires. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R191 | The external cables shall be strain relieved. | Specification identical to requirementReference H1 wired charger drawing |  |
| RSK_R192 | removed | -- | No external cables. Covered by PRD20.5. Comply with 60601-1-2. |
| RSK_R193 | The device shall contain internal anchoring/tie-wraps of wires. | Specification identical to requirementReference PCB drawings |  |
| RSK_R194 | The device shall contain ties around conductor bundles at the connector lead in. | Specification identical to requirementReference PCB drawings |  |
| RSK_R195 | The device shall contain redundant tie points for wires. | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| RSK_R196 | The device shall contain locking connectors on all internal harnesses and external cables | Specification identical to requirementReference E1 emitter and C1 cassette drawings |  |
| Collimator |  |  |  |
| RSK_R197 | The device collimator shall be encased within the enclosure. | Specification identical to requirementReference E1 emitter drawing |  |
| RSK_R198 | removed | -- | Dhruv says this is verified via heartbeat (RSK_R012). |
| Pucks |  |  |  |
| RSK_R199 | removed | -- | Not in RSK-P01 |
| RSK_R200 | removed | -- | Repeat of PRD18.7, unique labeling |
| RSK_R201 | removed | -- | Not in RSK-P01 |
| RSK_R202 | The pucks and emitter front face (puck attachment area) shall contain a magnet redundancies (more than one magnet per puck and emitter attachment area). | Specification identical to requirementReference pucks and E1 emitter drawings |  |
| RSK_R203 | removed | -- | Repeat of PRD18.7, unique labeling |
| Labels |  |  |  |
| RSK_R204 | removed | -- | Was not here before, but is RSK-P01. |
| RSK_R205 | removed | -- |  |
| RSK_R206 | removed | -- | Removed. Mitigation does not provide much value. Device already has bumpers, and lights on top. It clearly only goes in one orientation. (Mo spoke with Taylor) |
| RSK_R207 | removed | -- | Repeat of IFU.109 |
| RSK_R208 | removed | -- | Not in RSK-P01 |
| RSK_R209 | removed | -- | Not in RSK-P01All connectors shall be clearly labeled or indicated |
| RSK_R210 | removed | -- |  |
| RSK_R211 | removed | -- | Not in RSK-P01Marking and durability test |
| RSK_R212 | removed | -- | Not in RSK-P01 |
| RSK_R213 | removed | -- | Not in RSK-P01 |
| RSK_R214 | removed | -- |  |
| RSK_R215 | removed | -- |  |
| RSK_R216 | removed | -- |  |
| RSK_R217 | removed | -- |  |
| RSK_R218 | removed | -- |  |
| RSK_R219 | removed | -- |  |
| RSK_R220 | removed | -- | Not required. Risk mitigated via ISTA testing. |
| RSK_R221 | removed | -- | Repeat of PRD20.4 requirements. |
| RSK_R222 | removed | -- |  |
| RSK_R223 | removed | -- | Not in RSK assesment. |
| RSK_R224 | The emitter high voltage components shall contain a marking - Dangerous Voltage. | Specification identical to requirementReference label drawingsHigh voltage components are:-Monoblock LV PCB (ES-10019)-Monoblock LV PWS Rider (ES-10024) |  |
| RSK_R327 | removed | -- | Repeat of IFU.114 |
| Other |  |  |  |
| Misc |  |  |  |
| Manufacturing (To be verified in Phase 4 of Design Controls)Note that these manufacturing/process requirements and controls are not all-inclusive and additional requirements will be added in Development Phase 4 based on the manufacturing process and associated Process Failure Mode Effects Analysis (PFMEA). |  |  |  |
| RSK_R228 | removed | -- | Not in RSK-P01 |
| RSK_R259 | removed | -- |  |
| RSK_R243 | The device emitter, cassette, and footpedal shall be factory-paired prior to shipment. | To be added before Phase 4 | Revisit after 510kThis a feature that allows the user to pair and emitter & cassette over wifi, in case NFC isn't working.We could DELETE this requirement. And just rely on the fact the device will be "pre-paired" prior to shipment.The footpedal will HAVE to be "pre-paired" to the emitter anyways. There is not an option for the user to manually pair it. |
| RSK_R260 | The manufacturing work instructions shall include Torque Spec And Thread Locking Compound. | To be added before Phase 4 |  |
| RSK_R263 | The manufacturing work instructions shall include Calibration of Tracking to Correct Mechanical Misalignment. | To be added before Phase 4 |  |
| RSK_R264 | removed | -- |  |
| RSK_R265 | removed | -- | Verification testing to be determined during PFMEA. |
| RSK_R266 | The manufacturer shall maintain Documentation of the System (DHR). | To be added before Phase 4 |  |
| RSK_R267 | The manufacturing work instructions shall include Dose Verification. | To be added before Phase 4 |  |
| RSK_R285 | The manufacturing work instructions shall include a step to check leakage. | To be added before Phase 4 |  |
| RSK_R269 | The manufacturing work instructions shall include steps for OEM Specified Process For Surface Preparation And Bonding Procedures. | To be added before Phase 4 |  |
| RSK_R270 | The manufacturing work instructions shall include Dielectric testing. | To be added before Phase 4 |  |
| RSK_R274 | The manufactur shall measure and provide focal spot size data. | To be added before Phase 4 |  |
| RSK_R277 | The manufacturing work instructions shall include Post-assembly calibration. | To be added before Phase 4 |  |
| RSK_R279 | The manufacturing work instructions shall include Tube Inspection and seasoning. | To be added before Phase 4 |  |
| RSK_R281 | The manufacturing work instructions shall include the use of Crimping Tools, Sized Crimps, And Connectors. | To be added before Phase 4 |  |
| RSK_R284 | removed | -- | Repeat of RSK_R274 |
| RSK_R282 | removed | -- | Not in RSK-P01 |
| RSK_R324 | removed | -- | Not in RSK-P01 |

### Table 11
|  | Requirement Identifier | Requirement | Specification |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Export to USB | PRD4.15 | The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm.Reference SRSAM - traced to SRS-17.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD10.15The emitter display shall display the viewfinder. | PRD7.11 | The device shall work with wireless viewing hardware (wireless tablets and wireless monitors) | Reference SRSAM - traced to SRS-30.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | PRD8.15 | The emitter and tablet shall be able to pair to the cassette, and the foot pedal shall pair to the emitter. | Reference SRSAM - traced to SRS-7.2, 7.3, 7.6, and 7.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | PRD8.56 | The device shall allow the operator to take and view images without external internet connectivity. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | PRD9.32 | The software UI should indicate the state of the device (e.g. Powered on, Charging, Available for imaging, Emitting radiation, and Error State) | Reference SRS - verified via demoAM - don't know if I traced to every single thing possible, but this is traced to SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD10.32        The emitter shall be available for use in less than 180 seconds of initiating power on. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD10.9The emitter trigger(s) shall allow the operator to trigger an x-ray or photograph. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD10.9        The emitter trigger(s) shall allow the operator to trigger an x-ray or photograph. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD17.3The device shall support the use of a foot pedal with 2 triggers and 2 buttons. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD17.4The foot pedal right pedal (B) shall initiate a single x-ray exposure upon pressing and releasing when in radiographic mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD17.5The foot pedal right pedal (B) shall initiate DDR on the downpress and shall stop the exposure upon release when in radiographic mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD17.7        The left button (A) shall switch between Radiography and Photography modes. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD17.8The foot pedal right button (B) shall rotate the image 90 degrees. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD17.9The foot pedal left pedal (A) shall "Favorite" or Save the current image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.10        The x-ray tube shall operate between 0.04 - 0.40 mAs in 5 steps; the options shall be 0.04, 0.08, 0.16, 0.25, 0.40 mAs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.10        The x-ray tube shall operate between 0.04 - 0.40 mAs in 5 steps; the options shall be 0.04, 0.08, 0.16, 0.25, 0.40 mAs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.10        The x-ray tube shall operate between 0.04 - 0.40 mAs in 5 steps; the options shall be 0.04, 0.08, 0.16, 0.25, 0.40 mAs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.11        The x-ray exposure in serial radiographic mode shall be 40ms per frame, 5 frames per second, for a maximum of 20 seconds. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.3        The device shall monitor and log beam current, filament current, and monoblock temperature with each acquisition. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.3        The device shall monitor and log beam current, filament current, and monoblock temperature with each acquisition. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.8The x-ray tube shall operate between 40 kV to 80 kV in 10kV increments. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.8The x-ray tube shall operate between 40 kV to 80 kV in 10kV increments. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD2.9The x-ray tube beam current shall operate between 1mA to 2mA. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.22        The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2016) Digital Imaging and Communications in Medicine (DICOM) Set). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.4The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.4The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.4The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.4The device shall comply with IEC 60601-1 Edition 3.1 2012 Requirements for Medical Electrical Equipment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.8        The device shall comply with IEC 60601-2-28 Edition 3.0 2017 Requirements for x-ray Tube Assemblies. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD20.9        The device shall comply with IEC 60601-2-54 Edition 1.2 2018 Requirements for Medical electrical equipment for radiography. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD3.10The device shall display to the operator the status of the system via indicator LEDs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD3.10The device shall display to the operator the status of the system via indicator LEDs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD3.10        The device shall display to the operator the status of the system via indicator LEDs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD3.10        The device shall display to the operator the status of the system via indicator LEDs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD3.13The automatic collimator shall be able to adjust the aperture size at any given SID; selectable steps shall not exceed 0.8 cm in the length and width when in a plane orthogonal to the reference at a distance of 80 cm from the focal spot. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD3.3       The device shall only allow x-ray emissions within a source-to-detector (SID) distance between 25cm and 80cm. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.10The viewfinder shall display loading factors before taking an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.10The viewfinder shall display loading factors before taking an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.10The viewfinder shall display loading factors before taking an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.10        The viewfinder shall display loading factors before taking an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.11        The viewfinder shall provide positioning guidance in the form of angle and SID. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.11        The viewfinder shall provide positioning guidance in the form of angle and SID. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.11        The viewfinder shall provide positioning guidance in the form of angle and SID. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.12        The viewfinder shall provide guidance on the UI to aid in aligning x-ray axis to cassette axis. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.14The viewfinder shall display the non-active area uniquely than the active area. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.17        The viewfinder shall include a reference gauge so that the operator understands where the emitter is positioned in reference to the detector. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.19The viewfinder shall show the x-ray field projection for the puck that is selected. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.20The viewfinder shall display the imaging mode (Radiography or Photography). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.20The viewfinder shall display the imaging mode (Radiography or Photography). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.20        The viewfinder shall display the imaging mode (Radiography or Photography). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.3        The viewfinder shall display the optical image transformed into an image as seen from the Cassette. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.5The viewfinder shall calculate and display the collimated x-ray field. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.5        The viewfinder shall calculate and display the collimated x-ray field. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.6The viewfinder shall calculate and display the active area of the detector. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.6        The viewfinder shall calculate and display the active area of the detector. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.7        The viewfinder shall include a reference point to indicate the center of the x-ray field. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.7        The viewfinder shall include a reference point to indicate the center of the x-ray field. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD4.8        The viewfinder shall overlay the x-ray field and active area on the optical image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD5.22The cassette shall support x-ray emissions while wired charging. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD5.6The emitter shall display the status of the charging system. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD5.8The cassette shall display the status of the charging system. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD6.7        The device shall perform a startup procedure to check wireless comms and calibration. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD6.7        The device shall perform a startup procedure to check wireless comms and calibration. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD6.7        The device shall perform a startup procedure to check wireless comms and calibration. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD6.7        The device shall perform a startup procedure to check wireless comms and calibration. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.12The device shall work with MedAI supplied and customer supplied tablets over WiFi. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13        The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13      The emitter and cassette shall contain status indicators to inform user of device readiness (e.g. armed/disarmed) and x-ray emission. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.13      The emitter and cassette shall contain status indicators to inform user of device readiness (e.g. armed/disarmed) and x-ray emission. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.22The device shall automatically reconnect to a known WiFi network after inputting password the first time. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.24The device shall serve as a private WiFi Access Point. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD7.24        The device shall serve as a private WiFi Access Point. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.1The device shall allow the operator to switch between radiographic and photographic modes. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.1The device shall allow the operator to switch between radiographic and photographic modes. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.1The device shall allow the operator to switch between radiographic and photographic modes. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10        The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.10        The device idle state shall be distinguished from active state. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11        The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11        The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11        The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11        The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11        The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.11        The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12        The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12        The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12        The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12        The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12        The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12 The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12 The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.12 The device shall allow sending files to PACS in the DICOM format. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.13The device should provide confirmation that the image study has been successfully submitted to PACS or local storage (USB Drive). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.14        The cassette shall be able to send/stream a image(s) to display hardware within 1 second from trigger release. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.16The device shall contain debug and release modes for service operators. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.20The devices DDR cycle shall have a lag of no greater than 200 ms from the real time of the scan to the display of that frame on the Software UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.21        The device should self-terminate a DDR if any part of the projected collimated x-ray field is moved outside the active area. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.24The device shall limit the duty-cycle of single radiographs to a maximum of 200ms of exposure and 1800ms minimum of cooldown. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.25The device shall accept hyphens and spaces as part of name inputs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.26The device shall limit the duty-cycle of serial radiographic mode to a maximum of 20 seconds of duration and proportional cooldown with a maximum of 40 seconds of cooldown. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.26The device shall limit the duty-cycle of serial radiographic mode to a maximum of 20 seconds of duration and proportional cooldown with a maximum of 40 seconds of cooldown. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.3The device shall contain different indicators for each mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.3        The device shall contain different indicators for each mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.3        The device shall contain different indicators for each mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.31        The system shall be display the loading factors (kV, mAs) used for capturing the image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.47        The device shall support image queuing for use off-network and network submission when connected. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.48The system may allow viewing two images at a time for surgical comparison on large monitor(s), and pinning images for comparison. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.50        The Mobile Device App shall be compatible with Android devices. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.51The device shall support the WPA2 protocol. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.53The device shall enter an idle state when the device is not utilized for 5 minutes. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.53        The device shall enter an idle state when the device is not utilized for 5 minutes. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.54The device shall exit an idle state within 20 seconds upon detection of emitter or foot pedal activity. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.55        The device shall disallow x-ray acquisition when the device is in idle state |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.8The system should save images prior to shut down. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.8        The system should save images prior to shut down. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD8.8        The system should save images prior to shut down. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.1        All data presented on the software UI shall have a unit of measure or label |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.1        All data presented on the software UI shall have a unit of measure or label |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.10The software UI shall allow the operator to rotate images; 360 degrees of rotation in 90 degree increments. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.11The software UI should persist rotation adjustments. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.16        The software UI should display the network, device connection status, and signal strength, updating every 5 seconds. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.16        The software UI should display the network, device connection status, and signal strength, updating every 5 seconds. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.17        The software UI should display the PACS connection status. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.2        The MedAI Device App shall display the manufacturer contact information, a unique UDI, a message to refer to the MX1 IFU, and a warning that primary image interpretation shoul occur on DICOM displays. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.21The software UI shall display the SID during use. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.22The software UI shall display the dose after each image acquisition. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.22The software UI shall display the dose after each image acquisition. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.26The software UI should inform the operator if any fault occurs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.26The software UI should inform the operator if any fault occurs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.26The software UI should inform the operator if any fault occurs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.26        The software UI should inform the operator if any fault occurs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.26        The software UI should inform the operator if any fault occurs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.28The software UI shall provide an indication when a failure to capture an image occurs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.37        The software UI shall include a dropdown to select a puck before use. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.38The software UI shall display the source-to-skin distance (SSD). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.39The software UI shall allow operator to select puck collimation size from a series of preselected options. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.40The software UI shall allow the user to adjust brightness, contrast, and sharpness of an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.40The software UI shall allow the user to adjust brightness, contrast, and sharpness of an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.40The software UI shall allow the user to adjust brightness, contrast, and sharpness of an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.40The software UI shall allow the user to adjust brightness, contrast, and sharpness of an image. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.5The software UI should display the image immediately after exposure without the operator interacting with the UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.7The software UI shall allow the operator to select and view acquired images. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.7The software UI shall allow the operator to select and view acquired images. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.7The software UI shall allow the operator to select and view acquired images. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.8        The software UI shall allow the operator to independently manipulate the images. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| PRD9.9The software UI shall allow the operator to "pinch to zoom" images. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R001        The device SW shall alert the operator when irradiating. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R001        The device SW shall alert the operator when irradiating. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R001        The device SW shall alert the operator when irradiating. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R001        The device SW shall alert the operator when irradiating. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R001        The device SW shall alert the operator when irradiating. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R004The device SW communication protocol shall contain Sequence Number checks. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R005        The device SW shall queue up studies when network is not present. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R007The device SW communication protocol shall perform a Packet Validation. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R008        The device SW shall contain an interlock if any part of the projected collimated x-ray field is outside the active area. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R010The image processing algorithm shall be deterministic. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R012        The device SW shall ensure valid communication using a Heartbeat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R013The device SW shall utilize a hardware Watchdog. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R014The device SW shall contain a Message Checksum or Cyclic Redunacy Check (CRC) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R022        The device FW shall contain Filament Current Monitoring. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R024The device FW shall contain a filament current setpoint (firmware to disallow filament current other than specified setpoint). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R025        The device FW shall limit kV, exposure time, and tube current to values allowable in normal use. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R025        The device FW shall limit kV, exposure time, and tube current to values allowable in normal use. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R025        The device FW shall limit kV, exposure time, and tube current to values allowable in normal use. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R026        The device FW shall contain a low Beam Current Monitoring. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R028        The emitter shall contain a software interlock that disables the lasers when they are not pointed at the cassette. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R029        The device FW shall limit maximum exposure times for single and serial radiographic images. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R032The device shall contain an HV control and monitor. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R032The device shall contain an HV control and monitor. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R033The device FW shall initiate HV generation. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R033        The device FW shall initiate HV generation. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R041The device shall use of proprietary protocols. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R044The device shall contain a response verification (i.e message acknowledgement) between components. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R048        The device shall implement exposure data tracking. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R048        The device shall implement exposure data tracking. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R048        The device shall implement exposure data tracking. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R048        The device shall implement exposure data tracking. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R048        The device shall implement exposure data tracking. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R048        The device shall implement exposure data tracking. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R055The radiation sequence shall terminate within 1 frame post trigger release. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R056        The device shall contain a Start-up Check. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R056        The device shall contain a Start-up Check. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R056        The device shall contain a Start-up Check. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R058The device shall contain a Filament preheat. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R063        The device UI shall display technique factors post-imaging. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R064        The device shall only allow discrete technique choices when in manual mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R065The device shall contain a UI Button to Reset Image Adjustments. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R066        The Software UI shall display Error Messages. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R066        The Software UI shall display Error Messages. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R102The emitter shall contain a FW interlock to prevent x-ray emissions when physically plugged into an external power source. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R111The device shall contain a battery charge status indicator on the emitter and cassette UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R111The device shall contain a battery charge status indicator on the emitter and cassette UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R111The device shall contain a battery charge status indicator on the emitter and cassette UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R111        The device shall contain a battery charge status indicator on the emitter and cassette UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R112The device shall contain a Low battery warning on the Display UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R112        The device shall contain a Low battery warning on the Display UI. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R135        The device FW shall disallow x-ray emission in the event of a high kV fault. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R138        The device FW shall disallow x-ray emission in the event of a tube current fault. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R142The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R142        The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R142        The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R142        The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R143The device SW shall monitor the internal temperature(s) and enter a safe state in the event of under or over temp. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R143The device SW shall monitor the internal temperature(s) and enter a safe state in the event of under or over temp. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R143        The device SW shall monitor the internal temperature(s) and enter a safe state in the event of under or over temp. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R143        The device SW shall monitor the internal temperature(s) and enter a safe state in the event of under or over temp. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R144        The device SW shall contain bounds checking on all sensor data. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R146        The device shall verify the integrity of the safety-critical wireless systems within 1 second of every X ray exposure |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R148The device shall enter safe state (or warn the operator) in the event of under or over humidity |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R148The device shall enter safe state (or warn the operator) in the event of under or over humidity |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R232        The device shall display a warning in the event of poor wireless communication (COMS) quality. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R232        The device shall display a warning in the event of poor wireless communication (COMS) quality. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R232        The device shall display a warning in the event of poor wireless communication (COMS) quality. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R232        The device shall display a warning in the event of poor wireless communication (COMS) quality. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R237        The device shall contain Addressing to prevent connection to incorrect network(s) while pairing. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R237        The device shall contain Addressing to prevent connection to incorrect network(s) while pairing. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R239        The device shall contain DICOM ping to PACS upon data upload/export. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R243        The device emitter, cassette, and footpedal shall be factory-paired prior to shipment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R243        The device emitter, cassette, and footpedal shall be factory-paired prior to shipment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R249        The device shall perform RSSI checks on the foot pedal wireless connection. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R250The device shall limit the maximum duration time on DDR studies. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R250        The device shall limit the maximum duration time on DDR studies. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R250        The device shall limit the maximum duration time on DDR studies. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R250        The device shall limit the maximum duration time on DDR studies. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R254        The device shall perform a trigger interlock check sequence (w/wifi) on the detector. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R300The device software shall include Integrity checks. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R302        The device software shall contain a Firewall. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R302        The device software shall contain a Firewall. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R317The device software shall encrypt studies stored on the device. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R320The device software shall require credentials to access AP mode. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R330        The device software shall ping PACS server before sending packet(s). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| RSK_R338        The device software shall utilize AP client isolation to manage communications between cassette and emitter and cassette and tablet |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 12
| ID | Requirement | Temporary DR-SRS Sanity Checks | User Needs | NOTES / QUESTIONS / REDLINE COMMENTS |
| --- | --- | --- | --- | --- |
| 14. Manufacturability and Serviceability  (To be evaluated in Phase 4 of Design Controls) |  |  |  |  |
| BN1 | All critical x-ray system components shall be sourced from countries that do not pose as adversaries to the US, per EAR policy. | - | UN26. | Essentially MedAI cannot source from Cuba, Iran, North Korea, Syria, and the Crimea Region of Ukraine. As well as any restricted parties by the Dept of State. |
| BN2 | The manufacturing processes shall be designed to support the production of 250 units a year (~20 units a month) and an assembly and test time of less than five man hours. | - | N/A | 1,000 units X $40,000 per unit = 40MM in sales |
| BN3 | The device shall be designed to allow for in-house pilot production of up to 20-30 units before transfer to CM. | - | N/A |  |
| BN4 | The device shall be designed to operate within the CM fabrication and manufacturing constraints. | - | N/A |  |
| BN5 | The cassette componenets shall be able to be assembled entirely within one half of the shell. | - | UN1. |  |
| BN6 | The emitter shall be able to be assembled entirely within one half of the shell. | - | UN1. |  |
| BN7 | The device shall be serviceable by the manufacturer and approved third party services. | - | UN1. |  |
| BN8 | The emitter shall be serviceable by the easy replacement of the monoblock, battery pack, and display at the factory. | - | UN1. |  |
| BN9 | The cassette shall be serviceable by the easy replacement of the battery pack at the factory. | - | UN1. |  |
| BN10 | The cassette shall be serviceable by the replacement of the battery pack without separating the enclosure halves. | - | UN1. |  |
| 16. Reliability and COGS (*To be verified in Phase 4 of Design Controls) |  |  |  |  |
| BN11 | The COGS shall be no more than $25,000 per device when manufactured in volume. | - | N/A |  |
| BN12 | The device service life shall be a minimum of 5 years of operation, with an initial limited warranty of 1 year. | - | UN1. |  |
| BN13 | The device service life shall be able to be extended by 3 years with the replacement of the monoblock, battery packs, display. | - | UN1. | "That is correct... But this is hard to validate without extensive End of Life testing..." Hartman |

### Table 13
|  | Test Method | Total | Complete | Remaining |
| --- | --- | --- | --- | --- |
|  | Analysis | 71 | 1 | 70 |
|  | Analysis/Test | 13 | 0 | 13 |
|  |  | 84 | 1 | 0.0119047619 |
|  | Demo | 179 | 119 | 60 |
|  | Demo/Analysis | 1 | 0 | 1 |
|  | Demo/Test | 3 | 0 | 3 |
|  |  | 183 | 119 | 0.650273224 |
|  | Inspection | 145 | 144 | 1 |
|  | Inspection/Analysis | 2 | 2 | 0 |
|  | Inspection - MEMO | 16 | 16 | 0 |
|  |  | 163 | 162 | 0.9938650307 |
|  | Test | 97 | 14 | 83 |
|  | Test/Analysis | 7 | 0 | 7 |
|  | Test/Demo | 22 | 5 | 17 |
|  |  | 126 | 19 | 0.1507936508 |
|  |  | 556 | 301 | 0.5413669065 |
| PRD# | Requirement | Method | Test Step |  |
|  | Analysis |  |  |  |
|  | SW Systems |  |  |  |
| PRD6.1 | The critical safety circuits shall not depend on software to trigger a fault. | Analysis |  |  |
| PRD6.7 | The device shall perform a startup procedure to check wireless comms and calibration. | Analysis |  |  |
| PRD6.9 | The device shall perform tube conditioning if emitter is inactive for 3 months or greater. | Analysis |  |  |
| PRD7.48 | The device shall disable the high voltage interlock when the device is in an idle state. | Analysis |  |  |
| PRD8.12 | The device shall allow sending files to PACS in the DICOM format. | Analysis |  |  |
| PRD8.17 | The system shall store the most recent 1000 photographic images. | Analysis |  |  |
| PRD8.18 | The system shall store the most recent 1000 radiographic images. | Analysis |  |  |
| PRD8.19 | The system shall store the most recent 100 serial radiography series. | Analysis |  |  |
| PRD8.24 | The device shall limit the duty-cycle of radiographic mode to a maximum of 100ms of exposure and 900ms minimum of cooldown. | Analysis |  |  |
| PRD2.11 | The x-ray exposure in serial radiographic mode shall be 33ms per frame, 10 frames per second, for a maximum of 20 seconds. | Analysis |  |  |
| PRD2.3 | The device shall monitor monoblock health and predict remaining filament life for servicing and repair by the mfg. | Analysis |  |  |
| PRD8.16 | The device shall contain a debug mode for development | Analysis |  |  |
| PRD8.4 | The system should have various access role assignments for login. (ie admin, service, user, etc.) | Analysis | Confirm multiple role assignments exist for login. |  |
| RSK_R002 | The device shall contain audible signals/indicators set to 80 dB for < 2 mins. | Analysis |  |  |
| RSK_R003 | The device shall contain less than 5 unique alert signal types. | Analysis |  |  |
| RSK_R004 | The device SW communication protocol shall contain Sequence Number checks. | Analysis |  |  |
| RSK_R005 | The device SW shall Queue up studies when network is not present. | Analysis |  |  |
| RSK_R006 | The device SW shall retain most recent images (qty 1000). | Analysis |  |  |
| RSK_R007 | The device SW communication protocol shall perform a Packet Validation. | Analysis |  |  |
| RSK_R008 | The device SW shall contain an interlock for collimator positioning. | Analysis |  |  |
| RSK_R009 | The device SW shall utilize a secure connection for the remote UI. | Analysis |  |  |
| RSK_R010 | The image processing algorithm shall be deterministic. | Analysis |  |  |
| RSK_R012 | The device SW shall ensure valid communication using a Heartbeat. | Analysis |  |  |
| RSK_R013 | The device SW shall utilize a hardware Watchdog. | Analysis |  |  |
| RSK_R014 | The device SW shall contain a Message Checksum. | Analysis |  |  |
| RSK_R015 | The device SW shall perform component authentication. | Analysis |  |  |
| RSK_R019 | The device SW shall contain integrity checks. | Analysis |  |  |
| RSK_R021 | The device shall support Dead Pixel Correction. | Analysis |  |  |
| RSK_R033 | The device shall contain an HV gen that requires an active FW initiative. | Analysis |  |  |
| RSK_R041 | The device shall use of proprietary protocols (only able to pair with MedAI pedals). | Analysis |  |  |
| RSK_R042 | The device shall support Local storage in the cassette. | Analysis |  |  |
| RSK_R050 | The device SW shall require user provisioning approval for package set-up. | Analysis |  |  |
| RSK_R056 | The device shall contain a Start-up Check. | Analysis |  |  |
| RSK_R057 | The device shall perform a Power On Self Test. | Analysis |  |  |
| RSK_R058 | The device shall contain a Filament preheat. | Analysis |  |  |
| RSK_R059 | The device shall contain a Warmup Routine For the Control Systems. | Analysis |  |  |
| RSK_R060 | The device shall contain a Warmup Routine For the HV System. | Analysis |  |  |
| RSK_R062 | The device shall set the current to lowest level upon start-up. | Analysis |  |  |
| RSK_R144 | The device shall contain bounds checking on all sensor data. | Analysis |  |  |
| RSK_R146 | The device shall verify the integrity of the safety-critical wireless systems within 1 second of every X ray exposure | Analysis |  |  |
| RSK_R147 | The device shall have a means of detecting wireless packet drop in all safety critical systems. | Analysis |  |  |
| RSK_R018 | The device SW shall be robust to noise. | Analysis/Test |  |  |
| RSK_R040 | The device shall automatically delete old images on max storage being reached. | Analysis/Test |  |  |
| RSK_R043 | The device shall notify the operator If the network is not connected. | Analysis/Test |  |  |
| RSK_R044 | The device shall contain a response verification. | Analysis/Test |  |  |
| RSK_R052 | The device shall enter a safe state in the event of an overexposure. | Analysis/Test |  |  |
| RSK_R055 | The radiation sequence shall terminate within 1 frame post trigger release. | Analysis/Test |  |  |
| RSK_R198 | The device shall intermittently verify collimator function. | Analysis/Test |  |  |
|  | Wireless Systems/Coexistence |  |  |  |
| RSK_R237 | The device shall contain Addressing to prevent connection to incorrect network(s) while pairing. | Analysis |  |  |
| RSK_R238 | The foot pedal remote triggering function shall incorporate channel hopping. | Analysis |  |  |
| RSK_R239 | The device shall contain DICOM ping to PACS upon data upload/export. | Analysis |  |  |
| RSK_R240 | The device foot pedal remote trigger shall contain Forward error correction. | Analysis |  |  |
| RSK_R241 | The device shall contain heartbeat on the data display and wake procedures. | Analysis |  |  |
| RSK_R242 | The device shall contain loopback (sub-GHz reply confirmation) upon pairing. | Analysis |  |  |
| RSK_R243 | The device shall allow for manual pairing between the emitter and cassette. | Analysis |  |  |
| RSK_R244 | The device shall integrate packet ID Checks upon pairing. | Analysis |  |  |
| RSK_R245 | The device shall require a pairing procedure for accessories. | Analysis |  |  |
| RSK_R246 | The device shall perform pairing checks. | Analysis |  |  |
| RSK_R247 | The device shall ping to platform upon data upload/export. | Analysis |  |  |
| RSK_R248 | The device shall contain a radio interlock check sequence (w/wifi) to prevent detector reset. | Analysis |  |  |
| RSK_R249 | The device shall perform RSSI checks on the foot pedal wireless connection. | Analysis |  |  |
| RSK_R250 | The device shall limit the max time on DDR studies when initiated by a remote trigger. | Analysis |  |  |
| RSK_R252 | The device shall contain Sub-GHz radio heartbeat to wake the emitter and cassette. | Analysis |  |  |
| RSK_R253 | The device shall perform a trigger interlock check on the foot pedal. | Analysis |  |  |
| RSK_R254 | The device shall perform a trigger interlock check sequence (w/wifi) on the detector. | Analysis |  |  |
| RSK_R255 | The device shall contain a visual indicator for pairing. | Analysis |  |  |
| RSK_R256 | The device shall contain a UI indicator for the wireless connection status. | Analysis |  |  |
| RSK_R257 | The device shall perform a wifi Speed check and display warning for low speeds. | Analysis |  |  |
|  | EE Systems |  |  |  |
| RSK_R022 | The device FW shall contain Beam Current Monitoring. | Analysis |  |  |
| RSK_R025 | The device FW shall contain a limited kV, tube current, and filament current. | Analysis |  |  |
| RSK_R023 | The device FW shall contain a controlled low kV interlock fault. | Analysis/Test |  |  |
| RSK_R024 | The device FW shall contain a limited filament current setpoint (FW to disallow filament current above specified threshold). | Analysis/Test |  |  |
| RSK_R026 | The device FW shall contain a low Beam Current Monitoring/fault. | Analysis/Test |  |  |
| RSK_R027 | The device FW shall contain a Power ceiling (set upper limit on power). | Analysis/Test |  |  |
| RSK_R028 | The device lasers shall contain a Firmware Lockout. | Analysis/Test |  |  |
| RSK_R029 | The device firmware shall limit exposure for each study. | Analysis/Test |  |  |
| RSK_R032 | The device shall contain an HV block FW control and monitor. | Analysis |  |  |
| PRD7.30 | The All-in-One-Computer shall have a usb-a port to connect a flash drive to transfer images; However the device will not allow transfer of files from the flash drive to the device. | Analysis |  |  |
| PRD6.3 | The critical safety circuits shall be designed to be fault tolerant and fail safe. | Analysis |  |  |
| RSK_R082 | The device shall maintain high voltage ONLY during x-ray emission. | Analysis |  |  |
| RSK_R095 | The device shall restrict HV to within the monoblock. | Analysis |  |  |
| RSK_R101 | The device shall isolate power input signals to the pcb connectors. | Analysis |  |  |
| PRD8.23 | The device shall allow a DDR exposure between 1 and 40 s from the previous exposure. | Analysis |  |  |
| PRD8.26 | The device shall limit the duty-cycle of serial radiographic mode to a maximum of 20 seconds of duration and proportional cooldown with a maximum of 40 seconds of cooldown. | Analysis |  |  |
|  | DEMO |  |  |  |
|  | Without Radiation - device on - UI - viewable |  |  |  |
| PRD7.11 | The device shall work with viewing hardware in the form of a touchscreen display/AIO or tablet wirelessly. | Demo | Confirm that the main UI is touchscreen and wireless. |  |
| PRD4.16 | The viewfinder shall be viewable from a tablet | Demo | Verified by above |  |
| PRD7.29 | The All-in-One-Computer shall have a usb-a port to connect an off-the-shelf keyboard and a mouse; as a backup to the touch screen | Demo | Connect a keyboard and mouse to the cassette. Confirm mouse and keyboard show on UI. |  |
| PRD7.31 | The All-in-One-Computer shall have a ethernet port for wired ethernet connectivity | Demo | Connect ethernet cable to ethernet ports. Ensure that the device is not wirelessly connected to the internet. Confirm internet connectivity. |  |
| PRD4.4 | The viewfinder shall display the optical image | Demo | Confirm that optical images are visible on UI. |  |
| PRD4.10 | The viewfinder shall display loading factors before taking an image | Demo | Confirm loading factors are displayed on UI. |  |
| PRD4.14 | The viewfinder shall display the non-active area differently than the active area | Demo | Confirm that the viewfinder displays the non-active area differently than the active area |  |
| PRD5.6 | The emitter shall display the state of the charging system | Demo | Confirm that the emitter displays the state of the charging system |  |
| PRD5.7 | The emitter charging dock shall display the state of the charging system | Demo | Confirm that the emitter charging dock displays the state of the charging system |  |
| PRD5.8 | The cassette shall display the state of the charging system | Demo | Confirm that the cassette displays the state of the charging system |  |
| PRD4.8 | The viewfinder shall overlay optical image, x-ray field, active area, and active area center on the same image | Demo | Confirm that the viewfinder overlays the optical image, x-ray field, active area, and active area center on the same image |  |
| PRD4.9 | The viewfinder may overlay other contextual information to help the user self correct by aligning the field size to the detector | Demo | Confirm that the viewfinder overlays other contextual information |  |
| PRD4.11 | The viewfinder shall provide positioning guidance in the form of angle and SID to help get the desired x-ray image | Demo | Confirm that the viewfinder shows the angle and SID when the emitter is over the cassette. |  |
| PRD4.12 | The viewfinder shall provide guidance towards aligning x-ray axis to cassette axis | Demo | Confirm that the viewfinder provides guidance towards aligning the x-ray axis to cassette axis |  |
| PRD9.1 | All data presented on the display UI shall have a unit of measure or label and conform to the international standards for displaying of units. | Demo | Confirm that all data have standard units or labels. |  |
| PRD9.2 | The display UI shall contain the manufacturer contact information. | Demo | Confirm that the display UI displays the manufacturer contact information. |  |
| PRD9.6 | The display UI should display a minimum of two of the most recent images. | Demo | Confirm that the display shows at least two of the most recent images. |  |
| PRD9.16 | The display UI should display the network connection status in the main window. | Demo | Confirm that the UI displays the network connection status in the main window. |  |
| PRD9.17 | The display UI should display the PACS connection status in the main window. | Demo | Confirm that the UI displays the PACS connection status in the main window. |  |
| PRD9.18 | The display UI should display the Platform connection status in the main window. | Demo | Confirm that the UI displays the platform connection status in the main window. |  |
| PRD9.19 | The display UI should display the Battery status in the main window. | Demo | Confirm that the display UI displays the Battery status in the main window. |  |
| PRD9.20 | The display UI should display the Pairing status in the main window. | Demo | Confirm that the display UI displays the Pairing status in the main window. |  |
| PRD9.32 | The display UI should indicate state of the device (e.g. Powered on, Powered off, Charging, Available for imaging, Emitting radiation, and Error State) | Demo | Allow the device to go idle, confirm UI shows "idle" state. Allow the device to sleep, confirm UI shows "sleep" state. Wake the device, confirm the UI shows "wake" state. |  |
| PRD10.15 | The emitter display shall display the viewfinder | Demo | Confirm that the emitter display contains the viewfinder. |  |
| PRD10.16 | The emitter display shall display the remaining battery life | Demo | Confirm that the emitter display contains the battery life |  |
| PRD10.17 | The emitter display shall display the pairing status | Demo | Confirm that the emitter display contains the pairing status |  |
| PRD8.9 | The system shall have a Power-On screen to indicate ON state to operator. | Demo | The system shall have a Power-On screen to indicate ON state to operator. |  |
| PRD8.10 | The device idle state shall be distinguished from active state. | Demo | The device idle state shall be distinguished from active state. |  |
| PRD2.19 | The device may allow operator to select collimation size from a series of preselected options | Demo |  |  |
| PRD8.25 | The device shall accept hyphens and spaces as part of name inputs | Demo |  |  |
|  | Without Radiation - device on - UI - actions |  |  |  |
| PRD7.39 | The power button should not take longer than 3 seconds to display response to push. | Demo | Press the power button on the emitter, confirm that the device reacts within 3 seconds to push. Repeat with cassette. |  |
| PRD10.12 | The device should provide haptic feedback when the trigger is activated. | Demo | Confirm that the trigger provides haptic feedback (ie. clicking) |  |
| PRD10.7 | The emitter keypad should provide haptic feedback when buttons are pressed | Demo | Confirm that the emitter keys provides haptic feedback (ie. clicking) |  |
| PRD9.21 | The display UI shall display the SID during use. | Demo | Holding the emitter over the cassette, confirm that the display UI displays the SID. |  |
| PRD9.5 | The display UI should allow the operator to view images without interacting with the UI. | Demo | Take an image. Confirm ability to view images immediately without interacting with the UI. |  |
| PRD7.10 | The device shall allow the operator to view images at the point of imaging. | Demo | Verified by above |  |
| PRD9.4 | The display UI shall allow the operator to create patient studies | Demo | Confirm ability to create a patient study |  |
| PRD9.7 | The display UI shall allow the operator to select and view acquired images. | Demo | Select image to view. Confirm ability to click non-main image to view in main image window. |  |
| PRD9.8 | The display UI shall allow the operator to independently manipulate the images. | Demo | Manipulate 2 separate images. Confirm ability to change images independently from each other. |  |
| PRD9.9 | The display UI shall allow the operator to zoom images. | Demo | Confirm ability to zoom in on an image. |  |
| PRD9.10 | The display UI shall allow the operator to rotate images; 360 degrees of rotation in 90 degree increments. | Demo | Confirm ability to rotate image in 90 degree increments up to 360 degrees. |  |
| PRD9.11 | The display UI should persist rotation adjustments | Demo | Chose an image and rotate i by 90 degrees, confirm that clicking away from that image and returning to the image does not revert the rotation change. |  |
| PRD9.12 | The display UI shall allow the operator to independently adjust the sharpness, contrast ratio and brightness of the radiographic images. | Demo | Confirm ability to adjust the sharpness, contrast ratio, and brightness of a radiographic image. |  |
| PRD8.32 | The system shall allow the user to adjust brightness, contrast, and sharpness of an image. | Demo | Confirm ability to adjust the sharpness, contrast, and brightness of a photographic image. |  |
| PRD9.13 | The display UI may allow the operator to invert the colors of the x-ray image (Black / White). | Demo | Confirm ability to invert colors on a radiographic image. |  |
| PRD9.14 | The display UI should allow the operator to crop an image before sending to PACS | Demo | Confirm ability to crop an image. |  |
| PRD9.15 | The display UI may allow for annotations. | Demo | Confirm ability to add annotations to images. |  |
| PRD9.29 | The display UI may provide feedback when a trigger press occurs while interlock is not met. | Demo | Shoot the emitter while pointed away from the cassette, confirm no image taken and feedback from the device. |  |
| PRD9.30 | The display UI shall include a reference point so that the operator understands where the emitter is positioned in reference to the detector and intended anatomy. | Demo | Confirm that the reference point from emitter's position is displayed. |  |
| PRD9.3 | The display UI shall allow the operator to shutdown the system. | Demo | Confirm ability to shutdown the device. |  |
| PRD7.22 | The device shall automatically reconnect to a known wifi network after inputting password the first time | Demo | Connect to wifi network. Turn off device. Confirm the wifi is connected when device is turned on. |  |
| RSK_R063 | The device UI shall display technique factors post-imaging. | Demo |  |  |
| RSK_R065 | The device shall contain a UI Button to Reset Image Adjustments. | Demo |  |  |
| RSK_R066 | The UI shall Display Error Message. | Demo |  |  |
| RSK_R067 | The UI shall display the image mode. | Demo |  |  |
| RSK_R069 | The device shall contain an Active Area indicator on the UI. | Demo |  |  |
| RSK_R070 | The device shall allow for multiple UI end points. | Demo |  |  |
| RSK_R071 | The device shall display technique factors on the UI. | Demo |  |  |
| RSK_R072 | The device shall display technique factors on the emitter UI. | Demo |  |  |
| RSK_R073 | The device shall display SID On the UI. | Demo |  |  |
| RSK_R074 | The device shall contain a Remote UI that may be unlocked when in AP mode. | Demo |  |  |
| RSK_R233 | The device UI shall contain a clear and unique indication for severe faults. | Demo/Analysis |  |  |
| RSK_R068 | The device UI shall Not Provide Access to PHI. | Demo/Test |  |  |
| RSK_R111 | The device shall contain a battery charge status indicator on the emitter and cassette UI. | Demo |  |  |
| RSK_R112 | The device shall contain a Low battery warning on the Display UI. | Demo |  |  |
| RSK_R076 | The device shall display Feedback when images are acquired. | Demo |  |  |
| RSK_R232 | The device shall display a warning in the event of poor wireless COMS quality. | Demo |  |  |
| PRD8.33 | The system shall allow the user to invert the x ray images for viewing as needed | Demo |  |  |
| PRD8.34 | The system should be able to add dimensions and measure lengths and angles to an image | Demo |  |  |
| PRD8.35 | The system should be able to add indicates for Left or Right or Bilateral anatomy to an image | Demo |  |  |
| PRD8.36 | The system should be able to add indicates for Upright or Supine to an image | Demo |  |  |
| PRD8.37 | The system should be able to add indicates for Expiration or Inhalation to an image | Demo |  |  |
| PRD8.38 | The system should be able to add indicates for AP, PA, Lateral, and Oblique to an image | Demo |  |  |
| PRD8.39 | The system should display the image exposure index (Exposure Index) after each image is taken | Demo |  |  |
| PRD8.31 | The system shall be able to add the loading factors (kVp, mAs) used on the image | Demo |  |  |
|  | Battery |  |  |  |
| RSK_R126 | The emitter shall support charging in use. | Demo |  |  |
| RSK_R127 | The device shall support intermittent charging (ie. holster). | Demo |  |  |
| RSK_R133 | The device shall disallow removal of the battery packs by the operator. | Demo |  |  |
| RSK_R036 | The device shall contain an alert for the operator to replace the battery. | Demo/Test |  |  |
| RSK_R258 | The device shal disallow the use of non-compatible charger inputs. | Demo |  |  |
|  | Without Radiation - device on - technique factors |  |  |  |
| PRD10.2 | The emitter keypad shall allow the operator to select between radiography, serial radiography, photography, and AiLARA modes. | Demo | Using the emitter keypad, Confirm that all modes (DDR, single shot, photography) are selectable. |  |
| RSK_R064 | The device shall only allow discrete technique choices. | Demo |  |  |
| PRD8.3 | The device shall contain different indicators for each mode | Demo | Confirm that each mode has a unique indicator on the UI. |  |
| PRD8.1 | The device shall allow the operator to switch between DDR, single shot, and photographic modes. | Demo | Verified by above |  |
| PRD2.6 | The device shall allow operator to set minimum SSD, minimum SID, DDR max, total dose per patient study, field size limits. | Demo | Confirm ability to set SID, SSD, DDR max time, dose limit, and field size. |  |
| PRD2.8 | The x-ray tube shall operate between 40 kV to 80 kV. | Demo | In single-shot mode, confirm that the power range is 40 kV to 80 kV. |  |
| PRD2.13 | The device shall be able to perform single exposure x-rays up to 80 kV max | Demo | Verified by above |  |
| PRD2.12 | The device shall be able to perform DDR up to 60 kV max | Demo | In single-shot mode, confirm that the power range is 40 kV to 60 kV. |  |
| PRD2.9 | The x-ray tube beam current shall operate between 1mA to 2mA. | Demo | In single-shot mode, confirm that the current range is 1 mA to 2 mA. |  |
| PRD2.10 | The x-ray exposure time in radiographic mode shall be 33ms, 66ms, or 99ms. | Demo | In single-shot mode, confirm that the time range is 33ms to 99ms. |  |
| PRD10.3 | The emitter keypad shall display and allow the operator to select the appropriate loading factors (kV and mAS) for each exposure. | Demo | Verified by above |  |
| PRD8.6 | The system may allow the loading factors to be adjusted on the display interface for surgical application (Surgical Setting) | Demo | Confirm surgical mode is selectable. |  |
|  | Without Radiation - device on |  |  |  |
| PRD1.8 | The device shall be moved from room to room within a facility by one person. | Demo | Take one shot (any). Pick up the emitter and cassette and move them 20 feet. Set up again and take a second shot. |  |
| PRD1.3 | The device shall perform photography. | Demo | Take one shot in Photography Mode. |  |
| PRD3.4 | The device shall only allow x-ray emission above the minimum SSD | Demo | Verified by above |  |
| PRD5.14 | The emitter shall be chargeable via wired power connection | Demo | Connect emitter to wired charger, confirm charging status. |  |
| PRD7.28 | The emitter shall have a usb-c for power input | Demo | Verified by above |  |
| PRD5.15 | The emitter shall be chargeable via inductive charging dock | Demo | Connect emitter to charging dock, confirm charging status. |  |
| PRD5.23 | The cassette shall be chargeable via wired power connection | Demo | Connect cassette to wired charger, confirm charging status. |  |
| PRD7.25 | The cassette shall have a micro-usb service port(s), that is covered with a plug and requires a tool to access | Demo | Connect USB-C from service port to service station. Ensure connection and ability to provide servicing. |  |
| PRD7.27 | The emitter  shall have a micro-usb service port(s), that is covered with a plug and requires a tool to access | Demo | Connect USB-C from service port to service station. Ensure connection and ability to provide servicing. |  |
| PRD7.18 | The emitter should have a laser(s) to project a crosshair pattern at the center of the x-ray field as a means to assist in positioning the x-ray field. | Demo | Confirm laser pattern is projected from emitter to cassette surface. |  |
| PRD11.18 | The operator shall be able to view the emitter and cassette indicator LEDs while the device is in use. | Demo | Confirm that the emitter and cassette LEDs are on and visible. |  |
| PRD7.41 | All system components shall have an indicator when ON. | Demo | Confirm cassette and emitter have an "ON" indicator. |  |
| PRD7.42 | The power button on the cassette should provide instantaneous feedback, indicating the state of the device. | Demo | While device is "OFF" press the power button on the cassette. Confirm cassette response to button press. Repeat with device in "ON" state. |  |
| RSK_R051 | The device shall have a clear indicator for "armed" state. | Demo |  |  |
| RSK_R075 | The device shall contain Status Indicators On the Emitter And Cassette. | Demo |  |  |
| PRD3.10 | The device shall display to the operator the status of the tracking system (eg Armed verse Disarmed) via indicator LEDs | Demo |  |  |
| RSK_R089 | The device shall contain class II cross lasers to indicate active area. | Demo |  |  |
| RSK_R001 | The device SW shall alert the operator to system actions. | Demo/Test |  |  |
| PRD7.43 | The power button LED should respond when the operator initiates the shut off procedure via the UI. | Demo | While device is "ON", shut off the device using the UI. Confirm response to shut-down initiation. |  |
| PRD8.2 | The device shall be able to set the technique automatically in AiLARA mode | Demo | In AiLARA Mode, with a phantom on the cassette, hold the emitter over the cassette. Confirm technique factors are set and visible before initiating an image capture. |  |
| PRD8.8 | The system should save images prior to shut down. | Demo | Shut down system. Upon start-up confirm the images persist. |  |
| PRD8.11 | The device shall allow users to upload x-rays images and image series to the PACs server, USB Drive, Patient, Provider, and Platform | Demo | Confirm ability to upload to PACs server, USB Drive, Provider, and Platform. Confirm that all uploads show image send confirmation. |  |
| PRD8.13 | The device should provide confirmation that the image study has been successfully submitted to PACS, MedAI Platform, or local storage (Drive) | Demo | Verified by above |  |
| PRD2.25 | The device may provide dead pixel reporting to monitor image quality degradation | Demo |  |  |
| RSK_R011 | The device SW may require user confirmation before deletion. | Demo |  |  |
| RSK_R016 | The device SW shall control user access. | Demo |  |  |
| RSK_R017 | The device SW shall require credentials for AP mode connection. | Demo |  |  |
| RSK_R020 | The device SW shall integrate orientation tagging. | Demo |  |  |
| RSK_R034 | The device shall contain an AiLARA mode. | Demo |  |  |
| RSK_R047 | The tracking interlock shall disallow x-ray emission. | Demo |  |  |
| RSK_R154 | The device shall contain a viewing method to align the anatomy and detector (viewfinder). | Demo |  |  |
|  | Without Radiation - test finger |  |  |  |
| RSK_R176 | The device enclosure shall prevent operator access to high temperature parts. | Demo |  |  |
| RSK_R177 | The device enclosure shall prevent operator access to conductive components / contacts. | Demo |  |  |
| RSK_R178 | The device enclosure shall prevent operator access to moving parts. | Demo |  |  |
|  | With Radiation |  |  |  |
| PRD9.31 | The display UI shall always default the display of the image in the reference orientation of the emitter | Demo | Take an image, confirm the the display of the image is in reference orientation to the emitter. |  |
| PRD7.13 | The emitter shall contain status indicators to show user when x-ray emission is happening and status of device interlock | Demo | Take a DDR, confirm that the device indicates interlock changes and radiation. |  |
| PRD3.3 | The device shall only allow x-ray emission within the specified SID range | Demo | In single-shot mode, attempt to take and image at XXcm (below min SSD) and XXcm (above max SSD). Take one image at XX cm (within SSD). Confirm that only the image within SSD range was taken, |  |
| PRD1.1 | The device shall perform radiography. | Demo | Verified by above |  |
| PRD1.2 | The device shall perform serial radiography. | Demo | In serial radiography mode, take a 5 second capture. |  |
| PRD9.25 | The display UI shall inform the operator of radiation emission. | Demo | During the 5 second capture, confirm radiation emission indicator is visible. |  |
| PRD9.22 | The display UI shall display the dose during use. | Demo | During the 5 second capture, Confirm dose is displayed during use. |  |
| PRD9.23 | The display UI should display available and remaining DDR time | Demo | During the 5 second capture, Confirm timer is displayed. |  |
| PRD8.22 | The device may inform the operator of total DDR time during exposure. | Demo | Verified by above |  |
| PRD10.9 | The emitter trigger(s) shall allow the operator to trigger an x-ray, serial x-ray, or photograph | Demo | Verified by above |  |
| PRD9.24 | The display UI should alert the user of necessary cooldown period procedure post DDR. | Demo | Once 5 second capture is complete, confirm cooldown timer is shown. |  |
| PRD5.1 | The device shall be primarily battery operated. | Demo | Confirm device can be used while not connected to MAINS. |  |
| PRD5.13 | The emitter shall support Essential Performance while wireless charging. | Demo | While both the emitter and cassette are connected to a charger, take a single-shot x-ray. Confirm image is displayed. |  |
| PRD5.21 | The cassette fully charged battery shall support continuous operation while charging | Demo | Verified by above |  |
| PRD7.9 | The device shall allow the operator to take and view images without external internet connectivity. | Demo | Disconnect from the internet. Confirm ability to view images. |  |
| RSK_R048 | The device shall implement exposure data tracking. | Demo |  |  |
| RSK_R049 | The device shall contain a marking placement (L/R tags). | Demo |  |  |
| RSK_R053 | The device shall contain a debounce on all user inputs (buttons, triggers, touch). | Demo |  |  |
| RSK_R054 | The device shall restrict the operator from shooting multiple x-rays In <1 sec intervals while in single shot mode. | Demo |  |  |
| RSK_R077 | The device shall emit one x-ray exposure per one trigger pull when in single-shot mode. | Demo |  |  |
|  | Special Cases - device control app |  |  |  |
| PRD19.1 | The Device Control App shall be compatible with Android, iOS and Linux devices |  |  |  |
| PRD19.2 | The Device Control App shall not trigger x-ray emission |  |  |  |
| PRD19.3 | The Device Control App shall show single and serial radiography images |  |  |  |
| PRD19.4 | The Device Control App shall allow the user to set device technique factors |  |  |  |
| PRD19.5 | The Device Control App shall allow the user to manually collimate to smaller area |  |  |  |
| PRD19.6 | The Device Control App shall show device status indicators |  |  |  |
| PRD19.7 | The Device Control App shall show device and peripheral connectivity |  |  |  |
| PRD19.8 | The Device Control App shall show device information |  |  |  |
| PRD19.9 | The Device Control App shall allow the user to set device configurations |  |  |  |
| PRD19.10 | The Device Control App shall allow the user to shut down device |  |  |  |
| PRD19.12 | The Device Control App shall show notifications and alerts |  |  |  |
| PRD19.13 | The Device Control App shall allow the user to adjust image parameters |  |  |  |
|  | Special Cases - drape |  |  |  |
| PRD3.5 | The tracking system should function nominally when 50% of the LEDs are completely obstructed. | Demo | Cover 50% of cassette LEDs. Confirm ability to take an x-ray image. |  |
| PRD3.7 | The tracking system should function accurately with up to 5 standard medical drapes over the cassette. | Demo | Bunch drape over the cassette. Confirm ability to take an x-ray image. |  |
| PRD3.6 | The cassette LEDs shall be visible though a standard medical drape. | Demo | With drape over the cassette, confirm that LEDs are visible. |  |
| PRD13.1 | The device shall support use with standard medical drapes and clear sterile bags | Demo | Apply clear sterile cover over the cassette and emitter. Confirm ability to take an x-ray image. |  |
| PRD4.1 | The device shall allow the operator to view the detector active area while the cassette and emitter are draped prior to activating emission. | Demo | With drape over the cassette, confirm ability to view the active area. |  |
|  | Special Cases - Other |  |  |  |
| PRD3.8 | The tracking system shall function correctly outdoors in direct sunlight | Demo | Take the device outside and confirm that UI is visible. Take one shot (in any mode) and confirm image is taken properly. |  |
| PRD3.9 | The tracking system shall operate under high ambient light conditions within a bright surgical environment | Demo | Verified by above - Assuming surgical environment is XXX lumens |  |
| PRD5.2 | The device shall have less than 5 minutes of time when the device is unusable if the battery dies. | Demo | Using a device with an empty battery, plug into a charger. Attempt to turn on device and take an image every minute for 5 minutes. Confirm that the device is able to take an image within 5 minutes of plugging in to charger. |  |
| PRD2.23 | The device shall have a detector with an active area of 22cm x 22cm (9"x9") | Demo | Take one shot (Single Radiography Mode) on each detector |  |
| PRD8.5 | The system may allow admin operator to select and delete an image or images. | Demo | Attempt to delete images in operator mode, confirm that images persist. Change to admin mode, confirm the ability to delete images. |  |
| RSK_R102 | The device shall contain a HW interlock to prevent x-rays when plugged into an external power source. | Demo |  |  |
|  | Accessories |  |  |  |
| PRD7.8 | The device shall integrate with all future specified accessories, including Cart, Emitter Surgical Arm, and MedAI Clear Sterile Bags | Demo | Confirm pairing to accessories listed in IFU. (list accessories here) |  |
| PRD8.15 | The system shall be able to pair a cassette, emitter, foot pedal, and tablet together. | Demo | Verified by above |  |
| PRD12.9 | The device shall be able to be used while in the case with the lid open | Demo | Confirm ability to take an x-ray image while cassette is in the case. |  |
| RSK_R045 | The device shall exit idle state upon detection of emitter or foot pedal activity. | Demo |  |  |
| PRD17.1 | The device shall allow operator to turn off "handheld trigger" operation (foot pedal only) per state and local regulations | Demo | Confirm the ability to turn off "handheld trigger" operation (foot pedal only). |  |
| PRD1.5 | The device shall support remote hands free triggering of x-rays and DDR via a foot pedal | Demo | While paired with the foot pedal, press and hold trigger pedal. Confirm only one x-ray image is taken via foot-pedal 7 ft away from the device. |  |
| PRD17.2 | The foot pedal shall be wireless and work up to 7 feet away from device | Demo | Verified by above |  |
| PRD17.5 | The foot pedal B shall initiate DDR on the downpress and shall stop the exposure upon release | Demo | Verified by above |  |
| PRD17.3 | The device shall support the use of a two button foot pedal | Demo | While paired with the foot pedal, confirm ability to change mode. |  |
| PRD17.4 | The foot pedal A shall initiate a single x-ray exposure upon pressing | Demo | Verified by above |  |
| PRD1.4 | The device shall be hand held as its primary use case | Demo | Verified by above |  |
| PRD18.5 | The pucks size shall correspond to the size of the emitter. | Demo | Confirm the puck aligns properly to emitter. |  |
| RSK_R199 | The device shall allow for use of P00 Pucks for collimation. | Demo |  |  |
| PRD18.3 | The Detachable Part(s) shall be aligned and attached to the emitter | Demo | Verified by above |  |
| PRD18.4 | Puck label color shall match in color to the UI field. | Demo | Confirm puck color matches UI displayed color for puck. |  |
| PRD18.6 | The SID recommendation should show the color of the puck. | Demo | Confirm UI puck recommendation matches the color of the recommended puck label. |  |
| PRD18.1 | The detachment mechanism may incorporate operator positive feedback, either auditory or sensory. | Demo | When removing the puck, confirm feedback from device. |  |
| PRD7.3 | The emitter shall be capable of being supported and/or suspended in a non-permanent manner. | Demo | Place emitter on stand or cart. Confirm stability. |  |
| RSK_R164 | The device cart/accessories shall allow for non-motorized motions. | Demo |  |  |
| RSK_R165 | The device cart/accessories shall not allow motion in the vertical axis - no motion due to gravity. | Demo |  |  |
| RSK_R166 | The device cart/accessories shall have locking wheels. | Demo |  |  |
| RSK_R167 | The device shall contain a gas spring that has a force balanced to the cassette platform weight. | Demo |  |  |
| RSK_R168 | The cassette gas spring maximum retraction velocity shall be less than 5 m/s. | Demo |  |  |
|  | Platform |  |  |  |
| RSK_R037 | The device shall automatically upload images/studies to the platform. | Demo |  |  |
| RSK_R038 | The device may allow pairing by platform. | Demo |  |  |
| RSK_R039 | The device shall allow provisioning by the platform. | Demo |  |  |
|  | Inspection |  |  |  |
|  | Electrical Drawings/Spec Sheet |  |  |  |
| RSK_R096 | The device shall contain a redundant thermostat. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R097 | The device shall contain a redundant thermistor. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R098 | The device shall contain a Bleed-off circuit (for capacitive energies). | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R080 | The device shall contain a Power Supply rated to IEC 60950-1. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R086 | The device shall utilize EN320 plugs. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R087 | The device shall utilize IEC 60320 plugs. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R088 | The device shall only use an IEC-60601 rated external isolated power supply. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R079 | The device shall contain an AC/DC brick with surge protection specifications. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R081 | The device shall contain isolation from Mains - Overvoltage Cat 2 Power Supply. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R085 | The device shall contain current limiting devices. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R099 | The device shall contain an HV Transient Suppression on critical circuits. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R100 | The device charger shall comply to USB-PD specifications set by USB-IF. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R108 | The x-ray tube assy shall contain a temperature monitor on the anode. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R124 | The battery charger shall be integrated into the device. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R230 | The HV insulation shall integrate redundant safety methods (dielectric strength, floating loop, disable operation while plugged in, enclosure as a layer of insulation, arc monitoring - monoblock). | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.17 | The emitter shall have a Time of Flight (ToF) sensor for range finding capable of resolving 1 mm in depth within the specified SSD Range | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Electrical Drawing/Spec Sheet - battery |  |  |  |
| PRD5.3 | The emitter shall contain a rechargeable internal battery pack with integrated BMS | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD5.4 | The cassette shall contain a rechargeable internal battery pack with integrated BMS | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD5.9 | The device battery pack shall have a rated capacity in order to allow for air transit. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD5.26 | The device shall include coin cell battery to maintain RTC | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R125 | The device shall contain Rated battery chargers. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R131 | The device battery shall contain a Thermal fuse or CID. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R132 | Battery charging circuits shall be designed to accept 5V - 20V charging circuit power input. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R145 | The device shall contain a Temperature monitor FW (safe-state) on the battery compartment. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R234 | The device enclosure shall have a flame rating of V2 or better | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD5.16 | The charging power supplies shall contain ISO 60320 female plug to adapt to US and international plugs/outlets | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD5.17 | The charging power supplies shall be compatable with input voltage and frequency ranges 100-240 V and 50-60 Hz | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Electrical Drawing/Spec Sheet - monoblock/x-ray tube |  |  |  |
| PRD2.1 | The device shall have a monoblock that integrates the HVPS and x-ray tube. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.7 | The x-ray tube focal spot size shall be less than 100 um. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.2 | The x-ray tube assembly shall be shielded. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.14 | The device shall have a primary fixed collimation stage to confine the x-ray field coming out of the tube | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.15 | The primary fixed collimation shall collimate to 2 deg less than specified x-ray tube (43 deg.) | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R104 | The device shall contain a ceramic tube. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R109 | The device shall not contain a rotating anode. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R110 | The tube assembly shall contain a heat sink. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R157 | The x-ray tube assembly shall be composed of rated material of sufficient dielectric constant and thickness. | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - enclosure |  |  |  |
| RSK_R155 | Components and materials used in the device shall be derated to meet safety limits for the lifetime of the device per device requirements. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R169 | The device enclosure shall be composed of an impact and heat resistant plastic. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R172 | The device enclosure shall be composed of a non-chemically reactive plastic. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R171 | The device enclosure shall use low attenuating materials. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R174 | The device enclosure shall contain No openings in the shell/cover >1mm in diameter. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD13.12 | Any gap, that is not filled, shall be wider than 1 mm and have an aspect ratio of 3:1 in order to be cleaned | Inspection/Analysis | Refer to DRAWING/SPEC SHEET |  |
| RSK_R185 | The device enclosure shall have a min thickness of 1.4mm. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R186 | The device enclosure shall be sealed using a gasket or elastomeric seal. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R187 | The device enclosure shall be positioned at +- 0.25 degrees from the x-ray assembly. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R188 | The device enclosure surface shall allow appropriate clearance to facilitate airflow to and around circulation fans. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R189 | The device enclosure window piece shall not contain holes. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R197 | The device collimator shall be encased within the enclosure. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R182 | The device enclosure shall reduce falling and tipping risk by providing a surface with a high friction material. | Inspection | Refer to DRAWING/SPEC SHEETAlso verified via tip test per 60601-1 |  |
| RSK_R183 | The device shall contain flat resting surfaces. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R184 | The device shall contain a geometry to prevent tipping. | Inspection | Refer to DRAWING/SPEC SHEETAlso verified via tip test per 60601-1 |  |
| RSK_R149 | The device shall contain a pressure valve. | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - heat |  |  |  |
| RSK_R158 | The device shall contain a low/no flow fans. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R159 | The device fans shall be internally caged and shrouded. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R229 | The device internals shall incorporate thermal dissipation features (ie heat sink. heat pipes, fans vents, etc.). | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - wire/cables |  |  |  |
| RSK_R156 | Electrical connectors shall be fixed or strain relieved to prevent breakage or stress on connectors or other components. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R190 | The device shall integrate wire routing to prevent mechanical damage to the wires. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R193 | The device shall contain internal anchoring/tie-wraps of wires. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R194 | The device shall contain ties around conductor bundles at the connector lead in. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R195 | The device shall contain redundant tie points for wires. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.1 | The device shall not contain cables between the emitter and cassette. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.2 | The emitter and cassette shall not be permanently attached. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R191 | The external cables shall be strain relieved. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R192 | The external cables shall be shielded. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R196 | The device shall contain locking connectors on all internal harnesses and external cables | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - x-ray tube/alignment |  |  |  |
| RSK_R175 | The device enclosure shall align the tube within 2 degrees of the tracking camera focal axis. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.16 | The device shall have an automatic collimator to confine the x-ray field. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.21 | The device shall have an x-ray filter that is constructed of 6061 Aluminum. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.22 | The device shall utilize a digital flat field x-ray detector. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD2.24 | The detector shall contain shielding or have shielding behind the detector | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - viewfinder |  |  |  |
| PRD4.13 | The viewfinder shall update display at 30 fps | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - cassette |  |  |  |
| PRD14.5 | The cassette shall be able to be assembled entirely within one half of the shell. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.26 | The cassette shall have two interchangeable usb-c ports for power input and to connect accessories (eg. keyboard, mouse, display etc.) | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R150 | The cassette shall contain a desiccant. | Inspection |  |  |
|  | Mechanical Drawing/Spec Sheet - emitter |  |  |  |
| PRD14.6 | The emitter shall be able to be assembled entirely within one half of the shell. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.16 | The emitter shall have an IR optimized camera/sensor for IR Tracking System | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.14 | The emitter shall have an optical camera with autofocus for taking images with a resolution of 0.5mm at less than 25 cm separation. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.15 | The emitter shall have an optical camera for the Viewfinder display | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD11.8 | The emitter center of gravity shall be within 2 inches from the grip | Inspection/Analysis | Refer to DRAWING/SPEC SHEET |  |
|  | The emitter shall have a Time of Flight (ToF) sensor for range finding | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.19 | The lasers output power on the emitter shall be between 0.4 and 1.0 mW | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD10.13 | The emitter shall contain an LCD or OLED color display as the emitter display | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD10.14 | The emitter display shall have a minimum of 1,000 Nits | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R162 | The device shall not contain focusing optics for lasers. | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - Case |  |  |  |
| PRD12.2 | The case may be sized to be able to be carried on a plane (per FAA 22" x 14" x 9") | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD12.5 | The case shall have a handle and wheels to be transported by a single operator. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD12.6 | The case shall have a telescoping handle. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD12.7 | The case shall incorporate foam or other shock absorbing padding. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R153 | The device shall be primarily packaged in a hard shell case with closed cell internal foam or similar style case. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD12.11 | The case shall come packaged with desiccant to mitigate unacceptable levels of relative humidity. | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Mechanical Drawing/Spec Sheet - Other |  |  |  |
| PRD1.6 | The device shall be composed of two primary components the emitter and the cassette | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R202 | The device shall contain a Magnet Redundancies. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD7.37 | The power button shall be at least 10 mm in diameter. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R163 | The device shall not implement parts that could contribute to sudden expulsion of parts (ie. vacuum display, mechanical spring, gas pressure cylinder). | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R160 | The device shall have screw redundancies. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R161 | The device shall contain a redundant monitor. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R046 | The device shall contain a positioning system. | Inspection | ??? |  |
|  | Labeling/IFU |  |  |  |
| RSK_R226 | The device shall be provided with a physical copy of the IFU. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R227 | The device shall contain an IFU & eIFU that are identical in format and information. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD1.10 | The device shall include Accompanying Documents (IFU) with packaging | Inspection | Refer to DRAWING/SPEC SHEET |  |
| PRD11.16 | The device shall conform to the product Visual Brand Language (VBL) guidelines | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R204 | The device shall contain labeling for 'Do Not Step' | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R205 | The device shall contain labeling to Call Out Bottom of cassette. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R206 | The device shall contain labeling to Call Out "this side up" on the cassette. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R207 | The device shall contain labeling to Indicate "avoid spill". | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R208 | The device shall contain labelling on all components of the system. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R209 | The device shall contain labeling At Connector Junctions. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R210 | The device shall contain Visual Cues For Proper Connector Insertion. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R211 | The device shall contain high wear labels. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R212 | The device labeling shall require no more than a 7th grade reading level for use. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R213 | The device shall contain labeling with visual indicators/symbols. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R214 | The device shall contain emitter labeling. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R216 | The device shall contain a Label to Consult the IFU. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R217 | The device shall contain a Label - Do not stare into laser. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R218 | The device shall contain a Label - Do not trash. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R219 | The device shall contain a Label - do not use if enclosure is damaged. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R220 | The device shall contain a Label - fragile. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R221 | The device shall contain a Label - General warning - use as directed. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R222 | The device shall contain a Label - max and min temperature. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R223 | The device shall contain a Label - Rx Only. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R224 | The device shall contain a Label - X ray High voltage. | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Viewable - VVPR |  |  |  |
| PRD5.18 | The emitter charging dock shall be able to hold the emitter and align the charging coils | Inspection | Place emitter in charging dock. Confirm charging coils are aligned. |  |
| PRD7.6 | The cassette shall have marking to show the active area outline. | Inspection | Confirm active area outline on cassette. |  |
| PRD7.7 | The cassette shall have marking to show the top of the output image | Inspection | Confirm "top of image" indication on the cassette. |  |
| PRD7.32 | All device connection ports should be clearly labeled and recognisable to indicate connection points. | Inspection | Confirm labeling on all connection points on system contain symbols and wording. |  |
| PRD7.33 | All device connection ports should be labeled in both written and symbolic form. | Inspection | Verified by above |  |
| PRD10.1 | The emitter keypad shall contain 3 tactile buttons (up, down, mode). | Inspection | Confirm emitter keypad contains 3 tactile buttons (up, down, mode). |  |
| PRD10.8 | The emitter shall contain two trigger(s) | Inspection | Confirm trigger on emitter. |  |
| PRD11.9 | The cassette patient contacting surface shall be smooth. | Inspection | Confirm cassette surface is smooth. |  |
| PRD13.10 | The device shall have a smooth outer shell. | Inspection | Confirm emitter surface is smooth. |  |
| RSK_R179 | The device shall utilize covers and guards when necessary to prevent access to electrical, thermal, and moving components | Inspection | Confirm inability to access interior of device without the use of tools. |  |
| RSK_R201 | The device shall contain Posts to Prevent Pucks From Rotating. | Inspection | Apply puck to emitter. Confirm that that the emitter does not rotate freely while attached to the emitter. |  |
|  | Case/Overshipper |  |  |  |
| PRD12.1 | The device shall be able to be packaged in a hard shell case | Inspection | Confirm case is hard-shelled. |  |
| PRD12.8 | The case should provide compartments to hold all the fixed and detachable components of the device. | Inspection | Confirm all components of device have compartments and fit into case, |  |
| PRD12.10 | The case shall provide a means to mount the tablet on the inside lid of the case | Inspection | Mount tablet onto lid of case. Confirm stability. |  |
| PRD12.12 | The case shall be able to be packaged in a box. | Inspection | Confirm ability to place case in overshipper/box. |  |
| PRD12.13 | The box shall provide cutouts to be used as handles (or access case handles) | Inspection | Confirm overshipper/box contains handles. |  |
|  | Sterile Cover |  |  |  |
| PRD13.2 | The emitter shall interfaces to attach custom clear sterile cover(s) capable of being flush with optics/HMI components | Inspection | Place cover over the emitter. Confirm cover is flush with optics and HMI components. |  |
| PRD13.3 | The cassette shall interfaces to attach custom clear sterile cover(s) capable of being flush with IR LED components | Inspection | Place cover over the cassette. Confirm cover is flush with IR LEDs |  |
|  | Reference a MEMO |  |  |  |
| PRD13.5 | The cleaning procedure shall include standard materials and techniques. | Inspection - MEMO | Refer to MEMO |  |
| PRD14.1 | All critical x-ray system components shall be sourced from countries that do not pose as adversaries to the US, per EAR policy. | Inspection - MEMO | Refer to MEMO |  |
| PRD14.2 | The manufacturing processes shall be designed to support the production of 1,000 units a year (~20 units a week). | Inspection - MEMO | Refer to MEMO |  |
| PRD14.3 | The device shall be designed to allow for in-house pilot production of up to 100 units before transfer to CM. | Inspection - MEMO | Refer to MEMO |  |
| PRD14.4 | The device shall be designed to operate within the CM fabrication and manufacturing constraints. | Inspection - MEMO | Refer to MEMO |  |
| PRD14.7 | The device shall be serviceable by the manufacturer and approved third party services. | Inspection - MEMO | Refer to MEMO |  |
| PRD14.8 | The emitter shall be serviceable by the replacement of the monoblock | Inspection - MEMO | Refer to MEMO |  |
| PRD14.9 | The cassette shall be serviceable by the replacement of the detector | Inspection - MEMO | Refer to MEMO |  |
| PRD14.10 | The cassette shall be serviceable by the replacement of the battery pack without separating the enclosure halves. | Inspection - MEMO | Refer to MEMO |  |
| PRD15.1 | The device shall allow for transportation by air freight (cargo of plane) | Inspection - MEMO | Refer to MEMO |  |
| PRD15.4 | The device shall be stored in an ambient temperature of -10C +55C. | Inspection - MEMO | Refer to MEMO |  |
| PRD15.5 | The device shall be stored in a relative humidity of (non-condensing) 20-80%. | Inspection - MEMO | Refer to MEMO |  |
| PRD16.1 | The COGS shall be no more than $25,000 per device when manufactured in volume | Inspection - MEMO | Refer to MEMO |  |
| PRD16.4 | The device shall be able to maintain RTC for the length of the service life. | Inspection - MEMO | Refer to MEMO |  |
| RSK_R228 | MedAI shall create a MEMO that details the residual + scatter radiation information for the emitter. | Inspection - MEMO | Refer to MEMO |  |
| RSK_R225 | A new physical copy of the IFU shall be provided upon request. | Inspection - MEMO | Refer to MEMO |  |
| RSK_R090 | The IR LEDs intensity shall be below the injury threshold. | Inspection | Refer to MEMO |  |
| RSK_R103 | The device shall mechanically protect the tube. | Inspection | Refer to MEMO |  |
| RSK_R235 | The emitter shall have set maximum temperature and e-fields limits to prevent tube potting degradation. | Inspection | Refer to MEMO |  |
|  | Accessories |  |  |  |
| PRD18.10 | The device should have components that are large enough to easy to see if they are to be removed from the device. | Inspection | Confirm the detachable components are visible  from 5 ft away. |  |
| PRD18.11 | All detachable components should be recognisable as part of the device | Inspection | Confirm labeling on the detachable parts matches the system labeling. |  |
| RSK_R200 | The device may contain Color Coded Pucks. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R203 | The device shall contain labeling Directly On Pucks. | Inspection | Refer to DRAWING/SPEC SHEET |  |
| RSK_R215 | The device shall contain Foot Pedal labeling. | Inspection | Refer to DRAWING/SPEC SHEET |  |
|  | Test |  |  |  |
|  | Usability |  |  |  |
| PRD1.9 | The device shall be able to be packed, setup, and repacked without the use of a tool. | Test | To be observed during Formative use studies |  |
| PRD1.7 | The device shall allow operation by a lay person with minimal training. | Test | To be observed during Formative use studies |  |
| PRD10.4 | The emitter keypad shall allow for the operator to press an individual button without accidentally pressing a second button. | Test | To be observed during Formative use studies |  |
| PRD11.1 | The emitter should be comfortable to hold and move the emitter with all degrees of freedom in usable range during use | Test | To be observed during Formative use studies |  |
| PRD11.3 | The emitter shall be usable as intended with a left or right hand. | Test | To be observed during Formative use studies |  |
| PRD11.4 | The emitter shall be useable with one hand. | Test | To be observed during Formative use studies |  |
| PRD11.5 | The emitter may be useable with two hands (primary + support hand). | Test | To be observed during Formative use studies |  |
| PRD11.6 | The emitter shall allow the operator to simultaneously hold the emitter in a downward position and interact with the keypad buttons (via thumb) with one hand. | Test | To be observed during Formative use studies |  |
| PRD11.7 | The emitter should be usable in the forward and downward directions. | Test | To be observed during Formative use studies |  |
| PRD11.17 | The device touch points should be distinct and apparent to the operator (ie. separated via color or shape). | Test | To be observed during Formative use studies |  |
| PRD11.19 | The operator shall be able to comprehend information signals from the indicator LEDs while the device is in use. | Test | To be observed during Formative use studies |  |
| PRD13.4 | The operator shall be able to clean all commonly touched surfaces without disassembling the device. | Test | To be observed during Formative use studies |  |
| PRD10.5 | The emitter keypad buttons shall be controlled by the operator's thumb while holding the emitter. | Test/Demo | To be observed during Formative use studies |  |
| PRD10.6 | All emitter keypad buttons shall be accessible to the operator while holding the device with one hand. | Test/Demo | To be observed during Formative use studies |  |
| PRD10.10 | The operator shall be able to actuate the emitter trigger with one finger. | Test/Demo | To be observed during Formative use studies |  |
| PRD10.11 | The emitter shall allow the operator to simultaneously hold the emitter and actuate the trigger with one hand. | Test/Demo | To be observed during Formative use studies |  |
| PRD11.2 | The device shall be operable (trigger radiation and interact with emitter HMI)  with all combinations of soiled/not soiled and single-gloved/double-gloved hands (in surgical environment). | Test/Demo | To be observed during Formative use studies |  |
|  | Collimator |  |  |  |
| PRD2.17 | The automatic collimator shall be able to adjust aperture size and rotation to line up with the detector at any specified SID | Test |  |  |
| PRD2.18 | The device shall be able to be manually collimated down to 13x13cm at smallest SID, in discrete 1 cm increments | Test |  |  |
|  | Radiation + tracking |  |  |  |
| PRD2.4 | The device shall operate within a source-to-detector distance between 20cm and 45cm. | Test |  |  |
| PRD2.5 | The device shall operate within a source-to-skin distance between 20cm and 100cm. | Test |  |  |
| PRD2.20 | The edge of the automatic collimator leaves shall be designed to not create artifacts on x-ray image | Test |  |  |
| PRD3.1 | The positioning system shall compute the Source to Detector distance (SID). | Test |  |  |
| PRD3.2 | The device shall compute the Source to Skin distance (SSD). | Test |  |  |
| RSK_R105 | The device shall contain tube shielding/potting with an appropriate lead equivalent thickness to mitigate leakage. | Test |  |  |
| PRD8.21 | The device should self-terminate a DDR if the center point is moved outside the active area. | Test/Demo |  |  |
|  | Viewfinder Calculations |  |  |  |
| PRD4.5 | The viewfinder shall calculate and display the collimated x-ray field | Test |  |  |
| PRD4.6 | The viewfinder shall calculate and display the active area of the detector | Test |  |  |
| PRD4.7 | The viewfinder shall calculate and display the center of active area | Test |  |  |
| PRD4.15 | The viewfinder shall overlay the calculated active area with the actual active area within 4% | Test |  |  |
|  | Weight |  |  |  |
| PRD11.11 | The emitter shall weigh less than 6.5 lbs. and be distributed in a way that is ergonomic when held by hand. | Test |  | Create and write a VVPR procedure to verify the following requirements from the PRD and Risk mitigations: |
| PRD11.12 | The cassette shall weigh less than 14 lbs. | Test |  | Please copy paste the aforementioned table into the VVPR. This VVPR will be used to test the ERP for P01 to assist in finalizing the design for DCO.Once created, please link the VVPR to this ticket. |
| PRD11.13 | The device weight without packaging shall be less than or equal to 22 lbs. | Test |  | Write VVPR for PRD/RSK Requirement Testing - |
| PRD11.14 | The packaging shall weight less than or equal to 18 lbs. | Test |  |  |
| PRD11.15 | The device, including packaging, shall weight less than 40 lbs. | Test |  |  |
|  | GPS/Wireless Functions |  |  |  |
| PRD7.20 | The device should allow for cellular connectivity (eg 5G/4G/LTE) in the event the device is deployed in an area without network access | Test |  |  |
| PRD7.21 | The emitter and cassette should contain GPS tracking capabilities for asset tracking | Test |  |  |
| PRD7.23 | The emitter and cassette shall allow for WiFi connectivity with operating freq of 2.4/5/6 GHz for network access | Test |  |  |
| PRD7.24 | The device shall serve as a private WiFi Access Point for remote access. | Test |  |  |
|  | HV Interlocks and Fail Safes |  |  |  |
| RSK_R134 | The device HW shall open the HV interlock in the event of a filament current fault. | Test |  |  |
| RSK_R135 | The device HW shall open the HV interlock in the event of a high kV fault. | Test |  |  |
| RSK_R136 | The device HW open the HV interlock in the event of a x-ray exposure time (200 ms) interlock fault. | Test |  |  |
| RSK_R137 | The device shall open the HV interlock in the event of a fault. | Test |  |  |
| RSK_R138 | The device HW shall open the HV interlock in the event of a tube current fault. | Test |  |  |
| RSK_R139 | The device HW shall open the HV interlock in the event of an unexpected tube current fault. | Test |  |  |
| RSK_R078 | The device shall default to a safe state upon a fatal failure. | Test |  |  |
| RSK_R083 | The device shall fail safe when unexpected arcing occurs. | Test |  |  |
| RSK_R084 | The device shall enter a safe state in event of undercurrent. | Test |  |  |
| RSK_R092 | The device shall enter safe state when temperature threshold is exceeded | Test |  |  |
| RSK_R142 | The device shall monitor beam current and enter a safe state if the beam current is unexpected, too high, or too low | Test |  |  |
| RSK_R143 | The device shall Monitor the Internal Temperature(s) and fail safe in the event of under or over temp. | Test |  |  |
| RSK_R148 | The device shall enter safe state (or warn the operator) in the event of under or over humidity | Test |  |  |
| RSK_R093 | The device shall automatically shut down when high temperature threshold is exceeded. | Test |  |  |
| RSK_R140 | The device shall contain a temperature controlled power cut out in the event of overheat. | Test |  |  |
|  | Battery |  |  |  |
| RSK_R128 | The device shall contain a Fast charge feature. | Test |  |  |
| RSK_R129 | The device shall allow for throttled charging. | Test |  |  |
| RSK_R113 | The device BMS shall monitor and predicts battery life. | Test |  |  |
| RSK_R114 | The device BMS shall contain under-voltage protection. | Test |  |  |
| RSK_R115 | The device BMS shall contain Cell temp monitoring for under and over-temp protection. | Test |  |  |
| RSK_R116 | The device BMS shall contain an Open Circuit Protection. | Test |  |  |
| RSK_R117 | The device BMS shall contain Over-current protection. | Test |  |  |
| RSK_R118 | The device BMS shall contain Over-voltage protection. | Test |  |  |
| RSK_R119 | The device BMS shall contain an Overcharge protection circuit. | Test |  |  |
| RSK_R120 | The device BMS shall contain an Overcurrent draw protection. | Test |  |  |
| RSK_R121 | The device BMS shall monitor the battery Cell health. | Test |  |  |
| RSK_R123 | The device BMS shall contain cell-balancing. | Test |  |  |
| PRD5.12 | The emitter fully charged battery shall support a full day of operation with intermittent charging for nominal use case | Test/Demo |  |  |
| PRD5.10 | The device shall allow for charging while in surgery while maintaining sterility. | Test/Demo |  |  |
| PRD5.19 | The emitter charging system shall support throttled charging speeds | Test/Demo |  |  |
| PRD5.25 | The cassette charging system shall support throttled charging speeds | Test/Demo |  |  |
|  | EMC/EMI Testing |  |  |  |
| RSK_R094 | The device shall remove/reduce Concentrated E-Fields. | Test |  |  |
| RSK_R180 | The device enclosure shall be EMI shielded. | Test |  |  |
| RSK_R181 | The x-ray tube assembly shall be EMI shielded. | Test |  |  |
|  | Cleaning |  |  |  |
| PRD7.34 | All device labels shall not peel after repeated cleaning. | Test |  |  |
| PRD13.6 | The device shall be capable of withstanding being cleaned by isopropyl alcohol, cavicide (quaternary ammonium), bleach (hypochlorites), hydrogen peroxide, and water without reduction in performance | Test |  |  |
| PRD13.7 | The device shall be able to be cleaned without the need for a brush | Test |  |  |
| PRD13.8 | The device shall be able to be cleaned in less than 5 minutes. | Test |  |  |
| PRD13.9 | The disinfection time with Cavicide should be 3 minutes or less | Test |  |  |
| PRD13.11 | Any gaps shall be filled with sealant; any filler shall survive cleaning validation | Test |  |  |
|  | Force + Timing |  |  |  |
| PRD7.38 | The power button shall require between 1.7 N and 3 N of force to actuate. | Test |  |  |
| PRD5.11 | The emitter fully charged battery shall support 90 minutes of operation without intermittent charging for nominal use case | Test |  |  |
| PRD5.20 | The cassette fully charged battery shall support 120 minutes of operation without charging for nominal use case | Test |  |  |
| PRD7.40 | The device shall be ready to operate within 30 seconds from the time the power button is turned on under nominal conditions. | Test |  |  |
| PRD7.44 | The device shall enter an idle state when the device is not utilized for thirty (30) seconds. | Test |  |  |
| PRD7.45 | The device shall exit an idle state within two (2) seconds. | Test |  |  |
| PRD7.46 | The device shall enter an sleep state when the device is not utilized for one (1) minute. | Test |  |  |
| PRD7.47 | The device shall exit an sleep state within (5) seconds. | Test |  |  |
| PRD8.7 | The system shall save all image types within 10 seconds of the operator taking the image. | Test |  |  |
| PRD8.14 | The cassette shall be able to send/stream a image(s) to display hardware within 1 second | Test |  |  |
| PRD8.20 | A DDR cycle shall have a lag of no greater than 200 ms from the real time of the scan to the display of that frame on the UI. | Test |  |  |
| PRD7.36 | The device shall be available for use in less than 45 seconds of initiating power on | Test |  |  |
|  | Image Study |  |  |  |
| PRD10.18 | The emitter display should display x-ray imaging results onscreen detailed enough for determining x-ray image capture quality | Test |  |  |
|  | Uploads/Downloads |  |  |  |
| PRD8.41 | The device shall upload images, metadata, and device usage data to OP in a secure manner. | Test |  |  |
| PRD8.42 | The device shall provision providers and PACS server configurations from OP in a secure manner. | Test |  |  |
|  | IP testing |  |  |  |
| PRD15.2 | The emitter should be designed for IP54 and capable of passing IP33. | Test | Intertek |  |
| PRD15.3 | The cassette should be designed for IP54 and capable of passing IP33. | Test | Intertek |  |
|  | Ambient Settings (temp, humidity, etc) |  |  |  |
| PRD15.6 | The device shall operate within an ambient temperature of 0C to 40C. | Test |  |  |
| PRD15.7 | The device shall operate within a relative humidity of (non-condensing) 20-80%. | Test |  |  |
| PRD15.8 | The device shall operate at a pressure of 62 kpa to 106 kpa. | Test |  |  |
| RSK_R151 | The device shall contain a cooling system to maintain internal temperature below standard limits while within operating environmental conditions. | Test |  |  |
| RSK_R152 | The emitter and cassette enclosures shall protect the monoblock and detector using vibration and shock dampening materials. | Test |  |  |
|  | Ageing |  |  |  |
| PRD12.4 | The case shall be reusable. (Device can be packaged and repackaged in the case without degrading) | Test |  |  |
| PRD16.5 | The device shall be able to be stored for up to one year without use and still maintain EP. | Test/Inspection - MEMO | We should do a detailed review (and document it) of any materials that could degrade over time. We did not do aging for P00, so I think likely that we can justify not doing it for P01 assuming  a review of material stability/properties is favorable. We do need to consider battery life - in addition to battery testing per IEC62133 and UN 38.3, we should consider if there is performance testing we need to do in terms of being able to use our device for a certain # of times on a single battery charge - but this would be more to meet our use requirements (not aging testing related to storage). At Philips we had a device with a rechargeable battery and our rational for no aging was "In the case of the battery, the primary battery does not have a dated shelf life or an install before date since the battery is intended to be recharged and used from a charged state. The battery should be replaced based on the capacity indication." I would think we could use a documented review of materials/shelf life assessment to satisfy this requirement. Open to Rick's feedback.THis will be a little dependent on the manufacturers quality and specification of shelf life. Best case UM will state to charge system prior to usehopefully you are not producing Emitters and Cassettes and then putting them on the shelf for years. That is alot of capital to tie up. Long shelf lifes are usually for Disposales and a tired supply chain |  |
| PRD16.2 | The device service life shall be a minimum of 3 years of operation without component replacement; with an initial limited warranty of 1 year | Test/Inspection - MEMO | Refer to MEMO |  |
| PRD16.3 | The device service life shall be able to be extended by 3 years with the replacement of the monoblock, battery packs, display. | Test/Inspection - MEMO | Refer to MEMO |  |
|  | Accessories |  |  |  |
| PRD18.2 | The attachment mechanism should be simple enough for a single operator to hold the emitter in one hand, and the Detachable Part(s) in the other | Test |  |  |
| PRD18.7 | The force required to remove a puck from the emitter should not exceed 45 N. | Test |  |  |
| PRD18.8 | Pucks must be no more than 0.25lb each. | Test |  |  |
| PRD18.9 | The pucks should withstand a drop from 5 ft. | Test |  |  |
| RSK_R091 | The device shall contain a wireless detector trigger fault. | Test |  |  |
|  | Software Functions - Misc |  |  |  |
| PRD9.26 | The display UI should inform the operator when any fault occurs | Test/Demo |  |  |
| PRD9.27 | The display UI should provide a warning for low storage. | Test/Demo |  |  |
| PRD9.28 | The display UI shall provide an indication when a failure to capture an image occurs. | Test/Demo |  |  |
| PRD6.6 | The device shall perform a Power-On-Self-Test (POST) upon device startup to detect abnormal conditions including tube degradation, software modification, and battery health, collimator functionality. | Test/Analysis |  |  |
| PRD6.8 | The device shall be able to detect service issues and predictive maintenance of the batteries, monoblock, collimator. | Test/Analysis |  |  |
| PRD14.13 | The device's software system shall support unattended over the air (OTA) updates and diagnostics | Test/Analysis |  |  |
| PRD5.5 | The device shall monitor battery health and predict remaining battery life for servicing and repair by the mfg. | Test/Demo |  |  |
| PRD8.40 | The device shall connect to MedAI Platform (OP) and display connectivity status information | Test/Demo |  |  |
| PRD8.43 | The device should allow for AI algorithms to be downloaded from OP and run on the device. | Test/Demo |  |  |
| PRD8.44 | The device may allow OP to enable image sharing with Providers or Patients | Test/Demo |  |  |
| PRD8.27 | The manufacturer should be able to remotely disable the send to PACS/USB/Provider feature in the event that customer stops payments when the device is on lease. | Test/Demo |  |  |
| PRD8.28 | The manufacturer should be able to display an error message to alert the customer that a payment has been missed and device is disabled. | Test/Demo |  |  |
| PRD8.29 | The manufacturer should be able to remotely adjust device functionality; such as adjust Max kVp/mAs and turn DDR on/off | Test/Demo |  |  |
| PRD4.3 | The viewfinder shall display the optical image transformed to be from the reference frame of the cassette | Test/Demo |  |  |
|  | Hardware Functions - Misc |  |  |  |
| PRD6.2 | The critical safety circuits shall notify the software of a fault condition. | Test/Analysis |  |  |
| PRD6.4 | The faults initiated by the safety critical circuits shall render the system safe. | Test/Analysis |  |  |
| PRD6.5 | The device shall provide a fault tolerant high voltage interlock. | Test/Analysis |  |  |
| PRD14.11 | The system shall include all necessary sensors and external communication necessary to monitor the monoblock system and battery health to provide user notification(s) in advance of when servicing is needed | Test/Analysis |  |  |
| PRD7.12 | The device shall work with mfg supplied and customer supplied tablets wirelessly. | Test/Demo |  |  |
| TO BE SORTED |  |  |  |  |
| PRD7.49 | Tablet shall contain ability to record voice and sound |  |  |  |
| PRD7.50 | Device shall contain HIPAA compliant voice recognition |  |  |  |
| PRD7.51 | Device shall allow voice dictation into notes field |  |  |  |
| PRD7.52 | Device may allow voice control of user-initiated actions |  |  |  |
| PRD8.30 | Device shall allow the use of DICOM Modality Worklist for study management |  |  |  |
| PRD8.45 | Device may allow provisioning of patient data from MedAI Platform |  |  |  |
| PRD8.46 | Device shall allow the MedAI Platform to provision information to the device |  |  |  |
| PRD14.14 | Device shall remotely synchronize logs, fault codes, and images from device with MedAI Platform for preventative maintenance |  |  |  |
| PRD8.45 | Device may allow provisioning of patient data from MedAI Platform |  |  |  |
| PRD8.46 | Device shall allow the MedAI Platform to provision information to the device |  |  |  |
| PRD8.47 | Device shall remotely synchronize logs, fault codes, and images from device with MedAI Platform for preventative maintenance |  |  |  |
| RSK_R287 | The device shall follow development guidelines to mitigate breaking encryption. |  |  |  |
| RSK_R288 | The device software communication protocols shall validate data. |  |  |  |
| RSK_R289 | The device software should allow for encrypted Wifi technologies |  |  |  |
| RSK_R290 | The device software shall perform data validation to prevent compromised data and functionality. |  |  |  |
| RSK_R291 | The device software shall use two separate ports for SSE & REST. |  |  |  |
| RSK_R292 | The device software should reduce functionality based on risk for remote API invocation. |  |  |  |
| RSK_R293 | The device software shall not use of remote JSP(s). |  |  |  |
| RSK_R294 | The device software shall disallow shutdown from remote UI. |  |  |  |
| RSK_R295 | The device software shall disallow configuration functions from remote UI. |  |  |  |
| RSK_R296 | The device software shall undergo exploit testing before release. |  |  |  |
| RSK_R297 | The device software shall utilize the Principle of Least Privilege. |  |  |  |
| RSK_R298 | The device software shall disallow writing from USB to disk drive. |  |  |  |
| RSK_R299 | The control unit should utilize physical security to mitigate access to the EC and programmable boards. |  |  |  |
| RSK_R300 | The device software shall include Integrity checks. |  |  |  |
| RSK_R301 | The device software shall utilize Segmented networks. |  |  |  |
| RSK_R302 | The device software shall contain a Firewall. |  |  |  |
| RSK_R303 | The device software shall limit the privilege command set. |  |  |  |
| RSK_R304 | The manufacturer shall perform employee vetting and resume qualification association. |  |  |  |
| RSK_R305 | The device software shall allow for Offline install. |  |  |  |
| RSK_R306 | The device software shall contain Malware checking at build time. |  |  |  |
| RSK_R307 | The device software shall eprform integration checks. |  |  |  |
| RSK_R308 | The device software shall include a Password policy. |  |  |  |
| RSK_R309 | The device software shall contain an Account and IP lockout. |  |  |  |
| RSK_R310 | The device software shall utilize Modern development practices. |  |  |  |
| RSK_R311 | The device software shall Use newer/more secure protocols. |  |  |  |
| RSK_R312 | The device software shall utilize Segregation of software components. |  |  |  |
| RSK_R313 | The device software shall contain Cross-site request validation. |  |  |  |
| RSK_R314 | The device software shall include Exploit testing. |  |  |  |
| RSK_R315 | The device shall disable swagger-ui for production. |  |  |  |
| RSK_R316 | The device shall have SSH implementation. |  |  |  |
| RSK_R317 | The device software shall encrypt studies stored on the device. |  |  |  |
| RSK_R318 | The device shall allow the operator to initiate connection for remote desktop support. |  |  |  |
| RSK_R319 | The device software shall allow remote UI unlock when in AP mode. |  |  |  |
| RSK_R320 | The device software shall require credentials to access AP mode. |  |  |  |
| RSK_R321 | The device software shall require an unlock code to unlock the remote UI. |  |  |  |
| RSK_R322 | The device software shall utilize HTTPS when interfacing with MedAI Platform. |  |  |  |
| PRD7.53 | Device should support an ethernet, display port, and USB-A  connectors via a connector adapter |  |  |  |
| PRD8.47 | Device shall support image queuing for use off-network and network submission when connected |  |  |  |
| PRD8.48 | The system may view two images at a time for surgical comparison on large monitor(s) and pin images for comparison (Surgical Mode) |  |  |  |
| PRD9.34 | The display UI shall present time using the local time |  |  |  |
| PRD17.6 | The foot pedal should be designed for IP56 and capable of passing IP33. |  |  |  |
| PRD14.16 | The system shall enable a wipe of all images and patient identifying information from device |  |  |  |
| PRD20.31 | The device shall contain connectors tested Per IEC 60664-1, IEC 60512-4-1 Test 4a, IEC 60512-5-2-5b. |  |  |  |
| PRD7.54 | The device shall support the WPA2 protocol |  |  |  |

### Table 14
| ID | Summary | Requirement | NOTES | NOTES |
| --- | --- | --- | --- | --- |
| Note: These are specs generated from the Usability Req generation. Please ensure these line items or something similar are in your spec sheets with the proper Requirement(s) referenced. Please change values as they currently exist in the system. Given values are suggested values.THESE WERE-ADDED TO THE USE REQ PAGE |  |  |  |  |
| Alerts / Signals |  |  |  |  |
| USE_R027 | Alert - radiation | The device should inform the operator to leakage, residual, and stray radiation levels that pose risk to the operator(s) | Covered by IFU |  |
| USE_R101 | Document grid protection failure hazards | The device documentation shall make the operator aware of hazardous situations that may arise from grid protection failure. | PRD6.5 |  |
| USE_R102 | Document high voltage interlock failure hazards | The device documentation shall make the operator aware of hazardous situations that may arise from high voltage interlock failure. | PRD6.5 |  |
| USE_R015 | Alert - Emission | The Operator shall easily recognize when the device is emitting radiation. | IEC 60601-3 |  |
| USE_R028 | Alert - radiation emission | The device shall alert the operator of radiation emission | IEC 60601-3 |  |
| USE_R017 | Alert - Field out of Range | The system shall alert the operator once the emitter has been removed from the active area of the detector | IEC 60601-3 |  |
| USE_R031 | Alert - Shutdown | The system shall have an alert confirming that the operator has initiated the shutdown procedure | USE_R049 |  |
| USE_R034 | Alert - Troubleshoot | The device should notify the operator of required operator actions to troubleshoot the system during a known error | USE_R350 |  |
| USE_R006 | Alert - Activated Trigger | The device should provide feedback to the Operator when the trigger is activated | USE_R330 |  |
| USE_R037 | alerts - easy language | Error messages should contain reader friendly wording | USE_R347 & USE_R204 |  |
| USE_R029 | Alert - Reset | The reset button may issue a confirmation to reset changes made to an image | USE_R176 |  |
| USE_R026 | Alert - Poor Peripheral Connection | The device should notify the operator when a peripheral is not plugged | RSK_R136 |  |
| Cables |  |  |  |  |
| USE_R043 | Cable length | The connecting cables should be long enough to allow the operator to position the cassette easily around the intended anatomy | PRD6.3 & PRD11.22 |  |
| USE_R044 | Cables - bend | Cables shall have static bend radius not exceeding 4.5 x cable outer diameter and dynamic bend radius not exceeding 12 x outer diameter | PRD6.3 |  |
| USE_R046 | Cables - easy to plug in/out | Cables shall have a minimum pull strength of XX | Covered by 60601-1? |  |
| USE_R047 | Cables - trip hazard | The connecting cables should not be a trip hazard | PRD6.3 & PRD11.22 |  |
| Case |  |  |  |  |
| USE_R254 | Packaging Dimensions | The packaging shall be no larger than 47" x 20" x 9" |  |  |
| USE_R252 | Package padding | Foam or other shock absorbing padding shall be incorporated into the packaging of the equipment | PRD9.1 |  |
| USE_R282 | Primary Packaging Portability | The primary packaging (case) shall have handles and/or wheels to aid with transport and portability | PRD9.3 |  |
| USE_R051 | Case - proper storage compartments | The case should provide compartments for each of the three major components and the cables | PRD9.2 |  |
| USE_R281 | Primary Packaging Material(s) | The primary packaging (case) shall be made of lightweight material(s) | PRD11.20 |  |
| USE_R253 | Packaging Desiccant | The device should come packaged with desiccant to prevent unacceptable levels of relative humidity | RSK_R113 |  |
| Cassette Material(s) |  |  |  |  |
| USE_R192 | Impact absorption | The cassette may allow deflection to absorb impacts up to X" | IEC 60601-1 Test |  |
| USE_R312 | Shock absorption | The cassette must withstand a shock of X N/s of a X" diameter object | IEC 60601-1 Test |  |
| USE_R093 | Device material compatible | The device material should be compatible with most skin types | PRD12.16 |  |
| USE_R230 | Withstand weight - male arm | The device shall support the weight of a male arm in the 95th percentile | PRD11.3 |  |
| USE_R367 | Withstand weight - male leg | The device shall support the weight of a male leg in the recumbent and supine position in the 95th percentile + 10% | PRD11.3 |  |
| Cleaning |  |  |  |  |
| USE_R057 | Cleaning | Cleaning instructions and documentation of hazards and warnings for improper cleaning procedures shall be included in the IFU | PRD6.5 |  |
| USE_R062 | Cleaning Effect on Performance | The system should be able to be cleaned without a reduction in performance | PRD8.1 |  |
| Cables |  |  |  |  |
| USE_R077 | Control unit to cassette cable length | The control unit to cassette cable should be between X" and X" | PRD11.22 |  |
| USE_R265 | Patient Cable Length | The emitter to control unit cable shall not be greater than 3 meters | PRD11.22 |  |
| USE_R222 | Mains cable length | The control unit to AC mains cable should be between X" and X" | PRD11.22 |  |
| USE_R079 | Cable wrapping | The connecting cables should have cable wrapping features and be easily managed for storage | RSK_R004 |  |
| USE_R302 | Removable Patient Cables - Tool | The removable patient cables shall be easy to remove and reinstall with the use of a tool | RSK_R055 |  |
| Connectors |  |  |  |  |
| USE_R074 | Connectors - pull force | The connectors should withstand a pull force of X N | IEC 60601-1 Test |  |
| USE_R228 | Monitor & Peripheral Connection | The monitor and peripherals shall be easily connected to the Control Unit. | PRD5.11PRD5.12PRD5.13PRD5.14PRD5.15 |  |
| USE_R264 | Patient Cable Connection - Feedback | The device should provide haptic feedback when the patient cables are connected | USE_R073 |  |
| USE_R041 | Attachment mechanism feedback | All attachment mechanisms may incorporate operator positive feedback, either auditory or sensory. | USE_R073 |  |
| USE_R110 | Easy monitor connection | No tools should be required to connect the monitor to the device. | USE_R107 |  |
| USE_R294 | Re-attachable connector covers | The connector covers should easily reattach after removing peripherals | USE_R107 |  |
| Control Unit |  |  |  |  |
| USE_R301 | Removable control unit cover | The Control Unit cover(s) shall be easily removable from the Control Unit. | IEC 60601-1 (Connector cover req) |  |
| DDR |  |  |  |  |
| USE_R083 | DDR Termination - Field | The device should self-terminate the DDR sequence if the field exceeds the detector area | IEC 60601-3 (?) |  |
| USE_R084 | DDR Termination - SSD | The device should self-terminate the DDR sequence if the SSD becomes out of spec | IEC 60601-3 (?) |  |
| USE_R354 | Universal serial radiography features | The device shall have features that are universal to existing serial radiography systems | PRD1.2 |  |
| USE_R194 | Inform total DDR exposure time | The device may inform the operator of total DDR time during exposure | PRD2.4 |  |
| USE_R235 | No activity period of X s between DDR exposures | The device shall allow a DDR exposure after X s from the previous image | PRD3.14 |  |
| Emitter Dimensions |  |  |  |  |
| USE_R118 | Emitter Center of Mass | The weight of the emitter should be centered in an intuitive manor to reduce strain to the operator | USE_R118 |  |
| USE_R120 | Emitter Display Angle | The emitter display should be able to be angled +15 to -15 degrees from a 0 degree vertical viewing plane of the touchscreen. |  |  |
| USE_R144 | Emitter Weight Range | The Emitter should weigh between 3.15 and 6.9 lbs. | PRD11.16 & USE_R143 |  |
| Emitter Handle |  |  |  |  |
| USE_R098 | Distance between Emitter and Handle | The space between the emitter handle and the emitter must be at least X" | RSK_R160 |  |
| USE_R123 | Emitter Handle Comfort | The emitter shall have a handle that will allow comfortable use of the emitter during a test sequence and shall be designed to allow usage by at least 95% of operators. | USE_R156 |  |
| Emitter Keypad |  |  |  |  |
| USE_R097 | Display button size | The buttons used for the display should be at least 0.84” x 0.84” in order to accommodate a 99th percentile man finger width dimension. | PRD6.19 |  |
| USE_R128 | Emitter Keypad Force | The Emitter button(s) shall require between XX N and XX N of force to actuate | PRD6.19 |  |
| USE_R133 | Emitter Keypad Touch Type | The emitter keypad should be soft touch and the operator should be able to select with minimum effort | PRD6.19 |  |
| USE_R139 | Emitter UI Control | The Emitter UI buttons shall be controlled by the operator's thumb | PRD6.19 |  |
| USE_R199 | Keypad Active Area | The total emitter keypad active area shall be no greater than X" x X" | PRD6.19 |  |
| USE_R200 | Keypad to Handle Range | The distance from the handle to the keypad shall be <XXmm | PRD6.20 |  |
| USE_R106 | Easy access - keypad | All elements of the emitter keypad shall be accessible to the operator while in the intended single hand held position | USE_R126 |  |
| USE_R201 | Keypad Use - Emitter | The operator should be able to access and make use of the keypad without having the put the device down | USE_R126 |  |
| USE_R129 | Emitter Keypad Haptic Feedback | The emitter keypad should provide haptic feedback | USE_R330 |  |
| USE_R131 | Emitter Keypad Spacing | The emitter keypad buttons shall be at least X" apart | USE_R131 |  |
| Emitter Material |  |  |  |  |
| USE_R121 | Emitter Front Face Material | The emitter front face and LED windows shall be made of a clear material, free of cracks or crevices that can withstand repeated cleaning. | PRD8.1 & USE_R148 / USE_R149 |  |
| Emitter Trigger |  |  |  |  |
| USE_R249 | One Handed Use - Emitter Trigger | The Operator should be able to press and release the trigger with one hand while maintaining a steady position | PRD6.20 |  |
| USE_R335 | Trigger Location | The trigger shall be on the emitter | PRD6.20 |  |
| USE_R338 | Trigger to Handle Distance | The trigger shall be between X" and X" from the handle of the emitter | PRD6.20 |  |
| USE_R339 | Trigger to Keypad Distance Range | The distance between the trigger and the keypad shall be between X" and X" | PRD6.20 |  |
| USE_R336 | Trigger Pull Distance | The trigger shall require X" of movement from the resting position to initiate | PRD6.21 |  |
| USE_R137 | Emitter Trigger Force | The emitter Handle Button/Trigger shall be activated when exposed to a force between 4 to 20 ozf (1.1-5.6N). The button may incorporate operator positive feedback, either auditory or sensory. | PRD6.21 & USE_R330 |  |
| IFU |  |  |  |  |
| USE_R100 | Document grid protection system failures | The device documentation shall provide information to the operator to limit failures of the grid protection system. | PRD6.5 |  |
| USE_R103 | Document maximum weight | The device documentation shall provide explicit weight maximum(s) | PRD6.5 |  |
| USE_R116 | Education level | All accompanying documentation for the device shall be written at a level consistent with the educational level of the intended operator | PRD6.5 |  |
| USE_R087 | Detachable Part Use | The IFU shall indicate proper Detachable Part(s) use | PRD6.5 |  |
| USE_R072 | Connect monitor | Instructions for connecting the monitor should be provided in the IFU. | PRD6.5 |  |
| USE_R105 | Ease of Image Capture | Operator shall be able to easily capture an image (serial radiography, radiography, photography) after reading the IFU for the device. | PRD6.5 |  |
| USE_R164 | IFU - risk mitigation | Critical hazards/risks mitigated by Operator input shall be listed and explained in the IFU | PRD6.5 |  |
| USE_R165 | IFU - Set-up | Operator shall be able to assemble the components into a working device after reading the Assembly Instructions in the IFU. | PRD6.5 |  |
| USE_R166 | IFU Legibility | IFU should be printed with high contrast between text and background, and be easily legible | PRD6.5 |  |
| USE_R167 | IFU Legibility | The Operator shall be able to read and understand the IFU | PRD6.5 |  |
| USE_R168 | IFU Legibility - Font | The IFU should contain 12pt font minimum | PRD6.5 |  |
| USE_R169 | IFU states common damage conditions | The device IFU shall list common damage conditions the Operator is able to identify | PRD6.5 |  |
| USE_R269 | Pictorial instructions | Symbols and images should be used in instructions which are best explained graphically. | PRD6.5 |  |
| USE_R305 | Return Process | Operator shall be able to return the device components to the manufacturer for repair after reading the IFU for the device. | PRD6.5 |  |
| USE_R334 | Training Requirements - servicing | The Service Manual shall list proper training required for servicing | PRD6.5 |  |
| USE_R340 | Troubleshooting | The IFU should include all major troubleshooting activities the Operator can perform | PRD6.5 |  |
| USE_R365 | Warranty | Warranty information should be included in the instructions for use and given to the purchaser | PRD6.5 |  |
| Image Adjustment |  |  |  |  |
| USE_R182 | Image Adjustment - View and Sort | The images should be easily viewed and sorted by the operator | PRD6.23 |  |
| USE_R173 | Image Adjustment - Drag Edge | The image UI should allow drag and target on the image to the edge of the visible window and no farther | PRD6.24 |  |
| USE_R174 | Image Adjustment - Drag Lag | The image drag feature shall match the speed of the mouse | PRD6.24 |  |
| USE_R040 | Annotation of Images | Operator(s) should have the ability to append notes or annotations to exposures / images taken by the system to allow for review by other stakeholders and operators | PRD6.24 |  |
| USE_R146 | Exit image adjacent window | Exiting the image adjustment window should not reset the image | PRD6.24 |  |
| USE_R176 | Image Adjustment - Reset Changes | The device UI should allow the Operator to reset the image | PRD6.24 |  |
| USE_R181 | Image Adjustment - Undo | The device UI may allow the operator to undo the most recent action | PRD6.24 |  |
| USE_R170 | Image Adjustment | The function of the interface(s) used to manipulate images should be communicated to the operator via labeling, color or other visual cues. | PRD6.24 |  |
| USE_R190 | Image History Record | The operator facing interface should provide a history or notice of any and all changes made to an x-ray, captured image, or serial x-ray file. | PRD6.24 |  |
| USE_R171 | Image Adjustment - Adjustment Category | The interface should indicate which adjustment is being made at any time during the image adjustment process | PRD6.24 |  |
| USE_R180 | Image Adjustment - Contrast | ALL UI buttons and sliders should be in high contrast to the background | PRD6.24 |  |
| USE_R178 | Image Adjustment - Rotation Range | The rotation slider should allow 359 degrees of rotation in 1 degree increments | PRD6.25 |  |
| USE_R183 | Image Adjustment - Zoom Factor | The device UI shall zoom the image X% per mouse wheel full turn | PRD6.26 |  |
| USE_R369 | Zoom without mouse | The device may allow zoom without the use of a mouse | PRD6.26 |  |
| USE_R003 | Adjustable brightness | The brightness should allow adjustment between 0% and 100% | PRD6.27 |  |
| USE_R004 | Adjustable contrast | The contrast should allow adjustment between 0% and 100% | PRD6.27 |  |
| USE_R179 | Image Adjustment - Sharpness Range | The sharpness should allow adjustment between 0% and 100% | PRD6.27 |  |
| Imaging Modes |  |  |  |  |
| USE_R038 | Allow changing of modes | The device shall allow the Operator to switch between modes | PRD6.15 |  |
| USE_R069 | Confirm Mode - DDR | Operator shall be able to confirm the device is in serial radiography mode | PRD6.15 |  |
| USE_R070 | Confirm Mode - photography | Operator shall be able to confirm the device is in photography mode before use | PRD6.15 |  |
| USE_R071 | Confirm Mode - Radiography | Operator shall be able to confirm the device is in radiography mode before use | PRD6.15 |  |
| USE_R227 | Modes clearly indicated | The device shall contain different indicators for DDR, single shot, and photographic modes | PRD6.15 |  |
| USE_R322 | Switch Modes | The device should allow the Operator to switch between modes while Emitter is held. | PRD6.15 & PRD6.20 |  |
| USE_R163 | Idle Mode to Active State Time | The device should turn on from idle mode in 3 s | PRD6.13 |  |
| USE_R297 | Recognizable modes | Idle state shall easily be distinguished from active mode | PRD6.13 |  |
| Imaging Procedure |  |  |  |  |
| USE_R188 | Image Center | The Operator should be aware of the middle of the x-ray field before taking an image | In IEC 60601? |  |
| USE_R237 | No activity period of X s between photographic images | The device shall allow a photographic image after 1 s from the previous image | PRD1.6 |  |
| USE_R236 | No activity period of X s between single shot radiographic images | The device shall allow a single shot radiographic image after X s from the previous image | PRD3.13 |  |
| USE_R056 | Changing Technique Factors | The technique factors shall be able to be changed between imaging sessions | PRD6.17 |  |
| USE_R325 | Technique factors communication | The device shall communicate technique factor information to its operator(s) | PRD6.17 |  |
| USE_R327 | Time between capture and image saved | The device shall save all image types within X s of the operator taking the image | PRD5.1PRD5.2PRD5.3 |  |
| Labels |  |  |  |  |
| USE_R205 | Labeled active area | The Cassette shall have an active area marking | IEC 60601 Active area marking |  |
| USE_R295 | Reading distance | Direct labeling shall be readable from 10 ft away | IEC 60601-1 Section 7 |  |
| USE_R366 | Water detection sticker | Equipment components sensitive to fluids should contain a 'water detection' sticker which would inform the operator of a potentially hazardous compromise to the system | IEC 60601-1 Spillage |  |
| USE_R206 | Labeled images | Images should be referenceable to assist with radiation plan | PRD6.23 |  |
| USE_R241 | No peeling from cleaning | The device labels shall not peel after repeated cleaning | PRD8.1 |  |
| USE_R211 | Laser Brightness | The lasers on the emitter shall be between X and Y lumens of brightness | USE_R213 |  |
| LED |  |  |  |  |
| USE_R220 | LED visibility through drapes | The cassette LEDs shall be visible though a standard blue surgical drape | RSK_R002 & USE_R370 |  |
| USE_R214 | LED Brightness Range - Cassette | The LEDs on the Cassette shall be between X and Y lumens of brightness | RSK_R002 & USE_R370 |  |
| USE_R215 | LED Brightness Range - Emitter | The LEDs on the Emitter shall be between X and Y lumens of brightness | USE_R370 |  |
| USE_R216 | LED cycle interval | Blinking LEDs shall cycle between X and Y hertz | USE_R371 |  |
| USE_R217 | LED cycle length | Blinking LEDs cycles shall be at least X s in each state | USE_R371 |  |
| PACS |  |  |  |  |
| USE_R151 | Files sent to PACS in DICOM format | The device shall automatically send files to PACS in the DICOM format | PRD5.6 |  |
| USE_R255 | PACS - Check | The UI shall allow the operator to check the PACS package before sending | PRD5.6 |  |
| USE_R257 | PACS Menu | The device should have PACS menu options that mirror existing applications | PRD5.6 |  |
| USE_R258 | PACS Send Confirmation | The device should confirm image was sent to PACS | PRD5.6 |  |
| USE_R259 | PACS Send Confirmation | The device UI should display a confirmation screen before sending packet to PACS | PRD5.6 |  |
| USE_R260 | PACS Steps to Send | The device should have limited steps to send an image to PACS | PRD5.6 |  |
| USE_R155 | Format/clean hard drive | Hard drive space should be formatable once data has been sent to PACs servers. The system should prompt the operator to clean up disk space after the examination has been completed | PRD5.6 |  |
| USE_R261 | PACS Upload Status Bar | The system should display an uploading 'status bar' while data is being shared with the PACS servers | PRD5.6 |  |
| USE_R256 | PACS Image Qty | The device should be able to send more than one image at a time to PACS | PRD5.6 & PRD5.7 |  |
| Pucks |  |  |  |  |
| USE_R104 | Drop - Pucks | The pucks should withstand a drop from 5 ft | IEC 60601-1 Test |  |
| USE_R299 | Removable Components - Apparency | The system should have components that are easy to see if they are to be removed from the device | PRD6.4 |  |
| USE_R284 | Puck Edges | The puck posts should not be sharp | RSK_R080 & RSK_R017 |  |
| USE_R285 | Puck Labeling | The pucks labels should indicate their field size range | RSK_R177 |  |
| USE_R287 | Puck Post Min Dimension(s) | The puck posts must be at least X" off the front face of the emitter | RSK_R015 |  |
| USE_R207 | Labeling components | All detachable components should be easily recognisable as Imager Equipment | RSK_R063 |  |
| Positioning/Tracking System |  |  |  |  |
| USE_R320 | Stability of Emitter | The positioning system should account for the inherent decrease in stability which accompanies the use of a emitter | PRD3.6 |  |
| USE_R331 | Tracking System Error - Bunched Drape | The tracking system should have an error rate of no more than X when the drapes are bunched | PRD3.6 |  |
| USE_R332 | Tracking System Error - Covered LEDs | The tracking system should have an error rate of no more than X when X% of the LEDs are covered | PRD3.6 |  |
| USE_R196 | Interlock Limits | The interlock shall only activate in DDR and single shot mode | PRD3.7 |  |
| USE_R333 | Tracking System Error - Non-transparent Drape | The tracking system should have an error rate of no more than X when the cassette LEDs are covered with a drape of XX obacity | RSK_R002 |  |
| Power Switch/Button & Plug |  |  |  |  |
| USE_R224 | Mains Plug | This plug should allow for power to be obtained from a grounded wall outlet. | PRD6.29 |  |
| USE_R112 | Easy to use control unit power switch | The Control Unit Power Switch shall be easily switched on and off. | USE_R114 |  |
| USE_R278 | Power Switch - Apparency | The main power switch shall be easily recognizable. | USE_R114 |  |
| USE_R279 | Power Switch - Ease of Use | The main power switch shall be easy to press on and off | USE_R114 |  |
| USE_R291 | Push Button - Apparency | The power push button shall be easily recognizable. | USE_R114 |  |
| USE_R292 | Push Button Force | The power button shall require between X XN and XX N of force to actuate | USE_R114 |  |
| USE_R153 | Force to Activate Switch | The power switch shall require from 4 to 20 ozf to activate | USE_R114 |  |
| USE_R225 | Mains Plug Cover - Ease of Use | The Mains plug cover shall be easily removable from the plug. | USE_R114 |  |
| USE_R270 | Plus Apparency | The system shall have an easily recognizable plug. | USE_R114 |  |
| USE_R280 | Power-On screen | The device shall have a Power-On screen to indicate On state to operator | USE_R248 |  |
| Shut-Down |  |  |  |  |
| USE_R049 | Cancelation of Shut Down | The system should allow the operator to cancel the shutdown procedure | PRD6.8 |  |
| USE_R313 | Shutdown Command - Apparency | The system Shutdown command and its respective GUI button shall be easily recognizable by the operator. | PRD6.8 |  |
| USE_R324 | System shut-off | All UI displays and LEDs should turn off upon shut-off initialization | PRD6.8 |  |
| USE_R306 | Rocker Switch - Apparency | The rocker switch should be obvious to the operator during shut down | USE_R113 |  |
| Start-Up |  |  |  |  |
| USE_R293 | Pushbutton Ease of Press | The pushbutton shall be easy to press on and off | USE_R114 |  |
| System Level |  |  |  |  |
| USE_R193 | Impact Survivability | The device shall survive the foreseeable impacts (doorways, anatomy, tables, etc.) | IEC 60601-1 Test |  |
| USE_R321 | Stable device without mechanical support | The device shall be stable without the use of a fixed mechanical arm | PRD11.2 |  |
| USE_R353 | Universal radiography system features | The device shall have features that are universal to existing radiography systems | PRD1.1 |  |
| USE_R149 | Exterior Housing Surface | The exterior housing shall have a smooth surface. | USE_R148 |  |
| UI - General |  |  |  |  |
| USE_R066 | Colors | The UI should incorporate federally accepted colors and Imager colors only | IEC 60601-1 |  |
| USE_R152 | Font type | A sans serif font should be used | USE_R346 |  |
| USE_R158 | High contrast | Contrast between text and background should be high | USE_R346 |  |
| UI - Monitor |  |  |  |  |
| USE_R319 | SSD Display | The SSD shall be visible to the Operator during Use | IEC 60601 |  |
| USE_R226 | Manufacturer Contact Information | The device UI shall contain the manufacturer contact information | IEC 60601-1 Section 7 |  |
| USE_R068 | Confirm Mode | The UI shall signal to the operator which mode the device is in. | PRD6.15 |  |
| USE_R189 | Image Display - Number of Images | The device UI should display the most recent 5 images | PRD5.1PRD5.2PRD5.3 |  |
| USE_R229 | Monitor Interaction | The Operator should not need to interact with the monitor UI until the imaging session concludes | USE_R345 |  |
| USE_R359 | View Image - difficulty | The Operator should be able to view images without interacting with the UI | USE_R345 |  |
| USE_R316 | Smallest Image | The smallest image on the UI shall be at least XX | USE_R347 |  |
| USE_R344 | UI Image Viewing | The largest image on the UI shall be at least XX | USE_R347 |  |
| UI - Emitter |  |  |  |  |
| USE_R138 | Emitter UI Brightness | The Emitter UI shall be between X and Y lumens of brightness | USE_R347 |  |
| USE_R140 | Emitter UI Font Size | The Emitter UI text shall be at least X pnt font | USE_R347 |  |
| USE_R141 | Emitter UI Legibility | The text display on the emitter shall be easy to read | USE_R347 |  |
| Misc. |  |  |  |  |
| USE_R088 | Detector Quality | The system shall utilize a digital x-ray detector that is substantially equivalent/comparable to existing models | PRD1.1 |  |
| USE_R195 | Insulated packaging material | Packaging materials should be insulated to aid in the reduction of temperature change due to atmospheric conditions | PRD9.1 |  |
| USE_R268 | Photographic Camera Adjustment | Operator(s) should be able to manually or automatically focus the camera so that its photo is usable | PRD1.6 |  |
| USE_R372 | Software update steps | Software Updates shall be designed so that the operator will need to perform under three steps to update the software. | USE_R318 |  |
| From MEDTRONIC |  |  |  |  |
| General Product Requirements |  |  |  |  |
| Mounting, Portability, and Stability |  |  |  |  |
|  |  | The device shall be designed such that it can be placed on a clinical tabletop. To do so, the dimensions of the device shall be no more than X x X mm on the horizontal plane. |  |  |
|  |  | The transport case shall be designed in a way that the device is fully covered |  |  |
|  |  | The transport case shall allow insertion of the device in one direction only. |  |  |
|  |  | The power cord shall be 130 cm +/- 10%. |  |  |
| Cleaner and Solvent Resistance |  |  |  |  |
|  |  | The exterior of the device shall be cleanable with a cloth or sponge lightly moistened with a bactericide or germicide solution:• Mild dishwashing detergent• 70% isopropyl alcohol (rubbing alcohol)• 10% chlorine bleach (90% tap water)• Glutaraldehyde• Hospital disinfectant cleaners (phenolic-based: o-Phenylphenol 10.5%,o-Benzyl-p-chlorophenol 5.0%; Amphyl or equivalent)• Hydrogen peroxide• 15% ammonia (85% tap water)• Ammonia based household cleaners• Household cleaners (Alkyl Dimethyl Benzyl Ammonium Chloride 0.3%, 409 or equivalent)The pass/fail criteria will be that device surfaces, labels and other equivalent markings and shall resist removal or fading, smearing or blurring from disinfectants or cleaners. |  | Do we specify the exact cleaning procedure in a spec? |
|  |  | The case should be labeled with cleaning instructions. |  | Do we want to claim the case is cleanable? |
| Component Requirements |  |  |  |  |
| Individual Hardware Buttons |  |  |  |  |
|  |  | The device's operator interface shall include individual hardware buttons for the following functions:• “0/I” Switch, "MENU" Key to switch from one menu to another,• alarm control key labeled with appropriate symbols from IEC 60601-1-8 to pause an alarm, reset an alarm or pause the audio part of an alarm for 60 seconds,• "UP / UNFREEZE" Key to scroll up a menu or unfreeze curves,• "DOWN / FREEZE" Key to scroll down a menu or freeze curves,• "ENTER" Key to validate a setting, and• " ON/OFF" Key to start or stop the . |  | Must define all buttons/switches and their functions |
| LCD Screen |  |  |  |  |
|  |  | The LCD screen panel shall comply with the following constraints: Size 129.6 x 92.6 mm, monochrome with pixel resolution of at least 320 by 240. |  |  |
|  |  | The device should provide a means to manually adjust the contrast level of the LCD. |  | Don't know if this is something we would ever care about. Maybe for the monitor? |
|  |  | The device display system may include a screen saver function with the possibility to be disabled by the operator. |  | Optional |
| Audio Devices |  |  |  |  |
|  |  | The device shall have a means to check the operation of audible alarm function. |  |  |
|  |  | The system design shall prevent humidity from affecting performance. |  |  |
| Self Test |  |  |  |  |
|  |  | The device shall be ready to start  after completing at least the following set of POST (Power On Self Test) checks within 15 seconds: Status of power sources, status of critical memories integrity. In addition the device shall provide means to perform visual and auditory tests of LEDs and alarm buzzers. |  |  |
|  |  | The Safety Net shall comprise at least the following elements: Watchdogs for all safety critical processors, data acquisition channel integrity |  |  |
| Modes and Operation States |  |  |  |  |
| Start Up Transition Phase | N/A |  |  |  |
|  |  | Following power up or any CPU reset, the device shall execute Power On Self Test. |  |  |
| Power Down Transition Phase |  |  |  |  |
|  |  | If a Power Down occurs while the device is in an Active Radiation State, the device may complete the image capture, and store the most recent image in nonvolatile memory before turning off. Then on the next power up, the device will display the partially captured image. |  |  |
| Inactive  State |  |  |  |  |
|  |  | The device shall allow changing the mode while in Standby Mode |  |  |
|  |  | The device shall allow changing the settings while in Standby Mode |  |  |
| Active  State |  |  |  |  |
|  |  | The device shall not allow changing the mode while in Active Mode |  |  |
|  |  | The device shall not allow changing the settings while in Active Mode |  |  |
|  |  | During the imaging process, any mode changes shall apply at immediately |  |  |
|  |  | During armed state, the device shall allow changing the settings prior to its activation. |  |  |
| Service State |  |  |  |  |
|  |  | The device shall provide a “Service State” which shall allow the maintenance and the check-up of device system service functions. |  |  |
| Displayed Patient Data Monitoring Performance Requirements |  |  |  |  |
|  |  | The device shall have a time, kV, and current accuracy of X% |  |  |
|  |  | The device shall calculate and display dose |  |  |
| Operator Interface Requirements |  |  |  |  |
| Displayed Information |  |  |  |  |
|  |  | Upon startup, the device shall display manufacturer contact information, copyright notice, software version number. |  |  |
|  |  | The device shall indicate whether the operator interface controls are enabled or disabled. |  |  |
| Features and Capabilities of Information Storage |  |  |  |  |
| Events |  |  |  |  |
|  |  | The device shall provide a date and time stamp for all event log data. |  |  |
|  |  | The event log shall note any change to the system’s Real Time Clock by logging the Current Date/Time followed by the new Date/Time and a unique event code indicating the change |  |  |
| Detailed Monitoring |  |  |  |  |
|  |  | The device shall provide a date and time stamp for all stored images. |  |  |
| Counters |  |  |  |  |
|  |  | The device shall allow the operator to reset the patient total dose counter |  |  |
|  |  | The device shall provide a machine counter that shall count the number of hours spent in active  mode since the first use. |  |  |
|  |  | The device shall provide an Information Signals Menu where the 8 last signals are displayed with their occurring time and date. |  |  |
| Software |  |  |  |  |
|  |  | The software will meet the intended use |  |  |
|  |  | The software shall provide capabilities of displaying and transferring data of events, trends and detailed monitoring files. |  |  |
|  |  | The software shall provide capability of retrieving data files through memory devices or directly from the vent through an USB connection. |  |  |
|  |  | The software shall be able to identify each data file coming from the device with its device serial number. |  |  |
|  |  | The software shall include a warning popup message to prevent the user from recording the same data file on two different patient profiles and from associating one device to several patient profiles. |  |  |
|  |  | The software shall include a warning popup message to prevent the user from deleting data |  |  |
| Service Software |  |  |  |  |
|  |  | The software shall display and store event file data. |  |  |
|  |  | The software shall be able to upload device software. |  |  |
|  |  | The software shall provide capabilities of  performance tests management. |  |  |
|  |  | The software shall retrieve data files. |  |  |
|  |  | The software shall be able to identify each data file coming from the device with its device serial number. |  |  |
| Device’s USB Interface Capabilities |  |  |  |  |
|  |  | The software shall have a selection for USB device storage: real-time data collection or historical data transfer from the device internal memory. |  | Not sure if applicable |
|  |  | The user shall be able to select the duration of data record for real-time collection or the length of the past period for historical data retrieval. |  | Not sure if applicable |
|  |  | The device shall export its record files to a compatible USB memory device. A compatible USB device is a device formatted in 32 bit and between 256 MB and 4 GB capacity. |  | Not sure if applicable |
| Power Requirements |  |  |  |  |
|  |  | The device internal AC/DC power supply shall support 100 – 240 V and 50/60 Hz. |  |  |
|  |  | The device shall provide an external polarized AC cable to operate the device from AC mains power. |  | What is this? |
| IFU Requirements |  |  |  |  |
|  |  | The IFU shall include a caution to use a 32 bit formatted USB memory device. |  | Do we have a recommended storage size for the USB? |
|  |  | The IFU shall include a detailed description of all software displays. |  |  |
| Topics Requirements |  |  |  |  |
|  |  | The Instructions for Use shall include an accurate description of the device intended use. |  |  |
|  |  | The Instructions for Use shall include an accurate description of the Imager System characteristics and the installation procedure. |  |  |
|  |  | The Instructions for Use shall include an accurate description of using photographic, radiographic, and serial radiography modes. |  |  |
|  |  | The Instructions for Use shall include the list of pucks to achieve the desired exposure. |  |  |
|  |  | The user and clinician manual shall include a checklist to allow the user to understood the main topics of the IFU |  |  |
|  |  | The Instructions for Use shall include all available accessories |  |  |
|  |  | The IFU shall include a sentence or a symbol to remind the operator that the device shall be considered as waste electrical and electronic equipment. |  |  |
| Warnings and Cautions Requirements |  |  |  |  |
|  |  | The device IFU shall include a warning instructing the operator to read and to take account of the device intended use. The device IFU shall include a warning instructing the operator to read and to take account of environmental condition ranges for proper operation. |  |  |
|  |  | The device IFU shall include a warning instructing the operator to check that the device is properly assembled before operating. |  |  |
|  |  | The device IFU shall include a warning instructing the operator to read and to take account of the acceptable AC power supply characteristics. |  |  |
|  |  | The device IFU shall include a warning instructing the operator to check that the device surfaces before use |  |  |
|  |  | The device IFU shall include a warning instructing the operator to put the device in a safe place when in use. |  |  |
|  |  | The device IFU shall include a warning instructing the operator to check that the patient position couldn't lead to further pain/damage |  |  |
|  |  | The device IFU shall include a warning instructing the user to put the device in the case when using it in transport conditions. |  |  |
|  |  | The device IFU shall include a warning advising the operator/user to avoid using the device in a dusty environment. |  | Do we have limitations for dust? It doesn't seem like it. |
|  |  | The device IFU shall include a warning advising the operator/user against opening the device enclosure and advising the operator that only qualified personnel can service the device. |  |  |
|  |  | The device IFU shall include a warning advising the operator/user against powering on the device if the A/C power cord is damaged. |  |  |
|  |  | The device IFU shall include a warning advising the operator/user against use of liquid cleaner |  |  |
|  |  | The device IFU shall include a warning instructing the user to use approved or equivalent peripherals |  | Do we have specific peripherals spec'd? |
|  |  | The device IFU shall include a warning instructing the user to follow preventive maintenance schedule. |  | Do have a "schedule" |
|  |  | The device IFU shall include a warning instructing the user to check the file ID when using a USB memory device to transfer data between the device and the PC. |  |  |
|  |  | The device IFU shall include a warning instructing the user to clean the device regularly |  |  |
|  |  | The device IFU shall include a warning instructing the user to wash hands before manipulating the device and to regularly clean device and accessories |  |  |
|  |  | The device IFU shall include a warning instructing the user how to avoid condensation and overheating. |  | This includes into from ensuring fans are not obscured to appropriate use environment |
|  |  | The device IFU shall include a warning instructing the user to be careful not to obstruct the sound outlet. |  |  |
|  |  | The device IFU shall include a warning instructing the user to ensure the settings are compatible with the patient. |  |  |
|  |  | The device IFU shall include a warning instructing the user to check the settings when switching from one mode to another |  |  |
|  |  | The device IFU shall include a warning instructing the user not to store the the device for more than 2 years. |  | Related back to the shelf-life question |
|  |  | The device IFU shall include a warning instructing the user to ensure the I/O switch is in the Off (O) position before connecting and disconnecting the device to and from a power source. |  |  |
|  |  | The device IFU shall include a warning instructing the user to never use device or components that appear to be damaged and to contact the equipment supplier if any damage is discovered. |  |  |
|  |  | The device IFU shall include a warning instructing the user to use only the cleaning solutions recommended in the manuals and disconnect the components before cleaning. |  |  |
|  |  | The device IFU shall include a warning instructing the user not to leave power cables lying on the ground where they may pose a hazard. |  |  |
|  |  | The device IFU shall include a warning instructing the user that the device has to be used under responsibility and prescription of a doctor. |  |  |
|  |  | The device IFU shall include a warning informing the user that the manual describes how to respond to device alerts. |  |  |
|  |  | The device IFU shall include a warning instructing the user to always monitor the patient while the device is on, and to not leave the device on with the patient in the same room. |  | Seems like a bit much for our device |
|  |  | The device IFU shall include a warning instructing the user to handle the device carefully if the room temperature is high. |  | Do we even consider if they use the device out of environmental range? |
|  |  | The device IFU shall include a warning instructing the user to never immerse the device in liquid or allow any liquid to enter any device opening. |  |  |
|  |  | The device IFU shall include a warning instructing the user to wait for the device temperature to stabilize before using it after a transport or storage period. |  | Do we want to consider this? |
|  |  | The device IFU shall include a warning instructing the user that the device may exceed 41°C if the room temperature is above 35°C. |  | This depends on temperature testing, and what is decided as "operating temperature" |
|  |  | The device IFU shall include a warning instructing the user that the AC cable needs to be fastened to the device until click is felt. |  |  |
|  |  | The device IFU shall include a warning instructing the user to never expose the device to a direct flame. |  |  |
|  |  | The device IFU shall include a warning instructing the user to never expose any electrical part to water. |  |  |
| Servicing Manual Topics Requirements |  |  |  |  |
|  |  | The IFU shall include a schedule for preventive maintenance. |  |  |
|  |  | The IFU shall include a sentence or a symbol that reminds the operator that the device shall be considered as waste electrical and electronic equipment. |  |  |
|  |  | The device shall be labeled to indicate that the device shall be considered as waste electrical and electronic equipment. |  |  |
|  |  | A label reminding the operator to read the user manual shall be applied. |  |  |
|  |  | An Internal label shall be applied to warn about high voltage. |  |  |
|  |  | All device operational connectors and ports shall be labeled. |  |  |
| Product Traceability |  |  |  |  |
|  |  | All devices shall be identified by a unique serial number. |  |  |
|  |  | Device elements recognized as critical shall be identified by a unique batch number. Elements are defined as critical if their Risk Priority Index (RPI) in the FMEA is greater than 40 with severity greater than or equal to 4 and detection greater than 4. Additional elements can be added per manufacturing choice. |  |  |
|  |  | Device shall offer a way to check version number of PM and NUC |  |  |
| System Components |  |  |  |  |
|  |  | The assembly at first level shall include the following:1 Device 1 carrying Bag1 kit of 6 air inlet filters1 O2 connector1 CD Clinician’s manual (18 languages)1 double branch adult circuit (single branch for PB520)1 European power cord (type B)Customers will be prompted to order the following parts with every device:1 user manual printed (choose the correct language)1 power cord (if different from European power cord included in BOM above) |  | Do we want a list of major components listed in the DR? |
|  |  | The following items shall be available for usage with the device:• Clinical module software (for prescribers)• Service module software (for Home Care Providers)• USB cable for service (not allowed to be used in the homecare environment)• External battery• Circuits 5093500, 5093600, 5093900, 5094000• Valves packaged individually: 2 way (only PB560) and 3 way DAR valves (available with PB540)• Cart• Nurse call cable (from 540)• Car charger (DC power)• Dual Bags (pink and blue)• Carrying bag• FiO2 cell ( OOM102-1), cable and T piece (FIO2 kit) (only PB560) |  | Do we want a list of major components listed in the DR? |
| Shipping and Packaging Requirements |  |  |  |  |
|  |  | The device packaging and shipping shall comply with ADR and IATA regulations depending on the level of dangerous goods contained in the device. All finished goods units deemed critical to the safe operation of the device will be subjected to the ISTA-2A test standard ensuring that product integrity is maintained after the distribution environment. |  | Do we claim ISTA certification? |
|  |  | The device packaging shall include the storage duration recommended before use. |  | How do we want to claim shelf-life before use? According to Rick, there shouldn't be a long shelf life for capital equipment. |
|  |  | The device packaging shall have a shipping carton outer label containing information and symbols which comply with applicable standards. |  | Should be in Labels Spec sheet |

### Table 15
| Req written to apply to Imager | Notes | Add? |
| --- | --- | --- |
| Note: The following are requirements defined by Medtronic The details were changed to match the Imager. Details in Blue were not changed, so still relate to Medtronic's Ventilator. |  |  |
| General Product Requirements |  |  |
| Mounting, Portability, and Stability |  |  |
| The device shall be designed such that the it can be carried with a carrying case. |  | N |
| Operating Noise / Sound Levels |  |  |
| During normal usage, the noise level shall not exceed 30 dBA + 10% sound according to ISO 17510-1 2007 standard conditions. | May want to add as req/PRD | N |
| Features and Capabilities of Information Storage |  |  |
| Events |  |  |
| The device shall provide non-volatile data storage for at least 5000 events in an event log, including at least the following items:-  starts and stops - All confirmed  settings - All confirmed alert settings - All occurrences and ends of alarms with all their related actions:Inhibitions, cancellations, resets, acknowledge button presses. | Do we need audit log for all events? | Y |
| Product Traceability |  |  |
| The initial device graphical user interface and clinical software releases shall provide English (UK), English (US), French, Portuguese, Greek, Russian, Dutch, German, Polish, Turkish, Spanish, Italian, Japanese, Korean, Chinese, Finnish, Danish, Norwegian, Swedish | Do we only want to provide the IDU in English? There are no standards saying that we must provide it in other languages unless we go outside the US. | N |
| System Components |  |  |
| The device shall be compatible with all accessories for which compatibility claims are made in released labeling |  | N |

### Table 16
| UN13. | User Need Lookup | Operator should comfortably use the emitter and cassette in use positions where the emitter is pointing down and use positions where the emitter is pointing forward. |  | PRD2.24 | PRD Lookup | #N/A |  |  |  | UN LIST |  |  |  | PRD LIST |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NONE/Error |  |  |  | #N/A |  |  |  |  |  | UN1. | #REF! | Servicing Device UN | Manufacturer shall be able to service the device. | False | ID | Requirement | Specification |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN2. | #REF! | Patient Population UN | Operator shall use the device with adult and pediatric patients. | False | 1. General |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN3. | #REF! | Radiograph UN | Operator shall capture diagnostic radiographic images of extremities and shoulders. | False | PRD1.1 | The device shall be able to be packed, setup, and repacked without the use of a tool. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN4. | #REF! | DDR & Radioscopy UN | Operator shall capture diagnostic serial radiography, radioscopy of extremities and shoulders. | False | PRD1.2 | The device shall include Accompanying Documents (IFU) | Reference IFU Requirements |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN5. | #REF! | Photography UN | Operator shall capture photographic images of anatomies and objects. | False | 2. X-ray Imaging |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN6. | #REF! | Minimal PPE UN | Operator should use the device without surpassing their yearly occupational dose limits, per Code of Federal Regulations, Title 10, Part 20.1201. | False | PRD2.1 | The x-ray tube assembly shall be self-shielded. | -No additional shielding outside the monoblock-Complies with IEC 60601-1-3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN7. | #REF! | Layperson UN | Operator should be a medical professional and be able to transport, set up, use, and pack up the system alone without tools and with Accompanying Documents. | False | PRD2.2 | The device shall monitor and log beam current, filament current, and monoblock temperature with each acquisition. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN8. | #REF! | No Lead Lined Room UN | Operator shall be able to use the device without lead-lined rooms for radiation protection, if local regulations allow. | False | PRD2.3 | The x-ray tube focal spot size shall meet engineering specification for indicated anatomies and use enviroments | Specification identical to requirement (component specification - not to be verified at device top level) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN9. | #REF! | Battery Powered UN | Operator shall use the device while battery-powered. | False | PRD2.4 | The x-ray tube shall operate between 40 kV to 80 kV in 10kV increments. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN10. | #REF! | Environment UN | Operator shall use the device in the following environments: office, clinical. | False | PRD2.5 | The x-ray tube beam current shall operate between 1mA to 2mA. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN11. | #REF! | Packaging UN | Operator or Manufacturer shall be able to transport the packaged device safely in an automobile and airplane cargo. | False | PRD2.6 | The x-ray tube shall operate between 0.04 - 0.40 mAs in 5 steps; the options shall be 0.04, 0.08, 0.16, 0.25, 0.40 mAs. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN12. | #REF! | Disconnected Cassette+Emitter UN | Operator shall use the device without mechanically or electrically tethering the cassette and emitter together. | False | PRD2.7 | The x-ray exposure in serial radiographic mode shall be 40ms per frame, 5 frames per second, for a maximum of 20 seconds. | Test Points:@ 40kV, 0.04mAs 20 second DDR@ 60kV, 0.04mAs 20 second DDR@ 80kV, 0.04mAs 20 second DDR@ 40kV, 0.08mAs 20 second DDR@ 60kV, 0.08mAs 20 second DDR@ 80kV, 0.08mAs 20 second DDRAcceptance Criteria:Results in 100 + 1 pulses (frames)Sample 1st, 50th, and Last pulseMeasured Voltage ± 8% errorMeasured Current + 20% errorMeasured Time ± (10 % + 1ms) errorReference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN13. | #REF! | Ergonomic Shooting UN | Operator should comfortably use the emitter and cassette in use positions where the emitter is pointing down and use positions where the emitter is pointing forward. | False | PRD2.8 | The device shall be able to perform DDR up to 80 kV and 2mA. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN16. | #REF! | PACS et al UN | Operator shall send images and data to PACS and peripheral storage drives. | False | PRD2.9 | The device shall be able to perform single exposure x-rays up to 80 kV and 2mA. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN17. | #REF! | View and Post-Processing UN | Operator shall view an image and conduct post-processing (e.g. rotate, zoom, etc.) with and without internet connection. | False | PRD2.10 | The primary fixed collimation shall collimate to 43 deg. | The primary fixed collimation shall collimate to 43 deg +/- 0.5 degReference M10076 drawing |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN19. | #REF! | Import Patient Info UN | Operator may import patient information from an external source. | False | PRD2.11 | The device shall have a minimum Aluminum equivalent total x-ray beam filtration of 2.5 mm. | HVL > 2.5mm AL @70kVHVL > 2.9mm AL @80kV |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN20. | #REF! | Viewfinder UN | Operator shall view the anticipated x-ray beam illumination in order to distance, angle, align, and collimate the emitter, anatomy, and cassette for an intended radiograph. | False | PRD2.12 | The device shall utilize a digital flat field x-ray detector. | Reference cassette drawings for C1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN21. | #REF! | Tracking UN | Operator shall only be able to emit x-ray radiation while pointing the emitter at the cassette within allowable SID ranges | False | PRD2.13 | The device shall have a detector with an active area of 213.5mm x 213.5mm. | Reference cassette drawings for C1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN23. | #REF! | X-ray Technique UN | Operator should adjust loading factors (kVp, mAs) and acquisition type (Radiography, Photography) using HMI on the Emitter and Tablet. | False | PRD2.14 | The detector shall contain shielding or have shielding behind the detector. | Reference cassette drawings for C1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN25. | #REF! | Facility Metrics UN | Operator may have access to the device's images taken, image study dosage information for facility quality reviews. | False | PRD2.19 | The nominal x-ray exposure in Fluoroscopy Mode shall be 40ms per frame, 5 frames per second, for a maximum of 20 seconds. | Test Points:@ 40kV, 0.04mAs 20 second Fluro@ 50kV, 0.04mAs 20 second Fluro@ 60kV, 0.04mAs 20 second Fluro@ 64kV, 0.04mAs 20 second Fluro@ 40kV, 0.08mAs 20 second Fluro@ 50kV, 0.08mAs 20 second Fluro@ 60kV, 0.08mAs 20 second Fluro@ 64kV, 0.08mAs 20 second FluroAcceptance Criteria:Results in 100 + 1 pulses (frames)Sample 1st, 50th, and Last pulseMeasured Voltage ± 8% errorMeasured Current + 20% errorMeasured Time ± (10 % + 1ms) errorReference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN26. | #REF! | Regional Markets UN | Manufacturer should market the device in the United States, Canada, Mexico, and European Union. | False | PRD2.20 | The nominal voltage for Radioscopy shall be 80% or less than that of Radiography | Nominal voltage in fluoroscopy shall be 64kV |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN27. | #REF! | Foot Pedal UN | Operator shall be able to trigger acquisition wirelessly through a foot pedal. | False | PRD2.21 | The device shall provide a Low Dose Fluoroscopy Mode utilizing loading factors of 40 ms exposure time per frame, two point five (2.5) frames per second, for a maximum of 20 seconds duration | Test Points:@ 40kV, 0.04mAs 20 second Fluro@ 50kV, 0.04mAs 20 second Fluro@ 60kV, 0.04mAs 20 second Fluro@ 64kV, 0.04mAs 20 second Fluro@ 40kV, 0.08mAs 20 second Fluro@ 50kV, 0.08mAs 20 second Fluro@ 60kV, 0.08mAs 20 second Fluro@ 64kV, 0.08mAs 20 second FluroAcceptance Criteria:Results in 50 + 1 pulses (frames)Sample 1st, 50th, and Last pulseMeasured Voltage ± 8% errorMeasured Current + 20% errorMeasured Time ± (10 % + 1ms) errorReference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  | UN28. | #REF! | Weight Bearing x-ray UN | 75th Percentile American male patient shall stand on cassette for weight-bearing images of the foot and ankle. | False | 3. Positioning and Alignment |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.1 | The positioning system shall compute the Source to Detector distance (SID) within 5% error, through the full SID range and x-ray beam angles up to 30 deg realative to the normal axis of the cassette | The positioning system shall compute the Source to Detector distance (SID) with <5% error@ 25cm, 40cm, 60cm, 80cmComplies with IEC 60601-1-3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.2 | The positioning system shall compute the Source to Skin distance (SSD) such that calculated values are a) within error of 8% or 15mm (whichever is larger) and b) offset so that the calculated value is always less than the physical measured value. Tested at the SSD specified in PRD3.10. | The device shall compute the Source to Skin distance (SSD) with <  15mm error@ 25cm, 40cm, 60cm, 80cm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.3 | The device shall only allow x-ray emissions within a source-to-detector (SID) distance between 25cm and 80cm. | Reference SRSX-ray emissions disabled <25cm (LEDs red and will not emit)X-ray emissions allowed 25-80cm (LEDs green and able to emit)X-ray emissions Disabled > 80cm (LEDs red and will not emit) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.4 | The tracking system should function when 50% of the LEDs are not visible. | Tracking system functions (LEDs are green and device will emit x-rays) when 50% of the IR LEDs are covered |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.5 | The tracking system should function accurately with 1 standard medical drape over the cassette. | Tracking system functions (LEDs are green and device will emit x-rays) when 100% of the IR LEDs are covered by a single layer medical drape |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.6 | The tracking system shall operate under high ambient light conditions. | Tracking system functions (LEDs are green and device will emit x-rays)@2,000 lux (per IEC 60601-1 Subclause 7.1.2) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.7 | The device shall display to the operator the status of the system via indicator LEDs. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.8 | The automatic collimator to confine the x-ray field shall be able to adjust aperture size and rotation to line up with the detector at any specified SID. | @ 25cm, 40cm, 60cm, 80cm. Rotate the emitter 360 degrees in 45 degree increments and make sure the major & minor axes of the collimated field remain parallel to the walls of the active area at each increment. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.9 | The automatic collimator shall be able to adjust the aperture size at any given SID; selectable steps shall not exceed 0.8 cm in the length and width when in a plane orthogonal to the reference at a distance of 80 cm from the focal spot. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.10 | The device shall prevent x-ray emission when the calculated Source-to-Skin Distance is less than 130cm | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD3.11 | The device shall prevent hand-held Radioscopy and DDR | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 4. Viewfinder UI |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.1 | The device shall contain a viewfinder UI that allows the operator to view the detector active area while the cassette is draped, and provides a means to to align the anatomy and detector. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.2 | The viewfinder shall display the optical image transformed into an image as seen from the Cassette. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.3 | The viewfinder shall calculate and display the collimated x-ray field. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.4 | The viewfinder shall calculate and display the active area of the detector. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.5 | The viewfinder shall include a reference point to indicate the center of the x-ray field. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.6 | The viewfinder shall overlay the x-ray field and active area on the optical image. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.7 | The viewfinder shall display loading factors before taking an image. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.8 | The viewfinder shall provide positioning guidance in the form of angle and SID. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.9 | The viewfinder shall provide guidance on the UI to aid in aligning x-ray axis to cassette axis. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.10 | The viewfinder shall display the non-active area uniquely from the active area. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.11 | The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm.Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.12 | The viewfinder shall include a reference gauge so that the operator understands where the emitter is positioned in reference to the detector. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.13 | The viewfinder shall show the x-ray field projection for the puck that is selected. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD4.14 | The viewfinder shall display the imaging mode (Radiography, Radioscopy, or Photography). | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 5. Batteries and Charging |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.1 | The emitter shall contain a rechargeable internal battery pack with integrated BMS. | Reference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.2 | The cassette shall contain a rechargeable internal battery pack with integrated BMS. | Reference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.3 | The emitter shall display the status of the charging system. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.4 | The cassette shall display the status of the charging system. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.5 | The emitter and cassette battery packs shall have a rated capacity less than or equal to 100 WHr in order to allow for air transit. | Specification Identical to requirementReference MS-10010 and MS-10083 battery pack drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.6 | The emitter fully charged battery shall support 90 minutes of operation without intermittent charging for worst use case. | When subjected to the following use conditions, the Emitter battery shall last >90 minutes:80kV, 0.08mAs; 5s DDR, 15s wait, 5s DDR, 15s wait, 5s DDR, 15s wait, 15min wait. Repeat same DDR sequence every 15 minutes until device powers off. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.7 | The cassette fully charged battery shall support 90 minutes of operation without intermittent charging for worst use case. | When subjected to the following use conditions, the cassette battery shall last >90 minutes:80kV, 0.08mAs; 5s DDR, 15s wait, 5s DDR, 15s wait, 5s DDR, 15s wait, 15min wait. Repeat same DDR sequence every 15 minutes until device powers off. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.8 | The emitter shall be chargeable via wired power connection. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.9 | The charging power supplies shall contain ISO 60320 female plug to adapt to US and international plugs/outlets. | Specification Identical to requirementReference H1 Wired Charger drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.10 | The charging power supplies shall be compatible with input voltage and frequency ranges 100-240 V and 50-60 Hz. | Specification Identical to requirementReference H1 Wired Charger drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.11 | The cassette shall support x-ray emissions while wired charging. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.12 | The cassette shall be chargeable via wired power connection. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.13 | The emitter and cassette shall include a coin cell battery that is able to maintain a Real Time Clock (RTC) for a minimum of 3 months. | The minimum capacity of the coin cell battery shall be at least 4.3 mAh. (Reference: MEMO-P01-491 - Jetson RTC Battery Calculation, Rev A) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD5.14 | The emitter and cassette shall indicate when charging | The emitter and cassette UI shows a lighting bolt when charging |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 6. Critical Fault Monitoring |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD6.1 | The device shall perform a startup procedure to check wireless comms and calibration. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 7. Hardware System |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.1 | The device shall work with wireless viewing hardware (wireless tablets and wireless monitors) | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.2 | The device shall work with MedAI supplied and customer supplied Android tablets (with Android 10 or higher) over WiFi. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.3 | The emitter and cassette shall contain status indicators to inform user of armed, disarmed, and x-ray emission states. The loading state status shall have a yellow indicator. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.4 | The emitter shall have an optical camera for the Viewfinder display. | Specification identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.5 | The emitter shall have an IR optimized camera/sensor for IR Tracking System. | Specification identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.6 | The emitter shall contain Class 1 lasers to indicate the center of the x-ray field. | Specification identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.7 | The lasers output power on the emitter shall be between 0.4 and 1.0 mW. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.8 | The device shall automatically reconnect to a known WiFi network after inputting password the first time. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.9 | The emitter and cassette shall allow for WiFi connectivity using 2.4GHz and 5GHz bandwidths. | WiFi Module (M50817 ) shall be rated for 2.4GHz and 5GHz bandwidths |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.10 | The device shall serve as a private WiFi Access Point. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | Device Connections |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.21 | The cassette shall have service port(s), that is covered with a plug and requires a tool to access. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.22 | The cassette shall have 2 usb-c ports for power input and to connect accessories. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.23 | The emitter shall have service port(s), that is covered with a plug and requires a tool to access. | Specification Identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.24 | The emitter shall have a usb-c for power input. | Specification Identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD7.25 | The cassette shall have a usb-c port that can support connection to HDMI, Ethernet, and usb-a via a connector adapter. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 8. Software System |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.1 | The device shall allow the operator to switch between different imaging modes. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.2 | The device shall contain different indicators for each mode. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.3 | The system should save images upon end of acquisition | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.4 | The device idle state shall be distinguished from active state. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.5 | The device shall allow users to upload x-ray images and image series to the PACs server or local storage (USB Drive). | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.6 | The device shall allow sending files to PACS in the DICOM format. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.7 | The device should provide confirmation that the image study has been successfully submitted to PACS or local storage (USB Drive). | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.8 | The cassette shall be able to send/stream a image(s) to display hardware within 1 second from trigger release. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.9 | The emitter and tablet shall be able to pair to the cassette, and the foot pedal shall pair to the emitter. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.10 | The device shall contain debug and release modes for service operators. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.11 | Removed |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.12 | Removed |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.14 | The device shall only initiate x-rays when the computed x-ray field is contained within the image reception area. The device shall terminate x-rays if any part of the projected x-ray field is moved outside the image reception area. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.15 | The device shall limit the duty-cycle of single radiographs to a maximum of 200ms of exposure and 1800ms minimum of cooldown. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.16 | The device shall accept hyphens and spaces as part of name inputs. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.17 | The device shall limit the duty-cycle of serial radiographic and radioscopy mode to a maximum of 20 seconds of duration and proportional cooldown with a maximum of 40 seconds of cooldown. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.18 | The system shall be display the loading factors (kV, mAs) used for capturing the image. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.19 | The device shall support image queuing for use off-network and network submission when connected. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.20 | The system may allow viewing two images at a time for surgical comparison on large monitor(s), and pinning images for comparison. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.21 | The Mobile Device App shall be compatible with Android devices. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.22 | The device shall support at least the WPA2 protocol. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.23 | The device shall enter an idle state when the device is not utilized for 100 seconds. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.24 | The device shall exit an idle state within 30 seconds upon detection of emitter or foot pedal activity. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.25 | The device shall disallow x-ray acquisition when the device is in idle state | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.26 | The device shall allow the operator to take and view images without external internet connectivity. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.27 | The device shall normally prevent or stop x-ray acquisition if there is zero storage space for a full-length capture. Captures confirmed as sent to external storage (e.g. PACS), or images chosen by the user to be deleted, may be deleted before preventing x-rays. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.28 | The system shall provide a means to document the image orientation on both displayed and stored images | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.29 | The system shall provide a means to document the patient orientation for each image, when appropriate. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.30 | The live image displayed in fluoroscopy mode shall be displayed with less than a 0.350 second delay from irradiation to image appearance | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.31 | The system shall be able to perform exams during network communication activities (e.g. Sending to PACS) | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.32 | The device shall be able to enter Emergency Radioscopy Mode within 2 minutes of user initiation after a recoverable failure. | The MX1 shall be able to be powered ON and operator hit "Emergency Exam" button in 2 minutesReference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.33 | The device shall be able to recover all functions within 10 minutes. | The MX1 shall be able to be powered ON and operator shall be able to place patient info into to exam, take an x-ray, and view within 10 minutes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.34 | The Exam Screen shall include an "Irradiation Disabling Switch" which, when activated, will disable x-ray emissions until switched off. The Irradiation Disabling Switch may be activated at any time, including in the middle of an imaging sequence. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.35 | The System shall store all frames of a capture made in either DDR mode and Fluoroscopy Mode | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.36 | The SW system shall include a user-adjustable Timing Device that emits an audible warning after the limit has been exceeded. | Characteristics:- The operator shall be able to set the Timing Device to allow total emission times in an exam of up to 5 minutes without warning.- Any ray tube emission made without the Timing Device having been set shall cause a continuous audible warning signal during the loading.- Any x-ray tube emission made subsequent to the expirary of a previous set period shall cause a continuous audible warning signal during the loading- Resetting the Timing Device shall be possible, even during loading,- Means to control or reset the Timing Device cannot be the triggering switch or button.Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.37 | X-ray tube emission shall stop after the control is released and before more than one additional radiation pulse has been emitted. | "Loading Time" here is defined as the time between the start of the first pulse and the end of the last pulse. X-ray tube emission shall stop within 0.1 seconds of releasing any trigger, except when the loading time is less than 0.5 seconds. In that case, the emission may terminate within 0.5 seconds after the control is released. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.38 | The system shall provide means to set a limit, in normal use and no higher than 176 mGy/min, the maximum air kerma rate at the patient entrance reference point. Choosing to emit over this limit, when allowed, is Referred to as High Level Control. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.41 | RDSR (Radiation Dose Structured Reports) shall be created and exported for each exam, and have the capability to be sent to one or more destinations. | IEC 61910-1 Clauses 5.1.2 and 5.1.3, only "SHALL" features. Ignore Gantry angulations data requirements.Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.42 | The system shall limit DDR capture preview to 1 fps | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.43 | The system shall reduce DDR capture preview resolution to 25% of the image | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.44 | The system shall delay the first frame of a DDR capture preview by 2s | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.45 | The system shall intentionally delay the display of frames after the first frame of a DDR capture such that the frame rate is reduced to 1:5 frames. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD8.46 | The DDR preview shall display a warning text overlay to signify it as a preview | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 9. Software UI |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.1 | All data presented on the software UI shall have a unit of measure or label. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.2 | The MedAI Device App shall display the manufacturer contact information, a unique UDI, a message to refer to the MX1 IFU, and a warning that primary image interpretation shoul occur on DICOM displays. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.5 | The software UI should display the image or replay sequence after exposure without the operator interacting with the UI. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.7 | The software UI shall allow the operator to select and view acquired images. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.8 | The software UI shall allow the operator to independently manipulate the images. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.9 | The software UI shall allow the operator to "pinch to zoom" images. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.10 | The software UI shall allow the operator to rotate images; 360 degrees of rotation in 90 degree increments. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.11 | The software UI should persist rotation adjustments. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.16 | The software UI should display the network, device connection status, and signal strength, updating in under 90 seconds. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.17 | The software UI should display the PACS connection status. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.21 | The software UI shall display the SID during use. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.22 | The software UI shall display the dose after each image acquisition. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.26 | The software UI should inform the operator if any fault occurs. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.32 | The software UI should indicate the state of the device (e.g. Powered on, Charging, Available for imaging, Emitting radiation, and Error State) | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.35 | The software UI shall allow acquisition workflows while simultaneously uploading studies to PACS | MX1 system is able to take an x-ray image while loading to PACS. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.37 | The software UI shall allow the ability to select a puck before use. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.38 | The software UI shall display the source-to-skin distance (SSD). | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.39 | The software UI shall allow operator to select puck collimation size from a series of preselected options. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.40 | The software UI shall allow the user to adjust brightness, contrast, and sharpness of an image. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.41 | The software UI in non-emergency Radioscopy Mode shall display the Patients name and date of birth as well as the exam start date and time. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.42 | The software UI shall indicate the available image storage capacity at the beginning of an exam. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.43 | The software UI shall indiciate to the operator whether there is sufficient storage space to store a complete acquisition after selecting the mode and loading factors but prior to taking an image. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.44 | All displayed captured on the MedAI App Exam Screen shall be labeled with either "Live" or "Stored", as applicable. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.45 | Cumulative Air Kerma and Cumulative Dose Area Product during an exam shall be continuously displayed on the Device App and resets between Exams. | - Updated at least every 5 seconds- Accuracy of ±35% when greater than 100mGy, 5µGy*m2, and 6 mGy/min, respectively.Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.46 | The live capture shall always appear and be displayed in the same location on the monitor display. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.47 | The system shall provide an indication for when the x-ray beam axis is normal to the Active Area plane | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.48 | The software UI shall display "Emergency Mode" when the device is being used in Emergency Mode | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.49 | Choosing between Radioscopy and Radiography shall be available on the Device App or Emitter UI | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.50 | The software UI shall indicate when in LOW dose Radioscopy mode | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.51 | If a DDR or Radioscopy Capture is terminated for any reason other than releasing the trigger, the Device App shall notify them that a "Safety Feature" has ended the capture. | Feature to be released in Phase 4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.52 | When the device is set and positioned to exceed the air kerma maximum chosen for the High Level Control, the system shall emit an audible signal, unique to this warning, continuously. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.53 | The system shall provide the ability to inactivate any audible signals from the device, except for the High Level Control signal. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.54 | The Device shall sound an audible signal for initiation of x-ray tube emission. This sound in Fluoroscopy Mode shall be different than that of DDR Mode. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.55 | The Cumulative Reference Air Kerma and Reference Air Kerma Rate shall be clearly legible 2.5m from the display | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.56 | The Device shall by default display captures in Radiography Mode as light bones on dark background, and in Radioscopy Mode as dark bones on light background. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.57 | Removed |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD9.58 | The Device App shall display the Reference Air Kerma Rate in mGy/min continuously, updated every second, during Radioscopy emission. | - Updated at least every 1 seconds- Accuracy of ±35% when greater than 100mGy, 5µGy*m2, and 6 mGy/min, respectively.Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 10. HMI |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | Emitter |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.1 | The emitter keypad shall contain 3 tactile buttons | Specification Identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.2 | The emitter center button shall allow the operator to select between radiography and photography modes. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.3 | The emitter left button shall allow the operator to cycle between kV and right button shall allow the operator to cycle between mAs when in manual mode. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.4 | The emitter keypad buttons shall be controlled by the operator's thumb while holding the emitter with the same hand. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.5 | The emitter should provide haptic feedback when buttons and trigger are pressed. | The following buttons shall provide haptic feedback when pressed: Trigger 1 (inner handle), Trigger 2 (outer handle), UI buttons for OLED display (qty 3). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.6 | The emitter shall contain 2 trigger(s) for forward and downward x-ray emissions. | Specification Identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.7 | The emitter trigger(s) shall allow the operator to trigger an x-ray or photograph. Further x-ray or photograph capture shall not be allowed until the trigger is released. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.8 | The emiter trigger(s) shall be able to actuated with one finger. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.9 | The emitter shall contain an LDC display, that is a minimum size of 3.8" diagonally and has a minimum resolution of 720 x 720. | Specification Identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.10 | The emitter display shall display the viewfinder. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.11 | The emitter display shall display the remaining battery life in the form of a bar or percent. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.12 | The emitter display shall display the pairing status of the Foot Pedal. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.13 | The emitter keypad center button should gracefully shut off the emitter when press/hold for 3 seconds. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.14 | The emitter keypad center button should hard shut-off the emitter when press/hold for 10 seconds. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.15 | The emitter keypad center button should wake the emitter from idle with a single press of button. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.16 | The emitter should not take longer than 3 seconds to display a response once the power button is pushed. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.17 | The emitter shall be available for use in less than 180 seconds of initiating power on. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.18 | The emitter shall indicate when it is ON via the screen or an indicator light. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | Cassette |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.19 | The cassette shall contain a Monochrome OLED graphic display, that is at least 55mm x 13mm in size. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.20 | The cassette shall contain 2 buttons - 'Power Button' and 'Multi-Function Button'. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.21 | The cassette power button shall power the cassette ON with at least 2 presses. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.22 | The casssette should not take longer than 3 seconds to display a response once the power button is pushed. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.23 | The cassette shall be available for use in less than 180 seconds of initiating power on. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.24 | The cassette shall indicate when it is ON via the screen or an indicator light. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD10.25 | The cassette power button shall power the cassette OFF when press/hold for 3 seconds. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 11. Ergonomics |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.1 | The emitter should be comfortable to hold and move the emitter with all degrees of freedom in usable range during use. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.2 | The emitter shall be usable as intended with a left or right hand. | Emitter design shall be symmetrical |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.3 | The emitter shall be useable with one or two hands (primary + support hand). | The emitter shall be equal to or less than 8.0 lb |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.5 | The emitter shall allow the operator to simultaneously hold the emitter in a downward position and interact with the keypad buttons (via thumb) with one hand. | Specification Identical to requirementReference C1 cassette drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.6 | The emitter should be usable in the forward and downward directions. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.7 | The cassette active area shall support a minimum static load of 300 lbs with a 2X safety factor (test to 600 lb) | When 600 lb is applied to the cassette over an area of 0.1 m2 for 1 min, the cassette shall not:- Show any damage or permanent deflection greater than 5°. - BASIC SAFETY andESSENTIAL PERFORMANCE shall be maintained as defined by: MEMO-P01-441 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.8 | The weight of the cassette shall be such that it can be easily moved/trasported by a single person. | The cassette shall weigh equal to or less than 15.5 lbs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.9 | The weight of the packaging case shall be such that it can be easily moved/trasported by a single person. | The packaging case shall weigh equal to or less than 19 lbs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.10 | The weight of the device and packaging (when combined) shall be such that it can be moved/transported by a single person. | The device and packaging shall weigh equal to or less than 47 lbs (when combined) |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.11 | The operator shall be able to view the emitter and cassette indicator LEDs while the device is in use. | Specification Identical to requirementReference C1 cassette and E1 emitter drawings for detailed LED locations |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD11.12 | The weight of the foot pedal shall be such that it is easily moved/transported by a single person. | The foot pedal shall weigh equal to or less than 4 lbs. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 12. Packaging and Transportation |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD12.1 | The device shall be able to be packaged in a reusable hard shell case. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD12.2 | The case shall have a handle and wheels to be transported by a single operator. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD12.3 | The case shall have a telescoping handle. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD12.4 | The case shall incorporate foam or other shock absorbing padding. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD12.5 | The case should provide compartments to hold all the fixed and detachable components of the device. | Specification Identical to requirementReference P1 case drawings for detailed case specifications |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 13. Cleaning, Disinfection, and Sterile Bagging |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.1 | Removed |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.2 | The operator shall be able to clean all commonly touched surfaces without disassembling the device. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.3 | The cleaning procedure shall include standard materials and techniques. | The device shall be able to be cleaned using isopropyl alcohol and Cavicide. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.4 | The device shall be able to be cleaned in less than 5 minutes. | -The MX1 System enclosures materials and geometries must be deemed similar, or easier, to clean and disinfect than the P00 system, as determined by a third party lab. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.5 | The device disinfection time shall be 5 minutes or less with Cavicide. | -The MX1 System enclosures materials and geometries must be deemed similar, or easier, to clean and disinfect than the P00 system, as determined by a third party lab. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.6 | The device enclosures shall have sufficiently smooth outer shell to enable cleaning. | -The MX1 System enclosures materials and geometries must be deemed similar, or easier, to clean and disinfect than the P00 system, as determined by a third party lab. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD13.7 | The device shall be durable enough to withstand cleaning & disinfection for expected service life of the product. | - The E1 emitter and C1 cassette shall not show any major degradation such as rips, tears, or wear after 1,095 wipes with both Cavicide and 70% Isopropyl alcohol.- The F1 foot pedal and H1 charger shall not show any major degradation such as rips, tears, or wear after 548 wipes with both Cavicide and 70% Isopropyl alcohol. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 15. Operating Environment |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD15.1 | The device shall allow for transportation by air freight (cargo of plane). | Battery packs shall have a rated capacity less than 100 WHr |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD15.2 | The device shall be stored in an ambient temperature of -10C +55C. | Components sensitive to temperature shall be rated for storage within -10C to +55C. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD15.3 | The device shall be stored in a relative humidity of (non-condensing) 20-90%. | Components sensitive to humidity shall be rated for storage within 20-90% RH. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD15.4 | The device shall operate within an ambient temperature of 0C to +29.9C. | Components sensitive to temperature shall be rated for use within 0.0 C to +29.9C. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD15.5 | The device shall operate within a relative humidity of (non-condensing) 20-90%. | Components sensitive to humidity shall be rated for use within 20-90% RH. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD15.6 | The device shall operate at a pressure of 70 kpa to 106 kpa. | Components sensitive to pressure shall be rated for use within 70 to 106 kpa. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 16. Wireless Charger |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD16.1 | Emitter shall stop wireless charging for the duration of the x-ray acquisition | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD16.2 | Emitter shall allow x-ray emission when connected to wireless charger | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD16.3 | Removed |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD16.4 | Emitter shall only charge from one source when both wired and wireless chargers are present | Specification Identical to requirementCan be checked by plugging an inline current monitor into both power sources and confirming that both sources don't have current over one amp. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD16.5 | The wireless charger shall monitor internal temperatures and fail safe upon overtemp. | Overtemp limit set to 70CCan be checked by either setting the temp thershold below ambient or heating up the wireless charger in temp chamber and confirming that the W1 fails safe. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 17. Foot Pedal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.1 | The device shall support the use of a foot pedal with 2 triggers and 2 buttons. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.2 | The foot pedal right pedal (B) shall initiate a single x-ray exposure upon pressing and releasing when in radiographic mode. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.3 | The foot pedal right pedal (B) shall initiate DDR on the downpress and shall stop the exposure upon release when in radiographic mode. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.4 | The foot pedal should be rated to a liquid ingress rating of IPX8 per IEC 60529:1989/AMD2:2013/COR1:2019 | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.5 | The left button (A) shall switch between Radiography and Photography modes. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.6 | The foot pedal right button (B) shall rotate the image 90 degrees. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.7 | The foot pedal left pedal (A) shall "Favorite" or Save the current image. | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD17.8 | The foot pedal shall be wireless and work at a range up to 12 feet or more away from the emitter. | The foot pedal shall be able to trigger static and dynamic x-rays and change modes at a distance of 12 feet or more from the emitter. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 18. Collimation Pucks |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD18.1 | The device shall come with Collimation Pucks to collimate the x-ray field to smaller fields sizes than the automated collimator;  selectable steps shall be set no more than 0.8 cm apart (nominally) in the length and width when in a plane orthogonal to the reference at a distance of 80cm from the focal spot; the device's minimum selectable size shall not exceed 4 cm in length and width when in a plane orthogonal to the x-ray beam axis at a distance of 80 cm from the focal spot. | @25cm. Test at every manual collimation (puck) and automated collimation step.@ 40cm, 60cm, 80cm. Test at max automated collimation step.-The viewfinder shall overlay the calculated active area with the actual active area within 80% overlap.-The x-ray field measured along a diameter in the direction of greatest misalignment with the effective image reception area shall not extend beyond the boundary of the x-ray field area by more than 2 cm. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD18.2 | The emitter shall incorporate an attachment mechanism that allows an operator to hold the emitter in one hand and the attach or detach a puck with the other hand. | Specification Identical to requirementReference E1 emitter drawings |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD18.3 | The puck attachment mechanism should incorporate operator positive feedback when a puck is attached. | An audible click shall be heard when a puck is attached to an emitter. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD18.4 | The pucks shall not obstruct the ToF Sensors or Cameras. | With a puck attached to the emitter (any puck may be used), verify:- Puck does not appear in photograph when a photograph is taken- Puck does appear on viewfinder screen- Displayed SSD measurement is not < 5 cm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD18.5 | The pucks shall be uniquely identified so the operator can choose the appropriate puck for the desired collimation size. | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD18.6 | The pucks shall be packed in their own box and be able to be placed within the case | Specification Identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | 20. Applicable Standards |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.1 | The device design shall include processes defined in ISO 14971 Edition 3 2019, Application of risk management to medical devices | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.2 | The device shall conform to 21 CFR 1020.30:2018, PERFORMANCE STANDARDS FOR IONIZING RADIATION EMITTING PRODUCTS; Diagnostic x-ray systems and their major components. 21 CFR 1020.30(c), (h), (k), (l), (m), (n), and (o) shall be met by conforming to IEC 60601-1-3 and  60601-2-54. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.3 | The device shall conform to 21 CFR 1020.31:2015, PERFORMANCE STANDARDS FOR IONIZING RADIATION EMITTING PRODUCTS; Radiographic equipment by conforming to 60601-1-3 and 60601-2-54. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.4 | The device shall conform to 21 CFR 1020.32:2015, PERFORMANCE STANDARDS FOR IONIZING RADIATION EMITTING PRODUCTS; Fluoroscopic equipment. 21 CFR 1020.32(a), (b), (c), (d)(1), (d)(2), (d)(3)(i) – (iv), (d)(4), (f), (h), (i), (j), and (k) shall be met by conforming to IEC 60601-1-3, IEC 60601-2-54, and IEC 60601-2-43. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.5 | The device shall comply with IEC 60601-1 Edition 3.2 2020 Requirements for Medical Electrical Equipment. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.6 | The device  shall comply with IEC 60601-1-2 Edition 4.1 2020 Requirements for Medical Electrical Equipment. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.7 | The device shall comply with IEC 60601-1-3 Edition 2.2 2021 Requirements for Medical Electrical Equipment. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.8 | The device shall comply with IEC 60601-1-6 Edition 3.2 2020 Requirements for Medical Electrical Equipment, and IEC 62366-1 Edition 1.0 2015 Application of usability engineering to medical devices | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.9 | The device shall comply with IEC 60601-2-28 Edition 3.0 2017 Requirements for x-ray Tube Assemblies. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.10 | The device shall comply with IEC 60601-2-43 Edition 2.2 2019 Particular requirements for the basic safety and essential performance of X-ray equipment for interventional procedures | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.11 | The device shall comply with IEC 60601-2-54 Edition 2.0 2022 Requirements for Medical electrical equipment for radiography. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.12 | The x-ray Tube Assembly shall comply with IEC 60336:2005 for x-ray Tube Assemblies. | Complies with clause 201.7.2.102 of IEC 60601-2-28:2017 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.13 | The device shall comply with IEC 60601-2-43 Edition 2.2 2019 Particular requirements for the basic safety and essential performance of X-ray equipment for interventional procedures | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.14 | The device shall comply with IEC 60522 Edition 2.0 1999 for x-ray Tube Assemblies. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.15 | The device shall comply with IEC 62304 Edition 1.1 2015 for all software product lifecycle development. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.16 | The device shall comply with ISO 10993 Edition 5 2018 | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.17 | The device labeling shall comply with 21 CFR 801: Labeling. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.18 | The device shall comply with IEC 62133-2 Edition 1.0 2017-02 Secondary cells and batteries containing alkaline or other non-acid electrolytes - Safety requirements for portable sealed secondary cells, and for batteries made from them, for use in portable applications - Part 2: Lithium systems. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.19 | The device shall comply with section 38.3 of the UN Manual of Tests and Criteria (UN Transportation Testing) | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.20 | The device shall employ reasonable safeguards to prevent disclosure of any data classified as protected health information by and in accordance with the Health Insurance Portability and Accountability Act of 1996 (HIPAA). | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.21 | The device labeling shall comply with IEC 60825-1 Edition 2.0 2007-03 Safety of laser products - Part 1: Equipment classification, and requirements [Including: Technical Corrigendum 1 (2008), Interpretation Sheet 1 (2007), Interpretation Sheet 2 (2007)]. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.22 | The device shall comply with Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions, September 27, 2023. | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.23 | The device should support the DICOM Standard (NEMA PS 3.1 - 3.20 (2022) Digital Imaging and Communications in Medicine (DICOM) Set). | Reference SRS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.24 | The device shall be compliant to ANSI IEEE C63.27-2017 American National Standard For Evaluation Of Wireless Coexistence, and AAMI TIR69:2017/(R2020) Technical Information Report Risk management of radio-frequency wireless coexistence for medical devices and systems. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.25 | The device shall maintain essential performance after exposure to comply with ISTA 3A 2018 conditioning for Packaged-Products for Standard Parcel Delivery System Shipment 70 kg (150 lb) or Less. | After exposure to ISTA 3A conditioning for Standard Packaged Product, the device shall:-Maintain essential performance (per MEMO-P01-441)-Have no visible damage that affects safety or performance of the device |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.28 | The device shall comply with CISPR 11:2015 Industrial, scientific and medical equipment - Radio-frequency disturbance characteristics - Limits and methods of measurement. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.29 | The device shall comply with FCC 47 CFR Part 15 RADIO FREQUENCY DEVICES. | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.30 | The optional device tablet display shall comply with DICOM PS3.14 and IEC 62563-1 Edition 1.2 2021 for diagnostic image quality | Specification identical to requirementComplies per Third Party Test Lab |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False | PRD20.31 | The device shall comply with EPRC requirements that are not met via conformity to equivalent voluntary consensus standards: 21 CFR 1002 Subparts A, C, D, E, F; 21 CFR 1010.3; 21 CFR 1010.4; 21 CFR 1020.30 (a), (b), (d), (e), (g), (j), and (q); and 21 CFR 1020.31 (i), (d)(3)(v), and (g) | Specification identical to requirement |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  | False |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 17
| IMAGER LLC |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QSF-018 Risk and Hazard Analysis Template |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Issued by: Mgmt Rep |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Revision | DCO # | Description | Approved By | Approval Date | Digital Key |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A | 18-015 | Initial Release | Mgmt                                                                    Executive Mgmt | 2018-03-23 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B | 18-134 | Removal of Design Input Requirement and Design Record Tabs,Addition of Design Input Output Matrix Tab, Revision of Operator Need tab to remove justification | Mgmt                                                                    Executive Mgmt | 2018-12-12 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 18
| Pre-DV Tests Recommendation | Yes/No? | Destructive Level | Test Sequence | Protocol Status | Protocol Link |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy of x-ray Tube Voltage | Yes |  |  | Reviewed | EDE-P00-013-Accuracy of X-ray Tube Voltage_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of Loading Time | Yes |  |  | Reviewed | EDE-P00-006-Accuracy of Loading Time_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of Current Time Product | Yes |  |  | Reviewed | EDE-P00-007-Accuracy of Current Time Product_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Instability Hazards | yes |  |  | Reviewed | EDE-P00-011-Instability Hazards_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Push test | yes |  |  | Reviewed | EDE-P00-010-Push Test_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Drop Test | yes |  |  | Reviewed | EDE-P00-012-Drop Test_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| impact test | yes |  |  | Reviewed | EDE-P00-009-Impact Test_A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Excessive temperatures in ME EQUIPMENT | yes |  |  |  | EDE-P00-004 - Cassette Temperature TestsEDE-P00-005 - Control Unit Temperature Test |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| IP testing | yes |  |  | Reviewed |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Power Input | yes |  |  | initiated |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Dielectric strength for HV generator and x-ray Tube assembly ( the same as test from IEC 60601-2-7) | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Dielectric Withstand Test | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| The PERCENTAGE RIPPLE of the output voltage for ME EQUIPMENT with a CONSTANT POTENTIAL | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Earthing | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Leakage Current | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Interruption of Power Supply | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Static loadingTo perform test for imaging scanner ( x-ray source and Image detector). | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| elevation/vacuum testing | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| environmental testing (ball test) | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| cable pull & bend  test | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ESD | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| EFT | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| EMI | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Immunity | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SINGLE FAULT CONDITIONS | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SpillageTo perform test or to explain why spillage test was not performed, to put in the RMF | yes |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Creepage and Air Clearance | yes |  |  |  | waiting on EWS |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Methods of beam limitation in x-ray Equipment. | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Reproducibility of the Radiation Output in Radiography | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Linearity and Constancy in Radiography – IEC 60601-2-54, Clause | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Radiation data | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Test for dosimetric information | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| REFERENCE AIR KERMA RATE and the cumulative REFERENCE AIR KERMA and The overall uncertainty in the displayed values of the cumulative DOSE AREA PRODUCT | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Test for STRAY RADIATION | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| HVL | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Measured leakage radiation in the loading state | yes |  |  |  | waiting on final collimator design |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of marked and written indications. | yes |  |  | informal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Legibility of Markings | yes |  |  | informal |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Durability of Markings Test | no |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Abnormal Operation (fmea) | no |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accessibility of live parts | no |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| collimator lifecycle | No |  |  | not needed | N/A |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 19
| Test Description | Tests Recommended by Vlad to perform prior to DV | MREQ | IEC 60601-1, ED. 3;AAMI ES60601-1; CSA 60601-1: 2008 | IEC 60601-2-54 Clause | IEC60601-1-3: 2008) /Clause | Comment |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Legibility of Markings | X | MREQ-121 | 7.1.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Static loadingTo perform test for imaging scanner ( x-ray source and Image detector). | X | MREQ-124 | 9.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mechanical Strength TestPush test | X | MREQ-126 | 15.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Actuating part of control |  | MREQ-139 | 15.4.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Resistance to heat - Ball pressure test of thermoplastic parts |  |  | 8.8.4.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Excessive temperatures in ME EQUIPMENT | X | MREQ-128 | 11.1.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | MREQ-129 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | MREQ-130 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SINGLE FAULT CONDITIONS | X |  | 13.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Power Input | X | MREQ-116 | 4.11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Humidity treatment |  | MREQ-140 | 5.7 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Durability of Markings Test | X | MREQ-123 | 7.1.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Earthing | X | MREQ-122 | 8.6.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Dielectric Withstand Test | X | MREQ-133 | 8.8.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accessibility of live parts | X | - | 5.9.2 |  |  | The Imager system enclosures do not offer any way to access the internal live parts. We will not provide any tools to do so. The IFUs will clearly state that the equipment is not to be opened |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Leakage Current | X |  | 8.7a;8.7.3 A; 8.7.4.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Instability Hazards | X | MREQ-127 | 9.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Acoustic energy |  |  | 9.6.2.1 |  |  | does this apply? |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| X-Radiation |  |  | 10.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| ME EQUIPMENT intended to be connected to a power source by a plug |  |  | 8.4.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Thermal cycling |  |  | 8.9.3.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SpillageTo perform test or to explain why spillage test was not performed, to put in the RMF | X | - | 11.6 |  |  | N/A. Will write a rationale |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Cleaning and Sterilization |  | MREQ-138 | 11.6.7 |  |  | Test will be performed at Intertek after WuXi cleaning portion |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Abnormal Operation (fmea) | X | - | 13.2 |  |  | 13.2 refers to Single Fault Condition which generally is covered under FEMA risk assessment and mitigation for single fault failures. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Mold stress |  | MREQ-155 | 15.3.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Single impact of 6.78 Nm |  | MREQ-145 | 15.3.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Creepage and Air Clearance | X |  | 8.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Drop Test |  | MREQ-146 | 15.3.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | MREQ-147 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Interruption of Power Supply | X | MREQ-125 | 11.8 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Methods of beam limitation in x-ray Equipment. | X |  |  | 203.8.102 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Reproducibility of the Radiation Output in Radiography | X | MREQ-120 |  | 203.6.3.2.101. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Linearity and Constancy in Radiography – IEC 60601-2-54, Clause | X |  |  | 203.6.3.2.102 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of marked and written indications. | X |  |  | 203.8.102.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Correspondence between x-ray Field and Image Reception Area |  |  |  | 203.8.5.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Radiation data | X | MREQ-131 |  | 203.5.2.4.5.101 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Test for dosimetric information | X | MREQ-118 |  | 203.5.2.4.5.102 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| The PERCENTAGE RIPPLE of the output voltage for ME EQUIPMENT with a CONSTANT POTENTIAL | X | MREQ-115 |  | 203.4.101.2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| REFERENCE AIR KERMA RATE and the cumulative REFERENCE AIR KERMA and The overall uncertainty in the displayed values of the cumulative DOSE AREA PRODUCT | X | MREQ-118 |  | 203.6.4.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of x-ray Tube Voltage | X | MREQ-56 |  | 203.6.4.3.104.3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of Loading Time | X | MREQ-58 |  | 203.6.4.3.104.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Accuracy of Current Time Product | X | MREQ-114 |  | 203.6.4.3.104.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| limit the maximum REFERENCE AIR KERMA RATE to values given by local rules |  |  |  | 203.6.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Test for STRAY RADIATION | X | MREQ-119 |  | 203.13.6 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Determining the ATTENUATION OF RESIDUAL RADIATION ( N/A, to review application of product) |  |  |  | 203.11 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Dielectric strength for HV generator and x-ray Tube assembly ( the same as test from IEC 60601-2-7) | X | - |  | 201.8.8.3 |  | part of MREQ-116 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| HVL | X | MREQ-132 |  |  | 7.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Measured leakage radiation in the loading state | X | MREQ-117 |  |  | 12.4 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Measured leakage radiation when not in the loading state |  |  |  |  | 12.5 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 20
|  |  |  |  |  |  | Probably Needs to be Deleted or Discussed |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  | Not Addressed by PRD and Notify Engineering |  |  |  |  |  |  |
|  |  |  |  |  |  | Potentially Move to PRD b/c Overly Specific or Getting At Larger Problem |  |  |  |  |  |  |
|  |  |  |  |  |  | New User Need for Review |  |  |  |  |  |  |
| P00 Imager System Design Record - Operator Needs |  |  |  |  |  |  |  |  |  |  |  |  |
|  | Requirement ID | Source | Essential? | Theme | PRD Cnt | Design Input | Product Requirement | Design Output Evidence | Verification | Validation | Notes | Validation Strategy |
| UN3. |  | User Need | Y | GENERAL PERFORMANCE | #REF! | Device shall capture radiographic images of extremities, shoulders, and hips. | The device shall have a monoblock that integrates the HVPS and x-ray tube. |  |  |  |  |  |
|  |  |  |  |  |  |  | The x-ray tube focal spot size shall be less than 100 um. |  |  |  |  |  |
|  |  |  |  |  |  |  | The x-ray tube shall operate between 40 kV to 80 kV. |  |  |  |  |  |
|  |  |  |  |  |  |  | The x-ray tube beam current shall operate between 1mA to 2mA. |  |  |  |  |  |
|  |  |  |  |  |  |  | The x-ray exposure time in radiographic mode shall be 33ms, 66ms, or 99ms. |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall be able to perform single exposure x-rays up to 80 kV max |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall utilize a digital flat field x-ray detector. |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall have a detector with an active area of 22cm x 22cm (9"x9") |  |  |  |  |  |
|  | PRD2.9 |  |  |  |  |  | The system shall process static images to be diagnostically relevant. |  |  |  |  |  |
| UN4. |  | User Need | Y | GENERAL PERFORMANCE | #REF! | Device shall capture serial radiography or pulsed radioscopy of extremities, shoulders, and hips | The x-ray exposure in serial radiographic mode shall be 33ms per frame, 10 frames per second, for a maximum of 20 seconds. |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall be able to perform DDR up to 60 kV max |  |  |  |  |  |
|  |  |  |  |  |  |  | The system shall process serial images to be diagnostically relevant. |  |  |  |  |  |
| UN5. |  | User Need | N | GENERAL PERFORMANCE | #REF! | Device should capture photographic images . | The emitter shall have an optical camera with autofocus for taking pictures of anatomies and QR Codes |  |  |  |  |  |
| UN6. |  | User Need | N | GENERAL SAFETY | #REF! | Device radiation output shall remain within the Patient Dose limits without the use of lead aprons, unless required by regulation. |  |  |  |  |  |  |
|  |  |  |  |  |  |  | Photographic image  resolution shall be at least 0.5mm at less than 25 cm separation. |  |  |  |  |  |
| UN7. |  | User Need | N | GENERAL SAFETY | #REF! | Device radiation output shall remain within the operator and patient dose limits without the use of lead aprons, and without the use of lead lined rooms, unless required by state or federal regulations | The x-ray tube assembly shall be shielded. |  |  |  |  |  |
| UN8. |  | User Need | Y | GENERAL SAFETY | #REF! | Operator shall be able to use the device in non-lead lined rooms |  |  |  |  |  |  |
|  |  |  |  |  |  |  | The detector shall contain shielding or have shielding behind the detector |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should alert the user of necessary cooldown period procedure post DDR. |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall have an x-ray filter that is constructed of 6061 Aluminum. |  |  |  |  |  |
| UN21. |  | User Need |  | GENERAL SAFETY | #REF! | The Device shall maintain safe restrictions on x-ray emission during normal use (e.g. SID, alignment, SSD, etc.) per federal and state regulations |  |  |  |  |  |  |
| UN11. |  | User Need | N | GENERAL PERFORMANCE | #REF! | Operator should have the device available for use in less than 45 seconds of initiating power on, including warming procedures | The device shall be ready to operate within 30 seconds from the time the power button is turned on under nominal conditions. |  |  |  |  |  |
| UN12. |  | User Need | Y | GENERAL PERFORMANCE | #REF! | The device shall be able to be used without connecting to the wall outlet | The device shall be primarily battery operated. |  |  |  |  |  |
|  |  |  |  |  |  |  | The emitter shall contain a rechargeable internal battery pack. |  |  |  |  |  |
|  |  |  |  |  |  |  | The cassette shall contain a rechargeable internal battery pack. |  |  |  |  |  |
| UN13. |  | User Need | N | GENERAL PERFORMANCE | #REF! | The emitter and cassette batteries should be able to be charged while in use | The emitter shall support Essential Performance while wireless charging. |  |  |  |  |  |
|  |  |  |  |  |  |  | The emitter shall be chargeable via wired power connection |  |  |  |  |  |
|  |  |  |  |  |  |  | The emitter shall be chargeable via inductive charging dock |  |  |  |  |  |
| UN14. |  | User Need | Y | GENERAL PERFORMANCE | #REF! | The emitter and cassette batteries shall be able to be charged while in not in use |  |  |  |  |  |  |
| UN17. |  | User Need | N | ENVIRONMENTAL | #REF! | Operator shall be able to use the device in surgical environments | Human Interface elements (Lights, screens, etc) shall be visible in the bright surgical environment |  |  |  |  |  |
|  |  |  |  |  |  |  | Operator shall be able to view the detector active area while the cassette is obstructed by clear sterile bags or opaque surgical drapes |  |  |  |  |  |
| UN84. |  |  |  |  |  |  | Operator shall be able to use the device while covering the cassette in up to five drapes |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall support use with OTS drapes and clear sterile bags |  |  |  |  |  |
|  |  |  |  |  |  |  | The emitter shall have custom clear sterile cover(s) capable of being flush with optics/HMI components |  |  |  |  |  |
|  |  |  |  |  |  |  | The cassette shall have custom clear sterile cover(s) capable of being flush with IR LED components |  |  |  |  |  |
|  |  |  |  | ENVIRONMENTAL |  | Operator shall be able to use the device in home environments |  |  |  |  |  |  |
|  |  |  |  | ENVIRONMENTAL |  | Operator shall be able to use the device in ambulatory environments |  |  |  |  |  |  |
| UN18. |  | User Need | N | ENVIRONMENTAL | #REF! | Operator shall be able to use the device in typical outdoor athletic environments (grass, turf, asphalt, black top, etc.) | Human Interface elements (Lights, screens, etc) shall be visible in direct sunlight |  |  |  |  |  |
| UN11. |  | User Need | Y | ENVIRONMENTAL | #REF! | Operator shall be able to use the device in office and clinical environments |  |  |  |  |  |  |
| UN20. |  | User Need |  | ENVIRONMENTAL | #REF! | The packaged device shall be able to be transported safely in cargo of plane, cabin of plane, and automobile |  |  |  |  |  |  |
| UN21. |  | User Need |  | ENVIRONMENTAL | #REF! | Operator shall transport the deployed device safely in automobile |  |  |  |  |  |  |
| UN23. |  | User Need |  | USER INTERFACE | #REF! | Operator shall view all Human Interface elements (Lights, screens, etc) in the bright surgical environment and direct sunlight |  |  |  |  |  |  |
| UN24. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN25. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN26. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN27. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN16. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN17. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN30. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN19. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN20. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN41. |  | User Need |  | X-RAY IMAGING | #REF! | Device shall not be able to collimate larger than the detector active area |  |  |  |  |  |  |
| UN25. |  | User Need |  | X-RAY IMAGING | #REF! | Device shall collimate smaller than the active area, down to 13cmx13cm (~5x5 inches) at smallest SID, in discrete 1 cm increments |  |  |  |  |  |  |
| UN26. |  | User Need |  | X-RAY IMAGING | #REF! | Operator shall be able to visually compare the collimated area boundaries to anatomy landmarks for collimation alignment |  |  |  |  |  |  |
| UN44. |  | User Need |  | X-RAY IMAGING | #REF! | Operator should be able to collimate to the desired anatomy size with little or no knowledge on collimation techniques |  |  |  |  |  |  |
| UN45. |  | User Need |  | X-RAY IMAGING | #REF! | Operator shall be able to preselect collimation size from defined options |  |  |  |  |  |  |
| UN46. |  | User Need |  | X-RAY IMAGING | #REF! | Operator shall be able to ensure image is exposed correctly (Exposure Index) |  |  |  |  |  |  |
| UN27. |  | User Need |  | X-RAY IMAGING | #REF! | Operator shall be able to trigger radiation while 6ft or greater from the focal spot |  |  |  |  |  |  |
| UN34. |  | User Need |  | X-RAY IMAGING | #REF! | Operator shall understand x-ray beam center position |  |  |  |  |  |  |
| UN16. |  | User Need | N | USER INTERFACE | #REF! | Operator shall know the Device's current battery capacity and status (e.g. charging, not charging, damaged) to relevant accuracy |  |  |  |  |  |  |
| UN1. |  | User NeedHazard R1.123 | Y | USER INTERFACE | #REF! | The user interface shall inform the user when any fault occurs, or if the device requires service. | The display UI should inform the operator when any fault occurs |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should provide a warning for low storage. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall provide an indication when a failure to capture an image occurs. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI may provide feedback when a trigger press occurs while interlock is not met. |  |  |  |  |  |
| UN35. |  | User Need |  | USER INTERFACE | #REF! | Operator shall view x-ray imaging results on a screen detailed enough for determining x-ray image capture quality |  |  |  |  |  |  |
| UN22. |  | User Need |  | USER INTERFACE | #REF! | Operator shall view x-ray imaging results on a screen detailed enough for full diagnosis |  |  |  |  |  |  |
| UN23. |  | User Need |  | USER INTERFACE | #REF! |  |  |  |  |  |  |  |
| UN24. |  | User Need |  | USER INTERFACE | #REF! |  |  |  |  |  |  |  |
| UN39. |  |  |  | USER INTERFACE |  | The user interface should provide the user with detailed display of information to facilitate safe and effective use of the device. | Emitter Viewfinder shall display kV, mAs, SID, and mode as clear symbols or numbers. Operator shall be able to view factors before taking an image. |  |  |  |  |  |
|  |  |  |  |  |  |  | Emitter Viewfinder shall display the collimated x-ray boundaries for intended image, updating in less than 0.5 seconds. |  |  |  |  |  |
| UN40. |  | User Need |  | USER INTERFACE | #REF! | Operator shall recognize when the device is emitting radiation |  |  |  |  |  |  |
| UN48. |  | User Need |  | USER INTERFACE | #REF! | Operator shall know and be able to adjust the SID before taking the image, within limits allowed by other requirements. |  |  |  |  |  |  |
| UN49. |  | User Need |  | USER INTERFACE | #REF! |  |  |  |  |  |  |  |
| UN50. |  | User Need |  | USER INTERFACE | #REF! | Emitter Viewfinder shall display the collimated x-ray boundaries for intended image, updating in less than 0.5 seconds. |  |  |  |  |  |  |
| UN51. |  | User Need |  | USER INTERFACE | #REF! | Operator shall understand when tracking allows x-rays without noticeable delay |  |  |  |  |  |  |
| UN52. |  | User Need |  | USER INTERFACE | #REF! | Operator shall understand the means to align the emitter to detector active area to allow x-rays without noticeable delay |  |  |  |  |  |  |
| UN53. |  | User Need |  | USER INTERFACE | #REF! | Operator shall be able to view the detector active area while the cassette is obstructed by clear sterile bags or opaque surgical drapes |  |  |  |  |  |  |
| UN54. |  | User Need |  | USER INTERFACE | #REF! | Operator shall visually align the detector active area to anatomy landmarks |  |  |  |  |  |  |
| UN55. |  | User Need |  | USER INTERFACE | #REF! | Operator shall detect the device state (e.g. Powered on, Powered off, Charging, Available for imaging, Emitting radiation, and Error State) |  |  |  |  |  |  |
| UN56. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
| UN57. |  | User Need |  |  | #REF! |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  | All data presented on the display UI shall have a unit of measure or label and conform to the international standards for displaying of units. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall contain the manufacturer contact information. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display a minimum of two of the most recent images. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should persist rotation adjustments |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall allow the operator to independently adjust the sharpness, contrast ratio and brightness of the radiographic images. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI may allow the operator to invert the colors of the x-ray image (Black / White). |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should allow the operator to crop an image before sending to PACS |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI may allow for annotations. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display the network connection status in the main window. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display the PACS connection status in the main window. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display the Platform connection status in the main window. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display the Battery status in the main window. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display the Pairing status in the main window. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall display the SID during use. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall display the dose during use. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should display available and remaining DDR time |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should alert the user of necessary cooldown period procedure post DDR. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall inform the operator of radiation emission. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall include a reference point so that the operator understands where the emitter is positioned in reference to the detector and intended anatomy. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall always default the display of the image in the reference orientation of the emitter |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should indicate state of the device (active, idle, sleep) |  |  |  |  |  |
|  |  |  |  | USER INTERFACE |  | The user interface shall allow the user to control the device to facilitate safe and effective use. | The display UI shall allow the operator to shutdown the system. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall allow the operator to create patient studies |  |  |  |  |  |
|  |  |  |  |  |  |  | Operator shall be able to adjust loading factors and acquisition mode (e.g. single, DDR, photography, etc.) on the emitter |  |  |  |  |  |
|  |  |  |  |  |  |  | Operator or assistant should be able to adjust loading factors on an auxiliary control UI (tablet or mobile display) during surgery |  |  |  |  |  |
|  |  |  |  | USER INTERFACE |  | The user interface shall allow the user to view and manipulate image captures. | The display UI should allow the operator to view images without interacting with the UI. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall allow the operator to select and view acquired images. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall allow the operator to independently manipulate the images. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall allow the operator to rotate images; 360 degrees of rotation in 90 degree increments. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI shall allow the operator to independently adjust the sharpness, contrast ratio and brightness of the radiographic images. |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI may allow the operator to invert the colors of the x-ray image (Black / White). |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI should allow the operator to crop an image before sending to PACS |  |  |  |  |  |
|  |  |  |  |  |  |  | The display UI may allow for annotations. |  |  |  |  |  |
|  |  |  |  |  |  |  | Operator shall review loading factors on an image taken |  |  |  |  |  |
|  |  |  |  |  |  |  | Operator shall be able to invert the x ray images for viewing as needed |  |  |  |  |  |
|  |  |  |  |  |  |  | Operator should be able to zoom in a captured radiograph or photograph |  |  |  |  |  |
| UN103. |  |  |  | USER INTERFACE | #REF! | Operator should primarily interact with the software UI by hand or finger touch |  |  |  |  |  |  |
| UN104. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN105. |  |  |  | USER INTERFACE | #REF! | Operator shall be able to view PACS, Platform, and Network connectivity status on software UI |  |  |  |  |  |  |
| UN106. |  |  |  | USER INTERFACE | #REF! | Operator shall review loading factors on an image taken |  |  |  |  |  |  |
| UN107. |  |  |  | USER INTERFACE | #REF! | Operator shall be able to invert the x ray images for viewing as needed |  |  |  |  |  |  |
| UN108. |  |  |  | USER INTERFACE | #REF! | Operator should be able to zoom in a captured radiograph or photograph |  |  |  |  |  |  |
| UN15. |  | User Need | Y | HUMAN FACTORS | #REF! | Operator shall be able to setup or pack the device alone and without the use of a tool |  |  |  |  |  |  |
| UN9. UN10 |  | ISO15223-1IEC60601-1IEC 60601-1-2IEC 60601-2-54IEC60601-2-28FDA Final Rule - Use of Symbols in Labeling | N | HUMAN FACTORS | #REF! | Operator shall set up and operate the device using only information provided in the Accompanying Documents and Training |  |  |  |  |  |  |
| UN10. |  | User Need | Y | GENERAL | #REF! | Operator shall be able to read and understand labeling |  |  |  |  |  |  |
|  |  | User NeedRSK-P01-003 (various hazards, see labeling tab) |  |  |  |  | Labels and IFU shall meet FDA and ISO requirements |  |  |  |  |  |
| UN59. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall be able to access and interface with the Emitter UI without having to put down the emitter |  |  |  |  |  |  |
| UN60. |  | User Need |  | HUMAN FACTORS | #REF! | Operator should be able to view the emitter display while holding the emitter in use positions | IFU shall include all information necessary for safe and effective use of the P00 including safety information, set up and clinical use instructions, maintenance, use with accessories, and specifications. (see "Lableing" tab for detailed labeling requirements). |  |  |  |  |  |
| UN61. |  | User Need |  | HUMAN FACTORS | #REF! |  |  |  |  |  |  |  |
| UN28. |  | User Need |  | HUMAN FACTORS | #REF! | 75th Percentile American male patient shall be able to stand on cassette for weight-bearing images of the foot and ankle |  |  |  |  |  |  |
| UN63. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall be able to position the cassette in a horizontal and vertical orientation for imaging |  |  |  |  |  |  |
| UN64. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall move the emitter with all degrees of freedom in usable range |  |  |  |  |  |  |
| UN65. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall be able to operate the emitter when pointed in the downward and forward directions |  |  |  |  |  |  |
| UN66. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall hold the emitter steady and trigger x-rays with one hand for the duration of the image acquisition without motion blur |  |  |  |  |  |  |
| UN67. |  | User Need |  | HUMAN FACTORS | #REF! | Operator should hold the emitter outstretched for at least 30 seconds without discomfort or dropping |  |  |  |  |  |  |
| UN68. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall be able to trigger radiation without the use of their hands |  |  |  |  |  |  |
| UN69. |  | User Need |  | HUMAN FACTORS | #REF! | The emitter should be able to be suspended in a fixed position to image without the use of the operator's hands |  |  |  |  | Operator shall be able to perform normal procedures and have both hands available for use |  |
| UN70. |  | User Need |  | HUMAN FACTORS | #REF! | Operator shall be able to pick up, reposition, and transport between rooms a bagged cassette by the handle using wet soiled gloves |  |  |  |  | Bagged and gloved test pick up and put down |  |
| UN71. |  |  |  | HUMAN FACTORS | #REF! | Operator shall position patient anatomy for capture 1/2" from the edge of the cassette body (Bezel problem) |  |  |  |  |  |  |
| UN72. |  |  |  | HUMAN FACTORS | #REF! | Operator should not feel fatigue from holding the cassette with one hand for 30 seconds |  |  |  |  |  |  |
| UN73. |  |  |  | HUMAN FACTORS | #REF! | Operator should not feel fatigue from holding the emitter for the length of a DDR cycle plus 5 seconds |  |  |  |  |  |  |
| UN74. |  |  |  | HUMAN FACTORS | #REF! | Operator may power or wake all device components with one recognizable power button, if previously connected. |  |  |  |  | Get Dhruv's feasibility |  |
| UN75. |  |  |  | HUMAN FACTORS | #REF! | Operator shall be able to operate the device with either a left or right hand(s) |  |  |  |  |  |  |
| UN76. |  |  |  | HUMAN FACTORS | #REF! | Operator shall trigger radiation and interact with emitter UI with all combinations of soiled/not soiled and single-gloved/double-gloved hands |  |  |  |  |  |  |
| UN77. |  |  |  | HUMAN FACTORS | #REF! | Operator should connect either data or power connections into any port for that function in the emitter and cassette |  |  |  |  |  |  |
| UN78. |  |  |  | HUMAN FACTORS | #REF! | Patient arm shall be supported all the way without pinch points |  |  |  |  |  |  |
| UN79. |  |  |  | HUMAN FACTORS | #REF! | Operator should not need to look down at the foot pedal for which button to hit |  |  |  |  |  |  |
| UN80. |  |  |  | HUMAN FACTORS | #REF! | Operator should place the cassette under a patient's back or anatomy in supine position with one hand as comfortably as currently done |  |  |  |  |  |  |
| UN81. |  |  |  | HUMAN FACTORS | #REF! | Operator may perform surgical operations (including drilling, hammering, pushing, etc) on anatomy while resting on the cassette |  |  |  |  |  |  |
| UN82. |  |  |  | CLEANING AND STERILITY | #REF! | Operator shall be able to clean the device with standard cleaning agents used in clinics and hospitals. |  |  |  |  |  |  |
| UN83. |  |  |  | CLEANING AND STERILITY | #REF! | Operator shall be able to use the device in the surgical field without compromising the sterile field |  |  |  |  |  |  |
| UN84. |  |  |  | CLEANING AND STERILITY | #REF! | Operator shall use the device while covering the cassette in up to five drapes | Device shall be able to be cleaned or disinfected with isopropyl alcohol, cavicide (quaternary ammonium), bleach (hypochlorites), hydrogen peroxide, and water, without reduction in performance. |  |  |  |  |  |
| UN85. |  |  |  | CLEANING AND STERILITY | #REF! | Operator shall maintain sterility of the emitter and cassette while charging for surgical use |  |  |  |  |  |  |
| UN86. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN87. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN88. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN89. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN90. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN91. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  | The operator shall be able to clean all commonly touched surfaces without disassembling the device. |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall be able to be cleaned without the need for a brush |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall be able to be cleaned in less than 5 minutes. |  |  |  |  |  |
|  |  |  |  |  |  |  | The disinfection time with Cavicide should be 3 minutes or less |  |  |  |  |  |
|  |  |  |  |  |  |  | The device shall have a smooth outer shell. |  |  |  |  |  |
|  |  |  |  |  |  |  | Any gaps shall be filled with sealant; any filler shall survive cleaning validation |  |  |  |  |  |
| UN92. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should be able to have the loading factors set by the device with input of anatomy and view. | Any gap requiring disinfection, that is not filled, shall be wider than 1 mm and have an aspect ratio of 3:1 in order to be cleaned |  |  |  |  |  |
| UN93. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall be able to send images to the MedAI cloud platform | All inner edges shall be rounded sufficiently to enable cleaning |  |  |  |  |  |
| UN94. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall be able to send images to PACS or similar archive |  |  |  |  |  |  |
| UN95. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall send images and data to teleradiology service |  |  |  |  |  |  |
| UN96. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should securely send images to the patient |  |  |  |  |  |  |
| UN97. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should add virtual lengths and angles to images taken with the device, with ±0.5cm and ±2° accuracy. |  |  |  |  |  |  |
| UN98. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall indicate Left or Right or bilateral side anatomy on the image before submission to PACS |  |  |  |  |  |  |
| UN99. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall indicate Upright or Supine on the image before submission to PACS |  |  |  |  |  |  |
| UN100. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall indicate Expiration or Inhalation on the image before submission to PACS |  |  |  |  |  |  |
| UN101. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should indicate AP, PA, Lateral, and Oblique on the image before submission to PACS |  |  |  |  |  |  |
| UN102. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN109. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should view images or DDR video playback without initiation (auto-start) within 5 seconds of finishing capture. |  |  |  |  |  |  |
| UN110. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall select wireless network by selecting available options or keyboard input (for hidden networks) |  |  |  |  |  |  |
| UN111. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN112. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall populate PACS required data fields prior to or after collecting images |  |  |  |  |  |  |
| UN113. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall understand data entry fields that are required for PACS submission |  |  |  |  |  |  |
| UN114. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall configure required or available data entry fields for PACS submission during device setup |  |  |  |  |  |  |
| UN115. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall view separation between image studies to mitigate incorrect patient, anatomy, study info |  |  |  |  |  |  |
| UN116. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall send both processed and post-processed image to PACS |  |  |  |  |  |  |
| UN117. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall make post-processing radiograph image adjustments (e.g. Brightness, Sharpness, Contrast, Cropping, and Rotating) |  |  |  |  |  |  |
| UN118. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should securely queue images to be sent to a PACS server when network connection is restored |  |  |  |  |  |  |
| UN119. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall securely send images to a USB storage drive without network connection. |  |  |  |  |  |  |
| UN120. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should clear visibility of past images on UI |  |  |  |  |  |  |
| UN121. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall wipe all images and patient identifying information from device before service, etc |  |  |  |  |  |  |
| UN122. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall view time on the device as local time |  |  |  |  |  |  |
| UN123. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall view images and UI or stream to monitors without internet connection |  |  |  |  |  |  |
| UN124. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator may view two images at a time for surgical comparison on large monitor(s) and pin images for comparison |  |  |  |  |  |  |
| UN125. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall view, post-process, and send images on tablet or laptop without internet connection |  |  |  |  |  |  |
| UN126. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator may submit notes via voice dictations into image study record |  |  |  |  |  |  |
| UN127. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator may scan barcode or other identifier means to enter patient information automatically |  |  |  |  |  |  |
| UN128. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator may search past image studies for patient (from device or facility) and view images on device |  |  |  |  |  |  |
| UN129. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator should search and import patient information (e.g. name, birthday, patient ID, etc.) from external database |  |  |  |  |  |  |
| UN130. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator may measure lengths and angles on an x-ray image |  |  |  |  |  |  |
| UN131. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN132. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN133. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall view DDR on primary display with less than 0.5s delay, from radiation to frame display. |  |  |  |  |  |  |
| UN134. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator shall update the software system without sending the device back to the manufacturer |  |  |  |  |  |  |
| UN135. |  |  |  | SOFTWARE SYSTEM AND PLATFORM | #REF! | Operator may submit notes or dictations into image study record via voice dictation |  |  |  |  |  |  |
| UN136. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Manufacturer should be able to track location of device such that repossession is possible |  |  |  |  |  |  |
| UN137. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Manufacturer shall certify the use of the device for 4 years of operation, with an initial, complimentary limited warranty of 1 years with optional upgrade up to 7 years and 3 years service life |  |  |  |  |  |  |
| UN138. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Manufacturer shall acquire usage metrics from the device (how often user/facility is interacting with the device) |  |  |  |  |  |  |
| UN139. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Facility should receive a reshoot rate or analysis for imaging sessions for the past 3 months (Quarterly) |  |  |  |  |  |  |
| UN140. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Facility should have access to all images taken within 2 months; both submitted to PACS and rejected/reshoots |  |  |  |  |  |  |
| UN141. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Facility may have access to session and patient dosage accumulation from the past 2 months |  |  |  |  |  |  |
| UN142. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Manufacturer should be able to remotely disable send-to-"PACS/USB/Provider" functionality if contract obligations are not met |  |  |  |  |  |  |
| UN143. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Manufacturer should be able to push OTA updates to adjust functionality (Max kV/mAS and turn DDR ON/OFF) |  |  |  |  |  |  |
| UN144. |  |  |  | BUSINESS, INVESTOR, AND MARKETING | #REF! | Operator shall use the device in the United States, Canada, Mexico, European Union, and United Kingdom |  |  |  |  |  |  |
| UN145. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN146. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN147. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN148. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN149. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN150. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN151. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN152. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN153. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN154. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN155. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN156. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN157. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN158. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN159. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN160. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN161. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN162. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN163. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN164. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN165. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN166. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN167. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN168. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN169. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN170. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN171. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN172. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN173. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN174. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN175. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN176. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN177. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN178. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN179. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN180. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN181. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN182. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN183. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN184. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN185. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN186. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN187. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN188. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN189. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN190. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN191. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN192. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN193. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN194. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN195. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN196. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN197. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN198. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN199. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN200. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN201. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN202. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN203. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN204. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN205. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN206. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN207. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN208. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN209. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN210. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN211. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN212. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN213. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN214. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN215. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN216. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN217. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN218. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN219. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN220. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN221. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN222. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN223. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN224. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN225. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN226. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN227. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN228. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN229. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN230. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN231. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN232. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN233. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN234. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN235. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN236. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN237. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN238. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN239. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN240. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN241. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN242. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN243. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN244. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN245. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN246. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN247. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN248. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN249. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN250. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN251. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
| UN252. |  |  |  |  | #REF! |  |  |  |  |  |  |  |
