# PLN-P01-060 Rev A: MX1 Project Plan

## Metadata
- Document ID: PLN-P01-060
- Revision: A
- Prefix: PLN
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: unknown
- Source filename: PLN-P01-060 - MX1 Project Plan_A-Obsolete.docx
- Source path: Example QMS - MedAI/PLN-P01-060 - MX1 Project Plan_A-Obsolete.docx
- Extraction warnings: none

## Extracted Content
PURPOSE
This document describes the revised project plan for the MX1 Portable X-Ray System (“MX1”), Project P01. MX1 is the next evolution device of the current Imager Medical Imaging System, P00. There have been considerable updates to the project timeline and scope since the release of the original project plan, PLN-P01-023.
BACKGROUND
MedAI, Inc. (“MedAI”) submitted a 510(k) (K231372) to the FDA in April 2023 for the MX1 device. On February 6, 2024, the FDA issued a “not substantially equivalent” (NSE) letter to MedAI, which explained that the Agency did not find the MX1 device to be substantially equivalent to devices marketed in interstate commerce prior to May 23, 1976. Following receipt of this letter, MedAI decided to pursue a new 510(k) submission designed to specifically address the Agency’s concerns. Because the original 510(k) submission was made with an outdated design of MX1, and because a new indication for fluoroscopy will be included in the new submission, a new DHF was generated. Released documents that are not impacted by the changes will be used in the new DHF (i.e., BOMs, drawings). Refer to DHF-P01-008, MX1 Design History File Checklist.
REFERENCES
FDA 510(K) K231372, MX1 Portable X-ray System
PLN-P01-023 Rev. B, MX1 Project Plan
PLN-P01-027 Rev. A, K1 Cart Project Plan
PLN-P01-032 Rev. A, G1 Foot and Ankle Imaging System Project Plan
PLN-P01-044 Rev. B W1 Wireless Charger Project Plan
PLN-P01-061 Rev. A, MX1 Regulatory Plan
QSP-018 Rev. G, Design Control
DESIGN OVERVIEW
Sections 4.2 and 4.3 summarize the current design of MX1 and its accessories.
MX1 Portable X-Ray System
Emitter (E1)
The emitter is an optionally handheld or stand-mounted battery-powered device that is capable of emitting single shot X-rays and DDRs up to 80 kV and 2 mAs. The device has a joystick style handle with two triggers for both downward and forward imaging orientations. The X-ray source includes a monoblock that is composed of an X-ray tube and high voltage supply in a potted assembly. Collimation is achieved both automatically through a motorized system and manually through use of collimation pucks. The battery may be charged either through a wired connection or inductively. The emitter has a 3”x3” display, or “Viewfinder”, to assist the user in taking X-rays. The front clip allows the user to attach the emitter to mounting accessories. Indicator LEDs will also be visible to relay to the user the status of the system.
Cassette (C1)
The cassette consists of a 9” x 9” square detector encased in a plastic enclosure. The detector is offset within the enclosure to allow for imaging of hard to reach areas where the cassette border would cause interference. IR LEDs enable the tracking system to orient the emitter and cassette. Indicator LEDs will also be visible to relay to the user the status of the system. The cassette has a small display and two USB-C ports for wired charging and data transfer. The cassette has a detachable handle for transportation and mounting; an additional mount on the back of the device allows the user to attach the cassette to accessories.
Wired Charger (H1)
Two isolated USB-C charging bricks (AC/DC Converters) will be provided with the device. The charging brick will be a IEC 60601-1 compliant device that operates on the USB-C Power Delivery (PD) protocol. The isolated chargers will charge the cassette and emitter while also keeping the device isolated from Mains.
Wireless Foot Pedal (F1)
The wireless foot pedal consists of 2 pedals and 2 buttons. The device is battery powered via replaceable batteries. The pedals and buttons allow the user to change between modes, trigger X-ray emissions, rotate images, and favorite images via a wireless connection. Battery status is shown via Indicator LED at the top of the device.
Collimation Pucks & Pediatric Filter
The Collimation Pucks may be used to collimate the X-ray beam to very small field sizes. Each puck can be identified by the two bolded numbers at the end of each puck’s part number. The Pediatric Filter adds the necessary amount of filtration to safely image pediatric patients.
MedAI Device App (MedAI APP)
The MedAI Device App (“ODA”) will allow a tablet or monitor to be connected to the MX1 System via a wireless connection to display the primary UI. It will be available from the Google Play store for devices with Android version 10 and up.  Apple iOS may be supported in future software versions.
Accessories
MX1 is designed to work with a host of accessories to augment the device for its various markets and use cases. Several accessories are in development; each accessory has a unique DHF.
Cart (K1)
The mobile cart is intended for use in the storage, transport, positioning, and use of MX1. The cart includes a touchscreen DICOM monitor and provides means to charge the MX1 system. The cart also has a jointed arm for positioning the emitter.
Wireless Charger (W1)
The wireless charging holster or dock allows the user to place the emitter into the charger to wirelessly charge the device. The clip securely holds the emitter and will allow an operator to move and manipulate the emitter around as needed. The charger may be used as part of K1, G1, or L1 for emitter mounting for hands-free use.
Foot and Ankle System (G1)
The Foot and Ankle System provides a hands-free method of positioning the Emitter and Cassette for weight-bearing views of the foot and ankle.
Lab Kit (L1)
The Lab Kit is a table mounted arm that supports the emitter and tablet for hands free use. The Lab Kit is not intended to diagnose, treat, cure, or prevent any disease.
Tablet (T1)
The Tablet is a Samsung Galaxy S8+ that has been tested to meet DICOM requirements for diagnostic viewing with the MedAI Device App.
NEW DESIGN FEATURES
An additional fluoroscopy mode will be added that is capable of capturing fluoroscopic radiographs with the maximum technique factors of 64 kV and 0.08 mAs.
Additional safety features in software to support Radioscopically Guided Interventional Procedures, including but not limited to:
High Level Control: The system will limit the maximum air kerma rate at the patient entrance reference point to 176 mGy/min or less.
Additional dose calculations and calculation outputs, including Radiation Dose Structured Reports
The inclusion of a “Irradiation Disabling Switch” in the software UI
An Emergency Radioscopy Mode that allows the operator to recover critical functions before all device functions are available
New audible warning sounds and alarms
The software system will include the ability to invert images.
PROJECT TEAM RESOURCES AND FUNCTIONAL INTERFACING
Table 1: Project Team, Roles and Responsibilities
PROJECT AND DESIGN/DEVELOPMENT PROCESS MANAGEMENT
There are 5 major phases within the project plan. Moving from each phase requires passing through a project milestone, termed a phase gate. This is outlined, and all phase gates delineated in QSP-018 Design Control. Phase definitions and high level descriptions are included in QSP-018 and are summarized below in Section 6.2 for reference. QSP-018 also outlines authorship and approval responsibilities for all deliverables by function.
The project milestones and target completion date are listed in Table 2: Project Milestones.
Design History File (DHF) Checklist
All high level deliverables are outlined in the QSF-066 DHF Checklist.  Activities are grouped by Phase. Design History File Checklist number DHF-001 has been established for the MX1 Portable X-Ray System. Documentation references for applicable deliverables shall be added to the DHF checklist prior to closure of each phase.
Major Phase Definitions
Phase 1 - Initiation & Planning
The objectives of the Initiation and Planning Phase is to identify the scope of work, and determine a path forward for the remainder of the project. The clinical need and overall technical strategy is defined in this phase for creating a solution, as well as evaluation of the initial feasibility of a design concept. Design and Development planning ensures that the design process is controlled and that device quality objectives have been identified.
Phase 2 - Design
The objective of the Design Phase is to define the product design inputs & outputs, and establish initial risk analysis. The ultimate output of Phase 2 is a completed design ready for design freeze.
Inputs in this phase are the physical and performance requirements that are used for the basis of the design. This may include safety, customer related, and regulatory requirements, as well as functional, performance, interface, and material requirements. These requirements are documented and may be updated throughout the project.
Any design changes made after closure of Phase 2 must be assessed for impact on completed deliverables; assessments shall be documented and referenced in the Phase 3 and Phase 4 deliverables sections of the DHF checklist as applicable. All design changes from this point forward must also go through an Engineering Change Request per WI-005 - Engineering Change Request.
Phase 3 - Verification and Validation
The objective of this phase is to verify that the product meets design input requirements and demonstrate that the user needs and intended use requirements of the device have been satisfied by the device design. In this phase, devices are tested to ensure that all requirements identified in Phase 2 have been met. This phase also includes clinical evaluation deliverables and release of final device labeling.
Phase 4 - Design Transfer and Pre-Launch
The objective of this phase is to demonstrate that the device manufacturing process has been established and produces units that meet product specifications. This phase ensures that regulatory clearances/approvals and all other project deliverables are in place prior to release of product for human use.
Phase 5 - Commercialization and Postmarket Surveillance
Phase 5 constitutes formal exit from the design control process and triggers the beginning of the post production product life cycle. This phase requires continuous re-evaluation of risk based on the addition of new information from various sources such as production activities and customer feedback. Postmarket Surveillance activities are managed per WI-004, Post Market Surveillance.
Phase Reviews
For this project, phase reviews will occur individually for phases 1-4. Quality Engineering is responsible for DHF audits prior to moving to the next phase. Completion of DHF audits will be documented via Quality approval on the QSF-066 DHF checklist.
Project Schedule and Major Milestones
A project schedule has been established and an overview of major milestone dates is provided in Table 2 below.
Note: Because this Plan describes an iteration of the MX1 Project, the Project Milestones do not begin with a project kickoff.
Table 2: Project Milestones
Project Tracking and Task Management
Task management for individual design control deliverables will be tracked in the MedAI, Inc. instance of Monday.com through daily, weekly, and monthly meeting cadences.
Design Review Meetings (in addition to the phase reviews required per QSP-018) will be held as needed to review the design progress.
FUNCTIONAL PLANNING
Project Management
The project lead works collaboratively with Engineering, Quality and Regulatory to ensure that the MX1 product is functional, compliant, and satisfies the market needs. The project lead also manages the project schedule and ensures completion of the design control deliverables.
Quality Engineering
Quality Engineering, in conjunction with mechanical and electrical engineering, will establish component inspection requirements for all new MX1 device components in accordance with QSP-026, Statistical Techniques, and the MX1 DFMEA. All new suppliers will be assessed per QSP-007, Supplier Management, and those selected will be added to the approved supplier list in the Supplier Assessment Log, QSR-004. Quality Engineering will assist Project Management with management of design control deliverables to ensure all deliverables per QSP-018 are completed for the MX1 project.
MedAI does not intend to manufacture MX1 at the MedAI facility. Process validation, including the generation of a master validation plan, will be completed by the contract manufacturer. Quality Engineering will work with the contract manufacturer selected to perform an on-site audit prior to the start of MX1 production and distribution. A quality plan will be established between MedAI and the contract manufacturer to define change notification requirements and the QMS responsibilities for each party.
MedAI has been registered by Intertek, an MDSAP recognized auditing organization, as conforming to the requirements of ISO 13485:2016 for the United States, Australia, and Canada. MedAI’s management certification is applicable to the design and development, manufacture, and service of non-sterile x-ray equipment and software for the area of radiology. MedAI’s certificate is valid through February 23, 2026.
An updated Quality Plan for MX1 can be found in PLN-P01-062. The updated MX1 Risk Management Plan is documented as PLN-P01-063.
Regulatory Affairs
The MX1 device will require submission of a new traditional 510(k) in the United States. Because primary product code MAI is eligible for third party 510(k) review, MedAI plans to submit the 510(k) for review through Regulatory Technology Services (RTS). Submitting marketing applications for non-U.S. geographies is not in-scope for the MX1 product at this time. A detailed Regulatory Plan that includes submission timelines, language requirements, and applicable product standards can be found in PLN-P01-061.
Design Engineering
MedAI Engineering department consists of Mechanical, Electrical, X-Ray, Human Factors, and Software teams. Each engineering team is dedicated to the design and development of medical products that are compliant to applicable standards. The Hardware Team consisting of the Mechanical, Electrical, and X-Ray groups work to ensure compliance with all relevant IEC 60601 standards and the Software Team is responsible for compliance to IEC 62304. All parties perform Verification and validation exercises.
MANUFACTURING STRATEGY AND PLAN
Production Ramp Rate
To meet the business and sales requirements the following production ramp rate is planned:
2024
Q4: 50 units
2025
250 units
2026
500 units
Contract Manufacturer
MedAI has chosen Sanmina Corporation, Inc to manufacture the MX1 Portable X-ray System. The following qualifications were used to evaluate contract manufacturers. Sanmina met all the required and preferred qualifications.
The Operations team in collaboration with the Design Engineering team and Quality Engineering  will perform design transfer to Sanmina in Design Phase 4.
Infrastructure Requirements
Verification/validation units will be produced at the MedAI facility located at 100 Main St, Ste 700, Springfield, IL 60001. These units will be produced by members of the Design Engineering team in collaboration with Quality Engineering and Operations. MedAI may produce pilot production units at the MedAI facility to verify manufacturing work instructions. No new infrastructure, equipment, or resources are anticipated for building these units.
If MedAI decides to conduct full-scale production in-house, instead of using a contract manufacturer, then additional infrastructure and resources will be required. It is estimated at least 1,000 sq ft would be required to produce 250 units per quarter. Additionally, 500 sq ft for receiving/storage, and 500 sq ft for assembly of accessories (2,000 sq ft in total). Additional staff would be required, including potentially 1 full time QA engineer, and 3-4 full time assembly personnel.
MedAI plans to initially perform servicing in-house, then transfer servicing to Sanmina. MedAI estimates that 250 sq ft will be required for servicing. MedAI has sufficient space to set up a servicing area at this time.
MARKETING STRATEGY
The MX1 Portable X-Ray System will be marketed and sold in the following environments and countries.
Markets of Interest
Specialty Clinics
Clinical Labs
Hospital (non-operating rooms)
Hospital Emergency Department (ER)
Countries of Interest
USA
Training Requirements
TRA product training documents will be updated as applicable to include the MX1 device in their scope, or new documents may be created specific to MX1. The MX1 device will require end user and distributor training prior to device use (end customer) or placement with distributors. Training will be documented for each customer site and distributor.
DOCUMENT REVISION HISTORY
example.com/

### Table 1
| Responsible Member | Role | Responsibilities |
| --- | --- | --- |
| Hayden Beck (CEO) Reese Nash (COO) | Executive Management | Responsible for providing all departments with the resources necessary for successful completion of the project. Reviews project status and provides input during Phase reviews. Communicates project status to external stakeholders as applicable. |
| Devon Marsh (VP, Program Management) | Executive Management | Responsible for coordinating the project and its interdependencies, driving deliverable completion, and analyzing program risks. |
| Alex Hartman (Director, Engineering) | Technical Product Owner (TPO) | Responsible for managing the project schedule and interfacing with all teams to ensure completion of design control deliverables. |
| Eden Jameson (QE) Cameron Rivera (QE) | Quality (QA) | Responsible for authoring quality deliverables per QSP-018, including the quality plan and risk analysis documentation. Collaborates with Human Factors Team to determine usability deliverables. General review of design control deliverables for completeness and accuracy. Owns design phase closures. |
| Logan Howard (QA V&V) | Quality (QA) | Ensures that MX1 meets specified requirements and performs its intended functions without causing harm or errors. Works alongside the product development team and drives the verification and validation efforts. |
| Dhruv Vishwakarma(Director, Regulatory Affairs) | Regulatory (RA) | Responsible for all regulatory submissions/new marketing applications for MX1 (e.g. authorship of the 510(k), EU Technical File, Canadian license application, etc). Review of cross functional deliverables as it relates to compliance to regulatory standards. Authors regulatory plan and any associated regulatory strategy documentation. Review of labeling materials and labeling requirements in conjunction with the Enterprise Solutions team. |
| Quinn Becker(HF Team Lead) | Human Factors (HF) | Responsible for interfacing with customers and users on behalf of the engineering team. Develop user needs, develop product requirements, perform usability studies, and integrate MX1 into customer workflows and IT systems. |
| Drew Lane (ID Team Lead) | Industrial Design (ID) | Responsible for developing the concepts for manufactured product, considering requirements from engineering, user needs, sales/marketing, and management. |
| Drew Renton (ME Team Lead) | Product Development - Mechanical  Engineering | Responsible for all principal mechanical system design and V&V support. Including system packaging, enclosure design, and thermal design. |
| Keivan Darius(EE Team Lead) | Product Development - Electrical Engineering | Responsible for all principal electrical system design and V&V support. Including high voltage design, circuits and PCBA design, and embedded system firmware design. |
| Riley Compton(X-Ray Team Lead) | Project Development - X-Ray Science | Responsible for all principal X-ray system design and V&V support. Including X-ray emission, detection, and radiation safety. |
| Gage Carr(Device Software Team Lead) | Product Development - Software Engineering | Responsible for all principal software system design and V&V support. Including back-end and front-end device software design and tablet app design. |
| Mo Khosravanipour (Director, Operations) | Operations | Manages supply chain and oversees communication between MedAI and Sanmina, the MX1 Contract Manufacturer. Collaborates with Quality to facilitate design transfer. Drives process improvement efforts to increase efficiency. |
| Tatum Olsen (Sr. Director, Sales) | Sales | Responsible for managing internal sales representatives and external distributors. Leads customer acquisition efforts and generates revenue via sales. |
| Stevie Heath (Director, Marketing) | Marketing | Responsible for overseeing marketing strategies. |

### Table 2
| Project Milestones | Target Date |
| --- | --- |
| Phase 1 Review | March 2024 |
| Phase 2 Review | May 2024 |
| Phase 3 Review | June 2024 |
| Phase 4 Review | October 2024 |

### Table 3
| REQUIRED QUALIFICATIONS |
| --- |
| ISO 13485 FDA Registered Facility OSHA Radiation Certified (or willingness to become certified) Capacity for 20 units/mo in the first year, and 40 units/mo in the second year. Capacity to grow with future product lines. Purchasing/Receiving/Inspection of Components Inventory management and warehousing Experience with process validation (e.g. IQ/OQ/PQ, master validation plan) Willing to be audited by external regulatory agencies (e.g. FDA, notified bodies) relative to MedAI products. Willing to be audited by MedAI as part of supplier qualification and maintenance activities. Note: ISO 13485 strongly preferred over 9001. Especially if outsourcing any design activities, as mentioned below, it would be a must-have. |
| PREFERRED QUALIFICATIONS |
| Previous experience with X-ray emitting devices In-house Servicing & Maintenance Design Engineering Team - Assist with component swapping documentation / risk assessment / recommendations (as required) Process Engineering Team - Process improvement and value engineering (cost reduction) PCBA manufacturing (fabrication and assembly) Wire harness fabrication Injection molding fabrication Experience being audited by FDA and EU notified bodies, etc. Experience being audited by ETL, SGS, and UL for product certification |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Engineering Executive Mgmt Regulatory Affairs Sales Marketing | 15 Mar 2024 | 24-110 |
