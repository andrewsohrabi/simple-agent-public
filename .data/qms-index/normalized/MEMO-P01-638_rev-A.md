# MEMO-P01-638 Rev A: MX1 Software Development Configuration Management and Maintenance Practices

## Metadata
- Document ID: MEMO-P01-638
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-638 - MX1 Software Development Configuration Management and Maintenance Practices_A-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-638 - MX1 Software Development Configuration Management and Maintenance Practices_A-signed.docx
- Extraction warnings: none

## Extracted Content
PURPOSE
This document contains a summary of the software development lifecycle processes, configuration management, and maintenance practices for the software system within the MX1 Portable X-ray System.
SCOPE
The summaries of the following software plans and/or lifecycle processes are applicable to all design and development activities that produce software incorporated within the MX1 Portable X-ray System.
SUMMARY OF SOFTWARE DEVELOPMENT LIFECYCLE
QSP-020 - Software Development Lifecycle is applicable to all software designed and developed for use in any medical device produced by MedAI. For the purposes of the procedure, software includes firmware and other means for software-based control of the medical device.
Software Design and Development Activities
Software Requirements Analysis
Development of all software shall begin with Software Requirements Analysis. Generated from User Needs, system-level requirements are established to define the system, technical, functional, and user requirements that the end device should perform. From these requirements, software requirements are developed to define the portions of the system requirements fulfilled by software.
Software Architectural Design
The objective of this activity is to define the major structural components of the software, their externally visible properties, and the relationships among them based on the software requirements. Additionally, the architecture should be a logical representation of the software to be implemented. Segregation of major components and/or software items as a result of risk control measures should be demonstrated with the output of this activity.
Software Detailed Design
The objective of this activity is to refine the software items and interfaces defined within the Software Architecture Design into detailed descriptions, which specify how the software requirements are to be implemented.
Software Traceability Analysis
Software requirements defined during the Software Requirements Analysis activity should be traceable back to an overall system design requirement. The requirements shall also be traceable to detailed design specifications, software verification activities and outputs, and any risk control measures resulting from Software Risk Analysis.
Software Verification and Validation
Once software development is completed, it shall be placed under Configuration Management control. The final output shall be verified and validated in order to determine if the software built meets software system and overall device requirements.
Verification should be conducted starting with software unit, integration, and system testing. Integration and system-level testing may be combined as appropriate. If any anomalies are found during these activities, they shall be evaluated via the Software Problem Resolution Process.
The MX1 Software System shall be embedded onto the physical device for system-level testing. The software shall be implicitly validated during system testing if the device meets all system design requirements recorded in a Design Record for the MX1 Portable X-ray System.
Unresolved Anomalies
The objective of this activity is to document identified but unresolved anomalies and their effect on device performance.
A list of unresolved anomalies, if any exist, shall be included as part of the output for the Software Release (see Software Release Management in the Configuration Management section).
Software Risk Management Process
The Software Risk Management Process includes activities for identifying hazards, estimating and evaluating the associated risks, controlling the risks, and monitoring the effectiveness of the control. The software risk analysis is a subset of the overall Risk Management of the device. Software Risk Management activities shall occur throughout the entire development life cycle.
Software Problem Resolution Process
All anomalies found as a result of verification and validation activities or the Software Maintenance Process shall be documented as part of the Software Problem Resolution Process.
Any changes that must occur within released software to resolve a problem shall be handled as an Engineering Change Request defined as part of Device Modifications within QSP-018 - Design Control. Additionally, the Software Configuration Management processes shall be appropriately implemented during software-based Engineering Change Requests.
Any changes made to the software shall be reviewed against previous testing to determine if new verification and validation testing must be conducted. Any changes to software documentation or testing as a result of the software change shall be included as part of the input to the Engineering Change Request.
Trends related to the Software Problem Resolution Process shall be monitored and evaluated according to appropriate work instructions.
CONFIGURATION MANAGEMENT
Configuration Items
Configuration Items include custom MX1 Software Components, SOUPs, and any supporting items that impact the functionality of the MX1 Software System.
Prior to verification and validation activities, all software Configuration Items shall be placed under Configuration Management Control. The Configuration Items and their versions shall be documented.
Software Release Management
All Configuration Items shall be placed under Configuration Management control prior to the software and system verification and validation process..
At the time of verification, a set of software artifacts (e.g. software system image, binaries, applicable auxiliary configuration files, etc.) shall be versioned and released for testing. The files shall be placed in an access-controlled location on Google Cloud Platform until release per QSP-002 Document Control.
All software used in the operation and production of the MX1 System shall be verified and then validated prior to the software’s release. Once verification and validation activities have been completed, the documents and files referenced in this section shall be included in a QSF-033 - Engineering Change Request (ECR) for review to obtain approval for the release of a new version of the MX1 Software System. After release, the software files shall be stored in the controlled Quality Management System drive.
Software Change Control
Any changes made to the MX1 Software system shall be reviewed via risk management activities defined in QSP-019 - Risk Management prior to conducting necessary software verification and validation activities, which may include regression testing.
Any resulting changes that need to be made to any Configuration Item, software documentation, or verification and validation testing shall be managed per QSP-002 Document Control and documented using a QSF-033 - Engineering Change Request (ECR).
Configuration Status Accounting
Software changes made as part of the development process shall be traced via the revision level commit history. Changes made as part of Software Change Control shall be documented and maintained by QSF-033 - Engineering Change Requests (ECR). A Software System Release document shall contain changes and unresolved anomalies for a new software release.
The Software Configuration Management Process is further defined in PLN-P01-025 - Software Configuration Management Plan.
SOFTWARE MAINTENANCE PROCESS
As part of the Software Maintenance Process, all SOUPs shall be monitored periodically for updates.
SOUP monitoring may be conducted quarterly. At a minimum, SOUP monitoring shall be conducted at least once per year.
MedAI shall assess the impacts of SOUP updates and monitor ongoing development of SOUP components for critical safety, cybersecurity, and information security updates. A SOUP Monitoring Record shall be used to document this activity. Any SOUP update that mandates a software change shall be considered an anomaly.
All anomalies shall be handled by the Software Problem Resolution Process. Any software updates shall be managed through the Software Configuration Management Process.
Deliverables in the Software Maintenance Process shall be any additional documentation created or changed as a result of implementing the modification as required by the Software Problem Resolution Process.
DOCUMENT REVISION HISTORY
Digital Key:
example.com/

### Table 1
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 18 Apr 2024 | 24-156 |
