# PLN-P01-024 Rev E: MX1 Software Development Plan

## Metadata
- Document ID: PLN-P01-024
- Revision: E
- Prefix: PLN
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: PLN-P01-024 - MX1 Software Development Plan_E-signed.docx
- Source path: Example QMS - MedAI/PLN-P01-024 - MX1 Software Development Plan_E-signed.docx
- Extraction warnings: none

## Extracted Content
PURPOSE
This document contains the Software Development Plan for the Software System of the MX1 Portable X-Ray System, hereforth referenced as MX1 Software System, and the MX1 MedAI Device App.
SCOPE
This plan defines all software design and development activities conducted when producing software incorporated within the MX1 Software System and MX1 MedAI Device App only.
REFERENCES
IEC 62304 - Medical device software - Software life cycle processes (2015)
FDA Guidance for the Content of Premarket Submissions for Device Software Functions (2023)
DHF-P01-008 - MX1 Design History File Checklist
PLN-P01-066 - MX1 Security Management Plan
QSP-020 - Software Life Cycle Development
QSP-002 - Document Control
QSP-018 - Design Control
QSP-019 - Risk Management
WI-005 - Engineering Change Management
QSP-003 - Complaint Handling
QSF-035 - SOUP Monitoring Record
DEFINITIONS
Design History File (DHF): A compilation of documentation which describes the design history of a finished medical device
Software of Unknown Provenance (SOUP): Software item that is already developed and generally available and that has not been developed for the purpose of being incorporated into the medical device (also known as “off-the- shelf software”) or Software Item previously developed for which adequate records of the development processes are not available
Software Unit: Software Item that is not subdivided into other items
Software Item: Any identifiable part of a computer program, i.e., source code, object code, control code, control data, or a collection of these items
Software System: An Integrated collection of Software Items organized to accomplish a specific function or set of functions
Software Component: For the MX1 Software System, the largest/highest-level Software Items that comprise the Software System
Anomaly: Any condition that deviates from the expected based on requirements specifications, design documents, standards, etc. or from someone’s perceptions or experiences. Anomalies may be found during, but not limited to, the review, test, analysis, compilation, or use of medical device software or applicable documentation
SOFTWARE DEVELOPMENT PROCESS
The MX1 Software System development life cycle is comprised of the following activities:
Software Development Planning
Software Requirements Analysis
Software Architecture Design
Software Detailed Design
Software Risk Analysis
Software Unit Implementation
Software System Testing
Software Regression Testing
Software Configuration Management
Software Release
This plan shall address the process and/or methods by which each activity shall be achieved for successful release and incorporation into the MX1 System.
Software Safety Classification
The Software System designed for this product can contribute to a hazardous situation that may result in a non-serious injury as defined in IEC 62304. Therefore, the Software System has a safety classification of Class B. As a result, this plan describes the level of detail to which all document deliverables shall meet as required of Class B software.
All deliverables resulting from these development activities and corresponding tasks shall be documented and maintained in the DHF.
Software Requirements Analysis
User needs shall be the first requirements established at the start of a new medical device product. From these, product requirements are defined. Refer to QSP-018 - Design Controls for further details regarding the process for user needs and product-level requirements generation as well as the corresponding deliverables.
From this, software system-level requirements are developed to define the software project work that shall be completed to achieve the device system requirements. Software component-level requirements, interface requirements, and/or requirement dependencies may also be defined during this activity. These requirements shall be evaluated and revised as appropriate throughout the development process.
The deliverable from this activity is the Software Requirement Specifications document.
Software Architectural Design
Alongside the Software Requirements Analysis activity, a high-level architecture design shall be constructed to best address the functional and performance requirements for the overall MX1 Software System. Any architecture-based risk controls should be detailed during this activity.
Smaller, more focused architecture diagrams of the MX1 Software Components may also be constructed. The diagrams may illustrate the following:
Integration of Software Components within the MX1 Software System
Integration of Software Items within Software Components
Integration of SOUPs within the Software System
Interfaces between various Software Items and Components
Interfaces between Software Items and SOUPs
Interfaces between Software Items and software products external to the device
In conjunction with the Software Risk Analysis, the overall safety classification shall be confirmed during this activity.
The deliverable from this activity is the Software Architecture Document.
Software Detailed Design
The objective of the Software Detailed Design is to specify how the software requirements defined during the Software Requirements Analysis shall be implemented. Design details may include or be supplemented by flow and sequence diagrams, references to custom interface specifications, and/or other documentation highlighting implementation details.
The deliverable from this activity is the Software Design Specification document.
Software Unit Implementation
Coding guidelines may be developed to record best and/or preferred practices to produce consistent and readable code.
Software Units should be evaluated during code analysis activities in part to determine if the code meets these guidelines. Code and design analysis should be conducted to confirm implementation of software requirements and detailed design.
Further details of code analysis activities and corresponding deliverables shall be outlined in the Software Verification and Validation Plan.
VERIFICATION ACTIVITIES
Software Unit Verification
For the purposes of Software Unit Verification, unit testing may be applicable for certain Software Components in order to test safety-critical features. Unit testing shall primarily be conducted on Class B Software Units. Unit testing may also selectively be conducted on custom interfaces and Class A Software Units as appropriate.
A unit test plan shall include test procedures evaluated for adequacy and established acceptance criteria for Software Units. Further details about unit testing activities and corresponding deliverables from this activity shall be defined in the Software Verification and Validation Plan.
Software System Testing
Once the Software Units meet the acceptance criteria defined as part of the Software Unit Implementation and Verification activities, they shall be integrated into the Software System at one time. The Software System shall then be embedded into the physical device.
Thorough testing of the Software System shall be performed at the device level. This testing shall validate the system as a whole and implicitly revalidate the software-hardware integration.
A system-level test plan shall include test procedures evaluated for adequacy, expected outcomes, and established acceptance criteria. System-level testing shall ensure all critical software requirements defined during the Software Requirement Analysis activities are met. Requirements may be tested individually in isolation or in combinations, especially if dependencies between requirements exist.
Any anomalies found during Software System testing shall be documented and evaluated by the Software Problem Resolution Process.
Further details about the software system-level test plan and corresponding deliverables from this activity shall be defined in the Software Verification and Validation Plan.
Note: Any Software Integration testing deemed necessary shall be combined with Software System Testing activities.
Regression Testing
Regression testing may be conducted on occasion when new Software Units and/or Items are integrated with the overall Software System or when certain changes are made to the existing Software System.
Further details about conduction regression testing and corresponding deliverables from this activity shall be defined in the Software Verification and Validation Plan
SOFTWARE RELEASE
The Software System shall be pre-released to configuration management prior to certain verification and validation activities. Software System testing shall be completed before the software is released for installation onto the final device.
As part of the Software Problem Resolution Process, a list of Unresolved Anomalies may be created. Unresolved Anomalies are the issues that do not present a high-severity risk and may not be addressed in a current Software System release. They shall be documented as part of the Software Release Notes.
Additional deliverables from Software Release planning include the labeled software version (see Software Configuration Management Process) and software delivery procedure. Further details shall be defined in the Software Configuration Management Plan.
SOFTWARE MAINTENANCE PROCESS
The objective of this process is to determine how the software should be maintained as defined in QSP-020 - Software Development Lifecycle. SOUP monitoring may be conducted quarterly as part of the overall Software Maintenance Process. At a minimum, SOUP monitoring shall be conducted at least once per year.
When necessary, MedAI shall provide maintenance services to customers should SOUP monitoring result in updates to a released Software System.
CYBERSECURITY
The MX1 Security Management Plan is documented as PLN-P01-066. A Cybersecurity Assessment will be performed to identify software characteristics that could have an impact on security. The results of the Security Assessment should aid in evaluating the software via a Cybersecurity Risk Analysis. From this, a risk-based approach shall be utilized to determine if any additional cybersecurity controls need to be added to the device for risk mitigation.
Cybersecurity testing (e.g. penetration testing) shall be conducted to verify that the implemented software security controls are effective and to determine if additional controls need to be considered.
The following activities will be completed and documented throughout the device lifecycle:
Pre-Release Cybersecurity Activities
Cybersecurity Risk Analysis
The Cybersecurity Risk Analysis shall be conducted according to the Software Risk Analysis process, in accordance with WI-001, Risk Analysis and Evaluation. All hazards and potential mitigations shall be identified and documented. Security risk analysis will be documented separately from the overall MX1 Risk Analysis.
Safety, Security, and Privacy Requirements
Safety, security, and privacy requirements shall be recorded in the Software Requirement Specification deliverable.
Cybersecurity Traceability Matrix between Safety, Security, and Privacy Requirements and Cybersecurity Risk Analysis
This traceability activity shall be recorded within the overall Software Traceability Matrix.
Software Integrity Prior to Device Release/Distribution
Integrity check controls shall be put in place to ensure the released software loaded onto the device maintains its integrity during production of the device. For the device components, a hash of critical application and configuration files are recorded at build time and verified prior to installation to ensure the device integrity is not compromised during production. In the case of an unauthorized change, the integrity check will fail and disallow operators from using the device until remediation by authorized MedAI personnel.
The integrity check controls shall include checks for the following:
Application software modification since build
Service account (Linux user account, permissions or password) changes
Changes to host-based firewall rules
Application software configuration changes
Cybersecurity Labeling for User-facing Cybersecurity Processes
MedAI will provide instructions and/or recommendations regarding security and privacy in the MX1 device Instructions for Use (IFU) as they pertain to the usage of the MX1 system.
Post-Release Cybersecurity Activities
Software Integrity Maintenance Plan
Integrity check controls shall be put in place to ensure the released software loaded onto the device maintains its integrity. For the device components, checksums are checked at each device startup, as well as hourly while in operation, to ensure the device integrity is maintained and modifications have not occurred. In the case of an unauthorized change, the integrity check will fail and disallow operators from using the device until remediation by authorized MedAI personnel.
The integrity check control shall include checks for the following:
Application software modification since build
Service account (Linux user account, permissions or password) changes
Changes to host-based firewall rules
Application software configuration changes
Cybersecurity Vulnerability Identification and Risk Assessment
Cybersecurity vulnerabilities are monitored and identified through the Software Problem Resolution Process described in Section 13 of this plan. Inputs to this process include postmarket complaint data and periodic SOUP monitoring per QSP-020.
If vulnerabilities are identified, risk assessment will be conducted in accordance with Guidance for Industry and FDA Staff - Postmarket Management of Cybersecurity in Medical Devices (2016). The risk associated with the vulnerability will be categorized as either controlled (acceptable) or uncontrolled (unacceptable). For risks that remain uncontrolled, additional remediation should be implemented. Even when risks are controlled, MedAI may choose to deploy an additional control(s) as part of a “defense-in-depth” strategy. Typically, these changes would be considered a cybersecurity routine update or patch, a type of device enhancement.
Uncontrolled risk is present when there is unacceptable residual risk of patient harm due to insufficient risk mitigations and compensating controls. In assessing risk, manufacturers should consider the exploitability of the vulnerability and the severity of patient harm if exploited. For uncontrolled risks, MedAI will:
Remediate the vulnerabilities to reduce the risk of patient harm to an acceptable level.
If fixing the vulnerability may not be feasible or immediately practicable, MedAI will identify and implement risk mitigations and compensating controls to adequately mitigate the risk.
Customers and the user community will be provided with relevant information on recommended controls and residual cybersecurity risks so that they can take appropriate steps to mitigate the risk and make informed decisions regarding device use.
MedAI must report these vulnerabilities to the FDA according to 21 CFR part 806, unless reported under 21 CFR parts 803 or 1004, unless circumstances per Guidance for Industry and FDA Staff - Postmarket Management of Cybersecurity in Medical Devices (2016) VII. B. are met.
Vulnerability communications to customers for Uncontrolled Risks will occur within 30 days. Vulnerability communications to users will include:
A description of the vulnerability including an impact assessment based on the manufacturer’s current understanding.
A statement that manufacturer’s efforts are underway to address the risk of patient harm as expeditiously as possible.
A description of compensating controls, if any.
A statement that MedAI is working to fix the vulnerability, or provide a defense-in-depth strategy to reduce the probability of exploit and/or severity of harm, and will communicate regarding the availability of a fix in the future.
Software Update Delivery Plan
Software updates identified through the Software Problem Resolution Process and Software Maintenance Process shall be conducted by authorized MedAI employees. Software updates are served through Mender, a secure cloud-based software delivery service. The user shall be notified of an available update in the MX1 App. The notification requires a user confirmation before proceeding with the update.
SOFTWARE RISK ANALYSIS
Device Hazard Analysis and Risk Assessment of the device shall be conducted in accordance with QSP-019 - Risk Management Procedure. Any potential causes of the Software System that may contribute to a hazardous situation shall be identified and documented in the Risk Management File. The resulting risk assessment shall be used to evaluate and confirm the safety classifications of the overall Software System and of the individual Software Components, including those that contain SOUPs. Any risk mitigations arising from this activity shall be traced to a corresponding software requirement.
This Risk Assessment shall be reviewed and updated as the system and software requirements are updated.
SOFTWARE TRACEABILITY PROCESS
All software requirements defined during the Software Requirements Analysis should be traceable back to an overall system design requirement, which shall be linked to a user need. The requirements shall also be traceable to detailed design specifications, software verification activities and outputs, and any risk control measures resulting from the Software Risk Analysis.
The deliverable for this activity shall be the Software Traceability Matrix.
SOFTWARE CONFIGURATION MANAGEMENT PROCESS
Configuration Items
Configuration Items include custom Software Components, SOUPs, and any supporting items that impact the functionality of the Software System.
Prior to verification and validation activities, Configuration Items shall be placed under Configuration Management Control. The Configuration Items and their versions shall be documented.
Software Change Control
Further risk analysis, verification, and/or validation activities may be required if modifications are made to a Software System released as per the Software Release process.
Updates to the source code, verification and validation testing, and/or software documentation shall be recorded using QSF-033 - Engineering Change Request (ECR) and evaluated using the Change Control process outlined in WI-005 - Engineering Change Management.
Configuration Status Accounting
Software changes made as part of the development process shall be traced via the revision level commit history. Changes made as part of Software Change Control shall be documented and maintained by QSF-033 - Engineering Change Requests (ECR). The Software Release Notes shall document changes and unresolved anomalies for a new software release.
The Software Configuration Management Process shall be further defined in the Software Configuration Management Plan.
SOFTWARE PROBLEM RESOLUTION PROCESS
Anomalies may be discovered from external complaint analysis, the Software Maintenance Process, or during verification and validation activities. All anomalies shall be documented as part of the Software Problem Resolution Process.
Complaint Handling
Problems externally reported to MedAI shall be reported using QSF-012 - Complaint Record and evaluated using QSP-003 - Complaint Handling.
Software Maintenance Process
QSF-035 - SOUP Monitoring Record should be used to document problems arising from the Software Maintenance Process.
Software System Verification and Validation
Prior to conducting system-level verification activities, the Software System shall be placed under configuration management control. Any anomalies found after this activity shall be recorded as part of the test result documentation.
Anomalies shall be evaluated using a risk-based approach to determine if anomaly resolution is required. Justification for leaving anomalies unresolved shall be recorded in the document deliverable corresponding to the method by which they were discovered (i.e. Complaint Handling, Software Maintenance Process, or verification and validation activities). The remaining anomalies may be categorized based on the impact they may have on the system, operator, and/or patient.
Any resulting changes that need to be made to released software, software documentation, or verification and validation testing shall be handled via the Change Control process defined under Configuration Management.
For each Software System released, a Software Release Notes shall disclose any anomalies resolved with that version (excluding release of the initial Software System), as well as any remaining Unresolved Anomalies.
ACTIVITIES AND RESPONSIBILITIES
The following matrix lists the software development activities discussed in this plan and their respective document deliverables. The matrix also includes deliverables generated specifically for software with a Moderate Level of Concern as per the FDA Guidance for the Content of Premarket Submissions for Device Software Functions (2023.
The creation of the deliverables listed for the Software Unit Implementation, Software Unit Verification, and Software System Testing mark a milestone that indicates that the corresponding verification activity has been completed and that the following activity may be initiated.
Refer to  QSP-018 - Design Controls for the Responsible Departments and Required Approvers of the deliverables listed below.
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Process/Activity | Deliverables |
| --- | --- |
| Software Development Planning | Software Development Plan |
| Software Requirements Analysis | Software Requirements Specification (SRS) |
| Software Architectural Design | Software Architecture Document (SAD) |
| Software Detailed Design | Software Design Specification (SDS) |
| Software Unit Implementation | The deliverables from this activity shall be detailed in the Software Verification and Validation Plan |
| Software Unit Verification | The deliverables from this activity shall be detailed in the Software Verification and Validation Plan |
| Software System Testing | The deliverables from this activity shall be detailed in the Software Verification and Validation Plan |
| Software Release | Labeled Software Version |
|  | Software Release Notes |
| Software Maintenance Process | SOUP Monitoring Record |
| Cybersecurity | Cybersecurity Risk Analysis |
|  | Safety, Security, and Privacy Requirements shall be documented in the Software Requirement Specifications |
|  | Traceability between cybersecurity-related requirements and risk control measures shall be documented in the Software Traceability Matrix |
|  | Software Update Delivery Plan |
|  | Software Integrity Maintenance Plan |
|  | Device Instructions in the IFU for any user-facing cybersecurity processes |
| Software Risk Analysis | Software Risk Analysis |
| Software Traceability Process | Software Traceability Matrix |
| Software Configuration Management Process | Software Configuration Management Plan |
| Deliverables for the FDA Guidance for the Content of Premarket Submissions for Device Software Functions | Documentation Level Evaluation - A statement indicating the Documentation Level and a description of the rationale for that level |
|  | Software Description - A summary overview of the features and software operating environment |
|  | Software Development, Configuration Management, and Maintenance Practices - Summary of life cycle development plan and complete configuration management and maintenance plan document(s) OR A Declaration of Conformity to the FDA-recognized version of IEC 62304, including subclause 5.1 (Software development planning), clause 6 (software maintenance process), and clause 8 (software configuration management process), among others as applicable |
|  | Software Version History - A history of tested software versions including the date, version number, and a brief description of all changes relative to the previously tested software version. |
|  | Unresolved Software Anomalies - List of remaining unresolved software anomalies with an evaluation of the impact of each unresolved software anomaly on the device’s safety and effectiveness. |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Mgmt Rep Software Engineering Quality Engineering Regulatory Affairs | 02 May 2022 | 22-102 |
| B | Updates to Cybersecurity section to include more details about Software Update Delivery Process and Software Integrity Maintenance Plan | Software Engineering Quality Engineering Regulatory Affairs | 01 May 2023 | 23-143 |
| C | Changes to Activities and Responsibilities section to reflect document deliverable updates as per 2023 FDA Guidance for the Content of Premarket Submissions for Device Software Functions | Software Engineering Quality Engineering Regulatory Affairs | 21 Sep 2023 | 23-240 |
| D | Addition of references, including new Security Management Plan PLN-P01-066 | Software Engineering Quality Engineering Regulatory Affairs | 15 Mar 2024 | 24-112 |
| E | Addition of hosted Mender service for software updates | Software Engineering Quality Engineering Regulatory Affairs | 29 May 2024 | 24-328 |
