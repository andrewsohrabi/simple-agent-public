# PLN-P01-066 Rev D: MX1 Security Management Plan

## Metadata
- Document ID: PLN-P01-066
- Revision: D
- Prefix: PLN
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: PLN-P01-066 - MX1 Security Management Plan_D.docx
- Source path: Example QMS - MedAI/PLN-P01-066 - MX1 Security Management Plan_D.docx
- Extraction warnings: none

## Extracted Content
Purpose
The purpose of this document is to describe how cybersecurity risks shall be managed throughout the development lifecycle. It describes the plan to monitor, identify, and address, as appropriate, in a reasonable time, postmarket cybersecurity vulnerabilities and exploits. It describes the format and method for producing the software bill of materials. Finally, it describes how and when postmarket updates and/or patches to the device and related systems will be made.
The plan is meant to comply with 2023 FDA Guidance “Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions” and 2016 FDA Guidance “Postmarket Management of Cybersecurity in Medical Devices”.
Scope
The contents of this document is relevant to the architecture of the MX1 Software System v3.0.0 and above.
References
AAMI TIR57 “Principles for medical device security—Risk management”
NIST 2018 “Framework for Improving Critical Infrastructure Security”, v1.1.
2005 FDA Guidance “Cybersecurity for Networked Medical Devices Containing Off-the-shelf (OTS) Software”
2021 MITRE “Playbook for Threat Modeling Medical Devices”
2023 FDA Guidance “Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions”
Common Vulnerability Scoring System Version 3.1: Specification Document
Cybersecurity Risk Management
Security Risk Identification Methods
The following risk identification methods shall be followed and documented within RSK-P01-011 Rev B MX1 Security Risk Assessment.
Answering a subset of the “Questions that can be used to identify medical device security characteristics” derived from AAMI TIR57 “Principles for medical device security—Risk management”.
Vulnerability Monitoring of the SBOM
Threat Modeling following STRIDE
Threat modeling shall capture cybersecurity risks introduced through the supply chain, manufacturing, deployment, interoperation with other devices, maintenance/update activities, and decommission activities that might otherwise be overlooked in traditional safety risk assessment processes.
These methodologies were selected because they are inline with the AAMI TIR57 “Principles for medical device security—Risk management” conformance standard and also are consistent with the guidance in the 2021 MITRE “Playbook for Threat Modeling Medical Devices” (which was developed along with the FDA specifically for medical devices).
Security Risk Verification
Cybersecurity risk control measures shall be designed, developed, and verified following PLN-P01-065 MX1 Verification and Validation Plan.
Security Risk Likelihood Levels
Security Risk Severity (Impact) Levels
¹Confidential data is defined as data that is protected by data privacy standards, such as HIPAA or the GDPR.
²The following sources are considered essential data:
Device image files
Device image metadata
Device configuration
Device logs
Device source code
Encryption keys and authentication credentials
Patient Identifying Information (PII)
Security Risk Acceptability Levels
Security Risk Acceptability Matrix
Security Risk Overall Acceptability Criteria
Residual risks are evaluated by the same method and with the same criteria for risk acceptability as the initial risks. The residual risk will be determined to be acceptable or unacceptable. When unacceptable, further risk control options should be investigated. If further risk control is not practicable, a benefit-risk analysis shall be performed and the result documented.
Methods of Obtaining Post-Production Information
Post-production safety feedback shall be collected using these activities:
Collecting feedback from customers
Collecting problem reports/complaints from customers
Monitoring the MAUDE Alerts database
Secure Software Development
Development Environment Security & Secure Coding Standards
See Section 9. Cybersecurity of PLN-P01-024 Rev E MX1 Software Development Plan.
Vulnerability Testing
All OTS software shall be automatically searched in the National Institute of Standards and Technology’s (NIST) National Vulnerability Database (NVD) and Google’s Open Source Vulnerability Database (OSV) to find security vulnerabilities. All “High” and “Medium” severity findings shall be addressed.
Vulnerability Scanning for Self-Hosted OTS
MedAI utilizes a limited number of self-hosted off-the-shelf (OTS) tools, including BindPlane. To mitigate any potential security risks these tools may introduce, MedAI will conduct regular vulnerability scans on these tools and their environments every 30 days. These scans will be performed using the same tools and vulnerability databases applied to OTS tools operating on the MX1 Device, ensuring consistent security standards. Any vulnerabilities identified as “High” or “Medium” severity will be remediated within 90 days.
Penetration Testing
Internal penetration testing shall be performed to ensure controls are effective and to find any exploitable weaknesses in the software system. Third-party penetration testing may be performed as per the discretion of the Device Software and Quality Assurance teams.
Software Bill of Materials
A software bill of materials (SBOM) shall be produced for every publicly released version of the device. The SBOM shall follow the SPDX 2.3 file format. The SBOM shall include the direct and indirect software dependencies from the device.
Vulnerability Management
Known vulnerabilities shall be identified for the MedAI MX1 Software System as follows: This is in accordance with the guidance in sections V.A.4.B and VI.B of the September 2023 document “Cybersecurity in Medical Devices.”
All OTS components used in the system will be identified in the SBOM, and unique identifiers (CPEs, PURLs) will be located for each.
Vulnerabilities will be identified by searching the MITRE CVE List, NVD CPE Dictionary, Snyk Vulnerability Database, and OSV Database. Automated searches will be conducted using the NVD CPE API and OSV API, with results compiled into a CSV file.
The open-source vulnerability scanner "Bomber" will be used to scan for missed vulnerabilities, and any found will be added to the CSV.
All CVEs will be cross-referenced with CISA’s "Known Exploited Vulnerabilities Catalog."
Vulnerabilities will be reviewed to ensure applicability to the MedAI MX1 Software System, with misidentified CVEs removed. Applicable vulnerabilities will be assessed for risk and potential mitigations.
Continuous monitoring for new vulnerabilities will be set up using OSV and NVD databases for all released software versions.
Identification
Vulnerabilities alerts should be reviewed by an engineer once a quarter. The record of these reviews shall be stored in the resolution comments of the vulnerability alerts.
During the vulnerability reviews, every alert will be evaluated according to two metrics:
Its CVSS severity (Critical, High, Moderate, and Low), as indicated on the alert
The Safety-Severity Level of existing Risks identified in the RSK-P01-011 Rev B MX1 Security Risk Assessment.
The risk associated with the alert shall then be evaluated as “Controlled” or “Uncontrolled” according to the following matrix:
If there is no existing Risk in the RSK-P01-011 Rev B MX1 Security Risk Assessment that can be used to determine the Safety-Severity Level, then the engineer must notify the appropriate Project Leader who will coordinate with the Quality Assurance team to update the RSK-P01-011 Rev B MX1 Security Risk Assessment as appropriate. The vulnerability alert shall not be resolved until the new risks have been added as appropriate.
Alerts with a CVSS severity of “Medium” or “Low” may be ignored if there’s no chance of Catastrophic safety risk.
Alerts with a CVSS severity of “Critical” or “High” must be resolved once either:
It’s determined that the alert is not relevant. The “dismissal comment” must explain why the vulnerability won’t lead to a safety risk.
The vulnerability fix has been entered into the issue tracking system with an appropriate priority set to ensure the vulnerability is addressed. The “dismissal comment” must include a link to the issue id so that the fix can be traced.
Addressing Vulnerabilities
All “Uncontrolled” risks must be addressed and patches released as quickly as possible and within 30 days of the vulnerability being identified.
“Controlled” risks with a “Critical” or “High” CVSS severity level should typically still be addressed and the update released as part of the normal update schedule.
All other risks will be addressed based on their prioritization and available resources.
The resolution of security issues, including fixes, patches, or mitigations implemented, will be thoroughly documented.
Cybersecurity Metrics
The following cybersecurity metrics shall be calculated using automated and regularly scheduled manual methods:
Percentage of identified vulnerabilities that are updated or patched (defect density)
Duration from vulnerability identification to when it is updated or patched (mean time to remediate)
The “duration from when an update or patch is available to complete implementation in devices deployed in the field” metric, which is suggested by the 2023 FDA Guidance “Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions”, is not applicable as the software is hosted in the cloud by the manufacturer.
These metrics shall be reviewed on an annual basis.
Cybersecurity Instructions and Labeling
Device instructions for use and product specifications related to recommended cybersecurity controls appropriate for the intended use environment (e.g., anti-virus software, use of firewall) shall be included in the Instructions for Use or other customer facing documents (e.g., Release Note).
Information relating to vulnerabilities which have been determined to have the potential for creating unacceptable risk shall be communicated to users no later than 30 days after such determination has been made. The information will at a minimum describe the vulnerability and identify interim compensating controls (as applicable).
Software Updates
Validated software updates shall be developed following the PLN-P01-024 Rev E MX1 Software Development Plan which includes a mechanism for patch releases. See section 9.2.3. Software Update Delivery Plan of PLN-P01-024 Rev D MX1 Software Development Plan for details on how new updates and patches will be built.
The update process shall be evaluated as part of the threat modeling and shall be documented with RSK-P01-011 Rev A MX1 Security Risk Assessment.
Incident Response Plan
Incident response is divided into several phases: Preparation, Detection and Analysis, Containment Eradication and Recovery, and Post-Incident Activity. The following diagram describes how the phases are related to each other.
We describe each of the phases in more detail below.
Preparation
Preparation is intended to prevent security incidents before they happen. The following activities will be performed as a part of this phase:
Threat modeling
Security risk assessment
Vulnerability testing
Penetration testing
Static and dynamic code analysis
Detection and Analysis
Detection and Analysis is intended to alert and triage incidents as they happen. The following activities will be performed to detect incidents:
Logging of critical activities like login attempts
Logging of when external media is connected to the device
Security audit logs between different components of the device
Centralized logging and alerting system
Monitoring of security incidents occurring in CSPs.
Incidents will be categorized by the engineering team according to the following:
Containment Eradication and Recovery
This phase is for addressing and recovering from the security incident. The following activities may be performed at the discretion of the engineering team to eliminate security incidents.
Restore device configuration from old backups
Issue a patch release
Turn off certain features of the device
Rebooting or resetting the device
Decommission of device
Post-Incident Activity
This phase is for discussing lessons learned during the incident and ensuring better preparation in the future. It may include the following activities.
Post-incident retrospective meeting
New security controls
Updating organization procedures
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Label | Method of Access | Authentication Level | Nature of Vulnerabilities |
| --- | --- | --- | --- |
| Improbable (1) | Persistent physical access required | Multiple independent authentications required | Highly specialized conditions and equipment are required |
| Remote (2) | Temporary physical access required | Multi-factor authentication | Multiple conditions or equipment are required |
| Occasional (3) | Private network access required | Single factor | Specialized conditions required |
| Probable (4) | Local network or radio-frequency proximity required | Simple single factor (e.g., a short password or PIN) | Simple conditions required |
| Frequent (5) | Remote access required | None required | N/A |

### Table 2
| Label | Potential Impact to Device Operations | Potential Impact to Business Operations | Potential Impact to Data | Potential Impact to Other Organizations or the Environment |
| --- | --- | --- | --- | --- |
| Negligible (1) | No negative effects | No negative effects | No negative effects | No negative effects |
| Minor (2) | Insignificant loss of non-essential functions; temporary loss of device operation resulting in inconvenience | Customer complaints or limited damage to reputation | Loss of confidentiality of non-confidential data¹ | Device reveals information about itself beyond what is necessary |
| Moderate (3) | Significant loss of availability across a single or system communication can be tampered with | Loss of intellectual property; large damage to business reputation; minor legal issues | Loss of integrity of non-essential data²; loss of confidentiality of a small amount of confidential data¹ | Device becomes a vector to gather information about other systems on the same network |
| Serious (4) | Significant loss of availability across multiple devices | Serious legal issues; inability to perform critical mission functions | Loss of integrity of essential data²; loss of confidentiality of a large amount of confidential data¹ | Device becomes a vector to attack other systems |

### Table 3
| Label | Description |
| --- | --- |
| Acceptable | The security risk level is acceptable. |
| Attempt to Mitigate | The risk level is acceptable, but risk control measures that are feasible to implement within time and cost constraints should be implemented. |
| Unacceptable | The risk is unacceptable and must be mitigated. If it can’t be mitigated further, then a risk/benefit-analysis must be performed. |

### Table 4
| Likelihood \ Severity | Negligible (1) | Minor(2) | Moderate (3) | Serious (4) |
| --- | --- | --- | --- | --- |
| Frequent (5) | Acceptable | Attempt to Mitigate | Unacceptable | Unacceptable |
| Probable (4) | Acceptable | Attempt to Mitigate | Attempt to Mitigate | Unacceptable |
| Occasional (3) | Acceptable | Acceptable | Attempt to Mitigate | Attempt to Mitigate |
| Remote (2) | Acceptable | Acceptable | Acceptable | Attempt to Mitigate |
| Improbable (1) | Acceptable | Acceptable | Acceptable | Attempt to Mitigate |

### Table 5
| CVSS \ Safety-Severity Level | Negligible (1) | Minor (2) | Moderate (3) | Serious (4) | Catastrophic (5) |
| --- | --- | --- | --- | --- | --- |
| Critical | Controlled | Controlled | Uncontrolled | Uncontrolled | Uncontrolled |
| High | Controlled | Controlled | Controlled | Uncontrolled | Uncontrolled |
| Medium | Controlled | Controlled | Controlled | Controlled | Uncontrolled |
| Low | Controlled | Controlled | Controlled | Controlled | Controlled |

### Table 6
| Category | Definition |
| --- | --- |
| None | No effect to the organization’s ability to provide all services to all users |
| Low | Minimal effect; the organization can still provide all critical services to all users but has lost efficiency. |
| Medium | Organization has lost the ability to provide a critical service to a subset of system users. |
| High | Organization is no longer able to provide some critical services to any users. |

### Table 7
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Engineering Regulatory Affairs | 26 Apr 2024 | 24-194 |
| B | Updated Vulnerability Management Section | Quality Engineering Engineering Regulatory Affairs | 29 May 2024 | 24-316 |
| C | Updated the Vulnerability Management Section to include information about SBOM generation and MedAI’s vulnerability scanning process. Added Incident Response Section to describe how MedAI will respond to cyber security events. | See ECR-574 |  |  |
| D | Added Vulnerability Scanning for Self-Hosted OTS section | See ECR-602 |  |  |
