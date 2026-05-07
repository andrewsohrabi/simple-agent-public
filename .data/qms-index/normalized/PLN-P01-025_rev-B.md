# PLN-P01-025 Rev B: MX1 Software Configuration Management Plan

## Metadata
- Document ID: PLN-P01-025
- Revision: B
- Prefix: PLN
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: PLN-P01-025 - MX1 Software Configuration Management Plan_B-signed.docx
- Source path: Example QMS - MedAI/PLN-P01-025 - MX1 Software Configuration Management Plan_B-signed.docx
- Extraction warnings: none

## Extracted Content
PURPOSE
This document contains the Configuration Management Plan for the MX1 Software System and MX1 MedAI Device App.
SCOPE
This plan is applicable to all software design and development activities that produce software incorporated within the MX1 Software System and MX1 MedAI Device App only.
OTHER APPLICABLE DOCUMENTS
IEC 62304 - Medical device software - Software life cycle processes
PLN-P01-024 - P01 Software Development Plan
MEMO-P01-634 - MX1 Software Descriptions
MEMO-P01-640 - MX1 SOUP Descriptions
QSP-003 - Complaint Handling
QSF-033 - Engineering Change Request (ECR)
DEFINITIONS
Configuration Item (CI): An entity that can be uniquely identified at a given reference point
Version: An identified instance of a configuration item
Release: Particular version of a configuration item that is made available for a specific purpose
Design History File (DHF): A compilation of documentation which describes the design history of a finished medical device
Software of Unknown Provenance (SOUP): Software item that is already developed and generally available and that has not been developed for the purpose of being incorporated into the medical device.
Software Unit: Software Item that is not subdivided into other items
Software Item: Any identifiable part of a computer program, i.e., source code, object code, control code, control data, or a collection of these items
Software System: An Integrated collection of Software Items organized to accomplish a specific function or set of functions
Software Component: For the MX1 Software System, the largest/highest-level Software Items that comprise the Software System
Anomaly: any condition that deviates from the expected based on requirements specifications, design documents, standards, etc. or from someone’s perceptions or experiences. Anomalies may be found during, but not limited to, the review, test, analysis, compilation, or use of medical device software or applicable documentation
CONFIGURATION ITEMS
MX1 Software System
The MX1 Software System, consisting of all Software Components, shall have one unique version number.
The MedAI Device App is currently a component of the MX1 Software System and therefore will reflect the same version number as the overall software system.
The Software Components described in  MEMO-P01-634 - MX1 Software Descriptions shall be considered individual Configuration Items. Each Component shall have its own unique identifier.
The versioning guidelines for both are defined in the Configuration Item Versioning Conventions section of this plan.
SOUPs
Each SOUP used by and/or integrated within the MX1 Software System shall be considered an individual Configuration Item and shall have a version number or other unique identifier determined by the SOUP manufacturer. All detailed descriptions of the SOUPs shall be documented and maintained in  MEMO-P01-640 - MX1 SOUP Descriptions.
Supporting Items
Additional supporting items that impact the functionality of the MX1 Software System, including tools, items, or settings, may also be considered individual Configuration Items. Supporting items may include, but are not limited to: configuration files, calibration utilities, calibration scripts, test protocols, and test scripts.
These Items and their versions shall be documented and maintained in the DHF.
CONFIGURATION ITEM VERSIONING CONVENTIONS
Each MX1 Software Component shall be versioned and uniquely identifiable by git commit hashes. Other custom MedAI Configuration Items may be versioned in a similar manner.
The MX1 Software System shall be identified by a 3-digit version number using the “major.minor.patch” convention. Major updates shall indicate significant functionality that has been added to the software. Minor updates shall indicate improvements or fixes made to the existing functionality. Patches shall indicate non-user-facing system feature updates or fixes. Depending on the type of update, each digit of the version number shall increment by 1 as shown in the following table:
If a pre-released version of the MX1 Software System is used for internal or third party verification activities, the 3-digit version number should be appended with a Greek letter (e.g. v1.2.3-alpha). The appended letter should be incremented following the order of the Greek alphabet for ensuing pre-releases.
All features added or modified in the pre-released versions of the MX1 Software System shall be tested during verification activities against the final MX1 Software System release.
CONFIGURATION MANAGEMENT PROCESS
Source Code Version Control
During the software development process, MedAI shall adopt the Git Flow process as a source code version control method. All code repositories shall be stored in Github.
Git feature branches should be named in a way to provide traceability to a particular software design input. The process to merge a feature branch with a development branch begins with a developer initiating a pull request. After a review of the changes by Software Engineering, the feature branch should be accepted into the development branch if approved.
Software Development Environment
Any tool choice and build environment setup instructions shall be documented and maintained in a README file for each Software Component. The README files are stored alongside the relevant source code.
Tool changes and version updates shall be evaluated using a risk-based approach to determine verification and validation requirements.
Software Release Management
All Configuration Items shall be placed under Configuration Management control prior to the software and system verification and validation process. As part of this process, the controlled software components’ development branches shall be moved to release branches.
At the time of verification, the Continuous Integration server shall build a set of release files, including binaries with a unique build number, and any required auxiliary configuration files (if applicable). The files shall be placed in an access-controlled location on Google Cloud Platform until release per QSP-002 Document Control.
All software used in the operation and production of the MX1 System shall be verified and then validated prior to the software’s release. Once verification and validation activities have been completed, the documents and files referenced in this section shall be included in a QSF-033 - Engineering Change Request (ECR) for review to obtain approval for the release of a new version of the MX1 Software System.
The final software release ECR shall include a Software System Release document. This document will contain the Configuration Items and their unique identifiers. After release, the software files shall be stored in the controlled Quality Management System drive.
Software Change Control
If any part of the MX1 Software System is updated while under Configuration Management control, the changes shall be reviewed via risk management activities defined in QSP-019 - Risk Management prior to further software verification and validation activities, which may include regression testing.
Any resulting changes that need to be made to any Configuration Item, software documentation, or verification and validation testing shall be managed per QSP-002 Document Control and documented using a QSF-033 - Engineering Change Request (ECR).
Software Archival
Any software releases and additional Configuration Items prior to the latest release shall be archived for at least the lifetime of the device. Permissions to access the previous release versions should be restricted. Any related documentation shall follow document archival processes as defined in QSP-002 - Document Control.
CONFIGURATION STATUS ACCOUNTING
Software changes made prior to initial software release as part of the development process shall be traced via the revision level commit history. Changes made after initial release as part of Software Change Control shall be documented and maintained by QSF-033 - Engineering Change Requests (ECR). The Software Release Notes for each version shall detail changes that resulted in a new software release.
DOCUMENT REVISION HISTORY

### Table 1
| Versioning Convention (major.minor.patch) | Example version number: v1.2.3 |
| --- | --- |
| Major | v1.2.3 to v2.0.0 - Adding new functionality |
| Minor | v1.2.3 to v1.3.0 - User-facing functionality improvement |
| Patch | v1.2.3 to v1.2.4 - Bug fix or non-user facing change, speed performance enhancement, or cybersecurity improvement |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Management Representative Software Engineering Quality Engineering Regulatory Affairs | 24 Mar 2023 | 22-180 |
| B | Change of P01 references to MX1; Addition of pre-release version conventions in 5.4 CONFIGURATION ITEM VERSIONING CONVENTIONS | Engineering Quality Engineering Regulatory Affairs | 18 Apr 2024 | 24-135 |
