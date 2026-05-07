# ECR-587 Rev A: MX1 SW v4.0.1

## Metadata
- Document ID: ECR-587
- Revision: A
- Prefix: ECR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: v4.0
- Source filename: ECR-587 - MX1 SW v4.0.1_A-signed.docx
- Source path: Example QMS - MedAI/ECR-587 - MX1 SW v4.0.1_A-signed.docx
- Extraction warnings: none

## Extracted Content
ECR-587 - MX1 SW v4.0.1_A-signed
Sheet: Engineering Change Request
Sheet: Release Packet Decision Flow Ch
[Empty sheet]
Sheet: Regulatory Assessment
Sheet: Document Revision History
Sheet: Variables

### Table 1
|  | ENGINEERING CHANGE REQUEST |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | ECR# | ECR-587 | Product: | MX1 | Effective Start Date: | Immediately |  |
|  |  |  | Originator | Gage Carr (SW) | End Date (if applicable): | N/A |  |
|  | 1. RELEASE IS FOR: |  |  |  |  |  |  |
|  | False | Design & Development | True | V&V | False | Release to Production &Device Master Record Index (DMRI) |  |
|  |  |  |  |  |  | 510(k) (if applicable): |  |
|  | Skip to section 4 & 5 |  | Skip Section 3 |  | Fill sections 2 through 5 |  |  |
|  | 2. POST DESIGN FREEZE CHANGES |  |  |  |  |  |  |
|  | TYPE OF CHANGE: |  |  | CHANGE AFFECTS: |  |  |  |
|  | False | Change to existing Drawing(s) |  | False | Product in the Field |  |  |
|  | True | Change to existing Software/Firmware(s) |  | True | Product in Inventory |  |  |
|  | True | Change to existing BOM(s) |  | False | Product in Work in Progress |  |  |
|  | False | Change to existing Work Instruction(s) or Test Procedure(s) |  | False | Components/Material in Inventory |  |  |
|  | False | Other: |  | False | Tools/Fixtures in Manufacturing |  |  |
|  |  |  |  | False | Open Purchase Order(s) |  |  |
|  | REASON FOR CHANGE: |  |  | False | Other: |  |  |
|  | True | Product Improvement |  |  |  |  |  |
|  | False | Cost Reduction |  | IMPACT |  |  |  |
|  | False | Supplier Change |  | False | Incorporate into product in the field |  |  |
|  | False | Part Replacement (with Alternate or Equivalent Replacement) |  | True | Incorporate into product in inventory |  |  |
|  | False | Reliability/Performance Issue   (Corrective Action) |  | True | Incorporate for next build of product |  |  |
|  | False | Safety Issue   (Corrective Action) |  | False | Incorporate for next build of product once existing components/ have been used up |  |  |
|  | False | Response to NCR (include NCR# & Affected lot / qty): |  | False | Scrap Components/Material |  |  |
|  | False | Other: |  | False | Rework Components/Material |  |  |
|  |  |  |  | False | Incorporate as a temporary deviation for a specified order |  |  |
|  |  |  |  | False | Incorporate as a temporary deviation for a specified time or serial/lot number(s): |  |  |
|  |  |  |  | False | Other: |  |  |
|  | DESCRIPTION OF CHANGE |  |  |  |  |  |  |
|  | Changes to the ToF calibration process to fix a race condition that caused ToF calibration failure and fixed ownership of a /opt/medai/data/mysql to fix remote OS updates. |  |  |  |  |  |  |
|  | REASON FOR CHANGE |  |  |  |  |  |  |
|  | An issue was found with folder ownership, which caused remote updates to break. Additionally, issues were found with the ToF calibration process. |  |  |  |  |  |  |
|  | Is new Verification and/or Validation testing required? |  |  |  |  |  |  |
|  | False | Yes | If "Yes", provide the VVPR number(s) and any additional doc references: |  |  |  |  |
|  | True | No | If "No", provide justification: For ownership changes to /opt/medai/data, only directory restructuring and permission updates occurred, with no changes to functionality, logic, or user interfaces. Since these are non-functional infrastructure corrections with minimal risk and have been reviewed by MedAI engineers, no additional validation or verification is needed. Similarly, for the collimator-firmware fix, the change ensures uninterrupted Lidar TOF calibration by ignoring extra ICD commands, with no impact on system performance or behavior, thus no further V&V is required. |  |  |  |  |
|  | False | N/A | If "NA", provide justification: |  |  |  |  |
|  | Are changes to inputs or outputs of risk management required? |  |  |  |  |  |  |
|  | False | Yes | If "Yes", provide the updated Risk Document(s) and any additional doc references: |  |  |  |  |
|  | True | No | If "No", provide justification: There are no changes to the device's functionality or to existing risk mitigations. |  |  |  |  |
|  | False | N/A | If "NA", provide justification: |  |  |  |  |
|  | 3. REGULATORY ASSESSMENT |  |  |  |  |  |  |
|  | True | Regulatory Affiars to Complete Regulatory Impact Assessment in "Regulatory Assessment" Tab |  |  |  |  |  |
|  | 4. WORK INSTRUCTIONS |  |  |  |  |  |  |
|  | TRAINING: |  |  |  |  |  |  |
|  | Does this ECR include Manufacturing Work Instruction(s) (MWI) |  |  |  |  |  |  |
|  | False | Yes, select required training level per QSP-008 and QSR-003: |  |  |  |  |  |
|  |  | False | Level 1 (cannot be selected for initial release) |  |  |  |  |
|  |  | False | Level 2 |  |  |  |  |
|  |  | False | Level 3 |  |  |  |  |
|  | True | No, this ECR does not contain Manufacturing Work Instructions |  |  |  |  |  |
|  | 5. DOCUMENTS AFFECTED |  |  |  |  |  |  |
|  | Drawing Number / Doc Number & Description |  |  | New Revision (Include link to each document) |  | Temporary Link to Folder with all Files:  (folder to be deleted & files dispersed in electronic signatures folder once ECR has been signed) |  |
|  | BOM-101 - MX1 Portable X-ray System MAI, MX1 Software System v4.0.1 |  |  | A |  |  |  |
|  | S10002 - MX1 Emitter firmware (EM) |  |  | v4.0.1 |  |  |  |
|  | S10003 - MX1 Monoblock firmware (MB) |  |  | v4.0.1 |  |  |  |
|  | S10004 - MX1 Collimator firmware (COL) |  |  | v4.0.1 |  |  |  |
|  | S10005 - MX1 Cassette firmware (CAS) |  |  | v4.0.1 |  |  |  |
|  | S10006 - MX1 Foot pedal firmware (FP) |  |  | v4.0.1 |  |  |  |
|  | S10008 - MedAI Device App (ODA) - Android, .apk |  |  | v4.0.1 |  |  |  |
|  | SS-10036 - MX1 Emitter Jetson OS Image |  |  | v4.0.1 |  |  |  |
|  | SS-10037 - MX1 Cassette Jetson OS Image |  |  | v4.0.1 |  |  |  |
|  | ** The following updated software components are included in the software artifact delivered for SS-10036: |  |  |  |  |  |  |
|  | S10071 - MX1 Jetpack 5 OS (JO) |  |  | v4.0.1 |  |  |  |
|  | ** The following updated software components are included in the software artifact delivered for SS-10037: |  |  |  |  |  |  |
|  | S10071 - MX1 Jetpack 5 OS (JO) |  |  | v4.0.1 |  |  |  |
|  | MEMO-P01-766 - Resolution to Bugs Involving MariaDB Directory |  |  | A |  |  |  |
|  | MEMO-P01-765 - Changes to the MX1 Lidar TOF Calibration Sequence |  |  | A |  |  |  |
|  | DOCUMENT APPROVALS |  |  |  |  |  |  |
|  | Revision | DCO # | Description | Approved By | Eff. Date | Digital Key |  |
|  | A | 24-599 | Initial Release | Quality EngineeringEngineeringRegulatory Affairs | 2024-10-14 00:00:00 | example.com/ |  |

### Table 2
| DESIGN AND PROCESS CHANGE REVIEW | YES | NO | If yes, provide rationale, supporting documentation, etc. | COUNTRY/REGION | IMPACT |  | REGULATORY IMPACT ASSESSMENT - Include rationale and supporting data as applicable. Include regulatory change assessment flowchart pathways and/or key decisions points. |  |  | APPLICABLE GUIDANCES |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Is the modification to directly correct any device failure in the field or clinical study? | False | True |  | United States and United States Territories | False | Letter to File | These changes do not impact cybersecurity or other software on the device. The changes were made to fix an issue that caused ToF calibration to fail and fixed ownership of a /opt/medai/data/mysql to allow the remote OS update capability to function properly. The changes did not introduce new risks, modify existing risks, or change risk controls. The changes had no impact on function or performance |  |  | Guidance for Industry and FDA Staff - Deciding When to Submit a 510(k) for a Change to an Existing Device (2017) |
| Does the modification require clinical evaluation to determine if the device remains safe and effective? | False | True |  |  | False | Traditional 510(k) |  |  |  | Guidance for Industry and FDA Staff - Deciding When to Submit a 510(k) for a Software Change to an Existing Device (2017) |
| Is there a change to device performance, intended use, product labeling/IFU, or marketing claims? | False | True |  |  | False | Special 510(k) |  |  |  |  |
| Is the change to add a new manufacturing facility? | False | True |  |  | True | No Impact - ECR Assessment Only |  |  |  |  |
| Is the change to add new equipment or tools that directly affects the finished device? | False | True |  | Canada | False | License Amendment | The MX1 is currently not approved or marketed in Canada. |  |  | Guidance for the Interpretation of Significant Change of a Medical Device (2011) |
| Does the change add a new specification or test method or otherwise provide additional assurance of identity, strength, or reliability of the device? | False | True |  |  | False | Letter to File |  |  |  |  |
| Is there a change to a vendor, purchased components, or material? | False | True |  |  | True | No Impact - ECR Assessment Only |  |  |  |  |
| Is there a change in manufacturing method or process parameters? | False | True |  | Australia | False | Substantial Change Notification | The MX1 is currently not approved or marketed in Austrailia. |  |  | Changes affecting TGA-issued conformity assessment certificates (2021) |
| Is there a change to sterilization or shelf life? | False | True |  |  | False | Letter to File |  |  |  |  |
| Does the change affect the risk analysis? | False | True |  |  | True | No Impact - ECR Assessment Only |  |  |  |  |
| Is there an update to the GUDID database required? | False | True |  | Other: |  |  |  |  |  |  |

### Table 3
| MedAI, Inc. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QSF-033 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Engineering Change Request Form |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Issued By: | Mgmt Rep |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| DOCUMENT REVISION HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Revision | DCO # | Approved By | Description | Eff. Date | Digital Key |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A | 18-145 | Executive MgmtMgmt Rep | Initial Release | 2018-12-12 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B | 19-024 | Executive MgmtMgmt Rep | Revision to add software/firmware change type, revise documents affected section to include version number within both revision columns and source code to drawing column. | 2019-04-15 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C | 20-557 | Executive MgmtMgmt Rep | Revision to add change types and reasons for change. Consolidated repetitive inputs. | 2020-12-08 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| D | 21-090 | Executive MgmtMgmt Rep | Removed ECR single folder link. Now includes link to each document. | 2021-05-07 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E | 21-152 | Executive MgmtMgmt Rep | Added section for 510(k) Flowchart Pathway / Rational to Submit or Not Submit a New 510(k) | 2021-05-26 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F | 22-074 | Executive MgmtMgmt Rep | Added separate regulatory assessment tab. Added section for MWI training levels. | 2022-04-07 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| G | 23-162 | Regulatory AffairsMgmt Rep | Added Release Packet Decision Flow Chart tab. Reorganized pre-market and post-market response requirements to "design & development" and "release to production" requirements. | 2023-06-26 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| H | 24-199 | Regulatory AffairsQuality Engineering | Added V&V breakdown | 2024-05-01 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 4
| Verification/Validation | Status |
| --- | --- |
| Val | Planned |
| Verif | In-Progress |
|  | Complete |
