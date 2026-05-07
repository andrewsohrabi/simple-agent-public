# ECR-593 Rev A: MX1 BOM-055 Rev G

## Metadata
- Document ID: ECR-593
- Revision: A
- Prefix: ECR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: ECR-593 - MX1 BOM-055 Rev G_A-signed.docx
- Source path: Example QMS - MedAI/ECR-593 - MX1 BOM-055 Rev G_A-signed.docx
- Extraction warnings: none

## Extracted Content
ECR-593 - MX1 BOM-055 Rev G_A-signed
Sheet: Engineering Change Request
Sheet: Release Packet Decision Flow Ch
[Empty sheet]
Sheet: Regulatory Assessment
Sheet: Document Revision History
Sheet: Variables

### Table 1
| d |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | ENGINEERING CHANGE REQUEST |  |  |  |  |  |  |
|  | ECR# | 593.0 | Product: | MX1 | Effective Start Date: | Immediately |  |
|  |  |  | Originator | M. Khosravanipour | End Date (if applicable): | N/A |  |
|  | 1. RELEASE IS FOR: |  |  |  |  |  |  |
|  | False | Design & Development | True | V&V | False | Release to Production &Device Master Record Index (DMRI) |  |
|  |  |  |  |  |  | 510(k) (if applicable): |  |
|  | Skip Section(s) 2 & 3 |  | Skip Section(s) 3 |  | Fill sections 2 through 5 |  |  |
|  | 2. POST DESIGN FREEZE CHANGES |  |  |  |  |  |  |
|  | TYPE OF CHANGE: |  |  | CHANGE AFFECTS: |  |  |  |
|  | False | Change to existing Drawing(s) |  | False | Product in the Field |  |  |
|  | False | Change to existing Software/Firmware(s) |  | False | Product in Inventory |  |  |
|  | True | Change to existing BOM(s) |  | False | Product in Work in Progress |  |  |
|  | False | Change to existing Work Instruction(s) or Test Procedure(s) |  | True | Components/Material in Inventory |  |  |
|  | False | Other: |  | True | Tools/Fixtures in Manufacturing |  |  |
|  |  |  |  | False | Open Purchase Order(s) |  |  |
|  | REASON FOR CHANGE: |  |  | False | Other: |  |  |
|  | False | Product Improvement |  |  |  |  |  |
|  | False | Cost Reduction |  | IMPACT |  |  |  |
|  | False | Supplier Change |  | False | Incorporate into product in the field |  |  |
|  | True | Part Replacement (with Alternate or Equivalent Replacement) |  | False | Incorporate into product in inventory |  |  |
|  | False | Reliability/Performance Issue   (Corrective Action) |  | True | Incorporate for next build of product |  |  |
|  | False | Safety Issue   (Corrective Action) |  | False | Incorporate for next build of product once existing components/ have been used up |  |  |
|  | False | Response to NCR (include NCR# & Affected lot / qty): |  | False | Scrap Components/Material |  |  |
|  | False | Other: |  | False | Rework Components/Material |  |  |
|  |  |  |  | False | Incorporate as a temporary deviation for a specified order |  |  |
|  |  |  |  | False | Incorporate as a temporary deviation for a specified time or serial/lot number(s): |  |  |
|  |  |  |  | False | Other: |  |  |
|  | DESCRIPTION OF CHANGE |  |  |  |  |  |  |
|  | - BOM-055 updated to Rev G. Changes include the replacement of the existing sub-GHz antenna (M50054) with M51184 and M51183. Mounting hardware also added to support the new antennas.- Pediatric filter removed. Additionally, ES-10019 Rev B.1 was added in response to the part release.- Initial release of engineering evaluation memos supporting the sub-GHz antenna changes and wifi antenna location updates.- Drawing update for M11085 (Cassette Catch Plate). No change to part geometry or material. - Overshipper (M11160) labels updated to reflect correct FCC ID - Correct FCC ID is PD99260NG for WiFi Module. |  |  |  |  |  |  |
|  | REASON FOR CHANGE |  |  |  |  |  |  |
|  | - The sub-GHz antenna was replaced to use a module configuration with FCC certification. Additional mounting hardware was added to support the new antenna. - The wifi antennae were relocated to increase the distance from the patient contacting surface to the antenna to meet implementation reqts in wifi module FCC certification.- ES-10019 revised to enhance manfuacturability, address design issues, and increase compatibility with a (future) new x-ray tube.- Pediatric filter removed to align with regulatory decision to remove pediatrics from the indications for use.- MWIs updated to reflect BOM changes.- Additional tool added to ensure the manufacturability of the BOM changes.- RSK & QSR documents updated to reflect BOM changes.- Drawing update for M11085 to update CTQ, as original criteria was not able to be inspected with existing measurement equipment.- Labels updated to reflect correct FCC ID. |  |  |  |  |  |  |
|  | Is new Verification and/or Validation testing required? |  |  |  |  |  |  |
|  | False | Yes | If "Yes", provide the VVPR number(s) and any additional doc references: |  |  |  |  |
|  | True | No | If "No", provide justification: New anntenas demonstrated to function equivalently to original component. Additional mechanical testing for new hardware not required, as the materials and functions are similar to existing hardware components previously evaluated per IEC 60601-1. The updated adhesive has a similar overlap shear strength for ABS plastic, and while the adhesive type for the revised BOM is different, both adhesives are two part adhesives. Therefore, the new adhesive is not expected to impact previously completed testing. V&V impact assessment for ES-10019 Rev B and Rev B.1 included in ECRs-510 and -591. |  |  |  |  |
|  | False | N/A | If "NA", provide justification: |  |  |  |  |
|  | Are changes to inputs or outputs of risk management required? |  |  |  |  |  |  |
|  | True | Yes | If "Yes", provide the updated Risk Document(s) and any additional doc references: Updated RSK included below. DFMEA was updated to reflect the new parts. Changes to the HA reflective of removing pediatric indications will be included in Rev E of RSK-P01-010, which will be released with the revised IFU. |  |  |  |  |
|  | False | No | If "No", provide justification: |  |  |  |  |
|  | False | N/A | If "NA", provide justification: |  |  |  |  |
|  | 3. REGULATORY ASSESSMENT |  |  |  |  |  |  |
|  | True | Regulatory Affiars to Complete Regulatory Impact Assessment in "Regulatory Assessment" Tab |  |  |  |  |  |
|  | 4. WORK INSTRUCTIONS |  |  |  |  |  |  |
|  | TRAINING: |  |  |  |  |  |  |
|  | Does this ECR include Manufacturing Work Instruction(s) (MWI) |  |  |  |  |  |  |
|  | True | Yes, select required training level per QSP-008 and QSR-003: |  |  |  |  |  |
|  |  | False | Level 1 (cannot be selected for initial release) |  |  |  |  |
|  |  | True | Level 2 |  |  |  |  |
|  |  | False | Level 3 |  |  |  |  |
|  | False | No, this ECR does not contain Manufacturing Work Instructions |  |  |  |  |  |
|  | 5. DOCUMENTS AFFECTED |  |  |  |  | Temporary Link to Folder with all Files:  (folder to be deleted & files dispersed in electronic signatures folder once ECR has been signed) |  |
|  | Drawing Number / Doc Number & Description |  |  | New Revision (Include link to each document) |  | ECR-593 |  |
|  | BOMs |  |  |  |  |  |  |
|  | BOM-055 - MX1 (Top-level assembly) |  |  | G |  |  |  |
|  | BOM-008 - C1 Cassette |  |  | L |  |  |  |
|  | BOM-004 - E1 Emitter |  |  | K |  |  |  |
|  | BOM-037 - P1 Pelican Case |  |  | G |  |  |  |
|  | BOM-056 - MS-10627 Puck Box with Pucks |  |  | F |  |  |  |
|  | Drawings |  |  |  |  |  |  |
|  | MS-11089 - Cassette Top Populated |  |  | I |  |  |  |
|  | M51184 - Ezuiro Sub-GHz Antenna |  |  | A |  |  |  |
|  | MS-10136 - E1 Lower Internal ASSY |  |  | F |  |  |  |
|  | M51183 - Molex Sub-GHz Antenna |  |  | A |  |  |  |
|  | MS-10134 - Shell R Populated ASSY |  |  | F |  |  |  |
|  | M10951 - Emitter Sub-GHz Bracket |  |  | A |  |  |  |
|  | M51187 - 3M Acrylic Adhesive DP8705NS |  |  | A |  |  |  |
|  | MS-10627 - Puck Box with Pucks |  |  | E |  |  |  |
|  | M11085 - Cassette Catch Plate |  |  | B |  |  |  |
|  | M11160 - MX1 Cardboard Overshipper Label |  |  | C |  |  |  |
|  | A10065 - MX1 Cardboard Overshipper Label Artwork |  |  | C |  |  |  |
|  | MWIs |  |  |  |  |  |  |
|  | MWI-172 - MS-11089 - Cassette Top Populated |  |  | I |  |  |  |
|  | MWI-125 - MS-10136 - Lower Internal |  |  | I |  |  |  |
|  | MWI-124 - MS-10134 - Shell R Populated |  |  | G |  |  |  |
|  | MWI-267 - MS-10627 - Puck Box Assembly |  |  | F |  |  |  |
|  | DHRs |  |  |  |  |  |  |
|  | QSF-093 - C1 Cassette Design History Record |  |  | J |  |  |  |
|  | QSF-092 - E1 Emitter Design History Record |  |  | J |  |  |  |
|  | QSF-113 - DHR Form for MS-10627 Puck Box |  |  | B |  |  |  |
|  | Tools |  |  |  |  |  |  |
|  | T-207 - Cassette SubGig Antenna Location Fixture |  |  | A |  |  |  |
|  | Memo |  |  |  |  |  |  |
|  | MEMO-P01-771 - Cassette Wifi Antenna Placement Testing |  |  | A |  |  |  |
|  | MEMO-P01-772 - Sub Gig Antenna Testing |  |  | A |  |  |  |
|  | DFMEA |  |  |  |  |  |  |
|  | RSK-P01-012 - MX1 DFMEA |  |  | B |  |  |  |
|  | Inspection Plan |  |  |  |  |  |  |
|  | QSR-026 - Lot Acceptance Sampling |  |  | H |  |  |  |
|  | DOCUMENT APPROVALS |  |  |  |  |  |  |
|  | Revision | DCO # | Description | Approved By | Eff. Date | Digital Key |  |
|  | A | 24-617 | Initial Release | Quality EngineeringEngineeringRegulatory AffairsOps | 2024-10-24 00:00:00 | example.com/ |  |

### Table 2
| DESIGN AND PROCESS CHANGE REVIEW | YES | NO | If yes, provide rationale, supporting documentation, etc. | COUNTRY/REGION | IMPACT |  | REGULATORY IMPACT ASSESSMENT - Include rationale and supporting data as applicable. Include regulatory change assessment flowchart pathways and/or key decisions points. |  |  | APPLICABLE GUIDANCES |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Is the modification to directly correct any device failure in the field or clinical study? | False | True |  | United States and United States Territories | True | Letter to File | Regulatory Change Assessment Flowchart B:B1 - No, the device is not an IVD.B2 - No, the changes are not control mechanism, operating principle, or energy type changes.B3 - No, there is no change to cleaning or disinfection. MX1 is non-sterile.B4 - No, there is no change in packaging. MX1 does not have an expiration date.B5 - Yes, there is a change in design.B5.1 - No, the changes do not impact use of the device.B5.2 - No, new risks were not identified and existing risks were not modified.B5.3 - No, clincal data is not necessary.B5.4 - No, there were no unexpected issues from V&V activities.The design changes do not affect use of the device or device safety. A new 510k submission is not required.Decision: Documentation (LTF).  A Letter to File will be documented covering all changes in BOM-055 Rev F and G at the time of release to production.Labeling regulatory change assessment flowchart not included as the revised Indications for use and associated labeling (e.g., IFU) are not in this ECR, and will be reviewed by the FDA under 510k K241567. This update was made in response to an AINN letter during submission review. |  |  | Guidance for Industry and FDA Staff - Deciding When to Submit a 510(k) for a Change to an Existing Device (2017) |
| Does the modification require clinical evaluation to determine if the device remains safe and effective? | False | True |  |  | False | Traditional 510(k) |  |  |  | Guidance for Industry and FDA Staff - Deciding When to Submit a 510(k) for a Software Change to an Existing Device (2017) |
| Is there a change to device performance, intended use, product labeling/IFU, or marketing claims? | False | True |  |  | False | Special 510(k) |  |  |  |  |
| Is the change to add a new manufacturing facility? | False | True |  |  | False | No Impact - ECR Assessment Only |  |  |  |  |
| Is the change to add new equipment or tools that directly affects the finished device? | False | True |  | Canada | False | License Amendment | MX1 is currently not approved or marketed in Canada. |  |  | Guidance for the Interpretation of Significant Change of a Medical Device (2011) |
| Does the change add a new specification or test method or otherwise provide additional assurance of identity, strength, or reliability of the device? | False | True |  |  | False | Letter to File |  |  |  |  |
| Is there a change to a vendor, purchased components, or material? | True | False | New components added to MX1. The existing sub-GHz antenna was replaced with two different antennas to maintain FCC compliance. Mounting hardware was added to retain the anteannas in assembly. These changes do not impact device function or safety.New ES-10019 PCBA revision made to improve manfuacturability and address a design issue. There is no impact to device safety or effectiveness.Pediatric filter removed from BOM in response to updated indications, which will be reflected in IFU-MX1 Rev H. |  | True | No Impact - ECR Assessment Only |  |  |  |  |
| Is there a change in manufacturing method or process parameters? | False | True |  | Australia | False | Substantial Change Notification | MX1 is currently not approved or marketed in Australia. |  |  | Changes affecting TGA-issued conformity assessment certificates (2021) |
| Is there a change to sterilization or shelf life? | False | True |  |  | False | Letter to File |  |  |  |  |
| Does the change affect the risk analysis? | True | False | DFMEA was updated to include the new components. The updates did not introduce new risks/requirements and did not modify existing risks/mitigations. Pediatric-specific risks will be removed in RSK-P01-010 - MX1 Risk Assessment Rev E. |  | True | No Impact - ECR Assessment Only |  |  |  |  |
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
| H | 24-199 | Regulatory AffairsQuality Engineering | Added V&V breakdown | 2024-04-01 00:00:00 | example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Table 4
| Verification/Validation | Status |
| --- | --- |
| Val | Planned |
| Verif | In-Progress |
|  | Complete |
