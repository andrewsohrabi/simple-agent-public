# MEMO-P01-680 Rev A: MX1 Software System Verification via Code Review v3.1.0

## Metadata
- Document ID: MEMO-P01-680
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.1
- Source filename: MEMO-P01-680 - MX1 Software System Verification via Code Review v3.1.0_A.docx
- Source path: Example QMS - MedAI/MEMO-P01-680 - MX1 Software System Verification via Code Review v3.1.0_A.docx
- Extraction warnings: none

## Extracted Content
1. Purpose
The purpose of this MX1 Software System Verification Code Review is to verify Software Requirements Specifications from MEMO-P01-630 Rev. C that are architectural in nature or are not feasible to test at the system level.
2. Scope
The scope of this MX1 Software System Verification Code Review is to manually review the code quality of the source code comprising the MX1 Software System and the quality of custom MX1 supporting items that may impact the functionality of the source code relative to the software requirements in Table 1.
3. Methods
Locations and Personnel Responsibilities
The MX1 Software System shall be reviewed during a group review including the Software Engineers involved with the development of each Software Component. Additionally, one or more Software Engineers who did not author any part of the code under review shall be included in this activity.
Experimental Procedure
Record the version numbers of the reviewed Software Components and supporting items in the Formal Code Review Report.
During the code review, all relevant Software Components comprising the MX1 Software System shall be evaluated. Each Component and corresponding supporting items shall be evaluated individually. Supporting items may include, but are not limited to: configuration files, test protocols, and test scripts.
Table 1. Code Review
4. Conclusion
Overall Result:.
Pass
Fail
Other: _______
No anomalies or other concerns were discovered during the course of this review.
DOCUMENT REVISION HISTORY
Digital Key:
example.com/

### Table 1
| To: | File |
| --- | --- |
| From: | Cloud Services, Systems Intergration |

### Table 2
| Software Component/s: MedAI Device App (ODA) Version: v3.1.0 Author(s): Ari Inoue Reviewer(s): Pavel Sitnikov Review Date(s): 24 MAY 2024 |  |  |  |
| --- | --- | --- | --- |
| List the title and version(s) of any supporting items reviewed with this Software Component: MEMO-P01-630 - MX1 Software Requirements Specification, Rev. C |  |  |  |
| Requirements | Pass/ Fail | Author(s) | Reviewer |
| Luminance Calibration Curve |  |  |  |
| SRS-33.3    The SS shall apply the appropriate luminance calibration curve according to user selection | P | CI | PS 24 MAY 2024 |
| Security Risk Requirements |  |  |  |
| SRS-7.11    The SS shall use one-way communication between the Footpedal and Emitter | P | DH | GC, SP 29 MAY 2024 |
| SRS-28.9    The ODA local storage file names shall not point to a study or patient | P | CI | PS, GC 29 MAY 2024 |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-449 |  |
