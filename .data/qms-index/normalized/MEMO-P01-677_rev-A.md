# MEMO-P01-677 Rev A: MX1 Software System Verification via Code Review v3.0.0

## Metadata
- Document ID: MEMO-P01-677
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.0
- Source filename: MEMO-P01-677 - MX1 Software System Verification via Code Review v3.0.0_A.docx
- Source path: Example QMS - MedAI/MEMO-P01-677 - MX1 Software System Verification via Code Review v3.0.0_A.docx
- Extraction warnings: none

## Extracted Content
1. Purpose
The purpose of this MX1 Software System Verification Code Review is to verify Software Requirements Specifications from MEMO-P01-630 Rev. B that are architectural in nature or are not feasible to test at the system level.
2. Scope
The scope of this MX1 Software System Verification Code Review is to manually review the code quality of the source code comprising the MX1 Software System and the quality of custom MX1 supporting items that may impact the functionality of the source code relative to the software requirements in Table 1.
3. Methods
Locations and Personnel Responsibilities
The MX1 Software System shall be reviewed during a group review including the Software Engineers involved with the development of each Software Component. Additionally, one or more Software Engineers who did not author any part of the code under review shall be included in this activity..
Experimental Procedure
Record the version numbers of the reviewed Software Components and supporting items in the Formal Code Review Report.
During the code review, all relevant Software Components comprising the MX1 Software System shall be evaluated. Each Component and corresponding supporting items shall be evaluated individually. Supporting items may include, but are not limited to: configuration files, test protocols, and test scripts.
Table 1. Code Review
4. Conclusion
Overall Result:.
Pass
Fail
Other: _______
No anomalies or concerns were discovered during the course of this review.
DOCUMENT REVISION HISTORY
Digital Key:
example.com/

### Table 1
| To: | File |
| --- | --- |
| From: | Device Software, Systems Integration |

### Table 2
| Software Component/s: Emitter Firmware (EM), Cassette Firmware (CAS), Monoblock Firmware (MB), Cassette Orchestrator (CO), Emitter Orchestrator (EO) Version: v3.0.0 Author(s): Matt Palumbo, Dean Hilton, Reagan Cole, Banks Troutman Reviewer(s): Gage Carr, Reagan Cole Review Date(s): 23 MAY 2024 |  |  |  |
| --- | --- | --- | --- |
| List the title and version(s) of any supporting items reviewed with this Software Component: MEMO-P01-630 - MX1 Software Requirements Specification, Rev. B |  |  |  |
| Requirements | Pass/ Fail | Author(s) | Reviewer |
| ICD Packets |  |  |  |
| SRS-6.1    The SS shall use the proprietary Interface Control Document (ICD) protocol as a means to communicate between EO or CO and relevant firmware peripherals | P | MP, DH, BT | GC, SD 23 MAY 2024 |
| SRS-6.2    The SS shall perform Sequence Number checks in every firmware peripheral interaction in all firmware components, EO, and CO | P | MP | GC, SD 23 MAY 2024 |
| SRS-6.3    The SS shall perform a Checksum on all ICD packets | P | MP | GC, SD 23 MAY 2024 |
| SRS-6.4    The SS shall perform Packet Validation in all firmware components, EO, and CO | P | MP | GC, SD 23 MAY 2024 |
| SRS-6.5    The SS shall return a packet acknowledging successful peripheral communication between software and firmware components | P | MP | GC, SD 23 MAY 2024 |
| Firmware Watchdog |  |  |  |
| SRS-7.14    The SS shall utilize hardware watchdogs for all firmware components | P | MP | GC, SD 23 MAY 2024 |
| Humidity Faults |  |  |  |
| SRS-20.2    The SS shall detect and report any out-of-bounds emitter humidity event as a critical fault | P | MP, DH | GC, SD 23 MAY 2024 |
| SRS-20.15    The SS shall detect and report any out-of-bounds cassette humidity event as a critical fault | P | MP, DH | GC, SD 23 MAY 2024 |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-440 |  |
