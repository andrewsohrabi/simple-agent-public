# VVPR-P01-232 Rev B: SSD Accuracy v3.3.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-232
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.3.0
- Source filename: VVPR-P01-232 - SSD Accuracy v3.3.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-232 - SSD Accuracy v3.3.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this test protocol is to verify the MX1 Portable X-ray System calculates an accurate Source to Skin Distance (SSD) as per PRD3.2 at critical target SSDs with representative anatomies. It is essential that the MX1 system has a high degree of accuracy under a representative set of conditions encompassed by the device’s environment for use. Critical SSD distance is specified in PRD 3.10 as 30cm.
OBJECTIVE
To ensure the MX1 System accurately computes and displays the SSD at critical target SSDs and with representative anatomies. (PRD 3.2, PRD 3.10)
REFERENCES
RSK-P01-010 Rev. E - MX1 Risk Assessment
DR-P01-005 Rev. E - MX1 Design Inputs
E1 Drawing Rev D, Emitter
MEMO-P01-778 - SSD Characterization, Rev. A
MATERIALS
Note* The MX1 collimation pucks do not interfere with the LIDAR sensors since they are not in the field of view, and therefore not used in this test.
MX1 Portable X-ray System:
E1 Emitter BOM Rev. H
C1 Cassette BOM Rev. I
MX1 Software System v3.3.0
T-129 Rev. C - MX1 EOL Test Fixture
EQP-222 - Measuring Tape (or equivalent)
Hand Phantom (Sawbones 1511-69)
Leg Phantom (Sawbones 1301-160-1)
Nitrile Glove (MedPride MPR-50505, or similar)
S10045 MX1 Debug Window HTML, v1.2.0
SAMPLE SIZE
For this test, one (1) MX1 Portable X-ray System will be used. Per RSK-P01-010 Rev. E, Risk Assessment; improper SSD (R0465, R0612, R0613, R0614, R1106) carry a worst-case severity score of 4 (Delay of Procedure or Minor Radiation Stochastic Harm). Per QSP-026 Rev. C, Statistical Methods, no minimum sample size is required for tests with low risk (risk severity of 1 or 4) that output attribute data. Based on sample size constraints due to MX1 being capital equipment, and the worst-case nature of this test, an n=1 will be used.
METHODS
Locations and Personnel Responsibilities
This study shall take place in MedAI facilities.
MedAI personnel are responsible for execution of the protocol.
Training Requirements - This study does not require any special training.
Background
SSD refers to the distance between the x-ray source and the skin of the subject. The SSD can be visualized as a line orthogonal to the front face of the emitter from the x-ray source to the first point of contact on the subject’s skin.
The x-ray focal spot is inside the emitter at 4.76 cm from the front face surface (Refer to E1 Drawing Rev D, Emitter). The MX1 device includes this internal distance when reporting SSD.
Figure 1
Test Conditions
Source to Skin Distance (SSD)
The SSD to the cassette surface as well as representative hand phantom and knee phantom shall be assessed at the following SSDs:
25cm
30cm
35cm
Test Case:  Stand-Mounted
Rationale: The MX1 System must be able to maintain SSD accuracy while hand-held or stand-mounted. The sensors used to calculate SSD report at 5Hz (i.e. 5 times a second), which is sufficiently frequent enough for the safety interlocks while hand-held when there may be slight movement from the operator. Moreover, it is impractical to conduct this test using standard measurement equipment (i.e. measuring tape) while the MX1 System is hand-held. Therefore, all measurements will be taken while stand-mounted in a test-jig. Additionally, all MX1 x-ray modes use the same SSD readings, therefore only 1 representative x-ray mode (Single Radiography) will be tested.
Experimental Procedure
SSD Accuracy
With the MX1 on and connected to the M50133 with the MedAI Device App, place the Emitter & Cassette on the T-129 EOL Test Fixture.
Figure 2
Position the Emitter on the fixture such that the MX1 SSD reads 25cm (within 5mm).
S10045 MX1 Debug Window HTML displays the MX1-calculated SSD value to the tenth of a millimeter. Using a secondary device (e.g. laptop), use S10045 to retrieve and record the SSD value as the MX1 Reported SSD in Tables 1 and 2.
Using the EQP-222 - Measuring Tape, measure the distance from the focal spot marker on the side of the emitter to the cassette top. Record the value as the Measuring Tape SSD Measurement in Tables 1 and 2
Repeat steps above with the emitter at 30 and 35 cm SSD. Record the values in Tables 1 and 2.
Repeat steps 1 to 6 after placing the Hand and Knee Phantoms on the cassette active area.
Note* With the Hand and Knee Phantom tests, place the phantom on the cassette active area. Then, adjust the emitter until the MX1 reported SSD is at the target SSD before taking the measurements as defined in steps 3 to 5.
DATA COLLECTION FORMS
Table Definitions:
Target SSD: Target MX1-reported SSD, as emitter is adjusted
MX1 Reported SSD Measurement (cm): SSD value reported by the MX1 device in S10045 MX1 Debug Window HTML
Measuring Tape SSD Measurement (cm): Physical SSD measured from the focal-spot mark on the emitter to the top of the cassette or anatomy
Table 1: SSD Accuracy Data Sheet and Acceptance Criteria - SSD Accuracy Tolerance
Table 2: SSD Accuracy Data Sheet and Acceptance Criteria - Measured SSD > Reported SSD
ACCEPTANCE CRITERIA
The SSD accuracy specification was set by MedAI Design Requirements and Risk Assessment. The SSD accuracy specification of ± 8% cm OR 1.5 cm, whichever is larger, was established by MedAI based on the accuracy of the LIDAR system. Detailed acceptance criteria ranges for each target SSD are included in Tables 1 and 2.
Additionally, the MX1-reported SSD value must always be less than the physical measured SSD
The MX1 System must pass all specifications listed in Tables 1 and 2 with zero failures. SSD output values will have variable data recorded but all results will be treated as attribute (pass/fail); no variable data analysis is required.
If it becomes necessary to make changes to the test procedures or study scope after testing has been initiated the protocol changes and associated rationale will be documented in the test report as deviations to the protocol.
JUSTIFICATION FOR ACCEPTANCE CRITERIA
The acceptance criteria based on PRD3.2 aims to maintain safety in the distance from the x-ray source to patient skin surfaces as required by IEC and CFR regulations. When using LIDAR to provide means for this safety, the acceptance criteria is aimed to ensure that the distances provided are a) within a reasonable margin of accuracy and b) never showing a distance to the operator that is further than the distance to the patient skin surface in reality. This test aims to verify the safety offset proposed in MEMO-P01-778.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
Recorded By: AKEAFA MOMENDate: 11/4/24
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220
C1 Cassette Rev. I, SN: 1079
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System v3.3.0
Additional tools/equipment:
EQP-222 - Measuring Tape (or equivalent)
In the report section, fill in the following table for equipment used during this study:
RESULTS
Table 1: SSD Accuracy Data Sheet and Acceptance Criteria - SSD Accuracy Tolerance
Table 2: SSD Accuracy Data Sheet and Acceptance Criteria - Measured SSD > Reported SSD
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
Meeting the above acceptance criteria determines that the SSD distances provided by MX1 are a) within a reasonable margin of accuracy and b) accounts for the maximum error of the system and never shows a distance to the operator that is further than the distance to the patient skin surface in reality.
LIST OF APPENDICES
Appendices 1 - 3: Example images on how this procedure was executed.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1: Example SSD Display on S10045 MX1 Debug Window HTML
Appendix 2: Example Measuring Tape SSD Measurement
Appendix 3: Focal spot indicator on Emitter shell

### Table 1
| Target SSD | MX1 Reported SSD Measurement (cm) | Measuring Tape SSD Measurement (cm) | Acceptance Criteria Target SSD ± 8% or 1.5cm whichever is greater (% or cm) | PASS/FAIL |
| --- | --- | --- | --- | --- |
| No Anatomy on Cassette |  |  |  |  |
| 25 |  |  |  |  |
| 30 |  |  |  |  |
| 35 |  |  |  |  |
| Hand Phantom |  |  |  |  |
| 25 |  |  |  |  |
| 30 |  |  |  |  |
| 35 |  |  |  |  |
| Knee Phantom |  |  |  |  |
| 25 |  |  |  |  |
| 30 |  |  |  |  |
| 35 |  |  |  |  |

### Table 2
| Target SSD | MX1 Reported SSD Measurement (cm) | Measuring Tape SSD Measurement (cm) | Acceptance Criteria: Measured SSD > Reported SSD | PASS/FAIL |
| --- | --- | --- | --- | --- |
| No Anatomy on Cassette |  |  |  |  |
| 25 |  |  |  |  |
| 30 |  |  |  |  |
| 35 |  |  |  |  |
| Hand Phantom |  |  |  |  |
| 25 |  |  |  |  |
| 30 |  |  |  |  |
| 35 |  |  |  |  |
| Knee Phantom |  |  |  |  |
| 25 |  |  |  |  |
| 30 |  |  |  |  |
| 35 |  |  |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Engineering Regulatory Affairs | 04 Nov 2024 | 24-635 |

### Table 4
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Measuring Tape | EQP-222 | 02 MAY 2024 | 31 MAY 2029 |

### Table 5
| Target SSD | MX1 Reported SSD Measurement (cm) | Measuring Tape SSD Measurement (cm) | Acceptance Criteria Target SSD ± 8% or 1.5cm whichever is greater (% or cm) | PASS/FAIL |
| --- | --- | --- | --- | --- |
| No Anatomy on Cassette |  |  |  |  |
| 25 | 25.0 cm | 25.2 cm | 0.80% | PASS |
| 30 | 30.1 cm | 30.45 cm | 1.16% | PASS |
| 35 | 35.1 cm | 35.25 cm | 0.43% | PASS |
| Hand Phantom |  |  |  |  |
| 25 | 24.9 cm | 25.0 cm | 0.40% | PASS |
| 30 | 30.2 cm | 30.5 cm | 0.99% | PASS |
| 35 | 34.9 cm | 34.95 cm | 0.14% | PASS |
| Knee Phantom |  |  |  |  |
| 25 | 25.2 cm | 25.5 cm | 1.19% | PASS |
| 30 | 30.0 cm | 30.1 cm | 0.33% | PASS |
| 35 | 34.9 cm | 35.1 cm | 0.57% | PASS |

### Table 6
| Target SSD | MX1 Reported SSD Measurement (cm) | Measuring Tape SSD Measurement (cm) | Acceptance Criteria: Measured SSD > Reported SSD | PASS/FAIL |
| --- | --- | --- | --- | --- |
| No Anatomy on Cassette |  |  |  |  |
| 25 | 25.0 cm | 25.2 cm | 25.2 cm > 25.0 cm | PASS |
| 30 | 30.1 cm | 30.45 cm | 30.45 cm > 30.1 cm | PASS |
| 35 | 35.1 cm | 35.25 cm | 35.25 cm > 35.1 cm | PASS |
| Hand Phantom |  |  |  |  |
| 25 | 24.9 cm | 25.0 cm | 25.0 cm > 24.9 cm | PASS |
| 30 | 30.2 cm | 30.5 cm | 30.5 cm > 30.2 cm | PASS |
| 35 | 34.9 cm | 34.95 cm | 34.95 cm > 34.9 cm | PASS |
| Knee Phantom |  |  |  |  |
| 25 | 25.2 cm | 25.5 cm | 25.5 cm > 25.2 cm | PASS |
| 30 | 30.0 cm | 30.1 cm | 30.1 cm > 30.0 cm | PASS |
| 35 | 34.9 cm | 35.1 cm | 35.1 cm > 34.9 cm | PASS |

### Table 7
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-602 |  |
