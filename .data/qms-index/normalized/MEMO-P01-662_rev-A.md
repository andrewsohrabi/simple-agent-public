# MEMO-P01-662 Rev A: Essential Performance Determination

## Metadata
- Document ID: MEMO-P01-662
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-662 - Essential Performance Determination_A-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-662 - Essential Performance Determination_A-signed.docx
- Extraction warnings: none

## Extracted Content
Purpose
To identify the Essential Performance requirements of the MX1 Portable X-ray System. The MX1 System consists of the E1 emitter, C1 cassette, P1 case, and H1 wired charger. And optional accessories: W1 Wireless Charger, F1 wireless foot pedal, and T1 tablet.
References
IEC 60601-2-54 Edition 2.0 2022-09
IEC 60601-2-43 Edition 2.2 2019-10
Discussion
IEC 60601-2-54 describes required essential performance items for x-ray devices (shown in Table 201.101 of the standard and provided for reference below).
For the MX1 Portable X-ray System, Accuracy of loading factors (203.6.4.3.104), Reproducibility of the radiation output (203.6.3.2), and Imaging Performance (203.6.7) apply. There is no Automatic Control System (203.6.5), based on the exemption provided by the Risk Management File. IEC 60601-2-43 provides additional considerations for essential performance to be analyzed in the Risk Management File (shown in Table 201.101 of the standard and provided for reference below).
The following table (Table 1) shows all Essential Performance characteristics that the MX1 Portable X-ray System must fulfill. Note that a failure of Essential Performance for the MX1 Portable X-ray System is specific to incorrect performance (e.g. delivering an incorrect dose to the patient). The Essential Performance is independent of other User Needs and Performance Requirements set by MedAI. The Essential Performance characteristics identified are all associated with the E1 emitter; there are no IEC 60601-2-54 Essential Performance characteristics for the C1 cassette, P1 case, F1 wireless foot pedal, or the H1 wired charger.
Appendices
Appendix A - Essential Performance During Immunity & Other Design Verification Tests
Appendix B - Example Essential Performance Results Table
DOCUMENT REVISION HISTORY
Digital Key: example.com/
Appendix A - Essential Performance During Immunity & Other Design Verification Tests
Per IEC 60601-2-54:2022 - 202.101 Immunity testing of ESSENTIAL PERFORMANCE:
The MANUFACTURER may minimize the test requirements of the additional potential ESSENTIAL PERFORMANCE requirements listed in Table 201.101 to a practical level through the RISK MANAGEMENT PROCESS.
When selecting the requirements to be tested, the MANUFACTURER needs to take into account the sensitivity to the EMC environment, probability of EMC condition and severity, and probability and contribution to unacceptable RISK through the RISK MANAGEMENT PROCESS.
Through the risk management process, as documented in RSK-P01-010, MedAI has determined to minimize the essential performance requirements during immunity testing and other MedAI design verification tests that list essential performance as an acceptance criteria (other than full essential performance verification per IEC 60601-2-54:2022 as verified by a 3rd party test lab) to a practical level. This testing shall be limited to 203.6.4.3.104  Accuracy of LOADING FACTORS, which is a sample of full essential performance testing. For the MX1 System, this includes the following test points and pass criteria:
Essential Performance Test Points:
Single Radiographic Imaging: 40kV, 2mA/40ms (0.08mAs), 1 pulse
Single Radiographic Imaging: 40kV, 2mA/80ms (0.16mAs), 1 pulse
Single Radiographic Imaging: 80kV, 2mA/125ms (0.25mAs), 1 pulse
Single Radiographic Imaging: 80kV, 1mA/40ms (0.04mAs), 1 pulse
Single Radiographic Imaging: 40kV, 2mA/200ms (0.40mAs), 1 pulse
Serial Radiographic Imaging: 80kV, 2mA/40ms (0.08mAs), acquire for a minimum of 1 second/5 pulses, measure first or last pulse only
Radioscopic Imaging: 64kV, 1mA/40ms (0.04mAs), acquire for a minimum of 1 second/5 pulses, measure first or last pulse only
Radioscopic Imaging: 64kV, 2mA/40ms (0.08mAs), acquire for a minimum of 1 second/5 pulses, measure first or last pulse only
Pass Criteria (at all test points):
Measured Voltage ± 8% error
Measured Current + 20% error
Measured Time ± (10 % + 1ms) error
Note that Single Radiography covers the same electrical sources and signaling paths leading to irradiation as Serial Radiography and Radioscopy, and thus evaluation of a single pulse is sufficient for essential performance verification. (Reference: IEC 60601-2-54:2018 - Subclause 202.101)
Also note that the user only has the option to select kV (voltage) and mAs (current time product) during normal use. However, due to the very low mAs range of the MX1 System (0.04-0.40mAs) versus the accuracy allowable by the standard (10% + 0.20mAs), it is appropriate and more conservative to evaluate the accuracy of current (mA) and loading time (s) values (203.6.4.3.104.4 and 203.6.4.3.104.5) individually during immunity and other MedAI design verification tests that list essential performance as an acceptance criteria. If the system meets the requirements for accuracy of current (mA) and loading time (s), it will also meet the accuracy requirements for current time product (mAs). Therefore, current time product (mAs) accuracy (203.6.4.3.104.6) will only be evaluated during full essential performance verification as verified by a 3rd party test lab.
As per subclause 202.101, Recovery Management testing shall only be performed during immunity testing if there is a noticeable failure during immunity testing or in the course of performing design verification testing that list essential performance as acceptance criteria. Similarly, Display of last image hold radiogram or radioscopy replay image sequence (203.6.7.101) and Radiation Dose Documentation (201.4.102) testing will be performed in all tests except immunity testing and design verification testing.
The following details the full testing suite with pass criteria as determined by the design of the MX1 system. The test sequences shall be performed with discretion to all of the caveats and conditions discussed above.
Full Essential Performance Test Sequence
Recovery Management (4)
Power off emitter to engage recoverable failure
Wait for error state
Power cycle the cassette and emitter
Ensure no software integrity check failures are detected
Enter emergency mode
Ensure device is available for radioscopy by acquiring 2 second radioscopy sequence
Accuracy of Loading Factors (1) and Imaging Performance (3)
Acquire 8 sequences to verify accuracy of loading factors
Single Radiographic Imaging: 40kV, 2mA/40ms (0.08mAs), 1 pulse
Single Radiographic Imaging: 40kV, 2mA/80ms (0.16mAs), 1 pulse
Single Radiographic Imaging: 80kV, 2mA/125ms (0.25mAs), 1 pulse
Single Radiographic Imaging: 80kV, 1mA/40ms (0.04mAs), 1 pulse
Single Radiographic Imaging: 40kV, 2mA/200ms (0.40mAs), 1 pulse
Serial Radiographic Imaging: 80kV, 2mA/40ms (0.08mAs), acquire for a minimum of 1 second/5 pulses, measure first or last pulse only
Serial Radioscopic Imaging: 64kV, 1mA/40ms (0.04mAs), acquire for a minimum of 1 second/5 pulses, measure first or last pulse only
Serial Radioscopic Imaging: 64kV, 2mA/40ms (0.08mAs), acquire for a minimum of 1 second/5 pulses, measure first or last pulse only
Check pass criteria for all sequences
Measured Voltage ± 8% error
Measured Current + 20% error
Measured Time ± (10 % + 1ms) error
Check pass criteria during and after Serial Radioscopic acquisition sequences
“Live” label visible on UI
Display replays the radioscopy sequence
Previous radioscopy sequence is replaced by new live sequence
Radiation Dose Documentation (5)
Press “Complete Acquisition” button to end the procedure
Check pass criteria
Timestamp of generated RDSR file matches timestamp of acquisition
Appendix B - Example Essential Performance Results Table
Recovery Management
Accuracy of Loading Factors
Imaging Performance
RDSR Generation
Variable data will be recorded but will be treated as attribute (pass/fail) for analysis purposes. No variable data analysis will be performed.

### Table 1
| Table 1: Product Essential/Specific Performance |  |
| --- | --- |
| No. | Description |
| 1 | Accuracy of Loading Factors |
| 1a | IEC 60601-2-54:2022 - 203.6.4.3.104.3 Accuracy of X-RAY TUBE VOLTAGE The output voltage must be within an accuracy of ± 8% |
| 1b | IEC 60601-2-54:2022 - 203.6.4.3.104.4 Accuracy of X-RAY TUBE CURRENT The output current must be within an accuracy of ± 20% |
| 1c | IEC 60601-2-54:2022 - 203.6.4.3.104.5 Accuracy of LOADING TIME The output loading time must be within an accuracy of ± (10 % + 1 ms) |
| 1d | IEC 60601-2-54:2022 - 203.6.4.3.104.6  Accuracy of CURRENT TIME PRODUCT The output current time product must be within an accuracy of ± (10 % + 0.2 mAs) |
| 2 | Reproducibility of Loading Factors |
| 2a | IEC 60601-2-54:2022 - 203.6.3.2.101 Reproducibility of the RADIATION output in RADIOGRAPHY The coefficient of variation of MEASURED VALUES of AIR KERMA shall be not greater than 0.05 |
| 2b | IEC 60601-2-54:2022 - 203.6.3.2.102(a) Linearity and constancy in RADIOGRAPHY The quotients of the average of the MEASURED VALUES of AIR KERMA divided by the preselected values or the indicated values of CURRENT TIME PRODUCT, or the product of the values of X-RAY TUBE CURRENT and LOADING TIME, obtained shall not differ by more than 0.2 times the mean value of these quotients. |
| 3 | Imaging Performance |
| 3a | IEC 60601-2-54:2022 - 203.6.7.101 Display of last image hold radiogram or radioscopy replay image sequence IEC 60601-2-43:2019 - 203.6.7.101 Display of last image hold radiogram or radioscopy replay image sequence The system shall display the series of the most recent images of the most recent radioscopy irradiation event after termination of irradiation. The system shall subsequently replace the displayed radioscopy replay image sequence concurrently with reinitiation of radioscopic irradiation and indicate whether the displayed image is a radioscopy replay image sequence or from an ongoing radioscopy acquisition. |
| 4 | Recovery Management |
| 4a | IEC 60601-2-43:2019 - 201.4.101 Recovery Management The system shall recover all functions from any recoverable failure in less than 10 minutes. The system shall indicate emergency mode at the working position of the operator and include functions for radioscopy as per the last mode of operation. |
| 5 | Radiation Dose Documentation |
|  | IEC 60601-2-43:2019 - 201.4.102 Radiation Dose Documentation The system shall create RDSRs and have the ability to perform RDSR End of Procedure Transmission. |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs Exec Mgmt | 25 Apr 2024 | 24-195 |

### Table 3
| Test Point | Acceptance Criteria | Pass/Fail |
| --- | --- | --- |
| Recovery Management | Functions recovered within 10 minutes |  |

### Table 4
| Test Point | Voltage (kV) | Current (mA) | Time (s) | Acceptance Criteria | Pass/Fail |
| --- | --- | --- | --- | --- | --- |
| Single Radiographic Imaging: 40kV, 2mA/40ms (0.08mAs), 1 pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
| Single Radiographic Imaging: 40kV, 2mA/80ms (0.16mAs), 1 pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 71-89 ms |  |
| Single Radiographic Imaging: 80kV, 2mA/125ms (0.25mAs), 1 pulse |  |  |  | Voltage: 73.6-86.4 kV Current: 1.6-2.4 mA Time: 112-138 ms |  |
| Single Radiographic Imaging: 80kV, 1mA/40ms (0.04mAs), 1 pulse |  |  |  | Voltage: 73.6-86.4 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
| Single Radiographic Imaging: 40kV, 2mA/200ms (0.40mAs), 1 pulse |  |  |  | Voltage: 36.8-43.2 kV Current: 1.6-2.4 mA Time: 179-221 ms |  |
| Serial Radiographic Imaging: 80kV, 2mA/40ms (0.08mAs),  acquire for a minimum of 1 second/5 pulses, measure first pulse only |  |  |  | Voltage: 73.6-86.4 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |
| Radioscopic Imaging: 64kV, 1mA/40ms (0.04mAs),  acquire for a minimum of 1 second/5 pulses, measure first pulse only |  |  |  | Voltage: 58.8-69.1 kV Current: 0.8-1.2 mA Time: 35-45 ms |  |
| Radioscopic Imaging: 64kV, 2mA/40ms (0.08mAs),  acquire for a minimum of 1 second/5 pulses, measure first pulse only |  |  |  | Voltage: 58.8-69.1 kV Current: 1.6-2.4 mA Time: 35-45 ms |  |

### Table 5
| Test Point | Acceptance Criteria | Pass/Fail |
| --- | --- | --- |
| “Live” label | “Live” label visible on UI |  |
| Radioscopy replay sequence | Radioscopy sequence is replayed after acquisition |  |
| Sequence replacement | Previous radioscopy sequence is replaced by new live sequence |  |

### Table 6
| Test Point | Acceptance Criteria | Pass/Fail |
| --- | --- | --- |
| RDSR Generation | Timestamp of generated RDSR file is within 5 mins of current time |  |
