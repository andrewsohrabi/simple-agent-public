# VVPR-P01-157 Rev B: Radiography Linearity and Constancy Verification Protocol Report

## Metadata
- Document ID: VVPR-P01-157
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-157 - Radiography Linearity and Constancy Verification Protocol  Report_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-157 - Radiography Linearity and Constancy Verification Protocol  Report_B-signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify radiation output, linearity, and constancy are within specification for the MX1 Portable X-ray System.
OBJECTIVE
The objective of this study is to collect dosimetric information for inclusion in the MX1 Portable X-ray System Instructions for Use (IFU) in accordance with IEC 60601-2-54.
REFERENCES
IEC 60601-2-54 Edition 2.0 2022 Medical electrical equipment – Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy, Section 203.6.3.2 “Reproducibility of the RADIATION output”
IFU-MX1 Rev. D - Instructions for Use
MATERIALS
Equipment:
MX1 Portable X-ray System Rev E Components:
E1 Emitter
C1 Cassette
Laptop with Accu-Gold 2.0 Radiation Measurement Software
Radcal AGDM+ Accu-Gold Digitizer (EQP-109 or equivalent)
Radcal AGMS-DM+ Accu-Gold Multi-Sensor (EQP-110 or equivalent)
MX1 Testing Fixture (T-129)
Radiation PPE as necessary (ie. Lead vest, radiation monitor, etc.)
SAMPLE SIZE
This is a type test per IEC 60601-1:2020 Clause 5.2. Therefore, this test will utilize a sample size of 1.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL
Personnel Testing shall be conducted by members of the MedAI Engineering team.
Experimental Procedure
The procedure for imaging modalities of the device will follow IFU-MX1 - Instructions for Use for all possible loading factor configurations listed in Appendix A. Additional steps for verification are listed below as necessary.
Place the emitter and cassette into the MX1 Testing Fixture (T-129)
Turn the MX1 System on
System will automatically collimate to the max detector area
Pre-radiation/imaging
Place the EQP-110 (or equivalent) RadCal Multi-Sensor orthogonally centered about the focal spot at 90 cm SID.
Connect the EQP-110 (or equivalent) Multi-Sensor and EQP-109 (or equivalent) Digitizer to the PC
Initiate Accu-Gold software
Set the Multi-Sensor as the trigger sensor within the Accu-Gold software
Conduct any background noise correction as recommended by the Accu-Gold software/system
Dosimetric Measurements
Setup MX1 Emitter at initial loading factors per Appendix A
Trigger MX1 Emitter
Verify registration of dose in Accu-Gold software, copying dose data and acquisition time into table in Appendix A
Repeat steps 6.4.3.1-6.4.3.3 for the remaining loading factor combinations in Appendix A
Wait ~5 minutes
Repeat steps 6.4.3.1-6.4.3.5 until all 10 samples have been collected per loading factor combination
Data Calculation and Analysis
The data reported from the Accu-Gold software is reported in uGy. The mean of the ten air kerma values will be calculated. These values and the respective current time products will be used in this equation listed in Section 7 “ACCEPTANCE CRITERIA” of this document.
ACCEPTANCE CRITERIA
The acceptance criteria is defined in IEC 60601-2-54 Section 203.6.3.2.102 Linearity and constancy in RADIOGRAPHY, where the following is stated:
“For operation in RADIOGRAPHY the quotients of the average of the MEASURED VALUES of AIR KERMA divided by the preselected values or the indicated values of CURRENT TIME PRODUCT, or the product of the values of X-RAY TUBE CURRENT and LOADING TIME … shall not differ by more than 0,2 times the mean value of these quotients:”
APPENDICES
Appendix A - Data Sheet Templates with Loading Factor Combinations to be Tested
Table A.1: MX1 Air KermaData measured at 90 cm SID
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
Table A.2: Calculated Coefficient of Variation for Air Kerma Measurements
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
Table A.3: Equipment Table
Table A.4: Device Configuration
Recorded By: ______________________________________________   Date:_________________________________
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
No deviations from protocol.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table A.3: Equipment Table
MX1 Portable X-ray System Rev E  Components:
Table A.4: Device Configuration
Recorded By: Chris Holland   Date:    May 9, 2024
RESULTS
Data Collection
Table A.1: MX1 Air KermaData measured at 90 cm SID
Tested By: Riley Compton   Date:    May 9, 2024
Table A.2: Calculated Coefficient of Variation for Air Kerma Measurements
Tested By: Riley Compton   Date:    May 9, 2024
CONCLUSION
The MX1 system is compliant with standards of output linearity and constancy, as the difference quotient was lower than the summed quotient for all test points, as set out by IEC 60601-2-54 Section 203.6.3.2.102.
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Air Kerma for MX1 System |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loading Factor Combinations |  | Sample |  |  |  |  |  |  |  |  |  |  |
| Tube Voltage (kV) | Current-Time Product (mAs) |  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 40 | 0.25 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |
| 40 | 0.4 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |
| 64 | 0.08 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |
| 64 | 0.16 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |

### Table 2
| CV Calculations |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Tube Voltage (kV) | Current-Time Product (mAs) | Mean Kerma (uGy) |  |  | Pass/Fail |
| 40 | 0.25 |  |  |  |  |
| 40 | 0.40 |  |  |  |  |
| 64 | 0.08 |  |  |  |  |
| 64 | 0.16 |  |  |  |  |

### Table 3
| Equipment ID | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- |

### Table 4
| Device Serial Number: |  |
| --- | --- |
| Software Version: |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 09 May 2024 | 24-225 |

### Table 6
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 7
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| EQP-158 | Radcal AGDM+ Accu-Gold Digitizer | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| EQP-159 | Radcal AGMS-DM+ Accu-Gold Multi-Sensor | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| T-129 | EOL Testing Fixture | N/A | N/A | Chris Holland 5/9/24 |
| N/A | Accu-Gold 3.0 Radiation Measurement Software | N/A | N/A | Chris Holland 5/9/24 |

### Table 8
| Device Serial Number: | E1 Emitter: 1216    C1 Cassette: 1217 |
| --- | --- |
| Software Version: | v3.0.0-gamma |

### Table 9
| Air Kerma for MX1 System |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loading Factor Combinations |  | Sample |  |  |  |  |  |  |  |  |  |  |
| Tube Voltage (kV) | Current-Time Product (mAs) |  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 40 | 0.25 | Time (HH:MM) | 16:35 | 16:42 | 16:47 | 16:52 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 3.6 | 3.543 | 3.525 | 3.539 | 3.559 | 3.561 | 3.54 | 3.51 | 3.532 | 3.596 |
| 40 | 0.4 | Time (HH:MM) | 16:36 | 16:42 | 16:47 | 16:52 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 5.766 | 5.696 | 5.689 | 5.69 | 5.697 | 5.717 | 5.675 | 5.693 | 5.636 | 5.735 |
| 64 | 0.08 | Time (HH:MM) | 16:35 | 16:42 | 16:47 | 16:52 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 4.104 | 4.044 | 4.042 | 4.035 | 4.061 | 4.011 | 4.031 | 4.025 | 4.019 | 4.069 |
| 64 | 0.16 | Time (HH:MM) | 16:36 | 16:42 | 16:47 | 16:52 | 16:57 | 17:02 | 17:09 | 17:14 | 17:20 | 17:40 |
|  |  | Dose(uGy) | 8.362 | 8.187 | 8.201 | 8.187 | 8.19 | 8.189 | 8.216 | 8.197 | 8.199 | 8.263 |

### Table 10
| CV Calculations |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Tube Voltage (kV) | Current-Time Product (mAs) | Mean Kerma (uGy) |  |  | Pass/Fail |
| 40 | 0.25 | 3.5505 | 0.0465 | 2.84505 | Pass |
| 40 | 0.40 | 5.6994 |  |  |  |
| 64 | 0.08 | 4.0441 | 0.81813 | 10.1920625 | Pass |
| 64 | 0.16 | 8.2191 |  |  |  |

### Table 11
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | Engineering Quality Engineering Regulatory Affairs | 26 May 2024 | 24-269 |
