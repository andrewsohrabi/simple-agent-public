# VVPR-P01-156 Rev B: Radiation Output Reproducibility Verification Protocol Report

## Metadata
- Document ID: VVPR-P01-156
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-156 - Radiation Output Reproducibility Verification Protocol  Report_B-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-156 - Radiation Output Reproducibility Verification Protocol  Report_B-Signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify radiation output and reproducibility are within specification for the MX1 Portable X-ray System.
OBJECTIVE
The objective of this study is to collect dosimetric information for inclusion in the MX1 Portable X-ray System Instructions for Use (IFU) in accordance with IEC 60601-2-54.
REFERENCES
IEC 60601-2-54 Edition 2.0 2022 Medical electrical equipment – Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy, Section 203.6.3.2 “Reproducibility of the RADIATION output”
IFU-MX1 Rev. D- Instructions for Use
MATERIALS
Equipment:
MX1 Portable X-ray System Components:
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
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
.
Experimental Procedure
The procedure for imaging modalities of the device will follow IFU-MX1 - Instructions for Use for all possible loading factor configurations listed in Appendix A. Additional steps for verification are listed below as necessary.
Place the emitter and cassette into the MX1 Testing Fixture (T-129)
Turn the MX1 System on
System will automatically collimate to the max detector area
Pre-radiation/imaging
Place the EQP-110 RadCal Multi-Sensor (or equivalent) orthogonally centered about the focal spot at 90 cm SID.
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
The data reported from the Accu-Gold software is reported in uGy. The mean and standard deviation of the ten air kerma values will be calculated. These two values will be used to calculate the coefficient of variation:
ACCEPTANCE CRITERIA
The acceptance criteria is defined in IEC 60601-2-54 Section 203.6.3.2.101 Reproducibility of the RADIATION output in RADIOGRAPHY, where the following is stated: “The coefficient of variation of MEASURED VALUES of AIR KERMA shall be not greater than 0,05 for any combination of LOADING FACTORS.”
APPENDICES
Appendix A - Data Sheet Templates with Loading Factor Combinations to be Tested
Table A.1: Equipment Table
Table A.2: Device Configuration
Recorded By: ______________________________________________   Date:_________________________________
Table A.3: MX1 Air Kerma data measured at 90 cm SID
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
Table A.4: Calculated Coefficient of Variation for Air Kerma Measurements
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table A.3: Equipment Table
MX1 Portable X-ray System Rev E  Components:
Table A.4: Device Configuration
Tested By: Chris Holland   Date:    May 9, 2024
RESULTS
Data Collection
Table A.3: MX1 Air Kerma data measured at 90 cm SID
Tested By: Riley Compton   Date:    May 9, 2024
Table A.4: Calculated Coefficient of Variation for Air Kerma Measurements
Tested By: Riley Compton   Date:    May 9, 2024
CONCLUSION
The MX1 system is compliant with standards of output reproducibility, as the coefficient of variation for all four of the test points as set out by IEC 60601-2-54 Section 203.6.3.2.101 did not exceed the limit of 0.05.
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Equipment ID | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- |

### Table 2
| Device Serial Number: |  |
| --- | --- |
| Software Version: |  |

### Table 3
| Air Kerma for MX1 System |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Loading Factor Combinations |  | Sample |  |  |  |  |  |  |  |  |  |  |
| Tube Voltage (kV) | Current-Time Product (mAs) |  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| 40 | 0.4 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |
| 80 | 0.04 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |
| 40 | 0.25 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |
| 64 | 0.08 | Time (HH:MM) |  |  |  |  |  |  |  |  |  |  |
|  |  | Dose(uGy) |  |  |  |  |  |  |  |  |  |  |

### Table 4
| CV Calculations |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Tube Voltage (kV) | Current-Time Product (mAs) | Mean | Standard Deviation | Coefficient of Variation | CV Limit | Pass/Fail |
| 40 | 0.4 |  |  |  | 0.05 |  |
| 80 | 0.04 |  |  |  | 0.05 |  |
| 40 | 0.25 |  |  |  | 0.05 |  |
| 64 | 0.08 |  |  |  | 0.05 |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 08 May 2024 |  |

### Table 6
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 7
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| EQP-158 | Radcal AGDM+ Accu-Gold Digitizer | 13-July-2023 | 13-July-2024 | Chris Holland 5/9/24 |
| EQP-159 | Radcal AGMS-DM+ Accu-Gold Multi-Sensor | 13-July-2023 | 13-July-2024 | Chris Holland 5/9/24 |
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
| 40 | 0.4 | Time (HH:MM) | 16:35 | 16:41 | 16:46 | 16:51 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 5.8 | 5.69 | 5.68 | 5.67 | 5.687 | 5.631 | 5.694 | 5.68 | 5.688 | 5.723 |
| 80 | 0.04 | Time (HH:MM) | 16:35 | 16.41 | 16:47 | 16:52 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 3.219 | 3.133 | 3.099 | 3.116 | 3.108 | 3.12 | 3.112 | 3.11 | 3.135 | 3.187 |
| 40 | 0.25 | Time (HH:MM) | 16:35 | 16:42 | 16:47 | 16:52 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 3.6 | 3.543 | 3.525 | 3.539 | 3.559 | 3.561 | 3.54 | 3.51 | 3.532 | 3.596 |
| 64 | 0.08 | Time (HH:MM) | 16:35 | 16:42 | 16:47 | 16:52 | 16:57 | 17:02 | 17:08 | 17:14 | 17:20 | 17:39 |
|  |  | Dose(uGy) | 4.104 | 4.044 | 4.042 | 4.035 | 4.061 | 4.011 | 4.031 | 4.025 | 4.019 | 4.069 |

### Table 10
| CV Calculations |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Tube Voltage (kV) | Current-Time Product (mAs) | Mean | Standard Deviation | Coefficient of Variation | CV Limit | Pass/Fail |
| 40 | 0.4 | 5.6943 | 0.043606957 | 0.007658001 | 0.05 | Pass |
| 80 | 0.04 | 3.1339 | 0.038754068 | 0.012366083 | 0.05 | Pass |
| 40 | 0.25 | 3.5505 | 0.029125209 | 0.008203129 | 0.05 | Pass |
| 64 | 0.08 | 4.0441 | 0.027573941 | 0.006818313 | 0.05 | Pass |

### Table 11
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | Engineering Quality Engineering Regulatory Affairs | 25 May 2024 | 24-270 |
