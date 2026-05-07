# VVPR-P01-155 Rev B: Residual Radiation Verification Protocol Report

## Metadata
- Document ID: VVPR-P01-155
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-155 - Residual Radiation Verification Protocol  Report_B-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-155 - Residual Radiation Verification Protocol  Report_B-Signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify residual radiation measurements are within specification for the MX1 Portable X-ray System.
OBJECTIVE
The objective of this study is to collect dosimetric information for inclusion in the MX1 Portable X-ray System Instructions for Use (IFU) in accordance with IEC 60601-2-54.
REFERENCES
IEC 60601-2-54 Edition 2.0 2022 Medical electrical equipment – Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy Section  203.11.12  “Test for attenuation of residual radiation”
IFU-MX1 Rev. D- Instructions for Use
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
Attenuator for Residual Radiation (T-182)
SAMPLE SIZE
This is a type test per IEC 60601-1:2020 Clause 5.2. Therefore, this test will utilize a sample size of 1.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
Experimental Procedure
The procedure for imaging modalities of the device will follow IFU-MX1 - Instructions for Use for all possible loading factor configurations listed in Appendix A. Additional steps for verification are listed below as necessary.
Place the emitter and cassette into the MX1 Testing Fixture (T-064)
Turn the MX1 System on
System will automatically collimate to the max detector area
Bottom plate may be removed temporarily to manually set collimator position if aluminum attenuator blocks sensors for tracking
Place aluminum attenuator (T-182) directly underneath, flush with, the bottom plate of emitter
Pre-radiation/imaging
Place the EQP-110 (or equivalent) RadCal Multi-Sensor 10cm underneath the cassette bottom surface at point 1 in the diagram in Appendix A.
Connect the EQP-110 (or equivalent) Multi-Sensor and EQP-109 (or equivalent) Digitizer to the PC
Initiate Accu-Gold software
Set the Multi-Sensor as the trigger sensor within the Accu-Gold software
Conduct any background noise correction as recommended by the Accu-Gold software/system
Dosimetric Measurements
Setup MX1 Emitter at initial loading factors and SID per Appendix A
Trigger MX1 Emitter
Verify registration of dose in Accu-Gold software
Repeat steps 1-3 for the remaining loading factor combinations in Appendix A
Calculate air kerma at surface of detector based on the inverse square law as described in Section 6.6 of this protocol
Data Calculation and Analysis
Position A-I calculated dose will be normalized to dose in one hour (normalized dose rate uGy/hr) in the RadCal software. These numbers will be divided by three when entered into the data table in Appendix A, as the off-time enforced by the system after a DDR yields a maximum 33% duty cycle over 1 hour. The average of all measurements will be used to check compliance.
ACCEPTANCE CRITERIA
The average normalized dose rate must be less than 150 uGy/hr as per IEC 60601-2-54:2022 Ed 2 Table 203.106 for application category A.
APPENDICES
Appendix A - Data Sheet Templates with Loading Factor Combinations to be Tested
Figure A.1: Dosimeter Test Points
Table A.1: Air Kerma Measurements for Residual Radiation
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
Table A.2: Equipment Table
Table A.3: Device Configuration
Recorded By: ______________________________________________   Date:_________________________________
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
Note: No measurement points below the cassette accessible surface registered dose on the AccuGold AGMS-D+ sensor. The minimum dose to record on this dosimeter is 40 nGy, and the minimum dose rate with waveform analysis is 1 nGy/s, or 3.6 uGy/hr. All measurements were listed as below this value as a result.
Figure A.1: Dosimeter Test Points
Table A.1: Air Kerma Measurements for Residual Radiation
Tested By: Riley Compton   Date:    May 9, 2024
CONCLUSION
The MX1 system is compliant with standards of residual radiation, as all measured points from the accessible surface did not detect any air kerma rate above 3.6 uGy/hr, well below the limit of 150 uGy/hr as per IEC 60601-2-54:2022 Ed 2 Table 203.106 for application category A.
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Raw Dose behind MX1 Cassette (uGy) | Kerma Rate behind MX1 Cassette (uGy/hr) |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Nominal kV / mAs |  | Nominal kV / mAs |  |  |
| Test Point | 64 kV / 8 mAs | 80 kV / 8 mAs | Test Point | 64 kV / 8 mAs | 80 kV / 8 mAs |
| A |  |  | A |  |  |
| B |  |  | B |  |  |
| C |  |  | C |  |  |
| D |  |  | D |  |  |
| E |  |  | E |  |  |
| F |  |  | F |  |  |
| G |  |  | G |  |  |
| H |  |  | H |  |  |
| I |  |  | I |  |  |
| Average |  |  | Average |  |  |
|  |  |  | Limit | 150 | 150 |
|  |  |  | Pass/Fail |  |  |

### Table 2
| Equipment ID | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- |

### Table 3
| Device Serial Number: |  |
| --- | --- |
| Software Version: |  |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 08 May 2024 | 24-224 |

### Table 5
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 6
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| EQP-158 | Radcal AGDM+ Accu-Gold Digitizer | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| EQP-159 | Radcal AGMS-DM+ Accu-Gold Multi-Sensor | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| T-129 | EOL Testing Fixture | N/A | N/A | Chris Holland 5/9/24 |
| N/A | Accu-Gold 3.0 Radiation Measurement Software | N/A | N/A | Chris Holland 5/9/24 |

### Table 7
| Device Serial Number: | E1 Emitter: 1216    C1 Cassette: 1217 |
| --- | --- |
| Software Version: | v3.0.0-gamma |

### Table 8
| Raw Dose behind MX1 Cassette (nGy) | Kerma Rate behind MX1 Cassette (uGy/hr) |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
|  | Nominal kV / mAs |  | Nominal kV / mAs |  |  |
| Test Point | 64 kV / 8 mAs | 80 kV / 8 mAs | Test Point | 64 kV / 8 mAs | 80 kV / 8 mAs |
| A | < 40 nGy | < 40 nGy | A | < 3.6 uGy/hr | < 3.6 uGy/hr |
| B | < 40 nGy | < 40 nGy | B | < 3.6 uGy/hr | < 3.6 uGy/hr |
| C | < 40 nGy | < 40 nGy | C | < 3.6 uGy/hr | < 3.6 uGy/hr |
| D | < 40 nGy | < 40 nGy | D | < 3.6 uGy/hr | < 3.6 uGy/hr |
| E | < 40 nGy | < 40 nGy | E | < 3.6 uGy/hr | < 3.6 uGy/hr |
| F | < 40 nGy | < 40 nGy | F | < 3.6 uGy/hr | < 3.6 uGy/hr |
| G | < 40 nGy | < 40 nGy | G | < 3.6 uGy/hr | < 3.6 uGy/hr |
| H | < 40 nGy | < 40 nGy | H | < 3.6 uGy/hr | < 3.6 uGy/hr |
| I | < 40 nGy | < 40 nGy | I | < 3.6 uGy/hr | < 3.6 uGy/hr |
| Average | < 40 nGy | < 40 nGy | Average | < 3.6 uGy/hr | < 3.6 uGy/hr |
|  |  |  | Limit | 150 | 150 |
|  |  |  | Pass/Fail | Pass | Pass |

### Table 9
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | Engineering Quality Engineering Regulatory Affairs | 26 May 2024 | 24-271 |
