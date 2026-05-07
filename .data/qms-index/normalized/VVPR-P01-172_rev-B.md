# VVPR-P01-172 Rev B: Attenuation Equivalent Detector Verification Protocol Report

## Metadata
- Document ID: VVPR-P01-172
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-172 Attenuation Equivalent Detector Verification Protocol  Report_B-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-172 Attenuation Equivalent Detector Verification Protocol  Report_B-Signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to collect testing for quality equivalent filtration data of materials “located in the path of the X-RAY BEAM between the PATIENT and the X-RAY IMAGE RECEPTOR” in the MX1 Portable X-ray System, as described in IEC 60601-2-54.
OBJECTIVE
The primary purpose of this study is to collect the quality equivalent filtration data of in-path materials within the detector for the MX1 Portable X-ray System in accordance with IEC 60601-2-54 Clause 203.10  “ATTENUATION of the X-RAY BEAM between the PATIENT and the X-RAY IMAGE RECEPTOR”
REFERENCES
IEC 60601-1-3 Edition 2.2 2021-01
IEC 60601-2-54 Edition 2.0 2022-09
MATERIALS
Equipment:
VJ Integrated X-Ray Source (T-177 Rev A)
Lab Laptop with Accu-Gold 2.0 Radiation Measurement Software
Radcal AGDM+ Accu-Gold Digitizer (EQP-109 or equivalent)
Radcal AGMS-DM+ Accu-Gold Multi-Sensor (EQP-110 or equivalent)
Radiation PPE as necessary (ie. Lead vest, radiation monitor, etc.)
Narrow Beam Test Fixture (T-174 Rev A)
Beam Alignment Phantom (EQP-057)
8 mil Aluminum Shims (T-178 Rev A)
Mars1417x Detector (T-180 Rev A)
MX1 Cassette materials
Decal (M11104 Rev A)
Polycarbonate Shell (MS-11090 Rev D)
Carbon fiber plate with foam (MS-11088 Rev B)
iRay Mercu0909X (M50004 Rev A)
MATLAB QEF Script (Appendix A)
SAMPLE SIZE
Per IEC 60601-1-3:2021, “Readers of this collateral standard are reminded that, in accordance with IEC 60601-1, Clause 5, all the test procedures described are TYPE TESTS, intended to be carried out in a dedicated testing environment in order to determine compliance.” Clause 5 of IEC 60601-1:2020 states that tests in scope of the standard are considered Type Tests. Type Tests are performed on a single representative sample of the item being tested (n=1). Therefore, a sample size of n=1 will be used to collect the filtration data.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL.
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
Experimental Procedure
Place the Narrow Beam Test Fixture (T-174) below the VJ X-Ray (T-177) emitter
Place the Mars detector (T-180) under the Narrow Beam Fixture
Connect the Multi-Sensor (EQP-110 or equivalent) to the Digitizer (EQP-109 or equivalent) and the Digitizer to the laptop
Set the Multi-Sensor in line with the primary beam exiting the VJ X-Ray device
Initiate Accu-Gold software
Conduct any background noise correction as recommended by the Accu-Gold software/system
Turn on the detector
Pre-radiation/imaging
Prepare the detector for AED (Automatic Exposure Detection) mode
Use the Beam Alignment Phantom (EQP-057) to determine the location directly under the primary beam
Move the Narrow Beam Fixture such that the pinhole is aligned with the beam
Place Multi-Sensor such that the top half of the sensor is centered in the beam
Ensure no material is clamped in the Narrow Beam Fixture
Set VJ to 100 kV and 2 mA for 2000 ms
Trigger VJ exposure
Verify registration of dose and kV in Accu-Gold software and that the measured kV is 100 kV +- 0.5 kV and record the measurement in Table A.1 as 0 shims
Repeat steps 6.4.2.7-6.4.2.8 two additional times
Add 3.6 mm Al to the VJ’s beam path
Trigger VJ exposure
Verify registration of dose in Accu-Gold software and verify the HVL of the the beam as greater than or equal to 3.6 mm Al (e.g. the dose with 3.6 mm Al of additional attenuation must be greater than or equal to half of the dose from the first exposure)
Remove the 3.6mm Al from the VJ’s beam path
Dosimetric Measurements
Add 1 sheet of 8 mil aluminum (T-178) to the Narrow Beam Fixture and clamp it down
Trigger VJ
Verify registration of dose in Accu-Gold software,  label the measurement with the number of Al sheets present, and record the measurement in Table A.1
Repeat steps 6.4.3.1-6.5.3.3 until 11 Al sheets have been added, repeating each exposure twice for a total of three samples
Remove aluminum and replace with the cassette materials above the active imaging plane of the detector relevant to IEC 60601-2-54 Clause 203.10.1, cutting 8 inch squares of material if need be to fit into Narrow Beam Fixture. The iRay detector’s top panel shall be included, which can be removed from the iRay Mercu0909X M50004 unit.
Trigger VJ
Verify registration of dose in Accu-Gold software, label measurement according to kV and detector materials, and record dose measurement in Table A.2
Repeat steps 6.4.3.5 - 6.4.3.7 two additional times
Run MATLAB QEF Script to precisely estimate the aluminum equivalent of the detector materials.
Data Calculation and Analysis
The data reported from the Accu-Gold includes kerma and kerma rate, and each measurement point will be recorded in the data sheet template in Appendix A.
An exponential fit will be applied to the aluminum filtration measurements to create a model predicting thickness of aluminum for any given attenuated dose.
The attenuated dose of the detector materials under a 100 kV, 3.6 mm Al HVL beam will be compared against the model to calculate the QEF of the detector materials.
ACCEPTANCE CRITERIA
For the detector materials compromising the front panel of the MX1 cassette, the acceptance criteria in IEC 60601-2-54 clause 203.10.1 is used.
Total of all layers, excluding detector itself, composing the front panel of DIGITAL X-RAY IMAGING DEVICE [shall not exceed] 1,2 mm Al
APPENDICES
Appendix A - Data Sheet Template
Appendix A - Data Sheet Templates with Loading Factor Combinations to be Tested
Table A.1: Attenuated Kerma for Aluminum
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
Table A.2: Attenuated Kerma for MX1 Detector
Tested By: ______________________________________________   Date:_________________________________
______________________________________________             _________________________________
Table A.3: Equipment Table
Table A.4: Device Configuration
Recorded By: ______________________________________________   Date:_________________________________
MATLAB QEF Script
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table A.3: Equipment Table
Recorded By: Chris Holland   Date:    May 9, 2024
RESULTS
Alignment
Fixture was aligned with the focal spot phantom (EQP-057) and the Multi-Sensor was centered in the beam. The following alignment image was captured:
The dosimeter was centered in the beam, as shown with the following image:
Beam hardness verification:
Initial dose: 0.4669 mGy
Dose with 3.66 mmAl: 0.2396 mGy
Data Collection
Table A.1: Attenuated Kerma for Aluminum
Tested By: Riley Compton   Date:    May 9, 2024
Table A.2: Attenuated Kerma for MX1 Detector
Tested By: Riley Compton   Date:    May 9, 2024
CONCLUSION
The MX1 system is compliant with IEC 60601-2-54 clause 203.10.1, as the quality equivalent filtration of all materials in the cassette in the beam path before the imaging receptor area was 1.112 mm Al, less than the maximum value of 1.2 mm Al.
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Attenuated Kerma for Aluminum (mGy) |  |  |  |
| --- | --- | --- | --- |
| Number of Shims (8 mil shim) | Attenuated Kerma (mGy) |  |  |
|  | Sample 1 | Sample 2 | Sample 3 |
| 0 |  |  |  |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |
| 6 |  |  |  |
| 7 |  |  |  |
| 8 |  |  |  |
| 9 |  |  |  |
| 10 |  |  |  |
| 11 |  |  |  |

### Table 2
| Attenuated Kerma for MX1 Detector (mGy) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Sample | Measured Dose (mGy) | Average Dose (mGy) | Equivalent Al Thickness (mm Al) | IEC Maximum(mm Al) | Pass/Fail |
| 1 |  |  |  | 1.2 |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

### Table 3
| Equipment ID | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- |

### Table 4
| Device Serial Number: |  |
| --- | --- |
| Software Version: |  |

### Table 5
| %% Aluminum shim = 0.2036; %% mm per 8 mil shim baseline = 0000; % ENTER avg dose of baseline xx = [[1:11].*shim]; yy = [0 0 0 0 0 0 0 0 0 0 0]; % ENTER avg doses of attenuated beam from aluminum shims yy = yy/baseline; % divide by baseline dose rate fitAl = getFit(xx,yy); plotCurve('Al Filtration', xx, yy, fitAl); HVL = predX(0.5, fitAl); detectorDose = 0; % ENTER avg dose of attenuated beam from cassette materials kFullStack = detectorDose/baseline; detectorQEF = predX(kFullStack, fitAl); % Output QEF %% Functions function curveOutput = getFit(xVals, yVals) fitfun = fittype( @(a,b,c,x) a+b*exp(-c*x)); [curveOutput,gof] = fit(xVals',yVals',fitfun,'StartPoint',[1,1,1]); end function predictedVal = predX(yVal, fitted_curve) coeffs = coeffvalues(fitted_curve); predictedVal = log((yVal-coeffs(1))/coeffs(2))/-coeffs(3); end function plotCurve(titleStr, xx, yy, fitted_curve) hold off scatter(xx,yy); hold on plot(xx(1):0.01:xx(end),fitted_curve(xx(1):0.01:xx(end)), 'lineWidth', 2); legend('Measured', 'Model'); xlabel('Thickness (mm)'); ylabel('Normalized KERMA'); title(titleStr); end |
| --- |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Regulatory Affairs Engineering | 08 May 2024 | 24-220 |

### Table 7
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 8
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| EQP-158 | Radcal AGDM+ Accu-Gold Digitizer | 13-July-2023 | 13-July-2024 | Chris Holland 5/9/24 |
| EQP-159 | Radcal AGMS-DM+ Accu-Gold Multi-Sensor | 13-July-2023 | 13-July-2024 | Chris Holland 5/9/24 |
| T-177 Rev A | VJ Integrated X-Ray Source | N/A | N/A | Chris Holland 5/9/24 |
| T-174 Rev A | Narrow Beam Test Fixture | N/A | N/A | Chris Holland 5/9/24 |
| EQP-057 | Beam Alignment Phantom | N/A | N/A | Chris Holland 5/9/24 |
| T-180 Rev A | Mars 1417x Detector | N/A | N/A | Chris Holland 5/9/24 |
| T-178 Rev A | 8 mil Aluminum Shims | N/A | N/A | Chris Holland 5/9/24 |
| M11104 Rev A | Decal | N/A | N/A | Chris Holland 5/9/24 |
| MS-11090 Rev D | Polycarbonate Shell | N/A | N/A | Chris Holland 5/9/24 |
| M10238 Rev 2 | Carbon Fiber Sandwich Panel | N/A | N/A | Chris Holland 5/9/24 |
| M50004 Rev A | iRay Mercu0909X | N/A | N/A | Chris Holland 5/9/24 |
| N/A | MATLAB QEF Script (Appendix A) | N/A | N/A | Chris Holland 5/9/24 |
| N/A | Accu-Gold 3.0 Radiation Measurement Software | N/A | N/A | Chris Holland 5/9/24 |

### Table 9
| Attenuated Kerma for Aluminum (mGy) |  |  |  |
| --- | --- | --- | --- |
| Number of Shims (8 mil shim) | Attenuated Kerma (mGy) |  |  |
|  | Sample 1 | Sample 2 | Sample 3 |
| 0 | 0.4802 | 0.4809 | 0.4667 |
| 1 | 0.4532 | 0.4566 | 0.4501 |
| 2 | 0.4308 | 0.4321 | 0.4314 |
| 3 | 0.4177 | 0.4181 | 0.4172 |
| 4 | 0.4009 | 0.3933 | 0.3968 |
| 5 | 0.3810 | 0.3824 | 0.3846 |
| 6 | 0.3697 | 0.3645 | 0.3684 |
| 7 | 0.3540 | 0.3525 | 0.3561 |
| 8 | 0.3421 | 0.3405 | 0.3407 |
| 9 | 0.3317 | 0.3318 | 0.3244 |
| 10 | 0.3143 | 0.3152 | 0.3181 |
| 11 | 0.3055 | 0.3055 | 0.3069 |

### Table 10
| Attenuated Kerma for MX1 Detector (mGy) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Sample | Measured Dose (mGy) | Average Dose (mGy) | Equivalent Al Thickness (mm Al) | IEC Maximum(mm Al) | Pass/Fail |
| 1 | 0.3738 | 0.3735 | 1.112 | 1.2 | Pass |
| 2 | 0.3717 |  |  |  |  |
| 3 | 0.3751 |  |  |  |  |

### Table 11
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | Engineering Quality Engineering Regulatory Affairs | 25 May 2024 | 24-268 |
