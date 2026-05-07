# VVPR-P01-151 Rev B: Half Value Layer Verification Protocol Report

## Metadata
- Document ID: VVPR-P01-151
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-151 - Half Value Layer Verification Protocol  Report_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-151 - Half Value Layer Verification Protocol  Report_B-signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify the Half Value Layer (HVL) meets requirements for the MX1 Portable X-ray System
OBJECTIVE
The objective of this study is to collect dosimetric information for inclusion in the MX1 Portable X-ray System Instructions for Use (IFU) in accordance with IEC 60601-2-54.
REFERENCES
IEC 60601-1 Edition 3.2 2020 Medical electrical equipment - Part 1: General requirements for basic safety and essential performance
IEC 60601-1-3 Edition 2.2 2021-01 Medical electrical equipment - Part 1-3: General requirements for basic safety and essential performance - Collateral Standard: Radiation protection in diagnostic X-ray equipment. Section 7.1  “HALF-VALUE LAYERS and TOTAL FILTRATION in X-RAY EQUIPMENT”
IEC 60601-2-54 Edition 2.0 2022-09 - Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy
IFU-MX1 Rev. D - Instructions for Use
MATERIALS
Equipment:
MX1 Portable X-ray System Rev E  Components:
E1 Emitter
C1 Cassette
Laptop with Accu-Gold 2.0 Radiation Measurement Software
Radcal AGDM+ Accu-Gold Digitizer (EQP-109 or equivalent)
Radcal AGMS-DM+ Accu-Gold Multi-Sensor (EQP-110 or equivalent)
Narrow Beam Test Fixture (T-174)
MX1 Emitter Mounting Arm (K1, L1 or equivalent)
Radiation PPE as necessary (ie. Lead vest, radiation monitor, etc.)
Mars1417X Detector (T-180 Rev A)
8 mil Aluminum Shims (T-178 Rev A)
1 mil Aluminum Shims (T-179 Rev A)
Beam Alignment Phantom (EQP-057)
SAMPLE SIZE
This is a type test per IEC 60601-1:2020 Clause 5.2. Therefore, this test will utilize a sample size of 1.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
Experimental Procedure
The procedure for imaging modalities of the device will follow IFU-MX1 - Instructions for Use for all possible loading factor configurations listed in Appendix A. Additional steps for verification are listed below as necessary.
Place the Narrow Beam Fixture (T-174) under the MX1 Emitter Mounting Arm
Place the Mars detector (T-180) under the Narrow Beam Fixture
Connect the Multi-Sensor (EQP-110 or equivalent) to the Digitizer (EQP-109 or equivalent) and the Digitizer to the laptop
Turn the MX1 System on in engineering mode
System will automatically collimate to the max detector area
Turn on the detector
Initiate Accu-Gold software
Set the Multi-Sensor as the trigger sensor within the Accu-Gold software
Conduct any background noise correction as recommended by the Accu-Gold software/system
Pre-radiation/imaging
Place the Beam Alignment Phantom (EQP-057)  inside the Narrow Beam Fixture at the center of the stage
Set a technique of 40 kV 0.4 mAs on the E1 Emitter
Prepare the detector for (Automatic Exposure Detection (AED) mode
Trigger the E1 Emitter
Adjust the position of the E1 Emitter until the emitter’s beam is inline with the beam alignment phantom
Replace the Beam Alignment Phantom with the Multi-Sensor such that the top half of the sensor is centered in the beam
Ensure no aluminum is clamped in the Narrow Beam Fixture
Dosimetric Measurements
Trigger E1 Emitter remotely
Verify registration of dose in Accu-Gold software
Add 1 sheet of 8 mil aluminum (T-178) to the Narrow Beam Fixture and clamp it down
Trigger E1 Emitter remotely
Verify registration of dose in Accu-Gold software and label the measurement with the number of Al sheets present
Repeat steps 6.4.3.3-6.4.3.5 until the maximum Al sheets have been added for the respective kV based on table A.1 in Appendix A.
Remove all shims
Identify the test case that resulted in a measured dose immediately above half unattenuated value on the results table A.1, and record this value in table A.3
Replace the number of 8 mil shims identified in the previous step
Repeat steps 6.4.3.3-6.4.3.5 by adding shims in 1 mil (T-179) increments (up to a total of 7x 1 mil shims) rather than the 8 mil shims, as to increase the granularity of measurements near the HVL thickness
Record these values in table A.2 in Appendix A.
Repeat steps 6.4.3.1-6.4.3.11 at 50 kV, 60 kV, 70 kV, and 80 kV
Run the MATLAB HVL script in Appendix A to determine the HVL using dosimetric measurements
Record the calculated HVL values in table A.3.
Data Calculation and Analysis
The dose vs aluminum thickness is plotted and an exponential regression is performed to determine the expression for the dose vs Al thickness line. The regression’s equation should follow the form:
= normalized dose
= Al thickness
= unknowns determined with regression
is normalized by baseline, unattenuated dose measurements, such that a  of 1 indicates 0 mm Al of attenuation is present after the permanent filtration. With this expression, the precise thickness of aluminum required to reduce the initial dose to half its value can be calculated. An R2 of >0.99 is expected, otherwise the data should be recollected/remodeled.
ACCEPTANCE CRITERIA
The acceptance criteria for minimum permissible first HVL is defined in IEC 60601-1-3 Section 7.1 within Table 3:
The calculated HVL results per each kV setting shall be documented in the MX1 Instructions for Use (IFU-MX1).
APPENDICES
Appendix A - Data Sheet Templates with Loading Factor Combinations to be Tested
Table A.1: MX1 System Attenuated Kerma
Tested By: ______________________________________________   Date:_________________________________
Table A.2: Attenuated Kerma near HVL
Tested By: ______________________________________________   Date:_________________________________
Table A.3: MX1 System HVLs
(XX Shims)
Tested By: ______________________________________________   Date:_________________________________
MATLAB HVL Script
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
A technique of 64 kV 0.4 mAs was added. Intertek noted that it should be included per IEC standards as the nominal tube potential for the radioscopy mode. An acceptance criteria for 40 kV was also added through linear extrapolation, following the pattern of alternating 0.3/0.4 mm Al increments between 10 kV increments.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table A.3: Equipment Table
MX1 Portable X-ray System Rev E  Components:
Table A.4: Device Configuration
Recorded By: Chris Holland   Date:    May 9, 2024
RESULTS
Alignment
Fixture was placed in the same position as it was in for the pediatric filtration test (VVPR -P01-152). The following alignment image was captured during the pediatric filtration test:
It was verified that taking an image with the narrow beam fixture’s top plate raised above the bottom plate resulted in a circular image (not elliptical).
The dosimeter was centered in the beam, as shown with the following image:
Data Collection
Table A.1: MX1 System Attenuated Kerma
Tested By: Riley Compton   Date:    May 9, 2024
Table A.2: Attenuated Kerma near HVL
Tested By: Riley Compton   Date:    May 9, 2024
Table A.3: MX1 System HVLs
(XX Shims)
Tested By: Riley Compton   Date:    May 9, 2024
CONCLUSION
The MX1 system passes the IEC standards for beam quality, with all tested kVs exceeding the minimum permissible first HVL as defined in IEC 60601-1-3 Section 7.1 within Table 3.
REPORT APPROVAL

### Table 1
| X-RAY TUBE VOLTAGE Kv | Minimum permissible first HALF-VALUE LAYER mm Al |
| --- | --- |
| 50 | 1.8 |
| 60 | 2.2 |
| 70 | 2.5 |
| 80 | 2.9 |

### Table 2
| Attenuated Kerma for MX1 System (mGy) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Number of Shims (8 mil shim) | Tube Potential |  |  |  |  |
|  | 40 kV | 50 kV | 60 kV | 70 kV | 80 kV |
| 0 |  |  |  |  |  |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 9 |  |  |  |  |  |
| 10 | DNT |  |  |  |  |
| 11 | DNT |  |  |  |  |
| 12 | DNT | DNT |  |  |  |
| 13 | DNT | DNT |  |  |  |
| 14 | DNT | DNT | DNT |  |  |
| 15 | DNT | DNT | DNT |  |  |
| 16 | DNT | DNT | DNT | DNT |  |
| 17 | DNT | DNT | DNT | DNT |  |

### Table 3
| Attenuated Kerma near HVL (mGy) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Number of Additional 1 mil Shims | Tube Potential (Initial Number of 8 mil Shims) |  |  |  |  |
|  | 40 kV (__ Shims) | 50 kV (__ Shims) | 60 kV (__ Shims) | 70 kV (__ Shims) | 80 kV (__ Shims) |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |
| 6 |  |  |  |  |  |
| 7 |  |  |  |  |  |

### Table 4
| HVLs for MX1 System (mGy) |  |  |  |  |
| --- | --- | --- | --- | --- |
| Tube Potential | HVL (mm Al) |  |  |  |
|  | Rough HVL (Thickest measurement before half value) | Calculated, precise HVL | IEC Minimum | Pass/Fail |
| 40 kV |  |  | N/A |  |
| 50 kV |  |  | 1.8 |  |
| 60 kV |  |  | 2.2 |  |
| 70 kV |  |  | 2.5 |  |
| 80 kV |  |  | 2.9 |  |

### Table 5
| %% Aluminum shim = 0.2036; %% mm per 8 mil shim. Change to 1 mil here if need be. baseline = 0000; % ENTER avg dose of baseline xx = [[1:17].*shim]; % Change number of shims here yy = [0 0 0 0 0 0 0 0 0 0 0]; % ENTER avg doses of attenuated beam from aluminum shims yy = yy/baseline; % divide by baseline dose rate fitAl = getFit(xx,yy); plotCurve('Al Filtration', xx, yy, fitAl); HVL = predX(0.5, fitAl); %% Functions function curveOutput = getFit(xVals, yVals) fitfun = fittype( @(a,b,c,x) a+b*exp(-c*x)); [curveOutput,gof] = fit(xVals',yVals',fitfun,'StartPoint',[1,1,1]); end function predictedVal = predX(yVal, fitted_curve) coeffs = coeffvalues(fitted_curve); predictedVal = log((yVal-coeffs(1))/coeffs(2))/-coeffs(3); end function plotCurve(titleStr, xx, yy, fitted_curve) hold off scatter(xx,yy); hold on plot(xx(1):0.01:xx(end),fitted_curve(xx(1):0.01:xx(end)), 'lineWidth', 2); legend('Measured', 'Model'); xlabel('Thickness (mm)'); ylabel('Normalized KERMA'); title(titleStr); end |
| --- |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Regulatory Affairs Engineering | 08 May 2024 | 24-222 |

### Table 7
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 8
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| EQP-158 | Radcal AGDM+ Accu-Gold Digitizer | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| EQP-159 | Radcal AGMS-DM+ Accu-Gold Multi-Sensor | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| T-174 Rev A | Narrow Beam Test Fixture | N/A | N/A | Chris Holland 5/9/24 |
| EQP-057 | Beam Alignment Phantom | N/A | N/A | Chris Holland 5/9/24 |
| T-180 Rev A | Mars 1417x Detector | N/A | N/A | Chris Holland 5/9/24 |
| T-178 Rev A | 8 mil Aluminum Shims | N/A | N/A | Chris Holland 5/9/24 |
| T-179 Rev A | 1 mil Aluminum Shims | N/A | N/A | Chris Holland 5/9/24 |
| N/A | K1 Cart Arm | N/A | N/A | Chris Holland 5/9/24 |
| N/A | Accu-Gold 3.0 Radiation Measurement Software | N/A | N/A | Chris Holland 5/9/24 |

### Table 9
| Device Serial Number: | E1 Emitter: 1216    C1 Cassette: 1217 |
| --- | --- |
| Software Version: | v3.0.0-gamma |

### Table 10
| Attenuated Kerma for MX1 System (mGy) |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Number of Shims (8 mil shim) | Tube Potential |  |  |  |  |  |
|  | 40 kV | 50 kV | 60 kV | 64 kV | 70 kV | 80 kV |
| 0 | 0.006162 | 0.01229 | 0.01948 | 0.02216 | 0.02659 | 0.03476 |
| 1 | 0.005609 | 0.01147 | 0.01805 | 0.02063 | 0.02497 | 0.03302 |
| 2 | 0.005212 | 0.0106 | 0.01695 | 0.01945 | 0.02349 | 0.03129 |
| 3 | 0.00473 | 0.009813 | 0.01593 | 0.01823 | 0.02215 | 0.02972 |
| 4 | 0.004311 | 0.009111 | 0.01489 | 0.0172 | 0.02086 | 0.02827 |
| 5 | 0.003981 | 0.008487 | 0.01397 | 0.01621 | 0.01967 | 0.02681 |
| 6 | 0.003697 | 0.007963 | 0.01317 | 0.01569 | 0.01874 | 0.02544 |
| 7 | 0.003389 | 0.007481 | 0.01241 | 0.01473 | 0.01776 | 0.02438 |
| 8 | 0.003176 | 0.006998 | 0.01175 | 0.01385 | 0.01691 | 0.02324 |
| 9 | 0.002921 | 0.006554 | 0.01116 | 0.01309 | 0.01615 | 0.02235 |
| 10 | DNT | 0.006165 | 0.01056 | 0.01252 | 0.01531 | 0.02129 |
| 11 | DNT | 0.005813 | 0.01001 | 0.01189 | 0.0146 | 0.02046 |
| 12 | DNT | DNT | 0.009508 | 0.01132 | 0.014 | 0.01963 |
| 13 | DNT | DNT | 0.009071 | 0.01085 | 0.01337 | 0.01872 |
| 14 | DNT | DNT | DNT | DNT | 0.01276 | 0.01809 |
| 15 | DNT | DNT | DNT | DNT | 0.01223 | 0.0173 |
| 16 | DNT | DNT | DNT | DNT | DNT | 0.0167 |
| 17 | DNT | DNT | DNT | DNT | DNT | 0.01613 |

### Table 11
| Attenuated Kerma near HVL (mGy) |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Number of Additional 1 mil Shims | Tube Potential (Initial Number of 8 mil Shims) |  |  |  |  |  |
|  | 40 kV (8 Shims) | 50 kV (10 Shims) | 60 kV (11 Shims) | 64 kV (12 Shims) | 70 kV (13 Shims) | 80 kV (14 Shims) |
| 1 | 0.003142 | 0.006139 | 0.009909 | 0.01121 | 0.01343 | 0.01828 |
| 2 | 0.00311 | 0.006133 | 0.009866 | 0.01114 | 0.01328 | 0.01784 |
| 3 | 0.003054 | 0.006109 | 0.00979 | 0.01106 | 0.01313 | 0.01773 |
| 4 | 0.003022 | 0.006062 | 0.009655 | 0.01093 | 0.01303 | 0.01778 |
| 5 | 0.003025 | 0.005951 | 0.009613 | 0.01091 | 0.01295 | 0.01771 |
| 6 | 0.002946 | 0.005904 | 0.009618 | 0.01091 | 0.013 | 0.01754 |
| 7 | 0.002928 | 0.005857 | 0.009525 | 0.01074 | 0.01289 | 0.01734 |

### Table 12
| HVLs for MX1 System (mGy) |  |  |  |  |
| --- | --- | --- | --- | --- |
| Tube Potential | HVL (mm Al) |  |  |  |
|  | Rough HVL (Thickest measurement before half value) | Calculated, precise HVL | IEC Minimum | Pass/Fail |
| 40 kV | 1.6764 | 1.705 | 1.5 | Pass |
| 50 kV | 2.032 | 2.063 | 1.8 | Pass |
| 60 kV | 2.3114 | 2.377 | 2.2 | Pass |
| 64 kV | 2.4892 | 2.519 | 2.32 | Pass |
| 70 kV | 2.667 | 2.677 | 2.5 | Pass |
| 80 kV | 2.9972 | 3.036 | 2.9 | Pass |

### Table 13
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | Engineering Quality Engineering Regulatory Affairs | 25 May 2024 | 24-273 |
