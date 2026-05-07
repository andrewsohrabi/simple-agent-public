# VVPR-P01-152 Rev B: Pediatric Filtration Verification Protocol and Report

## Metadata
- Document ID: VVPR-P01-152
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-152- Pediatric Filtration Verification Protocol and Report_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-152- Pediatric Filtration Verification Protocol and Report_B-signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to  collect testing for total filtration data with the added pediatric filtration puck of the MX1 Portable X-ray System, as described in IEC 60601-2-54.
OBJECTIVE
The primary purpose of this study is to collect the total filtration data with the added pediatric filter for the MX1 Portable X-ray System in accordance with IEC 60601-2-54 Clause 203.7.1  “HALF-VALUE LAYERS and TOTAL FILTRATION in X-RAY EQUIPMENT”
REFERENCES
IEC 60601-1 Edition 3.2 2020
IEC 60601-1-3 Edition 2.2 2021-01
IEC 60601-2-54 Edition 2.0 2022-09
IFU-MX1 - Instructions for Use, Rev D
MATERIALS
Equipment:
MX1 Portable X-ray System (Rev E) Components:
E1 emitter including
Pediatric Filter attachment
C1 cassette)
Lab Laptop with Accu-Gold 2.0 Radiation Measurement Software
Radcal AGDM+ Accu-Gold Digitizer (EQP-109 or equivalent)
Radcal AGMS-DM+ Accu-Gold Multi-Sensor (EQP-110 or equivalent)
Radiation PPE as necessary (ie. Lead vest, radiation monitor, etc.)
Narrow Beam Test Fixture (T-174 Rev A)
MX1 Emitter Mounting Arm (K1, L1 or equivalent)
Beam Alignment Phantom (EQP-057)
8 mil Aluminum Shims (T-178 Rev A)
Mars1417x Detector (T-180 Rev A)
SAMPLE SIZE
Per IEC 60601-1-3:2021, “Readers of this collateral standard are reminded that, in accordance with IEC 60601-1, Clause 5, all the test procedures described are TYPE TESTS, intended to be carried out in a dedicated testing environment in order to determine compliance.” Clause 5 of IEC 60601-1:2020 states that tests in scope of the standard are considered Type Tests. Type Tests are performed on a single representative sample of the item being tested (n=1). Therefore, a sample size of n=1 will be used to collect the filtration data.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL.
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
Experimental Procedure
The procedure for imaging modalities of the device should follow IFU-MX1, Instructions for Use. Additional steps for verification are listed below as necessary.
Place the Narrow Beam Fixture (T-174) under the MX1 Emitter Mounting Arm
Place the Mars detector (T-180) under the Narrow Beam Fixture
Remove the front face assembly (MS-10132) and permanent added filter (M10073) from the E1 Emitter (to be performed by trained MedAI production personnel)
Place the E1 Emitter on the Mounting Arm
Connect the Multi-Sensor (EQP-110 or equivalent) to the Digitizer (EQP-109 or equivalent) and the Digitizer to the laptop
Turn the MX1 System on in engineering mode
System shall automatically collimate to the max detector area
Turn on the detector
Initiate Accu-Gold software
Set the Multi-Sensor as the trigger sensor within the Accu-Gold software
Conduct any background noise correction as recommended by the Accu-Gold software/system
Pre-radiation/imaging
Place the Beam Alignment Phantom (EQP-057) inside the Narrow Beam Fixture at the center of the stage
Set a technique of 40 kV 0.4 mAs on the E1 emitter
Prepare detector for Automatic Exposure Detection (AED) mode
Trigger the E1 emitter
Adjust the position of the E1 emitter until the emitter’s beam is inline with the Beam Alignment Phantom
Replace the Beam Alignment Phantom with the Multi-Sensor such that the top half of the sensor is centered in the beam
Ensure no aluminum is clamped in the Narrow Beam Fixture
Set a technique of 64 kV 0.4 mAs on the E1 emitter
Dosimetric Measurements
Trigger E1 Emitter
Verify registration of dose in Accu-Gold software
Add 1 sheet of 8 mil aluminum (T-178) to the Narrow Beam Fixture and clamp it down
Trigger E1 Emitter
Verify registration of dose in Accu-Gold software, label the measurement with the number of Al sheets present, and record it in Table A.1
Repeat steps 6.4.3.3-6.5.3.5 until 20 Al sheets have been added
Repeat steps 6.4.3.1-6.4.3.6 with a technique of 80 kV and 0.4 mAs
Remove aluminum in path and attach aluminum bracket, plastic bottom plate, and pediatric filter to E1 Emitter
Set kV to 64 kV and 0.4 mAs
Trigger E1 Emitter
Verify registration of dose in Accu-Gold software, label measurement according to kV and filtration, and record it in Table A.2
Repeat step 6.4.3.10 - 6.4.3.11 at 80 kV tube potentials
Run MATLAB script to precisely estimate the aluminum equivalent of the pediatric filter materials and record the results in Table A.2
Data Calculation and Analysis
The data reported from the Accu-Gold includes kerma and kerma rate, and each measurement point will be recorded in the data sheet template in Appendix A.
An exponential fit will be applied to the aluminum filtration measurements to create a model predicting thickness of aluminum for any given attenuated dose.
The attenuated dose of the emitter and pediatric filters will be compared against the model to calculate the QEF of the emitter and pediatric materials. This information shall be added to the MX1 Instructions for Use.
ACCEPTANCE CRITERIA
For the pediatric filtration puck, the acceptance criteria in IEC 60601-2-54 clause 203.7.1 is used.
X-RAY EQUIPMENT specified for pediatric applications shall be provided with means for placing an ADDED FILTER [and resulting in a total filtration] of not less than 0,1 mm Cu or 3,5 mm Al.
APPENDICES
Appendix A - Data Sheet Template
Table A.1:  Aluminum Attenuated Kerma
Tested By: ______________________________________________   Date:_________________________________
Table A.2:  Calculated Equivalent Aluminum Thickness
Tested By: ______________________________________________   Date:_________________________________
Table A.3: Equipment Table
Table A.4: Device Configuration
Recorded By: ______________________________________________   Date:_________________________________
MATLAB Script
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
The added filtration (emitter front face, internal aluminum filter) and pediatric filter were not connected to the emitter. Instead, they were placed in the narrow beam fixture in the beam’s path to better follow the spirit of IEC guidance, measuring all added filtration as a combined, aggregate block. The aggregate added, non-removable filtration for the MX1 system was also measured without the pediatric filter to generate more data for the MX1 IFU.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table A.3: Equipment Table
MX1 Portable X-ray System Rev E  Components:
Table A.4: Device Configuration
Tested By: Chris Holland   Date:    May 9, 2024
RESULTS
Alignment
Fixture was aligned with the focal spot phantom (EQP-057). The following alignment image was captured:
The dosimeter was centered in the beam, as shown with the following image:
At the end of testing, a final image was taken to show that the dosimeter had not moved significantly during testing. The final image is shown here:
Data Collection
Table A.1:  Aluminum Attenuated Kerma
Tested By: Riley Compton   Date:    May 9, 2024
Table A.2:  Calculated Equivalent Aluminum Thickness for MX1 Added Filtration with Pediatric Filter
Tested By: Riley Compton   Date:    May 9, 2024
Table A.3:  Calculated Equivalent Aluminum Thickness for MX1 Added Filtration without Pediatric Filter
Tested By: Riley Compton   Date:    May 9, 2024
CONCLUSION
The MX1 system pediatric filter is compliant with the criteria in IEC 60601-2-54 clause 203.7.1, with the summation of all ADDED FILTERS, removable and non-removable, exceeding the minimum total of 3.5 mm Al equivalent added filtration.
IEC defines ADDED FILTER as “Removable or irremovable FILTER positioned in the RADIATION BEAM to provide part or all of the ADDITIONAL FILTRATION.”
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Attenuated Kerma for Aluminum (mGy) |  |  |
| --- | --- | --- |
| Number of Shims (8 mil shim) | Tube Potential |  |
|  | 64 kV | 80 kV |
| 0 |  |  |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |
| 9 |  |  |
| 10 |  |  |
| 11 |  |  |
| 12 |  |  |
| 13 |  |  |
| 14 |  |  |
| 15 |  |  |
| 16 |  |  |
| 17 |  |  |
| 18 |  |  |
| 19 |  |  |
| 20 |  |  |

### Table 2
| Equivalent Thickness of Al for MX1 System With Pediatric Filter (mGy) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Tube Potential | Measured Dose (mGy) | Average Dose (mGy) | Equivalent Thickness Al (mm Al) | IEC Minimum (mm Al) | Pass/Fail |
| 64 kV |  |  |  | 3.5 |  |
| 80 kV |  |  |  | 3.5 |  |

### Table 3
| Equipment ID | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- |

### Table 4
| Device Serial Number: |  |
| --- | --- |
| Software Version: |  |

### Table 5
| %% Aluminum shim = 0.2036; %% mm per 8 mil shim baseline = 0000; % ENTER avg dose of baseline xx = [[1:11].*shim]; yy = [0 0 0 0 0 0 0 0 0 0 0]; % ENTER avg doses of attenuated beam from aluminum shims yy = yy/baseline; % divide by baseline dose rate fitAl = getFit(xx,yy); plotCurve('Al Filtration', xx, yy, fitAl); HVL = predX(0.5, fitAl); detectorDose = 0; % ENTER avg dose of attenuated beam from pediatric puck materials kPuck = detectorDose/baseline; puckQEF = predX(kFullStack, fitAl); % Output QEF %% Functions function curveOutput = getFit(xVals, yVals) fitfun = fittype( @(a,b,c,x) a+b*exp(-c*x)); [curveOutput,gof] = fit(xVals',yVals',fitfun,'StartPoint',[1,1,1]); end function predictedVal = predX(yVal, fitted_curve) coeffs = coeffvalues(fitted_curve); predictedVal = log((yVal-coeffs(1))/coeffs(2))/-coeffs(3); end function plotCurve(titleStr, xx, yy, fitted_curve) hold off scatter(xx,yy); hold on plot(xx(1):0.01:xx(end),fitted_curve(xx(1):0.01:xx(end)), 'lineWidth', 2); legend('Measured', 'Model'); xlabel('Thickness (mm)'); ylabel('Normalized KERMA'); title(titleStr); end |
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
| EQP-158 | Radcal AGDM+ Accu-Gold Digitizer | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| EQP-159 | Radcal AGMS-DM+ Accu-Gold Multi-Sensor | 13-Jul-2023 | 13-Jul-2024 | Chris Holland 5/9/24 |
| T-174 Rev A | Narrow Beam Test Fixture | N/A | N/A | Chris Holland 5/9/24 |
| EQP-057 | Beam Alignment Phantom | N/A | N/A | Chris Holland 5/9/24 |
| T-180 Rev A | Mars 1417x Detector | N/A | N/A | Chris Holland 5/9/24 |
| T-178 Rev A | 8 mil Aluminum Shims | N/A | N/A | Chris Holland 5/9/24 |
| M10469 LOT 10138 | Pediatric Filter Attachment | N/A | N/A | Chris Holland 5/9/24 |
| N/A | K1 Cart Arm | N/A | N/A | Chris Holland 5/9/24 |
| N/A | Accu-Gold 3.0 Radiation Measurement Software | N/A | N/A | Chris Holland 5/9/24 |

### Table 9
| Device Serial Number: | E1 Emitter: 1216    C1 Cassette: 1217 |
| --- | --- |
| Software Version: | v3.0.0-gamma |

### Table 10
| Attenuated Kerma for Aluminum (mGy) |  |  |
| --- | --- | --- |
| Number of Shims (8 mil shim) | Tube Potential |  |
|  | 64 kV | 80 kV |
| 0 | 0.03785 | 0.05628 |
| 1 | 0.03475 | 0.05250 |
| 2 | 0.03183 | 0.04818 |
| 3 | 0.02951 | 0.04504 |
| 4 | 0.02731 | 0.04246 |
| 5 | 0.02530 | 0.03983 |
| 6 | 0.02372 | 0.03763 |
| 7 | 0.02218 | 0.03542 |
| 8 | 0.02077 | 0.03345 |
| 9 | 0.01967 | 0.03181 |
| 10 | 0.01845 | 0.03028 |
| 11 | 0.01735 | 0.02868 |
| 12 | 0.01642 | 0.02786 |
| 13 | 0.01557 | 0.02632 |
| 14 | 0.01469 | 0.02506 |
| 15 | 0.01392 | 0.02393 |
| 16 | 0.01324 | 0.02304 |
| 17 | 0.01265 | 0.02204 |
| 18 | 0.01207 | 0.02101 |
| 19 | 0.01145 | 0.02016 |
| 20 | 0.01084 | 0.01925 |

### Table 11
| Equivalent Thickness of Al for MX1 System With Pediatric Filter (mGy) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Tube Potential | Measured Dose (mGy) | Average Dose (mGy) | Equivalent Thickness Al (mm Al) | IEC Minimum (mm Al) | Pass/Fail |
| 64 kV | 0.01164 | 0.01159 | 3.858 | 3.5 | Pass |
|  | 0.01161 |  |  |  |  |
|  | 0.01151 |  |  |  |  |
| 80 kV | 0.01928 | 0.01912 | 4.254 | 3.5 | Pass |
|  | 0.01914 |  |  |  |  |
|  | 0.01895 |  |  |  |  |

### Table 12
| Equivalent Thickness of Al for MX1 System Without Pediatric Filter (mGy) |  |  |
| --- | --- | --- |
| Tube Potential | Average Dose (mGy) | Equivalent Thickness Al (mm Al) |
| 64 kV | 0.02412 | 1.199 |
| 80 kV | 0.03431 | 1.553 |

### Table 13
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Report release | Engineering Quality Engineering Regulatory Affairs | 27 May 2024 | 24-272 |
