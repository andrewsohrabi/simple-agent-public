# MEMO-P01-672 Rev A: Dosimetric Indications

## Metadata
- Document ID: MEMO-P01-672
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-672 Dosimetric Indications_A-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-672 Dosimetric Indications_A-signed.docx
- Extraction warnings: none

## Extracted Content
OBJECTIVE
The objective of this study is to collect dosimetric information for inclusion in the MX1 Portable X-ray System Instructions for Use (IFU) in accordance with IEC 60601-2-54.
REFERENCES
IEC 60601-2-54 Edition 1.2 2018 Medical electrical equipment – Part 2-54: Particular requirements for the basic safety and essential performance of X-ray equipment for radiography and radioscopy Section  203.5.2.4.5  “Test for dosimetric information”
IFU-MX1 Rev. 2 - Instructions for Use
MATERIALS
Equipment:
MX1 Portable X-ray System Rev E (E1 Emitter, C1 Cassette)
M10469 Pediatric Filter Rev. A
Laptop with Accu-Gold 2.0 Radiation Measurement Software
Radcal AGDM+ Accu-Gold Digitizer (EQP-109, SN: 48-2813)
Radcal AGMS-DM+ Accu-Gold Multi-Sensor (EQP-110, SN: 43-1924)
PMMA Test Phantom (T-181 Rev A)
EOL Test Fixture (T-129 Rev A)
Radiation PPE as necessary (ie. Lead vest, radiation monitor, etc.)
MATLAB Script for Dose Fitting (Appendix A)
SAMPLE SIZE
This is a type test per IEC 60601-1:2020 Clause 5.2. Therefore, this test will utilize a sample size of 1.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
Training Requirements
Participants should be trained in how to operate the MX1 System and test equipment.
Background
For this test to generate a table of dose values in the IFU, multiple Source to Image Distances (SIDs) for the MX1 System will be tested.
Experimental Procedure
The procedure for imaging modalities of the device will follow IFU-MX1 - Instructions for Use for all possible loading factor configurations listed in Appendix A. Additional steps for verification are listed below as necessary.
Place the emitter and cassette into the EOL Test Fixture (T-129)
Turn the MX1 System on
System shall automatically collimate to the max detector area
Pre-radiation/imaging
Place the PMMA Phantom (T-181) on top of the cassette
Place the EQP-110 RadCal Multi-Sensor orthogonally centered about the focal spot at half way between the E1 emitter focal spot and the top of the PMMA Phantom per the configurations in Appendix A.
Connect the EQP-110 Multi-Sensor and EQP-109 Digitizer to the PC
Initiate Accu-Gold software
Set the Multi-Sensor as the trigger sensor within the Accu-Gold software
Conduct any background noise correction as recommended by the Accu-Gold software/system
Dosimetric Measurements
Setup MX1 Emitter at initial loading factors and initial SID per Appendix A
Trigger MX1 Emitter
Verify registration of dose in Accu-Gold software
Repeat steps 1-3 for the remaining loading factor combinations in Appendix A
Calculate air kerma at surface of detector based on the inverse square law as described in Section 5.6 of this protocol
Repeat steps 5.5.3.1-5.5.3.5 with pediatric filter present
Data Calculation and Analysis
The data reported from the Accu-Gold software is captured halfway between the focal spot and top of the PMMA block  (the location of the EQP-110 Radcal Multi-Sensor). Radiation dose outputs in the IFU will be expressed at a reference point 15 cm above the flat panel detector, which is what is reported to the user by the MX1 Software. Dose outputs at the surface of the detector will be calculated using the Inverse Square Law Equation:
Kr1 = Dose Reading Measured at ½ SID
d1 = RadCal SID
Kr2 =  Calculated Dose at the Surface of the Flat Panel Detector
d2 = Distance to reference point
Dose at the reference point 15 cm above the detector will therefore be calculated as:
The raw data will be placed into the MATLAB script in Appendix A. This script will produce fitted coefficients to the following model:
The script takes the kerma value and divides by the charge (Q) and multiplies by distance squared (d2) to isolate the relationship between the non-linear independent variable of tube potential (V) and kerma (K). This distance/charge normalized data is fit to this equation:
The resultant fit coefficients and a scatter plot of measurements vs line plot of regression will be reported in Appendix A.
APPENDICES
Appendix A - Data Sheet Templates with Loading Factor Combinations to be Tested
Raw data measured at halfway between PMMA and emitter using the methods in Section 5:
Table A.1: Standard Doses
Table A.2: Pediatric Doses
Calculated results at the reference point 15 cm above the flat panel detector per the methods in Section 5:
Table A.3: Standard Doses
Table A.4: Pediatric Doses
Table A.5
Graph A.1: Measured (Scatter Points) versus Predicted (Regression) Air Kerma
Graph A.2: Measured (Scatter Points) versus Predicted (Regression) Air Kerma with Pediatric Filter
MATLAB Script for Dose Model Fitting
DOCUMENT REVISION HISTORY
Digital Key:
example.com/

### Table 1
| To: | File |
| --- | --- |
| From: | Engineering |

### Table 2
| Air Kerma for MX1 System (mGy) At Halfway Between PMMA and Focal Spot |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SID (cm) | Distance to Dosimeter | Tube Voltage (kV) | Current-Time Product (mAs) |  |  |  |  |
|  |  |  | 0.04 | 0.08 | 0.16 | 0.25 | 0.40 |
| 40 | 12.6 | 40 | 0.03048 | 0.05679 | 0.1198 | 0.1863 | 0.2952 |
|  |  | 50 | 0.05896 | 0.1149 | 0.2396 | 0.3694 | 0.5864 |
|  |  | 60 | 0.09291 | 0.1857 | 0.3750 | 0.5706 | 0.9398 |
|  |  | 64 | 0.1051 | 0.2086 | 0.4212 | 0.6754 | 1.088 |
|  |  | 70 | 0.1315 | 0.2529 | 0.5036 | 0.8148 | 1.273 |
|  |  | 80 | 0.1605 | 0.3344 | 0.6565 | 1.041 | 1.662 |
| 60 | 19.3 | 40 | 0.01286 | 0.02415 | 0.05017 | 0.07913 | 0.1277 |
|  |  | 50 | 0.02562 | 0.04884 | 0.09975 | 0.1572 | 0.2514 |
|  |  | 60 | 0.03917 | 0.07867 | 0.1556 | 0.2445 | 0.3927 |
|  |  | 64 | 0.04499 | 0.08890 | 0.1795 | 0.2840 | 0.4550 |
|  |  | 70 | 0.05307 | 0.1080 | 0.2161 | 0.3404 | 0.5441 |
|  |  | 80 | 0.06983 | 0.1428 | 0.2819 | 0.4468 | 0.7170 |
| 80 | 29.9 | 40 | 0.005801 | 0.01016 | 0.02172 | 0.03348 | 0.05371 |
|  |  | 50 | 0.01073 | 0.02076 | 0.04208 | 0.06626 | 0.1054 |
|  |  | 60 | 0.01650 | 0.03214 | 0.06568 | 0.1084 | 0.1653 |
|  |  | 64 | 0.01954 | 0.03745 | 0.07482 | 0.1174 | 0.1878 |
|  |  | 70 | 0.02247 | 0.04526 | 0.09010 | 0.1414 | 0.2271 |
|  |  | 80 | 0.02748 | 0.05969 | 0.1178 | 0.1858 | 0.2957 |

### Table 3
| Air Kerma for MX1 System (mGy) At Halfway Between PMMA and Focal Spot |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SID (cm) | Distance to Dosimeter | Tube Voltage (kV) | Current-Time Product (mAs) |  |  |  |  |
|  |  |  | 0.04 | 0.08 | 0.16 | 0.25 | 0.40 |
| 40 | 12.6 | 40 | 0.01061 | 0.01891 | 0.04053 | 0.06324 | 0.1023 |
|  |  | 50 | 0.02517 | 0.04808 | 0.09932 | 0.1563 | 0.2567 |
|  |  | 60 | 0.04395 | 0.08695 | 0.1731 | 0.2759 | 0.4525 |
|  |  | 64 | 0.05246 | 0.1035 | 0.2092 | 0.3295 | 0.5288 |
|  |  | 70 | 0.06688 | 0.1346 | 0.2601 | 0.4140 | 0.6781 |
|  |  | 80 | 0.08782 | 0.1812 | 0.3804 | 0.5964 | 0.9083 |
| 60 | 19.3 | 40 | 0.004594 | 0.008214 | 0.01721 | 0.02641 | 0.04257 |
|  |  | 50 | 0.01075 | 0.02041 | 0.04198 | 0.06559 | 0.1054 |
|  |  | 60 | 0.01864 | 0.03646 | 0.07379 | 0.1164 | 0.1876 |
|  |  | 64 | 0.02220 | 0.04381 | 0.08809 | 0.1383 | 0.2217 |
|  |  | 70 | 0.02822 | 0.05551 | 0.1117 | 0.1761 | 0.2819 |
|  |  | 80 | 0.03859 | 0.07846 | 0.1557 | 0.2472 | 0.4034 |

### Table 4
| Air Kerma for MX1 System (mGy) At Reference Point |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SID (cm) | Distance to Reference Point (cm) | Tube Voltage (kV) | Current-Time Product (mAs) |  |  |  |  |
|  |  |  | 0.04 | 0.08 | 0.16 | 0.25 | 0.40 |
| 40 | 25 | 40 | 0.007742 | 0.014426 | 0.030431 | 0.047323 | 0.074986 |
|  |  | 50 | 0.014977 | 0.029186 | 0.060862 | 0.093834 | 0.148955 |
|  |  | 60 | 0.023601 | 0.047171 | 0.095256 | 0.144942 | 0.238724 |
|  |  | 64 | 0.026697 | 0.052988 | 0.106992 | 0.171562 | 0.276369 |
|  |  | 70 | 0.033403 | 0.064241 | 0.127922 | 0.206972 | 0.323362 |
|  |  | 80 | 0.04077 | 0.084943 | 0.166762 | 0.264431 | 0.422175 |
| 60 | 45 | 40 | 0.002366 | 0.004442 | 0.009229 | 0.014556 | 0.02349 |
|  |  | 50 | 0.004713 | 0.008984 | 0.018349 | 0.028916 | 0.046244 |
|  |  | 60 | 0.007205 | 0.014471 | 0.028622 | 0.044975 | 0.072235 |
|  |  | 64 | 0.008276 | 0.016353 | 0.033018 | 0.052241 | 0.083695 |
|  |  | 70 | 0.009762 | 0.019866 | 0.039751 | 0.062615 | 0.100085 |
|  |  | 80 | 0.012845 | 0.026267 | 0.051854 | 0.082187 | 0.131889 |
| 80 | 65 | 40 | 0.001227 | 0.00215 | 0.004596 | 0.007084 | 0.011365 |
|  |  | 50 | 0.00227 | 0.004393 | 0.008904 | 0.014021 | 0.022303 |
|  |  | 60 | 0.003491 | 0.006801 | 0.013898 | 0.022937 | 0.034977 |
|  |  | 64 | 0.004135 | 0.007924 | 0.015832 | 0.024842 | 0.039738 |
|  |  | 70 | 0.004755 | 0.009577 | 0.019065 | 0.02992 | 0.048054 |
|  |  | 80 | 0.005815 | 0.01263 | 0.024926 | 0.039315 | 0.06257 |

### Table 5
| Air Kerma for MX1 System (mGy) At Reference Point |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SID (cm) | Distance to Dosimeter | Tube Voltage (kV) | Current-Time Product (mAs) |  |  |  |  |
|  |  |  | 0.04 | 0.08 | 0.16 | 0.25 | 0.40 |
| 40 | 25 | 40 | 0.002695 | 0.004803 | 0.010295 | 0.016064 | 0.025986 |
|  |  | 50 | 0.006394 | 0.012213 | 0.025229 | 0.039703 | 0.065206 |
|  |  | 60 | 0.011164 | 0.022087 | 0.04397 | 0.070083 | 0.114942 |
|  |  | 64 | 0.013326 | 0.026291 | 0.05314 | 0.083698 | 0.134324 |
|  |  | 70 | 0.016989 | 0.034191 | 0.06607 | 0.105163 | 0.172248 |
|  |  | 80 | 0.022308 | 0.046028 | 0.096628 | 0.151495 | 0.230723 |
| 60 | 45 | 40 | 0.000845 | 0.001511 | 0.003166 | 0.004858 | 0.007831 |
|  |  | 50 | 0.001977 | 0.003754 | 0.007722 | 0.012065 | 0.019388 |
|  |  | 60 | 0.003429 | 0.006707 | 0.013573 | 0.021411 | 0.034508 |
|  |  | 64 | 0.004084 | 0.008059 | 0.016204 | 0.02544 | 0.040781 |
|  |  | 70 | 0.005191 | 0.010211 | 0.020547 | 0.032393 | 0.051854 |
|  |  | 80 | 0.007098 | 0.014432 | 0.02864 | 0.045471 | 0.074204 |

### Table 6
| Fitted Coefficients for Dose Model |  |  |  |
| --- | --- | --- | --- |
| Model | Coefficient |  |  |
|  | α | b | c |
| Standard | 1.4162 | 1.4588 | -189.98 |
| Pediatric | 0.0462 | 2.0839 | -61.091 |

### Table 7
| %% Normal (no pediatric filter) dose_data = [0.03050.05680.11980.18630.29520.0590.11490.23960.36940.58640.09290.18570.3750.57060.93980.10510.20860.42120.67541.0880.13150.25290.50360.81481.2730.16050.33440.65651.0411.6620.01290.02420.05020.07910.12770.02560.04880.09980.15720.25140.03920.07870.15560.24450.39270.0450.08890.17950.2840.4550.05310.1080.21610.34040.54410.06980.14280.28190.44680.7170.00580.01020.02170.03350.05370.01070.02080.04210.06630.10540.01650.03210.06570.10840.16530.01950.03740.07480.11740.18780.02250.04530.09010.14140.22710.02750.05970.11780.18580.2957]'; kV = repmat([40*ones(5,1)' 50*ones(5,1)' 60*ones(5,1)' 64*ones(5,1)' 70*ones(5,1)' 80*ones(5,1)']',3,1); mAs = repmat([0.04 0.08 0.16 0.25 0.40]',18,1); SID = [12.6 19.3 29.9]; SID = [SID(1)*ones(30,1)' SID(2)*ones(30,1)' SID(3)*ones(30,1)']'; dose_d_norm = dose_data.*(SID).^2; dose_d_mas_norm = dose_d_norm./mAs; %% fitfun = fittype( @(a,b,c,x) a*(x).^b+c); [kv2dose,gof] = fit(kV,dose_d_mas_norm,fitfun,'StartPoint',[1,2,0]); coeffs = coeffvalues(kv2dose) %% mAs_xx = [0.04, 0.08, 0.16, 0.25, 0.40]; kV_xx = [40 50 60 64 70 80]; colors = ["#0072BD" "#D95319" "#EDB120" "#7E2F8E" "#77AC30" "#4DBEEE" "#A2142F"]; subplot(1,3,1) for i = 1:6 if i == 1 hold off elseif i == 2 hold on end plot(mAs_xx, kv2dose(kV_xx(i)).*mAs_xx./12.6^2,'Color',colors(i)) end for i = 1:6 scatter(mAs_xx, dose_data(5*i-4:5*i),[],hex2rgb(colors(i)),"filled") end legend(["40 kV" "50 kV" "60 kV" "64 kV" "70 kV" "80 kV"]) title("Dose by kV and mAs for 12.6 cm SID") xlabel("Charge (mAs)") xlim([0.04 0.4]) ylabel("Dose (mGy)") subplot(1,3,2) for i = 1:6 if i == 1 hold off elseif i == 2 hold on end plot(mAs_xx, kv2dose(kV_xx(i)).*mAs_xx./19.3^2,'Color',colors(i)) end for i = 1:6 scatter(mAs_xx, dose_data(5*i-4+30:5*i+30),[],hex2rgb(colors(i)),"filled") end legend(["40 kV" "50 kV" "60 kV" "64 kV" "70 kV" "80 kV"]) title("Dose by kV and mAs for 19.3 cm SID") xlabel("Charge (mAs)") xlim([0.04 0.4]) ylabel("Dose (mGy)") subplot(1,3,3) for i = 1:6 if i == 1 hold off elseif i == 2 hold on end plot(mAs_xx, kv2dose(kV_xx(i)).*mAs_xx./29.9^2,'Color',colors(i)) end for i = 1:6 scatter(mAs_xx, dose_data(5*i-4+60:5*i+60),[],hex2rgb(colors(i)),"filled") end legend(["40 kV" "50 kV" "60 kV" "64 kV" "70 kV" "80 kV"]) title("Dose by kV and mAs for 29.9 cm SID") xlabel("Charge (mAs)") xlim([0.04 0.4]) ylabel("Dose (mGy)") %% Pediatric Filter dose_data_ped = [0.01060.01890.04050.06320.10230.02520.04810.09930.15630.25670.0440.08690.17310.27590.45250.05250.10350.20920.32950.52880.06690.13460.26010.4140.67810.08780.18120.38040.59640.90830.00460.00820.01720.02640.04260.01070.02040.0420.06560.10540.01860.03650.07380.11640.18760.02220.04380.08810.13830.22170.02820.05550.11170.17610.28190.03860.07850.15570.24720.4034]'; kV = repmat([40*ones(5,1)' 50*ones(5,1)' 60*ones(5,1)' 64*ones(5,1)' 70*ones(5,1)' 80*ones(5,1)']',2,1); mAs = repmat([0.04 0.08 0.16 0.25 0.40]',12,1); SID = [12.6 19.3]; SID = [SID(1)*ones(30,1)' SID(2)*ones(30,1)']'; dose_d_norm = dose_data_ped.*(SID).^2; dose_d_mas_norm = dose_d_norm./mAs; %% fitfun = fittype( @(a,b,c,x) a*(x).^b+c); [kv2dose,gof] = fit(kV,dose_d_mas_norm,fitfun,'StartPoint',[1,2,0]); coeffs = coeffvalues(kv2dose) %% mAs_xx = [0.04, 0.08, 0.16, 0.25, 0.40]; kV_xx = [40 50 60 64 70 80]; colors = ["#0072BD" "#D95319" "#EDB120" "#7E2F8E" "#77AC30" "#4DBEEE" "#A2142F"]; subplot(1,2,1) for i = 1:6 if i == 1 hold off elseif i == 2 hold on end plot(mAs_xx, kv2dose(kV_xx(i)).*mAs_xx./12.6^2,'Color',colors(i)) end for i = 1:6 scatter(mAs_xx, dose_data_ped(5*i-4:5*i),[],hex2rgb(colors(i)),"filled") end legend(["40 kV" "50 kV" "60 kV" "64 kV" "70 kV" "80 kV"]) title("Dose by kV and mAs for 12.6 cm SID") xlabel("Charge (mAs)") xlim([0.04 0.4]) ylabel("Dose (mGy)") subplot(1,2,2) for i = 1:6 if i == 1 hold off elseif i == 2 hold on end plot(mAs_xx, kv2dose(kV_xx(i)).*mAs_xx./19.3^2,'Color',colors(i)) end for i = 1:6 scatter(mAs_xx, dose_data_ped(5*i-4+30:5*i+30),[],hex2rgb(colors(i)),"filled") end legend(["40 kV" "50 kV" "60 kV" "64 kV" "70 kV" "80 kV"]) title("Dose by kV and mAs for 19.3 cm SID") xlabel("Charge (mAs)") xlim([0.04 0.4]) ylabel("Dose (mGy)") |
| --- |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering | 24 May 2024 | 24-228 |
