# VVPR-P01-150 Rev B: Focal Spot Size Measurement Verification Protocol Report

## Metadata
- Document ID: VVPR-P01-150
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-150 - Focal Spot Size Measurement Verification Protocol  Report_B-signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-150 - Focal Spot Size Measurement Verification Protocol  Report_B-signed.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify the focal spot size measurement meets requirements for the MX1 Portable X-ray System.
OBJECTIVE
The objective of this study is to collect focal spot dimension data for inclusion in the MX1 Portable X-ray System Instructions for Use (IFU) in accordance with IEC 60336.
REFERENCES
IEC 60336 Edition 2005 Medical electrical equipment - X-ray tube assemblies for medical diagnosis - Characteristics of focal spots
IFU-MX1 Rev. D - Instructions for Use
MATERIALS
Equipment:
MX1 Portable X-ray System Rev E  Components:
E1 Emitter
C1 Cassette
MX1 Focal Spot Test Fixture (T-175 Rev A)
Radiation PPE as necessary (ie. Lead vest, radiation monitor, etc.)
Beam Alignment Phantom (EQP-057)
Diagonomatic Pro-Project Slit Phantom (T-176 Rev A)
MATLAB Focal Spot Evaluation Script (Included in Appendix A)
Mars1417X Detector (T-180 Rev A)
SAMPLE SIZE
This is a type test per IEC 60601-1:2020 Clause 5.2. Therefore, this test will utilize a sample size of 1.
METHODS
Locations
This study shall take place in MedAI facilities in Springfield, IL
Personnel
Testing shall be conducted by members of the MedAI Engineering team.
Experimental Procedure
The procedure for imaging modalities of the device will follow IFU-MX1 - Instructions for Use for all possible loading factor configurations listed in Appendix A. Additional steps for verification are listed below as necessary.
Place and align the MX1 Focal Spot Test Fixture (T-175) on top of the iRay Mars detector (T-180) such that the edges are parallel to one another
Rotate the Testing Fixture 2-5 degrees relative to the detector
Place the emitter and cassette into the MX1 Testing Fixture
Ensure the XY Table on the MX1 Testing Fixture is at 0 degrees rotation
Turn the MX1 System on
System shall automatically collimate to the max detector area
Turn on the iRay Mars detector
Pre-radiation/imaging
Place the Beam Alignment Phantom (EQP-057) on stage in MX1 Testing Fixture
Set a technique of 80 kV 0.4 mAs on the E1 Emitter
Prepare the iRay Mars detector for Automatic Exposure Detection (AED) mode
Trigger the E1 Emitter
Adjust the XY table in the MX1 Testing Fixture to move the Beam Alignment Phantom closer to in-line with the focal spot emission axis
Repeat steps 6.5.2.2 - 6.5.2.5 until the Beam Alignment Phantom is centered
Set a technique of 80 kV and 0.08 mAs on the E1 emitter
Slit Measurements
Replace Beam Alignment Phantom with Slit Phantom (T-176)
Prepare the iRay Mars detector for AED mode
Trigger the E1 Emitter for 8 seconds (DDR)
Repeat steps 6.5.3.1-6.5.3.3 for a total of 5 exposures
Save radiographs to folder
Rotate Slit Phantom 90 degrees to measure the orthogonal axis to the first measurement
Repeat steps 6.5.3.1-6.5.3.5
Run MATLAB script to calculate focal spot dimensions with captured radiographs
Data Calculation and Analysis
Each row of pixels in the slit radiograph produces a plot similar to this:
All of these plots are oversampled at 100x the resolution, and shifted to align with one another, as to increase the effective resolution of the detector and reduce noise of the measurement. This also corrects for the 2-5 degree angle of the detector.
The curve is normalized to [0, 1] and the MATLAB script then calculates the width of the curve at a value of 0.15, or 15% of the maximum. Finally, the magnification is corrected by dividing by :
The result is the reported value for length or width. The combination of length and width are cross referenced by IEC 60336 Table 3 - Maximum permissible values of FOCAL SPOT dimensions for NOMINAL FOCAL SPOT VALUES to determine the IEC focal spot bracket.
ACCEPTANCE CRITERIA
The acceptance criteria for this study is the focal spot measurement shall have a combined IEC value ≤ 0.8. The results shall be documented in the MX1 Instructions for Use (IFU-MX1).
APPENDICES
Appendix A - Data Sheet Templates with the slit orientations to be tested and images of results
Appendix A - Data Sheet Templates
Table A.1: Focal Spot Measurements
Table A.2: Focal Spot Images and Measured Curves
Recorded By: _______________________________________Date:___________________
Table A.3: Equipment Table
MATLAB Focal Spot Script
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
It was decided to perform focal spot measurements on two units instead of one as they were both used for image study and validation.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Table A.3: Equipment Table
MX1 Portable X-ray System Rev E  Components:
Table A.4: Device Configuration (Unit 1)
Recorded By: Chris Holland   Date:    May 21, 2024
Table A.4.1: Device Configuration (Unit 2)
Recorded By: Chris Holland   Date:    May 24, 2024
RESULTS
Alignment
Unit 1
Stage was aligned with focal spot, as shown in image below:
Angle of slit was measured to be 2.78 degrees using image below:
Unit 2
Stage was aligned to focal spot, as shown in image below:
Angle was measured to be 2.94 degrees using image below
Data Collection
Table A.1: Focal Spot Measurements
Table A.2: Focal Spot Images and Measured Curves
Recorded By: Chris HollandDate: May 21, 2024
Table A.1.1: Unit 2 Focal Spot Measurements
Table A.2.1: Unit 2 Focal Spot Images and Measured Curves
Recorded By: Chris HollandDate: May 24, 2024
CONCLUSION
Both MX1 systems pass the MedAI requirement for focal spot dimensions for a 0.8 combined focal spot, as defined by IEC 60336.
Overall Result:.
Pass
Fail
Other:
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Full-Width 15% Max Measurements(Slit Radiograph) | IEC Focal Spot | Pass/Fail |  |
| --- | --- | --- | --- |
| Length (mm) | Width (mm) | Combined | MedAI 0.8 Requirement |

### Table 2
| Unit Under Test |  | Sign/Date: |
| --- | --- | --- |
| Serial Number: |  | Sign/Date: |
| Software Version: |  | Sign/Date: |

### Table 3
| Radiographs and Measured Curves |  |  |
| --- | --- | --- |
|  | Length (mm) | Width (mm) |
| Radiograph |  |  |
| Measured Curve |  |  |

### Table 4
| Equipment ID | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- |

### Table 5
| pitch = .100; sampleFactor = 100; subPitch = pitch/sampleFactor; magFactor = (972.5+2+7)/(257.95+53.0);%% denominator is slit to detector distance, numerator is slit to focal spot parent = 'placeholder\'; %% put parent folder here if not local dir img_size = [4352,3584]; imgLen = zeros(img_size); imgWid = zeros(img_size); path = strcat(parent, "length\"); files = dir(fullfile(path, "*.tif")); for i = 1:length(files) imgLen = imgLen + double(imread(strcat(path, files(i).name))); end path = strcat(parent, "width\"); files = dir(fullfile(path, "*.tif")); for i = 1:length(files) imgWid = imgWid + double(imread(strcat(path, files(i).name))); end parent = 'calibration\'; %% put parent folder here if not local dir img_size = [4352,3584]; imgLight = zeros(img_size); imgDark = zeros(img_size); path = strcat(parent, "light\"); files = dir(fullfile(path, "*.tif")); for i = 1:length(files) imgLight = imgLight + double(imread(strcat(path, files(i).name))); end path = strcat(parent, "dark\"); files = dir(fullfile(path, "*.tif")); for i = 1:length(files) imgDark = imgDark + double(imread(strcat(path, files(i).name))); end imgWid = (imgWid - imgDark)./imgLight; imgLen = (imgLen - imgDark)./imgLight; %% slitWid = cropImg(imgWid); slitLen = cropImg(imgLen); %% lsfWid = upsampleAngledSlit(slitWid, sampleFactor, 'slit'); lsfLen = upsampleAngledSlit(slitLen, sampleFactor, 'slit'); %% subplot(1,2,1) plotFocal(lsfWid, magFactor, subPitch,'FS Width') subplot(1,2,2) plotFocal(lsfLen, magFactor, subPitch, 'FS Length') %% Functions function [fw, leftIndex, rightIndex] = FWXM(curve, percMax) leftIndex = find(curve >= percMax*max(curve), 1, 'first'); rightIndex = find(curve >= percMax*max(curve), 1, 'last'); fw = (rightIndex-leftIndex); end function cropOut = cropImg(img) img = rescale(img); [~,cropBounds] = imcrop(rescale(log(img+1))); cropBounds = round(cropBounds); cropOut = img(cropBounds(2):cropBounds(2)+cropBounds(4),cropBounds(1):cropBounds(1)+cropBounds(3)); [~,cropBounds] = imcrop(rescale(cropOut)); cropBounds = round(cropBounds); cropOut = cropOut(cropBounds(2):cropBounds(2)+cropBounds(4),cropBounds(1):cropBounds(1)+cropBounds(3)); cropOut = rescale(cropOut); imshow(cropOut); if var(mean(cropOut,1)) < var(mean(cropOut,2)) % This translates the image if the slit is horizontal cropOut = cropOut'; end end function esf = upsampleAngledSlit(slit, sampleFactor, type) [slitRows, slitCols] = size(slit); Ibw = edge(slit,'canny',0.3); [H,theta,rho] = hough(Ibw,'Theta',-5.5:0.01:5.5); peaks  = houghpeaks(H,10); lines = houghlines(Ibw,theta,rho,peaks); figure, imshow(slit) hold on for k = 1:numel(lines) x1 = lines(k).point1(1); y1 = lines(k).point1(2); x2 = lines(k).point2(1); y2 = lines(k).point2(2); plot([x1 x2],[y1 y2],'Color','g','LineWidth', 2) end hold off w = waitforbuttonpress close theta = mean([lines.theta]); slitUps = imresize(slit,[slitRows,round(sampleFactor*slitCols)]); deltaMin = 1e3; thetaMin = 0; switch type case 'slit' [FW50M, left, right] = FWXM(slitUps(round(end/2),:),0.5); slitUpsBack = slitUps; slitUps = slitUps(:,max(1,left-5*FW50M):min(length(slitUps),right+5*FW50M)); case 'edge' end deltas = zeros(size([-1:0.01:1])); percLen = length([-1:0.01:1])+1; j = 1; for theta_adj = theta + [-1:0.01:1] pixelOffset = sampleFactor*(slitRows)*sind(theta_adj); offsetPerLine = pixelOffset/(height(slitUps)-1); edgeAdj = slitUps; for i = 1:height(slitUps) edgeAdj(i,:) = circshift(slitUps(i,:),round(offsetPerLine*(i+1-round(height(slitUps)/2)))); end switch type case 'slit' [FW50M1, left1, right1] = FWXM(mean(edgeAdj(1:round(end/2),:),1),0.5); [FW50MEnd, leftEnd, rightEnd] = FWXM(mean(edgeAdj(round(end/2):end,:),1),0.5); [~,peak1] = max(edgeAdj(1,:)); [~,peakEnd] = max(edgeAdj(end,:)); deltaLeft = leftEnd - left1; deltaRight = rightEnd - right1; deltaPeak = peakEnd - peak1; delta = abs(mean([deltaLeft,deltaRight,deltaPeak])); case 'edge' temp = trimdata(edgeAdj',.8*length(edgeAdj),Side="both")'; left = find(rescale(mean(temp(1:round(end/2),:),1))>0.5,1); right = find(rescale(mean(temp(round(end/2):end,:),1))>0.5,1); delta = abs(left-right); end if delta < deltaMin deltaMin = delta; thetaMin = theta_adj; end deltas(j) = delta; j = j + 1; disp(round(j/percLen*100,2)) end pixelOffset = sampleFactor*(slitRows)*sind(thetaMin); offsetPerLine = pixelOffset/(height(slitUps)-1); edgeAdj = slitUps; for i = 1:height(slitUps) edgeAdj(i,:) = circshift(slitUps(i,:),round(offsetPerLine*(i+1-round(height(slitUps)/2)))); end esf = rescale(mean(edgeAdj)); hold off plot(mean(edgeAdj(1:round(end/2),:),1)) hold on plot(mean(edgeAdj(round(end/2):end,:),1)) end function plotFocal(esf, magFactor, subPitch, titleStr) hold off plot(1000.*(0:subPitch/magFactor:(length(esf)-1)*subPitch/magFactor),esf) title(titleStr) xlabel('Distance in microns') ylabel('Relative Intensity') [FW15M, left, right] = FWXM(esf, 0.15); hold on plot((left:right)/magFactor,0.15*ones(right-left+1,1)) legend("Focal Spot", strcat("FW15M of ", num2str(FW15M/magFactor-20), "um")) end |
| --- |

### Table 6
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Regulatory Affairs Engineering | 08 May 2024 | 24-221 |

### Table 7
| Type of Document: | ☐ Interim Report | ☑ Final Report |
| --- | --- | --- |

### Table 8
| Equipment ID | Description | Last Calibration Date | Calibration Due Date | Signature & Date |
| --- | --- | --- | --- | --- |
| T-175 Rev A | MX1 Focal Spot Test Fixture | N/A | N/A | Chris Holland 5/21/24 |
| EQP-057 | Beam Alignment Phantom | N/A | N/A | Chris Holland 5/21/24 |
| T-176 Rev A | Diagonomatic Pro-Project Slit Phantom | N/A | N/A | Chris Holland 5/21/24 |
| T-180 Rev A | Mars 1417x Detector | N/A | N/A | Chris Holland 5/21/24 |

### Table 9
| Unit Under Test | dv25 | Sign/Date: Chris Holland 5/21/24 |
| --- | --- | --- |
| Serial Number: | E1 Emitter: 1220    C1 Cassette: 1079 | Sign/Date: Chris Holland 5/21/24 |
| Software Version: | v3.0.0-gamma | Sign/Date: Chris Holland 5/21/24 |

### Table 10
| Unit Under Test | dv23 | Sign/Date: Chris Holland 5/24/24 |
| --- | --- | --- |
| Serial Number: | E1 Emitter: 1220    C1 Cassette: 1079 | Sign/Date: Chris Holland 5/24/24 |
| Software Version: | v3.0.0-gamma | Sign/Date: Chris Holland 5/24/24 |

### Table 11
| Full-Width 15% Max Measurements(Slit Radiograph) | IEC Focal Spot | Pass/Fail |  |
| --- | --- | --- | --- |
| Length (mm) | Width (mm) | Combined | MedAI 0.8 Requirement |
| 1.52 | 1.06 | 0.8 | Pass |

### Table 12
| Radiographs and Measured Curves |  |  |
| --- | --- | --- |
|  | Length (mm) | Width (mm) |
| Radiograph |  |  |
| Measured Curve |  |  |

### Table 13
| Full-Width 15% Max Measurements(Slit Radiograph) | IEC Focal Spot | Pass/Fail |  |
| --- | --- | --- | --- |
| Length (mm) | Width (mm) | Combined | MedAI 0.8 Requirement |
| 1.57 | 1.18 | 0.8 | Pass |

### Table 14
| Radiographs and Measured Curves |  |  |
| --- | --- | --- |
|  | Length (mm) | Width (mm) |
| Radiograph |  |  |
| Measured Curve |  |  |

### Table 15
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | 28 May 2024 | 24-313 |
