# VVPR-P01-184 Rev B: MX1 Software System Foot Pedal Integration v3.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-184
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.0.0
- Source filename: VVPR-P01-184 - MX1 Software System Foot Pedal Integration v3.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-184 - MX1 Software System Foot Pedal Integration v3.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Foot pedal pairing via ODA
Acquisition mode switch via foot pedal
Image acquisition via foot pedal
Image rotation and “favoriting” via foot pedal
Foot pedal LED indications
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.0.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. B
IFU-MX1 - Instructions for Use, Rev. D
MATERIALS
E1 Emitter Rev. H
C1 Cassette Rev. I
F1 Foot Pedal Rev. B
M50133 Rev. A, Galaxy Tablet  S8+
APP MedAI Device App
Additional tools/equipment:
EQP-139 (or equivalent) Control Company Stopwatch 4YMT7
EQP-111 (or equivalent) RIGOL Programmable DC Power Supply
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 1. Foot Pedal Pairing - Requirements, Verification Steps, and Expected Results
Table 2. Foot Pedal Integration - Requirements, Verification Steps, and Expected Results
Table 3. Foot Pedal LED Indications - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
SRS-7.7 and SRS-7.9 in Table 1 were renumbered to SRS-7.10 and SRS-7.13, respectively. Original IDs were incorrect as a result of clerical error. No changes to the requirements or associated tests are required.
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220
C1 Cassette Rev. I, SN: 1079
F1 Foot Pedal Rev. B, Lot #: 10010
M50133 Galaxy Tablet S8+, Rev. A, MPN: R52X101FM1N
MX1 Software System v3.0.0
Additional tools/equipment:
EQP-139 - Control Company Stopwatch 4YMT7
EQP-111 -  RIGOL Programmable DC Power Supply
RESULTS
Table 1. Foot Pedal Pairing - Requirements, Verification Steps, and Expected Results
Table 2. Foot Pedal Integration - Requirements, Verification Steps, and Expected Results
Table 3. Foot Pedal LED Indications - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 through Appendix 19 - Verification Evidence as Specified in Results Table 1.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1. Emitter Without Foot Pedal
Appendix 2. Emitter With Foot Pedal
Appendix 3. Emitter Before Left Button Press
Appendix 4. Emitter After Left Button Press
Appendix 5. Emitter After Foot Pedal Disconnect And Button Press
Appendix 6. Emitter in Single Mode
Appendix 7. Emitter in Serial Radiography Following Left Button Press
Appendix 8. Emitter in Fluoro After Left Button Press
Appendix 9. Emitter in Photo Following Left Button Press
Appendix 10. Emitter in Single Following Left Button Press
Appendix 11. Single Image in App
Appendix 12. Photographic Image Displayed in App
Appendix 13. Radiographic Image
Appendix 14. Radiographic Image Rotated 90 Degrees Clockwise
Appendix 15. Image without Favorite Mark
Appendix 16. Image with Favorite Mark
Appendix 17. Serial Radiographic Capture, 24 frames for 5.38 second acquisition
Appendix 18. Foot Pedal Battery LED, Green
Appendix 19. Foot Pedal Battery LED, Red

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App, F1 Foot Pedal |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Foot Pedal Pairing via ODA |  |  |  |  |
| SRS-30.10 | The SS shall allow the user to set the foot pedal ID via the MedAI Device App | 1. Using an emitter that is not paired to a foot pedal, confirm that there is no foot pedal connection icon in the emitter display 2. Navigate to the Cassette Config page in the MedAI Device App 3. Enter the foot pedal ID into the input box. 4. Hit “Pair” button 5. Verify that the foot pedal connection icon appears in the emitter display 6. Press the mode switch button to confirm successful pairing | Foot pedal connection icon appears on the emitter display |  |  |
| SRS-7.7 | The SS shall allow for an foot pedal to be configured to communicate with a specified emitter |  | Acquisition mode switches upon left foot pedal button press |  |  |
| SRS-7.9 | For all imaging modes, the SS shall indicate a foot pedal's connection status to an emitter via icon in emitter touchscreen display |  |  |  |  |
|  |  | 1. Navigate to the Cassette Config page in the MedAI Device App 2. Enter the foot pedal ID “0” into the input box. 3. Hit “Pair” button 4. Attempt to change the imaging mode using the foot pedal | Foot pedal connection icon disappears from the emitter touchscreen display |  |  |
|  |  |  | Acquisition mode does NOT change upon left foot pedal button press |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Switch Acquisition Modes with Foot Pedal |  |  |  |  |
| SRS-22.3 | The SS shall switch between acquisition modes when a foot pedal left button press is detected | 1. Press the foot pedal left button to cycle through all acquisition modes 2. Verify the modes cycle in the following order: a. From single radiography to serial radiography, b. From serial radiography to fluoroscopy, c. From fluoroscopy to photography, and d. From photography to single radiography 3. Verify that the emitter display indicates the selected acquisition mode | Acquisition modes cycle in the following order: 1. Single to serial radiography 2. Serial radiography to fluoroscopy 3. Fluoroscopy to photography 4. Photography to single radiography |  |  |
|  |  |  | Single radiographic mode indication icon appears on emitter display when in single radiographic mode |  |  |
|  |  |  | Serial radiographic mode indication icon appears on emitter display when in serial radiographic mode |  |  |
|  |  |  | Fluoroscopic mode indication icon appears on emitter display when in fluoroscopy mode |  |  |
|  |  |  | Photographic mode indication icon appears on emitter display when in photographic mode |  |  |
|  | Test Case: Capture Single Radiographic Image with Foot Pedal |  |  |  |  |
| SRS-22.1 | The SS shall allow reception of triggers and button events from a paired foot pedal | 1. Ensure the system is in Single Radiography Mode 2. Press and release the right foot pedal | Single radiographic image displayed in App |  |  |
| SRS-22.2 | The SS shall support triggering of the device via a foot pedal that mimics the trigger behavior on the emitter |  |  |  |  |
| SRS-22.4 | In single radiographic mode, the SS shall allow initiation of a single x-ray exposure upon pressing and releasing the foot pedal right pedal (B) |  |  |  |  |
|  |  | 1. Ensure the system is in Single Radiography Mode 2. Press and release the right foot pedal | Only single radiographic image displayed in App |  |  |
|  | Test Case: Capture Serial Radiographic Images with Foot Pedal |  |  |  |  |
| SRS-22.5 | In serial radiographic and radioscopic modes, the SS shall allow the foot pedal right pedal (B) to initiate exposure on the downpress and shall stop the exposure upon release | 1. Ensure the system is in Serial Radiographic Mode 2. Start timer. Press and hold right foot pedal for at least 1 second before release. 3. Stop timer. Record time that foot pedal was depressed. | Series with an appropriate number of frames for the serial capture time recorded (5 frames/per second) is displayed |  |  |
|  | Test Case: Capture Radioscopic Images with Foot Pedal |  |  |  |  |
| SRS-22.5 | In serial radiographic and radioscopic modes, the SS shall allow the foot pedal right pedal (B) to initiate exposure on the downpress and shall stop the exposure upon release | 1. Ensure the system is in Radioscopic Mode 2. Start timer. Press and hold right foot pedal for at least 1 second before release. 3. Stop timer. Record time that foot pedal was depressed. | Series with an appropriate number of frames for the radioscopic capture time recorded (5 frames/per second) is displayed |  |  |
|  | Test Case: Capture Low Dose Radioscopic Images with Foot Pedal |  |  |  |  |
| SRS-22.5 | In serial radiographic and radioscopic modes, the SS shall allow the foot pedal right pedal (B) to initiate exposure on the downpress and shall stop the exposure upon release | 1. Ensure the system is in Low Dose Radiographic Mode 2. Start timer. Press and hold right foot pedal for at least 1 second before release. 3. Stop timer. Record time that foot pedal was depressed. | Series with an appropriate number of frames for the low dose radioscopic capture time recorded (2.5 frames/per second) is displayed |  |  |
|  | Test Case: Capture Photographic Images with Foot Pedal |  |  |  |  |
| SRS-22.6 | In photographic mode, the SS shall allow for the acquisition of a photo upon press and release of the foot pedal right pedal (B) | 1. Ensure the system is in Photographic Mode 2. Press and release the right foot pedal | Photographic image displayed in App |  |  |
|  | Test Case: Rotate Images |  |  |  |  |
| SRS-22.7 | The SS shall allow the foot pedal right button to rotate an acquired image by 90 degrees clockwise | 1. Capture a single radiographic image 2. Press the foot pedal right button once to rotate image | Single radiographic image is rotated by 90 degrees clockwise |  |  |
|  | Test Case: Mark/"Favorite" Image |  |  |  |  |
| SRS-22.8 | The SS shall allow the foot pedal left pedal to mark the most recently acquired image for export | 1. Ensure that ODA is open on the Acquisition Screen and that at least one capture has been acquired 2. Press the left foot pedal | "Favorite" mark (star) appears on recently acquired image in the camera roll |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
| Test Setup: A lab power supply shall be used to provide the foot pedal PCBA with voltage ranging from its maximum input voltage (4.5V) to its minimum input voltage (2.25V). |  |  |  |  |  |
|  | Test Case: Foot Pedal Battery Charge Indicator LED |  |  |  |  |
| SRS-22.9 | The SS shall indicate foot pedal battery charge level via bottom foot pedal LED | Begin with a lab PSU set to 4.5VDC and decrease voltage throughout the input voltage range. The following thresholds should result in the specified LED indication color and pattern. Thresholds: >/= 2.6 ->solid green <2.6 -> blinking red | LED color and pattern on device match the specified states |  |  |
|  | Test Case: Foot Pedal Communication Indicator LED |  |  |  |  |
| SRS-22.10 | The SS shall indicate foot pedal communication status via the top foot pedal LED | Press the foot pedal right button once to rotate the image | Top foot pedal LED blinks at the initial press and release |  |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 14 May 2024 | 24-237 |

### Table 6
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-139 | 9/12/2022 | 9/12/2024 |
| RIGOL Programmable DC Power Supply | EQP-111 | 02/20/2024 | 02/28/2025 |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App, F1 Foot Pedal |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Foot Pedal Pairing via ODA |  |  |  |  |
| SRS-30.10 | The SS shall allow the user to set the foot pedal ID via the MedAI Device App | 1. Using an emitter that is not paired to a foot pedal, confirm that there is no foot pedal connection icon in the emitter display 2. Navigate to the Cassette Config page in the MedAI Device App 3. Enter the foot pedal ID into the input box. 4. Hit “Pair” button 5. Verify that the foot pedal connection icon appears in the emitter display 6. Press the mode switch button to confirm successful pairing | Foot pedal connection icon appears on the emitter display | Expected outcome verified. See Appendix 2. Verified by RM 15MAY24 | P |
| SRS-7.10 | The SS shall allow for an foot pedal to be configured to communicate with a specified emitter |  | Acquisition mode switches upon left foot pedal button press | Expected outcome verified. See Appendix 4. Verified by RM 15MAY24 | P |
| SRS-7.13 | For all imaging modes, the SS shall indicate a foot pedal's connection status to an emitter via icon in emitter touchscreen display |  |  |  |  |
|  |  | 1. Navigate to the Cassette Config page in the MedAI Device App 2. Enter the foot pedal ID “0” into the input box. 3. Hit “Pair” button 4. Attempt to change the imaging mode using the foot pedal | Foot pedal connection icon disappears from the emitter touchscreen display | Expected outcome verified. See Appendix 5. Verified by RM 15MAY24 | P |
|  |  |  | Acquisition mode does NOT change upon left foot pedal button press | Expected outcome verified. See Appendix 5. Verified by RM 15MAY24 | P |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Switch Acquisition Modes with Foot Pedal |  |  |  |  |
| SRS-22.3 | The SS shall switch between acquisition modes when a foot pedal left button press is detected | 1. Press the foot pedal left button to cycle through all acquisition modes 2. Verify the modes cycle in the following order: a. From single radiography to serial radiography, b. From serial radiography to fluoroscopy, c. From fluoroscopy to photography, and d. From photography to single radiography 3. Verify that the emitter display indicates the selected acquisition mode | Acquisition modes cycle in the following order: 1. Single to serial radiography 2. Serial radiography to fluoroscopy 3. Fluoroscopy to photography 4. Photography to single radiography | Expected outcome verified. See Appendix 1. Expected outcome verified. See Appendix 6-10. Verified by RM 15MAY24 | P |
|  |  |  | Single radiographic mode indication icon appears on emitter display when in single radiographic mode | Expected outcome verified. See Appendix 6. Verified by RM 15MAY24 | P |
|  |  |  | Serial radiographic mode indication icon appears on emitter display when in serial radiographic mode | Expected outcome verified. See Appendix 7. Verified by RM 15MAY24 | P |
|  |  |  | Fluoroscopic mode indication icon appears on emitter display when in fluoroscopy mode | Expected outcome verified. See Appendix 8. Verified by RM 15MAY24 | P |
|  |  |  | Photographic mode indication icon appears on emitter display when in photographic mode | Expected outcome verified. See Appendix 9. Verified by RM 15MAY24 | P |
|  | Test Case: Capture Single Radiographic Image with Foot Pedal |  |  |  |  |
| SRS-22.1 | The SS shall allow reception of triggers and button events from a paired foot pedal | 1. Ensure the system is in Single Radiography Mode 2. Press and release the right foot pedal | Single radiographic image displayed in App | Expected outcome verified. See Appendix 11. Verified by RM 15MAY24 | P |
| SRS-22.2 | The SS shall support triggering of the device via a foot pedal that mimics the trigger behavior on the emitter |  |  |  |  |
| SRS-22.4 | In single radiographic mode, the SS shall allow initiation of a single x-ray exposure upon pressing and releasing the foot pedal right pedal (B) |  |  |  |  |
|  |  | 1. Ensure the system is in Single Radiography Mode 2. Press and release the right foot pedal | Only single radiographic image displayed in App | Expected outcome verified. See Appendix 11. Verified by RM 15MAY24 | P |
|  | Test Case: Capture Serial Radiographic Images with Foot Pedal |  |  |  |  |
| SRS-22.5 | In serial radiographic and radioscopic modes, the SS shall allow the foot pedal right pedal (B) to initiate exposure on the downpress and shall stop the exposure upon release | 1. Ensure the system is in Serial Radiographic Mode 2. Start timer. Press and hold right foot pedal for at least 1 second before release. 3. Stop timer. Record time that foot pedal was depressed. | Series with an appropriate number of frames for the serial capture time recorded (5 frames/per second) is displayed | Expected outcome verified. 24 frames from 5.38 seconds of pedal press and hold. See Appendix 17. Verified by AM 16MAY24 | P |
|  | Test Case: Capture Radioscopic Images with Foot Pedal |  |  |  |  |
| SRS-22.5 | In serial radiographic and radioscopic modes, the SS shall allow the foot pedal right pedal (B) to initiate exposure on the downpress and shall stop the exposure upon release | 1. Ensure the system is in Radioscopic Mode 2. Start timer. Press and hold right foot pedal for at least 1 second before release. 3. Stop timer. Record time that foot pedal was depressed. | Series with an appropriate number of frames for the radioscopic capture time recorded (5 frames/per second) is displayed | Expected outcome verified. 25 frames from 5.42 seconds of pedal press and hold. Verified by AM 16MAY24 | P |
|  | Test Case: Capture Low Dose Radioscopic Images with Foot Pedal |  |  |  |  |
| SRS-22.5 | In serial radiographic and radioscopic modes, the SS shall allow the foot pedal right pedal (B) to initiate exposure on the downpress and shall stop the exposure upon release | 1. Ensure the system is in Low Dose Radiographic Mode 2. Start timer. Press and hold right foot pedal for at least 1 second before release. 3. Stop timer. Record time that foot pedal was depressed. | Series with an appropriate number of frames for the low dose radioscopic capture time recorded (2.5 frames/per second) is displayed | Expected outcome verified. 12 frames from 5.26 seconds of pedal press and hold. Verified by AM 16MAY24 | P |
|  | Test Case: Capture Photographic Images with Foot Pedal |  |  |  |  |
| SRS-22.6 | In photographic mode, the SS shall allow for the acquisition of a photo upon press and release of the foot pedal right pedal (B) | 1. Ensure the system is in Photographic Mode 2. Press and release the right foot pedal | Photographic image displayed in App | Expected outcome verified. See Appendix 12. Verified by RM 15MAY24 | P |
|  | Test Case: Rotate Images |  |  |  |  |
| SRS-22.7 | The SS shall allow the foot pedal right button to rotate an acquired image by 90 degrees clockwise | 1. Capture a single radiographic image 2. Press the foot pedal right button once to rotate image | Single radiographic image is rotated by 90 degrees clockwise | Expected outcome verified. See Appendix 13-14. Verified by RM 15MAY24 | P |
|  | Test Case: Mark/"Favorite" Image |  |  |  |  |
| SRS-22.8 | The SS shall allow the foot pedal left pedal to mark the most recently acquired image for export | 1. Ensure that ODA is open on the Acquisition Screen and that at least one capture has been acquired 2. Press the left foot pedal | "Favorite" mark (star) appears on recently acquired image in the camera roll | Expected outcome verified. See Appendix 15-16. Verified by RM 15MAY24 | P |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, Tablet, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
| Test Setup: A lab power supply shall be used to provide the foot pedal PCBA with voltage ranging from its maximum input voltage (4.5V) to its minimum input voltage (2.25V). |  |  |  |  |  |
|  | Test Case: Foot Pedal Battery Charge Indicator LED |  |  |  |  |
| SRS-22.9 | The SS shall indicate foot pedal battery charge level via bottom foot pedal LED | Begin with a lab PSU set to 4.5VDC and decrease voltage throughout the input voltage range. The following thresholds should result in the specified LED indication color and pattern. Thresholds: >/= 2.6 ->solid green <2.6 -> blinking red | LED color and pattern on device match the specified states | Expected outcome verified. Foot pedal battery charge indicator LED turns green when PSU-supplied voltage is set to 4.5 V and orange when PSU set to 2.25 V. See Appendix 18 and 19. Verified by SP and AM 16MAY24 | P |
|  | Test Case: Foot Pedal Communication Indicator LED |  |  |  |  |
| SRS-22.10 | The SS shall indicate foot pedal communication status via the top foot pedal LED | Press the foot pedal right button once to rotate the image | Top foot pedal LED blinks at the initial press and release | Expected outcome verified. Top LED blinks at initial press and upon release of right button. Verified by AM 16MAY24 | P |

### Table 10
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-440 |  |
