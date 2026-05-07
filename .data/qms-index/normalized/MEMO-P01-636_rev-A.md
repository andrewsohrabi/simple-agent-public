# MEMO-P01-636 Rev A: MX1 Software System Unresolved Anomalies v3.0.0

## Metadata
- Document ID: MEMO-P01-636
- Revision: A
- Prefix: MEMO
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: v3.0
- Source filename: MEMO-P01-636 - MX1 Software System Unresolved Anomalies v3.0.0_A_Obsolete.docx
- Source path: Example QMS - MedAI/MEMO-P01-636 - MX1 Software System Unresolved Anomalies v3.0.0_A_Obsolete.docx
- Extraction warnings: none

## Extracted Content
1. PURPOSE
This document provides a detailed overview of the anomaly resolutions resulting from MX1 Software System (SS) v3.0.0 testing.
Each anomaly reference includes the corresponding requirement, verification steps, description of the anomaly, and how the anomaly may be addressed in a future software release, if applicable.
2. SCOPE
This document pertains to the anomalies discovered through MX1 SS v3.0.0 verification activities.
3. REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev B
VVPR-P01-175 - MX1 Software System, System Configuration, v3.0.0 Protocol and Report
VVPR-P01-176 - MX1 Software System, Power On/Off and Power States, v3.0.0 Protocol and Report
VVPR-P01-178 - MX1 Software System, Safety Interlocks, v3.0.0 Protocol and Report
VVPR-P01-186 - MX1 Software System, MedAI Device App, v3.0.0 Protocol and Report
Unresolved Anomalies
Table 1.  VVPR-P01-175 - MX1 Software System, System Configuration, v3.0.0 Protocol and Report - Unresolved Anomalies
Table 2.  VVPR-P01-176 - MX1 Software System, Power On/Off and Power States, v3.0.0 Protocol and Report - Unresolved Anomalies
Table 3. VVPR-P01-178 - MX1 Software System, Safety Interlocks, v3.0.0 Protocol and Report - Unresolved Anomalies
Table 4. VVPR-P01-186 - MX1 Software System, MedAI Device App, v3.0.0 Protocol and Report - Unresolved Anomalies
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Anomaly Description | Root Cause Analysis | Impact On Device Performance, Safety and Effectiveness | Planned Resolution and/or Mitigation/Workaround |
| --- | --- | --- | --- |
| Related to the following software requirements: SRS-1.11 In release mode, the SS shall restrict access to production-level accounts SRS-1.13 In release mode, the SS shall force logouts of any open maintenance mode terminals after 120 seconds of inactivity SRS-1.14 In release mode, the SS shall enforce the use of a restricted keyboard key set SRS-1.17 In release mode, the SS shall present a blank screen if an external display is connected to the emitter via service port SRS-1.18 In release mode, the SS shall perform an integrity check upon boot and every hour Anomaly Description: Once placed in release mode, the MX1 SS does not meet the above listed requirements as of v3.0.0. | A possible root cause for this anomaly may be that the scripts responsible for the implementation or enabling of the behaviors defined in the listed requirements contained incomplete commands and/or referenced incorrect paths. | The MX1 SS release mode is primarily a security measure to prevent bad actors from gaining access to sensitive and protected information stored on the emitter and cassette Jetsons. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in MX1 SS v3.1.0 |

### Table 2
| Anomaly Description | Root Cause Analysis | Impact On Device Performance, Safety and Effectiveness | Planned Resolution and/or Mitigation/Workaround |
| --- | --- | --- | --- |
| Related to the following software requirements: SRS-8.13 If in idle state, the SS shall exit idle state within 20 seconds of meeting an idle exit condition Anomaly Description: System consistently exits idle state within 30 seconds. | The requirement and related test method were defined before implementation. The primary component that extends the exit time is the detector, which takes nearly 30 seconds to wake properly after restoration of power. | Idle state is a feature to enhance system performance and behavior. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Requirement modification - SRS-8.13 will be updated to reflect the 30 second time to exit idle state. To be addressed in the next revision of MEMO-P01-630 - MX1 Software Requirement Specifications. |

### Table 3
| Anomaly Description | Root Cause Analysis | Impact On Device Performance, Safety and Effectiveness | Planned Resolution and/or Mitigation/Workaround |
| --- | --- | --- | --- |
| Related to the following software requirements: SRS-12.2 The SS shall terminate x-ray emission if any safety interlock is broken or a fault is detected during acquisition Anomaly Description: Upon plugging wired charger into emitter charging port, serial radiographic acquisition continues. | Root cause is failure of the emitter firmware (EM) to detect charger plug in. | The emitter should not be able to charge during x-ray emission. As a result, a mitigation was designed to detect wired charging and disallow x-ray emission. During analysis of the anomaly, it was determined that the emitter was not charging even though the wired charger was plugged into the emitter port. However, it is determined that, as a continued safety feature, a fix should be implemented. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in MX1 SS v3.1.0 |

### Table 4
| Anomaly Description | Root Cause Analysis | Impact On Device Performance, Safety and Effectiveness | Planned Resolution and/or Mitigation/Workaround |
| --- | --- | --- | --- |
| Related to the following software requirements: SRS-31.2 The SS shall allow operators to add, edit, and delete operator and/or physician name entries via the MedAI Device App SRS-38.4 The ODA shall provide a dropdown menu to select a physician name in the Exam Screen Anomaly Description: Failed to save modified or deleted saved entries in the User’s List. Additionally, failed to verify SRS-38.4 due to inability to save entries in the User’s List. Tapping the “Save” and “Delete” buttons in ODA did not result in the expected behavior. Existing user entry remained unchanged. | A root cause is undetermined as of the MX1 SS v3.0.0 release. | The primary intent of allowing the addition of physician and operator entries in the User’s List is to facilitate user needs with respect to management of DICOM files. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in a MX1 Software System release prior to commercial release of the MX1 System. |
| Related to the following software requirements: SRS-39.4 The SS shall allow the user to play back, pause, and step through serial radiographic images in the MedAI Device App ("DDR playback") Anomaly Description: For both serial radiographic and radioscopic captures, frame slider and presented images cycle rapidly if user attempts to tap on the slider during playback. This behavior is not present if the user pauses playback prior to using any of the other frame-slider related features. | A root cause is undetermined as of the MX1 SS v3.0.0 release. The “jolting” behavior is only present if the user attempts to tap on the slider itself or press the back/forward buttons while serial or radioscopic capture is playing. Once paused, the frame slider may be used as intended. | The frame slider in the MedAI Device App allows the user to pause or play back a serial radiographic or radioscopic capture. Users may also use the feature to move through captures frame by frame. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in a MX1 Software System release prior to commercial release of the MX1 System. |
| Related to the following software requirements: SRS-39.15 The SS shall allow the operator to scroll through and select previously acquired images in the camera roll via the MedAI Device App Anomaly Description: Images are not displayed in camera roll consistently. On occasion, images that are successfully acquired and displayed in the larger image panes do not appear in the camera roll. During this verification activity, the three photographic images acquired did not appear in the roll. | A root cause is undetermined as of the MX1 SS v3.0.0 release. | The camera roll feature is implemented to allow users to review images acquired at any time during an exam. It is primarily a feature intended to enhance the user experience. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in a MX1 Software System release prior to commercial release of the MX1 System. |
| Related to the following software requirements: SRS-34.2 The SS should update the wireless network connection status in the MedAI Device App every 5 seconds SRS-34.3    The SS should display the PACS connection status in the MedAI Device App Anomaly Description: Network connection status does not automatically update within 5 seconds. MedAI Cloud and Network Settings also does not display updated PACS connection status. A user-initiated activity (e.g. switching between screens) must occur before network connection status updates. | A root cause is undetermined as of the MX1 SS v3.0.0 release. | Display of network and PACS connection status is intended to facilitate users in determining their ability to retrieve MWL orders or send DICOM files to PACS. The delayed or incorrect status display has no impact on the system’s ability to import or export DICOM files. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in a MX1 Software System release prior to commercial release of the MX1 System. |
| Related to the following software requirements: SRS-37.3 The SS shall provide UI elements to clear all stored photographs, single radiographs, serial radiographs, or radioscopic captures via the MedAI Device App Anomaly Description: Low and regular dose radioscopic images remain stored on disk after tapping “Clear Images.” | A root cause is undetermined as of the MX1 SS v3.0.0 release. | The “Clear Images” feature allows the user to ensure continued use of the device by allowing regular clearing of images on disk. This feature has no impact on the primary intended use, safety, or essential performance of the MX1 system. | Software modification - To be fixed and verified in a MX1 Software System release prior to commercial release of the MX1 System. |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-440 |  |
