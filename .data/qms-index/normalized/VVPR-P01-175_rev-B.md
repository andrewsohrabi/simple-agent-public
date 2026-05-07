# VVPR-P01-175 Rev B: MX1 Software System System Configuration v3.0.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-175
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.0.0
- Source filename: VVPR-P01-175 - MX1 Software System System Configuration v3.0.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-175 - MX1 Software System System Configuration v3.0.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Release and debug modes
Software Services
System Logging
Emitter Jetson HDMI register settings
Cassette WiFi Ap
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.0.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. B
IFU-MX1 - Instructions for Use, Rev. D
MATERIALS
E1 Emitter Rev. H
C1 Cassette Rev. I
Additional tools:
EQP-139 (or equivalent) Control Company Stopwatch 4YMT7
External monitor
Mouse
Keyboard
USB Dongle
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 1. Software Services - Requirements, Verification Steps, and Expected Results
Table 2. Emitter Jetson Register Setting - Requirements, Verification Steps, and Expected Results
Table 3. System Logging - Requirements, Verification Steps, and Expected Results
Table 4. Cassette WiFi AP -  Requirements, Verification Steps, and Expected Results
Table 5. Release Mode - Requirements, Verification Steps, and Expected Results
Table 6. Debug Mode - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
Table 3, SRS-3.2 - Original test steps state to use “(jpegSaveQ)” as the term to grep, or search for, while navigating through the cassette-orchestrator logs. While a valid means of searching for most elements highlighted in SRS-3.2, jpegSaveQ will not return the set exposure time (set ms). The modified test steps state to grep with “METADATA”, which does return all elements of SRS-3.2 from the cassette-orchestrator logs.
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1204
C1 Cassette Rev. I, SN: 1205, 1079
MX1 Software System v3.0.0
EQP-139 Control Company Stopwatch 4YMT7
RESULTS
Table 1. Software Services - Requirements, Verification Steps, and Expected Results
Table 2. Emitter Jetson Register Setting - Requirements, Verification Steps, and Expected Results
Table 3. System Logging - Requirements, Verification Steps, and Expected Results
Table 4. Cassette WiFi AP -  Requirements, Verification Steps, and Expected Results
Table 5. Release Mode - Requirements, Verification Steps, and Expected Results
Table 6. Debug Mode - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: Pass with deviation
Anomalies - Refer to MEMO-P01-636 - MX1 Software System, v3.0.0, Unresolved Anomalies for resolution of the following anomalies found during the course of testing:
Table 5 - All failures in Table 5 resulted from the MX1 System failing to enter release mode as defined in SRS-1.11, SRS-1.13, SRS-1.14, SRS-1.17, and SRS-1.18.
LIST OF APPENDICES
Appendix 1 through Appendix 23 - Verification Evidence as Specified in Results Table 1-6.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1: Emitter Services Status
Appendix 2: Cassette Services Status
Appendix 3: Busy Box Drive Strength Initial Value
Appendix 4: Busy Box Drive Strength Successfully Changed
Appendix 5: Busy Box Drive Strength Verified After Reboot
Appendix 6: Busy Box Drive Strength Verified After Idle
Appendix 7: Emitter is logging
Appendix 8: Wifi List
Appendix 9:  Cassette Screen Network Name
Appendix 10: Wrong Network Password
Appendix 11: Tablet Connects with Correct Password
Appendix 12: Fail To SSH To Emitter In Release Mode
Appendix 13:  Fail To SSH To CassetteIn Release Mode
Appendix 14: Emitter Blank Screen In Release Mode
Appendix 15: Cassette Failing To have Blank Screen In Release Mode
Appendix 16: Successful Emitter SSH
Appendix 17: Successful Cassette SSH
Appendix 18: Journalctl X-ray Data
2024-05-25 cassette-1079   INFO 20:16:28.809759 [ec] (esr-17) <XRAY> <IMAGEPROC> <METADATA> Image: name: ri_single_50_40_1000_1710188186638389; batch index: 0; captureId: 102; mode: 0; setKv: 50; setExposureTime: 40; setFilamentCurrent: 2.000000; setBeamCurrent: 1.000000; pos_x: 0.000000; pos_y: 0.000000; pos_z: 0.000000; vec_x: 0.000000; vec_y: 0.000000; vec_z: 0.000000; ssd: 479; penetrationDistance: -479; sid: 0; collimatedArea: 1.000000; readBackKv: 49.970001; readBackFilamentCurrent: 5.000000; readBackBeamCurrent: 0.912000; monblockInductorTemp: 0.000000; monblockSideTemp: 0.000000; dap: 0.000008; cumulativeDap: 0.000008; dose: 8.227685; cumulativeDose: 8.227685; puckIndex: 0;
2024-05-25 cassette-1079   INFO 20:16:48.468293 [ec] (esr-17) <XRAY> <IMAGEPROC> <METADATA> Image: name: ri_single_60_40_1000_1710188206390772; batch index: 0; captureId: 103; mode: 0; setKv: 60; setExposureTime: 40; setFilamentCurrent: 2.000000; setBeamCurrent: 1.000000; pos_x: 0.000000; pos_y: 0.000000; pos_z: 0.000000; vec_x: 0.000000; vec_y: 0.000000; vec_z: 0.000000; ssd: 476; penetrationDistance: -476; sid: 0; collimatedArea: 1.000000; readBackKv: 59.950001; readBackFilamentCurrent: 5.000000; readBackBeamCurrent: 0.919000; monblockInductorTemp: 0.000000; monblockSideTemp: 0.000000; dap: 0.000013; cumulativeDap: 0.000013; dose: 12.928478; cumulativeDose: 12.928478; puckIndex: 0;
2024-05-25 cassette-1079   INFO 20:16:48.468293 [ec] (esr-17) <XRAY> <IMAGEPROC> <METADATA> Image: name: ri_single_60_40_1000_1710188206390772; batch index: 0; captureId: 103; mode: 0; setKv: 60; setExposureTime: 40; setFilamentCurrent: 2.000000; setBeamCurrent: 1.000000; pos_x: 0.000000; pos_y: 0.000000; pos_z: 0.000000; vec_x: 0.000000; vec_y: 0.000000; vec_z: 0.000000; ssd: 476; penetrationDistance: -476; sid: 0; collimatedArea: 1.000000; readBackKv: 59.950001; readBackFilamentCurrent: 5.000000; readBackBeamCurrent: 0.919000; monblockInductorTemp: 0.000000; monblockSideTemp: 0.000000; dap: 0.000013; cumulativeDap: 0.000013; dose: 12.928478; cumulativeDose: 12.928478; puckIndex: 0;
Appendix 19: Device is Green
Appendix 20: Device Specific Config
Appendix 21: Device is firing
Appendix 22: After Deletion Flashing Cyan

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Emitter Software Services |  |  |  |  |
| SRS-2.1 | On the emitter Jetson, the SS shall run the following software components as services from /opt/medai/bin/: 1. emitter-frontend 2. emitter-orchestrator 3. emitter-intermachine-proxy 4. xr-controller 5. idle-manager 6. medai-wifi-stability | 1. SSH into the emitter with ssh imager@<emitter-hostname> 2. Open a terminal and get a list of the running services with python3.8 -m mx1.services status 3. Save a truncated version of the output that shows the status of the services of interest | The following emitter services show a status of "active(running)": 1. emitter-frontend.service 2. emitter-orchestrator.service 3. emitter-inter-machine-proxy.service 4. xr-controller.service 5. idle-manager.service 6. wifi-stability.service |  |  |
|  | Test Case: Cassette Software Services |  |  |  |  |
| SRS-2.2 | On the cassette Jetson, the SS shall run the following software components as services from /opt/medai/bin/: 1. capture-presenter 2. cassette-inter-machine-proxy 3. cassette-orchestrator 4. iray-signaler 5. oled_overseer 6. image-reaper 7. connectivity-controller | 1. SSH into the cassette with ssh imager@<cassette-hostname> 2. Open a terminal and get a list of the running services with python3.8 -m mx1.services status 3. Save a truncated version of the output that shows the status of the services of interest | The following services show a status of "active(running)": 1. capture-presenter.service 2. cassette-inter-machine-proxy.service 3. cassette-orchestrator.service 4. iray-signaler.service 5. oled_overseer.service 6. image-reaper.service 7. connectivity-controller.service |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Check emitter jetson register after start up |  |  |  |  |
| SRS-4.9 | The SS shall set the emitter Jetson register 0x15b40138 to 0x0A0A0A07 during startup [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1.ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Returned value is 0x0A0A0A07 |  |  |
|  |  | 1. Ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to reset the register value: /bin/busybox devmem 0x15b40138 32 0x10101010 4. Reboot the emitter and type the following command: /bin/busybox devmem 0x15b40138 | Value is successfully set to 0x10101010 |  |  |
|  |  |  | Value is reset to 0x0A0A0A07 upon reboot |  |  |
|  | Test Case: Reset emitter jetson register after full idle |  |  |  |  |
| SRS-8.24 | Upon exiting any idle state, the SS shall reset the emitter Jetson register 0x15b40138 to 0x0A0A0A07 when commanding the emitter touchscreen display to wake from sleep state [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1. ssh into the emitter ssh imager@<emitter-hostname> 2. Allow the unit to enter idle state 3. Pull the trigger to exit idle state 4. Switch to super user via su and enter the correct credentials 5. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Value is set to 0x0A0A0A07 upon exiting idle |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Software System Logging |  |  |  |  |
| SRS-3.1 | The SS shall log system events and operator actions | 1. SSH into the emitter with ssh imager@<emitter-hostname> 2. Enter cat /var/log/syslog | grep medai -a 3. Save part of the output as evidence | The device has logs from medai software components in /var/log/syslog |  |  |
|  | Test Case: X-ray Metadata in Software Logs |  |  |  |  |
| SRS-3.2 | The SS shall log the following: 1. date and timestamp of acquisition, 2. full file path, 3. set kV, 4. set mA, 5. set ms, 6. readback kV, 7. readback mA, and 8. Monoblock tube temperature mA and ms shall be determined from set mAs value | 1. SSH into the cassette with ssh imager@<cassette-hostname> 2. Enter journalctl --user-unit cassette-orchestrator.service -f | grep "(jpegSaveQ)" 3. Place the system in single radiographic mode. Pull the emitter trigger to capture a single radiographic acquisition 4. Save the line that starts with "(jpegSaveQ) Sending" 5. Save the line that starts with "(jpegSaveQ) <IMGDBG> Saving processed:" | Logs contains date and timestamp of image acquisition |  |  |
|  |  |  | Logs full file path of acquired image |  |  |
|  |  |  | Logs contain set kV |  |  |
|  |  |  | Logs contain set mA |  |  |
|  |  |  | Logs contain set ms |  |  |
|  |  |  | Logs contain readback kV |  |  |
|  |  |  | Logs contain readback mA |  |  |
|  |  |  | Logs contain reported Monoblock temperature |  |  |

### Table 5
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). Ensure all safety interlocks are met. |  |  |  |  |
|  | Test Case: Cassette and Emitter Connection Status |  |  |  |  |
| SRS-7.9 | The SS shall allow for an emitter to be configured to communicate with a specified cassette | 1. SSH into the emitter 2. Record evidence of the contents of the device-specific config with cat /opt/medai/data/config/device-specific/device-specific-config.json 3. Capture a single radiographic acquisition 4. Edit the value of "cassette_network_name" in cat /opt/medai/data/config/device-specific/device-specific-config.json to "invalid" 5. Stop services with python3.8 -m mx1.services stop 6. Remove the current wifi connection with nmcli con delete cassette-<cassette-number> 7. Reboot the emitter 8. Change to photo mode | Before stopping the services, the MI LEDs are green |  |  |
|  |  |  | In the device-specific-config the value of "cassette_network_name" is "cassette-<cassette-number>" and the MI LEDs are either red or green |  |  |
|  |  |  | An x-ray was able to be captured |  |  |
| SRS-7.12 | For all imaging modes, the SS shall indicate an emitter’s connection status to a cassette via icon in emitter touchscreen display |  | After rebooting the emitter, the MI LEDs blink cyan |  |  |
|  |  |  | Before the WIFI connection is deleted, the cassette icon is present without a strike-through in photo and radiographic mode |  |  |
|  |  |  | After rebooting the emitter the cassette icon is present in both radiographic and photo mode with a strike-through |  |  |
|  | Test Case: Cassette WiFi AP |  |  |  |  |
| SRS-7.3 | The SS shall initiate a private WiFi Access Point (WiFi AP) upon startup of the cassette | 1. SSH into the emitter 2. List the WIFI networks available to the emitter with nmcli device wifi list | In the list of WiFi networks available to the emitter, the cassette's network is listed in green |  |  |
| SRS-7.1 | The SS shall use the WPA2 security standard for internal and external wireless communication |  | In the list of WIFIs the cassette's network is listed as WPA2 |  |  |
| SRS-7.5 | The SS shall display the cassette-hosted WiFi AP SSID and password on the cassette display | 1. Take an image of the cassette display. Verify that the cassette's SSID and password are shown. 2. Using a tablet, attempt to connect to the cassette WiFi with an incorrect password 3. Record evidence of the failure to connect the tablet to the cassette WiFi 4. Using the same tablet, connect to the cassette WiFi with the displayed password 5. Record evidence of a successful connection of the tablet to the cassette WiFi | The cassette display has the SSID and password present |  |  |
| SRS-7.4 | The SS shall require credentials to access the Cassette WiFi AP |  | Tablet fails to connect to cassette WiFi with incorrect password |  |  |
|  |  |  | Tablet is able to connect to cassette WiFi with password shown on cassette display |  |  |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Release Mode Precondition: Ensure the emitter and cassette do NOT have a wired network connection prior to placing the system into release mode. Place the emitter in release mode before the paired cassette. Emitter: Place the emitter in release mode using the following steps: 1. Connect a mouse and keyboard to the emitter service port via dongle. Note that an external monitor can be used but is not necessary; the terminal will be visible in the emitter display 2. Open a terminal 3. As root, run the following command: rm /imagerdebug 4. Restart the emitter Cassette: Place the cassette in release mode using the following steps: 1. Connect a mouse, keyboard, and external monitor to one of the cassette service ports via dongle 2. Open a terminal 3. As root, run the following command: rm /imagerdebug 4. Restart the cassette |  |  |  |  |
| SRS-1.10 | The SS shall contain a release mode for devices ready for distribution. Debug mode shall be disabled in release mode. | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the emitter with ssh imager@<emitter-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure |  |  |
| SRS-1.15 | In release mode, the SS shall disable SSH |  |  |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure |  |  |
| SRS-1.16 | In release mode, the SS shall present a blank screen if an external display is connected to the cassette via service port | 1. Connect a mouse, keyboard, and external monitor to the emitter service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to emitter |  |  |
| SRS-1.17 | In release mode, the SS shall present a blank screen if an external display is connected to the emitter via service port | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to cassette |  |  |
| SRS-1.13 | In release mode, the SS shall force logouts of any open maintenance mode terminals after 120 seconds of inactivity | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Use the esoteric key combination to open a terminal. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on emitter |  |  |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Use the esoteric key combination to open a terminal. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on cassette |  |  |
| SRS-1.14 | In release mode, the SS shall enforce the use of a restricted keyboard key set | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on emitter display |  |  |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on external monitor for cassette |  |  |
| SRS-1.11 | In release mode, the SS shall restrict access to production-level accounts | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Use the esoteric key combination to open a terminal 3. Using the command su -, attempt to log in as root user by using incorrect credentials for the unit under test 4. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on emitter |  |  |
|  |  | 1. Connect a mouse, keyboard, and monitor to a cassette service port via dongle 2. Use the esoteric key combination to open a terminal 3. Using the command su -, attempt to log in as root user by using incorrect credentials for the unit under test 4. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on cassette |  |  |
|  | Test Case: Integrity Check |  |  |  |  |
| SRS-1.18 | In release mode, the SS shall perform an integrity check upon boot and every hour | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Use the esoteric key combination to open a terminal 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.8 -m mx1.services stop To start: python3.8 -m mx1.services start 5. Reboot the emitter. Alternatively, wait one hour for the integrity check alarm to be triggered 6. Open another terminal session 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the emitter |  |  |
|  |  |  | Record evidence of the integrity check alarm on the emitter |  |  |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Use the esoteric key combination to open a terminal 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.8 -m mx1.services stop To start: python3.8 -m mx1.services start 5. Reboot the emitter. Alternatively, wait one hour for the integrity check alarm to be triggered 6. Open another terminal session 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the cassette |  |  |
|  |  |  | Record evidence of the integrity check alarm on the cassette |  |  |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Debug Mode Precondition: Emitter: Place the emitter in debug mode using the following steps: 1. Connect a mouse and keyboard to the emitter service port via dongle. Note that an external monitor can be used but is not necessary; the terminal will be visible in the emitter display 2. Use the esoteric key combo to open a terminal 3. As root user, run the following command: touch /imagerdebug 4. Restart the emitter Cassette: Place the cassette in debug mode using the following steps: 1. Connect a mouse, keyboard, and external monitor to one of the cassette service ports via dongle 2. Use the esoteric key combo to open a terminal 3. As root user, run the following command: touch /imagerdebug 4. Restart the cassette |  |  |  |  |
| SRS-1.7 | The SS shall contain a debug mode for development and production activities | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), SSH into the emitter with ssh imager@<emitter-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to emitter in debug mode |  |  |
| SRS-1.9 | In debug mode, the SS shall enable SSH |  |  |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to cassette in debug mode |  |  |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 21 May 2024 | 24-261 |

### Table 9
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-139 | 9/12/2022 | 9/12/2024 |

### Table 10
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Emitter Software Services |  |  |  |  |
| SRS-2.1 | On the emitter Jetson, the SS shall run the following software components as services from /opt/medai/bin/: 1. emitter-frontend 2. emitter-orchestrator 3. emitter-intermachine-proxy 4. xr-controller 5. idle-manager 6. medai-wifi-stability | 1. SSH into the emitter with ssh imager@<emitter-hostname> 2. Open a terminal and get a list of the running services with python3.8 -m mx1.services status 3. Save a truncated version of the output that shows the status of the services of interest | The following emitter services show a status of "active(running)": 1. emitter-frontend.service 2. emitter-orchestrator.service 3. emitter-inter-machine-proxy.service 4. xr-controller.service 5. idle-manager.service 6. wifi-stability.service | Expected outcome verified. See Appendix 1. Verified by DH 21MAY24 | P |
|  | Test Case: Cassette Software Services |  |  |  |  |
| SRS-2.2 | On the cassette Jetson, the SS shall run the following software components as services from /opt/medai/bin/: 1. capture-presenter 2. cassette-inter-machine-proxy 3. cassette-orchestrator 4. iray-signaler 5. oled_overseer 6. image-reaper 7. connectivity-controller | 1. SSH into the cassette with ssh imager@<cassette-hostname> 2. Open a terminal and get a list of the running services with python3.8 -m mx1.services status 3. Save a truncated version of the output that shows the status of the services of interest | The following services show a status of "active(running)": 1. capture-presenter.service 2. cassette-inter-machine-proxy.service 3. cassette-orchestrator.service 4. iray-signaler.service 5. oled_overseer.service 6. image-reaper.service 7. connectivity-controller.service | Expected outcome verified. See Appendix 2. Verified by DH 21MAY24 | P |

### Table 11
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Check emitter jetson register after start up |  |  |  |  |
| SRS-4.9 | The SS shall set the emitter Jetson register 0x15b40138 to 0x0A0A0A07 during startup [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1.ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Returned value is 0x0A0A0A07 | Expected outcome verified. See Appendix 3. Verified by DH 21MAY24 | P |
|  |  | 1. Ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to reset the register value: /bin/busybox devmem 0x15b40138 32 0x10101010 4. Reboot the emitter and type the following command: /bin/busybox devmem 0x15b40138 | Value is successfully set to 0x10101010 | Expected outcome verified. See Appendix 4. Verified by DH 21MAY24 | P |
|  |  |  | Value is reset to 0x0A0A0A07 upon reboot | Expected outcome verified. See Appendix 5. Verified by DH 21MAY24 | P |
|  | Test Case: Reset emitter jetson register after full idle |  |  |  |  |
| SRS-8.24 | Upon exiting any idle state, the SS shall reset the emitter Jetson register 0x15b40138 to 0x0A0A0A07 when commanding the emitter touchscreen display to wake from sleep state [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1. ssh into the emitter ssh imager@<emitter-hostname> 2. Allow the unit to enter idle state 3. Pull the trigger to exit idle state 4. Switch to super user via su and enter the correct credentials 5. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Value is set to 0x0A0A0A07 upon exiting idle | Expected outcome verified. See Appendix 6. Verified by DH 21MAY24 | P |

### Table 12
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Software System Logging |  |  |  |  |
| SRS-3.1 | The SS shall log system events and operator actions | 1. SSH into the emitter with ssh imager@<emitter-hostname> 2. Enter cat /var/log/syslog | grep medai -a 3. Save part of the output as evidence | The device has logs from medai software components in /var/log/syslog | Expected outcome verified. See Appendix 7. Verified by DH 21MAY24 | P |
|  | Test Case: X-ray Metadata in Software Logs |  |  |  |  |
| SRS-3.2 | The SS shall log the following: 1. date and timestamp of acquisition, 2. full file path, 3. set kV, 4. set mA, 5. set ms, 6. readback kV, 7. readback mA, and 8. Monoblock tube temperature mA and ms shall be determined from set mAs value | 1. SSH into the cassette with ssh imager@<cassette-hostname> 2. Enter journalctl --user-unit cassette-orchestrator.service -f | grep "METADATA" 3. Place the system in single radiographic mode. Pull the emitter trigger to capture a single radiographic acquisition 4. Save the line that starts with  “<XRAY> <IMAGEPROC> <METADATA>” * Deviation: Steps changed to grep for “METADATA” instead of (jpegSaveQ). See Protocol Deviations 1.a. | Logs contains date and timestamp of image acquisition | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |
|  |  |  | Logs full file path of acquired image | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |
|  |  |  | Logs contain set kV | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |
|  |  |  | Logs contain set mA | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |
|  |  |  | Logs contain set ms | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |
|  |  |  | Logs contain readback kV | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |
|  |  |  | Logs contain readback mA | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 |  |
|  |  |  | Logs contain reported Monoblock temperature | Expected outcome verified. See Appendix 18. Verified by AM 25MAY24 | P |

### Table 13
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). Ensure all safety interlocks are met. |  |  |  |  |
|  | Test Case: Cassette and Emitter Connection Status |  |  |  |  |
| SRS-7.9 | The SS shall allow for an emitter to be configured to communicate with a specified cassette | 1. SSH into the emitter 2. Record evidence of the contents of the device-specific config with cat /opt/medai/data/config/device-specific/device-specific-config.json 3. Capture a single radiographic acquisition 4. Edit the value of "cassette_network_name" in cat /opt/medai/data/config/device-specific/device-specific-config.json to "invalid" 5. Stop services with python3.8 -m mx1.services stop 6. Remove the current wifi connection with nmcli con delete cassette-<cassette-number> 7. Reboot the emitter 8. Change to photo mode | Before stopping the services, the MI LEDs are green | Expected outcome verified. See Appendix 19. Verified by DH 21MAY24 | P |
|  |  |  | In the device-specific-config the value of "cassette_network_name" is "cassette-<cassette-number>" and the MI LEDs are either red or green | Expected outcome verified. See Appendix 20. Verified by DH 21MAY24 | P |
|  |  |  | An x-ray was able to be captured | Expected outcome verified. See Appendix 21. Verified by DH 21MAY24 | P |
| SRS-7.12 | For all imaging modes, the SS shall indicate an emitter’s connection status to a cassette via icon in emitter touchscreen display |  | After rebooting the emitter, the MI LEDs blink cyan | Expected outcome verified. See Appendix 22. Verified by DH 21MAY24 | P |
|  |  |  | Before the WIFI connection is deleted, the cassette icon is present without a strike-through in photo and radiographic mode | Expected outcome verified. Verified by AM 28MAY24 | p |
|  |  |  | After rebooting the emitter the cassette icon is present in both radiographic and photo mode with a strike-through | Expected outcome verified. Verified by AM 28MAY24 | p |
|  | Test Case: Cassette WiFi AP |  |  |  |  |
| SRS-7.3 | The SS shall initiate a private WiFi Access Point (WiFi AP) upon startup of the cassette | 1. SSH into the emitter 2. List the WIFI networks available to the emitter with nmcli device wifi list | In the list of WiFi networks available to the emitter, the cassette's network is listed in green | Expected outcome verified. See Appendix 8. Verified by DH 21MAY24 | P |
| SRS-7.1 | The SS shall use the WPA2 security standard for internal and external wireless communication |  | In the list of WIFIs the cassette's network is listed as WPA2 | Expected outcome verified. See Appendix 8. Verified by DH 21MAY24 | P |
| SRS-7.5 | The SS shall display the cassette-hosted WiFi AP SSID and password on the cassette display | 1. Take an image of the cassette display. Verify that the cassette's SSID and password are shown. 2. Using a tablet, attempt to connect to the cassette WiFi with an incorrect password 3. Record evidence of the failure to connect the tablet to the cassette WiFi 4. Using the same tablet, connect to the cassette WiFi with the displayed password 5. Record evidence of a successful connection of the tablet to the cassette WiFi | The cassette display has the SSID and password present | Expected outcome verified. See Appendix 9. Verified by DH 21MAY24 | P |
| SRS-7.4 | The SS shall require credentials to access the Cassette WiFi AP |  | Tablet fails to connect to cassette WiFi with incorrect password | Expected outcome verified. See Appendix 10. Verified by DH 21MAY24 | P |
|  |  |  | Tablet is able to connect to cassette WiFi with password shown on cassette display | Expected outcome verified. See Appendix 11. Verified by DH 21MAY24 | P |

### Table 14
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Release Mode Precondition: Ensure the emitter and cassette do NOT have a wired network connection prior to placing the system into release mode. Place the emitter in release mode before the paired cassette. Emitter: Place the emitter in release mode using the following steps: 1. Connect a mouse and keyboard to the emitter service port via dongle. Note that an external monitor can be used but is not necessary; the terminal will be visible in the emitter display 2. Open a terminal 3. As root, run the following command: rm /imagerdebug 4. Restart the emitter Cassette: Place the cassette in release mode using the following steps: 1. Connect a mouse, keyboard, and external monitor to one of the cassette service ports via dongle 2. Open a terminal 3. As root, run the following command: rm /imagerdebug 4. Restart the cassette |  |  |  |  |
| SRS-1.10 | The SS shall contain a release mode for devices ready for distribution. Debug mode shall be disabled in release mode. | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the emitter with ssh imager@<emitter-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure | Expected outcome verified. See Appendix 11. Verified by DH 21MAY24 | P |
| SRS-1.15 | In release mode, the SS shall disable SSH |  |  |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure | Expected outcome verified. See Appendix 12. Verified by DH 21MAY24 | P |
| SRS-1.16 | In release mode, the SS shall present a blank screen if an external display is connected to the cassette via service port | 1. Connect a mouse, keyboard, and external monitor to the emitter service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to emitter | Expected outcome verified. See Appendix 14. Verified by DH 21MAY24 | P |
| SRS-1.17 | In release mode, the SS shall present a blank screen if an external display is connected to the emitter via service port | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to cassette | Failed See Appendix 15. Verified by DH 21MAY24 | F |
| SRS-1.13 | In release mode, the SS shall force logouts of any open maintenance mode terminals after 120 seconds of inactivity | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Use the esoteric key combination to open a terminal. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on emitter | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Use the esoteric key combination to open a terminal. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on cassette | Failed See Appendix 15. Verified by DH 21MAY24 | F |
| SRS-1.14 | In release mode, the SS shall enforce the use of a restricted keyboard key set | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on emitter display | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on external monitor for cassette | Failed See Appendix 15. Verified by DH 21MAY24 | F |
| SRS-1.11 | In release mode, the SS shall restrict access to production-level accounts | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Use the esoteric key combination to open a terminal 3. Using the command su -, attempt to log in as root user by using incorrect credentials for the unit under test 4. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on emitter | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  |  | 1. Connect a mouse, keyboard, and monitor to a cassette service port via dongle 2. Use the esoteric key combination to open a terminal 3. Using the command su -, attempt to log in as root user by using incorrect credentials for the unit under test 4. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on cassette | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  | Test Case: Integrity Check |  |  |  |  |
| SRS-1.18 | In release mode, the SS shall perform an integrity check upon boot and every hour | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Use the esoteric key combination to open a terminal 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.8 -m mx1.services stop To start: python3.8 -m mx1.services start 5. Reboot the emitter. Alternatively, wait one hour for the integrity check alarm to be triggered 6. Open another terminal session 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the emitter | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  |  |  | Record evidence of the integrity check alarm on the emitter | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Use the esoteric key combination to open a terminal 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.8 -m mx1.services stop To start: python3.8 -m mx1.services start 5. Reboot the emitter. Alternatively, wait one hour for the integrity check alarm to be triggered 6. Open another terminal session 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the cassette | Failed See Appendix 15. Verified by DH 21MAY24 | F |
|  |  |  | Record evidence of the integrity check alarm on the cassette | Failed See Appendix 15. Verified by DH 21MAY24 | F |

### Table 15
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Debug Mode Precondition: Emitter: Place the emitter in debug mode using the following steps: 1. Connect a mouse and keyboard to the emitter service port via dongle. Note that an external monitor can be used but is not necessary; the terminal will be visible in the emitter display 2. Use the esoteric key combo to open a terminal 3. As root user, run the following command: touch /imagerdebug 4. Restart the emitter Cassette: Place the cassette in debug mode using the following steps: 1. Connect a mouse, keyboard, and external monitor to one of the cassette service ports via dongle 2. Use the esoteric key combo to open a terminal 3. As root user, run the following command: touch /imagerdebug 4. Restart the cassette |  |  |  |  |
| SRS-1.7 | The SS shall contain a debug mode for development and production activities | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), SSH into the emitter with ssh imager@<emitter-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to emitter in debug mode | Expected outcome verified. See Appendix 16. Verified by DH 21MAY24 | P |
| SRS-1.9 | In debug mode, the SS shall enable SSH |  |  |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to cassette in debug mode | Expected outcome verified. See Appendix 17. Verified by DH 21MAY24 | P |

### Table 16
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-440 |  |
