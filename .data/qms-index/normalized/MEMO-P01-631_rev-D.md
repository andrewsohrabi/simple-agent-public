# MEMO-P01-631 Rev D: MX1 Software Design Specifications

## Metadata
- Document ID: MEMO-P01-631
- Revision: D
- Prefix: MEMO
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: unknown
- Source filename: MEMO-P01-631 - MX1 Software Design Specifications_D_Obsolete.docx
- Source path: Example QMS - MedAI/MEMO-P01-631 - MX1 Software Design Specifications_D_Obsolete.docx
- Extraction warnings: none

## Extracted Content
MEMO-P01-631 - MX1 Software Design Specifications_D_Obsolete
Sheet: Signoff
Sheet: Introduction
Sheet: JO
Sheet: EO
Sheet: CO
Sheet: IMP
Sheet: IRS
Sheet: XRC
Sheet: EMF
Sheet: IMWSCDDIRCC
Sheet: EM FW
Sheet: MB FW
Sheet: COL FW
Sheet: CAS FW
Sheet: FP FW
Sheet: CP
Sheet: ODA
Sheet: SOUPs

### Table 1
| MedAI MEDICAL, INC |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Document: | MEMO-P01-631 - MX1 Software Design Specifications |  |  |  |  |
| Project: | P01 |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |
| Revision | DCO # | Approved By | Eff. Date | Description | Digital Key |
| A | 24-198 | EngineeringQuality EngineeringRegulatory Affairs | 2024-04-25 00:00:00 | Initial Release | example.com/ |
| B | Refer to ECR-449 |  |  | Updates for MX1 SS v3.1.0 release | example.com/ |
| C | Refer to ECR-470 |  |  | Updates for MX1 SS v3.2.0 release | example.com/ |
| D | Refer to ECR-575 |  |  | Updates for MX1 SS v4.0.0 release | example.com/ |

### Table 2
| Purpose |  |
| --- | --- |
| The purpose of this document is to capture the detailed design specifications of the MX1 Software System (SS) and MedAI Device App (ODA) meant to support the device in meeting the overall device and software system requirements. |  |
| Scope |  |
| This document records the detailed design specifications for the individual Software Components that compose the MX1 Software System and MedAI Device App in accordance with the Software Detailed Design section defined in PLN-P01-024 - MX1 Software Development Plan. |  |
| References |  |
| IEC 62304:2015 - Medical device software - Software life cycle processes |  |
| PLN-P01-024 - MX1 Software Development Plan Rev. D |  |
| MEMO-P01-630 - MX1 Software Requirement Specifications Rev. E |  |
| Document Overview |  |
| The MX1 Software Design Specifications documents the specific design details to allow the MX1 Software System and MedAI Device App to meet requirements recorded in MEMO-P01-630 - MX1 Software Requirement Specifications.                 In this document, the Software Component Specifications are recorded in each individual tab. Each specification is linked to a driving MX1 Software System requirement. |  |

### Table 3
| Category | Related SRS ID | SDS ID | MX1 Jetpack 5 OS (JO) Design Specifications |
| --- | --- | --- | --- |
| Startup/OS Configuration | SRS-4.1SRS-4.2 | JO-SDS-1 | JO shall be based on the Ubuntu 20.04 LTS operating system |
| Startup/OS Configuration | SRS-4.1SRS-4.2 | JO-SDS-2 | JO shall use the ext4 filesystem. |
| Startup/OS Configuration | SRS-4.1SRS-4.10SRS-10.27 | JO-SDS-3 | JO shall set emitter Jetson register 0x15b40138 to 0x0A0A0A07 during startup |
| Startup/OS Configuration | SRS-7.3 | JO-SDS-4 | JO shall enable the cassette WiFi Access Point upon startup |
| Startup/OS Configuration | SRS-7.1 | JO-SDS-5 | JO shall enable the WPA2 protocol to encrypt all WiFi communications internal to the MX1 system |
| Iptables - Firewall for Emitter | SRS-7.10 | JO-SDS-6 | JO shall block all inbound connections to wlan0 except ports 8787 and 8788 on the emitter |
| Iptables - Firewall for Cassette | SRS-7.4 | JO-SDS-7 | JO shall block all inbound connections to wlan0 and eth0 on the cassette |
| Iptables - Firewall for Cassette | SRS-7.4 | JO-SDS-8 | JO shall block all inbound connections to wap0 on the cassette except 8080,8081,8082,8083,8084 (for AP clients) |
| Iptables - Firewall for Cassette | SRS-7.4 | JO-SDS-9 | JO shall block all inbound connections to wap0 on the cassette except 8080,8081,8082,8083,8086,8088, 8787, 8788 (for emitter) |
| Services/Permissions | SRS-48.4 | JO-SDS-10 | JO shall make imager user a member of sudo, video, audio, adm, dialout, i2c, input, gpio, gpiod, jtop groups |
| Services/Permissions | SRS-48.4 | JO-SDS-11 | JO shall set log file ownership to the adm/syslog group |
| Services/Permissions | SRS-48.4 | JO-SDS-12 | JO shall assign privilege to imager user to shut down the system by setting the appropriate permissions in sudoers system. |
| Integrity Checks | SRS-4.6 | JO-SDS-13 | JO shall record a hash of all files during build to support integrity checking. |
| Integrity Checks | SRS-4.6 | JO-SDS-14 | JO shall verify integrity of /etc/passwd and /etc/shadow by comparison to original hash at each boot. |
| Integrity Checks | SRS-4.6 | JO-SDS-15 | JO shall verify integrity of the configuration files by comparison to original hashes at each boot. |
| Integrity Checks | SRS-4.6 | JO-SDS-16 | JO shall verify integrity of /opt/imager/bin/iptables.sh by comparison to original hash at each boot. |
| System Logging | SRS-3.1SRS-3.2 | JO-SDS-17 | JO shall utilize rsyslog to organize and store logging for all cassette and emitter services |
| System Logging | SRS-3.1SRS-3.2 | JO-SDS-18 | JO shall store logs for all MedAI services in the /var/log/medai/[MedAI service name] log files |
| System Logging | SRS-3.1SRS-3.2 | JO-SDS-19 | JO shall organize log outputs with the following log levels:1. Critical (/var/log/medai/medai_crit.log)2. Error ( /var/log/medai/medai_error.log) 3. Warning (/var/log/medai/medai_warning)4. Informational (/var/log/medai/medai_info.log)5. Debug ( /var/log/medai/medai_debug.log)6. Trace (/var/log/medai/medai_trace.log) |
| Network | SRS-43.2SRS-43.3 | JO-SDS-20 | JO shall set the MTU for the eth0 connection at 192.168.8.188/24 to 9000 |
| WiFi AP Client Isolation | SRS-7.2SRS-7.3 | JO-SDS-21 | JO shall generate the cassette WiFi AP SSID by concatenating "cassette' and cassette serial number |
| WiFi AP Client Isolation | SRS-7.5 | JO-SDS-22 | JO shall assign 10.24.96.1 as the IP address for the cassette AP interface |
| WiFi AP Client Isolation | SRS-7.5 | JO-SDS-23 | JO shall assign 10.24.96.2 as the IP address and port for the emitter AP interface |
| WiFi AP Client Isolation | SRS-7.5 | JO-SDS-24 | JO shall assign an IP address from the range of 10.24.96.50 to 60 for non-emitter AP interfaces |
| Image Transfer - USB | SRS-44.6 | JO-SDS-25 | JO shall support export of images and videos to a single-function USB storage device. |
| Directory Structure | SRS-1.8SRS-1.9 | JO-SDS-26 | JO shall configure /opt/imager/bin/ for system setup scripts |
| Directory Structure | SRS-2.1SRS-2.2 | JO-SDS-27 | JO shall configure /opt/medai/bin/ for application binaries |
| Directory Structure | SRS-28.4SRS-34.2SRS-44.5 | JO-SDS-28 | JO shall configure /opt/medai/data/ for database storage |
| Directory Structure | SRS-28.1SRS-28.2SRS-28.4 | JO-SDS-29 | JO shall configure /opt/medai/data/images for image storage |
| Directory Structure | SRS-2.1SRS-2.2 | JO-SDS-30 | JO shall configure /opt/medai/data/config for application configuration files |
| Directory Structure | SRS-2.1SRS-2.2 | JO-SDS-31 | JO shall configure /opt/medai/data/config/device-specific/ for device specific configuration files |
| Directory Structure | SRS-2.1SRS-2.2SRS-3.1SRS-3.2SRS-28.1SRS-28.2 | JO-SDS-32 | JO shall partition the emitter and cassette Jetson NVMe drives in the following manner:/dev/nvme0n1p1 -> 32G  (for root A, /)/dev/nvme0n1p2 -> 32G  (for root B, /)/dev/nvme0n1p12 -> 30G (for /otatmp)/dev/nvme0n1p13 -> 30G  (for /var/log)/dev/nvme0n1p14 ->113G  (for /opt/medai/data) |
| Release Mode | SRS-1.9 | JO-SDS-33 | JO shall place the system in release mode if /imagerdebug flag is not present in the root directory upon system startup |
| Release Mode | SRS-1.9 | JO-SDS-34 | For release mode, JO shall enable the host-based firewalls (see IPTables section in this document) |
| Release Mode | SRS-1.9 | JO-SDS-35 | For release mode, JO shall implement an esoteric key combo for starting a terminal for maintenance |
| Release Mode | SRS-1.9 | JO-SDS-36 | For release mode, JO shall force logouts of open maintenance mode terminal after 120 seconds of inactivity |
| Debug Mode | SRS-1.8 | JO-SDS-37 | JO shall place the system in debug mode if /imagerdebug flag is present in the root directory upon system startup |
| Debug Mode | SRS-1.8 | JO-SDS-38 | For debug mode, JO shall enable SSH and use default port 22 |
| Debug Mode | SRS-1.8 | JO-SDS-39 | For debug mode, JO shall disable forced logouts of maintenance mode terminals |
| Debug Mode | SRS-1.8 | JO-SDS-40 | For debug mode, JO shall enable a minimal host-based firewall |

### Table 4
| Category | Related SRS ID | SDS ID | Emitter Orchestrator (EO) Design Specifications |
| --- | --- | --- | --- |
| Basic Safety/Interlocks | SRS-14.1 | EO-SDS-1 | EO shall evaluate the state of the system based on interlocks, heartbeats, and health signals. |
| Basic Safety/Interlocks | SRS-14.1SRS-14.2 | EO-SDS-2 | EO shall broadcast an interprocess signal called safe_state; when true, the system will be in safe state. |
| Basic Safety/Interlocks | SRS-14.1 | EO-SDS-3 | When in safe state, the emitter lasers shall be off, the LEDs shall be magenta, the monoblock shall be turned off, and x-rays shall be forbidden. |
| Basic Safety/Interlocks | SRS-14.1 | EO-SDS-4 | EO shall broadcast an interprocess signal called system_ready; when false, if not in safe state, LEDs will be blue, lasers off, monoblock on. |
| Basic Safety/Interlocks | SRS-19.1 | EO-SDS-5 | EO shall broadcast an intermachine signal indicating the health and status of the emitter and its systems called emitter_statuses_signal. |
| Basic Safety/Interlocks | SRS-14.1 | EO-SDS-6 | EO shall listen for an intermachine signal indicating the health and status of the cassette and its systems called cassette_statuses_signal. The system status will be determined by aggregating the statuses from the cassette with the statuses of the emitter. |
| Basic Safety/Interlocks | SRS-14.1 | EO-SDS-7 | EO shall disallow x-rays if any of the following are true: - SSD interlock not met- motion detected in fluoro mode- system is in idle state;- in safe state;- in photo mode;- emitter battery <= 15;- cassette battery <= 20% (not charging);- tracking distance not okay;- tracking not in bounds;- tracking not working;- emitter is plugged in;- emitter radio not connected;- system ready flag is false.- disable Xrays is true;- no captures remaining- less than 100 captures remaining and in ddr/fluoro mode |
| Basic Safety/Interlocks | SRS-14.1 | EO-SDS-8 | EO shall broadcast an intermachine signal called xrays_allowed_signal that indicates whether or not x-rays are allowed (see above for evaluation criteria). |
| Battery Life - ViewFinder | SRS-25.3 | EO-SDS-9 | EO shall query the state of the battery every half second, and brodcast the emitter_battery_info signal with that information. |
| Imaging Modes | SRS-23.3 | EO-SDS-10 | EO shall determine the shooting mode of the system- photo, single x-ray, DDR, or fluoro - and shall broadcast an intermachine stateful signal whenever it changes called mode_set. |
| Photographic Acquisition | SRS-23.2 | EO-SDS-11 | When the trigger is pressed in photo mode, EO shall broadcast a signal called photo_triggered_signal. |
| Photographic Acquisition | SRS-23.3 | EO-SDS-12 | When in photo mode, the lasers shall be off, and the LEDs shall be white. |
| Power States - Emitter Idle | SRS-10.18 | EO-SDS-13 | EO shall reset the idle timer when any keypress event is detected. |
| Power States - Emitter Idle | SRS-10.1 | EO-SDS-14 | EO shall monitor the IDLE icd flag from the emitter main, and broadcast changes to the idle state with the device_idle_signal. |
| Power States - Emitter Idle | SRS-10.7 | EO-SDS-15 | EO's EmitterImpl class shall listen for the idle_state_signal. If this signal indicates the system is idle, it shall use ICD commands to set the LED lights to breathe blue, shall turn off the emitter display, turn off the lasers, turn off ToF sensorsand turn off the monoblock. If the state changes away from idle, the monoblock shall be turned on, and LEDs/lasers shall be restored to their previous state. |
| Power States - Emitter Idle | SRS-10.1 | EO-SDS-16 | EO shall monitor the IDLE icd flag from the emitter main, and broadcast changes to the idle state with the device_idle_signal. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-17 | EO shall be responsible for listening for trigger presses and initiating appropriate responses. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-18 | EO shall look for device /dev/input/by-path/platform-gpio-keys-event, and listen to it as a keyboard input device if found. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-19 | EO shall interpret a BTN_6 keypress as a request to change the mode. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-20 | EO shall interpret a BTN_8 keypress as a trigger press/release. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-21 | EO shall interpret a BTN_9 keypress as a trigger press/release. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-22 | EO shall interpret a press of the right footpedal as a trigger press/release. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-23 | EO shall interpret the signal rest_trigger_signal as a trigger press/release. |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-24 | EO shall multiplex and aggregate all possible trigger press/release events into a single logical trigger_state signal. |
| Radiographic Acquisition | SRS-19.2 | EO-SDS-25 | When EO detects a trigger press when in radiographic mode, it shall determine whether to initiate a DDR or single x-ray based on how many milliseconds go by before the trigger is released. |
| Radiographic Acquisition | SRS-19.2 | EO-SDS-26 | The time threshold for determining DDR vs. single x-ray shall be configurable, and shall default to 300ms |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-27 | When a trigger is pressed in radiographic mode, EO shall send the intermachine broadcaster xray_pre_fire_signal, which contains all the known details and metadata about the upcoming x-ray (selected techniques, single/ddr, etc.) |
| Radiographic Acquisition | SRS-19.1 | EO-SDS-28 | When EO receives the cassette_ready signal (sent in response the the xray_pre_fire_signal), it shall send an ICD command to the monoblock to fire an x-ray. |
| Radiographic Acquisition - MI LEDs | SRS-16.14 | EO-SDS-29 | When in radiographic mode, the LEDs shall be green if the system is allowed to take an x-ray, or red if the system is healthy, but x-rays are not allowed. |
| Radiographic Acquisition - ViewFinder | SRS-16.8 | EO-SDS-30 | While a DDR is happening and while the DDR cooldown is happening immediately after a DDR, EO shall broadcast the ddr_gauge_set indicating the sate of DDR "fuel". |
| Tracking/Positioning - Lasers | SRS-16.16 | EO-SDS-31 | When in radiographic mode, the lasers shall be on if pointed at the cassette and in an acceptable firing position, based on tracking information. |
| Tracking/Positioning - Lasers | SRS-16.17 | EO-SDS-32 | When in radiographic mode, the lasers shall be on but blinking if pointed at the cassette, but not in a valid firing position, based on tracking information. |
| Emitter Debug Window | SRS-1.8 | EO-SDS-33 | EO shall implement a debugging endpoint to display signal and fault statuses using port 8088 |
| Emitter Shutdown | SRS-5.1 | EO-SDS-34 | EO shall initiate emitter shutdown upon receiving the shutdown_rq_from_emitter_signal |
| Emitter Shutdown | SRS-5.1 | EO-SDS-35 | EO shall interpret a 3200 ms BTN_6 keypress as a shutdown request and shall send the shutdown_rq_from_emitter_signal in response |
| Emitter Shutdown | SRS-5.5 | EO-SDS-36 | EO shall initiate emitter shutdown upon receiving the shutdown_rq_from_cassette_signal |
| Pulse Delay | SRS-19.16 | EO-SDS-37 | EO shall load and set the pulse delays as defined in emitter-orchestrator-config.json |
| Pulse Delay | SRS-19.16 | EO-SDS-38 | For a pulse delay value that is set for a particular exposure time, EO shall use that value for all exposure times equal to or greater than the specifed value until a different exposure time is met |
| DDR Enable/Disable | SRS-34.3 | EO-SDS-39 | EO shall subscribe to the signal ddrsAreAllowed_; when true, the system shall allow serial radiographic acquisition. |
| SSD Interlock Enable/Disable | SRS-34.4SRS-34.5 | EO-SDS-40 | EO shall broadcast an intermachine signal ssdInterlockSignal_; when true, the system shall allow radiographic acquisition when SSD is above the minimum bound |
| SSD Interlock Enable/Disable | SRS-34.4SRS-34.5 | EO-SDS-41 | EO shall subscribe to ssd_interlock_dist_signal_mm to determine if SSD interlock is enabled and at which value the minimum bound is set |
| Emitter Movement Interlock | SRS-14.23SRS-14.24 | EO-SDS-42 | EO shall set and monitor a timer to keep track of the time since the last reported emitter movement |
| Emitter Movement Interlock | SRS-14.23SRS-14.24 | EO-SDS-43 | EO shall disallow radioscopic acquisition if timer is a value of less than 1000 ms |
| Wireless Charging | SRS-27.1SRS-27.2SRS-27.3 | EO-SDS-44 | EO shall disable wireless charging via ICD command to EM for the duration of x-ray emission |

### Table 5
| Category | Related SRS ID | SDS ID | Cassette Orchestrator (CO) Design Specifications |
| --- | --- | --- | --- |
| Radiographic Acquisition | SRS-16.1 | CO-SDS-1 | CO shall be responsible for coordinating receiving x-ray images from the detector, processing those images, broadcasting those images, and saving those images. |
| Radiographic Acquisition | SRS-16.1 | CO-SDS-2 | CO shall listen for the  xray_pre_fire_signal. When received, it shall store the meta data and prepare to receive x-rays. When the preparation is complete, CO shall broadcast the cassette_ready intermachine signal. |
| Radiographic Acquisition | SRS-16.1 | CO-SDS-3 | CO shall listen for the iray_detector_raw_image_bytes_signal. When received, CO shall process the image, and broadcast the image via websocket to the tablet app. |
| Radiographic Acquisition | SRS-16.1 | CO-SDS-4 | CO shall listen for the xray_post_fire_signal, which shall contain meta data about the just-completed x-ray(s). Upon receiving this signal, CO shall save the images (with their meta data) and notify CP via REST about those images. |
| Basic Safety/Interlocks | SRS-14.15 | CO-SDS-5 | CO shall evaluate the state of the system based on interlocks, heartbeats, and health signals. |
| Basic Safety/Interlocks | SRS-14.15 | CO-SDS-6 | CO shall broadcast an interprocess signal called safe_state; when true, the system shall be in safe state. |
| Basic Safety/Interlocks | SRS-14.15 | CO-SDS-7 | When in safe state, the RBG LEDs shall be magenta, the detecor shall be turned off, and IR tracking LEDs shall be off. |
| Basic Safety/Interlocks | SRS-14.15 | CO-SDS-8 | CO shall broadcast an interprocess signal called system_ready; when false, if not in safe state, LEDs shall be blue, IR tracking LEDs off, detctor on. |
| Basic Safety/Interlocks | SRS-14.15 | CO-SDS-9 | CO shall broadcast an intermachine signal indicating the health and status of the cassette and its systems called cassette_statuses_signal. |
| Basic Safety/Interlocks | SRS-14.15 | CO-SDS-10 | CO shall listen for an intermachine signal indicating the health and status of the emitter and its systems called emitter_statuses_signal. The system status shall be determined by aggregating the statuses from the emitter with the statuses of the cassette. |
| Basic Safety/Interlocks | SRS-14.1 | CO-SDS-11 | CO shall listen for the intermachine signal xrays_allowed_signal, and track the status of that signal. |
| Basic Safety/Interlocks | SRS-16.14 | CO-SDS-12 | When in radiographic mode, the RGB LEDs shall be green if x-rays are allowed, and red if they are not. |
| Imaging Modes | SRS-23.3 | CO-SDS-13 | CO shall listen for an intermachine stateful signal called mode_set, and shall be change cassette behavior accordingly as the mode changes. |
| Photographic Acquisition | SRS-23.2 | CO-SDS-14 | CO shall listen for an intermachine signal called photo_ready_signal. When received, CO shall broadcast the image via websocket to the tablet app, save the image, and notify CP via REST that a new photo image has been received. |
| Imaging Modes | SRS-23.3 | CO-SDS-15 | When in photo mode, the LEDs shall be white, and the IR tracking LEDs shall be off. |
| Imaging Modes | SRS-21.4 | CO-SDS-16 | When in radiographic mode, the IR tracking LEDs shall be on. |
| Power States - Cassette Idle | SRS-10.3 | CO-SDS-17 | CO's CassetteImpl class shall listen for the idle_state_signal. If this signal indicates the system is idle, it shall use ICD commands to set the RGB LED lights to breathe blue, shall turn off IR tracking LEDs, and turn off the detector. If the state changes away from idle, the detector shall be turned on, and LEDs shall be restored to their previous state. |
| Battery Life | SRS-25.5 | CO-SDS-18 | CO shall query the state of the battery every half second, and brodcast the cassette_battery_info signal with that information. |
| Engineering Mode | SRS-1.2 | CO-SDS-19 | CO shall provide the following REST endpoint for mode switch: http://<ip address>:8081/remote_api?command=mode_set&mode=[imaging mode], where imaging mode can be set to photo, xray_manual, or ddr_manual |
| Engineering Mode | SRS-1.3 | CO-SDS-20 | CO shall provide the following REST endpoint for technique factor setting: http://<ip address>:8081/remote_api?command=set_techniques&kv=[tube voltage value]&exposure=[exposure time]&beam_current=[beam current] |
| Engineering Mode | SRS-1.5 | CO-SDS-21 | CO shall provide the following REST endpoint to simulate pulling the trigger: http://<ip address>:8081/remote_api?command=trigger_press&press_time=[time in milliseconds] |
| Engineering Mode | SRS-1.4 | CO-SDS-22 | CO shall provide the following REST endpoint to get metadata about the most recent x-ray acquisition: http://<ip address>:8081/last_xray_data |
| Engineering Mode | SRS-1.7 | CO-SDS-23 | CO shall provide the following REST endpoint to place the system into idlehttp://<ip address>:8081/remote_api?command=idle_state&state=true |
| Cassette Debug Window | SRS-1.8 | CO-SDS-24 | CO shall implement a debugging endpoint to display signal and fault statuses using port 8086 |
| Image Processing | SRS-28.3 | CO-SDS-25 | CO shall implement an image processing endpoint to manipulate brightness, contrast, and sharpness using port 8081 |
| Cassette Power Button Shutdown | SRS-5.2 | CO-SDS-26 | CO shall initiate cassette shutdown upon receiving the cassette_button_down_signal |
| Cassette Power Button Shutdown | SRS-5.2 | CO-SDS-27 | CO shall interpret a 3200 ms cassette power button press as a shutdown request and shall send the cassette_button_down_signal in response |
| ODA Shutdown | SRS-5.5 | CO-SDS-28 | CO shall initiate cassette shutdown upon receiving the shutdown_rq_from_cassette_signal |
| ODA Shutdown | SRS-5.5 | CO-SDS-29 | If shutdown is initiated using the shutdown_rq_from_cassette_signal, CO shall implement a 200 ms sleep to allow EO to hear the same signal before completing the cassette shutdown |
| Offset Calibration | SRS-46.1 | CO-SDS-30 | CO shall perform an offset correction by subtracting the dark frame from the light frame for each acquisition |
| Gain Calibration | SRS-46.2 | CO-SDS-31 | If the gain_mask_file attribute in device-specific-config.json points to a gain map, CO shall use the map to perform gain calibration for each offset corrected raw image |
| Defect Mapping | SRS-46.3 | CO-SDS-32 | If the image_pixel_mask attribute in device-specific-config.json points to a defect map, CO shall use the defect map as an interpolation mask for each offset corrected and gain calibrated raw image |
| General Image Calibration | SRS-46.1SRS-46.2SRS-46.3 | CO-SDS-33 | CO shall perform raw image calibration activities in the following sequence: offset calibration, gain calibration, and defect calibration |
| S-Curve Application | SSRS-28.5 | CO-SDS-34 | CO shall adjust image contrast by applying an s-curve to offset, gain, and defect calibrated raw images |
| Image Storage | SSRS-28.5 | CO-SDS-35 | CO shall scale JPEG2000 images and store as 16-bit images |
| SSD Interlock Enable/Disable | SRS-34.5 | CO-SDS-42 | CO shall broadcast an intermachine signal ssdInterlockDistanceInMm_ to send the set minimum SSD interlock bound. |
| SSD Interlock Enable/Disable | SRS-34.4 | CO-SDS-36 | CO shall broadcast an intermachine signal ssdInterlockEnabled_; when true, the system shall disallow x-ray emission below the set value for ssInterlockDistanceInMm_ |
| Low Resolution Serial Radiographic Images | SRS-43.21 | CO-SDS-37 | During initial streaming of serial radiographic frames, CO shall resize the frames to 25% of the original size (along both x and y dimensions). This percentage is configurable. |
| DDR Display Delay | SRS-43.31 | CO-SDS-45 | When acquiring a serial radiographic image, CO shall display an image with "DDR starting" message for 0.5 second prior to displaying the first acquired frame. |

### Table 6
| Category | Related SRS ID | SDS ID | Intermachine Signal Proxy (IMP) Design Specifications | Implemented? | Finalized? | Testable? | Document Reference? |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | False | False | False |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-1 | Each instance of IMP shall implement a network server broadcaster (NSB) for outgoing transmissions of intermachine signals. | True | True | True | (see diagram) |
| Device Component Communication | SRS-7.6 | IMP-SDS-2 | The network server broadcaster for outgoing signal transmissions shall use the websocket server protocol. | True | True | False | See MEMO-01-i-29. |
| Device Component Communication | SRS-7.6 | IMP-SDS-3 | Each instance of IMP shall implement a network client receiver (NCR) for listening to incoming transmissions of intermachine signals. | True | True | True | See Interface Control Document |
| Device Component Communication | SRS-7.6 | IMP-SDS-4 | The network client receiver for incoming signal transmissions shall use the websocket client protocol. | True | True | True | See RFC 1055 |
| Device Component Communication | SRS-7.6SRS-14.12SRS-14.16 | IMP-SDS-5 | One instance of intermachine proxy shall be run on the cassette, and another instance shall be run on the emitter. | True | True | True | See RFC 1055 |
| Device Component Communication | SRS-7.6SRS-14.1SRS-14.2 | IMP-SDS-6 | By default/convention, the NSB shall listen for clients on port 8765, and the NCR shall look for servers on port 8765. | True | True | True |  |
| Device Component Communication | SRS-7.6SRS-14.1SRS-14.2 | IMP-SDS-7 | At startup, the IMP shall start the NSB which shall wait for client requests on the default port. | True | True | True | (see diagram) |
| Device Component Communication | SRS-7.6 | IMP-SDS-8 | If an IP address is sent as a command line argument, at startup the IMP shall start the NCR, which attempt to connect to an NSB at that IP address on the default port. | True | True | True |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-9 | If an NCR attempts to make a client connection to an instance of IMP who's own NCR is not attached to a NSB, the IMP shall attempt to connect its NCR to an NSB at the same source IP address as the incoming client connection. | True | True | True | (see diagram) |
| Device Component Communication | SRS-7.6 | IMP-SDS-10 | On the cassette, the IMP shall be started with no command line parameters. | True | True | True | (see diagram) |
| Device Component Communication | SRS-7.6 | IMP-SDS-11 | On the emitter, the IMP shall be started with a command line argument of 10.24.96.1, the IP address of the cassette (on the wifi network AP maintained by the cassette). | True | True | True | (see diagram) |
| Device Component Communication | SRS-7.6 | IMP-SDS-12 | The IMP shall subscribe to the util::core_radio::websocket_proxy signal, with a parameter size limit of 2000 bytes. | True | True | True | (see diagram) |
| Device Component Communication | SRS-7.6 | IMP-SDS-13 | The IMP shall subscribe to the util::core_radio::jumbo_websocket_proxy, with a parameter size limit of 5844444 bytes. | True | True | False |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-14 | Packet format is as follows: [signal-id][serialized parameters][flags][sequence number], where signal id is two bytes, serialized parameters is however many bytes are appropriate to this signal, flags is one byte, and sequence number is one byte. | True | True | False |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-15 | When a packet is received, the NCR shall compare the sequence byte of the packet to the expected sequence byte, and post a warning in the log if they don't match. | True | True | False |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-16 | When a packet is received, the NCR shall inspect the first two bytes to determine the signal id. | True | True | False |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-17 | When a packet is received, the NCR shall query shared memory for a list of local processes that have subscribed to the signal associated with this packet. | True | True | False |  |
| Device Component Communication | SRS-7.6 | IMP-SDS-18 | When a packet is received, the NCR shall extract the serialized parameters from the packet, and copy them to the shared memory queue for each subscribed process. | True | True | False |  |
| Device Component Communication | SRS-7.6SRS-14.12SRS-14.16SRS-14.1SRS-14.2 | IMP-SDS-19 | The IMP shall implement a health/heartbeat signal called util::core_radio::improxy_status_signal. This signal shall be called once every second. This signal shall indicate the state of the IMP. | True | True | True |  |
|  |  |  |  | False | False | False |  |

### Table 7
| Category | Related SRS ID | SDS ID | iRay Signaler (IRS) Design Specifications |
| --- | --- | --- | --- |
| Signal Subscriptions | SRS-2.2 | IRS-SDS-1 | IRS shall subscribe to device_idle_signal |
| Signal Subscriptions | SRS-2.2 | IRS-SDS-2 | IRS shall subscribe to xray_pre_fire_signal |
| Signal Subscriptions | SRS-2.2 | IRS-SDS-3 | IRS shall subscribe to rest_gain_mode |
| Signal Subscriptions | SRS-2.2 | IRS-SDS-4 | IRS shall subscribe to xray_firing |
| Signal Subscriptions | SRS-2.2 | IRS-SDS-5 | IRS shall subscribe to last_image_received_signal |
| Signal Broadcasters | SRS-2.2 | IRS-SDS-6 | IRS shall broadcast image data across the iray_detector_raw_image_bytes_signal |
| Signal Broadcasters | SRS-2.2 | IRS-SDS-7 | IRS shall broadcast acquiring status as a bool across the detector_acquire_state signal |
| Signal Broadcasters | SRS-14.17 | IRS-SDS-8 | IRS shall broadcast the detector health as an int across the iray_health signal |
| Signal Broadcasters | SRS-19.2 | IRS-SDS-9 | IRS shall broadcast light frame image data across the iray_detector_light_image_bytes_signal |
| Signal Broadcasters | SRS-19.2 | IRS-SDS-10 | IRS shall broadcast dark frame image data across the iray_detector_dark_image_bytes_signal |
| Interfaces | SRS-2.2 | IRS-SDS-11 | IRS shall communicate with the iRay Mercu 0909X via the iRay SDK |
| Gain Mode | SRS-2.2 | IRS-SDS-13 | IRS shall set the gain mode to the received value from the rest_gain_mode signal when engineering mode is enabled |
| Gain Mode | SRS-46.2 | IRS-SDS-14 | IRS shall calculate the gain mode using the received values for SID, tube voltage (kV), beam current (mA), and exposure time (ms) |
| Detector Timing | SRS-2.2 | IRS-SDS-15 | IRS shall set the detector frame rate to 3 frames per second for single radiographic acquisions |
| Detector Timing | SRS-2.2 | IRS-SDS-16 | IRS shall set the detector frame rate to 6 frames per second for serial radiographic acquistions |
| Acquistion | SRS-2.2 | IRS-SDS-17 | IRS shall start a detector acquisition when the xray_pre_fire_signal is received, the detector is ready, and the mode is not photo |
| Acquistion | SRS-2.2 | IRS-SDS-18 | IRS shall terminate a detector acquisition when the last_image_received_signal is received |
| Health | SRS-2.2SRS-14.1SRS-14.2SRS-14.17 | IRS-SDS-19 | IRS shall send a health value of -2 when the detector fails to initialize |
| Health | SRS-2.2SRS-14.1SRS-14.2SRS-14.17 | IRS-SDS-20 | IRS shall send a health value of 0 when the detector successfully initializes |
| Health | SRS-2.2SRS-14.1SRS-14.2SRS-14.17 | IRS-SDS-21 | IRS shall send a health value of 1 when the detector is healthy |

### Table 8
| Category | Related SRS ID | SDS ID | XR Controller (XRC) Design Specifications |
| --- | --- | --- | --- |
| xr-controller common | SRS-2.2 | XRC-SDS-1 | XRC shall display the application version (git hash) using a launch argument |
| xr-controller common | SRS-2.2 | XRC-SDS-2 | XRC shall send logs to syslog |
| xr-controller common | SRS-2.2 | XRC-SDS-3 | XRC shall load configurable values from a JSON file |
| xr-controller common | SRS-2.2 | XRC-SDS-4 | XRC shall broadcast an application health signal |
| Camera flow | SRS-18.1 | XRC-SDS-5 | XRC shall use Framos cameras and software drivers |
| Camera flow | SRS-18.1 | XRC-SDS-6 | XRC shall interface with cameras using MIPI CSI-2 |
| Camera flow | SRS-18.1 | XRC-SDS-7 | XRC shall interface with Nvidia Argus or V4L2 to control cameras and capture images |
| Camera flow | SRS-18.1 | XRC-SDS-8 | XRC shall interface and manage up to three cameras using a common interface |
| Camera flow | SRS-18.1 | XRC-SDS-9 | XRC shall be able to control the resolution and bit depth of the camera |
| Camera flow | SRS-18.1 | XRC-SDS-10 | XRC shall be able to control camera frame rate and exposure |
| Camera flow | SRS-18.1 | XRC-SDS-11 | XRC shall be able to control the streaming state of the cameras |
| Camera flow | SRS-18.1 | XRC-SDS-12 | XRC shall use CUDA unified memory for camera image buffers |
| Camera flow | SRS-18.1 | XRC-SDS-13 | XRC shall load camera lens and distortion calibration data for each camera |
| Tracking system | SRS-14.3 | XRC-SDS-14 | XRC shall configure the tracking camera to operate at the same frame rate as the Cassette LED pattern update rate to 50Hz |
| Tracking system | SRS-14.3 | XRC-SDS-15 | XRC shall process the camera images to detect the Cassette tracking LEDs and decode the unique LED brightness patterns |
| Tracking system | SRS-14.3 | XRC-SDS-16 | XRC shall use statistical error correction to maintain LED pattern IDs |
| Tracking system | SRS-14.3 | XRC-SDS-17 | XRC shall match the unique LED IDs with known positions of the LEDs relative to the cassette detector image plane center |
| Tracking system | SRS-14.3 | XRC-SDS-18 | XRC shall use the solvePnP function in OpenCV to estimate the pose of the cassette relative to the emitter tracking camera |
| Tracking system | SRS-14.3 | XRC-SDS-19 | XRC shall broadcast raw tracking data using the signalling system |
| Tracking system | SRS-14.3 | XRC-SDS-20 | XRC shall broadcast tracking system status using the signalling system |
| Transform manager | SRS-16.1 | XRC-SDS-21 | XRC shall use quaternions to represent orientation |
| Transform manager | SRS-16.1 | XRC-SDS-22 | XRC shall create a representation of the emitter physical components as nodes of a graph |
| Transform manager | SRS-16.1 | XRC-SDS-23 | XRC shall define the relative positions of the emitter physical components in a common coordinate space |
| Transform manager | SRS-16.1 | XRC-SDS-24 | XRC shall use the graph model to change the reference frame of the raw tracking data to cassette space from tracking camera space |
| Transform manager | SRS-16.1 | XRC-SDS-25 | XRC shall use the graph model to calculate the pose of the focal spot relative to the cassette origin |
| Transform manager | SRS-16.1 | XRC-SDS-26 | XRC may use a kalman filter to remove noise from the focal spot pose data |
| Transform manager | SRS-16.1 | XRC-SDS-27 | XRC shall use the graph model to calculate the pose of the viewfinder camera relative to the cassette origin |
| Transform manager | SRS-16.1 | XRC-SDS-28 | XRC shall use the graph model to calculate the pose of the imaging camera relative to the cassette origin |
| Transform manager | SRS-16.1 | XRC-SDS-29 | XRC shall calculate SID using the focal spot pose |
| Transform manager | SRS-16.1 | XRC-SDS-30 | XRC shall calculate rvec and tvec values from the viewfinder camera pose |
| Transform manager | SRS-16.1 | XRC-SDS-31 | XRC shall calculate rvec and tvec values from the imaging camera pose |
| Transform manager | SRS-16.1 | XRC-SDS-32 | XRC shall broadcast SID using the signalling system |
| Transform manager | SRS-16.1 | XRC-SDS-33 | XRC shall broadcast viewfinder camera rvec anc tvec using the signalling system |
| Transform manager | SRS-16.1 | XRC-SDS-34 | XRC shall broadcast imaging camera rvec and tvec using the signalling system |
| CPA Manager | SRS-16.7 | XRC-SDS-35 | XRC shall use focal spot pose to calculate x-ray field area |
| CPA Manager | SRS-16.7 | XRC-SDS-36 | XRC shall generate vectors from the focal spot to evenly spaced points along the edge of a selected puck or collimator af a selected size |
| CPA Manager | SRS-16.7 | XRC-SDS-37 | XRC shall compute the intersection of each generated vector with the detector image plane |
| CPA Manager | SRS-16.7 | XRC-SDS-38 | XRC shall generate x-ray field areas using the computed intersection points as a contour |
| CPA Manager | SRS-16.7 | XRC-SDS-39 | XRC shall generate the x-ray field area at the detector image plane with at least 80% overlap of the actual x-ray field and with no more than a 2cm edge misalignment in the direction of greatest misalignment |
| CPA Manager | SRS-16.7 | XRC-SDS-40 | XRC shall calculate the maximum collimation size to limit the x-ray field to the detector active area |
| CPA Manager | SRS-16.7SRS-18.7SRS-18.8 | XRC-SDS-41 | XRC shall determine if the collimated field area is fully contained in the detector active area |
| CPA Manager | SRS-16.7 | XRC-SDS-42 | XRC shall calculate the collimator rotation angle to keep the collimated field square with the detector active area |
| CPA Manager | SRS-16.7SRS-18.7SRS-18.8 | XRC-SDS-43 | XRC shall broadcast the collimation size and rotation angle using the signalling system |
| CPA Manager | SRS-16.7SRS-18.7SRS-18.8 | XRC-SDS-44 | XRC shall allow for the selection of a number puck |
| CPA Manager | SRS-16.7 | XRC-SDS-45 | XRC shall allow for the selection of a fixed collimation size |
| CPA Manager | SRS-16.7SRS-18.7SRS-18.8 | XRC-SDS-46 | XRC shall allow for the selection of automated collimation |
| CPA Manager | SRS-18.7SRS-18.8 | XRC-SDS-47 | XRC shall apply a puck bounds overlay based on the largest puck collimated field |
| CPA Manager | SRS-16.7 | XRC-SDS-48 | XRC shall broadcast the x-ray field area contour using the signalling system |
| CPA Manager | SRS-16.7 | XRC-SDS-49 | XRC shall broadcast the x-ray field in bounds state using the signalling system |
| CPA Manager | SRS-16.7 | XRC-SDS-50 | XRC shall calculate the area in mm^2 of the x-ray field area |
| CPA Manager | SRS-16.7 | XRC-SDS-51 | XRC shall broadcast the field area using the signalling system |
| LIDAR Comm Manager | SRS-16.1 | XRC-SDS-52 | XRC shall connect to the lidar data serial port |
| LIDAR Comm Manager | SRS-16.1 | XRC-SDS-53 | XRC shall broadcast a signal to request lidar enable or disable |
| LIDAR Comm Manager | SRS-16.1 | XRC-SDS-54 | XRC shall receive lidar streaming data using the ICD packet structure |
| LIDAR Comm Manager | SRS-16.1 | XRC-SDS-55 | XRC shall calculate the SSD in mm using the center 2x2 block of each of the four lidar data sets |
| LIDAR Comm Manager | SRS-16.1 | XRC-SDS-56 | XRC shall broadcast the SSD using the signalling system |
| Viewfinder | SRS-16.7 | XRC-SDS-57 | XRC shall receive image from the viewfinder camera |
| Viewfinder | SRS-16.7 | XRC-SDS-58 | XRC shall process the viewfinder image to correct for fisheye distortion |
| Viewfinder | SRS-16.7 | XRC-SDS-59 | XRC shall change the perspective of the viewfinder camera image to be orthogonal to the cassette image plane |
| Viewfinder | SRS-16.7 | XRC-SDS-60 | XRC shall generate a video stream with the detector active area outline superimposed onto a stream from the viewfinder camera |
| Viewfinder | SRS-16.7 | XRC-SDS-61 | XRC shall generate a video stream with the collimated x-ray field area superimposed onto a stream from the viewfinder camera |
| Viewfinder | SRS-16.7 | XRC-SDS-62 | XRC shall generate a video stream with an indicator of emitter tilt superimposed onto a stream from the viewfinder camera |
| Viewfinder | SRS-16.7 | XRC-SDS-63 | XRC shall generate a video stream with an indicator emitter target superimposed onto a stream from the viewfinder camera |
| Viewfinder | SRS-16.7 | XRC-SDS-64 | XRC shall stream the generated image using interprocess shared memory |
| Viewfinder | SRS-16.7 | XRC-SDS-65 | XRC shall subscribe to photo_triggered_signal to trigger a photo capture |
| Viewfinder | SRS-16.7 | XRC-SDS-66 | XRC shall broadcast the photo_ready_signal signal containing captured photo jpeg data |
| IPC Proxy | SRS-16.8 | XRC-SDS-67 | XRC shall use interprocess shared memory to communicate with the emitter frontend |
| IPC Proxy | SRS-16.8 | XRC-SDS-68 | XRC shall use flatbuffer for serialization of interprocess messages |
| IPC Proxy | SRS-16.8 | XRC-SDS-69 | XRC shall receive and transmit flatbuffer messages |
| IPC Proxy | SRS-16.8 | XRC-SDS-70 | XRC shall transmit viewfinder video using CUDA  unified memory buffers registered with interprocess shared memory |
| IPC Proxy | SRS-16.8 | XRC-SDS-71 | XRC shall transmit the emitter battery level |
| IPC Proxy | SRS-16.8 | XRC-SDS-72 | XRC shall transmit the current technique kV |
| IPC Proxy | SRS-16.8 | XRC-SDS-73 | XRC shall transmit the current technique mAs |
| IPC Proxy | SRS-16.8 | XRC-SDS-74 | XRC shall transmit the system mode |
| IPC Proxy | SRS-16.8 | XRC-SDS-75 | XRC shall transmit the SID |
| IPC Proxy | SRS-16.8 | XRC-SDS-76 | XRC shall transmit the remaining DDR time |
| IPC Proxy | SRS-7.8SRS-23.5 | XRC-SDS-77 | XRC shall transmit the cassette connection state |
| IPC Proxy | SRS-16.8 | XRC-SDS-78 | XRC shall transmit the foot pedal connection state |
| IPC Proxy | SRS-16.8 | XRC-SDS-79 | XRC shall receive kV change requests |
| IPC Proxy | SRS-16.8 | XRC-SDS-80 | XRC shall receive mAs change requests |
| IPC Proxy | SRS-16.8 | XRC-SDS-81 | XRC shall receive a heartbeat |

### Table 9
| Category | Related SRS ID | SDS ID | Emitter Frontend (EMF) Design Specifications |
| --- | --- | --- | --- |
| Emitter frontend common | SRS-2.2 | EMF-SDS-1 | EMF shall display the application version (git hash) using a launch argument |
| Emitter frontend common | SRS-3.1 | EMF-SDS-2 | EMF shall send logs to syslog |
| Emitter frontend common | SRS-2.2 | EMF-SDS-3 | EMF shall load configurable values from a JSON file |
| GUI | SRS-16.1 | EMF-SDS-4 | EMF shall display a viewfinder image with active area and field overlays |
| GUI | SRS-16.8 | EMF-SDS-5 | EMF shall display the SID as a  sliding gauge |
| GUI | SRS-13.2 | EMF-SDS-6 | EMF shall display the current technique kV and mAs |
| GUI | SRS-19.11 | EMF-SDS-7 | EMF shall display the remaining serial radiographic or radioscopic time |
| GUI | SRS-7.9 | EMF-SDS-9 | EMF shall display the foot pedal connection state |
| GUI | SRS-25.3 | EMF-SDS-10 | EMF shall display the emitter battery state |
| GUI | SRS-11.3 | EMF-SDS-11 | EMF shall display the system mode |
| GUI | SRS-16.26 | EMF-SDS-14 | EMF shall display the computed SSD value |
| GUI | SRS-16.27 | EMF-SDS-15 | EMF shall set the SSD display green or red to indicate whether the SSD interlock is met or not |
| GUI | SRS-16.8 | EMF-SDS-32 | EMF shall set the SID gauge bubble to green or red to indicate whether the SID interlock is met or not |
| HMI | SRS-13.4 | EMF-SDS-16 | EMF shall respond to emitter HMI buttons to request kV and mAs changes |
| IPC proxy | SRS-16.1 | EMF-SDS-17 | EMF shall use interprocess shared memory to communicate with the emitter frontent |
| IPC proxy | SRS-16.1 | EMF-SDS-18 | EMF shall use flatbuffer for serialization of interprocess messages |
| IPC proxy | SRS-16.1 | EMF-SDS-19 | EMF shall receive and transmit flatbuffer messages |
| IPC proxy | SRS-16.1 | EMF-SDS-20 | EMF shall receive viewfinder video using CUDA  unified memory buffers registered with interprocess shared memory |
| IPC proxy | SRS-25.3 | EMF-SDS-21 | EMF shall receive the emitter battery level |
| IPC proxy | SRS-13.2 | EMF-SDS-22 | EMF shall receive the current technique kV |
| IPC proxy | SRS-13.2 | EMF-SDS-23 | EMF shall receive the current technique mAs |
| IPC proxy | SRS-11.3 | EMF-SDS-24 | EMF shall receive the system mode |
| IPC proxy | SRS-16.8 | EMF-SDS-25 | EMF shall receive the SID |
| IPC proxy | SRS-19.11 | EMF-SDS-26 | EMF shall receive the remaining DDR time |
| IPC proxy | SRS-7.8 | EMF-SDS-27 | EMF shall receive the cassette connection state |
| IPC proxy | SRS-7.9 | EMF-SDS-28 | EMF shall receive the foot pedal connection state |
| IPC proxy | SRS-13.4 | EMF-SDS-29 | EMF shall transmit kV change requests |
| IPC proxy | SRS-13.5 | EMF-SDS-30 | EMF shall transmit mAs change requests |
| IPC proxy | SRS-16.1 | EMF-SDS-31 | EMF shall transmit a heartbeat |

### Table 10
| Category | Related SRS ID | SDS ID | Idle Manager (IM) Design Specifications |
| --- | --- | --- | --- |
| Signal Subscriptions | SRS-10.1 | IM-SDS-1 | IM shall subscribe to device_idle_signal |
| Signal Subscriptions | SRS-10.1 | IM-SDS-2 | IM shall subscribe to rest_idle_signal |
| Signal Subscriptions | SRS-10.1 | IM-SDS-3 | IM shall subscribe to manual_idle_signal |
| Signal Broadcasters | SRS-10.1 | IM-SDS-4 | IM shall broad the device idle state via the idle_state_signal |
| Idle Condition | SRS-10.1 | IM-SDS-5 | IM shall broadcast the received state from rest_idle_signal if an engineering mode command has not been received and the idle state has changed |
| Idle Condition | SRS-10.1 | IM-SDS-6 | IM shall solely obey engineering mode idle state commands if they are enabled and a engineering mode command is sent |
| Category | Related SRS ID | SDS ID | WiFi Stability (WS) Design Specifications |
| Network Configuration | SRS-7.2 | WS-SDS-1 | WS shall use the SSID specified in /opt/medai/data/config/device-specific/device-specific-config.json |
| Network Reconnection | SRS-7.2 | WS-SDS-2 | WS shall check wifi connection status every 2 seconds |
| Network Reconnection | SRS-7.2 | WS-SDS-3 | WS shall attempt to reconnect to the cassette's wifi if there is no valid connection present |
| Network Reconnection | SRS-7.2 | WS-SDS-4 | WS shall sleep for 15 seconds before checking the network status again |
| Category | Related SRS ID | SDS ID | Cassette Display Driver (CDD) Design Specifications |
| Signal Subscriptions | SRS-25.5 | CDD-SDS-1 | CDD shall subscribe to cassette_battery_info signal |
| Signal Subscriptions | SRS-10.6 | CDD-SDS-2 | CDD shall subscribe to idle_state_signal |
| Signal Subscriptions | SRS-10.6 | CDD-SDS-3 | CDD shall subscribe to system_ready_signal |
| Signal Subscriptions | SRS-21.11 | CDD-SDS-4 | CDD shall subscribe to safe_state_signal |
| Battery Indications | SRS-25.7 | CDD-SDS-5 | CDD shall display no bars when the battery level is at or below 10% |
| Battery Indications | SRS-25.7 | CDD-SDS-6 | CDD shall display a single bar when the battery percentage is in the range (10%, 30%] |
| Battery Indications | SRS-25.7 | CDD-SDS-7 | CDD shall display two bars when the battery percentage is in the range (30%, 50%] |
| Battery Indications | SRS-25.7 | CDD-SDS-8 | CDD shall display three bars when the battery percentage is in the range (50%, 70%] |
| Battery Indications | SRS-25.7 | CDD-SDS-9 | CDD shall display four bars when the battery percentage is in the range (70%, 90%] |
| Battery Indications | SRS-25.7 | CDD-SDS-10 | CDD shall display five bars when the battery percentage is in the range (90%, 100%] |
| Battery Indications | SRS-25.6 | CDD-SDS-11 | CDD shall display the battery percentage as an integer above the battery indicator |
| Battery Indications | SRS-26.4 | CDD-SDS-12 | CDD shall display a lightning bolt within the battery indicator when the cassette is charging |
| Device State | SRS-10.6 | CDD-SDS-12 | CDD shall display "Idle" when true is received from idle_state_signal |
| Device State | SRS-10.6 | CDD-SDS-13 | CDD shall display "Ready" when true is received from system_ready_signal |
| Device State | SRS-10.6 | CDD-SDS-14 | CDD shall display "Not Ready" when true is received from system_ready_signal |
| Device State | SRS-21.11 | CDD-SDS-15 | CDD shall display "Error" when true is received from safe_state_signal |
| Device State | SRS-25.8 | CDD-SDS-16 | CDD shall display "Low Battery!" when the cassette battery percentage is 20% or less |
| Category | Related SRS ID | SDS ID | Image Reaper (IR) Design Specifications |
| Image Culling | SRS-28.6 | IR-SDS-1 | IR shall scan the disk for image captures on a configurable time period |
| Image Culling | SRS-28.6 | IR-SDS-2 | IR shall remove captures on scheduled scans to satisfy the configurable disk quota |
| Image Culling | SRS-28.6 | IR-SDS-3 | IR shall retain a configurable number of captures on scheduled scans |
| Image Culling | SRS-28.6 | IR-SDS-4 | IR shall notify CP of captures removed during scheduled scans |
| Image Culling | SRS-28.6 | IR-SDS-5 | IR shall remove captures on demand via the REST API |
| Category | Related SRS ID | SDS ID | Connectivity Controller (CC) Design Specifications |
| Connectivity Controller | SRS-34.1SRS-34.2SRS-34.3SRS-34.4SRS-34.5SRS-34.6 | CC-SDS-1 | CC shall query Network Manager for a list of available wireless networks |
| Connectivity Controller | SRS-34.1SRS-34.2SRS-34.3SRS-34.4SRS-34.5SRS-34.6 | CC-SDS-2 | CC shall relay available wireless networks to CP |

### Table 11
| Category | Related SRS ID | SDS ID | Emitter Firmware (EM) Design Specifications |
| --- | --- | --- | --- |
| Interfaces | SRS-6.1SRS-14.7 | EM-SDS-1 | EM shall serially connect to EO via MLVDS, 115200 baud, 8 bit frame, no parity, 1 stop bit. |
| Interfaces | SRS-6.1SRS-14.7 | EM-SDS-2 | EM shall serially communicate with EO using ICD protocol packets. |
| Interfaces | SRS-6.1 | EM-SDS-3 | EM shall provide R/W registers for control and status. |
| Interfaces | SRS-6.1 | EM-SDS-4 | EM shall provide a register to identify as Device Type 3. |
| Interfaces | SRS-6.1 | EM-SDS-5 | EM shall accept incoming ICD protocol packets that match Device Type 3. |
| Interfaces | SRS-6.1 | EM-SDS-6 | EM shall encode ICD protocol packet replies with Device Type 3. |
| Interfaces | SRS-6.1 | EM-SDS-7 | EM shall receive detector reset request signals from MB via LVDS. |
| Interfaces | SRS-6.1 | EM-SDS-8 | EM shall transmit detector reset requests to CAS via Sub-GHz radio. |
| Interfaces | SRS-6.1 | EM-SDS-9 | EM shall receive pedal/button events from FP via Sub-GHz radio. |
| Events | SRS-6.1 | EM-SDS-10 | EM shall provide an event mask register to enable/disable events. |
| Events | SRS-6.1 | EM-SDS-11 | EM shall provide a register to read event flags. |
| Events | SRS-6.1SRS-14.7 | EM-SDS-12 | EM shall signal event flag changes to EO via interrupt. |
| Events | SRS-10.1SRS-10.12 | EM-SDS-13 | EM shall set an event flag if idle for > 5 minutes. |
| Events | SRS-23.5SRS-7.9 | EM-SDS-14 | EM shall set an event flag if FP is not paired. |
| Events | SRS-23.5SRS-7.9 | EM-SDS-15 | EM shall set an event flag if FP Sub-GHz radio is detected. |
| Events | SRS-10.20SRS-10.21 | EM-SDS-16 | EM shall set an event flag if FP event is received. |
| Events | SRS-22.2SRS-21.1 | EM-SDS-17 | EM shall set an event flag if humidity is out-of-range. |
| Events | SRS-22.1SRS-21.1 | EM-SDS-18 | EM shall set an event flag if temperature is out-of-range. |
| Events | SRS-14.21SRS-26.3 | EM-SDS-19 | EM shall set an event flag if battery is wired charging or wireless charging. |
| Events | SRS-22.3SRS-21.1 | EM-SDS-20 | EM shall set an event flag if supply or regulator voltages are out-of-range. |
| Firmware version | SRS-6.1 | EM-SDS-21 | EM shall provide a register to read the firmware version. |
| Mode Indicator RGB LEDs | SRS-4.4 | EM-SDS-22 | EM shall initialize the RGB LED array to blue upon startup. |
| Mode Indicator RGB LEDs | SRS-5.3SRS-10.11SRS-16.10SRS-16.11SRS-16.14SRS-16.15SRS-21.5SRS-23.3 | EM-SDS-23 | EM shall provide a mode indicator register to control the RGB LED array. |
| Mode Indicator RGB LEDs | SRS-5.3SRS-10.11SRS-16.10SRS-16.11SRS-16.14SRS-16.15SRS-21.5SRS-23.3 | EM-SDS-24 | EM shall provide for setting the RGB LED array color. |
| Mode Indicator RGB LEDs | SRS-5.3SRS-10.11SRS-16.10SRS-16.11SRS-16.14SRS-16.15SRS-21.5SRS-23.3 | EM-SDS-25 | EM shall provide for enabling/disabling the RGB LED array. |
| Mode Indicator RGB LEDs | SRS-5.3 | EM-SDS-26 | EM shall provide for blinking the RGB LED array. |
| Mode Indicator RGB LEDs | SRS-10.11 | EM-SDS-27 | EM shall provide for fading the RGB LED array. |
| Buzzer | SRS-19.12SRS-19.13SRS-19.15 | EM-SDS-28 | EM shall provide a register to control the buzzer. |
| Buzzer | SRS-19.12SRS-19.13SRS-19.15 | EM-SDS-29 | EM shall provide for setting the buzzer duration in milliseconds. |
| Buzzer | SRS-19.12SRS-19.13SRS-19.15 | EM-SDS-30 | EM shall provide for setting the buzzer frequency in kilohertz. |
| Accelerometer | SRS-10.12 | EM-SDS-31 | EM shall provide a register to control the accelerometer. |
| Accelerometer | SRS-10.12 | EM-SDS-32 | EM shall provide for disabling the accelerometer. |
| Accelerometer | SRS-10.12 | EM-SDS-33 | EM shall provide for enabling the accelerometer. |
| Accelerometer | SRS-10.12SRS-10.16 | EM-SDS-34 | EM shall provide for resetting the accelerometer idle timer. |
| Foot Pedal | SRS-7.7SRS-7.9 | EM-SDS-35 | EM shall provide a register to read foot pedal/button status. |
| BMS Battery | SRS-14.19SRS-25.3 | EM-SDS-36 | EM shall provide a register to read the battery charge percentage. |
| Humidity | SRS-22.2 | EM-SDS-37 | EM shall provide a register to read the humidity percentage. |
| Humidity | SRS-22.2 | EM-SDS-38 | EM shall set an event flag if the emitter internal humidity is higher than 90%. |
| Temperature | SRS-22.1 | EM-SDS-39 | EM shall provide a register to read the temperature in degrees C. |
| Temperature | SRS-22.1 | EM-SDS-40 | EM shall set an event flag if the monoblock sensor reports a temperature of 70 degrees C or higher. |
| Temperature | SRS-22.1 | EM-SDS-41 | EM shall set an event flag if the emitter handle temperature sensor reports a value of 48 degrees C or higher. |
| Temperature | SRS-22.1 | EM-SDS-42 | EM shall set an event flag if the emitter PCB temperature sensor reports a value of 75 degrees C or higher. |
| Temperature | SRS-22.1 | EM-SDS-43 | EM shall set an event flag if the emitter PMUX temperature sensor reports a value of 100 degrees C or higher. |
| Power control | SRS-4.1SRS-5.3SRS-21.7SRS-21.6 | EM-SDS-44 | EM shall provide a register to control powered peripherals. |
| Power control | SRS-21.7SRS-10.8SRS-10.9SRS-10.22 | EM-SDS-45 | EM shall provide for enabling/disabling COL power. |
| Power control | SRS-21.6SRS-10.22 | EM-SDS-46 | EM shall provide for enabling/disabling MB power. |
| Power control | SRS-4.1SRS-5.3SRS-10.22 | EM-SDS-47 | EM shall provide for enabling/disabling EM power. |
| Voltage monitors | SRS-22.3 | EM-SDS-48 | EM shall provide registers to monitor analog voltages. |
| Voltage monitors | SRS-22.3 | EM-SDS-49 | EM shall set an event flag if the BAT_V_MON voltage rail reading is outside of the [18 V, 39 V] range. |
| Voltage monitors | SRS-22.3 | EM-SDS-50 | EM shall set an event flag if the DISPLAY_V_MON voltage rail reading is outside of the [18 V, 39 V] range. |
| Voltage monitors | SRS-22.3 | EM-SDS-51 | EM shall set an event flag if the COL_V_MON voltage rail reading is outside of the [18 V, 39 V] range. |
| Voltage monitors | SRS-22.3 | EM-SDS-52 | EM shall set an event flag if the 5V0 voltage rail reading is outside of the [4.8 V, 5.25 V] range. |
| Voltage monitors | SRS-22.3 | EM-SDS-53 | EM shall set an event flag if the LED_5V0 voltage rail reading is outside of the [4.8 V, 5.2 V] range. |
| System shutdown | SRS-5.1 | EM-SDS-54 | EM shall provide a register to shut down the system. |
| MCU UID | SRS-6.1SRS-14.7 | EM-SDS-55 | EM shall provide a register to read the MCU Unique ID. |
| Watchdog | SRS-7.11 | EM-SDS-56 | EM shall initialize the independent watchdog timer to 2 seconds. |
| Watchdog | SRS-7.11 | EM-SDS-57 | EM shall pet the independent watchdog every superloop cycle. |
| Wireless Charging | SRS-27.1SRS-27.2SRS-27.3 | EM-SDS-58 | EM shall provide a register to allow EO to enable and disable charging |
| Firmware upgrade | SRS-45.1 | EM-SDS-59 | EM shall support one boot jump image. |
| Firmware upgrade | SRS-45.1 | EM-SDS-60 | EM shall support two upgrade capable application images. |
| Firmware upgrade | SRS-45.1 | EM-SDS-61 | EM shall support two application information NVM sectors. |
| Firmware upgrade | SRS-45.1 | EM-SDS-62 | EM shall support one logging NVM sector. |
| Firmware upgrade | SRS-45.1 | EM-SDS-63 | EM shall store 'run next' requests in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | EM-SDS-64 | EM shall track 'jump' attempts in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | EM-SDS-65 | EM shall provide runtime context. |
| Firmware upgrade | SRS-45.1 | EM-SDS-66 | EM shall support runtime context switching. |
| Firmware upgrade | SRS-45.1 | EM-SDS-67 | EM shall support upgrade requests. |
| Firmware upgrade | SRS-45.1 | EM-SDS-68 | EM shall provide upgrade debug LED indication. |
| Firmware upgrade | SRS-45.1 | EM-SDS-69 | EM shall erase upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | EM-SDS-70 | EM shall provide upgrade flash sector erase status. |
| Firmware upgrade | SRS-45.1 | EM-SDS-71 | EM shall support upgrade section size variations. |
| Firmware upgrade | SRS-45.1 | EM-SDS-72 | EM shall enforce upgrade section size limits. |
| Firmware upgrade | SRS-45.1 | EM-SDS-73 | EM shall support upgrade section address checking. |
| Firmware upgrade | SRS-45.1 | EM-SDS-74 | EM shall enforce upgrade section timeouts. |
| Firmware upgrade | SRS-45.1 | EM-SDS-75 | EM shall support upgrade section flash programming. |
| Firmware upgrade | SRS-45.1 | EM-SDS-76 | EM shall support upgrade flash programming error checking. |
| Firmware upgrade | SRS-45.1 | EM-SDS-77 | EM shall support upgrade flashed application CRC32 checking. |
| Firmware upgrade | SRS-45.1 | EM-SDS-78 | EM shall provide upgrade completion status. |
| Firmware upgrade | SRS-45.1 | EM-SDS-79 | EM shall provide upgrade error status. |
| Firmware upgrade | SRS-45.1 | EM-SDS-80 | EM shall re-lock upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | EM-SDS-81 | EM shall support soft restart. |
| Firmware upgrade | SRS-45.1 | EM-SDS-82 | EM shall reinitialize upon restarts. |
| Firmware upgrade | SRS-45.1 | EM-SDS-83 | EM shall switch applications for failed jump attempts. |

### Table 12
| Category | Related SRS ID | SDS ID | Monoblock Firmware (MB) Design Specifications |
| --- | --- | --- | --- |
| Interfaces | SRS-6.1SRS-14.7 | MB-SDS-1 | MB shall serially connect to EO via MLVDS, 115200 baud, 8 bit frame, no parity, 1 stop bit. |
| Interfaces | SRS-6.1SRS-14.7 | MB-SDS-2 | MB shall serially communicate with EO using ICD protocol packets. |
| Interfaces | SRS-6.1 | MB-SDS-3 | MB shall provide R/W registers for control and status. |
| Interfaces | SRS-6.1 | MB-SDS-4 | MB shall provide a register to identify as Device Type 1. |
| Interfaces | SRS-6.1 | MB-SDS-5 | MB shall accept incoming ICD protocol packets that match Device Type 1. |
| Interfaces | SRS-6.1 | MB-SDS-6 | MB shall encode ICD protocol packet replies with Device Type 1. |
| Interfaces | SRS-6.1 | MB-SDS-7 | MB shall signal detector reset requests to EM via LVDS. |
| Events | SRS-21.1SRS-6.1SRS-14.7 | MB-SDS-8 | MB shall provide an event mask register to enable/disable events. |
| Events | SRS-21.1SRS-6.1SRS-14.7 | MB-SDS-9 | MB shall provide a register to read event flags. |
| Events | SRS-21.1 | MB-SDS-10 | MB shall signal event flag changes to EO via LVDS interrupt. |
| Events | SRS-22.12 | MB-SDS-11 | MB shall set an event flag if the exposure time exceeds 220ms. |
| Events | SRS-22.7 | MB-SDS-12 | MB shall set an event flag if tube voltage averaged during X-ray emission is outside of set point [40 - 80kV, 1kV steps, +/- 8%] |
| Events | SRS-22.8 | MB-SDS-13 | MB shall set an event flag if tube voltage is non-zero when X-ray emission is not intentional. |
| Events | SRS-22.9 | MB-SDS-14 | MB shall set an event flag if beam current averaged during X-ray emission is outside of set point [0.5 - 2.0mA, 0.1mA steps, +/-20%] |
| Events | SRS-22.10 | MB-SDS-15 | MB shall set an event flag if beam current is non-zero when X-ray emission is not intentional. |
| Events | SRS-22.11 | MB-SDS-16 | MB shall set an event flag if temperature is out-of-range. |
| Events | SRS-22.13 | MB-SDS-17 | MB shall set an event flag if supply or regulator voltages are out-of-range. |
| Events | SRS-20.9 | MB-SDS-18 | MB shall set an event flag if PWS voltage is out-of-range. |
| Firmware version | SRS-6.1 | MB-SDS-19 | MB shall provide a register to read the firmware version. |
| X-ray control |  | MB-SDS-20 | MB shall provide a register to set X-ray technique mode. [single, series] |
| X-ray control | SRS-12.1SRS-12.2SRS-12.7 | MB-SDS-21 | MB shall provide a register to set X-ray technique tube voltage set point. [40 - 80kV, 1kV steps] |
| X-ray control | SRS-12.3 | MB-SDS-22 | MB shall provide a register to set X-ray technique tube current set point. [0.5 - 2.0mA, 0.1mA steps] |
| X-ray control | SRS-12.3 | MB-SDS-23 | MB shall provide a register to set X-ray technique exposure time. [single 30ms - 200ms, series 10ms - 50ms, 1ms steps] |
| X-ray control | SRS-19.4 | MB-SDS-24 | MB shall limit repeated series X-ray exposures to 20s. |
| X-ray control | SRS-13.1SRS-19.1SRS-19.2SRS-19.3 | MB-SDS-25 | MB shall provide a register to start/stop X-ray acquisition. |
| X-ray control | SRS-22.7 | MB-SDS-26 | MB shall provide a register to read tube voltage averaged during X-ray acquisition. |
| X-ray control | SRS-22.9 | MB-SDS-27 | MB shall provide a register to read tube current averaged during X-ray acquisition. |
| Detector control | SRS-19.2 | MB-SDS-28 | MB shall signal detector reset requests to EM 30ms before X-ray acquisition. |
| Temperature | SRS-22.11 | MB-SDS-29 | MB shall provide a register to read the temperature in degrees C. |
| Temperature | SRS-22.11 | MB-SDS-30 | MB shall set an event flag if the monoblock temperature sensor on inductor reports a temperature of 80 degrees C or higher |
| Temperature | SRS-22.11 | MB-SDS-31 | MB shall set an event flag if the monoblock temperature sensor on monoblock side reports a temperature of 80 degrees C or higher |
| Voltage monitors | SRS-22.13 | MB-SDS-32 | MB shall provide registers to monitor analog voltages. |
| Voltage monitors | SRS-22.13 | MB-SDS-33 | MB shall set an event flag if the BAT_V_MON voltage rail reading is outside of the [18V, 40V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-34 | MB shall set an event flag if the 3V0 voltage rail reading is outside of the [2.6V, 3.4V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-35 | MB shall set an event flag if the 3V3 voltage rail reading is outside of the [3V, 3.5V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-36 | MB shall set an event flag if the 3V3_ANA voltage rail reading is outside of the [3.1V, 3.5V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-37 | MB shall set an event flag if the 5V0 voltage rail reading is outside of the [4.7V, 5.3V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-38 | MB shall set an event flag if the M15V0 voltage rail reading is outside of the [-15.15V, -14.5V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-39 | MB shall set an event flag if the P15V0 voltage rail reading is outside of the [14.5V, 15.15V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-40 | MB shall set an event flag if the 15V0_FIL voltage rail reading is outside of the [14.5V, 15.5V] range |
| Voltage monitors | SRS-22.13 | MB-SDS-41 | MB shall set an event flag if the XRAY_PWS voltage rail reading is outside of the [9V, 86V] range |
| MCU UID | SRS-6.1SRS-6.4 | MB-SDS-42 | MB shall provide a register to read the MCU Unique ID. |
| Watchdog | SRS-7.11 | MB-SDS-43 | MB shall initialize the independent watchdog timer to 2 seconds. |
| Watchdog | SRS-7.11 | MB-SDS-44 | MB shall pet the independent watchdog every superloop cycle. |
| Firmware upgrade | SRS-45.1 | MB-SDS-45 | MB shall support one boot jump image. |
| Firmware upgrade | SRS-45.1 | MB-SDS-46 | MB shall support two upgrade capable application images. |
| Firmware upgrade | SRS-45.1 | MB-SDS-47 | MB shall support two application information NVM sectors. |
| Firmware upgrade | SRS-45.1 | MB-SDS-48 | MB shall support one logging NVM sector. |
| Firmware upgrade | SRS-45.1 | MB-SDS-49 | MB shall store 'run next' requests in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | MB-SDS-50 | MB shall track 'jump' attempts in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | MB-SDS-51 | MB shall provide runtime context. |
| Firmware upgrade | SRS-45.1 | MB-SDS-52 | MB shall support runtime context switching. |
| Firmware upgrade | SRS-45.1 | MB-SDS-53 | MB shall support upgrade requests. |
| Firmware upgrade | SRS-45.1 | MB-SDS-54 | MB shall provide upgrade debug LED indication. |
| Firmware upgrade | SRS-45.1 | MB-SDS-55 | MB shall unlock upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | MB-SDS-56 | MB shall erase upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | MB-SDS-57 | MB shall provide upgrade flash sector erase status. |
| Firmware upgrade | SRS-45.1 | MB-SDS-58 | MB shall support upgrade section size variations. |
| Firmware upgrade | SRS-45.1 | MB-SDS-59 | MB shall enforce upgrade section size limits. |
| Firmware upgrade | SRS-45.1 | MB-SDS-60 | MB shall support upgrade section address checking. |
| Firmware upgrade | SRS-45.1 | MB-SDS-61 | MB shall enforce upgrade section timeouts. |
| Firmware upgrade | SRS-45.1 | MB-SDS-62 | MB shall support upgrade section flash programming. |
| Firmware upgrade | SRS-45.1 | MB-SDS-63 | MB shall support upgrade flash programming error checking. |
| Firmware upgrade | SRS-45.1 | MB-SDS-64 | MB shall support upgrade flashed application CRC32 checking. |
| Firmware upgrade | SRS-45.1 | MB-SDS-65 | MB shall provide upgrade completion status. |
| Firmware upgrade | SRS-45.1 | MB-SDS-66 | MB shall provide upgrade error status. |
| Firmware upgrade | SRS-45.1 | MB-SDS-67 | MB shall re-lock upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | MB-SDS-68 | MB shall support soft restart. |
| Firmware upgrade | SRS-45.1 | MB-SDS-69 | MB shall reinitialize upon restarts. |
| Firmware upgrade | SRS-45.1 | MB-SDS-70 | MB shall switch applications for failed jump attempts. |

### Table 13
| Category | Related SRS ID | SDS ID | Collimator Firmware (COL) Design Specifications |
| --- | --- | --- | --- |
| Interfaces | SRS-6.1SRS-14.7 | COL-SDS-1 | COL shall serially connect to EO via MLVDS, 115200 baud, 8 bit frame, no parity, 1 stop bit. |
| Interfaces | SRS-6.1SRS-14.7 | COL-SDS-2 | COL shall serially communicate with EO using ICD protocol packets. |
| Interfaces | SRS-6.1 | COL-SDS-3 | COL shall provide R/W registers for control and status. |
| Interfaces | SRS-6.1 | COL-SDS-4 | COL shall provide a register to identify as Device Type 4. |
| Interfaces | SRS-6.1 | COL-SDS-5 | COL shall accept incoming ICD protocol packets that match Device Type 4. |
| Interfaces | SRS-6.1 | COL-SDS-6 | COL shall encode ICD protocol packet replies with Device Type 4. |
| Interfaces | SRS-6.1SRS-16.20 | COL-SDS-7 | COL shall serially connect to XRC via LVDS, 115200 baud, 8 bit frame, no parity, 1 stop bit. |
| Interfaces | SRS-6.1SRS-16.20 | COL-SDS-8 | COL shall serially transmit ToF data to XRC using ICD protocol packets. |
| Interfaces | SRS-6.1SRS-16.20 | COL-SDS-9 | COL shall encode ToF ICD protocol packets with Device Type 4. |
| Events | SRS-21.1SRS-6.1SRS-14.7 | COL-SDS-10 | COL shall provide an event mask register to enable/disable events. |
| Events | SRS-21.1SRS-6.1SRS-14.7 | COL-SDS-11 | COL shall provide a register to read event flags. |
| Events | SRS-22.4SRS-22.5 | COL-SDS-12 | COL shall signal event flag changes to EO via LVDS interrupt. |
| Events | SRS-16.20 | COL-SDS-13 | COL shall set an event flag if Time of Flight is not calibrated. |
| Events | SRS-22.4 | COL-SDS-14 | COL shall set an event flag if homing has not been completed. |
| Events | SRS-22.4 | COL-SDS-15 | COL shall set an event flag if homing or aperture set timed out after 5s. |
| Events | SRS-22.5 | COL-SDS-16 | COL shall set an event flag if supply or regulator voltages are out-of-range. |
| Firmware version | SRS-6.1SRS-14.7 | COL-SDS-17 | COL shall provide a register to read the firmware version. |
| Lasers | SRS-10.8SRS-10.22SRS-16.16SRS-16.17SRS-16.18 | COL-SDS-18 | COL shall provide a register to control the lasers. |
| Lasers | SRS-10.8SRS-10.22SRS-16.16SRS-16.17SRS-16.18SRS-21.7 | COL-SDS-19 | COL shall provide for enabling/disabling the lasers. |
| Lasers | SRS-16.17 | COL-SDS-20 | COL shall provide for blinking the lasers. |
| Time of Flight | SRS-16.20 | COL-SDS-21 | COL shall provide a register to control ToF. |
| Time of Flight | SRS-16.20 | COL-SDS-22 | COL shall provide for calibrating ToF. |
| Time of Flight | SRS-16.20 | COL-SDS-23 | COL shall provide for enabling/disabling ToF. |
| Time of Flight | SRS-16.20 | COL-SDS-24 | COL shall provide for setting ToF stream rate 1-15Hz. |
| Time of Flight | SRS-16.20 | COL-SDS-25 | COL shall provide ranging data for 4 ToF sensors. |
| Time of Flight | SRS-16.20 | COL-SDS-26 | COL shall format ranging data for each ToF sensor as an 8x8 zone. |
| Time of Flight | SRS-16.20 | COL-SDS-27 | COL shall report ranging data from 0-4000mm. |
| Collimation | SRS-14.4 | COL-SDS-28 | COL shall provide a register to home the aperture. |
| Collimation | SRS-14.4 | COL-SDS-29 | COL shall provide for homing orientation offsets. |
| Collimation | SRS-14.4 | COL-SDS-30 | COL shall provide for reading homing status. |
| Collimation | SRS-14.4 | COL-SDS-31 | COL shall clear an event flag if homing completes. |
| Collimation | SRS-14.4 | COL-SDS-32 | COL shall set an event flag if homing timed out after 5s. |
| Collimation | SRS-14.4 | COL-SDS-33 | COL shall provide a register for setting aperture orientation/size [0-359 degrees, 5.5-18.4mm] |
| Collimation | SRS-14.4 | COL-SDS-34 | COL shall maintain the requested aperture orientation within +/- 0.5 degrees. |
| Collimation | SRS-14.4 | COL-SDS-35 | COL shall maintain the requested aperture size within +/- 0.25mm. |
| Collimation | SRS-14.4 | COL-SDS-36 | COL shall provide for reading aperture set status. |
| Collimation | SRS-14.4 | COL-SDS-37 | COL shall set an event flag if aperture set timed out after 5s. |
| Voltage monitors | SRS-22.5 | COL-SDS-38 | COL shall provide registers to monitor analog voltages. |
| Voltage monitors | SRS-22.5 | COL-SDS-39 | COL shall set an event flag if the VMOTOR_24V voltage rail reading is outside of the [19 V, 29 V] range |
| Voltage monitors | SRS-22.5 | COL-SDS-40 | COL shall set an event flag if the 5V0 voltage rail reading is outside of the [4.5 V, 5.5 V] range |
| MCU UID | SRS-6.1SRS-16.20 | COL-SDS-41 | COL shall provide a register to read the MCU Unique ID. |
| Watchdog | SRS-7.11 | COL-SDS-42 | COL shall initialize the independent watchdog timer to 2 seconds. |
| Watchdog | SRS-7.11 | COL-SDS-43 | COL shall pet the independent watchdog every superloop cycle. |
| Firmware upgrade | SRS-45.1 | COL-SDS-44 | COL shall support one boot jump image. |
| Firmware upgrade | SRS-45.1 | COL-SDS-45 | COL shall support two upgrade capable application images. |
| Firmware upgrade | SRS-45.1 | COL-SDS-46 | COL shall support two application information NVM sectors. |
| Firmware upgrade | SRS-45.1 | COL-SDS-47 | COL shall support one logging NVM sector. |
| Firmware upgrade | SRS-45.1 | COL-SDS-48 | COL shall store 'run next' requests in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | COL-SDS-49 | COL shall track 'jump' attempts in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | COL-SDS-50 | COL shall provide runtime context. |
| Firmware upgrade | SRS-45.1 | COL-SDS-51 | COL shall support runtime context switching. |
| Firmware upgrade | SRS-45.1 | COL-SDS-52 | COL shall support upgrade requests. |
| Firmware upgrade | SRS-45.1 | COL-SDS-53 | COL shall provide upgrade debug LED indication. |
| Firmware upgrade | SRS-45.1 | COL-SDS-54 | COL shall erase upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | COL-SDS-55 | COL shall provide upgrade flash sector erase status. |
| Firmware upgrade | SRS-45.1 | COL-SDS-56 | COL shall support upgrade section size variations. |
| Firmware upgrade | SRS-45.1 | COL-SDS-57 | COL shall enforce upgrade section size limits. |
| Firmware upgrade | SRS-45.1 | COL-SDS-58 | COL shall support upgrade section address checking. |
| Firmware upgrade | SRS-45.1 | COL-SDS-59 | COL shall enforce upgrade section timeouts. |
| Firmware upgrade | SRS-45.1 | COL-SDS-60 | COL shall support upgrade section flash programming. |
| Firmware upgrade | SRS-45.1 | COL-SDS-61 | COL shall support upgrade flash programming error checking. |
| Firmware upgrade | SRS-45.1 | COL-SDS-62 | COL shall support upgrade flashed application CRC32 checking. |
| Firmware upgrade | SRS-45.1 | COL-SDS-63 | COL shall provide upgrade completion status. |
| Firmware upgrade | SRS-45.1 | COL-SDS-64 | COL shall provide upgrade error status. |
| Firmware upgrade | SRS-45.1 | COL-SDS-65 | COL shall support soft restart. |
| Firmware upgrade | SRS-45.1 | COL-SDS-66 | COL shall reinitialize upon restarts. |
| Firmware upgrade | SRS-45.1 | COL-SDS-67 | COL shall switch applications for failed jump attempts. |

### Table 14
| Category | Related SRS ID | SDS ID | Cassette Firmware (CAS) Design Specifications |
| --- | --- | --- | --- |
| Interfaces | SRS-6.1SRS-14.8 | CAS-SDS-1 | CAS shall serially connect to EO via MLVDS, 115200 baud, 8 bit frame, no parity, 1 stop bit. |
| Interfaces | SRS-6.1SRS-14.8 | CAS-SDS-2 | CAS shall serially communicate with EO using ICD protocol packets. |
| Interfaces | SRS-6.1 | CAS-SDS-3 | CAS shall provide R/W registers for control and status. |
| Interfaces | SRS-6.1 | CAS-SDS-4 | CAS shall provide a register to identify as Device Type 5. |
| Interfaces | SRS-6.1 | CAS-SDS-5 | CAS shall accept incoming ICD protocol packets that match Device Type 5. |
| Interfaces | SRS-6.1 | CAS-SDS-6 | CAS shall encode ICD protocol packet replies with Device Type 5. |
| Interfaces | SRS-6.1 | CAS-SDS-7 | CAS shall receive Detector Reset Requests from EM via Sub-GHz radio. |
| Interfaces | SRS-10.3 | CAS-SDS-8 | CAS shall sleep/wake detector via control line XRAY_ON. |
| Interfaces | SRS-10.3 | CAS-SDS-9 | CAS shall reset detector sensor memory via control line SYNC. |
| Interfaces | SRS-10.3 | CAS-SDS-10 | CAS shall sense detector acquiring via control line EXPO. |
| Events | SRS-6.1 | CAS-SDS-11 | CAS shall provide an event mask register to enable/disable events. |
| Events | SRS-6.1 | CAS-SDS-12 | CAS shall provide an event flags register to read events. |
| Events | SRS-6.1SRS-14.8 | CAS-SDS-13 | CAS shall signal event flag changes to CO via LVDS interrupt. |
| Events | SRS-5.2 | CAS-SDS-14 | CAS shall set an event flag if the power off button is pressed. |
| Events | SRS-23.5SRS-7.8SRS-10.15 | CAS-SDS-15 | CAS shall set an event flag if CAS is not paired. |
| Events | SRS-22.15 | CAS-SDS-16 | CAS shall set an event flag if humidity is out-of-range. |
| Events | SRS-22.14 | CAS-SDS-17 | CAS shall set an event flag if temperature is out-of-range. |
| Events | SRS-14.20SRS-26.2SRS-26.4 | CAS-SDS-18 | CAS shall set an event flag if battery is wired charging. |
| Events | SRS-22.16 | CAS-SDS-19 | CAS shall set an event flag if voltages are out-of-range. |
| Firmware version | SRS-6.1 | CAS-SDS-20 | CAS shall provide a register to read the firmware version. |
| Mode Indicator RGB LEDs | SRS-4.5 | CAS-SDS-21 | CAS shall initialize the RGB LED array to blue. |
| Mode Indicator RGB LEDs | SRS-5.4SRS-10.4SRS-16.12SRS-16.13SRS-16.14SRS-16.15SRS-21.2SRS-23.3 | CAS-SDS-22 | CAS shall provide a mode indicator register to control the RGB LED array. |
| Mode Indicator RGB LEDs | SRS-5.4SRS-10.4SRS-16.12SRS-16.13SRS-16.14SRS-16.15SRS-21.2SRS-23.3 | CAS-SDS-23 | CAS shall provide for setting the RGB LED array color. |
| Mode Indicator RGB LEDs | SRS-5.4SRS-10.4SRS-16.12SRS-16.13SRS-16.14SRS-16.15SRS-21.2SRS-23.3SRS-23.4 | CAS-SDS-24 | CAS shall provide for enabling/disabling the RGB LED array. |
| Mode Indicator RGB LEDs | SRS-5.4 | CAS-SDS-25 | CAS shall provide for blinking the RGB LED array. |
| Mode Indicator RGB LEDs | SRS-10.4 | CAS-SDS-26 | CAS shall provide for fading the RGB LED array. |
| Tracking IR LEDs | SRS-10.5SRS-21.4 | CAS-SDS-27 | CAS shall provide a register for enabling/disable the IR LEDs. |
| Tracking IR LEDs | SRS-16.2SRS-16.3SRS-16.8SRS-16.10SRS-16.11SRS-16.12SRS-16.13SRS-16.21 | CAS-SDS-28 | CAS shall project light patterns that uniquely identify each IR LED. |
| Buzzer | SRS-19.14 | CAS-SDS-29 | CAS shall provide a register to control the buzzer. |
| Buzzer | SRS-19.14 | CAS-SDS-30 | CAS shall provide for setting the buzzer duration in milliseconds. |
| Buzzer | SRS-19.14 | CAS-SDS-31 | CAS shall provide for setting the buzzer frequency in kilohertz. |
| Sub-GHz radio | SRS-7.6 | CAS-SDS-32 | CAS shall provide a register to set the detector radio address. |
| BMS Battery | SRS-25.5SRS-26.4 | CAS-SDS-33 | CAS shall provide a register to read the battery charge percentage. |
| Humidity | SRS-22.15 | CAS-SDS-34 | CAS shall provide a register to read the humidity percentage. |
|  | SRS-22.15 | CAS-SDS-35 | CAS shall set an event flag if the cassette internal humidity is higher than 90%. |
| Temperature | SRS-22.14 | CAS-SDS-36 | CAS shall provide a register to read the temperature in degrees C. |
|  | SRS-22.14 | CAS-SDS-37 | CAS shall set an event flag if the cassette PCB temperature sensor 1 reports a temperature of 91 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-38 | CAS shall set an event flag if the cassette PCB temperature sensor 2 reports a temperature of 54 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-39 | CAS shall set an event flag if the cassette PCB temperature sensor 3 reports a temperature of 51 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-40 | CAS shall set an event flag if the cassette PCB temperature sensor 4 reports a temperature of 99 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-41 | CAS shall set an event flag if the cassette PCB temperature sensor 5 reports a temperature of 70 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-42 | CAS shall set an event flag if the cassette PCB temperature sensor 6 reports a temperature of 61 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-43 | CAS shall set an event flag if the cassette PCB temperature sensor 7 reports a temperature of 79 degrees C or higher |
|  | SRS-22.14 | CAS-SDS-44 | CAS shall set an event flag if the cassette PCB temperature sensor 8 reports a temperature of 60 degrees C or higher |
|  |  | CAS-SDS-45 | CAS shall set an event flag if the cassette PCB temperature sensor 9 reports a temperature of 65 degrees C or higher |
| Detector sleep | SRS-10.3 | CAS-SDS-46 | CAS shall provide a register to sleep/wake the detector. |
| Power control | SRS-10.3 | CAS-SDS-47 | CAS shall provide a register to control powered peripherals. |
| Power control | SRS-10.3 | CAS-SDS-48 | CAS shall provide for enabling/disabling DET power. |
|  | SRS-5.2 | CAS-SDS-49 | CAS shall provide for enabling/disabling CAS power. |
| Voltage monitors | SRS-22.16 | CAS-SDS-50 | CAS shall provide registers to monitor analog voltages. |
| Voltage monitors | SRS-22.16 | CAS-SDS-51 | CAS shall set an event flag if the CM_PWR voltage rail reading is outside of the [9.8 V, 21 V] range. |
| Voltage monitors | SRS-22.16 | CAS-SDS-52 | CAS shall set an event flag if the MCU_5V0 voltage rail reading is outside of the [4.78 V, 5.15 V] range. |
| Voltage monitors | SRS-22.16 | CAS-SDS-53 | CAS shall set an event flag if the JET_3V3 voltage rail reading is outside of the [3.15 V, 3.4 V] range. |
| Voltage monitors | SRS-22.16 | CAS-SDS-54 | CAS shall set an event flag if the DET_22V voltage rail reading is outside of the [19 V, 23 V] range. |
| Voltage monitors | SRS-22.16 | CAS-SDS-55 | CAS shall set an event flag if the JET_5V0 voltage rail reading is outside of the [4.8 V, 5.15 V] range. |
| System shutdown | SRS-5.2 | CAS-SDS-56 | CAS shall provide a register to shut down the system. |
| MCU UID | SRS-6.1 | CAS-SDS-57 | CAS shall provide a register to read the MCU Unique ID. |
| Watchdog | SRS-7.11 | CAS-SDS-58 | CAS shall initialize the independent watchdog timer to 2 seconds. |
| Watchdog | SRS-7.11 | CAS-SDS-59 | CAS shall pet the independent watchdog every superloop cycle. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-60 | CAS shall support one boot jump image. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-61 | CAS shall support two upgrade capable application images. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-62 | CAS shall support two application information NVM sectors. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-63 | CAS shall support one logging NVM sector. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-64 | CAS shall store 'run next' requests in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-65 | CAS shall track 'jump' attempts in the logging NVM sector. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-66 | CAS shall provide runtime context. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-67 | CAS shall support runtime context switching. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-68 | CAS shall support upgrade requests. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-69 | CAS shall provide upgrade debug LED indication. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-70 | CAS shall erase upgrade flash sectors. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-71 | CAS shall provide upgrade flash sector erase status. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-72 | CAS shall support upgrade section size variations. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-73 | CAS shall enforce upgrade section size limits. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-74 | CAS shall support upgrade section address checking. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-75 | CAS shall enforce upgrade section timeouts. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-76 | CAS shall support upgrade section flash programming. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-77 | CAS shall support upgrade flash programming error checking. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-78 | CAS shall support upgrade flashed application CRC32 checking. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-79 | CAS shall provide upgrade completion status. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-80 | CAS shall provide upgrade error status. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-81 | CAS shall support soft restart. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-82 | CAS shall reinitialize upon restarts. |
| Firmware upgrade | SRS-45.1 | CAS-SDS-83 | CAS shall switch applications for failed jump attempts. |

### Table 15
| Category | Related SRS ID | SDS ID | Foot Pedal Firmware (FP) Design Specifications |
| --- | --- | --- | --- |
| Interfaces | SRS-7.7SRS-7.9 | FP-SDS-1 | FP shall connect to EM via Sub-GHz radio. |
| Interfaces | SRS-6.1SRS-7.7 | FP-SDS-3 | FP shall encode radio packet replies with MCU Unique ID. |
| Events | SRS-24.1 | FP-SDS-4 | FP shall transmit an asynchronous event packet when any pedal/button is pressed/released. |
| Events | SRS-24.3SRS-24.8 | FP-SDS-5 | FP shall set/clear an event bit when the left pedal is pressed/released. |
| Events | SRS-24.2SRS-24.4SRS-24.5 | FP-SDS-6 | FP shall set/clear an event bit when the right pedal is pressed/released. |
| Events | SRS-24.3 | FP-SDS-7 | FP shall set/clear an event bit when the left button is pressed/released. |
| Events | SRS-24.7 | FP-SDS-8 | FP shall set/clear an event bit when the right button is pressed/released. |
| Buttons/Pedals | SRS-24.1SRS-10.20SRS-10.21 | FP-SDS-9 | FP shall detect when any pedal/button is pressed/released. |
| Radios | SRS-24.1 | FP-SDS-10 | FP shall transmit a radio heartbeat packet once per second to notify EM of connectivity. |
| Radios | SRS-24.1 | FP-SDS-11 | FP shall inhibit wireless transmission for battery voltage <= 2.6V. |
| LEDs | SRS-24.9 | FP-SDS-12 | FP shall set the battery LED green for battery voltage > 2.9V. |
| LEDs | SRS-24.9 | FP-SDS-13 | FP shall blink the battery LED red for battery voltage <= 2.9V. |
| LEDs | SRS-24.10 | FP-SDS-14 | FP shall blink the radio LED orange when actively transmitting non-heartbeat packets. |
| Watchdog | SRS-7.11 | FP-SDS-15 | FP shall initialize the independent watchdog timer to 2 seconds. |
| Watchdog | SRS-7.11 | FP-SDS-16 | FP shall pet the independent watchdog every superloop cycle. |
| Radios | SRS-45.19 | FP-SDS-17 | FP shall enable forward error correction (FEC) on the foot pedal radios upon initialization |
| Radios | SRS-45.19 | FP-SDS-18 | FP shall enable CRC32 capabilites on the foot pedal radios upon initialization |

### Table 16
| Category | Related SRS ID | SDS ID | Capture-presenter (CP) Design Specifications |
| --- | --- | --- | --- |
| Interface | SRS-14.18SRS-30.5 | CP-SDS-1 | CP shall provide Rest API over HTTP |
| Interface | SRS-30.5 | CP-SDS-2 | CP shall provide WebSocket connection for full-duplex communication with ODA |
| Interface | SRS-14.18 | CP-SDS-3 | CP shall utilise JSON as data transmission format for the API |
| API | SRS-44.3 | CP-SDS-4 | CP shall provide endpoint for CO to post metadata and filename of newly acquired Xrays |
| API | SRS-39.1SRS-39.2SRS-39.3SRS-39.4SRS-39.5SRS-39.6SRS-39.7 | CP-SDS-5 | CP shall provide endpoint for CO to post system notifications to be displayed in the UI |
| API | SRS-18.2SRS-18.3SRS-18.6 | CP-SDS-6 | CP shall provide endpoint for CO to post current settings for Collimation system |
| API | SRS-42.16SRS-44.7 | CP-SDS-7 | CP shall provide endpoint for ODA to manipulate studies state (Start study, Complete study, Close study) |
| API | SRS-44.3 | CP-SDS-8 | CP shall provide endpoint for ODA to pull Xrays information for current study |
| API | SRS-44.3 | CP-SDS-9 | CP shall provide endpoint for ODA to pull Xrays information for all studies |
| API | SRS-31.3 | CP-SDS-10 | CP shall provide endpoint for ODA to initiate export of the Xrays to PACS server |
| API | SRS-44.6 | CP-SDS-11 | CP shall provide endpoint for ODA to initiate export of the Xrays to selected USB drive |
| API | SRS-43.11SRS-43.12SRS-43.13 | CP-SDS-12 | CP shall provide endpoint for ODA to adjust Brightness, Contrast and Sharpness of a concrete Xray |
| API | SRS-42.8 | CP-SDS-13 | CP shall provide endpoint for ODA to pull Modality Worklist from selected RIS server |
| API | SRS-31.3 | CP-SDS-14 | CP shall provide endpoint for ODA to manipulate PACS and RIS server list (add,edit.delete) |
| API | SRS-33.2 | CP-SDS-15 | CP shall provide endpoint for ODA to manipulate Doctors list (add,edit,delete) |
| API | SRS-18.2SRS-18.3SRS-18.6 | CP-SDS-16 | CP shall provide endpoint for ODA to post updates for Collimation sytem settings |
| API | SRS-34.1SRS-34.2SRS-38.1SRS-38.2 | CP-SDS-17 | CP shall provide endpoint for ODA to pull current networks settings from the cassette |
| API | SRS-34.1SRS-34.2SRS-38.1SRS-38.2 | CP-SDS-18 | CP shall provide endpoint for ODA to upate network settings for the cassette |
| API | SRS-41.1 | CP-SDS-19 | CP shall provide endpoint for ODA to pull system information from the cassette |
| API | SRS-31.3 | CP-SDS-20 | CP shall provide endpoint for ODA to manipulate parameters for export to DICOM files |
| API | SRS-5.5 | CP-SDS-21 | CP shall provide endpoint for ODA to initiate cassette shutdown |
| Database | SRS-44.3 | CP-SDS-22 | CP shall store Xrays metadata and image names in persistent manner |
| Database | SRS-44.3 | CP-SDS-23 | CP shall store information about Xray image adjustments in persistent manner |
| Database | SRS-44.5 | CP-SDS-24 | CP shall store Studies information is persistent manner |
| Database | SRS-31.3 | CP-SDS-25 | CP shall store configured PACS and RIS servers information in persistent manner |
| Database | SRS-33.2 | CP-SDS-26 | CP shall store Doctors list in persistent manner |
| Database | SRS-32.1 | CP-SDS-27 | CP shall store set parameters for export to DICOM files in persistent manner |
| Database | SRS-44.3SRS-44.5SRS-31.3SRS-33.2SRS-32.1 | CP-SDS-28 | CP shall maintain data integrity across device restarts |
| Database | SRS-44.3SRS-44.5SRS-31.3SRS-33.2SRS-32.1 | CP-SDS-29 | CP shall maintain data consistency across multiple simultaneous connections |
| Logging | SRS-3.1 | CP-SDS-30 | CP shall publish decent amount of log records allowing to trace major issues within the application |
| Logging | SRS-3.1 | CP-SDS-31 | CP shall publish log records into system log |
| Logging | SRS-3.1 | CP-SDS-32 | CP shall provide tags when logging records from different modules of the application |
| Operation System integration | SRS-38.1SRS-38.2 | CP-SDS-33 | CP shall call system scripts to read current network settings of the cassette |
| Operation System integration | SRS-38.1SRS-38.2 | CP-SDS-34 | CP shall call system scripts to set network settings for the cassette |
| Operation System integration | SRS-44.6 | CP-SDS-35 | CP shall call LSBLK system utility to get information about connected USB drives |
| Operation System integration | SRS-34.1SRS-34.2SRS-38.1SRS-38.2 | CP-SDS-36 | CP shall call NMCLI system utility to get information about available Wi-Fi networks |
| DICOM | SRS-44.3 | CP-SDS-37 | CP shall provide module to convert data for Studies and Xrays from inner format to DICOM format |
| DICOM | SRS-44.3 | CP-SDS-38 | CP shall utilize DCM4CHE library to implement DICOM module |
| DICOM | SRS-31.4 | CP-SDS-39 | CP shall be able to ping PACS and RIS servers using ECHO command |
| DICOM | SRS-38.3 | CP-SDS-40 | CP shall periodically check PACS and RIS servers availability |
| DICOM | SRS-44.3SRS-44.4 | CP-SDS-41 | CP shall be able to send exported DICOM files to selected PACS server |
| DICOM | SRS-44.6 | CP-SDS-42 | CP shall be able to save exported DICOM files to selected USB drive |
| DICOM | SRS-44.3SRS-44.6 | CP-SDS-49 | CP shall be able to generate DICOM files using both JPEG and JPEG2000 image formats |
| DICOM | SRS-44.3SRS-44.6 | CP-SDS-50 | CP shall be able to choose between JPEG and JPEG2000 formats basing on what corresponding PACS server supports |
| DICOM | SRS-42.8 | CP-SDS-43 | CP shall be able to get modality worklist data from selected RIS server |
| DICOM | SRS-42.8 | CP-SDS-44 | CP shall be able to fill exported DICOM file with data loaded from RIS server |
| DICOM | SRS-44.3SRS-44.5SRS-44.6 | CP-SDS-45 | CP shall implement DICOM export in non-blocking manner (background worker) |
| Security | SRS-44.5 | CP-SDS-46 | CP shall encode PHI and other sensitive data with cryptographic algorithm (AES256) |
| Security | SRS-44.6 | CP-SDS-47 | CP shall protect persisted data with password |
| Security | SRS-44.7 | CP-SDS-48 | CP shall generate unique password per unit |

### Table 17
| Category | Related SRS ID | SDS ID | MedAI Device App (ODA) Design Specifications |
| --- | --- | --- | --- |
| Interface | SRS-7.2 | ODA-SDS-1 | ODA shall be able to lookup HTTP and WebSocket connections at IP 10.24.96.1 when tablet connected to cassette's AP |
| Interface | SRS-7.2SRS-18.2SRS-18.3SRS-30.5 | ODA-SDS-2 | ODA shall be able to communicate with Capture Presenter service over HTTP on port 8080 |
| Interface | SRS-43.2SRS-43.3SRS-43.4SRS-39.1SRS-39.2SRS-39.3SRS-39.5SRS-39.6SRS-39.7 | ODA-SDS-3 | ODA shall be able to listen to incoming messages from Cassette Orchestrator service over WebSocket on port 8082 |
| Interface | SRS-16.20SRS-16.21SRS-24.7SRS-24.8 | ODA-SDS-4 | ODA shall be able to listen to incoming messages from Cassette Orchestrator service over WebSocket on port 8083 |
| Interface | SRS-7.2SRS-18.2SRS-18.3SRS-30.5 | ODA-SDS-5 | ODA shall be able to listen to incoming messages from Capture Presenter service over WebSocket on port 8080 |
| Interface | SRS-43.11SRS-43.12SRS-43.13SRS-43.18 | ODA-SDS-6 | ODA shall be able to sent to outgoing image processing requests to Cassette Orchestrator service over HTTP on port 8081 |
| Interface | SRS-18.2SRS-18.3SRS-18.6 | ODA-SDS-7 | ODA shall be able to sent to outgoing puck/collimation selection messages to Cassette Orchestrator service over HTTP on port 8081 |
| Interface | SRS-7.2SRS-18.2SRS-18.3SRS-30.5 | ODA-SDS-8 | ODA shall utilize Rest API and JSON data format when communicating with Capture Presenter service |
| Landing page | SRS-42.11 | ODA-SDS-9 | ODA shall implement form to input new Study details, such as: 1. Patient name, birthdate2. Doctor name3. Study description4. Body part examined5. Side of anatomy |
| Landing page | SRS-42.8 | ODA-SDS-10 | ODA shall implement controls to load and select RIS server to get worklist items from |
| Landing page | SRS-42.8SRS-42.9 | ODA-SDS-11 | ODA shall implement worklist controls including: filtering, reload, selection |
| Landing page | SRS-42.14 | ODA-SDS-12 | ODA shall implement controls to clear study form, and to start new study |
| Landing page | SRS-32.1SRS-42.11 | ODA-SDS-13 | ODA shall implement validation of input fields in accordance with current DICOM config |
| Acquisition page | SRS-43.1 | ODA-SDS-14 | ODA shall display last taken Xray and make it most notable and clearly visible (main display area) |
| Acquisition page | SRS-43.1 | ODA-SDS-15 | ODA shall display previously taken Xray in manner allowing side-to-side comparison with last Xray (secondary display area) |
| Acquisition page | SRS-43.10 | ODA-SDS-16 | ODA shall display a list of all Xrays taken during current exam (roll area) |
| Acquisition page | SRS-43.1SRS-43.10 | ODA-SDS-17 | ODA shall allow selection of Xray from the list to be displayed in main display area |
| Acquisition page | SRS-43.5SRS-43.6 | ODA-SDS-18 | ODA shall display Xray acquisitions params for Xray in main and secondary display areas including:1. Tube voltage in kV2. Exposure time product in mAs3. Dosage in mGy4. Dose Area Product in mGy*cm^25. Acquisition date and time |
| Acquisition page | SRS-43.1SRS-43.2SRS-43.3SRS-43.21SRS-43.22SRS-43.23SRS-43.24 | ODA-SDS-19 | ODA shall be able to display streaming frames of serial Xray capture (DDR) in main display area |
| Acquisition page | SRS-43.4SRS-43.21SRS-43.22SRS-43.23SRS-43.24 | ODA-SDS-20 | ODA shall allow to playback serial Xray captures (DDR) in main display area |
| Acquisition page | SRS-43.4 | ODA-SDS-21 | ODA shall implement controls to control DDR playback, like: play/stop, prev/next frame |
| Acquisition page | SRS-43.15SRS-43.16 | ODA-SDS-22 | ODA shall implement allow to zoom and pan Xray in main display area |
| Acquisition page | SRS-43.17 | ODA-SDS-23 | ODA shall implement controls to rotate single and serial Xrays |
| Acquisition page | SRS-43.11SRS-43.12SRS-43.13 | ODA-SDS-24 | ODA shall implement controls to adjust contrast, sharpness and brightness of single Xray |
| Acquisition page | SRS-43.18 | ODA-SDS-25 | ODA shall implement controls to revert any adjustments made to Xray image |
| Acquisition page | SRS-42.16 | ODA-SDS-26 | ODA shall implement button to cancel current exam and return to landing page |
| Acquisition page | SRS-44.1 | ODA-SDS-27 | ODA shall implement button to complete current exam and redirect to Library page |
| Acquisition page | SRS-16.8SRS-16.20 | ODA-SDS-28 | ODA shall display SID and SSD values streamed from cassette |
| Library page | SRS-44.1 | ODA-SDS-29 | ODA shall display list of all Xrays taken by unit grouped by study |
| Library page | SRS-44.1 | ODA-SDS-30 | ODA shall sort grouped Xrays in chronological order |
| Library page | SRS-44.2 | ODA-SDS-31 | ODA shall allow to select one or several Xray within a study |
| Library page | SRS-44.3 | ODA-SDS-32 | ODA shall provide dialog with form allowing to initialize export of a selected Xrays to selected PACS server |
| Library page | SRS-44.6 | ODA-SDS-33 | ODA shall provide dialog with form allowing to initialize export of a selected Xrays to selected USB drive |
| Library page | SRS-44.7 | ODA-SDS-34 | ODA shall provide "Complete exam" dialog which allows to review study info and taken xrays if there is an active study |
| Library page | SRS-43.9 | ODA-SDS-35 | ODA shall implement carousel like display view to one-by-one review of selected Xrays (Xray view area) |
| Library page | SRS-43.4 | ODA-SDS-36 | ODA shall allow to playback serial Xray captures (DDR) in Xray view area |
| Library page | SRS-43.4 | ODA-SDS-37 | ODA shall implement controls to control DDR playback in Xray view area, like: play/stop, prev/next frame |
| Library page | SRS-43.15SRS-43.16 | ODA-SDS-38 | ODA shall implement allow to zoom and pan Xray in  Xray view area |
| Library page | SRS-43.17 | ODA-SDS-39 | ODA shall implement controls to rotate single and serial Xrays |
| Library page | SRS-43.11SRS-43.12SRS-43.13 | ODA-SDS-40 | ODA shall implement controls to adjust contrast, sharpness and brightness of single Xray |
| Library page | SRS-43.18 | ODA-SDS-41 | ODA shall implement controls to revert any adjustments made to Xray image |
| Library page | SRS-44.3SRS-44.6 | ODA-SDS-42 | ODA shall display checkmark above Xrays which have been already exported to PACS or USB drive |
| Library page | SRS-39.13 | ODA-SDS-81 | ODA shall allow to playback radioscopic captures in Xray view area |
| Library page | SRS-39.13 | ODA-SDS-82 | ODA shall default to pasuing and displaying the last frame of radioscopic captures after acquisition |
| DICOM Servers page | SRS-44.4 | ODA-SDS-43 | ODA shall display list of configured PACS and RIS servers |
| DICOM Servers page | SRS-31.3 | ODA-SDS-44 | ODA shall implement controls to add, update and delete DICOM server |
| DICOM Servers page | SRS-31.1 | ODA-SDS-45 | ODA shall provide dialog with form to enter information about new DICOM server or to update information about existing DICOM server |
| DICOM Servers page | SRS-31.1 | ODA-SDS-46 | ODA shall have following fields at add/edit DICOM server form:1. Role (PACS, RIS)2. Name3. Location4. IP or hostname5. Port number6. Application entity title |
| DICOM Servers page | SRS-31.3 | ODA-SDS-47 | ODA shall validate inputs of add/edit DICOM server form so that all fields are required and have proper formatting |
| DICOM Servers page | SRS-31.4 | ODA-SDS-48 | ODA shall implement button to test connection with selected DICOM server |
| Doctors page | SRS-33.1SRS-33.2 | ODA-SDS-49 | ODA shall display llist of doctors registered on a unit |
| Doctors page | SRS-33.1SRS-33.2 | ODA-SDS-50 | ODA shall implement controls to add, update and delete doctor |
| Doctors page | SRS-33.1SRS-33.2 | ODA-SDS-51 | ODA shall provide dialog with form to enter information about new doctor or to update infromation about existing doctor |
| Doctors page | SRS-33.1SRS-33.2 | ODA-SDS-52 | ODA shall have following fields at add/edit doctor form:1. Identifier2. First name3. Last name4. Email5. Phone Number |
| Doctors page | SRS-33.1SRS-33.2 | ODA-SDS-53 | ODA shall validate inputs for add/edit doctor form in accordance with current DICOM config |
| Network Settings page | SRS-38.1SRS-38.2 | ODA-SDS-54 | ODA shall display current state of network connection at the cassette |
| Network Settings page | SRS-38.1SRS-38.2 | ODA-SDS-55 | ODA shall implement form to enter or update network settings of the cassette |
| Network Settings page | SRS-38.1SRS-38.2 | ODA-SDS-56 | ODA shall provide selector of avilable Wi-Fi networks |
| Network Settings page | SRS-38.1SRS-38.2 | ODA-SDS-57 | ODA shall provide buttons to Save or Revert updated network settings |
| DICOM Fields page | SRS-32.1 | ODA-SDS-58 | ODA shall display list of supported DICOM tags |
| DICOM Fields page | SRS-42.13 | ODA-SDS-59 | ODA shall display list of supported body parts |
| DICOM Fields page | SRS-32.1 | ODA-SDS-60 | ODA shall implement controls to mark DICOM tags as mandatory/optional for the export |
| DICOM Fields page | SRS-42.13 | ODA-SDS-61 | ODA shall implement controls to mark body parts visible/hidden at Landing form |
| Collimation page | SRS-18.2SRS-18.3SRS-18.6 | ODA-SDS-62 | ODA shall implement controls to switch between automatic, puck and manual collimation |
| Collimation page | SRS-18.2 | ODA-SDS-63 | ODA shall implement controls to select aperture size when in manual collimation mode |
| Collimation page | SRS-18.3 | ODA-SDS-64 | ODA shall implement controls to select puck number when in puck collimation mode |
| Menu | SRS-38.1SRS-38.2 | ODA-SDS-81 | ODA shall implement menu to navigate through app pages |
| Menu | SRS-38.1SRS-38.2 | ODA-SDS-82 | ODA shall implement panel for quick access to network connection status and active DICOM servers info |
| Menu | SRS-41.1 | ODA-SDS-65 | ODA shall implement panel with information about connected cassette. Information may include:1. Serial number2. Software version3. Date of manufacture4. Application entity title |
| Menu | SRS-39.1 | ODA-SDS-66 | ODA shall implement panel to display list of recent notifications received from the cassette |
| Menu | SRS-40.1 | ODA-SDS-67 | ODA shall implement panel to display legal information including company name and address and UDI |
| General | SRS-30.5 | ODA-SDS-68 | ODA shall display connection error message when connection with CP or CO can't be established |
| General | SRS-30.5 | ODA-SDS-69 | ODA shall not allow operate Xray lists or controls when connection with cassette services can't be established |
| General | SRS-30.5SRS-30.7SRS-30.8SRS-30.9 | ODA-SDS-70 | ODA shall implement button to re-try cassette connection |
| General | SRS-30.5SRS-30.7SRS-30.8SRS-30.9 | ODA-SDS-71 | ODA shall timeout the re-try connection attempt after 20 seconds |
| General | SRS-39.2 | ODA-SDS-72 | ODA shall display popup messages when info or warning notifications are being received form the cassette |
| General | SRS-21.10 | ODA-SDS-73 | ODA shall display error dialog when fatal error notification is being received from the cassette |
| General | SRS-5.5 | ODA-SDS-74 | ODA shall display confirmation modal when shutdown button is tapped |
| General | SRS-5.5 | ODA-SDS-75 | ODA shall display confirmation dialog box indicating successful shutdown |
| DICOM Device Type | SRS-37.1SRS-37.5 | ODA-SDS-76 | ODA shall implement controls to select display device type |
| DICOM Luminance Calibration Curve | SRS-37.3 | ODA-SDS-77 | When device type is set to "Samsung Galaxy Tab S8+", ODA shall apply a luminance calibration curve to radiographic images before display |
| DICOM Luminance Calibration Curve | SRS-37.3 | ODA-SDS-78 | ODA shall set the pixel brightness for each image as defined in gamma.dart when applying the calibration curve |
| Screen Brightness | SRS-37.2 | ODA-SDS-79 | When device type is set to "Samsung Galaxy Tab S8+", ODA shall set the tablet's brightness to the maximum value |
| Warning Banner | SRS-37.4 | ODA-SDS-80 | When device type is set to "Other", ODA shall set a persistent warning banner at the top of the screen |

### Table 18
| Related SRS ID | SDS ID | Category | SOUP Design Specifications | Verification method | Implemented? | Finalized? | Testable? | Document Reference? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | OpenCV - TS | The TS shall use the OpenCV threshold function to create a binary image from tracking camera output | Only used to convert image to binary from 8bit. May not need to be tested. | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV findContours function to identify the tracking LEDs in the tracking camera binary image |  | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV moments function to compute the position and size of the tracking LED contrours | Tested with data from findContours. Image in, check if data matches expected | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV solvePnP function to compute the x-ray tube focal spot location using the tracking LED positions | Data points in, check if output tvec and rotation is as expected | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV Rodrigues function to create a rotation matrix from the solvePnP outputs | May not need testing. Data in, data out | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV RQDecomp3x3 function to compute euler angles from the tracking position rotation matrix | May not need testing. Data in, data out | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV KalmanFilter implementation to reduce measurement noise in the position tracking outputs | Create data, add noise, run through filter, check if data points are within acceptable range | False | False | False |  |
|  |  | OpenCV - TS | The TS shall use the OpenCV fitEllipse function to compute the x-ray field on the detector image plane | Create ellipse, get points long it, run points though function, check against original ellipse | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV imread function to open radiographic images from disk | Open file, check raw image data or check last accessed time in OS | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV resize function to reduce the size of the radiographic image for histogram calculations | May not need testing. Check image size before and after | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV calcHist function to calculate a histogram distribution of the radiographic image | Make image from histogram? | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV convertTo function to change the bit depth of the radiographic image to enable saving as a jpeg | Check type before and after | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV imwrite function to save the processed radiographic images to disk | Save test image, check with OS that file exists | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV VideoWriter.open function to create serial radiography files on disk |  | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV VideoWriter.write function to push image frames to the serial radiogray files | Run through video output steps, check that a file exists | False | False | False |  |
|  |  | OpenCV - IP | IP shall use the OpenCV VideoWriter.release function to render the serial radiography files |  | False | False | False |  |
|  |  | Exiv2 - IP | IP shall use the Exiv2 readMetadata function to load existing meta data from an image |  |  |  |  |  |
|  |  | Exiv2 - IP | P Shall use the Exiv2 ImageFactory::open function to load images for metadata editing |  |  |  |  |  |
|  |  | Exiv2 - IP | P Shall use the Exiv2 xmpData function to parse the metadata XMP packet |  |  |  |  |  |
|  |  | Exiv2 - IP | P Shall use the Exiv2 setXmpData function apply the metadata XMP packet to an image |  |  |  |  |  |
|  |  | Exiv2 - IP | P Shall use the Exiv2 writeMetadata function to save metadata to an image |  |  |  |  |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_InitCamera function to initialize a connection to the tracking camera | Call function, check camera status | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_PixelClock function to set the internal clock speed on the tracking camera | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_SetAutoParameter function to adjust various image capture properties of the tracking camera | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_SetSensorScaler function to set the tracking camera image sensor scaling property | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_AOI function to set the tracking camera image sensor area of interest | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_LUT function to set the tracking camera internal lookup table values | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_Exposure function to set the tracking camera image exposure time | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_SetGainBoost function to set the tracking camera gain boost property | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_SetHardwareGain function to set the tracking camera hardware gain property | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_IO function to enable and read the tracking camera GPIO pins | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_DeviceFeature function to set the tracking camera shutter mode | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_SetExternalTrigger function to set the tracking camera trigger mode | Set, read back | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_EnableEvent function to enable the tracking camera image ready event |  | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_CaptureVideo function to start the tracking camera video capture |  | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_GetImageMem function to read image data from the tracking camera |  | False | False | False |  |
|  |  | IDS | The TS shall use the IDS Software Suite is_WaitEvent function to listen for image ready events |  | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK http_listener.support function to configure the MC HTTP server | Create web server and try to reach it? | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK http_listener.open function to start the MC HTTP server |  | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK http_request.reply(http_response) function to reply to HTTP requests from CP |  | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK http_request.extract_json function to find JSON data in HTTP requests |  | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK http_client.request function to create and send HTTP requests to CP | Send request to self | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK web::json::value::parse function to parse JSON data structures from files or streams | Read sample JSON file and extract values | False | False | False |  |
|  |  | CPP Rest SDK | MC shall use the CPP Rest SDK getJsonValue function to store or create JSON data |  | False | False | False |  |
|  |  | Boost Log | Too many functions not testable independently; see MEMO-P00-030 - Imager SOUP Descriptions for further description. |  | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function _InitSocketAPI() to initialize the API | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevGetLibraryConfigOptions to retrieve default options. | Run sample program | False | False | False |  |
|  |  |  | The gigev_access submodule shall use the teledyne library function GevSetLibraryConfigOptions to set the log level option to normal. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevGetCameraList to get access to the connected detector | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevOpenCamera to activate the detector | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevGetGenICamXML_FileName to retrieve the xml configuration file for the detector | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevGetCameraInterfaceOptions to retrieve the default detector options | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevSetCameraInterfaceOptions to set performance options for the detector. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevGetFeatureNodeMap to retrieve default features of the detector | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevSetFeatureValueAsString to set the trigger mode to snapshot, and to toggle standby mode. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevInitImageTransfer to initialize memory fo rimage transfers. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevStartImageTransfer to start image transfers. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevAbortTransfer to abort image transfers. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevFreeTransfer to clean up transfers memory when cleaning up and shutting down. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevWaitForNextImage to await the arrival of the next image when radiographic images are generated. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevCloseCamera to close the camera when cleaning up and shutting down. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function GevApiUninitialize to close the API when cleaning up and shutting down. | Run sample program | False | False | False |  |
|  |  | Teledyne | The gigev_access submodule shall use the teledyne library function _CloseSocketAPI to close the socket API when cleaning up and shutting down. | Run sample program | False | False | False |  |
