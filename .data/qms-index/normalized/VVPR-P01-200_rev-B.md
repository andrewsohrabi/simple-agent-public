# VVPR-P01-200 Rev B: MX1 Software System System Configuration v3.2.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-200
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.2.0
- Source filename: VVPR-P01-200 - MX1 Software System System Configuration v3.2.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-200 - MX1 Software System System Configuration v3.2.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following features:
Emitter Jetson HDMI Register Setting
Release and Debug modes
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.2.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. D
IFU-MX1 - Instructions for Use, Rev. D
MATERIALS
E1 Emitter BOM Rev. H
C1 Cassette BOM Rev. I
F1 Foot Pedal BOM Rev. B
M50133 Rev. A, Galaxy Tablet  S8+
Mouse
Keyboard
External monitor
USB-C hub and ethernet adapter
MX1 Software System v3.2.0
APP MedAI Device App
Additional tools/equipment:
EQP-139 (or equivalent) Control Company Stopwatch 4YMT7
In the report section, fill in the following table for equipment used during this study:
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Follow the steps outlined below. The MX1 Instructions for Use (IFU-MX1) should be used to guide operation of the device as needed.
Table 1. Emitter Jetson HDMI Register - Requirements, Verification Steps, and Expected Results
Table 2. Release Mode - Requirements, Verification Steps, and Expected Results
Table 3. Debug Mode - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1220
C1 Cassette Rev. I, SN: 1221
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System v3.2.0
APP MedAI Device App v3.2.0
Additional tools/equipment:
EQP-275 - Control Company Stopwatch 4YMT7
RESULTS
Table 1. Emitter Jetson HDMI Register - Requirements, Verification Steps, and Expected Results
Table 2. Release Mode - Requirements, Verification Steps, and Expected Results
Table 3. Debug Mode - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
Anomalies:
Table 2, SRS-1.17 - SID gauge is visible on external monitor connected to the emitter.
LIST OF APPENDICES
Appendix 1 and 21 - Verification Evidence as Specified in Results Tables 1 through 3.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1
root@emitter-dv24:/home/imager# /bin/busybox devmem 0x15b40138
0x0A0A0A07
root@emitter-dv24:/home/imager#
Appendix 2
Last login: Fri Sep 13 16:10:17 2024 from 172.16.11.179
imager@emitter-dv24:~$ su
Password:
root@emitter-dv24:/home/imager# /bin/busybox devmem 0x15b40138 32 0x10101010
root@emitter-dv24:/home/imager# exit
exit
imager@emitter-dv24:~$ exit
logout
Connection to emitter-dv24.local closed.
(base) gagecarr@Gages-MBP-2 ~ % ssh imager@emitter-dv24.local
Last login: Fri Sep 13 20:23:22 2024 from fe80::140f:53f4:c391:a69%eth1
imager@emitter-dv24:~$ su
Password:
root@emitter-dv24:/home/imager# /bin/busybox devmem 0x15b40138
0x0A0A0A07
root@emitter-dv24:/home/imager#
Appendix 3
imager@emitter-dv24:~$ su
Password:
root@emitter-dv24:/home/imager# /bin/busybox devmem 0x15b40138 32 0x10101010
root@emitter-dv24:/home/imager# /bin/busybox devmem 0x15b40138
0x0A0A0A07
root@emitter-dv24:/home/imager#
Appendix 4
imager@nuc8-flashing-1:~$ ssh imager@emitter-dv24.local
The authenticity of host 'emitter-dv24.local (172.16.11.226)' can't be established.
ECDSA key fingerprint is SHA256:4AEnM9G7NfaTh5XwgMW9iHAebv95aF5yiIEwT9AKNeY.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'emitter-dv24.local,172.16.11.226' (ECDSA) to the list of known hosts.
imager@emitter-dv24.local: Permission denied (publickey).
imager@nuc8-flashing-1:~$
Appendix 5
Appendix 6
Appendix 7
Appendix 8
Appendix 9
Appendix 10
Appendix 11
root@emitter-dv24:/home/imager# cat /root/inithashes
#!/bin/bash
echo "Rebuilding list of system file hashes..."
hashdeep -c sha256 -r -o f /bin /boot /etc /lib /opt /sbin /home > /root/file_hashes
ls -la /root/file_hashes
echo "File hashes written to /root/file_hashes"
root@emitter-dv24:/home/imager# cat /root/integritycheck.log
#written by integritycheck.py
System Integrity Check 2024-09-16 14:29:22
Checking for application software modification since build:
Integrity check for emitter applications
b''
OK, No changes
Checking for user service files modification since build:
Integrity check for emitter service files
b''
OK, No changes
Checking for user account changes since build:
b''
OK, No changes
Checking for firewall rule changes since build:
b''
OK, No changes
Checking for application configuration changes since build:
Checking emitter application configuration changes
b'/opt/medai/data/config/emitter-orchestrator-config.json\n'
Config files changed!
Using cassette address because emitter flag is set
Major alarm, alerting application
Command 'curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d '{ "level": "major", "reason": "Application configuration changes since build" }' example.com/' returned non-zero exit status 28.
Can't alert application to send major alarm!
The retry count is now 5, sleeping for 10 seconds
b'Platform fault received'
Integrity Check Complete
root@emitter-dv24:/home/imager# timed out waiting for input: auto-logout
Appendix 12
Appendix 13
root@cassette-dv24:/home/imager# cat /root/integritycheck.log
#written by integritycheck.py
System Integrity Check 2024-09-16 14:19:04
Checking for application software modification since build:
Integrity check for cassette applications
b''
OK, No changes
Checking for user service files modification since build:
Integrity check for cassette service files
b''
OK, No changes
Checking for user account changes since build:
b''
OK, No changes
Checking for firewall rule changes since build:
b''
OK, No changes
Checking for application configuration changes since build:
Checking cassette application configuration changes
b'/opt/medai/data/config/cassette-orchestrator-config.json\n'
Config files changed!
Using localhost address because emitter flag is not set
Major alarm, alerting application
Command 'curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d '{ "level": "major", "reason": "Application configuration changes since build" }' example.com/' returned non-zero exit status 7.
Can't alert application to send major alarm!
The retry count is now 5, sleeping for 10 seconds
Command 'curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d '{ "level": "major", "reason": "Application configuration changes since build" }' example.com/' returned non-zero exit status 7.
Can't alert application to send major alarm!
The retry count is now 4, sleeping for 10 seconds
Command 'curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d '{ "level": "major", "reason": "Application configuration changes since build" }' example.com/' returned non-zero exit status 7.
Can't alert application to send major alarm!
The retry count is now 3, sleeping for 10 seconds
b'Platform fault received'
Integrity Check Complete
Appendix 14
wil@wil-xps15:~/Downloads$ ssh -i id_ed25519 imager@100.127.179.121
Last login: Mon Sep 16 15:10:18 2024 from 100.75.143.54
imager@emitter-dv24:~$
Appendix 15
(base) gagecarr@Gages-MBP-2 .ssh % ssh -i id_ed25519 imager@100.73.42.123
Warning: Permanently added '100.73.42.123' (ED25519) to the list of known hosts.
Last login: Mon Sep 16 14:55:11 2024 from fe80::140f:53f4:c391:a69%eth1
imager@cassette-dv24:~$
Appendix 16
Appendix 17
Appendix 18
imager@cassette-dv24:~$ su -
Password:
su: Authentication failure
imager@cassette-dv24:~$
Appendix 19
imager@emitter-dv24:~$ su -
Password:
su: Authentication failure
imager@emitter-dv24:~$
Appendix 20
wil@wil-xps15:~/Downloads$ ssh imager@emitter-dv24.local
The authenticity of host 'emitter-dv24.local (172.16.11.226)' can't be established.
ED25519 key fingerprint is SHA256:AzLJ91uqNoTPCqd3gN9tcqNnb84hoethUQopCS4kQkU.
This host key is known by the following other names/addresses:
~/.ssh/known_hosts:139: [hashed name]
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'emitter-dv24.local' (ED25519) to the list of known hosts.
imager@emitter-dv24.local's password:
Last login: Mon Sep 16 15:29:11 2024 from 100.75.143.54
Appendix 21
wil@wil-xps15:~/Downloads$ ssh -i id_ed25519 imager@100.127.179.121
Last login: Mon Sep 16 15:10:18 2024 from 100.75.143.54
imager@emitter-dv24:~$

### Table 1
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter |  |  |  |  |
| Test Setup: | The emitter is powered on. |  |  |  |  |
|  | Test Case: Check emitter jetson register after start up |  |  |  |  |
| SRS-4.9 | The SS shall set the emitter Jetson register 0x15b40138 to 0x0A0A0A07 during startup [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1.ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Returned value is 0x0A0A0A07 |  |  |
|  |  | 1. Ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to reset the register value: /bin/busybox devmem 0x15b40138 32 0x10101010 4. Reboot the emitter and type the following command: /bin/busybox devmem 0x15b40138 | Value is successfully set to 0x10101010 |  |  |
|  |  |  | Value is reset to 0x0A0A0A07 upon reboot |  |  |
|  | Test Case: Reset emitter jetson register after full idle |  |  |  |  |
| SRS-8.24 | Upon exiting any idle state, the SS shall reset the emitter Jetson register 0x15b40138 to 0x0A0A0A07 when commanding the emitter touchscreen display to wake from sleep state [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1. ssh into the emitter ssh imager@<emitter-hostname> 2. Allow the unit to enter idle state 3. Pull the trigger to exit idle state 4. Switch to super user via su and enter the correct credentials 5. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Value is set to 0x0A0A0A07 upon exiting idle |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Release Mode Precondition: Ensure the emitter and cassette do NOT have a wired network connection prior to placing the system into release mode. Also, place the emitter in release mode before the paired cassette. Emitter: Place the emitter in release mode using the following steps: 1. Connect a mouse and keyboard to the emitter service port via dongle. Note that an external monitor can be used but is not necessary; the terminal will be visible in the emitter display 2. Open a terminal 3. As root, run /root/inithashes 4. As root, run the following command: rm /imagerdebug 5. Restart the emitter Cassette: Place the cassette in release mode using the following steps: 1. Connect a mouse, keyboard, and external monitor to one of the cassette service ports via dongle 2. Open a terminal 3. As root, run /root/inithashes 4. As root, run the following command: rm /imagerdebug 5. Restart the cassette Additional Note: Retrieve each device's SSH key from /home/imager/.ssh/id_ed25519 in order to SSH in via Tailscale |  |  |  |  |
| SRS-1.10 | The SS shall contain a release mode for devices ready for distribution. Debug mode shall be disabled in release mode. | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the emitter with ssh imager@<emitter-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure |  |  |
| SRS-1.15 | In release mode, the SS shall require certificate-based authentication to allow remote access via SSH |  |  |  |  |
|  |  | 1. Connect a secondary device (e.g. laptop) to the cassette's wifi AP 2. From the secondary device, attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure |  |  |
|  |  | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the emitter via Tailscale 3. Record evidence of a successful SSH connection | Successful SSH connection to emitter in release mode |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette via Tailscale 3. Record evidence of a successful SSH connection | Successful SSH connection to cassette in release mode |  |  |
| SRS-1.16 | In release mode, the SS shall present a blank screen if an external display is connected to the cassette via service port | 1. Connect a mouse, keyboard, and external monitor to the cassette service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to cassette |  |  |
| SRS-1.17 | In release mode, the SS shall present a blank screen if an external display is connected to the emitter via service port | 1. Connect a mouse, keyboard, and external monitor to a emitter service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to emitter |  |  |
| SRS-1.13 | In release mode, the SS shall force logouts of any open maintenance mode terminals after 120 seconds of inactivity | 1. Connect the emitter to a network via ethernet cable 2. Use Tailscale to SSH into the emitter. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on emitter |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. Use Tailscale to SSH into the cassette. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on cassette |  |  |
| SRS-1.14 | In release mode, the SS shall enforce the use of a restricted keyboard key set | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on emitter display |  |  |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on external monitor for cassette |  |  |
| SRS-1.11 | In release mode, the SS shall restrict access to production-level accounts | 1. Connect a mouse, keyboard, and external monitor to the emitter service port via dongle 2. In the terminal login screen, use the command su - and attempt to log in as root user by using incorrect credentials for the unit under test 3. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on emitter |  |  |
|  |  | 1. Connect a mouse, keyboard, and monitor to a cassette service port via dongle 2. In the terminal login screen, use the command su - and attempt to log in as root user by using incorrect credentials for the unit under test 3. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on cassette |  |  |
|  | Test Case: Integrity Check |  |  |  |  |
| SRS-1.18 | In release mode, the SS shall perform an integrity check upon boot and every hour | 1. Connect a mouse, keyboard, and external monitor to the emitter service port via dongle 2. In the terminal login screen on the external monitor, log in as root. 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.11 -m mx1.services stop To start: python3.11 -m mx1.services start 5. Reboot the emitter. Alternatively, wait one hour for the integrity check alarm to be triggered 6. In the terminal login screen, log in as root. 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the emitter |  |  |
|  |  |  | Record evidence of the integrity check alarm on the emitter |  |  |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. In the terminal login screen on the external monitor, log in as root. 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.11 -m mx1.services stop To start: python3.11 -m mx1.services start 5. Reboot the cassette. Alternatively, wait one hour for the integrity check alarm to be triggered 6. In the terminal login screen, log in as root. 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the cassette |  |  |
|  |  |  | Record evidence of the integrity check alarm on the cassette |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Debug Mode Precondition: Emitter: Place the emitter in debug mode using the following steps: 1. Use Tailscale to SSH into the emitter 2. As root user, run the following command: touch /imagerdebug 3. Restart the emitter Cassette: Place the cassette in debug mode using the following steps: 1. Use Tailscale to SSH into the cassette 2. As root user, run the following command: touch /imagerdebug 3. Restart the cassette |  |  |  |  |
| SRS-1.7 | The SS shall contain a debug mode for development and production activities | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), SSH into the emitter with ssh imager@<emitter-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to emitter in debug mode |  |  |
| SRS-1.9 | In debug mode, the SS shall enable SSH |  |  |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to cassette in debug mode |  |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 13 Sep 2024 | 24-534 |

### Table 6
| Equipment Name | EQP # | Last Calibrated Date | Calibration Due Date |
| --- | --- | --- | --- |
| Control Company Stopwatch 4YMT7 | EQP-275 | 05/01/2024 | 05/01/2026 |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter |  |  |  |  |
| Test Setup: | The emitter is powered on. |  |  |  |  |
|  | Test Case: Check emitter jetson register after start up |  |  |  |  |
| SRS-4.9 | The SS shall set the emitter Jetson register 0x15b40138 to 0x0A0A0A07 during startup [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1.ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Returned value is 0x0A0A0A07 | Expected Outcome Verified. Verified by GC 13SEPT24 See appendix 1. | P |
|  |  | 1. Ssh into the emitter: ssh imager@<emitter-hostname> 2. Switch to super user via su and enter the correct credentials 3. Type the following command to reset the register value: /bin/busybox devmem 0x15b40138 32 0x10101010 4. Reboot the emitter and type the following command: /bin/busybox devmem 0x15b40138 | Value is successfully set to 0x10101010 | Expected Outcome Verified. Verified by GC 13SEPT24 See appendix 2. | P |
|  |  |  | Value is reset to 0x0A0A0A07 upon reboot |  |  |
|  | Test Case: Reset emitter jetson register after full idle |  |  |  |  |
| SRS-8.24 | Upon exiting any idle state, the SS shall reset the emitter Jetson register 0x15b40138 to 0x0A0A0A07 when commanding the emitter touchscreen display to wake from sleep state [IEC 60601-1-2:7 ELECTROMAGNETIC EMISSIONS requirements for ME EQUIPMENT and ME SYSTEMS] | 1. ssh into the emitter ssh imager@<emitter-hostname> 2. Allow the unit to enter idle state 3. Pull the trigger to exit idle state 4. Switch to super user via su and enter the correct credentials 5. Type the following command to get the value: /bin/busybox devmem 0x15b40138 | Value is set to 0x0A0A0A07 upon exiting idle | Expected Outcome Verified. Verified by GC 13SEPT24 See appendix 3. | P |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Release Mode Precondition: Ensure the emitter and cassette do NOT have a wired network connection prior to placing the system into release mode. Also, place the emitter in release mode before the paired cassette. Emitter: Place the emitter in release mode using the following steps: 1. Connect a mouse and keyboard to the emitter service port via dongle. Note that an external monitor can be used but is not necessary; the terminal will be visible in the emitter display 2. Open a terminal 3. As root, run /root/inithashes 4. As root, run the following command: rm /imagerdebug 5. Restart the emitter Cassette: Place the cassette in release mode using the following steps: 1. Connect a mouse, keyboard, and external monitor to one of the cassette service ports via dongle 2. Open a terminal 3. As root, run /root/inithashes 4. As root, run the following command: rm /imagerdebug 5. Restart the cassette Additional Note: Retrieve each device's SSH key from /home/imager/.ssh/id_ed25519 in order to SSH in via Tailscale |  |  |  |  |
| SRS-1.10 | The SS shall contain a release mode for devices ready for distribution. Debug mode shall be disabled in release mode. | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the emitter with ssh imager@<emitter-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 4. | P |
| SRS-1.15 | In release mode, the SS shall require certificate-based authentication to allow remote access via SSH |  |  |  |  |
|  |  | 1. Connect a secondary device (e.g. laptop) to the cassette's wifi AP 2. From the secondary device, attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Verify that there is a failure to connect via SSH. Record evidence of the failure. | Record evidence of message displayed in terminal of SSH failure | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 5. | P |
|  |  | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the emitter via Tailscale 3. Record evidence of a successful SSH connection | Successful SSH connection to emitter in release mode | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 14. | P |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette via Tailscale 3. Record evidence of a successful SSH connection | Successful SSH connection to cassette in release mode | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 15. | P |
| SRS-1.16 | In release mode, the SS shall present a blank screen if an external display is connected to the cassette via service port | 1. Connect a mouse, keyboard, and external monitor to the cassette service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to cassette | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 6. | P |
| SRS-1.17 | In release mode, the SS shall present a blank screen if an external display is connected to the emitter via service port | 1. Connect a mouse, keyboard, and external monitor to a emitter service port via dongle 2. Record evidence of the screen displayed on the external monitor | Blank screen is displayed on external display connected to emitter | See Appendix 7. The emitter frontend protrudes onto the screen. | F |
| SRS-1.13 | In release mode, the SS shall force logouts of any open maintenance mode terminals after 120 seconds of inactivity | 1. Connect the emitter to a network via ethernet cable 2. Use Tailscale to SSH into the emitter. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on emitter | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 16. | P |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. Use Tailscale to SSH into the cassette. Additionally, start a timer 3. Leave the terminal session open and inactive 4. Stop timer when the terminal session closes 5. Verify that the terminal session closes after 120 seconds of inactivity | Open maintenance mode terminal session closes after 120 seconds of inactivity on cassette | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 17. | P |
| SRS-1.14 | In release mode, the SS shall enforce the use of a restricted keyboard key set | 1. Connect a mouse and keyboard to the emitter service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on emitter display | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 8. | P |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. Attempt to open a terminal window with control + alt + T 3. Take an image for evidence that a terminal does not open | Terminal window does not appear on external monitor for cassette | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 9. | P |
| SRS-1.11 | In release mode, the SS shall restrict access to production-level accounts | 1. Connect a mouse, keyboard, and external monitor to the emitter service port via dongle 2. In the terminal login screen, use the command su - and attempt to log in as root user by using incorrect credentials for the unit under test 3. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on emitter | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 18. | P |
|  |  | 1. Connect a mouse, keyboard, and monitor to a cassette service port via dongle 2. In the terminal login screen, use the command su - and attempt to log in as root user by using incorrect credentials for the unit under test 3. Record evidence of the failure to enter as root user | Failure to enter as root user using incorrect credentials on cassette | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 19. | P |
|  | Test Case: Integrity Check |  |  |  |  |
| SRS-1.18 | In release mode, the SS shall perform an integrity check upon boot and every hour | 1. Connect a mouse, keyboard, and external monitor to the emitter service port via dongle 2. In the terminal login screen on the external monitor, log in as root. 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.11 -m mx1.services stop To start: python3.11 -m mx1.services start 5. Reboot the emitter. Alternatively, wait one hour for the integrity check alarm to be triggered 6. In the terminal login screen, log in as root. 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the emitter | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 10. | P |
|  |  |  | Record evidence of the integrity check alarm on the emitter | Expected Outcome Verified. Verified by GC 16SEPT24 See appendix 11. | P |
|  |  | 1. Connect a mouse, keyboard, and external monitor to a cassette service port via dongle 2. In the terminal login screen on the external monitor, log in as root. 3. Modify any configuration file in /opt/medai/data/config. Record evidence of the modification. 4. Use the following command to restart the services: To stop: python3.11 -m mx1.services stop To start: python3.11 -m mx1.services start 5. Reboot the cassette. Alternatively, wait one hour for the integrity check alarm to be triggered 6. In the terminal login screen, log in as root. 7. Access /root/integritycheck.log. Record evidence of the integrity check alarm. | Record evidence of the modified configuration file on the cassette | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 12. | P |
|  |  |  | Record evidence of the integrity check alarm on the cassette | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 13. | P |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Device Components Needed: | E1 Emitter, C1 Cassette, M50133 Galaxy Tablet S8+, APP MedAI Device App |  |  |  |  |
| Test Setup: | All device components are powered on and connected. The system is in single radiographic mode. Loading factors are set at the lowest settings (40 kV and 0.04 mAs). |  |  |  |  |
|  | Test Case: Debug Mode Precondition: Emitter: Place the emitter in debug mode using the following steps: 1. Use Tailscale to SSH into the emitter 2. As root user, run the following command: touch /imagerdebug 3. Restart the emitter Cassette: Place the cassette in debug mode using the following steps: 1. Use Tailscale to SSH into the cassette 2. As root user, run the following command: touch /imagerdebug 3. Restart the cassette |  |  |  |  |
| SRS-1.7 | The SS shall contain a debug mode for development and production activities | 1. Connect the emitter to a network via ethernet cable 2. From a secondary device (e.g. laptop), SSH into the emitter with ssh imager@<emitter-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to emitter in debug mode | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 20. | P |
| SRS-1.9 | In debug mode, the SS shall enable SSH |  |  |  |  |
|  |  | 1. Connect the cassette to a network via ethernet cable 2. From a secondary device (e.g. laptop), attempt to SSH into the cassette with ssh imager@<cassette-hostname> 3. Record evidence of a successful SSH connection | Successful SSH connection to cassette in debug mode | Expected Outcome Verified. Verified by WP 16SEPT24 See appendix 21. | P |

### Table 10
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-470 |  |
