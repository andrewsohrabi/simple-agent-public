# VVPR-P01-170 Rev C: MX1 Software System Penetration Test Protocol and Report

## Metadata
- Document ID: VVPR-P01-170
- Revision: C
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-170 - MX1 Software System Penetration Test Protocol and Report_C.docx
- Source path: Example QMS - MedAI/VVPR-P01-170 - MX1 Software System Penetration Test Protocol and Report_C.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirement Specifications as it relates to software security-specific features and/or risk mitigations.
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirement Specifications, specifically as they pertain to the security of the MX1 Software System.
MEMO-P01-630 Rev. B, MX1 Software Requirements Specification (SRS), includes requirements applicable to MX1 Software System (including the MedAI Device App) version 3.1.0.
This is to be accomplished through outlining and subsequently performing penetration testing against the subject MX1 System.
REFERENCES
MEMO-P01-630 - MX1 Software Requirement Specifications Rev. C
IFU-MX1 - Instructions for Use Rev. D
MATERIALS
MX1 Portable X-ray System
E1 Emitter
C1 Cassette
F1 Foot Pedal
T1 Tablet
APP MedAI Device App
H1 Charger
W1 Wireless Charger
Malicious input tools
OWASP ZAP (HTTP Proxy)
sqlmap (SQL Injection)
nmap (Port scanner)
nikto (Web Application Scanner)
wireshark (network sniffer)
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main St Suite 700 Springfield, IL 60001. To be performed by trained MedAI engineering staff.
Where screenshot evidence is not attached, the personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified.
Overview
A penetration test is a dedicated attack against a network connected system. The focus of this test is to perform attacks similar to those of a malicious remote attacker and to attempt to infiltrate the system, alter or remove data, or cause degradation of system performance. The goal of the penetration testing to be performed against the MedAI MX1 System is to evaluate the system and exploit any available flaws in the system security.
The target MX1 System is an embedded appliance for taking x-ray images in medical settings. The main operating system is based on Ubuntu 20.04LTS. The appliance consists of a Cassette which contains an x-ray detector, an Emitter which contains an x-ray source, and a Tablet which is the primary user interface. The Cassette and Emitter operate on Ubuntu, while the Tablet operates on Android 12.
Available pathways for network based attacks are:
Two external network interfaces from the Cassette: 1) WiFi in AP mode, and 2) WiFi in DHCP mode
The Emitter has one external interface, WiFi in DHCP mode.
The Tablet has one external interface, WiFi in DHCP mode.
Methodology
The tester approaches the system as would a remote unauthenticated attacker, and includes:
Identification of the externally accessible TCP and UDP ports
Identification of the service software versions referenced to known exploits
Identification of network broadcasts
Crawling of web services to identify remotely available resources
The remotely available services are to be subjected to various abuses and examined for side effects
Though the testing is applied as it would be to a typical web application, this device is not intended to be directly internet-facing. The security model is somewhat different because the operator uses the Tablet via the cassette-hosted WiFi for the primary user interface. The Tablet application runs as a logged-in user but there is not an HDMI display or keyboard/mouse in this implementation. The system is required to be stored in an environment with restricted access.
The following tools are to be used to discover remotely available resources and to apply malicious input:
OWASP ZAP (HTTP Proxy)
sqlmap (SQL Injection)
nmap (Port scanner)
nikto (Web Application Scanner)
wireshark (network sniffer)
Test Setup
The Ubuntu operating system on the Cassette and Emitter components runs on one JETSON XAVIER NX Module in each component. Networking is facilitated by a Intel 9260 module in each component. The Tablet is a Samsung Galaxy Tab S8+.
Note: The intent of this cybersecurity penetration test report is to test the security of the MX1 Software System. The security and functionality of the MedAI Device App and its communication to the rest of the MX1 system is not impacted by the specific hardware hosting the App.
The Cassette acts as a WiFi access point for the Emitter and Tablet. The Cassette also acts as a DHCP client to external networks through the WiFi interface.
The Tablet is the primary user interface. There is no mouse, keyboard, or primary video display connected directly to the Jetson modules.
The Cassette external network interface is meant for access to customer image storage servers (PACS).
Jetson modules with connected Intel 9260 networking modules are to be configured with MX1 Software for the purposes of this test. The report shall document the software used for testing.
Firewall setup - remove the /imagerdebug file and reboot. These will enable the release mode of the MX1 system.
Experimental Procedure
Follow the steps outlined in Tables 1-4 below. The test steps shall be:
External Port Scanning - Scan for open TCP/UDP ports visible from each available network interface of each device
Enumeration of API calls - Traffic between ODA and Cassette captured and scanned with ZAP, SQL and command injections, file inclusions
Verification of AP Client Isolation - Demonstrate that an AP client cannot see traffic of another client
Verify limited sudo privileges - User imager should have limited sudo privileges, shutdown,reboot,suspend and nmcli only
Table 1. External Port Scanning - Requirements, Verification Steps, and Expected Results
Table 2. Enumeration of Capture Presenter API - Requirements, Verification Steps, and Expected Results
Table 3. Verification of AP Client Isolation - Requirements, Verification Steps, and Expected Results
Table 4. Verify limited sudo privileges - Requirements, Verification Steps, and Expected Results
Data Analysis
All of the verification tests in Tables 1 through 4 shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Evidence/Test Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements in Tables 1 through 4 per the expected results documented in the “Expected Result/Pass Criteria” column.
PROTOCOL APPROVAL
Digital Key: example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
E1 Emitter Rev. H, SN: 1206
C1 Cassette Rev. I, SN: 1207
M50133 Galaxy Tablet S8+ Rev. A, MPN: R52T504E84B
MX1 Software System, v3.1.0-gamma
RESULTS
Table 1. External Port Scanning - Requirements, Verification Steps, and Expected Results
Table 2. Enumeration of Capture Presenter API - Requirements, Verification Steps, and Expected Results
Table 3. Verification of AP Client Isolation - Requirements, Verification Steps, and Expected Results
Table 4. Verify limited sudo privileges - Requirements, Verification Steps, and Expected Results
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 through Appendix 3 - Verification Evidence as Specified in Results Tables 1 through 4.
Appendix 1 - URL list
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
example.com/
Appendix 2 - SQLmap runs
doctorpost.txt
POST example.com/ HTTP/1.1
user-agent: Dart/3.4 (dart:io)
content-type: application/json;charset=UTF-8
accept: application/json;charset=UTF-8
content-length: 62
host: 10.24.96.1:8080
{"identifier":"5rfb","firstName":"rfghyf","lastName":"ffyrfr"}
┌──(kali㉿kali)-[~]
└─$ sqlmap -r doctorpost.txt --proxy example.com/ --level 5 --risk 3 --batch
___
__H__
___ ___["]_____ ___ ___ {1.8.2#stable}
|_ -| . ["] | .'| . |
|___|_ [,]_|_|_|__,| _|
|_|V... |_| example.com/
[!] legal disclaimer: Usage of sqlmap for attacking targets without prior mutual consent is illegal. It is the end user's responsibility to obey all applicable local, state and federal laws. Developers assume no liability and are not responsible for any misuse or damage caused by this program
[*] starting @ 11:32:48 /2024-05-25/
[11:32:48] [INFO] parsing HTTP request from 'doctorpost.txt'
JSON data found in POST body. Do you want to process it? [Y/n/q] Y
[11:32:48] [INFO] testing connection to the target URL
...
...
...
[11:34:19] [CRITICAL] all tested parameters do not appear to be injectable. You can give it a go with the switch '--text-only' if the target page has a low percentage of textual content (~100.00% of page content is text). If you suspect that there is some kind of protection mechanism involved (e.g. WAF) maybe you could try to use option '--tamper' (e.g. '--tamper=space2comment') and/or switch '--random-agent'
[11:34:19] [WARNING] HTTP error codes detected during run:
500 (Internal Server Error) - 110 times
[*] ending @ 11:34:19 /2024-05-25/
====================================================================
studypost.txt
POST example.com/ HTTP/1.1
user-agent: Dart/3.4 (dart:io)
content-type: application/json;charset=UTF-8
accept: application/json;charset=UTF-8
content-length: 297
host: 10.24.96.1:8080
{"patient":{"id":"8ugf","first_name":"ghcvgh","last_name":"fytguu","birthdate":"2024-05-18T00:00:00.000","npi":null},"doctor":{"id":"5rfb","first_name":"rfghyf","last_name":"ffyrfr","birthdate":null,"npi":null},"doctorGuid":"5rfb","patientOrientation":"RIGHT","description":"","bodyPart":"FINGER"}
┌──(kali㉿kali)-[~]
└─$ sqlmap -r studypost.txt --proxy example.com/ --level 5 --risk 3 --batch
___
__H__
___ ___[']_____ ___ ___ {1.8.2#stable}
|_ -| . [)] | .'| . |
|___|_ [)]_|_|_|__,| _|
|_|V... |_| example.com/
[!] legal disclaimer: Usage of sqlmap for attacking targets without prior mutual consent is illegal. It is the end user's responsibility to obey all applicable local, state and federal laws. Developers assume no liability and are not responsible for any misuse or damage caused by this program
[*] starting @ 12:29:30 /2024-05-25/
[12:29:30] [INFO] parsing HTTP request from 'studypost.txt'
...
...
...
[13:38:24] [CRITICAL] all tested parameters do not appear to be injectable. You can give it a go with the switch '--text-only' if the target page has a low percentage of textual content (~100.00% of page content is text). If you suspect that there is some kind of protection mechanism involved (e.g. WAF) maybe you could try to use option '--tamper' (e.g. '--tamper=space2comment') and/or switch '--random-agent'
[13:38:24] [WARNING] HTTP error codes detected during run:
400 (Bad Request) - 26642 times
[*] ending @ 13:38:24 /2024-05-25/
===========================================================
modepost.txt
POST example.com/ HTTP/1.1
user-agent: Dart/3.4 (dart:io)
content-type: application/json;charset=UTF-8
accept: application/json;charset=UTF-8
content-length: 40
host: 10.24.96.1:8080
{"mode":"PHOTO","dose":0.0,"ddr_fuel":0}
┌──(kali㉿kali)-[~]
└─$ sqlmap -r modepost.txt --proxy example.com/ --level 5 --risk 3 --batch
___
__H__
___ ___[)]_____ ___ ___ {1.8.2#stable}
|_ -| . [.] | .'| . |
|___|_ [(]_|_|_|__,| _|
|_|V... |_| example.com/
[!] legal disclaimer: Usage of sqlmap for attacking targets without prior mutual consent is illegal. It is the end user's responsibility to obey all applicable local, state and federal laws. Developers assume no liability and are not responsible for any misuse or damage caused by this program
[*] starting @ 13:40:24 /2024-05-25/
[13:40:24] [INFO] parsing HTTP request from 'modepost.txt'
...
...
...
[14:07:07] [CRITICAL] all tested parameters do not appear to be injectable. If you suspect that there is some kind of protection mechanism involved (e.g. WAF) maybe you could try to use option '--tamper' (e.g. '--tamper=space2comment') and/or switch '--random-agent'
[14:07:07] [WARNING] HTTP error codes detected during run:
400 (Bad Request) - 26609 times
[*] ending @ 14:07:07 /2024-05-25/
Appendix 3 - ZAP Report
See 2024-05-25 ZAP Scan Report
(Available Upon Request)
REPORT APPROVAL
Digital Key: example.com/

### Table 1
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Case: External Port Scan |  |  |  |  |  |
| SRS-44.2 | The SS shall configure the firewall to reject ICMP packets | Scan using nmap of wlan0 from external network after cassette firewall is enabled: $ nmap 10.44.1.245 -v -p 22,53,8080,8081,8082,8083,8086,8765 | Must say ‘host is down’ because pings are blocked |  |  |
| SRS-7.6 SRS-44.1 | The SS shall utilize a host-based firewall to protect the system's internal network from inbound connections from external network The SS shall have no open unsecured ports | Host now blocks ping, scan anyway should show ports blocked also $ nmap 10.44.1.245 -v -p 22,53,8080,8081,8082,8083,8086,8765 -Pn | Must say “filtered”, or “closed” for all ports |  |  |
| SRS-7.7 | The SS shall implement firewall rules to block non-emitter AP clients from reaching emitter services | Use nmap from an AP client, verify that emitter ports not visible:$ nmap 10.24.96.1 -v -p 8086,8088,8765 | Must say ‘filtered” |  |  |

### Table 2
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Case: Enumeration of Capture Presenter API |  |  |  |  |  |
| SRS-44.3 | The SS shall utilize APIs that are resistant to web application security threats | 1. Modify capture-presenter to listen on port 8888 instead of default port 8080: Environment="MICRONAUT_SERVER_PORT=8888" 2. Use SSH to tunnel port 8080 to mitmproxy: $ ssh kali@10.42.0.147 -L 10.24.96.1:8080:localhost:8080 3. Use mitmproxy to forward the traffic on port 8888 to ZAP proxy listener: # mitmproxy -p 8080 --mode upstream:example.com/ 4. Configure ZAP proxy to listen on 8000 then forward to Jetson capture-presenter's port 8888 5. Filter out false positives by resending flagged requests manually 6. Check ZAP scan log for security alerts above low severity | ZAP scan log shows no alerts with a severity greater than “low” Attach ZAP active scan log to appendix for evidence |  |  |
|  |  | Prior to running the following tests, use the proxy setup from Table 2 to enumerate and capture URLs from the Tablet’s perspective. Record evidence of the capture. | None - This step is for data collection used for the following tests. |  |  |
|  |  | 1. From the collected list, select URLs that cause database writes using client input fields for abuse. 2. Prepare sample information for testing. 3. Use SQLmap to abuse each json parameter. | Verify the output contains the string “all tested parameters do not appear to be injectable” all tested parameters do not appear to be injectable. |  |  |

### Table 3
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Case: AP Client Isolation |  |  |  |  |  |
| SRS-7.8 | The SS shall utilize AP client isolation to isolate WiFi communications between cassette and emitter and cassette and tablet | Verify that the ap-isolation feature of hostapd is functioning: With an emitter and tablet attached, devices in release mode, run ‘netdiscover’ from a simulated attacker client. | The command result must indicate that only the AP’s address should be discoverable by the attacker. The emitter and tablet should not be discoverable. |  |  |

### Table 4
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| SRS-44.4 | The SS shall utilize the Principle of Least Privilege | Verify that an attempt to list sudo capabilities shows limited privileges $ sudo -l Enter the password | Verify the following output is received: User imager may run the following commands on <hostname>: (ALL : ALL) NOPASSWD: /sbin/poweroff,/sbin/reboot,/usr/bin/systemctl suspend,/usr/bin/nmcli |  |  |
|  |  | Verify that an attempt to use sudo to root privilege is logged to /var/log/auth.log $ sudo su - Enter password for <user> Verify the activity is logged Open /var/log/auth.log | Verify the following output is shown after the first command: $ sudo su - [sudo] password for imager: Sorry, user imager is not allowed to execute '/bin/su -' as root on <hostname> From the log, provide evidence was logged of the attempted login |  |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | SW Engineering Quality Engineering Regulatory Affairs | 26 Apr 2024 | 24-159 |
| B | Updated SRS IDs and added SRS-44.1 to Table 1 | Engineering Quality Engineering Regulatory Affairs | 29 May 2024 | 24-324 |

### Table 6
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Case: External Port Scan |  |  |  |  |  |
| SRS-44.2 | The SS shall configure the firewall to reject ICMP packets | Scan using nmap of wlan0 from external network after cassette firewall is enabled: $ nmap 10.44.1.245 -v -p 22,53,8080,8081,8082,8083,8086,8765 | Must say ‘host is down’ because pings are blocked | $ nmap 10.44.1.225 -v -p 22,53,8080,8081,8082,8083,8086,8765 Starting Nmap 7.94SVN ( example.com/ ) at 2024-05-24 13:32 EDT Initiating Ping Scan at 13:32 Scanning 10.44.1.225 [2 ports] Completed Ping Scan at 13:33, 3.06s elapsed (1 total hosts) Nmap scan report for 10.44.1.225 [host down] Read data files from: /usr/bin/../share/nmap Note: Host seems down. If it is really up, but blocking our ping probes, try -Pn Nmap done: 1 IP address (0 hosts up) scanned in 3.11 seconds Verified by MM 29MAY24 | P |
| SRS-7.6 SRS-44.1 | The SS shall utilize a host-based firewall to protect the system's internal network from inbound connections from external network The SS shall have no open unsecured ports | Host now blocks ping, scan anyway should show ports blocked also $ nmap 10.44.1.245 -v -p 22,53,8080,8081,8082,8083,8086,8765 -Pn | Must say “filtered”, or “closed” for all ports | $ nmap 10.44.1.225 -v -p 22,53,8080,8081,8082,8083,8086,8765 -Pn Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower. Starting Nmap 7.94SVN ( example.com/ ) at 2024-05-24 13:34 EDT Initiating Parallel DNS resolution of 1 host. at 13:34 Completed Parallel DNS resolution of 1 host. at 13:34, 0.00s elapsed Initiating Connect Scan at 13:34 Scanning 10.44.1.225 [8 ports] Completed Connect Scan at 13:34, 3.03s elapsed (8 total ports) Nmap scan report for 10.44.1.225 Host is up. PORT STATESERVICE 22/tcp   filtered ssh 53/tcp   filtered domain 8080/tcp filtered http-proxy 8081/tcp filtered blackice-icecap 8082/tcp filtered blackice-alerts 8083/tcp filtered us-srv 8086/tcp filtered d-s-n 8765/tcp filtered ultraseek-http Read data files from: /usr/bin/../share/nmap Nmap done: 1 IP address (1 host up) scanned in 3.07 seconds Verified by MM 29MAY24 | P |
| SRS-7.7 | The SS shall implement firewall rules to block non-emitter AP clients from reaching emitter services | Use nmap from an AP client, verify that emitter ports not visible:$ nmap 10.24.96.1 -v -p 8086,8088,8765 | Must say ‘filtered” | $ nmap 10.24.96.1 -v -p 8086,8088,8765 Starting Nmap 7.94SVN ( example.com/ ) at 2024-05-24 13:41 EDT Initiating Ping Scan at 13:41 Scanning 10.24.96.1 [2 ports] Completed Ping Scan at 13:41, 0.00s elapsed (1 total hosts) Initiating Parallel DNS resolution of 1 host. at 13:41 Completed Parallel DNS resolution of 1 host. at 13:41, 13.02s elapsed Initiating Connect Scan at 13:41 Scanning 10.24.96.1 [3 ports] Completed Connect Scan at 13:41, 1.35s elapsed (3 total ports) Nmap scan report for 10.24.96.1 Host is up (0.0032s latency). PORT STATESERVICE 8086/tcp filtered d-s-n 8088/tcp filtered radan-http 8765/tcp filtered ultraseek-http Read data files from: /usr/bin/../share/nmap Nmap done: 1 IP address (1 host up) scanned in 14.41 seconds Verified by MM 29MAY24 | P |

### Table 7
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Case: Enumeration of Capture Presenter API |  |  |  |  |  |
| SRS-44.3 | The SS shall utilize APIs that are resistant to web application security threats | 1. Modify capture-presenter to listen on port 8888 instead of default port 8080: Environment="MICRONAUT_SERVER_PORT=8888" 2. Use SSH to tunnel port 8080 to mitmproxy: $ ssh kali@10.42.0.147 -L 10.24.96.1:8080:localhost:8080 3. Use mitmproxy to forward the traffic on port 8888 to ZAP proxy listener: # mitmproxy -p 8080 --mode upstream:example.com/ 4. Configure ZAP proxy to listen on 8000 then forward to Jetson capture-presenter's port 8888 5. Filter out false positives by resending flagged requests manually 6. Check ZAP scan log for security alerts above low severity | ZAP scan log shows no alerts with a severity greater than “low” Attach ZAP active scan log to appendix for evidence | Expected outcome verified. See Appendix 32024-05-25-ZAP-Report-MX1.html Verified by MM 29MAY24 | P |
|  |  | Prior to running the following tests, use the proxy setup from Table 2 to enumerate and capture URLs from the Tablet’s perspective. Record evidence of the capture. | None - This step is for data collection used for the following tests. | Expected outcome verified. See Appendix 1 |  |
|  |  | 1. From the collected list, select URLs that cause database writes using client input fields for abuse. 2. Prepare sample information for testing. 3. Use SQLmap to abuse each json parameter. | Verify the output contains the string “all tested parameters do not appear to be injectable” all tested parameters do not appear to be injectable. | Expected outcome verified. See Appendix 2 Verified by MM 29MAY24 | P |

### Table 8
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| Test Case: AP Client Isolation |  |  |  |  |  |
| SRS-7.8 | The SS shall utilize AP client isolation to isolate WiFi communications between cassette and emitter and cassette and tablet | Verify that the ap-isolation feature of hostapd is functioning: With an emitter and tablet attached, devices in release mode, run ‘netdiscover’ from a simulated attacker client. | The command result must indicate that only the AP’s address should be discoverable by the attacker. The emitter and tablet should not be discoverable. | $ sudo netdiscover -r 10.24.96.0/24 Currently scanning: Finished!   |   Screen View: Unique Hosts 4 Captured ARP Req/Rep packets, from 2 hosts.   Total size: 240 _____________________________________________________________________________ IP        At MAC Address Count Len  MAC Vendor / Hostname ----------------------------------------------------------------------------- 10.24.96.1  9c:b6:d0:4b:fb:5c  3 180  Rivet Networks 10.24.96.57 b0:7d:64:57:3c:21  1  60  Intel Corporate Expected outcome verified. Verified by MM 29MAY24 | P |

### Table 9
| Requirement ID | Software Requirement Specification | Verification Steps | Pass Criteria | Evidence/Test Result | PASS/FAIL |
| --- | --- | --- | --- | --- | --- |
| SRS-44.4 | The SS shall utilize the Principle of Least Privilege | Verify that an attempt to list sudo capabilities shows limited privileges $ sudo -l Enter the password | Verify the following output is received: User imager may run the following commands on <hostname>: (ALL : ALL) NOPASSWD: /sbin/poweroff,/sbin/reboot,/usr/bin/systemctl suspend,/usr/bin/nmcli | imager@cassette-dv22:~$ sudo -l Matching Defaults entries for imager on cassette-dv22: env_reset, mail_badpass, secure_path=/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin\:/snap/bin User imager may run the following commands on cassette-dv22: (ALL : ALL) NOPASSWD: /sbin/poweroff, /sbin/reboot, /usr/bin/systemctl suspend, /usr/bin/nmcli Expected outcome verified. Verified by MM 29MAY24 | P |
|  |  | Verify that an attempt to use sudo to root privilege is logged to /var/log/auth.log $ sudo su - Enter password for <user> Verify the activity is logged Open /var/log/auth.log | Verify the following output is shown after the first command: $ sudo su - [sudo] password for imager: Sorry, user imager is not allowed to execute '/bin/su -' as root on <hostname> From the log, provide evidence was logged of the attempted login | imager@cassette-dv22:~$ sudo su - [sudo] password for imager: Sorry, user imager is not allowed to execute '/usr/bin/su -' as root on cassette-dv22. May 24 18:28:56 cassette-dv22 sudo:   imager : command not allowed ; TTY=pts/2 ; PWD=/home/imager ; USER=root ; COMMAND=/usr/bin/su - Expected outcome verified. Verified by MM 29MAY24 | P |

### Table 10
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| C | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-449 |  |
