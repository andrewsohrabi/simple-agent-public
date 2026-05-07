# RSK-P01-011 Rev A: MX1 Security Risk Assessment

## Metadata
- Document ID: RSK-P01-011
- Revision: A
- Prefix: RSK
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: unknown
- Source filename: RSK-P01-011 - MX1 Security Risk Assessment_A-Obsolete.docx
- Source path: Example QMS - MedAI/RSK-P01-011 - MX1 Security Risk Assessment_A-Obsolete.docx
- Extraction warnings: none

## Extracted Content
RSK-P01-011 - MX1 Security Risk Assessment_A-Obsolete
Sheet: Signoff
Sheet: Introduction
Sheet: Security Risk Identification -
Sheet: Threat Modeling - STRIDE
Sheet: Security Architecture Views
Sheet: Asset Inventory
Sheet: Interface Inventory
Sheet: Security Risks
Sheet: Known Vulnerabilities
Sheet: Security Risk Controls
Sheet: Residual Security Risk Conclusi

### Table 1
| Document: | RSK-P01-011 - MX1 Security Risk Assessment |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Project: | P01 |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |
| Revision | DCO # | Approved By | Description | Eff. Date | Digital Key |
| A | 24-192 | Quality EngineeringEngineeringRegulatory Affairs | Initial release | 2024-04-29 00:00:00 | example.com/ |

### Table 2
| Purpose |  |
| --- | --- |
| The purpose of this document is to summarize the outputs of the cybersecurity risk management process, including threat modeling, for the device. |  |
| Scope |  |
| The design specifications in this revision are applicable to MX1 Software System and MedAI Device App releases v1.0.0, 1.0.1, 1.0.2, 1.1.0, 1.1.1, 1.1.2, 1.1.3, 1.1.4, 1.1.5, 1.1.6, 1.1.7, 1.1.8, 1.2.0, 2.0.0, 2.0.1, and 2.0.2. |  |
| Definitions |  |
| Asset: A person, structure, facility, information, records, information technology systems and resources, material, process, relationships, or reputation that has value. |  |
| Attack Tree: A tree diagram showing the sequence of events that could lead to an asset being attacked. |  |
| CVSS: The Common Vulnerability Scoring System (CVSS) provides a way to capture the principal characteristics of a vulnerability and produce a numerical score reflecting its severity. The numerical score can then be translated into a qualitative representation (such as low, medium, high, and critical) to help organizations properly assess and prioritize their vulnerability management processes. |  |
| Cyber Device: Generally, any medical device that connects to the internet. More specifically, it is a device that (1) includes software validated, installed, or authorized by the sponsor as a device or in a device (2) has the ability to connect to the internet, and (3) contains any such technological characteristics validated, installed, or authorized by the sponsor that could be vulnerable to cybersecurity threats. |  |
| Cybersecurity Signal: Any information which indicates the potential for, or confirmation of, a cybersecurity vulnerability or exploit that affects, or could affect a medical device. |  |
| Data and Systems Security: The operational state of a medical device in which information assets (data and systems) are reasonably protected from degradation of confidentiality, integrity, and availability |  |
| Exploit: An exploit is an instance where a vulnerability or vulnerabilities have been exercised (accidentally or intentionally) by a threat and could impact the safety or essential performance of a medical device or use a medical device as a vector to compromise a connected device or system. |  |
| SBOM: Software Bill Of Materials |  |
| SOUP: Software Of Unknown Provenance is software that has not been developed with a known software development process or which has unknown or no safety-related properties. |  |
| STRIDE: A model for identifying cybersecurity threats using the mnemonic for Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, and Elevation of privilege. |  |
| Threat: A threat is any circumstance or event with the potential to adversely impact the device, organizational operations (including mission, functions, image, or reputation), organizational assets, individuals, or other organizations through an information system via unauthorized access, destruction, disclosure, modification of information, and/or denial of service. Threats exercise vulnerabilities, which may impact the safety or essential performance of the device. |  |
| Vulnerability: A vulnerability is a weakness in an information system, system security procedures, internal controls, human behavior, or implementation that could be exploited by a threat. |  |
| Essential Performance: The performance of a clinical function, other than related to basic safety, where loss or degradation beyond the limits specified by the manufacturer results in an unacceptable risk |  |
| References |  |
| AAMI TIR57 “Principles for medical device security—Risk management” |  |
| NIST 2018 “Framework for Improving Critical Infrastructure Security”, v1.1. |  |
| 2005 FDA Guidance “Cybersecurity for Networked Medical Devices Containing Off-the-shelf (OTS) Software” |  |
| 2021 MITRE “Playbook for Threat Modeling Medical Devices” |  |
| 2023 FDA Guidance “Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions” |  |
| Common Vulnerability Scoring System Version 3.1: Specification Document |  |
| Security Assessment Team |  |
| Name | Organization |
| Bimba Shrestha | Innolitics |
| Gage Carr | MedAI |
| Pavel Sitnikov | MedAI |
| Mickey McGown | MedAI |
| Ari Inoue | MedAI |
| Dhruv Vishwakarma | MedAI |

### Table 3
| Characteristics Related to Safety - Essential Performance |  |  |
| --- | --- | --- |
| 1. What is the essential performance of the device? | Documented in MEMO-P01-441 - Essential Performance Determination |  |
| 2. Can the device be used to affect the patient? For example, warming, electrical impulses, x-rays, deliver fluids? | Yes |  |
| 3. Is the medical device intended to modify the patient environment? Factors that should be considered include temperature, humidity, atmospheric gas composition, pressure, light. | No |  |
| 4. Is the device implanted in a patient? | No |  |
| 5. Is the device intended to make medical decisions or support medical decisions made by a doctor? | Yes |  |
| 5.1 What controls are in place to ensure the integrity of algorithms or data generated from algorithms used to make or suggest medical decisions? | No user access for backendAlgorithms not configurable by end userNo algorithms are used to make or suggest medical decisionsSee Security Controls |  |
| 5.2. Can the operation of the algorithm be affected by outside influences or the results be intercepted and changed? | No |  |
| 5.3. What controls are in place to ensure the integrity of measured or collected clinical or physiological data used to make or suggest medical decisions (e.g., diagnostic ECG data or ultrasound images)? | Encryption at restNo direct user access to image dataHTTPS to external connections |  |
| 6. Does the device control other devices or is it controlled by other devices, either directly or remotely? | Device can be controlled for remote supportSecurity controls for tailscaleFootpedal trigger (SubGhz, filtered by footpedal ID configured at mfg) |  |
| 7. If present on a network, could the device be used as a launching pad (pivot point) for malware on the network? | Yes |  |
| 8. Is there a process to check for known vulnerabilities and for applying fixes? | Yes |  |
| Characteristics Related to Safety - Data Storage - PII/Private Data Assets |  |  |
| 1. Does the device store PII Data? | Yes |  |
| 2. Is the data encrypted when at rest? | Yes |  |
| 3. Are backups created and stored elsewhere? | No |  |
| 3.1. Are backups encrypted? | N/A |  |
| 3.2. Are they encrypted before transport or transmission to the other site? | N/A |  |
| 3.3. Is there a limit on how long backups are kept or on how many versions are kept? | N/A |  |
| 4. Is patient data scrubbed or obscured to remove personal information when not required? | N/A |  |
| 5. How is the integrity of the data preserved on the device and during transfer? | On device, the PHI is being encrypted by symmetric 'AES/CBC/PKCS7Padding' algorithm.The device uses a secure connection (HTTPS) to transfer data to MedAI Platform |  |
| 6. Is the data encrypted when not at rest (i.e., in volatile memory)? | No |  |
| 7. Is encryption fixed or configurable? | Fixed |  |
| 7.1. Is encryption based on a publicly recognized standard (recommended) or on a home-grown solution? | Yes |  |
| 7.2. Is encryption/decryption tested against a validated compatible system? | Yes |  |
| 7.3. How are keys managed? | Keys are managed by Google |  |
| 8. Does the device use or maintain billing or financial information? | No |  |
| 9. Do portions of the data have different levels of sensitivity (PII/non-PII)? | Yes |  |
| 9.1. Is all data encrypted, or just PII? | No |  |
| 9.2. Is any of the PII subject to additional restrictions according to local laws? | No |  |
| 10. Can the data be classified according to sensitivity, and can the principle of least privilege be applied? | No |  |
| 11. Is export of private data or PII (e.g., through removable media or the network) logged and attributable to an authenticated user in a security audit trail? | No |  |
| Characteristics Related to Safety - Data Storage - Non-PII Data Assets |  |  |
| 1. What kinds of non-PII or personal data does the device store (e.g., diagnostic data or therapy parameters)? | Device operation configuration dataDevice identification data |  |
| 2. Does the device store network or other user credentials that could be used to gain access to other systems? I.e., logon to a wireless network or authenticate to an LDAP or similar user authentication system? | No |  |
| 3. Does the device store configuration or calibration data? | Yes |  |
| 3.1. Is configuration or calibration data critical to the operation of the device or safety of the patient? For example, if calibration settings are changed, could it cause harm to the patient? | Yes |  |
| 3.2. What is the impact if configuration data or controls (e.g., configuration interface menus) are compromised? | None |  |
| 3.3. Are controls in place to ensure the integrity of the configuration data? | Startup integrity checking from factory calibration/configuration |  |
| 4. Are measurements taken? For example, the variables that are measured and the accuracy and the precision of the measurement results. | No |  |
| 5. Does the device employ removable media? | Yes |  |
| 6. Are there any other methods to copy, remove, or place data on the device? Consider other methods that data could be copied to or from the device. Examples might include USB ports, PCIe ports (for removable PCIe drives), service ports, wireless access points, etc. | The user is able to export DICOM studies with PII to USB drives. No data is able to be copied onto the device. |  |
| Characteristics Related to Safety - Data Transfer |  |  |
| 1. What types of data are transferred to and from the device? | DICOM studies |  |
| 1.1. Does the data contain PII? | Yes |  |
| 1.2. What methods could potentially be used to intercept data between endpoints? | Man in the middle attacksPACS server spoofing |  |
| 1.3. Are controls in place to ensure the appropriate confidentiality, integrity, and authenticity of the data before and after transfer? For example, encryption and digital signatures. | Yes, the data is encrypted at rest on device |  |
| 2. If an adverse event occurs (either to the patient or the device), can the data be recovered or restored? | Not unless it was sent to the PACS server |  |
| 3. Is availability of the data transferred to or from the device critical to its operation? | No |  |
| 3.1. If data transfer to/from the device is interrupted, does that interruption represent a safety risk? | No |  |
| 3.2. If data transfer to/from the device is interrupted, is critical data cached so that it can be re-transmitted once connectivity is restored? | Yes |  |
| 4. Does the device transfer data via wired connection? | Yes, DICOM studies can be sent over a wired connection to the network |  |
| 4.1. What kinds of communication ports are available on the device? | See Interface Inventory |  |
| 4.1.1. Are communication ports based on standards? For example, standard Ethernet? | See Interface Inventory |  |
| 4.1.2. Are there any proprietary communication ports? | See Interface Inventory |  |
| 4.1.3. Are there serial (e.g., RS-232) ports available? | Yes, under tooled cover |  |
| 4.1.4. Are communication ports internal or external to the device? Are they easily accessible? | See Interface Inventory |  |
| 4.1.5. Are communication ports enabled by default, or do they need to be enabled before they can be used? | See Interface Inventory |  |
| 5. Does the device transfer data via wireless? | See Interface Inventory |  |
| 5.1. What kinds of wireless connectivity does the device use? For example, Bluetooth, Wi-Fi, NFC, custom radio, etc.? | See Interface Inventory |  |
| 5.2. How are the wireless communications secured? Consider what standards are implemented and supported. | See Interface Inventory |  |
| 5.3. Can security levels be negotiated with the peer? | No |  |
| 6. How are communication protocols (e.g., TCP/IP) implemented? | Base Linux Networking Stack is used |  |
| 6.1. Are they implemented using COTS software or part of an underlying OTS operating system? Consider the selection of technologies based on the maturity of the implementation(s). | Part of underlying OTS OS |  |
| 6.1.1. Are they implemented using a custom solution? Consider introducing more stringent testing processes to ensure the protocol stack is robust enough for deployment into a hostile operating environment, such as a hospital network. | No |  |
| 6.1.2. How is the implementation verified in terms of standards for the protocol? | N/A |  |
| 6.1.3. Can it be verified or certified by an outside body? | N/A |  |
| Characteristics Related to Safety - Authentication & Authorization |  |  |
| 1. How does the device authenticate users or services? | See Authentication Security Controls |  |
| 2. How are credentials managed? | See Authentication Security Controls |  |
| 2.1. Are credentials updated on a regular basis? | No |  |
| 2.2. Are credentials updated via remote configuration change, or made locally? | No |  |
| 3. Are there any hard-coded or default accounts on the device? | No |  |
| 4. How is credential expiration managed? | Not managed |  |
| 4.1. How often are users required to update their credentials? | No end-user account for OS |  |
| 4.2. Is credential expiration based on time, number of uses, or other? | Not managed |  |
| 4.3. Can the expiration of credentials be managed from the device? If so, what authority is required to configure it? | Not managed |  |
| 5. How does the device implement person authentication? That is does the device allow a person to authenticate with the device, via a login screen, remote session, or other method? | Service is via remote SSH session using Tailscale VPN, the credentials are in DHR. |  |
| 6. Does the device support multi-factor authentication? | No. |  |
| 7. How is authentication managed, based on the operating environment? | Unique credentials for imager user and root user are set at factory and recorded in DHR. |  |
| 8. Does the device support multiple authorization levels or roles? | The imager user has application-level privileges only, root access is required for system-level privileges. |  |
| 9. Do authenticated user sessions have a timeout? Consider the operating environment in which the device will be used. In some cases, different session timeouts may be necessary, or the ability to disable a timeout may be desirable. | Idle sessions are logged out in two minutes. |  |
| 10. Do user accounts become disabled after a certain number of failed login attempts? | No |  |
| 10.1. Are they re-enabled after a period of time to prevent attackers having a simple means to conduct a denial-of-service attack on accounts? | N/A |  |
| 11. How are user accounts audited? | Remote session access is logged (/var/log/auth.log)Account configuration is checked for tampering at boot and hourly. |  |
| 11.1. Are there controls that disable accounts after a certain amount of inactivity (e.g., 3 months between logons) has passed? | No |  |
| 12. How can audit reports of accounts be generated? Consider methods aside from centrally located authentication providers (e.g., LDAP) of generating lists of users from the device so that unnecessary user accounts may be deactivated. | N/A for the device itself, signals system fault if account added or credentials changed. |  |
| 13. Does the device authenticate with other systems/software/services, or allow other systems/software/services to authenticate with it? | Authenticates to MedAI Platform, Logging server, Update server. |  |
| 13.1. How does the device authenticate with other systems, and how do other systems authenticate with the device? | After admin approval, the device is granted a token by the service. |  |
| 13.2. How are credentials managed for system/software service authentication? For example, a remote system may require credentials that are not stored in an existing LDAP directory and must be coded into the software that authenticates with other systems. A good example of this is authentication with an external Web service. | Not sure how or where each application persistently stores the JWT |  |
| 13.3. If credentials for authenticating with other systems are stored on the device, are they encrypted? If so, are best practices used to manage the decryption key? | Not sure how or where each application persistently stores the JWT |  |
| 13.4. Are credentials stored in hardware (e.g., a TPM)? | No |  |
| Characteristics Related to Safety - Auditing |  |  |
| 1. What actions or activities are logged or audited on the device? Activities that should be considered include the following: |  |  |
| 1.1. all user, device, or process authentication activity; | Yes |  |
| 1.2. user actions/activity; | Yes |  |
| 1.3. changes to system configuration or calibration; | Yes |  |
| 1.4. system-level actions/activity (e.g., health of the device, sensor readings); | Yes |  |
| 1.5. network activity (e.g., remote network connections, data transferred to the device); | Yes |  |
| 1.6. software/firmware updates; and | Yes |  |
| 1.7. export of sensitive data. | Yes |  |
| 2. How is audit data stored? | In persistent storage on the device and in a central log service. |  |
| 2.1. Is the audit trail stored in clear-text, or encrypted? | clear-text |  |
| 2.2. Where is the audit trail kept on the device and who has access to it? | Logs are in the /var/log directory accessible only to the root user. |  |
| 2.3. Can audit data be erased or otherwise tampered with? | Logs can only be modified by the root user. |  |
| 2.4. What are the limitations of audit data storage? | Log storage capacity is 32GB |  |
| 2.5. What authority is required to remove an audit record? | Logs can only be removed by the root user. |  |
| 2.6. Is audit data stored with system data? | Yes |  |
| 3. Can audit data be transferred from the device? | Logs are transferred to a logging service. |  |
| 4. What mechanisms are in place to ensure that audit data is not tampered with during transfer? | Logs are transferred using HTTPS. |  |
| 5. Who is authorized to access audit data? | See Authentication Security Controls |  |
| 6. Are there any other types of access to audit data? | See Authentication Security Controls |  |
| Characteristics Related to Safety - Physical Security |  |  |
| 1. What physical properties does the device exhibit that allow access to information, data, etc., on the device? For example, accessible Ethernet or USB ports, removable hard drives, etc. | USB port for DICOM store to flash drive. Port also supports screen/keyboard/network for servicing. |  |
| 2. Does the device have externally accessible data ports? For example, network ports, serial ports, USB ports. | USB port for DICOM store to flash drive. Port also supports screen/keyboard/network for servicing. |  |
| 2.1. Are external ports enabled by default? | Yes |  |
| 2.2. Can external ports be disabled? | No |  |
| 2.3. When external ports are used, is the activity audited or logged? | Yes |  |
| 2.4. Are there physical locks or other mechanisms available to block access to external ports? | Emitter ports are behind a cover. |  |
| 3. Does the device have a screen? | Yes, the ODA on the tablet, the frontend on the emitter |  |
| 3.1. Is data displayed on the screen sensitive in nature? For example, are passwords displayed in clear-text? | Passwords are not displayed in clear-text. |  |
| 3.2. Is patient data viewable on the screen? Clear text or encrypted/hidden? | Patient data is displayed in clear-text. |  |
| 3.1. Is the device intended to be used in a public area? | No? |  |
| 4. Are there any internal ports on the device? For example, internal USB connections, JTAG connectors, debugging ports? | Internal USB for flashing and servicing. |  |
| 4.1. Are internal ports disabled, either via hardware (e.g., jumpers) or software controls? | No |  |
| 4.2. Can debugging ports be disabled in production configuration of the device? | No |  |
| 5. Are there any anti-tampering mechanisms on the device? | No |  |
| 5.1. Is there any way to detect physical intrusion into a device, actively or passively? For example, labels, adhesives, tamper evident seals, physical covers. | Labels/seals? |  |
| 5.2. Are there software or electronic mechanisms to detect intrusion into a device? | Software integrity is checked at boot and hourly. |  |
| Characteristics Related to Safety - Device/System Updates |  |  |
| 1. Can updates be performed remotely? | Yes |  |
| 1.1. How does the device owner obtain software updates for the device? For example, downloaded over the Internet, provided on CD or USB drive. | Downloaded |  |
| 1.1.1. If software updates are downloaded over the Internet, how is the authenticity of the download confirmed? For example, secure hash, keyed hashed message authentication code, or digital signature? | Updates are digitally signed. |  |
| 1.1.2. If software updates are provided on removable media, how is the authenticity of the software confirmed? | N/A |  |
| 1.1.3. When software updates are provided, what controls are in place to ensure the media (downloaded or otherwise) is free of malware/viruses? | The deb installers are not being scanned for malware. |  |
| 2. Are software updates performed physically (in-person) by the manufacturer? For example, using a specific fixture or hardware required to apply the software update. | No |  |
| 2.1. Are any other physical controls in place to ensure that software updates to the device can be performed only by the manufacturer or trained service provider? For example, USB dongle required to authenticate with the device before software updates can be applied. | No |  |
| 2.2. Are application updates and operating system updates handled separately or bundled? | Can update application or OS plus application |  |
| 3. Is the authenticity of a software update checked before it is applied? For example, does the device require signing of software updates? | Device requires the update to be digitally signed. |  |
| 4. Are software updates validated in any other way prior to installation on the device? For example, deep inspection of the software update for specific design features. | No |  |
| 5. Once software updates are installed on the device, what measures are in place to roll-back an update if an unsafe condition occurs? | A previous version can be applied via update. |  |
| 6. Can the configuration of the device be altered either remotely (e.g., over a network) or physically on the device? | Configuration of the device requires root access. |  |
| 6.1. What method (e.g., physical access or network) is used to alter the device configuration? | Login via SSH |  |
| 6.2. What kinds of changes can be made to the configuration of the device? | User account, System parameters, Application parameters |  |
| 6.3. Can configuration changes place the device in an unsafe state? | Yes |  |
| 6.4. Could configuration changes affect patient safety? | Yes |  |
| 6.5. Can the configuration of the device be returned to a known safe state after being changed? | Can be reflashed |  |
| 6.6. What authentication controls are in place to limit access to configuration changes? | Credentials required for SSH, Credentials for access to Tailscale |  |
| 6.7. Are configuration changes validated prior to being applied? | At factory |  |
| 6.8. Are device configuration changes validated prior to being applied? For example, are the changes within known safe limits on the device? | At factory |  |
| 6.9. Can configuration settings affect user access or privileges? | Yes |  |
| 7. Does the device use COTS operating system or software? | Yes, substantially Ubuntu 20 LTS |  |
| 7.1. What is the verification/validation strategy for COTS software updates? | Internal V&V, SBOM |  |
| 7.2. Are COTS software updates applied by the manufacturer or the device owner? | Manufacturer |  |
| 7.3. How often are COTS software updates made to the device? | As needed |  |
| 7.4. Are there special dispensations for applying critical security updates to the COTS software? For example, hotfixes or zero day security fixes. | No |  |
| 8. Are any other COTS or Software of Unknown Provenance (SOUP) used as part of the device? | Yes |  |
| 8.1. Have COTS or SOUP components been evaluated from a security standpoint? | Yes |  |
| 9. Is there a process for monitoring security issues related to your device after-market? | Yes |  |
| 10. Is there a process whereby users can report a security issue with the device? | Yes |  |
| 11. Are COTS and other components used in the device monitored for security updates? | Yes |  |
| 12. Is there a response process for when security issues are detected? Any process should analyze the security issue and recommend a course of action commensurate with the safety and security risks. | Need an incident response plan |  |
| Characteristics Related to Safety - Hardening |  |  |
| 1. What measures are taken to ensure the system is hardened from external access via exposed interfaces? For example, network connections (wired or wireless), proximal access (physical access to the device). | External display has kiosk mode to prevent terminal access.Network access requires credentials. |  |
| 2. Have unused or unnecessary user accounts been disabled on the device? For example, operating system “guest” accounts, database administrative accounts with default passwords. | Yes |  |
| 2.1. What accounts are necessary for the intended operation of the device or system? | The imager user account. |  |
| 2.2. Have credentials for default accounts been changed? | Yes at factory, recorded in DHR |  |
| 3. Are any custom software applications or services installed on the device? | Yes |  |
| 3.1. What account/access level do those services execute under? | At the imager user evel. |  |
| 3.2. Do those accounts have their privilege levels configured as to not allow them to execute in unintended ways? | Yes |  |
| 4. Does the system use a COTS operating system or software? For example, Windows, Oracle. | Yes, Ubuntu 20 LTS |  |
| 4.1. Have resources such as the National Checklist Program Repository, NIST SP 800-70, been reviewed for guidance regarding configuration of operating systems and off-the-shelf software? | Yes |  |
| 4.2. What measures are taken to ensure that COTS software has been hardened on the device as to not allow inappropriate access, escalation of privilege, etc.? | Pentest |  |
| 5. How is device integrity ensured post manufacturing? | Device checks software integrity and issues system fault if tampering is detected. |  |
| 5.1. Are there controls in place to ensure that the device is not tampered with during transport or delivery? | No |  |
| 5.2. How do you ensure integrity of software updates or other installation media after it is delivered? For example, hashes, digital signatures? | Digital signatures |  |
| 6. How is the integrity of the device ensured when it is initially connected to a network for the first time? For example, are there methods to ensure that zero-day or recently identified vulnerabilities that have active agents looking to exploit those vulnerabilities (worms, viruses, etc.) cannot infect the device when it is first connected to the network? | System does not allow externally initiated connections and has no open ports. |  |
| 7. Are there adequate controls to ensure that malware, viruses, or other unwanted software is not introduced on the device or a component of the device during manufacture or assembly? |  |  |
| 7.1. Is there an audit policy in place as part of the manufacturing process to continually monitor these controls? |  |  |
| 8. Have components been evaluated against databases of known vulnerabilities (e.g., CVE)? |  |  |
| Characteristics Related to Safety - Emergency Access |  |  |
| 1. Is there a need for emergency access given the intended use of the device? |  |  |
| 2. Does the device implement emergency access features (e.g., “break glass” functionality)? |  |  |
| 3. What actions are allowed when operating in emergency-access mode? For example, can the user operate the device in a limited capacity, change settings, update software? |  |  |
| 4. What is the intended purpose of the emergency mode? For example, is it to use the device for a limited time or a limited set of features in an emergency situation? |  |  |
| 5. What kinds of changes might be made to the device while in emergency mode? |  |  |
| 6. What kinds of information or assets can the operator access while in emergency access mode? For example, PII data, user credentials, other sensitive information. |  |  |
| 7. Can an operator update the operating system or software on the device in emergency access mode? |  |  |
| 8. What actions are logged or audited when operating in emergency-access mode? |  |  |
| 8.1. Are audit data or logs accessible during emergency use? |  |  |
| 8.2. Can audit data or logs be modified during emergency use? |  |  |
| 8.3. Are additional actions audited or logged that are not during normal operation? |  |  |
| 8.4. What detail level are actions audited at, when operating in emergency access mode? Is it possible to audit in a higher level of detail when operating in emergency-access mode? |  |  |
| 9. What constraints are placed on emergency access that enforce expected behavior? |  |  |
| Characteristics Related to Safety - Malware/Virus Protection |  |  |
| 1. Is the device susceptible to viruses or malware? For example, does it operate using a COTS operating system or similar that is known to have malware or viruses? |  |  |
| 2. Does the device employ malware or virus protection software? |  |  |
| 2.1. How often are virus or malware definitions updated or deployed to the device? |  |  |
| 3. Are there other controls on the device to prevent malware, virus, or other unwanted modifications? |  |  |
| 4. Does the device include Intrusion Detection or Prevention systems (IDS/IPS)? |  |  |
| 4.1. How often are IDS/IPS signatures updated on the device? |  |  |
| 4.2. How are those IDS/IPS signature updates deployed? |  |  |
| 5. Does the device use application or process whitelisting? |  |  |
| 5.1. How are changes to the whitelist performed? |  |  |
| 6. What level of access is required to install, modify, remove, or disable malware or other protection measures? |  |  |
| 6.1. Are modifications to malware, virus, or other security protections logged? |  |  |
| 6.2. Does the design define how malware or intrusion protection logs are reviewed or transmitted? |  |  |
| 7. Does the design consider potential coexistence of protection applications (e.g., malware monitoring or intrusion detection) with the system software and applications? |  |  |
| Characteristics Related to Safety - Backup/Disaster Recovery |  |  |
| 1. What methods/processes are used to ensure security of device or system backups? |  |  |
| 2. How is the confidentiality of data on device backups ensured? |  |  |
| 2.1. Are device backups encrypted? |  |  |
| 2.2. Are there other security measures to ensure confidentiality of backup data? |  |  |
| 3. How is the integrity of device backups ensured? |  |  |
| 3.1. Are there multiple backups that can be compared? |  |  |
| 3.2. Are there hash values or similar mechanisms produced for each backup to ensure integrity? |  |  |
| 4. After a device is restored using a device backup, what processes or procedures are in place to validate that the device has been returned to a known good state. |  |  |
| Characteristics Related to Safety - Labeling |  |  |
| 1. What instructions are provided for the secure use of the device? |  |  |
| 1.1. If there is sensitive information (e.g., PII) displayed on the screen, are there instructions on the proper use of the device to ensure such data is not publicly visible? For example, ensuring the device is turned away from publicly accessible spaces if it contains PII. |  |  |
| 1.2. Are there instructions or labeling available that provide instruction on the use of the device in different operating environments? For example, documentation describing the differences in how a device may operate when in an acute care setting vs. other settings. |  |  |
| 2. Are instructions provided for the secure configuration and deployment of the device? |  |  |
| 2.1. What configuration(s) of the device are considered “secure” in different operating environments? |  |  |
| 2.2. Is there a baseline or recommended network configuration to support secure deployment of the device? |  |  |
| 2.3. Are there configuration settings that network administrators need to be aware of for the device to function properly? For example, network firewalls may need to be configured to allow certain open ports. |  |  |
| 3. What instructions are provided for the secure disposal of the device? |  |  |
| 3.1. Is there sensitive information stored on the device, such as PII or other confidential information? |  |  |
| 3.2. Are instructions provided for how to properly dispose of the device once it has reached the end of its functional life to ensure that data stored on the device is also destroyed appropriately? |  |  |

### Table 4
| System Item |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Name | Organization | Date |  |  |  |
| Bimba Shrestha | Innolitics | 2024-03-04 00:00:00 |  |  |  |
| Gage Carr | MedAI | 2024-03-04 00:00:00 |  |  |  |
| Threats |  |  |  |  |  |
| What follows is the list of identified threats, their STRIDE threat type, a description of the sequence of events leading to the threat, and notes about their mitigation. Most of the threats trace to security risks, which in turn trace to requirements that have risk control measures. |  |  |  |  |  |
| System Item | STRIDE Type | Description | Mitigation Notes | Security Risks |  |
| Mender Server | Tampering | Malicious actor compromises system update via upstream software vulnerability which modifies the behavior of the device in unexpected ways. | - System updates are signed during deployment using secure signing algorithm. | - SR1 |  |
| MedAI Platform | Elevation of Privileges | Malicious actor gains unauthorized access to web pages containing sensitive data because of broken access control in web application | - Ensure that all url endpoitns in the MedAI Platform have access control validation | - SR2 |  |
| MedAI Platform | Spoofing | Malicious actor guesses or obtains a user’s reset password link and accesses their account by creating a new password | - Password reset links are long and randomly generated, and they expire after some set period of time | - SR3 |  |
| MedAI Platform | Spoofing | Malicious actor brute force guesses user’s password | - Use a strong password strength criteria | - SR3 |  |
| MedAI Platform | Infromation Disclosure | Malicious actor reads sensitive data over-the-wire through an unsecured HTTP connection | - Use HTTPS for all connections | - SR2 |  |
| MedAI Platform | Tampering | Malicious actor provides intentionally malformed data to the application to perform SQL injection on database | - Sanitize inputs before making SQL queries | - SR3 |  |
| MedAI Device App | Information Disclosure | Malicious actor gains access to sensitive data or details about app internals via error stack traces or overly informative error messages | - No PHI in error messages and logs | - SR2 |  |
| MedAI Platform | Spoofing | Malicious actor gains remote access to server running web application by unnecessarily open ports or a misconfigured firewall | N/A | - SR3 |  |
| MedAI Platform | Tampering | Malicious actor exploits known vulnerability in outdated software package that the application is using | - Procedure to do an assessment- CVE scan at build time | - SR2 |  |
| MedAI Platform | Repudiation | Malicious actor’s attempts to exploit system go undetected because high value events such as logins are not properly logged. | - Log all failed login attempts | - SR4 |  |
| MedAI Platform | Denial of Service | Malicious actor overwhelms web application servers with abnormal traffic, causing it to become unavailable | - Have a secure WAF or DNS | - SR5 |  |
| MedAI Platform | Spoofing | Malicious actor modifies DNS entries on DNS platform to redirect all queries to a malicious spoofed server | - Enable 2 factor authentication on all developer accounts | - SR3 |  |
| MedAI Device App | Spoofing | Malicious actor could access data "over-the-wire" between the Cassette and the MedAI Device App | - Devices connected are isolated and there is no routing there | - SR2 |  |
| MedAI Device App | Tampering | Malicious actor uses other apps on tablet that causes the ODA to behave in unexpected ways | - Add something to the user manual saying that the app shoudl be run in kiosk mode | - SR5 |  |
| MedAI Device App | Tampering | Malicious operator tries to use the android app on a device that isn't supported, casuing it to behave unexpectedly | - Add a warning when the app launches on a device that isn't one of the supported ones. | - SR5 |  |
| MedAI Device App | Tampering | Malicious actor compromises new release of android application, leading to security vulnerabilities downstream | - Software development procedure states that code reviews must be done | - SR1 |  |
| MedAI Device App | Information Disclosure | Malicious actor could gain access to the tablet's file system and then get access to the stored PHI | - No PHI in tablet local storage- Local storage file names should not point to study or patient in any way | - SR2 |  |
| MedAI Device App | Repudiation | Malicious activity from operator goes undetected because of insufficient logging, leading to security risks later | - Ensure sufficient logging and upload logs to a centralized location | - SR4 |  |
| MedAI Device App | Spoofing | Malicious actor uses android tablet in unexpected ways | - Add authentication between Capture Present and the MedAI Device App | - SR2 |  |
| Tailscale | Elevation of Privileges | Malicious actor gains access to Tailscale account and then gains access to software running on Cassette | - Implement Access Level Controls (ACLs) in Tailscale, and require two factor authenication | - SR6 |  |
| Cassette/Emitter/Footpedal | Tampering | Malicious actor compromises system update via upstream software vulnerability which modifies the behavior of the Cassette/Emitter/Footpedal in unexpected ways. | - System updates are signed during deployment using secure signing algorithm. | - SR1 |  |
| Cassette/Emitter/Footpedal | Repudiation | Malicious operator inserts corrupted USB drive into cassette USB port and claims to have not done so | - Syslog logs when a USB interface is intialized | - SR4 |  |
| Cassette/Emitter/Footpedal | Information Disclosure | Malicious actor gains access to Cassette or Emitter SSD and then gains access to sensitive data | - Physical security- Requires tools to access SSD | - SR2 |  |
| Cassette/Emitter/Footpedal | Tampering | Malicious actor loads malicious software onto Cassette/Emitter from external media | - Physical security- Use of proprietary servicing ports for loading software/firmware | - SR7 |  |
| Cassette/Emitter/Footpedal | Information Disclosure | Malicious actor gains access to sensitive data via emitter-cassette communication line | - Credentials required to access cassette WiFi APTablet and emitter communications with cassette managed via AP client isolation | - SR2 |  |
| Cassette/Emitter/Footpedal | Spoofing | Malicious actor spoofs cassette and tablet/emitter communication packets | "Use WPA2 protocol to encrypt internal and external WiFi communicationsCredentials required to access cassette WiFi APTablet and emitter communications with cassette managed via AP client isolationData packets encoded via proprietary communication protocol" | - SR2 |  |
| Cassette/Emitter/Footpedal | Spoofing | Malicious actor spoofs emitter and foot pedal communications packets | Communication is one-way (foot pedal to emitter) over proprietary communication protocol (ICD) | - SR2 |  |
| Cassette/Emitter/Footpedal | Information Disclosure | Malicious actor gains access to sensitive data over an insecure Wifi network | "IFU - ""Warning: Do not connect the MX1 device to any open, unsecure or public networks. All networks should be encrypted using WPA2 or higher, including in Non-Professional and Home Environments.""Use WPA2 protocol to encrypt internal and external WiFi communications" | - SR7 |  |
| Cassette/Emitter/Footpedal | Elevation of Privileges | Malicious actor gains unauthorized access to production/service-level accounts | Release mode restricts access to production-level accounts (kiosk mode) | - SR7 |  |
| Cassette/Emitter/Footpedal | Spoofing | Malicious actor brute force guesses a service account's password | Release mode restricts access to production-level accounts (kiosk mode) | - SR3 |  |
| Cassette/Emitter/Footpedal | Information Disclosure | Malicious actor performs man-in-the-middle attack of external connection to cassette | Host-based firewall for traffic from external networks | - SR3 |  |

### Table 5
| Global System View |  |
| --- | --- |
| The following cybersecurity diagrams have been provided as recommended by 2023 FDA Guidance “Cybersecurity in Medical Devices: Quality System Considerations and Content of Premarket Submissions”. |  |
| Updatability and Patchability View |  |
| The following diagram shows the high level flow of creating and managing keys and Artifact (system update) signatures. After creating and signing the update Artifact, it is made available to the devices running the Mender Client by uploading it to the Mender Server (which runs within Mender's secure fully managed cloud environment). During the update installation process, the Mender Client running on the device will verify the Artifact using the corresponding public key(s). The Artifact will only be installed if the verification is successful. The Mender Client will abort the update process and report an error to the Mender Server if there is a failure at any point.For more details, see the Mender Artifact Creation Documentation. |  |
| Use Case View |  |
| To be added in Phase 3 |  |

### Table 6
| Asset Inventory |
| --- |
| The following assets were identified: |
| MedAI Platform Database |
| DICOM files from the device |
| Capture Presenter Database |
| Image files |
| Modality Worklist |
| ELK Log Server |
| Local logs on Cassette/Emitter |
| Local logs on the MedAI Device App |
| Github developer logins |
| Google cloud logins |
| Gitlab developer logins |
| Mender user logins |
| ELK user logins |
| Physician information stored on Cassette |
| Google drive user logins |
| Configuration files for Cassette |
| Configuration files for Emitter |
| Company reputation |
| MedAI Device App android code sign certificate |
| Mender artifact signer |
| Cassette encryption key |
| MedAI Platform user logins |
| Source code |
| Intellectural property in binary builds |
| Wifi password |

### Table 7
| Interface Inventory |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Interface | Name | Type | External | Description |  |
| INT-001 | Remote Support | VPN Link | True | Tailscale services are used for remote support from MedAI to end user. The protocol used is Wireguard (ref). User Access Control is defined by Tailscale admin through the server. Authentication is controlled by Google Authentication to the Tailnet server and password protected access from server to device. Passwords are currently maintained in the DHR. |  |
| INT-002 | Play Store | HTTPS | True | This connection is used to query and update the ODA. The facility admin will set up and manage the Google Account that is used to access the Google Play Store. |  |
| INT-003 | Frontend Data | Websockets | False | This interface is used to transfer images and metadata to the frontend interface on ODA. The hosted network on the Cassette requires WPA2 encryption. The network credentials are set in factory and displayed on the cassettte display. The credentials can only be changed by a credentialed user. The number of clients allowed is restricted to 5. |  |
| INT-004 | Frontend Configuration | HTTP | False | This interface is used to control device parameters to the frontend interface on ODA. The hosted network on the Cassette requires WPA2 encryption. The network credentials are set in factory and displayed on the cassettte display. The credentials can only be changed by a credentialed user. The number of clients allowed is restricted to 5. |  |
| INT-005 | DICOM Local | DICOM | True | - This communication path is used to store DICOMs created by the device to the PACS, and to retrieve the Modality Worklist from the PACS.- PACS connection details are configured on the MedAI Device App before this communication path is used.- This communication path is strictly between two DICOM supporting entities within the customer's secure Intranet.- DICOM connection details, including HTTPS/SSL configuration follow the security best practices at the customer's site. |  |
| INT-006 | MedAI Platform | HTTPS | True | The interface is used to communicate images, metadata, and PHI to the MedAI Platform. Authentication is performed by the Google Identity Platform (GIP). |  |
| INT-007 | DICOM Remote | DICOM | True | - This communication path is used to store DICOMs created by the device to the PACS, and to retrieve the Modality Worklist from the PACS.- PACS connection details are configured on the MedAI Device App before this communication path is used.- This communication path is strictly between two DICOM supporting entities within the customer's secure Intranet.- DICOM connection details, including HTTPS/SSL configuration follow the security best practices at the customer's site. |  |
| INT-008 | Device Logging | HTTPS | True | This interface is used to log device performance information to a cloud logging service (Google BindPlane). Authentication is performed by Google Authentication. |  |
| INT-009 | Software Updates | HTTPS | True | This interface is used to deliver qualified and signed software updates to devices and to collect some inventory information. Server-side authentication is performed through generation of a web token from the device upon first request. The Mender service is hosted by MedAI and users are managed by <TBD> and requires 2FA. |  |
| INT-010 | Emitter Cassette High Level | Websockets | False | This interface is used to transfer technical information to perform basic operations on the device, such as coordination of x-ray acquisition. This interface is firewalled and not accessible to end users without root access. |  |
| INT-011 | Emitter Cassette Low Level | Sub-Ghz | False | This interface is used for low latency coordination between the emitter and cassette to perform time-sensitive operations, such as acquisition timing. Authentication is performed by checking for the factory-set ID of the client. |  |
| INT-012 | Backend Internal | HTTP | False | This interface is used to communicate device parameter information to be passed to the ODA. This interface is firewalled and not accessible to end users without root access. |  |
| INT-013 | Footpedal Emitter Low Level | Sub-Ghz | False | This interface is used for low latency coordination between the emitter and footpedal to perform time-sensitive operations, such as triggering. Authentication is performed by checking for the factory-set ID of the client. |  |
| INT-014 | Emitter Service High Level | USB-C | False | One interface is used only for power and cannot access data in the system. The second interface is used for screen and keyboard control of the emitter Jetson for debugging. A limited kiosk mode is implemented to disallow user access to a terminal. Access is controlled by a esoteric password only available to MedAI. Both ports are secured by a tooled cover. |  |
| INT-015 | Emitter Service Low Level | Micro USB | False | One interface is used to flash the emitter firmware over a UART connection. The second interface is used to flash the Jetson. Both ports are secured by a tooled cover. |  |
| INT-016 | Cassette Service High Level | USB-C | False | Both interfaces can be used for screen and keyboard control of the cassette Jetson for debugging. A limited kiosk mode is implemented to disallow user access to a terminal. Access is controlled by a esoteric password only available to MedAI. |  |
| INT-017 | Cassette Service Low Level | Micro USB | False | One interface is used to flash the cassette firmware over a UART connection. The second interface is used to flash the Jetson. Both ports are secured by a tooled cover. |  |
| INT-018 | Tablet Logging | HTTPS | True | This interface is used to log tablet performance information to a cloud logging service (Google BindPlane). Authentication is performed by Google Authentication. |  |

### Table 8
| Security Risks |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  | Pre-Mitigation |  |  |  |  | Post-Mitigation |  |  |  |
| ID | Name | Threat Actors | Assets | Sequence of Events | Leads to Safety Risks | Severity | Likelihood | Acceptability | Mitigation | New Hazards from Mitigation | Severity | Likelihood | Acceptability |  |
| SR1 | Compromised Software Update | Organized criminal organizations | Availability of device, PHI or other systems on the network | 1. Malicious actor exploits supply side vulnerability and compromises the integrity of a software release.2. The compromised software release is deployed to production.3. Insecure software is used to make device unavailable, or launch a ransomware attack. | - Delayed procedure- Increased exposure to radiation | Serious (4) | Remote (2) | Attempt to Mitigate | Refer to Security Risk Controls | Nothing substantial. Note that these mitigations could hinder the ability to quickly deploy software updates. | Serious (4) | Improbable (1) | Acceptable |  |
| SR2 | Sensitive Data Exposure | Organized criminal organizations, sloppy developers, unintentional misuse | Image data or related PHI | 1. Malicious actor gains access to PHI or other sensitive data by exploiting software vulnerability.2. Sensitive data is leaked | None | Moderate (3) | Ocassional (3) | Attempt to Mitigate | Refer to Security Risk Controls | None | Moderate (3) | Remote (2) | Acceptable |  |
| SR3 | Unauthorized Access | Organized criminal organizations | PHI or other systems on the network | 1. Malicious actor gains access to device or connected systems by exploiting software vulnerability.2. Unauthorized access is used to make device unavailable, or launch a ransomware attack. | None | Minor (2) | Ocassional (3) | Attempt to Mitigate | - SRS-7.4 | None | Minor (2) | Improbable (1) | Acceptable |  |
| SR4 | Insufficient Logging | Organized criminal organizations | Availability of device, PHI or other systems on the network | 1. Device is compromised.2. Because of insufficient logging, admins are unaware that the device is compromised. | None | Moderate (3) | Ocassional (3) | Attempt to Mitigate | - SRS-3.6- SRS-3.7- SRS-3.8 | None | Moderate (3) | Remote (2) | Acceptable |  |
| SR5 | Unavailable Device | Organized criminal organizations | Availability of device | 1. Malicious actor exploits improper or insufficient data validation to cause the device to behave in unexpected ways and potentially become unavailable.2. Device is temporarily unavailable. | - Delayed procedure- Increased exposure to radiation | Moderate (3) | Ocassional (3) | Attempt to Mitigate | Refer to Security Risk Controls | None | Moderate (3) | Improbable (1) | Acceptable |  |
| SR6 | Unauthorized Access to Running Device | Organized criminal organizations | Availability of device, PHI or other systems on the network | 1. Malicious actor gains access to running device by exploiting software vulnerability2. Unauthorized access is used to make device unavailable or gain access to PHI | - Delayed procedure- Increased exposure to radiation | Serious (4) | Remote (2) | Attempt to Mitigate | Refer to Security Risk Controls | None | Serious (4) | Improbable (1) | Acceptable |  |
| SR7 | Launch Point for Network Attack | Organized criminal organizations | Other systems on the network | 1. Device is compromised.2. Device is used to launch attacks on other systems on the network.3. Ransomware disrupts clinical workflows. | None related to this device | Serious (4) | Ocassional (3) | Attempt to Mitigate | - SRS-7.7 | None | Serious (4) | Improbable (1) | Acceptable |  |
| SR8 | Sensitive Data Exposure | Organized criminal organizations | Information in logs | 1. Malicious actor gains access to private embedded key in apk2. Private key is used to decrypt logs3. PHI in logs is leaked | None | Serious (4) | Ocassional (3) | Attempt to Mitigate | Refer to Security Risk Controls | None | Improbable (1) | Moderate (3) | Acceptable |  |

### Table 9
| Known Vulnerabilities |  |  |
| --- | --- | --- |
| To be completed in Phase 3 |  |  |

### Table 10
| Security Risk Controls |  |  |  |
| --- | --- | --- | --- |
| A variety of risk control measures have been implemented related to cybersecurity risks and threats. These are presented below, categorized according to the concepts Identify, Authentication, Authorization, Confidentiality, Cryptography, Detection, Integrity, Resiliency, and Updateability. |  |  |  |
| SRS ID | Category | Requirement |  |
| SRS-XX | Authentication | The MedAI Platform shall use randomly generated links for password reset |  |
| SRS-XX | Authentication | The MedAI Platform password reset links shall expire |  |
| SRS-44.5 | Authentication | The SS shall enforce a strong password strength criteria |  |
| SRS-XX | Authentication | Multi-factor authentication shall be enabled on all developer accounts |  |
| SRS-44.6 | Authentication | The SS shall authenticate communication between the Capture Presenter and the ODA |  |
| SRS-7.4 | Authentication | The SS shall require credentials to access the Cassette WiFi AP |  |
| SRS-XX | Authorization | The MedAI Platform shall perform access control validation on all URL endpoints |  |
| SRS-XX | Authorization | Tailscale accounts shall be configured to use access control |  |
| SRS-1.11 | Authorization | The SS in release mode shall restrict access to production-level accounts |  |
| SRS-XX | Code, Data, and Execution Integrity | The MedAI Platform shall sanitize all inputs before making queries to the database |  |
| SRS-XX | Code, Data, and Execution Integrity | The SS software dependencies shall be evaluated and determined to be widely-used |  |
| SRS-28.2 | Code, Data, and Execution Integrity | The ODA shall warn users when launched on an unsupported android tablet |  |
| SRS-XX | Code, Data, and Execution Integrity | There shall be a software development procedure that requires code to be reviewed before getting merged in |  |
| SRS-XX | Code, Data, and Execution Integrity | The SS shall use proprietary servicing ports for loading software/firmware |  |
| SRS-44.7 | Confidentiality | The SS shall not allow a direct SSH route to the Cassette |  |
| SRS-XX | Confidentiality | The User Manual shall include a recommendation to configure the android tablet to run the ODA in kiosk mode |  |
| SRS-28.9 | Confidentiality | The ODA local storage file names shall not point to a study or patient |  |
| SRS-XX | Confidentiality | The SSD on the Cassette and Emitter shall be inaccessible without tools |  |
| SRS-7.12 | Confidentiality | The SS shall use one-way communication between the Footpedal and Emitter |  |
| SRS-XX | Confidentiality | The User Manual shall include a warning to not connect the device to insecure networks |  |
| SRS-7.2 | Cryptography | The SS shall use HTTPS for all non-internal connections |  |
| SRS-7.1 | Cryptography | The SS shall use the WPA2 protocol to encrypt WiFi communications |  |
| SRS-3.5 | Event Detection and Logging | The SS shall not log any PHI |  |
| SRS-3.6 | Event Detection and Logging | The SS shall log all failed login attempts |  |
| SRS-3.7 | Event Detection and Logging | The SS shall log when a USB interface is initialized |  |
| SRS-45.18 | Firmware and Software Updates | The SS shall use a software update system that signs updates during deployment using a secure signing algorithm |  |
| SRS-45.19 | Firmware and Software Updates | The SS shall use a software update system that uses role-based access control |  |
| SRS-XX | Resiliency and Recovery | The MedAI Platform shall have a secure web application firewall |  |
| SRS-7.7 | Resiliency and Recovery | The SS shall have a host-based firewall |  |
| SRS-45.17 | Resiliency and Recovery | The SS shall be architected in a way such that the deployed version of the software can be "rolled back" to a previous version in less than an hour |  |
| SRS-3.8 | Resiliency and Recovery | The SS shall generate a security audit log that captures any external access to the cassette or emitter. |  |

### Table 11
| To be done in Phase 3 |
| --- |
