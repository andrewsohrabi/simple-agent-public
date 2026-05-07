# MEMO-P01-633 Rev B: MX1 System and Software Architecture Design

## Metadata
- Document ID: MEMO-P01-633
- Revision: B
- Prefix: MEMO
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-633 - MX1 System and Software Architecture Design_B.docx
- Source path: Example QMS - MedAI/MEMO-P01-633 - MX1 System and Software Architecture Design_B.docx
- Extraction warnings: none

## Extracted Content
1. Purpose
The purpose of this document is to describe the architecture of the Software System within the MX1 Portable X-ray System, its various components, and the manner in which those components interface with each other.
2. Scope
The contents of this document is relevant to the architecture of the MX1 Software System v3.1.0.
3. References
IEC 60601-1 - Medical electrical equipment – Part 1: General requirements for basic safety and essential performance
IEC 62304 - Medical device software - Software life cycle processes
RSK-P01-010 - MX1 Risk Assessment, Rev. C
PLN-P01-024 - MX1 Software Development Plan, Rev. D
MEMO-P01-634 - MX1 Software Descriptions, Rev. B
MEMO-P01-630 - MX1 Software Requirement Specifications, Rev. C
MEMO-P01-640 - MX1 OTS (SOUP) Report, Rev. B
4. Definitions
Software of Unknown Provenance (SOUP): Software item that is already developed and generally available and that has not been developed for the purpose of being incorporated into the medical device (also known as “off-the-shelf software”) or Software Item previously developed for which adequate records of the development processes are not available
Off-the-shelf (OTS) Software: Pre-developed software that is readily available and not specifically designed for integration into a particular medical device
Software Unit: Software Item that is not subdivided into other items
Software Item: Any identifiable part of a computer program, i.e., source code, object code, control code, control data, or a collection of these items
Software Component: For the MX1 Software System, the largest/highest-level Software Items that comprise the Software System
Software System: An Integrated collection of Software Items organized to accomplish a specific function or set of functions
5. Conventions
The diagrams within this document display the decomposed Software Components that comprise the overall Software System of the MX1 Portable X-ray System. Each of the components have been divided into smaller Software Items, and the diagrams show how the Items may interface with each other.
Diagram descriptions:
Software Ecosystem Diagram - This diagram denotes the decomposed Software Components that comprise the overall MX1 Software System. The diagram also highlights the high-level hardware platforms upon which each component resides and the communication bus and/or protocol used by interfacing Software Components.
Software System Communications Diagram - In this diagram, the Software Components are further divided into smaller, logically segregated Software Items. Data flow and interactions between the different Software Items and across Software Components are highlighted here.
Network Details Diagram - This networking diagram presents an overview of the private and secure wireless communications occurring across device components. Additionally, it showcases the system’s ability to connect to wireless networks external to the system to support features such as RIS/PACS integration.
6. Software Safety Segregation
The main Software Components are separated in the following ways:
The firmware components are located on separate processors. Further details are provided in MEMO-P01-634 - MX1 Software Descriptions.
The software components are logically separated as indicated in the Software System Communications Diagram in this document.
Other more specific methods of segregating Software Items may be listed for Software Components depending on safety classification.
7. Software Safety Classification
As stated in PLN-P01-024 - MX1 Software Development Plan, the MX1 Software System has an overall safety classification of Class B as defined in IEC 62304.
RSK-P01-010 - MX1 Risk Assessment indicates that all software-related risks are considered “broadly acceptable” after the application of mitigations. In order to duly ensure the MX1 Portable X-ray System meets safety and essential performance requirements, certain Software Components are categorized and verified as Class B components based on the pre-mitigation scores, even though the overall residual risk is acceptable.
The decomposed Software Items for all the Software Components are indicated in diagrams below. The output of RSK-P01-010 - MX1 Risk Assessment drives which of those Software Items critical to device safety and/or essential performance may drive a Class B categorization:
If any Software Item is linked to a hazard that results in a moderate risk pre-mitigation, the corresponding parent Software Component is categorized as Class B.
If the risk assessment may be used to confirm that all Software Items of a Software Component are Class A, then that Software Component classification may also be reduced to Class A.
The safety classification for all Software Components in the MX1 Software System is Class B.
8. Software Ecosystem Diagram
This diagram details the components in the MX1 system with communication interfaces and protocols designated by lines. The software system has tightly controlled interfaces between the components, providing separation of concerns that are described in MEMO-P01-634 - MX1 Software Descriptions.
9. Software System Communications Diagram
The Software System Communications Diagram details the various sequenced signals that are used for multi-component functions.
Trigger Signals
This diagram shows the sequence of events involved in generating a trigger event. The trigger signal can be generated from 2 physical triggers, the foot pedal, or over REST for engineering mode (only used for testing and debugging). The subsequent x-ray signals diagram follows this trigger signal.
X-ray Signals
This diagram shows the sequence of events that follow a trigger signal in order to perform an x-ray acquisition. As this sequence is complex and involves many components of the system, the diagram is broken down into 10 steps and substeps. The steps are colored discretely to allow for easier visualization.
10. Network Details Diagram
The Network Details Diagram details wired and wireless communication protocols and connections.
11. Firewall
MX1 uses firewalls and security rules to ensure the various communication pathways are secure. The rules describing access and allowances are as follows.
Emitter Jetson: wlan0 as a DHCP client
Disallow all inbound TCP/UDP with exceptions:
Allow established connection responses
Allow inbound TCP on ports 8787 and 8788
Allow all outbound connections
Allow ping
Cassette Jetson: WiFi AP interface wap0
Disallow all inbound TCP/UDP with exceptions:
Allow established connection responses
Allow inbound TCP on ports 8080,8081,8082,8083,8084 for all communication
Allow inbound TCP on ports 8086, 8088, 8787, 8788 for emitter only
Allow all outbound connections
12. Firmware Diagrams
All MX1 firmware components are built on similar architectures that provide separation of concerns through discrete handlers for each function, such as laser control or power management. These handlers act independently except for some particular cases where information needs to be passed between them. The software components on the Jetson can only command and communicate with any firmware peripherals through the ICD protocol.
Emitter-firmware
Emitter-firmware controls various functions in the emitter, as described in MEMO-P01-634 - MX1 Software Descriptions.
Collimator-firmware
Collimator-firmware controls the functions of the collimator, as described in MEMO-P01-634.
Monoblock-firmware
Monoblock-firmware controls the functions of the monoblock, as described in MEMO-P01-634.
Cassette-firmware
Cassette-firmware controls various functions in the cassette, as described in MEMO-P01-634.
Footpedal-firmware
Footpedal-firmware controls the functions of the footpedal, as described in MEMO-P01-634.
13. SOUP Integration
MEMO-P01-640 - MX1 OTS (SOUP) Report Descriptions lists all the SOUPs used by each MX1 Software Component as well as their versions, manufacturers, licenses, and general function within the MX1 Software System.
All SOUPs are integrated within the overall MX1 Software System and/or individual Software Items recorded in the diagrams above. There are no SOUPs that exist as individual Software Items.
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| To: | File |
| --- | --- |
| From: | Device Software, Systems Integration, Cloud Services |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 25 Apr 2024 | 24-187 |
| B | Updated revisions of References | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-449 |  |
