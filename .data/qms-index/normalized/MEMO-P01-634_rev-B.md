# MEMO-P01-634 Rev B: MX1 Software Descriptions

## Metadata
- Document ID: MEMO-P01-634
- Revision: B
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-634 - MX1 Software Descriptions_B-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-634 - MX1 Software Descriptions_B-signed.docx
- Extraction warnings: none

## Extracted Content
1. PURPOSE
This document is intended to define and highlight the major features of the software components comprising the Software System on the MX1 Portable X-ray System.
2. SCOPE
This document describes the primary features, responsibilities, and, if applicable, developmental and operational environments of the individual software components comprising the MX1 Software System.
3. REFERENCES
MEMO-P01-637 - MX1 Software Documentation Level Evaluation Rev. B
MEMO-P01-460 - MX1 OTS (SOUP) Report Rev. B
4. OVERVIEW OF THE MX1 SOFTWARE SYSTEM
Please refer to MEMO-P01-460 - MX1 OTS (SOUP) Report for further information regarding SOUPs used within the MX1 Software System. All SOUPs are integrated within the MX1 Software System and MedAI Device App (i.e. no standalone SOUPs). As a result, no additional documentation is provided to device operators as they will not need or be able to directly interact with any SOUPs with the MX1 system.
5. EMITTER
6. CASSETTE
7. MX1 JETPACK 5 OS
8. MedAI DEVICE APP
9. FOOT PEDAL
DOCUMENT REVISION HISTORY
Digital Key:
example.com/

### Table 1
| Emitter Orchestrator (EO) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Primary integrator and controller for the Emitter. This component is responsible for coordinating communications and interactions between emitter-specific software components, cassette, and foot pedal. |
| Features/Responsibilities | Primary features/responsibilities: Interfacing with and coordinating communication between emitter-specific software components, cassette, and foot pedal Coordinating technique factor settings based on results from XR Controller outputs Managing trigger inputs Coordinating x-ray emission based on system safety interlocks Monitoring for critical faults Placing system into a safe state in response to detected faults Interfaces with the following software components: Cassette Orchestrator XR Controller Intermachine Signal Proxy Emitter Firmware Collimator Firmware Monoblock Firmware Foot Pedal Firmware |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter Jetson Xavier NX |

### Table 2
| XR (eXtended Reality) Controller (XRC) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Responsible for processing interaction between real world and virtual data to determine emitter position with respect to a cassette. Additionally, responsible for features provided via ViewFinder. |
| Features/Responsibilities | Manages/interfaces with the following hardware components: Tracking camera Viewfinder camera Imaging camera Time of Flight (ToF) sensors Collimator Emitter Touchscreen Display Primary features/responsibilities: Tracking / Localization Collimator position and projection calculations Time of Flight (ToF) data processing Viewfinder image processing Imaging camera photo/video capture Device orientation Link to Emitter Frontend Provides data to/for: Tracking related interlocks SSD related features X-ray image processing Dose calculations Collimator control Image/video stream to Qt front end application |
| Development and Operational Environment |  |
| Language | C++, CUDA C |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter Jetson Xavier NX |

### Table 3
| Emitter Frontend (EMF) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Primary interface for emitter display and HMI. Display elements are written in Qt. Application must be open source to comply with Qt license restrictions, no proprietary MedAI code. |
| Features/Responsibilities | Frontend application to show on the emitter display. Responsible for showing Viewfinder feed Imaging camera preview X-ray image preview Emitter user interface Image and system data is provided by the XR Controller using the IPC Communicator. Interfaces with Physical emitter buttons (through XR Controller) Technique selection Collimator mode selection System options/configuration |
| Development and Operational Environment |  |
| Language | C++ with Qt language extensions |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter Jetson Xavier NX |

### Table 4
| Intermachine Signal Proxy (IMP) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Primary facilitator of intermachine communications between emitter and cassette via shared memory. IMP is an executable software component that resides on both the emitter and cassette Jetsons. |
| Features/Responsibilities | Primary responsibilities: Manages and coordinates intermachine communications between paired emitter and cassette Broadcasts signals received from other software processes Interfaces with the following software components: Cassette Orchestrator Emitter Orchestrator Idle manager Cassette display driver Iray Signaler |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter and Cassette Jetson Xavier NX |

### Table 5
| Idle Manager (IM) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Responsible for monitoring system entry and exit into idle state. |
| Features/Responsibilities | Primary responsibilities: Monitor system entry and exit into idle state Interfaces with the following software components: Cassette Orchestrator (via intermachine proxy) Cassette Display Driver (via intermachine proxy) Emitter Orchestrator |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter Jetson Xavier NX |

### Table 6
| Wifi Stability (WS) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Responsible for monitoring wifi connection status and reconnecting if needed. |
| Features/Responsibilities | Primary responsibilities: Monitor emitter wifi connection status |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter Jetson Xavier NX |

### Table 7
| Emitter Firmware (EM) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Firmware for the emitter main PCB’s imagerontroller unit |
| Features/Responsibilities | Interfaces with the following software system component: Emitter Orchestrator Cassette Firmware Foot pedal Firmware Interfaces with the following hardware system components: Monoblock Collimator Interfaces with the following board level hardware components: USART M-LVDS transceiver RGB LEDs Battery Interface Temperature sensors Humidity sensor Inertial Measurement Unit Sub-GHz radios LVDS signaling transceiver Buzzer Power control circuitry Analog to Digital Converters Primary responsibilities: ICD communications RGB LED display Battery level and charging status Temperature reporting Humidity reporting Idle detection Detector acquisition timing Foot pedal button press reception Audible indication for end of X-ray Collimator/Monoblock power on/off control Voltage regulator monitoring Charge control |
| Development and Operational Environment |  |
| Language | C |
| Operating System (if applicable) | N/A |
| Hardware Platform | Emitter STM32L433VCT6 Microcontroller Unit |

### Table 8
| Collimator Firmware (COL) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Firmware for the collimator PCB’s imagerontroller unit. |
| Features/Responsibilities | Interfaces with the following software system components: Emitter Orchestrator XR Controller Interfaces with the following board level hardware components: USART M-LVDS transceiver Brushless DC Motor drivers Phototransistor Quadrature encoder signals Lasers Time of Flight (ToF) sensors Analog to Digital ConvertersPrimary responsibilities: ICD communications Collimator homing Collimator closed loop control Laser on/off control Distance ranging Voltage regulator monitoring |
| Development and Operational Environment |  |
| Language | C |
| Operating System (if applicable) | N/A |
| Hardware Platform | Collimator STM32F103RCT6 Microcontroller Unit |

### Table 9
| Monoblock Firmware (MB) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Firmware for the monoblock LV PCB board’s imagerontroller unit |
| Features/Responsibilities | Interfaces with the following software system component: Emitter Orchestrator Interface with the following hardware system components: Emitter Interfaces with the following board level hardware components: USART M-LVDS transceiver Pulse Width Modulated timers Digital to Analog Converters Timeout circuits LVDS signaling transceiver Temperature sensors Analog to Digital Converters Primary responsibilities: ICD communications High voltage generation Filament control X-ray timing control X-ray voltage reporting X-ray current reporting X-ray timeout detection Detector acquire reset request Temperature reporting Voltage regulator monitoring |
| Development and Operational Environment |  |
| Language | C |
| Operating System (if applicable) | N/A |
| Hardware Platform | Monoblock STM32F446RET6 Microcontroller Unit |

### Table 10
| Cassette Orchestrator (CO) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Primary integrator and controller for the Cassette. This component is responsible for coordinating communications and interactions between cassette-specific software components, emitter, and wirelessly-accessible MedAI Device App. |
| Features/Responsibilities | Primary features/responsibilities: Interfacing with and coordinating communication between cassette-specific software components, emitter, and MedAI Device App Coordinating detector operation based on system safety interlocks Monitoring for critical faults Placing system into a safe state in response to detected faults Managing Cassette display and HMI Interfaces with the following software components: Emitter Orchestrator Intermachine Signal Proxy Cassette Firmware iRay Signaler Capture Presenter MedAI Device App |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Cassette Jetson Xavier NX |

### Table 11
| iRay Signaler (IRS) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Responsible for monitoring the health status of the Mercu0909X detector. Also responsible for coordinating communications between the detector and other software components. |
| Features/Responsibilities | Interfaces with the following hardware component: Mercu0909X detector Primary responsibilities: Monitor detector health status Coordinating communications during x-ray acquisition Interfaces with the following software components: Cassette orchestrator Emitter orchestrator (via intermachine proxy |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Cassette Jetson Xavier NX |

### Table 12
| Cassette Display Driver (CDD) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Responsible for displaying cassette battery status, cassette-hosted WiFi name, cassette-hosted WiFi status, and other cassette statuses |
| Features/Responsibilities | Interfaces with the following hardware component: Cassette OLED display Primary responsibilities: Display cassette battery status Display WiFi information Display other cassette statuses and information Interfaces with the following software components: Cassette orchestrator Idle manager (via intermachine proxy) |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Cassette Jetson Xavier NX |

### Table 13
| Intermachine Signal Proxy (IMP) |
| --- |
| Refer to IMP entry in the Emitter section above. |

### Table 14
| Capture Presenter (CP) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Capture Presenter refers to the Java backend service running on the Cassette Jetson Xavier NX. This component is primarily responsible for managing communications between the MedAI Device App and other software components. |
| Features/Responsibilities | Interfaces with the following software components: MedAI Device App Cassette Orchestrator Interfaces with the following software services external to the MX1 Software System: RIS/PACS server, if configured Primary responsibilities: Coordinating communications with MedAI Device App Relaying newly acquired images on MedAI Device App Relaying error messages for display on MedAI Device App Secure storage of the following: Image metadata (e.g. corresponding technique factors, dosage) Associated patient information Network configurations RIS/PACS configurations Queued DICOM Studies Logging user-initiated events |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Cassette Jetson Xavier NX |

### Table 15
| Image Reaper (IR) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Image Reaper refers to the service that culls the oldest images stored on the Cassette Jetson Xavier NX in accordance with a set disk space quota |
| Features/Responsibilities | Interfaces with the following software components: Capture Presenter Primary responsibilities: Run timed disk scans to report capture count quotas and disk space quotas Run on-demand capture removals initiated by REST API |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Cassette Jetson Xavier NX |

### Table 16
| Connectivity Controller (CC) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Connectivity Controller refers to the service that retrieves a list of available WiFi networks available to the Cassette Jetson Xavier NX. |
| Features/Responsibilities | Interfaces with the following software components: Capture Presenter Primary responsibilities: Queries NetworkManager for list of available client WiFi APs Reports available WiFi APs to Capture Presenter and supports connection attempts |
| Development and Operational Environment |  |
| Language | C++ |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Cassette Jetson Xavier NX |

### Table 17
| Cassette Firmware (CAS) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Firmware for the cassette main PCB’s imagerontroller unit |
| Features/Responsibilities | Interfaces with the following software system component: Cassette Orchestrator Interfaces with the following hardware system component: X-ray Detector Interfaces with the following board level hardware components: USART M-LVDS transceiver IR LEDs RGB LEDs Battery interface Temperature sensors Humidity sensor Sub-GHz radio Power control circuitry Analog to Digital Converters Primary responsibilities: ICD communications IR LED pattern projection RGB LED display Battery level and charging status Temperature reporting Humidity reporting Detector acquisition timing Detector power on/off control Voltage regulator monitoring |
| Development and Operational Environment |  |
| Language | C |
| Operating System (if applicable) | N/A |
| Hardware Platform | Cassette STM32L433VCT6 Microcontroller Unit |

### Table 18
| MX1 Jetpack 5 OS (JO) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Manages Ubuntu OS on Emitter and Cassette Jetsons |
| Features/Responsibilities | Primary responsibilities: System startup checks System and application logging storage/management Management of network communications between cassette and emitter Management of wireless network communications between cassette and wirelessly accessed (e.g. Android tablet) MedAI Device App |
| Development and Operational Environment |  |
| Language | Bash |
| Operating System (if applicable) | Ubuntu 22.04 |
| Hardware Platform | Emitter and Cassette Jetson Xavier NX |

### Table 19
| MedAI Device App (ODA) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Primary user interface to display acquired images and capture user-initiated actions related to image processing and transmission. |
| Features/Responsibilities | Interfaces with the following software components: Capture Presenter Cassette Orchestrator Primary responsibilities: Display of acquired images Relaying user-initiated image processing requests Relaying network and PACS configuration information Capturing and relaying user-input data, including patient information DICOM calibration features Provides/receives data to/for: All user-input data is sent to Capture Presenter for processing and storage |
| Development and Operational Environment |  |
| Language | Dart |
| Operating System (if applicable) | Android OS - Android 10 or above |
| Hardware Platform | Tested on Samsung Galaxy Tab S8+ |

### Table 20
| Foot Pedal Firmware (FP) |  |
| --- | --- |
| General Description + Primary Features |  |
| Description | Firmware for the foot pedal PCB’s imagerontroller unit |
| Features/Responsibilities | Interfaces with the following software system component: Emitter Firmware Interfaces with the following board level hardware components: Foot pedals/buttons Sub-GHz radio Analog to Digital Converters Bi-color LEDs Primary responsibilities: Relaying foot pedal/button presses to emitter Wireless transmit indicator Battery status indicator |
| Development and Operational Environment |  |
| Language | C |
| Operating System (if applicable) | N/A |
| Hardware Platform | Foot pedal STM32F446RET6 Microcontroller Unit |

### Table 21
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 29 Apr 2024 | 24-182 |
| B | Added reference to MEMO-P01-640 | Engineering Quality Engineering Regulatory Affairs | 27 May 2024 | 24-286 |
