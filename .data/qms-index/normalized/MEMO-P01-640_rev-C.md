# MEMO-P01-640 Rev C: MX1 OTS SOUP Report

## Metadata
- Document ID: MEMO-P01-640
- Revision: C
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-640 - MX1 OTS SOUP Report_C-Signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-640 - MX1 OTS SOUP Report_C-Signed.docx
- Extraction warnings: none

## Extracted Content
MEMO-P01-640 - MX1 OTS SOUP Report_C-Signed
Sheet: Sign Off
Sheet: Introduction
Sheet: Description of OTS Software

### Table 1
| MedAI Inc. |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Document: | MEMO-P01-640 - MX1 OTS (SOUP) Report |  |  |  |  |
| Project Number: | P01 |  |  |  |  |
| To: | File |  |  |  |  |
| From: | Device Software, Systems Integration, Cloud Services |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |
| Revision | Description | DCO # | Approved By | Eff. Date | Digital Key |
| A | Initial Release | 24-188 | EngineeringQuality EngineeringRegulatory Affairs | 2024-04-26 00:00:00 | example.com/ |
| B | Updated to include additional SOUPs as of MX1 SS v3.1.0 release and general reformatting of OTS Report | Refer to ECR-449 |  |  | example.com/ |
| C | Updated to reflect SOUP/OTS SW list as of MX1 SS v3.3.0 release and general reformatting of OTS Report | 24-656 | EngineeringQuality EngineeringRegulatory Affairs | 2025-01-16 00:00:00 | example.com/ |

### Table 2
| Purpose |  |
| --- | --- |
| The purpose of this document is to provide the Off-the-shelf (OTS) Software documentation for the device per the 2023 FDA Guidance Off-The-Shelf Software Use in Medical Devices. |  |
| Scope |  |
| The design specifications in this revision are applicable to MX1 Software System and MedAI Device App release as of 3.3.0 |  |
| Risk Assessment of OTS Software |  |
| Risk Management of OTS Software Items was performed as part of the medical device risk management. See the Risk Assessment. Additional notes about the risk assessment for each OTS Software Item are also included in the tables within the Description of OTS Software on this document. |  |
| Software Testing as Part of Verification and Validation |  |
| The requirements that the OTS Software Items must fulfill, and the verification and validation of these requirements, are described in the Software Requirements Specification |  |

### Table 3
| Remote APT Dependencies |  |
| --- | --- |
| Title | Remote APT Dependencies |
| Manufacturer | Open-source software |
| Version | See SBOM |
| Risk Assessment of OTS Software | Risk assessment of OTS is done as a part of risk assessment for the full device. See RSK-P01-010 for details. |
| Any OTS Software documentation that will be provided to the end user. | N/A as the end-user doesn’t have access |
| Why is this OTS Software appropriate for this medical device? | Most of the OTS software is widely used, industry standard. Implementing the same functionality from scratch would be more error prone than using this OTS library. |
| What are the expected design limitations of the OTS Software? | None |
| What Hardware Specifications are there for the OTS Software to function properly? | The OTS runs on the Nvidia Jetson Xavier NX which is a part of the device. This is the same hardware that is used in validation. |
| What Software Specifications are there for the OTS Software to function properly? | The OTS runs on a Linux system image that is shipped with the device. The end-user does not have access to the Linux system image and therefore cannot be configured. The Linux system image is the same one that's used in validation. |
| What aspects of the OTS Software and system can (and/or must) be installed/configured? | None |
| What steps are permitted (or must be taken) to install and/or configure the product? | None |
| How often will the configuration need to be changed (by the end-user)? | Never |
| What education and training are suggested or required for the user of the OTS Software? | None |
| What measures have been designed into the medical device to prevent the operation of any non-specified OTS Software? | None, as the user doesn’t have control over this OTS software’s runtime environment |
| What is the OTS Software intended to do? | nvidia-l4t-jetson-multimedia-api: A collection of lower-level APIs that support flexible application development on the Nvidia Jetson.nvidia-l4t-3d-core, nvidia-l4t-pva: Package that provides the bootloader, kernel, necessary firmwares, NVIDIA drivers for various accelerators present on Jetson modules.vpi2-dev: A software library that implements computer vision (CV) and image processing (IP) algorithms on several computing Nvidia hardware platforms. curl: A computer software project providing a library (libcurl) and command-line tool (curl) for transferring data using various network protocols.liboping-dev: Software package for pinging multiple hosts in parallel using IPv4 or IPv6 transparently.apt-utils: Package contains commandline utilities related to package management with APT. python3.11: A widely used, high-level programming language used for a wide variety of applications.python3.11-dev: Development files for Python.gcc-11: Widely used, industry standard Linux C compiler.g++-11: Widely used, industry standard Linux C++ compiler.cuda-nvcc-11-4: Software package providing tooling for interfacing with and writing applications for Nvidia GPUs.gpiod: C library and tools for interacting with the linux GPIO character device (gpiod stands for GPIO device).libgpiod-dev: Development files for C library and tools for interacting with the linux GPIO character device (gpiod stands for GPIO device).hashdeep: A widely used program to compute, match, and audit hashsets.libgtk-3-dev: A package that contains the header and development files which are needed for building GTK applications. libnm-dev: A package that manages ethernet, Wi-Fi, mobile broadband (WWAN), and PPPoE devices, and provides VPN integration with a variety of different VPN services.nvidia-vpi: A software library that implements computer vision (CV) and image processing (IP) algorithms on several computing Nvidia hardware platforms.nvidia-vpi-dev: Development files for a soft…[truncated] |
| What are the links with other software including software outside the medical device? | N/A as this software is integrated directly into the system image of the device. |
| Describe testing, verification, and validation of the OTS Software | The OTS software was tested indirectly by the system verification.Note that it is not necessary to identify specific OTS versions as they are configured within the system images. All verification was thus performed with the exact OTS versions that will be used in the final device. |
| Provide the results of the testing | See the following ECRs for the system verification records:ECR-440 - MX1 SW v3.0.0ECR-449 - MX1 SW v3.1.0ECR-470 - MX1 SW v3.2.0ECR-574 - MX1 SW v3.2.1ECR-602 - MX1 SW v3.3.0 |
| Is there a current list of OTS Software problems (bugs) and access to updates? | Yes, the software is open source and has a public bug tracker and updates are provided by the community or could be provided by the manufacturer if needed. Also, cybersecurity vulnerabilities are identified per PLN-P01-066 - MX1 Security Management Plan |
| What measures have been designed into the medical device to prevent the introduction of incorrect versions? | The software runs on the Linux system image that's a part of the device. It uses the same configuration in validation. |
| How will you maintain the OTS Software configuration? | In the system image, the specified versions of the OTS software are pinned. |
| Where and how will you store the OTS Software? | The OTS software is bundled into the system image. |
| How will you ensure proper installation of the OTS Software? | The OTS software is bundled into the system image so no installation instructions are required. |
| How will you ensure proper maintenance and life cycle support for the OTS Software? | Bug fixes and maintenance are provided by the community and version updates can be applied in a controlled manner following our software development plan, including design controls and configuration management. Cybersecurity vulnerabilities are monitored per PLN-P01-066 - MX1 Security Management Plan |
| ODA Dependencies |  |
| Title | MedAI Device App Dependencies |
| Manufacturer | Open Source Software |
| Version | See SBOM |
| Risk Assessment of OTS Software | Risk assessment of OTS is done as a part of risk assessment for the full device. See RSK-P01-010 for details. |
| Any OTS Software documentation that will be provided to the end user. | N/A as the end-user doesn’t have access |
| Why is this OTS Software appropriate for this medical device? | Most of the OTS software is widely used, industry standard. Implementing the same functionality from scratch would be more error prone than using this OTS library. |
| What are the expected design limitations of the OTS Software? | None |
| What Hardware Specifications are there for the OTS Software to function properly? | The ODA is intended for, but not limited to, Galaxy S8 Tablets. Galaxy S8 Tablets were used for validation. |
| What Software Specifications are there for the OTS Software to function properly? | The ODA requires Android 10 or higher to run. The OS software used in validation is Android 14 |
| What aspects of the OTS Software and system can (and/or must) be installed/configured? | Only the finalized ODA apk will be available for installation. |
| What steps are permitted (or must be taken) to install and/or configure the product? | The MedAI Device App will come installed on the MedAI K1 cart and the provided Android Tablet. Additionally the ODA will be available on the Google Play Store for download to Android devices that can support the ODA |
| How often will the configuration need to be changed (by the end-user)? | Never |
| What education and training are suggested or required for the user of the OTS Software? | None |
| What measures have been designed into the medical device to prevent the operation of any non-specified OTS Software? | The ODA apk is precompiled and therefore non-specified OTS Software cannot be added |
| What is the OTS Software intended to do? | micronaut-http-client: Component of Micronaut Framework to supply HTTP requests to services which support Rest API.reactor-core: Library that provides Pub-Sub pattern of communication to modules of the appreactive-streams: Extension to reactor-core lib which add Streams API syntax for Pub-Sub patternslf4j-api: Provides unified Log API to app modulesmicronaut-runtime: A modern, JVM-based, full-stack framework for building modular, easily testable microservice and serverless applicationsmicronaut-inject: Component of Micronaut Framework that provides DI pattern integration for the appjavax.annotation-api: Common annotations for Java Platformjakarta.inject-api: Annotations supporting integration of DI pattern for Java Platformjakarta.annotation-api: Common annotations supporting implementation of Rest API on Java Platformsnakeyaml: Library that supports parsing of yaml data serialization languagemicronaut-core-reactive: Component of Micronaut Framework that provides integration of Pub-Sub patternmicronaut-aop: Component of Micronaut Framework that provides AOP integrationvalidation-api: Common annotations for validation model that can add constraints to the fields, methods, or classes.micronaut-jackson-databind: Component of Micronaut framework that provides integration with Jackson libraryjackson-core: Jackson is a high-performance JSON processor for Java.micronaut-http-client-core: Component of Micronaut Framework that provides a simple initerface for performing HTTP requests.micronaut-websocket: Component of Micronaut Framework that provides base classes to implement WebSocket servermicronaut-http-netty: Component of Micronaut Framework that provides integration with Netty frameworknetty-buffer: Netty is an NIO client server framework which enables quick and easy development of network applications such as protocol servers and clients.micronaut-data-hibernate-jpa: Component of Micronaut Framework that provides integration with H…[truncated] |
| What are the links with other software including software outside the medical device? | N/A as this software is integrated directly into the system image of the device. |
| Describe testing, verification, and validation of the OTS Software | The OTS software was tested indirectly by the system verification.Note that it is not necessary to identify specific OTS versions as they are configured within the system images. All verification was thus performed with the exact OTS versions that will be used in the final device. |
| Provide the results of the testing | See the following ECRs for the system verification records:ECR-440 - MX1 SW v3.0.0ECR-449 - MX1 SW v3.1.0ECR-470 - MX1 SW v3.2.0ECR-574 - MX1 SW v3.2.1ECR-602 - MX1 SW v3.3.0 |
| Is there a current list of OTS Software problems (bugs) and access to updates? | Yes, the software is open source and has a public bug tracker and updates are provided by the community or could be provided by the manufacturer if needed. Also, cybersecurity vulnerabilities are identified per PLN-P01-066 - MX1 Security Management Plan |
| What measures have been designed into the medical device to prevent the introduction of incorrect versions? | The software runs on the Linux system image and Android system that is a part of the device. It uses the same configuration in validation. |
| How will you maintain the OTS Software configuration? | In the ODA apk, the specified OTS versions are pinned. |
| Where and how will you store the OTS Software? | The OTS software is bundled into the system image and ODA. |
| How will you ensure proper installation of the OTS Software? | The OTS software is bundled into the system image and ODA. The only installation required is a single download of a precompiled apk. |
| How will you ensure proper maintenance and life cycle support for the OTS Software? | Bug fixes and maintenance are provided by the community and version updates can be applied in a controlled manner following our software development plan, including design controls and configuration management. Cybersecurity vulnerabilities are monitored per PLN-P01-066 - MX1 Security Management Plan |
| Self Compiled Dependencies |  |
| Title | Self Compiled Dependencies |
| Manufacturer | Open-source software |
| Version | See SBOM |
| Risk Assessment of OTS Software | Risk assessment of OTS is done as a part of risk assessment for the full device. See RSK-P01-010 for details. |
| Any OTS Software documentation that will be provided to the end user. | N/A as the end-user doesn’t have access |
| Why is this OTS Software appropriate for this medical device? | Most of the OTS software is widely used, industry standard. Implementing the same functionality from scratch would be more error prone than using this OTS library. |
| What are the expected design limitations of the OTS Software? | None |
| What Hardware Specifications are there for the OTS Software to function properly? | The OTS runs on the Nvidia Jetson Xavier NX which is a part of the device. This is the same hardware that is used in validation. |
| What Software Specifications are there for the OTS Software to function properly? | The OTS runs on a Linux system image that is shipped with the device. The end-user does not have access to the Linux system image and therefore cannot be configured. The Linux system image is the same one that's used in validation. |
| What aspects of the OTS Software and system can (and/or must) be installed/configured? | None |
| What steps are permitted (or must be taken) to install and/or configure the product? | None |
| How often will the configuration need to be changed (by the end-user)? | Never |
| What education and training are suggested or required for the user of the OTS Software? | None |
| What measures have been designed into the medical device to prevent the operation of any non-specified OTS Software? | None, as the user doesn’t have control over this OTS software’s runtime environment |
| What is the OTS Software intended to do? | cmake: Industry standard method of generating make files, which aids in the build process of C++ projectslibicu: A set of C/C++ ibraries providing Unicode and globalization support for software applications.framos: A software package providing tools and libraries for working with FRAMOS imaging and embedded vision technologies.qt: An application framework that can be used for developing C++ applications with a graphical user interface.graalvm: A high-performance runtime that provides support for multiple programming languages and execution modes. boost: A commonly used collection of peer-reviewed, portable C++ source libraries that work well with the C++ Standard Library, and add easy implementations of desired functionality.xmp: A library for parsing and handling Adobe's Extensible Metadata Platform (XMP) used in digital media.fmt: An open-source formatting library for C++ that provides a fast and safe alternative to built in output functions.opencv: An industry-standard open-source computer vision and machine learning software library containing algorithms for real-time image processing and analysis. |
| What are the links with other software including software outside the medical device? | N/A as this software is integrated directly into the system image of the device. |
| Describe testing, verification, and validation of the OTS Software | The OTS software was tested indirectly by the system verification.Note that it is not necessary to identify specific OTS versions as they are configured within the system images. All verification was thus performed with the exact OTS versions that will be used in the final device. |
| Provide the results of the testing | See the following ECRs for the system verification records:ECR-440 - MX1 SW v3.0.0ECR-449 - MX1 SW v3.1.0ECR-470 - MX1 SW v3.2.0ECR-574 - MX1 SW v3.2.1ECR-602 - MX1 SW v3.3.0 |
| Is there a current list of OTS Software problems (bugs) and access to updates? | Yes, the software is open source and has a public bug tracker and updates are provided by the community or could be provided by the manufacturer if needed. Also, cybersecurity vulnerabilities are identified per PLN-P01-066 - MX1 Security Management Plan |
| What measures have been designed into the medical device to prevent the introduction of incorrect versions? | The software runs on the Linux system image that's a part of the device. It uses the same configuration in validation. |
| How will you maintain the OTS Software configuration? | In the system image, the specified versions of the OTS software are pinned. |
| Where and how will you store the OTS Software? | The OTS software is bundled into the system image. |
| How will you ensure proper installation of the OTS Software? | The OTS software is bundled into the system image so no installation instructions are required. |
| How will you ensure proper maintenance and life cycle support for the OTS Software? | Bug fixes and maintenance are provided by the community and version updates can be applied in a controlled manner following our software development plan, including design controls and configuration management. Cybersecurity vulnerabilities are monitored per PLN-P01-066 - MX1 Security Management Plan |
| Git Installed Dependencies |  |
| Title | Git Installed Dependencies |
| Manufacturer | Open-source software |
| Version | See SBOM |
| Risk Assessment of OTS Software | Risk assessment of OTS is done as a part of risk assessment for the full device. See RSK-P01-010 for details. |
| Any OTS Software documentation that will be provided to the end user. | N/A as the end-user doesn’t have access |
| Why is this OTS Software appropriate for this medical device? | Most of the OTS software is widely used, industry standard. Implementing the same functionality from scratch would be more error prone than using this OTS library. |
| What are the expected design limitations of the OTS Software? | None |
| What Hardware Specifications are there for the OTS Software to function properly? | The OTS runs on the Nvidia Jetson Xavier NX which is a part of the device. This is the same hardware that is used in validation. |
| What Software Specifications are there for the OTS Software to function properly? | The OTS runs on a Linux system image that is shipped with the device. The end-user does not have access to the Linux system image and therefore cannot be configured. The Linux system image is the same one that's used in validation. |
| What aspects of the OTS Software and system can (and/or must) be installed/configured? | None |
| What steps are permitted (or must be taken) to install and/or configure the product? | None |
| How often will the configuration need to be changed (by the end-user)? | Never |
| What education and training are suggested or required for the user of the OTS Software? | None |
| What measures have been designed into the medical device to prevent the operation of any non-specified OTS Software? | None, as the user doesn’t have control over this OTS software’s runtime environment |
| What is the OTS Software intended to do? | httplib: A simple, header-only HTTP/HTTPS library for C++.u8g2-arm-linux: A graphics library for monochrome displays, optimized for ARM Linux devices.eigen3: A high-performance C++ template library for linear algebra, including matrices, vectors, numerical solvers, and related algorithms.flatbuffers: An efficient cross-platform serialization library for C++, developed by Google, designed for performance-critical applications. |
| What are the links with other software including software outside the medical device? | N/A as this software is integrated directly into the system image of the device. |
| Describe testing, verification, and validation of the OTS Software | The OTS software was tested indirectly by the system verification.Note that it is not necessary to identify specific OTS versions as they are configured within the system images. All verification was thus performed with the exact OTS versions that will be used in the final device. |
| Provide the results of the testing | See the following ECRs for the system verification records:ECR-440 - MX1 SW v3.0.0ECR-449 - MX1 SW v3.1.0ECR-470 - MX1 SW v3.2.0ECR-574 - MX1 SW v3.2.1ECR-602 - MX1 SW v3.3.0 |
| Is there a current list of OTS Software problems (bugs) and access to updates? | Yes, the software is open source and has a public bug tracker and updates are provided by the community or could be provided by the manufacturer if needed. Also, cybersecurity vulnerabilities are identified per PLN-P01-066 - MX1 Security Management Plan |
| What measures have been designed into the medical device to prevent the introduction of incorrect versions? | The software runs on the Linux system image that's a part of the device. It uses the same configuration in validation. |
| How will you maintain the OTS Software configuration? | In the system image, the specified versions of the OTS software are pinned. |
| Where and how will you store the OTS Software? | The OTS software is bundled into the system image. |
| How will you ensure proper installation of the OTS Software? | The OTS software is bundled into the system image so no installation instructions are required. |
| How will you ensure proper maintenance and life cycle support for the OTS Software? | Bug fixes and maintenance are provided by the community and version updates can be applied in a controlled manner following our software development plan, including design controls and configuration management. Cybersecurity vulnerabilities are monitored per PLN-P01-066 - MX1 Security Management Plan |
