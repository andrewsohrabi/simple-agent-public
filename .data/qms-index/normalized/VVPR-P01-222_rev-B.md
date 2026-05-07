# VVPR-P01-222 Rev B: MX1 Cloud Service Provider Verification via Inspection Protocol and Report

## Metadata
- Document ID: VVPR-P01-222
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-222 - MX1 Cloud Service Provider Verification via Inspection Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-222 - MX1 Cloud Service Provider Verification via Inspection Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this document is to demonstrate that the Cloud Service Providers (CSPs) used by the MX1 Portable X-ray System’s Software System meet rigorous standards for data protection, compliance, and risk management. This includes demonstrating adherence to industry best practices for confidentiality, integrity, and availability of device-related data, as well as ensuring compliance with applicable cybersecurity standards.
OBJECTIVE AND SCOPE
Mender, Tailscale, and Google Cloud Platform (specifically, Google Logging services) are the three cloud service providers used by the MX1 Software System. This objective of this protocol is intended to evaluate the adequacy of those services against requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.2.1 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. F
IFU-MX1 - Instructions for Use, Rev. G
MATERIALS
None
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Suite 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
For each of the specifications in the tables below, review and record the associated reference material and record the results. If the reference is insufficient to verify the specification, images or pictures from a completed unit may be used. Note that the SRS ID references and the associated requirements were taken directly from MEMO-P01-630 Rev. F
Table 1. Mender - Specifications to be Verified via Inspection
Table 2. Tailscale- Specifications to be Verified via Inspection
Table 3. Google Cloud Platform (Google Logging) - Specifications to be Verified via Inspection
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Reference and Evidence/Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Evidence/Result” and “Pass/Fail” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
None
RESULTS
Table 1. Mender - Specifications to be Verified via Inspection
Table 2. Tailscale- Specifications to be Verified via Inspection
Table 3. Google Cloud Platform (Google Logging) - Specifications to be Verified via Inspection
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
No anomalies were found during the course of testing.
LIST OF APPENDICES
Appendix 1 and 4 - Verification Evidence as Specified in Results Table 1.
REPORT APPROVAL
Digital Key:
example.com/
Appendix 1. RBAC - Mender
Appendix 2. Scale Testing - Mender
Appendix 3. HTTPS - Mender
Appendix 4. Security Defense Mechanisms - Mender
Appendix 5. RBAC - Tailscale
Appendix 6. Scale Testing - Tailscale
Appendix 7. Security Audit - Tailscale
Appendix 8. Principle of Least Privilege - Google Logging
Appendix 10. Third-Party Testing - Google Logging
Appendix 11. DDOS Protection - Google Logging
Appendix 12. Vulnerability Scanning - Google Logging
Appendix 13. HTTPS - Tailscale
Appendix 14. Security Mechanisms - Tailscale
Appendix 15. Vulnerability Scanning - Tailscale

### Table 1
| SRS ID | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference Mender documentation |  |  |  |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference Mender documentation |  |  |  |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference Mender documentation |  |  |  |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference Mender documentation |  |  |  |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference Mender documentation |  |  |  |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference Mender documentation |  |  |  |

### Table 2
| Req# | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference Tailscale documentation |  |  |  |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference Tailscale documentation |  |  |  |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference Tailscale documentation |  |  |  |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference Tailscale documentation |  |  |  |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference Tailscale documentation |  |  |  |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference Tailscale documentation |  |  |  |

### Table 3
| Req# | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference Google Logging documentation |  |  |  |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference Google Logging documentation |  |  |  |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference Google Logging documentation |  |  |  |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference Google Logging documentation |  |  |  |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference Google Logging documentation |  |  |  |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference Google Logging documentation |  |  |  |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 09 Oct 2024 | 24-590 |

### Table 5
| SRS ID | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference Mender documentation | Refer to “Role-Based Access Control” in official Mender documentation: example.com/ | “Role-Based Access Control” includes explanation for role-based access features supported by Mender as well as instructions on how to configure roles See Appendix 1 Verified by AM 09OCT24 | PASS |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference Mender documentation | Refer to Mender engineering support site: example.com/ | Mender is regularly scale tested to ensures that “it can scale up to hundreds of thousands devices per customer” as per engineering FAQ. See Appendix 2 Verified by AM 09OCT24 | PASS |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference Mender documentation | Refer to VVPR-P01-222, Attachment 1 - Executive Summary Security Assessment of Mender, 2023-12-19 | Mender uses 3rd party security consultants to perform penetration tests/security assessments of all products See Attachment 1 as an executive summary of 3rd party audit findings from 12-19-2023 Verified by AM 09OCT24 | PASS |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference Mender documentation | Refer to Mender engineering support site: example.com/ | Mender clients on devices (e.g. the MX1) can only communicate to the Mender server via HTTPS. See Appendix 3 Verified by AM 09OCT24 | PASS |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference Mender documentation | Refer to “Security” in official Mender documentation:: example.com/ | Mender provides multiple security mechanisms (e.g. no open ports, key-based client authentication, non-configurable rate limits) on both the client and server-side to mitigate against high-volume attacks on linked devices See Appendix 4 Verified by AM 09OCT24 | PASS |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference Mender documentation | Refer to VVPR-P01-222, Attachment 2 - Evidence of Hosted Mender vulnerability handling and vulnerability scans | Mender generally scans on a monthly basis and attempts to resolve dependency vulnerabilities within 30 days For information and evidence of dependency updates and vulnerability scans, see the attachment, see Attachment 2 Verified by AM 09OCT24 | PASS |

### Table 6
| Req# | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference Tailscale documentation | Refer to “User roles” in official Tailscale documentation: example.com/ | “User roles” includes explanation for role-based access features supported by Tailscale as well as instructions on how to configure roles See Appendix 5 Verified by AM 09OCT24 | PASS |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference Tailscale documentation | Refer to “Performance Best Practices” in official Tailscale documentation: example.com/ | Tailscale conducts various levels of performance testing and supports thousands of teams and products See Appendix 6 Verified by AM 09OCT24 | PASS |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference Tailscale documentation | Refer to Tailscale Security Page: example.com/ | Tailscale undergoes security audits with the security firm Latacora See Appendix 7 Verified by AM 09OCT24 | PASS |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference Tailscale documentation | Refer to Tailscale Security Page: example.com/ | Tailscale offers end-to-end encryption See Appendix 13 Verified by AM 09OCT24 | PASS |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference Tailscale documentation | Refer to Tailscale Security Page: example.com/ | Tailscale offers multiple security defense mechanisms to mitigate against large-volume attacks, including Access Control Lists, See Appendix 14 Verified by AM 09OCT24 | PASS |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference Tailscale documentation | Refer to Tailscale Security Page: example.com/ | Tailscale publishes regular security bulletins disclosing any security issues and/or vulnerabilities See Appendix 15 Verified by AM 09OCT24 | PASS |

### Table 7
| Req# | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference Google Logging documentation | Refer to example.com/ | Google Cloud allows users to implement the Principle of Least Privilege. See Appendix 8 Verified by GC 09OCT24 | PASS |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference Google Logging documentation | Refer to example.com/ | Google Logging operates on GCP’s global infrastructure, which has been tested to scale for millions of users and devices Verified by GC 09OCT24 | PASS |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference Google Logging documentation | Refer to example.com/ | Google Logging operates on GCP which undergoes regular third-party security audits and complies with a wide range of standards, including ISO/IEC 27001, SOC 2, etc. See Appendix 10 Verified by GC 09OCT24 | PASS |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference Google Logging documentation | Refer to example.com/ | Google Logging enforces HTTPS See Appendix 10 Verified by GC 09OCT24 | PASS |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference Google Logging documentation | Refer to example.com/ | Google Logging operates on GCP which has traffic management features, safeguarding the service from high-volume attacks. See Appendix 11 Verified by GC 09OCT24 | PASS |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference Google Logging documentation | Refer to example.com/ | Google Logging operates on GCP which performs regular vulnerability scans as part of its security operations. GCP is FedRAMP complaint that requires all vulnerabilities to be addressed within 90 days. See Appendix 12 Verified by GC 09OCT24 | PASS |

### Table 8
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-574 |  |
