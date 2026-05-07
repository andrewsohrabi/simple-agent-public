# VVPR-P01-231 Rev B: MX1 Cloud Service Provider Verification via Inspection BindPlane Protocol and Report

## Metadata
- Document ID: VVPR-P01-231
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-231 - MX1 Cloud Service Provider Verification via Inspection BindPlane Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-231 - MX1 Cloud Service Provider Verification via Inspection BindPlane Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this document is to demonstrate that the Cloud Service Providers (CSPs) used by the MX1 Portable X-ray System’s Software System meet rigorous standards for data protection, compliance, and risk management. This includes demonstrating adherence to industry best practices for confidentiality, integrity, and availability of device-related data, as well as ensuring compliance with applicable cybersecurity standards.
OBJECTIVE AND SCOPE
BindPlane is a telemetry-collecting cloud service provider used by the MX1 Software System. It is an observability pipeline that provides the ability to collect, refine, and ship metrics, logs, and traces to any destination. In the MX1 devices, it is responsible for configuring and routing system logs from the devices to Google Cloud Logging. This objective of this protocol is intended to evaluate the adequacy of those services against requirements set by MedAI for the MX1 Software System and MedAI Device App as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.3.0 release.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. G
MATERIALS
None
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Suite 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
For each of the specifications in the tables below, review and record the associated reference material and record the results. If the reference is insufficient to verify the specification, images or pictures from a completed unit may be used. Note that the SRS ID references and the associated requirements were taken directly from MEMO-P01-630 Rev. G
Table 1. Mender - Specifications to be Verified via Inspection
Data Analysis
All of the verification tests shall be treated as attribute and marked with either Pass or Fail, with required outputs entered into the Reference and Evidence/Result column or as an attachment to the report. No further data analysis is required for this procedure.
ACCEPTANCE CRITERIA
The acceptance criteria for this protocol is 100% PASS for all requirements per the expected results documented in the “Evidence/Result” and “Pass/Fail” column.
PROTOCOL APPROVAL
Digital Key:
example.com/
REPORT SECTION
PROTOCOL DEVIATIONS
Table 1 was retitled to reference BindPlane instead of Mender. Incorrect title was a result of a clerical error.
DEVICES, COMPONENTS, OR EQUIPMENT USED
None
RESULTS
Table 1. BindPlane - Specifications to be Verified via Inspection
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
Appendix 1. RBAC - BindPlane
Appendix 2. Scale Testing - BindPlane/Google Cloud
Appendix 3. Security Audits - BindPlane
Appendix 4. HTTPS - BindPlane

### Table 1
| SRS ID | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference BindPlane documentation |  |  |  |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference BindPlane documentation |  |  |  |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference BindPlane documentation |  |  |  |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference BindPlane documentation |  |  |  |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference BindPlane documentation |  |  |  |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference BindPlane documentation |  |  |  |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 01 Nov 2024 | 24-629 |

### Table 3
| SRS ID | Specification/Acceptance Criteria | Reference | Evidence/Result | PASS/FAIL |
| --- | --- | --- | --- | --- |
| SRS-44.5 | CSPs must support the configuration of role-based access controls following the principle of least privilege as well as instructions for implementing them Reference BindPlane documentation | Refer to “Role-Based Access Control” in official Bindplane documentation: example.com/ | “Role-Based Access Control” includes explanation for role-based access features supported by Bindplane as well as instructions on how to configure roles See Appendix 1 Verified by RN 01NOV24 | PASS |
| SRS-44.6 | CSPs must be scale tested to ensure it can scale to support thousands of users and devices connecting to it Reference BindPlane documentation | Refer to Bindplane engineering support site: example.com/ Refer to Cloud Logging Documentation: example.com/ | Log data doesn’t pass through Bindplane, it is directly sent to Google Cloud. See Appendix 2 Verified by RN 01NOV24 | PASS |
| SRS-44.7 | CSP must have processes to continuously go through third-party security audits and address issues as they arise Reference BindPlane documentation | Refer to Google’s Compliance Resource Center Documentation: example.com/ | We self-host BindPlane on Google Cloud Platform. Google Cloud Platform regularly goes through 3rd third audits. See Appendix 3 Verified by RN 01NOV24 | PASS |
| SRS-44.8 | CSPs must use HTTPS for all connections Reference BindPlane documentation | Refer to BindPlane certificate hosted on MedAI instance of Google Cloud Platform in Appendix 4 | We perform SSL Termination at the load balancer, ensuring all connections are over HTTPS See Appendix 4 Verified by RN 01NOV24 | PASS |
| SRS-44.9 | CSPs must provide mechanisms to safeguard against high-volume attacks, such as DDoS through traffic management and access controls Reference BindPlane documentation | Refer to Bindplane engineering support site: example.com/ Refer to Cloud Logging Documentation: example.com/ | Log data doesn’t pass through Bindplane, it is directly sent to Google Cloud. See Appendix 2 Verified by RN 01NOV24 | PASS |
| SRS-44.10 | CSPs must scan for vulnerabilities and have at least a 90-day timeline to resolution with customer notification Reference BindPlane documentation | Refer to Bindplane engineering support site: example.com/ Refer to PLN-P01-066 - MX1 Security Management Plan, Rev D section Vulnerability Scanning for Self-Hosted OTS | We self-host Bindplane and check for and apply updates regularly. See Appendix 3 Verified by RN 01NOV24 | PASS |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-602 |  |
