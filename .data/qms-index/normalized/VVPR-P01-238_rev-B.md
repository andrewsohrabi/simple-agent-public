# VVPR-P01-238 Rev B: MX1 Software System ODA Fuzz Testing v3.4.0 Protocol and Report

## Metadata
- Document ID: VVPR-P01-238
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: v3.4.0
- Source filename: VVPR-P01-238 - MX1 Software System ODA Fuzz Testing v3.4.0 Protocol and Report_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-238 - MX1 Software System ODA Fuzz Testing v3.4.0 Protocol and Report_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to demonstrate that MX1 Portable X-ray System’s Software System meets the requirements as stated in MEMO-P01-630 - MX1 Software Requirements Specification as it relates to the following feature:
SRS-44.3 - The SS shall utilize Application Programming Interfaces (APIs) that are resistant to web application security threats
OBJECTIVE AND SCOPE
The primary objective of this study is to verify the software system-level requirements set by MedAI for the MX1 Software System (SS) and MedAI Device App (ODA) as documented in MEMO-P01-630 - MX1 Software Requirements Specification as part of the v3.4.0 release.
The scope of this study is limited to the verification of ability of MX1 Software System to handle unexpected or malformed data from user facing text input data fields and forms in ODA’s UI.
The scope of this study is limited to use cases when MX1’s Operator fills in information necessary to start the exam and administrate/configure the device.
Refer to the Appendices for an overview of all ODA text input data fields being tested in this protocol
Note that this study is being conducted to provide support for clarifications made during the FDA Interactive Request Responses for K241567. No changes or modifications have been made to the test features as part of MX1 SS v3.4.0.
REFERENCES
MEMO-P01-630 - MX1 Software Requirements Specification, Rev. H
MATERIALS
MX1 Software System v3.4.0
Flutter SDK 3.24.5 or higher
Docker version 24.0.7 or higher
Test Machine (see Test Setup section below for further details)
SAMPLE SIZE
This is a software verification test, and therefore will utilize a sample size of 1.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI office building: 100 Main Street Ste 700 Springfield, IL 60001. To be performed by trained MedAI medical engineering staff.
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Test Setup
Test Machine  - Github Server or Local Runner, or any other machine (virtual or physical) capable of running Docker and Flutter applications (e.g. laptop)
Docker Software 24.0.7 or higher
Docker is a software platform that allows for the deployment of software applications in standardized OS-virtualized packages called containers. In this protocol, a Docker container will host MX1 software components to allow for running these tests virtually.
Flutter SDK 3.24.5 or higher
Software tools used to compile and run tests and emulate behaviour of real tablet devices.
Test Procedure
On a test machine, start the Docker container hosting the MX1 software components
Checkout the mcx-tablet repository using Git
In mcx-tablet, navigate into the medai-device-app module directory
Run the following commands to perform significant amounts (not less than 50) of operations that fill the ODA text input data fields with randomly generated malformed data.  Note that these commands will automatically launch ODA in the virtual environment prior to test execution.
flutter test test/fuzz_tests/users_form_fuzz_test.dart -d windows --dart-define 'FUZZ_ITERATIONS=50'
flutter test test/fuzz_tests/dicom_server_form_fuzz_test.dart -d windows --dart-define 'FUZZ_ITERATIONS=50'
flutter test test/fuzz_tests/study_form_fuzz_test.dart -d windows --dart-define 'FUZZ_ITERATIONS=50'
Wait for the test executions to complete
Collect test logs from the stdout of each execution
Review the test execution log summarizing the data. Investigate and document exceptions (failures or other reported faults), if any
Data Analysis
The test report generated shall be reviewed to determine whether ODA experienced any fault conditions during the course of testing.
If any faults occur, each occurrence shall be assessed to determine and document the root cause.
ACCEPTANCE CRITERIA
The acceptance criteria for passing this verification test shall be as follows:
Output log of test executions indicates that ODA did not crash or experience any faults while processing all the provided malformed input data.
APPENDICES
Appendix A through E - All ODA screens with text input data fields. Note that the text fields under test are listed under each ODA screenshot.
PROTOCOL APPROVAL
Digital Key:
example.com/
Appendix A: “Start Exam” Screen
Appendix B: “Add/Edit Users” Entry Form in “Users” Screen
Appendix C: “Add/Edit DICOM Server” Entry Form in “DICOM Servers” Screen
Appendix D: “Network Settings” Screen
Appendix E: “Patient Data Review” Entry Forms in “Send to Drive” and “Send to PACS” Screens
REPORT SECTION
Recorded By: PAVEL SITNIKOVDate: 12/4/24
PROTOCOL DEVIATIONS
None
DEVICES, COMPONENTS, OR EQUIPMENT USED
MX1 Software System v3.4.0
Test Machine - A Windows 10 Laptop was used to run this protocol using Docker version 27.2.0 and Flutter SDK 3.24.5.
RESULTS
Refer to these attachments for the full test outputs generated during the execution of this protocol. These test outputs include all the generated malformed data inputs used to test against ODA.
VVPR-P01-238, Attachment 1 - ODA Fuzz Test Results, users_form_fuzz_test, MX1 SS v3.4.0
VVPR-P01-238, Attachment 2 - ODA Fuzz Test Results, study_form_fuzz_test, MX1 SS v3.4.0
VVPR-P01-238, Attachment 3 - ODA Fuzz Test Results, dicom_server_form_fuzz_test, MX1 SS v3.4.0
The users_form_fuzz_test tests the text input data fields for the ODA screen in Appendix B. The users_form_fuzz_test execution result summary is as follows:
22:53 +50: loading C:/work/git/mcx-tablet/medai-device-app/test/fuzz_tests/users_form_fuzz_test.dart
[I]  Finished 50 iterations. Recorded 0 errors.
[I]  Total execution time: 0:22:50.557723. Average iteration time: 0:00:04.449000.
22:53 +50: All tests passed!
The study_form_fuzz_test tests the text input data fields for the ODA screen in Appendixes A and E. The study_form_fuzz_test execution result summary is as follows:
21:10 +50: loading C:/work/git/mcx-tablet/medai-device-app/test/fuzz_tests/study_form_fuzz_test.dart
[I]  Finished 50 iterations. Recorded 0 errors.
[I]  Total execution time: 0:21:06.689050. Average iteration time: 0:00:02.477000.
21:10 +50: All tests passed!
The dicom_server_form_fuzz_test tests the text input data fields for the ODA screen in Appendixes C and D. The dicom_server_form_fuzz_test result summary is as follows:
22:32 +50: loading C:/work/git/mcx-tablet/medai-device-app/test/fuzz_tests/dicom_server_form_fuzz_test.dart
[I]  Finished 50 iterations. Recorded 0 errors.
[I]  Total execution time: 0:22:29.316903. Average iteration time: 0:00:04.109000.
22:32 +50: All tests passed!
As shown by test logs above, no failures were reported during the testing.
CONCLUSION
Overall Result:.
Pass
Fail
Other: _______
There were 0 recorded faults reported during the execution of this protocol.
The MedAI Device App and all other MX1 software components continued to perform as intended and did not crash during the course of testing.
ATTACHMENTS
VVPR-P01-238, Attachment 1 - VVPR-P01-238, Attachment 1 - ODA Fuzz Test Results, users_form_fuzz_test, MX1 SS v3.4.0
VVPR-P01-238, Attachment 2 - VVPR-P01-238, Attachment 2 - ODA Fuzz Test Results, study_form_fuzz_test, MX1 SS v3.4.0
VVPR-P01-238, Attachment 3 - VVPR-P01-238, Attachment 3 - ODA Fuzz Test Results, dicom_server_form_fuzz_test, MX1 SS v3.4.0
REPORT APPROVAL
Digital Key:
example.com/
VVPR-P01-238, Attachment 1 - ODA Fuzz Test Results, users_form_fuzz_test, MX1 SS v3.4.0
VVPR-P01-238, Attachment 2 - ODA Fuzz Test Results, study_form_fuzz_test, MX1 SS v3.4.0
VVPR-P01-238, Attachment 3 - ODA Fuzz Test Results, dicom_server_form_fuzz_test, MX1 SS v3.4.0

### Table 1
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 04 Dec 2024 | 24-698 |

### Table 2
| Relevant Text Data Input Fields in “Start Exam” Screen |
| --- |
| Text Input Data Field |
| Patient First Name |
| Patient Last Name |
| Patient Identifier |
| Patient Date of Birth |
| Description and Notes |

### Table 3
| Text Data Input Fields in “Add/Edit Users” Screen |
| --- |
| Text Input Data Field |
| User Identifier |
| User First Name |
| User Last Name |
| User Email |
| User Phone Number |

### Table 4
| Text Data Input Fields in “Add/Edit DICOM Servers” Screen |
| --- |
| Text Input Data Field |
| DICOM Server Name |
| DICOM Server Location |
| DICOM Server IP Address |
| DICOM Server Port Number |
| DICOM Server Application Entity Title |

### Table 5
| Text Data Input Fields in “Network Settings” Screen |
| --- |
| Text Input Data Field |
| WiFi Network Password |

### Table 6
| Text Data Input Fields in “Send to Drive/PACS” Screen |
| --- |
| Text Input Data Field |
| Patient First Name |
| Patient Last Name |
| Patient Identifier |
| Patient Date of Birth |
| Description and Notes |

### Table 7
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Engineering Quality Engineering Regulatory Affairs | Refer to ECR-643 |  |
