# MEMO-P01-671 Rev A: MX1 Device and App Software Usability Formative Evaluation

## Metadata
- Document ID: MEMO-P01-671
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-671 - MX1 Device and App Software Usability Formative Evaluation_A-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-671 - MX1 Device and App Software Usability Formative Evaluation_A-signed.docx
- Extraction warnings: none

## Extracted Content
Study Purpose
This document describes Formative Usability Evaluation of the v3.1.0-alpha software release for the MX1 Portable X-Ray System. This study is intended to gather feedback from MedAI employees not involved in the design or implementation of new software features in order to inform the design process.
Objective
This Formative Usability effort aims to evaluate the design and implementation of v3.1.0-alpha software release. The collected feedback will be used to evaluate these newly added features and any necessary changes, and will be considered for and incorporated into future execution of the design.
References
IEC 60601-1-6 Edition 3.2 2020-07 General requirements for basic safety and essential performance – Collateral standard: Usability
IEC 62366-1: Edition 1.1 2020-06, Application of usability engineering to medical devices
IEC 62366-2: 2016, Guidance on application of usability engineering to medical devices
PLN-P01-064 Rev. B - MX1 Usability Plan
Guidance for Industry and FDA Staff - Applying Human Factors and Usability Engineering to Medical Devices (2016)
Materials
An MX1 Portable X-ray System unit assembled for design verification/validation purposes shall be used for this test. This unit will be substantially equivalent to that of the final units marketed, utilizing equivalent materials, manufacturing processes, instructions and training.
This MX1 system, with software version v3.1.0-alpha, consists of the:
E1 Emitter
C1 Cassette
T1 tablet loaded with the MedAI Imaging App (APP), Software Version v3.1.0-alpha
Methods
Test Environment
This study was conducted at MedAI Headquarters in Springfield, Illinois.
Test Participants
Test participants consisted of MedAI personnel who have not had prior experience with the current device and app software version.
Test Personnel
Test personnel consisted of MedAI personnel who have prior knowledge of MX1 System design and protocols.
Procedure
MedAI test personnel will walk test participants through the new v3.1.0-alpha software features, allowing them to interact with the device and app to perform expected aspects of use. Participants will be facilitated for feedback including any undue complexity in the way information is presented, confusion involving the layout or flow of the UI, or difficulty using each function. All feedback will be recorded.
List of new software features to run through:
Device
Viewfinder - Mode Indicators
App
Collimation Menu, including Pediatric Filter selection
Cassette Settings Menu, including Audible Signals, Exam Timer, Maximum Allowable Air Kerma Rate, and Mode Limitation
Cassette Config Drawer, including Image Deletion
Exam Setup Page, including Device Mode indication, and Image Storage Capacity
Acquisition Page, including Device Mode selection, Collimation Selector, Low Dose Mode toggle, Cumulative Dose indicator, Dose Area Product indicator, Current Dose Rate indicator, Loading Timer, Stop X-rays button, and DDR limitations/warnings during capture
Device Mode indication on Exam Setup Page and Device Mode selection on Acquisition Page
Image Storage Indication on Exam Setup Page
Emergency Mode
Acceptance Criteria
There were no defined acceptance criteria for this evaluation; rather, this study serves as an assessment of the proposed software features and implementation of these features on the device and app to identify undue complexity within the use process. The data collected will be evaluated to inform further design iterations.
Results
A total of 3 test participants were collected for the study, and all feedback is included in Table 1.
Table 1. Participant Feedback
Conclusion
Participants understood and confirmed the purpose, form, and function of each new software element. While many names, text, or display form (flat box for a selectable option) were not immediately obvious to the untrained participants, a brief explanation corrected any misunderstanding. From there, all elements were confirmed to meet expectations on form and function. There was strong confirmation on the placement of these elements within the workflow.
Substantial improvements may include:
Adjusting the slider scale for Loading Time Limiter to a range of 1s to 300s or so, or converting the element to a number input
Enlarging the “mute” button for Loading Time Limiter on the Acquisition page to be easier to touch with a finger
These substantial improvements shall be communicated to the software development team, along with the results of this evaluation, for future risk and requirements assessment for development.
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Device/App Section | Feedback |
| --- | --- |
| Cassette Settings | For Mode Limitation: Is mode enable/disable persistent across device restarts? For the Loading Time Limiter and Dose Rate Limiter: Kinda hard to to tell with the sliders what you’re selecting until you’ve selected something Ability to type in a number would be better instead of sliders Have a wheel selector instead of sliders General: Easy to understand, makes sense |
| Collimation Page | For Pediatric Filter Toggle: Easy, makes sense |
| Exam Setup Page | For Device Mode Selection: The button doesn’t immediately jump out Having Emergency Mode and Device Mode buttons next to each other is a bit confusing, names are too similar For Image Storage: Two out of three participants thought it indicated how many images were currently stored, not how many were left Having the number displayed as a fraction of total capacity or as a bar that empties would make it more obvious Possibly changing the name from “Image Storage” to “Image Storage Capacity” or something more obvious would also be helpful Otherwise, the indication jumps out |
| Acquisition Page | For Low Dose Mode Toggle: Make the Low Dose Mode toggle obviously unselectable when in modes other than Fluoro Mode Just "Low" on the button isn’t really descriptive enough if you don’t already know what it is For Collimation Selector: Just the icon on the button isn’t really descriptive enough if you don’t already know what it is For Loading Time Limiter Indication: The icon for turning it off and on is a bit small, seems like it could be accidentally pressed For DDR capture warning: One out of three participants didn’t really notice the warning while the other two thought it was obvious General: In the bottom area where previously captured images go, maybe put something there to more obviously indicate its empty when there are no previously captured images |
| Viewfinder | For Device Mode Indication: Its obvious, makes sense |
| Cassette Config | For Image Deletion: Display the number of images currently stored Otherwise seems useful, makes sense |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering | 12 May 2024 | 24-231 |
