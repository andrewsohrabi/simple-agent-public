# MEMO-P01-412 Rev B: P01 Use Cases

## Metadata
- Document ID: MEMO-P01-412
- Revision: B
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-412 - P01 Use Cases_B-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-412 - P01 Use Cases_B-signed.docx
- Extraction warnings: none

## Extracted Content
Purpose
A primary design feature for the MX1 is battery operation. Batteries have a usable time before requiring charging that relates to 1) battery capacity and 2) the usage of the device. The power that is consumed by each component (Emitter / Cassette) to run their respective functions for each of the device states (Run, Idle, etc.)
The Engineering Team has requested the User Needs Team provide guidance on appropriate use cases and working time needed for the anticipated markets. This guidance will guide the engineering team to design towards appropriate device capabilities and provide the basis for testing.
Scope
The descriptions and decisions related to battery life in anticipated markets are only in relation to MX1 and its specific use cases. The guidance from this document and the User Needs Team is and always will be limited by the research and evaluation done by the team. Furthermore, as research and evaluation continue to develop, the descriptions, decisions, and guidance will continue to develop.
Market Breakdowns
Emergency Medical Services (EMS)
Interviews
2021-09-28 Casey Compton
Notes and Assumptions
Per IEC60601-1-12:2014+AMD1:2020, the EMS ENVIRONMENT includes:
responding to and providing life support at the scene of an emergency to a patient reported as experiencing injury or illness in a pre-hospital setting, and transporting the patient, while continuing such life support care, to an appropriate professional healthcare facility for further care.
providing monitoring, treatment or diagnosis during transport between professional healthcare facilities.
The MX1 device is intended to be used at the scene of an emergency in a pre-hospital setting It is not intended for use in vehicles, including cars, buses, trains, boats, planes, helicopters or during patient transport. The analysis of the EMS use case, as described below, only applies to MX1 use at the scene of an emergency.
EMS (Emergency Medical Services) respond to emergency situations with limited but fast treatment options. Their time frame for action including uses for x-ray modalities during these situations is quick, lasting only around 15 minutes onsite. Situations potentially involve two or more patients, who may need multiple radiographs each in a worst case scenario. EMS has almost no precedent for x-ray use (apart from stroke CT units, which are highly specialized). However, in an effort to accommodate use that we cannot anticipate, three full-power DDR cycles and nine images will be targeted for every situation, corresponding to three patients with a DDR cycle and three images each.
Situations are the on-scene part of a larger Response, which involves 1) travel to the location, 2) the situation activity, and 3) travel to a hospital or care center. Responses involving x-ray use will likely involve patient travel to a hospital after MX1 use. Travel time to a Situation is conservatively estimated to be 15 minutes. Activity at a Situation, which incorporates x-ray use described above, is conservatively estimated to be 15 minutes. During a 15-minute Situation where the x-ray machine is used, the unit is assumed to be on and actively tracking or attempting to track
While traveling to and from a Situation, the unit cannot be charged or used and will therefore be turned off.. Therefore, the unit is assumed to be off during travel. An ambulance might be handling responses constantly for 24 hours, though we will assume that they can swap units every shift or 12 hours for a fully charged unit.  A use case can then be written:
Ambulance Use
Travel to scene for15 minutes, device remains off
Device powers on and captures 3 DDRs and 9 images over 15 minutes then powered off
Travel to hospital for 15 minutes, device remains off
Repeat for 12 hours
Hospital Emergency Department
Interviews
2021-09-28 Lindsey Hartfield
2021-08-24 Lindsey Hartfield
2021-08-03 Marlowe Quinn
2021-12-13 Marlowe Quinn
2021-12-09 Morgan Whitfield
Notes and Assumptions
The Hospital Emergency Department is a difficult environment for devices because of the 24-hour operation and the fast-paced workflows that may speed or slow down drastically. Large ERs like Grady hospital will likely see a consistent throughput and very little time for rest or charging devices, and that is the case we will take. If we can design a device that keeps up with the usage needs of a Level 1 Trauma center like Grady, it will likely work for any emergency department.
A device likely will be used and stationed specifically at the ER, used by either a Rad Tech or a Doctor. The user will simply pass it to the next shift user, so we can expect 24-hour usage of the device. Some interviewees have described a typical “lull” period where many devices are charged for around an hour, but these periods occur when fewer patients walk in, which cannot be anticipated. Here, we will describe the current status quo x-ray device use, then the workflow that MedAI might pursue.
Physicians typically see a patient every 10 minutes when the ER is busy. Certainly not every patient needs x-rays, but here we will assume that every other patient needs images. 3 images per patient is very standard, and we’re going to assume that each exam also includes a DDR to account for the chance that a physician utilizes it for interventional procedures such as reductions.
Rad techs are staffed for the ER, seeing patients quickly in ‘rotations’ where the devices substitute in and out. Interviewees described rotations as 20 to 30 minutes, seeing 7 to 20 patients in each rotation. We’re going to assume the longer rotation period and median patient count of 12 patients in 30 minutes. No charging is done during these rotations.
Emergency Room Doctor Use
Device travel to patient  for 3 minutes. On and not charging.
5s DDR then 3 Images over 2 minutes. Device either acquiring or active and tracking
Device travel to charger for 3 minutes. On and not charging.
Charge for 12 minutes
Repeat continuously
Emergency Room Rad Tech Use
Travel to Patient for1 minute. On and not charging.
3 full-power images over 1.5 minutes. Device acquiring or active and tracking.1.5 minutes)
Repeat above 11 times (12 times or 30 minutes total)
Charge for 30 minutes
Repeat continuously
Operating Room (Hospital & ASC)
Interviews
2022-02-24 Dr. Jamie Eldridge
2021-09-15 Greer Knox
2021-09-29 Jett Kingsley
Notes and Assumptions
Ambulatory Surgery Centers (ASCs) are geared to tackle as many surgeries as possible in an 8 hour window. They have back-to-back surgeries all day. Trauma surgeries, which handle cases over a 24 hour period, likely see more intense use of DDR (e.g., unplanned exploratory viewing), but won't necessarily happen back-to-back. For the trauma case, we assume Dr. Jamie Eldridge’s ASC surgery experience of the thickest anatomies that may be targeted with P01 (hips), double the bursts of usage within the surgery. The device is not meant to last an entire day of surgery, though it is meant to last a typical surgery length, or the patient may be exposed to unnecessary risk from waiting for a replacement x-ray machine.
To estimate an ASC handling shorter cases with many pictures, we took inspiration from Dr. Knox’s surgical days and significant overuse (10 pictures + DDR in each 20 minute surgery). As before, the device is not meant to last an entire day, though it must last the span of each surgery when fully charged
The estimated use scenarios are
Trauma Surgeries:
All captures on highest power (kV and mAs)
5 second DDR in triplets over a minute Burst:
5s DDR, 15s wait, 5s DDR, 15s wait, 5s DDR, 15s wait
Bursts (minute duration) will happen 6 times over the 1.5 hour surgery:
1 min burst, 14 min wait, repeat 5 more times
Charging between images/surgeries, beginning each surgery with a full battery.
Hand Surgeries:
50kV, 66mAs for all images
10 images spread evenly, and a 5 second DDR+Image for the final image
2 images, 300s wait, 2 images, 300s wait, 2 images, 300s wait, 2 images, 300s wait, 5s DDR, 2 images, 295s wait.
These 20 minute surgeries happen consecutively for 8 hours, 24 times total then cool/charge overnight and between images/surgeries, though the device is only required to survive a 20 minute surgery.
Clinic
Clinics, like ASCs, are designed to schedule as many quick visits as possible. The typical goal of clinic visits is patient diagnoses, pre-treatment planning, and post-treatment evaluation. However, we have seen with the Imager P00 a confidence to perform guided injections and other interventions using our device and its DDR. Therefore, we have included a full-length DDR and 3 images every 10 minutes in our use assumptions. Note that most clinics wouldn’t see a patient every 10 minutes, but that might be the case for a busy clinic with a large amount of patients. Also, it is assumed that providers within these clinics will not charge the unit except for during their lunch break, so we want to challenge the design team with no-charge blocks of 4-hour half days. After the day, it can charge and cool overnight.
Clinical Use (8 hrs):
50kV, 66ms for all images, preferably full power if we can make it. Idle as normal.
Patient every 10 minutes
1 full-length DDR and 3 images each patient
Continue for 4 hour shift (24 patients total) NO CHARGING
Charge for 1 hour
Repeat another shift
Charge overnight to full battery.
Sports Medicine and Mobile Imaging
The device use expected in the Sports Medicine and Mobile Imaging markets was decided to likely not exceed an extent surpassing the markets listed above. The team did not find significant evidence of charging availability or patient throughput challenges found in above markets. Therefore, we will not be designing battery life tests with these markets in mind, until further evidence presents itself.
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| To: | File |
| --- | --- |
| From: | User Needs Team |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Hardware Engineering Quality Engineering Regulatory Affairs | 12/8/2022 | 22-066 |
| B | Remove references to use and charging in vehicles. Minor updates to use case details. Remove P00 duty cycle information. | Quality Engineering Engineering Regulatory Affairs | 06 Apr 2023 | 23-115 |
