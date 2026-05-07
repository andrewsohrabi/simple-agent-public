# VVPR-P01-162 Rev D: MX1 Battery Life Test Protocol and Report

## Metadata
- Document ID: VVPR-P01-162
- Revision: D
- Prefix: VVPR
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-162 - MX1 Battery Life Test Protocol and Report_D-Signed.docx
- Source path: Example QMS - MedAI/VVPR-P01-162 - MX1 Battery Life Test Protocol and Report_D-Signed.docx
- Extraction warnings: none

## Extracted Content
D Protocol Section
STUDY PURPOSE
The purpose of this study is to evaluate the battery life of the MX1 Portable X-ray System under simulated use conditions.
OBJECTIVE AND SCOPE
This study will compare measured MX1 battery life data in a use case simulation with the battery life specifications documented in DR-P01-005 Rev. A. This study applies to the MX1 Portable X-ray System’s C1 Cassette with the MS-10083 Cassette Battery Pack, E1 Emitter with the MS-10010 Emitter Battery Pack, and the H1 Rev. A Wired Charger.
REFERENCES
DR-P01-005 Rev. A - MX1 Design Inputs
RSK-P01-010 Rev. B - MX1 Risk Assessment
MEMO-P01-412 Rev. B - P01 Use Cases
QSP-026 Rev. B - Statistical Techniques
VVPR-P01-088 Rev. B  - Engineering Mode Protocol and Report
MATERIALS
This test will be performed on a MX1 System design verification unit and will utilize engineering scripts to control simulated use behavior and timing.
The MX1 major components, positioned statically to emit x-rays, including:
MX1 Emitter E1, Rev. I
MX1 Cassette C1, Rev. H
The MX1 Cassette and Emitter, before each test, will be charged to full battery with an H1 Rev. A Wired Charger for over 2 hours to ensure the battery is fully charged, aside from indicators.
Engineering Scripts (Appendix B)
Radiation detecting equipment to detect any radiation output
Radcal AGDM+ Accu-Gold Digitizer (EQP-109 or equivalent)
Radcal AGMS-DM+ Accu-Gold Multi-Sensor (EQP-110 or equivalent)
Appropriate radiation shielding and monitoring equipment
SAMPLE SIZE
A sample size of fourMX1 systems will be used. The resultant performance of the test will be recorded as variable data with each unit.
This verification is tested with a 90/90 confidence/reliability level, per QSP-026, Statistical Methods. Although the data will be analyzed as variable to the required 90/90 confidence/reliability level, the sample size is limited by the inherent constraints of MX1 being a capital equipment device.
METHODS
Test Environments
MedAI Office; Springfield, IL
Test Personnel
MedAI engineering personnel shall facilitate the:
Preparation of the device for use;
Execution of the test; and
Observation, collection, and analysis of the test data
TEST PROCEDURE BACKGROUND
A worst-case simulated use case that exceeds expected normal use was chosen based on risk of delayed procedure as adapted from the P01 Use Cases memo, MEMO-P01-412. Software scripts were written in engineering mode to execute the tests without constant human involvement.
The use case and test criteria can be described as Battery Depletion, the duration the device continues before it cannot take an x-ray, because it does not have the battery capacity to execute the next capture. This testing will record the duration from beginning of the use case scenario to the last successful capture.
USE CASE DESCRIPTION and SCRIPT TIMING REQUIREMENTS
Battery Depletion Use Case
All captures on highest power for serial radiography (80 kV and 0.08 mAs)
No charging, device idling as normal
Take three 5 second DDRs within a minute: 5s DDR (25 images), 15s wait, 5s DDR (25 images), 15s wait, 5s DDR (25 images), and 15s wait
Repeat step 8.1.3 every 15 minutes until the device fails to take an x-ray due to battery depletion.
Record time when device fails to take an x-ray due to battery depletion (15% E1 battery or lower and 20% C1 battery or lower)
Scripts created for the above test to be performed are documented in Appendix B
PROCEDURE
Set up the subject MX1 device in a test jig
Document the devices’ serial numbers and jig’s tool number in the report
Place a radiation detector on the Cassette, in the path of the beam.
Power on the system and ensure that Engineering Mode is on
Reference VVPR-P01-088 Rev B. Section 6.d.vi.
Verify both Emitter and Cassette are fully charged, as specified in section 4.3
Install the mx1-controller to your environment: example.com/
Disconnect any H1 Wired Chargers from the emitter and cassette and immediately execute the script “battery_life_test.py”
Use the Accu-Gold application to monitor the first few x-ray bursts with the radiation sensor to ensure radiation is emitted.
Periodically monitor the system to see if a component is powered off from battery depletion, signaling the end of the test.
Once test is complete, connect H1 Wired Chargers to the emitter and cassette
Run the following command on the C1 cassette: “ journalctl --user-unit cassette-orchestrator.service -f | grep -e “; battery %, isCharging, is Plugged, respectively:” ” and locate the first entry where the C1 cassette battery was less than 20% or the first entry where the E1 emitter battery was less than 15%. Locate the timestamp for that entry.
Calculate the duration of the test using the start time and then timestamp found in 9.12
Record the result data in the report in the Battery Life Test Data Collection Form in Appendix A.
Verify via the software log that the test sequence described in Sections 8.1.3-8.1.4 (Three 5 second DDRs within a minute: 5s DDR, 15s wait, 5s DDR, 15s wait, 5s DDR, 15s wait. Repeat every 15 minutes until the device failed to shoot due to battery depletion) was accurately performed by the engineering script throughout the test.
Power off the subject system
Repeat procedure for the total number of subject device samples
DATA AND ANALYSIS
The system is tested with an output data point of the duration the device operates before it cannot take an x-ray because it does not have the battery capacity to execute the next capture.
Data for battery depletion use case will be reported as variable data and analyzed using one-sided statistical tolerance limits with the confidence/reliability levels detailed in the acceptance criteria section of this protocol.
ACCEPTANCE CRITERIA
The calculated lower tolerance limit  () must match or exceed the specified minimum battery depletion requirement for the worst case use case, where  is the mean duration of battery depletion,  is the standard deviation of battery depletion duration, and  is the coverage factor that determines the confidence in data falling within the calculated limit. Appendix 1 in QSP-026, identifies a  k value equal to 3.187for a sample size of 4 when calculating limits for One-Sided Factors with 90% Reliability and 90% Confidence.
Table 1. Acceptance Criteria
APPENDICES
Appendix A - MX1 Battery Life Test Data Collection Form
Table A.1 -  MX1 Battery Life Test Results
Performed By: ___________________________________________    Date: ___________________
____________________________________________            ____________________
Appendix B - Engineering Test Script for Battery Life Testing
Appendix B - Engineering Test Scripts for Battery Life Testing - battery_life_test.py
import time
import logging
from mx1control.device import MX1Device
HOLD_TIME = 5000
KV = 80
MAS = 0.08
NUM_DDR = 3
SPEED_UP_FACTOR = 1
DDR_SLEEP_TIME = 15 / SPEED_UP_FACTOR
LONG_SLEEP_TIME = 900 / SPEED_UP_FACTOR
IDLE_TIME = 300 / SPEED_UP_FACTOR
WAKE_UP_TIME = 30 / SPEED_UP_FACTOR
LOG_SAVE_NAME = "battery_life_test_results.txt"
DEVICE_NAME = "0.0.0.0" #"0.0.0.0" is if its ran locally on a cassette
mx1 = MX1Device(DEVICE_NAME)
def setup_logger():
#logging.basicConfig(filename=LOG_SAVE_NAME,format='%(asctime)s %(levelname)-4s %(message)s', level=logging.INFO, force=True, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# if the below isn't what you want, comment it and uncomment logging.basiConfig
#logger.propagate = False
formatter = logging.Formatter('%(asctime)s %(levelname)-4s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(LOG_SAVE_NAME)
file_handler.setLevel(logging.INFO)  # Set the desired logging level for the file
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
return logger
def run(logger : logging.Logger):
while True:
logger.info("Exiting idle")
# the REST API set idle command doesn't look like it works atm
# Work around for now
# Switch mode to camera (this is just for safety so not xray is taken if it was not in idle for any reason)
# Pull Trigger
# device is now awake
mx1.set_mode("photo")
mx1.trigger_pull(press_time=100)
logger.info(f"Waiting {WAKE_UP_TIME} for the device to wake up")
time.sleep(WAKE_UP_TIME)
mx1.set_mode("ddr_manual")
time.sleep(3) # This is here to ensure it enters the mode properly before trigger pullin
for i in range(NUM_DDR):
logger.info(f"Taking DDR {i+1}")
mx1.capture(
"serial",
kv=KV,
mas=MAS,
capture_time=HOLD_TIME,
retrieve_images=False
)
time.sleep(DDR_SLEEP_TIME)
logger.info(f"Wating {IDLE_TIME} to enter idle")
time.sleep(IDLE_TIME)
logger.info(f"Entering idle")
difference_sleep = LONG_SLEEP_TIME - IDLE_TIME - WAKE_UP_TIME - DDR_SLEEP_TIME
logger.info(f"Sleeping an additional {difference_sleep} for a total {LONG_SLEEP_TIME} minus wake up time of {WAKE_UP_TIME} and sleep time of {DDR_SLEEP_TIME} after the last DDR")
time.sleep(difference_sleep)
if __name__ == "__main__":
logger = setup_logger()
run(logger)
PROTOCOL APPROVAL
Digital Key: example.com/
Report Section
PROTOCOL DEVIATIONS
The H1 charger was not used to charge the system components during the execution of this protocol. Rather, a component of H1 was used (M50011), achieving the same result. This deviation has no effect on the test results.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Sample 1 System
Software System Version: v3.0.0
E1 Emitter SN: 1216
C1 Cassette SN: 1217
Sample 2 System
Software System Version: v3.0.0
E1 Emitter SN: 1220
C1 Cassette SN: 1079
Sample 3 System
Software System Version: v3.0.0
E1 Emitter SN: 1206
C1 Cassette SN: 1207
Sample 4 System
Software System Version: v3.0.0
E1 Emitter SN: 1204
C1 Cassette SN: 1205
M50011 Rev. A
Lot unknown, but the component was received as part of PO22086. All M50011 components in inventory were received as part of the same purchase order, but were assigned different lots due to being received in different batches.
RESULTS
Results of battery life testing are presented in Table 1 below.
Table 1. MX1 Battery Life Test Results
Performed By: _____Noah Jensen______    Date: ___5/24/2024____
CONCLUSION
The lower tolerance limit of the 4 sampled units proves with 90% confidence and 90% reliability that the MX1 surpasses the 90 minute pass criteria.
APPENDICES
Appendix A - Test Output Logs for Each Sample Tested
Appendix A - Test Output Logs for Each Sample Tested
dv23 - battery_life_test_results.txt
2024-05-22 01:15:39 INFO Exiting idle
2024-05-22 01:15:40 INFO Waiting 30.0 for the device to wake up
2024-05-22 01:16:02 INFO Exiting idle
2024-05-22 01:16:02 INFO Waiting 30.0 for the device to wake up
2024-05-22 01:16:35 INFO Taking DDR 1
2024-05-22 01:16:57 INFO Taking DDR 2
2024-05-22 01:17:18 INFO Taking DDR 3
2024-05-22 01:17:40 INFO Waiting 300.0 to enter idle
2024-05-22 01:22:40 INFO Entering idle
2024-05-22 01:22:40 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 01:31:55 INFO Exiting idle
2024-05-22 01:31:56 INFO Waiting 30.0 for the device to wake up
2024-05-22 01:32:29 INFO Taking DDR 1
2024-05-22 01:32:51 INFO Taking DDR 2
2024-05-22 01:33:12 INFO Taking DDR 3
2024-05-22 01:33:35 INFO Waiting 300.0 to enter idle
2024-05-22 01:33:35 INFO Entering idle
2024-05-22 01:33:35 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 01:42:48 INFO Exiting idle
2024-05-22 01:42:49 INFO Waiting 30.0 for the device to wake up
2024-05-22 01:43:22 INFO Taking DDR 1
2024-05-22 01:43:48 INFO Taking DDR 2
2024-05-22 01:44:15 INFO Taking DDR 3
2024-05-22 01:44:41 INFO Waiting 300.0 to enter idle
2024-05-22 01:59:43 INFO Exiting idle
2024-05-22 01:59:44 INFO Waiting 30.0 for the device to wake up
2024-05-22 02:00:17 INFO Taking DDR 1
2024-05-22 02:00:46 INFO Taking DDR 2
2024-05-22 02:01:10 INFO Taking DDR 3
2024-05-22 02:01:37 INFO Waiting 300.0 to enter idle
2024-05-22 02:06:37 INFO Entering idle
2024-05-22 02:06:37 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 02:15:53 INFO Exiting idle
2024-05-22 02:15:53 INFO Waiting 30.0 for the device to wake up
2024-05-22 02:16:27 INFO Taking DDR 1
2024-05-22 02:16:53 INFO Taking DDR 2
2024-05-22 02:17:19 INFO Taking DDR 3
2024-05-22 02:17:36 INFO Waiting 300.0 to enter idle
2024-05-22 02:22:36 INFO Entering idle
2024-05-22 02:22:36 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 02:33:29 INFO Exiting idle
2024-05-22 02:33:30 INFO Waiting 30.0 for the device to wake up
2024-05-22 02:34:03 INFO Taking DDR 1
2024-05-22 02:34:30 INFO Taking DDR 2
2024-05-22 02:34:57 INFO Taking DDR 3
2024-05-22 02:35:23 INFO Waiting 300.0 to enter idle
2024-05-22 02:40:23 INFO Entering idle
2024-05-22 02:40:23 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 02:49:39 INFO Exiting idle
2024-05-22 02:49:39 INFO Waiting 30.0 for the device to wake up
2024-05-22 02:50:13 INFO Taking DDR 1
2024-05-22 02:51:01 INFO Taking DDR 2
2024-05-22 02:51:27 INFO Taking DDR 3
2024-05-22 02:51:53 INFO Waiting 300.0 to enter idle
2024-05-22 02:56:54 INFO Entering idle
2024-05-22 02:56:54 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 03:06:09 INFO Exiting idle
2024-05-22 03:06:09 INFO Waiting 30.0 for the device to wake up
2024-05-22 03:06:43 INFO Taking DDR 1
2024-05-22 03:07:09 INFO Taking DDR 2
2024-05-22 03:07:36 INFO Taking DDR 3
2024-05-22 03:08:02 INFO Waiting 300.0 to enter idle
2024-05-22 03:13:02 INFO Entering idle
2024-05-22 03:13:02 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-22 03:22:17 INFO Exiting idle
2024-05-22 03:22:18 INFO Waiting 30.0 for the device to wake up
2024-05-22 03:22:52 INFO Taking DDR 1
2024-05-22 03:23:18 INFO Taking DDR 2
2024-05-22 03:23:45 INFO Taking DDR 3
2024-05-22 03:24:07 INFO Waiting 300.0 to enter idle
---Operator ended test at 3:25:08 due to battery life—
dv25 - battery_life_test_results.txt
2024-05-23 00:19:23 INFO Exiting idle
2024-05-23 00:19:24 INFO Waiting 30.0 for the device to wake up
2024-05-23 00:19:58 INFO Taking DDR 1
2024-05-23 00:20:25 INFO Taking DDR 2
2024-05-23 00:20:52 INFO Taking DDR 3
2024-05-23 00:21:19 INFO Wating 300.0 to enter idle
2024-05-23 00:26:19 INFO Entering idle
2024-05-23 00:26:19 INFO Sleeping an addtional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 00:35:34 INFO Exiting idle
2024-05-23 00:35:35 INFO Waiting 30.0 for the device to wake up
2024-05-23 00:36:08 INFO Taking DDR 1
2024-05-23 00:36:35 INFO Taking DDR 2
2024-05-23 00:37:03 INFO Taking DDR 3
2024-05-23 00:37:30 INFO Waiting 300.0 to enter idle
2024-05-23 00:42:30 INFO Entering idle
2024-05-23 00:42:30 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 00:51:45 INFO Exiting idle
2024-05-23 00:51:46 INFO Waiting 30.0 for the device to wake up
2024-05-23 00:52:19 INFO Taking DDR 1
2024-05-23 00:52:46 INFO Taking DDR 2
2024-05-23 00:53:16 INFO Taking DDR 3
2024-05-23 00:53:43 INFO Waiting 300.0 to enter idle
2024-05-23 00:58:43 INFO Entering idle
2024-05-23 00:58:43 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 01:07:54 INFO Exiting idle
2024-05-23 01:07:55 INFO Waiting 30.0 for the device to wake up
2024-05-23 01:08:28 INFO Taking DDR 1
2024-05-23 01:08:55 INFO Taking DDR 2
2024-05-23 01:09:22 INFO Taking DDR 3
2024-05-23 01:09:49 INFO Waiting 300.0 to enter idle
2024-05-23 01:14:49 INFO Entering idle
2024-05-23 01:14:49 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 01:24:05 INFO Exiting idle
2024-05-23 01:24:05 INFO Waiting 30.0 for the device to wake up
2024-05-23 01:24:39 INFO Taking DDR 1
2024-05-23 01:25:06 INFO Taking DDR 2
2024-05-23 01:25:33 INFO Taking DDR 3
2024-05-23 01:26:00 INFO Waiting 300.0 to enter idle
2024-05-23 01:31:00 INFO Entering idle
2024-05-23 01:31:00 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 01:40:15 INFO Exiting idle
2024-05-23 01:40:16 INFO Waiting 30.0 for the device to wake up
2024-05-23 01:40:49 INFO Taking DDR 1
2024-05-23 01:41:16 INFO Taking DDR 2
2024-05-23 01:41:42 INFO Taking DDR 3
2024-05-23 01:42:09 INFO Waiting 300.0 to enter idle
2024-05-23 01:47:09 INFO Entering idle
2024-05-23 01:47:09 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 01:56:24 INFO Exiting idle
2024-05-23 01:56:25 INFO Waiting 30.0 for the device to wake up
2024-05-23 01:56:58 INFO Taking DDR 1
2024-05-23 01:57:20 INFO Taking DDR 2
2024-05-23 01:57:47 INFO Taking DDR 3
2024-05-23 01:58:13 INFO Waiting 300.0 to enter idle
2024-05-23 02:03:13 INFO Entering idle
2024-05-23 02:03:13 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 02:12:28 INFO Exiting idle
2024-05-23 02:12:29 INFO Waiting 30.0 for the device to wake up
2024-05-23 02:13:02 INFO Taking DDR 1
2024-05-23 02:13:30 INFO Taking DDR 2
2024-05-23 02:13:57 INFO Taking DDR 3
2024-05-23 02:14:23 INFO Waiting 300.0 to enter idle
2024-05-23 02:19:23 INFO Entering idle
2024-05-23 02:19:23 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 02:28:39 INFO Exiting idle
2024-05-23 02:28:39 INFO Waiting 30.0 for the device to wake up
2024-05-23 02:29:13 INFO Taking DDR 1
2024-05-23 02:29:40 INFO Taking DDR 2
2024-05-23 02:30:07 INFO Taking DDR 3
2024-05-23 02:30:34 INFO Waiting 300.0 to enter idle
2024-05-23 02:35:34 INFO Entering idle
2024-05-23 02:35:34 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 02:44:49 INFO Exiting idle
2024-05-23 02:44:50 INFO Waiting 30.0 for the device to wake up
2024-05-23 02:45:24 INFO Taking DDR 1
2024-05-23 02:45:51 INFO Taking DDR 2
2024-05-23 02:46:18 INFO Taking DDR 3
2024-05-23 02:46:45 INFO Waiting 300.0 to enter idle
2024-05-23 02:51:45 INFO Entering idle
2024-05-23 02:51:45 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 03:01:00 INFO Exiting idle
2024-05-23 03:01:00 INFO Waiting 30.0 for the device to wake up
2024-05-23 03:01:34 INFO Taking DDR 1
2024-05-23 03:02:01 INFO Taking DDR 2
2024-05-23 03:02:29 INFO Taking DDR 3
--- Operator stopped test at 03:02:45 due to battery life —
dv22 - battery_life_test_results.txt
2024-05-23 20:16:46 INFO Exiting idle
2024-05-23 20:16:46 INFO Waiting 30.0 for the device to wake up
2024-05-23 20:17:20 INFO Taking DDR 1
2024-05-23 20:17:41 INFO Taking DDR 2
2024-05-23 20:18:03 INFO Taking DDR 3
2024-05-23 20:18:24 INFO Waiting 300.0 to enter idle
2024-05-23 20:23:24 INFO Entering idle
2024-05-23 20:23:24 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 20:32:39 INFO Exiting idle
2024-05-23 20:32:40 INFO Waiting 30.0 for the device to wake up
2024-05-23 20:33:13 INFO Taking DDR 1
2024-05-23 20:33:35 INFO Taking DDR 2
2024-05-23 20:33:56 INFO Taking DDR 3
2024-05-23 20:34:17 INFO Waiting 300.0 to enter idle
2024-05-23 20:39:17 INFO Entering idle
2024-05-23 20:39:17 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 20:48:33 INFO Exiting idle
2024-05-23 20:48:33 INFO Waiting 30.0 for the device to wake up
2024-05-23 20:49:07 INFO Taking DDR 1
2024-05-23 20:49:28 INFO Taking DDR 2
2024-05-23 20:49:50 INFO Taking DDR 3
2024-05-23 20:50:11 INFO Waiting 300.0 to enter idle
2024-05-23 20:55:11 INFO Entering idle
2024-05-23 20:55:11 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 21:04:26 INFO Exiting idle
2024-05-23 21:04:27 INFO Waiting 30.0 for the device to wake up
2024-05-23 21:05:00 INFO Taking DDR 1
2024-05-23 21:05:22 INFO Taking DDR 2
2024-05-23 21:07:08 WARNING Timed out waiting for image, could indicate battery life
2024-05-23 21:07:23 INFO Taking DDR 3
2024-05-23 21:09:09 WARNING Timed out waiting for image, could indicate battery life
2024-05-23 21:09:24 INFO Waiting 300.0 to enter idle
2024-05-23 21:14:24 INFO Entering idle
2024-05-23 21:14:24 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 21:21:49 INFO Exiting idle
2024-05-23 21:21:50 INFO Waiting 30.0 for the device to wake up
2024-05-23 21:22:23 INFO Taking DDR 1
2024-05-23 21:22:45 INFO Taking DDR 2
2024-05-23 21:23:06 INFO Taking DDR 3
2024-05-23 21:23:27 INFO Waiting 300.0 to enter idle
2024-05-23 21:28:27 INFO Entering idle
2024-05-23 21:28:27 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 21:37:42 INFO Exiting idle
2024-05-23 21:37:43 INFO Waiting 30.0 for the device to wake up
2024-05-23 21:38:16 INFO Taking DDR 1
2024-05-23 21:38:38 INFO Taking DDR 2
2024-05-23 21:38:59 INFO Taking DDR 3
2024-05-23 21:39:21 INFO Waiting 300.0 to enter idle
2024-05-23 21:44:21 INFO Entering idle
2024-05-23 21:44:21 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 21:53:36 INFO Exiting idle
2024-05-23 21:53:36 INFO Waiting 30.0 for the device to wake up
2024-05-23 21:54:10 INFO Taking DDR 1
2024-05-23 21:54:31 INFO Taking DDR 2
2024-05-23 21:54:53 INFO Taking DDR 3
2024-05-23 21:55:14 INFO Waiting 300.0 to enter idle
2024-05-23 22:00:14 INFO Entering idle
2024-05-23 22:00:14 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 22:09:29 INFO Exiting idle
2024-05-23 22:09:30 INFO Waiting 30.0 for the device to wake up
2024-05-23 22:10:03 INFO Taking DDR 1
2024-05-23 22:10:25 INFO Taking DDR 2
2024-05-23 22:10:46 INFO Taking DDR 3
2024-05-23 22:11:07 INFO Waiting 300.0 to enter idle
2024-05-23 22:16:07 INFO Entering idle
2024-05-23 22:16:07 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 22:25:23 INFO Exiting idle
2024-05-23 22:25:23 INFO Waiting 30.0 for the device to wake up
2024-05-23 22:25:57 INFO Taking DDR 1
2024-05-23 22:26:18 INFO Taking DDR 2
2024-05-23 22:26:40 INFO Taking DDR 3
2024-05-23 22:27:01 INFO Waiting 300.0 to enter idle
2024-05-23 22:32:01 INFO Entering idle
2024-05-23 22:32:01 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-23 22:41:16 INFO Exiting idle
2024-05-23 22:41:17 INFO Waiting 30.0 for the device to wake up
2024-05-23 22:41:50 INFO Taking DDR 1
2024-05-23 22:42:12 INFO Taking DDR 2
--- Operator stopped test at 22:42:24 due to battery life ---
dv24 - battery_life_test_results.txt
2024-05-23 23:56:24 INFO Exiting idle
2024-05-23 23:56:25 INFO Waiting 30.0 for the device to wake up
2024-05-23 23:56:58 INFO Taking DDR 1
2024-05-23 23:57:20 INFO Taking DDR 2
2024-05-23 23:57:41 INFO Taking DDR 3
2024-05-23 23:58:03 INFO Waiting 300.0 to enter idle
2024-05-24 00:03:03 INFO Entering idle
2024-05-24 00:03:03 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 00:12:18 INFO Exiting idle
2024-05-24 00:12:19 INFO Waiting 30.0 for the device to wake up
2024-05-24 00:12:52 INFO Taking DDR 1
2024-05-24 00:13:14 INFO Taking DDR 2
2024-05-24 00:13:35 INFO Taking DDR 3
2024-05-24 00:13:56 INFO Waiting 300.0 to enter idle
2024-05-24 00:18:56 INFO Entering idle
2024-05-24 00:18:56 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 00:28:11 INFO Exiting idle
2024-05-24 00:28:12 INFO Waiting 30.0 for the device to wake up
2024-05-24 00:28:45 INFO Taking DDR 1
2024-05-24 00:29:07 INFO Taking DDR 2
2024-05-24 00:29:28 INFO Taking DDR 3
2024-05-24 00:29:49 INFO Waiting 300.0 to enter idle
2024-05-24 00:34:50 INFO Entering idle
2024-05-24 00:34:50 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 00:44:05 INFO Exiting idle
2024-05-24 00:44:05 INFO Waiting 30.0 for the device to wake up
2024-05-24 00:44:39 INFO Taking DDR 1
2024-05-24 00:45:00 INFO Taking DDR 2
2024-05-24 00:45:22 INFO Taking DDR 3
2024-05-24 00:46:09 INFO Waiting 300.0 to enter idle
2024-05-24 00:51:09 INFO Entering idle
2024-05-24 00:51:09 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 01:00:25 INFO Exiting idle
2024-05-24 01:00:25 INFO Waiting 30.0 for the device to wake up
2024-05-24 01:00:59 INFO Taking DDR 1
2024-05-24 01:01:25 INFO Taking DDR 2
2024-05-24 01:01:52 INFO Taking DDR 3
2024-05-24 01:02:18 INFO Waiting 300.0 to enter idle
2024-05-24 01:07:19 INFO Entering idle
2024-05-24 01:07:19 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 01:16:34 INFO Exiting idle
2024-05-24 01:16:34 INFO Waiting 30.0 for the device to wake up
2024-05-24 01:17:08 INFO Taking DDR 1
2024-05-24 01:17:34 INFO Taking DDR 2
2024-05-24 01:18:01 INFO Taking DDR 3
2024-05-24 01:18:27 INFO Waiting 300.0 to enter idle
2024-05-24 01:23:27 INFO Entering idle
2024-05-24 01:23:27 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 01:32:42 INFO Exiting idle
2024-05-24 01:32:43 INFO Waiting 30.0 for the device to wake up
2024-05-24 01:33:16 INFO Taking DDR 1
2024-05-24 01:33:38 INFO Taking DDR 2
2024-05-24 01:34:04 INFO Taking DDR 3
2024-05-24 01:34:30 INFO Waiting 300.0 to enter idle
2024-05-24 01:39:30 INFO Entering idle
2024-05-24 01:39:30 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 01:48:45 INFO Exiting idle
2024-05-24 01:48:46 INFO Waiting 30.0 for the device to wake up
2024-05-24 01:49:19 INFO Taking DDR 1
2024-05-24 01:49:46 INFO Taking DDR 2
2024-05-24 01:50:13 INFO Taking DDR 3
2024-05-24 01:50:39 INFO Waiting 300.0 to enter idle
2024-05-24 01:55:39 INFO Entering idle
2024-05-24 01:55:39 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 02:04:54 INFO Exiting idle
2024-05-24 02:04:55 INFO Waiting 30.0 for the device to wake up
2024-05-24 02:05:29 INFO Taking DDR 1
2024-05-24 02:05:55 INFO Taking DDR 2
2024-05-24 02:06:22 INFO Taking DDR 3
2024-05-24 02:06:49 INFO Waiting 300.0 to enter idle
2024-05-24 02:11:49 INFO Entering idle
2024-05-24 02:11:49 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
2024-05-24 02:21:04 INFO Exiting idle
2024-05-24 02:21:05 INFO Waiting 30.0 for the device to wake up
2024-05-24 02:21:38 INFO Taking DDR 1
2024-05-24 02:22:05 INFO Taking DDR 2
2024-05-24 02:22:31 INFO Taking DDR 3
2024-05-24 02:22:57 INFO Waiting 300.0 to enter idle
2024-05-24 02:27:57 INFO Entering idle
2024-05-24 02:27:57 INFO Sleeping an additional 555.0 for a total 900.0 minus wake up time of 30.0 and sleep time of 15.0 after the last DDR
--- Operator stopped test at 02:36:18 due to battery life ---
REPORT APPROVAL
Digital Key:
example.com/

### Table 1
| Use Case | Confidence/Reliability | Acceptance Criteria |
| --- | --- | --- |
| Intense Procedures (minutes) | 90/90 | Battery depletion ≥ 90 minutes |

### Table 2
| Sample | Device SN | Duration (min) | Average Duration (min) | Standard Deviation | K-Factor | Lower Tolerance Limit (90/90) | Pass Criteria | PASS/ FAIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  | 3.187a |  | ≥ 90 minutes |  |
| 2 |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |
| aThe k-Factor derived from the Tchebysheff theorem, which does not assume normality, would be k=3.2. The more conservative k-Factor 3.187was chosen from the One-sided Factors Table of QSP-026 for 90% confidence/90% reliability and an n=4. |  |  |  |  |  |  |  |  |

### Table 3
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Engineering Quality Engineering Regulatory Affairs | 20 May 2024 | 24-150 |
| B | Updated script and procedure to match v3.0.0 Software Release | Engineering Quality Engineering Regulatory Affairs | 21 May 2024 | 24-267 |
| C | Increased sample size to 4 and adjusted K value accordingly | Engineering Quality Engineering Regulatory Affairs | 24 May 2024 | 24-284 |

### Table 4
| Sample | DeviceSN | Duration (minutes) | Average Duration(minutes) | Standard Deviation | K-Factor | Lower Tolerance Limit (90/90) | Pass Criteria | PASS/ FAIL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DV23 (1216) | 129 | 149.25 | 13.5246 | 3.187a | 106.115 | ≥ 90 minutes | PASS |
| 2 | DV25 (1220) | 163 |  |  |  |  |  |  |
| 3 | DV22 (1206) | 145 |  |  |  |  |  |  |
| 4 | DV24(1204) | 160 |  |  |  |  |  |  |
| aThe k-Factor derived from the Tchebysheff theorem, which does not assume normality, would be k=3.2. The more conservative k-Factor 3.187 was chosen from the One-sided Factors Table of QSP-026 for 90% confidence/90% reliability and an n=4. |  |  |  |  |  |  |  |  |

### Table 5
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| D | Report | Engineering Quality Engineering Regulatory Affairs | 26 May 2024 | 24-291 |
