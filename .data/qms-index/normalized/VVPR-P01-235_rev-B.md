# VVPR-P01-235 Rev B: Battery Model Accuracy Verification Protocol

## Metadata
- Document ID: VVPR-P01-235
- Revision: B
- Prefix: VVPR
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: VVPR-P01-235 - Battery Model Accuracy Verification Protocol_B.docx
- Source path: Example QMS - MedAI/VVPR-P01-235 - Battery Model Accuracy Verification Protocol_B.docx
- Extraction warnings: none

## Extracted Content
Protocol Section
STUDY PURPOSE
The purpose of this study is to verify the accuracy of the Open Circuit Voltage to State of Charge (OCV/SOC) function and coulomb counter of the MAX17205 Fuel Gauge on the Emitter Battery Pack (MS-10010) and the Cassette Battery Pack (MS-10083).
REFERENCES
Analog Devices MAX17205 IC
MWI-207 Rev C - WS-009 Workstation Installation
MWI-211 Rev C - MS-10010 Emitter Battery Pack Verification
MWI-214 Rev C - MS-10083 Cassette Battery Pack Verification
S10041 - Cassette MAX17205 Configuration File version 1.2
S10043 - Emitter MAX17205 Configuration File version 1.2
MATERIALS
Testing Equipment
T-115 Rev A - MX1 BMS Interface
M50603 - Ethernet Cable, 6ft
M50646 - ITECH Power Supply IT6952A
MS-10476 Rev A - Electronic Load to PMUX Harness, Male
M50607 - Keysight Technologies Dual Input Electronic Load EL34243A
M50566 - Network switch
M50629 - RS-232 Cable, Male to Female
M51122 Banana Jack Cable, Black
M51125 Banana Jack Cable, Red
Dell Computer (Windows 11)
Monitor
Equipment Under Test
MS-10010 BOM Rev I - Emitter Battery Pack
MS-10083 BOM Rev J - Cassette Battery Pack
SAMPLE SIZE
This test verifies a series of defined registers, and will therefore utilize a sample size of 1 of each battery.
METHODS
Locations and Personnel Responsibilities
Verification - To be performed at MedAI, Inc. office building: 100 Main Street Ste 700 Springfield, IL 60001
Personnel who verified the outcome/expected result shall enter with the test result their initials and the date the requirement was verified. Screenshot/photo evidence shall be attached when applicable.
Experimental Procedure
Setup
Arrange all parts on the workbench as shown in the image.
Plug in the HDMI cable between the monitor and workstation computer
Plug in the USB cables for the mouse and keyboard to the front of the workstation computer
Plug in the USB connector of the [USB to Serial Cable Harness] to the back of the workstation computer
Plug in the USB cable of T-115 to the back of the workstation computer
Plug in one ethernet cable between the [Network Switch] and the back of the electronic load
Plug in one ethernet cable between the [Network Switch] and the workstation computer
Plug in facility provided network drop into the [Network Switch]
Plug in the workstation computer’s power cord to the back of the computer and an AC outlet
Plug in the monitor’s power cord to the back of the monitor and an AC outlet
Plug in the electronic load’s power cord to the back of the electronic load and an AC outlet
Plug in the power supply’s power cord to the back of the power supply and an AC outlet
Plug the [RS-232 Cable, Male to Female] into the back of the power supply and into the [RS-232 to USB Adapter].
Plug in the [RS-232 to USB Adapter] into the back of the workstation computer USB port.
Connect the Red Banana Jack Cable (M51125) between the power supply positive output and Electronic Load Channel 1 positive output.
Connect the BlackBanana Jack Cable (M51122) between the power supply negative output and Electronic Load Channel 1 negative output.
Plug in the banana plug connection of the [Electronic Load to PMUX harness, Male] to Electronic Load channel 1, so red cables are connected and black cables are connected.
Press the power buttons on the workstation computer and monitor to power on
Press the power buttons on the power supply and DC Electronic Load to power on
On the workstation computer, open MedAI Diagnostic Tool executable folder.
Run “MedAI Diagnostic Tool.exe”.
Click the Settings gear icon in the upper right hand corner to open the Settings window
Click the Scan Ports button under the Power Supply COM Port label and select the COM port of the power supply. Click [Test Connection…] and confirm response
Find the Electronic load VISA address by clicking “Utilities”, “I/O Config”, then “LAN Status”.
Copy the Electronic Load VISA address into the textbox on the ODT settings window, and click the [Test Connection…] button to verify that a response is received
Click the Scan Ports button under the BMS Programming Fixture COM Port, and select COM port of T-115.
Click “Apply” and “OK”.
Execution
Connect the data cable tail of the Cassette Battery Pack (MS-10083) to the MX1 BMS interface, and the power cable tail to the exposed XT-60 connector of MS-10476.
Ensure the “Cassette Battery Model Verification Test” is the only test selected.
Press “Connect BMS” and “RUN”.
When prompted by the software, disconnect the banana plugs from the power supply channels and click “ABORT”
When prompted by the software, connect the banana plugs back to the power supply channels and click “ABORT”
Wait until the test returns “PASS” (Typically Multiple Hours)
Disconnect the Cassette Battery Pack power and data cables, and reconnect the banana plugs to the correct power supply channels.
Connect the Emitter Battery Pack (MS-10010) in the same fashion.
Deselect the “Cassette Battery Model Verification Test” and Select the “Emitter Battery model Verification Test”. Ensure that this test is the only one selected.
Press “Connect BMS” and “RUN”.
When prompted by the software, disconnect the banana plugs from the power supply channels and click “ABORT”
When prompted by the software, connect the banana plugs back to the power supply channels and click “ABORT”
Wait until the test returns “PASS” (Typically Multiple Hours)
Data Analysis
The MedAI Diagnostics Software develops a log for each test that was run with the following format: “<BatteryType>ModelVerificationMMdd_HHmm”, where:
<BatteryType> is either “Cassette” or “Emitter”
MMdd is the date that the test started
HHmm is the time that the test started, using a 24 hour clock.
The data shall be analyzed and the following calculations shall be made:
Fuel Gauge Measurement Accuracy
Compare the reported cell voltage from the fuel gauge (VCELL and AVGVCELL) to the average reported cell voltage from the BMS (CellXVoltage)
Compare the reported battery current from the fuel gauge to the actual current reported by the electronic load and power supply.
Fuel Gauge Coulomb Count Accuracy
Compare the raw coulomb count value reported by the fuel gauge (QH) to the calculated passed charge as determined by the integral of the current passed through the electronic load.
Fuel Gauge Voltage Model Accuracy
Compare the Open Circuit voltage during relaxation stages and the calculated state of charge to the VFOCV and the VFSOC registers reported by the fuel gauge
ACCEPTANCE CRITERIA
Measurement Accuracy
The Fuel Gauge reported cell voltage shall be within ±5% of the BMS average reported cell voltage
The Fuel Gauge reported current during relaxation shall have a mean value in the range of [-1,1]mA
The Fuel Gauge reported current during use shall be within ±5% of the electronic load reported current
Coulomb Count Accuracy
The Fuel Gauge reported change in capacity from the beginning of the test shall be within ±10% of the calculated passed charge.
Voltage Model Accuracy
The Fuel Gauge reported voltage model shall be within ±5% of the calculated voltage model.
APPENDICES
Appendix A: Battery Model Verification Test Code:
Git Hash: 2ba088d6eb69b3085097f563e797f92ded6b8077
using CsvHelper;
using PluginBase;
using PMUX_Test_Plugin;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
namespace ODT
{
internal class ModelVerificationTest : Test
{
#region Basic ODT Test Variables
private readonly string _name;
private readonly string _desc;
public override string Name => _name;
public override string Description => _desc;
#endregion
#region Test Hardware Variables
private BQ76952? _bq;
private MAX17205? _max;
private ElectronicLoad? _el;
private string _address;
private string _visa_addr;
private string _spBMS;
public string VisaAddress
{
get { return _visa_addr; }
set { _visa_addr = value; }
}
public string SERIAL_PORT_BMS
{
get { return _spBMS; }
set { _spBMS = value; }
}
private readonly ElectronicLoad.ELECTRONICLOADCHANNEL _batt_load_channel = ElectronicLoad.ELECTRONICLOADCHANNEL.CH1;
#endregion
#region Battery-Specific Variables
private readonly BATTERY _battType;
private readonly double _batt_chg_voltage;
private readonly double _batt_c_nom;
private readonly int _batt_cell_count;
private readonly double _batt_typ_current;
#endregion
private string filename = "";
private Stopwatch _sw_CC = new(); //stopwatch to count elapsed time between current measurements
private bool _abort;
public override async Task<int> Abort()
{
_abort = true;
return 0;
}
public ModelVerificationTest(BATTERY battType)
{
_battType = battType;
_name = $"{battType} Battery Model Verification Test";
_desc = $"Verifies the OCV/SOC model of the {battType} Battery Fuel Gauge ";
switch (_battType)
{
case BATTERY.CASSETTE:
_batt_chg_voltage = 16.8;
_batt_c_nom = 5.6;
_batt_cell_count = 4;
//_batt_cap_nom = 5600;
_batt_typ_current = 70 / (3.6 * _batt_cell_count);
break;
case BATTERY.EMITTER:
_batt_chg_voltage = 33.6;
_batt_c_nom = 2.8;
_batt_cell_count = 8;
//_batt_cap_nom = 2800;
_batt_typ_current = 50 / (3.6 * _batt_cell_count);
break;
}
}
#region Battery Helper Functions
///// <summary>
///// Hit the Battery pack with a small amount of current
///// to pull BMS out of Short Circuit Detection Loop
///// </summary>
///// <param name="v">To set a specific voltage, or use 0 to set the max battery voltage </param>
//public async Task prechargePack(double v = 0)
//{
//    if (v == 0)
//    {
//        PWS.SetVoltage(_batt_chg_voltage); // Set power supply to battery charging voltage
//    }
//    else
//    {
//        PWS.SetVoltage(v);
//    }
//    PWS.SetCurrentLimit(_batt_c_nom / 20); // Set really low charge current limit because we dont want to actually charge
//    await ToggleBQDsgFet(false);
//    PWS.PowerOn(true);
//    await Task.Delay(500);
//    await ToggleBQDsgFet(true);
//    PWS.PowerOn(false); //Disable PWS
//}
///// <summary>
///// Convert Ushort to String
///// </summary>
///// <param name="value"></param>
///// <returns></returns>
//public string ushort_to_string(ushort value)
//{
//    string binaryString = Convert.ToString(value, 2).PadLeft(16, '0');
//    return binaryString;
//}
public void SendDebugMessage(string message)
{
OnDebugMessage(new DebugMessageArgs(message));
}
/// <summary>
/// Toggle the Discharge FET of the BMS.
/// </summary>
/// <param name="enable">Boolean to </param>
public async Task ToggleBQDsgFet(bool enable)
{
if (_bq is not null)
{
await ToggleBQFetEn(false);
_bq.Update_manu_status();
int try_count = 0;
while (_bq._manu_status.DSG_TEST == !enable)
{
_bq.commandOnly((ushort)bqReg.BQ76_SUB_CMD_ONLY_DSGTEST);
await Task.Delay(100);
_bq.Update_manu_status();
await Task.Delay(100);
if (try_count > 10)
{
SendDebugMessage("failed to toggle discharge fet after " + try_count + " tries");
SendDebugMessage(_bq._manu_status.status.ToString());
break;
}
}
}
else
{
SendDebugMessage("Uninitialized Hardware!");
}
}
/// <summary>
/// Toggle the Charge FET of the BMS.
/// </summary>
/// <param name="enable"></param>
public async Task ToggleBQChgFet(bool enable)
{
if (_bq is not null)
{
await ToggleBQFetEn(false);
_bq.Update_manu_status();
int try_count = 0;
while (_bq._manu_status.CHG_TEST == !enable)
{
_bq.commandOnly((ushort)bqReg.BQ76_SUB_CMD_ONLY_CHGTEST);
await Task.Delay(100);
_bq.Update_manu_status();
await Task.Delay(100);
if (try_count > 10)
{
SendDebugMessage("failed to toggle charge fet after " + try_count + " tries");
SendDebugMessage(_bq._manu_status.status.ToString());
break;
}
}
}
else
{
SendDebugMessage("Uninitialized Hardware!");
}
}
/// <summary>
/// Toggle the Precharge FET of the BMS
/// </summary>
/// <param name="enable"></param>
public async Task ToggleBQPchgFet(bool enable)
{
if (_bq is not null)
{
_bq.Update_manu_status();
while (_bq._manu_status.PCHG_TEST == !enable)
{
_bq.commandOnly((ushort)bqReg.BQ76_SUB_CMD_ONLY_PCHGTEST);
await Task.Delay(100);
_bq.Update_manu_status();
await Task.Delay(100);
}
}
else
{
SendDebugMessage("Uninitialized Hardware!");
}
}
/// <summary>
/// Allow for the toggling of the FETs of the BMS
/// </summary>
/// <param name="enable"></param>
public async Task ToggleBQFetEn(bool enable)
{
if (_bq is not null)
{
int trycount = 0;
_bq.Update_manu_status();
while (_bq._manu_status.FET_EN == !enable)
{
_bq.commandOnly((ushort)bqReg.BQ76_SUB_CMD_ONLY_FET_ENABLE);
trycount++;
await Task.Delay(100);
_bq.Update_manu_status();
await Task.Delay(100);
if (trycount > 10)
{
SendDebugMessage("Failed to toggle FET_EN after 10 tries");
break;
}
}
}
else
{
SendDebugMessage("Uninitialized Hardware!");
}
}
#endregion
#region Data Logging Functions
public bool InitData()
{
string[] actual_MAX_REG_NAMES = Enum.GetNames(typeof(maxReg));
bqReg[] bqCMDsOfInterest =
{
bqReg.BQ76_DIR_CMD_Cell1Voltage,
bqReg.BQ76_DIR_CMD_Cell2Voltage,
bqReg.BQ76_DIR_CMD_Cell3Voltage,
bqReg.BQ76_DIR_CMD_Cell4Voltage,
bqReg.BQ76_DIR_CMD_Cell5Voltage,
bqReg.BQ76_DIR_CMD_Cell6Voltage,
bqReg.BQ76_DIR_CMD_Cell7Voltage,
bqReg.BQ76_DIR_CMD_Cell8Voltage,
bqReg.BQ76_DIR_CMD_StackVoltage,
bqReg.BQ76_DIR_CMD_LDPinVoltage,
bqReg.BQ76_DIR_CMD_PACKPinVoltage,
bqReg.BQ76_DIR_CMD_CC2Current,
bqReg.BQ76_DIR_CMD_TS1Temperature,
bqReg.BQ76_DIR_CMD_TS2Temperature,
bqReg.BQ76_DIR_CMD_TS3Temperature,
bqReg.BQ76_DIR_CMD_IntTemperature
};
string[] actual_DIR_CMD_NAMES =
{
"BQ76_DIR_CMD_Cell1Voltage",
"BQ76_DIR_CMD_Cell2Voltage",
"BQ76_DIR_CMD_Cell3Voltage",
"BQ76_DIR_CMD_Cell4Voltage",
"BQ76_DIR_CMD_Cell5Voltage",
"BQ76_DIR_CMD_Cell6Voltage",
"BQ76_DIR_CMD_Cell7Voltage",
"BQ76_DIR_CMD_Cell8Voltage",
"BQ76_DIR_CMD_StackVoltage",
"BQ76_DIR_CMD_LDPinVoltage",
"BQ76_DIR_CMD_PACKPinVoltage",
"BQ76_DIR_CMD_CC2Current",
"BQ76_DIR_CMD_TS1Temperature",
"BQ76_DIR_CMD_TS2Temperature",
"BQ76_DIR_CMD_TS3Temperature",
"BQ76_DIR_CMD_IntTemperature"
};
FileStream stream;
StreamWriter writer;
CsvWriter csv;
try
{
stream = System.IO.File.Open(filename, FileMode.Append);
writer = new StreamWriter(stream);
csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
}
catch (Exception e)
{
OnDebugMessage(new DebugMessageArgs(e.ToString()));
return false;
}
csv.WriteField("DateTime");
for (int i = 0; i < actual_MAX_REG_NAMES.Length; i++)
{
csv.WriteField(actual_MAX_REG_NAMES[i]);
}
for (int i = 0; i < bqCMDsOfInterest.Length; i++)
{
csv.WriteField(actual_DIR_CMD_NAMES[i]);
}
csv.WriteField("PWS_CURRENT");
csv.WriteField("EL_CURRENT");
csv.WriteField("TIME_ELAPSED");
csv.WriteField("PASSED_CHARGE_mAH");
csv.WriteField("PWS_VOLTAGE");
csv.WriteField("EL_VOLTAGE");
csv.NextRecord();
writer.Close();
return true;
}
/// <summary>
/// Logs Data from the BMS, including Cell Voltages, Currents, Pack Voltages, statuses, and
/// </summary>
/// <returns></returns>
public int LogData()
{
bqReg[] bqCMDsOfInterest =
{
bqReg.BQ76_DIR_CMD_Cell1Voltage,
bqReg.BQ76_DIR_CMD_Cell2Voltage,
bqReg.BQ76_DIR_CMD_Cell3Voltage,
bqReg.BQ76_DIR_CMD_Cell4Voltage,
bqReg.BQ76_DIR_CMD_Cell5Voltage,
bqReg.BQ76_DIR_CMD_Cell6Voltage,
bqReg.BQ76_DIR_CMD_Cell7Voltage,
bqReg.BQ76_DIR_CMD_Cell8Voltage,
bqReg.BQ76_DIR_CMD_StackVoltage,
bqReg.BQ76_DIR_CMD_LDPinVoltage,
bqReg.BQ76_DIR_CMD_PACKPinVoltage,
bqReg.BQ76_DIR_CMD_CC2Current,
bqReg.BQ76_DIR_CMD_TS1Temperature,
bqReg.BQ76_DIR_CMD_TS2Temperature,
bqReg.BQ76_DIR_CMD_TS3Temperature,
bqReg.BQ76_DIR_CMD_IntTemperature
};
if (_max is not null && _el is not null && _bq is not null)
{
try
{
// do is not null check, return/throw accordingly
_max.ReadAllRegisters();
double pws_current = PWS.MeasureCurrent();
double el_current = _el.MeasureLoadCurrent(_batt_load_channel);
double el_voltage = _el.MeasureLoadVoltage(_batt_load_channel);
double pws_voltage = PWS.MeasureVoltage();
double elapsed_hr = _sw_CC.Elapsed.TotalHours;
double passed_charge_mAh = (pws_current - el_current) / 1000 * elapsed_hr;
_sw_CC.Restart();
FileStream stream;
StreamWriter writer;
CsvWriter csv;
try
{
// log filename will be "BatteryCycleAfterProgrammingddMM_HHmm.csv"
stream = System.IO.File.Open(filename, FileMode.Append);
writer = new StreamWriter(stream);
csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
}
catch (Exception e)
{
OnDebugMessage(new DebugMessageArgs(e.ToString()));
return 0;
}
csv.WriteField(DateTime.Now.ToString());
foreach (maxReg reg in Enum.GetValues(typeof(maxReg)))
{
csv.WriteField(_max.registers[(int)reg]);
}
for (int i = 0; i < bqCMDsOfInterest.Length; i++)
{
Int16 value = 0;
_bq.getStatus((ushort)bqCMDsOfInterest[i], 2, ref value);
csv.WriteField(value);
}
csv.WriteField(pws_current);
csv.WriteField(el_current);
csv.WriteField(elapsed_hr);
csv.WriteField(passed_charge_mAh);
csv.WriteField(pws_voltage);
csv.WriteField(el_voltage);
csv.NextRecord();
writer.Close();
return 1;
}
catch (Exception ex)
{
Debug.WriteLine(ex.ToString());
return 0;
}
}
SendDebugMessage("Uninitialized Hardware!");
return 0;
}
#endregion
#region Battery State Functions
/// <summary>
/// Discharges Battery Pack until a voltage, time limit is reached.
/// </summary>
/// <param name="dsg_current">Current to discharge the battery pack</param>
/// <param name="min_voltage">Voltage to achieve to stop discharge</param>
/// <param name="discharge_time">Time limit for discharge</param>
/// <returns>0 if one of the conditions are met</returns>
public async Task<int> DischargePack(double dsg_current, double min_voltage, double discharge_time)
{
if (_bq is null || _el is null)
{
SendDebugMessage("Uninitialized Hardware!");
return -1;
}
// make return plain Task, change back later if needed
Stopwatch _sw = new();
_el.SetLoadCurrent(dsg_current, _batt_load_channel);
SendDebugMessage("Starting Discharge");
_sw.Start();
_el.Connect(_batt_load_channel);
while (_bq.GetPackOutputVoltage() > min_voltage)
{
await Task.Delay(1000);
//Send BMS Safety Alert Data to Debug Window
_bq.Update_safety_alert_a();
LogData();
//Case to break loop if Battery fully discharges
if (_bq._bq76_safety_alert_a.CUV)
{
_sw.Stop();
discharge_time = _sw.Elapsed.TotalHours;
_el.Disconnect(_batt_load_channel);
SendDebugMessage("Battery Fully Discharged!");
SendDebugMessage($"Discharge time: {discharge_time:F2} hr");
return 0;
}
//Case to break loop if "ABORT" button is pressed (used for debugging)
if (_abort)
{
_abort = false;
_sw.Stop();
discharge_time = _sw.Elapsed.TotalHours;
_el.Disconnect(_batt_load_channel);
SendDebugMessage("Abort Pressed!");
SendDebugMessage($"Discharge time: {discharge_time:F2} hr");
return 0;
}
//Case to break loop if time limit is reached
if (_sw.Elapsed.TotalHours > discharge_time)
{
_sw.Stop();
discharge_time = _sw.Elapsed.TotalHours;
_el.Disconnect(_batt_load_channel);
SendDebugMessage("Time Limit Reached!");
SendDebugMessage($"Discharge time: {discharge_time:F2} hr");
return 0;
}
}
//Actions to take when Battery reaches target discharge voltage
_sw.Stop();
discharge_time = _sw.Elapsed.TotalHours;
_el.Disconnect(_batt_load_channel);
SendDebugMessage("Battery Target Voltage Reached!");
SendDebugMessage($"Discharge time: {discharge_time:F2} hr");
return 0;
}
/// <summary>
/// Charges Battery Pack
/// </summary>
/// <param name="chg_current">Maximum charge current</param>
/// <param name="charge_time">Time Limit for the charging cycle</param>
/// <returns>True if charger reached cutoff current, false if process was aborted</returns>
public async Task<int> ChargePack(double chg_current, double charge_time)
{
Stopwatch _sw = new();
double max_voltage = _batt_chg_voltage;
double cutoff = (_batt_c_nom / 50);
PWS.SetVoltage(max_voltage);
PWS.SetCurrentLimit(chg_current);
SendDebugMessage("Starting Charge");
PWS.PowerOn(true);
_sw.Restart();
//wait for charging to start. If charging takes more than 30s to start, then battery is fully charged.
while (PWS.MeasureCurrent() < cutoff)
{
if (_sw.Elapsed.TotalSeconds > 30)
{
_sw.Stop();
PWS.PowerOn(false);
SendDebugMessage("Battery Not Charging!");
SendDebugMessage("Charge time: 0");
return 0;
}
}
_sw.Restart();
while (PWS.MeasureCurrent() > cutoff)
{
LogData();
await Task.Delay(1000);
//Case to break loop if "ABORT" button is pressed (used for debugging)
if (_abort)
{
_abort = false;
PWS.PowerOn(false);
_sw.Stop();
charge_time = _sw.Elapsed.TotalHours;
SendDebugMessage("Abort Pressed!");
SendDebugMessage($"Charge time: {charge_time:F2} hr");
return 0;
}
//Case to break loop if time limit is reached
if (_sw.Elapsed.TotalHours > charge_time)
{
_sw.Stop();
charge_time = _sw.Elapsed.TotalHours;
PWS.PowerOn(false);
SendDebugMessage("Time Limit Reached!");
SendDebugMessage($"Charge time: {charge_time:F2} hr");
return 0;
}
}
//Actions to take when Battery reaches fully charged state.
_sw.Stop();
charge_time = _sw.Elapsed.TotalHours;
PWS.PowerOn(false);
SendDebugMessage("Fully Charged!");
SendDebugMessage($"Charge time: {charge_time:F2} hr");
return 0;
}
/// <summary>
/// Relaxes the Pack Cells
/// </summary>
/// <param name="relax_time_hr">Time (in hours) for pack to relax</param>
/// <returns>True if relaxation time was achieved, false if process was aborted</returns>
public async Task<int> RelaxPack(double relax_time_hr)
{
Stopwatch _sw = new();
SendDebugMessage("Starting Relaxation");
//Disable fets when relaxing
await ToggleBQDsgFet(false);
await ToggleBQChgFet(false);
_sw.Restart();
while (_sw.Elapsed.TotalHours < relax_time_hr)
{
await Task.Delay(1000);
LogData();
if (_abort)
{
_abort = false;
_sw.Stop();
relax_time_hr = _sw.Elapsed.TotalHours;
await ToggleBQDsgFet(true);
await ToggleBQChgFet(true);
SendDebugMessage("Abort Pressed!");
SendDebugMessage($"Relax time: {relax_time_hr:F2} hr");
return 0;
}
}
_sw.Stop();
relax_time_hr = _sw.Elapsed.TotalHours;
await ToggleBQDsgFet(true);
await ToggleBQChgFet(true);
SendDebugMessage("Relaxation time ended!");
SendDebugMessage($"Relax time: {relax_time_hr:F2} hr");
return 0;
}
#endregion
public override async Task<int> Run()
{
_bq = new BQ76952(_BMS);
_max = new MAX17205(_BMS);
_el = new ElectronicLoad(_visa_addr);
filename = $"{_battType}ModelVerification{DateTime.Now:MMdd_HHmm}.csv";
InitData();
//string ID = _el.GetID();
OnRunStatusChanged(new StatusEventArgs(RunStatuses.Testing));
// Set up battery and ensure that FETs are all ON, and that short circuit protection is OFF
_bq.Update_manu_status();
_bq.DisableShortCircuitFETFault();
await ToggleBQFetEn(false);
await ToggleBQChgFet(true);
await ToggleBQDsgFet(true);
await Task.Delay(500);
_sw_CC = new Stopwatch();
_sw_CC.Restart();
await DischargePack(_batt_c_nom, 0, 2);
await ChargePack(_batt_c_nom, 3);
SendDebugMessage("Unplug Power Supply and press ABORT");
await RelaxPack(100);
SendDebugMessage($"Starting Model Verification. Timestamp: {DateTime.Now}");
_bq.Update_safety_alert_a();
while (!_bq._bq76_safety_alert_a.CUV)
{
await DischargePack(_batt_c_nom / 10, 0, 5.0 / 60);
await RelaxPack(0.25f);
_bq.Update_safety_alert_a();
}
SendDebugMessage("Unplug Power Supply and press ABORT");
await RelaxPack(100);
await ChargePack(_batt_c_nom, 3);
return 0;
}
}
}
PROTOCOL APPROVAL
Digital Key: example.com/
REPORT SECTION
Recorded By: TAYLOR BECKHAMDate: 12/19/24
PROTOCOL DEVIATIONS
A small number of deviations to improve test script stability were required in order to ensure continued smooth functioning of the test. However, these changes had no impact on testing performed, the expected behaviour of the test or the device under test. The updated script commit is 94a5bd55969bd2c48e570e55ed9a35909e154641.
DEVICES, COMPONENTS, OR EQUIPMENT USED
Equipment:
EQP-201 - ITECH Power Supply IT6952A - Cal Date: 07/11/2024
EQP-211 - Keysight Technologies Dual Input Electronic Load EL34243A - Cal Date: 02/02/2024
Devices Under Test:
MS-10010 BOM Rev I - Emitter Battery Pack - SN: AV57727-0005
MS-10083 BOM Rev J - Cassette Battery Pack - SN:AV58217-0003
RESULTS AND DISCUSSION
Fuel Gauge Measurement Accuracy, Coulomb Count Accuracy and Voltage Model Accuracy are presented in Figures 1 through 3 and Tables 1 and 2. Details about the raw data analysis and the code used to create these graphs can be found in Attachment 3.
Reported cell voltage accuracy, shown in Figure 1, was calculated by comparing the VCELL register on the MAX17205 IC (green) with the BMS cell voltage value (blue). The %error (red) was found to be within ± 5% at all times during the test.
Figure 1: MS-10083 and MS-10010 Cell Voltage and % Error
Reported current accuracy (fuel gauge reported current vs. electronic load reported current) during active use, shown in Table 1, was calculated by averaging the %error between the measured current and reported current at all times when the battery was charging or discharging. Mean current errors were well within the ±5% allowable range. Note, percent errors were averaged due to a slight offset in reporting time between the fuel gauge and the electronic load.
Table 1: Current Measurement Error During Use:
Reported current accuracy during relaxation was calculated by averaging the fuel gauge current values reported when the battery pack was in a relaxation state. This average current value was shown to be within the allowable range of ± 1mA, as shown in Table 2.
Table 2: Current Measurement Error During Relaxation:
Coulomb Counter Accuracy measurements are shown in Figure 2, where the calculated passed charge (green) was compared to the fuel gauge reported passed charge also referred to as the reported change in capacity (blue). The Mean Absolute Scaled Error (MASE error) (black) can be seen to be less than 2% at all times which is well within the allowable ±10% range.
Figure 2: MS-10083 and MS-10010 Passed Charge and % Error
Open Circuit Voltage Model (also referred to as the Fuel Gage voltage model) comparison is shown in Figure 3, where the internal curve (red) is plotted alongside the calculated curve (blue). The MASE Error (black) is shown to be within the ±5% allowable range at all points along the curve.
Figure 3: MS-10083 and MS-10010 OCV/SOC Relation and % Error
CONCLUSIONS
Overall Result:.
Pass
Fail
Other: _______
Both the Emitter Battery Pack (MS-10010) and the Cassette Battery Pack (MS-10083) passed the acceptance criteria for all fuel gauge monitoring accuracy tests including:
Voltage Monitoring Accuracy
Current Monitoring Accuracy
Fuel Gauge Coulomb Count Accuracy
Fuel Gauge Voltage Model Accuracy
As these are all of the major parameters used by the fuel gauge to monitor the State of Charge of the battery packs, MS-10010 and MS-10083 have been demonstrated to report intuitive State of Charge forecasts to the MX1 Software System. See Appendix A for detailed information regarding IC parameter measurements versus State of Charge direct measurement.
APPENDICES
Appendix A: Parameter Measurement vs SoC Direct Measurement
ATTACHMENTS
VVPR-P01-235 Attachment 1: CASSETTEModelVerification1204_1055.csv
VVPR-P01-235 Attachment 2: EMITTERModelVerification1207_0301.csv
VVPR-P01-235 Attachment 3: Battery Model Verification Data Analysis.ipynb
REPORT APPROVAL
Digital Key: example.com/
Appendix A: Parameter Measurement vs SoC Direct Measurement
The choice to measure the inputs to the fuel gauge rather than measuring the true SoC of the battery pack is due to the advanced nature of the fuel gauge, coupled with the difficulty in measuring a true State of Charge value on a battery pack.
SoC can be defined as [Available Capacity]/[Maximum Battery Capacity]. While this is theoretically simple to calculate, in practice, available capacity changes depending on the situation. For short term capacity updates, we rely on the coulomb counter integrated into the MAX17205 IC, which can provide a direct count of the amount of charge going into or out of the battery by doing an integration of the measured current with a trapezoidal approximation. Any immediate charge/discharge test for validation would heavily rely on this part of the fuel gauge. However, coulomb counter measurements introduce a compounding error that causes the value reported to become significantly more incorrect over time. This compounding error can be seen in Figure A-1. In order to account for this drift, the MAX17205 uses a voltage model to measure the voltage of the battery pack and compare it to a known curve to correct for inaccuracies. This model is used at an unknown rate during use, so it’s very difficult to determine whether the Voltage Model or the Coulomb Counter is being used to make a measurement. The mixing of these models is shown in Figure A-2.
Figure A-1: Coulomb Count Inaccuracy over Time (Source: MAX17205 Datasheet)
Figure A-2: SoC Reporting (Source: MAX17205 Datasheet)
Measuring the true State of Charge of a battery pack is difficult to define in variable load conditions because batteries have a non-linear voltage response to changing load conditions. Because the MX1 has a large variation in current draw, the available capacity can differ wildly, especially during x-ray events or when the device changes between different power states.
Because an unstable SoC reading is unintuitive for an end user, the Max17205 essentially has to change the way it reports available capacity through a converge to empty feature. The fuel gauge calculates the expected time to reach empty, and then changes the rate at which the reported SoC decreases so that the actual SoC and reported SoC converge at 0. This convergence is shown in an example in Figure A-3.  However, that functionality makes it difficult to make the claim that the reported state of charge is always within 5% of the “true” state of charge.
Figure A-3: MAX17205 Converge to Empty Feature (Source: MAX17205 Datasheet)
VVPR-P01-235 Attachment 1: CASSETTEModelVerification1204_1055.csv
VVPR-P01-235 Attachment 2: EMITTERModelVerification1207_0301.csv
VVPR-P01-235 Attachment 3: Battery Model Verification Data Analysis.ipynb

### Table 1
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Refer to ECR-646 |  |  |

### Table 2
| Mean Current Errors |  |  |
| --- | --- | --- |
| Part Number | Charging | Discharging |
| MS-10083 | -0.08% | -0.77% |
| MS-10010 | -0.19% | -0.75% |

### Table 3
| Average Relax Current |  |
| --- | --- |
| MS-10083 | -0.41 mA |
| MS-10010 | -0.53 mA |

### Table 4
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| B | Initial Release | Refer to ECR-654 |  |  |
