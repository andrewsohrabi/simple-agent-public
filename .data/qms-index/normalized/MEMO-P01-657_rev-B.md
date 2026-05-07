# MEMO-P01-657 Rev B: MX1 Isolation Diagram

## Metadata
- Document ID: MEMO-P01-657
- Revision: B
- Prefix: MEMO
- Latest revision: True
- Signed: False
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-657 - MX1 Isolation Diagram_B.docx
- Source path: Example QMS - MedAI/MEMO-P01-657 - MX1 Isolation Diagram_B.docx
- Extraction warnings: none

## Extracted Content
MEMO-P01-657 - MX1 Isolation Diagram_B
Sheet: Revision History
Sheet: Isolation Diagram
Sheet: Isolation List (Emitter)
Sheet: Isolation List (Cassette)

### Table 1
| MedAI MEDICAL, INC |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Document: | MEMO-P01-657 - MX1 Isolation Diagram |  |  |  |  |
| Project Name: | MX1 Portable X-ray System |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |
| Revision | Description | DCO # | Approved By | Eff. Date | Digital Key |
| A | Initial Release | 24-166 | EngineeringQuality EngineeringRegulatory Affairs | 25 Apr 2024 | example.com/ |
| B | Modify Cassette Isolation diagram to add Intertek Requested Changes | Refer ECR-599 |  |  | example.com/ |

### Table 2
| The diagram is maintained within LucidChart and a link to the master document is below:example.com/ |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Table 3
| Location | Component Description | MedAI Part Number | Means of Protection | Working voltage(VDC) | Single fault guarded voltage(VDC) | Conductor exposed internally | Conductor exposed to patient | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | Inductive Rx Coil | M50015 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E2 | Battery Pack | MS-10010 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E3 | Display PCB | ES-10005 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E4 | Collimator Sensor PCB | ES-10020 | 2 MOOP | 5.0 | 5.0 | Yes | No |  |
| E5 | Emitter Power Input PCB | ES-10015 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E6 | Silicone Button Cover | M10004 | 2 MOOP | 3.3 | 33.6 | Yes | No |  |
| E7 | Battery Management PCB | ES-10022 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E8 | Monoblock LV Control PCB (unpotted) | ES-10019 | 2 MOOP | 81.0 | 81.0 | Yes | No |  |
| E9 | Monoblock | MS-10007 | 2 MOOP | 80000.0 | 80000.0 | No | No | Insulated X-RAY TUBE ASSEMBLY, dielectric strength tested to 88KV. |
| E10, E11 | Line Lasers | M10065,M10066 | 2 MOOP | 5.0 | 5.0 | Yes | No |  |
| E12,E13 | Collimator Motors | M50005 | 2 MOOP | 24.0 | 33.6 | Yes | No |  |
| E14 | Collimator Driver PCB | ES-10008 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E15 | Service Port | M10255 | 2 MOOP | 5.0 | 5.0 | Yes | No |  |
| E16 | Emitter Main PCB | ES-10003 | 2 MOOP | 33.6 | 33.6 | Yes | No |  |
| E17 | Sensor PCB | ES-10006 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E18 | Tracking Camera | M50008 | 2 MOOP | 3.3 | 33.6 | Yes | No |  |
| E19 | Imaging Camera | M50010 | 2 MOOP | 3.3 | 33.6 | Yes | No |  |
| E20 | Viewfinder Camera | M50009 | 2 MOOP | 3.3 | 33.6 | Yes | No |  |
| E21 | Fan | M50244 | 2 MOOP | 24.0 | 33.6 | Yes | No |  |
| E22 | NFC Antenna | M50056 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E23, E24 | Sub-GHz Antennas | M50057 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E25, E26 | WiFi Antennas | M50274 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E28 | Emitter LED PCB | ES-10021 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E29 | Forward Button Slider | M10253 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E30 | Forward Button Cover | M10252 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E31 | Downward Button Interface | M50232 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E32 | Downward Button Slider | M10251 | 2 MOOP | 5.0 | 33.6 | Yes | No |  |
| E33 | HMI Display | M50000 | 2 MOOP | 3.3 | 33.6 | Yes | No |  |

### Table 4
| Location | Component Description | MedAI Part Number | Means of Protection | Working Voltage | Single fault guarded voltage | Conductor exposed internally | Conductor exposed to patient | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Isolated USB-C Connector | E51172 | 2 MOPP | 20VDC | 24VDC | Yes | Yes |  |
| C2 | Cassette Main USB Isolation | NA | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C3 | Battery Pack | MS-10083 | 2 MOPP | 16.8VDC | 20VDC | Yes | No |  |
| C4 | Battery Management PCB | ES-10023 | 2 MOPP | 16.8VDC | 20VDC | Yes | No |  |
| C5 | Power Button | E50906 | 2 MOPP | 5VDC | 5VDC | Yes | Yes |  |
| C6 | Select Button | E50906 | 2 MOPP | 5VDC | 5VDC | Yes | Yes |  |
| C7 | Cassette Main PCB | ES-10004 | 2 MOPP | 24VDC | 24VDC | Yes | No |  |
| C8 | Cassette Display | M50003 | 2 MOPP | 12VDC | 20VDC | Yes | No |  |
| C9,C10 | LED Tracking PCBs | ES-10036, ES-10038 | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C11 | Service Port | M11017 | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C12 | Detector Panel | M50004 | 2 MOPP | 24VDC | 24VDC | Yes | No |  |
| C14 | Sub-GHz Antennas | M50057 | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C15 | WiFi Antenna | M50194 | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C16 | Cassette Button PCB | ES-10029 | 2 MOPP | 5VDC | 5VDC | Yes | No |  |
| C17 | Cassette Fan | MS-10385 | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C18 | Cassette Handle Connector Screw Base | M11135 | 2 MOPP | 120VAC | 240VAC | No | Yes | To Applied Part |
| C19 | H1 Charger Brick | M50011 | 2 MOPP | 120VAC | 240VAC | Yes | No |  |
| C20 | Non-Isolated USB-C Connector | E51172 | 2 MOPP | 5VDC | 20VDC | Yes | No |  |
| C21 | Cassette Enclosure | MS-11090 | 2 MOPP | 120VAC | 240VAC | No | No |  |
| C22 | Cassette Enclosure | MS-11090 | 2 MOPP | 24VDC | 24VDC | No | No | To Cassette Main PCB |
| C23 | Cassette Enclosure | MS-11090 | 2 MOPP | 24VDC | 24VDC | No | No | To Battery Packs |
| C24 | Cassette Enclosure | MS-11090 | 2 MOPP | 24VDC | 24VDC | No | No | To Detector |
