# RSK-P01-012 Rev C: MX1 DFMEA

## Metadata
- Document ID: RSK-P01-012
- Revision: C
- Prefix: RSK
- Latest revision: False
- Signed: False
- Obsolete: True
- Software version: unknown
- Source filename: RSK-P01-012 - MX1 DFMEA_C-Obsolete.docx
- Source path: Example QMS - MedAI/RSK-P01-012 - MX1 DFMEA_C-Obsolete.docx
- Extraction warnings: none

## Extracted Content
RSK-P01-012 - MX1 DFMEA_C-Obsolete
Sheet: Signoff
Sheet: DFMEA
Sheet: Sheet4

### Table 1
| MedAI MEDICAL, INC |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Document: | RSK-P01-012 - MX1 DFMEA |  |  |  |  |  |
| Project: | MX1 |  |  |  |  |  |
| APPROVALS / DOCUMENT REVISION HISTORY |  |  |  |  |  |  |
| Revision | Description | DCO # | Approved By | Eff. Date | Digital Key |  |
| A | Initial Release | Refer to ECR-543 |  |  | example.com/ |  |
| B | Addition of the following parts to reflect BOM-055 Rev G updates:- M51184- M51183- M10951- M51187Removed M50057 (replaced with M51184)M10755 moved to different assembly to reflect BOM updatesRevised radiation harms to align with new scoring | Refer to ECR-593 |  |  | example.com/ |  |
| C | Addition of parts to reflect BOM-055 Rev H, I, J, & K updates. | Refer to ECR-631 |  |  | example.com/ |  |

### Table 2
| 0 | Subassy | Subassy Description | Part Number | Description | Intended Function of Component | Potential Hazard/Failure Modes | Possible Causes of Failure | Possible Effects of the Hazard/Failure |  | S1 | O1 | RPN 1 | Current Controls | Req ID | Actions to eliminate cause or enhance prevention or detection | S2 | O2 | RPN 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  | Product Effect(s)(Impact due to Failure Mode) | Patient Effect(s)(Harms or Health Hazard due to Failure Mode) |  |  |  |  |  |  |  |  |  |
| DRSK0001 | MX1 | Top-level assembly | E1 | Emitter | Takes x-ray images, displays preview to user | Fails to take x-ray image | Sudden Opening via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Mechanical Load and Impact Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1849 | MX1 | Top-level assembly | E1 | Emitter | Takes x-ray images, displays preview to user | Fails to take x-ray image | Misalignment due to Mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Mechanical Load and Impact Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1850 | MX1 | Top-level assembly | E1 | Emitter | Takes x-ray images, displays preview to user | Fails to take x-ray image | Jetson Fault or Failure | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1851 | MX1 | Top-level assembly | E1 | Emitter | Takes x-ray images, displays preview to user | Fails to display preview to user | Jetson Fault or Failure | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1852 | MX1 | Top-level assembly | E1 | Emitter | Takes x-ray images, displays preview to user | Fails to display preview to user | Sudden Opening via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Mechanical Load and Impact Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0002 | MX1 | Top-level assembly | C1 | Cassette | Hold Patient Anatomy, Process Images, Enable Tracking with E1 | Fails to Hold Patient Anatomy | Sudden Opening via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Mechanical Load and Impact Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1853 | MX1 | Top-level assembly | C1 | Cassette | Hold Patient Anatomy, Process Images, Enable Tracking with E2 | Fails to Enable Tracking | Misalignment due to Mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Mechanical Load and Impact Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1854 | MX1 | Top-level assembly | C1 | Cassette | Hold Patient Anatomy, Process Images, Enable Tracking with E3 | Fails to Process Images | Jetson Fault or Failure | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0003 | MX1 | Top-level assembly | H1 | Wired Charger | Can be used to charge E1 and C1, can be pluged into W1 | Fails to Charge E1 | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1855 | MX1 | Top-level assembly | H1 | Wired Charger | Can be used to charge E1 and C1, can be pluged into W1 | Fails to Charge C1 | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1856 | MX1 | Top-level assembly | H1 | Wired Charger | Can be used to charge E1 and C1, can be pluged into W1 | Fails to Power W1 | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1857 | MX1 | Top-level assembly | H1 | Wired Charger | Can be used to charge E1 and C1, can be pluged into W1 | Fails to Charge E1 | Improper specifications | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1858 | MX1 | Top-level assembly | H1 | Wired Charger | Can be used to charge E1 and C1, can be pluged into W1 | Fails to Charge C1 | Improper specifications | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1859 | MX1 | Top-level assembly | H1 | Wired Charger | Can be used to charge E1 and C1, can be pluged into W1 | Fails to Power W1 | Improper specifications | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0004 | MX1 | Top-level assembly | P1 | Pelican Case | Contains E1 and C1, Protects Device During Transport | Fails to Contain Device | Sudden opening via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0005 | MX1 | Top-level assembly | M10637 | Physical IFU-MX1 Booklet | Provides information to the operator on use of the device | Fails to provide information to the operator | Poor specifications/formatting | Operator unaware of how to use the device | Delay of Procedure | 4.0 | 3.0 | 12 | Usability Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0006 | MX1 | Top-level assembly | MS-10627 | Puck Box with Pucks | Holds pucks during transport | Mechanical Failure - Box degrades/falls apart | Overstrain during shipping, weakened due to moisture/water | Pucks are loose and could damage devices in case. | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0007 | MX1 | Top-level assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Failure to allow proper spacing | Improper specifications - too small | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0008 | MX1 | Top-level assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Failure to allow proper spacing | Improper specifications - too large | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0009 | MX1 | Top-level assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Failure to maintain integrity of product | Improper material choice | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0010 | MX1 | Top-level assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Failure to maintain integrity of product | Improper material choice | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0011 | MX1 | Top-level assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Allows ingress | Improper seal | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0012 | MX1 | Top-level assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Fails to close and lock | Improper locking mechanism | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0013 | MX1 | Top-level assembly | M50703 | Packaging Tape, Clear, 2",  MIL 2.0 | Clear Tape for Overshipper | Falis to adhere | Improper material choice | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1860 | MX1 | Top-level assembly | M11177 | Physical MX1 Emergency Instructions Booklet | Provides information to operator | Degrades over time | Material Choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1861 | MX1 | Top-level assembly | M11177 | Physical MX1 Emergency Instructions Booklet | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1862 | MX1 | Top-level assembly | M11177 | Physical MX1 Emergency Instructions Booklet | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Usability Test | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2276 | MX1 | Top-level assembly | M11495 | Quick Guide | Provides information to operator | Degrades over time | Material Choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2277 | MX1 | Top-level assembly | M11495 | Quick Guide | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2278 | MX1 | Top-level assembly | M11495 | Quick Guide | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Usability Test | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0014 | MS-10334 | Overshipper Assembly | M10789 | MX1 Cardboard Overshipper | Protect outer shell of case from scratches during shipping | Fails to protect primary package | Fails to protect primary package | Minor scuffing to plastic case | Minor Inconvenience | 1.0 | 4.0 | 4 | None Needed | N/A | No further planned remediation | 1.0 | 4.0 | 4 |
| DRSK0015 | MS-10334 | Overshipper Assembly | M11160 | MX1 Cardboard Overshipper Label | Product Labeling | Protect outer shell of case from scratches during shipping | Fails to protect primary package | Fails to protect primary package | Minor scuffing to plastic case | 4.0 | 1.0 | 4.0 | None Needed | N/A | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK0016 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0017 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0018 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0019 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0020 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0021 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0022 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0023 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0024 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0025 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0026 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0027 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0028 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0029 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0030 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0031 | MS-10334 | Overshipper Assembly | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0032 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0033 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0034 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0035 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0036 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0037 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0038 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0039 | MS-10334 | Overshipper Assembly | M50313 | UN 3481 Label | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0040 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0041 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0042 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0043 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0044 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0045 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0046 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0047 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0048 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0049 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0050 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0051 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0052 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0053 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0054 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0055 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0056 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0057 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0058 | E1 | Emitter ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0059 | E1 | Emitter ASSY | M10247 | P01 Emitter External Cowling, IM | Covers fans | Fail to maintain proper clearances | Mechanical damage from external forces - exposes fan | Basic safety compromised; still operable | Minor Injury (fan blades) | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0060 | E1 | Emitter ASSY | M10247 | P01 Emitter External Cowling, IM | Covers fans | Fail to maintain proper clearances | Mechanical damage from external forces - exposes heat sink | Basic safety compromised; still operable | Minor burn | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0061 | E1 | Emitter ASSY | M10247 | P01 Emitter External Cowling, IM | Covers fans | Fail to maintain proper clearances | Single Fault on heat sink | Basic safety compromised; still operable | Moderate burn | 7.0 | 2.0 | 14 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0062 | E1 | Emitter ASSY | M10247 | P01 Emitter External Cowling, IM | Covers fans | Fail to maintain proper clearances | Fault condition - temp control failure | Basic safety compromised; still operable | Moderate burn | 7.0 | 2.0 | 14 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0063 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0064 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0065 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0066 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0067 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0068 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0069 | E1 | Emitter ASSY | MS-10268 | P01 Emitter Shell L, with thread inserts | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0070 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0071 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0072 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0073 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0074 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0075 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0076 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0077 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0078 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0079 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0080 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0081 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0082 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0083 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0084 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0085 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0086 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0087 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0088 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0089 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0090 | E1 | Emitter ASSY | MS-10134 | Shell R Populated ASSY | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0091 | E1 | Emitter ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Does not direct status LEDs | Incorrect material choice | Status unclear to operator | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR | PRD3.7 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0092 | E1 | Emitter ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0093 | E1 | Emitter ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0094 | E1 | Emitter ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0095 | E1 | Emitter ASSY | M10278 | Emitter Screw Plugs | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0096 | E1 | Emitter ASSY | M10278 | Emitter Screw Plugs | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0097 | E1 | Emitter ASSY | M10278 | Emitter Screw Plugs | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0098 | E1 | Emitter ASSY | M10278 | Emitter Screw Plugs | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0099 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0100 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0101 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0102 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0103 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0104 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Rod breaks | Mechanical damage from external forces | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0105 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Rod unable to trigger button press | Too short | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0106 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to activate (click) | Improper positioning | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Debounce on trigger | RSK_R053 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0107 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1456 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1457 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during single acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0108 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during single acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0109 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Dislodged pins | Mechanical damage from external forces | Trigger falls off; device still operable | Moderate Dissatisfaction | 1.0 | 2.0 | 2 | Incoming inspection | QSP-014 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0110 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Exposed metal components | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0111 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Exposed metal components | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0112 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0113 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0114 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0115 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Fails to seal | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0116 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0117 | E1 | Emitter ASSY | M10251 | Downward Button Slide, IM | Actuates button press from operator | Patient/operator skin reaction | Incorrect material choice | Potential skin irritation to operator | Minor Injury | 4.0 | 2.0 | 8 | Comply to ISO 10993 | PRD20.14 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0118 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Structural integrity compromized | Mechanical damage from external forces | Loss of means of protection; device still operable (limited) | Minor Electrical shock | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK0119 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0120 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0121 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0122 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0123 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0124 | E1 | Emitter ASSY | M10255 | P01 Emitter Service Port Cover, IM | Covers service port | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0125 | E1 | Emitter ASSY | M10491 | Socket button head screw M3x0.5 x 6 Stainless Steel, with thread locker | Fastener | Loose internal or external componenets | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0126 | E1 | Emitter ASSY | M10491 | Socket button head screw M3x0.5 x 6 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0127 | E1 | Emitter ASSY | M10482 | Socket button head screw M3x0.5 x 35 Stainless Steel, with thread locker | Fastener | Loose internal or external componenets | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0128 | E1 | Emitter ASSY | M10482 | Socket button head screw M3x0.5 x 35 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0129 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0130 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0131 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0132 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0133 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0134 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0135 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0136 | E1 | Emitter ASSY | M50157 | Thread Forming Screw M2, 6mm Long | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0137 | E1 | Emitter ASSY | M50264 | PH Thread Forming Screw M3, 16mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0138 | E1 | Emitter ASSY | M50264 | PH Thread Forming Screw M3, 16mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0139 | E1 | Emitter ASSY | M50264 | PH Thread Forming Screw M3, 16mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0140 | E1 | Emitter ASSY | M50264 | PH Thread Forming Screw M3, 16mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0141 | E1 | Emitter ASSY | M50264 | PH Thread Forming Screw M3, 16mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0142 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0143 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0144 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0145 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0146 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0147 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0148 | E1 | Emitter ASSY | M10159 | Label: Emitter Ra Left | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0149 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0150 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0151 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0152 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0153 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0154 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0155 | E1 | Emitter ASSY | M10160 | Label: Emitter Ra Right | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0156 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0157 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0158 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0159 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0160 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0161 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0162 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0163 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0164 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0165 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0166 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0167 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0168 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0169 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0170 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0171 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0172 | E1 | Emitter ASSY | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0173 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Fail to remain transparent | Improper material choice | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0174 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Fail to maintain proper electrical clearances | Improper specification - too thin | Product operable, may not maintain electrical safety | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0175 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0176 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0177 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0178 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 |  | 4.0 | 2.0 | 8 |
| DRSK0179 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0180 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0181 | MS-10132 | Front Face ASSY | M10248 | P01 Emitter Front Face, IM | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0182 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Fail to remain transparent | Improper material choice | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0183 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Fail to maintain proper electrical clearances | Improper specification - too thin | Product operable, may not maintain electrical safety | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0184 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0185 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 |  | 0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 |  | 0 |
| DRSK0186 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0187 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0188 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0189 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0190 | MS-10132 | Front Face ASSY | M10387 | Camera Cover Glass | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0191 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0192 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0193 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0194 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0195 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0196 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0197 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0198 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0199 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0200 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0201 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0202 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0203 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0204 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0205 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0206 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0207 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0208 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0209 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1712 | MS-10132 | Front Face ASSY | M50128 | PU Bumper, 3/8" OD, 5/32" High | Protects interior of device | Internal parts are subjected to ingress | Use outside of temperature range | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0210 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Fail to remain transparent | Improper material choice | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0211 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Fail to maintain proper electrical clearances | Improper specification - too thin | Product operable, may not maintain electrical safety | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0212 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0213 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0214 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0215 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0216 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0217 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0218 | MS-10132 | Front Face ASSY | M10386 | ToF Cover Glass | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0219 | MS-10132 | Front Face ASSY | M50173 | Neodymium Magnet, N55, 1/4" dia x 1/8" thick | Attaches pucks to front face | Falls out of front face | Mechanical damage from external forces | Unable to attach pucks | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0220 | MS-10132 | Front Face ASSY | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0221 | MS-10132 | Front Face ASSY | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin)Parts fall into sterile bag during surgery | Temporary Discomfort | 1.0 | 2.0 | 2.0 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0222 | MS-10132 | Front Face ASSY | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0223 | MS-10132 | Front Face ASSY | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0224 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0225 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0226 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0227 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0228 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0229 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0230 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0231 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0232 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0233 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0234 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0235 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0236 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0237 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0238 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0239 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0240 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0241 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0242 | MS-10134 | Shell R Populated ASSY | MS-10267 | P01 Emitter Shell R, with thread inserts | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0243 | MS-10134 | Shell R Populated ASSY | MS-10136 | E1 Lower Internal ASSY | Sheet Metal bracket that holds the coil and attachs to the PCB | Frame structural integrity fails | Improper material choice - warps/bends | Intermittent charging | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0244 | MS-10134 | Shell R Populated ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil misaligned | Improper geometry | Unable to charge wirelessly/slow charging speed | Delay of Procedure | 4.0 | 3.0 | 12 | Charging Verification Test | PRD5.12 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0245 | MS-10134 | Shell R Populated ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil misaligned | Mechanical damage from external forces | Unable to charge wirelessly/slow charging speed | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0246 | MS-10134 | Shell R Populated ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil disconnected | Poor coil wire routing | Unable to charge wirelessly | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0247 | MS-10134 | Shell R Populated ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil inefficient | Excessive spacing from outer surface | Slow charging speed | Delay of Procedure | 4.0 | 3.0 | 12 | Charging Verification Test | PRD5.12 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0248 | MS-10134 | Shell R Populated ASSY | MS-10010 | Emitter Battery Pack | Holds LEDs in place on the device | Fails to hold LED in proper position | Mechanical damage from external forces | LED status not visible to operator | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0249 | MS-10134 | Shell R Populated ASSY | MS-10401 | HMI and Display PCBA Assembly | N/A - Assembled in house |  |  |  |  |  |  |  |  |  |  |  |  |  |
| DRSK1713 | MS-10401 | HMI and Display PCBA Assembly | MS-10001 | HMI Assembly | Displays data on viewfinder | Display failure | High humidity | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12.0 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1714 | MS-10401 | HMI and Display PCBA Assembly | ES-10005 | P01 Emitter Display PCBA | Displays data to viewfinder | PCB failure | Individual component failure (open/shorts/etc) | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12.0 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1715 | MS-10401 | HMI and Display PCBA Assembly | M50001 | Phillips Rounded Head Thread Forming Screw: #1-1/2in | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1716 | MS-10401 | HMI and Display PCBA Assembly | M50034 | Phillips Rounded Head Thread Forming Screw: #1-1/4in | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1863 | MS-10001 | HMI Assembly | M50000 | Capacitive Touchscreen Display | Provides information to operator, used to display information about tracking and image to be taken | Fails to stay attached | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1864 | MS-10001 | HMI Assembly | M50000 | Capacitive Touchscreen Display | Provides information to operator, used to display information about tracking and image to be taken | Display is obscured | Too much pressure applied to screen - breaks | Display not visible to operator | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1865 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1866 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1867 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Degradation over time | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1868 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1869 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Desktop Review with Nelson Labs | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1870 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1871 | MS-10001 | HMI Assembly | M10002 | Front Bezel, HMI | Protects interior of device, captures display | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1872 | MS-10001 | HMI Assembly | M10003 | Back Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1873 | MS-10001 | HMI Assembly | M10003 | Back Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1874 | MS-10001 | HMI Assembly | M10003 | Back Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Degradation over time | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1875 | MS-10001 | HMI Assembly | M10003 | Back Bezel, HMI | Protects interior of device, captures display | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1876 | MS-10001 | HMI Assembly | M10003 | Back Bezel, HMI | Protects interior of device, captures display | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1877 | MS-10001 | HMI Assembly | M10003 | Back Bezel, HMI | Protects interior of device, captures display | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1878 | MS-10001 | HMI Assembly | M10004 | Silicone Molded Button, HMI | Trigger an Image | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1879 | MS-10001 | HMI Assembly | M10004 | Silicone Molded Button, HMI | Trigger an Image | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1880 | MS-10001 | HMI Assembly | M10004 | Silicone Molded Button, HMI | Trigger an Image | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1881 | MS-10001 | HMI Assembly | M10004 | Silicone Molded Button, HMI | Trigger an Image | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1882 | MS-10001 | HMI Assembly | M10005 | ALS Light Pipe, HMI | Protects interior of device, Direct status LEDs to outside of enclosure | Does not direct status LEDs | Incorrect material choice | Status unclear to operator | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1883 | MS-10001 | HMI Assembly | M10005 | ALS Light Pipe, HMI | Protects interior of device, Direct status LEDs to outside of enclosure | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1884 | MS-10001 | HMI Assembly | M10005 | ALS Light Pipe, HMI | Protects interior of device, Direct status LEDs to outside of enclosure | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1885 | MS-10001 | HMI Assembly | M10005 | ALS Light Pipe, HMI | Protects interior of device, Direct status LEDs to outside of enclosure | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Desktop Review with Nelson Labs | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1886 | MS-10001 | HMI Assembly | M10306 | Display Adhesive, HMICut from M50172 3M 9495LE 12"x12" sheet | Seals E1 Screen Edge | Possible Ingress of foreign liquids | Incorrect Adhesive Choice | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1887 | MS-10001 | HMI Assembly | M50001 | Phillips Rounded Head Thread Forming Screw: #1-1/2in | Fastener | Mechaincal Connection Failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1888 | MS-10001 | HMI Assembly | M50034 | PHILLIPS ROUNDED HEAD THREAD FORMING SCREW: #1-1/4IN | Fastener | Mechaincal Connection Failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1889 | MS-10001 | HMI Assembly | M50002 | PHILLIPS ROUNDED HEAD THREAD FORMING SCREW: #1-1/8IN | Fastener | Mechaincal Connection Failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0250 | MS-10134 | Shell R Populated ASSY | MS-10276 | P01 Emitter Bulkhead w/inserts IM | Protects interior of device and holds heatsinks | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0251 | MS-10134 | Shell R Populated ASSY | MS-10276 | P01 Emitter Bulkhead w/inserts IM | Protects interior of device and holds heatsinks | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0252 | MS-10134 | Shell R Populated ASSY | MS-10276 | P01 Emitter Bulkhead w/inserts IM | Protects interior of device and holds heatsinks | Structural integrity compromized | Mechanical damage from external forces | Loss of means of protection; device still operable (limited) | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK0253 | MS-10134 | Shell R Populated ASSY | M10379 | Emitter Ferrite Mount | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0254 | MS-10134 | Shell R Populated ASSY | M10379 | Emitter Ferrite Mount | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0255 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0256 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0257 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0258 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0259 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0260 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0261 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0262 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0263 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0264 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0265 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0266 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0267 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0268 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0269 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0270 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0271 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0272 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0273 | MS-10134 | Shell R Populated ASSY | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0274 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0275 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0276 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0277 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0278 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0279 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0280 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0281 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0282 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0283 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0284 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0285 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0286 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0287 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0288 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0289 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0290 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0291 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0292 | MS-10134 | Shell R Populated ASSY | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0293 | MS-10134 | Shell R Populated ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Monoblock/Jetson Overheat | Fan speed too slow | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0294 | MS-10134 | Shell R Populated ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Jetson/Monoblock Overheat | Fan stops | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0295 | MS-10134 | Shell R Populated ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Fan Overheat | Debris blocks fan | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Covered fan | RSK_R159 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0296 | MS-10134 | Shell R Populated ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Fan stops | Ingress of dust into enclosure | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Shrouded and covered fan | RSK_R159 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0297 | MS-10134 | Shell R Populated ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Biohazard | Contaminates get into device | Dirty interior | Infection | 1.0 | 2.0 | 2 | Covered fan | RSK_R159 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0298 | MS-10134 | Shell R Populated ASSY | M10370 | FFC, 0.5mm pitch, 33 CKT, 165mm, OSC, Shielded | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0299 | MS-10134 | Shell R Populated ASSY | M10370 | FFC, 0.5mm pitch, 33 CKT, 165mm, OSC, Shielded | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0300 | MS-10134 | Shell R Populated ASSY | M10370 | FFC, 0.5mm pitch, 33 CKT, 165mm, OSC, Shielded | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0301 | MS-10134 | Shell R Populated ASSY | MS-10314 | Emitter Main - Power Input Power Harness | Transmits power between PMUX and Emitter Main PCB | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Minor fire | Minor burn | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0302 | MS-10134 | Shell R Populated ASSY | MS-10314 | Emitter Main - Power Input Power Harness | Transmits power between PMUX and Emitter Main PCB | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0303 | MS-10134 | Shell R Populated ASSY | MS-10314 | Emitter Main - Power Input Power Harness | Transmits power between PMUX and Emitter Main PCB | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0304 | MS-10134 | Shell R Populated ASSY | MS-10314 | Emitter Main - Power Input Power Harness | Transmits power between PMUX and Emitter Main PCB | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0305 | MS-10134 | Shell R Populated ASSY | MS-10313 | Emitter Main - Power Input Data Harness | Transmits data between PMUX and Emitter Main PCB | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Minor fire | Minor burn | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0306 | MS-10134 | Shell R Populated ASSY | MS-10313 | Emitter Main - Power Input Data Harness | Transmits data between PMUX and Emitter Main PCB | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0307 | MS-10134 | Shell R Populated ASSY | MS-10313 | Emitter Main - Power Input Data Harness | Transmits data between PMUX and Emitter Main PCB | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0308 | MS-10134 | Shell R Populated ASSY | MS-10313 | Emitter Main - Power Input Data Harness | Transmits data between PMUX and Emitter Main PCB | Insulation worn by friction over time causing electrical short | Strain relief points become disconnected | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0309 | MS-10134 | Shell R Populated ASSY | MS-10313 | Emitter Main - Power Input Data Harness | Transmits data between PMUX and Emitter Main PCB | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0310 | MS-10134 | Shell R Populated ASSY | MS-10151 | Forward Button Harness | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0311 | MS-10134 | Shell R Populated ASSY | MS-10151 | Forward Button Harness | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0312 | MS-10134 | Shell R Populated ASSY | MS-10151 | Forward Button Harness | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0313 | MS-10134 | Shell R Populated ASSY | MS-10151 | Forward Button Harness | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0314 | MS-10134 | Shell R Populated ASSY | MS-10263 | Downward Button Harness | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0315 | MS-10134 | Shell R Populated ASSY | MS-10263 | Downward Button Harness | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0316 | MS-10134 | Shell R Populated ASSY | MS-10263 | Downward Button Harness | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0317 | MS-10134 | Shell R Populated ASSY | MS-10263 | Downward Button Harness | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0318 | MS-10134 | Shell R Populated ASSY | M50232 | Flush Silicone Translucent Boot | Covers Trigger to seal and dampen button feel | Button does not actuate | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0319 | MS-10134 | Shell R Populated ASSY | MS-10099 | Handle Thermistor Assembly | Measures ambient temperature in the emitter handle | Thermistor failure | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0320 | MS-10134 | Shell R Populated ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Does not direct status LEDs | Incorrect material choice | Status unclear to operator | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR | PRD3.7 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0321 | MS-10134 | Shell R Populated ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0322 | MS-10134 | Shell R Populated ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0323 | MS-10134 | Shell R Populated ASSY | M10254 | Emitter Mode Light Pipe, IM | Protects interior of device, Direct status LEDs to outside of enclosure | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0324 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0325 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0326 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0327 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0328 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0329 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Rod breaks | Mechanical damage from external forces | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0330 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Rod unable to trigger button press | Too short | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0331 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to activate (click) | Improper positioning | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Debounce on trigger | RSK_R053 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0332 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1458 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 |  | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0333 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during single acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1459 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during single acquisition ; uncontrolled radiation output | Negligible Radiation Tissue Reaction | 1.0 | 2.0 |  | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0334 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Dislodged pins | Mechanical damage from external forces | Trigger falls off; device still operable | Moderate Dissatisfaction | 1.0 | 2.0 | 2 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0335 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Exposed metal components | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0336 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Exposed metal components | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0337 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0338 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0339 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0340 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Fails to seal | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0341 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0342 | MS-10134 | Shell R Populated ASSY | M10253 | Forward Button Slide, IM | Actuates button press from operator | Patient/operator skin reaction | Incorrect material choice | Potential skin irritation to operator | Minor Injury | 4.0 | 2.0 | 8 | Comply to ISO 10993 | PRD20.14 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0343 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0344 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0345 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0346 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0347 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0348 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Rod breaks | Mechanical damage from external forces | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0349 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Rod unable to trigger button press | Too short | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0350 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to activate (click) | Improper positioning | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Debounce on trigger | RSK_R053 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0351 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1460 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 |  | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0352 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during single acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1461 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during single acquisition ; uncontrolled radiation output | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0353 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Dislodged pins | Mechanical damage from external forces | Trigger falls off; device still operable | Moderate Dissatisfaction | 1.0 | 2.0 | 2 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0354 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Exposed metal components | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0355 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Exposed metal components | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0356 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0357 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0358 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0359 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Fails to seal | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0360 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0361 | MS-10134 | Shell R Populated ASSY | M10252 | Forward Button Cover, IM | Actuates button press from operator | Patient/operator skin reaction | Incorrect material choice | Potential skin irritation to operator | Minor Injury | 4.0 | 2.0 | 8 | Comply to ISO 10993 | PRD20.14 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0362 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0363 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0364 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0365 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0366 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0367 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Rod breaks | Mechanical damage from external forces | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0368 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Rod unable to trigger button press | Too short | Product inoperable; unable to use trigger | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0369 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to activate (click) | Improper positioning | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Debounce on trigger | RSK_R053 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0370 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1462 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during DDR acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Maximum time set on DDR | PRD2.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0371 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to deactivate (unclick) | Button mechanism fails - component failure | Trigger initiates; button does not unpress during single acquisition | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1463 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to deactivate (unclick) | Sticky button - improper geometry | Trigger initiates; button does not unpress during single acquisition ; uncontrolled radiation output | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Debounce on trigger | RSK_R053 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0372 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Dislodged pins | Mechanical damage from external forces | Trigger falls off; device still operable | Moderate Dissatisfaction | 1.0 | 2.0 | 2 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0373 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Exposed metal components | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0374 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Exposed metal components | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0375 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0376 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0377 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0378 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Fails to seal | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0379 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Cleaning Verification Test | PRD13.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0380 | MS-10134 | Shell R Populated ASSY | M10256 | Forward Button Base, IM | Actuates button press from operator | Patient/operator skin reaction | Incorrect material choice | Potential skin irritation to operator | Minor Injury | 4.0 | 2.0 | 8 | Comply to ISO 10993 | PRD20.14 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0381 | MS-10134 | Shell R Populated ASSY | M10319 | Emitter Handle Strain Relief Bracket | Relives strain on cables in handle | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0382 | MS-10134 | Shell R Populated ASSY | M10319 | Emitter Handle Strain Relief Bracket | Relives strain on cables in handle | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0383 | MS-10134 | Shell R Populated ASSY | M10388 | Cable tie mount, 2 pos, 130 deg | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0384 | MS-10134 | Shell R Populated ASSY | M10388 | Cable tie mount, 2 pos, 130 deg | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0385 | MS-10134 | Shell R Populated ASSY | M10491 | Socket button head screw M3x0.5 x 6 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0386 | MS-10134 | Shell R Populated ASSY | M50269 | Thread Forming Screw M2.5, 6mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0387 | MS-10134 | Shell R Populated ASSY | M50269 | Thread Forming Screw M2.5, 6mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0388 | MS-10134 | Shell R Populated ASSY | M50269 | Thread Forming Screw M2.5, 6mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0389 | MS-10134 | Shell R Populated ASSY | M50269 | Thread Forming Screw M2.5, 6mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0390 | MS-10134 | Shell R Populated ASSY | M10485 | Shoulder Screw, 4 mm Shoulder Dia, 8 mm Shoulder Len, M3, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0391 | MS-10134 | Shell R Populated ASSY | M50175 | Spring, 0.25" Long, 0.210" OD, 0.174" ID | Applies pressure to ceiling interface (bulkhead and heatsinks) | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Accumulation of dust internally; device inoperable over time | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0392 | MS-10134 | Shell R Populated ASSY | M50175 | Spring, 0.25" Long, 0.210" OD, 0.174" ID | Applies pressure to ceiling interface (bulkhead and heatsinks) | Internal parts are subjected to ingress | Improper geometry | Accumulation of dust internally; device inoperable over time | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0393 | MS-10134 | Shell R Populated ASSY | M50295 | Acrylic Adhesive Tape 3/8" | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0401 | MS-10134 | Shell R Populated ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0402 | MS-10134 | Shell R Populated ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0409 | MS-10134 | Shell R Populated ASSY | M50153 | Thermal Paste, TC3 | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0410 | MS-10134 | Shell R Populated ASSY | M50153 | Thermal Paste, TC3 | Thermal protection for the cassette | Overheat | Improper material choice | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0411 | MS-10134 | Shell R Populated ASSY | M50326 | Cable Tie, 15" | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0412 | MS-10134 | Shell R Populated ASSY | M50326 | Cable Tie, 15" | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0413 | MS-10134 | Shell R Populated ASSY | M50386 | Ferrite, Trigger | Reduces EMI | Does not reduce EMI to device | Improper specifications | Device malfunction, may still be operable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0414 | MS-10134 | Shell R Populated ASSY | M50386 | Ferrite, Trigger | Reduces EMI | Fails to reduce EMI from device | Improper specifications | None | Interference with other electronics | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0415 | MS-10134 | Shell R Populated ASSY | M10246 | P01 Emitter Cowling Base, IM | Covers fans | Fail to maintain proper clearances | Mechanical damage from external forces - exposes fan | Basic safety compromised; still operable | Minor Injury (fan blades) | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0416 | MS-10134 | Shell R Populated ASSY | M10246 | P01 Emitter Cowling Base, IM | Covers fans | Fail to maintain proper clearances | Mechanical damage from external forces - exposes heat sink | Basic safety compromised; still operable | Minor burn | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0417 | MS-10134 | Shell R Populated ASSY | M10246 | P01 Emitter Cowling Base, IM | Covers fans | Fail to maintain proper clearances | Fault condition - temp control failure | Basic safety compromised; still operable | Moderate burn | 7.0 | 2.0 | 14 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0418 | MS-10134 | Shell R Populated ASSY | M10246 | P01 Emitter Cowling Base, IM | Covers fans | Fail to maintain proper clearances | Single Fault on heat sink | Basic safety compromised; still operable | Moderate burn | 7.0 | 2.0 | 14 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0419 | MS-10134 | Shell R Populated ASSY | M50198 | PH Thread Forming Screw M3, 8mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0420 | MS-10134 | Shell R Populated ASSY | M50198 | PH Thread Forming Screw M3, 8mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0421 | MS-10134 | Shell R Populated ASSY | M50198 | PH Thread Forming Screw M3, 8mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0422 | MS-10134 | Shell R Populated ASSY | M50198 | PH Thread Forming Screw M3, 8mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0423 | MS-10134 | Shell R Populated ASSY | M50045 | FFC, 0.5mm pitch, 14 Ckt, 76mm | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0424 | MS-10134 | Shell R Populated ASSY | M50045 | FFC, 0.5mm pitch, 14 Ckt, 76mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0425 | MS-10134 | Shell R Populated ASSY | M50045 | FFC, 0.5mm pitch, 14 Ckt, 76mm | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0426 | MS-10134 | Shell R Populated ASSY | M50045 | FFC, 0.5mm pitch, 14 Ckt, 76mm | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0427 | MS-10134 | Shell R Populated ASSY | M50045 | FFC, 0.5mm pitch, 14 Ckt, 76mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0428 | MS-10134 | Shell R Populated ASSY | M50045 | FFC, 0.5mm pitch, 14 Ckt, 76mm | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1842 | MS-10134 | Shell R Populated ASSY | M10951 | MX1 Emitter Sub-GHz Bracket | Mounts M51183 antenna to Emitter Shell | Fails to mount antenna to Emitter Shell | Sudden disconnect via mechanical damage | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 1.0 | 1 |
| DRSK1843 | MS-10134 | Shell R Populated ASSY | M51183 | Molex Sub-GHz Antenna | Wireless connectivity/communication with Cassette | Antenna fails | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1844 | MS-10134 | Shell R Populated ASSY | M51184 | Ezuiro Sub-GHz Antenna | Wireless connectivity/communication with Foot Pedal | Antenna fails | Mechanical damage from external forces | Foot Pedal inoperable | Minor Dissatisfaction or no results when using Foot Pedal | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1845 | MS-10134 | Shell R Populated ASSY | M51187 | 3M Acrylic Adhesive DP8705NS | Adheres bracket to shell | Adhesive fails, bracket not attached to Emitter Shell | Improper/Inadequate adhesive | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 1.0 | 1 |
| DRSK1846 | MS-10134 | Shell R Populated ASSY | M10755 | Emitter Sub-GHz Coax Cowling | Retains both antenna connections to module | Part becomes dislodged | Mechanical damage from external forces | Antennas no longer have an additional level of retention | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 1.0 | 1 |
| DRSK1847 | MS-10134 | Shell R Populated ASSY | M10483 | Socket button head screw M3x0.5 x 4 Stainless Steel, with thread locker | Connects Emitter Sub-GHz Coax Cowling to Emitter Main | Parts become dislodged | Mechanical damage from external forces | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 1.0 | 1 |
| DRSK1890 | MS-10134 | Shell R Populated ASSY | M10750 | HMI Handle FFC Guard | Prevents FFC being crushed during assembly, acts a point of strain relief for cables traveling through the handle | Excessive Wear on Cables | Improper Geometry | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1891 | MS-10134 | Shell R Populated ASSY | M10750 | HMI Handle FFC Guard | Prevents FFC being crushed during assembly, acts a point of strain relief for cables traveling through the handle | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1892 | MS-10134 | Shell R Populated ASSY | M10750 | HMI Handle FFC Guard | Prevents FFC being crushed during assembly, acts a point of strain relief for cables traveling through the handle | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1893 | MS-10134 | Shell R Populated ASSY | M10750 | HMI Handle FFC Guard | Prevents FFC being crushed during assembly, acts a point of strain relief for cables traveling through the handle | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1894 | MS-10134 | Shell R Populated ASSY | M10750 | HMI Handle FFC Guard | Prevents FFC being crushed during assembly, acts a point of strain relief for cables traveling through the handle | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1895 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Structural failure under weight or load | Material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1896 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Structural failure under weight or load | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1897 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1898 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1899 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 2.0 | 8 | Compliance to ISO 10993 | PRD20.16 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1900 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1901 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Fails to insulate | Material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1902 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1903 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1904 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Material Choice | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1905 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1906 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1907 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1908 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1909 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1910 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1911 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1912 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1913 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10244 | P01 Emitter Shell L, IM | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1914 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Structural failure under weight or load | Material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1915 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Structural failure under weight or load | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1916 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1917 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1918 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 2.0 | 8 | Compliance to ISO 10993 | PRD20.16 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1919 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1920 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Fails to insulate | Material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1921 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1922 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1923 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Material Choice | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1924 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1925 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1926 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1927 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1928 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1929 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1930 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1931 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1932 | MS-10268 | P01 Emitter Shell L, with thread inserts | M10243 | P01 Emitter Shell R, IM | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1933 | MS-10267 | P01 Emitter Shell R, with thread inserts | M50179 | IUTB-M3-Hi-TechFastenersInc. | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1934 | MS-10267 | P01 Emitter Shell R, with thread inserts | M50180 | Thread Insert, IBB-M3-4 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1935 | MS-10010 | Emitter Battery Pack | MS-10096 | Emitter Battery Pack - Harness | Allows Power Transfer from Battery Pack to PMUX | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1936 | MS-10010 | Emitter Battery Pack | MS-10096 | Emitter Battery Pack - Harness | Allows Power Transfer from Battery Pack to PMUX | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1937 | MS-10010 | Emitter Battery Pack | MS-10096 | Emitter Battery Pack - Harness | Allows Power Transfer from Battery Pack to PMUX | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1938 | MS-10010 | Emitter Battery Pack | MS-10096 | Emitter Battery Pack - Harness | Allows Power Transfer from Battery Pack to PMUX | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1939 | MS-10010 | Emitter Battery Pack | MS-10096 | Emitter Battery Pack - Harness | Allows Power Transfer from Battery Pack to PMUX | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1940 | MS-10010 | Emitter Battery Pack | M10079 | Emitter Rib Support | Provides Support for Battery Cells, Assists in Constraining Battery Pack within E1 | Excessive Wear on Cables | Improper Geometry | Product inoperable | Moderate Fire | 7.0 | 2.0 | 14 | Incoming Inspection | QSP-014 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1941 | MS-10010 | Emitter Battery Pack | M10079 | Emitter Rib Support | Provides Support for Battery Cells, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1942 | MS-10010 | Emitter Battery Pack | M10079 | Emitter Rib Support | Provides Support for Battery Cells, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1943 | MS-10010 | Emitter Battery Pack | M10079 | Emitter Rib Support | Provides Support for Battery Cells, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1944 | MS-10010 | Emitter Battery Pack | M10079 | Emitter Rib Support | Provides Support for Battery Cells, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1945 | MS-10010 | Emitter Battery Pack | M10326 | Emitter End Cap | Protects BMS, Assists in Constraining Battery Pack within E1 | Excessive Wear on Cables | Improper Geometry | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1946 | MS-10010 | Emitter Battery Pack | M10326 | Emitter End Cap | Protects BMS, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1947 | MS-10010 | Emitter Battery Pack | M10326 | Emitter End Cap | Protects BMS, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1948 | MS-10010 | Emitter Battery Pack | M10326 | Emitter End Cap | Protects BMS, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1949 | MS-10010 | Emitter Battery Pack | M10326 | Emitter End Cap | Protects BMS, Assists in Constraining Battery Pack within E1 | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1950 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Misalignment | Improper Geometry | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1951 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1952 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1953 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1954 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1955 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Misalignment | Adhesive Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1956 | MS-10010 | Emitter Battery Pack | M10086 | Square Foam Spacer | Protects BMS, Shock Asorbtion | Fails to Absorb Shock | Material Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1957 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Misalignment | Improper Geometry | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1958 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1959 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1960 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1961 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1962 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Misalignment | Adhesive Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1963 | MS-10010 | Emitter Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Fails to Absorb Shock | Material Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1964 | MS-10010 | Emitter Battery Pack | M10287 | EBP Fish Paper Profile | Insulate Battery Cells From BMS | Misalignment | Improper Geometry | Short, Device Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1965 | MS-10010 | Emitter Battery Pack | M10287 | EBP Fish Paper Profile | Insulate Battery Cells From BMS | Structural failure under weight or load | Structural failure due to fatigue | Short, Device Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1966 | MS-10010 | Emitter Battery Pack | M10287 | EBP Fish Paper Profile | Insulate Battery Cells From BMS | Structural failure under weight or load | Material Choice | Short, Device Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1967 | MS-10010 | Emitter Battery Pack | M10287 | EBP Fish Paper Profile | Insulate Battery Cells From BMS | Structural failure under weight or load | Part degrades from aging | Short, Device Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1968 | MS-10010 | Emitter Battery Pack | M10287 | EBP Fish Paper Profile | Insulate Battery Cells From BMS | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Short, Device Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1969 | MS-10010 | Emitter Battery Pack | M10287 | EBP Fish Paper Profile | Insulate Battery Cells From BMS | Misalignment | Adhesive Choice | Short, Device Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1970 | MS-10010 | Emitter Battery Pack | M10329 | EBP Fish Paper Large Profile | Insulate Battery Cells | Misalignment | Improper Geometry | Conductors not insulated | Minor Electrical shock | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1971 | MS-10010 | Emitter Battery Pack | M10329 | EBP Fish Paper Large Profile | Insulate Battery Cells | Structural failure under weight or load | Structural failure due to fatigue | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1972 | MS-10010 | Emitter Battery Pack | M10329 | EBP Fish Paper Large Profile | Insulate Battery Cells | Structural failure under weight or load | Material Choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1973 | MS-10010 | Emitter Battery Pack | M10329 | EBP Fish Paper Large Profile | Insulate Battery Cells | Structural failure under weight or load | Part degrades from aging | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1974 | MS-10010 | Emitter Battery Pack | M10329 | EBP Fish Paper Large Profile | Insulate Battery Cells | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1975 | MS-10010 | Emitter Battery Pack | M10329 | EBP Fish Paper Large Profile | Insulate Battery Cells | Misalignment | Adhesive Choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1976 | MS-10010 | Emitter Battery Pack | M50023 | 18650 Rechargeable Battery Cells | Provide Power to E1 | Fails to Provide Sufficient Power | Cell Choice | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1977 | M50023 | Emitter Battery Pack | M50023 | 18651 Rechargeable Battery Cells | Provide Power to E2 | Power Failure After Repeated Use | Cell Choice | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1978 | MS-10010 | Emitter Battery Pack | M50342 | Heat Shrink, FIT221 3/16 | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1979 | MS-10010 | Emitter Battery Pack | M50344 | Heat Shrink, FIT221 1/8 BLK | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1980 | MS-10010 | Emitter Battery Pack | M50206 | Shrink Wrap (124mm) | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1981 | MS-10010 | Emitter Battery Pack | M50317 | Shrink Wrap (145mm) | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1982 | MS-10010 | Emitter Battery Pack | M50346 | Hot Melt, Bostik 2124 | Provides intercell Support | Structural failure under weight or load | Adhesive Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1983 | MS-10010 | Emitter Battery Pack | M50346 | Hot Melt, Bostik 2124 | Provides intercell Support | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1984 | MS-10010 | Emitter Battery Pack | M50346 | Hot Melt, Bostik 2124 | Provides intercell Support | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1985 | MS-10010 | Emitter Battery Pack | M50351 | Kapton Tape, K250-3/4 | Contrains Cables on Pack, Assists in Constraining Cap | Structural failure under weight or load | Adhesive Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1986 | MS-10010 | Emitter Battery Pack | M50351 | Kapton Tape, K250-3/4 | Contrains Cables on Pack, Assists in Constraining Cap | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1987 | MS-10010 | Emitter Battery Pack | M50351 | Kapton Tape, K250-3/4 | Contrains Cables on Pack, Assists in Constraining Cap | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1988 | MS-10010 | Emitter Battery Pack | M50347 | CTR250/1.125 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1989 | MS-10010 | Emitter Battery Pack | M50892 | STR156/1.25/062 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1990 | MS-10010 | Emitter Battery Pack | M50896 | CT250/1.25 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1991 | MS-10010 | Emitter Battery Pack | M50353 | CT250/1.375 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1992 | MS-10010 | Emitter Battery Pack | M50378 | AIM SAC305 flux core solder | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1993 | MS-10010 | Emitter Battery Pack | M50378 | AIM SAC305 flux core solder | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1994 | MS-10010 | Emitter Battery Pack | M50378 | AIM SAC305 flux core solder | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1995 | MS-10010 | Emitter Battery Pack | M10358 | Label: Emitter Battery Pack | Provides information to operator | Detaches from surface | Adhesive Choice | Label information not available | Operator Inconvenience or Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1996 | MS-10010 | Emitter Battery Pack | M10358 | Label: Emitter Battery Pack | Provides information to operator | Degrades over time | Material Choice | Label information not available | Operator Inconvenience or Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1997 | MS-10096 | Emitter Battery Pack - Harness | ES-10023 | Emitter Battery Management System (BMS) | Facilitates Power Transfer to PMUX from Cells | PCB failure | Individual component failure (open/shorts/etc) | Battery Failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use Battery compliant with IEC-62133 | PRD20.18 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1998 | MS-10096 | Emitter Battery Pack - Harness | M50355 | UL 1213 16 RED | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1999 | MS-10096 | Emitter Battery Pack - Harness | M50356 | UL 1213 16 BLK | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2000 | MS-10096 | Emitter Battery Pack - Harness | M50357 | UL 1213 26 WHT | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2001 | MS-10096 | Emitter Battery Pack - Harness | M50358 | UL 1213 26 BLK | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2002 | MS-10096 | Emitter Battery Pack - Harness | M50359 | UL 1213 26 BLU | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2003 | MS-10096 | Emitter Battery Pack - Harness | M50374 | UL 1213 26 GRN | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2004 | MS-10096 | Emitter Battery Pack - Harness | M50360 | UL 1213 24 YEL | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2005 | MS-10096 | Emitter Battery Pack - Harness | M50361 | UL 1213 24 BLU | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2006 | MS-10096 | Emitter Battery Pack - Harness | M50362 | UL 1213 24 BRW | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2007 | MS-10096 | Emitter Battery Pack - Harness | M50363 | UL 1213 24 GRY | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2008 | MS-10096 | Emitter Battery Pack - Harness | M50364 | UL 1213 24 ORG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2009 | MS-10096 | Emitter Battery Pack - Harness | M50365 | UL 1213 24 VIO | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2010 | MS-10096 | Emitter Battery Pack - Harness | M50366 | UL 1213 24 WHT | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2011 | MS-10096 | Emitter Battery Pack - Harness | M50319 | Contact, 1.25mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2012 | MS-10096 | Emitter Battery Pack - Harness | M50048 | Conn Plug, 1.25mm CLIK-mate, 4 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2013 | MS-10096 | Emitter Battery Pack - Harness | M50276 | XT60 Connector, Female | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2014 | MS-10276 | Emitter Bulkhead w/inserts IM | M10245 | P01 Emitter Bulkhead, IM | Holds E1 Heatsinks | Structural failure under weight or load | Material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2015 | MS-10276 | Emitter Bulkhead w/inserts IM | M10245 | P01 Emitter Bulkhead, IM | Holds E1 Heatsinks | Structural failure under weight or load | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2016 | MS-10276 | Emitter Bulkhead w/inserts IM | M10245 | P01 Emitter Bulkhead, IM | Holds E1 Heatsinks | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2017 | MS-10276 | Emitter Bulkhead w/inserts IM | M10245 | P01 Emitter Bulkhead, IM | Holds E1 Heatsinks | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2018 | MS-10276 | Emitter Bulkhead w/inserts IM | M10245 | P01 Emitter Bulkhead, IM | Holds E1 Heatsinks | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2019 | MS-10276 | Emitter Bulkhead w/inserts IM | M10245 | P01 Emitter Bulkhead, IM | Holds E1 Heatsinks | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2020 | MS-10276 | Emitter Bulkhead w/inserts IM | M50180 | Thread Insert, IBB-M3-4 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2021 | MS-10314 | Emitter Main - Power Input Power Harness | M50229 | Conn Plug, 2.00mm CLIK-mate, 4 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2022 | MS-10314 | Emitter Main - Power Input Power Harness | M50320 | Contact, 2.00mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2023 | MS-10314 | Emitter Main - Power Input Power Harness | M50061 | Wire, 22 AWG, Black | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2024 | MS-10314 | Emitter Main - Power Input Power Harness | M50063 | Wire, 22 AWG, Red | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2025 | MS-10314 | Emitter Main - Power Input Power Harness | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2026 | MS-10314 | Emitter Main - Power Input Power Harness | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2027 | MS-10314 | Emitter Main - Power Input Power Harness | M50385 | Ferrite, Emitter Main Power | Filter EMI | Fails to filter enough EMI | Incorrect Part Specification | No Effect | Possible Issues with nearby electronics | 4.0 | 2.0 | 8 | Compliance to IEC  60601-2 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2028 | MS-10313 | Emitter Main - Power Input Data Harness | M50230 | Conn Plug, 1.25mm CLIK-mate, 6 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2029 | MS-10313 | Emitter Main - Power Input Data Harness | M50319 | Contact, 1.25mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2030 | MS-10313 | Emitter Main - Power Input Data Harness | M50105 | Wire, 28 AWG, White | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2031 | MS-10313 | Emitter Main - Power Input Data Harness | M50106 | Wire, 28 AWG, Green | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2032 | MS-10313 | Emitter Main - Power Input Data Harness | M50107 | Wire, 28 AWG, Yellow | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2033 | MS-10313 | Emitter Main - Power Input Data Harness | M50255 | Wire, 28 AWG, Blue | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2034 | MS-10313 | Emitter Main - Power Input Data Harness | M50256 | Wire, 28 AWG, Slate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2035 | MS-10313 | Emitter Main - Power Input Data Harness | M50257 | Wire, 28 AWG, Brown | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2036 | MS-10313 | Emitter Main - Power Input Data Harness | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2037 | MS-10313 | Emitter Main - Power Input Data Harness | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2038 | MS-10151 | Forward Button Harness | MS-50164 | Push Button,  P9-111121W | Triggers the taking of an image | Fails to be depressed | Incorrect Part Specification | Button Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2039 | MS-10151 | Forward Button Harness | M50104 | Wire, 28 AWG, Black | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2040 | MS-10151 | Forward Button Harness | M50120 | Wire, 28 AWG, Red | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2041 | MS-10151 | Forward Button Harness | M50050 | Conn Plug, 1.25mm CLIK-mate, 2 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2042 | MS-10151 | Forward Button Harness | M50319 | Contact, 1.25mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2043 | MS-10151 | Forward Button Harness | M50156 | Heat Shrink, 2.11mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2044 | MS-10151 | Forward Button Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2045 | MS-10151 | Forward Button Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Button Inoperable | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2046 | MS-10151 | Forward Button Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Button Inoperable | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2047 | MS-10151 | Forward Button Harness | M50424 | Heat Shrink, 3.10mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2048 | MS-10263 | Downward Button Harness | MS-50164 | Push Button,  P9-111121W | Triggers the taking of an image | Fails to be depressed | Incorrect Part Specification | Button Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2049 | MS-10263 | Downward Button Harness | M50104 | Wire, 28 AWG, Black | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2050 | MS-10263 | Downward Button Harness | M50120 | Wire, 28 AWG, Red | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2051 | MS-10263 | Downward Button Harness | M50050 | Conn Plug, 1.25mm CLIK-mate, 2 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2052 | MS-10263 | Downward Button Harness | M50319 | Contact, 1.25mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2053 | MS-10263 | Downward Button Harness | M50156 | Heat Shrink, 2.11mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2054 | MS-10263 | Downward Button Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2055 | MS-10263 | Downward Button Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Button Inoperable | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2056 | MS-10263 | Downward Button Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Button Inoperable | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2057 | MS-10263 | Downward Button Harness | M50424 | Heat Shrink, 3.10mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2058 | MS-10099 | Handle Thermistor Assembly | M50072 | Conn Plug, 1mm Pico-Lock, 2 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2059 | MS-10099 | Handle Thermistor Assembly | M50047 | Contact, 1mm, Pico-Lock | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Button Inoperable | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2060 | MS-10099 | Handle Thermistor Assembly | M50119 | Thermistor 10Kohm | Monitors Temperature in Handle | Fails to report temperature | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2061 | MS-10099 | Handle Thermistor Assembly | M50119 | Thermistor 10Kohm | Monitors Temperature in Handle | Fails to report temperature | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0429 | MS-10136 | E1 Lower Internal ASSY | MS-10378 | Symmetrical LED Bracket ASSY | Holds LEDs in place on the device | Fails to hold LED in proper position | Mechanical damage from external forces | LED status not visible to operator | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0430 | MS-10136 | E1 Lower Internal ASSY | ES-10021 | Emitter LED PCBA | Indicator LEDs for operator | Short LED | Faulty LED | LEDs not displayed; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0431 | MS-10136 | E1 Lower Internal ASSY | ES-10021 | Emitter LED PCBA | Indicator LEDs for operator | Open LED | Faulty LED | LEDs not displayed; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0432 | MS-10136 | E1 Lower Internal ASSY | MS-10148 | Emitter Main ASSY | Assem that controls power and data within the emitter (motherboard) | Thermistor failure | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0433 | MS-10136 | E1 Lower Internal ASSY | MS-10405 | Jetson with Heat pipe Assembly | Controls Device, Ensures Device is Fuctioning Properly, Staying at a regulated temperature | Jetson Thermal Fault, Jetson Failure | Insufficient Power | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2062 | MS-10136 | E1 Lower Internal ASSY | MS-10405 | Jetson with Heat pipe Assembly | Controls Device, Ensures Device is Fuctioning Properly, Staying at a regulated temperature | Jetson Thermal Fault, Jetson Failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2063 | MS-10136 | E1 Lower Internal ASSY | MS-10405 | Jetson with Heat pipe Assembly | Controls Device, Ensures Device is Fuctioning Properly, Staying at a regulated temperature | Jetson Thermal Fault, Jetson Failure | Thermal Fault | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0434 | MS-10136 | E1 Lower Internal ASSY | MS-10155 | E1 X-ray ASSY | Generate x-rays beam | Fails to Generate x-ray beam | Insufficient Power | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2064 | MS-10136 | E1 Lower Internal ASSY | MS-10155 | E1 X-ray ASSY | Generate x-rays beam | Fails to Generate x-ray beam | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2065 | MS-10136 | E1 Lower Internal ASSY | MS-10155 | E1 X-ray ASSY | Generate x-rays beam | Fails to Generate x-ray beam | Thermal Fault | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0435 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Radiation | Inadequate input V | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0436 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0437 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | ISTA Testing | PRD20.26 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0438 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | Overcurrent | Operator Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0439 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | May disable laser guidance but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0440 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Control signal open | Loose connection to connector | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Locking connectors | RSK_R196 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0441 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Improper specifications/geometry | Incorrect beam alignment shown; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0442 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Incorrect material choice | Incorrect beam alignment shown; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0443 | MS-10136 | E1 Lower Internal ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Misalligned (stackup error) | Incorrect beam alignment shown; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0444 | MS-10136 | E1 Lower Internal ASSY | MS-10149 | Collimator Bracket-Camera ASSY | Assem that contains all necessary sensors to detect and measure the device's physical enviroment | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0445 | MS-10136 | E1 Lower Internal ASSY | MS-10272 | Monoblock Heat Pipe ASSY, w/ Hardware | Draw heat from one location to another | Jetson/Monoblock Overheat | Improper pipe routing | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0446 | MS-10136 | E1 Lower Internal ASSY | MS-10272 | Monoblock Heat Pipe ASSY, w/ Hardware | Draw heat from one location to another | Jetson/Monoblock Overheat | Mechanical damage from external forces | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0447 | MS-10136 | E1 Lower Internal ASSY | MS-10272 | Monoblock Heat Pipe ASSY, w/ Hardware | Draw heat from one location to another | Jetson/Monoblock Overheat | Insufficient heat transfer capacity | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0448 | MS-10136 | E1 Lower Internal ASSY | M10307 | LVPS Heat Sink Bracket | Stabalizes camera mount; interfaces with power supply monoblock components to heatsink | Structural integrity compromized | Mechanical damage from external forces | Reduced monoblock performance; reduced cooling; device still operable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0449 | MS-10136 | E1 Lower Internal ASSY | M10307 | LVPS Heat Sink Bracket | Stabalizes camera mount; interfaces with power supply monoblock components to heatsink | Structural integrity compromized | Improper geometry | Reduced monoblock performance; reduced cooling; device still operable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0450 | MS-10136 | E1 Lower Internal ASSY | MS-10048 | Emitter Main Bracket, Rear ASSY | Holds emitter main in place | Misalligned | Sudden disconnect via mechanical damage | Damage to emitter main board; Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0451 | MS-10136 | E1 Lower Internal ASSY | M10371 | FFC, 0.5mm pitch, 60 CKT, 70mm, OSC, Shielded | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0452 | MS-10136 | E1 Lower Internal ASSY | M10371 | FFC, 0.5mm pitch, 60 CKT, 70mm, OSC, Shielded | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0453 | MS-10136 | E1 Lower Internal ASSY | M10371 | FFC, 0.5mm pitch, 60 CKT, 70mm, OSC, Shielded | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0454 | MS-10136 | E1 Lower Internal ASSY | M10372 | FFC, 0.5mm pitch, 28 CKT, 103mm, Shielded | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0455 | MS-10136 | E1 Lower Internal ASSY | M10372 | FFC, 0.5mm pitch, 28 CKT, 103mm, Shielded | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0456 | MS-10136 | E1 Lower Internal ASSY | M10372 | FFC, 0.5mm pitch, 28 CKT, 103mm, Shielded | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0457 | MS-10136 | E1 Lower Internal ASSY | M10381 | Pico-Lock Cable Assy, 4 Circuit, 150mm, Twisted | Transmits power and signal for LED PCBs | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Minor fire | Minor burn | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0458 | MS-10136 | E1 Lower Internal ASSY | M10381 | Pico-Lock Cable Assy, 4 Circuit, 150mm, Twisted | Transmits power and signal for LED PCBs | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0459 | MS-10136 | E1 Lower Internal ASSY | M10381 | Pico-Lock Cable Assy, 4 Circuit, 150mm, Twisted | Transmits power and signal for LED PCBs | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0460 | MS-10136 | E1 Lower Internal ASSY | M10381 | Pico-Lock Cable Assy, 4 Circuit, 150mm, Twisted | Transmits power and signal for LED PCBs | Insulation worn by friction over time causing electrical short | Strain relief points become disconnected | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0461 | MS-10136 | E1 Lower Internal ASSY | M10381 | Pico-Lock Cable Assy, 4 Circuit, 150mm, Twisted | Transmits power and signal for LED PCBs | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0465 | MS-10136 | E1 Lower Internal ASSY | M50274 | WiFi Antenna, 250mm | Connects emitter to cassette | Becomes displaced | Sudden disconnect via mechanical damage | Inability to initiate trigger | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0466 | MS-10136 | E1 Lower Internal ASSY | M50274 | WiFi Antenna, 250mm | Connects emitter to cassette | Becomes displaced | Sudden disconnect via mechanical damage | Unable to view images on tablet | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0467 | MS-10136 | E1 Lower Internal ASSY | M50274 | WiFi Antenna, 250mm | Connects emitter to cassette | Becomes displaced | Sudden disconnect via mechanical damage | Inability to send Images outside of device | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0468 | MS-10136 | E1 Lower Internal ASSY | M10235 | Antenna Mounting Bracket | Holds antennaes in place | Misalligned | Sudden disconnect via mechanical damage | Loss of communication between components; Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0469 | MS-10136 | E1 Lower Internal ASSY | M50098 | Mountable Cable Tie | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0470 | MS-10136 | E1 Lower Internal ASSY | M50098 | Mountable Cable Tie | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0471 | MS-10136 | E1 Lower Internal ASSY | M10388 | Cable tie mount, 2 pos, 130 deg | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0472 | MS-10136 | E1 Lower Internal ASSY | M10388 | Cable tie mount, 2 pos, 130 deg | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0473 | MS-10136 | E1 Lower Internal ASSY | M10297 | Emitter Monoblock Foam Pad #1 | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0474 | MS-10136 | E1 Lower Internal ASSY | M10297 | Emitter Monoblock Foam Pad #1 | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0475 | MS-10136 | E1 Lower Internal ASSY | M10297 | Emitter Monoblock Foam Pad #1 | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0476 | MS-10136 | E1 Lower Internal ASSY | M10298 | Emitter Monoblock Foam Pad #2 | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0477 | MS-10136 | E1 Lower Internal ASSY | M10298 | Emitter Monoblock Foam Pad #2 | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0478 | MS-10136 | E1 Lower Internal ASSY | M10298 | Emitter Monoblock Foam Pad #2 | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0479 | MS-10136 | E1 Lower Internal ASSY | M10483 | Socket button head screw M3x0.5 x 4 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0480 | MS-10136 | E1 Lower Internal ASSY | M10491 | Socket button head screw M3x0.5 x 6 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0481 | MS-10136 | E1 Lower Internal ASSY | M10479 | Socket button head screw M2 x 0.4 x 4  Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0482 | MS-10136 | E1 Lower Internal ASSY | M10484 | Socket button head screw M2.5x0.5 x 8 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0483 | MS-10136 | E1 Lower Internal ASSY | M10490 | Socket button head screw M3x0.5 x 12 Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0484 | MS-10136 | E1 Lower Internal ASSY | M50291 | Thermal Pad, TG-A1250, 1mm thick, 20mm x 20mm Square | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0485 | MS-10136 | E1 Lower Internal ASSY | M10376 | Emitter Main Thermal Pad | Gap filler for heat transfer | Overheat | Insufficient thermal conductivity specifications | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0486 | MS-10136 | E1 Lower Internal ASSY | M50153 | Thermal Paste, TC3 | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0487 | MS-10136 | E1 Lower Internal ASSY | M50153 | Thermal Paste, TC3 | Thermal protection for the cassette | Overheat | Improper material choice | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0488 | MS-10136 | E1 Lower Internal ASSY | M50295 | Acrylic Adhesive Tape 3/8" | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0500 | MS-10136 | E1 Lower Internal ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0501 | MS-10136 | E1 Lower Internal ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0508 | MS-10136 | E1 Lower Internal ASSY | MS-10369 | Lit Cleat Cap ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging revceiving coil | Receiver coil misaligned | Improper geometry | Unable to charge wirelessly/slow charging speed | Delay of Procedure | 4.0 | 3.0 | 12 | Charging Verification Test | PRD5.11/5.12 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0509 | MS-10136 | E1 Lower Internal ASSY | MS-10369 | Lit Cleat Cap ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging revceiving coil | Receiver coil misaligned | Mechanical damage from external forces | Unable to charge wirelessly/slow charging speed | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0510 | MS-10136 | E1 Lower Internal ASSY | MS-10369 | Lit Cleat Cap ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging revceiving coil | Receiver coil inefficient | Excessive spacing from outer surface | Slow charging speed | Delay of Procedure | 4.0 | 3.0 | 12 | Charging Verification Test | PRD5.11/5.12 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0511 | MS-10136 | E1 Lower Internal ASSY | MS-10369 | Lit Cleat Cap ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging revceiving coil | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0512 | MS-10136 | E1 Lower Internal ASSY | MS-10369 | Lit Cleat Cap ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging revceiving coil | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0513 | MS-10136 | E1 Lower Internal ASSY | MS-10369 | Lit Cleat Cap ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging revceiving coil | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Compliance to ISO 10993 | PRD20.16 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0514 | MS-10136 | E1 Lower Internal ASSY | M10582 | RX Coil Bracket | Positions RX coil in Power Cleat ASSY | Mechanical damage from external forces | Improper geometry | Wireless Charge inoperable | Low Operator Inconvenience | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0515 | MS-10136 | E1 Lower Internal ASSY | M10583 | RX Coil Tape | Affix RX coil to Bracket | Rx Coil becomes loose | Poor adhesion | Wireless Charge inoperable | Low Operator Inconvenience | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0516 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | Assem that contains the Power Muxing PCB, coil bracket, and inductive coil | Frame structural integrity fails | Improper material choice - warps/bends | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0517 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | PCB failure | Individual component failure (open/shorts/etc) | Inability to charge device; damage device | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0518 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | PCB failure | Monoblock connector failure | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0519 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | PCB failure | Battery Connector failure | Inability to charge device | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0520 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | PCB failure | Emitter main connector failure | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0521 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | PCB failure | USB-C Connector failure | Inability to charge device | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0522 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | Battery Damage | Individual component failure (open/shorts/etc) | Battery failure - potential combustion | Major Fire | 7.0 | 2.0 | 14.0 | Battery 60601-1 Compliant | RSK_R125 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0523 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | Overheats | Shorting of a component along main power line | Potential for burnt internal parts; device inoperable | Major Fire | 7.0 | 2.0 | 14.0 | Battery 60601-1 Compliant | QSP-014 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0524 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | Overheats | Shorting of a component along main power line | Potential for burnt internal parts; device inoperable | Major Fire | 7.0 | 2.0 | 14.0 | Battery 60601-1 Compliant | RSK_R125 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0525 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | Overheats | Shorting of a component along main power line | Potential for burnt internal parts; device inoperable | Major Fire | 7.0 | 2.0 | 14.0 | Verification Testing | RSK_R125 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0526 | MS-10136 | E1 Lower Internal ASSY | ES-10015 | Emitter Power Input PCBA | PCB to manage input power from USBC, inductive coil, and output to rest of the system | Overheats | High charging current while device is off | Inability to charge device;damage device | Major Fire | 7.0 | 2.0 | 14.0 | Battery 60601-1 Compliant | RSK_R125 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK0527 | MS-10136 | E1 Lower Internal ASSY | M50034 | Phillips Rounded Head Thread Forming Screw: #1-1/4in | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin)Parts fall into sterile bag during surgery | Temporary Discomfort | 1.0 | 2.0 | 2.0 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0528 | MS-10136 | E1 Lower Internal ASSY | M50034 | Phillips Rounded Head Thread Forming Screw: #1-1/4in | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0529 | MS-10136 | E1 Lower Internal ASSY | M10785 | RX Coil Thermistor Retaining Bracket | Captures and applies contact force to thermistor | Thermistor out of place | Improper geometry | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0530 | MS-10136 | E1 Lower Internal ASSY | M10785 | RX Coil Thermistor Retaining Bracket | Captures and applies contact force to thermistor | Thermistor out of place | Improper material selection | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0531 | MS-10136 | E1 Lower Internal ASSY | M10487 | Socket button head screw M2x0.4 x 2 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Impact causes damage | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0532 | MS-10136 | E1 Lower Internal ASSY | M50153 | Thermal Paste, TC3 | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0533 | MS-10136 | E1 Lower Internal ASSY | M50153 | Thermal Paste, TC3 | Thermal protection for the cassette | Overheat | Improper material choice | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0534 | MS-10136 | E1 Lower Internal ASSY | M50002 | Phillips Rounded Head Thread Forming Screw: #1-1/8in | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0535 | MS-10136 | E1 Lower Internal ASSY | M50002 | Phillips Rounded Head Thread Forming Screw: #1-1/8in | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin)Parts fall into sterile bag during surgery | Temporary Discomfort | 1.0 | 2.0 | 2.0 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0536 | MS-10136 | E1 Lower Internal ASSY | M50002 | Phillips Rounded Head Thread Forming Screw: #1-1/8in | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0537 | MS-10136 | E1 Lower Internal ASSY | M50002 | Phillips Rounded Head Thread Forming Screw: #1-1/8in | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1718 | MS-10136 | E1 Lower Internal ASSY | M10908 | Emitter WIFI Coax Cowling | Retains antenna connection to module | Parts become dislodged | Mechanical damage from external forces | Antennas no longer have an additional level of retention | Delay of Procedure | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0538 | MS-10145 | UB 5035-15, Gasket Applied | M10054 | Heatsink, UB5035-15 Modified | Expels heat | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0539 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0540 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0541 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0542 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0543 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0544 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0545 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0546 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0547 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0548 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0549 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0550 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0551 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0552 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0553 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0554 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0555 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0556 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0557 | MS-10145 | UB 5035-15, Gasket Applied | M10143 | Gasket, UB 5035-15 | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0558 | MS-10144 | UB 60-15, Gasket Applied | M10053 | Heatsink, UB60-15 Modified | Expels heat | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0559 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0560 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0561 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0562 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0563 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0564 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0565 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0566 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0567 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0568 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0569 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0570 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0571 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0572 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0573 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0574 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0575 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0576 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0577 | MS-10144 | UB 60-15, Gasket Applied | M10142 | Gasket, UB 60-15 | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0578 | MS-10135 | Fan-Duct ASSY | MS-10097 | Fan Assembly | Expels heat from heat sink | Monoblock/Jetson Overheat | Fan speed too slow | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0579 | MS-10135 | Fan-Duct ASSY | MS-10097 | Fan Assembly | Expels heat from heat sink | Jetson/Monoblock Overheat | Fan stops | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0580 | MS-10135 | Fan-Duct ASSY | MS-10097 | Fan Assembly | Expels heat from heat sink | Fan Overheat | Debris blocks fan | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Covered fan | RSK_R159 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0581 | MS-10135 | Fan-Duct ASSY | MS-10097 | Fan Assembly | Expels heat from heat sink | Fan stops | Ingress of dust into enclosure | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Covered fan | RSK_R159 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0582 | MS-10135 | Fan-Duct ASSY | MS-10097 | Fan Assembly | Expels heat from heat sink | Biohazard | Contaminates get into device | Dirty interior | Infection | 1.0 | 2.0 | 2 | Covered fan | RSK_R159 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0583 | MS-10135 | Fan-Duct ASSY | M10249 | Cooling Duct, IM | Part of airduct for heatsinks; fan attatchment | Structural integrity compromized | Mechanical damage from external forces | Loss of means of protection; device still operable (limited) | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0584 | MS-10135 | Fan-Duct ASSY | M10249 | Cooling Duct, IM | Part of airduct for heatsinks; fan attatchment | Fan damage | Mechanical damage from external forces | Reduced cooling; device still operable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0585 | MS-10135 | Fan-Duct ASSY | M50127 | Phillips Head Thread Forming Screw M4, 20mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0586 | MS-10135 | Fan-Duct ASSY | M50127 | Phillips Head Thread Forming Screw M4, 20mm Long | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0587 | MS-10135 | Fan-Duct ASSY | M50127 | Phillips Head Thread Forming Screw M4, 20mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0588 | MS-10135 | Fan-Duct ASSY | M50127 | Phillips Head Thread Forming Screw M4, 20mm Long | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2066 | MS-10097 | Fan Assembly | M50048 | Conn-Housing-PL4POS-1.25mm | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Overheating | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2067 | MS-10097 | Fan Assembly | M50244 | Fan, 50mm Blower, PWM control, E speed, conformal coated, tach | Provide Cooling to E1 | Fails to Provide Cooling to E1 | Incorrect Part Specification | Overheating | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2068 | MS-10097 | Fan Assembly | M50319 | Contact, 1.25mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Overheating | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2069 | MS-10097 | Fan Assembly | M50424 | Heat Shrink, 3.10mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0589 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | Overheats | Shorting of a component along main power line | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0590 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | Overheats | Shorting of a component along main power line | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0591 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | Unregulated Power rail | Individual component failure | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0592 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | Fan Damage | Individual component failure | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0593 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | Unregulated Power rail | Individual component failure | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0594 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0595 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0596 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0597 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0598 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0599 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0600 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0601 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0602 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0603 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0604 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0605 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0606 | MS-10148 | Emitter Main ASSY | ES-10003 | Emitter Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, and WIFI modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0607 | MS-10148 | Emitter Main ASSY | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0608 | MS-10148 | Emitter Main ASSY | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0609 | MS-10148 | Emitter Main ASSY | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Unable to save images | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0610 | MS-10148 | Emitter Main ASSY | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Unable to take image | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0611 | MS-10148 | Emitter Main ASSY | M50817 | Multiprotocol Modules Intel Wireless-AC 9260, 2230, 2x2 AC+BT, Gigabit, No vPro | Provides Connectivity toTablet and Cassette | No Wifi | Mechanical damage | Device unable to communicate with PACS server | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0612 | MS-10148 | Emitter Main ASSY | M10483 | Socket button head screw M3x0.5 x 4 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0614 | MS-10405 | Jetson with Heat pipe Assembly | MS-10146 | Xavier NX Heat Pipe ASSY | Draw heat from one location to another | Jetson/Monoblock Overheat | Improper pipe routing | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0615 | MS-10405 | Jetson with Heat pipe Assembly | MS-10146 | Xavier NX Heat Pipe ASSY | Draw heat from one location to another | Jetson/Monoblock Overheat | Mechanical damage from external forces | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0616 | MS-10405 | Jetson with Heat pipe Assembly | MS-10146 | Xavier NX Heat Pipe ASSY | Draw heat from one location to another | Jetson/Monoblock Overheat | Insufficient heat transfer capacity | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0617 | MS-10405 | Jetson with Heat pipe Assembly | M50101 | JETSON XAVIER NX Module | Runs device and ensures proper functioning | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0618 | MS-10405 | Jetson with Heat pipe Assembly | M50101 | JETSON XAVIER NX Module | Runs device and ensures proper functioning | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0619 | MS-10405 | Jetson with Heat pipe Assembly | M10494 | Xavier NX screw, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0620 | MS-10405 | Jetson with Heat pipe Assembly | M50082 | Jetson Xavier NX Leaf spring | Runs device and ensures proper functioning | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0621 | MS-10405 | Jetson with Heat pipe Assembly | M50082 | Jetson Xavier NX Leaf spring | Provides impact resistance to Jetson | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0622 | MS-10405 | Jetson with Heat pipe Assembly | M50153 | Thermal Paste, TC3 | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0623 | MS-10405 | Jetson with Heat pipe Assembly | M50153 | Thermal Paste, TC3 | Thermal protection for the cassette | Overheat | Improper material choice | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2070 | MS-10146 | Xavier NX Heat Pipe ASSY | M10058 | Xavier NX Cold Side Contact Block | Connects heat pipes to heatsinks for heat transfer | Failure to transfer heat | Contact prevented via mechanical damage | Jetson overheats, faults | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2071 | MS-10146 | Xavier NX Heat Pipe ASSY | M10055 | Xavier NX Heat Pipe | Facilitates transfer of heat from monoblock to heatsink | Failure to transfer heat | Mechanical damage from outside forces | Jetson overheats, faults | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2072 | MS-10146 | Xavier NX Heat Pipe ASSY | M10057 | Xavier NX Hot Side Contact Block | Connects Jetson to heat pipes for heat transfer | Failure to transfer heat | Contact prevented via mechanical damage | Jetson overheats, faults | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0624 | MS-10155 | E1 X-ray ASSY | MS-10007 | Monoblock | Assem that contains an X-ray tube and power supply in a potted enclosure | Fails to reduce EMI from device | Improper specifications | Interference with other electronics; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1719 | MS-10155 | E1 X-ray ASSY | MS-10007 | Monoblock | Assem that contains an X-ray tube and power supply in a potted enclosure | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK0625 | MS-10155 | E1 X-ray ASSY | M10019 | Monoblock FR Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Tube insulation degrades | Long term repeated use | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0626 | MS-10155 | E1 X-ray ASSY | M10019 | Monoblock FR Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0627 | MS-10155 | E1 X-ray ASSY | M10019 | Monoblock FR Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0628 | MS-10155 | E1 X-ray ASSY | M10019 | Monoblock FR Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0629 | MS-10155 | E1 X-ray ASSY | M10052 | Monoblock FL Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0630 | MS-10155 | E1 X-ray ASSY | M10052 | Monoblock FL Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0631 | MS-10155 | E1 X-ray ASSY | M10052 | Monoblock FL Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0632 | MS-10155 | E1 X-ray ASSY | M10067 | LVPS Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0633 | MS-10155 | E1 X-ray ASSY | M10067 | LVPS Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0634 | MS-10155 | E1 X-ray ASSY | M10067 | LVPS Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0635 | MS-10155 | E1 X-ray ASSY | M10068 | LVB Right Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0636 | MS-10155 | E1 X-ray ASSY | M10068 | LVB Right Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0637 | MS-10155 | E1 X-ray ASSY | M10068 | LVB Right Bracket | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0638 | MS-10155 | E1 X-ray ASSY | M10348 | Monoblock Mount, Overmolded | Assem that contains an X-ray tube and power supply in a potted enclosure | Potting compound breaks | Mechanical damage from external forces | Arcs w/ Failure of potting compound; device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0639 | MS-10155 | E1 X-ray ASSY | M10348 | Monoblock Mount, Overmolded | Assem that contains an X-ray tube and power supply in a potted enclosure | Misalignment | Improper geometry | Cameras and monoblock no longer aligned; device still operable | Negligible Radiation Tissue | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0640 | MS-10155 | E1 X-ray ASSY | M10348 | Monoblock Mount, Overmolded | Assem that contains an X-ray tube and power supply in a potted enclosure | Overheat | Dielectric Failure | Monoblock damage; Arcs; Device inoperable | If operator/patient touches power cable/charger while operating device; Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0641 | MS-10155 | E1 X-ray ASSY | M10331 | LVPS THERMAL PAD | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0642 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure | Loss of safety monitoring | Delay of Procedure | 4.0 | 3.0 | 12 | Heartbeat Monitoring | RSK_R012 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0643 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | Overheats | Individual component failure (open/shorts/etc) | Potential for burnt internal parts; device inoperable | Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0644 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Produces excess emissions | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0645 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Produces excess emissions | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0646 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0647 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0648 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0649 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0650 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | Switching failure | FW failure | Incorrect reading | Delay of Procedure | 4.0 | 3.0 | 12 | FW has Heartbeat | RSK_R012 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0651 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1ISTA Test | PRD20.5PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0652 | MS-10155 | E1 X-ray ASSY | ES-10019 | Monoblock LV PCBA | PCB that controls the inputs to the monoblock and provides safety monitoring | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0653 | MS-10155 | E1 X-ray ASSY | ES-10024 | Monoblock LV PWS Rider | PCB that supplies power to the Monoblock LV PCB | Overheats | Individual component failure (open/shorts/etc) | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0654 | MS-10155 | E1 X-ray ASSY | M50098 | Mountable Cable Tie | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0655 | MS-10155 | E1 X-ray ASSY | M50098 | Mountable Cable Tie | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0656 | MS-10155 | E1 X-ray ASSY | M10486 | Hex socket countersunk head screw M3x0.5 x 5 Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0657 | MS-10155 | E1 X-ray ASSY | M10483 | Socket button head screw M3x0.5 x 4 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0658 | MS-10155 | E1 X-ray ASSY | M10489 | Socket button head screw M3x0.5 x 8 Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0659 | MS-10155 | E1 X-ray ASSY | M10490 | Socket button head screw M3x0.5 x 12 Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0660 | MS-10155 | E1 X-ray ASSY | M10375 | LV Thermal Pad | Gap filler for heat transfer | Overheat | Insufficient thermal conductivity specifications | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0661 | MS-10155 | E1 X-ray ASSY | M10375 | LV Thermal Pad | Gap filler for heat transfer | Overheat | Improper geometry | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0669 | MS-10155 | E1 X-ray ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0670 | MS-10155 | E1 X-ray ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0677 | MS-10008 | Collimator | M50167 | P01_drivebelts | Transfer collimator motor power to pulleys | Belt Slip | Excessive belt length | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0678 | MS-10008 | Collimator | M50167 | P01_drivebelts | Transfer collimator motor power to pulleys | Belt Slip | Insufficient belt tension specified | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0679 | MS-10008 | Collimator | M50167 | P01_drivebelts | Transfer collimator motor power to pulleys | Belt Slip | Wear | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0680 | MS-10008 | Collimator | M50167 | P01_drivebelts | Transfer collimator motor power to pulleys | Belt failure | Excessive belt tension | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0681 | MS-10008 | Collimator | M50167 | P01_drivebelts | Transfer collimator motor power to pulleys | Belt failure | Pulley misalignment | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0682 | MS-10008 | Collimator | M10479 | Socket button head screw M2 x 0.4 x 4  Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0683 | MS-10008 | Collimator | M10264 | P01_encoderdisk | Optical pattern to sense collimator movement | Fails to adhere | Improper adhesive selection | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0684 | MS-10008 | Collimator | M10029 | Cap, Chassis | Holds drive mechanics, motors, and collimator pcb | Structural integrity compromized | Mechanical damage from external forces | Loose components cause shorts, Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0685 | MS-10008 | Collimator | M10029 | Cap, Chassis | Holds drive mechanics, motors, and collimator pcb | Fastener failure | Mechanical damage from external forces - Vibration | Loose components cause shorts, Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0686 | MS-10008 | Collimator | M10035 | Bearing, Short | Bearing interface for moving parts | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0687 | MS-10008 | Collimator | M10035 | Bearing, Short | Bearing interface for moving parts | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0688 | MS-10008 | Collimator | M10035 | Bearing, Short | Bearing interface for moving parts | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0689 | MS-10008 | Collimator | M10035 | Bearing, Short | Bearing interface for moving parts | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0690 | MS-10008 | Collimator | M10036 | Bearing, Long | Bearing interface for moving parts | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0691 | MS-10008 | Collimator | M10036 | Bearing, Long | Bearing interface for moving parts | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0692 | MS-10008 | Collimator | M10036 | Bearing, Long | Bearing interface for moving parts | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0693 | MS-10008 | Collimator | M10036 | Bearing, Long | Bearing interface for moving parts | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0694 | MS-10008 | Collimator | M10034 | Bearing, Main | Bearing interface for moving parts | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0695 | MS-10008 | Collimator | M10034 | Bearing, Main | Bearing interface for moving parts | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0696 | MS-10008 | Collimator | M10034 | Bearing, Main | Bearing interface for moving parts | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0697 | MS-10008 | Collimator | M10034 | Bearing, Main | Bearing interface for moving parts | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0698 | MS-10008 | Collimator | M10339 | Chassis, Collimator | Holds drive mechanics, motors, and collimator pcb | Structural integrity compromized | Mechanical damage from external forces | Loose components cause shorts, Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0699 | MS-10008 | Collimator | M10339 | Chassis, Collimator | Holds drive mechanics, motors, and collimator pcb | Fastener failure | Mechanical damage from external forces - Vibration | Loose components cause shorts, Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0700 | MS-10008 | Collimator | M10342 | DISTAL DISK M10025 WITH COATING | Bearing interface for moving parts | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0701 | MS-10008 | Collimator | M10342 | DISTAL DISK M10025 WITH COATING | Bearing interface for moving parts | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0702 | MS-10008 | Collimator | M10342 | DISTAL DISK M10025 WITH COATING | Bearing interface for moving parts | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0703 | MS-10008 | Collimator | M10342 | DISTAL DISK M10025 WITH COATING | Bearing interface for moving parts | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0704 | MS-10008 | Collimator | M10341 | INTERLEAF DISK M10024 WITH COATING | Bearing interface for moving parts | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0705 | MS-10008 | Collimator | M10341 | INTERLEAF DISK M10024 WITH COATING | Bearing interface for moving parts | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0706 | MS-10008 | Collimator | M10341 | INTERLEAF DISK M10024 WITH COATING | Bearing interface for moving parts | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0707 | MS-10008 | Collimator | M10341 | INTERLEAF DISK M10024 WITH COATING | Bearing interface for moving parts | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0708 | MS-10008 | Collimator | M10023 | Leaf, Collimator | Restricts the x ray field to the desired size | Unintentional gaps in leaves | Leaf material too thick | Loss of radiation protection, device still operable | Negligible Radiation Stochastic Harm | 1.0 | 3.0 | 3 | Compliance to IEC 60601-2-54 | PRD20.3 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0709 | MS-10008 | Collimator | M10023 | Leaf, Collimator | Restricts the x ray field to the desired size | Allows excess radiation | Leaf material too thin | Loss of radiation protection, device still operable | Negligible Radiation Stochastic Harm | 1.0 | 3.0 | 3 | Compliance to IEC 60601-2-54 | PRD20.3 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0710 | MS-10008 | Collimator | M10340 | 1st DISK M10022 WITH COATING | Bearing interface for moving parts | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0711 | MS-10008 | Collimator | M10340 | 1st DISK M10022 WITH COATING | Bearing interface for moving parts | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0712 | MS-10008 | Collimator | M10340 | 1st DISK M10022 WITH COATING | Bearing interface for moving parts | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0713 | MS-10008 | Collimator | M10340 | 1st DISK M10022 WITH COATING | Bearing interface for moving parts | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0714 | MS-10008 | Collimator | M10021 | Wheel, Pin | Output pulley collimator leaves pivot on | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0715 | MS-10008 | Collimator | M10021 | Wheel, Pin | Output pulley collimator leaves pivot on | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0716 | MS-10008 | Collimator | M10021 | Wheel, Pin | Output pulley collimator leaves pivot on | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0717 | MS-10008 | Collimator | M10021 | Wheel, Pin | Output pulley collimator leaves pivot on | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0718 | MS-10008 | Collimator | M10495 | Sheet Metal Motor Cable Bracket | Strain relief for Motor Flex Cable | Cable Damaged | Mechanical Damage from External Forces | Collimator Motor Inoperative; beam is under-collimated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1464 | MS-10008 | Collimator | M10495 | Sheet Metal Motor Cable Bracket | Strain relief for Motor Flex Cable | Cable Damaged | Mechanical Damage from External Forces | Collimator Motor Inoperative; beam is over-collimated and image needs to be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 |  |
| DRSK0719 | MS-10008 | Collimator | M10495 | Sheet Metal Motor Cable Bracket | Strain relief for Motor Flex Cable | Braket Breaks | Mechanical Damage from External Forces | Collimator Motor Inoperative; beam is under-collimated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1465 | MS-10008 | Collimator | M10495 | Sheet Metal Motor Cable Bracket | Strain relief for Motor Flex Cable | Braket Breaks | Mechanical Damage from External Forces | Collimator Motor Inoperative; beam is over-collimated and image needs to be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 |  |
| DRSK0720 | MS-10008 | Collimator | MS-10026 | Wheel, Cam, ASSY | Applies cam movement to move leaves | Excessive wear | Improper geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0721 | MS-10008 | Collimator | MS-10026 | Wheel, Cam, ASSY | Applies cam movement to move leaves | Excessive wear | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0722 | MS-10008 | Collimator | MS-10026 | Wheel, Cam, ASSY | Applies cam movement to move leaves | Seized pulleys | Insufficient clearances | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0723 | MS-10008 | Collimator | MS-10026 | Wheel, Cam, ASSY | Applies cam movement to move leaves | Collimator position inaccurate | Excessive clearances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0724 | MS-10008 | Collimator | MS-10030 | Drive ASSY | Motopr and input pulley | No collimation movement | Loose Drive Pulley set screws | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0725 | MS-10008 | Collimator | M50473 | Hex Button Head Screw M3 0.5 x 5mm | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0726 | MS-10008 | Collimator | M50473 | Hex Button Head Screw M3 0.5 x 5mm | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0727 | MS-10008 | Collimator | M50473 | Hex Button Head Screw M3 0.5 x 5mm | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0728 | MS-10008 | Collimator | M50473 | Hex Button Head Screw M3 0.5 x 5mm | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0729 | MS-10008 | Collimator | ES-10020 | Collimator Sensor PCBA | Range detection sensor | Communication loss | Poor board to board connection | System display error; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0730 | MS-10008 | Collimator | ES-10020 | Collimator Sensor PCBA | Range detection sensor | Communication loss | Improper drive signal | System display error; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0731 | MS-10008 | Collimator | ES-10020 | Collimator Sensor PCBA | Range detection sensor | Communication bus | No proper pullup | Erratic communication; System display error; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0732 | MS-10008 | Collimator | ES-10020 | Collimator Sensor PCBA | Range detection sensor | Communication bus | Poor board to board connection | Erratic communication; System display error; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0733 | MS-10008 | Collimator | ES-10008 | Collimator PCBA | Controls the inputs to the motors to achieve the desired collimation output | PCB failure | Individual component failure | Collimator jams; beam is under-collimated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1466 | MS-10008 | Collimator | ES-10008 | Collimator PCBA | Controls the inputs to the motors to achieve the desired collimation output | PCB failure | Individual component failure | Collimator jams; beam is over-collimated and image needs to be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0734 | MS-10008 | Collimator | ES-10008 | Collimator PCBA | Controls the inputs to the motors to achieve the desired collimation output | PCB failure | Firmware failure | Comunication loss, device not operable | Delay of Procedure | 4.0 | 3.0 | 12 | Heartbeat Monitoring | RSK_R012 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0735 | MS-10008 | Collimator | ES-10008 | Collimator PCBA | Controls the inputs to the motors to achieve the desired collimation output | FW miscommunication with Jetson | Firmware failure | Incorrect reporting, device not operable | Delay of Procedure | 4.0 | 3.0 | 12 | Heartbeat Monitoring | RSK_R012 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0736 | MS-10008 | Collimator | ES-10008 | Collimator PCBA | Controls the inputs to the motors to achieve the desired collimation output | Encoder failure | Mechanical damage from external forces | Incorrect reporting, device not operable | Delay of Procedure | 4.0 | 3.0 | 12 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0737 | MS-10008 | Collimator | M50040 | FFC, 0.5mm pitch, 8 CKT, 50mm | Transfer signal between Collimator Main PCB and Collimator sensor PCB | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Minor fire | Minor burn | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0738 | MS-10008 | Collimator | M50295 | Acrylic Adhesive Tape 3/8" | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1720 | MS-10008 | Collimator | M50289 | Anaerobic adhesive, Loctite 242 | Connects components together | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2073 | MS-10026 | Wheel, Cam, ASSY | M50170 | COTS_Dowel m2x8 McM91585A214 | Acts as drive path between Wheel and tungsten leaf | Does not drive leaf | Mechanical Damage | Collimator becomes in operable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2074 | MS-10026 | Wheel, Cam, ASSY | M10027 | Wheel, Cam | Drives leaves, is driven by belt | Does not drive leaf | Mechanical Damage | Collimator becomes in operable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0739 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Reaction time of aperture size/rotation change is too slow. | Motor/encoder/electronics can not react fast enough to satisfy operator | No product effect | Moderate Dissatisfaction | 1.0 | 5.0 | 5 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 5.0 | 5 |
| DRSK0740 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator aperture opening not accurate over required range | Misalignment of collimator to source | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 4.0 | 4 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0741 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator aperture opening not accurate over required range | Collimater clearances/tolerances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 4.0 | 4 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0742 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator aperture opening not accurate over required range | Encoder step size insufficient | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 4.0 | 4 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0744 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator Jam | Operation outside of specified temperature range | Collimator fails to collimate beam; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0745 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Motor failure | Operation outside of specified temperature range | Collimator fails to collimate beam; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0746 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Motor failure | Operation outside of specified temperature range | Slow collimation; device still operable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0747 | MS-10200 | Collimator - Line Laser ASSY | MS-10008 | P01 Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Homing failure | Homing LED/sensor failure | Collimator fails to collimate beam; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0748 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0749 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | ISTA Testing | PRD20.26 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0750 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | Overcurrent | Operator Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0751 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | May disable laser guidance but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0752 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Control signal open | Loose connection to connector | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Locking connectors | RSK_R196 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0753 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Improper specifications/geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0754 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0755 | MS-10200 | Collimator - Line Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Misalligned (stackup error) | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0756 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0757 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | ISTA Testing | PRD20.26 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0758 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | Overcurrent | Operator Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0759 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | May disable laser guidance but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0760 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Control signal open | Loose connection to connector | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Locking connectors | RSK_R196 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0761 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Improper specifications/geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0762 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Incorrect material choice | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0763 | MS-10200 | Collimator - Line Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Misalligned (stackup error) | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0764 | MS-10200 | Collimator - Line Laser ASSY | M50474 | Zinc-Plated Steel, M2, 4 mm Long, T6 Torx Drive | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0765 | MS-10200 | Collimator - Line Laser ASSY | M50474 | Zinc-Plated Steel, M2, 4 mm Long, T6 Torx Drive | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0766 | MS-10200 | Collimator - Line Laser ASSY | M50474 | Zinc-Plated Steel, M2, 4 mm Long, T6 Torx Drive | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0767 | MS-10200 | Collimator - Line Laser ASSY | M50474 | Zinc-Plated Steel, M2, 4 mm Long, T6 Torx Drive | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0768 | MS-10200 | Collimator - Line Laser ASSY | M10479 | Socket button head screw M2 x 0.4 x 4  Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0769 | MS-10149 | Collimator Bracket-Camera ASSY | M10076 | Static Collimator | Restricts x-ray field | x-ray field too small | Incorrect geometry | X-ray exposure area smaller than expected; image must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0770 | MS-10149 | Collimator Bracket-Camera ASSY | M10076 | Static Collimator | Restricts x-ray field | x-ray field too large | Incorrect geometry | Greater x-ray exposure; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0771 | MS-10149 | Collimator Bracket-Camera ASSY | M10076 | Static Collimator | Restricts x-ray field | Insufficient attenuation | Incorrect material | Greater x-ray exposure; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0772 | MS-10149 | Collimator Bracket-Camera ASSY | M10076 | Static Collimator | Restricts x-ray field | Insufficient attenuation | Incorrect thickness | Greater x-ray exposure; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0773 | MS-10149 | Collimator Bracket-Camera ASSY | M10073 | Collimator-Optics Bracket | Holds sensor module and collimator onto monoblock | Misalligned | Mechanical damage from external forces | Tracking inaccurate, beam extends past active area | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0774 | MS-10149 | Collimator Bracket-Camera ASSY | ES-10006 | Sensor PCBA | PCB that carries all the sensors and connections | PCB failure | Individual component failure (open/shorts/etc) | Sensor Failure | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | IFU - Verify device functionality | 4.0 | 3.0 | 12 |
| DRSK0775 | MS-10149 | Collimator Bracket-Camera ASSY | ES-10006 | Sensor PCBA | PCB that carries all the sensors and connections | PCB failure | Individual component failure (open/shorts/etc) | Sensor Failure | Instrument failure, no results - surgery | 7.0 | 2.0 | 14 | Compliance to IEC 60601-1 | PRD20.5 | IFU - Verify device functionality | 7.0 | 2.0 | 14 |
| DRSK0776 | MS-10149 | Collimator Bracket-Camera ASSY | ES-10006 | Sensor PCBA | PCB that carries all the sensors and connections | Unregulated Power rail | Individual component failure (open/shorts/etc) | Damaged camera module; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0777 | MS-10149 | Collimator Bracket-Camera ASSY | M50010 | Framos Sensor, FSM-IMX577C- 01S-V1B | Allows user to align active area and x-ray field | Viewfinder failure | Individual component failure (open/shorts/etc) | Viewfinder not available | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0778 | MS-10149 | Collimator Bracket-Camera ASSY | M50010 | Framos Sensor, FSM-IMX577C- 01S-V1B | Takes photographic images | Imaging module failure | Individual component failure (open/shorts/etc) | Inability to take photo images | Low Operator Inconvenience | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0779 | MS-10149 | Collimator Bracket-Camera ASSY | M50008 | Framos Sensor, FSM-IMX335M- 02O-V1A | Allows user to align active area and x-ray field | Viewfinder failure | Individual component failure (open/shorts/etc) | Viewfinder not available | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0780 | MS-10149 | Collimator Bracket-Camera ASSY | M50008 | Framos Sensor, FSM-IMX335M- 02O-V1A | Uses IR to determine distances | Misalligned | Mechanical damage from external forces | Tracking inaccurate, beam extends past active area | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0781 | MS-10149 | Collimator Bracket-Camera ASSY | M50008 | Framos Sensor, FSM-IMX335M- 02O-V1A | Uses IR to determine distances | IR module fails | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0782 | MS-10149 | Collimator Bracket-Camera ASSY | M50009 | Framos Sensor, FSM-IMX715C- 01S-V1A | Allows user to align active area and x-ray field | Viewfinder failure | Individual component failure (open/shorts/etc) | Viewfinder not available | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0783 | MS-10149 | Collimator Bracket-Camera ASSY | M50009 | Framos Sensor, FSM-IMX715C- 01S-V1A | Uses IR to determine distances | Misalligned | Mechanical damage from external forces | Tracking inaccurate, beam extends past active area | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0784 | MS-10149 | Collimator Bracket-Camera ASSY | M50009 | Framos Sensor, FSM-IMX715C- 01S-V1A | Uses IR to determine distances | IR module fails | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0785 | MS-10149 | Collimator Bracket-Camera ASSY | M10483 | Socket button head screw M3x0.5 x 4 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0786 | MS-10149 | Collimator Bracket-Camera ASSY | M10479 | Socket button head screw M2 x 0.4 x 4  Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0787 | MS-10149 | Collimator Bracket-Camera ASSY | M10330 | Static Collimator Adhesive | Restricts x-ray field | x-ray field too small | Incorrect geometry | X-ray exposure area smaller than expected; image must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0788 | MS-10149 | Collimator Bracket-Camera ASSY | M10330 | Static Collimator Adhesive | Restricts x-ray field | x-ray field too large | Incorrect geometry | Greater x-ray exposure; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0789 | MS-10149 | Collimator Bracket-Camera ASSY | M10330 | Static Collimator Adhesive | Restricts x-ray field | Insufficient attenuation | Incorrect material | Greater x-ray exposure; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0790 | MS-10149 | Collimator Bracket-Camera ASSY | M10330 | Static Collimator Adhesive | Restricts x-ray field | Insufficient attenuation | Incorrect thickness | Greater x-ray exposure; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | IEC 60601-3 Testing (Intertek) | PRD20.7 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0791 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | MS-10147 | Monoblock Heat Pipe ASSY | Thermal system that will pull heat from the CPU and monoblock and vent external to the enclosure | Collimator position inaccurate | Excessive clearances | Misalgnment of X-ray beam to detector. | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0792 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | MS-10100 | Monoblock Thermistor Assembly | Measures monoblock temperature | Thermistor failure | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0793 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50035 | Shoulder Screw, SS-13.2 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0794 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50035 | Shoulder Screw, SS-13.2 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0795 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50035 | Shoulder Screw, SS-13.2 | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0796 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50035 | Shoulder Screw, SS-13.2 | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0797 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50036 | E-ring, 2.3mm groove | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0798 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50036 | E-ring, 2.3mm groove | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0799 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50036 | E-ring, 2.3mm groove | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0800 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50036 | E-ring, 2.3mm groove | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0801 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50124 | Spring, S001YJ0D | Applied pressure to between monoblock and heat pipe assembly | Monoblock Overheat | Insufficient contact pressure | Jetson throttle - device inoperable (restart required) | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0802 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50153 | Thermal Paste, TC3 | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0803 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50153 | Thermal Paste, TC3 | Thermal protection for the cassette | Overheat | Improper material choice | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0804 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M10232 | Monoblock Thermistor Retaining Bracket | Holds thermistor in place | Thermistor out of place | Improper geometry | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0805 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M10232 | Monoblock Thermistor Retaining Bracket | Holds thermistor in place | Thermistor out of place | Improper material selection | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0806 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M10487 | Socket button head screw M2x0.4 x 2 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0814 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0815 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2075 | MS-10147 | Monoblock Heat Pipe ASSY | M10060 | Monoblock Cold Side Contact Block | Connects heat pipes to heatsinks for heat transfer | Failure to transfer heat | Contact prevented via mechanical damage | Monoblock overheats, faults | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2076 | MS-10147 | Monoblock Heat Pipe ASSY | M10059 | Monoblock Hot Side Contact Block | Connects monoblock to heat pipes for heat transfer | Failure to transfer heat | Contact prevented via mechanical damage | Monoblock overheats, faults | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2077 | MS-10147 | Monoblock Heat Pipe ASSY | M10056 | Monoblock Heat Pipe | Facilitates transfer of heat from monoblock to heatsink | Failure to transfer heat | Mechanical damage from outside forces | Monoblock overheats, faults | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2078 | MS-10100 | Monoblock Thermistor Assembly | M50072 | Conn Plug, 1mm Pico-Lock, 2 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Improper temperature read out | Monoblock faults, Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2079 | MS-10100 | Monoblock Thermistor Assembly | M50047 | Contact, 1mm, Pico-Lock | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Improper temperature read out | Monoblock faults, Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2080 | MS-10100 | Monoblock Thermistor Assembly | M50119 | Thermistor 10Kohm | Reads temperature of heat pipe assembly | Head cracks | Mechanical Damage | Improper temperature read out | Monoblock faults, Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0838 | MS-10222 | RH Laser ASSY | MS-10093 | E1 RH Laser Harness | Transfer low voltage power to Laser | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Low Operator Inconvenience | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1721 | MS-10222 | RH Laser ASSY | M10924 | Laser Mount Core | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1722 | MS-10222 | RH Laser ASSY | M10924 | Laser Mount Core | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1723 | MS-10222 | RH Laser ASSY | M10926 | Laser Mount R Base | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1724 | MS-10222 | RH Laser ASSY | M10926 | Laser Mount R Base | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0850 | MS-10222 | RH Laser ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0851 | MS-10222 | RH Laser ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1725 | MS-10222 | RH Laser ASSY | M10924 | Laser Mount Core | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1726 | MS-10222 | RH Laser ASSY | M10924 | Laser Mount Core | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1727 | MS-10222 | RH Laser ASSY | M10925 | Laser Mount L Base | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Incoming Inspection | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1728 | MS-10222 | RH Laser ASSY | M10925 | Laser Mount L Base | Holds line laser to indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2081 | MS-10093 | E1 RH Laser Harness | M10066 | RH Laser | Generates line laser | Does not generate laser | Mechanical Damage | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Low Operator Inconvenience | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2082 | MS-10093 | E1 RH Laser Harness | M50114 | Cable-ASSY, Pico-Lock, 2 Circuit, 100mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2083 | MS-10093 | E1 RH Laser Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2084 | MS-10093 | E1 RH Laser Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2085 | MS-10093 | E1 RH Laser Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0874 | MS-10221 | LH Laser ASSY | MS-10094 | E1 LH Laser Harness | Transfer low voltage power to Laser | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Low Operator Inconvenience | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0886 | MS-10221 | LH Laser ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0887 | MS-10221 | LH Laser ASSY | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2086 | MS-10094 | E1 LH Laser Harness | M10065 | LH Laser | Generates line laser | Does not generate laser | Mechanical Damage | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Low Operator Inconvenience | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2087 | MS-10094 | E1 LH Laser Harness | M50114 | Cable-ASSY, Pico-Lock, 2 Circuit, 100mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2088 | MS-10094 | E1 LH Laser Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2089 | MS-10094 | E1 LH Laser Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2090 | MS-10094 | E1 LH Laser Harness | M50238 | Solder, RoHS Compliant, No clean | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Laser inoperable | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0894 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Motor not responsive | Connector and/or drive electronics fail over extended life cycle | Collimator stops working | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0895 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Motor not responsive | Insufficient Torque | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0896 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Motor not responsive | Output Shaft Bearing Failure | Wheel displaced; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0897 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Overheat | Failure of thermal cutout circuit | Potential for burnt internal parts; device life impacted; device still operable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0898 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Overheat | Overheats due to high housing temperatures | Potential for burnt internal parts; device life impacted; device still operable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0899 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Overheat | Motor Lock (motor icomponents nternally fail) | Potential for burnt internal parts; device inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0900 | MS-10030 | Drive ASSY | M50005 | Brushless DC Motor | Moves leaves in collimator | Motor not repsonsive | Use outside of recommended humidity range | Collimator stops working | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0901 | MS-10030 | Drive ASSY | M10033 | Mount, Motor, DF32 | Motopr and input pulley | No collimation movement | Loose Drive Pulley set screws | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0902 | MS-10030 | Drive ASSY | MS-10031 | Pulley, DriveR, ASSY | Motopr and input pulley | No collimation movement | Loose Drive Pulley set screws | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0903 | MS-10030 | Drive ASSY | M10486 | Hex socket countersunk head screw M3x0.5 x 5 Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1729 | MS-10030 | Drive ASSY | M50897 | Epoxy - 3M DP420, Black | Retention method of pulley to motor shaft | Fails to hold components together | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2091 | MS-10031 | Pulley, DriveR, ASSY | M50168 | COTS_SetScrew M2X0.4-3 McM91390A090 | Retention method of pulley to motor shaft | Fails to hold components together | Set screw backs out | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2092 | MS-10031 | Pulley, DriveR, ASSY | M50169 | COTS_Flange SDP_sdpsi_a_6a50p008fa | Retention method of belt to pulley | Fails to hold belt in place | Flange become unseated | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2093 | MS-10031 | Pulley, DriveR, ASSY | M10032 | Pulley, DriveR | Engages with belt and motor to drive collimator | Fails to drive belt | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK0905 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0906 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0907 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0908 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0909 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0910 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0911 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0912 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0913 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0914 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0915 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0916 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0917 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0918 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0919 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0920 | C1 | Cassette Main Assy | M10178 | GTIN LABEL | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0921 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0922 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0923 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0924 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0925 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0926 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0927 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0928 | C1 | Cassette Main Assy | M10162 | SAFETY RA LABEL | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0929 | C1 | Cassette Main Assy | M11081 | Battery Cover | Covers Battery pack | Falls off from impact | Mechanical damage from external forces | Battery falls out; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0930 | C1 | Cassette Main Assy | M11081 | Battery Cover | Covers Battery pack | Falls off from impact | Mechanical damage from external forces | Allows access to internals; device still operable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0931 | C1 | Cassette Main Assy | M10492 | Torx Flat Head Screws, M3 x 0.50 x 8 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0932 | C1 | Cassette Main Assy | MS-10083 | Cassette Battery Pack | Battery pack with integrated battery management system (BMS) | Becomes dislodged from enclosure | Mechanical damage from external forces | Inability to charge device | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0933 | C1 | Cassette Main Assy | M10407 | Cassette Battery Mounting Foam | Secures and Protects Battery from Internal Movement | Adhesive Failure | Aging | Battery connections fail and short | Battery Hard Short - Fire Risk | 4.0 | 3.0 | 12 | Battery Protection on BMS | PRD5.2 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0934 | C1 | Cassette Main Assy | MS-11139 | Cassette Handle Assembly | Allows operator to hold and carry the cassette | Handle breaks | Aging | Device falls and is damaged, device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0935 | C1 | Cassette Main Assy | M50213 | Acrylic Adhesive Tape 1" | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0936 | C1 | Cassette Main Assy | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | Contrains grand majority of hardware, ensure detector is not exposed and is aligned properly | Fails to contain hardware or keep parts aligned | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2094 | MS-10083 | Cassette Battery Pack | MS-10312 | Cassette Battery Pack - Harness | Allows Power Transfer from Battery Pack to C1 | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK2095 | MS-10083 | Cassette Battery Pack | MS-10312 | Cassette Battery Pack - Harness | Allows Power Transfer from Battery Pack to C1 | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK2096 | MS-10083 | Cassette Battery Pack | MS-10312 | Cassette Battery Pack - Harness | Allows Power Transfer from Battery Pack to C1 | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2097 | MS-10083 | Cassette Battery Pack | MS-10312 | Cassette Battery Pack - Harness | Allows Power Transfer from Battery Pack to C1 | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2098 | MS-10083 | Cassette Battery Pack | MS-10312 | Cassette Battery Pack - Harness | Allows Power Transfer from Battery Pack to C1 | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2099 | MS-10083 | Cassette Battery Pack | MS-10312 | Cassette Battery Pack - Harness | Allows Power Transfer from Battery Pack to C1 | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2100 | MS-10083 | Cassette Battery Pack | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Excessive Wear on Cables | Improper Geometry | Product inoperable | Minor Fire | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK2101 | MS-10083 | Cassette Battery Pack | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2102 | MS-10083 | Cassette Battery Pack | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2103 | MS-10083 | Cassette Battery Pack | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2104 | MS-10083 | Cassette Battery Pack | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2105 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Misalignment | Improper Geometry | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2106 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2107 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2108 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2109 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2110 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Misalignment | Adhesive Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2111 | MS-10083 | Cassette Battery Pack | M10317 | Square Foam Spacer with CBP Cut | Protects BMS, Shock Asorbtion | Fails to Absorb Shock | Material Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2112 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Misalignment | Improper Geometry | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2113 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2114 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2115 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2116 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2117 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Misalignment | Adhesive Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2118 | MS-10083 | Cassette Battery Pack | M10328 | Square Thick Foam Spacer | Protects BMS Bottom, Shock Asorbtion | Fails to Absorb Shock | Material Choice | Loss of Components on BMS, Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2119 | MS-10083 | Cassette Battery Pack | M50416 | INF-18650-2x2 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2120 | MS-10083 | Cassette Battery Pack | M50367 | AV-07-10031 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2121 | MS-10083 | Cassette Battery Pack | M50893 | CT250/2.50 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2122 | MS-10083 | Cassette Battery Pack | M50894 | STR156/0.75/062 | Connect Battery Cells Electrically | Fails to Connects Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2123 | MS-10083 | Cassette Battery Pack | M50023 | 18650 Rechargeable Battery Cells | Provide Power to E1 | Fails to Provide Sufficient Power | Cell Choice | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2124 | MS-10083 | Cassette Battery Pack | M50342 | Heat Shrink, FIT221 3/16 | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2125 | MS-10083 | Cassette Battery Pack | M50345 | Heat Shrink, FIT221 1/4 | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2126 | MS-10083 | Cassette Battery Pack | M50346 | Hot Melt, Bostik 2124 | Provides intercell Support | Structural failure under weight or load | Adhesive Choice | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2127 | MS-10083 | Cassette Battery Pack | M50346 | Hot Melt, Bostik 2124 | Provides intercell Support | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2128 | MS-10083 | Cassette Battery Pack | M50346 | Hot Melt, Bostik 2124 | Provides intercell Support | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2129 | MS-10083 | Cassette Battery Pack | M50352 | Kapton Tape, K250-3/4 | Contrains Cables on Pack, Assists in Constraining Cap | Structural failure under weight or load | Adhesive Choice | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2130 | MS-10083 | Cassette Battery Pack | M50352 | Kapton Tape, K250-3/4 | Contrains Cables on Pack, Assists in Constraining Cap | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2131 | MS-10083 | Cassette Battery Pack | M50352 | Kapton Tape, K250-3/4 | Contrains Cables on Pack, Assists in Constraining Cap | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2132 | MS-10083 | Cassette Battery Pack | M51163 | CT200/1.50 | Connect Battery Cells Electrically | Fails to Connect Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2133 | MS-10083 | Cassette Battery Pack | M50349 | Radius Tab, CTR200/1.00 | Connect Battery Cells Electrically | Fails to Connect Cells | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2134 | MS-10083 | Cassette Battery Pack | M50375 | Shrink Wrap (83mm) | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2135 | MS-10083 | Cassette Battery Pack | M50378 | AIM SAC305 flux core solder | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2136 | MS-10083 | Cassette Battery Pack | M50378 | AIM SAC305 flux core solder | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2137 | MS-10083 | Cassette Battery Pack | M50378 | AIM SAC305 flux core solder | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2138 | MS-10083 | Cassette Battery Pack | M10359 | Label: Cassette Battery Pack | Provides information to operator | Detaches from surface | Adhesive Choice | Label information not available | Operator Inconvenience or Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2139 | MS-10083 | Cassette Battery Pack | M10359 | Label: Cassette Battery Pack | Provides information to operator | Degrades over time | Material Choice | Label information not available | Operator Inconvenience or Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2140 | MS-10312 | Cassette Battery Pack - Harness | ES-10022 | Cassette Battery Management System (BMS) | Facilitates Power Transfer to Cassette from Cells | PCB failure | Individual component failure (open/shorts/etc) | Battery Failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use battery pack and cells certified to IEC 62133-2 | PRD20.18 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2141 | MS-10312 | Cassette Battery Pack - Harness | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Excessive Wear on Cables | Improper Geometry | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2142 | MS-10312 | Cassette Battery Pack - Harness | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2143 | MS-10312 | Cassette Battery Pack - Harness | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Material Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2144 | MS-10312 | Cassette Battery Pack - Harness | M10084 | Cassette End Cap | Provides Support for Battery Cells, Assists in Constraining Battery Pack within C1 | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2145 | MS-10312 | Cassette Battery Pack - Harness | M50366 | UL 1213 24 WHT | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2146 | MS-10312 | Cassette Battery Pack - Harness | M50371 | UL 1213 24 GRN | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2147 | MS-10312 | Cassette Battery Pack - Harness | M50361 | UL 1213 24 BLU | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2148 | MS-10312 | Cassette Battery Pack - Harness | M50372 | UL 1213 22 BLK | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2149 | MS-10312 | Cassette Battery Pack - Harness | M50373 | UL 1213 22 RED | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2150 | MS-10312 | Cassette Battery Pack - Harness | M50355 | UL 1213 16 RED | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2151 | MS-10312 | Cassette Battery Pack - Harness | M50356 | UL 1213 16 BLK | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2152 | MS-10312 | Cassette Battery Pack - Harness | M50363 | UL 1213 24 GRY | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2153 | MS-10312 | Cassette Battery Pack - Harness | M50364 | UL 1213 24 ORG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2154 | MS-10312 | Cassette Battery Pack - Harness | M50362 | UL 1213 24 BRW | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2155 | MS-10312 | Cassette Battery Pack - Harness | M50320 | Contact, 2.00mm CLIK-mate | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2156 | MS-10312 | Cassette Battery Pack - Harness | M50052 | Conn Plug, 2.00mm CLIK-mate, 8 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2157 | MS-10312 | Cassette Battery Pack - Harness | M50229 | Conn Plug, 2.00mm CLIK-mate, 4 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0937 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0938 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0939 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0940 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0941 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0942 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0943 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11087 | Cassette Bottom Populated | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0944 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0945 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0946 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0947 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0948 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0949 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0950 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | MS-11089 | Cassette Top Populated | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2158 | MS-10511 | Cassette Enclosure Without Battery/Battery Cover/Cover Screws | M50217 | Socket button head screw, M5 x 0.80 x 10 Stainless Steel | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0952 | MS-11089 | Cassette Top Populated | M11037 | Foam Spacer | Shock absorbtion and mounting | Fails to absorb shock (physical) | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0953 | MS-11089 | Cassette Top Populated | M11037 | Foam Spacer | Shock absorbtion and mounting | Fails to hold detector in place | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0954 | MS-11089 | Cassette Top Populated | M11037 | Foam Spacer | Shock absorbtion and mounting | Detector misaligned | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0955 | MS-11089 | Cassette Top Populated | M11037 | Foam Spacer | Shock absorbtion and mounting | Failed to dampen the vibration/shock | Faulty mounts | Damaged detector non-functional | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0956 | MS-11089 | Cassette Top Populated | M11037 | Foam Spacer | Shock absorbtion and mounting | Fails to absorb shock (physical) | Storage outside of recommended temperature range | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0957 | MS-11089 | Cassette Top Populated | M11037 | Foam Spacer | Shock absorbtion and mounting | Failed to dampen the vibration/shock | Storage outside of recommended temperature range | Damaged detector non-functional | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0958 | MS-11089 | Cassette Top Populated | M11044 | Bottom Foam Bumper | Shock absorbtion and mounting | Fails to absorb shock (physical) | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0959 | MS-11089 | Cassette Top Populated | M11044 | Bottom Foam Bumper | Shock absorbtion and mounting | Fails to hold detector in place | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0960 | MS-11089 | Cassette Top Populated | M11044 | Bottom Foam Bumper | Shock absorbtion and mounting | Detector misaligned | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0961 | MS-11089 | Cassette Top Populated | M11044 | Bottom Foam Bumper | Shock absorbtion and mounting | Failed to dampen the vibration/shock | Faulty mounts | Damaged detector non-functional | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0962 | MS-11089 | Cassette Top Populated | M11044 | Bottom Foam Bumper | Shock absorbtion and mounting | Fails to absorb shock (physical) | Storage outside of recommended temperature range | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0963 | MS-11089 | Cassette Top Populated | M11044 | Bottom Foam Bumper | Shock absorbtion and mounting | Failed to dampen the vibration/shock | Storage outside of recommended temperature range | Damaged detector non-functional | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0964 | MS-11089 | Cassette Top Populated | M11045 | Top Foam Bumper | Shock absorbtion and mounting | Fails to absorb shock (physical) | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0965 | MS-11089 | Cassette Top Populated | M11045 | Top Foam Bumper | Shock absorbtion and mounting | Fails to hold detector in place | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0966 | MS-11089 | Cassette Top Populated | M11045 | Top Foam Bumper | Shock absorbtion and mounting | Detector misaligned | Mechanical damage from external forces | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0967 | MS-11089 | Cassette Top Populated | M11045 | Top Foam Bumper | Shock absorbtion and mounting | Failed to dampen the vibration/shock | Faulty mounts | Damaged detector non-functional | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0968 | MS-11089 | Cassette Top Populated | M11045 | Top Foam Bumper | Shock absorbtion and mounting | Fails to absorb shock (physical) | Storage outside of recommended temperature range | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0969 | MS-11089 | Cassette Top Populated | M11045 | Top Foam Bumper | Shock absorbtion and mounting | Failed to dampen the vibration/shock | Storage outside of recommended temperature range | Damaged detector non-functional | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0970 | MS-11089 | Cassette Top Populated | M50506 | FFC, 0.5mm pitch, 24 CKT, 178mm | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0971 | MS-11089 | Cassette Top Populated | M50506 | FFC, 0.5mm pitch, 24 CKT, 178mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0972 | MS-11089 | Cassette Top Populated | M50506 | FFC, 0.5mm pitch, 24 CKT, 178mm | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0973 | MS-11089 | Cassette Top Populated | M50506 | FFC, 0.5mm pitch, 24 CKT, 178mm | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0974 | MS-11089 | Cassette Top Populated | M11079 | Cassette Bottom Edge Seal | Stops ingress to cassette | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0975 | MS-11089 | Cassette Top Populated | M11079 | Cassette Bottom Edge Seal | Stops ingress to cassette | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0976 | MS-11089 | Cassette Top Populated | M11079 | Cassette Bottom Edge Seal | Stops ingress to cassette | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0977 | MS-11089 | Cassette Top Populated | M11079 | Cassette Bottom Edge Seal | Stops ingress to cassette | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0978 | MS-11089 | Cassette Top Populated | M11079 | Cassette Bottom Edge Seal | Stops ingress to cassette | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0979 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Connects electrical components | Fail to allow correct electrical signal to pass through | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0980 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Epoxy potting could catch fire due to extreme temperature | Minor injury | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0981 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0982 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Connects electrical components | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0983 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0984 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0985 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0986 | MS-11089 | Cassette Top Populated | M50437 | FFC, 0.5mm pitch, 22 CKT, 52mm | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0987 | MS-11089 | Cassette Top Populated | M50004 | Detector | Mounts cassette to cart mount | Plate becomes detached | Sudden disconnect via mechanical damage | Product inoperable; shorted components | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0988 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | Fails to produce image | Device overheats | Reduced image quality | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0989 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | No image captured from detector | Detector connection lost | No x-ray acquisition | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0990 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | No image captured from detector | Power loss | No x-ray acquisition | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0991 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | No image captured from detector | Trigger Cable failure | X-ray emission without image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0992 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | No image captured from detector | Power Cable failure | X-ray emission without image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0993 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | No image captured from detector | Ethernet adapter failure | X-ray emission without image | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK0994 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | Dead pixels on detector | Detector damage or degradation | Poor image quality | Delay of Procedure | 4.0 | 5.0 | 20 | None Needed | N/A | No further planned remediation | 4.0 | 5.0 | 20 |
| DRSK0995 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | Fails to produce image | Device used in ambient temperature <10C | Reduced image quality | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0996 | MS-11089 | Cassette Top Populated | M50004 | Detector | Creates images from radiation | Fails to produce image | Device used in ambient temperature <10C | Reduced image quality | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK0997 | MS-11089 | Cassette Top Populated | M50341 | Ethernet Cassette Main to Detector | Cable from cassette to cassette main | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK0998 | MS-11089 | Cassette Top Populated | MS-11086 | Cassette Main - Modules Populated | Houses Majority of C1 Hardaware | Parts Misalligned | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK0999 | MS-11089 | Cassette Top Populated | MS-11113 | Cassette + Inserts and Light Pipes | Holds detector in place | Detector misaligned | Mount screw(s) came off | Field of view offset, xray image not useable, must be re-captured | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1000 | MS-11089 | Cassette Top Populated | MS-10384 | Tile Boards and Mounting Bracket | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1001 | MS-11089 | Cassette Top Populated | MS-10384 | Tile Boards and Mounting Bracket | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1002 | MS-11089 | Cassette Top Populated | MS-11049 | Button Harness | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1003 | MS-11089 | Cassette Top Populated | MS-11049 | Button Harness | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1004 | MS-11089 | Cassette Top Populated | MS-11049 | Button Harness | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1005 | MS-11089 | Cassette Top Populated | MS-11049 | Button Harness | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1006 | MS-11089 | Cassette Top Populated | MS-11049 | Button Harness | Connects electrical components | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1007 | MS-11089 | Cassette Top Populated | MS-11049 | Button Harness | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1008 | MS-11089 | Cassette Top Populated | MS-11074 | Cassette Tracking to Main Harness | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1009 | MS-11089 | Cassette Top Populated | MS-11074 | Cassette Tracking to Main Harness | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1010 | MS-11089 | Cassette Top Populated | MS-11074 | Cassette Tracking to Main Harness | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1011 | MS-11089 | Cassette Top Populated | MS-11074 | Cassette Tracking to Main Harness | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1012 | MS-11089 | Cassette Top Populated | MS-11074 | Cassette Tracking to Main Harness | Connects electrical components | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1013 | MS-11089 | Cassette Top Populated | MS-11074 | Cassette Tracking to Main Harness | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1014 | MS-11089 | Cassette Top Populated | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1015 | MS-11089 | Cassette Top Populated | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1016 | MS-11089 | Cassette Top Populated | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1017 | MS-11089 | Cassette Top Populated | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1018 | MS-11089 | Cassette Top Populated | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1019 | MS-11089 | Cassette Top Populated | MS-11048 | Detector Trigger Cable | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1020 | MS-11089 | Cassette Top Populated | MS-11048 | Detector Trigger Cable | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1021 | MS-11089 | Cassette Top Populated | MS-11048 | Detector Trigger Cable | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1022 | MS-11089 | Cassette Top Populated | MS-11048 | Detector Trigger Cable | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1023 | MS-11089 | Cassette Top Populated | MS-11048 | Detector Trigger Cable | Connects electrical components | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1024 | MS-11089 | Cassette Top Populated | MS-11048 | Detector Trigger Cable | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1025 | MS-11089 | Cassette Top Populated | MS-11047 | DETECTOR POWER HARNESS | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1026 | MS-11089 | Cassette Top Populated | MS-11047 | DETECTOR POWER HARNESS | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1027 | MS-11089 | Cassette Top Populated | MS-11047 | DETECTOR POWER HARNESS | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1028 | MS-11089 | Cassette Top Populated | MS-11047 | DETECTOR POWER HARNESS | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1029 | MS-11089 | Cassette Top Populated | MS-11047 | DETECTOR POWER HARNESS | Connects electrical components | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1030 | MS-11089 | Cassette Top Populated | MS-11047 | DETECTOR POWER HARNESS | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1031 | MS-11089 | Cassette Top Populated | MS-11114 | Tracking Board with Hardware | Enables Tracking from C1 to E1 | Tracking Failure | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2159 | MS-11089 | Cassette Top Populated | MS-11114 | Tracking Board with Hardware | Enables Tracking from C1 to E1 | Tracking Failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1033 | MS-11089 | Cassette Top Populated | MS-11110 | Cassette Display Assembly | Bracket that holds the antennaes above the isolation board | Fails to stay attached | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1034 | MS-11089 | Cassette Top Populated | MS-11110 | Cassette Display Assembly | Bracket that holds the antennaes above the isolation board | Display is obscured | Too much pressure applied to screen - breaks | Display not visible to operator | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1035 | MS-11089 | Cassette Top Populated | MS-11088 | Carbon Plate and Foam | Distributes Weight in C1 when loaded, adds structural support for C1 when is loaded | Fails to support loaded C1 | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2160 | MS-11089 | Cassette Top Populated | MS-11088 | Carbon Plate and Foam | Distributes Weight in C1 when loaded, adds structural support for C1 when is loaded | Fails to support loaded C1 | Sudden disconnect via mechanical damage | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2161 | MS-11089 | Cassette Top Populated | MS-11088 | Carbon Plate and Foam | Distributes Weight in C1 when loaded, adds structural support for C1 when is loaded | Fails to support loaded C1 | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2162 | MS-11089 | Cassette Top Populated | MS-11088 | Carbon Plate and Foam | Distributes Weight in C1 when loaded, adds structural support for C1 when is loaded | Fails to support loaded C1 | Improper allignment | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1036 | MS-11089 | Cassette Top Populated | M10478 | Socket button head Torx screw M3 x 0.50 x 6, Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1037 | MS-11089 | Cassette Top Populated | M50066 | Wifi Antenna | Connects emitter to cassette | Becomes displaced | Sudden disconnect via mechanical damage | Inability to initiate trigger | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1038 | MS-11089 | Cassette Top Populated | M50066 | Wifi Antenna | Connects emitter to cassette | Becomes displaced | Sudden disconnect via mechanical damage | Unable to view images on tablet | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1039 | MS-11089 | Cassette Top Populated | M50066 | Wifi Antenna | Connects emitter to cassette | Becomes displaced | Sudden disconnect via mechanical damage | Inability to send Images outside of device | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1040 | MS-11089 | Cassette Top Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1041 | MS-11089 | Cassette Top Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1042 | MS-11089 | Cassette Top Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1043 | MS-11089 | Cassette Top Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1044 | MS-11089 | Cassette Top Populated | M50445 | Torx Plus Rounded Head Thread Forming Screw M3, 8mm Long | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1848 | MS-11089 | Cassette Top Populated | M51184 | Ezuiro Sub-GHz Antenna | Wireless connectivity/communication with Emitter | Antenna fails | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1055 | MS-11089 | Cassette Top Populated | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1056 | MS-11089 | Cassette Top Populated | M50259 | Cable tie, 1.8mm wide, 71mm long | Holds wires/cables/electrical components in place | Fails to hold component in place | Creepage and clearance underspecified | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1063 | MS-11089 | Cassette Top Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1064 | MS-11089 | Cassette Top Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin)Parts fall into sterile bag during surgery | Temporary Discomfort | 1.0 | 2.0 | 2.0 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1065 | MS-11089 | Cassette Top Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1066 | MS-11089 | Cassette Top Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1067 | MS-11089 | Cassette Top Populated | M50213 | Acrylic Adhesive Tape 1" | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1068 | MS-11089 | Cassette Top Populated | MS-11103 | P01 Cassette Jetson Heat Pipe Assy | Cool the Jetson in C1 | Fails to sufficiently cool Jetson | Sudden disconnect via mechanical damage | Insufficient Cooling, Jetson Temperature Fault, Device Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2163 | MS-11089 | Cassette Top Populated | MS-11103 | P01 Cassette Jetson Heat Pipe Assy | Cool the Jetson in C1 | Fails to sufficiently cool Jetson | Improper allignment | Insufficient Cooling, Jetson Temperature Fault, Device Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1069 | MS-11089 | Cassette Top Populated | M10480 | Socket Button Head Screw M2.5 0.45mm x 4mm SS, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1070 | MS-11089 | Cassette Top Populated | M50452 | Tracking Tile Board FFC | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1071 | MS-11089 | Cassette Top Populated | M11166 | Wifi Cowling | Retains antenna connectors | Mechanical Damage | Mechanical damage from external forces | Antennas no longer have an additional level of retention | Delay of Procedure | 4.0 | 3.0 | 12 | Spec material to withstand expected environmental forces | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1072 | MS-11089 | Cassette Top Populated | M11167 | Sub-GHz Cowling | Retains antenna connectors | Mechanical Damage | Mechanical damage from external forces | Antennas no longer have an additional level of retention | Delay of Procedure | 4.0 | 3.0 | 12 | Spec material to withstand expected environmental forces | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1073 | MS-11089 | Cassette Top Populated | M10772 | Fan Wire Shield | Prevents fan wires from getting pinched during assembly | Mechanical Damage | Mechanical damage from external forces | Becomes loose and rattles inside of cassette. | Minor Dissatisfaction | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1076 | MS-11089 | Cassette Top Populated | M11169 | Jetson Additional Cooling Block | Retains heat pipe | Mechanical Failure | Mechanical damage from external forces | Decreased thermal performance Risk of short circuit | Minor discomfort | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1078 | MS-11089 | Cassette Top Populated | M50083 | Socket button head screw M3x0.5 x 10 Stainless Steel | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1080 | MS-11089 | Cassette Top Populated | M50882 | 18-8 Stainless Steel Hex Flat Head Screws M2 x 0.40 mm Thread Size, 4 mm Long | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1083 | MS-11089 | Cassette Top Populated | M50281 | Socket button head screw M3x0.5 x 12 Stainless Steel | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1084 | MS-11089 | Cassette Top Populated | M50856 | Silicone Oil 5000 CST | Lubricant to assist during assembly | Lubricant ineffective | Improper storage before application | Difficulty during assembly. Silicone dampeners damaged during assembly. Minor Dissatisfaction | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1085 | MS-11089 | Cassette Top Populated | M50289 | Anaerobic adhesive, Loctite 242 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1086 | MS-11089 | Cassette Top Populated | M50289 | Anaerobic adhesive, Loctite 242 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin)Parts fall into sterile bag during surgery | Temporary Discomfort | 1.0 | 2.0 | 2.0 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1087 | MS-11089 | Cassette Top Populated | M50289 | Anaerobic adhesive, Loctite 242 | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1088 | MS-11089 | Cassette Top Populated | M50289 | Anaerobic adhesive, Loctite 242 | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1730 | MS-11089 | Cassette Top Populated | M50215 | 36-768-ND-SCREW | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12.0 | No further planned remediation | N/A | No further planned remediation | 4.0 | 3.0 | 12.0 |
| DRSK1731 | MS-11089 | Cassette Top Populated | M11168 | Tile Ribbon Cable Retainer | Retains Flat Flex Cable | Parts become dislodged | Improper specifications | Product operable; long term reliability may be compromised | Minor Dissatisfaction | 1.0 | 3.0 | 12.0 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 12.0 |
| DRSK1732 | MS-11089 | Cassette Top Populated | M50342 | Heat Shrink, FIT221 3/16 | Provides insulation between heat pipe and surrounding environment | Break down over time | Elevated Temperature | No insulation between heat pipe and environment. Risk of short circuit | Delay of Procedure | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12.0 |
| DRSK2164 | MS-11089 | Cassette Top Populated | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | Cools Charging Components on the Cassette Tracking Board | Jetson Overheat | Improper pipe routing | Jetson throttle - device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2165 | MS-11089 | Cassette Top Populated | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | Cools Charging Components on the Cassette Tracking Board | Jetson Overheat | Mechanical damage from external forces | Jetson throttle - device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2166 | MS-11089 | Cassette Top Populated | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | Cools Charging Components on the Cassette Tracking Board | Jetson Overheat | Insufficient heat transfer capacity | Jetson throttle - device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2167 | MS-11089 | Cassette Top Populated | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | Cools 5V0 Regulator on the Cassette Tracking Board | Jetson Overheat | Improper pipe routing | Jetson throttle - device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2168 | MS-11089 | Cassette Top Populated | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | Cools 5V0 Regulator on the Cassette Tracking Board | Jetson Overheat | Mechanical damage from external forces | Jetson throttle - device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2169 | MS-11089 | Cassette Top Populated | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | Cools 5V0 Regulator on the Cassette Tracking Board | Jetson Overheat | Insufficient heat transfer capacity | Jetson throttle - device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2170 | MS-11049 | Button Harness | M50182 | Conn Plug, 1.50mm CLIK-mate, 4 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2171 | MS-11049 | Button Harness | M50107 | Wire, 28 AWG, Yellow, Alpha Wire 422807 YL005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2172 | MS-11049 | Button Harness | M50104 | Wire, 28 AWG, Black, Alpha Wire 422807 BK005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2173 | MS-11049 | Button Harness | M50106 | Wire, 28 AWG, Green, Alpha Wire 422807 GR005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2174 | MS-11049 | Button Harness | M50430 | Heat Shrink, 3mm, 3:1 Shrink, Adhesive lined | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2175 | MS-11049 | Button Harness | M50187 | Contact, 1.50mm CLIK-mate, 24-28 AWG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2176 | MS-11074 | Cassette Tracking to Main Harness | M50427 | CLIK-Mate Plug Housing,  1.50mm Pitch, Single Row,  Positive Lock, 9 Circuits,  Natural | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2177 | MS-11074 | Cassette Tracking to Main Harness | M50201 | Wire, 24 AWG, Red | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2178 | MS-11074 | Cassette Tracking to Main Harness | M50242 | Wire, 24 AWG, Black | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2179 | MS-11074 | Cassette Tracking to Main Harness | M50187 | Contact, 1.50mm CLIK-mate, 24-28 AWG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2180 | MS-11048 | Detector Trigger Cable | M50186 | Lemo 6 Pos, FGG.0B.306 | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2181 | MS-11048 | Detector Trigger Cable | M50105 | Wire, 28 AWG, White, Alpha Wire 422807 WH005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2182 | MS-11048 | Detector Trigger Cable | M50104 | Wire, 28 AWG, Black, Alpha Wire 422807 BK005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2183 | MS-11048 | Detector Trigger Cable | M50106 | Wire, 28 AWG, Green, Alpha Wire 422807 GR005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2184 | MS-11048 | Detector Trigger Cable | M50107 | Wire, 28 AWG, Yellow, Alpha Wire 422807 YL005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2185 | MS-11048 | Detector Trigger Cable | M50224 | Heat Shrink, 0.205" ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2186 | MS-11048 | Detector Trigger Cable | M50223 | Heat Shrink,  0.045" ID supplied, 1.2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2187 | MS-11048 | Detector Trigger Cable | M50182 | Conn Plug, 1.50mm CLIK-mate, 4 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2188 | MS-11048 | Detector Trigger Cable | M50187 | Contact, 1.50mm CLIK-mate, 24-28 AWG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2189 | MS-11047 | DETECTOR POWER HARNESS | M50191 | LEMO 4 Pos, FGG.0B.304 | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2190 | MS-11047 | DETECTOR POWER HARNESS | M50181 | Click Mate Plug 5POS 1.50MM | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2191 | MS-11047 | DETECTOR POWER HARNESS | M50242 | Wire, 24 AWG, Black, Alpha Wire, 3050 BK005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2192 | MS-11047 | DETECTOR POWER HARNESS | M50201 | Wire, 24 AWG, Red, Alpha Wire, 3050 RD005 | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2193 | MS-11047 | DETECTOR POWER HARNESS | M50156 | Heat Shrink, 2.11mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2194 | MS-11047 | DETECTOR POWER HARNESS | M50224 | Heat Shrink, 0.205" ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2195 | MS-11047 | DETECTOR POWER HARNESS | M50187 | Contact, 1.50mm CLIK-mate, 24-28 AWG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Product Inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1089 | MS-11088 | Carbon Plate and Foam | M10238 | Carbon Sandwich | Distribues weight to allow higher force to be applied to surface of the cassette | Plate breaks | Poor weight distribution | Hole in enclosure; loss of electrical safety | Moderate Electrical Shock | 7.0 | 2.0 | 14.0 | Plate weight test (600lb load) | PRD11.10 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1090 | MS-11088 | Carbon Plate and Foam | M11035 | Carbon Fiber Plate Adhesive | Distribues weight to allow higher force to be applied to surface of the cassette | Fail to protect operator/patient against single fault | Creepage and clearance underspecified | Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1091 | MS-11088 | Carbon Plate and Foam | M11046 | Carbon Fiber Plate Foam | Distribues weight to allow higher force to be applied to surface of the cassette | Plate breaks | Poor weight distribution | Hole in enclosure; loss of electrical safety | Moderate Electrical Shock | 7.0 | 2.0 | 14.0 | Plate weight test (600lb load) | PRD11.10 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1092 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1093 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1094 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1095 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1096 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1097 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1098 | MS-11087 | Cassette Bottom Populated | MS-11092 | Cassette Enclosure Bottom with Inserts | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1099 | MS-11087 | Cassette Bottom Populated | MS-10385 | CASSETTE FAN HARNESS | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1100 | MS-11087 | Cassette Bottom Populated | MS-10385 | CASSETTE FAN HARNESS | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1101 | MS-11087 | Cassette Bottom Populated | MS-10385 | CASSETTE FAN HARNESS | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1102 | MS-11087 | Cassette Bottom Populated | MS-10385 | CASSETTE FAN HARNESS | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1103 | MS-11087 | Cassette Bottom Populated | MS-10385 | CASSETTE FAN HARNESS | Connects electrical components | Shorts | Improper crimp specification | Product inoperable | Minor Fire | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1104 | MS-11087 | Cassette Bottom Populated | MS-10385 | CASSETTE FAN HARNESS | Connects electrical components | Shorts | Sudden disconnect via mechanical damage | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1105 | MS-11087 | Cassette Bottom Populated | M11085 | Cassette Catch Plate | Mounts cassette to cart mount | Plate breaks | Sudden disconnect via mechanical damage | Product inoperable; shorted components | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1106 | MS-11087 | Cassette Bottom Populated | M11040 | Cassette Bottom Bumper | Shock absorbtion and mounting | Fails to stay in place | Mechanical damage from external forces | Scew imaging surface | Minor Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1107 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1108 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin  Plastic | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1109 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1110 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1111 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1112 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1113 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1114 | MS-11087 | Cassette Bottom Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1115 | MS-11087 | Cassette Bottom Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1116 | MS-11087 | Cassette Bottom Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1117 | MS-11087 | Cassette Bottom Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1118 | MS-11087 | Cassette Bottom Populated | M50443 | Torx Plus Rounded Head Thread Forming Screw M3, 5mm Long | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1119 | MS-11087 | Cassette Bottom Populated | M11083 | CASSETTE DUCT COVER | Thermal protection for the cassette - directs airflow for thermal control;Prevents access to fan and heatsinks | Fails to direct airflow | Mechanical damage from external forces | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1120 | MS-11087 | Cassette Bottom Populated | M11083 | CASSETTE DUCT COVER | Thermal protection for the cassette - directs airflow for thermal control;Prevents access to fan and heatsinks | Acdess to fan and heatsink | Cover detaches | Reduced performance | Minor Burn or Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1121 | MS-11087 | Cassette Bottom Populated | M11101 | VESA Mount Plate | Mounts cassette to cart mount | Plate breaks | Sudden disconnect via mechanical damage | Product inoperable; shorted components | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1122 | MS-11087 | Cassette Bottom Populated | M11102 | VESA Mount Plate Adhesive | Mounts cassette to cart mount | Plate becomes detached | Sudden disconnect via mechanical damage | Product inoperable; shorted components | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1123 | MS-11087 | Cassette Bottom Populated | M10492 | Torx Flat Head Screws, M3 x 0.50 x 8 Stainless Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1124 | MS-11087 | Cassette Bottom Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1125 | MS-11087 | Cassette Bottom Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin)Parts fall into sterile bag during surgery | Temporary Discomfort | 1.0 | 2.0 | 2.0 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1126 | MS-11087 | Cassette Bottom Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1127 | MS-11087 | Cassette Bottom Populated | M50888 | Anaerobic adhesive, Loctite 403 | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1128 | MS-11087 | Cassette Bottom Populated | M50856 | Silicone Oil 5000 CST | Lubricant to assist during assembly | Lubricant ineffective | Improper storage before application | Difficulty during assembly. Silicone dampeners damaged during assembly. Minor Dissatisfaction | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2196 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Structural failure under weight or load | Material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2197 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Structural failure under weight or load | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2198 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2199 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2200 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 2.0 | 8 | Compliance to ISO 10993 | PRD20.16 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2201 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2202 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Fails to insulate | Material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2203 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2204 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2205 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Material Choice | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2206 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2207 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2208 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2209 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2210 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2211 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2212 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2213 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2214 | MS-11092 | Cassette Enclosure Bottom with Inserts | M11084 | Bottom Enclosure Ducted | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2215 | MS-11092 | Cassette Enclosure Bottom with Inserts | M50414 | IBB-M3-6 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2216 | MS-10385 | CASSETTE FAN HARNESS | M50428 | Fan, DB0590505H1A-BT0 | Provides Cooling to C1 Heatsinks | Fails to Provide Cooling to C1 | Incorrect Part Specification | Overheating | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2217 | MS-10385 | CASSETTE FAN HARNESS | M50430 | Heat Shrink, 3mm, 3:1 shrink, Adhesive lined | Provides insulation | Detaches from surface | Improper material/size choice | Conductors not insulated | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2218 | MS-10385 | CASSETTE FAN HARNESS | M50429 | Rubber Grommet, 1/4" Hole, 3/32" Thick, 1/8" ID | Seals Cable Pass through for fan harness | Fails to seal | Incorrect Part Specification | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2219 | MS-10385 | CASSETTE FAN HARNESS | M50421 | Conn Plug, Pico-Clasp, 4 Pos | Connects to Corrosponding Connector | Fails Connection | Incorrect Part Specification | Jetson overheats, faults | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2220 | MS-10385 | CASSETTE FAN HARNESS | M50422 | Contact, Pico-Clasp, Tin Plated, 28-32 AWG | Electrically Connects Components | Fails Electrical Connection | Incorrect Part Specification | Jetson overheats, faults | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1129 | MS-11110 | Cassette Display Assembly | M50449 | Self Tapper M3 5mm | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1130 | MS-11110 | Cassette Display Assembly | M50449 | Self Tapper M3 5mm | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1131 | MS-11110 | Cassette Display Assembly | M50449 | Self Tapper M3 5mm | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1132 | MS-11110 | Cassette Display Assembly | M50449 | Self Tapper M3 5mm | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1133 | MS-11110 | Cassette Display Assembly | M50003 | CASSETTE SCREEN | Bracket that holds the antennaes above the isolation board | Fails to stay attached | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1134 | MS-11110 | Cassette Display Assembly | M50003 | CASSETTE SCREEN | Bracket that holds the antennaes above the isolation board | Display is obscured | Too much pressure applied to screen - breaks | Display not visible to operator | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1135 | MS-11110 | Cassette Display Assembly | M11069 | Display Carrier | Bracket that holds display screen | Misalligned | Mechanical damage from external forces | Screen obstructed | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1136 | MS-11110 | Cassette Display Assembly | ES-10037 | MX1 Cassette Display PCBA | Provides tracking to device | Open LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1137 | MS-11110 | Cassette Display Assembly | ES-10037 | MX1 Cassette Display PCBA | Controls display output/input | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1138 | MS-11110 | Cassette Display Assembly | M10404 | Display Adhesive, Cassette Display | Holds display screen in place | Display detaches | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1139 | MS-11110 | Cassette Display Assembly | M10450 | DISPLAY ASSEMBLY ESD ADHESIVE | Insulates display from possible ESD event | Adhesive failure | Poor Surface Finish | ESD event destroys display board or cassette main | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1140 | MS-11091 | Cassette Button Carrier with Inserts | MS-11082 | Button Cover Assembly | Turns Cassette ON/OFF | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1141 | MS-11091 | Cassette Button Carrier with Inserts | MS-11082 | Button Cover Assembly | Turns Cassette ON/OFF | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1142 | MS-11091 | Cassette Button Carrier with Inserts | MS-11082 | Button Cover Assembly | Turns Cassette ON/OFF | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1143 | MS-11091 | Cassette Button Carrier with Inserts | MS-11082 | Button Cover Assembly | Turns Cassette ON/OFF | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1144 | MS-11091 | Cassette Button Carrier with Inserts | MS-11082 | Button Cover Assembly | Turns Cassette ON/OFF | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1145 | MS-11091 | Cassette Button Carrier with Inserts | M11036 | T-ROD | Connects the silicon button cover to the button(s) | Rod breaks | Mechanical damage from external forces | Product inoperable; Unable to power on | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1146 | MS-11091 | Cassette Button Carrier with Inserts | M11036 | T-ROD | Connects the silicon button cover to the button(s) | Rod unable to trigger button press | Too short | Product inoperable; Unable to power on | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1147 | MS-11091 | Cassette Button Carrier with Inserts | M10478 | Socket button head Torx screw M3 x 0.50 x 6, Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1148 | MS-11091 | Cassette Button Carrier with Inserts | ES-10029 | BUTTON BOARD | Relays button push action to cassette main | Overvoltage | Individual component failure (open/shorts/etc) | Damage detector; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1149 | MS-11091 | Cassette Button Carrier with Inserts | ES-10029 | BUTTON BOARD | Connects the silicon button cover to the button(s) | Rod breaks | Mechanical damage from external forces | Product inoperable; Unable to power on | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1150 | MS-11091 | Cassette Button Carrier with Inserts | ES-10029 | BUTTON BOARD | Connects the silicon button cover to the button(s) | Rod unable to trigger button press | Too short | Product inoperable; Unable to power on | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1151 | MS-11091 | Cassette Button Carrier with Inserts | M11078 | CASSETTE MOLDED BUTTON | Turns Cassette ON/OFF | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1152 | MS-11091 | Cassette Button Carrier with Inserts | M11078 | CASSETTE MOLDED BUTTON | Turns Cassette ON/OFF | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1153 | MS-11091 | Cassette Button Carrier with Inserts | M11078 | CASSETTE MOLDED BUTTON | Turns Cassette ON/OFF | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1154 | MS-11091 | Cassette Button Carrier with Inserts | M11078 | CASSETTE MOLDED BUTTON | Turns Cassette ON/OFF | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1155 | MS-11091 | Cassette Button Carrier with Inserts | M11078 | CASSETTE MOLDED BUTTON | Turns Cassette ON/OFF | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2221 | MS-11082 | Button Cover Assembly | M11080 | Cassette Button Carrier | Ridigdly attaches buttons to C1 Shell | Structural failure under weight or load | Material Choice | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2222 | MS-11082 | Button Cover Assembly | M11080 | Cassette Button Carrier | Ridigdly attaches buttons to C1 Shell | Structural failure under weight or load | Material Plastically Deforms | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2223 | MS-11082 | Button Cover Assembly | M11080 | Cassette Button Carrier | Ridigdly attaches buttons to C1 Shell | Structural failure under weight or load | High Temperature damages the material | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2224 | MS-11082 | Button Cover Assembly | M11080 | Cassette Button Carrier | Ridigdly attaches buttons to C1 Shell | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2225 | MS-11082 | Button Cover Assembly | M11080 | Cassette Button Carrier | Ridigdly attaches buttons to C1 Shell | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Loose Component | Minor Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK2226 | MS-11082 | Button Cover Assembly | M50414 | IBB-M3-6 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1156 | MS-10384 | Tile Boards and Mounting Bracket | ES-10038 | MX1 Cassette Angled Tracking PCB | Controls display output/input | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1157 | MS-10384 | Tile Boards and Mounting Bracket | ES-10038 | MX1 Cassette Angled Tracking PCB | Provides tracking to device | PCB failure | Individual component failure (open/shorts/etc) | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1158 | MS-10384 | Tile Boards and Mounting Bracket | ES-10038 | MX1 Cassette Angled Tracking PCB | Provides tracking to device | Short LED | Faulty LED | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1159 | MS-10384 | Tile Boards and Mounting Bracket | ES-10038 | MX1 Cassette Angled Tracking PCB | Provides tracking to device | Open LED | Faulty LED | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1160 | MS-10384 | Tile Boards and Mounting Bracket | ES-10038 | MX1 Cassette Angled Tracking PCB | Provides tracking to device | Short LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1161 | MS-10384 | Tile Boards and Mounting Bracket | ES-10038 | MX1 Cassette Angled Tracking PCB | Provides tracking to device | Open LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1162 | MS-10384 | Tile Boards and Mounting Bracket | MS-11112 | Cassette Side Board Mount Assembly | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1163 | MS-10384 | Tile Boards and Mounting Bracket | MS-11112 | Cassette Side Board Mount Assembly | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1164 | MS-10384 | Tile Boards and Mounting Bracket | M10478 | Socket button head Torx screw M3 x 0.50 x 6, Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2227 | MS-10384 | Tile Boards and Mounting Bracket | M11298 | Tile Board Spacer | Insulate Tile Board from Tile Bracket | Misalignment | Improper Geometry | Conductors not insulated | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2228 | MS-10384 | Tile Boards and Mounting Bracket | M11298 | Tile Board Spacer | Insulate Tile Board from Tile Bracket | Structural failure under weight or load | Structural failure due to fatigue | Conductors not insulated | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2229 | MS-10384 | Tile Boards and Mounting Bracket | M11298 | Tile Board Spacer | Insulate Tile Board from Tile Bracket | Structural failure under weight or load | Material Choice | Conductors not insulated | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2230 | MS-10384 | Tile Boards and Mounting Bracket | M11298 | Tile Board Spacer | Insulate Tile Board from Tile Bracket | Structural failure under weight or load | Part degrades from aging | Conductors not insulated | Minor Electrical shock | 4.0 | 3.0 | 12 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2231 | MS-10384 | Tile Boards and Mounting Bracket | M11298 | Tile Board Spacer | Insulate Tile Board from Tile Bracket | Structural failure under weight or load | Structural failure due to fatigue/repeated use | Conductors not insulated | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2232 | MS-11112 | Cassette Side Board Mount Assembly | M11111 | LED TILE SHEET METAL BRACKET | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2233 | MS-11112 | Cassette Side Board Mount Assembly | M11111 | LED TILE SHEET METAL BRACKET | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2234 | MS-11112 | Cassette Side Board Mount Assembly | M50436 | Self-Clinching Nut M3 | Fastener | Mechanical Connection Failure | Component Choice | Tracking Misalignment, device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1165 | MS-11086 | Cassette Main - Modules Populated | M50817 | Intel Wireless-AC 9260, 2230, 2x2 AC+BT, Gigabit, No vPro | Provides Connectivity to Tablet and Emitter | No Wifi | Mechanical damage | Device unable to communicate with PACS server | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1166 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Short LED | Faulty LED | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1167 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Open LED | Faulty LED | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1168 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Short LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1169 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Open LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1170 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Overheats | Shorting of a component along main power line | Potential for burnt internal parts; device inoperable | Major Fire | 7.0 | 2.0 | 14 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1171 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Battery Damage | Individual component failure (open/shorts/etc) | Battery failure - potential combustion | Major Fire | 7.0 | 2.0 | 14 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1172 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Unregulated Power rail | Individual component failure | Potential for burnt internal parts; device inoperable | Major Fire | 7.0 | 2.0 | 14 | Incoming Inspection | QSP-014 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1173 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | Overheats | Individual component failure (open/shorts/etc) | Increased touch temperatures | Minor Burn | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1174 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Cassette failure | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1175 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1176 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1177 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1178 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1179 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1180 | MS-11086 | Cassette Main - Modules Populated | ES-10004 | Cassette Main PCBA | PCB that controls the power distribution and contains CPU, NVMe, WIFI, and LTE modules | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1181 | MS-11086 | Cassette Main - Modules Populated | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1182 | MS-11086 | Cassette Main - Modules Populated | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1183 | MS-11086 | Cassette Main - Modules Populated | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Unable to save images | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1184 | MS-11086 | Cassette Main - Modules Populated | M51021 | NVMe Viking - M.2 2230 256GB | SW updates | Memory Failure | Sudden disconnect via mechanical damage | Unable to take image | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1185 | MS-11086 | Cassette Main - Modules Populated | M10478 | Socket button head Torx screw M3 x 0.50 x 6, Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1186 | MS-11086 | Cassette Main - Modules Populated | M11076 | Molded USB C Port | Covers USB C port | Fails to cover port | Mechanical damage from external forces | No device effect; potential for damage to port | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1187 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Essential performance testing following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1188 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Parts become dislodged | Screw and bolt diameter/threads too small, nut center hole diameter too big | Parts fall from device onto patient anatomy (intact skin) | Temporary Discomfort | 1.0 | 2.0 | 2 | Visual inspection for loose components following ISTA 3A simulated shipping | PRD20.26 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1189 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Connector mechanical failure | Material failure - Brittle | Interior shorting; Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1190 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1191 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1192 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1193 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1194 | MS-11086 | Cassette Main - Modules Populated | M50157 | Thread-Forming Screws for Thin Plastic | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1195 | MS-11086 | Cassette Main - Modules Populated | M50214 | Rubber PCB Isolation Grommet | Silicone bushing that mounts/supports boards | Becomes detached | Sudden disconnect via mechanical damage | No Device Effect | No Patient Effect | 1.0 | 1.0 | 1 | None Needed | N/A | No further planned remediation | 1.0 | 1.0 | 1 |
| DRSK1196 | MS-11086 | Cassette Main - Modules Populated | M10459 | Cassette USB Ferrite Cap | Attaches Ferrite | Mechanical Failure | Mechanical Shock | Ferrite becomes loose - could cause a data harness failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1197 | MS-11086 | Cassette Main - Modules Populated | M50423 | USB 3.2 Type-C IP67 Rated Jumper | USB Port connector | Misalignment | Mechanical Shock | Broken Data or Power Cable; Cannot Charge or Communicate | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1198 | MS-11086 | Cassette Main - Modules Populated | M50423 | USB 3.2 Type-C IP67 Rated Jumper | USB Port connector | Port Wear-Out | Repetitive Use | Broken Data or Power Cable; Cannot Charge or Communicate | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1199 | MS-11086 | Cassette Main - Modules Populated | M50432 | Screw, M2 x 0.40mm x 4mmL | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1733 | MS-11086 | Cassette Main - Modules Populated | M50538 | Ferrite | EMC Choke | Mechanical Failure | Not properly constrained | EMC issues; impact other devices | Low Operator Inconvenience | 1.0 | 3.0 | 3 | IEC 60601-2 Testing | PRD20.6 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1201 | MS-11103 | P01 Cassette Heat Pipe and Block | M11097 | Heat Sink Plate | Expels heat | Overheat | Improper specifications | Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1202 | MS-11103 | P01 Cassette Heat Pipe and Block | M11099 | Heat Sink Plate Gasket | Seals heat sink to plate | Overheat | Improper specifications | Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1203 | MS-11103 | P01 Cassette Heat Pipe and Block | M11100 | P01 Cassette Heat Sink, UB50-9B | Stops ingress to the heat sink | Ingress to heat sink | Mechanical damage from external forces | Product inoperable; shorted components | Delay of Procedure | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1204 | MS-11103 | P01 Cassette Heat Pipe and Block | M11100 | P01 Cassette Heat Sink, UB50-9B | Stops ingress to the heat sink | Ingress to heat sink | Too thin | Product inoperable; shorted components | Delay of Procedure | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1205 | MS-11103 | P01 Cassette Heat Pipe and Block | M10479 | Socket button head screw M2 x 0.4 x 4  Zinc-Plated Alloy Steel, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1206 | MS-11103 | P01 Cassette Heat Pipe and Block | M11098 | Heat Sink Plate TIM | Expels heat | Overheat | Improper specifications | Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1207 | MS-11103 | P01 Cassette Heat Pipe and Block | M10494 | Xavier NX screw, with thread locker | Fastener | Internal components become accessible | Screws loosen over time | Internal components become accessible | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1208 | MS-11103 | P01 Cassette Heat Pipe and Block | M50082 | Jetson Xavier NX Leaf Spring | Runs device and ensures proper functioning | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1209 | MS-11103 | P01 Cassette Heat Pipe and Block | M50082 | Jetson Xavier NX Leaf Spring | Provides impact resistance to Jetson | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1210 | MS-11103 | P01 Cassette Heat Pipe and Block | M50101 | Jetson Xavier NX | Runs device and ensures proper functioning | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1211 | MS-11103 | P01 Cassette Heat Pipe and Block | M50101 | Jetson Xavier NX | Runs device and ensures proper functioning | Jetson failure | Sudden disconnect via mechanical damage | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1212 | MS-11103 | P01 Cassette Heat Pipe and Block | M50153 | Thermal Paste, TC3 | Gap filler for heat transfer | Overheat | Improper specifications | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of monoblock/jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1213 | MS-11103 | P01 Cassette Heat Pipe and Block | M50153 | Thermal Paste, TC3 | Thermal protection for the cassette | Overheat | Improper material choice | Reduced performance | Operator Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1214 | MS-11103 | P01 Cassette Heat Pipe and Block | M50617 | Copper Foil Tape Stock | Gap filler for heat transfer | Too thin/Too Thick | Improper material choice | Thermal Performance Reduction; Early Jetson Shutdown | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-2-28 | PRD20.9 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1215 | MS-11103 | P01 Cassette Heat Pipe and Block | MS-11093 | Cassette Jetson Heat Pipe ASSY | Draw heat from one location to another | Jetson/Monoblock Overheat | Improper pipe routing | Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1216 | MS-11103 | P01 Cassette Heat Pipe and Block | MS-11093 | Cassette Jetson Heat Pipe ASSY | Draw heat from one location to another | Jetson/Monoblock Overheat | Mechanical damage from external forces | Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1217 | MS-11103 | P01 Cassette Heat Pipe and Block | MS-11093 | Cassette Jetson Heat Pipe ASSY | Draw heat from one location to another | Jetson/Monoblock Overheat | Insufficient heat transfer capacity | Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8 | Thermal monitoring of jetson | PRD2.3 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2235 | MS-11093 | Cassette Jetson Heat Pipe Assembly | M11094 | P01 Cassette Jetson Heat Pipe | Takes heat from Jetson Heat Block to Heat Sinks | Failure to transfer heat | Contact prevented via mechanical damage | Jetson overheats, faults | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2236 | MS-11093 | Cassette Jetson Heat Pipe Assembly | M11095 | P01 Cassette Jetson Heat Pipe Mirrored | Takes heat from Jetson Heat Block to Heat Sinks | Failure to transfer heat | Mechanical damage from outside forces | Jetson overheats, faults | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2237 | MS-11093 | Cassette Jetson Heat Pipe Assembly | M11096 | P01 Cassette Jetson Heat Pipe Mounting Block | Pulls Heat from Jetson to transfer to Heat Pipes | Failure to transfer heat | Contact prevented via mechanical damage | Jetson overheats, faults | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1219 | MS-11113 | Cassette + Inserts and Light Pipes | M50234 | LIGHT PIPE TOP CAP | Brings tracking LED to cassette surface | Broken light pipe | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1220 | MS-11113 | Cassette + Inserts and Light Pipes | M50234 | LIGHT PIPE TOP CAP | Brings status LED to visible surface | Broken light pipe | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1221 | MS-11113 | Cassette + Inserts and Light Pipes | M11052 | VISIBLE LED LIGHT PIPE CAP | Brings status LED to visible surface | Broken light pipe | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1222 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1223 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1224 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1225 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1226 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1227 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1228 | MS-11113 | Cassette + Inserts and Light Pipes | M11104 | Cassette Detector Area Decal | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1229 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1230 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1231 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1232 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1233 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1234 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1235 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11090 | CASSETTE TOP - NEW DUCTED - ASSY | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1734 | MS-11113 | Cassette + Inserts and Light Pipes | M11165 | Cassette silicone led light pipes | Directs tracking LED light to cassette surface | Becomes dislodged | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1735 | MS-11113 | Cassette + Inserts and Light Pipes | M11165 | Cassette silicone led light pipes | Directs tracking LED light to cassette surface | Does not direct status LEDs | Incorrect material choice | Status unclear to operator | Delay of Procedure | 4.0 | 2.0 | 8 | Tracking VVPR | VVPR-P01-101 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1736 | MS-11113 | Cassette + Inserts and Light Pipes | M11165 | Cassette silicone led light pipes | Directs tracking LED light to cassette surface | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1737 | MS-11113 | Cassette + Inserts and Light Pipes | M11165 | Cassette silicone led light pipes | Directs tracking LED light to cassette surface | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2238 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Structural failure under weight or load | Material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2239 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Structural failure under weight or load | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2240 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2241 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2242 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 2.0 | 8 | Desktop Review with Nelson Labs | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2243 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2244 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Fails to insulate | Material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2245 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2246 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2247 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Material Choice | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2248 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2249 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2250 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2251 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2252 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2253 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2254 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2255 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2256 | MS-11090 | Cassette Top - New ducted Assembly | M11077 | CASSETTE TOP - NEW DUCTED | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC  60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2257 | MS-11090 | Cassette Top - New ducted Assembly | M50376 | IBB-M5-6 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2258 | MS-11090 | Cassette Top - New ducted Assembly | M50377 | IBB-M3-12 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2259 | MS-11090 | Cassette Top - New ducted Assembly | M50413 | M3 x 0.5 SPIROL HEATED INSERT | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2260 | MS-11090 | Cassette Top - New ducted Assembly | M50414 | IBB-M3-6 | Fastener | Mechanical Connection Failure | Material Choice | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1236 | MS-11114 | Tracking Board with Hardware | ES-10036 | Tracking PCBA | Connects the silicon button cover to the button(s) | Rod unable to trigger button press | Too short | Product inoperable; Unable to power on | Delay of Procedure | 4.0 | 2.0 | 8 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1237 | MS-11114 | Tracking Board with Hardware | ES-10036 | Tracking PCBA | Provides tracking to device | PCB failure | Individual component failure (open/shorts/etc) | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1238 | MS-11114 | Tracking Board with Hardware | ES-10036 | Tracking PCBA | Provides tracking to device | Short LED | Faulty LED | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1239 | MS-11114 | Tracking Board with Hardware | ES-10036 | Tracking PCBA | Provides tracking to device | Open LED | Faulty LED | LEDs failure, device inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | LED redundancy | PRD3.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1240 | MS-11114 | Tracking Board with Hardware | ES-10036 | Tracking PCBA | Provides tracking to device | Short LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1241 | MS-11114 | Tracking Board with Hardware | ES-10036 | Tracking PCBA | Provides tracking to device | Open LED | Faulty RGB LED | RGB LEDs failure | Moderate Operator Inconvenience | 1.0 | 3.0 | 3 | LED redundancy | PRD3.4 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1242 | MS-11114 | Tracking Board with Hardware | M50434 | Snap-in Supports with Female Connection | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1243 | MS-11114 | Tracking Board with Hardware | M50434 | Snap-in Supports with Female Connection | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1244 | MS-11114 | Tracking Board with Hardware | M50434 | Snap-in Supports with Female Connection | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1245 | MS-11114 | Tracking Board with Hardware | M50434 | Snap-in Supports with Female Connection | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1246 | MS-11114 | Tracking Board with Hardware | M50450 | Phillips Rounded Head Thread-Forming Screws for Plastic, 18-8 Stainless Steel, M3.5 Screw Size, 8 mm Long | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1247 | MS-11114 | Tracking Board with Hardware | M50450 | Phillips Rounded Head Thread-Forming Screws for Plastic, 18-8 Stainless Steel, M3.5 Screw Size, 8 mm Long | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1248 | MS-11114 | Tracking Board with Hardware | M50450 | Phillips Rounded Head Thread-Forming Screws for Plastic, 18-8 Stainless Steel, M3.5 Screw Size, 8 mm Long | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1249 | MS-11114 | Tracking Board with Hardware | M50450 | Phillips Rounded Head Thread-Forming Screws for Plastic, 18-8 Stainless Steel, M3.5 Screw Size, 8 mm Long | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1250 | MS-11114 | Tracking Board with Hardware | M50214 | Rubber PCB Isolation Grommet | Silicone bushing that mounts/supports boards | Becomes detached | Sudden disconnect via mechanical damage | No Device Effect | No Patient Effect | 1.0 | 1.0 | 1 | None Needed | N/A | No further planned remediation | 1.0 | 1.0 | 1 |
| DRSK1251 | MS-11139 | Cassette Handle Assembly | M11130 | Cassette Handle Top | Provides grip for operator | Drops Cassette | Material fails - cannot support load | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1252 | MS-11139 | Cassette Handle Assembly | M11130 | Cassette Handle Top | Provides grip for operator | Grip uncomfortable to operator | Non-ergonomic design | None | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1253 | MS-11139 | Cassette Handle Assembly | M11131 | Cassette Handle Bottom | Provides grip for operator | Drops Cassette | Material fails - cannot support load | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1254 | MS-11139 | Cassette Handle Assembly | M11131 | Cassette Handle Bottom | Provides grip for operator | Grip uncomfortable to operator | Non-ergonomic design | None | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1255 | MS-11139 | Cassette Handle Assembly | M11134 | Cassette Handle Strap | Allows operator to access grip and supports weight | Drops Cassette | Material fails - cannot support load | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1256 | MS-11139 | Cassette Handle Assembly | M11135 | Cassette Handle Connector Screw Base | Mates handle and strap to cassette | Does not keep handle attached to cassette | Incorrect design | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1257 | MS-11139 | Cassette Handle Assembly | M11136 | Cassette Handle Connector Screw Cap | Mates handle and strap to cassette | Does not keep handle attached to cassette | Incorrect design | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1258 | MS-11139 | Cassette Handle Assembly | M50494 | Flat Head Thread-Forming Screws for Plastic Torx, Zinc-Plated Steel, M3 Screw Size, 16.000 mm Long | Holds parts together | Components are not kept held together | Improper thread spec | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1259 | MS-11139 | Cassette Handle Assembly | M50495 | 18-8 Stainless Steel Narrow Cheese Head Slotted Screws M3 x 0.5 mm Thread, 10 mm Long, 4.00 mm Head Diameter | Holds parts together | Components are not kept held together | Improper thread spec | Product inoperable | Minor injury if dropped on a person | 4.0 | 1.0 | 4 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK1270 | H1 | Wired Charger | M50011 | 100 Watt USB-C PD Medical Desktop Power Supply | Charges emitter and cassette | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1271 | H1 | Wired Charger | M50011 | 100 Watt USB-C PD Medical Desktop Power Supply | Charges emitter and cassette | Enclosure cracks | Drops | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1272 | H1 | Wired Charger | M50011 | 100 Watt USB-C PD Medical Desktop Power Supply | Charges emitter and cassette | Electrical failure | Internal damage/electrical short | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1273 | H1 | Wired Charger | M50011 | 100 Watt USB-C PD Medical Desktop Power Supply | Charges emitter and cassette | Electrical failure | Under spec'd | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1274 | H1 | Wired Charger | M50011 | 100 Watt USB-C PD Medical Desktop Power Supply | Charges emitter and cassette | Electrical failure | Under spec'd | Product inoperable | Moderate Electrical Shock | 7.0 | 3.0 | 21 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1275 | H1 | Wired Charger | M50316 | H1 Outlet NEMA-C7 Cable | Charges emitter and cassette | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1276 | H1 | Wired Charger | M50316 | H1 Outlet NEMA-C7 Cable | Charges emitter and cassette | Enclosure cracks | Drops | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1277 | H1 | Wired Charger | M50316 | H1 Outlet NEMA-C7 Cable | Charges emitter and cassette | Electrical failure | Internal damage/electrical short | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1278 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Electrical failure | Internal damage/electrical short | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1279 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1280 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Illegible | Improper color choice | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1281 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1282 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1283 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1284 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1285 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1286 | H1 | Wired Charger | M10172 | Label:Wired Charger Brick Ra | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1287 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Electrical failure | Internal damage/electrical short | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Charger OTS IEC 60601-1 Certified | RSK_R125 | 2MOP in Cassette | 4.0 | 3.0 | 12 |
| DRSK1288 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1289 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Illegible | Improper color choice | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1290 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1291 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1292 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1293 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1294 | H1 | Wired Charger | M10173 | Label: Wired Charger Brick Cord | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1295 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1296 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1297 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1298 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1299 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1300 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1301 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1302 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1303 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1304 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1305 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1306 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1307 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1308 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1309 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1310 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1311 | H1 | Wired Charger | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1312 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Failure to allow proper spacing | Improper specifications - too small | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1313 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Failure to allow proper spacing | Improper specifications - too small | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1314 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Failure to maintain integrity of product | Improper material choice | Potential damage to product external | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1315 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Failure to maintain integrity of product | Improper material choice | Potential damage to product internal | Delay of Procedure | 4.0 | 2.0 | 8 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1316 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Allows ingress | Improper seal | Potential damage to product internal | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1317 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Fails to close and lock | Improper locking mechanism | Potential damage to product external | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1318 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Excess internal pressure | No pressure release feature | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1319 | P1 | Case | M50322 | Pelican Case, iM2720 Storm Travel Case | Hold and transport device | Excess internal pressure | No pressure release feature | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1320 | P1 | Case | M50619 | Velocity Systems Computer Sleeve Large | Hold and transport device | Failure to maintain integrity of product | Improper specifications | Damage to components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1321 | P1 | Case | M10613 | Pelican Foam Top | Hold Cassette in Case | Fails to hold cassette in place | Improper specifications | Damage to components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1322 | P1 | Case | M10614 | Pelican Foam Bottom | Holds Emitter, Puck Box, Tablet, and Charging Cables in Case | Fails to hold device in place | Improper specifications | Damage to components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1323 | P1 | Case | M10608 | Lid Foam Half | Holds Cassette in Case | Fails to hold cassette in place | Improper specifications | Damage to components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1324 | P1 | Case | M10625 | Velcro Strip | Adheres to case and provides mounting for Computer Sleeve | Failure to maintain integrity of product | Improper specifications | Damage to components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1325 | P1 | Case | M10361 | Pelican Case MedAI Logo | Provides Information to Operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1326 | P1 | Case | M10361 | Pelican Case MedAI Logo | Adheres to case and provides mounting for Computer Sleeve | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1327 | P1 | Case | M10361 | Pelican Case MedAI Logo | Adheres to case and provides mounting for Computer Sleeve | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1328 | P1 | Case | M10178 | Label: Standardized GTIN | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Damage other components; Product inoperable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1329 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1330 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1331 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1332 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1333 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1334 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1335 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1336 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1337 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Usability Test | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1338 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1339 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1340 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1341 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1342 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1343 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1344 | P1 | Case | M10178 | Label: Standardized GTIN | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2279 | P1 | Case | M50240 | Hot Glue, 3M, 3792LM | Used to attach Lid Foam to P1 | Structural failure under weight or load | Adhesive Choice | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2280 | P1 | Case | M50240 | Hot Glue, 3M, 3792LM | Used to attach Lid Foam to P1 | Structural failure under weight or load | Structural failure due to fatigue | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK2281 | P1 | Case | M50240 | Hot Glue, 3M, 3792LM | Used to attach Lid Foam to P1 | Structural failure under weight or load | Part degrades from aging | Loose Component | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1346 | MS-10627 | Puck Box with Pucks | M10101 | Puck #1 (5cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter; increases exposure | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1347 | MS-10627 | Puck Box with Pucks | M10101 | Puck #1 (5cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1348 | MS-10627 | Puck Box with Pucks | M10101 | Puck #1 (5cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1349 | MS-10627 | Puck Box with Pucks | M10101 | Puck #1 (5cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1350 | MS-10627 | Puck Box with Pucks | M10101 | Puck #1 (5cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1351 | MS-10627 | Puck Box with Pucks | M10102 | Puck #2 (6cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1352 | MS-10627 | Puck Box with Pucks | M10102 | Puck #2 (6cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1353 | MS-10627 | Puck Box with Pucks | M10102 | Puck #2 (6cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1354 | MS-10627 | Puck Box with Pucks | M10102 | Puck #2 (6cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1355 | MS-10627 | Puck Box with Pucks | M10102 | Puck #2 (6cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1356 | MS-10627 | Puck Box with Pucks | M10103 | Puck #3 (7cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1357 | MS-10627 | Puck Box with Pucks | M10103 | Puck #3 (7cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1358 | MS-10627 | Puck Box with Pucks | M10103 | Puck #3 (7cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1359 | MS-10627 | Puck Box with Pucks | M10103 | Puck #3 (7cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1360 | MS-10627 | Puck Box with Pucks | M10103 | Puck #3 (7cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1361 | MS-10627 | Puck Box with Pucks | M10104 | Puck #4 (8cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1362 | MS-10627 | Puck Box with Pucks | M10104 | Puck #4 (8cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1363 | MS-10627 | Puck Box with Pucks | M10104 | Puck #4 (8cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1364 | MS-10627 | Puck Box with Pucks | M10104 | Puck #4 (8cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1365 | MS-10627 | Puck Box with Pucks | M10104 | Puck #4 (8cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1366 | MS-10627 | Puck Box with Pucks | M10105 | Puck #5 (9cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1367 | MS-10627 | Puck Box with Pucks | M10105 | Puck #5 (9cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1368 | MS-10627 | Puck Box with Pucks | M10105 | Puck #5 (9cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1369 | MS-10627 | Puck Box with Pucks | M10105 | Puck #5 (9cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1370 | MS-10627 | Puck Box with Pucks | M10105 | Puck #5 (9cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1371 | MS-10627 | Puck Box with Pucks | M10106 | Puck #6 (10cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1372 | MS-10627 | Puck Box with Pucks | M10106 | Puck #6 (10cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1373 | MS-10627 | Puck Box with Pucks | M10106 | Puck #6 (10cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1374 | MS-10627 | Puck Box with Pucks | M10106 | Puck #6 (10cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1375 | MS-10627 | Puck Box with Pucks | M10106 | Puck #6 (10cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1376 | MS-10627 | Puck Box with Pucks | M10107 | Puck #7 (11cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1377 | MS-10627 | Puck Box with Pucks | M10107 | Puck #7 (11cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1378 | MS-10627 | Puck Box with Pucks | M10107 | Puck #7 (11cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1379 | MS-10627 | Puck Box with Pucks | M10107 | Puck #7 (11cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1380 | MS-10627 | Puck Box with Pucks | M10107 | Puck #7 (11cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1381 | MS-10627 | Puck Box with Pucks | M10108 | Puck #8 (12cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1382 | MS-10627 | Puck Box with Pucks | M10108 | Puck #8 (12cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1383 | MS-10627 | Puck Box with Pucks | M10108 | Puck #8 (12cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1384 | MS-10627 | Puck Box with Pucks | M10108 | Puck #8 (12cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1385 | MS-10627 | Puck Box with Pucks | M10108 | Puck #8 (12cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1386 | MS-10627 | Puck Box with Pucks | M10109 | Puck #9 (13cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1387 | MS-10627 | Puck Box with Pucks | M10109 | Puck #9 (13cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1388 | MS-10627 | Puck Box with Pucks | M10109 | Puck #9 (13cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1389 | MS-10627 | Puck Box with Pucks | M10109 | Puck #9 (13cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1390 | MS-10627 | Puck Box with Pucks | M10109 | Puck #9 (13cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1391 | MS-10627 | Puck Box with Pucks | M10110 | Puck #10 (14cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1392 | MS-10627 | Puck Box with Pucks | M10110 | Puck #10 (14cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1393 | MS-10627 | Puck Box with Pucks | M10110 | Puck #10 (14cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1394 | MS-10627 | Puck Box with Pucks | M10110 | Puck #10 (14cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1395 | MS-10627 | Puck Box with Pucks | M10110 | Puck #10 (14cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1396 | MS-10627 | Puck Box with Pucks | M10111 | Puck #11 (15cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1397 | MS-10627 | Puck Box with Pucks | M10111 | Puck #11 (15cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1398 | MS-10627 | Puck Box with Pucks | M10111 | Puck #11 (15cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1399 | MS-10627 | Puck Box with Pucks | M10111 | Puck #11 (15cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1400 | MS-10627 | Puck Box with Pucks | M10111 | Puck #11 (15cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1401 | MS-10627 | Puck Box with Pucks | M10112 | Puck #12 (16cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1402 | MS-10627 | Puck Box with Pucks | M10112 | Puck #12 (16cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1403 | MS-10627 | Puck Box with Pucks | M10112 | Puck #12 (16cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1404 | MS-10627 | Puck Box with Pucks | M10112 | Puck #12 (16cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1405 | MS-10627 | Puck Box with Pucks | M10112 | Puck #12 (16cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1406 | MS-10627 | Puck Box with Pucks | M10113 | Puck #13 (17cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1407 | MS-10627 | Puck Box with Pucks | M10113 | Puck #13 (17cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1408 | MS-10627 | Puck Box with Pucks | M10113 | Puck #13 (17cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1409 | MS-10627 | Puck Box with Pucks | M10113 | Puck #13 (17cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1410 | MS-10627 | Puck Box with Pucks | M10113 | Puck #13 (17cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1411 | MS-10627 | Puck Box with Pucks | M10114 | Puck #14 (18cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1412 | MS-10627 | Puck Box with Pucks | M10114 | Puck #14 (18cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1413 | MS-10627 | Puck Box with Pucks | M10114 | Puck #14 (18cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1414 | MS-10627 | Puck Box with Pucks | M10114 | Puck #14 (18cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1415 | MS-10627 | Puck Box with Pucks | M10114 | Puck #14 (18cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1416 | MS-10627 | Puck Box with Pucks | M10115 | Puck #15 (19cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1417 | MS-10627 | Puck Box with Pucks | M10115 | Puck #15 (19cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1418 | MS-10627 | Puck Box with Pucks | M10115 | Puck #15 (19cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1419 | MS-10627 | Puck Box with Pucks | M10115 | Puck #15 (19cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1420 | MS-10627 | Puck Box with Pucks | M10115 | Puck #15 (19cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1421 | MS-10627 | Puck Box with Pucks | M10116 | Puck #16 (20cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1422 | MS-10627 | Puck Box with Pucks | M10116 | Puck #16 (20cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1423 | MS-10627 | Puck Box with Pucks | M10116 | Puck #16 (20cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1424 | MS-10627 | Puck Box with Pucks | M10116 | Puck #16 (20cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1425 | MS-10627 | Puck Box with Pucks | M10116 | Puck #16 (20cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1426 | MS-10627 | Puck Box with Pucks | M10117 | Puck #17 (21cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1427 | MS-10627 | Puck Box with Pucks | M10117 | Puck #17 (21cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1428 | MS-10627 | Puck Box with Pucks | M10117 | Puck #17 (21cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1429 | MS-10627 | Puck Box with Pucks | M10117 | Puck #17 (21cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1430 | MS-10627 | Puck Box with Pucks | M10117 | Puck #17 (21cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1431 | MS-10627 | Puck Box with Pucks | M10118 | Puck #18 (22cm@1m) | Further collimates the active beam | Fails to filter x-ray beam | Improper specifications/geometry - too thin | Improper beam energy leaving emitter | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1432 | MS-10627 | Puck Box with Pucks | M10118 | Puck #18 (22cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - hole too small | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1433 | MS-10627 | Puck Box with Pucks | M10118 | Puck #18 (22cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too little | Improper specifications/geometry - too thick | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1434 | MS-10627 | Puck Box with Pucks | M10118 | Puck #18 (22cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry- hole to large | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1435 | MS-10627 | Puck Box with Pucks | M10118 | Puck #18 (22cm@1m) | Further collimates the active beam | Fails to collimate x-ray beam - too high | Improper specifications/geometry - too thin | Improper collimation | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1437 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Holds pucks during transport | Mechanical Failure - Box degrades/falls apart | Overstrain during shipping, weakened due to moisture/water | Pucks are loose and could damage devices in case. | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1438 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2.0 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1439 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1440 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | Usability Test | Usability | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1441 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | Comply to IEC 60601-2-54 | PRD20.9 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1442 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1443 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1444 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1445 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1446 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1447 | MS-10627 | Puck Box with Pucks | M10790 | MX1 Puck Box | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1449 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | Usability Test | Usability | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1450 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | Comply to IEC 60601-2-54 | Usability | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1451 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Detaches from surface | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1452 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Detaches from surface | Improper adhesive | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1453 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Degrades over time | Improper material choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1454 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Degrades over time | Improper material choice - ink | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | P00 Equivalent Material (MEMO) | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1455 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2.0 | Validation of cleaning method | PRD13.13 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1546 | MS-10221 | LH Laser ASSY | MS-10094 | E1 LH Laser Harness | Transfer low voltage power to Laser | Insulation worn by friction over time causing electrical short | Strain relief points become disconnected | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1547 | MS-10221 | LH Laser ASSY | MS-10094 | E1 LH Laser Harness | Transfer low voltage power to Laser | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Product inoperable | Minor Electrical Shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1550 | MX1 | Top-level assembly | MX1 | Top-level assembly |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| DRSK1551 | MS-10334 | Overshipper Assembly | MS-10334 | Overshipper Assembly | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1552 | MS-10334 | Overshipper Assembly | MS-10334 | Overshipper Assembly | Protect outer shell of case from scratches during shipping | Fails to close and lock | Improper locking mechanism | Potential damage to product external | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1553 | MS-10334 | Overshipper Assembly | MS-10334 | Overshipper Assembly | Protects device while in case | Failure to allow proper spacing | Improper specifications -cavities too large | Potential damage to product internal | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1554 | C1 | Cassette Main Assy | C1 | Cassette Main Assy | Connects components together | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1555 | C1 | Cassette Main Assy | C1 | Cassette Main Assy | Polycarbonate enclosure for cassette | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1556 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1557 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1558 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Electrical shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1559 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1560 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1561 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1562 | MS-11089 | Cassette Top Populated | MS-11089 | Cassette Top Populated | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1563 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1564 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1565 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1566 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1567 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1568 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1569 | MS-11087 | Cassette Bottom Populated | MS-11087 | Cassette Bottom Populated | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1570 | MS-11110 | Cassette Display Assembly | MS-11110 | Cassette Display Assembly | Bracket that holds the antennaes above the isolation board | Fails to stay attached | Improper allignment | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1571 | MS-11110 | Cassette Display Assembly | MS-11110 | Cassette Display Assembly | Bracket that holds the antennaes above the isolation board | Display is obscured | Too much pressure applied to screen - breaks | Display not visible to operator | Delay of Procedure | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1572 | MS-11091 | Cassette Button Carrier with Inserts | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Internally shorts | Mechanically fault (buttons) | Buttons do not work; device inoperable | Delay of Procedure | 4.0 | 2.0 | 8.0 | DEMO VVPR - Haptic Feedback | PRD10.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1573 | MS-11091 | Cassette Button Carrier with Inserts | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Improper spacings | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1574 | MS-11091 | Cassette Button Carrier with Inserts | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Fails to insulate | Improper material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1575 | MS-11091 | Cassette Button Carrier with Inserts | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Fails to seal | Button pulled out/dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1576 | MS-11091 | Cassette Button Carrier with Inserts | MS-11091 | Cassette Button Carrier with Inserts | Turns Cassette ON/OFF | Button surface degradation | Improper material choice | Basic safety compromised-ingress; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1577 | MS-10384 | Tile Boards and Mounting Bracket | MS-10384 | Tile Boards and Mounting Bracket | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1578 | MS-10384 | Tile Boards and Mounting Bracket | MS-10384 | Tile Boards and Mounting Bracket | Aligns LED tile boards in cassette | Misaligns LEDs | Mechanical damage from external forces | User unable to view LED indicators; tracking failure; device inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1579 | MS-11113 | Cassette + Inserts and Light Pipes | MS-11113 | Cassette + Inserts and Light Pipes | Holds detector in place | Detector misaligned | Mount screw(s) came off | Field of view offset, xray image not useable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1581 | E1 | Emitter ASSY | E1 | Emitter ASSY | Polycarbonate enclosure for emitter | Connector mechanical failure | Material failure - Brittle | Parts fall from device | Minor Injury | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2261 | MS-10268 | Emitter Shell L, with thread inserts | M50180 | Thread Insert, IBB-M3-4 | Provide method for clamping half shells together | Half Shells begin to come apart | Degradation over time | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1582 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1583 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1584 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1585 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1586 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1587 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1588 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1589 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1590 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1591 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1592 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1593 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1594 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1595 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1596 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1597 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1598 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1599 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1600 | MS-10132 | Front Face ASSY | MS-10132 | Front Face ASSY | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1601 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1602 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1603 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1604 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1605 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1606 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1607 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1608 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1609 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1610 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1611 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1612 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1613 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1614 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1615 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1616 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1617 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1618 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Drop Test | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1619 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1620 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Connects electrical components | Insulation worn by friction over time | Strain relief points become disconnected | Conductors not insulated | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1621 | MS-10134 | Shell R Populated ASSY | MS-10134 | Shell R Populated ASSY | Connects electrical components | Harnesses could be damaged over time by other internal parts | Poor strain relief implementation | Exposed Conductors or Exposed Connection Ends | Minor Electrical Shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1622 | MS-10136 | E1 Lower Internal ASSY | MS-10136 | E1 Lower Internal ASSY | Sheet Metal bracket that holds the coil and attachs to the PCB | Frame structural integrity fails | Improper material choice - warps/bends | Intermittent charging | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1623 | MS-10141 | Power Cleat ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil misaligned | Improper geometry | Unable to charge wirelessly/slow charging speed | Delay of Procedure | 4.0 | 3.0 | 3.0 | Charging Verification Test | PRD5.11/5.12 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1624 | MS-10141 | Power Cleat ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil misaligned | Mechanical damage from external forces | Unable to charge wirelessly/slow charging speed | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1625 | MS-10141 | Power Cleat ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil disconnected | Poor coil wire routing | Unable to charge wirelessly | Delay of Procedure | 4.0 | 3.0 | 3.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1626 | MS-10141 | Power Cleat ASSY | MS-10141 | Power Cleat ASSY | Enclosure face and bracket for Emitter Power Input PCBA and wireless charging receiving coil | Receiver coil inefficient | Excessive spacing from outer surface | Slow charging speed | Delay of Procedure | 4.0 | 3.0 | 3.0 | Charging Verification Test | PRD5.11/5.12 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2262 | MS-10369 | Lit Cleat Cap ASSY | M10250 | Power Cleat Cap, IM | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Use of Common Engineering Plastics | RSK_R169 | No further planned remediation | 4.0 | 1.0 | 4 |
| DRSK2263 | MS-10369 | Lit Cleat Cap ASSY | M10368 | Cleat Cap Light Pipe | Indicates whether device is charging or not | Becomes dislodged | Mechanical damage from external forces | Unable to indicate charge state | Minor Dissatisfaction | 1.0 | 2.0 | 2 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1627 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1628 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1629 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1630 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1631 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1632 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | ISTA TestDrop Test | PRD20.26PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1633 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1634 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1635 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1636 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1637 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1638 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1639 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1640 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1641 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1642 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1643 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Ingress testing performed on DV unit - Intertek | PRD20.4 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1644 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1645 | MS-10145 | UB 5035-15, Gasket Applied | MS-10145 | UB 5035-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1646 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1647 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1648 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12.0 | ISTA TestDrop Test | PRD20.26PRD20.4 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1649 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1650 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1651 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1652 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to insulate | Incorrect material choice | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8.0 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1653 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thick | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1654 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Too thin | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1655 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Wrong material | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1656 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too small | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1657 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Width too big | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1658 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper adhesive | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1659 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Incorrect bonding process | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1660 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Improper allignment - between enclosure shells | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1661 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Fails to seal (from ingress) | Adhesive joint - too small | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1662 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Degradation over time | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1663 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Mechanical damage from external forces | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1664 | MS-10144 | UB 60-15, Gasket Applied | MS-10144 | UB 60-15, Gasket Applied | Protects interior of device | Internal parts are subjected to ingress | Shifts out of place/becomes dislodged | Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1665 | MS-10135 | Fan-Duct ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Monoblock/Jetson Overheat | Fan speed too slow | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 2.0 | Thermal monitoring of monoblock/jetson | PRD2.2 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1666 | MS-10135 | Fan-Duct ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Jetson/Monoblock Overheat | Fan stops | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 2.0 | Thermal monitoring of monoblock/jetson | PRD2.2 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1667 | MS-10135 | Fan-Duct ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Fan Overheat | Debris blocks fan | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8.0 | Covered fan | RSK_R159 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1668 | MS-10135 | Fan-Duct ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Fan stops | Ingress of dust into enclosure | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 8.0 | Covered fan | RSK_R159 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1669 | MS-10135 | Fan-Duct ASSY | MS-10135 | Fan-Duct ASSY | Expels heat from heat sink | Biohazard | Contaminates get into device | Dirty interior | Infection | 7.0 | 2.0 | 2.0 | Covered fan | RSK_R159 | No further planned remediation | 7.0 | 2.0 | 14 |
| DRSK1670 | MS-10148 | Emitter Main ASSY | MS-10148 | Emitter Main ASSY | Assem that controls power and data within the emitter (motherboard) | Thermistor failure | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1671 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Reaction time of aperture size/rotation change is too slow. | Motor/encoder/electronics can not react fast enough to satisfy operator | No product effect | Moderate Dissatisfaction | 1.0 | 5.0 | 5.0 | None Needed | N/A | No further planned remediation | 1.0 | 5.0 | 5 |
| DRSK1672 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator aperture opening not accurate over required range | Misalignment of collimator to source | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 4.0 | 16.0 | Compliance to IEC 60601-1-2-54 | PRD20.3 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1673 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator aperture opening not accurate over required range | Collimater clearances/tolerances | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 4.0 | 16.0 | Compliance to IEC 60601-1-2-54 | PRD20.3 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1674 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator aperture opening not accurate over required range | Encoder step size insufficient | Misalignment of X-ray beam to detector; improper anatomy irradiated | Negligible Radiation Tissue Reaction | 1.0 | 4.0 | 16.0 | Compliance to IEC 60601-1-2-54 | PRD20.3 | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1676 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Collimator Jam | Operation outside of specified temperature range | Collimator fails to collimate beam; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1677 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Motor failure | Operation outside of specified temperature range | Collimator fails to collimate beam; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1678 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Motor failure | Operation outside of specified temperature range | Slow collimation; device still operable | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1679 | MS-10008 | Collimator | MS-10008 | Collimator | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Homing failure | Homing LED/sensor failure | Collimator fails to collimate beam; Device inoperable | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1680 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Motorized collimator that moves Tungsten leaves to restricts the x ray field to the desired size | Radiation | Inadequate input V | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1681 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3.0 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1682 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3.0 | ISTA Testing | PRD20.25 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1683 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | OvercurrentMay disable laser guidance but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1684 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Control signal open | Loose connection to connector | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 3.0 | 3.0 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1685 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Improper specifications/geometry | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1686 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Incorrect material choice | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1687 | MS-10200 | Collimator - Line Laser ASSY | MS-10200 | Collimator - Line Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Misalligned (stackup error) | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1688 | MS-10149 | Collimator Bracket-Camera ASSY | MS-10149 | Collimator Bracket-Camera ASSY | Assem that contains all necessary sensors to detect and measure the device's physical enviroment | PCB failure | Individual component failure (open/shorts/etc) | Device inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | Incoming PCB Inspection | QSP-014 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1689 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | Draw heat from one location to another | Jetson/Monoblock Overheat | Improper pipe routing | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 2.0 | Thermal monitoring of monoblock/jetson | PRD2.2 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1690 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | Draw heat from one location to another | Jetson/Monoblock Overheat | Mechanical damage from external forces | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 2.0 | Thermal monitoring of monoblock/jetson | PRD2.2 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1691 | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | MS-10272 | Monoblock Heat Pipe ASSY, with Hardware | Draw heat from one location to another | Jetson/Monoblock Overheat | Insufficient heat transfer capacity | Tube death/Jetson throttle - device inoperable (full servicing required) | Delay of Procedure | 4.0 | 2.0 | 2.0 | Thermal monitoring of monoblock/jetson | PRD2.2 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1692 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3.0 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1693 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3.0 | ISTA Testing | PRD20.25 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1694 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | OvercurrentMay disable laser guidance but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1695 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Control signal open | Loose connection to connector | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 3.0 | 3.0 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1696 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Improper specifications/geometry | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1697 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Incorrect material choice | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1698 | MS-10222 | RH Laser ASSY | MS-10222 | RH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Misalligned (stackup error) | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1699 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Incorrect geometry | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3.0 | Incoming Inspection | QSP-014 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1700 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Crosshair misaligned from x-ray axis | Loose fit of lasers in mount causing movement | Product operable, may result in misaligned image | Negligible Radiation Tissue Reaction | 1.0 | 3.0 | 3.0 | ISTA Testing | PRD20.25 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1701 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Power supply malfunction | Vcc exceeds 3.3V | OvercurrentMay disable laser guidance but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1702 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Control signal open | Loose connection to connector | Laser does not turn on when activatedLaser guidance disables but system functional; Operator can take x-ray under interlock | Operator Dissatisfaction | 1.0 | 3.0 | 3.0 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1703 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Improper specifications/geometry | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1704 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Incorrect material choice | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1705 | MS-10221 | LH Laser ASSY | MS-10221 | LH Laser ASSY | Indicate x-ray axis | Incorrect placement/alignment | Misalligned (stackup error) | Incorrect beam alignment shown; device still operable | Negligible Radiation Tissue Reaction | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1706 | MS-10030 | Drive ASSY | MS-10030 | Drive ASSY | Motopr and input pulley | No collimation movement | Loose Drive Pulley set screws | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 3.0 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1707 | H1 | Wired Charger | H1 | Wired Charger | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label information | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1708 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Provides information to operator | Degrades over time | Improper cleaning method | Operator unable to see label informatiom | Minor Dissatisfaction or Instrument failure, no results | 1.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1709 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Protect outer shell of case from scratches during shipping | Fails to close and lock | Improper locking mechanism | Potential damage to product external | Delay of Procedure | 4.0 | 2.0 | 2.0 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1710 | MS-10334 | Puck Box with Pucks | MS-10334 | Puck Box with Pucks | Protects device while in case | Failure to allow proper spacing | Improper specifications -cavities too large | Potential damage to product internal | Delay of Procedure | 4.0 | 2.0 | 2.0 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1711 | MS-10627 | Puck Box with Pucks | MS-10627 | Puck Box with Pucks | Holds pucks during transport | Mechanical Failure - Box degrades/falls apart | Overstrain during shipping, weakened due to moisture/water | Pucks are loose and could damage devices in case. | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1741 | MS-10007 | Monoblock | M10158 | Label: Monoblock RA | Provides information to operator | Detaches from surface | Improper material choice | Service personnel unable to see label information | None | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1742 | MS-10007 | Monoblock | M10158 | Label: Monoblock RA | Provides information to operator | Degrades over time | Improper material choice | Service personnel unable to see label information | None | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1743 | MS-10007 | Monoblock | M10462 | Angled Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1744 | MS-10007 | Monoblock | M10463 | Front Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1745 | MS-10007 | Monoblock | M10464 | Top Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1746 | MS-10007 | Monoblock | M10465 | Middle Right Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1747 | MS-10007 | Monoblock | M10466 | Middle Left Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1748 | MS-10007 | Monoblock | M10467 | Back Mirrored Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1749 | MS-10007 | Monoblock | M10468 | Back Horizontal Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1750 | MS-10007 | Monoblock | M10473 | Ceramic Cover Foil | EMC shielding | Detaches from surface | Improper material choice; adhesive degrades over time | EMI interference leads to Device Inoperablility | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1-2 | PRD20.6 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1751 | MS-10007 | Monoblock | M10611 | Label: Monoblock SN and Date | Provides information to operator | Detaches from surface | Improper material choice | Service personnel unable to see label information | None | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1752 | MS-10007 | Monoblock | M10611 | Label: Monoblock SN and Date | Provides information to operator | Degrades over time | Improper material choice | Service personnel unable to see label information | None | 1.0 | 3.0 | 3 | None Needed | N/A | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1753 | MS-10007 | Monoblock | M50156 | Heat Shrink, 2.11mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1755 | MS-10007 | Monoblock | M50325 | Thermally Conductive Epoxy, BT-301 | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1756 | MS-10007 | Monoblock | M50550 | Mini mate cable ASSY, 2 pin, Teflon, 20 AWG, 3 in | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1757 | MS-10007 | Monoblock | MS-10505 | MB Thermistor ASSY | Measures monoblock temperature | Thermistor failure | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1758 | MS-10579 | Potted Monoblock | M10015 | Ceramic End Cap Large - Unplated | Electrically insulates while allowing heat to pass through efficiently | Overheat | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1759 | MS-10579 | Potted Monoblock | M10015 | Ceramic End Cap Large - Unplated | Electrically insulates while allowing heat to pass through efficiently | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1760 | MS-10579 | Potted Monoblock | M10452 | Monoblock Shell Bottom Cap | Monoblock enclosure face and EMC shielding | Enclosure Damage | Sudden mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1761 | MS-10579 | Potted Monoblock | M10453 | Monoblock Shell | Monoblock enclosure face and EMC shielding | Enclosure Damage | Sudden mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1762 | MS-10579 | Potted Monoblock | M10454 | Monoblock Shell Top Cap | Monoblock enclosure face and EMC shielding | Enclosure Damage | Sudden mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1763 | MS-10579 | Potted Monoblock | M10455 | Monoblock Transformer Shell | Monoblock enclosure face and EMC shielding | Enclosure Damage | Sudden mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1764 | MS-10579 | Potted Monoblock | M10456 | Monoblock Transformer Cap | Monoblock enclosure face and EMC shielding | Enclosure Damage | Sudden mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1765 | MS-10579 | Potted Monoblock | M10458 | Top Cap Adhesive backed Ultem Sheet | Dielectric Holdoff | Dielectric failure | Improper material choice; dielectric strength | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1766 | MS-10579 | Potted Monoblock | M10458 | Top Cap Adhesive backed Ultem Sheet | Dielectric Holdoff | Dielectric failure | Improper material choice; adhesive degrades over time | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1767 | MS-10579 | Potted Monoblock | M10577 | Transformer shell insulation | Dielectric Holdoff | Dielectric failure | Improper material choice; dielectric strength | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1768 | MS-10579 | Potted Monoblock | M10577 | Transformer shell insulation | Dielectric Holdoff | Dielectric failure | Improper material choice; adhesive degrades over time | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1769 | MS-10579 | Potted Monoblock | M10578 | Monoblock shell bottom cap insulation | Dielectric Holdoff | Dielectric failure | Improper material choice; dielectric strength | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1770 | MS-10579 | Potted Monoblock | M10578 | Monoblock shell bottom cap insulation | Dielectric Holdoff | Dielectric failure | Improper material choice; adhesive degrades over time | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1771 | MS-10579 | Potted Monoblock | M50539 | Board Support Screw Mount Nylon 4.50mm | Seperates parts | Fails to maintain seperation between parts | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1773 | MS-10579 | Potted Monoblock | M50598 | Electrically conductive epoxy, Electro-Bond 06 | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1774 | MS-10579 | Potted Monoblock | M50613 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 3mm long | Joins components | Fails to hold components together | Too short | Damage other components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1775 | MS-10579 | Potted Monoblock | M50613 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 3mm long | Joins components | Fails to hold components together | Too long | Damage other components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1776 | MS-10579 | Potted Monoblock | M50613 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 3mm long | Joins components | Fails to hold components together | Screws loosen over time | Damage other components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1777 | MS-10579 | Potted Monoblock | M50613 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 3mm long | Joins components | Fails to hold components together | No thread locking | Damage other components; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1781 | MS-10579 | Potted Monoblock | M51168 | Lead Solder | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1783 | MS-10579 | Potted Monoblock | MS-10623 | Monoblock, Tube ASSY | Generates X-rays | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1784 | MS-10579 | Potted Monoblock | MS-10623 | Monoblock, Tube ASSY | Generates X-rays | Braze Failure | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2264 | MS-10579 | Potted Monoblock | M51179 | SYLGARD 170 Silicone Elastomer Part A 22.6KG | Mix with M51180 to create dielectric material of monoblock | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2265 | MS-10579 | Potted Monoblock | M51180 | SYLGARD 170 Silicone Elastomer Part B 22.6KG | Mix with M51179 to create dielectric material of monoblock | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2266 | MS-10579 | Potted Monoblock | M51219 | DOWSIL OS-20 Fluid | Cleans and primes parts for potting | Fails to bond parts together | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2267 | MS-10579 | Potted Monoblock | M51220 | Lead Solder, Amerway CWSN63WRAP3, Diameter .032" | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2268 | MS-10579 | Potted Monoblock | M51221 | DOWSIL P5200 Adhesion Promoter Red | Primes parts for potting | Fails to bond parts together | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 2.0 | 8 | None Needed | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK2269 | MS-10579 | Potted Monoblock | M51222 | Solder, Kester Solder, 24-7068-7603, 0.02" | Make Electrical Bridge Between components | Structural failure under weight or load | Material Choice | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2270 | MS-10579 | Potted Monoblock | M51222 | Solder, Kester Solder, 24-7068-7603, 0.02" | Make Electrical Bridge Between components | Structural failure under weight or load | Structural failure due to fatigue | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2271 | MS-10579 | Potted Monoblock | M51222 | Solder, Kester Solder, 24-7068-7603, 0.02" | Make Electrical Bridge Between components | Structural failure under weight or load | Part degrades from aging | Product Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1785 | MS-10623 | Monoblock, Tube ASSY | M10010 | COPPER HEAT SLUG | Expels Heat | Overheat | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1786 | MS-10623 | Monoblock, Tube ASSY | M50595 | Hook-up Wire 20AWG 1C SOLID | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1787 | MS-10623 | Monoblock, Tube ASSY | M50596 | Smooth Edge Lug Terminal Through Hole Connector | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1788 | MS-10623 | Monoblock, Tube ASSY | M50599 | Silver conductive grease | Gap filler for heat transfer | Overheat | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1789 | MS-10623 | Monoblock, Tube ASSY | M50612 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 18mm Long | Joins components | Fails to hold components together | Too short | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1790 | MS-10623 | Monoblock, Tube ASSY | M50612 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 18mm Long | Joins components | Fails to hold components together | Too long | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1791 | MS-10623 | Monoblock, Tube ASSY | M50612 | 316 SS Button Head Hex Drive Screw, M3 x 0.5mm, 18mm Long | Joins components | Fails to hold components together | Screws loosen over time | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1792 | MS-10623 | Monoblock, Tube ASSY | MS-10393 | 80kV Xray Potted Grenade | Generates X-rays but contains leakage radiation | M11178 | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1793 | MS-10623 | Monoblock, Tube ASSY | MS-10393 | 80kV Xray Potted Grenade | Generates X-rays but contains leakage radiation | Braze Failure | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1794 | MS-11235 | Monoblock, Power Assembly | MS-10477 | Monoblock,TXF, Exxelia Assy | Generates high voltage | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1795 | MS-11235 | Monoblock, Power Assembly | MS-10622 | Monoblock, High Voltage ASSY | Generates high voltage | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1796 | MS-11235 | Monoblock, Power Assembly | M51168 | Lead Solder | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1797 | MS-10477 | Monoblock,TXF, Exxelia Assy | E51446 | Fixed Ind 4.7UH 10.5A 11.5 Mohm | Part of the transformer circuit | Component failure | Individual component failure (open/shorts/etc) | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1798 | MS-10477 | Monoblock,TXF, Exxelia Assy | ES-10047 | P01 Monoblock TXFR | PCBA passes power from Dc-Dc boost and into transforner | PCB failure | Individual component failure (open/shorts/etc) | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1799 | MS-10477 | Monoblock,TXF, Exxelia Assy | M10327 | Monoblock AlN Inductor Heat Shunt | Expels Heat | Overheat | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1801 | MS-10477 | Monoblock,TXF, Exxelia Assy | M50546 | Thermally Conductive Epoxy, Thermo-Bond 24 | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1802 | MS-10477 | Monoblock,TXF, Exxelia Assy | MS-10269 | Exxelia Transformer | Steps up voltage | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1804 | MS-10477 | Monoblock,TXF, Exxelia Assy | M51168 | Lead Solder | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1805 | MS-10622 | Monoblock, High Voltage ASSY | ES-10025 | Dean Tech VM Positive | Steps up voltage | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1806 | MS-10622 | Monoblock, High Voltage ASSY | ES-10026 | Dean Tech VM Negative | Steps down voltage | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1807 | MS-10622 | Monoblock, High Voltage ASSY | ES-10046 | P01 Monoblock HV | PCBA passes signals outside monoblock and contains high voltage componenets | PCB failure | Individual component failure (open/shorts/etc) | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1808 | MS-10622 | Monoblock, High Voltage ASSY | M50539 | Board Support Screw Mount Nylon 4.50mm | Seperates parts | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1809 | MS-10622 | Monoblock, High Voltage ASSY | M50594 | 18 AWG Hook-Up Wire 19/30 White 25kV | Highly insulated electrical connection | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1810 | MS-10622 | Monoblock, High Voltage ASSY | M50595 | Hook-up Wire 20AWG 1C SOLID | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1811 | MS-10622 | Monoblock, High Voltage ASSY | MS-10423 | Filament transformer 3:2 Turns | Electrically isolates | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1812 | MS-10622 | Monoblock, High Voltage ASSY | MS-10424 | Filament transformer, 4:3 Turns | Electrically isolates | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2272 | MS-10622 | Monoblock, High Voltage ASSY | M51220 | Lead Solder, Diameter .032" | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2273 | MS-10622 | Monoblock, High Voltage ASSY | M51279 | DP-100 | Seals backside of 10 pin connector | Fails to seal | Improper Assembly | Device Inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Verification of build during EOL Testing | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2274 | MS-10622 | Monoblock, High Voltage ASSY | M51280 | Micro Bead Resin FIller | Promote adhesion between DP-100 and Sylguard | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1814 | MS-10622 | Monoblock, High Voltage ASSY | M51168 | Lead Solder | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1813 | MS-10622 | Monoblock, High Voltage ASSY | M50238 | Solder, Kester Solder, 90-7482-3320, 0.02" | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1815 | MS-10423 | Filament transformer 3:2 Turns | M50509 | FERRITE CORE TOROID 5.5UH T38 | Provides Isolation | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1816 | MS-10423 | Filament transformer 3:2 Turns | M50510 | 28 AWG Wire 19/40 18kV Clear | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1817 | MS-10423 | Filament transformer 3:2 Turns | M50511 | 28 AWG Wire 19/40 18kV Black | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1818 | MS-10423 | Filament transformer 3:2 Turns | M50618 | 832HD Epoxy Potting Compound | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1819 | MS-10424 | Filament transformer, 4:3 Turns | M50509 | FERRITE CORE TOROID 5.5UH T38 | Provides Isolation | Dielectric failure | Improper specifications | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1820 | MS-10424 | Filament transformer, 4:3 Turns | M50511 | 28 AWG Wire 19/40 18kV Black | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1821 | MS-10424 | Filament transformer, 4:3 Turns | M50512 | 28 AWG Wire 19/40 18kV White | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1822 | MS-10424 | Filament transformer, 4:3 Turns | M50618 | 832HD Epoxy Potting Compound | Bonds surfaces | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1823 | MS-10505 | MB Closed Loop Thermistor Assembly | M50156 | Heat Shrink, 2.11mm ID supplied, 2:1 Shrink | Provides insulation | Detaches from surface | Improper material/size choice | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK2275 | MS-10505 | MB Closed Loop Thermistor Assembly | M51222 | Solder, Kester Solder, 24-7068-7603, 0.02" | Joins components electrically | Fails to hold components together | Sudden disconnect via mechanical damage | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1825 | MS-10505 | MB Closed Loop Thermistor Assembly | M50324 | Thermistor, NRL1104F3380B1F | Measures monoblock temperature | Thermistor failure | Mechanical damage from external forces | Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1826 | MS-10505 | MB Closed Loop Thermistor Assembly | M50682 | Cable-ASSY, Pico-Lock 1.0mm, 2 Circuit, 150mm | Connects electrical components | Unable to carry proper current and voltage through pin | Improper specification per Electrical requirements - underspecified | Monblock failure; Product inoperable | Delay of Procedure | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1827 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11170 | Charging Circuit Heat Block Bracket | Retains heat pipe | Mechanical Damage | Mechanical damage from external forces | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1828 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11151 | Charging Circuit Heat Block Heat Pipe | Transfers heat | Heat pipe looses internal fluid | Mechanical damage from external forces | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1829 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M50881 | 18-8 Stainless Steel Hex Flat Head ScrewsM3 x 0.50 mm Thread Size, 4 mm Long | Joins components | Fails to hold components together | Too short | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1830 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M50881 | 18-8 Stainless Steel Hex Flat Head ScrewsM3 x 0.50 mm Thread Size, 4 mm Long | Joins components | Fails to hold components together | Too long | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1831 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M50881 | 18-8 Stainless Steel Hex Flat Head ScrewsM3 x 0.50 mm Thread Size, 4 mm Long | Joins components | Fails to hold components together | Screws loosen over time | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1832 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11150 | Charging Circuit Heat Block | Retains heat pipe | Mechanical Damage | Mechanical damage from external forces | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1833 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11171 | Charging Inductor TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1834 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11172 | Switching MOSFET TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1835 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11173 | Cassette MOSFET TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1836 | MS-11179 | Cassette Main Charging Heat Block Assembly with Heat Pipe | M11178 | Cassette Main Charging IC TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1837 | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | M11147 | 5V0 Regulator Heat Block Heat Pipe | Transfers heat | Heat pipe looses internal fluid | Mechanical damage from external forces | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1838 | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | M11146 | 5V0 Regulator Heat Block | Retains heat pipe | Mechanical Damage | Mechanical damage from external forces | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1839 | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | M11154 | 5V0 Regulator Heat Block Large Inductor TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1840 | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | M11155 | 5V0 Regulator Heat Block DC DC TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1841 | MS-11180 | 5V0 Regulator Heat Block Assembly with Heat Pipe | M11156 | 5V0 Regulaor Heat Block Small Inductor TIM | Gap filler for heat transfer | Overheat | Improper specifications | Reduced performance; charging rate throttled | Operator Dissatisfaction | 1.0 | 3.0 | 3 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 1.0 | 3.0 | 3 |
| DRSK1842 | F1 | Foot Pedal | MS-50006 | Wireless Footpedal GP211 (without PCBA) | Protects interior of device | Enclosure Damaged - large | Mechanical damage from external forces | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 3.0 | 12 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1843 | F1 | Foot Pedal | MS-50006 | Wireless Footpedal GP211 (without PCBA) | Protects interior of device | Enclosure Damaged - small | Mechanical damage from external forces | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1844 | F1 | Foot Pedal | MS-50006 | Wireless Footpedal GP211 (without PCBA) | Protects interior of device | Operator/Patient skin reaction | Non-biocompatible material choice | No effect | Potential for skin irritation | 4.0 | 3.0 | 12 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1845 | F1 | Foot Pedal | MS-50006 | Wireless Footpedal GP211 (without PCBA) | Protects interior of device | Fails to maintain clearances | Improper specifications | Basic safety compromised; still operable | Minor Electrical shock | 4.0 | 2.0 | 8 | Compliance to IEC 60601-1 | PRD20.5 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1846 | F1 | Foot Pedal | MS-50006 | Wireless Footpedal GP211 (without PCBA) | Protects interior of device | Structural integrity compromized | Incorrect material choice | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Polycarbonate material choice | RSK_R169 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1847 | F1 | Foot Pedal | MS-50006 | Wireless Footpedal GP211 (without PCBA) | Protects interior of device | Structural integrity compromized | Improper geometry | Device inoperable - ingress | Delay of Procedure | 4.0 | 2.0 | 8 | Incoming Inspection | QSP-014 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1848 | F1 | Foot Pedal | M10165 | Label: Foot Pedal Ra | Provides information to operator | Degrades over time | Material Choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1849 | F1 | Foot Pedal | M10165 | Label: Foot Pedal Ra | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1850 | F1 | Foot Pedal | M10165 | Label: Foot Pedal Ra | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Usability Test | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1851 | F1 | Foot Pedal | M10168 | Label: Foot Pedal Emit Button | Provides information to operator | Degrades over time | Material Choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1852 | F1 | Foot Pedal | M10168 | Label: Foot Pedal Emit Button | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1853 | F1 | Foot Pedal | M10168 | Label: Foot Pedal Emit Button | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Usability Test | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1854 | F1 | Foot Pedal | M10169 | Label: Foot Pedal Save | Provides information to operator | Degrades over time | Material Choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1855 | F1 | Foot Pedal | M10169 | Label: Foot Pedal Save | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1856 | F1 | Foot Pedal | M10169 | Label: Foot Pedal Save | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Usability Test | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1856 | F1 | Foot Pedal | ES-10007 | Footpedal PCBA | Triggers E1 to take an image | Fails to trigger E1 to take an image | Individual component failure (open/shorts/etc) | Unable to take image via F1 | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10176 | Label: Foot Pedal Packing | Provides information to operator | Degrades over time | Material Choice | Operator unable to see label information | Minor Dissatisfaction | 1.0 | 2.0 | 2 | None Needed | N/A | No further planned remediation | 1.0 | 2.0 | 2 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10176 | Label: Foot Pedal Packing | Provides information to operator | Illegible | Improper font choice/size | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Comply to IEC 60601-2-54 | PRD20.11 | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10176 | Label: Foot Pedal Packing | Provides information to operator | Illegible | Improper color choice | Operator unable to see label information | Delay of Procedure | 4.0 | 2.0 | 8 | Usability Test | N/A | No further planned remediation | 4.0 | 2.0 | 8 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10345 | F1 Shipping Box | Protect outer shell of F1 from scratches and damage during shipping | Failure to allow proper spacing | Improper specifications - too small | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10345 | F1 Shipping Box | Protect outer shell of F1 from scratches and damage during shipping | Failure to allow proper spacing | Improper specifications - too large | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10345 | F1 Shipping Box | Protect outer shell of F1 from scratches and damage during shipping | Failure to maintain integrity of product | Improper material choice | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10345 | F1 Shipping Box | Protect outer shell of F1 from scratches and damage during shipping | Failure to maintain integrity of product | Improper material choice | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | ISTA Testing | PRD20.26 | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10345 | F1 Shipping Box | Protect outer shell of F1 from scratches and damage during shipping | Allows ingress | Improper seal | Potential damage to product internal | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |
| DRSK1856 | MS-10344 | F1 Shipping Box Assembly | M10345 | F1 Shipping Box | Protect outer shell of F1 from scratches and damage during shipping | Fails to close and lock | Improper locking mechanism | Potential damage to product external | Delay of Procedure | 4.0 | 3.0 | 12 | None Needed | N/A | No further planned remediation | 4.0 | 3.0 | 12 |

### Table 3
|  | 0 |
| --- | --- |
| 2.0 | DRSK0001 |
| 3.0 | DRSK1849 |
| 4.0 | DRSK1850 |
| 5.0 | DRSK1851 |
| 6.0 | DRSK1852 |
| 7.0 | DRSK0002 |
| 8.0 | DRSK1853 |
| 9.0 | DRSK1854 |
| 10.0 | DRSK0003 |
| 11.0 | DRSK1855 |
| 12.0 | DRSK1856 |
| 13.0 | DRSK1857 |
| 14.0 | DRSK1858 |
| 15.0 | DRSK1859 |
| 16.0 | DRSK0004 |
| 17.0 | DRSK0005 |
| 18.0 | DRSK0006 |
| 19.0 | DRSK0007 |
| 20.0 | DRSK0008 |
| 21.0 | DRSK0009 |
| 22.0 | DRSK0010 |
| 23.0 | DRSK0011 |
| 24.0 | DRSK0012 |
| 25.0 | DRSK0013 |
| 26.0 | DRSK1860 |
| 27.0 | DRSK1861 |
| 28.0 | DRSK1862 |
| 29.0 | DRSK0014 |
| 30.0 | DRSK0015 |
| 31.0 | DRSK0016 |
| 32.0 | DRSK0017 |
| 33.0 | DRSK0018 |
| 34.0 | DRSK0019 |
| 35.0 | DRSK0020 |
| 36.0 | DRSK0021 |
| 37.0 | DRSK0022 |
| 38.0 | DRSK0023 |
| 39.0 | DRSK0024 |
| 40.0 | DRSK0025 |
| 41.0 | DRSK0026 |
| 42.0 | DRSK0027 |
| 43.0 | DRSK0028 |
| 44.0 | DRSK0029 |
| 45.0 | DRSK0030 |
| 46.0 | DRSK0031 |
| 47.0 | DRSK0032 |
| 48.0 | DRSK0033 |
| 49.0 | DRSK0034 |
| 50.0 | DRSK0035 |
| 51.0 | DRSK0036 |
| 52.0 | DRSK0037 |
| 53.0 | DRSK0038 |
| 54.0 | DRSK0039 |
| 55.0 | DRSK0040 |
| 56.0 | DRSK0041 |
| 57.0 | DRSK0042 |
| 58.0 | DRSK0043 |
| 59.0 | DRSK0044 |
| 60.0 | DRSK0045 |
| 61.0 | DRSK0046 |
| 62.0 | DRSK0047 |
| 63.0 | DRSK0048 |
| 64.0 | DRSK0049 |
| 65.0 | DRSK0050 |
| 66.0 | DRSK0051 |
| 67.0 | DRSK0052 |
| 68.0 | DRSK0053 |
| 69.0 | DRSK0054 |
| 70.0 | DRSK0055 |
| 71.0 | DRSK0056 |
| 72.0 | DRSK0057 |
| 73.0 | DRSK0058 |
| 74.0 | DRSK0059 |
| 75.0 | DRSK0060 |
| 76.0 | DRSK0061 |
| 77.0 | DRSK0062 |
| 78.0 | DRSK0063 |
| 79.0 | DRSK0064 |
| 80.0 | DRSK0065 |
| 81.0 | DRSK0066 |
| 82.0 | DRSK0067 |
| 83.0 | DRSK0068 |
| 84.0 | DRSK0069 |
| 85.0 | DRSK0070 |
| 86.0 | DRSK0071 |
| 87.0 | DRSK0072 |
| 88.0 | DRSK0073 |
| 89.0 | DRSK0074 |
| 90.0 | DRSK0075 |
| 91.0 | DRSK0076 |
| 92.0 | DRSK0077 |
| 93.0 | DRSK0078 |
| 94.0 | DRSK0079 |
| 95.0 | DRSK0080 |
| 96.0 | DRSK0081 |
| 97.0 | DRSK0082 |
| 98.0 | DRSK0083 |
| 99.0 | DRSK0084 |
| 100.0 | DRSK0085 |
| 101.0 | DRSK0086 |
| 102.0 | DRSK0087 |
| 103.0 | DRSK0088 |
| 104.0 | DRSK0089 |
| 105.0 | DRSK0090 |
| 106.0 | DRSK0091 |
| 107.0 | DRSK0092 |
| 108.0 | DRSK0093 |
| 109.0 | DRSK0094 |
| 110.0 | DRSK0095 |
| 111.0 | DRSK0096 |
| 112.0 | DRSK0097 |
| 113.0 | DRSK0098 |
| 114.0 | DRSK0099 |
| 115.0 | DRSK0100 |
| 116.0 | DRSK0101 |
| 117.0 | DRSK0102 |
| 118.0 | DRSK0103 |
| 119.0 | DRSK0104 |
| 120.0 | DRSK0105 |
| 121.0 | DRSK0106 |
| 122.0 | DRSK0107 |
| 123.0 | DRSK1456 |
| 124.0 | DRSK1457 |
| 125.0 | DRSK0108 |
| 126.0 | DRSK0109 |
| 127.0 | DRSK0110 |
| 128.0 | DRSK0111 |
| 129.0 | DRSK0112 |
| 130.0 | DRSK0113 |
| 131.0 | DRSK0114 |
| 132.0 | DRSK0115 |
| 133.0 | DRSK0116 |
| 134.0 | DRSK0117 |
| 135.0 | DRSK0118 |
| 136.0 | DRSK0119 |
| 137.0 | DRSK0120 |
| 138.0 | DRSK0121 |
| 139.0 | DRSK0122 |
| 140.0 | DRSK0123 |
| 141.0 | DRSK0124 |
| 142.0 | DRSK0125 |
| 143.0 | DRSK0126 |
| 144.0 | DRSK0127 |
| 145.0 | DRSK0128 |
| 146.0 | DRSK0129 |
| 147.0 | DRSK0130 |
| 148.0 | DRSK0131 |
| 149.0 | DRSK0132 |
| 150.0 | DRSK0133 |
| 151.0 | DRSK0134 |
| 152.0 | DRSK0135 |
| 153.0 | DRSK0136 |
| 154.0 | DRSK0137 |
| 155.0 | DRSK0138 |
| 156.0 | DRSK0139 |
| 157.0 | DRSK0140 |
| 158.0 | DRSK0141 |
| 159.0 | DRSK0142 |
| 160.0 | DRSK0143 |
| 161.0 | DRSK0144 |
| 162.0 | DRSK0145 |
| 163.0 | DRSK0146 |
| 164.0 | DRSK0147 |
| 165.0 | DRSK0148 |
| 166.0 | DRSK0149 |
| 167.0 | DRSK0150 |
| 168.0 | DRSK0151 |
| 169.0 | DRSK0152 |
| 170.0 | DRSK0153 |
| 171.0 | DRSK0154 |
| 172.0 | DRSK0155 |
| 173.0 | DRSK0156 |
| 174.0 | DRSK0157 |
| 175.0 | DRSK0158 |
| 176.0 | DRSK0159 |
| 177.0 | DRSK0160 |
| 178.0 | DRSK0161 |
| 179.0 | DRSK0162 |
| 180.0 | DRSK0163 |
| 181.0 | DRSK0164 |
| 182.0 | DRSK0165 |
| 183.0 | DRSK0166 |
| 184.0 | DRSK0167 |
| 185.0 | DRSK0168 |
| 186.0 | DRSK0169 |
| 187.0 | DRSK0170 |
| 188.0 | DRSK0171 |
| 189.0 | DRSK0172 |
| 190.0 | DRSK0173 |
| 191.0 | DRSK0174 |
| 192.0 | DRSK0175 |
| 193.0 | DRSK0176 |
| 194.0 | DRSK0177 |
| 195.0 | DRSK0178 |
| 196.0 | DRSK0179 |
| 197.0 | DRSK0180 |
| 198.0 | DRSK0181 |
| 199.0 | DRSK0182 |
| 200.0 | DRSK0183 |
| 201.0 | DRSK0184 |
| 202.0 | DRSK0185 |
| 203.0 | DRSK0186 |
| 204.0 | DRSK0187 |
| 205.0 | DRSK0188 |
| 206.0 | DRSK0189 |
| 207.0 | DRSK0190 |
| 208.0 | DRSK0191 |
| 209.0 | DRSK0192 |
| 210.0 | DRSK0193 |
| 211.0 | DRSK0194 |
| 212.0 | DRSK0195 |
| 213.0 | DRSK0196 |
| 214.0 | DRSK0197 |
| 215.0 | DRSK0198 |
| 216.0 | DRSK0199 |
| 217.0 | DRSK0200 |
| 218.0 | DRSK0201 |
| 219.0 | DRSK0202 |
| 220.0 | DRSK0203 |
| 221.0 | DRSK0204 |
| 222.0 | DRSK0205 |
| 223.0 | DRSK0206 |
| 224.0 | DRSK0207 |
| 225.0 | DRSK0208 |
| 226.0 | DRSK0209 |
| 227.0 | DRSK1712 |
| 228.0 | DRSK0210 |
| 229.0 | DRSK0211 |
| 230.0 | DRSK0212 |
| 231.0 | DRSK0213 |
| 232.0 | DRSK0214 |
| 233.0 | DRSK0215 |
| 234.0 | DRSK0216 |
| 235.0 | DRSK0217 |
| 236.0 | DRSK0218 |
| 237.0 | DRSK0219 |
| 238.0 | DRSK0220 |
| 239.0 | DRSK0221 |
| 240.0 | DRSK0222 |
| 241.0 | DRSK0223 |
| 242.0 | DRSK0224 |
| 243.0 | DRSK0225 |
| 244.0 | DRSK0226 |
| 245.0 | DRSK0227 |
| 246.0 | DRSK0228 |
| 247.0 | DRSK0229 |
| 248.0 | DRSK0230 |
| 249.0 | DRSK0231 |
| 250.0 | DRSK0232 |
| 251.0 | DRSK0233 |
| 252.0 | DRSK0234 |
| 253.0 | DRSK0235 |
| 254.0 | DRSK0236 |
| 255.0 | DRSK0237 |
| 256.0 | DRSK0238 |
| 257.0 | DRSK0239 |
| 258.0 | DRSK0240 |
| 259.0 | DRSK0241 |
| 260.0 | DRSK0242 |
| 261.0 | DRSK0243 |
| 262.0 | DRSK0244 |
| 263.0 | DRSK0245 |
| 264.0 | DRSK0246 |
| 265.0 | DRSK0247 |
| 266.0 | DRSK0248 |
| 267.0 | DRSK0249 |
| 268.0 | DRSK1713 |
| 269.0 | DRSK1714 |
| 270.0 | DRSK1715 |
| 271.0 | DRSK1716 |
| 272.0 | DRSK1863 |
| 273.0 | DRSK1864 |
| 274.0 | DRSK1865 |
| 275.0 | DRSK1866 |
| 276.0 | DRSK1867 |
| 277.0 | DRSK1868 |
| 278.0 | DRSK1869 |
| 279.0 | DRSK1870 |
| 280.0 | DRSK1871 |
| 281.0 | DRSK1872 |
| 282.0 | DRSK1873 |
| 283.0 | DRSK1874 |
| 284.0 | DRSK1875 |
| 285.0 | DRSK1876 |
| 286.0 | DRSK1877 |
| 287.0 | DRSK1878 |
| 288.0 | DRSK1879 |
| 289.0 | DRSK1880 |
| 290.0 | DRSK1881 |
| 291.0 | DRSK1882 |
| 292.0 | DRSK1883 |
| 293.0 | DRSK1884 |
| 294.0 | DRSK1885 |
| 295.0 | DRSK1886 |
| 296.0 | DRSK1887 |
| 297.0 | DRSK1888 |
| 298.0 | DRSK1889 |
| 299.0 | DRSK0250 |
| 300.0 | DRSK0251 |
| 301.0 | DRSK0252 |
| 302.0 | DRSK0253 |
| 303.0 | DRSK0254 |
| 304.0 | DRSK0255 |
| 305.0 | DRSK0256 |
| 306.0 | DRSK0257 |
| 307.0 | DRSK0258 |
| 308.0 | DRSK0259 |
| 309.0 | DRSK0260 |
| 310.0 | DRSK0261 |
| 311.0 | DRSK0262 |
| 312.0 | DRSK0263 |
| 313.0 | DRSK0264 |
| 314.0 | DRSK0265 |
| 315.0 | DRSK0266 |
| 316.0 | DRSK0267 |
| 317.0 | DRSK0268 |
| 318.0 | DRSK0269 |
| 319.0 | DRSK0270 |
| 320.0 | DRSK0271 |
| 321.0 | DRSK0272 |
| 322.0 | DRSK0273 |
| 323.0 | DRSK0274 |
| 324.0 | DRSK0275 |
| 325.0 | DRSK0276 |
| 326.0 | DRSK0277 |
| 327.0 | DRSK0278 |
| 328.0 | DRSK0279 |
| 329.0 | DRSK0280 |
| 330.0 | DRSK0281 |
| 331.0 | DRSK0282 |
| 332.0 | DRSK0283 |
| 333.0 | DRSK0284 |
| 334.0 | DRSK0285 |
| 335.0 | DRSK0286 |
| 336.0 | DRSK0287 |
| 337.0 | DRSK0288 |
| 338.0 | DRSK0289 |
| 339.0 | DRSK0290 |
| 340.0 | DRSK0291 |
| 341.0 | DRSK0292 |
| 342.0 | DRSK0293 |
| 343.0 | DRSK0294 |
| 344.0 | DRSK0295 |
| 345.0 | DRSK0296 |
| 346.0 | DRSK0297 |
| 347.0 | DRSK0298 |
| 348.0 | DRSK0299 |
| 349.0 | DRSK0300 |
| 350.0 | DRSK0301 |
| 351.0 | DRSK0302 |
| 352.0 | DRSK0303 |
| 353.0 | DRSK0304 |
| 354.0 | DRSK0305 |
| 355.0 | DRSK0306 |
| 356.0 | DRSK0307 |
| 357.0 | DRSK0308 |
| 358.0 | DRSK0309 |
| 359.0 | DRSK0310 |
| 360.0 | DRSK0311 |
| 361.0 | DRSK0312 |
| 362.0 | DRSK0313 |
| 363.0 | DRSK0314 |
| 364.0 | DRSK0315 |
| 365.0 | DRSK0316 |
| 366.0 | DRSK0317 |
| 367.0 | DRSK0318 |
| 368.0 | DRSK0319 |
| 369.0 | DRSK0320 |
| 370.0 | DRSK0321 |
| 371.0 | DRSK0322 |
| 372.0 | DRSK0323 |
| 373.0 | DRSK0324 |
| 374.0 | DRSK0325 |
| 375.0 | DRSK0326 |
| 376.0 | DRSK0327 |
| 377.0 | DRSK0328 |
| 378.0 | DRSK0329 |
| 379.0 | DRSK0330 |
| 380.0 | DRSK0331 |
| 381.0 | DRSK0332 |
| 382.0 | DRSK1458 |
| 383.0 | DRSK0333 |
| 384.0 | DRSK1459 |
| 385.0 | DRSK0334 |
| 386.0 | DRSK0335 |
| 387.0 | DRSK0336 |
| 388.0 | DRSK0337 |
| 389.0 | DRSK0338 |
| 390.0 | DRSK0339 |
| 391.0 | DRSK0340 |
| 392.0 | DRSK0341 |
| 393.0 | DRSK0342 |
| 394.0 | DRSK0343 |
| 395.0 | DRSK0344 |
| 396.0 | DRSK0345 |
| 397.0 | DRSK0346 |
| 398.0 | DRSK0347 |
| 399.0 | DRSK0348 |
| 400.0 | DRSK0349 |
| 401.0 | DRSK0350 |
| 402.0 | DRSK0351 |
| 403.0 | DRSK1460 |
| 404.0 | DRSK0352 |
| 405.0 | DRSK1461 |
| 406.0 | DRSK0353 |
| 407.0 | DRSK0354 |
| 408.0 | DRSK0355 |
| 409.0 | DRSK0356 |
| 410.0 | DRSK0357 |
| 411.0 | DRSK0358 |
| 412.0 | DRSK0359 |
| 413.0 | DRSK0360 |
| 414.0 | DRSK0361 |
| 415.0 | DRSK0362 |
| 416.0 | DRSK0363 |
| 417.0 | DRSK0364 |
| 418.0 | DRSK0365 |
| 419.0 | DRSK0366 |
| 420.0 | DRSK0367 |
| 421.0 | DRSK0368 |
| 422.0 | DRSK0369 |
| 423.0 | DRSK0370 |
| 424.0 | DRSK1462 |
| 425.0 | DRSK0371 |
| 426.0 | DRSK1463 |
| 427.0 | DRSK0372 |
| 428.0 | DRSK0373 |
| 429.0 | DRSK0374 |
| 430.0 | DRSK0375 |
| 431.0 | DRSK0376 |
| 432.0 | DRSK0377 |
| 433.0 | DRSK0378 |
| 434.0 | DRSK0379 |
| 435.0 | DRSK0380 |
| 436.0 | DRSK0381 |
| 437.0 | DRSK0382 |
| 438.0 | DRSK0383 |
| 439.0 | DRSK0384 |
| 440.0 | DRSK0385 |
| 441.0 | DRSK0386 |
| 442.0 | DRSK0387 |
| 443.0 | DRSK0388 |
| 444.0 | DRSK0389 |
| 445.0 | DRSK0390 |
| 446.0 | DRSK0391 |
| 447.0 | DRSK0392 |
| 448.0 | DRSK0393 |
| 449.0 | DRSK0394 |
| 450.0 | DRSK0395 |
| 451.0 | DRSK0396 |
| 452.0 | DRSK0397 |
| 453.0 | DRSK0398 |
| 454.0 | DRSK0399 |
| 455.0 | DRSK0400 |
| 456.0 | DRSK0401 |
| 457.0 | DRSK0402 |
| 458.0 | DRSK0403 |
| 459.0 | DRSK0404 |
| 460.0 | DRSK0405 |
| 461.0 | DRSK0406 |
| 462.0 | DRSK0407 |
| 463.0 | DRSK0408 |
| 464.0 | DRSK0409 |
| 465.0 | DRSK0410 |
| 466.0 | DRSK0411 |
| 467.0 | DRSK0412 |
| 468.0 | DRSK0413 |
| 469.0 | DRSK0414 |
| 470.0 | DRSK0415 |
| 471.0 | DRSK0416 |
| 472.0 | DRSK0417 |
| 473.0 | DRSK0418 |
| 474.0 | DRSK0419 |
| 475.0 | DRSK0420 |
| 476.0 | DRSK0421 |
| 477.0 | DRSK0422 |
| 478.0 | DRSK0423 |
| 479.0 | DRSK0424 |
| 480.0 | DRSK0425 |
| 481.0 | DRSK0426 |
| 482.0 | DRSK0427 |
| 483.0 | DRSK0428 |
| 484.0 | DRSK1842 |
| 485.0 | DRSK1843 |
| 486.0 | DRSK1844 |
| 487.0 | DRSK1845 |
| 488.0 | DRSK1846 |
| 489.0 | DRSK1847 |
| 490.0 | DRSK1890 |
| 491.0 | DRSK1891 |
| 492.0 | DRSK1892 |
| 493.0 | DRSK1893 |
| 494.0 | DRSK1894 |
| 495.0 | DRSK1895 |
| 496.0 | DRSK1896 |
| 497.0 | DRSK1897 |
| 498.0 | DRSK1898 |
| 499.0 | DRSK1899 |
| 500.0 | DRSK1900 |
| 501.0 | DRSK1901 |
| 502.0 | DRSK1902 |
| 503.0 | DRSK1903 |
| 504.0 | DRSK1904 |
| 505.0 | DRSK1905 |
| 506.0 | DRSK1906 |
| 507.0 | DRSK1907 |
| 508.0 | DRSK1908 |
| 509.0 | DRSK1909 |
| 510.0 | DRSK1910 |
| 511.0 | DRSK1911 |
| 512.0 | DRSK1912 |
| 513.0 | DRSK1913 |
| 514.0 | DRSK1914 |
| 515.0 | DRSK1915 |
| 516.0 | DRSK1916 |
| 517.0 | DRSK1917 |
| 518.0 | DRSK1918 |
| 519.0 | DRSK1919 |
| 520.0 | DRSK1920 |
| 521.0 | DRSK1921 |
| 522.0 | DRSK1922 |
| 523.0 | DRSK1923 |
| 524.0 | DRSK1924 |
| 525.0 | DRSK1925 |
| 526.0 | DRSK1926 |
| 527.0 | DRSK1927 |
| 528.0 | DRSK1928 |
| 529.0 | DRSK1929 |
| 530.0 | DRSK1930 |
| 531.0 | DRSK1931 |
| 532.0 | DRSK1932 |
| 533.0 | DRSK1933 |
| 534.0 | DRSK1934 |
| 535.0 | DRSK1935 |
| 536.0 | DRSK1936 |
| 537.0 | DRSK1937 |
| 538.0 | DRSK1938 |
| 539.0 | DRSK1939 |
| 540.0 | DRSK1940 |
| 541.0 | DRSK1941 |
| 542.0 | DRSK1942 |
| 543.0 | DRSK1943 |
| 544.0 | DRSK1944 |
| 545.0 | DRSK1945 |
| 546.0 | DRSK1946 |
| 547.0 | DRSK1947 |
| 548.0 | DRSK1948 |
| 549.0 | DRSK1949 |
| 550.0 | DRSK1950 |
| 551.0 | DRSK1951 |
| 552.0 | DRSK1952 |
| 553.0 | DRSK1953 |
| 554.0 | DRSK1954 |
| 555.0 | DRSK1955 |
| 556.0 | DRSK1956 |
| 557.0 | DRSK1957 |
| 558.0 | DRSK1958 |
| 559.0 | DRSK1959 |
| 560.0 | DRSK1960 |
| 561.0 | DRSK1961 |
| 562.0 | DRSK1962 |
| 563.0 | DRSK1963 |
| 564.0 | DRSK1964 |
| 565.0 | DRSK1965 |
| 566.0 | DRSK1966 |
| 567.0 | DRSK1967 |
| 568.0 | DRSK1968 |
| 569.0 | DRSK1969 |
| 570.0 | DRSK1970 |
| 571.0 | DRSK1971 |
| 572.0 | DRSK1972 |
| 573.0 | DRSK1973 |
| 574.0 | DRSK1974 |
| 575.0 | DRSK1975 |
| 576.0 | DRSK1976 |
| 577.0 | DRSK1977 |
| 578.0 | DRSK1978 |
| 579.0 | DRSK1979 |
| 580.0 | DRSK1980 |
| 581.0 | DRSK1981 |
| 582.0 | DRSK1982 |
| 583.0 | DRSK1983 |
| 584.0 | DRSK1984 |
| 585.0 | DRSK1985 |
| 586.0 | DRSK1986 |
| 587.0 | DRSK1987 |
| 588.0 | DRSK1988 |
| 589.0 | DRSK1989 |
| 590.0 | DRSK1990 |
| 591.0 | DRSK1991 |
| 592.0 | DRSK1992 |
| 593.0 | DRSK1993 |
| 594.0 | DRSK1994 |
| 595.0 | DRSK1995 |
| 596.0 | DRSK1996 |
| 597.0 | DRSK1997 |
| 598.0 | DRSK1998 |
| 599.0 | DRSK1999 |
| 600.0 | DRSK2000 |
| 601.0 | DRSK2001 |
| 602.0 | DRSK2002 |
| 603.0 | DRSK2003 |
| 604.0 | DRSK2004 |
| 605.0 | DRSK2005 |
| 606.0 | DRSK2006 |
| 607.0 | DRSK2007 |
| 608.0 | DRSK2008 |
| 609.0 | DRSK2009 |
| 610.0 | DRSK2010 |
| 611.0 | DRSK2011 |
| 612.0 | DRSK2012 |
| 613.0 | DRSK2013 |
| 614.0 | DRSK2014 |
| 615.0 | DRSK2015 |
| 616.0 | DRSK2016 |
| 617.0 | DRSK2017 |
| 618.0 | DRSK2018 |
| 619.0 | DRSK2019 |
| 620.0 | DRSK2020 |
| 621.0 | DRSK2021 |
| 622.0 | DRSK2022 |
| 623.0 | DRSK2023 |
| 624.0 | DRSK2024 |
| 625.0 | DRSK2025 |
| 626.0 | DRSK2026 |
| 627.0 | DRSK2027 |
| 628.0 | DRSK2028 |
| 629.0 | DRSK2029 |
| 630.0 | DRSK2030 |
| 631.0 | DRSK2031 |
| 632.0 | DRSK2032 |
| 633.0 | DRSK2033 |
| 634.0 | DRSK2034 |
| 635.0 | DRSK2035 |
| 636.0 | DRSK2036 |
| 637.0 | DRSK2037 |
| 638.0 | DRSK2038 |
| 639.0 | DRSK2039 |
| 640.0 | DRSK2040 |
| 641.0 | DRSK2041 |
| 642.0 | DRSK2042 |
| 643.0 | DRSK2043 |
| 644.0 | DRSK2044 |
| 645.0 | DRSK2045 |
| 646.0 | DRSK2046 |
| 647.0 | DRSK2047 |
| 648.0 | DRSK2048 |
| 649.0 | DRSK2049 |
| 650.0 | DRSK2050 |
| 651.0 | DRSK2051 |
| 652.0 | DRSK2052 |
| 653.0 | DRSK2053 |
| 654.0 | DRSK2054 |
| 655.0 | DRSK2055 |
| 656.0 | DRSK2056 |
| 657.0 | DRSK2057 |
| 658.0 | DRSK2058 |
| 659.0 | DRSK2059 |
| 660.0 | DRSK2060 |
| 661.0 | DRSK2061 |
| 662.0 | DRSK0429 |
| 663.0 | DRSK0430 |
| 664.0 | DRSK0431 |
| 665.0 | DRSK0432 |
| 666.0 | DRSK0433 |
| 667.0 | DRSK2062 |
| 668.0 | DRSK2063 |
| 669.0 | DRSK0434 |
| 670.0 | DRSK2064 |
| 671.0 | DRSK2065 |
| 672.0 | DRSK0435 |
| 673.0 | DRSK0436 |
| 674.0 | DRSK0437 |
| 675.0 | DRSK0438 |
| 676.0 | DRSK0439 |
| 677.0 | DRSK0440 |
| 678.0 | DRSK0441 |
| 679.0 | DRSK0442 |
| 680.0 | DRSK0443 |
| 681.0 | DRSK0444 |
| 682.0 | DRSK0445 |
| 683.0 | DRSK0446 |
| 684.0 | DRSK0447 |
| 685.0 | DRSK0448 |
| 686.0 | DRSK0449 |
| 687.0 | DRSK0450 |
| 688.0 | DRSK0451 |
| 689.0 | DRSK0452 |
| 690.0 | DRSK0453 |
| 691.0 | DRSK0454 |
| 692.0 | DRSK0455 |
| 693.0 | DRSK0456 |
| 694.0 | DRSK0457 |
| 695.0 | DRSK0458 |
| 696.0 | DRSK0459 |
| 697.0 | DRSK0460 |
| 698.0 | DRSK0461 |
| 699.0 | DRSK0465 |
| 700.0 | DRSK0466 |
| 701.0 | DRSK0467 |
| 702.0 | DRSK0468 |
| 703.0 | DRSK0469 |
| 704.0 | DRSK0470 |
| 705.0 | DRSK0471 |
| 706.0 | DRSK0472 |
| 707.0 | DRSK0473 |
| 708.0 | DRSK0474 |
| 709.0 | DRSK0475 |
| 710.0 | DRSK0476 |
| 711.0 | DRSK0477 |
| 712.0 | DRSK0478 |
| 713.0 | DRSK0479 |
| 714.0 | DRSK0480 |
| 715.0 | DRSK0481 |
| 716.0 | DRSK0482 |
| 717.0 | DRSK0483 |
| 718.0 | DRSK0484 |
| 719.0 | DRSK0485 |
| 720.0 | DRSK0486 |
| 721.0 | DRSK0487 |
| 722.0 | DRSK0488 |
| 723.0 | DRSK0493 |
| 724.0 | DRSK0494 |
| 725.0 | DRSK0495 |
| 726.0 | DRSK0496 |
| 727.0 | DRSK0497 |
| 728.0 | DRSK0498 |
| 729.0 | DRSK0499 |
| 730.0 | DRSK0500 |
| 731.0 | DRSK0501 |
| 732.0 | DRSK0502 |
| 733.0 | DRSK0503 |
| 734.0 | DRSK0504 |
| 735.0 | DRSK0505 |
| 736.0 | DRSK0506 |
| 737.0 | DRSK0507 |
| 738.0 | DRSK0508 |
| 739.0 | DRSK0509 |
| 740.0 | DRSK0510 |
| 741.0 | DRSK0511 |
| 742.0 | DRSK0512 |
| 743.0 | DRSK0513 |
| 744.0 | DRSK0514 |
| 745.0 | DRSK0515 |
| 746.0 | DRSK0516 |
| 747.0 | DRSK0517 |
| 748.0 | DRSK0518 |
| 749.0 | DRSK0519 |
| 750.0 | DRSK0520 |
| 751.0 | DRSK0521 |
| 752.0 | DRSK0522 |
| 753.0 | DRSK0523 |
| 754.0 | DRSK0524 |
| 755.0 | DRSK0525 |
| 756.0 | DRSK0526 |
| 757.0 | DRSK0527 |
| 758.0 | DRSK0528 |
| 759.0 | DRSK0529 |
| 760.0 | DRSK0530 |
| 761.0 | DRSK0531 |
| 762.0 | DRSK0532 |
| 763.0 | DRSK0533 |
| 764.0 | DRSK0534 |
| 765.0 | DRSK0535 |
| 766.0 | DRSK0536 |
| 767.0 | DRSK0537 |
| 768.0 | DRSK1718 |
| 769.0 | DRSK0538 |
| 770.0 | DRSK0539 |
| 771.0 | DRSK0540 |
| 772.0 | DRSK0541 |
| 773.0 | DRSK0542 |
| 774.0 | DRSK0543 |
| 775.0 | DRSK0544 |
| 776.0 | DRSK0545 |
| 777.0 | DRSK0546 |
| 778.0 | DRSK0547 |
| 779.0 | DRSK0548 |
| 780.0 | DRSK0549 |
| 781.0 | DRSK0550 |
| 782.0 | DRSK0551 |
| 783.0 | DRSK0552 |
| 784.0 | DRSK0553 |
| 785.0 | DRSK0554 |
| 786.0 | DRSK0555 |
| 787.0 | DRSK0556 |
| 788.0 | DRSK0557 |
| 789.0 | DRSK0558 |
| 790.0 | DRSK0559 |
| 791.0 | DRSK0560 |
| 792.0 | DRSK0561 |
| 793.0 | DRSK0562 |
| 794.0 | DRSK0563 |
| 795.0 | DRSK0564 |
| 796.0 | DRSK0565 |
| 797.0 | DRSK0566 |
| 798.0 | DRSK0567 |
| 799.0 | DRSK0568 |
| 800.0 | DRSK0569 |
| 801.0 | DRSK0570 |
| 802.0 | DRSK0571 |
| 803.0 | DRSK0572 |
| 804.0 | DRSK0573 |
| 805.0 | DRSK0574 |
| 806.0 | DRSK0575 |
| 807.0 | DRSK0576 |
| 808.0 | DRSK0577 |
| 809.0 | DRSK0578 |
| 810.0 | DRSK0579 |
| 811.0 | DRSK0580 |
| 812.0 | DRSK0581 |
| 813.0 | DRSK0582 |
| 814.0 | DRSK0583 |
| 815.0 | DRSK0584 |
| 816.0 | DRSK0585 |
| 817.0 | DRSK0586 |
| 818.0 | DRSK0587 |
| 819.0 | DRSK0588 |
| 820.0 | DRSK2066 |
| 821.0 | DRSK2067 |
| 822.0 | DRSK2068 |
| 823.0 | DRSK2069 |
| 824.0 | DRSK0589 |
| 825.0 | DRSK0590 |
| 826.0 | DRSK0591 |
| 827.0 | DRSK0592 |
| 828.0 | DRSK0593 |
| 829.0 | DRSK0594 |
| 830.0 | DRSK0595 |
| 831.0 | DRSK0596 |
| 832.0 | DRSK0597 |
| 833.0 | DRSK0598 |
| 834.0 | DRSK0599 |
| 835.0 | DRSK0600 |
| 836.0 | DRSK0601 |
| 837.0 | DRSK0602 |
| 838.0 | DRSK0603 |
| 839.0 | DRSK0604 |
| 840.0 | DRSK0605 |
| 841.0 | DRSK0606 |
| 842.0 | DRSK0607 |
| 843.0 | DRSK0608 |
| 844.0 | DRSK0609 |
| 845.0 | DRSK0610 |
| 846.0 | DRSK0611 |
| 847.0 | DRSK0612 |
| 848.0 | DRSK0614 |
| 849.0 | DRSK0615 |
| 850.0 | DRSK0616 |
| 851.0 | DRSK0617 |
| 852.0 | DRSK0618 |
| 853.0 | DRSK0619 |
| 854.0 | DRSK0620 |
| 855.0 | DRSK0621 |
| 856.0 | DRSK0622 |
| 857.0 | DRSK0623 |
| 858.0 | DRSK2070 |
| 859.0 | DRSK2071 |
| 860.0 | DRSK2072 |
| 861.0 | DRSK0624 |
| 862.0 | DRSK1719 |
| 863.0 | DRSK0625 |
| 864.0 | DRSK0626 |
| 865.0 | DRSK0627 |
| 866.0 | DRSK0628 |
| 867.0 | DRSK0629 |
| 868.0 | DRSK0630 |
| 869.0 | DRSK0631 |
| 870.0 | DRSK0632 |
| 871.0 | DRSK0633 |
| 872.0 | DRSK0634 |
| 873.0 | DRSK0635 |
| 874.0 | DRSK0636 |
| 875.0 | DRSK0637 |
| 876.0 | DRSK0638 |
| 877.0 | DRSK0639 |
| 878.0 | DRSK0640 |
| 879.0 | DRSK0641 |
| 880.0 | DRSK0642 |
| 881.0 | DRSK0643 |
| 882.0 | DRSK0644 |
| 883.0 | DRSK0645 |
| 884.0 | DRSK0646 |
| 885.0 | DRSK0647 |
| 886.0 | DRSK0648 |
| 887.0 | DRSK0649 |
| 888.0 | DRSK0650 |
| 889.0 | DRSK0651 |
| 890.0 | DRSK0652 |
| 891.0 | DRSK0653 |
| 892.0 | DRSK0654 |
| 893.0 | DRSK0655 |
| 894.0 | DRSK0656 |
| 895.0 | DRSK0657 |
| 896.0 | DRSK0658 |
| 897.0 | DRSK0659 |
| 898.0 | DRSK0660 |
| 899.0 | DRSK0661 |
| 900.0 | DRSK0662 |
| 901.0 | DRSK0663 |
| 902.0 | DRSK0664 |
| 903.0 | DRSK0665 |
| 904.0 | DRSK0666 |
| 905.0 | DRSK0667 |
| 906.0 | DRSK0668 |
| 907.0 | DRSK0669 |
| 908.0 | DRSK0670 |
| 909.0 | DRSK0671 |
| 910.0 | DRSK0672 |
| 911.0 | DRSK0673 |
| 912.0 | DRSK0674 |
| 913.0 | DRSK0675 |
| 914.0 | DRSK0676 |
| 915.0 | DRSK0677 |
| 916.0 | DRSK0678 |
| 917.0 | DRSK0679 |
| 918.0 | DRSK0680 |
| 919.0 | DRSK0681 |
| 920.0 | DRSK0682 |
| 921.0 | DRSK0683 |
| 922.0 | DRSK0684 |
| 923.0 | DRSK0685 |
| 924.0 | DRSK0686 |
| 925.0 | DRSK0687 |
| 926.0 | DRSK0688 |
| 927.0 | DRSK0689 |
| 928.0 | DRSK0690 |
| 929.0 | DRSK0691 |
| 930.0 | DRSK0692 |
| 931.0 | DRSK0693 |
| 932.0 | DRSK0694 |
| 933.0 | DRSK0695 |
| 934.0 | DRSK0696 |
| 935.0 | DRSK0697 |
| 936.0 | DRSK0698 |
| 937.0 | DRSK0699 |
| 938.0 | DRSK0700 |
| 939.0 | DRSK0701 |
| 940.0 | DRSK0702 |
| 941.0 | DRSK0703 |
| 942.0 | DRSK0704 |
| 943.0 | DRSK0705 |
| 944.0 | DRSK0706 |
| 945.0 | DRSK0707 |
| 946.0 | DRSK0708 |
| 947.0 | DRSK0709 |
| 948.0 | DRSK0710 |
| 949.0 | DRSK0711 |
| 950.0 | DRSK0712 |
| 951.0 | DRSK0713 |
| 952.0 | DRSK0714 |
| 953.0 | DRSK0715 |
| 954.0 | DRSK0716 |
| 955.0 | DRSK0717 |
| 956.0 | DRSK0718 |
| 957.0 | DRSK1464 |
| 958.0 | DRSK0719 |
| 959.0 | DRSK1465 |
| 960.0 | DRSK0720 |
| 961.0 | DRSK0721 |
| 962.0 | DRSK0722 |
| 963.0 | DRSK0723 |
| 964.0 | DRSK0724 |
| 965.0 | DRSK0725 |
| 966.0 | DRSK0726 |
| 967.0 | DRSK0727 |
| 968.0 | DRSK0728 |
| 969.0 | DRSK0729 |
| 970.0 | DRSK0730 |
| 971.0 | DRSK0731 |
| 972.0 | DRSK0732 |
| 973.0 | DRSK0733 |
| 974.0 | DRSK1466 |
| 975.0 | DRSK0734 |
| 976.0 | DRSK0735 |
| 977.0 | DRSK0736 |
| 978.0 | DRSK0737 |
| 979.0 | DRSK0738 |
| 980.0 | DRSK1720 |
| 981.0 | DRSK2073 |
| 982.0 | DRSK2074 |
| 983.0 | DRSK0739 |
| 984.0 | DRSK0740 |
| 985.0 | DRSK0741 |
| 986.0 | DRSK0742 |
| 987.0 | DRSK0744 |
| 988.0 | DRSK0745 |
| 989.0 | DRSK0746 |
| 990.0 | DRSK0747 |
| 991.0 | DRSK0748 |
| 992.0 | DRSK0749 |
| 993.0 | DRSK0750 |
| 994.0 | DRSK0751 |
| 995.0 | DRSK0752 |
| 996.0 | DRSK0753 |
| 997.0 | DRSK0754 |
| 998.0 | DRSK0755 |
| 999.0 | DRSK0756 |
| 1000.0 | DRSK0757 |
| 1001.0 | DRSK0758 |
| 1002.0 | DRSK0759 |
| 1003.0 | DRSK0760 |
| 1004.0 | DRSK0761 |
| 1005.0 | DRSK0762 |
| 1006.0 | DRSK0763 |
| 1007.0 | DRSK0764 |
| 1008.0 | DRSK0765 |
| 1009.0 | DRSK0766 |
| 1010.0 | DRSK0767 |
| 1011.0 | DRSK0768 |
| 1012.0 | DRSK0769 |
| 1013.0 | DRSK0770 |
| 1014.0 | DRSK0771 |
| 1015.0 | DRSK0772 |
| 1016.0 | DRSK0773 |
| 1017.0 | DRSK0774 |
| 1018.0 | DRSK0775 |
| 1019.0 | DRSK0776 |
| 1020.0 | DRSK0777 |
| 1021.0 | DRSK0778 |
| 1022.0 | DRSK0779 |
| 1023.0 | DRSK0780 |
| 1024.0 | DRSK0781 |
| 1025.0 | DRSK0782 |
| 1026.0 | DRSK0783 |
| 1027.0 | DRSK0784 |
| 1028.0 | DRSK0785 |
| 1029.0 | DRSK0786 |
| 1030.0 | DRSK0787 |
| 1031.0 | DRSK0788 |
| 1032.0 | DRSK0789 |
| 1033.0 | DRSK0790 |
| 1034.0 | DRSK0791 |
| 1035.0 | DRSK0792 |
| 1036.0 | DRSK0793 |
| 1037.0 | DRSK0794 |
| 1038.0 | DRSK0795 |
| 1039.0 | DRSK0796 |
| 1040.0 | DRSK0797 |
| 1041.0 | DRSK0798 |
| 1042.0 | DRSK0799 |
| 1043.0 | DRSK0800 |
| 1044.0 | DRSK0801 |
| 1045.0 | DRSK0802 |
| 1046.0 | DRSK0803 |
| 1047.0 | DRSK0804 |
| 1048.0 | DRSK0805 |
| 1049.0 | DRSK0806 |
| 1050.0 | DRSK0807 |
| 1051.0 | DRSK0808 |
| 1052.0 | DRSK0809 |
| 1053.0 | DRSK0810 |
| 1054.0 | DRSK0811 |
| 1055.0 | DRSK0812 |
| 1056.0 | DRSK0813 |
| 1057.0 | DRSK0814 |
| 1058.0 | DRSK0815 |
| 1059.0 | DRSK0816 |
| 1060.0 | DRSK0817 |
| 1061.0 | DRSK0818 |
| 1062.0 | DRSK0819 |
| 1063.0 | DRSK0820 |
| 1064.0 | DRSK0821 |
| 1065.0 | DRSK2075 |
| 1066.0 | DRSK2076 |
| 1067.0 | DRSK2077 |
| 1068.0 | DRSK2078 |
| 1069.0 | DRSK2079 |
| 1070.0 | DRSK2080 |
| 1071.0 | DRSK0838 |
| 1072.0 | DRSK1721 |
| 1073.0 | DRSK1722 |
| 1074.0 | DRSK1723 |
| 1075.0 | DRSK1724 |
| 1076.0 | DRSK0843 |
| 1077.0 | DRSK0844 |
| 1078.0 | DRSK0845 |
| 1079.0 | DRSK0846 |
| 1080.0 | DRSK0847 |
| 1081.0 | DRSK0848 |
| 1082.0 | DRSK0849 |
| 1083.0 | DRSK0850 |
| 1084.0 | DRSK0851 |
| 1085.0 | DRSK0852 |
| 1086.0 | DRSK0853 |
| 1087.0 | DRSK0854 |
| 1088.0 | DRSK0855 |
| 1089.0 | DRSK0856 |
| 1090.0 | DRSK0857 |
| 1091.0 | DRSK1725 |
| 1092.0 | DRSK1726 |
| 1093.0 | DRSK1727 |
| 1094.0 | DRSK1728 |
| 1095.0 | DRSK2081 |
| 1096.0 | DRSK2082 |
| 1097.0 | DRSK2083 |
| 1098.0 | DRSK2084 |
| 1099.0 | DRSK2085 |
| 1100.0 | DRSK0874 |
| 1101.0 | DRSK0879 |
| 1102.0 | DRSK0880 |
| 1103.0 | DRSK0881 |
| 1104.0 | DRSK0882 |
| 1105.0 | DRSK0883 |
| 1106.0 | DRSK0884 |
| 1107.0 | DRSK0885 |
| 1108.0 | DRSK0886 |
| 1109.0 | DRSK0887 |
| 1110.0 | DRSK0888 |
| 1111.0 | DRSK0889 |
| 1112.0 | DRSK0890 |
| 1113.0 | DRSK0891 |
| 1114.0 | DRSK0892 |
| 1115.0 | DRSK0893 |
| 1116.0 | DRSK2086 |
| 1117.0 | DRSK2087 |
| 1118.0 | DRSK2088 |
| 1119.0 | DRSK2089 |
| 1120.0 | DRSK2090 |
| 1121.0 | DRSK0894 |
| 1122.0 | DRSK0895 |
| 1123.0 | DRSK0896 |
| 1124.0 | DRSK0897 |
| 1125.0 | DRSK0898 |
| 1126.0 | DRSK0899 |
| 1127.0 | DRSK0900 |
| 1128.0 | DRSK0901 |
| 1129.0 | DRSK0902 |
| 1130.0 | DRSK0903 |
| 1131.0 | DRSK1729 |
| 1132.0 | DRSK2091 |
| 1133.0 | DRSK2092 |
| 1134.0 | DRSK2093 |
| 1135.0 | DRSK0904 |
| 1136.0 | DRSK0905 |
| 1137.0 | DRSK0906 |
| 1138.0 | DRSK0907 |
| 1139.0 | DRSK0908 |
| 1140.0 | DRSK0909 |
| 1141.0 | DRSK0910 |
| 1142.0 | DRSK0911 |
| 1143.0 | DRSK0912 |
| 1144.0 | DRSK0913 |
| 1145.0 | DRSK0914 |
| 1146.0 | DRSK0915 |
| 1147.0 | DRSK0916 |
| 1148.0 | DRSK0917 |
| 1149.0 | DRSK0918 |
| 1150.0 | DRSK0919 |
| 1151.0 | DRSK0920 |
| 1152.0 | DRSK0921 |
| 1153.0 | DRSK0922 |
| 1154.0 | DRSK0923 |
| 1155.0 | DRSK0924 |
| 1156.0 | DRSK0925 |
| 1157.0 | DRSK0926 |
| 1158.0 | DRSK0927 |
| 1159.0 | DRSK0928 |
| 1160.0 | DRSK0929 |
| 1161.0 | DRSK0930 |
| 1162.0 | DRSK0931 |
| 1163.0 | DRSK0932 |
| 1164.0 | DRSK0933 |
| 1165.0 | DRSK0934 |
| 1166.0 | DRSK0935 |
| 1167.0 | DRSK0936 |
| 1168.0 | DRSK2094 |
| 1169.0 | DRSK2095 |
| 1170.0 | DRSK2096 |
| 1171.0 | DRSK2097 |
| 1172.0 | DRSK2098 |
| 1173.0 | DRSK2099 |
| 1174.0 | DRSK2100 |
| 1175.0 | DRSK2101 |
| 1176.0 | DRSK2102 |
| 1177.0 | DRSK2103 |
| 1178.0 | DRSK2104 |
| 1179.0 | DRSK2105 |
| 1180.0 | DRSK2106 |
| 1181.0 | DRSK2107 |
| 1182.0 | DRSK2108 |
| 1183.0 | DRSK2109 |
| 1184.0 | DRSK2110 |
| 1185.0 | DRSK2111 |
| 1186.0 | DRSK2112 |
| 1187.0 | DRSK2113 |
| 1188.0 | DRSK2114 |
| 1189.0 | DRSK2115 |
| 1190.0 | DRSK2116 |
| 1191.0 | DRSK2117 |
| 1192.0 | DRSK2118 |
| 1193.0 | DRSK2119 |
| 1194.0 | DRSK2120 |
| 1195.0 | DRSK2121 |
| 1196.0 | DRSK2122 |
| 1197.0 | DRSK2123 |
| 1198.0 | DRSK2124 |
| 1199.0 | DRSK2125 |
| 1200.0 | DRSK2126 |
| 1201.0 | DRSK2127 |
| 1202.0 | DRSK2128 |
| 1203.0 | DRSK2129 |
| 1204.0 | DRSK2130 |
| 1205.0 | DRSK2131 |
| 1206.0 | DRSK2132 |
| 1207.0 | DRSK2133 |
| 1208.0 | DRSK2134 |
| 1209.0 | DRSK2135 |
| 1210.0 | DRSK2136 |
| 1211.0 | DRSK2137 |
| 1212.0 | DRSK2138 |
| 1213.0 | DRSK2139 |
| 1214.0 | DRSK2140 |
| 1215.0 | DRSK2141 |
| 1216.0 | DRSK2142 |
| 1217.0 | DRSK2143 |
| 1218.0 | DRSK2144 |
| 1219.0 | DRSK2145 |
| 1220.0 | DRSK2146 |
| 1221.0 | DRSK2147 |
| 1222.0 | DRSK2148 |
| 1223.0 | DRSK2149 |
| 1224.0 | DRSK2150 |
| 1225.0 | DRSK2151 |
| 1226.0 | DRSK2152 |
| 1227.0 | DRSK2153 |
| 1228.0 | DRSK2154 |
| 1229.0 | DRSK2155 |
| 1230.0 | DRSK2156 |
| 1231.0 | DRSK2157 |
| 1232.0 | DRSK0937 |
| 1233.0 | DRSK0938 |
| 1234.0 | DRSK0939 |
| 1235.0 | DRSK0940 |
| 1236.0 | DRSK0941 |
| 1237.0 | DRSK0942 |
| 1238.0 | DRSK0943 |
| 1239.0 | DRSK0944 |
| 1240.0 | DRSK0945 |
| 1241.0 | DRSK0946 |
| 1242.0 | DRSK0947 |
| 1243.0 | DRSK0948 |
| 1244.0 | DRSK0949 |
| 1245.0 | DRSK0950 |
| 1246.0 | DRSK0951 |
| 1247.0 | DRSK2158 |
| 1248.0 | DRSK0952 |
| 1249.0 | DRSK0953 |
| 1250.0 | DRSK0954 |
| 1251.0 | DRSK0955 |
| 1252.0 | DRSK0956 |
| 1253.0 | DRSK0957 |
| 1254.0 | DRSK0958 |
| 1255.0 | DRSK0959 |
| 1256.0 | DRSK0960 |
| 1257.0 | DRSK0961 |
| 1258.0 | DRSK0962 |
| 1259.0 | DRSK0963 |
| 1260.0 | DRSK0964 |
| 1261.0 | DRSK0965 |
| 1262.0 | DRSK0966 |
| 1263.0 | DRSK0967 |
| 1264.0 | DRSK0968 |
| 1265.0 | DRSK0969 |
| 1266.0 | DRSK0970 |
| 1267.0 | DRSK0971 |
| 1268.0 | DRSK0972 |
| 1269.0 | DRSK0973 |
| 1270.0 | DRSK0974 |
| 1271.0 | DRSK0975 |
| 1272.0 | DRSK0976 |
| 1273.0 | DRSK0977 |
| 1274.0 | DRSK0978 |
| 1275.0 | DRSK0979 |
| 1276.0 | DRSK0980 |
| 1277.0 | DRSK0981 |
| 1278.0 | DRSK0982 |
| 1279.0 | DRSK0983 |
| 1280.0 | DRSK0984 |
| 1281.0 | DRSK0985 |
| 1282.0 | DRSK0986 |
| 1283.0 | DRSK0987 |
| 1284.0 | DRSK0988 |
| 1285.0 | DRSK0989 |
| 1286.0 | DRSK0990 |
| 1287.0 | DRSK0991 |
| 1288.0 | DRSK0992 |
| 1289.0 | DRSK0993 |
| 1290.0 | DRSK0994 |
| 1291.0 | DRSK0995 |
| 1292.0 | DRSK0996 |
| 1293.0 | DRSK0997 |
| 1294.0 | DRSK0998 |
| 1295.0 | DRSK0999 |
| 1296.0 | DRSK1000 |
| 1297.0 | DRSK1001 |
| 1298.0 | DRSK1002 |
| 1299.0 | DRSK1003 |
| 1300.0 | DRSK1004 |
| 1301.0 | DRSK1005 |
| 1302.0 | DRSK1006 |
| 1303.0 | DRSK1007 |
| 1304.0 | DRSK1008 |
| 1305.0 | DRSK1009 |
| 1306.0 | DRSK1010 |
| 1307.0 | DRSK1011 |
| 1308.0 | DRSK1012 |
| 1309.0 | DRSK1013 |
| 1310.0 | DRSK1014 |
| 1311.0 | DRSK1015 |
| 1312.0 | DRSK1016 |
| 1313.0 | DRSK1017 |
| 1314.0 | DRSK1018 |
| 1315.0 | DRSK1019 |
| 1316.0 | DRSK1020 |
| 1317.0 | DRSK1021 |
| 1318.0 | DRSK1022 |
| 1319.0 | DRSK1023 |
| 1320.0 | DRSK1024 |
| 1321.0 | DRSK1025 |
| 1322.0 | DRSK1026 |
| 1323.0 | DRSK1027 |
| 1324.0 | DRSK1028 |
| 1325.0 | DRSK1029 |
| 1326.0 | DRSK1030 |
| 1327.0 | DRSK1031 |
| 1328.0 | DRSK2159 |
| 1329.0 | DRSK1033 |
| 1330.0 | DRSK1034 |
| 1331.0 | DRSK1035 |
| 1332.0 | DRSK2160 |
| 1333.0 | DRSK2161 |
| 1334.0 | DRSK2162 |
| 1335.0 | DRSK1036 |
| 1336.0 | DRSK1037 |
| 1337.0 | DRSK1038 |
| 1338.0 | DRSK1039 |
| 1339.0 | DRSK1040 |
| 1340.0 | DRSK1041 |
| 1341.0 | DRSK1042 |
| 1342.0 | DRSK1043 |
| 1343.0 | DRSK1044 |
| 1344.0 | DRSK1848 |
| 1345.0 | DRSK1048 |
| 1346.0 | DRSK1049 |
| 1347.0 | DRSK1050 |
| 1348.0 | DRSK1051 |
| 1349.0 | DRSK1052 |
| 1350.0 | DRSK1053 |
| 1351.0 | DRSK1054 |
| 1352.0 | DRSK1055 |
| 1353.0 | DRSK1056 |
| 1354.0 | DRSK1057 |
| 1355.0 | DRSK1058 |
| 1356.0 | DRSK1059 |
| 1357.0 | DRSK1060 |
| 1358.0 | DRSK1061 |
| 1359.0 | DRSK1062 |
| 1360.0 | DRSK1063 |
| 1361.0 | DRSK1064 |
| 1362.0 | DRSK1065 |
| 1363.0 | DRSK1066 |
| 1364.0 | DRSK1067 |
| 1365.0 | DRSK1068 |
| 1366.0 | DRSK2163 |
| 1367.0 | DRSK1069 |
| 1368.0 | DRSK1070 |
| 1369.0 | DRSK1071 |
| 1370.0 | DRSK1072 |
| 1371.0 | DRSK1073 |
| 1372.0 | DRSK1076 |
| 1373.0 | DRSK1078 |
| 1374.0 | DRSK1080 |
| 1375.0 | DRSK1083 |
| 1376.0 | DRSK1084 |
| 1377.0 | DRSK1085 |
| 1378.0 | DRSK1086 |
| 1379.0 | DRSK1087 |
| 1380.0 | DRSK1088 |
| 1381.0 | DRSK1730 |
| 1382.0 | DRSK1731 |
| 1383.0 | DRSK1732 |
| 1384.0 | DRSK2164 |
| 1385.0 | DRSK2165 |
| 1386.0 | DRSK2166 |
| 1387.0 | DRSK2167 |
| 1388.0 | DRSK2168 |
| 1389.0 | DRSK2169 |
| 1390.0 | DRSK2170 |
| 1391.0 | DRSK2171 |
| 1392.0 | DRSK2172 |
| 1393.0 | DRSK2173 |
| 1394.0 | DRSK2174 |
| 1395.0 | DRSK2175 |
| 1396.0 | DRSK2176 |
| 1397.0 | DRSK2177 |
| 1398.0 | DRSK2178 |
| 1399.0 | DRSK2179 |
| 1400.0 | DRSK2180 |
| 1401.0 | DRSK2181 |
| 1402.0 | DRSK2182 |
| 1403.0 | DRSK2183 |
| 1404.0 | DRSK2184 |
| 1405.0 | DRSK2185 |
| 1406.0 | DRSK2186 |
| 1407.0 | DRSK2187 |
| 1408.0 | DRSK2188 |
| 1409.0 | DRSK2189 |
| 1410.0 | DRSK2190 |
| 1411.0 | DRSK2191 |
| 1412.0 | DRSK2192 |
| 1413.0 | DRSK2193 |
| 1414.0 | DRSK2194 |
| 1415.0 | DRSK2195 |
| 1416.0 | DRSK1089 |
| 1417.0 | DRSK1090 |
| 1418.0 | DRSK1091 |
| 1419.0 | DRSK1092 |
| 1420.0 | DRSK1093 |
| 1421.0 | DRSK1094 |
| 1422.0 | DRSK1095 |
| 1423.0 | DRSK1096 |
| 1424.0 | DRSK1097 |
| 1425.0 | DRSK1098 |
| 1426.0 | DRSK1099 |
| 1427.0 | DRSK1100 |
| 1428.0 | DRSK1101 |
| 1429.0 | DRSK1102 |
| 1430.0 | DRSK1103 |
| 1431.0 | DRSK1104 |
| 1432.0 | DRSK1105 |
| 1433.0 | DRSK1106 |
| 1434.0 | DRSK1107 |
| 1435.0 | DRSK1108 |
| 1436.0 | DRSK1109 |
| 1437.0 | DRSK1110 |
| 1438.0 | DRSK1111 |
| 1439.0 | DRSK1112 |
| 1440.0 | DRSK1113 |
| 1441.0 | DRSK1114 |
| 1442.0 | DRSK1115 |
| 1443.0 | DRSK1116 |
| 1444.0 | DRSK1117 |
| 1445.0 | DRSK1118 |
| 1446.0 | DRSK1119 |
| 1447.0 | DRSK1120 |
| 1448.0 | DRSK1121 |
| 1449.0 | DRSK1122 |
| 1450.0 | DRSK1123 |
| 1451.0 | DRSK1124 |
| 1452.0 | DRSK1125 |
| 1453.0 | DRSK1126 |
| 1454.0 | DRSK1127 |
| 1455.0 | DRSK1128 |
| 1456.0 | DRSK2196 |
| 1457.0 | DRSK2197 |
| 1458.0 | DRSK2198 |
| 1459.0 | DRSK2199 |
| 1460.0 | DRSK2200 |
| 1461.0 | DRSK2201 |
| 1462.0 | DRSK2202 |
| 1463.0 | DRSK2203 |
| 1464.0 | DRSK2204 |
| 1465.0 | DRSK2205 |
| 1466.0 | DRSK2206 |
| 1467.0 | DRSK2207 |
| 1468.0 | DRSK2208 |
| 1469.0 | DRSK2209 |
| 1470.0 | DRSK2210 |
| 1471.0 | DRSK2211 |
| 1472.0 | DRSK2212 |
| 1473.0 | DRSK2213 |
| 1474.0 | DRSK2214 |
| 1475.0 | DRSK2215 |
| 1476.0 | DRSK2216 |
| 1477.0 | DRSK2217 |
| 1478.0 | DRSK2218 |
| 1479.0 | DRSK2219 |
| 1480.0 | DRSK2220 |
| 1481.0 | DRSK1129 |
| 1482.0 | DRSK1130 |
| 1483.0 | DRSK1131 |
| 1484.0 | DRSK1132 |
| 1485.0 | DRSK1133 |
| 1486.0 | DRSK1134 |
| 1487.0 | DRSK1135 |
| 1488.0 | DRSK1136 |
| 1489.0 | DRSK1137 |
| 1490.0 | DRSK1138 |
| 1491.0 | DRSK1139 |
| 1492.0 | DRSK1140 |
| 1493.0 | DRSK1141 |
| 1494.0 | DRSK1142 |
| 1495.0 | DRSK1143 |
| 1496.0 | DRSK1144 |
| 1497.0 | DRSK1145 |
| 1498.0 | DRSK1146 |
| 1499.0 | DRSK1147 |
| 1500.0 | DRSK1148 |
| 1501.0 | DRSK1149 |
| 1502.0 | DRSK1150 |
| 1503.0 | DRSK1151 |
| 1504.0 | DRSK1152 |
| 1505.0 | DRSK1153 |
| 1506.0 | DRSK1154 |
| 1507.0 | DRSK1155 |
| 1508.0 | DRSK2221 |
| 1509.0 | DRSK2222 |
| 1510.0 | DRSK2223 |
| 1511.0 | DRSK2224 |
| 1512.0 | DRSK2225 |
| 1513.0 | DRSK2226 |
| 1514.0 | DRSK1156 |
| 1515.0 | DRSK1157 |
| 1516.0 | DRSK1158 |
| 1517.0 | DRSK1159 |
| 1518.0 | DRSK1160 |
| 1519.0 | DRSK1161 |
| 1520.0 | DRSK1162 |
| 1521.0 | DRSK1163 |
| 1522.0 | DRSK1164 |
| 1523.0 | DRSK2227 |
| 1524.0 | DRSK2228 |
| 1525.0 | DRSK2229 |
| 1526.0 | DRSK2230 |
| 1527.0 | DRSK2231 |
| 1528.0 | DRSK2232 |
| 1529.0 | DRSK2233 |
| 1530.0 | DRSK2234 |
| 1531.0 | DRSK1165 |
| 1532.0 | DRSK1166 |
| 1533.0 | DRSK1167 |
| 1534.0 | DRSK1168 |
| 1535.0 | DRSK1169 |
| 1536.0 | DRSK1170 |
| 1537.0 | DRSK1171 |
| 1538.0 | DRSK1172 |
| 1539.0 | DRSK1173 |
| 1540.0 | DRSK1174 |
| 1541.0 | DRSK1175 |
| 1542.0 | DRSK1176 |
| 1543.0 | DRSK1177 |
| 1544.0 | DRSK1178 |
| 1545.0 | DRSK1179 |
| 1546.0 | DRSK1180 |
| 1547.0 | DRSK1181 |
| 1548.0 | DRSK1182 |
| 1549.0 | DRSK1183 |
| 1550.0 | DRSK1184 |
| 1551.0 | DRSK1185 |
| 1552.0 | DRSK1186 |
| 1553.0 | DRSK1187 |
| 1554.0 | DRSK1188 |
| 1555.0 | DRSK1189 |
| 1556.0 | DRSK1190 |
| 1557.0 | DRSK1191 |
| 1558.0 | DRSK1192 |
| 1559.0 | DRSK1193 |
| 1560.0 | DRSK1194 |
| 1561.0 | DRSK1195 |
| 1562.0 | DRSK1196 |
| 1563.0 | DRSK1197 |
| 1564.0 | DRSK1198 |
| 1565.0 | DRSK1199 |
| 1566.0 | DRSK1733 |
| 1567.0 | DRSK1201 |
| 1568.0 | DRSK1202 |
| 1569.0 | DRSK1203 |
| 1570.0 | DRSK1204 |
| 1571.0 | DRSK1205 |
| 1572.0 | DRSK1206 |
| 1573.0 | DRSK1207 |
| 1574.0 | DRSK1208 |
| 1575.0 | DRSK1209 |
| 1576.0 | DRSK1210 |
| 1577.0 | DRSK1211 |
| 1578.0 | DRSK1212 |
| 1579.0 | DRSK1213 |
| 1580.0 | DRSK1214 |
| 1581.0 | DRSK1215 |
| 1582.0 | DRSK1216 |
| 1583.0 | DRSK1217 |
| 1584.0 | DRSK2235 |
| 1585.0 | DRSK2236 |
| 1586.0 | DRSK2237 |
| 1587.0 | DRSK1219 |
| 1588.0 | DRSK1220 |
| 1589.0 | DRSK1221 |
| 1590.0 | DRSK1222 |
| 1591.0 | DRSK1223 |
| 1592.0 | DRSK1224 |
| 1593.0 | DRSK1225 |
| 1594.0 | DRSK1226 |
| 1595.0 | DRSK1227 |
| 1596.0 | DRSK1228 |
| 1597.0 | DRSK1229 |
| 1598.0 | DRSK1230 |
| 1599.0 | DRSK1231 |
| 1600.0 | DRSK1232 |
| 1601.0 | DRSK1233 |
| 1602.0 | DRSK1234 |
| 1603.0 | DRSK1235 |
| 1604.0 | DRSK1734 |
| 1605.0 | DRSK1735 |
| 1606.0 | DRSK1736 |
| 1607.0 | DRSK1737 |
| 1608.0 | DRSK2238 |
| 1609.0 | DRSK2239 |
| 1610.0 | DRSK2240 |
| 1611.0 | DRSK2241 |
| 1612.0 | DRSK2242 |
| 1613.0 | DRSK2243 |
| 1614.0 | DRSK2244 |
| 1615.0 | DRSK2245 |
| 1616.0 | DRSK2246 |
| 1617.0 | DRSK2247 |
| 1618.0 | DRSK2248 |
| 1619.0 | DRSK2249 |
| 1620.0 | DRSK2250 |
| 1621.0 | DRSK2251 |
| 1622.0 | DRSK2252 |
| 1623.0 | DRSK2253 |
| 1624.0 | DRSK2254 |
| 1625.0 | DRSK2255 |
| 1626.0 | DRSK2256 |
| 1627.0 | DRSK2257 |
| 1628.0 | DRSK2258 |
| 1629.0 | DRSK2259 |
| 1630.0 | DRSK2260 |
| 1631.0 | DRSK1236 |
| 1632.0 | DRSK1237 |
| 1633.0 | DRSK1238 |
| 1634.0 | DRSK1239 |
| 1635.0 | DRSK1240 |
| 1636.0 | DRSK1241 |
| 1637.0 | DRSK1242 |
| 1638.0 | DRSK1243 |
| 1639.0 | DRSK1244 |
| 1640.0 | DRSK1245 |
| 1641.0 | DRSK1246 |
| 1642.0 | DRSK1247 |
| 1643.0 | DRSK1248 |
| 1644.0 | DRSK1249 |
| 1645.0 | DRSK1250 |
| 1646.0 | DRSK1251 |
| 1647.0 | DRSK1252 |
| 1648.0 | DRSK1253 |
| 1649.0 | DRSK1254 |
| 1650.0 | DRSK1255 |
| 1651.0 | DRSK1256 |
| 1652.0 | DRSK1257 |
| 1653.0 | DRSK1258 |
| 1654.0 | DRSK1259 |
| 1655.0 | DRSK1270 |
| 1656.0 | DRSK1271 |
| 1657.0 | DRSK1272 |
| 1658.0 | DRSK1273 |
| 1659.0 | DRSK1274 |
| 1660.0 | DRSK1275 |
| 1661.0 | DRSK1276 |
| 1662.0 | DRSK1277 |
| 1663.0 | DRSK1278 |
| 1664.0 | DRSK1279 |
| 1665.0 | DRSK1280 |
| 1666.0 | DRSK1281 |
| 1667.0 | DRSK1282 |
| 1668.0 | DRSK1283 |
| 1669.0 | DRSK1284 |
| 1670.0 | DRSK1285 |
| 1671.0 | DRSK1286 |
| 1672.0 | DRSK1287 |
| 1673.0 | DRSK1288 |
| 1674.0 | DRSK1289 |
| 1675.0 | DRSK1290 |
| 1676.0 | DRSK1291 |
| 1677.0 | DRSK1292 |
| 1678.0 | DRSK1293 |
| 1679.0 | DRSK1294 |
| 1680.0 | DRSK1295 |
| 1681.0 | DRSK1296 |
| 1682.0 | DRSK1297 |
| 1683.0 | DRSK1298 |
| 1684.0 | DRSK1299 |
| 1685.0 | DRSK1300 |
| 1686.0 | DRSK1301 |
| 1687.0 | DRSK1302 |
| 1688.0 | DRSK1303 |
| 1689.0 | DRSK1304 |
| 1690.0 | DRSK1305 |
| 1691.0 | DRSK1306 |
| 1692.0 | DRSK1307 |
| 1693.0 | DRSK1308 |
| 1694.0 | DRSK1309 |
| 1695.0 | DRSK1310 |
| 1696.0 | DRSK1311 |
| 1697.0 | DRSK1312 |
| 1698.0 | DRSK1313 |
| 1699.0 | DRSK1314 |
| 1700.0 | DRSK1315 |
| 1701.0 | DRSK1316 |
| 1702.0 | DRSK1317 |
| 1703.0 | DRSK1318 |
| 1704.0 | DRSK1319 |
| 1705.0 | DRSK1320 |
| 1706.0 | DRSK1321 |
| 1707.0 | DRSK1322 |
| 1708.0 | DRSK1323 |
| 1709.0 | DRSK1324 |
| 1710.0 | DRSK1325 |
| 1711.0 | DRSK1326 |
| 1712.0 | DRSK1327 |
| 1713.0 | DRSK1328 |
| 1714.0 | DRSK1329 |
| 1715.0 | DRSK1330 |
| 1716.0 | DRSK1331 |
| 1717.0 | DRSK1332 |
| 1718.0 | DRSK1333 |
| 1719.0 | DRSK1334 |
| 1720.0 | DRSK1335 |
| 1721.0 | DRSK1336 |
| 1722.0 | DRSK1337 |
| 1723.0 | DRSK1338 |
| 1724.0 | DRSK1339 |
| 1725.0 | DRSK1340 |
| 1726.0 | DRSK1341 |
| 1727.0 | DRSK1342 |
| 1728.0 | DRSK1343 |
| 1729.0 | DRSK1344 |
| 1730.0 | DRSK1345 |
| 1731.0 | DRSK1346 |
| 1732.0 | DRSK1347 |
| 1733.0 | DRSK1348 |
| 1734.0 | DRSK1349 |
| 1735.0 | DRSK1350 |
| 1736.0 | DRSK1351 |
| 1737.0 | DRSK1352 |
| 1738.0 | DRSK1353 |
| 1739.0 | DRSK1354 |
| 1740.0 | DRSK1355 |
| 1741.0 | DRSK1356 |
| 1742.0 | DRSK1357 |
| 1743.0 | DRSK1358 |
| 1744.0 | DRSK1359 |
| 1745.0 | DRSK1360 |
| 1746.0 | DRSK1361 |
| 1747.0 | DRSK1362 |
| 1748.0 | DRSK1363 |
| 1749.0 | DRSK1364 |
| 1750.0 | DRSK1365 |
| 1751.0 | DRSK1366 |
| 1752.0 | DRSK1367 |
| 1753.0 | DRSK1368 |
| 1754.0 | DRSK1369 |
| 1755.0 | DRSK1370 |
| 1756.0 | DRSK1371 |
| 1757.0 | DRSK1372 |
| 1758.0 | DRSK1373 |
| 1759.0 | DRSK1374 |
| 1760.0 | DRSK1375 |
| 1761.0 | DRSK1376 |
| 1762.0 | DRSK1377 |
| 1763.0 | DRSK1378 |
| 1764.0 | DRSK1379 |
| 1765.0 | DRSK1380 |
| 1766.0 | DRSK1381 |
| 1767.0 | DRSK1382 |
| 1768.0 | DRSK1383 |
| 1769.0 | DRSK1384 |
| 1770.0 | DRSK1385 |
| 1771.0 | DRSK1386 |
| 1772.0 | DRSK1387 |
| 1773.0 | DRSK1388 |
| 1774.0 | DRSK1389 |
| 1775.0 | DRSK1390 |
| 1776.0 | DRSK1391 |
| 1777.0 | DRSK1392 |
| 1778.0 | DRSK1393 |
| 1779.0 | DRSK1394 |
| 1780.0 | DRSK1395 |
| 1781.0 | DRSK1396 |
| 1782.0 | DRSK1397 |
| 1783.0 | DRSK1398 |
| 1784.0 | DRSK1399 |
| 1785.0 | DRSK1400 |
| 1786.0 | DRSK1401 |
| 1787.0 | DRSK1402 |
| 1788.0 | DRSK1403 |
| 1789.0 | DRSK1404 |
| 1790.0 | DRSK1405 |
| 1791.0 | DRSK1406 |
| 1792.0 | DRSK1407 |
| 1793.0 | DRSK1408 |
| 1794.0 | DRSK1409 |
| 1795.0 | DRSK1410 |
| 1796.0 | DRSK1411 |
| 1797.0 | DRSK1412 |
| 1798.0 | DRSK1413 |
| 1799.0 | DRSK1414 |
| 1800.0 | DRSK1415 |
| 1801.0 | DRSK1416 |
| 1802.0 | DRSK1417 |
| 1803.0 | DRSK1418 |
| 1804.0 | DRSK1419 |
| 1805.0 | DRSK1420 |
| 1806.0 | DRSK1421 |
| 1807.0 | DRSK1422 |
| 1808.0 | DRSK1423 |
| 1809.0 | DRSK1424 |
| 1810.0 | DRSK1425 |
| 1811.0 | DRSK1426 |
| 1812.0 | DRSK1427 |
| 1813.0 | DRSK1428 |
| 1814.0 | DRSK1429 |
| 1815.0 | DRSK1430 |
| 1816.0 | DRSK1431 |
| 1817.0 | DRSK1432 |
| 1818.0 | DRSK1433 |
| 1819.0 | DRSK1434 |
| 1820.0 | DRSK1435 |
| 1821.0 | DRSK1739 |
| 1822.0 | DRSK1740 |
| 1823.0 | DRSK1437 |
| 1824.0 | DRSK1438 |
| 1825.0 | DRSK1439 |
| 1826.0 | DRSK1440 |
| 1827.0 | DRSK1441 |
| 1828.0 | DRSK1442 |
| 1829.0 | DRSK1443 |
| 1830.0 | DRSK1444 |
| 1831.0 | DRSK1445 |
| 1832.0 | DRSK1446 |
| 1833.0 | DRSK1447 |
| 1834.0 | DRSK1448 |
| 1835.0 | DRSK1449 |
| 1836.0 | DRSK1450 |
| 1837.0 | DRSK1451 |
| 1838.0 | DRSK1452 |
| 1839.0 | DRSK1453 |
| 1840.0 | DRSK1454 |
| 1841.0 | DRSK1455 |
| 1842.0 | DRSK1546 |
| 1843.0 | DRSK1547 |
| 1844.0 | DRSK1549 |
| 1845.0 | DRSK1550 |
| 1846.0 | DRSK1551 |
| 1847.0 | DRSK1552 |
| 1848.0 | DRSK1553 |
| 1849.0 | DRSK1554 |
| 1850.0 | DRSK1555 |
| 1851.0 | DRSK1556 |
| 1852.0 | DRSK1557 |
| 1853.0 | DRSK1558 |
| 1854.0 | DRSK1559 |
| 1855.0 | DRSK1560 |
| 1856.0 | DRSK1561 |
| 1857.0 | DRSK1562 |
| 1858.0 | DRSK1563 |
| 1859.0 | DRSK1564 |
| 1860.0 | DRSK1565 |
| 1861.0 | DRSK1566 |
| 1862.0 | DRSK1567 |
| 1863.0 | DRSK1568 |
| 1864.0 | DRSK1569 |
| 1865.0 | DRSK1570 |
| 1866.0 | DRSK1571 |
| 1867.0 | DRSK1572 |
| 1868.0 | DRSK1573 |
| 1869.0 | DRSK1574 |
| 1870.0 | DRSK1575 |
| 1871.0 | DRSK1576 |
| 1872.0 | DRSK1577 |
| 1873.0 | DRSK1578 |
| 1874.0 | DRSK1579 |
| 1875.0 | DRSK1581 |
| 1876.0 | DRSK2261 |
| 1877.0 | DRSK1582 |
| 1878.0 | DRSK1583 |
| 1879.0 | DRSK1584 |
| 1880.0 | DRSK1585 |
| 1881.0 | DRSK1586 |
| 1882.0 | DRSK1587 |
| 1883.0 | DRSK1588 |
| 1884.0 | DRSK1589 |
| 1885.0 | DRSK1590 |
| 1886.0 | DRSK1591 |
| 1887.0 | DRSK1592 |
| 1888.0 | DRSK1593 |
| 1889.0 | DRSK1594 |
| 1890.0 | DRSK1595 |
| 1891.0 | DRSK1596 |
| 1892.0 | DRSK1597 |
| 1893.0 | DRSK1598 |
| 1894.0 | DRSK1599 |
| 1895.0 | DRSK1600 |
| 1896.0 | DRSK1601 |
| 1897.0 | DRSK1602 |
| 1898.0 | DRSK1603 |
| 1899.0 | DRSK1604 |
| 1900.0 | DRSK1605 |
| 1901.0 | DRSK1606 |
| 1902.0 | DRSK1607 |
| 1903.0 | DRSK1608 |
| 1904.0 | DRSK1609 |
| 1905.0 | DRSK1610 |
| 1906.0 | DRSK1611 |
| 1907.0 | DRSK1612 |
| 1908.0 | DRSK1613 |
| 1909.0 | DRSK1614 |
| 1910.0 | DRSK1615 |
| 1911.0 | DRSK1616 |
| 1912.0 | DRSK1617 |
| 1913.0 | DRSK1618 |
| 1914.0 | DRSK1619 |
| 1915.0 | DRSK1620 |
| 1916.0 | DRSK1621 |
| 1917.0 | DRSK1622 |
| 1918.0 | DRSK1623 |
| 1919.0 | DRSK1624 |
| 1920.0 | DRSK1625 |
| 1921.0 | DRSK1626 |
| 1922.0 | DRSK2262 |
| 1923.0 | DRSK2263 |
| 1924.0 | DRSK1627 |
| 1925.0 | DRSK1628 |
| 1926.0 | DRSK1629 |
| 1927.0 | DRSK1630 |
| 1928.0 | DRSK1631 |
| 1929.0 | DRSK1632 |
| 1930.0 | DRSK1633 |
| 1931.0 | DRSK1634 |
| 1932.0 | DRSK1635 |
| 1933.0 | DRSK1636 |
| 1934.0 | DRSK1637 |
| 1935.0 | DRSK1638 |
| 1936.0 | DRSK1639 |
| 1937.0 | DRSK1640 |
| 1938.0 | DRSK1641 |
| 1939.0 | DRSK1642 |
| 1940.0 | DRSK1643 |
| 1941.0 | DRSK1644 |
| 1942.0 | DRSK1645 |
| 1943.0 | DRSK1646 |
| 1944.0 | DRSK1647 |
| 1945.0 | DRSK1648 |
| 1946.0 | DRSK1649 |
| 1947.0 | DRSK1650 |
| 1948.0 | DRSK1651 |
| 1949.0 | DRSK1652 |
| 1950.0 | DRSK1653 |
| 1951.0 | DRSK1654 |
| 1952.0 | DRSK1655 |
| 1953.0 | DRSK1656 |
| 1954.0 | DRSK1657 |
| 1955.0 | DRSK1658 |
| 1956.0 | DRSK1659 |
| 1957.0 | DRSK1660 |
| 1958.0 | DRSK1661 |
| 1959.0 | DRSK1662 |
| 1960.0 | DRSK1663 |
| 1961.0 | DRSK1664 |
| 1962.0 | DRSK1665 |
| 1963.0 | DRSK1666 |
| 1964.0 | DRSK1667 |
| 1965.0 | DRSK1668 |
| 1966.0 | DRSK1669 |
| 1967.0 | DRSK1670 |
| 1968.0 | DRSK1671 |
| 1969.0 | DRSK1672 |
| 1970.0 | DRSK1673 |
| 1971.0 | DRSK1674 |
| 1972.0 | DRSK1676 |
| 1973.0 | DRSK1677 |
| 1974.0 | DRSK1678 |
| 1975.0 | DRSK1679 |
| 1976.0 | DRSK1680 |
| 1977.0 | DRSK1681 |
| 1978.0 | DRSK1682 |
| 1979.0 | DRSK1683 |
| 1980.0 | DRSK1684 |
| 1981.0 | DRSK1685 |
| 1982.0 | DRSK1686 |
| 1983.0 | DRSK1687 |
| 1984.0 | DRSK1688 |
| 1985.0 | DRSK1689 |
| 1986.0 | DRSK1690 |
| 1987.0 | DRSK1691 |
| 1988.0 | DRSK1692 |
| 1989.0 | DRSK1693 |
| 1990.0 | DRSK1694 |
| 1991.0 | DRSK1695 |
| 1992.0 | DRSK1696 |
| 1993.0 | DRSK1697 |
| 1994.0 | DRSK1698 |
| 1995.0 | DRSK1699 |
| 1996.0 | DRSK1700 |
| 1997.0 | DRSK1701 |
| 1998.0 | DRSK1702 |
| 1999.0 | DRSK1703 |
| 2000.0 | DRSK1704 |
| 2001.0 | DRSK1705 |
| 2002.0 | DRSK1706 |
| 2003.0 | DRSK1707 |
| 2004.0 | DRSK1708 |
| 2005.0 | DRSK1709 |
| 2006.0 | DRSK1710 |
| 2007.0 | DRSK1711 |
| 2008.0 | DRSK1741 |
| 2009.0 | DRSK1742 |
| 2010.0 | DRSK1743 |
| 2011.0 | DRSK1744 |
| 2012.0 | DRSK1745 |
| 2013.0 | DRSK1746 |
| 2014.0 | DRSK1747 |
| 2015.0 | DRSK1748 |
| 2016.0 | DRSK1749 |
| 2017.0 | DRSK1750 |
| 2018.0 | DRSK1751 |
| 2019.0 | DRSK1752 |
| 2020.0 | DRSK1753 |
| 2021.0 | DRSK1754 |
| 2022.0 | DRSK1755 |
| 2023.0 | DRSK1756 |
| 2024.0 | DRSK1757 |
| 2025.0 | DRSK1758 |
| 2026.0 | DRSK1759 |
| 2027.0 | DRSK1760 |
| 2028.0 | DRSK1761 |
| 2029.0 | DRSK1762 |
| 2030.0 | DRSK1763 |
| 2031.0 | DRSK1764 |
| 2032.0 | DRSK1765 |
| 2033.0 | DRSK1766 |
| 2034.0 | DRSK1767 |
| 2035.0 | DRSK1768 |
| 2036.0 | DRSK1769 |
| 2037.0 | DRSK1770 |
| 2038.0 | DRSK1771 |
| 2039.0 | DRSK1772 |
| 2040.0 | DRSK1773 |
| 2041.0 | DRSK1774 |
| 2042.0 | DRSK1775 |
| 2043.0 | DRSK1776 |
| 2044.0 | DRSK1777 |
| 2045.0 | DRSK1778 |
| 2046.0 | DRSK1779 |
| 2047.0 | DRSK1780 |
| 2048.0 | DRSK1781 |
| 2049.0 | DRSK1782 |
| 2050.0 | DRSK1783 |
| 2051.0 | DRSK1784 |
| 2052.0 | DRSK2264 |
| 2053.0 | DRSK2265 |
| 2054.0 | DRSK2266 |
| 2055.0 | DRSK2267 |
| 2056.0 | DRSK2268 |
| 2057.0 | DRSK2269 |
| 2058.0 | DRSK2270 |
| 2059.0 | DRSK2271 |
| 2060.0 | DRSK1785 |
| 2061.0 | DRSK1786 |
| 2062.0 | DRSK1787 |
| 2063.0 | DRSK1788 |
| 2064.0 | DRSK1789 |
| 2065.0 | DRSK1790 |
| 2066.0 | DRSK1791 |
| 2067.0 | DRSK1792 |
| 2068.0 | DRSK1793 |
| 2069.0 | DRSK1794 |
| 2070.0 | DRSK1795 |
| 2071.0 | DRSK1796 |
| 2072.0 | DRSK1797 |
| 2073.0 | DRSK1798 |
| 2074.0 | DRSK1799 |
| 2075.0 | DRSK1800 |
| 2076.0 | DRSK1801 |
| 2077.0 | DRSK1802 |
| 2078.0 | DRSK1803 |
| 2079.0 | DRSK1804 |
| 2080.0 | DRSK1805 |
| 2081.0 | DRSK1806 |
| 2082.0 | DRSK1807 |
| 2083.0 | DRSK1808 |
| 2084.0 | DRSK1809 |
| 2085.0 | DRSK1810 |
| 2086.0 | DRSK1811 |
| 2087.0 | DRSK1812 |
| 2088.0 | DRSK2272 |
| 2089.0 | DRSK2273 |
| 2090.0 | DRSK2274 |
| 2091.0 | DRSK1814 |
| 2092.0 | DRSK1813 |
| 2093.0 | DRSK1815 |
| 2094.0 | DRSK1816 |
| 2095.0 | DRSK1817 |
| 2096.0 | DRSK1818 |
| 2097.0 | DRSK1819 |
| 2098.0 | DRSK1820 |
| 2099.0 | DRSK1821 |
| 2100.0 | DRSK1822 |
| 2101.0 | DRSK1823 |
| 2102.0 | DRSK1824 |
| 2103.0 | DRSK2275 |
| 2104.0 | DRSK1825 |
| 2105.0 | DRSK1826 |
| 2106.0 | DRSK1827 |
| 2107.0 | DRSK1828 |
| 2108.0 | DRSK1829 |
| 2109.0 | DRSK1830 |
| 2110.0 | DRSK1831 |
| 2111.0 | DRSK1832 |
| 2112.0 | DRSK1833 |
| 2113.0 | DRSK1834 |
| 2114.0 | DRSK1835 |
| 2115.0 | DRSK1836 |
| 2116.0 | DRSK1837 |
| 2117.0 | DRSK1838 |
| 2118.0 | DRSK1839 |
| 2119.0 | DRSK1840 |
| 2120.0 | DRSK1841 |
