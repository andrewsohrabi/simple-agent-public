# MEMO-P01-660 Rev A: Gamma Curve Consistency Assessment

## Metadata
- Document ID: MEMO-P01-660
- Revision: A
- Prefix: MEMO
- Latest revision: True
- Signed: True
- Obsolete: False
- Software version: unknown
- Source filename: MEMO-P01-660 Gamma Curve Consistency Assessment_A-signed.docx
- Source path: Example QMS - MedAI/MEMO-P01-660 Gamma Curve Consistency Assessment_A-signed.docx
- Extraction warnings: none

## Extracted Content
PURPOSE
The purpose of this study is to demonstrate the diagnostic display performance of the MedAI Device Application v4.1.0 is consistent with the diagnostic display performance of the MedAI Device Application v1.1.3 previously validated by Image Quality Labs (See 3P-P01-21 Rev B.) by comparing the application of the gamma curve correction in the image processing of  these MX1 software versions and showing equivalency.
OBJECTIVE AND SCOPE
Maintaining consistency in image processing algorithms across software versions is crucial for ensuring the accuracy and reliability of output images. The gamma curve adjustment, in particular, is essential for achieving correct luminance levels in processed images.
The primary objective of this study is to verify that the application of the gamma curve adjustment algorithm to test images produces consistent results after moving this processing from tablet (v1.1.3) to MX1 device (v4.1.0)
The scope for this study is limited to the verification of equivalence of the brightness of identical test images processed only by Software system v1.1.3 and Software system v4.1.0
MATERIALS
Tablet Application:
v1.1.3: example.com/ Use 1.1.3 tag
v4.1.0: example.com/ Use 4.1.0 tag
MX1 Software:
example.com/ (S10008 - MedAI Device App (ODA) - Android) Use tags 1.1.3 (git hash: 0d5867e6e26b4653abe0948802d36b366564a19d) and 4.1.0 (git hash: b3ec9fae913803155fd28ab34727e4f0aaa7f277)
Test Image:
xray_00.tif
METHODS
The evaluation process consisted of a practical test using real-world conditions. Version v1.1.3 of the software was installed on a Galaxy S9 Tablet, and a mock X-ray image was processed through the MX1 Software System. The output image was then downloaded for analysis. This procedure was identically replicated using version v4.1.0 of the software on the same device.
Matrices of pixel brightness values of each image were dumped into a text file to quantify and verify the consistency in the gamma curve application between the two versions. Side-by-side comparisons of the contents of the result files were then performed.
A special test image was created to make the process of validation and interpretation of the results easier. It represents a repeating set of 1 pixel wide lines where the brightness of each line goes from 0x000000 to 0xFFFFFF. (see Fig. 1).
Figure 1. Test image
TEST PROCEDURE
v1.1.3  image and pixels dump acquisition:
Place test image under /home/imager/test/savedXrays folder on the cassette
Run MX1 backend and disable device side DICOM calibration by executing command: curl "http://<cassette’s_ip_or_hostname>:8081/set_dicom_tablet?type=none"
Run v1.1.3 of ODA on Galaxy S9 Tablet
Initiate mock x-ray acquisition by executing command: curl "http://<cassette’s_ip_or_hostname>:8081/remote_api?command=trigger_press&press_time=100"
Connect tablet to dev machine so that its filesystem can be explored and copy <capture_guid>-gamma and <capture_guid>-pixels files from /data/data/com.medai.app/app_flutter folder on the tablet
capture_guid can be found in tablets logs of in MX1’s database after execution following query: SELECT guid FORM capture;
v4.1.0 image and pixels dump acquisition:
If test image not yet copied repeat Step #1 from previous paragraph
Run MX1 backend and enable DICOM adjustment gamma for Galaxy S9 Tablet displays  by executing command: curl "http://<cassette’s_ip_or_hostname>:8081/set_dicom_tablet?type=galaxy"
Run v4.1.0 of ODA on Galaxy S9 Tablet
Repeat Steps #4-6 from previous paragraph
Results analysis
In pixels dump files acquired from both versions of ODA verify that there are no values exceeding minimum and maximum values of adjusted gamma (see Table 1.).
Compare two pixel dump files using your favorite tool to identify equivalence or differences of adjusted brightness values.
Android Studio offers files comparison feature with visual highlighting of differences if any.
There is also command line tool called diff which compares provided files line by line and prints differences to STDOUT
Table 1. Updated Tone Map
RESULTS & VALIDATION
Figure 2. Result image with adjusted gamma copied from ODA v1.1.3
Figure 3. Result image with adjusted gamma copied from ODA v4.1.0
Figure 4. Screenshot of compare view of Android Studio indicating no differences in files of pixel-by-pixel dump from v1.1.3 and v4.1.0
CONCLUSION
The code analysis and image comparison data confirm that the fundamental image processing approach, particularly the gamma curve adjustments, has been consistently maintained from software version v1.1.3 to v4.1.0.
ATTACHMENTS
MEMO-P01-660 Attachment 1: Test Image xray_00.tif
MEMO-P01-660 Attachment 2: Result image with adjusted gamma copied from ODA v1.1.3 - 113_tablet.jpg
MEMO-P01-660 Attachment 3: Result image with adjusted gamma copied from ODA v4.1.0 - 410_galaxy_tablet.jpg
MEMO-P01-660 Attachment 4: Pixel-by-pixel dump of adjusted image from ODA v1.1.3 - 1_1_3.txt
MEMO-P01-660 Attachment 5: Pixel-by-pixel dump of adjusted image from ODA v4.1.0 - 4_1_0.txt
DOCUMENT REVISION HISTORY
Digital Key: example.com/

### Table 1
| Original Brightness Value | Updated Brightness Value |
| --- | --- |
| 0 | 20 |
| 15 | 27 |
| 30 | 34 |
| 45 | 42 |
| 60 | 50 |
| 75 | 59 |
| 90 | 68 |
| 105 | 78 |
| 120 | 88 |
| 135 | 101 |
| 150 | 113 |
| 165 | 127 |
| 180 | 144 |
| 195 | 161 |
| 210 | 181 |
| 225 | 202 |
| 240 | 224 |
| 255 | 252 |

### Table 2
| Rev | Description of Change | Approved By | Approval Date: | DCO # |
| --- | --- | --- | --- | --- |
| A | Initial Release | Quality Engineering Engineering Regulatory Affairs | 17 Feb 2025 | 25-097 |
