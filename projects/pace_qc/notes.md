| Bit | Flag Name  | Description                                                                     | L2 Mask Default | L3 Mask Default |
| --- | ---------- | ------------------------------------------------------------------------------- | --------------- | --------------- |
| 00  | ATMFAIL    | Atmospheric correction failure                                                  | ON              | —               |
| 01  | LAND       | Pixel is over land                                                              | ON              | ON              |
| 02  | PRODWARN   | One or more product algorithms generated a warning                              | —               | —               |
| 03  | HIGLINT    | Sunglint: reflectance exceeds threshold                                         | ON              | —               |
| 04  | HILT       | Observed radiance very high or saturated                                        | ON              | ON              |
| 05  | HISATZEN   | Sensor view zenith angle exceeds threshold                                      | ON              | —               |
| 06  | COASTZ     | Pixel is in shallow water                                                       | —               | —               |
| 07  | CLDSHDW    | Pixel is in cloud shadow                                                         | —               | —               |
| 08  | STRAYLIGHT | Probable stray light contamination                                              | ON              | ON              |
| 09  | CLDICE     | Probable cloud or ice contamination                                             | ON              | ON              |
| 10  | COCCOLITH  | Coccolithophores detected                                                       | ON              | —               |
| 11  | TURBIDW    | Turbid water detected                                                           | —               | —               |
| 12  | HISOLZEN   | Solar zenith exceeds threshold                                                  | ON              | —               |
| 13  | spare      | Reserved for future use                                                         | —               | —               |
| 14  | LOWLW      | Very low water-leaving radiance                                                 | ON              | —               |
| 15  | CHLFAIL    | Chlorophyll algorithm failure                                                   | ON              | —               |
| 16  | NAVWARN    | Navigation quality is suspect                                                   | ON              | —               |
| 17  | ABSAER     | Absorbing aerosols determined                                                   | —               | —               |
| 18  | spare      | Reserved for future use                                                         | —               | —               |
| 19  | MAXAERITER | Maximum iterations reached for NIR iteration                                    | ON              | —               |
| 20  | MODGLINT   | Moderate sun glint contamination                                                | —               | —               |
| 21  | CHLWARN    | Chlorophyll out-of-bounds                                                       | ON              | —               |
| 22  | ATMWARN    | Atmospheric correction is suspect                                               | ON              | —               |
| 23  | spare      | Reserved for future use                                                         | —               | —               |
| 24  | SEAICE     | Probable sea ice contamination                                                  | —               | —               |
| 25  | NAVFAIL    | Navigation failure                                                              | ON              | —               |
| 26  | FILTER     | Pixel rejected by user-defined filter OR insufficient data for smoothing filter | —               | —               |
| 27  | spare      | Reserved for future use                                                         | —               | —               |
| 28  | BOWTIEDEL  | Deleted off-nadir pixels                                                        | —               | —               |
| 29  | HIPOL      | High degree of polarization determined                                          | —               | —               |
| 30  | PRODFAIL   | Failure in any product                                                          | —               | —               |
| 31  | spare      | Reserved for future use                                                         | —               | —               |



To exclude during Tchla validation:

| Flag Name  | Bit Number |
| ---------- | ---------- |
| LAND       | 1          |
| HIGLINT    | 3          |
| HILT       | 4          |
| STRAYLIGHT | 8          |
| CLDICE     | 9          |
| ATMFAIL    | 0          |
| LOWLW      | 14         |
| FILTER     | 26         |
| NAVFAIL    | 25         |
| NAVWARN    | 16         |