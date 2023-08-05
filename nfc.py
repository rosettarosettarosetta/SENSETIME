from se_openhw.kit.nano.hat import NFC
sensor = NFC()

while True:
    uid = sensor.read_uid()
    if uid is None:
        continue
    print("Found card with UID:", [hex(i) for i in uid])