import time
from gtts import gTTS
from se_openhw.kit.nano.hat import EnvironmentalSensor
from se_openhw.kit.nano import Speaker
from playsound import playsound
playsound("wav/startup.wav")

sensor = EnvironmentalSensor()
sp = Speaker()

playsound("wav/measurement.wav")

temperature, gas, relative_humidity, pressure, altitude = sensor.read()

temperature_string="气温是 %0.1f 摄氏度,    " % temperature
gas_string="挥发性有机物浓度为 %d 欧姆,    " % gas
hum_string="相对湿度为 %0.1f %%,    " % relative_humidity
pre_string="气压为 %0.3f 百帕,    " % pressure
alt_string="海拔为 %0.2f 米.    " % altitude

is_temp_normal=bool(temperature>21 and temperature<27)
is_gas_normal=bool(gas>9990 and gas<11000)
is_rh_normal=bool(relative_humidity>40 and relative_humidity<65)
is_pre_normal=bool(pressure>990 and pressure<1100)

if is_temp_normal and is_gas_normal and is_rh_normal and is_pre_normal:
    is_normal_string="环境符合生产标准！ "
else:
    is_normal_string="环境不符合生产标准！"
    if is_gas_normal==False:
        is_normal_string+="挥发性有机物不在标准范围！"
    if is_temp_normal==False:
        is_normal_string+="温度不在标准范围！"
    if is_rh_normal==False:
        is_normal_string+="相对湿度不在标准范围！"
    if is_pre_normal==False:
        is_normal_string+="气压不在标准范围！"

print("Temperature: %0.1f C" % temperature)
print("Gas: %d ohm" % gas)
print("Humidity: %0.1f %%" % relative_humidity)
print("Pressure: %0.3f hPa" % pressure)
print("Altitude = %0.2f meters" % altitude)

speak_string=temperature_string+gas_string+hum_string+pre_string+alt_string+is_normal_string
print(speak_string)
tts=gTTS(text=speak_string, lang='zh-cn')
print("generate tts")

tts.save(r'speak.wav')
print("generate wav")

playsound(r'speak.wav')