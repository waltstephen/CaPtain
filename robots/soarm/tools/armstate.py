"""Read-only state of the SO-101 bus (no torque change): position, torque enable (reg 40), temperature per servo."""
import json

import scservo_sdk as s

port, pk = s.PortHandler("/dev/ttyACM0"), s.PacketHandler(0)
assert port.openPort() and port.setBaudRate(1000000)
print(json.dumps({i: {"pos": pk.read2ByteTxRx(port, i, 56)[0], "torque": pk.read1ByteTxRx(port, i, 40)[0],
                      "temp": pk.read1ByteTxRx(port, i, 63)[0]} for i in range(1, 7)}))
port.closePort()
