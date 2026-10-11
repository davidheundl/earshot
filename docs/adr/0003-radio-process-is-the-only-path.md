# Agents talk only through the radio process

Agents never exchange anything directly: every byte between Drones is a frame pushed to the radio process, which decides delivery from range, shadowing, half duplex, collisions, airtime budget and Jam Zones (the same `RadioMedium` the lockstep runner uses). The radio is clocked by the world's state snapshots. We chose ZeroMQ (PUB/SUB + PUSH/PULL on localhost) over raw UDP for topic filtering and no framing code. This is what makes partition, healing and channel-load claims in the pitch honest.
