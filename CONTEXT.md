# Earshot

A swarm of cheap drones that perch on flat roofs to listen, fuse sound and RF energy into a shared probability map of threats, and decide among themselves who flies out to verify with a camera. The same swarm guards an airport against drones and a game reserve against poachers: a Scenario names its Threat Class.

## Swarm

**Drone**:
One swarm member: a body (in Webots or the 2D twin) plus its own Agent.
_Avoid_: node, unit, robot

**Agent**:
The decision-making brain of exactly one Drone; sees only its own sensors and the radio.
_Avoid_: controller, bot

**Mode**:
What a Drone is doing right now: Patrol, Listen, Verify or Fill Gap.
_Avoid_: state, behaviour

**Patrol**:
Flying its Sector; nearly deaf to sound, hears RF.

**Listen**:
Landed on a Perch Site with motors off; hears sound with a bearing.
_Avoid_: perch mode, hover-listen

**Verify**:
Flying to a Suspicion with the camera on.
_Avoid_: investigate, inspect

**Fill Gap**:
Covering part of a neighbour's vacated Sector.

**Verifier**:
The Drone currently in Verify; the only one rendering camera images.

**Follow**:
A Verifier staying with the threat it confirmed, camera on, so it stays Confirmed and the Threat Log gets its path; ends when the threat is lost or after `follow_for` seconds. It faces its Quarry and matches its speed, closing in when it falls behind. Still Mode Verify.
_Avoid_: chase, pursue, track (a Track is a fused Suspicion), lock (that is a Claim)

**Quarry**:
What a Follower knows of the threat it follows: position and velocity, from its own camera sightings and the Confirmed Track, coasting between them.
_Avoid_: target (a Source), track

**Guide**:
A demo mock (`guide: true`, `scenarios/muc_demo.yaml`): a Verifier in flight is told where its intruder really is, flies straight to it, confirms and follows. Detection stays honest; this is the one place truth reaches an Agent, never in an evaluation.
_Avoid_: oracle, cheat

**Sector**:
The area a Drone is responsible for while patrolling; re-partitioned when a neighbour leaves.

**Gap**:
A Sector left uncovered because its Drone went to Verify.

**Listener**:
A Drone in Listen whose bearings help localise a particular Suspicion.

**Array Drone**:
A Drone permanently perched on one Perch Site whose sound comes from a real microphone array in the room.
_Avoid_: fixed node, static sensor

**Perch Site**:
A known flat roof from the 3D city model where a Drone may land to Listen; in a Reserve, a lookout tower's platform or a camp roof.
_Avoid_: landing spot, roof (alone)

## World

**City**:
The static prior knowledge every Drone carries: buildings, Perch Sites, rail lines, roads.
_Avoid_: map (ambiguous with Probability Map)

**Reserve**:
A City that is a protected wildlife area (tools/make_savanna.py): dirt tracks instead of streets, lookout towers and camps for Perch Sites, kopjes, thickets and waterholes, game instead of fans.
_Avoid_: park (ambiguous), savanna (that is the biome, and the Scenario's name)

**Source**:
Something that emits sound and possibly RF: a drone, heavy vehicle, train or animal. Ground truth; Agents never see it. Its `kind` says what it is ("elephant herd", "poacher pickup"); an intruder is a Source the swarm should find.
_Avoid_: target, emitter, threat (a Source is only a threat if it is an intruder of the Threat Class)

**Poacher**:
An intruder Source in a Reserve: a vehicle off the tracks, its crew often on a UHF handheld radio.

**Shot**:
A rifle fired in a Reserve: an Event, not a Source. Every tower within a few km hears it once, with a sharp bearing, as Evidence for the Threat Class where it was fired; there is nothing to see, find or follow.
_Avoid_: gunshot source, bang

**Beam**:
What a sound classifier is given for one Source in one 1 s window: that Source's sound, every other Source 6 dB down, and the background and ego-noise. Rendered by the World from waveforms; Agents never see it.
_Avoid_: mix (that is everything the microphone gets, unsteered)

**Nuisance**:
A Source whose true Target Class is Nothing but which sounds like a threat (a fan, an AC unit; in a Reserve the game: an elephant's rumble, a lion's roar).

**Scenario**:
A seeded script of Sources, Jam Zones and Faults for one run.

**Event**:
A change to a running Scenario at a sim time: scripted in the Scenario or sent by the operator from the dashboard.
_Avoid_: hotkey (that is only how an operator sends one), action

**Jam Zone**:
An area where the radio delivers nothing.

**Fault**:
A scripted sensor defect on one Drone (stuck bearing, deaf, phantom detections).

## Belief

**Probability Map**:
One Drone's grid over the City holding, per cell, the probability of each Target Class.
_Avoid_: heatmap, belief map, situational picture

**Target Class**:
Drone, Vehicle, Train or Nothing.

**Threat Class**:
The Target Class a Scenario's swarm hunts: Drone at the airport, Vehicle in a Reserve. Only its Tracks are Suspicions; where the City explains it (a vehicle on a track), it is Explained.
_Avoid_: target type

**Evidence**:
One raw reading in an Observation (a bearing, an RF level, a camera sighting); fused additively, exactly once: its Observation is accepted once per (origin, seq).
_Avoid_: measurement (when meaning the radio message)

**Suspicion**:
A region of the Probability Map worth acting on; broadcast as a Track. Overlapping Suspicions count as one danger, at their likelihood-weighted mean.

**Track**:
A fused Suspicion with location, uncertainty ellipse, class and status. State, not Evidence: a newer Track replaces the older one.

**Ghost**:
A Suspicion where unrelated evidence happens to cross (overlapping RF rings, bearing lines from different Sources), with no Source behind it.
_Avoid_: false positive (that is the outcome of verifying a Ghost)

**Threat Log**:
The operator's record of every danger a Verifier checked: what the camera recognised, when, by which Drone, and the path while it is followed. Built by the runners from decision events and camera labels; Agents never read it.
_Avoid_: alert list, incident log

**Explained**:
A Suspicion best accounted for without a new Source: by City knowledge (rumble on the rail line is a train), or as a side lobe of a Confirmed drone (a drone peak within 150 m of a confirmed point, other than the one that point follows). Nobody verifies it.

**Faulty Sensor**:
The hypothesis that a Drone's own sensor, not the world, explains its reports.

## Radio

**Observation**:
The radio message carrying one Drone's Evidence from one sensor window: up to two readings, sent together.

**Claim**:
A Drone's bid to take a task, valid until its lease expires.
_Avoid_: lock, reservation

**Digest**:
A summary of which messages a Drone holds per origin, used to heal after a partition.

**Request**:
A Drone asking a neighbour for the messages a Digest showed it is missing.

**Message Store**:
Every frame a Drone has accepted, at most once; doubles as duplicate filter and healing buffer.

**Hop**:
One retransmission by a forwarding Drone.

**Partition**:
The swarm split into groups that cannot hear each other (distance or Jam Zone).
