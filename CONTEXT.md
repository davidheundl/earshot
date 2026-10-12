# Earshot

A swarm of cheap drones that perch on flat roofs to listen, fuse sound and RF energy into a shared probability map of threats, and decide among themselves who flies out to verify with a camera.

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
A known flat roof from the 3D city model where a Drone may land to Listen.
_Avoid_: landing spot, roof (alone)

## World

**City**:
The static prior knowledge every Drone carries: buildings, Perch Sites, rail lines, roads.
_Avoid_: map (ambiguous with Probability Map)

**Source**:
Something that emits sound and possibly RF: a drone, heavy vehicle or train. Ground truth; Agents never see it.
_Avoid_: target, emitter, threat (a Source is only a threat if it is a drone)

**Nuisance**:
A Source whose true Target Class is Nothing but which sounds drone-like (a fan, an AC unit).

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

**Evidence**:
A raw Observation; fused additively, exactly once per (origin, seq).
_Avoid_: measurement (when meaning the radio message)

**Suspicion**:
A region of the Probability Map worth acting on; broadcast as a Track. Overlapping Suspicions count as one danger, at their likelihood-weighted mean.

**Track**:
A fused Suspicion with location, uncertainty ellipse, class and status. State, not Evidence: a newer Track replaces the older one.

**Ghost**:
A Suspicion where unrelated evidence happens to cross (overlapping RF rings, bearing lines from different Sources), with no Source behind it.
_Avoid_: false positive (that is the outcome of verifying a Ghost)

**Explained**:
A Suspicion best accounted for by City knowledge (rumble on the rail line is a train); nobody verifies it.

**Faulty Sensor**:
The hypothesis that a Drone's own sensor, not the world, explains its reports.

## Radio

**Observation**:
The radio message carrying one Evidence item.

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
