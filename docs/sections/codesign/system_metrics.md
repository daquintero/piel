# System Metrics

## Power Analysis

Let us first begin considering the digital design metrics that are important for us to understand the electrical operation characteristics of mixed electronic-photonic systems. Most of these are redefined to include photonic loads based on *Digital Integrated Circuits, A Design Perspective* by Jan Rabaey. Page numbers are provided accordingly.

### Power Consumption Definitions

#### Peak Power

A design can have a peak power $P_{peak}$ which basically involves the maximum power consumption possible by the total photonic and electronic system. If we have a single supply voltage to our system $V_{supply}$, then we can define it as:

$$
P_{peak} = i_{peak}V_{supply} = \text{max}\left(p(t)\right)
$$

If we have $N$ multiple supply voltages to our system, which is more likekly the case in a mixed-signal digital and analogue supplies, and potentially another external photonic electrical supply, then we can define the total peak power of the system as:

$$
P_{peak,possible} = \sum_0^N i_{peak,N}V_{supply,N} = \sum_0^N \text{max}\left(p_N(t)\right)
$$

In reality, we need to think of the maximum power that can be drawn by the maximum power consuming operation, which may not involve all supplies operating at the maximum draw. However, if you want a conservative estimate, you can assume that is it possible that all supplies are operating at their maximum current draw $i_{peak}$. An generic maximum power defined as the highest consuming state of the system, where not all supplies are operating at their maximum $i_{peak}$ power can then be defined dependent on an consumption efficiency parameter $\eta_N$ for the highest operation load:


$$
P_{peak,operation} = \sum_0^N \eta_N i_{peak,N}V_{supply,N} = \text{max} \left( \sum_0^N \left(p_N(t)\right) \right)
$$

#### Average Power

What may be more likely is that you might be operating this integrated electronic-photonic system with a set of instructions over a long period of time. A set of example of this would be through encoding communication channels, or arbitrary unitary operations, sensing a  sample, etc. In this case, there is a power consumption over a period of time. We can describe this in terms of the average power consumption of the whole system over a period of time $T$:


$$
P_{average} = \int_0^T p(t) dt
$$

If we have multiple supplies, as is likely to be the case, then we can consider it to be:

$$
P_{average} = \sum_0^N \frac{V_{supply,N}}{T} \int_0^T i_{supply,N} (t) dt
$$

### Power Consumption Sources

#### In terms of photonic loads

We can decompose this in terms of dynamic and static power. When some transistors are switching, they are consuming dynamic power that they do not consume when they are in an idle state. This applies similarly to a photonic load. A resistive photonic load such as a thermo-optic heater is consuming power in an idle state constantantly whenever any signal is being applied. A capacitive load such as a electro-optic carrier depletion phase shifter gets charged and discharged according to the voltage that is applied and the total power consumption is dependent on the total switching events. The same could be said for the thermo-optic phase shifters in terms of a PWM-based modulation and so on. It is just important to understand the sources of power consumption and heat dissipation in our circuits in order for us to be able to optimise for it.

We know we want to minimise the static power consumption of the circuit on the idle state, which means that resistive loads such as heaters are no-gos in terms of VLSI-photonics without a suitable cooling solution.

In terms of dynamic loads, we know that the more switching events we have, and more components we have, the higher the total power consumption.

Let us evaluate how our circuit operates for these devices. In terms of a carrier depletion modulator, we can consider the electrical connection as some resistive wire and the junction load to be a capacitor. We can describe this from first-principles as a basic RC circuit, with the following relationship:

$$ V_{out,RC}(t) = (1-e^{-t/\tau}) $$

Our time constant $\tau = RC$. And we know that $50\% V_{out,RC} = 0.69\tau$ and $90\% V_{out,RC} = 2.2\tau$ based on Eq. 1.13, page 34 on Rabay.

Energy input from signal source to charge a capacitor, independent of
series resistance R, although this determines rise times.
$$E_{in} =  \int_{0}^{\infty} i_{in}(t) v_{in}(t) dt = V \int_{0}^{\infty} C \frac{dV_{out}}{dt} dt = (CV) \int_{0}^{V} dV_{out} = CV^2$$


During charge-up, the energy stored in resistor is:
$$E_c = \int_{0}^{\infty} i_c (t) V_{out} (t) dt = \int_{0}^{\infty} C \frac{dV_{out}}{dt} dt = C \int_{0}^{V} V_{out} dV_{out} = \frac{CV^2}{2}$$



The other half of the energy gets dissipated in the resistor during
rising edge, and the rest of the capacitor energy gets dissipated on the
falling edge. This means that per transition there is about
$E_{load,loss} = \frac{CV^2}{2}$ of energy dissipation. This does not
account the energy dissipation per stage, but we need to account for it
for a super low power based design. There is also the characterisation
considerations, and corresponding capacitances.

Another important relationship of the $RC$ time constant is also in the driving of the device, when, in a switching event, the driving switch can be considered as a voltage-controlled-capacitor (gate) modulating-resistor (source-drain). This is important because we can see, when we drive a switching event, and when we consider our signal drivers, the effect of their fundamental components on the rest of the circuit.

## Time Analysis

### Signal Propagation Definitions

Our signals will change given that we have control over how we affect our photonic circuit. Say, we define two boundary conditions of our signals in a transition between 10% and 90%. We define the time of change in between these transitions as the rise or fall time depending on the direction of the change. In digital electronics, section 1.3 of Rabay, we call this change the propagation delay of our transitions. Now, this is very important for a range of reasons.

Mainly this has an effect on the speed of our system, and also on the power consumption of the system. It has an important effect on how we design our driving electronics for our photonics loads.

Rabay describes the importance of this definition very well:

> The rise/fall time of a signal is largely determined by the strength of the driving gate, and the load presented by the node itself, which sums the contributions of the connecting gates (fan-out) and the wiring parasitics.

We will explore this definition in the context of our drivers and loads thoroughly. An important relationship worth remembering is that in a simple RC series circuit, it takes $2.2 \tau = 2.2 RC$ to reach the 90% signal transition point.


### Driving, Propagation Delay & Fanout

If we consider each of our modulators, as a load, we must also consider how we are driving them.
