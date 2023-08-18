# Motivation

## Why is `piel` a microservice?

[microservices.io](https://microservices.io) says:

> Microservices - also known as the microservice architecture - is an architectural style that structures an application as a collection of services that are:
> * Independently deployable
> * Loosely coupled
> * Organized around business capabilities
> * Owned by a small team
> * The microservice architecture enables an organization to deliver large, complex applications rapidly, frequently, reliably and sustainably.

My view is that there are some incredible open-source projects related to electronic and photonic design that have been developed for some time, and are in active development. However, what has been missing is a way to get them talking to each other in order to enable a design flow functionality that leverages the best aspects of all tools together. `piel` is meant to be the fabric that enables the interconnectivity and design flow in between multiple projects.

If this design flow is useful for others who use other tools, because this is a microservice and it is so decoupled, it is very easy for others to extend functions for their own flows. This is trying to leverage the strengths of multiple open-source projects to achieve a design flow better than proprietary co-design tools - only dependent on the further functionality of each individual project.

Another thing `piel` aims to provide is a resolved environment for using all these tools together, which is nontrivial if you have ever dealt with dependency conflicts.

## Why is it called `piel`?

`piel` is an acronym of **P**hotonic **I**ntegrated **EL**ectronics tools. `piel` in Spanish also means skin, and in a funny way of thinking, it aims to compose together a body of electronic and photonic design tool projects.

## Open-Source Motivation

Standing on the shoulders of open-source giants.

As scientists, not as software developers, sometimes we have a knack for reinventing the (software) wheel in our projects whilst not creating interfaces for utility connectivity of the work we do. I see a tendency to focus on solving a specific circumstantial problem, rather than allowing others to clearly understand the context and issue, let the knowledge from multiple disciplines contribute to a good solution, and generalise the functionality of our toolset to achieve more in the future.

So much understanding and information is lost in closed-source modelling systems, in proprietary badly written software that does not enable us to debug them, or even care to sort out our problems when we have a massive chip design deadline. Open-source software frees us from the chains that hold back our innovation of the future. It allows us to make things better, to have good discussions openly, to improve the tools we all use, to work together as a community, and to get different people involved. We can direct our field of research in a meaningful direction towards where we want to go - not where a monopolistic multinational company can get most profit.

This project aims to be a set of easy functions that allow interconnecting and getting the most out of existing tools to design electronics and photonics. It aims to provide as much easy connectivity between amazing open-source projects that have been developed for a very long time. It aims to provide interfaces that could be extended to integrate closed-source design tools to be useful to more than a few existing teams with existing microelectronic, and photonic design flows.
