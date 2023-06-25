# SAX Co-Design

## Implementation Principle

The methodology of interconnection between photonics and electronics design can be done in the time and frequency domain. However, one of the most basic simulation implementations is determining how an electronic system implements a photonic operation. This means, for a given mapping between an electronic signal to a photonic one, how does the full photonic system change?

This is where frequency domain solver tools like `sax` come into play for photonics.

One pseudo electronic-photonic simulation currently available has been demonstrated in [PhotonTorch 09_XOR_task_with_MZI](https://docs.photontorch.com/examples/09_XOR_task_with_MZI.html). We want to extend this type of functionality into a co-design between electronic and photonic tools.
