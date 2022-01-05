# ContextTransitionPredictability
The source codes for computing context transition predictability.

contextTransitionPredictability.py is the file that calculates context transition predictability based on the input file.
entropyCompute.py aims to calculate the relevant entropy and predictability.

The input data format is: [userid, locatonid_1@timestamp_1, ... , locatonid_n@timestamp_n].
The output file format is: [userid, entropy or predictability].

For example:

Input content:
1462,4b19f917f964a520abe623e3#Train Station@1333529414000,
4b19f917f964a520abe623e3#Train Station@1333549777000,
4b283516f964a520e19024e3#Train Station@1333550869000,
4b19f917f964a520abe623e3#Train Station@1333565041000,
4bbac8b753649c742f7249fb#Office@1333568100000,
4b600990f964a520e3d329e3#Train Station@1333568978000,
4b4aa852f964a520678c26e3#Train Station@1333614770000,
4b283516f964a520e19024e3#Train Station@1333615697000,
4b283516f964a520e19024e3#Train Station@1333647322000,
4b55c776f964a520a2ef27e3#Train Station@1333873221000,
4b4a79ccf964a520bc8826e3#Train Station@1333873488000

Context Transition Predictability output content:
1462,0.885632226078601



