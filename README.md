# ContextTransitionPredictability

## Paper 

Please cite our paper if you find our work useful for your research:

```tex
@article{zhang2022beyond,
  title={Beyond the limits of predictability in human mobility prediction: context-transition predictability},
  author={Zhang, Chao and Zhao, Kai and Chen, Meng},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```

## The source codes for computing context transition predictability

contextTransitionPredictability.py is the file that calculates context transition predictability based on the input file.<br />
entropyCompute.py aims to calculate the relevant entropy and predictability.

The input data format is: [userid, locatonid_1@timestamp_1, ... , locatonid_n@timestamp_n].<br />
The output file format is: [userid, entropy or predictability].

## For example:

Input content(a partial trajectory extracted from the dataset):<br />
1462,4b19f917f964a520abe623e3#Train Station@1333529414000,<br />
4b19f917f964a520abe623e3#Train Station@1333549777000,<br />
4b283516f964a520e19024e3#Train Station@1333550869000,<br />
4b19f917f964a520abe623e3#Train Station@1333565041000,<br />
4bbac8b753649c742f7249fb#Office@1333568100000,<br />
4b600990f964a520e3d329e3#Train Station@1333568978000,<br />
4b4aa852f964a520678c26e3#Train Station@1333614770000,<br />
4b283516f964a520e19024e3#Train Station@1333615697000,<br />
4b283516f964a520e19024e3#Train Station@1333647322000,<br />
4b55c776f964a520a2ef27e3#Train Station@1333873221000,<br />
4b4a79ccf964a520bc8826e3#Train Station@1333873488000

Context Transition Predictability output given temporal information:<br />
1462,0.885632226078601



