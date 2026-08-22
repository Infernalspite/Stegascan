# Research Foundations (Phase 0)

Every heuristic in this project maps to a documented, real attack
technique or a real published defense, rather than an invented one.
This is the grounding work done before any detection code was written.

## Attack side: steganographic malware in weights

**EvilModel** (Wang, Liu & Cui, IEEE ISCC 2021) demonstrated that
malware can be hidden inside a trained neural network by targeting
redundant weights and overwriting them with malware bytes — a 36.9 MB
malware sample embedded inside a 178 MB AlexNet model with ~1% accuracy
loss, undetected by antivirus engines because the file *structure*
never changes.

**EvilModel 2.0** (Wang et al., Computers & Security 2022) extended
this with MSB reservation, fast substitution, and half substitution,
embedding payloads approaching half the model's size, validated across
550 models and nineteen real malware samples.

**StegoNet** (Liu et al., ACSAC 2020) independently modified
least-significant bits of weights and proposed trigger mechanisms to
activate a hidden payload later.

**"Steganographic Capacity of Selected/Deep Learning Models"** (arXiv
2306.17189, 2308.15502) overwrites the LSBs of *every* weight across a
layer rather than hand-picking redundant ones — this full-tensor
pattern is exactly what `core/scan_steganography.py`'s bit-depth sweep
is built to find.

## Attack side: backdoors / trojans

**BadNets** (Gu, Liu, Dolan-Gavitt & Garg, IEEE Access 2019) established
the foundational threat model: a network that performs at
state-of-the-art accuracy on the user's own validation data, yet
behaves arbitrarily badly on attacker-chosen trigger inputs — including
a backdoored street-sign classifier that misidentified stop signs as
speed-limit signs when a small sticker was present, directly analogous
to this project's corner-pixel trigger on a digit classifier.

## Defense side: what real backdoor detection looks like

- **Neural Cleanse** (Wang et al., IEEE S&P 2019) reverse-engineers a
  minimal trigger per output label, on the insight that a genuinely
  backdoored label needs a much smaller, more consistent perturbation
  to force than any legitimate label — `core/scan_backdoor.py`'s
  brute-force patch grid is a hackathon-scale simplification of this
  same core idea.
- **Spectral Signatures** (Tran, Li & Madry, NeurIPS 2018) showed
  poisoned training examples leave a trace in the top singular vectors
  of a network's learned feature-representation covariance.
- **Activation Clustering** (Chen et al., 2018) clusters penultimate-layer
  activations; poisoned and clean inputs separate into different
  clusters because the backdoor uses a different internal pathway.
- **STRIP** (Gao et al., ACSAC 2019) is a runtime defense: superimpose
  an incoming input onto known-clean images and measure prediction
  entropy — a triggered input dominates regardless of the blend,
  producing abnormally low entropy.
- **Fine-Pruning** (Liu, Dolan-Gavitt & Garg, RAID 2018) is a
  mitigation, not a detector: prune neurons dormant on clean inputs,
  then fine-tune, removing the backdoor while preserving accuracy.

## References

1. Wang, Z., Liu, C., & Cui, X. (2021). *EvilModel: Hiding Malware
   Inside of Neural Network Models*. IEEE ISCC. arXiv:2107.08590.
2. Wang, Z. et al. (2022). *EvilModel 2.0: Bringing Neural Network
   Models into Malware Attacks*. Computers & Security, 120, 102807.
3. Liu, T. et al. (2020). *StegoNet: Turn Deep Neural Network into a
   Stegomalware*. ACSAC, 928–938.
4. Baird, A. et al. (2023). *Steganographic Capacity of Deep Learning
   Models*. arXiv:2306.17189, arXiv:2308.15502.
5. Gu, T., Liu, K., Dolan-Gavitt, B., & Garg, S. (2019). *BadNets:
   Evaluating Backdooring Attacks on Deep Neural Networks*. IEEE
   Access, 7, 47230–47244.
6. Wang, B. et al. (2019). *Neural Cleanse: Identifying and Mitigating
   Backdoor Attacks in Neural Networks*. IEEE S&P, 707–723.
7. Tran, B., Li, J., & Madry, A. (2018). *Spectral Signatures in
   Backdoor Attacks*. NeurIPS 31, 8000–8010.
8. Chen, B. et al. (2018). *Detecting Backdoor Attacks on Deep Neural
   Networks by Activation Clustering*. arXiv:1811.03728.
9. Gao, Y. et al. (2019). *STRIP: A Defence Against Trojan Attacks on
   Deep Neural Networks*. ACSAC, 113–125.
10. Liu, K., Dolan-Gavitt, B., & Garg, S. (2018). *Fine-Pruning:
    Defending Against Backdooring Attacks on Deep Neural Networks*.
    RAID.
