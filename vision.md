# Vision

## The Bigger Idea

The experiment plan tests a specific hypothesis about learned segmentation and PHM layers on three domains. But the underlying idea is broader than that.

The intuition: intelligence doesn't scale by being faster or having more memory. It scales by thinking in more dimensions. A flatworm processes its world in a few dimensions — light/dark, chemical gradient, touch. A human processes the same world in thousands of dimensions simultaneously — spatial, temporal, semantic, emotional, social, causal, counterfactual. The difference isn't speed. It's the richness of the representation.

Current AI models are more like the flatworm than we admit. Every token is a flat vector. The model is fast and has enormous memory, but its per-token representation is structurally impoverished. Making the vector longer (d=512 to d=4096) adds more features but doesn't add structure — it's still a flat list. A 4096-dimensional flatworm is still a flatworm.

What if the right scaling axis isn't making the vector longer, but making the representation deeper — matrices, tensors, algebraic objects with internal structure? And what if the right input pipeline isn't a hand-designed tokenizer that assumes we know what the atoms are, but a learned system that discovers the atoms from raw data?

## Inputs and Outputs

The architecture should be agnostic to what it processes. Define the input, define the output, and the model figures out the middle. This means:

- **Text:** Input = raw UTF-8 bytes. Output = predicted next bytes.
- **Images:** Input = raw pixel bytes (RGB values). Output = predicted next pixels.
- **Audio:** Input = raw PCM samples as bytes. Output = predicted next samples.
- **Robot proprioception:** Input = raw sensor readings (joint angles, forces, IMU). Output = predicted next sensor readings.
- **Proteins:** Input = amino acid sequences as bytes. Output = predicted next amino acids.
- **Mathematics:** Input = formal expressions as byte streams. Output = predicted next bytes.
- **Physical systems:** Input = measurements (positions, velocities, fields) as byte streams. Output = predicted next measurements.

In every case, the model sees raw bytes. It learns to segment them. It processes them through structured representations. It predicts what comes next. The objective is always the same: minimize surprise. Shannon's insight: prediction and compression are the same thing. A model that predicts well understands the structure of its input.

The three-domain experiment (text, images, audio) is the minimum viable test. But the architecture isn't limited to three domains. If it works on three, there's no architectural reason it wouldn't work on thirty. You'd just feed it more byte streams.

## Structured Outputs

The current experiment predicts flat byte probabilities — a vector of 256 values (one per possible next byte). But if the internal representation is a matrix, why is the output a vector?

A future version could predict structured outputs:
- Instead of "what is the next byte?" → "what is the next 8x8 block of bytes?" (a matrix-valued prediction)
- Instead of predicting one token at a time → predicting a structured chunk (a tensor of future bytes with confidence for each)
- For robot control: output a matrix representing joint torques across multiple timesteps simultaneously

This is conceptually close to multi-token prediction (Meta, 2024) but richer. MTP predicts N independent next tokens. Structured output prediction would predict an N-dimensional object where the dimensions interact — the prediction for byte 3 depends on the prediction for byte 7 because they're part of the same structured chunk.

This is Phase 5 territory. The initial experiment sticks with flat byte prediction to isolate the input/representation variables.

## The Algebra Discovery Question

PHM layers learn their multiplication rules from data via Kronecker products. This means the model discovers its own algebra — its own rules for how dimensions interact within each token.

Quaternion algebra couples 4 dimensions with specific rules (i*j=k, j*k=i, k*i=j). The model might discover quaternion-like rules, or something completely different. The learned algebra might:

- Be the same across all layers (a universal computational primitive)
- Differ by layer depth (early layers use simple algebra, deep layers use complex algebra)
- Differ by domain (the model uses different math for text vs images — discovered, not designed)
- Change over training (starting simple, growing more complex as the model learns)

Analyzing what algebra the model discovers is a research contribution independent of whether the model achieves good loss. If the model converges to quaternion-like rules, that tells us something about the structure of the problems it's solving. If it discovers something novel, that's potentially a mathematical insight, not just an engineering one.

## The Co-Evolution Feedback Loop

The segmenter decides what the atoms are. The PHM layers decide what algebra to use on those atoms. These two systems are trained jointly — they co-evolve.

The segmenter learns to create chunks that are useful for the PHM algebra to process. The PHM algebra learns rules that work well for the chunks the segmenter creates. Neither system alone determines the representation. The representation emerges from their interaction.

This is different from every existing architecture:
- Standard transformers: fixed tokenizer, fixed algebra (real-valued linear)
- BLT: learned tokenizer, fixed algebra (real-valued linear)
- PHM transformers: fixed tokenizer, learned algebra
- This project: learned tokenizer, learned algebra

The co-evolution might produce something neither component would discover alone. The segmenter might learn to create chunks that have algebraic structure (e.g., byte patterns that naturally form quaternion-like groups). The algebra might learn rules that make certain segmentations more useful than others. This is the part we can't predict — it's emergent.

## Connection to the Brain

The brain does something like co-evolution between segmentation and representation:

- Sensory cortex learns to segment raw input into useful chunks (edge detection, phoneme detection, object segmentation). These aren't pre-programmed — they develop through experience and differ across individuals.
- The same cortical column architecture processes these chunks using structured computation. The computation isn't flat — it involves oscillatory binding (theta-gamma coupling), dendritic branching (multiple nonlinear computations per neuron), and multi-scale grid cell encoding.
- Mountcastle's uniformity principle: the same algorithm everywhere. The cortex is one repeated module that learns different representations from different inputs.

The brain's segmentation and computation co-evolve during development. A child learning to read develops letter-level and word-level segmentation in visual cortex simultaneously with semantic representations in temporal cortex. Neither precedes the other — they bootstrap each other.

Our experiment is a simplified computational version of this developmental process. If it works at 50M parameters on three domains, the question becomes: what happens at 500M parameters on thirty domains? Does the co-evolution produce increasingly sophisticated domain-specific strategies from a single domain-general architecture?

## The Hardware Future

Current GPUs do one thing well: multiply matrices of real numbers. This is why flat-vector transformers dominate — the hardware is optimized for exactly that computation.

Photonic tensor processors can do something GPUs can't: native matrix-matrix operations in a single physical step. A 32x32 matrix-valued token could be multiplied by a 32x32 weight matrix in one optical propagation, using wavelength multiplexing for rows and spatial multiplexing for columns. Lab demonstrations exist (Nature Photonics, 2025) at 34 TOPS/mm^2 with O(N^3) parallelism.

If the learned-representations experiment shows that structured representations are worth the compute cost, the natural next step is hardware that makes them cheap. The software experiment validates the idea. The hardware makes it practical.

But that's years away. For now, standard GPUs handle PHM layers just fine — they decompose into batched matrix multiplications that tensor cores execute efficiently. The experiment is bottlenecked by ideas, not hardware.

## Beyond This Experiment

If the experiment works — if learned segmentation + structured representations + multi-modal training produces emergent domain-specific behavior — the natural follow-ups are:

1. **More domains:** Add code, mathematics, protein sequences, robot sensor data. See if the architecture continues to differentiate without domain-specific engineering.

2. **Richer neurons:** Replace the standard neuron (weight * input + bias → activation) with something more brain-like — KAN edge functions, dendritic branching, complex-valued oscillatory dynamics. See if richer per-neuron computation complements the richer per-token representation.

3. **Structured outputs:** Predict matrices or tensors instead of flat byte probabilities. See if the model can output structured objects that capture multi-dimensional predictions simultaneously.

4. **Embodiment:** Connect the model to a simulated robot. Sensor inputs as raw byte streams, motor outputs as predicted byte streams. The same architecture that processes text and images now processes proprioception and generates actions. No separate policy network, no reward function — just prediction.

5. **Scale:** If all of the above works at 50-200M parameters, what happens at 1B? At 10B? Does the co-evolution between segmentation and algebra produce qualitatively different behavior at scale?

These are years of research. The three-domain byte-level PHM experiment is week one.
