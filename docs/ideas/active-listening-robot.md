# Idea: An Actively Listening Robot (not part of EXP-0001)

- **Status:** idea, parked.
- **Source:** the author, 2026-10-28.

## Problem

Robots cannot react to a speaker in real time with nods, smiles, frowns or "mhm" at the right moments. The right reaction depends on two things:
- **Slow context:** what is being said and how the conversation is going.
- **Fast cues:** prosody, pauses, pitch and energy changes, gaze, which unfold in the order of 100 ms.

A large model understands the context but is too slow to place a nod in a pause.

## Mapping onto this project's architecture

| | Slow model | Fast model |
|---|---|---|
| Input | Chunks of the conversation (transcript + audio), maybe vision | Microphone features (energy, pitch, voice activity, pause length), maybe face/gaze, at ~50–100 Hz |
| Output | Speech when the robot talks, plus a **conversation-context latent** (topic, sentiment, whether the speaker expects agreement, emotional tone) | Backchannel actions at high rate: head nods, facial expression, short vocalisations, gaze shifts |
| Rate | ~1 Hz or per utterance | ≥ 20 Hz |

The fast model does **not** need to work out the emotional state of the conversation. It gets that from the latent and only times and shapes the reaction from low-level cues.

## Questions shared with EXP-0001

- How small can the fast reaction model be, given the context latent?
- Does the latent only need to say "what kind of conversation this is" (the task-ID analogue)? Or must it carry the current state, e.g. "the speaker just said something sad", so that staleness matters?
- Timing: backchannels must land within a few hundred ms of a cue. That bounds the acceptable latent age.

## Literature to search when this is picked up

Not yet searched or verified. Search terms: backchannel prediction, turn-taking / voice activity projection, listener head/face generation, social robot backchanneling, real-time multimodal dialogue systems.
