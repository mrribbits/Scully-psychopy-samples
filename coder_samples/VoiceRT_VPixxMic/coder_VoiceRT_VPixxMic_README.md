# VoiceRT_VPixxMic (Coder): word reading with DATAPixx3 microphone recording

A word appears and the participant reads it aloud. The DATAPixx3 records its MIC input on
every trial. Recording starts on the same screen refresh that shows the word and carries
Pixel Mode B trigger 1. Each trial is saved as a `.wav` file, and a rough voice onset and
voice RT go into the data file. Built for PsychoPy 2026.2.4 and the MEG lab display
(1920×1080, 120 Hz, 60.6 cm wide, 95 cm viewing distance → 52.5 px/deg).

This sample is Coder only, with no Builder twin yet. It reuses the audio-scheduler pattern
from `ChoiceRT_VPixxAudio`, here for recording instead of playback, and the trigger pixel
from `ChoiceRT_VPixxPixelMode`.

Keep `conditions.csv` next to `VoiceRT_VPixxMic.py`. You need `pypixxlib` installed in
PsychoPy's Python, and the microphone plugged into the DATAPixx3's **MIC** input.

## Running it
1. Open `VoiceRT_VPixxMic.py` in PsychoPy's **Coder** window and click **Run** (not Pilot:
   Pixel Mode needs a full-screen window).
2. To test without the DATAPixx3, tick `dummy_vpixx` in the start dialog. The computer's
   default microphone is then recorded through PsychoPy. The timing is **not** precise and
   no triggers are sent.
3. Press **Esc** to stop early. The data collected so far are saved.

## Flow
```
instructions          press 7 or 2
trials (8 words x 2 reps = 16, random)
  fixation cross 0.5 s (60 frames, timed from when the cross appears)
  word 2.0 s          trigger 1 + recording starts on the word's first frame
  blank               while the recording is copied, saved and analysed
thanks                3 s
```

## Trigger
| Event | Pixel (R, G, B) | MEG trigger |
|---|---|---|
| word appears / recording starts | (0, 0, 1) | 1 |

The trigger lasts one frame (8.33 ms). As in the other samples, it fires one frame before the
word is visible on the projector.

## How the recording is locked to the screen
1. **Once, at the start** (section 6): `DPxInitAudCodec()`, `DPxSetMicSource('MIC', 50)`,
   `DPxSetMicLRMode('mono')`, and the mic group delay read with `DPxGetMicGroupDelay(48000)`.
2. **Start of each trial** (section 11): `DPxSetMicBuff(address, size)` resets where the
   recording goes. `DPxSetMicSched(0, 48000, 'hz', 0)` + `DPxDisableMicSchedCountdown()`
   sets up a recording that runs until it is told to stop. It is not started yet.
3. **The frame the word is due:** `DPxStartMicSched()` + `DPxWriteRegCacheAfterVideoSync()`.
   The DATAPixx3 holds the start until the next refresh, which is the `win.flip()` that
   shows the word and the trigger pixel.
4. **The frame the word disappears (2.0 s later):** `DPxStopMicSched()` +
   `DPxWriteRegCacheAfterVideoSync()`, timed to that refresh in the same way.
5. **Then:** `DPxGetMicBuffWriteAddr()` gives how much was recorded (saved as
   `mic.samplesWritten`), and `DPxReadRam` copies the first 2.0 s to the PC. pypixxlib
   returns the samples as unsigned 16-bit numbers, and `.view(np.int16)` turns them into
   signed audio. The result is written to a `.wav` file and the voice onset is found.

**Note on `DPxReadRam`:** on the MEG lab PC, `DPxReadRam(address, n)` returns an array of `n`
values but fills only the first `n` *bytes* (`n/2` samples); the rest are zeros. Asking for
96,000 gave exactly 1 s of audio followed by digital silence, even though the DATAPixx3 had
recorded the full 2 s (`mic.samplesWritten` ≈ 96,033). The script therefore asks for
`2 × 96,000` and keeps the first 96,000, which works whichever way a pypixxlib version counts.

The recording is stopped on the refresh that removes the word rather than after a fixed
sample count, so each recording spans exactly the time the word was on screen.

Between steps 3 and the flip, nothing may call `DPxUpdateRegCache`, because it would erase
the pending start. `saveVpixxTime` only runs *after* the flip, so it is safe.

## Microphone settings (section 1)
- `micGain = 50`: VPixx suggests about 100 for a microphone, but 100 clipped loud voices in
  the MEG lab, so 50 gives twice the headroom. Check `mic.peak` in the data. A value of 32767
  or 32768 means the recording clipped (the range is −32768 to +32767), so lower the gain. A
  value of only a few hundred means it is too quiet, so raise the gain.
- `micChannelMode = 'mono'` stores the average of the left and right channels. If
  recordings are about half as loud as expected, try `'left'`.
- `micMemoryAddress = 0`: each trial overwrites the last recording in DATAPixx3 memory. If
  you combine this with tone playback, keep it away from the tone at 16,000,000.

## Voice onset (a rough check)
The MEG room is noisy. In test recordings the background was steady broadband noise from
about 60 Hz to 4 kHz, roughly a quarter of the voice's level, and speech stood out most
clearly between 300 Hz and 3 kHz. The detector:

1. Band-pass filters the recording to 300–3000 Hz (4th-order Butterworth, run forwards and
   backwards so nothing is shifted in time). This also removes any DC offset and mains hum.
2. Measures the loudness (RMS) of each 5 ms piece.
3. Takes the background loudness as the median of the first 300 ms.
4. Reports the onset as the first piece that is more than 2× the background and stays above
   that for at least 20 ms. If that never happens, the trial gets `noVoice = 1`.

The settings are `onsetBandHz`, `baselineSec`, `onsetFactor` and `onsetHoldSec` in section 1.
On test recordings, the onsets fell where the voice visibly starts, and a noise-only
recording gave no onset. Breaths, lip smacks and coughs can still fool it. For publication,
measure onsets offline from the `.wav` files (e.g. in Praat).

An earlier version measured the unfiltered sound against the loudest piece. With this much
background noise it either found no voice or reported an impossible onset of about 0.1 s.

## Data
Files are saved as `data/<participant>_<session>_<date>.*`, plus one
`data/<participant>_<session>_<date>_trialNN.wav` per trial (48 kHz, 16-bit, mono). Sample 0
of each `.wav` is the refresh that showed the word.

| Column | Meaning |
|---|---|
| `word` | the word shown (from `conditions.csv`) |
| `word.started` | PsychoPy clock time of the refresh that showed the word |
| `word.vp` | DATAPixx3 clock time at that refresh (empty in dummy mode) |
| `word.trigger` | the trigger value sent (1) |
| `wavFile` | this trial's recording |
| `voiceOnset` | raw voice onset, s from the first sample (empty if `noVoice`) |
| `voiceRT` | `voiceOnset − mic.groupDelay − 1 frame (8.33 ms)`: s from the word becoming visible |
| `noVoice` | 1 = no clear voice found |
| `mic.groupDelay` | the DATAPixx3 audio chip's delay (s); 0 in dummy mode |
| `mic.samplesWritten` | samples the DATAPixx3 recorded this trial; should be about 96,000 (less = the `.wav` ends in silence, and a warning is logged) |
| `mic.peak` | largest absolute sample, used to check for clipping (32767 or 32768 = clipped) |
| `dummy_vpixx_used` | True if the run used no DATAPixx3. The `dummy_vpixx` column is the start-dialog tick box; they differ only if pypixxlib was missing |

## Script layout
| Section | What it does |
|---|---|
| 1 | settings (display, timing, trigger, microphone, onset detection) |
| 2 | sizes in pixels; recording length in samples |
| 3–5 | start dialog, data files, window |
| 6 | dummy-mode decision; full-screen check; connect, Pixel Mode B, mic on; `atexit` cleanup; dummy microphone |
| 7 | trigger pixel; `saveVpixxTime`, `readRecording`, `saveWav`, `findVoiceOnset` |
| 8 | stimuli, keyboard, `saveAndQuit`, Esc check |
| 9 | clocks |
| 10 | instructions |
| 11 | trials loop: set up the schedule; frame loop (cross, then word + trigger + start); read, save, analyse |
| 12 | thank-you screen; save and quit |

## MEG note
Speaking moves the head and jaw and activates face and neck muscles. Expect large artifacts
in OPM-MEG data from about the voice onset onward. The period between the word appearing and
the voice onset is usually what you analyse.
