#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VoiceRT_VPixxMic.py  -  PsychoPy CODER sample
=====================================================================

Voice reaction time task: a word appears and the participant reads it
aloud. The DATAPixx3 records the microphone on every trial, starting on
the exact screen refresh that shows the word, and a Pixel Mode B MEG
trigger (trigger 1) marks that same refresh.

There is no Builder version of this sample (yet). It is built from the
same pieces as two other Coder samples, so read those first if anything
here is unclear:
    ChoiceRT_VPixxAudio      - the DATAPixx3 audio scheduler, started on
                               a screen refresh (here: recording, not
                               playing)
    ChoiceRT_VPixxPixelMode  - Pixel Mode B triggers

Written for PsychoPy 2026.2.4 and the Scully Center OPM-MEG lab:
    display 1920 x 1080 px, 120 Hz, 60.6 cm wide, viewed from 95 cm
    DATAPixx3 between the stimulus PC and the projector
    microphone plugged into the DATAPixx3's MIC input
    response keys: 7 = left index finger, 2 = right index finger

---------------------------------------------------------------------
THE TASK
---------------------------------------------------------------------
    instructions        press 7 or 2 to start
    16 trials, random order (8 words x 2), each one:
        fixation cross      0.5 s
        word                2.0 s  -> MEG trigger 1 + recording starts
                                      (read the word aloud)
        (blank screen while the recording is saved, a fraction of a second)
    thank-you screen    3 s

---------------------------------------------------------------------
HOW TO RUN IT
---------------------------------------------------------------------
    1. Open this file in PsychoPy's Coder window (File > Open).
    2. Keep conditions.csv in the same folder as this file.
    3. Click "Run" (NOT "Pilot": pilot mode uses a small window, and Pixel
       Mode triggers need a full-screen window - see section 6).
    4. Fill in the start dialog. Tick "dummy_vpixx" to test on a computer
       with no DATAPixx3: the computer's own microphone is then recorded
       through PsychoPy, NOT precisely timed, and no triggers are sent.
    Press Esc at any time to stop. The data recorded so far are saved.

---------------------------------------------------------------------
HOW THE RECORDING WORKS (read this first)
---------------------------------------------------------------------
The microphone is not recorded by the stimulus PC. It is recorded by the
DATAPixx3, VPixx's box between the PC and the projector, into the
DATAPixx3's own memory.
    1. Once, at the start: switch the audio on and choose the MIC input
       and its gain (section 6).
    2. Start of each trial: give the DATAPixx3 a "microphone schedule":
       record 48000 samples per second into memory until told to stop.
       The schedule is set up but not started yet.
    3. On the frame the word appears: tell it "start", timed to arrive
       exactly when the screen refreshes. The first sample of the
       recording is taken at that refresh, to well under 1 ms. The MEG
       trigger is the top-left pixel of that same frame, so trigger 1,
       the word and sample 0 of the recording all share one refresh.
    4. On the frame the word disappears (2.0 s later): tell it "stop",
       timed to that refresh in the same way.
    5. Then: copy the first 2.0 s of the recording from the DATAPixx3 to
       the PC, save it as a .wav file, and find where the voice starts.

---------------------------------------------------------------------
TWO TIMING CORRECTIONS (voiceRT includes both)
---------------------------------------------------------------------
    projector lag  In the MEG lab, things locked to a screen refresh (the
                   Pixel Mode trigger, and here the recording) happen 1
                   frame (8.33 ms) BEFORE the image is on the projector
                   screen. So the recording starts 8.33 ms before the
                   participant can see the word.
    mic delay      The DATAPixx3's audio chip (its "CODEC") takes a
                   little time to turn sound into numbers, so each sample
                   is stored slightly after the sound arrived. pypixxlib
                   reports this as the "group delay".
    voiceRT = voiceOnset - micGroupDelay - projectorLag
    (voiceOnset is the raw time of the voice, from the first sample.)
    The trigger is ALSO 1 frame early, so for MEG analysis remember the
    word appears 8.33 ms after trigger 1.

---------------------------------------------------------------------
HOW THE VOICE ONSET IS FOUND (a rough check, not a final analysis)
---------------------------------------------------------------------
The MEG room is noisy: in test recordings the background was steady
broadband noise from about 60 Hz to 4 kHz, and speech stood out most
clearly between 300 Hz and 3 kHz. So:
    1. Keep only 300-3000 Hz (a band-pass filter). This also removes any
       constant offset and mains hum.
    2. Cut the result into 5 ms pieces and measure the loudness (RMS) of
       each.
    3. Background loudness = the median of the first 300 ms, before
       anyone could have started reading the word aloud.
    4. The voice onset is the first piece that is more than 2x the
       background AND stays above it for at least 20 ms (so a click or a
       single loud piece doesn't count).
If that never happens, the trial counts as "no voice" (noVoice = 1) and
voiceOnset is left empty. In the test recordings this put the onsets
where the voice visibly starts, and found nothing in noise alone.
Coughs, lip smacks and breaths can still fool it. For publication,
measure onsets offline from the .wav files (e.g. by eye in Praat).
"""

# =====================================================================
# 0. IMPORT THE LIBRARIES WE NEED
# =====================================================================
#   visual   - windows and things to draw (text, shapes, ...)
#   core     - clocks and quitting
#   data     - the data file, the trial list and the trials loop
#   gui      - the start dialog box
#   logging  - writes messages to the console and to a .log file
#   keyboard - PsychoPy's precise keyboard
import psychopy
from psychopy import visual, core, data, gui, logging
from psychopy.hardware import keyboard
from psychopy.constants import NOT_STARTED, STARTED, FINISHED

import math      # Python's maths library: tan(), radians(), ...
import os        # file and folder names
import atexit    # lets us name a function that must run when Python quits
import wave      # writes .wav sound files (part of Python itself)

# scipy.signal (part of PsychoPy's Python) filters the recording before
# the voice onset is measured (section 7).
from scipy import signal

# "numpy" is a maths library for long lists of numbers, such as the
# 96,000 numbers that make up a 2-second recording. "np" is its short name.
import numpy as np

# pypixxlib is VPixx's Python library for talking to the DATAPixx3. If it
# is not installed, remember that instead of crashing; section 6 then
# switches to dummy mode.
try:
    from pypixxlib import _libdpx as dp
    pypixxlibInstalled = True
except ImportError:
    dp = None
    pypixxlibInstalled = False


# =====================================================================
# 1. SETTINGS - EDIT HERE
# =====================================================================

# ---- The display (MEG lab projector) --------------------------------
screenSizePix = (1920, 1080)  # width, height in pixels
screenWidthCm = 60.6          # width of the image on the screen, in cm
viewingDistanceCm = 95        # distance from the eyes to the screen, in cm
refreshRate = 120             # Hz
background = (-1, -1, -1)     # black. Black also keeps the trigger pixel
                              # "off" between triggers.

# ---- Stimulus sizes, in degrees of visual angle ---------------------
fixationSizeDeg = 0.6         # width and height of the central cross
wordHeightDeg = 1.0           # height of the letters

# ---- Trial timing, seconds ------------------------------------------
fixationDuration = 0.5        # cross alone before the word
wordDuration = 2.0            # word on screen = length of the recording
thanksDuration = 3.0          # how long the thank-you screen stays up

# ---- Trials ----------------------------------------------------------
conditionsFile = 'conditions.csv'  # one row per word
nReps = 2                          # 8 words x 2 = 16 trials

# ---- Keys to start ---------------------------------------------------
startKeys = ['7', '2']   # either index-finger button starts the task

# ---- MEG trigger (VPixx Pixel Mode B) -------------------------------
# The DATAPixx3 reads the BLUE value of the screen's top-left pixel and
# copies it to the 8 MEG trigger lines. Blue = 1 switches on line 1.
# (See ChoiceRT_VPixxPixelMode for the full explanation.)
TRIG_WORD = (0, 0, 1)   # word appears (and recording starts) -> trigger 1
TRIG_OFF = (0, 0, 0)    # black pixel -> all lines off

# ---- The recording ---------------------------------------------------
sampleRate = 48000        # samples per second
micSource = 'MIC'         # the DATAPixx3's MIC input ('LINE' = line input)
micGain = 50              # VPixx suggests about 100 for a microphone and
                          # 1 for line level, but 100 clipped loud voices
                          # in the MEG lab, so 50 gives 2x headroom. Check
                          # the mic.peak column: 32767 or 32768 = clipped
                          # (lower the gain); a few hundred = too quiet.
micChannelMode = 'mono'   # 'mono' stores the average of the left and
                          # right channels. If recordings are half as
                          # loud as you expect, try 'left'.

# Where in the DATAPixx3's memory the recording is stored. Each trial
# overwrites the one before (it is copied to the PC in between). This
# sample uses no other DATAPixx3 memory, so 0 is safe. If you combine it
# with ChoiceRT_VPixxAudio, keep this away from the tone's 16,000,000.
micMemoryAddress = 0

# ---- Voice-onset detection (see the top of the file) ----------------
onsetBandHz = (300, 3000)  # keep only this frequency band (speech)
onsetWindowSec = 0.005     # 5 ms pieces
baselineSec = 0.300        # first 300 ms = background noise
onsetFactor = 2.0          # voice = more than 2x the background loudness
onsetHoldSec = 0.020       # ...for at least 20 ms in a row

# ---- Projector lag (see the top of the file) ------------------------
projectorLagSec = 1.0 / refreshRate   # 1 frame = 8.33 ms


# =====================================================================
# 2. WORK OUT SIZES AND THE RECORDING LENGTH
# =====================================================================
# Degrees -> pixels, as in the other samples (about 52.5 px per degree).
cmPerDeg = 2 * viewingDistanceCm * math.tan(math.radians(0.5))
pixPerCm = screenSizePix[0] / screenWidthCm
pixPerDeg = cmPerDeg * pixPerCm
fixationSizePix = round(fixationSizeDeg * pixPerDeg)   # 32 px
wordHeightPix = round(wordHeightDeg * pixPerDeg)       # 52 px

# Number of samples in one recording: 2.0 s x 48000 = 96000.
nRecSamples = int(round(wordDuration * sampleRate))

# Room reserved in DATAPixx3 memory: the recording plus 1 s spare, in
# bytes (2 bytes per sample). The recording is stopped by the script, so
# it may run a few samples past nRecSamples; the spare room catches them.
micBufferBytes = 2 * int(round((wordDuration + 1.0) * sampleRate))


# =====================================================================
# 3. THE START DIALOG
# =====================================================================
expName = 'VoiceRT_VPixxMic'
expInfo = {
    'participant': '',
    'session': '001',
    'dummy_vpixx': False,
}

PILOTING = core.setPilotModeFromArgs()
if PILOTING:
    expInfo['participant'] = 'pilot'

dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if not dlg.OK:          # the experimenter pressed Cancel
    core.quit()

expInfo['date'] = data.getDateStr()
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopy.__version__


# =====================================================================
# 4. DATA FILES
# =====================================================================
# data/<participant>_<session>_<date>.csv / .log / .psydat, plus one .wav
# per trial: data/<participant>_<session>_<date>_trial01.wav, ...
thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(thisDir)

fileStem = os.path.join(thisDir, 'data', '%s_%s_%s' % (
    expInfo['participant'], expInfo['session'], expInfo['date']))
os.makedirs(os.path.dirname(fileStem), exist_ok=True)

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=True, saveWideText=True,
    dataFileName=fileStem, sortColumns='time')
thisExp.addData('piloting', PILOTING)

logging.console.setLevel(logging.WARNING)
logFile = logging.LogFile(fileStem + '.log', level=logging.INFO)


# =====================================================================
# 5. THE WINDOW
# =====================================================================
win = visual.Window(
    size=screenSizePix if not PILOTING else (800, 600),
    fullscr=not PILOTING,
    screen=0,
    color=background, colorSpace='rgb',
    units='pix',
    allowGUI=False,
    blendMode='avg', useFBO=True,
)
win.mouseVisible = False
expInfo['frameRate'] = win.getActualFrameRate()


# =====================================================================
# 6. CONNECT TO THE DATAPixx3: PIXEL MODE B AND THE MICROPHONE
# =====================================================================
# ---- Dummy mode? -----------------------------------------------------
DUMMY_VPIXX = str(expInfo['dummy_vpixx']) == 'True'

if not pypixxlibInstalled and not DUMMY_VPIXX:
    logging.warning('pypixxlib is not installed, so dummy_vpixx was turned '
                    'on: NO TRIGGERS, and the recording is NOT precisely timed')
    DUMMY_VPIXX = True

# (The dialog's tick box is saved as 'dummy_vpixx'; this column is the mode
#  that actually ran, which differs only if pypixxlib was missing.)
thisExp.addData('dummy_vpixx_used', DUMMY_VPIXX)

# ---- Pixel Mode needs a full-screen window ---------------------------
# The DATAPixx3 reads the top-left pixel of the whole SCREEN. In a small
# (Pilot) window that pixel belongs to the desktop, so stop here.
if not DUMMY_VPIXX and (PILOTING or not win.fullscr):
    message = ('VPixx Pixel Mode triggers need a FULL-SCREEN window in RUN '
               'mode. Switch from Pilot to Run, or tick dummy_vpixx to test '
               'without the DATAPixx3.')
    logging.error(message)
    win.close()
    raise RuntimeError(message)

if not DUMMY_VPIXX:
    # ---- Real DATAPixx3 ----------------------------------------------
    # Most dp.DPx... commands only change a local copy of the device's
    # settings (the "register cache"). Nothing reaches the DATAPixx3
    # until dp.DPxWriteRegCache() sends that copy.
    dp.DPxOpen()                        # connect over USB
    dp.DPxStopAllScheds()               # stop anything left running
    dp.DPxEnableDoutPixelModeB()        # top-left blue value -> trigger lines
    dp.DPxInitAudCodec()                # switch the audio chip on
    dp.DPxSetMicSource(micSource, micGain)   # MIC input and its gain
    dp.DPxSetMicLRMode(micChannelMode)       # which channel(s) to store
    dp.DPxWriteRegCache()               # send the settings above

    # The audio chip's delay for this sample rate (see the top of the
    # file). Saved in every row of the data file.
    micGroupDelay = dp.DPxGetMicGroupDelay(sampleRate)
    logging.exp('DATAPixx3 microphone ready: %s input, gain %s, '
                'group delay %.3f ms'
                % (micSource, micGain, 1000 * micGroupDelay))

    def closeVpixx():
        """Stop recording, turn Pixel Mode off, disconnect."""
        try:
            dp.DPxStopAllScheds()
            dp.DPxDisableDoutPixelMode()
            dp.DPxWriteRegCache()
            dp.DPxClose()
        except Exception:
            pass   # already closed or unplugged

    # Run closeVpixx when the script ends, however it ends.
    atexit.register(closeVpixx)

else:
    # ---- Dummy mode ---------------------------------------------------
    # Record from the computer's default microphone through PsychoPy.
    # Timing is NOT precise (often tens of ms off), so use this for
    # testing the script only. streamBufferSecs must be longer than one
    # recording.
    from psychopy.sound import Microphone
    micGroupDelay = 0.0
    dummyMic = Microphone(sampleRateHz=sampleRate, channels=1,
                          streamBufferSecs=wordDuration + 2.0)


# =====================================================================
# 7. THE TRIGGER PIXEL AND HELPER FUNCTIONS
# =====================================================================
# A 1 x 1 pixel square in the top-left corner, drawn on every flip.
# In pixel units (0, 0) is the screen CENTRE, so the corner is at
# (-width/2, +height/2); 0.5 puts the square in the middle of that pixel.
screenWidth, screenHeight = win.size
triggerPixel = visual.Rect(
    win, units='pix', width=1, height=1,
    pos=(-screenWidth / 2 + 0.5, screenHeight / 2 - 0.5),
    colorSpace='rgb255', fillColor=TRIG_OFF, lineColor=TRIG_OFF,
    lineWidth=0, interpolate=False, autoLog=False)
triggerPixel.setAutoDraw(True)


def setTriggerPixel(colour):
    """Colour the trigger pixel from the next flip on (e.g. TRIG_WORD)."""
    triggerPixel.fillColor = colour
    triggerPixel.lineColor = colour


def saveVpixxTime(columnName):
    """Save the DATAPixx3's clock time in the data file.

    Run with win.callOnFlip(), so it runs straight AFTER the refresh that
    showed the word. It must not run before that refresh: its
    DPxUpdateRegCache() would erase the pending "start recording".
    """
    if not DUMMY_VPIXX:
        dp.DPxUpdateRegCache()
        thisExp.addData(columnName, dp.DPxGetTime())


def readRecording():
    """Copy this trial's recording from the DATAPixx3 to the PC.

    Returns a numpy array of nRecSamples whole numbers, -32768 to +32767.
    """
    if not DUMMY_VPIXX:
        # The recording was stopped on the flip that removed the word.
        # Ask the DATAPixx3 where it would write the NEXT sample: everything
        # before that address is this trial's recording. 2 bytes = 1 sample.
        dp.DPxUpdateRegCache()
        bytesWritten = dp.DPxGetMicBuffWriteAddr() - micMemoryAddress
        samplesWritten = bytesWritten // 2
        thisExp.addData('mic.samplesWritten', samplesWritten)
        if samplesWritten < nRecSamples:
            logging.warning('Recording shorter than expected: %d of %d samples'
                            % (samplesWritten, nRecSamples))
        # Read exactly nRecSamples (2.0 s) from the start of the recording.
        # If fewer were written, the missing end is filled with zeros.
        nRead = max(0, min(samplesWritten, nRecSamples))
        samples = np.zeros(nRecSamples, dtype=np.int16)
        if nRead > 0:
            # DPxReadRam(address, n) returns n 16-bit numbers, but on the
            # MEG lab PC's pypixxlib only the first n BYTES (n/2 samples) are
            # actually filled in; the rest come back as zeros. (Asking for
            # 96000 gave exactly 1 s of audio followed by digital silence.)
            # So we ask for twice as many and keep the first nRead. This
            # works whichever way a pypixxlib version counts: if it counts
            # samples, we just read 2 s of spare memory we don't use.
            raw = dp.DPxReadRam(micMemoryAddress, 2 * nRead)
            raw = np.asarray(raw, dtype=np.uint16)[:nRead]
            # pypixxlib hands the numbers back as UNSIGNED (0 to 65535);
            # .view(np.int16) reads the same bits as SIGNED audio samples
            # (-32768 to +32767).
            samples[:nRead] = raw.view(np.int16)
        return samples
    else:
        dummyMic.stop()
        clip = dummyMic.getRecording()
        # clip.samples are -1 to +1, one column per channel. Take the
        # first channel and scale to 16-bit, like the DATAPixx3 data.
        samples = np.asarray(clip.samples, dtype=float)
        if samples.ndim > 1:
            samples = samples[:, 0]
        samples = samples[:nRecSamples]
        return np.clip(samples * 32767, -32768, 32767).astype(np.int16)


def saveWav(samples, fileName):
    """Save 16-bit mono samples as a .wav file."""
    with wave.open(fileName, 'wb') as wavFile:
        wavFile.setnchannels(1)          # mono
        wavFile.setsampwidth(2)          # 2 bytes = 16 bits per sample
        wavFile.setframerate(sampleRate)
        wavFile.writeframes(samples.astype('<i2').tobytes())


def findVoiceOnset(samples):
    """Rough voice onset in seconds from the first sample, or None.

    See "HOW THE VOICE ONSET IS FOUND" at the top of the file.
    """
    # Step 1: keep only 300-3000 Hz. butter() designs the filter;
    # sosfiltfilt() runs it forwards and then backwards, so the filtered
    # sound is not shifted in time (important: we are measuring timing).
    filt = signal.butter(4, onsetBandHz, btype='bandpass',
                         fs=sampleRate, output='sos')
    x = samples.astype(float)
    x = signal.sosfiltfilt(filt, x - x.mean())

    # Step 2: cut into 5 ms pieces (240 samples each) and measure the
    # loudness (RMS = square root of the mean of the squares) of each.
    winLen = int(round(onsetWindowSec * sampleRate))
    nWins = len(x) // winLen
    pieces = x[:nWins * winLen].reshape(nWins, winLen)
    rms = np.sqrt(np.mean(pieces ** 2, axis=1))

    # Step 3: background loudness = median of the first 300 ms (60 pieces).
    nBase = int(round(baselineSec / onsetWindowSec))
    if nWins <= nBase:
        return None
    threshold = onsetFactor * np.median(rms[:nBase])

    # Step 4: the first piece after the background period that starts a
    # run of at least 20 ms (4 pieces) all above the threshold.
    nHold = max(1, int(round(onsetHoldSec / onsetWindowSec)))
    above = rms > threshold
    for i in range(nBase, nWins - nHold + 1):
        if above[i:i + nHold].all():
            return i * onsetWindowSec
    return None                     # never clearly louder: no voice


# =====================================================================
# 8. STIMULI, KEYBOARD AND QUITTING
# =====================================================================
taskInstructions = visual.TextStim(
    win, name='taskInstructions',
    text=('On each trial a word will appear in the middle of the screen.\n\n'
          'Read it aloud, clearly, as quickly as you can.\n\n'
          'Please keep your head still and look at the cross between words.\n\n'
          'Press either button to start.'),
    font='Arial', height=32, wrapWidth=1400, color='white', units='pix')

thanksText = visual.TextStim(
    win, name='thanksText',
    text='Thank you!\n\nThe task is complete.',
    font='Arial', height=32, wrapWidth=1400, color='white', units='pix')

fixation = visual.ShapeStim(
    win, name='fixation', vertices='cross', units='pix',
    size=(fixationSizePix, fixationSizePix), pos=(0, 0),
    lineWidth=1.0, lineColor='white', fillColor='white', interpolate=True)

# The word. Its text is set at the start of each trial.
wordStim = visual.TextStim(
    win, name='word', text='', font='Arial', height=wordHeightPix,
    color='white', units='pix', pos=(0, 0))

kb = keyboard.Keyboard()


def saveAndQuit():
    """Save the data files, close the window and stop the script."""
    thisExp.saveAsWideText(fileStem + '.csv', delim='auto')
    thisExp.saveAsPickle(fileStem)
    thisExp.abort()
    logging.flush()
    win.close()
    core.quit()            # closeVpixx() runs now, through atexit


def checkForEscape():
    """Quit (saving the data) if the experimenter pressed Esc."""
    if kb.getKeys(keyList=['escape'], waitRelease=False, clear=False):
        logging.warning('Esc pressed: stopping the experiment')
        saveAndQuit()


# =====================================================================
# 9. CLOCKS
# =====================================================================
globalClock = core.Clock()
logging.setDefaultClock(globalClock)
trialClock = core.Clock()
frameTolerance = 0.001


# =====================================================================
# 10. INSTRUCTIONS SCREEN
# =====================================================================
kb.clearEvents()
taskInstructions.setAutoDraw(True)
thisExp.timestampOnFlip(win, 'instruct.started')
while True:
    checkForEscape()
    keys = kb.getKeys(keyList=startKeys, waitRelease=False)
    if keys:
        thisExp.addData('ready.keys', keys[-1].name)
        thisExp.addData('ready.rt', keys[-1].rt)
        break
    win.flip()
taskInstructions.setAutoDraw(False)
thisExp.addData('instruct.stopped', globalClock.getTime())
thisExp.nextEntry()


# =====================================================================
# 11. THE TRIALS LOOP
# =====================================================================
trials = data.TrialHandler2(
    trialList=data.importConditions(conditionsFile),
    nReps=nReps, method='random', extraInfo=expInfo, name='trials')
thisExp.addLoop(trials)

trialNum = 0
for thisTrial in trials:
    trialNum += 1
    word = str(thisTrial['word'])

    # ---- Prepare the trial -------------------------------------------
    wordStim.text = word
    wavName = '%s_trial%02d.wav' % (os.path.basename(fileStem), trialNum)
    thisExp.addData('wavFile', wavName)
    thisExp.addData('mic.groupDelay', micGroupDelay)

    if not DUMMY_VPIXX:
        # Where to store the recording: start at micMemoryAddress, with
        # micBufferBytes of room. Doing this every trial also moves the
        # "write here next" address back to the start, so each recording
        # overwrites the last one instead of following on after it.
        dp.DPxSetMicBuff(micMemoryAddress, micBufferBytes)
        # The schedule: start at once when told (onset 0), record at
        # sampleRate samples per second ('hz'), and keep going until the
        # script stops it (count 0, countdown off). We stop it ourselves,
        # on the refresh that removes the word, because a sample count
        # stopped the recording after only half the requested length.
        dp.DPxSetMicSched(0, sampleRate, 'hz', 0)
        dp.DPxDisableMicSchedCountdown()
        dp.DPxWriteRegCache()     # send it (not started yet)

    fixation.status = NOT_STARTED
    wordStim.status = NOT_STARTED
    setTriggerPixel(TRIG_OFF)
    kb.clearEvents()

    thisExp.timestampOnFlip(win, 'trial.started')
    win.callOnFlip(trialClock.reset)
    win.flip()

    # ---- Run the trial: one pass of this loop per frame -------------
    while True:
        tThisFlip = win.getFutureFlipTime(clock=trialClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)

        # Every frame starts with the trigger pixel black, so trigger 1
        # lasts exactly one frame (8.33 ms).
        setTriggerPixel(TRIG_OFF)

        # ---- fixation cross: on at the start of the trial ------------
        if fixation.status == NOT_STARTED and tThisFlip >= 0.0 - frameTolerance:
            fixation.setAutoDraw(True)
            fixation.status = STARTED
            # When the cross appears, on the global clock: our best guess
            # now, replaced by the actual flip time once the flip happens.
            fixation.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(fixation, 'tStartRefresh')
            thisExp.timestampOnFlip(win, 'fixation.started')

        # ---- the word: trigger 1 and recording start, same frame -----
        # Timed from when the cross actually appeared, so the cross is up
        # for exactly fixationDuration (60 frames at 120 Hz).
        if (wordStim.status == NOT_STARTED and fixation.status == STARTED
                and tThisFlipGlobal >= fixation.tStartRefresh + fixationDuration - frameTolerance):
            fixation.setAutoDraw(False)      # the word replaces the cross
            fixation.status = FINISHED
            wordStim.setAutoDraw(True)
            wordStim.status = STARTED
            wordStim.tStartRefresh = tThisFlipGlobal
            win.timeOnFlip(wordStim, 'tStartRefresh')   # actual flip time
            thisExp.timestampOnFlip(win, 'word.started')

            # MEG trigger 1 in the top-left pixel of this frame.
            setTriggerPixel(TRIG_WORD)
            thisExp.addData('word.trigger', TRIG_WORD[2])

            if not DUMMY_VPIXX:
                # Mark "start recording" in the local copy of the settings,
                # and send it to the DATAPixx3, which holds it until the
                # next screen refresh: the one win.flip() below shows.
                dp.DPxStartMicSched()
                dp.DPxWriteRegCacheAfterVideoSync()
                # IMPORTANT: nothing may call dp.DPxUpdateRegCache() between
                # these lines and the flip, or the "start" is erased.
            else:
                # Dummy mode: start PsychoPy's recording right after the
                # flip (NOT precisely timed).
                win.callOnFlip(dummyMic.start)

            # After the flip: save the DATAPixx3 clock at that refresh.
            win.callOnFlip(saveVpixxTime, 'word.vp')

        # ---- end of the word = end of the recording ------------------
        if (wordStim.status == STARTED
                and tThisFlipGlobal > wordStim.tStartRefresh + wordDuration - frameTolerance):
            thisExp.addData('word.stopped', tThisFlipGlobal)
            break

        # Dummy mode only: keep PsychoPy's microphone buffer emptied.
        if DUMMY_VPIXX and wordStim.status == STARTED and hasattr(dummyMic, 'poll'):
            dummyMic.poll()

        checkForEscape()
        win.flip()

    # ---- End of the trial: blank screen, save the recording ----------
    wordStim.setAutoDraw(False)
    wordStim.status = FINISHED
    setTriggerPixel(TRIG_OFF)
    if not DUMMY_VPIXX:
        # Stop the recording on the refresh that removes the word, the same
        # way it was started: the DATAPixx3 holds the "stop" until then.
        dp.DPxStopMicSched()
        dp.DPxWriteRegCacheAfterVideoSync()
    win.flip()          # show the blank screen while we work

    samples = readRecording()
    saveWav(samples, os.path.join(os.path.dirname(fileStem), wavName))

    voiceOnset = findVoiceOnset(samples)
    if voiceOnset is not None:
        thisExp.addData('voiceOnset', voiceOnset)
        thisExp.addData('voiceRT', voiceOnset - micGroupDelay - projectorLagSec)
        thisExp.addData('noVoice', 0)
    else:
        thisExp.addData('noVoice', 1)   # leave voiceOnset / voiceRT empty

    # Clipping check: the largest sample size this trial. 32767 (top of
    # the range) or 32768 (bottom: -32768) means the recording clipped
    # and the gain is too high.
    thisExp.addData('mic.peak', int(np.abs(samples.astype(np.int32)).max()))

    thisExp.addData('trial.stopped', globalClock.getTime())
    thisExp.nextEntry()


# =====================================================================
# 12. THANK-YOU SCREEN
# =====================================================================
thanksText.setAutoDraw(True)
thisExp.timestampOnFlip(win, 'thanksText.started')
win.callOnFlip(trialClock.reset)
win.flip()
while trialClock.getTime() < thanksDuration:
    checkForEscape()
    win.flip()
thanksText.setAutoDraw(False)
win.flip()

saveAndQuit()
