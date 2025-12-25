#!/usr/bin/env python3
# ============================================================
#  Image Noise Type Analyzer
#  Author: chat GPT 5.0
#  Date:   2025-11-30
#
#  Purpose:
#  Find the reference headshot AndrewReal.jpg in any of the following YT videos
#  Andrew - The Problem Prince  ( Channel 4  Documentary ).mp4' 
#  Andrew Mountbatten Windsor The downfall of a former British prince  ITV News.mp4' 
#  David Frost interviews Prince Andrew on TV-am.mp4' 
#  Frost over the World - Prince Andrew - 16 May 08.mp4' 
#  GALA Prince Andrew  ce quil faut connaître.mp4' 
#  Interview with Prince Andrew.mp4' 
#  Le prince Andrew descend la tour Shard en rappel.mp4' 
#  MARIAGES ROYAUX. Quand Sarah Ferguson a dit oui au Prince Andrew.mp4' 
#  New York Federal Prosecutors Reach Out To Prince Andrew Over Jeffrey Epstein.mp4' 
#  On This Day 26 June 1982 – Prince Andrew Speaks of His Experience in the Falklands.mp4' 
#  Prince Andrew and Sarah Ferguson profile  interview (1986).mp4' 
#  Prince Andrew in U.S..mp4' 
#  Prince Andrew Interview 2010.mp4' 
#  Prince Andrews “Hot and Sweaty” Adrenaline Rush in Falklands War - Uncut Interview (2002).mp4' 
#  Princess Diana at Sarah Fergusons wedding.mp4' 
#  Raw Footage Windsor Castle Fire (1992).mp4' 
#  ROYAL WEDDING 1986 - Andrew  Sarah (1 of 9).mp4' 
#  Sarah Ferguson refuserait de se marier avec le Prince Andrew, Pourquoi compliquer les choses .mp4' 
#  Youre a monster – Prince Andrew and Sarah Ferguson Interview on Royal Wedding (1986).mp4'

# ============================================================

import cv2
import numpy as np
import os
from PIL import Image

# ============================================================
# CONFIG
# ============================================================

DIRECTORY = "case-giuffre-andrew-maxwell/video/work/"
PDF_LIST = [
    f for f in os.listdir(DIRECTORY)
    if f.lower().endswith(".mp4")
]

REF_PATH = "AndrewReal.jpg"
FRAME_STEP = 2 # was 2
FACE_SIZE = 96 #was 128
MATCH_THRESHOLD = 40  # combined hash score threshold (lower = stricter), was 20
SAVE_HITS = True

CASCADE_PATH = "haarcascade_frontalface_default.xml"
CONF_HAAR = 1.1

# ============================================================
# HASH FUNCTIONS
# ============================================================

def ahash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = gray.mean()
    return (gray > avg).flatten().astype(np.uint8)

def dhash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.flatten().astype(np.uint8)

def colorhash(img):
    hsv = cv2.cvtColor(cv2.resize(img, (8, 8)), cv2.COLOR_BGR2HSV)
    h_mean = float(hsv[:, :, 0].mean())
    s_mean = float(hsv[:, :, 1].mean())
    return np.array([h_mean, s_mean])

def hamming(a, b):
    return np.count_nonzero(a != b)

# ============================================================
# FACE DETECTION
# ============================================================

def detect_biggest_face(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=CONF_HAAR,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda r: r[2] * r[3])

# ============================================================
# TIME FORMATTING
# ============================================================

def format_time(seconds):
    ms = int((seconds % 1) * 1000)
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def format_time_filename(seconds):
    ms = int((seconds % 1) * 1000)
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}-{m:02d}-{s:02d}.{ms:03d}"

# ============================================================
# PROCESS A SINGLE VIDEO
# ============================================================

def process_video(video_path, ah_ref, dh_ref, ch_ref, cascade):

    video_short = os.path.splitext(os.path.basename(video_path))[0]
    HITS_DIR = f"matches_{video_short}"

    if SAVE_HITS:
        os.makedirs(HITS_DIR, exist_ok=True)

    print(f"\n==============================")
    print(f"[VIDEO] {video_short}")
    print("==============================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Frames: {total}, FPS: {fps:.2f}")

    frame_id = 0
    hits = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # skip frames
        if frame_id % FRAME_STEP != 0:
            frame_id += 1
            continue

        rect = detect_biggest_face(frame, cascade)
        if rect is None:
            frame_id += 1
            continue

        x, y, w, h = rect
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            frame_id += 1
            continue

        face_norm = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

        ah_f = ahash(face_norm)
        dh_f = dhash(face_norm)
        ch_f = colorhash(face_norm)

        score = (
            hamming(ah_ref, ah_f)
            + hamming(dh_ref, dh_f)
            + np.linalg.norm(ch_ref - ch_f)
        )

        print(f"[{frame_id:06d}] score={score:.2f}")

        if score < MATCH_THRESHOLD:

            # Timestamp
            t = frame_id / fps
            ts_display = format_time(t)
            ts_file = format_time_filename(t)

            print(f">>> MATCH FOUND at frame {frame_id} (time {ts_display})")
            hits.append((frame_id, score, ts_display))

            # Draw face rectangle + timestamp
            vis = frame.copy()
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(vis, ts_display, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

            # Save with timestamp in filename
            if SAVE_HITS:
                out = os.path.join(HITS_DIR, f"hit_{frame_id:06d}_{ts_file}.jpg")
                cv2.imwrite(out, vis)

            if score < 10:
                print(">>> STRONG MATCH — stopping early.")
                break

        frame_id += 1

    cap.release()

    # Write report
    report_path = os.path.join(HITS_DIR, "hit_report.txt")
    with open(report_path, "w") as f:
        for (fid, score, ts) in hits:
            f.write(f"Frame {fid}, score={score:.2f}, time={ts}\n")

    print(f"[INFO] Report written to {report_path}")
    print(f"[INFO] Hits: {hits}")

# ============================================================
# MAIN — LOOP OVER ALL VIDEOS
# ============================================================

def main():

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade.")

    # Load reference
    ref = cv2.imread(REF_PATH)
    if ref is None:
        raise RuntimeError(f"Cannot load reference: {REF_PATH}")

    rect = detect_biggest_face(ref, cascade)
    if rect is None:
        print("[WARN] No face in reference — using entire image")
        ref_face = cv2.resize(ref, (FACE_SIZE, FACE_SIZE))
    else:
        x, y, w, h = rect
        ref_face = cv2.resize(ref[y:y+h, x:x+w], (FACE_SIZE, FACE_SIZE))

    # Reference hashes
    ah_ref = ahash(ref_face)
    dh_ref = dhash(ref_face)
    ch_ref = colorhash(ref_face)

    print("[INFO] Reference hashes created.")

    # Loop over videos
    for video_path in PDF_LIST:
        process_video(video_path, ah_ref, dh_ref, ch_ref, cascade)


if __name__ == "__main__":
    main()
