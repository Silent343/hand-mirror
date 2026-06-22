"""
hand_mirror.py
==============
Main module for Hand Mirror 3D.

Detects up to two hands in real-time using MediaPipe Hands and
renders them as 3D robotic hands (black/white prosthesis style) on a
fixed side panel. The rotation, tilt, and position of the 3D model
faithfully follow the 21 3D landmarks provided by MediaPipe, so
when you tilt or rotate your hand, the model moves in the exact same way.

Dependencies:
    - opencv-python  >= 4.8
    - mediapipe      >= 0.10
    - numpy          >= 1.24

Usage::

    python hand_mirror.py

Controls:
    q — quit
"""

import cv2
import mediapipe as mp
import numpy as np
import math


# ──────────────────────────────────────────────────────────────────────────────
#  Global Constants
# ──────────────────────────────────────────────────────────────────────────────

PANEL_W: int = 380
"""Width in pixels of the left side panel where the hands are rendered."""

PANEL_H: int = 480
"""Height in pixels of the side panel (equal to the camera frame height)."""

Z_AMPLIFY: float = 2.6
"""
Amplification factor for the Z-axis of the landmarks.

MediaPipe delivers relatively small Z values. Multiplying them by this
factor makes the tilt and depth of the hand clearly perceptible in the
perspective projection. Increasing this value exaggerates the tilt; 
decreasing it softens it.
"""

HAND_DISPLAY_SCALE: int = 125
"""
Rendering scale of the hand on the panel (pixel units).

Controls the visual size of the 3D model. Larger values make the hand
bigger; smaller values shrink it to fit better in each slot.
"""

# ── Robotic color palette (OpenCV BGR format) ───────────────────────

COL_PANEL_BG: tuple = (18, 19, 24)
"""Background color of the side panel — almost blue-black."""

COL_GRID: tuple = (32, 34, 44)
"""Color of the panel's background grid."""

COL_DIVIDER: tuple = (55, 58, 74)
"""Color of the dividing lines and panel borders."""

COL_LABEL: tuple = (170, 175, 195)
"""Color for the label text (light bluish-gray)."""

CASING_WHITE: tuple = (238, 240, 245)
"""Main color of the robotic hand casing — slightly cool white."""

CASING_SHADE: tuple = (175, 178, 188)
"""Shade/outline color of the casing to give volume."""

RUBBER_BLACK: tuple = (28, 30, 36)
"""Color of the rubber pads — deep black with a slight blue tint."""

JOINT_SILVER: tuple = (200, 205, 215)
"""Silver color for the fill of the mechanical joints."""

JOINT_RING: tuple = (120, 125, 140)
"""Color of the outer ring of each joint (darker metallic tone)."""

SCREW_COL: tuple = (90, 95, 110)
"""Color of the screws and fastening details."""

# ── Hand topology (MediaPipe landmark indices) ─────────────────

BONES: list[tuple] = [
    # Thumb:  wrist→CMC, CMC→MCP, MCP→IP, IP→tip
    (0, 1, 0), (1, 2, 1), (2, 3, 1), (3, 4, 2),
    # Index:  MCP→PIP, PIP→DIP, DIP→tip
    (5, 6, 1), (6, 7, 1), (7, 8, 2),
    # Middle: MCP→PIP, PIP→DIP, DIP→tip
    (9, 10, 1), (10, 11, 1), (11, 12, 2),
    # Ring:   MCP→PIP, PIP→DIP, DIP→tip
    (13, 14, 1), (14, 15, 1), (15, 16, 2),
    # Pinky:  MCP→PIP, PIP→DIP, DIP→tip
    (17, 18, 1), (18, 19, 1), (19, 20, 2),
]
"""
List of hand bones as tuples ``(a, b, kind)``.

Each bone connects landmark ``a`` with landmark ``b``.

``kind`` indicates the visual style:

- ``0`` — palm segment (casing, medium thickness)
- ``1`` — middle/proximal phalanx (white casing with rubber stripe)
- ``2`` — distal phalanx / tip (black rubber pad)
"""

PALM_HULL: list[int] = [0, 1, 5, 9, 13, 17]
"""
Indices of the landmarks that form the convex hull of the palm.

Used to draw the filled polygon of the robotic palm.
The order follows the perimeter: wrist (0), thumb CMC (1),
index MCP (5), middle MCP (9), ring MCP (13), pinky MCP (17).
"""


# ──────────────────────────────────────────────────────────────────────────────
#  Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def lerp_arr(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation (lerp) between two NumPy arrays.

    Calculates ``a + (b - a) * t``, useful for smoothing the transition between
    the previous state and the new state of the landmarks frame by frame.

    Args:
        a: Source array (previous state).
        b: Target array (new state).
        t: Interpolation factor in the range ``[0.0, 1.0]``.
           ``0.0`` returns ``a``; ``1.0`` returns ``b``.

    Returns:
        Interpolated array with the same shape as ``a`` and ``b``.
    """
    return a + (b - a) * t


def rotate_y(pts: np.ndarray, ang: float) -> np.ndarray:
    """
    Rotates a set of 3D points around the Y-axis.

    Applies the standard rotation matrix for the Y-axis::

        | cos(ang)   0   sin(ang) |
        |    0       1      0     |
        | -sin(ang)  0   cos(ang) |

    Args:
        pts: Array of shape ``(N, 3)`` with ``(x, y, z)`` coordinates.
        ang: Rotation angle in radians. Positive = right turn.

    Returns:
        Array of shape ``(N, 3)`` with the rotated points.
    """
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return pts @ R.T


def rotate_x(pts: np.ndarray, ang: float) -> np.ndarray:
    """
    Rotates a set of 3D points around the X-axis.

    Applies the standard rotation matrix for the X-axis::

        | 1     0        0    |
        | 0   cos(ang) -sin(ang) |
        | 0   sin(ang)  cos(ang) |

    Args:
        pts: Array of shape ``(N, 3)`` with ``(x, y, z)`` coordinates.
        ang: Rotation angle in radians. Positive = tilts downwards.

    Returns:
        Array of shape ``(N, 3)`` with the rotated points.
    """
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return pts @ R.T


def project(pts3d: np.ndarray, cx: int, cy: int, fov: int = 650) -> list[tuple]:
    """
    Applies perspective projection to a set of 3D points.

    Uses the classic projection formula with a centered vanishing point::

        px = cx + x * fov / (z + fov)
        py = cy + y * fov / (z + fov)

    A higher ``fov`` value flattens the perspective (more orthographic);
    a lower value exaggerates it (more "fisheye").

    Args:
        pts3d: Array of shape ``(N, 3)`` with scaled 3D coordinates.
        cx: X coordinate of the projection center on the panel (pixels).
        cy: Y coordinate of the projection center on the panel (pixels).
        fov: Focal distance in pixels (synthetic field-of-view).
             Defaults to ``650``.

    Returns:
        List of tuples ``(px, py, z)`` where ``px`` and ``py`` are the
        projected coordinates in pixels and ``z`` is the original depth
        (used for drawing order).
    """
    out = []
    for x, y, z in pts3d:
        # Avoid division by zero or perspective inversion
        zz = z + fov
        if zz <= 1:
            zz = 1
        px = int(cx + x * fov / zz)
        py = int(cy + y * fov / zz)
        out.append((px, py, z))
    return out


def draw_capsule(
    img: np.ndarray,
    p1: tuple,
    p2: tuple,
    thickness: int,
    color: tuple,
    outline: tuple | None = None,
) -> None:
    """
    Draws a 2D capsule (segment with semicircular ends) on an image.

    Combines a thick line with two circles at the ends to achieve
    the capsule effect. Optionally draws a thin outline on top.

    Args:
        img: BGR NumPy image being drawn on (modified in-place).
        p1: Tuple ``(x, y)`` of the first end of the capsule.
        p2: Tuple ``(x, y)`` of the second end of the capsule.
        thickness: Diameter of the capsule in pixels.
        color: BGR color for the capsule fill.
        outline: BGR color of the thin outline stroke (1px thickness).
                 If ``None``, no outline is drawn.

    Returns:
        None. The image is modified directly.
    """
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    r = thickness // 2
    # Circles at the ends to round the capsule
    cv2.circle(img, p1, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r, color, -1, cv2.LINE_AA)
    if outline is not None:
        cv2.line(img, p1, p2, outline, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
#  Class: RoboticHandRenderer
# ──────────────────────────────────────────────────────────────────────────────

class RoboticHandRenderer:
    """
    Renders 3D robotic hand models from MediaPipe landmarks.

    Maintains the smoothed state (position and fade alpha) of up to two hands
    (``"Left"`` and ``"Right"``). Each call to :meth:`update` ingests the 21
    real landmarks of the current frame; each call to :meth:`draw` projects and
    paints the resulting model onto a NumPy panel.

    The visual appearance imitates a robotic prosthesis:

    - White casing on proximal and middle phalanges.
    - Black rubber pads on the distal phalanges (tips).
    - Silver mechanical joints with a central screw.
    - Central black panel on the palm.

    Attributes:
        state (dict): Dictionary with keys ``"Left"`` and ``"Right"``.
                      Each value is a dict with:

                      - ``"pts"`` — ``(21, 3)`` array of normalized and 
                        smoothed landmarks, or ``None`` if the hand
                        has not been detected yet.
                      - ``"alpha"`` — current opacity ``[0.0, 1.0]``
                        for the fade-in / fade-out effect.
    """

    def __init__(self) -> None:
        """
        Initializes the renderer with an empty state for both hands.

        Both hands start with ``pts = None`` and ``alpha = 0.0`` (invisible).
        """
        self.state: dict = {
            "Left":  {"pts": None, "alpha": 0.0},
            "Right": {"pts": None, "alpha": 0.0},
        }

    def update(self, label: str, landmarks) -> None:
        """
        Updates the state of a hand with the landmarks of the current frame.

        Performs three preprocessing steps on the 21 landmarks:

        1. **Centering**: translates all points so that the wrist
           (landmark 0) is at the origin ``(0, 0, 0)``.
        2. **Normalization**: divides by the wrist→MCP distance of the middle
           finger (landmark 9), making the size independent of the
           distance to the camera.
        3. **Z Amplification**: multiplies the depth axis by
           :data:`Z_AMPLIFY` so the tilt is visible in the
           perspective projection.

        Then applies linear interpolation (:func:`lerp_arr`) between the previous
        state and the new one to smooth the movement, and increases the fade-in alpha.

        Args:
            label: Visual label of the hand, ``"Left"`` or ``"Right"``.
            landmarks: Sequence of 21 MediaPipe landmark objects, each
                       with ``x``, ``y``, ``z`` attributes normalized
                       in the range ``[0, 1]``.

        Returns:
            None. Modifies ``self.state[label]`` in-place.
        """
        # Extracts the 21 3D coordinates into a NumPy array
        raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # Centers on the wrist (landmark 0 → origin)
        raw = raw - raw[0]

        # Normalizes by hand size (wrist→middle MCP distance)
        hand_size = np.linalg.norm(raw[9]) + 1e-6   # +eps avoids division by zero
        raw = raw / hand_size

        # Amplifies depth so the tilt is noticeable
        raw[:, 2] *= Z_AMPLIFY

        st = self.state[label]
        if st["pts"] is None:
            # First detection: no smoothing to avoid jumping from the origin
            st["pts"] = raw
        else:
            # Temporal smoothing: blend 35% new + 65% previous state
            st["pts"] = lerp_arr(st["pts"], raw, 0.35)

        # Fade-in: increases opacity up to 1.0
        st["alpha"] = min(1.0, st["alpha"] + 0.15)

    def fade_out(self, label: str) -> None:
        """
        Gradually reduces the opacity of an undetected hand (fade-out).

        Called every frame in which the ``label`` hand does not appear in the
        MediaPipe results. When ``alpha`` reaches ``0.0``, the hand
        stops being drawn.

        Args:
            label: Label of the hand to fade, ``"Left"`` or ``"Right"``.

        Returns:
            None. Modifies ``self.state[label]["alpha"]`` in-place.
        """
        st = self.state[label]
        st["alpha"] = max(0.0, st["alpha"] - 0.08)

    def draw(self, panel: np.ndarray, label: str, cx: int, cy: int) -> None:
        """
        Projects and draws the 3D robotic hand model on the panel.

        The method performs the following steps in order:

        1. Verifies that there is data and the alpha is visible.
        2. Applies a slight fixed 3/4 view rotation (``rotate_x`` +
           ``rotate_y``) so the hand doesn't look completely flat.
        3. Scales the landmarks by :data:`HAND_DISPLAY_SCALE` and projects
           them with perspective using :func:`project`.
        4. Draws the palm as a white convex polygon with an inner black panel
           and screws.
        5. Sorts the bones by average depth (Z) from furthest to closest
           (painter's algorithm) and draws them as capsules with the style
           corresponding to their ``kind``.
        6. Draws the mechanical joints (silver circle + ring + screw)
           at each bone end.
        7. Adds the ``"L"`` or ``"R"`` label below the model.

        Args:
            panel: BGR NumPy image of the side panel where it's drawn.
                   Modified in-place.
            label: Label of the hand to draw, ``"Left"`` or ``"Right"``.
            cx: X coordinate of the model's anchor center on the panel.
            cy: Y coordinate of the model's anchor center on the panel.

        Returns:
            None. Modifies ``panel`` in-place.
        """
        st = self.state[label]

        # Does not draw if the hand hasn't been detected yet or is totally invisible
        if st["alpha"] < 0.02 or st["pts"] is None:
            return

        alpha = st["alpha"]
        pts = st["pts"].copy()

        # ── 3/4 View: small static rotation to give a sense of volume ──
        # rotate_x slightly lifts the base; rotate_y turns a bit to the left
        pts = rotate_x(pts, 0.18)
        pts = rotate_y(pts, -0.12)

        # Scales to pixels and projects with perspective
        pts_scaled = pts * HAND_DISPLAY_SCALE
        proj = project(pts_scaled, cx, cy)
        P = [(p[0], p[1]) for p in proj]   # 2D projected coordinates
        Z = [p[2] for p in proj]            # original depths

        # ── 1. Robotic Palm ──────────────────────────────────────────────────
        hull_pts = np.array([P[i] for i in PALM_HULL], np.int32)

        # Fills the palm with alpha blend for the fade-in effect
        overlay = panel.copy()
        cv2.fillConvexPoly(overlay, cv2.convexHull(hull_pts), CASING_WHITE)
        cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, panel)

        # Palm outline
        cv2.polylines(panel, [cv2.convexHull(hull_pts)], True, CASING_SHADE, 2, cv2.LINE_AA)

        # Inner black panel (robotic accent — simulates the dark back of the prosthesis)
        inner = cv2.convexHull(hull_pts).reshape(-1, 2)
        center = inner.mean(axis=0)
        inner_small = ((inner - center) * 0.55 + center).astype(np.int32)
        cv2.fillConvexPoly(panel, inner_small, RUBBER_BLACK)

        # Decorative screws on the wrist and MCP of the index and pinky
        for sp in [P[0], P[5], P[17]]:
            cv2.circle(panel, sp, 3, SCREW_COL, -1, cv2.LINE_AA)

        # ── 2. Bones (painter's algorithm: from far to near) ─────────────────
        bone_draw = []
        for a, b, kind in BONES:
            # Average depth of the bone
            zavg = (Z[a] + Z[b]) / 2
            bone_draw.append((zavg, a, b, kind))

        # Sort from highest Z (far) to lowest Z (near) for correct overlapping
        bone_draw.sort(key=lambda x: -x[0])

        for zavg, a, b, kind in bone_draw:
            p1, p2 = P[a], P[b]

            # Base thickness according to segment type
            base_thick = (20 if kind == 2 else 15) if kind != 0 else 17

            # Thickness adjustment by perspective: closer bones look thicker
            depth_factor = max(0.6, min(1.4, 650 / (zavg + 650)))
            thick = max(6, int(base_thick * depth_factor))

            if kind == 2:
                # Distal phalanx → black rubber pad with light outline
                draw_capsule(panel, p1, p2, thick, RUBBER_BLACK, CASING_SHADE)
            else:
                # Proximal/middle phalanx → white casing with central black rubber stripe
                draw_capsule(panel, p1, p2, thick, CASING_WHITE, CASING_SHADE)
                draw_capsule(panel, p1, p2, max(3, thick // 3), RUBBER_BLACK)

            # Mechanical joint at the proximal end of the bone
            r = max(4, thick // 2)
            cv2.circle(panel, p1, r, JOINT_SILVER, -1, cv2.LINE_AA)  # silver fill
            cv2.circle(panel, p1, r, JOINT_RING,   2,  cv2.LINE_AA)  # outer ring
            cv2.circle(panel, p1, max(2, r // 2), SCREW_COL, -1, cv2.LINE_AA)  # screw

        # Final joint at the tip of each finger (landmarks 4, 8, 12, 16, 20)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(panel, P[tip], 5, JOINT_SILVER, -1, cv2.LINE_AA)

        # Identification label below the model
        cv2.putText(
            panel,
            "R" if label == "Right" else "L",
            (cx - 6, cy + 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LABEL, 1, cv2.LINE_AA,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Class: HandMirrorDetector
# ──────────────────────────────────────────────────────────────────────────────

class HandMirrorDetector:
    """
    Main controller for the Hand Mirror 3D system.

    Manages the video capture loop, MediaPipe Hands processing, and
    the final composition of the window (3D panel + camera feed).

    The resulting window is divided into two horizontal sections:

    - **Left** — panel of :data:`PANEL_W` px width with the two
      3D robotic hands (top slot = left, bottom slot = right).
    - **Right** — mirrored webcam feed with the MediaPipe skeleton
      superimposed.

    Attributes:
        window_name (str): OpenCV window title.
        mp_hands: ``mediapipe.solutions.hands`` module.
        mp_drawing: ``mediapipe.solutions.drawing_utils`` module.
        hands: Configured instance of ``MediaPipe.Hands`` for real-time
               detection of up to 2 hands.
        renderer (RoboticHandRenderer): 3D renderer instance.
    """

    def __init__(self) -> None:
        """
        Initializes MediaPipe Hands and the robotic renderer.

        Configures the detector with:

        - ``max_num_hands = 2`` — detects both hands simultaneously.
        - ``min_detection_confidence = 0.7`` — initial detection threshold.
        - ``min_tracking_confidence = 0.5`` — continuous tracking threshold.
        """
        self.window_name: str = "Hand Mirror 3D  (q = quit)"
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.renderer = RoboticHandRenderer()

    def _build_panel(self, h: int) -> np.ndarray:
        """
        Builds the empty left side panel with grid and labels.

        Creates a NumPy image of ``h × PANEL_W`` pixels with:

        - Dark background (:data:`COL_PANEL_BG`).
        - Grid lines every 40 px (:data:`COL_GRID`).
        - Right border (:data:`COL_DIVIDER`).
        - ``"HAND MIRROR 3D"`` title at the top.
        - ``"LEFT"`` and ``"RIGHT"`` labels for each slot.
        - Horizontal dividing line at ``h // 2``.

        Args:
            h: Height of the panel in pixels (must match the camera frame).

        Returns:
            NumPy array of shape ``(h, PANEL_W, 3)`` and dtype ``uint8``.
        """
        panel = np.full((h, PANEL_W, 3), COL_PANEL_BG, dtype=np.uint8)

        # Background grid
        for y in range(0, h, 40):
            cv2.line(panel, (0, y), (PANEL_W, y), COL_GRID, 1)
        for x in range(0, PANEL_W, 40):
            cv2.line(panel, (x, 0), (x, h), COL_GRID, 1)

        # Right border of the panel
        cv2.line(panel, (PANEL_W - 1, 0), (PANEL_W - 1, h), COL_DIVIDER, 2)

        # Title and top separator
        cv2.putText(panel, "HAND MIRROR 3D", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_LABEL, 1, cv2.LINE_AA)
        cv2.line(panel, (12, 38), (PANEL_W - 12, 38), COL_DIVIDER, 1)

        # Labels for each slot
        cv2.putText(panel, "LEFT", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_LABEL, 1, cv2.LINE_AA)
        cv2.putText(panel, "RIGHT", (12, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_LABEL, 1, cv2.LINE_AA)

        # Horizontal dividing line between the two slots
        cv2.line(panel, (12, h // 2), (PANEL_W - 12, h // 2), COL_DIVIDER, 1)

        return panel

    def run(self) -> None:
        """
        Executes the main capture, processing, and visualization loop.

        In each iteration of the loop:

        1. Reads a webcam frame and mirrors it horizontally.
        2. Resizes the frame to :data:`PANEL_H` pixels in height if necessary.
        3. Converts BGR → RGB and passes it through ``MediaPipe.Hands.process()``.
        4. For each detected hand:

           a. Draws the MediaPipe skeleton on the camera frame.
           b. Corrects the hand label (left/right) due to mirroring.
           c. Calls :meth:`RoboticHandRenderer.update` with the landmarks.

        5. Calls :meth:`RoboticHandRenderer.fade_out` for absent hands.
        6. Builds the side panel with :meth:`_build_panel`.
        7. Renders the two 3D models on the panel.
        8. Combines panel + frame with ``np.hstack`` and shows the window.
        9. Exits if the user presses ``q``.

        Raises:
            RuntimeError: If the webcam cannot be opened (``cv2.VideoCapture``
                          fails or returns ``isOpened() == False``).

        Returns:
            None. Blocks until the user closes the window.
        """
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Error: could not open the webcam.")

        print("🎥 Camera started. Show your hands and tilt them!")
        print("Press 'q' to quit.\n")

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    break

                # Mirrors horizontally for a natural mirror effect
                frame = cv2.flip(frame, 1)
                fh, fw, _ = frame.shape

                # Resizes to the panel height if it differs
                if fh != PANEL_H:
                    ratio = PANEL_H / fh
                    frame = cv2.resize(frame, (int(fw * ratio), PANEL_H))
                    fh, fw, _ = frame.shape

                # MediaPipe requires RGB, not BGR
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                # Set of visual labels detected in this frame
                visible: set[str] = set()

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_lm, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                    ):
                        # Draws the MediaPipe skeleton over the camera feed
                        self.mp_drawing.draw_landmarks(
                            frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(
                                color=(120, 200, 255), thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(
                                color=(60, 110, 200), thickness=2),
                        )

                        # Label correction due to frame mirroring:
                        # MediaPipe sees the mirrored frame, so its "Left"
                        # corresponds to the user's visual right hand.
                        mp_label = handedness.classification[0].label
                        visual = "Right" if mp_label == "Left" else "Left"

                        # Updates the renderer state with the real landmarks
                        self.renderer.update(visual, hand_lm.landmark)
                        visible.add(visual)

                # Fade-out for hands not detected in this frame
                for lbl in ("Left", "Right"):
                    if lbl not in visible:
                        self.renderer.fade_out(lbl)

                # Builds the panel and renders the two 3D models
                panel = self._build_panel(fh)
                cx = PANEL_W // 2
                self.renderer.draw(panel, "Left",  cx, fh // 4 + 95)
                self.renderer.draw(panel, "Right", cx, fh * 3 // 4 + 80)

                # Combines left panel + right camera feed
                combined = np.hstack([panel, frame])
                cv2.imshow(self.window_name, combined)

                # Quit upon pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Releases resources even if an exception occurs
            camera.release()
            cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HandMirrorDetector().run()