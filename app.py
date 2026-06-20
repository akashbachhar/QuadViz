"""
Quadruped Leg Visualizer — PyQt6 + pyqtgraph OpenGL, with IK and body pose.

Run:
    pip install pyqt6 pyqtgraph PyOpenGL numpy
    python quadruped_gui.py

Layout:
  * Top-left   : hardware-accelerated 3D view (pyqtgraph.opengl).
  * Right panel: Body pose (roll/pitch/yaw/x/y/height, "Pin feet"),
                 12 joint-angle sliders, inverse-kinematics inputs.
  * Bottom strip: link lengths, live foot positions, gait/reset buttons.
"""

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import sys
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ----------------------------------------------------------------------
# Kinematics
# ----------------------------------------------------------------------
def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_R(roll, pitch, yaw):
    r, p, y = np.deg2rad([roll, pitch, yaw])
    return Rz(y) @ Ry(p) @ Rx(r)


def _wrap(a):
    return (a + 180) % 360 - 180


class Leg:
    """3-DOF leg via composed rotations, expressed in the BODY frame.
       haa: hip abduction about body x   (frontal/YZ splay)
       hfe: hip flexion about local y    (sagittal swing, composed on haa)
       kfe: knee about local y           (relative to thigh)"""

    def __init__(self, hip, side, link_lengths=(0.3, 0.5, 0.5),
                 haa=0.0, hfe=0.0, kfe=0.0, color="red"):
        self.hip = np.array(hip, float)       # hip position in body frame
        self.side = side                       # +1 left, -1 right
        self.L = list(link_lengths)
        self.color = color
        self.set_angles(haa, hfe, kfe)

    def set_angles(self, haa, hfe, kfe):
        self.haa, self.hfe, self.kfe = haa, hfe, kfe
        self.points = self._fk()

    def _fk(self):
        L1, L2, L3 = self.L
        h = np.deg2rad([self.haa, self.hfe, self.kfe])
        A = self.hip
        R1 = Rx(h[0]); B = A + R1 @ np.array([0, self.side * L1, 0])   # abduction
        R2 = R1 @ Ry(h[1]); C = B + R2 @ np.array([0, 0, -L2])         # thigh
        R3 = R2 @ Ry(h[2]); D = C + R3 @ np.array([0, 0, -L3])         # shank
        return np.array([A, B, C, D])

    @property
    def end(self):
        return self.points[-1]

    def solve_ik(self, target_body, limits, current):
        """target_body: desired foot position in the BODY frame.
        Returns (status, angles_deg|None); status 'ok'|'reach'|'limit'."""
        L1, L2, L3 = self.L
        px, py, pz = np.array(target_body, float) - self.hip
        r = np.hypot(py, pz)
        sols = []
        if r > 1e-9 and abs(self.side * L1 / r) <= 1.0:
            base = np.arctan2(pz, py)
            d_ab = np.arccos(np.clip(self.side * L1 / r, -1, 1))
            for s_ab in (+1, -1):
                haa = base + s_ab * d_ab
                w = pz * np.cos(haa) - py * np.sin(haa)
                U, W = -px, -w
                d2 = U * U + W * W
                c2 = (d2 - L2 * L2 - L3 * L3) / (2 * L2 * L3)
                if abs(c2) <= 1.0:
                    for s_k in (+1, -1):
                        th2 = s_k * np.arccos(np.clip(c2, -1, 1))
                        th1 = np.arctan2(U, W) - np.arctan2(L3 * np.sin(th2),
                                                            L2 + L3 * np.cos(th2))
                        sols.append(np.array([_wrap(np.degrees(haa)),
                                              _wrap(np.degrees(th1)),
                                              _wrap(np.degrees(th2))]))
        if not sols:
            return "reach", None
        keys = ["haa", "hfe", "kfe"]
        ok = lambda s: all(limits[keys[i]][0] - 1e-6 <= s[i] <= limits[keys[i]][1] + 1e-6
                           for i in range(3))
        cur = np.array(current, float)
        valid = [s for s in sols if ok(s)]
        if valid:
            return "ok", min(valid, key=lambda s: np.sum((s - cur) ** 2))
        return "limit", min(sols, key=lambda s: np.sum((s - cur) ** 2))


class Quadruped:
    LEGS = {                       # name -> (front/back sign, left/right sign, color)
        "FL": (+1, +1, "red"),
        "FR": (+1, -1, "blue"),
        "BL": (-1, +1, "green"),
        "BR": (-1, -1, "purple"),
    }

    def __init__(self, body_length=0.6, body_width=0.4,
                 link_lengths=(0.3, 0.5, 0.5), stance=(8, 40, -80)):
        self.body_length, self.body_width = body_length, body_width
        self.legs = {}
        for name, (sx, sy, col) in self.LEGS.items():
            hip = np.array([sx * body_length / 2, sy * body_width / 2, 0.0])
            self.legs[name] = Leg(hip, side=sy, link_lengths=link_lengths,
                                  haa=stance[0], hfe=stance[1], kfe=stance[2], color=col)

    def hips_loop(self):
        return np.array([self.legs[k].hip for k in ["FL", "FR", "BR", "BL", "FL"]])


def trot(t, leg, period=0.6, A_hfe=25, A_kfe=35, base=(8, 40, -80)):
    offset = {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}[leg]
    ph = ((t / period) + offset) % 1.0
    haa = base[0]
    hfe = base[1] + A_hfe * np.sin(2 * np.pi * ph)
    kfe = base[2] - A_kfe * max(0.0, np.sin(2 * np.pi * ph))
    return haa, hfe, kfe


# ----------------------------------------------------------------------
# GUI constants
# ----------------------------------------------------------------------
JOINTS = ["haa", "hfe", "kfe"]
JOINT_LABEL = {"haa": "Hip abduction", "hfe": "Hip flexion", "kfe": "Knee"}
RANGES = {"haa": (-60, 60), "hfe": (-30, 120), "kfe": (-150, 30)}
DEFAULT = {"haa": 8, "hfe": 40, "kfe": -80}
LEG_ORDER = ["FL", "FR", "BL", "BR"]
AXES = ["x", "y", "z"]
LINK_LABEL = ["L1 hip", "L2 thigh", "L3 shank"]

GLCOLOR = {
    "red":    (0.86, 0.18, 0.18, 1.0),
    "blue":   (0.20, 0.42, 0.95, 1.0),
    "green":  (0.13, 0.70, 0.24, 1.0),
    "purple": (0.62, 0.24, 0.82, 1.0),
    "black":  (0.85, 0.85, 0.90, 1.0),   # body frame, light on dark bg
    "bad":    (0.95, 0.15, 0.15, 1.0),
}

POSE = [("roll", -30, 30, 0.0, "\u00b0"),
        ("pitch", -30, 30, 0.0, "\u00b0"),
        ("yaw", -30, 30, 0.0, "\u00b0"),
        ("x", -0.30, 0.30, 0.0, " m"),
        ("y", -0.30, 0.30, 0.0, " m"),
        ("height", 0.20, 1.20, 0.72, " m")]


class RobotView(gl.GLViewWidget):
    """Main 3D view plus a screen-corner orientation gizmo: x/y/z arrows that
    mirror the camera rotation, like a CAD viewer. The gizmo is a small second
    GLViewWidget overlaid in the bottom-left, with its camera slaved to this
    view's azimuth/elevation."""

    GZ = 120  # gizmo size in px

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gizmo = gl.GLViewWidget(parent=self)
        self.gizmo.setFixedSize(self.GZ, self.GZ)
        _fmt = QtGui.QSurfaceFormat()
        _fmt.setAlphaBufferSize(8)
        self.gizmo.setFormat(_fmt)
        self.gizmo.setAttribute(QtCore.Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        self.gizmo.setBackgroundColor(pg.mkColor(0, 0, 0, 0))   # transparent
        self.gizmo.setCameraPosition(distance=3.2)
        self.gizmo.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._build_gizmo()
        self._sync_timer = QtCore.QTimer(self)
        self._sync_timer.timeout.connect(self._sync_gizmo)
        self._sync_timer.start(40)

    def _build_gizmo(self):
        I = np.eye(3)
        Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)   # +z -> +x
        Rxm90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # +z -> +y
        defs = [((1, 0, 0), (0.90, 0.22, 0.22, 1.0), "x", Ry90),
                ((0, 1, 0), (0.30, 0.80, 0.32, 1.0), "y", Rxm90),
                ((0, 0, 1), (0.32, 0.52, 1.00, 1.0), "z", I)]
        for axis, color, name, R in defs:
            a = np.array(axis, float)
            shaft = gl.GLLinePlotItem(pos=np.vstack([[0, 0, 0], a * 0.72]),
                                      color=color, width=3, antialias=True)
            self.gizmo.addItem(shaft)
            md = gl.MeshData.cylinder(rows=2, cols=16, radius=[0.07, 0.0], length=0.28)
            head = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader="shaded")
            M = np.eye(4); M[:3, :3] = R; M[:3, 3] = a * 0.72
            head.setTransform(QtGui.QMatrix4x4(*M.flatten().tolist()))
            self.gizmo.addItem(head)
            try:
                lab = gl.GLTextItem(
                    pos=a * 1.15, text=name,
                    color=QtGui.QColor(int(color[0] * 255),
                                       int(color[1] * 255),
                                       int(color[2] * 255)))
                self.gizmo.addItem(lab)
            except Exception:
                pass  # GLTextItem unavailable on very old pyqtgraph

    def _sync_gizmo(self):
        self.gizmo.setCameraPosition(azimuth=self.opts["azimuth"],
                                     elevation=self.opts["elevation"],
                                     distance=3.2)

    def _place_gizmo(self):
        m = 10
        self.gizmo.move(m, self.height() - self.gizmo.height() - m)
        self.gizmo.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_gizmo()

    def showEvent(self, e):
        super().showEvent(e)
        self._place_gizmo()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quadruped Visualizer — OpenGL · FK · IK · body pose")
        self.resize(1340, 960)

        self.robot = Quadruped()
        self._t = 0.0
        self.pinned = {}                 # leg -> world foot target
        self.unreachable = set()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # ============ TOP ROW: view (left) + side controls (right) ============
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, stretch=1)

        # ---- OpenGL 3D view (top-left) ----
        self.view = RobotView()
        self.view.setBackgroundColor(pg.mkColor("#0e1116"))
        self.view.setCameraPosition(pos=pg.Vector(0, 0, 0.35),
                                    distance=3.0, elevation=18, azimuth=-60)
        top.addWidget(self.view, stretch=3)

        # ---- right side panel (scrollable) ----
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(480)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        top.addWidget(scroll, stretch=2)

        side = QtWidgets.QWidget()
        scroll.setWidget(side)
        sv = QtWidgets.QVBoxLayout(side)
        sv.setSpacing(10)

        self.pose_ctrl, self.pose_lab = {}, {}
        self.length_ctrl = {}
        self.sliders, self.value_labels = {}, {}
        self.coord_boxes, self.ik_status = {}, {}

        for leg in LEG_ORDER:
            sv.addWidget(self._slider_block(leg))
        sv.addWidget(self._ik_block())
        sv.addWidget(self._pose_block())
        sv.addStretch(1)

        # ============ BOTTOM STRIP: lengths + feet + buttons ============
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(10)
        root.addLayout(bottom)

        bottom.addWidget(self._length_block())

        self.foot_card = QtWidgets.QLabel()
        self.foot_card.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.foot_card.setStyleSheet(
            "QLabel { background:#1f2430; border-radius:10px; padding:12px; }")
        bottom.addWidget(self._wrap_titled("Foot positions (world, live)",
                                           self.foot_card), stretch=1)

        ctrl = QtWidgets.QGroupBox("Controls")
        ctrl.setStyleSheet("QGroupBox{font-weight:bold;}")
        cv = QtWidgets.QVBoxLayout(ctrl)
        self.play_btn = QtWidgets.QPushButton("\u25B6  Play gait")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        reset_btn = QtWidgets.QPushButton("Reset stance")
        reset_btn.clicked.connect(self._reset)
        for b in (self.play_btn, reset_btn):
            b.setMinimumHeight(34); b.setMinimumWidth(130); cv.addWidget(b)
        bottom.addWidget(ctrl)

        # ---- animation timer ----
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)

        self._build_view()
        self._sync_pins()
        self._refresh()

    # -- panel builders --------------------------------------------------
    def _pose_block(self):
        box = QtWidgets.QGroupBox("Body pose")
        box.setStyleSheet("QGroupBox{font-weight:bold;}")
        v = QtWidgets.QVBoxLayout(box)
        for name, lo, hi, default, unit in POSE:
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(name.capitalize()); lab.setMinimumWidth(64)
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setRange(0, 1000)
            s.setProperty("lo", lo); s.setProperty("hi", hi); s.setProperty("unit", unit)
            s.setValue(int(round((default - lo) / (hi - lo) * 1000)))
            s.valueChanged.connect(self._on_pose)
            val = QtWidgets.QLabel(); val.setMinimumWidth(58)
            row.addWidget(lab); row.addWidget(s, 1); row.addWidget(val)
            self.pose_ctrl[name] = s
            self.pose_lab[name] = val
            v.addLayout(row)
        opts = QtWidgets.QHBoxLayout()
        self.pin_chk = QtWidgets.QCheckBox("Pin feet to ground")
        self.pin_chk.setChecked(True)
        self.pin_chk.toggled.connect(self._on_pin_toggle)
        opts.addWidget(self.pin_chk)
        self.pose_warn = QtWidgets.QLabel("")
        self.pose_warn.setTextFormat(QtCore.Qt.TextFormat.RichText)
        opts.addWidget(self.pose_warn); opts.addStretch(1)
        v.addLayout(opts)
        return box

    def _pose_value(self, name):
        s = self.pose_ctrl[name]
        lo, hi = s.property("lo"), s.property("hi")
        return lo + (hi - lo) * s.value() / 1000.0

    def _length_block(self):
        box = QtWidgets.QGroupBox("Link lengths")
        box.setStyleSheet("QGroupBox{font-weight:bold;}")
        row = QtWidgets.QHBoxLayout(box)
        defaults = self.robot.legs["FL"].L
        for i, label in enumerate(LINK_LABEL):
            row.addWidget(QtWidgets.QLabel(label))
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(0.05, 1.0); sb.setSingleStep(0.01); sb.setDecimals(2)
            sb.setValue(defaults[i])
            sb.valueChanged.connect(self._on_length)
            self.length_ctrl[i] = sb
            row.addWidget(sb)
        return box

    def _slider_block(self, leg):
        box = QtWidgets.QGroupBox(f"{leg}  \u2014  joint angles")
        color = Quadruped.LEGS[leg][2]
        box.setStyleSheet(f"QGroupBox{{font-weight:bold; color:{color};}}")
        v = QtWidgets.QVBoxLayout(box)
        for j in JOINTS:
            row = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel(JOINT_LABEL[j]); name.setMinimumWidth(95)
            lo, hi = RANGES[j]
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setRange(lo, hi); s.setValue(DEFAULT[j])
            s.valueChanged.connect(self._on_slider)
            val = QtWidgets.QLabel(f"{DEFAULT[j]:>4}\u00b0"); val.setMinimumWidth(40)
            row.addWidget(name); row.addWidget(s, 1); row.addWidget(val)
            self.sliders[(leg, j)] = s
            self.value_labels[(leg, j)] = val
            v.addLayout(row)
        return box

    def _ik_block(self):
        box = QtWidgets.QGroupBox("Inverse kinematics  \u2014  type a world foot target")
        box.setStyleSheet("QGroupBox{font-weight:bold;}")
        grid = QtWidgets.QGridLayout(box)
        grid.setHorizontalSpacing(6)
        for c, head in enumerate(["", "x", "y", "z", "", ""]):
            lab = QtWidgets.QLabel(head); lab.setStyleSheet("font-weight:bold; color:#555;")
            grid.addWidget(lab, 0, c)
        for r, leg in enumerate(LEG_ORDER, start=1):
            color = Quadruped.LEGS[leg][2]
            tag = QtWidgets.QLabel(leg); tag.setStyleSheet(f"font-weight:bold; color:{color};")
            grid.addWidget(tag, r, 0)
            for c, ax in enumerate(AXES, start=1):
                sb = QtWidgets.QDoubleSpinBox()
                sb.setRange(-3.0, 3.0); sb.setSingleStep(0.02); sb.setDecimals(2)
                sb.setMinimumWidth(64)
                self.coord_boxes[(leg, ax)] = sb
                grid.addWidget(sb, r, c)
            go = QtWidgets.QPushButton("Go"); go.setMaximumWidth(40)
            go.clicked.connect(lambda _, lg=leg: self._go(lg))
            grid.addWidget(go, r, 4)
            st = QtWidgets.QLabel(""); st.setMinimumWidth(96)
            self.ik_status[leg] = st
            grid.addWidget(st, r, 5)
        return box

    @staticmethod
    def _wrap_titled(title, widget):
        box = QtWidgets.QGroupBox(title)
        box.setStyleSheet("QGroupBox{font-weight:bold;}")
        lay = QtWidgets.QVBoxLayout(box); lay.addWidget(widget)
        return box

    # -- body transform --------------------------------------------------
    def body_R(self):
        return body_R(self._pose_value("roll"),
                      self._pose_value("pitch"),
                      self._pose_value("yaw"))

    def body_p(self):
        return np.array([self._pose_value("x"),
                         self._pose_value("y"),
                         self._pose_value("height")])

    def to_world(self, pts_body):
        return (self.body_R() @ np.atleast_2d(pts_body).T).T + self.body_p()

    # -- OpenGL scene ----------------------------------------------------
    def _build_view(self):
        grid = gl.GLGridItem(); grid.setSize(3, 3); grid.setSpacing(0.25, 0.25)
        grid.setColor((255, 255, 255, 40))
        self.view.addItem(grid)

        self.body_item = gl.GLLinePlotItem(width=5, antialias=True, mode="line_strip",
                                           color=GLCOLOR["black"])
        self.view.addItem(self.body_item)
        self.leg_items, self.foot_items = {}, {}
        for name, leg in self.robot.legs.items():
            li = gl.GLLinePlotItem(width=4, antialias=True, mode="line_strip",
                                   color=GLCOLOR[leg.color])
            fi = gl.GLScatterPlotItem(size=12, color=GLCOLOR[leg.color])
            self.view.addItem(li); self.view.addItem(fi)
            self.leg_items[name] = li; self.foot_items[name] = fi

    def _refresh(self):
        rows = []
        body_world = self.to_world(self.robot.hips_loop())
        self.body_item.setData(pos=body_world)
        for name, leg in self.robot.legs.items():
            haa = self.sliders[(name, "haa")].value()
            hfe = self.sliders[(name, "hfe")].value()
            kfe = self.sliders[(name, "kfe")].value()
            leg.set_angles(haa, hfe, kfe)
            pw = self.to_world(leg.points)
            col = GLCOLOR["bad"] if name in self.unreachable else GLCOLOR[leg.color]
            self.leg_items[name].setData(pos=pw, color=col)
            self.foot_items[name].setData(pos=pw[-1:].copy())
            fx, fy, fz = pw[-1]
            for ax, val in zip(AXES, (fx, fy, fz)):
                sb = self.coord_boxes[(name, ax)]
                if not sb.hasFocus():
                    sb.blockSignals(True); sb.setValue(float(val)); sb.blockSignals(False)
            c = Quadruped.LEGS[name][2]
            rows.append(
                f"<tr><td style='padding-right:10px'>"
                f"<span style='color:{c};font-size:15px'>&#9679;</span> "
                f"<b style='color:{c}'>{name}</b></td>"
                f"<td style='color:#9aa4b2'>x</td><td style='color:#e8eaed;padding-right:8px'>{fx:+.2f}</td>"
                f"<td style='color:#9aa4b2'>y</td><td style='color:#e8eaed;padding-right:8px'>{fy:+.2f}</td>"
                f"<td style='color:#9aa4b2'>z</td><td style='color:#e8eaed'>{fz:+.2f}</td></tr>")
        self.foot_card.setText(
            "<div style='font-family:Menlo,Consolas,monospace;font-size:13px'>"
            "<table cellspacing='6'>" + "".join(rows) + "</table></div>")
        for name, lab in self.pose_lab.items():
            unit = self.pose_ctrl[name].property("unit")
            v = self._pose_value(name)
            lab.setText(f"{v:+.2f}{unit}" if "m" in unit else f"{v:+.0f}{unit}")

    # -- pin helpers -----------------------------------------------------
    def _sync_pins(self):
        for name, leg in self.robot.legs.items():
            self.pinned[name] = self.to_world(leg.end)[0]

    def _solve_pose(self):
        R, p = self.body_R(), self.body_p()
        self.unreachable = set()
        warn = []
        for name, leg in self.robot.legs.items():
            target_body = R.T @ (self.pinned[name] - p)
            cur = [self.sliders[(name, j)].value() for j in JOINTS]
            status, angles = leg.solve_ik(target_body, RANGES, cur)
            if status == "ok":
                for j, a in zip(JOINTS, angles):
                    s = self.sliders[(name, j)]
                    s.blockSignals(True); s.setValue(int(round(a))); s.blockSignals(False)
                    self.value_labels[(name, j)].setText(f"{int(round(a)):>4}\u00b0")
            else:
                self.unreachable.add(name); warn.append(name)
        self.pose_warn.setText(
            f"<span style='color:#c47f00'>&#9651; can't hold: {', '.join(warn)}</span>"
            if warn else "")
        self._refresh()

    # -- handlers --------------------------------------------------------
    def _on_slider(self):
        for (leg, j), s in self.sliders.items():
            self.value_labels[(leg, j)].setText(f"{s.value():>4}\u00b0")
        self.unreachable = set()
        self._refresh()
        if self.pin_chk.isChecked():
            self._sync_pins()

    def _on_pose(self):
        if self.pin_chk.isChecked():
            self._solve_pose()
        else:
            self._refresh()
            self._sync_pins()

    def _on_pin_toggle(self, on):
        if on:
            self._sync_pins()
        else:
            self.unreachable = set()
            self.pose_warn.setText("")
            self._refresh()

    def _on_length(self):
        L = [self.length_ctrl[i].value() for i in range(3)]
        for leg in self.robot.legs.values():
            leg.L = list(L)
            leg.set_angles(leg.haa, leg.hfe, leg.kfe)
        self.unreachable = set()
        self._refresh()
        if self.pin_chk.isChecked():
            self._sync_pins()

    def _go(self, leg):
        world_target = np.array([self.coord_boxes[(leg, ax)].value() for ax in AXES])
        target_body = self.body_R().T @ (world_target - self.body_p())
        cur = [self.sliders[(leg, j)].value() for j in JOINTS]
        status, angles = self.robot.legs[leg].solve_ik(target_body, RANGES, cur)
        lab = self.ik_status[leg]
        if status == "ok":
            for j, a in zip(JOINTS, angles):
                s = self.sliders[(leg, j)]
                s.blockSignals(True); s.setValue(int(round(a))); s.blockSignals(False)
            self.pinned[leg] = world_target
            self.unreachable.discard(leg)
            self._on_slider_silent()
            lab.setText("<span style='color:#1a8a3a'>&#10003; reached</span>")
        elif status == "limit":
            lab.setText("<span style='color:#c47f00'>&#9651; joint limit</span>")
        else:
            lab.setText("<span style='color:#c0392b'>&#10007; out of reach</span>")

    def _on_slider_silent(self):
        # update labels + plot without re-syncing pins (used after IK Go)
        for (leg, j), s in self.sliders.items():
            self.value_labels[(leg, j)].setText(f"{s.value():>4}\u00b0")
        self._refresh()

    def _reset(self):
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)
        for name, lo, hi, default, unit in POSE:
            s = self.pose_ctrl[name]
            s.blockSignals(True)
            s.setValue(int(round((default - lo) / (hi - lo) * 1000)))
            s.blockSignals(False)
        for key, s in self.sliders.items():
            s.blockSignals(True); s.setValue(DEFAULT[key[1]]); s.blockSignals(False)
        for lab in self.ik_status.values():
            lab.setText("")
        self.unreachable = set(); self.pose_warn.setText("")
        for (leg, j), s in self.sliders.items():
            self.value_labels[(leg, j)].setText(f"{s.value():>4}\u00b0")
        self._refresh()
        self._sync_pins()

    def _toggle_play(self, on):
        self.play_btn.setText("\u23F8  Pause" if on else "\u25B6  Play gait")
        if on:
            self._t = 0.0; self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        self._t += self.timer.interval() / 1000.0
        for leg in LEG_ORDER:
            haa, hfe, kfe = trot(self._t, leg)
            for j, a in zip(JOINTS, (haa, hfe, kfe)):
                s = self.sliders[(leg, j)]
                s.blockSignals(True); s.setValue(int(round(a))); s.blockSignals(False)
        self._on_slider()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()