import sys
import numpy as np
from scipy.signal import fftconvolve
from matplotlib.image import imread
from skimage.transform import resize
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QSlider, QLabel, QHBoxLayout, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- Fonctions Utilitaires ---

def create_smiley(N, radius):
    y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
    mask_face = x**2 + y**2 <= radius**2
    eye_radius = int(radius * 0.13)
    eye_y = -int(radius * 0.35)
    eye_x_offset = int(radius * 0.45)
    mask_eye_l = (x + eye_x_offset)**2 + (y - eye_y)**2 <= eye_radius**2
    mask_eye_r = (x - eye_x_offset)**2 + (y - eye_y)**2 <= eye_radius**2
    mouth_radius = int(radius * 0.65)
    mouth_y = int(radius * 0.25)
    theta = np.linspace(np.pi/6, 5*np.pi/6, 100)
    mouth_x = (mouth_radius * np.cos(theta)).astype(int)
    mouth_y_arc = (mouth_radius * np.sin(theta)).astype(int) + mouth_y
    smiley = np.zeros((N, N))
    smiley[mask_face] = 0.2
    smiley[mask_eye_l] = 1.0
    smiley[mask_eye_r] = 1.0
    for mx, my in zip(mouth_x, mouth_y_arc):
        ix = N//2 + mx
        iy = N//2 + my
        if 0 <= ix < N and 0 <= iy < N:
            smiley[iy-1:iy+2, ix-1:ix+2] = 1.0
    return smiley

def gamma_correction(img, gamma=0.5):
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-10)
    return np.power(img_norm, gamma)


# --- Classe Principale ---

class OpticalSimulation(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- Paramètres ---
        self.N = 256
        self.R_aperture = 0.5
        self.width_trace = 0.08 
        self.strength_trace = 25.0 
        self.curvature = 0.4
        self.pressure = 3.0
        self.fingerprint_freq = 500
        self.global_aberration = 0.7
        
        x = np.linspace(-1, 1, self.N)
        y = np.linspace(-1, 1, self.N)
        self.X, self.Y = np.meshgrid(x, y)
        self.mask_aperture = (self.X**2 + self.Y**2) < self.R_aperture**2
        self.amplitude = np.zeros((self.N, self.N))
        self.amplitude[self.mask_aperture] = 1.0
        
        # Import Image
        try:
            img = imread('image copie.png') 
            if img.ndim == 3: img = np.mean(img, axis=2)
            img = resize(img, (self.N, self.N), mode='reflect', anti_aliasing=True)
            img = (img - img.min()) / (img.max() - img.min() + 1e-10)
            self.objet = img
        except Exception:
            self.objet = create_smiley(self.N, radius=int(self.N * 0.15))

        # --- GUI Setup ---
        self.setWindowTitle("Simulateur Optique & Reconstruction Complète")
        self.setGeometry(50, 50, 1100, 950)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # ==========================================
        # SECTION 1 : SIMULATION (HAUT - FIXÉ)
        # ==========================================
        
        # On crée un conteneur pour TOUT le haut
        self.top_container = QWidget()
        top_layout = QVBoxLayout(self.top_container)
        top_layout.setContentsMargins(0, 0, 0, 0) # Marges minimales
        
        # --> Titre Haut
        lbl_sim = QLabel("1. SIMULATION TEMPS RÉEL (Effet de la trace)")
        lbl_sim.setStyleSheet("font-weight: bold; background-color: #ddd; padding: 5px;")
        lbl_sim.setFixedHeight(30) # Fixe la hauteur du titre
        top_layout.addWidget(lbl_sim)

        # --> Sliders Haut (On groupe tout sur 2 lignes compactes)
        sim_controls_1 = QHBoxLayout()
        sim_controls_2 = QHBoxLayout()
        
        # Ligne 1 de sliders
        self.create_slider(sim_controls_1, "Rotation", self.update_simulation, 0, 180, 0, "slider_sim")
        self.lbl_angle = QLabel("0°")
        sim_controls_1.addWidget(self.lbl_angle)
        self.create_slider(sim_controls_1, "Largeur", self.update_sim_params, 1, 100, int(self.width_trace*100), "slider_width")
        self.create_slider(sim_controls_1, "Force", self.update_sim_params, 0, 100, int(self.strength_trace), "slider_strength")
        
        # Ligne 2 de sliders
        self.create_slider(sim_controls_2, "Courbure", self.update_sim_params, 0, 100, int(self.curvature*100), "slider_curvature")
        self.create_slider(sim_controls_2, "Pression", self.update_sim_params, 0, 100, int(self.pressure*10), "slider_pressure")
        self.create_slider(sim_controls_2, "Aberration", self.update_sim_params, 0, 100, int(self.global_aberration*100), "slider_global_aberration")
        self.create_slider(sim_controls_2, "Empreinte", self.update_sim_params, 0, 100, int(self.fingerprint_freq/10000*100), "slider_fingerprint")

        top_layout.addLayout(sim_controls_1)
        top_layout.addLayout(sim_controls_2)

        # --> Graphiques Haut (Très plats)
        # figsize=(largeur, hauteur_faible)
        self.fig_sim = Figure(figsize=(15, 3)) 
        self.canvas_sim = FigureCanvas(self.fig_sim)
        top_layout.addWidget(self.canvas_sim)
        
        # Ajout des subplots (1 ligne, 4 colonnes)
        self.ax_obj = self.fig_sim.add_subplot(141)
        self.ax_pupil = self.fig_sim.add_subplot(142)
        self.ax_psf = self.fig_sim.add_subplot(143)
        self.ax_img = self.fig_sim.add_subplot(144)
        self.fig_sim.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.05, wspace=0.1)

        # On fixe la hauteur de la figure du haut pour qu'elle ne prenne pas trop de place
        # La figure du haut prend 40% de la hauteur, celle du bas 60% (adaptatif)
        self.canvas_sim.setMinimumHeight(200)
        self.canvas_sim.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.top_container, stretch=2)

        # ==========================================
        # SECTION 2 : RECONSTRUCTION (BAS - EXTENSIBLE)
        # ==========================================
        
        lbl_recon = QLabel("2. RECONSTRUCTION MULTI-ANGLES (Comparaison)")
        lbl_recon.setStyleSheet("font-weight: bold; background-color: #cfc; padding: 5px;")
        lbl_recon.setFixedHeight(30)
        main_layout.addWidget(lbl_recon)

        recon_controls = QHBoxLayout()
        recon_controls.addWidget(QLabel("Nombre de captures :"))
        self.slider_recon = QSlider(Qt.Horizontal)
        self.slider_recon.setRange(1, 12)
        self.slider_recon.setValue(3)
        self.slider_recon.setTickPosition(QSlider.TicksBelow)
        self.slider_recon.valueChanged.connect(self.update_reconstruction)
        self.lbl_count = QLabel("3")
        self.lbl_count.setStyleSheet("font-weight: bold; color: blue;")
        recon_controls.addWidget(self.slider_recon)
        recon_controls.addWidget(self.lbl_count)
        main_layout.addLayout(recon_controls)

        # Canvas Bas (Prend toute la place restante)
        self.fig_recon = Figure(figsize=(12, 6))
        self.canvas_recon = FigureCanvas(self.fig_recon)
        self.canvas_recon.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.canvas_recon, stretch=3)
        
        self.ax_recon_orig = self.fig_recon.add_subplot(131)
        self.ax_recon_result = self.fig_recon.add_subplot(132)
        self.ax_recon_diff = self.fig_recon.add_subplot(133)
        self.fig_recon.subplots_adjust(wspace=0.1, left=0.05, right=0.95, top=0.90, bottom=0.05)

        # Initialisation
        self.init_plots_structure()
        self.update_simulation(0)
        self.update_reconstruction()

    def create_slider(self, layout, label, callback, vmin, vmax, val, attr_name):
        """Helper pour créer des sliders compacts"""
        layout.addWidget(QLabel(label + ":"))
        sl = QSlider(Qt.Horizontal)
        sl.setRange(vmin, vmax)
        sl.setValue(val)
        sl.valueChanged.connect(callback)
        setattr(self, attr_name, sl)
        layout.addWidget(sl)

    def init_plots_structure(self):
        # Simulation
        self.ax_obj.imshow(self.objet, cmap='gray')
        self.ax_obj.set_title("Objet Original", fontsize=8)
        self.ax_obj.axis('off')
        
        self.im_pupil = self.ax_pupil.imshow(np.zeros((self.N, self.N)), cmap='twilight', extent=[-1,1,-1,1], vmin=-np.pi, vmax=np.pi)
        self.ax_pupil.set_title("Phase", fontsize=8)
        self.ax_pupil.axis('off')

        zoom = self.N // 4
        self.im_psf = self.ax_psf.imshow(np.zeros((2*zoom, 2*zoom)), cmap='inferno')
        self.ax_psf.set_title("PSF", fontsize=8)
        self.ax_psf.axis('off')

        self.im_final = self.ax_img.imshow(np.zeros((self.N, self.N)), cmap='gray', vmin=0, vmax=1)
        self.ax_img.set_title("Convolution", fontsize=8)
        self.ax_img.axis('off')

        # Reconstruction
        self.im_recon_orig = self.ax_recon_orig.imshow(self.objet, cmap='gray', vmin=0, vmax=1)
        self.ax_recon_orig.set_title("1. Original", fontsize=10)
        self.ax_recon_orig.axis('off')
        
        self.im_recon_result = self.ax_recon_result.imshow(np.zeros((self.N, self.N)), cmap='gray', vmin=0, vmax=1)
        self.ax_recon_result.set_title("2. Reconstruction", fontsize=10)
        self.ax_recon_result.axis('off')
        
        self.im_recon_diff = self.ax_recon_diff.imshow(np.zeros((self.N, self.N)), cmap='hot', vmin=0, vmax=0.1)
        self.ax_recon_diff.set_title("3. Différence", fontsize=10)
        self.ax_recon_diff.axis('off')

    def update_sim_params(self):
        self.width_trace = self.slider_width.value() / 100.0
        self.curvature = self.slider_curvature.value() / 50.0
        self.strength_trace = self.slider_strength.value()
        self.pressure = self.slider_pressure.value() / 10.0
        self.fingerprint_freq = self.slider_fingerprint.value() * 10
        self.global_aberration = self.slider_global_aberration.value() / 50.0
        self.update_simulation()
        
    def compute_complex_pupil(self, angle_deg):
        theta = np.radians(angle_deg)
        Xr = self.X * np.cos(theta) + self.Y * np.sin(theta)
        Yr = -self.X * np.sin(theta) + self.Y * np.cos(theta)
        
        Xr_curved = Xr - self.curvature * (Yr**2)
        base_trace = np.exp(-Xr_curved**2 / (2*self.width_trace**2))
        pressure_profile = np.exp(-Yr**2 / (self.pressure + 0.1)) 
        fingerprint = 0.2 * np.sin(self.fingerprint_freq * Xr_curved) * base_trace
        aberration = self.global_aberration * (self.X**2 + self.Y**2)
        total_phase = self.strength_trace * base_trace * pressure_profile + fingerprint * 10 + aberration
        
        return self.amplitude * np.exp(1j * total_phase * self.mask_aperture)

    def update_simulation(self, angle_deg=None):
        if angle_deg is None: angle_deg = self.slider_sim.value()
        self.lbl_angle.setText(f"{angle_deg}°")
        
        pupil_complex = self.compute_complex_pupil(angle_deg)
        psf_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_complex)))
        psf = np.abs(psf_fft)**2
        psf = psf / (psf.sum() + 1e-20)
        img_convolved = fftconvolve(self.objet, psf, mode='same')
        
        self.im_pupil.set_data(np.angle(pupil_complex))
        mid = self.N // 2
        zoom = self.N // 4
        psf_crop = psf[mid-zoom:mid+zoom, mid-zoom:mid+zoom]
        self.im_psf.set_data(psf_crop)
        self.im_psf.set_clim(0, psf.max() * 0.05) 
        img_view = gamma_correction(img_convolved, gamma=0.45)
        self.im_final.set_data(img_view)
        self.canvas_sim.draw()

    def update_reconstruction(self):
        n_angles = self.slider_recon.value()
        self.lbl_count.setText(str(n_angles))
        
        angles_capture = np.linspace(0, 180, n_angles, endpoint=False)
        captured_images_fft = [] 
        psf_ffts = []            
        
        for angle in angles_capture:
            pupil_complex = self.compute_complex_pupil(angle)
            psf_fft_raw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_complex)))
            psf = np.abs(psf_fft_raw)**2
            psf /= psf.sum()
            img_blurred = fftconvolve(self.objet, psf, mode='same')
            
            captured_images_fft.append(np.fft.fft2(np.fft.ifftshift(img_blurred)))
            psf_ffts.append(np.fft.fft2(np.fft.ifftshift(psf)))

        numerateur = np.zeros_like(captured_images_fft[0], dtype=complex)
        denominateur = np.zeros_like(captured_images_fft[0], dtype=float)
        K = 0.0005 
        
        for G, H in zip(captured_images_fft, psf_ffts):
            numerateur += G * np.conj(H)
            denominateur += np.abs(H)**2
            
        F_final = numerateur / (denominateur + K)
        img_reconstructed = np.fft.ifft2(F_final)
        img_reconstructed = np.abs(np.fft.fftshift(img_reconstructed))

        self.im_recon_orig.set_data(self.objet)
        img_recon_clipped = np.clip(img_reconstructed, 0, 1)
        self.im_recon_result.set_data(gamma_correction(img_recon_clipped, 0.6))
        self.ax_recon_result.set_title(f"2. Reconstruction ({n_angles} angles)", fontsize=10)

        img_diff = np.abs(self.objet - img_recon_clipped)
        self.im_recon_diff.set_data(img_diff)
        max_diff = img_diff.max()
        clim_max = max_diff if max_diff > 1e-3 else 0.1
        self.im_recon_diff.set_clim(0, clim_max)

        self.canvas_recon.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpticalSimulation()
    window.show()
    sys.exit(app.exec_())