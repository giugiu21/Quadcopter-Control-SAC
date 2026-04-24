#Script LUA vuoto nella scena

import numpy as np

# ===========================================================================
# 1. UTILITY MATEMATICHE
# ===========================================================================
def skew(v: np.ndarray) -> np.ndarray:
    """S(.) usato nel paper (es. nell'eq. 2.26)."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

#Vettore corrispondente estratto dalla matrice skewsimmetrica
def vee(M: np.ndarray) -> np.ndarray:
    """vee(S(v)) = v. Necessario per costruire e_R su SO(3)."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def Rz(psi: float) -> np.ndarray:
    """Rotazione attorno a z di angolo psi (yaw)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def quat_to_R(q):
    """Quaternione CoppeliaSim (x,y,z,w) -> matrice di rotazione R."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])


# ===========================================================================
# 2. PARAMETRI DEL QUADROTOR
# ===========================================================================
class QuadrotorParams:
    """
    Notazione del paper:
      m   = massa
      JB  = matrice di inerzia in body frame
      L   = lunghezza dei bracci (4 bracci ortogonali)
      cf  = coefficiente di spinta
      c_t = coefficiente di coppia di reazione (c_tau nel paper)
      g   = gravita'
    Da adattare al modello che usi nella scena CoppeliaSim.
    """
    def __init__(self):
        self.m   = 0.5                                # [kg] 0.12 in teoria? ricca che stai a fa
        self.g   = 9.81                                 # [m/s^2]
        self.L   = 0.17                                 # [m]
        self.cf  = 1.0e-5                               # [N/(rad/s)^2]
        self.c_t = 1.0e-7                               # [Nm/(rad/s)^2]
        self.JB  = np.diag([3.2e-3, 3.2e-3, 5.5e-3])    # [kg m^2]


# ===========================================================================
# 3. WRENCH MAP & ALLOCATION MATRIX 
# ===========================================================================
class AllocationReference:
    """
    F~ del paper:

       [ f ]   [ cf      cf      cf      cf    ] [ u1 ]
       [m_x] = [ 0       cf*L    0      -cf*L  ] [ u2 ]
       [m_y]   [-cf*L    0       cf*L   0      ] [ u3 ]
       [m_z]   [ c_t    -c_t     c_t   -c_t    ] [ u4 ]

    det(F~) = -8 * cf^3 * c_t * L^2  (vedi paper) -> invertibile.
    """
    def __init__(self, p: QuadrotorParams):
        cf, ct, L = p.cf, p.c_t, p.L
        self.F_tilde = np.array([
            [ cf,    cf,    cf,    cf  ],
            [ 0,     cf*L,  0,    -cf*L],
            [-cf*L,  0,     cf*L,  0   ],
            [ ct,   -ct,    ct,   -ct  ]
        ])
        self.F_tilde_inv = np.linalg.inv(self.F_tilde)


# ===========================================================================
# 4. OUTER LOOP -- POSITION CONTROL
# ===========================================================================
class PositionPD:
    """
       f_d = m * p_R_ddot_d
             - K1 (p_R - p_d)
             - K2 (p_R_dot - p_d_dot)
             + m * g * e3
    """
    def __init__(self, K1, K2, m, g, i_clamp=2.0):
        self.K1 = np.diag(K1)
        self.K2 = np.diag(K2)
        self.m = m
        self.g = g
        self.e3 = np.array([0.0, 0.0, 1.0])
        #self.integ = np.zeros(3)
        self.i_clamp = i_clamp

    def reset(self):
        #self.integ[:] = 0.0
        pass

    def compute(self, p, p_dot, p_d, p_d_dot, p_d_ddot, dt):
        e_p = p - p_d
        e_v = p_dot - p_d_dot

        #self.integ += e_p * dt
        #self.integ = np.clip(self.integ, -self.i_clamp, self.i_clamp)

        f_d = ( self.m * p_d_ddot
                - self.K1 @ e_p
                - self.K2 @ e_v
                + self.m * self.g * self.e3 )
        return f_d, e_p, e_v


# ===========================================================================
# 5. CALCOLO DI R_B^d  
# ===========================================================================
def desired_attitude(f_d: np.ndarray, psi_d: float):

    """
    z_B^d  = f_d / ||f_d||
    x_Bp_d = Rz(psi_d) [1 0 0]^T
    y_B^d  = (z_B^d x x_Bp_d) / ||...||
    x_B^d  = y_B^d x z_B^d
    """
    norm_fd = np.linalg.norm(f_d)
    if norm_fd < 1e-9:
        return np.eye(3) # se f_d è quasi nullo, non possiamo definire z_B^d. la matrice di rotazione diventa identità e 

    z_B_d  = f_d / norm_fd
    x_Bp_d = Rz(psi_d) @ np.array([1.0, 0.0, 0.0])

    y_B_d_unnorm = np.cross(z_B_d, x_Bp_d)
    n = np.linalg.norm(y_B_d_unnorm)
    if n < 1e-9:
        x_arb = np.array([1.0, 0.0, 0.0])
        if abs(z_B_d @ x_arb) > 0.9:
            x_arb = np.array([0.0, 1.0, 0.0])
        y_B_d = np.cross(z_B_d, x_arb)
        y_B_d /= np.linalg.norm(y_B_d)
    else:
        y_B_d = y_B_d_unnorm / n

    x_B_d = np.cross(y_B_d, z_B_d)
    return np.column_stack((x_B_d, y_B_d, z_B_d))


# ===========================================================================
# 6. INNER LOOP -- ATTITUDE CONTROL
# ===========================================================================
class AttitudePD:
    """
       m = -Km1 * e_R - Km2 * e_omega
           + omega_R x JB omega_R
           - JB ( S(omega_R) R^T R_d omega_d - R^T R_d omega_d_dot )

       e_R     = 0.5 * vee(R_d^T R - R^T R_d)
       e_omega = omega_R - R^T R_d omega_d
    """
    def __init__(self, Km1, Km2, JB, i_clamp=0.5):
        self.Km1 = np.diag(Km1)
        self.Km2 = np.diag(Km2)
        self.JB  = JB
        self.integ = np.zeros(3)
        self.i_clamp = i_clamp

    def reset(self):
        self.integ[:] = 0.0

    def compute(self, R, omega_R, R_d, omega_d, omega_d_dot, dt):
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        e_omega = omega_R - R.T @ R_d @ omega_d

        self.integ += e_R * dt
        self.integ = np.clip(self.integ, -self.i_clamp, self.i_clamp)

        ff = ( skew(omega_R) @ R.T @ R_d @ omega_d
               - R.T @ R_d @ omega_d_dot )

        m = ( - self.Km1 @ e_R
              - self.Km2 @ e_omega
              # - self.Kmi @ self.integ
              + np.cross(omega_R, self.JB @ omega_R)
              - self.JB @ ff )
        return m, e_R, e_omega


# ===========================================================================
# 7. CONTROLLORE COMPLETO
# ===========================================================================
class QuadrotorController:
    """

       p_d, psi_d --> [Position PD] --f_d--> [build R_d]
                                                  |
                                                  v
                       [Attitude PD] -----------> m
                                                  |
                       proiezione f_d . z_B ---> f
                                                  |
                       --> (f, m) applicati DIRETTAMENTE al body, senza dover passare dalla matrice di allocazione
                       di fatto è quella che permette di passare da forze ad input, o meglio spinning rates dei propellers
    """
    def __init__(self, params: QuadrotorParams):
        self.p = params
        self.alloc = AllocationReference(params)  # solo per riferimento, non viene usata a meno che non vogliamo dare gli inputs direttamente a livello di propellers e non di forze

        # da m=0.5 kg, JB ~ diag(3.2e-3, 3.2e-3, 5.5e-3).
        # Errore steady-state su cerchio (raggio 1 m, omega 0.5 rad/s): ~4 mm.
        # REGOLA DI TUNING (utile se cambi massa/inerzia):
        #   - I gain di attitude scalano ~linearmente con JB. Se il tuo
        #     drone ha JB doppia, raddoppia Km1 e Km2.
        #   - I gain di posizione scalano ~linearmente con m. Stessa cosa.
        #   - Tieni il rapporto critico K2 ~ 2*sqrt(K1*m) (smorzamento ~1).
        #     Idem per attitude: Km2 ~ 2*sqrt(Km1*JB_ii).
        #   - Inner loop deve essere PIU' RAPIDO dell'outer (Fig. 2.9):
        #     larghezza di banda attitude ~ 5-10x quella di posizione.
        self.pos = PositionPD(
            K1=[3.0, 3.0, 5.0],
            K2=[2.5, 2.5, 4.0],
            m=params.m, g=params.g
        )
        self.att = AttitudePD(
            Km1=[2.5, 2.5, 1.0],
            Km2=[0.5, 0.5, 0.3],
            JB=params.JB
        )

    def step(self, state, ref, dt):
        # Outer loop 
        f_d, e_p, e_v = self.pos.compute(
            state['p'], state['p_dot'],
            ref['p_d'], ref['p_d_dot'], ref['p_d_ddot'], dt
        )
        # R_d 
        R_d = desired_attitude(f_d, ref['psi_d'])
        
        z_B = state['R'][:, 2]
        f_scalar = max(float(f_d @ z_B), 0.0)
        # Inner loop
        m, e_R, e_omega = self.att.compute(
            state['R'], state['omega'],
            R_d, ref['omega_d'], ref['omega_d_dot'], dt
        )
        return {
            'f': f_scalar, 'm': m, 'f_d': f_d, 'R_d': R_d,
            'e_p': e_p, 'e_v': e_v, 'e_R': e_R, 'e_omega': e_omega
        }


# ===========================================================================
# 8. PIANIFICATORE DI TRAIETTORIA 
# ===========================================================================
def min_jerk_segment(p0, p1, T, t):
    """
    Polinomio di grado 5 (min-jerk) tra p0 e p1 in tempo T.
    Restituisce p(t), p_dot(t), p_ddot(t).
    Garantisce posizione, velocita' e accelerazione continue ai bordi
    (importante per non avere salti su p_d_ddot dell'eq. 2.24).
    """
    if T <= 0:
        return p1.copy(), np.zeros_like(p1), np.zeros_like(p1)
    s = np.clip(t / T, 0.0, 1.0)
    h   = 10*s**3 - 15*s**4 + 6*s**5
    hd  = (30*s**2 - 60*s**3 + 30*s**4) / T
    hdd = (60*s    - 180*s**2 + 120*s**3) / (T*T)
    p   = p0 + (p1 - p0) * h
    pd  = (p1 - p0) * hd
    pdd = (p1 - p0) * hdd
    return p, pd, pdd


#Classe da pulire? perchè ci stanno 3 trajectory diverse? non facciamo solo quella lineare percorso più breve?
class TrajectoryPlanner:
    """
      Fase A) sequenza di waypoint connessi con tratti min-jerk di durata T_seg.
      Fase B) traiettoria continua (cerchio o lemniscata) eseguita all'infinito
              dopo l'ultimo waypoint (oppure hover se continuous=None).
    """
    def __init__(self, mode='hover', A=None, B=None, waypoints=None,
                T_seg=4.0, speed=0.3, psi_d=0.0,
                continuous_kwargs=None):
        """
        Planner di traiettoria

        mode:
        - 'hover'      : resta fermo nel punto A
        - 'line'       : vai da A a B
        - 'waypoints'  : segue una lista di waypoint con min-jerk
        - 'circle'     : traiettoria circolare
        - 'lemniscate' : traiettoria a otto

        Parametri principali:
        A        : punto iniziale o punto di hover
        B        : punto finale, usato per mode='line'
        waypoints: lista di punti, usata per mode='waypoints'
        T_seg    : durata dei segmenti min-jerk
        speed    : velocità nominale per la linea
        psi_d    : yaw desiderato
        """

        self.mode = mode
        self.T_seg = float(T_seg)
        self.speed = float(speed)
        self.psi_d = psi_d
        self.cont_kw = continuous_kwargs or {}

        if A is None:
            A = [0.0, 0.0, 1.5]

        self.A = np.array(A, dtype=float)
        self.B = None if B is None else np.array(B, dtype=float)

        if self.mode == 'hover':
            self.wps = [self.A.copy()]
            self.t_wp_end = 0.0
        elif self.mode == 'line':
            if self.B is None:
                raise ValueError("Per mode='line' devi specificare anche B.")

            self.wps = [self.A.copy(), self.B.copy()]
            self.t_wp_end = 0.0
            #------------------
        elif self.mode == 'waypoints':
            if waypoints is None or len(waypoints) < 1:
                raise ValueError("Per mode='waypoints' devi specificare almeno un waypoint.")

            self.wps = [np.array(w, dtype=float) for w in waypoints]
            self.t_wp_end = self.T_seg * max(0, len(self.wps) - 1)

        elif self.mode in ['circle', 'lemniscate']:
            self.wps = [self.A.copy()]
            self.t_wp_end = 0.0

        else:
            raise ValueError(
                f"mode='{self.mode}' non riconosciuta. "
                "Usa 'hover', 'line', 'waypoints', 'circle' o 'lemniscate'."
            )
            

    def _circle(self, tau):
        kw = self.cont_kw
        radius = kw.get('radius', 1.0)
        omega  = kw.get('omega',  0.5)
        height = kw.get('height', self.wps[-1][2])
        center = np.array(kw.get('center', [self.wps[-1][0]-radius,
                                            self.wps[-1][1],
                                            height]))
        c, s = np.cos(omega*tau), np.sin(omega*tau)
        p   = center + np.array([radius*c, radius*s, 0.0])
        pd  = np.array([-radius*omega*s,    radius*omega*c,    0.0])
        pdd = np.array([-radius*omega**2*c, -radius*omega**2*s, 0.0])
        return p, pd, pdd
    
    def _hover(self, tau):
        """
        Hovering su un punto fisso A.

        Restituisce:
        p   = posizione desiderata costante
        pd  = velocità desiderata nulla
        pdd = accelerazione desiderata nulla
        """
        p = self.A.copy()
        pd = np.zeros(3)
        pdd = np.zeros(3)

        return p, pd, pdd

    
    def _line(self, tau):
        """
        Segmento rettilineo A -> B con profilo min-jerk.
        Parte da fermo, arriva da fermo, senza salto di velocità.
        """
    def _line(self, t):
        """
        Segmento rettilineo A -> B con profilo min-jerk.
        Parte da fermo e arriva da fermo.
        """

        A = self.A
        B = self.B

        delta = B - A
        L = np.linalg.norm(delta)

        if L < 1e-9:
            return A.copy(), np.zeros(3), np.zeros(3)

        T = L / self.speed

        if t >= T:
            return B.copy(), np.zeros(3), np.zeros(3)

        return min_jerk_segment(A, B, T, t)

    def _lemniscate(self, tau):
        kw = self.cont_kw
        a      = kw.get('a', 1.0)
        omega  = kw.get('omega', 0.5)
        height = kw.get('height', self.wps[-1][2])
        center = np.array(kw.get('center', [self.wps[-1][0],
                                            self.wps[-1][1],
                                            height]))
        eps = 1e-3
        def pos(tt):
            th = omega*tt
            c, s = np.cos(th), np.sin(th)
            d = 1 + s*s
            return np.array([a*c/d, a*s*c/d, 0.0])
        p   = pos(tau)
        pd  = (pos(tau+eps) - pos(tau-eps)) / (2*eps)
        pdd = (pos(tau+eps) - 2*pos(tau) + pos(tau-eps)) / (eps*eps)
        return center + p, pd, pdd

    def __call__(self, t):

        if self.mode == 'hover':
            p, pd, pdd = self._hover(t)

        elif self.mode == 'line':
            p, pd, pdd = self._line(t)

        elif self.mode == 'waypoints':
            if t < self.t_wp_end and len(self.wps) >= 2:
                idx = int(t // self.T_seg)
                idx = min(idx, len(self.wps) - 2)

                t_local = t - idx * self.T_seg

                p, pd, pdd = min_jerk_segment(
                    self.wps[idx],
                    self.wps[idx + 1],
                    self.T_seg,
                    t_local
                )
            else:
                p = self.wps[-1].copy()
                pd = np.zeros(3)
                pdd = np.zeros(3)

        elif self.mode == 'circle':
            p, pd, pdd = self._circle(t)

        elif self.mode == 'lemniscate':
            p, pd, pdd = self._lemniscate(t)

        else:
            raise ValueError(f"mode='{self.mode}' non riconosciuta")

        return {
            'p_d': p,
            'p_d_dot': pd,
            'p_d_ddot': pdd,
            'psi_d': self.psi_d,
            'omega_d': np.zeros(3),
            'omega_d_dot': np.zeros(3),
        }

# ===========================================================================
# 9. INTERFACCIA COPPELIASIM
# ===========================================================================
class CoppeliaInterface:

    def __init__(self, quad_name='/scene 1/Quadcopter'):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.quad = self.sim.getObject(quad_name)
        self.reference_drawing = None
        self.prop_joints = [self.sim.getObject(f'/Quadcopter/propeller[{i}]/joint')
                    for i in range(4)]
        
    def get_state(self):
        p   = np.array(self.sim.getObjectPosition(self.quad, -1))
        q   = self.sim.getObjectQuaternion(self.quad, -1)
        R   = quat_to_R(q)
        v_lin, v_ang = self.sim.getObjectVelocity(self.quad)
        p_dot   = np.array(v_lin)
        omega_w = np.array(v_ang)             # world frame
        omega_R = R.T @ omega_w               # body frame
        return {'p': p, 'p_dot': p_dot, 'R': R, 'omega': omega_R}

    def apply_wrench(self, f_scalar, m_body, R_world):
        """
        f_scalar : spinta totale lungo z_B (scalare, >=0)   [N]
        m_body   : coppia espressa nel body frame, vettore   [N m]
        R_world  : R = W_R_B per portarli in world frame
        addForceAndTorque vuole forze/coppie nel WORLD frame.
        """
        F_world = R_world @ np.array([0.0, 0.0, f_scalar])
        M_world = R_world @ m_body
        self.sim.addForceAndTorque(self.quad,
                                   F_world.tolist(),
                                   M_world.tolist())
        
    def apply_propellers(self, u_lambda):
        """
        applica la spinta f_i = cf * u_lambda[i] lungo l'asse z_Pi di ogni elica.
        """
        cf = self.params.cf
        for i, prop_handle in enumerate(self.props):
            f_i = cf * max(u_lambda[i], 0.0)
            #  R_Pi (orientamento elica nel mondo)
            q = self.sim.getObjectQuaternion(prop_handle, -1)
            R_Pi = quat_to_R(q)
            # La spinta e' lungo z_Pi (terza colonna)
            F_world = R_Pi @ np.array([0.0, 0.0, f_i])
            # Coppia di reazione attorno a z_Pi
            sign = 1.0 if i % 2 == 0 else -1.0
            tau_world = R_Pi @ np.array([0.0, 0.0, sign * self.params.c_t * u_lambda[i]])
            self.sim.addForceAndTorque(prop_handle, F_world.tolist(), tau_world.tolist())

    def spin_propellers(self, u_lambda):

        for i, u_i in enumerate(u_lambda):
            omega = np.sqrt(max(u_i, 0.0))
            sign = 1.0 if i % 2 == 0 else -1.0
            self.sim.setJointTargetVelocity(self.prop_joints[i], sign * omega)

    def start(self):
        self.sim.setStepping(True)
        self.sim.setFloatParam(self.sim.floatparam_simulation_time_step, 0.005)  # 5 ms
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()

    def step(self):
        self.sim.step()

    def time(self):
        return self.sim.getSimulationTime()
    
    def clear_reference_drawing(self):
        self.sim.removeDrawingObject(self.reference_drawing)
        self.reference_drawing = None
        self.sim.step()


    def draw_reference_point(self, planner, color=[1.0, 0.0, 0.0]):
        """
        Disegna il punto di riferimento su cui il drone deve fare hovering.

        Usa planner(0) per ottenere p_d, cioè la posizione desiderata.
        """
        ref = planner(0.0)
        p = ref['p_d']

        self.reference_drawing = self.sim.addDrawingObject(
            self.sim.drawing_points,
            10,      # size
            0.0,
            -1,
            1,
            color
        )

        self.sim.addDrawingObjectItem(
            self.reference_drawing,
            p.tolist()
        )

    
    def draw_reference_trajectory(self, planner, t_total, dt, color=[0.0, 0.0, 1.0]):

        self.reference_drawing = self.sim.addDrawingObject(
            self.sim.drawing_lines,  
            2,                       
            0.0,                    
            -1,                    
            0,                       
            color                    
        )
        
        t = 0.0
        last_p = None
        while t <= t_total:
            ref = planner(t)
            p = ref['p_d']
            
            if last_p is not None:
                item = list(last_p) + list(p)
                self.sim.addDrawingObjectItem(self.reference_drawing, item)
                
            last_p = p.copy()
            t += dt

# ===========================================================================
# 10.  PLOTS
# ===========================================================================
class Logger:
    def __init__(self):
        self.t, self.p, self.p_d = [], [], []
        self.e_p, self.e_R = [], []
        self.f, self.m = [], []

    def log(self, t, state, ref, out):
        self.t.append(t)
        self.p.append(state['p'].copy())
        self.p_d.append(ref['p_d'].copy())
        self.e_p.append(out['e_p'].copy())
        self.e_R.append(out['e_R'].copy())
        self.f.append(out['f'])
        self.m.append(out['m'].copy())

    def to_arrays(self):
        return {
            't':   np.array(self.t),
            'p':   np.array(self.p),
            'p_d': np.array(self.p_d),
            'e_p': np.array(self.e_p),
            'e_R': np.array(self.e_R),
            'f':   np.array(self.f),
            'm':   np.array(self.m),
        }

    def plot(self, mode="hover"):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib non installato: salto i plot.")
            return
        d = self.to_arrays()
        labels = ['x', 'y', 'z']

        fig, ax = plt.subplots(3, 2, figsize=(12, 8))
        if mode=="hover":
            fig.suptitle("Test HOVER", fontsize=18)
        else:
            fig.suptitle("Test POINT-TO-POINT", fontsize=18)
        
        for i in range(3):
            ax[i, 0].plot(d['t'], d['p'][:, i],   label=f'p_{labels[i]}')
            ax[i, 0].plot(d['t'], d['p_d'][:, i], '--',
                          label=f'p_d_{labels[i]}')
            ax[i, 0].set_ylabel(f'pos {labels[i]} [m]')
            ax[i, 0].grid(True); ax[i, 0].legend()
            ax[i, 1].plot(d['t'], d['e_R'][:, i], label=f'e_R_{labels[i]}')
            ax[i, 1].set_ylabel('e_R')
            ax[i, 1].grid(True); ax[i, 1].legend()
        ax[-1, 0].set_xlabel('t [s]'); ax[-1, 1].set_xlabel('t [s]')
        #fig.suptitle('Tracking di posizione (sx) e errore di assetto (dx)')
        fig.tight_layout()

        fig2, ax2 = plt.subplots(2, 1, figsize=(10, 5))
        ax2[0].plot(d['t'], d['f']);  ax2[0].set_ylabel('f [N]'); ax2[0].grid(True)
        ax2[1].plot(d['t'], d['m']);  ax2[1].set_ylabel('m [Nm]')
        ax2[1].set_xlabel('t [s]');   ax2[1].grid(True)
        ax2[1].legend(['m_x', 'm_y', 'm_z'])
        fig2.suptitle('Comandi (f, m)')
        fig2.tight_layout()
        plt.show()


# ===========================================================================
# 11. MAIN
# ===========================================================================
def main():
    params = QuadrotorParams()
    ctrl   = QuadrotorController(params)
    sim    = CoppeliaInterface(quad_name='/Quadcopter')
    logger = Logger()

    # planner = TrajectoryPlanner(
    #     waypoints=[[3.0, 0.0, 0.05],
    #                [0.0, 0.0, 1.5],
    #                [1.0, 0.0, 1.5]],
    #     T_seg=4.0,
    #     continuous='circle',
    #     continuous_kwargs={'radius': 1.0, 'omega': 0.5, 'height': 1.5,
    #                        'center': [0.0, 0.0, 1.5]},
    #     psi_d=0.0
    # )

    dt = 0.005
    t_total = 10.0

    for i in range(2):
        if(i==0): #faccio hovering
            planner = TrajectoryPlanner(
                T_seg=4.0,
                mode='hover',
                A=[0.0, 0.0, 1.5],
                speed=0.3,
                psi_d=0.0
            )
            sim.start()
            
            sim.draw_reference_point(planner, color=[1.0, 0.0, 0.0])
            sim.sim.setObjectPosition(sim.quad, -1, [0.0, 0.0, 1.5])
            t = 0.0
            try:
                while t < t_total:
                    state = sim.get_state()
                    ref   = planner(t)
                    out   = ctrl.step(state, ref, dt)
                    sim.apply_wrench(out['f'], out['m'], state['R'])
                    u_lambda = ctrl.alloc.F_tilde_inv @ np.hstack(([out['f']], out['m']))
                    sim.spin_propellers(np.clip(u_lambda, 0.0, None))
                    logger.log(t, state, ref, out)
                    sim.step()
                    t += dt
            finally:
                sim.clear_reference_drawing()
                sim.stop()
                logger.plot(mode="hover")

        else: #faccio A->B
            planner = TrajectoryPlanner(
                # Per 'line' non servono waypoint: A e B sono gia' la traiettoria.
                T_seg=4.0,
                mode='line',
                A=[0.0, 0.0, 1.5],
                B=[1.0, 0.0, 1.5],
                speed=0.3,
                psi_d=0.0
            )
            sim.start()
            sim.sim.setObjectPosition(sim.quad, -1, [0.0, 0.0, 1.5])
            sim.draw_reference_trajectory(planner, t_total, dt, color=[1.0, 0.0, 0.0])
            t = 0.0
            try:
                while t < t_total:
                    state = sim.get_state()
                    ref   = planner(t)
                    out   = ctrl.step(state, ref, dt)
                    sim.apply_wrench(out['f'], out['m'], state['R'])
                    u_lambda = ctrl.alloc.F_tilde_inv @ np.hstack(([out['f']], out['m']))
                    sim.spin_propellers(np.clip(u_lambda, 0.0, None))
                    logger.log(t, state, ref, out)
                    sim.step()
                    t += dt
            finally:
                sim.clear_reference_drawing()
                sim.stop()
                logger.plot(mode="line")


if __name__ == '__main__':
    main()