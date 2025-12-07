import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
import os
import pygame


# Set the SDL_VIDEODRIVER to dummy to run Pygame headless
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

class Domino:
    """Represents a single domino with physics and state."""
    def __init__(self, grid_pos, dimensions):
        self.grid_pos = grid_pos
        self.w, self.d, self.h = dimensions
        self.state = 'standing'  # 'standing', 'falling', 'fallen'
        self.angle_rad = 0.0
        self.angular_velocity = 0.0
        self.hit_by = None  # To prevent multiple triggers from one source

    def update(self, dt, gravity):
        """Update domino physics if it's falling."""
        if self.state != 'falling':
            return False

        # Apply a simple gravity model to accelerate the fall
        self.angular_velocity += math.sin(self.angle_rad + 0.1) * gravity * dt
        self.angle_rad += self.angular_velocity * dt

        if self.angle_rad >= math.pi / 2:
            self.angle_rad = math.pi / 2
            self.angular_velocity = 0
            self.state = 'fallen'
            return True  # Just fell
        return False

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.pos = [x, y]
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = random.randint(15, 30)
        self.color = color
        self.radius = random.uniform(2, 4)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            color = self.color + (alpha,)
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Place dominoes on a grid and start a chain reaction to topple them all."
    user_guide = "Use arrow keys to move and space to place dominoes. After placing all, use arrows to select one and press shift to push it."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 12, 8
    MAX_DOMINOES = 15
    SIMULATION_TIME_SECONDS = 15
    FPS = 30

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_DOMINO_STAND = (60, 220, 120)
    COLOR_DOMINO_FALL = (255, 80, 80)
    COLOR_DOMINO_SELECT = (100, 150, 255)
    COLOR_PARTICLE_HIT = (255, 200, 150)

    # Physics
    GRAVITY = 0.8
    MOMENTUM_TRANSFER = 0.7
    INITIAL_PUSH_VELOCITY = 0.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 18)

        # Iso projection constants
        self.tile_w_half = 24
        self.tile_h_half = 12
        self.origin_x = self.WIDTH // 2
        self.origin_y = 120
        self.domino_dims = (self.tile_w_half * 0.3, self.tile_w_half * 0.8, self.tile_h_half * 5)

        self.dominoes = []
        self.particles = []
        self.placed_domino_coords = set()
        self.cursor_pos = [0, 0]
        self.selection_idx = 0
        self.phase = 'placement'
        self.timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = 0
        self.prev_shift_held = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed python's random for particle generation
            random.seed(seed)

        self.phase = 'placement'
        self.dominoes = []
        self.particles = []
        self.placed_domino_coords = set()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selection_idx = 0
        self.timer = self.SIMULATION_TIME_SECONDS * self.FPS
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = 0
        self.prev_shift_held = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.phase == 'placement':
            self._handle_placement_input(movement, space_press)
            if len(self.dominoes) == self.MAX_DOMINOES:
                self.phase = 'selection'
                if self.dominoes:
                    self.selection_idx = 0
        
        elif self.phase == 'selection':
            self._handle_selection_input(movement, shift_press)

        elif self.phase == 'reaction':
            newly_fallen, new_hits = self._update_physics()
            reward += newly_fallen * 0.1
            self.score += newly_fallen
            for hit_pos in new_hits:
                self._spawn_particles(hit_pos, 10, self.COLOR_PARTICLE_HIT)
            
            self.timer -= 1
        
        self._update_particles()
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        
        # truncated is always False as per requirements
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_placement_input(self, movement, space_press):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if space_press and len(self.dominoes) < self.MAX_DOMINOES:
            pos_tuple = tuple(self.cursor_pos)
            if pos_tuple not in self.placed_domino_coords:
                self.dominoes.append(Domino(pos_tuple, self.domino_dims))
                self.placed_domino_coords.add(pos_tuple)

    def _handle_selection_input(self, movement, shift_press):
        if not self.dominoes: return
        
        if movement == 1 or movement == 3: self.selection_idx = (self.selection_idx - 1 + len(self.dominoes)) % len(self.dominoes)
        elif movement == 2 or movement == 4: self.selection_idx = (self.selection_idx + 1) % len(self.dominoes)
        
        if shift_press:
            selected_domino = self.dominoes[self.selection_idx]
            selected_domino.state = 'falling'
            selected_domino.angular_velocity = self.INITIAL_PUSH_VELOCITY
            self.phase = 'reaction'

    def _update_physics(self):
        newly_fallen = 0
        new_hits = []
        dt = 1.0 / self.FPS
        
        falling_dominoes = [d for d in self.dominoes if d.state == 'falling']
        standing_dominoes = [d for d in self.dominoes if d.state == 'standing']

        for domino in self.dominoes:
            if domino.update(dt, self.GRAVITY):
                newly_fallen += 1
        
        for faller in falling_dominoes:
            if faller.angle_rad < 0.1: continue # Not falling enough to hit
            
            faller_tip_x = faller.grid_pos[0]
            faller_tip_y = faller.grid_pos[1] + math.sin(faller.angle_rad) * (faller.h / self.tile_h_half / 2)

            for stander in standing_dominoes:
                if stander.hit_by is not None: continue

                dist_sq = (stander.grid_pos[0] - faller_tip_x)**2 + (stander.grid_pos[1] - faller_tip_y)**2
                if dist_sq < 0.5**2: # Collision radius
                    stander.state = 'falling'
                    stander.angular_velocity = faller.angular_velocity * self.MOMENTUM_TRANSFER
                    stander.hit_by = faller
                    
                    hit_pos_screen = self._project_iso(stander.grid_pos[0], stander.grid_pos[1])
                    new_hits.append(hit_pos_screen)

        return newly_fallen, new_hits

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        num_fallen = sum(1 for d in self.dominoes if d.state == 'fallen')
        
        if self.phase == 'reaction' and self.timer <= 0:
            reward = -10 if num_fallen < self.MAX_DOMINOES else 0
            return True, reward
        
        if num_fallen == self.MAX_DOMINOES and len(self.dominoes) == self.MAX_DOMINOES:
            # +5 for toppling all, +50 for doing it in time
            return True, 55

        if self.steps >= 2000: # Max episode length
            return True, -20

        return False, 0
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "phase": self.phase}

    def _project_iso(self, x, y, z=0):
        screen_x = self.origin_x + (x - y) * self.tile_w_half
        screen_y = self.origin_y + (x + y) * self.tile_h_half - z
        return int(screen_x), int(screen_y)

    def _render_game(self):
        self._draw_grid()
        
        render_queue = []
        if self.phase == 'placement' or self.phase == 'selection':
            self._draw_cursor(render_queue)

        for i, domino in enumerate(self.dominoes):
            color = self.COLOR_DOMINO_STAND
            if domino.state != 'standing':
                color = self.COLOR_DOMINO_FALL
            if self.phase == 'selection' and i == self.selection_idx:
                color = self.COLOR_DOMINO_SELECT
            
            _, sort_key = self._project_iso(domino.grid_pos[0], domino.grid_pos[1])
            
            # FIX: Pass all required arguments for _draw_iso_cuboid to the render queue.
            render_queue.append((
                sort_key,
                'domino',
                self.screen,
                domino.grid_pos[0],
                domino.grid_pos[1],
                0,  # gz
                domino.w,
                domino.d,
                domino.h,
                domino.angle_rad,
                color,
                False  # is_wireframe
            ))

        render_queue.sort(key=lambda item: item[0])

        for _, item_type, *args in render_queue:
            if item_type in ('domino', 'cursor'):
                self._draw_iso_cuboid(*args)
        
        for p in self.particles:
            p.draw(self.screen)

    def _draw_grid(self):
        for i in range(self.GRID_WIDTH + 1):
            p1 = self._project_iso(i, 0)
            p2 = self._project_iso(i, self.GRID_HEIGHT)
            pygame.gfxdraw.line(self.screen, p1[0], p1[1], p2[0], p2[1], self.COLOR_GRID)
        for i in range(self.GRID_HEIGHT + 1):
            p1 = self._project_iso(0, i)
            p2 = self._project_iso(self.GRID_WIDTH, i)
            pygame.gfxdraw.line(self.screen, p1[0], p1[1], p2[0], p2[1], self.COLOR_GRID)

    def _draw_cursor(self, render_queue):
        if self.phase == 'placement':
            x, y = self.cursor_pos
            _, sort_key = self._project_iso(x, y)
            is_occupied = tuple(self.cursor_pos) in self.placed_domino_coords
            color = (255, 0, 0, 100) if is_occupied else self.COLOR_CURSOR
            render_queue.append((sort_key, 'cursor', self.screen, x, y, 0, *self.domino_dims, 0, color, True))

    def _draw_iso_cuboid(self, surface, gx, gy, gz, w, d, h, angle_rad, color, is_wireframe=False):
        # 8 vertices of the cuboid in local space, centered on its base
        local_verts = [
            (-w/2, -d/2, 0), (w/2, -d/2, 0), (w/2, d/2, 0), (-w/2, d/2, 0),
            (-w/2, -d/2, h), (w/2, -d/2, h), (w/2, d/2, h), (-w/2, d/2, h)
        ]

        # Apply rotation around the front-bottom edge (axis along X)
        rotated_verts = []
        for lv in local_verts:
            x, y, z = lv
            # The pivot is at y = d/2
            y_piv, z_piv = y - d/2, z
            new_y = y_piv * math.cos(angle_rad) - z_piv * math.sin(angle_rad)
            new_z = y_piv * math.sin(angle_rad) + z_piv * math.cos(angle_rad)
            rotated_verts.append((x, new_y + d/2, new_z))

        # Project to screen space
        screen_verts = [self._project_iso(gx + rv[0]/self.tile_w_half, gy + rv[1]/self.tile_w_half, gz + rv[2]) for rv in rotated_verts]

        # Define faces by vertex indices
        faces = [
            (0, 1, 2, 3),  # Bottom
            (4, 5, 6, 7),  # Top
            (0, 1, 5, 4),  # Front
            (2, 3, 7, 6),  # Back
            (1, 2, 6, 5),  # Right
            (0, 3, 7, 4)   # Left
        ]
        
        face_draw_order = [0, 3, 5, 2, 4] 
        
        for i in face_draw_order:
            face_verts_indices = faces[i]
            points = [screen_verts[j] for j in face_verts_indices]
            
            if is_wireframe:
                pygame.gfxdraw.aapolygon(surface, points, color)
            else:
                shade_factor = 1.0
                if i in [0]: shade_factor = 0.6 # Bottom
                if i in [5]: shade_factor = 0.7 # Left
                if i in [2]: shade_factor = 0.9 # Front
                if i in [4]: shade_factor = 1.0 # Top

                face_color = (
                    int(color[0] * shade_factor),
                    int(color[1] * shade_factor),
                    int(color[2] * shade_factor)
                )
                pygame.gfxdraw.filled_polygon(surface, points, face_color)
                pygame.gfxdraw.aapolygon(surface, points, (0,0,0,50)) # Outline

    def _render_ui(self):
        phase_text_str = f"PHASE: {self.phase.upper()}"
        phase_surf = self.font_main.render(phase_text_str, True, self.COLOR_TEXT)
        self.screen.blit(phase_surf, (self.WIDTH // 2 - phase_surf.get_width() // 2, 10))

        if self.phase == 'reaction':
            timer_str = f"TIME: {self.timer / self.FPS:.1f}"
            timer_surf = self.font_info.render(timer_str, True, self.COLOR_TEXT)
            self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 10, 10))

        if self.phase == 'placement':
            count_str = f"DOMINOES TO PLACE: {self.MAX_DOMINOES - len(self.dominoes)}"
        else:
            count_str = f"TOPPLED: {self.score} / {self.MAX_DOMINOES}"
        
        count_surf = self.font_info.render(count_str, True, self.COLOR_TEXT)
        self.screen.blit(count_surf, (10, self.HEIGHT - count_surf.get_height() - 10))
        
        if self.phase == 'placement':
            hint_str = "[ARROWS] Move Cursor  [SPACE] Place Domino"
        elif self.phase == 'selection':
            hint_str = "[ARROWS] Select Domino  [SHIFT] Start Chain Reaction"
        else:
            hint_str = ""
        
        if hint_str:
            hint_surf = self.font_info.render(hint_str, True, self.COLOR_TEXT)
            self.screen.blit(hint_surf, (self.WIDTH // 2 - hint_surf.get_width() // 2, self.HEIGHT - hint_surf.get_height() - 10))

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color))

if __name__ == '__main__':
    # Un-comment the line below to run with a visible display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Check if we are running in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. No display will be shown.")
        # Run a short episode automatically in headless mode
        total_reward = 0
        terminated = False
        for _ in range(200): # Place some dominoes
             obs, reward, terminated, truncated, info = env.step([random.randint(0,4), 1, 0])
             if len(env.dominoes) >= env.MAX_DOMINOES:
                 break
        
        obs, reward, terminated, truncated, info = env.step([0, 0, 1]) # Push one
        
        while not terminated:
            obs, reward, terminated, truncated, info = env.step([0,0,0])
            total_reward += reward
        print(f"Headless episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
    else:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Domino Topple Environment")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset(seed=42)
                    total_reward = 0
                    print("--- ENV RESET ---")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
                pygame.time.wait(2000)
                obs, info = env.reset(seed=42)
                total_reward = 0
                print("--- ENV RESET ---")
                
            clock.tick(GameEnv.FPS)
            
        pygame.quit()