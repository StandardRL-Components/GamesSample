import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:50:24.985129
# Source Brief: brief_00666.md
# Brief Index: 666
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a strategic game.
    The player deploys quantum structures and terraforms the landscape to
    defend against increasingly intense quantum storms.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Defend your core by building quantum structures and terraforming the landscape to neutralize incoming energy storms."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to build or terraform, and use shift to cycle between modes."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 32
    GRID_ROWS = 20
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_GRID = (30, 30, 50)
    COLOR_STORM = (255, 50, 50)
    COLOR_STRUCTURE_1 = (50, 150, 255)
    COLOR_STRUCTURE_2 = (40, 200, 255)
    COLOR_STRUCTURE_3 = (60, 100, 255)
    COLOR_STRUCTURE_4 = (100, 180, 255)
    COLOR_TERRAFORM = (50, 255, 100)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_ENERGY_HIGH = (0, 255, 128)
    COLOR_ENERGY_LOW = (255, 0, 64)

    # Game parameters
    MAX_STEPS = 1000
    INITIAL_ENERGY = 1000
    STRUCTURE_COST = [0, 100, 150, 200, 250] # Cost for type 1, 2, 3, 4
    TERRAFORM_COST = 25
    PARTICLE_DAMAGE = 20
    PARTICLE_REGEN = 1.0 # Energy gained per absorbed particle

    # Modes
    MODE_TERRAFORM = 0
    MODE_BUILD_1 = 1
    MODE_BUILD_2 = 2
    MODE_BUILD_3 = 3
    MODE_BUILD_4 = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont('Consolas', 16, bold=True)
        self.font_large = pygame.font.SysFont('Consolas', 24, bold=True)

        self.render_mode = render_mode
        self.structures = []
        self.particles = []
        self.effects = []
        self.terraform_grid = np.zeros((self.GRID_COLS, self.GRID_ROWS, 2), dtype=np.float32)

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy = self.INITIAL_ENERGY
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.structures.clear()
        self.particles.clear()
        self.effects.clear()
        self.terraform_grid.fill(0)

        self.storm_speed = 0.5
        self.storm_spawn_timer = 0
        self.storm_spawn_interval = 20
        self.storm_pattern = 0
        
        self.unlocked_modes = [self.MODE_TERRAFORM, self.MODE_BUILD_1]
        self.current_mode_idx = 1 # Start with build mode
        
        self.space_was_held = False
        self.shift_was_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0.1 # Survival reward

        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        space_press = space_held and not self.space_was_held
        shift_press = shift_held and not self.shift_was_held
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        self._handle_input(movement, space_press, shift_press)
        
        energy_lost, particles_absorbed = self._update_game_state()

        # Calculate rewards
        reward -= energy_lost * 0.5
        if particles_absorbed > 0:
            reward += particles_absorbed * 0.05 # Small reward for absorption
        
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.steps >= self.MAX_STEPS:
                self.score += 100 # Win bonus
            else:
                self.score -= 100 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_press, shift_press):
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # --- Mode Cycling (Shift) ---
        if shift_press:
            # Sfx: UI_Mode_Switch.wav
            self.current_mode_idx = (self.current_mode_idx + 1) % len(self.unlocked_modes)

        # --- Action (Space) ---
        if space_press:
            mode = self.unlocked_modes[self.current_mode_idx]
            cx, cy = self.cursor_pos
            
            if mode == self.MODE_TERRAFORM:
                if self.energy >= self.TERRAFORM_COST:
                    self.energy -= self.TERRAFORM_COST
                    self._apply_terraform(cx, cy)
                    # Sfx: Terraform_Activate.wav
            
            elif mode >= self.MODE_BUILD_1:
                cost = self.STRUCTURE_COST[mode]
                if self.energy >= cost and not self._is_structure_at(cx, cy):
                    self.energy -= cost
                    self.structures.append(self._create_structure(cx, cy, mode))
                    # Sfx: Structure_Place.wav

    def _update_game_state(self):
        self._update_difficulty()
        self._spawn_storms()
        energy_lost, particles_absorbed = self._update_and_collide_particles()
        self._update_effects()
        return energy_lost, particles_absorbed

    def _update_difficulty(self):
        # Increase storm speed
        if self.steps > 0 and self.steps % 200 == 0:
            self.storm_speed = min(2.0, self.storm_speed + 0.05)
        
        # Introduce new patterns
        if self.steps == 400: self.storm_pattern = 1
        if self.steps == 600: self.storm_pattern = 2
        if self.steps == 800: self.storm_pattern = 3

        # Unlock new structures
        if self.steps == 250 and self.MODE_BUILD_2 not in self.unlocked_modes:
            self.unlocked_modes.append(self.MODE_BUILD_2)
        if self.steps == 500 and self.MODE_BUILD_3 not in self.unlocked_modes:
            self.unlocked_modes.append(self.MODE_BUILD_3)
        if self.steps == 750 and self.MODE_BUILD_4 not in self.unlocked_modes:
            self.unlocked_modes.append(self.MODE_BUILD_4)

    def _spawn_storms(self):
        self.storm_spawn_timer -= 1
        if self.storm_spawn_timer <= 0:
            self.storm_spawn_timer = self.storm_spawn_interval
            num_particles = self.np_random.integers(2, 5)

            for _ in range(num_particles):
                # Pattern 0: Top to bottom
                if self.storm_pattern == 0:
                    x = self.np_random.uniform(0, self.SCREEN_WIDTH)
                    y = -10
                    vx = self.np_random.uniform(-0.1, 0.1)
                    vy = self.storm_speed
                # Pattern 1: Left to right
                elif self.storm_pattern == 1:
                    x = -10
                    y = self.np_random.uniform(0, self.SCREEN_HEIGHT)
                    vx = self.storm_speed
                    vy = self.np_random.uniform(-0.1, 0.1)
                # Pattern 2: Corner bursts
                elif self.storm_pattern == 2:
                    corner = self.np_random.choice(4)
                    if corner == 0: x, y = -10, -10
                    elif corner == 1: x, y = self.SCREEN_WIDTH + 10, -10
                    elif corner == 2: x, y = -10, self.SCREEN_HEIGHT + 10
                    else: x, y = self.SCREEN_WIDTH + 10, self.SCREEN_HEIGHT + 10
                    angle = math.atan2((self.SCREEN_HEIGHT/2) - y, (self.SCREEN_WIDTH/2) - x)
                    vx = math.cos(angle) * self.storm_speed
                    vy = math.sin(angle) * self.storm_speed
                # Pattern 3: Spiral inwards
                else:
                    side = self.np_random.choice(4)
                    if side == 0: x, y, vx, vy = self.np_random.uniform(0, self.SCREEN_WIDTH), -10, 0, self.storm_speed
                    elif side == 1: x, y, vx, vy = self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT), -self.storm_speed, 0
                    elif side == 2: x, y, vx, vy = self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10, 0, -self.storm_speed
                    else: x, y, vx, vy = -10, self.np_random.uniform(0, self.SCREEN_HEIGHT), self.storm_speed, 0
                
                self.particles.append({'pos': np.array([x, y], dtype=np.float32), 
                                       'vel': np.array([vx, vy], dtype=np.float32),
                                       'trail': []})

    def _update_and_collide_particles(self):
        energy_lost = 0
        particles_absorbed = 0
        
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            # Apply terraforming force
            grid_x = int(p['pos'][0] / self.CELL_WIDTH)
            grid_y = int(p['pos'][1] / self.CELL_HEIGHT)
            if 0 <= grid_x < self.GRID_COLS and 0 <= grid_y < self.GRID_ROWS:
                force = self.terraform_grid[grid_x, grid_y]
                p['vel'] += force
                # Dampen velocity to prevent instability
                speed = np.linalg.norm(p['vel'])
                if speed > self.storm_speed * 2.0:
                    p['vel'] = (p['vel'] / speed) * self.storm_speed * 2.0

            # Apply structure forces
            for s in self.structures:
                s_pos = np.array([s['pos'][0] * self.CELL_WIDTH + self.CELL_WIDTH/2, 
                                  s['pos'][1] * self.CELL_HEIGHT + self.CELL_HEIGHT/2])
                dist_vec = s_pos - p['pos']
                dist = np.linalg.norm(dist_vec)

                if s['type'] == self.MODE_BUILD_4 and dist < s['radius'] * 2: # Repulsor
                    # Sfx: Repulsor_Hum.wav (loop)
                    repel_force = -dist_vec / max(1, dist**2) * s['power']
                    p['vel'] += repel_force
                
            p['trail'].append(p['pos'].copy())
            if len(p['trail']) > 5: p['trail'].pop(0)
            p['pos'] += p['vel']

            # Collision with structures
            collided = False
            for s in self.structures:
                s_pos_px = np.array([s['pos'][0] * self.CELL_WIDTH + self.CELL_WIDTH / 2,
                                     s['pos'][1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2])
                if np.linalg.norm(p['pos'] - s_pos_px) < s['radius']:
                    # Sfx: Particle_Absorb.wav
                    self.energy = min(self.INITIAL_ENERGY, self.energy + self.PARTICLE_REGEN)
                    s['energy'] = min(s['capacity'], s['energy'] + 1)
                    particles_to_remove.append(i)
                    self.effects.append({'type': 'absorb', 'pos': p['pos'], 'life': 10, 'radius': s['radius']})
                    particles_absorbed += 1
                    collided = True
                    break
            if collided: continue
            
            # Boundary and ground collision checks
            if not (0 <= p['pos'][0] < self.SCREEN_WIDTH and 0 <= p['pos'][1] < self.SCREEN_HEIGHT):
                if p['pos'][0] < -50 or p['pos'][0] > self.SCREEN_WIDTH + 50 or \
                   p['pos'][1] < -50 or p['pos'][1] > self.SCREEN_HEIGHT + 50:
                    particles_to_remove.append(i) # Remove if far off-screen
            elif p['pos'][1] >= self.SCREEN_HEIGHT - 1: # Hit bottom "ground"
                # Sfx: Ground_Impact.wav
                self.energy -= self.PARTICLE_DAMAGE
                energy_lost += self.PARTICLE_DAMAGE
                particles_to_remove.append(i)
                self.effects.append({'type': 'impact', 'pos': p['pos'], 'life': 15, 'radius': 10})

        # Remove particles in reverse to avoid index errors
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]
            
        return energy_lost, particles_absorbed

    def _update_effects(self):
        self.effects = [e for e in self.effects if e['life'] > 0]
        for e in self.effects:
            e['life'] -= 1

    def _check_termination(self):
        if self.energy <= 0:
            self.game_over = True
            self.energy = 0
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy,
            "mode": self.unlocked_modes[self.current_mode_idx],
            "unlocked_modes": len(self.unlocked_modes)
        }

    def _render_game(self):
        self._render_background_grid()
        self._render_terraform_effects()
        self._render_structures()
        self._render_particles()
        self._render_effects()
        self._render_cursor()

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_terraform_effects(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                force_vec = self.terraform_grid[x, y]
                magnitude = np.linalg.norm(force_vec)
                if magnitude > 0:
                    px = int(x * self.CELL_WIDTH + self.CELL_WIDTH / 2)
                    py = int(y * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
                    alpha = min(255, int(magnitude * 5000))
                    radius = int(self.CELL_WIDTH * 0.4)
                    
                    # Draw glow effect
                    for i in range(radius, 0, -2):
                        glow_alpha = int(alpha * (1 - i / radius)**2)
                        if glow_alpha > 0:
                            pygame.gfxdraw.filled_circle(self.screen, px, py, i, (*self.COLOR_TERRAFORM, glow_alpha))
    
    def _render_structures(self):
        for s in self.structures:
            px = int(s['pos'][0] * self.CELL_WIDTH + self.CELL_WIDTH / 2)
            py = int(s['pos'][1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2)
            color = s['color']
            
            # Draw glow
            for i in range(int(s['radius'] * 1.5), s['radius'], -2):
                alpha = 60 * (1 - (i - s['radius']) / (s['radius'] * 0.5))
                pygame.gfxdraw.aacircle(self.screen, px, py, i, (*color, int(alpha)))
            
            # Draw main shape
            if s['type'] == self.MODE_BUILD_1: # Triangle
                points = [(px, py - s['radius']), (px - s['radius'], py + s['radius']), (px + s['radius'], py + s['radius'])]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif s['type'] == self.MODE_BUILD_2: # Square
                rect = (px - s['radius'], py - s['radius'], s['radius']*2, s['radius']*2)
                pygame.draw.rect(self.screen, color, rect)
            elif s['type'] == self.MODE_BUILD_3: # Circle (AoE)
                pygame.gfxdraw.aacircle(self.screen, px, py, s['radius'], color)
                pygame.gfxdraw.filled_circle(self.screen, px, py, s['radius'], color)
            elif s['type'] == self.MODE_BUILD_4: # Diamond (Repulsor)
                points = [(px, py - s['radius']), (px + s['radius'], py), (px, py + s['radius']), (px - s['radius'], py)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_particles(self):
        for p in self.particles:
            px, py = int(p['pos'][0]), int(p['pos'][1])
            
            # Draw trail
            if len(p['trail']) > 1:
                for i in range(len(p['trail']) - 1):
                    alpha = int(150 * (i / len(p['trail'])))
                    start_pos = (int(p['trail'][i][0]), int(p['trail'][i][1]))
                    end_pos = (int(p['trail'][i+1][0]), int(p['trail'][i+1][1]))
                    pygame.draw.line(self.screen, (*self.COLOR_STORM, alpha), start_pos, end_pos, 2)

            # Draw particle glow
            pygame.gfxdraw.filled_circle(self.screen, px, py, 6, (*self.COLOR_STORM, 60))
            pygame.gfxdraw.filled_circle(self.screen, px, py, 4, (*self.COLOR_STORM, 120))
            # Draw particle core
            pygame.gfxdraw.filled_circle(self.screen, px, py, 2, self.COLOR_STORM)

    def _render_effects(self):
        for e in self.effects:
            alpha = int(255 * (e['life'] / (15 if e['type'] == 'impact' else 10)))
            if alpha <= 0: continue
            pos = (int(e['pos'][0]), int(e['pos'][1]))
            
            if e['type'] == 'impact':
                color = (*self.COLOR_STORM, alpha)
                radius = int(e['radius'] * (1.0 - e['life']/15.0))
                pygame.gfxdraw.aacircle(self.screen, *pos, radius, color)
                pygame.gfxdraw.aacircle(self.screen, *pos, radius+1, color)
            elif e['type'] == 'absorb':
                color = (*self.COLOR_TEXT, alpha)
                radius = int(e['radius'] * (e['life']/10.0))
                pygame.gfxdraw.aacircle(self.screen, *pos, radius, color)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px = cx * self.CELL_WIDTH
        py = cy * self.CELL_HEIGHT
        
        # Flashing alpha for cursor
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        color = (*self.COLOR_CURSOR, int(alpha))
        
        rect = (px, py, self.CELL_WIDTH, self.CELL_HEIGHT)
        pygame.draw.rect(self.screen, color, rect, 2)

    def _render_ui(self):
        # --- Energy Bar ---
        energy_ratio = self.energy / self.INITIAL_ENERGY
        bar_width = int((self.SCREEN_WIDTH - 20) * energy_ratio)
        bar_color = [int(c1 * energy_ratio + c2 * (1 - energy_ratio)) for c1, c2 in zip(self.COLOR_ENERGY_HIGH, self.COLOR_ENERGY_LOW)]
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, 10, self.SCREEN_WIDTH - 20, 20))
        if bar_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, 10, bar_width, 20))
        
        # --- Text Rendering Helper ---
        def draw_text(text, font, pos, color=self.COLOR_TEXT, shadow_color=self.COLOR_TEXT_SHADOW):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
            self.screen.blit(text_surf, pos)

        # --- Info Text ---
        draw_text(f"ENERGY: {int(self.energy)}", self.font_small, (15, 12))
        
        steps_text = f"STEP: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_small.render(steps_text, True, self.COLOR_TEXT)
        draw_text(steps_text, self.font_small, (self.SCREEN_WIDTH - steps_surf.get_width() - 15, 12))

        # --- Mode Text ---
        mode = self.unlocked_modes[self.current_mode_idx]
        if mode == self.MODE_TERRAFORM: mode_str = "MODE: TERRAFORM"
        else: mode_str = f"MODE: BUILD T{mode}"
        draw_text(mode_str, self.font_large, (10, self.SCREEN_HEIGHT - 35))

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "SURVIVED" if self.steps >= self.MAX_STEPS else "ENERGY DEPLETED"
            draw_text(end_text, self.font_large, (self.SCREEN_WIDTH/2 - self.font_large.size(end_text)[0]/2, self.SCREEN_HEIGHT/2 - 20))

    def close(self):
        pygame.quit()
        
    # --- Internal Helper Functions ---
    def _is_structure_at(self, x, y):
        return any(s['pos'] == [x, y] for s in self.structures)

    def _create_structure(self, x, y, type):
        s = {'pos': [x, y], 'type': type, 'energy': 0}
        if type == self.MODE_BUILD_1:
            s.update({'color': self.COLOR_STRUCTURE_1, 'radius': 8, 'capacity': 50})
        elif type == self.MODE_BUILD_2:
            s.update({'color': self.COLOR_STRUCTURE_2, 'radius': 10, 'capacity': 100})
        elif type == self.MODE_BUILD_3:
            s.update({'color': self.COLOR_STRUCTURE_3, 'radius': 12, 'capacity': 75})
        elif type == self.MODE_BUILD_4:
            s.update({'color': self.COLOR_STRUCTURE_4, 'radius': 10, 'capacity': 0, 'power': 0.05})
        return s

    def _apply_terraform(self, x, y):
        self.effects.append({'type': 'impact', 'pos': (x*self.CELL_WIDTH+10, y*self.CELL_HEIGHT+10), 'life': 15, 'radius': 15})
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    dist_sq = dx*dx + dy*dy
                    if dist_sq == 0: continue
                    
                    force_vec = -np.array([dx, dy]) / dist_sq
                    self.terraform_grid[nx, ny] += force_vec * 0.02
                    # Clamp force to avoid instability
                    mag = np.linalg.norm(self.terraform_grid[nx, ny])
                    if mag > 0.1:
                        self.terraform_grid[nx, ny] = (self.terraform_grid[nx, ny] / mag) * 0.1

if __name__ == '__main__':
    # --- Interactive Human Play ---
    # This block will not run in the headless environment but is useful for local testing.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Quantum Storm")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated and not truncated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])

    print(f"Game Over. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
    env.close()