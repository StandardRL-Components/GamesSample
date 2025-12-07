import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a tower defense game.
    The player must defend a central "Octave Core" from incoming "Soundwaves"
    by placing and upgrading energy-absorbing towers.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Defend the central 'Octave Core' from incoming 'Soundwaves' by placing and "
        "upgrading energy-absorbing towers."
    )
    user_guide = (
        "Use arrow keys to move the cursor. Press space to build or upgrade a tower. "
        "Press shift to pause the game."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_CELL_SIZE = 40
    GRID_W = SCREEN_WIDTH // GRID_CELL_SIZE
    GRID_H = SCREEN_HEIGHT // GRID_CELL_SIZE

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (30, 20, 60)
    COLOR_CORE = (255, 255, 255)
    COLOR_CORE_GLOW = (200, 200, 255)
    COLOR_TOWER_BASE = np.array([0, 150, 255])
    COLOR_TOWER_UPGRADED = np.array([200, 100, 255])
    COLOR_WAVE_SPECTRUM = [(255, 50, 50), (255, 255, 50), (50, 255, 50)]
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR_VALID = (0, 255, 0, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)

    # Game Parameters
    CORE_MAX_HEALTH = 100
    INITIAL_RESOURCES = 100
    MAX_WAVES = 20
    MAX_STEPS = 15000  # Extended for longer games
    WAVE_COOLDOWN_FRAMES = 150 # Time between waves

    TOWER_COST = 50
    TOWER_UPGRADE_COST_BASE = 75
    TOWER_MAX_LEVEL = 5
    TOWER_BASE_RADIUS = 20
    TOWER_RESOURCE_GAIN = 5

    WAVE_BASE_SPEED = 1.0
    WAVE_SPEED_INCREMENT = 0.1
    WAVE_BASE_COUNT = 3
    WAVE_DAMAGE = 10

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 72)
        self.font_subtitle = pygame.font.Font(None, 40)
        
        self.render_mode = render_mode

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.is_paused = False
        self.core_health = 0
        self.resources = 0
        self.current_wave_number = 0
        self.towers = []
        self.waves = []
        self.particles = []
        self.starfield = []
        self.cursor_grid_pos = [0, 0]
        self.time_since_last_wave = 0
        self.wave_in_progress = False
        self.last_space_press = False
        self.last_shift_press = False

        self._generate_starfield()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_condition_met = False
        self.is_paused = False
        
        self.core_health = self.CORE_MAX_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.current_wave_number = 0
        
        self.towers = []
        self.waves = []
        self.particles = []
        
        self.cursor_grid_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        self.time_since_last_wave = 0
        self.wave_in_progress = False
        
        self.last_space_press = False
        self.last_shift_press = False
        
        # Re-generate starfield with the new seed for determinism
        self._generate_starfield()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # Handle pause toggle (edge-triggered)
        if shift_held and not self.last_shift_press:
            self.is_paused = not self.is_paused
        self.last_shift_press = shift_held
        
        # Player can always move cursor and try to build, even when paused
        self._handle_cursor_movement(movement)
        action_reward = self._handle_tower_actions(space_held)
        reward += action_reward

        if not self.is_paused and not self.game_over:
            self.steps += 1
            reward += 0.01  # Small survival reward per frame

            # Update game state
            deflection_reward = self._update_waves()
            reward += deflection_reward
            self._update_particles()
            self._manage_waves()
        
        self.last_space_press = space_held
        
        # Check termination conditions
        terminated = self._check_termination()
        if terminated and not self.win_condition_met:
             reward -= 100.0 # Penalty for losing
        elif terminated and self.win_condition_met:
             reward += 100.0 # Bonus for winning

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.game_over:
            return True
        if self.core_health <= 0:
            self.game_over = True
            return True
        if self.current_wave_number > self.MAX_WAVES and not self.wave_in_progress:
            self.game_over = True
            self.win_condition_met = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _handle_cursor_movement(self, movement):
        if movement == 1: self.cursor_grid_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_grid_pos[1] += 1  # Down
        elif movement == 3: self.cursor_grid_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_grid_pos[0] += 1  # Right
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, self.GRID_W - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, self.GRID_H - 1)

    def _handle_tower_actions(self, space_held):
        # Edge-triggered action
        if not space_held or self.last_space_press:
            return 0.0

        cursor_coords = (
            self.cursor_grid_pos[0] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2,
            self.cursor_grid_pos[1] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE // 2
        )

        # Check if a tower exists at the cursor
        existing_tower = None
        for tower in self.towers:
            if tower['grid_pos'] == self.cursor_grid_pos:
                existing_tower = tower
                break
        
        if existing_tower:
            # Upgrade tower
            if existing_tower['level'] < self.TOWER_MAX_LEVEL:
                cost = self.TOWER_UPGRADE_COST_BASE * existing_tower['level']
                if self.resources >= cost:
                    self.resources -= cost
                    existing_tower['level'] += 1
                    return 5.0 # Upgrade reward
        else:
            # Place new tower
            if self.resources >= self.TOWER_COST:
                self.resources -= self.TOWER_COST
                self.towers.append({
                    'pos': np.array(cursor_coords, dtype=float),
                    'grid_pos': list(self.cursor_grid_pos),
                    'level': 1,
                    'last_hit_time': 0
                })
                return 1.0 # Small reward for placing
        return 0.0

    def _manage_waves(self):
        if self.wave_in_progress or self.current_wave_number > self.MAX_WAVES:
            return

        self.time_since_last_wave += 1
        if self.time_since_last_wave > self.WAVE_COOLDOWN_FRAMES:
            self.time_since_last_wave = 0
            self.current_wave_number += 1
            if self.current_wave_number <= self.MAX_WAVES:
                self._spawn_wave()
                self.wave_in_progress = True

    def _spawn_wave(self):
        num_waves = self.WAVE_BASE_COUNT + self.current_wave_number - 1
        speed = self.WAVE_BASE_SPEED + (self.current_wave_number - 1) * self.WAVE_SPEED_INCREMENT
        
        for _ in range(num_waves):
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -10.0])
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10.0])
            elif edge == 2: # Left
                pos = np.array([-10.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
            else: # Right
                pos = np.array([self.SCREEN_WIDTH + 10.0, self.np_random.uniform(0, self.SCREEN_HEIGHT)])

            target = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
            target += self.np_random.uniform(-50, 50, 2)
            
            direction = target - pos
            direction = direction / np.linalg.norm(direction)
            
            self.waves.append({
                'pos': pos,
                'vel': direction * speed,
                'color': self.COLOR_WAVE_SPECTRUM[self.np_random.integers(len(self.COLOR_WAVE_SPECTRUM))],
                'id': self.np_random.random()
            })

    def _update_waves(self):
        deflection_reward = 0
        core_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])
        core_radius = 30

        for wave in reversed(self.waves):
            wave['pos'] += wave['vel']

            if np.linalg.norm(wave['pos'] - core_pos) < core_radius:
                self.core_health -= self.WAVE_DAMAGE
                self._create_particles(wave['pos'], 30, self.COLOR_CORE, 3, 0.5)
                self.waves.remove(wave)
                continue

            for tower in self.towers:
                tower_radius = self.TOWER_BASE_RADIUS + (tower['level'] - 1) * 5
                dist = np.linalg.norm(wave['pos'] - tower['pos'])
                
                if dist < tower_radius and (self.steps - tower.get('last_hit_time', 0) > 10):
                    tower['last_hit_time'] = self.steps
                    
                    normal = (wave['pos'] - tower['pos']) / dist
                    wave['vel'] = wave['vel'] - 2 * np.dot(wave['vel'], normal) * normal
                    wave['pos'] = tower['pos'] + normal * (tower_radius + 1)
                    
                    self.resources = min(999, self.resources + self.TOWER_RESOURCE_GAIN * tower['level'])
                    deflection_reward += 1.0
                    
                    self._create_particles(wave['pos'], 10, self.COLOR_TOWER_UPGRADED, 2, 0.3)
                    break

            if not ((-20 < wave['pos'][0] < self.SCREEN_WIDTH + 20) and (-20 < wave['pos'][1] < self.SCREEN_HEIGHT + 20)):
                self.waves.remove(wave)
        
        if not self.waves:
            self.wave_in_progress = False
            
        return deflection_reward

    def _create_particles(self, pos, count, color, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1.0) * speed
            velocity = np.array([math.cos(angle), math.sin(angle)]) * vel_mag
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': self.np_random.uniform(0.5, 1.0) * life,
                'max_life': life,
                'color': color
            })

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['life'] -= 1 / self.metadata['render_fps']
            if p['life'] <= 0:
                self.particles.remove(p)

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
            "core_health": self.core_health,
            "resources": self.resources,
            "wave": self.current_wave_number,
            "towers": len(self.towers)
        }

    def _generate_starfield(self):
        if not hasattr(self, 'np_random'):
             self.reset(seed=0) # ensure np_random exists
        self.starfield = []
        for _ in range(150):
            self.starfield.append({
                'pos': (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                'size': self.np_random.integers(1, 3),
                'brightness': self.np_random.integers(50, 151)
            })

    def _render_game(self):
        for star in self.starfield:
            c = star['brightness']
            self.screen.set_at(star['pos'], (c, c, c))
            
        for x in range(0, self.SCREEN_WIDTH, self.GRID_CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        core_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        health_pulse = 0.5 + 0.5 * math.sin(self.steps * 0.1)
        for i in range(15, 0, -1):
            alpha = 50 * (1 - i / 15) * health_pulse
            color = (*self.COLOR_CORE_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, core_pos[0], core_pos[1], 30 + i, color)
        pygame.gfxdraw.aacircle(self.screen, core_pos[0], core_pos[1], 30, self.COLOR_CORE)
        pygame.gfxdraw.filled_circle(self.screen, core_pos[0], core_pos[1], 30, self.COLOR_CORE)

        for tower in self.towers:
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            level_ratio = (tower['level'] - 1) / (self.TOWER_MAX_LEVEL - 1) if self.TOWER_MAX_LEVEL > 1 else 0
            
            # Manual linear interpolation to replace np.lerp
            interpolated_color = self.COLOR_TOWER_BASE * (1 - level_ratio) + self.COLOR_TOWER_UPGRADED * level_ratio
            color = tuple(interpolated_color.astype(int))

            radius = int(self.TOWER_BASE_RADIUS + (tower['level'] - 1) * 5)
            
            for i in range(radius // 2, 0, -2):
                alpha = 80 * (1 - i / (radius//2))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i, (*color, alpha))
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            
            level_text = self.font_ui.render(str(tower['level']), True, self.COLOR_BG)
            self.screen.blit(level_text, level_text.get_rect(center=pos))

        for wave in self.waves:
            pos = (int(wave['pos'][0]), int(wave['pos'][1]))
            pulse = 1 + 0.2 * math.sin(self.steps * 0.3 + wave['id'] * 10)
            size = int(6 * pulse)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, wave['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, wave['color'])

        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)
            
        if not self.game_over:
            self._render_cursor()
            
    def _render_cursor(self):
        grid_x, grid_y = self.cursor_grid_pos
        rect = pygame.Rect(grid_x * self.GRID_CELL_SIZE, grid_y * self.GRID_CELL_SIZE, 
                           self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
        
        is_occupied = any(t['grid_pos'] == self.cursor_grid_pos for t in self.towers)
        can_afford_place = self.resources >= self.TOWER_COST
        can_afford_upgrade = False
        if is_occupied:
            tower = next(t for t in self.towers if t['grid_pos'] == self.cursor_grid_pos)
            if tower['level'] < self.TOWER_MAX_LEVEL:
                can_afford_upgrade = self.resources >= self.TOWER_UPGRADE_COST_BASE * tower['level']

        color = self.COLOR_CURSOR_INVALID
        if (is_occupied and can_afford_upgrade) or (not is_occupied and can_afford_place):
            color = self.COLOR_CURSOR_VALID

        s = pygame.Surface((self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        def draw_text(text, font, color, pos, anchor="topleft"):
            shadow = font.render(text, True, (0,0,0))
            content = font.render(text, True, color)
            shadow_rect = shadow.get_rect(**{anchor: (pos[0]+2, pos[1]+2)})
            content_rect = content.get_rect(**{anchor: pos})
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(content, content_rect)
            
        draw_text(f"RESOURCES: {self.resources}", self.font_ui, self.COLOR_TEXT, (10, 10))
        draw_text(f"CORE HEALTH: {max(0, self.core_health)}%", self.font_ui, self.COLOR_TEXT, (10, 40))
        
        wave_str = f"WAVE: {min(self.current_wave_number, self.MAX_WAVES)} / {self.MAX_WAVES}"
        if self.current_wave_number == 0:
            wave_str = "PREPARING..."
        draw_text(wave_str, self.font_ui, self.COLOR_TEXT, (self.SCREEN_WIDTH - 10, 10), anchor="topright")

        if self.is_paused and not self.game_over:
            draw_text("PAUSED", self.font_title, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), anchor="center")
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            title_text = "VICTORY" if self.win_condition_met else "CORE DESTROYED"
            subtitle_text = "All waves defended!" if self.win_condition_met else "The melody has faded."
            
            draw_text(title_text, self.font_title, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30), anchor="center")
            draw_text(subtitle_text, self.font_subtitle, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30), anchor="center")

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It requires pygame to be installed and will open a window.
    # The main GameEnv class does not require a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Octave Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    
    action = [0, 0, 0]

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        action[0] = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata['render_fps'])
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000)

    env.close()