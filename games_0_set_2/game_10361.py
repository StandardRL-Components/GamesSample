import gymnasium as gym
import os
import pygame
import math
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a digital grid, dodging deadly lasers while sabotaging targets and solving puzzles for upgrades."
    user_guide = "Use arrow keys to move. Hold space to shrink or interact with objects (targets/puzzles). Hold shift to grow."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000
    VICTORY_CONDITION = 3

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_LASER = (255, 20, 20)
    COLOR_LASER_GLOW = (255, 100, 100)
    COLOR_TARGET = (255, 200, 0)
    COLOR_TARGET_INACTIVE = (80, 70, 0)
    COLOR_PUZZLE = (150, 50, 255)
    COLOR_PUZZLE_INACTIVE = (50, 20, 80)
    COLOR_TEXT = (220, 220, 240)

    # Player
    PLAYER_BASE_SPEED = 4.0
    PLAYER_MIN_SIZE = 8
    PLAYER_MAX_SIZE = 20
    PLAYER_SIZE_CHANGE_RATE = 0.5

    # Lasers
    LASER_INITIAL_SPEED = 1.0
    LASER_SPEED_INCREMENT = 0.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # This custom font avoids loading external files
        self._init_custom_font()

        # Initialize all state variables to prevent AttributeError
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_size = 0.0
        self.player_target_size = 0.0
        self.player_speed = 0.0
        self.lasers = []
        self.targets = []
        self.puzzles = []
        self.particles = []
        self.sabotaged_count = 0
        self.laser_speed = 0.0
        self.interaction_cooldown = 0
        self.upgrades = {"speed": False, "shield": False, "size": False}
        self.shield_active = False
        self.grace_period = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.sabotaged_count = 0
        self.grace_period = self.FPS * 2  # 2-second grace period for stability

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_target_size = (self.PLAYER_MIN_SIZE + self.PLAYER_MAX_SIZE) / 2
        self.player_size = self.player_target_size
        self.player_speed = self.PLAYER_BASE_SPEED

        self.laser_speed = self.LASER_INITIAL_SPEED
        self.lasers = self._initialize_lasers()

        self.targets = [
            {'pos': np.array([100, 100]), 'active': True, 'radius': 25},
            {'pos': np.array([self.WIDTH - 100, self.HEIGHT - 100]), 'active': True, 'radius': 25},
            {'pos': np.array([100, self.HEIGHT - 100]), 'active': True, 'radius': 25},
        ]
        self.puzzles = [
            {'pos': np.array([self.WIDTH - 100, 100]), 'active': True, 'radius': 20}
        ]

        self.particles = []
        self.interaction_cooldown = 0
        self.upgrades = {"speed": False, "shield": False, "size": False}
        self.shield_active = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.01  # Small survival reward

        self.steps += 1
        if self.interaction_cooldown > 0:
            self.interaction_cooldown -= 1
        if self.grace_period > 0:
            self.grace_period -= 1

        # 1. Handle Input and Intent
        # Movement
        move_vec = np.zeros(2, dtype=np.float32)
        if movement == 1: move_vec[1] = -1  # Up
        elif movement == 2: move_vec[1] = 1   # Down
        elif movement == 3: move_vec[0] = -1  # Left
        elif movement == 4: move_vec[0] = 1   # Right

        # Interaction / Size Change
        interaction_triggered = False
        if space_held and self.interaction_cooldown == 0:
            # Check for target interaction
            for target in self.targets:
                if target['active'] and np.linalg.norm(self.player_pos - target['pos']) < target['radius'] + self.player_size:
                    reward += self._sabotage_target(target)
                    interaction_triggered = True
                    break
            # Check for puzzle interaction
            if not interaction_triggered:
                for puzzle in self.puzzles:
                    if puzzle['active'] and np.linalg.norm(self.player_pos - puzzle['pos']) < puzzle['radius'] + self.player_size:
                        reward += self._solve_puzzle(puzzle)
                        interaction_triggered = True
                        break

        if not interaction_triggered:
            if space_held:  # Shrink
                self.player_target_size -= self.PLAYER_SIZE_CHANGE_RATE
                self._spawn_particles(self.player_pos, 5, self.COLOR_PLAYER, 1.0, 'shrink')
            if shift_held:  # Grow
                self.player_target_size += self.PLAYER_SIZE_CHANGE_RATE
                self._spawn_particles(self.player_pos, 5, self.COLOR_PLAYER, 1.0, 'grow')

        # 2. Update Game State
        # Player position
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        self.player_pos += move_vec * self.player_speed
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_size, self.WIDTH - self.player_size)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_size, self.HEIGHT - self.player_size)

        # Player size
        max_size = self.PLAYER_MAX_SIZE * 1.5 if self.upgrades['size'] else self.PLAYER_MAX_SIZE
        self.player_target_size = np.clip(self.player_target_size, self.PLAYER_MIN_SIZE, max_size)
        self.player_size += (self.player_target_size - self.player_size) * 0.2  # Smooth interpolation

        # Lasers
        for laser in self.lasers:
            laser['angle'] += laser['dir'] * self.laser_speed * 0.02

        # Particles
        self._update_particles()

        # 3. Check Collisions and Termination
        terminated = False
        truncated = False
        if self._check_laser_collision():
            if self.shield_active:
                self.shield_active = False
                self._spawn_particles(self.player_pos, 30, (200, 200, 255), 3.0, 'burst')
            else:
                reward = -100.0
                self.game_over = True
                terminated = True
                self._spawn_particles(self.player_pos, 50, self.COLOR_PLAYER, 4.0, 'burst')

        if self.sabotaged_count >= self.VICTORY_CONDITION:
            reward += 100.0
            self.game_over = True
            terminated = True

        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _sabotage_target(self, target):
        target['active'] = False
        self.sabotaged_count += 1
        self.laser_speed += self.LASER_SPEED_INCREMENT
        self.interaction_cooldown = 15  # 0.5s cooldown
        self._spawn_particles(target['pos'], 40, self.COLOR_TARGET, 3.0, 'burst')
        return 10.0

    def _solve_puzzle(self, puzzle):
        puzzle['active'] = False
        self.interaction_cooldown = 15

        available_upgrades = [k for k, v in self.upgrades.items() if not v]
        if available_upgrades:
            upgrade_to_grant = self.np_random.choice(available_upgrades)
            self.upgrades[upgrade_to_grant] = True
            if upgrade_to_grant == 'speed':
                self.player_speed *= 1.5
            elif upgrade_to_grant == 'shield':
                self.shield_active = True

        self._spawn_particles(puzzle['pos'], 40, self.COLOR_PUZZLE, 3.0, 'burst')
        return 5.0

    def _check_laser_collision(self):
        if self.grace_period > 0:
            return False

        for laser in self.lasers:
            p1 = laser['origin']
            p2 = p1 + np.array([math.cos(laser['angle']), math.sin(laser['angle'])]) * 1000

            # Simplified line-circle collision
            d = p2 - p1
            f = p1 - self.player_pos
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - self.player_size**2
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                discriminant = math.sqrt(discriminant)
                t1 = (-b - discriminant) / (2*a)
                t2 = (-b + discriminant) / (2*a)
                if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_lasers()
        self._render_targets_and_puzzles()
        self._render_particles()
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sabotaged": self.sabotaged_count,
            "upgrades": self.upgrades
        }

    # --- RENDER METHODS ---
    def _render_background(self):
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_lasers(self):
        for laser in self.lasers:
            start_pos = laser['origin']
            end_pos = start_pos + np.array([math.cos(laser['angle']), math.sin(laser['angle'])]) * 1000

            # Glow effect
            pygame.draw.aaline(self.screen, self.COLOR_LASER_GLOW, start_pos, end_pos, 4)
            # Core beam
            pygame.draw.aaline(self.screen, self.COLOR_LASER, start_pos, end_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(start_pos[0]), int(start_pos[1]), 5, self.COLOR_LASER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(start_pos[0]), int(start_pos[1]), 5, self.COLOR_LASER_GLOW)

    def _render_targets_and_puzzles(self):
        pulse = (math.sin(self.steps * 0.2) + 1) / 2

        for target in self.targets:
            color = self.COLOR_TARGET if target['active'] else self.COLOR_TARGET_INACTIVE
            if target['active']:
                alpha = int(150 + pulse * 105)
                color = (min(255, color[0]+alpha//4), min(255, color[1]+alpha//4), color[2])
            pygame.gfxdraw.filled_circle(self.screen, int(target['pos'][0]), int(target['pos'][1]), target['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, int(target['pos'][0]), int(target['pos'][1]), target['radius'], color)

        for puzzle in self.puzzles:
            color = self.COLOR_PUZZLE if puzzle['active'] else self.COLOR_PUZZLE_INACTIVE
            if puzzle['active']:
                alpha = int(150 + pulse * 105)
                color = (min(255, color[0]+alpha//4), color[1], min(255, color[2]+alpha//4))
            r = puzzle['radius']
            points = [
                (puzzle['pos'][0], puzzle['pos'][1] - r), (puzzle['pos'][0] + r, puzzle['pos'][1]),
                (puzzle['pos'][0], puzzle['pos'][1] + r), (puzzle['pos'][0] - r, puzzle['pos'][1])
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_player(self):
        if self.game_over: return

        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        size = int(self.player_size)

        # Glow
        glow_pulse = (math.sin(self.steps * 0.15) + 1) / 2
        glow_radius = int(size * (1.5 + glow_pulse * 0.5))
        glow_alpha = int(50 + glow_pulse * 50)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, glow_alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Shield
        if self.shield_active:
            shield_pulse = (math.sin(self.steps * 0.4) + 1) / 2
            shield_radius = int(size * 1.4)
            shield_color = (200, 200, 255, int(100 + shield_pulse * 100))
            pygame.gfxdraw.aacircle(self.screen, px, py, shield_radius, shield_color)
            pygame.gfxdraw.aacircle(self.screen, px, py, shield_radius-1, shield_color)

        # Core
        pygame.gfxdraw.filled_circle(self.screen, px, py, size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, size, self.COLOR_PLAYER)

        # Size bar
        bar_w, bar_h = 30, 4
        bar_x, bar_y = px - bar_w // 2, py + size + 5
        max_s = self.PLAYER_MAX_SIZE * 1.5 if self.upgrades['size'] else self.PLAYER_MAX_SIZE
        fill_ratio = (self.player_size - self.PLAYER_MIN_SIZE) / (max_s - self.PLAYER_MIN_SIZE)
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x, bar_y, int(bar_w * fill_ratio), bar_h))

    def _render_ui(self):
        # Sabotage count
        self._render_text(f"TARGETS: {self.sabotaged_count}/{self.VICTORY_CONDITION}", 10, 10)

        # Upgrades
        self._render_text("UPGRADES:", 10, 30)
        if self.upgrades['speed']: self._render_text("SPD", 90, 30, color=(100, 255, 100))
        if self.upgrades['shield']: self._render_text("SHD", 125, 30, color=(100, 100, 255))
        if self.upgrades['size']: self._render_text("SZE", 160, 30, color=(255, 100, 100))

    # --- PARTICLE SYSTEM ---
    def _spawn_particles(self, pos, count, color, speed, p_type):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(0.5, 1.0) * speed
            elif p_type == 'shrink':
                vel = np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)]) * speed
            elif p_type == 'grow':
                angle = self.np_random.uniform(0, 2 * math.pi)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed * -0.5  # implosion
            else:
                vel = np.zeros(2)

            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95  # friction
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / 40
            color = (*p['color'], int(alpha * 255))
            radius = int(p['radius'] * alpha)
            if radius > 0:
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (int(p['pos'][0]) - radius, int(p['pos'][1]) - radius), special_flags=pygame.BLEND_RGBA_ADD)


    # --- HELPERS ---
    def _initialize_lasers(self):
        return [
            {'origin': np.array([0, 0]), 'angle': self.np_random.uniform(0, math.pi/2), 'dir': 1},
            {'origin': np.array([self.WIDTH, self.HEIGHT]), 'angle': self.np_random.uniform(math.pi, 3*math.pi/2), 'dir': -1},
            {'origin': np.array([self.WIDTH/2, -20]), 'angle': self.np_random.uniform(math.pi/4, 3*math.pi/4), 'dir': 1},
        ]

    def _init_custom_font(self):
        self.FONT_MAP = {
            'A': [0x7E, 0x11, 0x11, 0x11, 0x7E], 'B': [0x7F, 0x49, 0x49, 0x49, 0x36],
            'C': [0x3E, 0x41, 0x41, 0x41, 0x22], 'D': [0x7F, 0x41, 0x41, 0x22, 0x1C],
            'E': [0x7F, 0x49, 0x49, 0x41, 0x41], 'F': [0x7F, 0x09, 0x09, 0x01, 0x01],
            'G': [0x3E, 0x41, 0x49, 0x49, 0x7A], 'H': [0x7F, 0x08, 0x08, 0x08, 0x7F],
            'I': [0x00, 0x41, 0x7F, 0x41, 0x00], 'J': [0x20, 0x40, 0x41, 0x3F, 0x01],
            'K': [0x7F, 0x08, 0x14, 0x22, 0x41], 'L': [0x7F, 0x40, 0x40, 0x40, 0x40],
            'M': [0x7F, 0x02, 0x0C, 0x02, 0x7F], 'N': [0x7F, 0x04, 0x08, 0x10, 0x7F],
            'O': [0x3E, 0x41, 0x41, 0x41, 0x3E], 'P': [0x7F, 0x09, 0x09, 0x09, 0x06],
            'Q': [0x3E, 0x41, 0x51, 0x21, 0x5E], 'R': [0x7F, 0x09, 0x19, 0x29, 0x46],
            'S': [0x46, 0x49, 0x49, 0x49, 0x31], 'T': [0x01, 0x01, 0x7F, 0x01, 0x01],
            'U': [0x3F, 0x40, 0x40, 0x40, 0x3F], 'V': [0x1F, 0x20, 0x40, 0x20, 0x1F],
            'W': [0x3F, 0x40, 0x38, 0x40, 0x3F], 'X': [0x63, 0x14, 0x08, 0x14, 0x63],
            'Y': [0x07, 0x08, 0x70, 0x08, 0x07], 'Z': [0x61, 0x51, 0x49, 0x45, 0x43],
            '0': [0x3E, 0x51, 0x49, 0x45, 0x3E], '1': [0x00, 0x42, 0x7F, 0x40, 0x00],
            '2': [0x42, 0x61, 0x51, 0x49, 0x46], '3': [0x21, 0x41, 0x45, 0x4B, 0x31],
            '4': [0x18, 0x14, 0x12, 0x7F, 0x10], '5': [0x27, 0x45, 0x45, 0x45, 0x39],
            '6': [0x3C, 0x4A, 0x49, 0x49, 0x30], '7': [0x01, 0x71, 0x09, 0x05, 0x03],
            '8': [0x36, 0x49, 0x49, 0x49, 0x36], '9': [0x06, 0x49, 0x49, 0x29, 0x1E],
            ':': [0x00, 0x36, 0x36, 0x00, 0x00], '/': [0x00, 0x01, 0x02, 0x04, 0x08],
            ' ': [0x00, 0x00, 0x00, 0x00, 0x00]
        }

    def _render_text(self, text, x, y, size=2, color=None):
        if color is None: color = self.COLOR_TEXT
        text = text.upper()
        for char in text:
            if char in self.FONT_MAP:
                pattern = self.FONT_MAP[char]
                for row_idx, row in enumerate(pattern):
                    for col_idx in range(7):
                        if (row >> col_idx) & 1:
                            pygame.draw.rect(self.screen, color, (x + (6 - col_idx) * size, y + row_idx * size, size, size))
            x += 6 * size

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This allows playing the game manually for testing and visualization
    # To run, you will need to `pip install pygame`
    # It is not included in the environment's dependencies.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Circuit Sabotage")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    total_reward = 0
    
    while not done:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()