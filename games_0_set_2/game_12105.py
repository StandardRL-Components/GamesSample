import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:06:00.465577
# Source Brief: brief_02105.md
# Brief Index: 2105
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A puzzle/strategy Gymnasium environment where the agent matches seasonal tiles
    to launch attacks at a shared energy pool.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Match pairs of different seasonal tiles to launch attacks against a central energy core. "
        "Build momentum for faster attacks and use fusion charges for a powerful boost."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to select a tile, then an adjacent "
        "different tile to perform a match. Hold shift when matching to use a powerful Fusion Attack."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_ROWS, GRID_COLS = 6, 10
    TILE_SIZE = 36
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * TILE_SIZE) // 2
    GRID_MARGIN_Y = 80

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_GRID = (40, 30, 60)
    COLOR_SPRING = (50, 220, 150)
    COLOR_AUTUMN = (255, 120, 30)
    COLOR_ENERGY = (180, 50, 255)
    COLOR_MOMENTUM = (255, 220, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255)

    # Game Parameters
    MAX_ENERGY = 1000
    MAX_MOMENTUM = 100
    MAX_STEPS = 1000
    BASE_DAMAGE = 50
    MOMENTUM_GAIN_ON_MATCH = 15
    MOMENTUM_DECAY_PER_STEP = 0.1
    FUSION_REWARD = 5.0
    WIN_REWARD = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy_pool = 0
        self.momentum = 0
        self.fusion_charges = 0
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0]
        self.selected_tile_pos = None
        self.projectiles = []
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.projectile_damage_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.energy_pool = self.MAX_ENERGY
        self.momentum = self.MAX_MOMENTUM / 4
        self.fusion_charges = 1
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.visual_cursor_pos = [float(self.cursor_pos[0]), float(self.cursor_pos[1])]
        self.selected_tile_pos = None
        
        self.projectiles = []
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.projectile_damage_reward = 0

        self._generate_board()
        while not self._has_valid_moves():
            self._generate_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # --- Handle Input and Game Logic ---
        action_taken = self._handle_input(movement, space_pressed, shift_pressed)
        
        if action_taken:
            match_reward, fusion_used = action_taken
            reward += match_reward
            if fusion_used:
                reward += self.FUSION_REWARD
        else:
            reward -= 0.01 # Small penalty for idle steps

        # --- Update Game State ---
        self._update_projectiles()
        self._update_particles()
        
        # Update momentum
        self.momentum = max(0, self.momentum - self.MOMENTUM_DECAY_PER_STEP)
        
        # Update visual cursor
        self.visual_cursor_pos[0] += (self.cursor_pos[0] - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (self.cursor_pos[1] - self.visual_cursor_pos[1]) * 0.4

        # --- Calculate Step Reward ---
        # Reward for damage is added when projectiles hit
        reward += self.projectile_damage_reward
        self.projectile_damage_reward = 0 # Reset for next step

        self.score += reward
        
        # --- Check Termination ---
        terminated = self.energy_pool <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        if terminated and not self.game_over:
            self.game_over = True
            if self.energy_pool <= 0:
                reward += self.WIN_REWARD
                self.score += self.WIN_REWARD
                # Victory particle explosion
                for _ in range(200):
                    self._create_particles(
                        (self.SCREEN_WIDTH // 2, 35), 
                        random.choice([self.COLOR_ENERGY, self.COLOR_MOMENTUM, (255,255,255)]), 
                        1, 
                        (self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5)),
                        60
                    )

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        if movement == 2: self.cursor_pos[0] += 1  # Down
        if movement == 3: self.cursor_pos[1] -= 1  # Left
        if movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] %= self.GRID_ROWS
        self.cursor_pos[1] %= self.GRID_COLS

        # Handle selection/match
        if space_pressed or shift_pressed:
            r, c = self.cursor_pos
            if self.grid[r, c] == 0: # Cannot select empty space
                return None

            if self.selected_tile_pos is None:
                self.selected_tile_pos = (r, c)
                # SFX: select_tile.wav
            else:
                is_fused_attack = shift_pressed and self.fusion_charges > 0
                return self._attempt_match(self.selected_tile_pos, (r, c), is_fused_attack)
        return None

    def _attempt_match(self, pos1, pos2, is_fused):
        r1, c1 = pos1
        r2, c2 = pos2

        # Check for adjacency
        is_adjacent = abs(r1 - r2) + abs(c1 - c2) == 1
        if not is_adjacent:
            self.selected_tile_pos = pos2 # Select the new tile instead
            return None

        type1 = self.grid[r1, c1]
        type2 = self.grid[r2, c2]

        # Check for different, non-empty types
        if type1 != 0 and type2 != 0 and type1 != type2:
            # MATCH SUCCESS
            # SFX: match_success.wav
            damage_multiplier = 2.0 if is_fused else 1.0
            damage = self.BASE_DAMAGE * damage_multiplier

            start_pixel_pos = (
                self.GRID_MARGIN_X + (c1 + c2) / 2 * self.TILE_SIZE + self.TILE_SIZE / 2,
                self.GRID_MARGIN_Y + (r1 + r2) / 2 * self.TILE_SIZE + self.TILE_SIZE / 2
            )
            self._launch_attack(start_pixel_pos, damage, type1, type2)
            
            if is_fused:
                self.fusion_charges -= 1
            
            self.grid[r1, c1] = 0
            self.grid[r2, c2] = 0
            self._refill_board()
            
            self.momentum = min(self.MAX_MOMENTUM, self.momentum + self.MOMENTUM_GAIN_ON_MATCH)
            
            # Award a fusion charge for a high-momentum match
            if self.momentum > self.MAX_MOMENTUM * 0.9 and self.np_random.random() < 0.25:
                self.fusion_charges = min(3, self.fusion_charges + 1)
                # SFX: fusion_charge_gain.wav

            self.selected_tile_pos = None
            return 1.0, is_fused # Reward for match, and if fusion was used
        else:
            # MATCH FAIL
            # SFX: match_fail.wav
            self.selected_tile_pos = pos2 # Select the new tile
            return None

    def _launch_attack(self, start_pos, damage, type1, type2):
        projectile_speed = 3 + 8 * (self.momentum / self.MAX_MOMENTUM)
        
        # Color based on the two matched tiles
        color1 = self.COLOR_SPRING if type1 == 1 else self.COLOR_AUTUMN
        color2 = self.COLOR_SPRING if type2 == 1 else self.COLOR_AUTUMN
        
        projectile = {
            "pos": list(start_pos),
            "damage": damage,
            "speed": projectile_speed,
            "color1": color1,
            "color2": color2,
            "angle": 0
        }
        self.projectiles.append(projectile)

    def _update_projectiles(self):
        self.projectile_damage_reward = 0
        for p in self.projectiles[:]:
            target_pos = (self.SCREEN_WIDTH / 2, 35)
            angle = math.atan2(target_pos[1] - p["pos"][1], target_pos[0] - p["pos"][0])
            p["pos"][0] += math.cos(angle) * p["speed"]
            p["pos"][1] += math.sin(angle) * p["speed"]
            p["angle"] += 0.2

            # Create trail
            if self.steps % 2 == 0:
                self._create_particles(p["pos"], p["color1"], 1, (0,0), 15, 0.5)
                self._create_particles(p["pos"], p["color2"], 1, (0,0), 15, 0.5)

            # Check for hit
            if math.hypot(p["pos"][0] - target_pos[0], p["pos"][1] - target_pos[1]) < 20:
                self.energy_pool = max(0, self.energy_pool - p["damage"])
                self.projectile_damage_reward += p["damage"] * 0.01 # Reward scaling
                self.projectiles.remove(p)
                # SFX: impact.wav
                # Create impact explosion
                for _ in range(50):
                    self._create_particles(
                        target_pos, 
                        random.choice([p["color1"], p["color2"], self.COLOR_ENERGY]), 
                        1, 
                        (self.np_random.uniform(-3, 3), self.np_random.uniform(-2, 2)),
                        40
                    )
    
    def _create_particles(self, pos, color, count, velocity_range, lifetime, size_mult=1.0):
        for _ in range(count):
            vel = list(velocity_range) if isinstance(velocity_range, (list, tuple)) else [self.np_random.uniform(-velocity_range, velocity_range), self.np_random.uniform(-velocity_range, velocity_range)]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifetime": lifetime,
                "max_life": lifetime,
                "color": color,
                "size": self.np_random.uniform(3, 6) * size_mult,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # Gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _refill_board(self):
        for c in range(self.GRID_COLS):
            empty_count = 0
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    self.grid[r + empty_count, c] = self.grid[r, c]
                    self.grid[r, c] = 0
            
            for r in range(empty_count):
                self.grid[r, c] = self.np_random.integers(1, 3) # 1 or 2
        
        if not self._has_valid_moves():
            # Anti-softlock: if no moves, reshuffle
            self._generate_board()

    def _generate_board(self):
        self.grid = self.np_random.integers(1, 3, size=(self.GRID_ROWS, self.GRID_COLS))

    def _has_valid_moves(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r,c]
                # Check right
                if c + 1 < self.GRID_COLS and self.grid[r, c+1] != tile_type:
                    return True
                # Check down
                if r + 1 < self.GRID_ROWS and self.grid[r+1, c] != tile_type:
                    return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_effects()
        self._render_board_and_tiles()
        self._render_cursor_and_selection()
        self._render_projectiles_and_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "energy": self.energy_pool,
            "momentum": self.momentum,
            "fusion_charges": self.fusion_charges,
        }

    # --- Rendering Methods ---
    def _render_background_effects(self):
        # Slow moving starfield
        for i in range(50):
            x = (hash(i*10) * 3 + self.steps * 0.1) % self.SCREEN_WIDTH
            y = (hash(i*30) * 5 + self.steps * 0.05) % self.SCREEN_HEIGHT
            size = hash(i) % 2 + 1
            pygame.draw.rect(self.screen, (30,20,50), (x, y, size, size))

    def _render_board_and_tiles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile_type = self.grid[r, c]
                rect = pygame.Rect(
                    self.GRID_MARGIN_X + c * self.TILE_SIZE,
                    self.GRID_MARGIN_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                # Draw grid cell background
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1, border_radius=4)

                if tile_type > 0:
                    self._draw_tile(rect, tile_type)

    def _draw_tile(self, rect, tile_type):
        color = self.COLOR_SPRING if tile_type == 1 else self.COLOR_AUTUMN
        core_color = (200, 255, 230) if tile_type == 1 else (255, 180, 100)

        # Use gfxdraw for anti-aliasing
        pygame.gfxdraw.box(self.screen, rect.inflate(-4, -4), (*color, 150))
        pygame.gfxdraw.rectangle(self.screen, rect.inflate(-4, -4), color)
        
        center_x, center_y = rect.center
        inner_rect = rect.inflate(-16, -16)
        pygame.gfxdraw.box(self.screen, inner_rect, (*core_color, 200))
        pygame.gfxdraw.rectangle(self.screen, inner_rect, core_color)


    def _render_cursor_and_selection(self):
        # Draw selected tile highlight
        if self.selected_tile_pos:
            r, c = self.selected_tile_pos
            rect = pygame.Rect(
                self.GRID_MARGIN_X + c * self.TILE_SIZE,
                self.GRID_MARGIN_Y + r * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )
            pygame.draw.rect(self.screen, (255, 255, 0), rect.inflate(2, 2), 3, border_radius=6)

        # Draw cursor
        vr, vc = self.visual_cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + vc * self.TILE_SIZE,
            self.GRID_MARGIN_Y + vr * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        
        # Pulsing effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        line_width = int(2 + pulse * 2)
        color = tuple(int(c * (0.8 + pulse * 0.2)) for c in self.COLOR_CURSOR)
        pygame.draw.rect(self.screen, color, cursor_rect, line_width, border_radius=6)


    def _render_projectiles_and_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_life"]))
            color = (*p["color"], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"] * (p["lifetime"] / p["max_life"])), color)

        for p in self.projectiles:
            x, y = int(p["pos"][0]), int(p["pos"][1])
            # Draw two rotating points for a "hybrid" feel
            offset1 = 10 * math.sin(p["angle"])
            offset2 = 10 * math.cos(p["angle"])
            pygame.gfxdraw.filled_circle(self.screen, int(x + offset1), int(y + offset2), 8, p["color1"])
            pygame.gfxdraw.filled_circle(self.screen, int(x - offset1), int(y - offset2), 8, p["color2"])
            pygame.gfxdraw.aacircle(self.screen, int(x + offset1), int(y + offset2), 8, p["color1"])
            pygame.gfxdraw.aacircle(self.screen, int(x - offset1), int(y - offset2), 8, p["color2"])


    def _render_ui(self):
        # --- Energy Pool ---
        energy_ratio = self.energy_pool / self.MAX_ENERGY
        bar_width = 300
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        # Background bar
        pygame.draw.rect(self.screen, (50, 50, 80), (bar_x, 20, bar_width, 20), border_radius=5)
        # Foreground bar
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (bar_x, 20, bar_width * energy_ratio, 20), border_radius=5)
        # Text
        energy_text = self.font_main.render(f"ENERGY: {int(self.energy_pool)} / {self.MAX_ENERGY}", True, self.COLOR_TEXT)
        self.screen.blit(energy_text, (bar_x + bar_width + 10, 20))

        # --- Momentum Bar ---
        momentum_ratio = self.momentum / self.MAX_MOMENTUM
        mom_bar_width = 200
        mom_bar_x = 10
        pygame.draw.rect(self.screen, (80, 80, 50), (mom_bar_x, 20, mom_bar_width, 10), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM, (mom_bar_x, 20, mom_bar_width * momentum_ratio, 10), border_radius=3)
        mom_text = self.font_small.render("MOMENTUM", True, self.COLOR_TEXT)
        self.screen.blit(mom_text, (mom_bar_x, 5))

        # --- Fusion Charges ---
        fusion_text = self.font_small.render("FUSION [SHIFT]", True, self.COLOR_TEXT)
        self.screen.blit(fusion_text, (10, self.SCREEN_HEIGHT - 45))
        for i in range(self.fusion_charges):
            pygame.gfxdraw.filled_circle(self.screen, 30 + i * 25, self.SCREEN_HEIGHT - 20, 8, self.COLOR_ENERGY)
            pygame.gfxdraw.aacircle(self.screen, 30 + i * 25, self.SCREEN_HEIGHT - 20, 8, self.COLOR_MOMENTUM)

        # --- Score & Steps ---
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, self.SCREEN_HEIGHT - 45))
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, self.SCREEN_HEIGHT - 25))

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Fusion Attacker")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    print("\n" + env.user_guide)
    print("Q: Quit")
    
    while not done:
        # Action defaults
        movement = 0 # None
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                done = True

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
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            done = True

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()