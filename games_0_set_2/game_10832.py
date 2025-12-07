import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:01:25.741170
# Source Brief: brief_00832.md
# Brief Index: 832
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for Boson "cards"
class Boson:
    def __init__(self, grid_pos, boson_type, colors, cell_size, grid_offset):
        self.grid_pos = list(grid_pos)  # [row, col]
        self.boson_type = boson_type
        self.colors = colors
        self.cell_size = cell_size
        self.grid_offset = grid_offset
        
        self.target_pix_pos = self._get_target_pixel_pos()
        self.current_pix_pos = list(self.target_pix_pos)
        
        self.is_matched = False
        self.pulse = random.uniform(0, 2 * math.pi)
        self.creation_time = pygame.time.get_ticks()

    def _get_target_pixel_pos(self):
        x = self.grid_offset[0] + self.grid_pos[1] * self.cell_size + self.cell_size / 2
        y = self.grid_offset[1] + self.grid_pos[0] * self.cell_size + self.cell_size / 2
        return [x, y]

    def update(self):
        # Smooth interpolation for movement
        self.target_pix_pos = self._get_target_pixel_pos()
        self.current_pix_pos[0] += (self.target_pix_pos[0] - self.current_pix_pos[0]) * 0.25
        self.current_pix_pos[1] += (self.target_pix_pos[1] - self.current_pix_pos[1]) * 0.25
        self.pulse += 0.1

    def draw(self, surface):
        pos = (int(self.current_pix_pos[0]), int(self.current_pix_pos[1]))
        color = self.colors[self.boson_type]
        
        # Pulse effect
        pulse_size = 2 * math.sin(self.pulse)
        
        # Outer glow
        glow_radius = int(self.cell_size * 0.4 + pulse_size)
        glow_color = (*color, 60)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], glow_radius, glow_color)
        
        # Core
        core_radius = int(self.cell_size * 0.3)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], core_radius, color)
        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], core_radius, (255, 255, 255))
        
        # Inner highlight
        highlight_radius = int(core_radius * 0.5)
        highlight_color = (min(255, c+80) for c in color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], highlight_radius, (*highlight_color, 200))

# Helper class for particles
class Particle:
    def __init__(self, pos, vel, color, lifespan, size):
        self.pos = list(pos)
        self.vel = list(vel)
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan
        self.size = size

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1 # a little gravity
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color_with_alpha = (*self.color, alpha)
            size = int(self.size * (self.lifespan / self.max_lifespan))
            if size > 0:
                pygame.draw.circle(surface, color_with_alpha, (int(self.pos[0]), int(self.pos[1])), size)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A cosmic match-3 puzzle game. Swap fundamental particles, create chains, "
        "and flip gravity to reach the target score before time runs out."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to swap with an adjacent particle "
        "in the direction of your last move. Press shift to flip gravity."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 8, 8
        self.CELL_SIZE = 40
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.GRID_OFFSET = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y)
        self.MAX_STEPS = 60 * 60 # 60 FPS for 60 seconds

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_BG_FLIPPED = (25, 5, 10)
        self.COLOR_GRID = (50, 50, 80, 100)
        self.COLOR_CURSOR = (255, 255, 0)
        self.BOSON_COLORS = [
            (220, 50, 50),   # Strong Force (Red)
            (50, 100, 220),  # Weak Force (Blue)
            (50, 220, 100),  # Higgs (Green)
            (200, 50, 200),  # Graviton (Purple) - unlockable
            (220, 150, 50)   # Photon (Orange) - unlockable
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.target_score = 0
        self.grid = []
        self.cursor_pos = []
        self.last_move_dir = (0, 0)
        self.gravity_dir = 1
        self.unlocked_boson_types = 3
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.chain_count = 0
        self.background_stars = []
        
        self._generate_stars()
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for debugging and not part of the standard env
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.target_score = 100
        self.unlocked_boson_types = 3
        
        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.last_move_dir = (0, 0) # (dr, dc)
        self.gravity_dir = 1 # 1 for down, -1 for up
        self.particles = []
        self.chain_count = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._populate_grid()
        
        return self._get_observation(), self._get_info()

    def _populate_grid(self):
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self._create_boson(r, c)
        
        # Ensure no initial matches
        while self._find_and_mark_matches():
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r][c].is_matched:
                        self._create_boson(r, c)

    def _create_boson(self, r, c):
        boson_type = self.np_random.integers(0, self.unlocked_boson_types)
        self.grid[r][c] = Boson((r, c), boson_type, self.BOSON_COLORS, self.CELL_SIZE, self.GRID_OFFSET)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        reward = 0

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        self._handle_movement(movement)
        if space_pressed: self._handle_swap()
        if shift_pressed: self._handle_gravity_flip()

        # --- Update Game State ---
        self._update_bosons_and_particles()
        
        # --- Match and Gravity Logic ---
        matches_found = self._find_and_mark_matches()
        if matches_found:
            self.chain_count += 1
            reward += self._process_matches()
        else:
            self.chain_count = 0

        self._apply_gravity_and_refill()

        # --- Unlock new bosons ---
        if self.score > 2000 and self.unlocked_boson_types < 5:
            self.unlocked_boson_types = 5
        elif self.score > 1000 and self.unlocked_boson_types < 4:
            self.unlocked_boson_types = 4

        # --- Check Termination ---
        terminated = self.time_left <= 0 or self.score >= self.target_score
        if terminated:
            self.game_over = True
            if self.score >= self.target_score:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        dr, dc = 0, 0
        if movement == 1: dr = -1  # Up
        elif movement == 2: dr = 1   # Down
        elif movement == 3: dc = -1  # Left
        elif movement == 4: dc = 1   # Right
        
        if dr != 0 or dc != 0:
            self.last_move_dir = (dr, dc)
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dr, 0, self.GRID_ROWS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dc, 0, self.GRID_COLS - 1)

    def _handle_swap(self):
        r1, c1 = self.cursor_pos
        dr, dc = self.last_move_dir
        r2, c2 = r1 + dr, c1 + dc

        if 0 <= r2 < self.GRID_ROWS and 0 <= c2 < self.GRID_COLS:
            # Perform swap
            self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
            self.grid[r1][c1].grid_pos = [r1, c1]
            self.grid[r2][c2].grid_pos = [r2, c2]
            
            # Check if swap creates a match
            if not self._find_and_mark_matches():
                # If not, swap back
                self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]
                self.grid[r1][c1].grid_pos = [r1, c1]
                self.grid[r2][c2].grid_pos = [r2, c2]
            else:
                # Valid swap, reset markers for processing
                for r in range(self.GRID_ROWS):
                    for c in range(self.GRID_COLS):
                        self.grid[r][c].is_matched = False
                # SFX: Play swap sound

    def _handle_gravity_flip(self):
        self.gravity_dir *= -1
        # SFX: Play gravity flip sound

    def _find_and_mark_matches(self):
        to_match = set()
        # Horizontal
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS - 2):
                if self.grid[r][c].boson_type == self.grid[r][c+1].boson_type == self.grid[r][c+2].boson_type:
                    to_match.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS - 2):
                if self.grid[r][c].boson_type == self.grid[r+1][c].boson_type == self.grid[r+2][c].boson_type:
                    to_match.update([(r, c), (r+1, c), (r+2, c)])
        
        if not to_match:
            return False

        for r, c in to_match:
            self.grid[r][c].is_matched = True
        return True

    def _process_matches(self):
        num_matched = 0
        reward = 0
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] and self.grid[r][c].is_matched:
                    num_matched += 1
                    self.score += 1
                    self._create_particles(self.grid[r][c].current_pix_pos, self.grid[r][c].colors[self.grid[r][c].boson_type], 10)
                    self.grid[r][c] = None
        
        if num_matched > 0:
            reward += 0.1 * num_matched # Continuous feedback
            # SFX: Play match sound
        if num_matched >= 3:
            reward += 1.0
        if self.chain_count >= 2: # Chain reaction of 2+ separate clears
             reward += 5.0 * (self.chain_count - 1)
        
        return reward

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_cells = []
            # Gravity pulls bosons down (or up)
            if self.gravity_dir == 1: # Down
                for r in range(self.GRID_ROWS - 1, -1, -1):
                    if self.grid[r][c] is None:
                        empty_cells.append(r)
                    elif empty_cells:
                        dest_r = empty_cells.pop(0)
                        self.grid[dest_r][c] = self.grid[r][c]
                        self.grid[dest_r][c].grid_pos[0] = dest_r
                        self.grid[r][c] = None
                        empty_cells.append(r)
            else: # Up
                for r in range(self.GRID_ROWS):
                    if self.grid[r][c] is None:
                        empty_cells.append(r)
                    elif empty_cells:
                        dest_r = empty_cells.pop(0)
                        self.grid[dest_r][c] = self.grid[r][c]
                        self.grid[dest_r][c].grid_pos[0] = dest_r
                        self.grid[r][c] = None
                        empty_cells.append(r)

            # Refill empty cells from the top (or bottom)
            refill_row = 0 if self.gravity_dir == 1 else self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS):
                actual_r = r if self.gravity_dir == 1 else self.GRID_ROWS - 1 - r
                if self.grid[actual_r][c] is None:
                    self._create_boson(actual_r, c)
                    # Position new bosons off-screen to fall in
                    self.grid[actual_r][c].current_pix_pos[1] -= self.gravity_dir * self.CELL_SIZE

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        bg_color = self.COLOR_BG if self.gravity_dir == 1 else self.COLOR_BG_FLIPPED
        self.screen.fill(bg_color)
        for star in self.background_stars:
            pygame.draw.circle(self.screen, star[2], (int(star[0]), int(star[1])), star[3])
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_HEIGHT))

        # Draw bosons
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c]:
                    self.grid[r][c].draw(self.screen)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # Time
        time_sec = max(0, self.time_left // 60)
        time_text = self.font_ui.render(f"TIME: {time_sec}", True, (255, 255, 255))
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Target Score
        target_text = self.font_ui.render(f"TARGET: {self.target_score}", True, (200, 200, 200))
        self.screen.blit(target_text, (10, 35))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            if self.score >= self.target_score:
                msg = "LEVEL COMPLETE"
                color = (100, 255, 100)
            else:
                msg = "TIME UP"
                color = (255, 100, 100)
            msg_surf = self.font_msg.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH//2 - msg_surf.get_width()//2, self.HEIGHT//2 - msg_surf.get_height()//2))

    def _update_bosons_and_particles(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c]:
                    self.grid[r][c].update()
        
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifespan > 0]
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(20, 40)
            size = random.uniform(2, 5)
            self.particles.append(Particle(pos, vel, color, lifespan, size))

    def _generate_stars(self):
        self.background_stars = []
        for _ in range(100):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.choice([1, 1, 1, 2])
            brightness = random.randint(50, 150)
            color = (brightness, brightness, brightness)
            self.background_stars.append((x, y, color, size))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "target_score": self.target_score,
            "chain_count": self.chain_count,
            "unlocked_bosons": self.unlocked_boson_types
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # Manual play mode
    # For human play, we need a real display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Use a separate display for human play
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Boson Flip")
    clock = pygame.time.Clock()
    
    while running:
        # --- Event Handling ---
        action = [0, 0, 0] # [movement, space, shift]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # --- Game Step ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated:
            # Allow reset on key press
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # Convert observation back to a surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit to 60 FPS

    env.close()