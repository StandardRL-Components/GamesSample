
# Generated: 2025-08-28T06:13:24.178605
# Source Brief: brief_02867.md
# Brief Index: 2867

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to 'click' the selected cell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect all 20 gems before time runs out. Avoid the traps! Each move and click counts."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.NUM_GEMS = 20
        self.NUM_TRAPS = 3
        self.MAX_TIME = 60.0  # seconds
        self.FPS = 30 # For time calculation

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TRAP = (120, 30, 30)
        self.COLOR_TRAP_ACTIVE = (255, 50, 50)
        self.GEM_COLORS = [
            (50, 150, 255), (50, 255, 150), (255, 255, 100),
            (255, 100, 255), (100, 255, 255)
        ]

        # Grid layout calculation
        self.GRID_AREA_HEIGHT = self.SCREEN_HEIGHT - 60
        self.CELL_SIZE = min(self.SCREEN_WIDTH // self.GRID_COLS, self.GRID_AREA_HEIGHT // self.GRID_ROWS)
        self.GRID_WIDTH = self.CELL_SIZE * self.GRID_COLS
        self.GRID_HEIGHT = self.CELL_SIZE * self.GRID_ROWS
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2 + 30

        # Initialize state variables
        self.grid = []
        self.gems = {}
        self.traps = set()
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.time_remaining = 0.0
        self.game_over = False
        self.win = False
        self.last_space_press = False
        self.particles = []
        self.triggered_trap_pos = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.win = False
        self.last_space_press = False
        self.particles.clear()
        self.triggered_trap_pos = None

        # Reset cursor
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.visual_cursor_pos = self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1])

        # Generate grid layout
        self.gems.clear()
        self.traps.clear()
        all_positions = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        self.np_random.shuffle(all_positions)

        gem_positions = all_positions[:self.NUM_GEMS]
        trap_positions = all_positions[self.NUM_GEMS:self.NUM_GEMS + self.NUM_TRAPS]

        for i, pos in enumerate(gem_positions):
            self.gems[pos] = {
                "color": self.np_random.choice(self.GEM_COLORS),
                "type": self.np_random.integers(0, 3) # 0:circle, 1:rect, 2:diamond
            }
        for pos in trap_positions:
            self.traps.add(pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            # shift_held is unused per brief

            # Update time
            self.time_remaining -= 1.0 / self.FPS
            
            # --- Handle Input ---
            # 1. Movement
            prev_cursor_pos = list(self.cursor_pos)
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            
            # Clamp cursor to grid
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

            # 2. Click (Spacebar) - on press, not hold
            clicked = space_held and not self.last_space_press
            if clicked:
                pos_tuple = tuple(self.cursor_pos)
                if pos_tuple in self.gems:
                    # Collect gem
                    reward += 1.0
                    self.score += 1
                    self.gems_collected += 1
                    self._create_particles(pos_tuple, self.gems[pos_tuple]["color"], 20)
                    del self.gems[pos_tuple]
                    # Play gem collect sound
                elif pos_tuple in self.traps:
                    # Hit trap
                    reward -= 10.0
                    self.game_over = True
                    self.triggered_trap_pos = pos_tuple
                    self._create_particles(pos_tuple, self.COLOR_TRAP_ACTIVE, 50)
                    # Play trap sound
                else:
                    # Clicked empty cell
                    reward -= 0.1
                    # Play empty click sound

            self.last_space_press = space_held

            # --- Check Termination Conditions ---
            if self.gems_collected >= self.NUM_GEMS:
                self.win = True
                self.game_over = True
                reward += 100.0
                # Play win sound
            
            if self.time_remaining <= 0:
                self.game_over = True
                # Play time up sound

        # Update steps and particles
        self.steps += 1
        self._update_particles()
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_grid()
        self._render_gems_and_traps()
        self._render_particles()
        self._render_cursor()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "gems_collected": self.gems_collected,
        }

    def _get_pixel_pos(self, grid_x, grid_y, center=False):
        px = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return [px, py]

    def _render_grid(self):
        for x in range(self.GRID_COLS + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_HEIGHT))
        for y in range(self.GRID_ROWS + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_WIDTH, py))

    def _render_gems_and_traps(self):
        # Render gems
        for pos, gem_data in self.gems.items():
            center_px = self._get_pixel_pos(pos[0], pos[1], center=True)
            radius = int(self.CELL_SIZE * 0.3)
            gem_type = gem_data["type"]
            color = gem_data["color"]
            
            if gem_type == 0: # Circle
                pygame.gfxdraw.filled_circle(self.screen, center_px[0], center_px[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, center_px[0], center_px[1], radius, color)
            elif gem_type == 1: # Square
                rect = pygame.Rect(center_px[0] - radius, center_px[1] - radius, radius * 2, radius * 2)
                pygame.draw.rect(self.screen, color, rect)
            else: # Diamond
                points = [
                    (center_px[0], center_px[1] - radius),
                    (center_px[0] + radius, center_px[1]),
                    (center_px[0], center_px[1] + radius),
                    (center_px[0] - radius, center_px[1]),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Render traps
        for pos in self.traps:
            center_px = self._get_pixel_pos(pos[0], pos[1], center=True)
            size = int(self.CELL_SIZE * 0.3)
            color = self.COLOR_TRAP_ACTIVE if pos == self.triggered_trap_pos else self.COLOR_TRAP
            
            # Draw a thick 'X'
            pygame.draw.line(self.screen, color, (center_px[0] - size, center_px[1] - size), (center_px[0] + size, center_px[1] + size), 3)
            pygame.draw.line(self.screen, color, (center_px[0] - size, center_px[1] + size), (center_px[0] + size, center_px[1] - size), 3)

    def _render_cursor(self):
        target_px = self._get_pixel_pos(self.cursor_pos[0], self.cursor_pos[1])
        
        # Smooth interpolation for visual cursor
        lerp_factor = 0.4
        self.visual_cursor_pos[0] = (1 - lerp_factor) * self.visual_cursor_pos[0] + lerp_factor * target_px[0]
        self.visual_cursor_pos[1] = (1 - lerp_factor) * self.visual_cursor_pos[1] + lerp_factor * target_px[1]

        cursor_rect = pygame.Rect(int(self.visual_cursor_pos[0]), int(self.visual_cursor_pos[1]), self.CELL_SIZE, self.CELL_SIZE)
        
        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill((*self.COLOR_CURSOR, 60)) # Yellow, semi-transparent
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), 3) # Border
        self.screen.blit(s, cursor_rect.topleft)

    def _render_ui(self):
        # Score and Gems
        score_text = f"SCORE: {self.score}"
        gems_text = f"GEMS: {self.gems_collected} / {self.NUM_GEMS}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        gems_surf = self.font_small.render(gems_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(gems_surf, (150, 10))

        # Timer
        time_str = f"TIME: {max(0, self.time_remaining):.1f}"
        time_color = self.COLOR_TRAP_ACTIVE if self.time_remaining < 10 else self.COLOR_TEXT
        time_surf = self.font_small.render(time_str, True, time_color)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else self.COLOR_TRAP_ACTIVE
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a semi-transparent background for the text
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, grid_pos, color, count):
        center_px = self._get_pixel_pos(grid_pos[0], grid_pos[1], center=True)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.uniform(15, 30) # frames
            self.particles.append({
                "pos": list(center_px),
                "vel": velocity,
                "lifetime": lifetime,
                "max_life": lifetime,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # gravity
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_life"]))
            color = (*p["color"], alpha)
            size = int(3 * (p["lifetime"] / p["max_life"]))
            if size > 0:
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (int(p["pos"][0] - size), int(p["pos"][1] - size)))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment visually
if __name__ == '__main__':
    env = GameEnv()
    
    # --- For manual play ---
    # To control, focus on the Pygame window.
    # Arrow keys to move, Space to click.
    
    obs, info = env.reset()
    done = False
    running = True
    
    # Create a display window
    pygame.display.set_caption("Gem Collector")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    while running:
        # Action defaults to NO-OP
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the env's internal screen to the display screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        env.clock.tick(env.FPS)

    env.close()