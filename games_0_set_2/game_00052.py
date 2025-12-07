
# Generated: 2025-08-27T12:29:05.734575
# Source Brief: brief_00052.md
# Brief Index: 52

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Define a simple structure for pixels for clarity
Pixel = namedtuple("Pixel", ["color_idx", "value"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrows to move the cursor and aim. Press Space to push the selected row or column in the direction of your last movement."
    )

    game_description = (
        "A tricky puzzle game. Slide rows and columns to merge pixels and match the target grid before you run out of moves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.FONT_S = pygame.font.Font(None, 24)
        self.FONT_M = pygame.font.Font(None, 32)
        self.FONT_L = pygame.font.Font(None, 48)

        self._init_colors()

        self.level_configs = {
            1: {"size": 4, "moves": 20, "colors": 2, "shuffles": 5},
            2: {"size": 5, "moves": 30, "colors": 3, "shuffles": 10},
            3: {"size": 6, "moves": 40, "colors": 4, "shuffles": 15},
        }
        self.max_level = len(self.level_configs)
        
        self.reset()
        
        self.validate_implementation()

    def _init_colors(self):
        self.COLOR_BG = (25, 30, 45)
        self.COLOR_GRID_BG = (40, 45, 65)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_CURSOR_FILL = (255, 200, 0, 50)
        self.PIXEL_COLORS = [
            (230, 50, 50),   # Red
            (50, 200, 80),   # Green
            (60, 130, 255),  # Blue
            (240, 230, 80),  # Yellow
            (200, 90, 220),  # Purple
        ]
        self.PIXEL_TEXT_COLOR = (20, 20, 20)
        self.PARTICLE_COLORS = [(255, 255, 100), (255, 180, 50)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_complete = False
        self.win_message = ""
        
        self.level = 1
        self._setup_level()
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def _setup_level(self):
        config = self.level_configs[self.level]
        self.grid_size = config["size"]
        self.moves_left = config["moves"]
        self.max_moves = config["moves"]
        self.num_colors = config["colors"]
        
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]
        self.last_move_dir = 4 # Default to right

        # Generate target grid
        self.target_grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        num_pixels = self.np_random.integers(self.grid_size * self.grid_size // 2, self.grid_size * self.grid_size)
        for _ in range(num_pixels):
            r, c = self.np_random.integers(0, self.grid_size, size=2)
            if self.target_grid[r][c] is None:
                color = self.np_random.integers(0, self.num_colors)
                value = self.np_random.integers(1, 4)
                self.target_grid[r][c] = Pixel(color, value)

        # Create player grid by shuffling target
        self.player_grid = [row[:] for row in self.target_grid]
        for _ in range(config["shuffles"]):
            push_dir = self.np_random.integers(1, 5)
            index = self.np_random.integers(0, self.grid_size)
            self._perform_push(push_dir, index, reward_calc=False)

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.game_over = self._check_termination()

        if self.game_over:
            if self.level_complete and space_held and not self.last_space_held:
                if self.level < self.max_level:
                    self.level += 1
                    self._setup_level()
                    self.game_over = False
                    self.level_complete = False
                    self.win_message = ""
                else: # Final win
                    self.win_message = "YOU CLEARED ALL LEVELS!"
            return self._get_observation(), 0, True, False, self._get_info()
        
        # 1. Handle cursor movement (no move cost)
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            self.last_move_dir = 1
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.grid_size - 1, self.cursor_pos[0] + 1)
            self.last_move_dir = 2
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            self.last_move_dir = 3
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.grid_size - 1, self.cursor_pos[1] + 1)
            self.last_move_dir = 4

        # 2. Handle push action (costs a move)
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            mismatch_before = self._calculate_mismatch()
            
            # Use last_move_dir to determine push direction and axis
            push_dir = self.last_move_dir
            index = self.cursor_pos[0] if push_dir in [3, 4] else self.cursor_pos[1]
            
            merges = self._perform_push(push_dir, index)
            # sfx: push_sound

            mismatch_after = self._calculate_mismatch()
            
            reward += (mismatch_before - mismatch_after) * 1.0 # Progress reward
            reward += merges * 5.0 # Merge bonus
            
            self.score += reward
            self.moves_left -= 1

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.level_complete: # Only add terminal rewards once
            if self.moves_left <= 0:
                reward -= 50.0 # Loss penalty
                self.score -= 50.0
                self.win_message = "OUT OF MOVES!"
            else: # Must be a win
                reward += 100.0
                self.score += 100.0
                self.level_complete = True
                if self.level == self.max_level:
                    self.win_message = "FINAL LEVEL COMPLETE!"
                else:
                    self.win_message = f"LEVEL {self.level} COMPLETE! PRESS SPACE"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _perform_push(self, direction, index, reward_calc=True):
        merges = 0
        if direction in [1, 2]: # Up/Down (Column)
            line = [self.player_grid[r][index] for r in range(self.grid_size)]
        else: # Left/Right (Row)
            line = self.player_grid[index][:]

        reverse = direction in [2, 4] # Down or Right
        
        # 1. Filter out Nones
        pixels = [p for p in line if p is not None]
        if reverse:
            pixels.reverse()

        # 2. Merge
        i = 0
        while i < len(pixels) - 1:
            if pixels[i].color_idx == pixels[i+1].color_idx and pixels[i].value + pixels[i+1].value <= 9:
                pixels[i] = Pixel(pixels[i].color_idx, pixels[i].value + pixels[i+1].value)
                if reward_calc:
                    # sfx: merge_sound
                    self._create_particles(direction, index, i if not reverse else self.grid_size - 1 - i)
                del pixels[i+1]
                merges += 1
            i += 1
            
        # 3. Rebuild line
        if reverse:
            pixels.reverse()
        
        padding = [None] * (self.grid_size - len(pixels))
        
        if direction in [1, 3]: # Up or Left
            new_line = pixels + padding
        else: # Down or Right
            new_line = padding + pixels
            
        # 4. Update grid
        if direction in [1, 2]: # Column
            for r in range(self.grid_size):
                self.player_grid[r][index] = new_line[r]
        else: # Row
            self.player_grid[index] = new_line
            
        return merges

    def _calculate_mismatch(self):
        mismatch = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.player_grid[r][c] != self.target_grid[r][c]:
                    mismatch += 1
        return mismatch

    def _check_termination(self):
        if self.game_over:
            return True
        if self.moves_left <= 0:
            return True
        if self._calculate_mismatch() == 0:
            return True
        if self.steps >= 1000:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid dimensions and positions
        self.cell_size = 400 // (self.grid_size + 2)
        grid_pixel_size = self.cell_size * self.grid_size
        
        player_grid_x = (self.SCREEN_WIDTH // 2 - grid_pixel_size) // 2
        target_grid_x = self.SCREEN_WIDTH // 2 + player_grid_x
        grid_y = (self.SCREEN_HEIGHT - grid_pixel_size) // 2

        # Update and draw particles
        self._update_and_draw_particles(player_grid_x, grid_y)

        # Draw grids
        self._draw_grid("YOUR GRID", self.player_grid, player_grid_x, grid_y)
        self._draw_grid("TARGET", self.target_grid, target_grid_x, grid_y)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            player_grid_x + cursor_c * self.cell_size,
            grid_y + cursor_r * self.cell_size,
            self.cell_size, self.cell_size
        )
        # Blinking effect
        if (pygame.time.get_ticks() // 300) % 2 == 0:
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
            
            # Draw push direction indicator
            self._draw_push_indicator(cursor_rect)

    def _draw_grid(self, title, grid, start_x, start_y):
        title_surf = self.FONT_M.render(title, True, self.COLOR_UI_TEXT)
        self.screen.blit(title_surf, (start_x + (self.grid_size * self.cell_size - title_surf.get_width()) // 2, start_y - 35))

        grid_rect = pygame.Rect(start_x, start_y, self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                pixel = grid[r][c]
                rect = pygame.Rect(start_x + c * self.cell_size + 2, start_y + r * self.cell_size + 2, self.cell_size - 4, self.cell_size - 4)
                if pixel:
                    pygame.draw.rect(self.screen, self.PIXEL_COLORS[pixel.color_idx], rect, border_radius=4)
                    val_surf = self.FONT_S.render(str(pixel.value), True, self.PIXEL_TEXT_COLOR)
                    self.screen.blit(val_surf, val_surf.get_rect(center=rect.center))
    
    def _draw_push_indicator(self, cursor_rect):
        center = cursor_rect.center
        size = self.cell_size // 4
        points = []
        if self.last_move_dir == 1: # Up
            points = [(center[0], center[1] - size), (center[0] - size, center[1]), (center[0] + size, center[1])]
        elif self.last_move_dir == 2: # Down
            points = [(center[0], center[1] + size), (center[0] - size, center[1]), (center[0] + size, center[1])]
        elif self.last_move_dir == 3: # Left
            points = [(center[0] - size, center[1]), (center[0], center[1] - size), (center[0], center[1] + size)]
        elif self.last_move_dir == 4: # Right
            points = [(center[0] + size, center[1]), (center[0], center[1] - size), (center[0], center[1] + size)]
        
        if points:
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CURSOR)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CURSOR)


    def _render_ui(self):
        level_text = f"Level: {self.level}/{self.max_level}"
        moves_text = f"Moves: {self.moves_left}/{self.max_moves}"
        score_text = f"Score: {int(self.score)}"
        
        level_surf = self.FONT_M.render(level_text, True, self.COLOR_UI_TEXT)
        moves_surf = self.FONT_M.render(moves_text, True, self.COLOR_UI_TEXT)
        score_surf = self.FONT_M.render(score_text, True, self.COLOR_UI_TEXT)

        self.screen.blit(level_surf, (20, 15))
        self.screen.blit(moves_surf, (20, 45))
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 20, 15))

        if self.win_message:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_surf = self.FONT_L.render(self.win_message, True, self.COLOR_CURSOR)
            self.screen.blit(win_surf, win_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)))

    def _create_particles(self, direction, index, sub_index):
        grid_pixel_size = self.cell_size * self.grid_size
        start_x = (self.SCREEN_WIDTH // 2 - grid_pixel_size) // 2
        start_y = (self.SCREEN_HEIGHT - grid_pixel_size) // 2
        
        if direction in [1, 2]: # Column
            px, py = start_x + index * self.cell_size, start_y + sub_index * self.cell_size
        else: # Row
            px, py = start_x + sub_index * self.cell_size, start_y + index * self.cell_size
        
        center_x = px + self.cell_size // 2
        center_y = py + self.cell_size // 2

        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": vel,
                "lifetime": random.randint(15, 30),
                "color": random.choice(self.PARTICLE_COLORS),
                "size": random.uniform(2, 5)
            })

    def _update_and_draw_particles(self, grid_x, grid_y):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifetime"] -= 1
            p["size"] -= 0.1
            if p["lifetime"] <= 0 or p["size"] <= 0:
                self.particles.remove(p)
            else:
                pos = (int(p["pos"][0]), int(p["pos"][1]))
                pygame.draw.circle(self.screen, p["color"], pos, max(0, int(p["size"])))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
            "mismatch": self._calculate_mismatch(),
        }
    
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    print(env.user_guide)

    while not done:
        action = [0, 0, 0] # Default no-op
        
        # Simple human input mapping
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Check for quit event
        quit_event = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_event = True
                break
        if quit_event:
            break

        # Only step if an action is taken for this turn-based game
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Since auto_advance is False, we control the speed.
        # A small delay prevents the loop from running too fast on human input.
        pygame.time.wait(30) 

    env.close()