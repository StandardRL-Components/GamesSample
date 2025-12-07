
# Generated: 2025-08-27T14:36:25.622113
# Source Brief: brief_00736.md
# Brief Index: 736

        
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

    user_guide = (
        "Arrows: Move cursor. Hold Shift + ↑/↓: Select color. Space: Fill square."
    )

    game_description = (
        "Reveal the hidden pixel art by filling the grid. Manage your limited color supply!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = self.GRID_SIZE * self.GRID_SIZE * 2 # Generous step limit
        self.WIN_THRESHOLD = 0.9
        
        # Colors (10 main colors + 1 empty color)
        self.COLORS = [
            pygame.Color("#ff6666"), pygame.Color("#ffb366"), pygame.Color("#ffff66"),
            pygame.Color("#b3ff66"), pygame.Color("#66ff66"), pygame.Color("#66ffb3"),
            pygame.Color("#66ffff"), pygame.Color("#66b3ff"), pygame.Color("#b366ff"),
            pygame.Color("#ff66ff"), pygame.Color("#404050") # 11th color is for empty cells
        ]
        self.COLOR_BG = pygame.Color("#202028")
        self.COLOR_GRID_LINES = pygame.Color("#303040")
        self.COLOR_CURSOR = pygame.Color("#FFFFFF")
        self.COLOR_TEXT = pygame.Color("#E0E0E0")
        self.COLOR_WIN = pygame.Color("#77FF77")
        self.COLOR_LOSE = pygame.Color("#FF7777")

        # Layout
        self.CELL_SIZE = 32
        self.GRID_OFFSET_X = 40
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.UI_X = self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE + 40
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Game State (initialized in reset)
        self.grid = None
        self.target_image = None
        self.color_counts = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_accuracy_milestone = 0
        self.particles = []
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def _generate_target_image(self):
        image = np.full((self.GRID_SIZE, self.GRID_SIZE), 10, dtype=int) # Fill with empty
        pattern_type = self.np_random.integers(0, 3)

        if pattern_type == 0: # Smiley Face
            face_color = self.np_random.integers(0, 5)
            eye_color = self.np_random.integers(5, 10)
            image[2:8, 2:8] = face_color
            image[2, 2] = image[2, 7] = image[7, 2] = image[7, 7] = 10
            image[3, 4] = image[3, 6] = eye_color
            image[5, 3:7] = eye_color
            image[6, 3] = image[6, 6] = 10
        elif pattern_type == 1: # Heart
            heart_color = self.np_random.integers(0, 10)
            image[3, 2] = image[3, 3] = image[3, 6] = image[3, 7] = heart_color
            image[2, 4] = image[2, 5] = heart_color
            image[4:8, 1:9] = heart_color
            image[8, 2:8] = heart_color
            image[9, 3:7] = heart_color
        else: # Spaceship
            ship_color = self.np_random.integers(0, 5)
            wing_color = self.np_random.integers(0, 5)
            flame_color = self.np_random.integers(5, 10)
            image[4, 2] = image[4, 7] = wing_color
            image[5, 1:9] = wing_color
            image[2:5, 4:6] = ship_color
            image[5, 3:7] = ship_color
            image[6, 4:6] = flame_color
            image[7, 4] = image[7, 5] = self.np_random.integers(5, 10)

        return image

    def _calculate_initial_counts(self):
        counts = np.zeros(len(self.COLORS) - 1, dtype=int)
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.target_image[r, c]
                if color_idx < len(counts):
                    counts[color_idx] += 1
        return counts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_accuracy_milestone = 0
        self.particles = []

        self.target_image = self._generate_target_image()
        self.color_counts = self._calculate_initial_counts()
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), 10, dtype=int)
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_color_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        movement, space_press, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input ---
        if shift_held: # Color selection mode
            if movement == 1: # Up
                self.selected_color_idx = (self.selected_color_idx - 1) % (len(self.COLORS) - 1)
            elif movement == 2: # Down
                self.selected_color_idx = (self.selected_color_idx + 1) % (len(self.COLORS) - 1)
        else: # Cursor movement mode
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

        # --- Handle Action ---
        if space_press:
            r, c = self.cursor_pos
            if self.grid[r, c] == 10 and self.color_counts[self.selected_color_idx] > 0:
                # Place color
                self.grid[r, c] = self.selected_color_idx
                self.color_counts[self.selected_color_idx] -= 1
                # SFX: Place tile
                
                self._create_particles(c, r, self.COLORS[self.selected_color_idx])

                # Calculate placement reward
                if self.grid[r, c] == self.target_image[r, c]:
                    reward += 1
                else:
                    reward -= 1
            # else: SFX: Error/Buzz
        
        self.score += reward
        self._update_particles()
        self.steps += 1

        # --- Check for Milestone Rewards ---
        correct_cells = np.sum(self.grid == self.target_image)
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        accuracy = correct_cells / total_cells if total_cells > 0 else 0
        
        current_milestone = int(accuracy * 10) # Each 10% is a milestone
        if current_milestone > self.last_accuracy_milestone:
            milestone_reward = (current_milestone - self.last_accuracy_milestone) * 5
            reward += milestone_reward
            self.score += milestone_reward
            self.last_accuracy_milestone = current_milestone

        # --- Check Termination Conditions ---
        terminated = False
        if accuracy >= self.WIN_THRESHOLD:
            reward += 50
            self.score += 50
            terminated = True
            self.win = True
        elif np.any(self.color_counts < 0) or (self.color_counts[self.selected_color_idx] == 0 and np.sum(self.grid == 10) > 0 and self._is_color_needed(self.selected_color_idx)):
             reward -= 50
             self.score -= 50
             terminated = True
             self.win = False
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.win = False

        if terminated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_color_needed(self, color_idx):
        """Check if a color is still needed to complete the image."""
        needed_mask = (self.target_image == color_idx)
        unfilled_mask = (self.grid == 10)
        return np.any(needed_mask & unfilled_mask)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_target_preview()
        self._render_grid()
        self._render_cursor()
        self._render_particles()

    def _render_ui(self):
        self._render_palette()
        self._render_text_info()
        if self.game_over:
            self._render_game_over()

    def _render_grid(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r, c]
                color = self.COLORS[color_idx]
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

    def _render_cursor(self):
        r, c = self.cursor_pos
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        thickness = 2 + int(pulse * 2)
        rect = pygame.Rect(
            self.GRID_OFFSET_X + c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, thickness)

    def _render_palette(self):
        bar_h = 28
        bar_w = 150
        for i in range(len(self.COLORS) - 1):
            y = self.GRID_OFFSET_Y + i * (bar_h + 5)
            
            # Draw background
            bg_rect = pygame.Rect(self.UI_X, y, bar_w, bar_h)
            pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, bg_rect)

            # Draw fill
            initial_count = np.sum(self.target_image == i)
            fill_ratio = self.color_counts[i] / initial_count if initial_count > 0 else 0
            fill_w = int(bar_w * fill_ratio)
            fill_rect = pygame.Rect(self.UI_X, y, fill_w, bar_h)
            pygame.draw.rect(self.screen, self.COLORS[i], fill_rect)

            # Draw border and highlight
            border_color = self.COLOR_CURSOR if i == self.selected_color_idx else self.COLOR_TEXT
            border_width = 2 if i == self.selected_color_idx else 1
            pygame.draw.rect(self.screen, border_color, bg_rect, border_width)

    def _render_text_info(self):
        correct_cells = np.sum(self.grid == self.target_image)
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        accuracy = (correct_cells / total_cells) * 100 if total_cells > 0 else 0

        info_texts = [
            f"Score: {self.score}",
            f"Steps: {self.steps}/{self.MAX_STEPS}",
            f"Accuracy: {accuracy:.1f}%"
        ]
        for i, text in enumerate(info_texts):
            surf = self.font_main.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surf, (self.UI_X, self.HEIGHT - 90 + i * 30))

    def _render_target_preview(self):
        preview_size = 10
        offset_x = self.UI_X + 25
        offset_y = 20
        
        # Obscure based on accuracy
        correct_cells = np.sum(self.grid == self.target_image)
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        accuracy = correct_cells / total_cells if total_cells > 0 else 0
        reveal_count = int(total_cells * accuracy)
        
        revealed_indices = self.np_random.permutation(total_cells)[:reveal_count]

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                idx_1d = r * self.GRID_SIZE + c
                rect = pygame.Rect(offset_x + c * preview_size, offset_y + r * preview_size, preview_size, preview_size)
                if idx_1d in revealed_indices:
                    color = self.COLORS[self.target_image[r, c]]
                else:
                    color = self.COLOR_GRID_LINES
                pygame.draw.rect(self.screen, color, rect)
        
        preview_border = pygame.Rect(offset_x, offset_y, self.GRID_SIZE * preview_size, self.GRID_SIZE * preview_size)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, preview_border, 1)
        
        text_surf = self.font_main.render("Target", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (offset_x + (preview_border.width - text_surf.get_width()) // 2, offset_y - 20))

    def _create_particles(self, c, r, color):
        px = self.GRID_OFFSET_X + (c + 0.5) * self.CELL_SIZE
        py = self.GRID_OFFSET_Y + (r + 0.5) * self.CELL_SIZE
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(10, 20)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
        self.particles = [p for p in self.particles if p[4] > 0]

    def _render_particles(self):
        for p in self.particles:
            x, y, _, _, lifetime, color = p
            size = max(0, int((lifetime / 20) * 4))
            pygame.draw.rect(self.screen, color, (int(x - size/2), int(y - size/2), size, size))
    
    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_WIN if self.win else self.COLOR_LOSE
        
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        correct_cells = np.sum(self.grid == self.target_image)
        total_cells = self.GRID_SIZE * self.GRID_SIZE
        accuracy = correct_cells / total_cells if total_cells > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": accuracy,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    pygame.display.set_caption("Pixel Art Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Accuracy: {info['accuracy']:.2f}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()

        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()