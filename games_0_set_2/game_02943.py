
# Generated: 2025-08-28T06:27:51.780177
# Source Brief: brief_02943.md
# Brief Index: 2943

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move the selector. Press space to rotate the selected tile and its neighbors."
    )

    game_description = (
        "An isometric puzzle game. Rotate tiles to make the entire 5x5 grid a single color. "
        "Each rotation costs one move. You have a limited number of moves per stage. "
        "There are three stages of increasing difficulty."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 5
    MOVES_PER_STAGE = 20
    MAX_STEPS = 30 * 90  # 90 seconds total at 30 FPS

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SELECTOR = (255, 255, 100)
    
    TILE_COLORS = [
        (227, 85, 85),   # Red
        (85, 190, 227),  # Blue
        (85, 227, 131),  # Green
        (227, 213, 85),  # Yellow
    ]
    TILE_OUTLINE_COLORS = [
        (180, 50, 50),
        (50, 150, 180),
        (50, 180, 90),
        (180, 170, 50),
    ]

    # --- Isometric Projection ---
    TILE_WIDTH = 64
    TILE_HEIGHT = 32
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100

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
        
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.current_stage = None
        self.moves_left = None
        self.target_colors = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win = None
        
        # Input handling for turn-based logic with auto_advance
        self.space_was_pressed = False
        self.move_cooldown = 0
        
        # Animation
        self.animation_queue = []
        self.ANIMATION_DURATION = 10 # frames

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_stage = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        
        # Determine target colors for all 3 stages at the start
        self.target_colors = self.np_random.integers(0, len(self.TILE_COLORS), size=3).tolist()

        self.space_was_pressed = False
        self.move_cooldown = 0
        self.animation_queue = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.moves_left = self.MOVES_PER_STAGE
        target_color = self.target_colors[self.current_stage - 1]
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), target_color, dtype=int)
        
        mismatches_by_stage = [5, 10, 15]
        num_mismatches = mismatches_by_stage[self.current_stage - 1]
        
        possible_coords = [(r, c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE)]
        mismatch_coords = self.np_random.choice(len(possible_coords), num_mismatches, replace=False)
        
        for idx in mismatch_coords:
            r, c = possible_coords[idx]
            # Ensure the new color is not the target color
            possible_new_colors = [i for i in range(len(self.TILE_COLORS)) if i != target_color]
            new_color = self.np_random.choice(possible_new_colors)
            self.grid[r, c] = new_color

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        
        movement, space_held, _ = action
        
        self._handle_input(movement, space_held)
        self._update_animations()

        # Only check for win/loss conditions when no animations are playing
        if not self.animation_queue:
            is_solved = np.all(self.grid == self.target_colors[self.current_stage - 1])
            
            if is_solved:
                reward += 5.0
                self.score += 50
                if self.current_stage == 3:
                    self.win = True
                    self.game_over = True
                    reward += 100.0
                else:
                    self.current_stage += 1
                    self._setup_stage()
            
            elif self.moves_left <= 0:
                self.game_over = True
                reward -= 100.0

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            if not self.win: # Penalize for timeout
                reward -= 50.0

        # Continuous reward
        target_color = self.target_colors[self.current_stage - 1]
        matches = np.sum(self.grid == target_color)
        mismatches = (self.GRID_SIZE * self.GRID_SIZE) - matches
        reward += (matches * 0.1) - (mismatches * 0.02)
        
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # --- Handle Cursor Movement ---
        self.move_cooldown = max(0, self.move_cooldown - 1)
        if movement != 0 and self.move_cooldown == 0:
            r, c = self.cursor_pos
            if movement == 1: r -= 1  # Up
            elif movement == 2: r += 1  # Down
            elif movement == 3: c -= 1  # Left
            elif movement == 4: c += 1  # Right
            
            self.cursor_pos = (r % self.GRID_SIZE, c % self.GRID_SIZE)
            self.move_cooldown = 5 # 5-frame cooldown

        # --- Handle Rotation Action ---
        if space_held and not self.space_was_pressed and self.moves_left > 0 and not self.animation_queue:
            # sound placeholder: self.play_sound("rotate")
            self.moves_left -= 1
            r, c = self.cursor_pos
            
            tiles_to_rotate = [(r, c)]
            if r > 0: tiles_to_rotate.append((r - 1, c))
            if r < self.GRID_SIZE - 1: tiles_to_rotate.append((r + 1, c))
            if c > 0: tiles_to_rotate.append((r, c - 1))
            if c < self.GRID_SIZE - 1: tiles_to_rotate.append((r, c + 1))
            
            for tr, tc in tiles_to_rotate:
                self.grid[tr, tc] = (self.grid[tr, tc] + 1) % len(self.TILE_COLORS)
                self.animation_queue.append([tr, tc, self.ANIMATION_DURATION])

        self.space_was_pressed = space_held

    def _update_animations(self):
        if not self.animation_queue:
            return
        
        new_queue = []
        for anim in self.animation_queue:
            anim[2] -= 1 # Decrement timer
            if anim[2] > 0:
                new_queue.append(anim)
        self.animation_queue = new_queue

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
            "stage": self.current_stage,
            "moves_left": self.moves_left,
        }

    def _to_iso(self, r, c):
        px = self.ORIGIN_X + (c - r) * (self.TILE_WIDTH / 2)
        py = self.ORIGIN_Y + (c + r) * (self.TILE_HEIGHT / 2)
        return int(px), int(py)

    def _render_game(self):
        # Create a dictionary of animations for quick lookup
        anim_dict = {(r, c): timer for r, c, timer in self.animation_queue}

        # Draw tiles from back to front
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                px, py = self._to_iso(r, c)
                color_index = self.grid[r, c]
                
                scale = 1.0
                if (r, c) in anim_dict:
                    timer = anim_dict[(r, c)]
                    progress = (self.ANIMATION_DURATION - timer) / self.ANIMATION_DURATION
                    scale = 1.0 + 0.2 * math.sin(progress * math.pi)

                scaled_w = self.TILE_WIDTH * scale
                scaled_h = self.TILE_HEIGHT * scale

                points = [
                    (px, py - scaled_h / 2),
                    (px + scaled_w / 2, py),
                    (px, py + scaled_h / 2),
                    (px - scaled_w / 2, py),
                ]

                # Draw filled tile and anti-aliased outline
                pygame.gfxdraw.filled_polygon(self.screen, points, self.TILE_COLORS[color_index])
                pygame.gfxdraw.aapolygon(self.screen, points, self.TILE_OUTLINE_COLORS[color_index])

        # Draw selector on top
        self._render_selector()
        
        # Draw game over/win text
        if self.game_over:
            text = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 150, 150)
            text_surf = self.font_game_over.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _render_selector(self):
        r, c = self.cursor_pos
        px, py = self._to_iso(r, c)

        # Pulsing effect for the selector
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        alpha = int(150 + 105 * pulse)
        
        # Draw a slightly larger outline for the selector
        selector_w = self.TILE_WIDTH * 1.1
        selector_h = self.TILE_HEIGHT * 1.1
        
        points = [
            (px, py - selector_h / 2),
            (px + selector_w / 2, py),
            (px, py + selector_h / 2),
            (px - selector_w / 2, py),
        ]
        
        # Draw multiple polygons for thickness
        for i in range(3):
            offset_points = [(p[0], p[1] - i) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, offset_points, (*self.COLOR_SELECTOR, alpha))
            offset_points = [(p[0], p[1] + i) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, offset_points, (*self.COLOR_SELECTOR, alpha))

    def _render_ui(self):
        # --- Stage Display (Top-Left) ---
        stage_text = f"Stage: {self.current_stage}"
        stage_surf = self.font_ui.render(stage_text, True, self.COLOR_UI_TEXT)
        stage_rect = stage_surf.get_rect(topleft=(15, 15))

        # Border indicating target color
        border_rect = stage_rect.inflate(10, 10)
        target_color_idx = self.target_colors[self.current_stage - 1]
        pygame.draw.rect(self.screen, self.TILE_COLORS[target_color_idx], border_rect, border_radius=5)
        
        self.screen.blit(stage_surf, stage_rect)

        # --- Moves Display (Top-Right) ---
        moves_text = f"Moves: {self.moves_left}"
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_UI_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(moves_surf, moves_rect)

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This part is for human play and visualization
    # It will not be part of the final submission for the library
    import sys

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("ChromaShift")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Optional: auto-reset after a pause
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already the rendered screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()
    sys.exit()