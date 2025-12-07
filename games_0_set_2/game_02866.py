
# Generated: 2025-08-27T21:39:42.699053
# Source Brief: brief_02866.md
# Brief Index: 2866

        
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
        "Controls: ↑↓←→ to move the selector. Press Space to change the color of the selected tile group."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Change the color of connected tile groups to make the entire board a single color before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame and Display ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Constants ---
        self.GRID_SIZE = 8
        self.MAX_MOVES = 20
        self.BOARD_AREA_WIDTH = 360
        self.TILE_SIZE = self.BOARD_AREA_WIDTH // self.GRID_SIZE
        self.BOARD_OFFSET_X = (self.SCREEN_WIDTH - self.BOARD_AREA_WIDTH) // 2
        self.BOARD_OFFSET_Y = (self.SCREEN_HEIGHT - self.BOARD_AREA_WIDTH) // 2
        self.ANIMATION_DELAY_PER_STEP = 2 # Frames of delay for each step in the wave
        self.ANIMATION_DURATION = 15 # Frames for color to transition

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINES = (40, 60, 80)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLORS = [
            pygame.Color(255, 80, 80),   # Red
            pygame.Color(80, 255, 80),   # Green
            pygame.Color(80, 120, 255),  # Blue
            pygame.Color(255, 220, 80),  # Yellow
        ]
        
        # --- Game State ---
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self.moves_left = 0
        self.selector_pos = np.array([0, 0])
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.last_space_held = False
        self.animations = {} # Dict mapping (r,c) to animation data
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self.np_random.integers(0, len(self.COLORS), size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        self.moves_left = self.MAX_MOVES
        self.selector_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.last_space_held = False
        self.animations.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        self._update_animations()

        movement = action[0]
        space_held = action[1] == 1
        
        if not self.game_over:
            # Handle movement
            if movement == 1: self.selector_pos[1] -= 1 # Up
            elif movement == 2: self.selector_pos[1] += 1 # Down
            elif movement == 3: self.selector_pos[0] -= 1 # Left
            elif movement == 4: self.selector_pos[0] += 1 # Right
            self.selector_pos = np.clip(self.selector_pos, 0, self.GRID_SIZE - 1)

            # Handle action (space press on rising edge)
            if space_held and not self.last_space_held:
                # sound_effect: "click.wav"
                self.moves_left -= 1
                self._trigger_color_change()
                
                # Check for termination and calculate rewards post-move
                is_win = np.all(self.grid == self.grid[0, 0])
                is_loss = self.moves_left <= 0

                if is_win:
                    self.game_over = True
                    self.win_message = "YOU WIN!"
                    reward = 50.0
                    # sound_effect: "win_fanfare.wav"
                elif is_loss:
                    self.game_over = True
                    self.win_message = "GAME OVER"
                    reward = -10.0
                    # sound_effect: "loss_buzzer.wav"
                else:
                    # Continuous reward for progress
                    target_color = self.grid[0, 0]
                    reward = float(np.sum(self.grid == target_color))
        
        self.last_space_held = space_held
        terminated = self.game_over
        
        # Update score with the reward from this step
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _trigger_color_change(self):
        start_pos = tuple(self.selector_pos)
        target_color_idx = self.grid[start_pos]
        new_color_idx = (target_color_idx + 1) % len(self.COLORS)
        
        # Use BFS to find all connected tiles and their distance for the wave effect
        q = [(start_pos, 0)] # (pos, depth)
        visited = {start_pos}
        
        while q:
            (r, c), depth = q.pop(0)
            
            # Update grid model immediately
            old_color_idx = self.grid[r, c]
            self.grid[r, c] = new_color_idx
            
            # Create animation object
            self.animations[(r, c)] = {
                "start_color": self.COLORS[old_color_idx],
                "end_color": self.COLORS[new_color_idx],
                "delay": depth * self.ANIMATION_DELAY_PER_STEP,
                "progress": 0.0,
            }

            # Check neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                   (nr, nc) not in visited and self.grid[nr, nc] == target_color_idx:
                    visited.add((nr, nc))
                    q.append(((nr, nc), depth + 1))

    def _update_animations(self):
        finished_anims = []
        for pos, anim in self.animations.items():
            if anim["delay"] > 0:
                anim["delay"] -= 1
            elif anim["progress"] < 1.0:
                anim["progress"] += 1.0 / self.ANIMATION_DURATION
            
            if anim["progress"] >= 1.0:
                finished_anims.append(pos)
        
        for pos in finished_anims:
            del self.animations[pos]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid and tiles
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile_rect = pygame.Rect(
                    self.BOARD_OFFSET_X + c * self.TILE_SIZE,
                    self.BOARD_OFFSET_Y + r * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )
                
                color = self.COLORS[self.grid[r, c]]
                anim = self.animations.get((r, c))
                if anim and anim['delay'] == 0:
                    # Interpolate color if animating
                    progress = min(1.0, anim['progress'])
                    color = color.lerp(anim['end_color'], progress)

                pygame.draw.rect(self.screen, color, tile_rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, tile_rect, 1)

        # Draw selector
        selector_x = self.BOARD_OFFSET_X + self.selector_pos[0] * self.TILE_SIZE
        selector_y = self.BOARD_OFFSET_Y + self.selector_pos[1] * self.TILE_SIZE
        selector_rect = pygame.Rect(selector_x, selector_y, self.TILE_SIZE, self.TILE_SIZE)
        
        # Pulsing effect for selector
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, line_width)

    def _render_ui(self):
        # Display moves left
        moves_text_surf = self.font_main.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text_surf, (20, 20))

        # Display score
        score_text_surf = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text_surf, score_rect)

        # Display game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            
            win_surf = self.font_title.render(self.win_message, True, self.COLOR_SELECTOR)
            win_rect = win_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            overlay.blit(win_surf, win_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for visualization
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Color Flood Puzzle")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Unused in this game
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing a reset
            pygame.time.wait(1000)

        clock.tick(30) # Limit to 30 FPS for human play

    env.close()