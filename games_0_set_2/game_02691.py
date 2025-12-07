
# Generated: 2025-08-28T05:38:20.576415
# Source Brief: brief_02691.md
# Brief Index: 2691

        
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
        "Controls: ↑ to move up, ↓ down, ← left, → right. Collect all green cells in 25 moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a robot through a grid-based maze to collect all energy cells before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        
        # Game parameters
        self.TOTAL_MOVES = 25
        self.NUM_CELLS = 10
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_ROBOT = (0, 150, 255)
        self.COLOR_CELL = (50, 255, 50)
        self.COLOR_CELL_GLOW = (50, 255, 50, 60) # RGBA
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_VICTORY = (100, 255, 100)
        self.COLOR_DEFEAT = (255, 100, 100)
        self.COLOR_OVERLAY = (0, 0, 0, 150) # RGBA

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24)
        self.font_game_over = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # State variables (initialized in reset)
        self.robot_pos = None
        self.energy_cells = None
        self.moves_remaining = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_outcome = ""
        
        # Initialize state
        # self.reset() is called by the wrapper or user, but we can call it once
        # to ensure all variables are populated for validation.
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.moves_remaining = self.TOTAL_MOVES
        
        # Generate unique positions for robot and cells
        all_positions = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_positions)
        
        self.robot_pos = all_positions.pop()
        self.energy_cells = [all_positions.pop() for _ in range(self.NUM_CELLS)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # If game is over, do not process new actions, just return the final state
        if self.game_over:
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )

        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        
        # --- Process Movement ---
        if movement != 0: # 0 is no-op
            self.moves_remaining -= 1
            reward = -0.1 # -0.1 per move
            # // Move sound effect
            
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1  # Right
            
            new_x = self.robot_pos[0] + dx
            new_y = self.robot_pos[1] + dy

            # Check boundaries
            if 0 <= new_x < self.GRID_W and 0 <= new_y < self.GRID_H:
                self.robot_pos = (new_x, new_y)
        
        # --- Check for Energy Cell Collection ---
        if self.robot_pos in self.energy_cells:
            self.energy_cells.remove(self.robot_pos)
            self.score += 1
            reward += 5 # +5 for collecting a cell
            # // Collect sound effect
        
        self.steps += 1
        terminated = self._check_termination()

        # --- Calculate Terminal Rewards ---
        if terminated:
            if self.game_outcome == "VICTORY!":
                reward += 50
                # // Victory sound effect
            else: # "OUT OF MOVES"
                reward -= 50
                # // Failure sound effect

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        if self.score >= self.NUM_CELLS:
            self.game_over = True
            self.game_outcome = "VICTORY!"
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            self.game_outcome = "OUT OF MOVES"
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw energy cells with glow
        for cell_pos in self.energy_cells:
            cx = int(cell_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
            cy = int(cell_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            radius = int(self.CELL_SIZE * 0.25)
            glow_radius = int(radius * 1.8)

            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_CELL_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (cx - glow_radius, cy - glow_radius))

            pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, self.COLOR_CELL)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, self.COLOR_CELL)

        # Draw robot
        robot_rect = pygame.Rect(
            self.robot_pos[0] * self.CELL_SIZE,
            self.robot_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        # Inflate to leave a margin, making it look like a smaller square in the cell
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, robot_rect.inflate(-8, -8))

    def _render_ui(self):
        moves_text = f"Moves: {self.moves_remaining}"
        score_text = f"Cells: {self.score} / {self.NUM_CELLS}"
        
        moves_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(moves_surf, (10, 10))
        self.screen.blit(score_surf, (150, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            outcome_color = self.COLOR_VICTORY if self.game_outcome == "VICTORY!" else self.COLOR_DEFEAT
            
            text_surf = self.font_game_over.render(self.game_outcome, True, outcome_color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid-Bot Maze")
    clock = pygame.time.Clock()
    
    print("\n" + env.user_guide)
    print("Press 'R' to reset, 'ESC' to quit.")
    
    running = True
    while running:
        movement_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("\n--- Game Reset ---")
                if event.key == pygame.K_ESCAPE:
                    running = False

        if movement_action != 0 and not env.game_over:
            action = [movement_action, 0, 0] # Space/Shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_remaining']}, Terminated: {terminated}")
            
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()