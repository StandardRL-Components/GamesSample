
# Generated: 2025-08-28T04:26:15.060867
# Source Brief: brief_05250.md
# Brief Index: 5250

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Collect all 7 white herbs to win. Avoid the red creatures."
    )

    # Short, user-facing description of the game
    game_description = (
        "Navigate a cursed forest grid, gathering herbs to lift the curse "
        "while evading deadly creatures."
    )

    # Frames only advance when an action is received
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.CELL_SIZE = 40
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        self.MAX_STEPS = 1000
        self.HERBS_TO_COLLECT = 7
        
        # Colors (Primary: bright, Background: dark)
        self.COLOR_BG = (10, 25, 10)
        self.COLOR_TREE = (40, 25, 15)
        self.COLOR_GRID = (20, 40, 20)
        
        self.COLOR_PLAYER = (100, 255, 100)
        self.COLOR_PLAYER_GLOW = (100, 255, 100, 50)
        
        self.COLOR_HERB = (255, 255, 255)
        self.COLOR_HERB_GLOW = (255, 255, 255, 60)
        
        self.COLOR_CREATURE = (255, 50, 50)
        self.COLOR_CREATURE_FLICKER = (200, 0, 0)
        self.COLOR_CREATURE_GLOW = (255, 0, 0, 70)
        
        self.COLOR_TEXT = (220, 220, 220)
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Game State (initialized in reset)
        self.player_pos = None
        self.herb_positions = None
        self.creature_positions = None
        self.creature_paths = None
        self.creature_path_indices = None
        self.trees = None
        
        self.herbs_collected = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.np_random = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.herbs_collected = 0
        self.game_over = False
        
        # --- Define World Layout ---
        self.trees = [
            (1, 1), (1, 2), (2, 1), (2, 2), # Top-left block
            (7, 1), (8, 1), (7, 2), (8, 2), # Top-right block
            (1, 7), (1, 8), (2, 7), (2, 8), # Bottom-left block
            (7, 7), (8, 7), (7, 8), (8, 8), # Bottom-right block
            (4, 4), (5, 4), (4, 5), (5, 5)  # Center block
        ]
        
        # --- Define Creature Paths ---
        self.creature_paths = [
            # Path 1: Clockwise around top-left trees
            [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2), (3,3), (2,3), (1,3), (0,3), (0,2), (0,1)],
            # Path 2: Counter-clockwise around bottom-right trees
            [(9,9), (8,9), (7,9), (6,9), (6,8), (6,7), (6,6), (7,6), (8,6), (9,6), (9,7), (9,8)],
            # Path 3: Horizontal patrol in the middle
            [(0,6), (1,6), (2,6), (3,6), (4,6), (5,6), (6,6), (7,6), (8,6), (9,6), (8,6), (7,6), (6,6), (5,6), (4,6), (3,6), (2,6), (1,6)]
        ]
        self.creature_path_indices = [0, 0, 0]
        self.creature_positions = [p[i] for p, i in zip(self.creature_paths, self.creature_path_indices)]
        
        # --- Place Player and Herbs ---
        self.player_pos = [4, 9] # Start at bottom-center
        
        occupied_cells = set(self.trees)
        occupied_cells.add(tuple(self.player_pos))
        for pos in self.creature_positions:
            occupied_cells.add(tuple(pos))
            
        self.herb_positions = []
        while len(self.herb_positions) < self.HERBS_TO_COLLECT:
            pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
            if pos not in occupied_cells:
                self.herb_positions.append(list(pos))
                occupied_cells.add(pos)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # --- Calculate pre-move distances for reward shaping ---
        dist_to_herb_before = self._get_dist_to_nearest(self.player_pos, self.herb_positions)
        dist_to_creature_before = self._get_dist_to_nearest(self.player_pos, self.creature_positions)

        # --- 1. Update Player Position ---
        if movement != 0: # no-op
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_pos = [
                (self.player_pos[0] + dx) % self.GRID_WIDTH,
                (self.player_pos[1] + dy) % self.GRID_HEIGHT
            ]
            
            # Prevent moving onto trees
            if tuple(new_pos) not in self.trees:
                self.player_pos = new_pos
        
        # --- 2. Update Creature Positions ---
        for i in range(len(self.creature_paths)):
            self.creature_path_indices[i] = (self.creature_path_indices[i] + 1) % len(self.creature_paths[i])
            self.creature_positions[i] = self.creature_paths[i][self.creature_path_indices[i]]
        
        # --- 3. Calculate Post-Move Rewards and Check Collisions ---
        
        # Continuous reward based on distance change
        dist_to_herb_after = self._get_dist_to_nearest(self.player_pos, self.herb_positions)
        dist_to_creature_after = self._get_dist_to_nearest(self.player_pos, self.creature_positions)
        
        if dist_to_herb_before > 0:
            reward += (dist_to_herb_before - dist_to_herb_after) * 0.1
        if dist_to_creature_before > 0:
            reward += (dist_to_creature_before - dist_to_creature_after) * -0.2

        # Event-based reward: Herb collection
        if self.player_pos in self.herb_positions:
            self.herb_positions.remove(self.player_pos)
            self.herbs_collected += 1
            reward += 10
            # sfx: herb_collect.wav
            if self.herbs_collected == self.HERBS_TO_COLLECT:
                reward += 100 # Win bonus
                self.game_over = True
                # sfx: game_win.wav
        
        # Event-based reward: Creature collision
        if tuple(self.player_pos) in [tuple(p) for p in self.creature_positions]:
            reward -= 100 # Loss penalty
            self.game_over = True
            # sfx: player_death.wav
        
        # --- 4. Update Step Counter and Termination ---
        self.steps += 1
        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_dist_to_nearest(self, pos, target_list):
        if not target_list:
            return 0
        # Manhattan distance
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in target_list)

    def _grid_to_pixel(self, x, y):
        return (
            int(self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE / 2),
            int(self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2)
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.HEIGHT - self.GRID_OFFSET_Y))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.WIDTH - self.GRID_OFFSET_X, py))

        # Render trees
        for x, y in self.trees:
            rect = pygame.Rect(self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_TREE, rect)

        # Render herbs
        for x, y in self.herb_positions:
            px, py = self._grid_to_pixel(x, y)
            glow_radius = int(self.CELL_SIZE * 0.7)
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_HERB_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_HERB_GLOW)
            pygame.draw.circle(self.screen, self.COLOR_HERB, (px, py), int(self.CELL_SIZE * 0.2))

        # Render creatures
        flicker = self.steps % 4 < 2
        for x, y in self.creature_positions:
            px, py = self._grid_to_pixel(x, y)
            color = self.COLOR_CREATURE if flicker else self.COLOR_CREATURE_FLICKER
            size = int(self.CELL_SIZE * 0.4) if flicker else int(self.CELL_SIZE * 0.35)
            # Glow
            glow_radius = int(self.CELL_SIZE * 0.8)
            pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_CREATURE_GLOW)
            pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_CREATURE_GLOW)
            # Core
            pygame.draw.rect(self.screen, color, (px - size//2, py - size//2, size, size))

        # Render player
        px, py = self._grid_to_pixel(self.player_pos[0], self.player_pos[1])
        glow_radius = int(self.CELL_SIZE * 0.75)
        pygame.gfxdraw.filled_circle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, px, py, glow_radius, self.COLOR_PLAYER_GLOW)
        size = int(self.CELL_SIZE * 0.6)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (px - size//2, py - size//2, size, size))
        
        # Render UI
        herb_text = self.font_small.render(f"Herbs: {self.herbs_collected} / {self.HERBS_TO_COLLECT}", True, self.COLOR_TEXT)
        self.screen.blit(herb_text, (10, 10))
        
        # Render Game Over/Win Text
        if self.game_over:
            if self.herbs_collected == self.HERBS_TO_COLLECT:
                end_text = "CURSE LIFTED"
            else:
                end_text = "CAUGHT BY THE FOREST"
            
            text_surf = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "herbs_collected": self.herbs_collected,
            "player_pos": self.player_pos,
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
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Cursed Forest")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print(env.user_guide)

    while not done:
        # --- Human Controls ---
        movement_action = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        action = [movement_action, 0, 0] # Space/Shift are not used

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Game Reset ---")
                
                # Step the environment only on a key press for turn-based gameplay
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                print(f"Step: {info['steps']}, Action: {movement_action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

        # --- Rendering ---
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit FPS for human play

    print("\n--- Game Over ---")
    print(f"Final Score: {info['score']:.2f}")
    print(f"Total Steps: {info['steps']}")
    
    # Keep the window open for a bit to see the final screen
    pygame.time.wait(3000)
    env.close()