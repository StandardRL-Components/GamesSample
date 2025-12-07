
# Generated: 2025-08-28T03:44:27.026568
# Source Brief: brief_05020.md
# Brief Index: 5020

        
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
        "Controls: Arrow keys to move selector. Space to activate a cell and match a falling block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match falling colored blocks to a grid in a rhythmic, fast-paced puzzle game. Activate the correct color in the correct column to clear blocks, build combos, and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    MAX_STEPS = 1000
    TARGET_BLOCKS = 100
    MAX_CONSECUTIVE_MISSES = 5
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SELECTOR = (255, 255, 0)
    
    COLORS = {
        "RED": (255, 70, 70),
        "GREEN": (70, 255, 70),
        "BLUE": (70, 70, 255),
    }
    GRID_COLORS = {
        "RED": (100, 20, 20),
        "GREEN": (20, 100, 20),
        "BLUE": (20, 20, 100),
    }
    COLOR_NAMES = ["RED", "GREEN", "BLUE"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        self.grid_area_height = 360
        self.cell_size = self.grid_area_height // self.GRID_SIZE
        self.grid_width = self.cell_size * self.GRID_SIZE
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_area_height) // 2

        self.game_state_attributes = [
            'steps', 'score', 'game_over', 'selector_pos', 'grid_colors',
            'blocks', 'particles', 'block_fall_speed', 'blocks_cleared',
            'blocks_spawned', 'consecutive_misses', 'combo', 'prev_space_held',
            'np_random'
        ]
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        grid_color_indices = self.np_random.integers(0, len(self.COLOR_NAMES), size=(self.GRID_SIZE, self.GRID_SIZE))
        self.grid_colors = [[self.COLOR_NAMES[idx] for idx in row] for row in grid_color_indices]

        self.blocks = []
        self.particles = []
        
        self.block_fall_speed = 1.5
        self.blocks_cleared = 0
        self.blocks_spawned = 0
        self.consecutive_misses = 0
        self.combo = 0
        
        self.prev_space_held = False

        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_action, _ = action
        space_held = space_action == 1
        
        self.steps += 1
        reward = -0.01  # Small time penalty

        self._handle_input(movement, space_held)
        
        match_reward = self._process_matches(space_held)
        reward += match_reward

        self._update_blocks()
        miss_reward = self._check_misses()
        reward += miss_reward
        
        self._update_particles()
        
        if self.steps % 50 == 0:
            self.block_fall_speed += 0.02
        
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, movement, space_held):
        if movement == 1: # Up
            self.selector_pos[1] = max(0, self.selector_pos[1] - 1)
        elif movement == 2: # Down
            self.selector_pos[1] = min(self.GRID_SIZE - 1, self.selector_pos[1] + 1)
        elif movement == 3: # Left
            self.selector_pos[0] = max(0, self.selector_pos[0] - 1)
        elif movement == 4: # Right
            self.selector_pos[0] = min(self.GRID_SIZE - 1, self.selector_pos[0] + 1)

    def _process_matches(self, space_held):
        reward = 0
        space_press = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        if not space_press:
            return 0

        sel_x, sel_y = self.selector_pos
        grid_color_name = self.grid_colors[sel_y][sel_x]
        
        matched_block = None
        # Find the lowest block in the selected column to match
        lowest_y = -1
        for block in self.blocks:
            if block['col'] == sel_x and block['y'] > lowest_y:
                lowest_y = block['y']
                matched_block = block
        
        if matched_block and matched_block['color_name'] == grid_color_name:
            # --- Successful Match ---
            # SFX: Match success
            self.blocks.remove(matched_block)
            self._create_particles(
                matched_block['x'] + self.cell_size / 2, 
                matched_block['y'] + self.cell_size / 2, 
                self.COLORS[matched_block['color_name']]
            )
            
            reward += 1.0
            self.score += 1
            self.blocks_cleared += 1
            self.consecutive_misses = 0
            self.combo += 1
            
            if self.combo > 1:
                reward += 5.0
                self.score += self.combo # Combo bonus score
            
            if self.blocks_spawned < self.TARGET_BLOCKS:
                self._spawn_block()
        else:
            # --- Failed Match ---
            # SFX: Match fail
            self.combo = 0
            
        return reward

    def _update_blocks(self):
        for block in self.blocks:
            block['y'] += self.block_fall_speed

    def _check_misses(self):
        reward = 0
        blocks_to_remove = []
        for block in self.blocks:
            if block['y'] > self.SCREEN_HEIGHT:
                # SFX: Miss
                blocks_to_remove.append(block)
                reward -= 1.0
                self.score -= 1
                self.consecutive_misses += 1
                self.combo = 0
                if self.blocks_spawned < self.TARGET_BLOCKS:
                    self._spawn_block()
        
        self.blocks = [b for b in self.blocks if b not in blocks_to_remove]
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _check_termination(self):
        if self.blocks_cleared >= self.TARGET_BLOCKS:
            return True, 100.0
        if self.consecutive_misses >= self.MAX_CONSECUTIVE_MISSES:
            return True, -10.0
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
        return False, 0.0

    def _spawn_block(self):
        if self.blocks_spawned >= self.TARGET_BLOCKS:
            return

        col = self.np_random.integers(0, self.GRID_SIZE)
        color_name = self.np_random.choice(self.COLOR_NAMES)
        
        self.blocks.append({
            'col': col,
            'x': self.grid_offset_x + col * self.cell_size,
            'y': -self.cell_size,
            'color_name': color_name,
        })
        self.blocks_spawned += 1

    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'max_life': 30,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                color_name = self.grid_colors[y][x]
                pygame.draw.rect(self.screen, self.GRID_COLORS[color_name], rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        # Draw falling blocks
        for block in self.blocks:
            rect = pygame.Rect(
                int(block['x']), int(block['y']),
                self.cell_size, self.cell_size
            )
            color = self.COLORS[block['color_name']]
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        selector_rect = pygame.Rect(
            self.grid_offset_x + sel_x * self.cell_size,
            self.grid_offset_y + sel_y * self.cell_size,
            self.cell_size, self.cell_size
        )
        # Pulsing glow effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        thickness = int(2 + pulse * 3)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, thickness)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = p['size'] * (p['life'] / p['max_life'])
            pygame.gfxdraw.filled_circle(
                self.screen, int(p['pos'][0]), int(p['pos'][1]), int(max(0, size)), color
            )

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Misses
        miss_text = self.font_medium.render(f"MISSES: {self.consecutive_misses}/{self.MAX_CONSECUTIVE_MISSES}", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (self.SCREEN_WIDTH - miss_text.get_width() - 10, 10))

        # Blocks Cleared
        blocks_text = self.font_small.render(f"CLEARED: {self.blocks_cleared}/{self.TARGET_BLOCKS}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (10, 40))

        # Combo
        if self.combo > 1:
            combo_text = self.font_large.render(f"x{self.combo}", True, self.COLOR_SELECTOR)
            pos_x = self.grid_offset_x + self.grid_width + 20
            pos_y = self.SCREEN_HEIGHT / 2 - combo_text.get_height() / 2
            self.screen.blit(combo_text, (pos_x, pos_y))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.blocks_cleared >= self.TARGET_BLOCKS:
                end_text = "VICTORY!"
            else:
                end_text = "GAME OVER"
                
            text_surface = self.font_large.render(end_text, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surface, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_cleared": self.blocks_cleared,
            "consecutive_misses": self.consecutive_misses,
            "combo": self.combo,
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # pygame.display.set_caption("Color Grid Match")
    # window = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    # running = True
    # while running:
    #     movement = 0 # none
    #     space = 0 # released
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
        
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: movement = 1
    #     elif keys[pygame.K_DOWN]: movement = 2
    #     elif keys[pygame.K_LEFT]: movement = 3
    #     elif keys[pygame.K_RIGHT]: movement = 4
        
    #     if keys[pygame.K_SPACE]: space = 1
        
    #     action = [movement, space, 0] # shift is not used
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Draw to the window
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     window.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     if terminated:
    #         print(f"Game Over! Final Info: {info}")
    #         pygame.time.wait(2000)
    #         obs, info = env.reset()
        
    #     env.clock.tick(30)
    # env.close()

    # --- Random Agent Test ---
    print("--- Running Random Agent Test ---")
    total_reward = 0
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if (i+1) % 200 == 0:
            print(f"Step {i+1}, Info: {info}, Reward: {reward:.2f}")
        if terminated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
    env.close()