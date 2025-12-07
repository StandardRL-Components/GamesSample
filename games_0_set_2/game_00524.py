
# Generated: 2025-08-27T13:54:46.938994
# Source Brief: brief_00524.md
# Brief Index: 524

        
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
        "Controls: ←→ to move the falling block. Hold Space to drop it faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as you can. Risky placements near the edge earn bonus points, but if a block falls, you lose!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GROUND = (180, 180, 180)
        self.COLOR_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (230, 57, 70),  # Red
            (241, 125, 10), # Orange
            (255, 190, 11), # Yellow
            (6, 214, 160),  # Green
            (21, 150, 225), # Blue
            (111, 75, 242), # Indigo
            (181, 75, 242)  # Violet
        ]

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_Y = self.HEIGHT - 40
        self.WIN_CONDITION_BLOCKS = 15
        self.MAX_STEPS = 2000 # Increased for more gameplay potential
        self.BLOCK_HEIGHT = 20
        self.PLAYER_MOVE_SPEED = 8

        # Initialize state variables
        self.stacked_blocks = []
        self.falling_block = None
        self.steps = 0
        self.score = 0.0
        self.blocks_stacked = 0
        self.base_fall_speed = 0.0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Base platform
        base_width = 150
        base_rect = pygame.Rect(
            (self.WIDTH - base_width) / 2,
            self.GROUND_Y,
            base_width,
            self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [(base_rect, self.COLOR_GROUND)]
        
        self.steps = 0
        self.score = 0.0
        self.blocks_stacked = 0
        self.game_over = False
        self.base_fall_speed = 3.0

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()

    def _spawn_new_block(self):
        width = self.np_random.integers(low=60, high=121)
        color = self.np_random.choice(self.BLOCK_COLORS)
        
        x_pos = self.np_random.integers(low=0, high=self.WIDTH - width + 1)
        
        self.falling_block = {
            "rect": pygame.Rect(x_pos, 0, width, self.BLOCK_HEIGHT),
            "color": tuple(color)
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Action Handling ---
        if movement == 3:  # Left
            self.falling_block["rect"].x -= self.PLAYER_MOVE_SPEED
        elif movement == 4:  # Right
            self.falling_block["rect"].x += self.PLAYER_MOVE_SPEED

        # Clamp to screen boundaries
        self.falling_block["rect"].x = max(0, self.falling_block["rect"].x)
        self.falling_block["rect"].x = min(self.WIDTH - self.falling_block["rect"].width, self.falling_block["rect"].x)

        # --- Game Logic: Gravity & Fast Drop ---
        current_fall_speed = self.base_fall_speed
        if space_held:
            current_fall_speed *= 5  # Fast drop
            # Sound: Fast drop whoosh sfx

        self.falling_block["rect"].y += current_fall_speed

        # --- Collision & Stacking Logic ---
        top_block_rect, _ = self.stacked_blocks[-1]
        
        # Check if the falling block has landed on the stack
        if self.falling_block["rect"].colliderect(top_block_rect) and self.falling_block["rect"].bottom >= top_block_rect.top:
            
            # Snap position vertically for perfect stacking
            self.falling_block["rect"].bottom = top_block_rect.top

            # Calculate horizontal overlap
            overlap_left = max(self.falling_block["rect"].left, top_block_rect.left)
            overlap_right = min(self.falling_block["rect"].right, top_block_rect.right)
            overlap_width = overlap_right - overlap_left
            
            # Termination condition: Block falls off
            if overlap_width <= 1: # Use <=1 to avoid float precision issues
                terminated = True
                self.game_over = True
                reward = -100.0
                self.score = -100.0
                # Sound: Game over / block fall sfx
            else:
                # Successful placement
                # Sound: Block place sfx
                self.stacked_blocks.append((self.falling_block["rect"], self.falling_block["color"]))
                self.blocks_stacked += 1
                
                base_reward = 1.0
                reward += base_reward
                self.score += base_reward

                # Risk/reward scoring based on overhang
                overhang_ratio = 1.0 - (overlap_width / self.falling_block["rect"].width)
                if overhang_ratio > 0.5:  # Risky placement
                    risk_reward = 2.0
                    reward += risk_reward
                    self.score += risk_reward
                    # Sound: Risky placement chime sfx
                elif overhang_ratio < 0.2:  # Safe placement
                    safe_penalty = -0.2
                    reward += safe_penalty
                    self.score += safe_penalty
                
                # Victory condition
                if self.blocks_stacked >= self.WIN_CONDITION_BLOCKS:
                    terminated = True
                    self.game_over = True
                    win_reward = 100.0
                    reward += win_reward
                    self.score += win_reward
                    # Sound: Victory fanfare sfx
                else:
                    # Increase difficulty and spawn next block
                    self.base_fall_speed += 0.1
                    self._spawn_new_block()
        else:
            # Continuous reward for being aligned over the target
            fb_rect = self.falling_block["rect"]
            if fb_rect.left >= top_block_rect.left and fb_rect.right <= top_block_rect.right:
                reward += 0.1

        # --- Step and Max Steps Termination ---
        self.steps += 1
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Draw ground line
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y + self.BLOCK_HEIGHT), (self.WIDTH, self.GROUND_Y + self.BLOCK_HEIGHT), 2)
        
        # Draw all stacked blocks
        for block_rect, color in self.stacked_blocks:
            self._draw_block(block_rect, color)

        # Draw the currently falling block
        if self.falling_block and not self.game_over:
            self._draw_block(self.falling_block["rect"], self.falling_block["color"], is_falling=True)
            # Draw a projection line to help aiming
            top_block_rect, _ = self.stacked_blocks[-1]
            proj_color = (*self.falling_block["color"], 100)
            pygame.draw.line(self.screen, proj_color, 
                             self.falling_block["rect"].midbottom,
                             (self.falling_block["rect"].centerx, top_block_rect.top), 1)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_block(self, rect, color, is_falling=False):
        shadow_color = tuple(max(0, c - 40) for c in color)
        highlight_color = tuple(min(255, c + 40) for c in color)

        pygame.draw.rect(self.screen, color, rect)
        
        # 3D effect with highlights and shadows
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, shadow_color, (rect.left + 1, rect.bottom - 1), rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow_color, rect.topright, (rect.right - 1, rect.bottom - 1), 2)

        if is_falling:
            # Add a subtle glow for better visibility
            glow_surface = pygame.Surface((rect.width + 12, rect.height + 12), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*color, 35), glow_surface.get_rect(), border_radius=8)
            self.screen.blit(glow_surface, (rect.x - 6, rect.y - 6), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score display
        score_text = self.font_big.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Blocks stacked display
        blocks_text = self.font_big.render(f"BLOCKS: {self.blocks_stacked}/{self.WIN_CONDITION_BLOCKS}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, (self.WIDTH - blocks_text.get_width() - 20, 10))
        
        if self.game_over:
            outcome_text = "VICTORY!" if self.blocks_stacked >= self.WIN_CONDITION_BLOCKS else "GAME OVER"
            outcome_color = (255, 215, 0) if self.blocks_stacked >= self.WIN_CONDITION_BLOCKS else (230, 57, 70)
            
            outcome_surf = self.font_big.render(outcome_text, True, outcome_color)
            outcome_rect = outcome_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(outcome_surf, outcome_rect)
            
            reset_surf = self.font_small.render("Call reset() to play again", True, self.COLOR_TEXT)
            reset_rect = reset_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(reset_surf, reset_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_stacked": self.blocks_stacked,
        }

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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played.
    # For training, it should be "rgb_array".
    render_mode = "human" 

    if render_mode == "human":
        # Override the render_mode for human play
        GameEnv.metadata["render_modes"] = ["human", "rgb_array"]
        GameEnv.render = lambda self: pygame.display.update()
        
        env = GameEnv(render_mode="human")
        env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Block Stacker")
    else:
        env = GameEnv()

    obs, info = env.reset()
    terminated = False
    
    # Simple keyboard agent for human play
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            action.fill(0) # Reset actions
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1

            env.clock.tick(30) # 30 FPS for human play
        else:
            # In a non-human mode, just sample random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if render_mode == "human":
            env._get_observation() # Draw the state
            env.render() # Update the display

    if render_mode == "human":
        # Keep window open for a bit after game over
        pygame.time.wait(2000)
        pygame.quit()
    
    print(f"Game Over! Final Info: {info}")