
# Generated: 2025-08-28T01:07:50.131331
# Source Brief: brief_04006.md
# Brief Index: 4006

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use Left/Right arrows to move the block. Press Space to drop it instantly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as you can. Earn points for risky overhangs, but be careful! "
        "If a block falls off the stack, you lose. Reach a height of 20 to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    WIN_HEIGHT = 20

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_PLAYFIELD = (50, 60, 70)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    # Game element sizes
    BLOCK_WIDTH = 60
    BLOCK_HEIGHT = 20
    PLAYFIELD_WIDTH = 300

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)
        
        self.playfield_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.BLOCK_HEIGHT,
            self.PLAYFIELD_WIDTH,
            self.BLOCK_HEIGHT
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.stacked_blocks = []
        self.height = 0
        self.fall_speed = 10.0
        self.blocks_placed_since_speed_increase = 0
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def _spawn_new_block(self):
        color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
        color = self.BLOCK_COLORS[color_idx]
        start_x = self.SCREEN_WIDTH / 2 - self.BLOCK_WIDTH / 2 + self.np_random.integers(-50, 51)
        
        self.falling_block = {
            "rect": pygame.Rect(start_x, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "color": color
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        terminated = False

        # --- 1. Handle Player Input ---
        if movement == 3:  # Left
            self.falling_block["rect"].x -= 8
        elif movement == 4:  # Right
            self.falling_block["rect"].x += 8
        
        self.falling_block["rect"].left = max(0, self.falling_block["rect"].left)
        self.falling_block["rect"].right = min(self.SCREEN_WIDTH, self.falling_block["rect"].right)

        # --- 2. Update Block Position ---
        if space_held:
            landing_y, _ = self._find_landing_spot()
            self.falling_block["rect"].bottom = landing_y
        else:
            self.falling_block["rect"].y += self.fall_speed
        
        # --- 3. Check for Landing & State Update ---
        landing_y, target_obj = self._find_landing_spot()
        
        if self.falling_block["rect"].bottom >= landing_y:
            # A landing event occurred this frame
            if landing_y > self.SCREEN_HEIGHT:
                # Landed off-screen, which is a loss
                reward = -100
                terminated = True
                # sfx: game_over_sound
            else:
                # Landed on a valid surface
                self.falling_block["rect"].bottom = landing_y # Snap to position
                target_rect = target_obj["rect"] if target_obj else None
                
                place_reward, place_terminated = self._place_block(target_rect)
                reward += place_reward
                terminated = terminated or place_terminated
                # sfx: block_place_sound
                
                if not terminated:
                    self._spawn_new_block()

        # --- 4. Update Counters & Check Termination ---
        self.steps += 1
        self.score += reward
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated:
            self.game_over = True
            if self.height >= self.WIN_HEIGHT:
                self.win = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_landing_spot(self):
        """Finds the highest surface directly beneath the falling block."""
        falling_rect = self.falling_block["rect"]
        
        potential_surfaces = [{"rect": self.playfield_rect, "obj": None}]
        for block in self.stacked_blocks:
            potential_surfaces.append({"rect": block["rect"], "obj": block})

        valid_spots = []
        for surface in potential_surfaces:
            surface_rect = surface["rect"]
            if (falling_rect.left < surface_rect.right and 
                falling_rect.right > surface_rect.left and 
                falling_rect.bottom <= surface_rect.top + 1):
                valid_spots.append(surface)
        
        if not valid_spots:
            return self.SCREEN_HEIGHT + 1, None

        highest_spot = min(valid_spots, key=lambda s: s["rect"].top)
        return highest_spot["rect"].top, highest_spot["obj"]

    def _place_block(self, target_rect):
        """Finalizes block placement, calculates reward, and updates state."""
        reward = 0.1 # Base reward for successful placement

        if target_rect: # Landed on another block
            dx = abs(self.falling_block["rect"].centerx - target_rect.centerx)
            if dx < 2:
                reward = -0.2
            else:
                overhang_bonus = min(2.0, 0.1 * dx)
                reward = 1.0 + overhang_bonus
        
        self.stacked_blocks.append(self.falling_block.copy())
        self.height = len(self.stacked_blocks)
        
        self.blocks_placed_since_speed_increase += 1
        if self.blocks_placed_since_speed_increase >= 5:
            self.fall_speed += 0.2
            self.blocks_placed_since_speed_increase = 0

        terminated = False
        if self.height >= self.WIN_HEIGHT:
            reward += 100
            terminated = True
            # sfx: win_sound
        
        return reward, terminated
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_PLAYFIELD, self.playfield_rect, border_radius=3)

        for block in self.stacked_blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block["color"]), block["rect"], width=2, border_radius=3)

        if self.falling_block and not self.game_over:
            rect = self.falling_block["rect"]
            color = self.falling_block["color"]
            
            glow_rect = rect.inflate(12, 12)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, 60), s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)

            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, width=2, border_radius=3)

    def _render_ui(self):
        height_text = self.font_ui.render(f"Height: {self.height} / {self.WIN_HEIGHT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (15, 15))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 45))
        
        speed_text = self.font_ui.render(f"Fall Speed: {self.fall_speed:.1f} px/f", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 15, 15))

        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            bg_rect = text_rect.inflate(40, 40)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": self.height,
        }
    
    def close(self):
        pygame.quit()

# This block allows for direct human play and visualization
if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Must be set for headless, but we override for human play
    
    try:
        # For human play, we want a real window
        os.environ["SDL_VIDEODRIVER"] = "x11" if os.name == "posix" else "windows"
    except Exception:
        pass # Fallback to dummy if display is not available

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Stacker")

    total_reward = 0
    running = True
    while running:
        movement = 0
        space = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30)

    env.close()