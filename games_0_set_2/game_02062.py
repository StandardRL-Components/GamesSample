
# Generated: 2025-08-27T19:08:59.212942
# Source Brief: brief_02062.md
# Brief Index: 2062

        
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
        "Controls: ←→ to move the platform left and right to catch the falling blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as possible in this side-view arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.screen_width = 640
        self.screen_height = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (40, 60, 90)
        self.COLOR_PLATFORM = (255, 255, 255)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TARGET_LINE = (255, 200, 0, 150)
        self.background = self._create_background()
        
        # Game constants
        self.max_steps = 1500
        self.target_height = 250
        self.platform_width = 120
        self.platform_height = 10
        self.platform_speed = 8
        self.initial_fall_speed = 2.0
        self.fall_speed_increment = 0.15

        # Initialize state variables
        self.platform_x = 0
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.fall_speed = 0
        self.stack_height = 0
        self.steps = 0
        self.score = 0
        self.game_over_message = ""

        # This will be called once in __init__
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        
        self.platform_x = self.screen_width / 2
        self.platform_y = self.screen_height - 25
        
        self.stacked_blocks = []
        self.particles = []
        
        self.fall_speed = self.initial_fall_speed
        self.stack_height = 0
        
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]
        
        # --- 1. Handle Input ---
        if movement == 3:  # Left
            self.platform_x -= self.platform_speed
        elif movement == 4:  # Right
            self.platform_x += self.platform_speed
        
        # Clamp platform position
        self.platform_x = max(
            self.platform_width / 2, 
            min(self.platform_x, self.screen_width - self.platform_width / 2)
        )

        # --- 2. Update Game State ---
        self._update_falling_block()
        
        # Check for landing
        landed_info = self._check_landing()
        if landed_info["landed"]:
            # sfx: land
            old_height = self.stack_height
            self.stacked_blocks.append(self.falling_block)
            self._calculate_stack_height()
            
            # Spawn particles on successful placement
            self._spawn_particles(self.falling_block["rect"].midbottom, self.falling_block["color"])
            
            # Spawn the next block
            self._spawn_new_block()
            
            # --- 3. Calculate Reward ---
            reward += 0.5  # Reward for catching
            height_gain = self.stack_height - old_height
            if height_gain > 0:
                reward += height_gain * 1.0 # Reward for increasing height
            
        self._update_particles()
        
        # --- 4. Check Termination Conditions ---
        # A. Unstable placement
        if landed_info["unstable"]:
            # sfx: topple
            reward -= 100
            terminated = True
            self.game_over_message = "UNSTABLE!"
        # B. Block missed
        elif self.falling_block and self.falling_block["rect"].top > self.screen_height:
            # sfx: miss
            reward -= 100
            terminated = True
            self.game_over_message = "MISSED!"
        # C. Reached target height
        elif self.stack_height >= self.target_height:
            # sfx: win
            reward += 100
            terminated = True
            self.game_over_message = "GOAL!"
        
        # D. Max steps reached
        self.steps += 1
        if self.steps >= self.max_steps:
            terminated = True
            if not self.game_over_message:
                self.game_over_message = "TIME UP"

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_new_block(self):
        width = self.np_random.integers(50, 91)
        height = self.np_random.integers(20, 31)
        # Ensure block spawns fully within screen bounds horizontally
        x = self.np_random.integers(width // 2, self.screen_width - width // 2)
        y = -height
        
        # Generate a bright, saturated color
        hue = self.np_random.random()
        color = pygame.Color(0)
        color.hsva = (hue * 360, 90, 95, 100)
        
        self.falling_block = {
            "rect": pygame.Rect(x - width // 2, y, width, height),
            "color": tuple(color)[:3]
        }

    def _update_falling_block(self):
        if not self.falling_block: return
        
        self.falling_block["rect"].y += self.fall_speed
        
        # Increase fall speed over time
        if self.steps > 0 and self.steps % 75 == 0:
            self.fall_speed += self.fall_speed_increment
            
    def _check_landing(self):
        if not self.falling_block:
            return {"landed": False, "unstable": False}

        fb_rect = self.falling_block["rect"]
        
        # Find the highest surface(s) directly below the falling block
        highest_support_y = self.platform_y
        potential_supports = [pygame.Rect(self.platform_x - self.platform_width / 2, self.platform_y, self.platform_width, self.platform_height)]

        for sb in self.stacked_blocks:
            sb_rect = sb["rect"]
            # Check for horizontal overlap
            if fb_rect.left < sb_rect.right and fb_rect.right > sb_rect.left:
                # If this block is higher than the current highest support
                if sb_rect.top < highest_support_y:
                    highest_support_y = sb_rect.top
                    potential_supports = [sb_rect]
                # If it's at the same height, add it to the list of supports
                elif sb_rect.top == highest_support_y:
                    potential_supports.append(sb_rect)

        # Check if the block's bottom will cross the support surface in this frame
        if fb_rect.bottom >= highest_support_y:
            # Snap block to the surface
            fb_rect.bottom = highest_support_y
            
            # Calculate how much of the block's base is supported
            total_support_width = 0
            for support_rect in potential_supports:
                overlap_left = max(fb_rect.left, support_rect.left)
                overlap_right = min(fb_rect.right, support_rect.right)
                if overlap_right > overlap_left:
                    total_support_width += (overlap_right - overlap_left)

            # Check for instability
            if total_support_width < fb_rect.width * 0.20: # Less than 20% support
                return {"landed": False, "unstable": True}
            
            return {"landed": True, "unstable": False}
        
        return {"landed": False, "unstable": False}

    def _calculate_stack_height(self):
        if not self.stacked_blocks:
            self.stack_height = 0
            return
        
        min_y = min(b["rect"].top for b in self.stacked_blocks)
        self.stack_height = max(0, self.platform_y - min_y)

    def _get_observation(self):
        # --- Render all game elements ---
        self.screen.blit(self.background, (0, 0))
        
        # Target line
        target_y = self.platform_y - self.target_height
        if target_y > 0:
            self._draw_dashed_line(
                self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.screen_width, target_y)
            )
            
        # Stacked blocks
        for block in self.stacked_blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, (0,0,0), block["rect"], 2) # Outline

        # Falling block (with glow)
        if self.falling_block:
            fb_rect = self.falling_block["rect"]
            glow_color = self.falling_block["color"]
            # Draw glow effect
            for i in range(4, 0, -1):
                glow_surface = pygame.Surface((fb_rect.width + i*4, fb_rect.height + i*4), pygame.SRCALPHA)
                pygame.draw.rect(glow_surface, (*glow_color, 20), glow_surface.get_rect(), border_radius=i*2)
                self.screen.blit(glow_surface, (fb_rect.left - i*2, fb_rect.top - i*2))
            
            pygame.draw.rect(self.screen, self.falling_block["color"], fb_rect)
            pygame.draw.rect(self.screen, (255,255,255), fb_rect, 2) # Bright outline

        # Platform
        platform_rect = pygame.Rect(
            self.platform_x - self.platform_width / 2, self.platform_y,
            self.platform_width, self.platform_height
        )
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, platform_rect, border_radius=3)
        pygame.draw.rect(self.screen, (50,50,50), platform_rect, 2, border_radius=3)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["radius"]))
            
        # UI
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Display Stack Height
        height_text = f"Height: {int(self.stack_height)} / {self.target_height}"
        text_surf = self.font_ui.render(height_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (15, 10))
        
        # Display Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.screen_width - 15, 10))
        self.screen.blit(score_surf, score_rect)

        # Display Game Over message
        if self.game_over_message:
            go_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_TEXT)
            go_rect = go_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            # Add a semi-transparent background for readability
            bg_surf = pygame.Surface(go_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, go_rect.topleft)
            self.screen.blit(go_surf, go_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": self.stack_height,
        }

    def _create_background(self):
        bg = pygame.Surface((self.screen_width, self.screen_height))
        for y in range(self.screen_height):
            # Linear interpolation between top and bottom colors
            interp = y / self.screen_height
            color = [
                self.COLOR_BG_TOP[i] * (1 - interp) + self.COLOR_BG_BOTTOM[i] * interp
                for i in range(3)
            ]
            pygame.draw.line(bg, color, (0, y), (self.screen_width, y))
        return bg

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": self.np_random.uniform(2, 5),
                "lifetime": self.np_random.integers(20, 40),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Gravity
            p["lifetime"] -= 1
            p["radius"] -= 0.05
            if p["lifetime"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _draw_dashed_line(self, surf, color, start_pos, end_pos, width=2, dash_length=10):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dl = dash_length
        if x1 == x2: # Vertical
            for i in range(y1, y2, dl * 2):
                pygame.draw.line(surf, color, (x1, i), (x1, i + dl), width)
        elif y1 == y2: # Horizontal
            for i in range(x1, x2, dl * 2):
                pygame.draw.line(surf, color, (i, y1), (i + dl, y1), width)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test game-specific assertion from brief
        self.reset()
        self.platform_width = 20
        self.initial_fall_speed = 1
        self.fall_speed = 1
        for _ in range(50):
            _, _, terminated, _, _ = self.step(self.action_space.sample())
            if terminated:
                break
        assert not terminated, "Random agent did not survive 50 steps with specified test conditions."

        print("✓ Implementation validated successfully")

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # To run the game interactively
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    obs, info = env.reset()
    terminated = False
    
    action = env.action_space.sample()
    action[0] = 0 # No-op initially
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Default action is no-op
        movement_action = 0
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        
        action = np.array([movement_action, 0, 0]) # Space and Shift are unused

        if terminated:
            # Show game over screen for 2 seconds then reset
            end_time = pygame.time.get_ticks() + 2000
            while pygame.time.get_ticks() < end_time:
                # Keep drawing the final state
                frame_to_show = np.transpose(obs, (1, 0, 2))
                surf = pygame.surfarray.make_surface(frame_to_show)
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                env.clock.tick(30)
            
            obs, info = env.reset()
            terminated = False

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame_to_show = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_to_show)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30)
        
    env.close()