
# Generated: 2025-08-28T00:15:43.471896
# Source Brief: brief_01553.md
# Brief Index: 1553

        
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
        "Stack falling blocks as high as possible. A wobbly tower will collapse! Reach 100 units to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLATFORM = (100, 100, 110)
        self.COLOR_TARGET_LINE = (255, 0, 0, 150)
        self.COLOR_CURSOR = (255, 255, 255, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (52, 152, 219), (231, 76, 60), (46, 204, 113),
            (241, 196, 15), (155, 89, 182), (26, 188, 156)
        ]

        # Game parameters
        self.MAX_STEPS = 2000
        self.WIN_HEIGHT = 100
        self.PLATFORM_WIDTH = 150
        self.PLATFORM_HEIGHT = 20
        self.BLOCK_WIDTH = 50
        self.BLOCK_HEIGHT = 15
        self.CURSOR_SPEED = 6
        self.INITIAL_FALL_SPEED = 1.0 # pixels per step
        self.FALL_SPEED_INCREASE = 0.05
        self.FAST_DROP_MULTIPLIER = 5

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.platform = None
        self.stacked_blocks = []
        self.falling_block = None
        self.fall_speed = 0
        self.current_stack_height = 0
        self.center_of_mass_x = 0
        self.instability = 0
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.platform = pygame.Rect(
            (self.WIDTH - self.PLATFORM_WIDTH) // 2,
            self.HEIGHT - self.PLATFORM_HEIGHT,
            self.PLATFORM_WIDTH,
            self.PLATFORM_HEIGHT
        )

        self.stacked_blocks = []
        self.particles = []
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.current_stack_height = 0
        self.center_of_mass_x = self.platform.centerx
        self.instability = 0

        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1

        if self.game_over:
            self._update_particles() # Keep particles moving on end screen
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held (action[2]) is unused

        # --- Handle Input & Game Logic ---
        if self.falling_block:
            # Move cursor/block horizontally
            if movement == 3:  # Left
                self.falling_block["rect"].x -= self.CURSOR_SPEED
            elif movement == 4: # Right
                self.falling_block["rect"].x += self.CURSOR_SPEED
            
            # Clamp to screen bounds
            self.falling_block["rect"].x = np.clip(self.falling_block["rect"].x, 0, self.WIDTH - self.BLOCK_WIDTH)

            # Move block down (fast drop if space is held)
            current_fall_speed = self.fall_speed * self.FAST_DROP_MULTIPLIER if space_held else self.fall_speed
            self.falling_block["rect"].y += current_fall_speed

            # Check for landing
            landing_surface = self.platform
            highest_block = None
            if self.stacked_blocks:
                # Find the highest block the falling block could land on
                potential_supports = [b for b in self.stacked_blocks if self.falling_block["rect"].colliderect(b["rect"])]
                if potential_supports:
                    highest_block = min(potential_supports, key=lambda b: b["rect"].top)
                    landing_surface = highest_block["rect"]
            
            if self.falling_block["rect"].bottom >= landing_surface.top:
                # Block has landed on a surface
                self.falling_block["rect"].bottom = landing_surface.top
                
                # Check if it landed on a valid support (not off the side)
                if self.falling_block["rect"].right < self.platform.left or self.falling_block["rect"].left > self.platform.right:
                    # Dropped completely off the platform stack
                    # Sound: fall_off.wav
                    reward += -5
                    self.game_over = True
                    self._create_splash(self.falling_block["rect"].center, self.falling_block["color"], 10)
                    self.falling_block = None
                else:
                    # Landed on stack/platform, now check stability
                    placement_info = self._place_block()
                    reward += placement_info["reward"]
                    if placement_info["collapse"]:
                        self.game_over = True
                    elif not self.game_over:
                        self._spawn_new_block()
        
        # Update particles
        self._update_particles()

        # Increase difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed += self.FALL_SPEED_INCREASE
            
        # Check for win condition
        if self.current_stack_height >= self.WIN_HEIGHT and not self.game_over:
            self.win = True
            self.game_over = True
            reward += 100 # Goal-oriented reward

        self.score += reward
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _place_block(self):
        new_block = self.falling_block
        self.falling_block = None

        all_blocks = self.stacked_blocks + [new_block]
        
        # Calculate new center of mass
        total_mass = 0
        weighted_sum_x = 0
        for block in all_blocks:
            mass = block["rect"].width * block["rect"].height
            total_mass += mass
            weighted_sum_x += block["rect"].centerx * mass
        
        new_com_x = weighted_sum_x / total_mass if total_mass > 0 else self.platform.centerx

        # Check for collapse
        if new_com_x < self.platform.left or new_com_x > self.platform.right:
            # Sound: collapse.wav
            self._create_collapse_explosion(all_blocks)
            self.stacked_blocks = []
            return {"reward": -50, "collapse": True}
        
        # If stable, add to stack
        # Sound: place_block.wav
        self.stacked_blocks.append(new_block)
        self.center_of_mass_x = new_com_x

        # Update stack height
        new_height = self.platform.top - min(b["rect"].top for b in self.stacked_blocks)
        height_gain = new_height - self.current_stack_height
        self.current_stack_height = new_height

        # Calculate rewards
        reward = 10  # For successful placement
        reward += height_gain * 0.1 # For height gain
        offset = abs(new_block["rect"].centerx - self.platform.centerx)
        reward -= offset * 0.01 # Penalty for being off-center

        # Create landing particles
        self._create_splash(new_block["rect"].midbottom, new_block["color"], 15)

        return {"reward": reward, "collapse": False}

    def _spawn_new_block(self):
        x = self.np_random.integers(self.BLOCK_WIDTH // 2, self.WIDTH - self.BLOCK_WIDTH // 2)
        color = random.choice(self.BLOCK_COLORS)
        self.falling_block = {
            "rect": pygame.Rect(x - self.BLOCK_WIDTH // 2, 0, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
            "color": color
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Target height line
        target_y = self.platform.top - self.WIN_HEIGHT
        if target_y > 0:
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.WIDTH, target_y), 1)

        # Platform
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, self.platform)
        pygame.draw.rect(self.screen, tuple(max(0, c-20) for c in self.COLOR_PLATFORM), self.platform, 2)

        # Instability/wobble calculation
        self.instability = (self.center_of_mass_x - self.platform.centerx) / (self.platform.width / 2.0)
        wobble_factor = self.instability * 0.1 # A scalar for visual lean

        # Stacked blocks
        for block in self.stacked_blocks:
            y_offset = self.platform.top - block["rect"].centery
            x_lean = wobble_factor * y_offset
            x_lean += math.sin(self.steps * 0.15 + y_offset * 0.1) * self.instability * 2.5
            
            render_rect = block["rect"].copy()
            render_rect.x += int(x_lean)
            
            # Use gfxdraw for anti-aliasing
            border_color = tuple(max(0, c-40) for c in block["color"])
            pygame.gfxdraw.box(self.screen, render_rect, block["color"])
            pygame.gfxdraw.rectangle(self.screen, render_rect, border_color)


        # Falling block
        if self.falling_block:
            # Draw cursor projection line
            r, g, b, a = self.COLOR_CURSOR
            pygame.gfxdraw.vline(self.screen, self.falling_block["rect"].centerx, 0, self.HEIGHT, (r,g,b,a))
            
            # Draw falling block
            border_color = tuple(max(0, c-40) for c in self.falling_block["color"])
            pygame.gfxdraw.box(self.screen, self.falling_block["rect"], self.falling_block["color"])
            pygame.gfxdraw.rectangle(self.screen, self.falling_block["rect"], border_color)

        # Particles
        for p in self.particles:
            p_size = int(p["size"] * (p["life"] / p["max_life"]))
            if p_size > 0:
                alpha = int(255 * (p["life"] / p["max_life"]))
                color = p["color"] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), p_size, color)

    def _render_ui(self):
        height_text = self.font_ui.render(f"Height: {self.current_stack_height:.1f} / {self.WIN_HEIGHT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (10, 10))

        score_text = self.font_ui.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            
            # Draw a semi-transparent background for the text
            bg_surf = pygame.Surface(text_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0,0,0,120))
            self.screen.blit(bg_surf, text_rect)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "height": self.current_stack_height}

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # Gravity on particles
            p["life"] -= 1

    def _create_splash(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(-math.pi/2 - 0.5, -math.pi/2 + 0.5)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                "pos": list(pos), "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(20, 40), "max_life": 40,
                "color": color, "size": self.np_random.integers(2, 5)
            })

    def _create_collapse_explosion(self, blocks):
        for block in blocks:
            for _ in range(8):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 5)
                self.particles.append({
                    "pos": list(block["rect"].center), "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "life": self.np_random.integers(40, 70), "max_life": 70,
                    "color": block["color"], "size": self.np_random.integers(3, 7)
                })

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("Validating implementation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")