
# Generated: 2025-08-28T02:38:25.176344
# Source Brief: brief_04516.md
# Brief Index: 4516

        
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
        "Controls: ←→ to move the block. Hold Shift to move faster. Hold Space to drop the block quickly."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced stacking game. Build a tower 25 blocks high before the timer runs out. Unstable placements will cause your tower to collapse!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_Y = self.SCREEN_HEIGHT - 40
        self.TARGET_HEIGHT = 25
        self.TIME_LIMIT_SECONDS = 60
        self.FPS = 30
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Block properties
        self.BLOCK_WIDTH = 100
        self.BLOCK_HEIGHT = 15
        
        # Physics and controls
        self.BASE_FALL_SPEED = 1.0
        self.FAST_DROP_SPEED = 15.0
        self.NORMAL_MOVE_SPEED = 5.0
        self.FAST_MOVE_SPEED = 10.0
        self.DIFFICULTY_INTERVAL = 500 # Steps before fall speed increases
        self.FALL_SPEED_INCREASE = 0.05
        
        # Stability
        self.MIN_OVERLAP_RATIO = 0.1 # If overlap is less than 10% of width, tower collapses
        self.MAX_WOBBLE_MAG = 2.5 # Max pixels of visual wobble

        # Colors
        self.COLOR_BG_TOP = (15, 20, 35)
        self.COLOR_BG_BOTTOM = (30, 40, 60)
        self.COLOR_GROUND = (60, 70, 90)
        self.COLOR_FALLING_BLOCK = (255, 100, 100)
        self.COLOR_GHOST_BLOCK = (255, 100, 100, 100)
        self.COLOR_STACK_BASE = (100, 220, 255)
        self.COLOR_STACK_TOP = (40, 80, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 30)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.collapse_animation = None
        self.stacked_blocks = []
        self.falling_block = None
        self.base_fall_speed = self.BASE_FALL_SPEED
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.collapse_animation = None
        self.base_fall_speed = self.BASE_FALL_SPEED
        
        # Create ground block
        ground_block = {
            "x": self.SCREEN_WIDTH / 2 - self.BLOCK_WIDTH * 2,
            "y": self.GROUND_Y,
            "w": self.BLOCK_WIDTH * 4,
            "h": self.BLOCK_HEIGHT,
            "color": self.COLOR_GROUND,
            "wobble_mag": 0,
            "wobble_offset": 0,
        }
        self.stacked_blocks = [ground_block]
        
        self.particles = []
        self._generate_falling_block()
        
        return self._get_observation(), self._get_info()

    def _generate_falling_block(self):
        start_x = self.np_random.uniform(
            self.BLOCK_WIDTH, self.SCREEN_WIDTH - self.BLOCK_WIDTH * 2
        )
        self.falling_block = {
            "x": start_x,
            "y": -self.BLOCK_HEIGHT,
            "w": self.BLOCK_WIDTH,
            "h": self.BLOCK_HEIGHT,
        }

    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            # Let animations play out but don't advance game logic
            self.steps += 1
            terminated = self.steps >= self.MAX_STEPS + self.FPS * 2 # Allow 2s for animation
            return self._get_observation(), 0.0, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Game Logic ---
        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.base_fall_speed += self.FALL_SPEED_INCREASE
            
        # 1. Update falling block based on action
        move_speed = self.FAST_MOVE_SPEED if shift_held else self.NORMAL_MOVE_SPEED
        if movement == 3: # Left
            self.falling_block["x"] -= move_speed
        elif movement == 4: # Right
            self.falling_block["x"] += move_speed
        
        # Clamp to screen bounds
        self.falling_block["x"] = np.clip(self.falling_block["x"], 0, self.SCREEN_WIDTH - self.BLOCK_WIDTH)
        
        # 2. Apply gravity
        fall_speed = self.FAST_DROP_SPEED if space_held else self.base_fall_speed
        self.falling_block["y"] += fall_speed
        
        # 3. Collision and landing check
        landed = False
        bottom_block = self.stacked_blocks[-1]
        
        falling_rect = pygame.Rect(self.falling_block["x"], self.falling_block["y"], self.falling_block["w"], self.falling_block["h"])
        bottom_rect = pygame.Rect(bottom_block["x"], bottom_block["y"], bottom_block["w"], bottom_block["h"])
        
        if falling_rect.bottom >= bottom_rect.top and falling_rect.colliderect(bottom_rect):
            landed = True
            
            # Snap block to top of stack
            self.falling_block["y"] = bottom_rect.top - self.BLOCK_HEIGHT
            
            # Calculate overlap for stability and reward
            overlap = max(0, min(falling_rect.right, bottom_rect.right) - max(falling_rect.left, bottom_rect.left))
            
            if overlap < self.BLOCK_WIDTH * self.MIN_OVERLAP_RATIO:
                # Tower collapse
                # sfx: tower_collapse.wav
                self.game_over = True
                reward += -100.0
                self._start_collapse_animation()
            else:
                # Successful placement
                # sfx: block_place.wav
                reward += 1.0 # Base reward for placement
                reward += (overlap / self.BLOCK_WIDTH) * 0.1 # Bonus for good placement
                
                overhang = self.BLOCK_WIDTH - overlap
                wobble_mag = (abs(falling_rect.centerx - bottom_rect.centerx) / (bottom_rect.w / 2)) * self.MAX_WOBBLE_MAG
                
                new_block = {
                    "x": self.falling_block["x"],
                    "y": self.falling_block["y"],
                    "w": self.BLOCK_WIDTH,
                    "h": self.BLOCK_HEIGHT,
                    "color": self._get_stack_color(len(self.stacked_blocks)),
                    "wobble_mag": wobble_mag,
                    "wobble_offset": self.np_random.uniform(0, 2 * math.pi)
                }
                self.stacked_blocks.append(new_block)
                self.score += 1
                self._create_particles(falling_rect.midbottom, new_block['color'])
                
                if len(self.stacked_blocks) - 1 >= self.TARGET_HEIGHT:
                    self.win = True
                    self.game_over = True
                    reward += 100.0
                else:
                    self._generate_falling_block()
        
        # 4. Check if block fell off
        if not landed and self.falling_block["y"] > self.SCREEN_HEIGHT:
            # sfx: block_miss.wav
            reward += -5.0
            self._generate_falling_block()
        
        # 5. Update particles
        self._update_particles()
        
        # 6. Check termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True # Time's up
            reward += -50.0 # Penalty for running out of time
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_stack_color(self, height):
        ratio = min(1.0, (height - 1) / self.TARGET_HEIGHT)
        r = self.COLOR_STACK_BASE[0] + ratio * (self.COLOR_STACK_TOP[0] - self.COLOR_STACK_BASE[0])
        g = self.COLOR_STACK_BASE[1] + ratio * (self.COLOR_STACK_TOP[1] - self.COLOR_STACK_BASE[1])
        b = self.COLOR_STACK_BASE[2] + ratio * (self.COLOR_STACK_TOP[2] - self.COLOR_STACK_BASE[2])
        return (int(r), int(g), int(b))

    def _start_collapse_animation(self):
        self.collapse_animation = []
        for block in self.stacked_blocks[1:]: # Don't animate ground
            self.collapse_animation.append({
                "x": block["x"],
                "y": block["y"],
                "w": block["w"],
                "h": block["h"],
                "color": block["color"],
                "vx": self.np_random.uniform(-3, 3),
                "vy": self.np_random.uniform(-5, 0),
                "vr": self.np_random.uniform(-5, 5),
                "angle": 0
            })

    def _update_collapse_animation(self):
        if self.collapse_animation is None:
            return
        for block in self.collapse_animation:
            block['vy'] += 0.5 # Gravity
            block['x'] += block['vx']
            block['y'] += block['vy']
            block['angle'] += block['vr']

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "x": pos[0],
                "y": pos[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "radius": self.np_random.uniform(3, 7),
                "color": color,
                "life": self.np_random.integers(15, 30),
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['radius'] -= 0.2
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            ratio = y / self.SCREEN_HEIGHT
            r = self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio
            g = self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio
            b = self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        if self.collapse_animation:
            self._update_collapse_animation()
            for block in self.collapse_animation:
                surf = pygame.Surface((block['w'], block['h']), pygame.SRCALPHA)
                surf.fill(block['color'])
                rotated_surf = pygame.transform.rotate(surf, block['angle'])
                rect = rotated_surf.get_rect(center=(block['x'] + block['w']/2, block['y'] + block['h']/2))
                self.screen.blit(rotated_surf, rect)
            return

        # Draw stacked blocks with wobble
        for i, block in enumerate(self.stacked_blocks):
            wobble = 0
            if block["wobble_mag"] > 0:
                wobble = math.sin(self.steps * 0.2 + block["wobble_offset"]) * block["wobble_mag"]
            
            rect = (int(block["x"] + wobble), int(block["y"]), block["w"], block["h"])
            pygame.draw.rect(self.screen, block["color"], rect)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in block["color"]), rect, 1) # Outline

        # Draw ghost block
        if self.falling_block and not self.game_over:
            ghost_y = self.stacked_blocks[-1]["y"] - self.BLOCK_HEIGHT
            ghost_rect = (int(self.falling_block["x"]), int(ghost_y), self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
            s = pygame.Surface((self.BLOCK_WIDTH, self.BLOCK_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_GHOST_BLOCK)
            self.screen.blit(s, ghost_rect)

        # Draw falling block
        if self.falling_block and not self.game_over:
            rect = (int(self.falling_block["x"]), int(self.falling_block["y"]), self.falling_block["w"], self.falling_block["h"])
            pygame.draw.rect(self.screen, self.COLOR_FALLING_BLOCK, rect)
            pygame.draw.rect(self.screen, (255,255,255), rect, 2) # Bright outline
            
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.aacircle(self.screen, int(p['x']), int(p['y']), int(p['radius']), p['color'])

    def _render_text(self, text, pos, font, color=None, shadow_color=None):
        color = color or self.COLOR_TEXT
        shadow_color = shadow_color or self.COLOR_TEXT_SHADOW
        
        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
        
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_ui(self):
        # UI elements
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        height = max(0, len(self.stacked_blocks) - 1)
        
        self._render_text(f"TIME: {time_left:.1f}", (10, 10), self.font_ui)
        self._render_text(f"HEIGHT: {height}/{self.TARGET_HEIGHT}", (self.SCREEN_WIDTH - 200, 10), self.font_ui)
        
        if self.game_over:
            if self.win:
                msg = "TOWER COMPLETE!"
                color = (100, 255, 100)
            elif self.collapse_animation:
                msg = "TOWER COLLAPSED!"
                color = (255, 80, 80)
            else: # Time's up
                msg = "TIME'S UP!"
                color = (255, 200, 0)
            
            text_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2 + 3, self.SCREEN_HEIGHT/2 + 3))
            self.screen.blit(text_surf, text_rect)
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": len(self.stacked_blocks) - 1,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Controls ---
    # Create a mapping from pygame keys to the MultiDiscrete action space
    key_map = {
        pygame.K_LEFT: (3, 0, 0),  # Movement: left
        pygame.K_RIGHT: (4, 0, 0), # Movement: right
        pygame.K_SPACE: (0, 1, 0), # Space held
        pygame.K_LSHIFT: (0, 0, 1),# Shift held
        pygame.K_RSHIFT: (0, 0, 1),# Shift held
    }

    obs, info = env.reset()
    done = False
    
    # Use a separate pygame window for rendering
    pygame.display.set_caption("Block Stacker")
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    while not done:
        # --- Action Calculation ---
        # Start with a default no-op action
        action = np.array([0, 0, 0]) 
        
        # Poll for pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Check for held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info['score']:.2f}, Height: {info['height']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()