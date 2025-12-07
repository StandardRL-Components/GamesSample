
# Generated: 2025-08-28T01:29:09.578338
# Source Brief: brief_04124.md
# Brief Index: 4124

        
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
        "Controls: ←→ to move block. ↓ to soft drop. Press space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks as high as possible. Perfect placements earn bonus points. Don't let the stack topple!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    TARGET_HEIGHT = 300 # Pixels from base

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 40, 50)
    COLOR_BASE = (100, 110, 120)
    COLOR_BASE_GLOW = (130, 140, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GAMEOVER = (255, 50, 50, 180)
    COLOR_TARGET_LINE = (255, 215, 0, 100)

    BLOCK_COLORS = [
        ((255, 80, 80), (255, 120, 120)),   # Red
        ((80, 120, 255), (120, 160, 255)),  # Blue
        ((80, 255, 80), (120, 255, 120)),   # Green
        ((255, 255, 80), (255, 255, 120)),  # Yellow
        ((200, 80, 255), (220, 120, 255)),  # Purple
        ((255, 160, 80), (255, 180, 120)),  # Orange
        ((80, 255, 255), (120, 255, 255)),  # Cyan
    ]

    # Game Physics
    BLOCK_HEIGHT = 15
    FALL_SPEED = 1.5
    MOVE_SPEED = 6
    PERFECT_THRESHOLD = 1.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_gameover = pygame.font.SysFont("impact", 60)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.np_random = None
        self.stack = []
        self.falling_block = None
        self.particles = []
        self.last_perfect_landing = False

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.last_perfect_landing = False

        # Create the base platform
        base_width = 200
        self.base_y = self.SCREEN_HEIGHT - 30
        self.stack = [{
            'x': self.SCREEN_WIDTH / 2,
            'y': self.base_y,
            'w': base_width,
            'h': self.BLOCK_HEIGHT,
            'color': self.COLOR_BASE,
            'glow': self.COLOR_BASE_GLOW
        }]
        
        self._spawn_new_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_pressed = action[1] == 1  # Boolean
            
            # --- Handle Input ---
            self._handle_input(movement, space_pressed)
            
            # --- Update Game Logic ---
            reward = self._update_game_state()
            self._update_particles()
            
            self.score += reward

            # --- Check Termination Conditions ---
            stack_height_pixels = self._get_stack_height()
            if stack_height_pixels >= self.TARGET_HEIGHT:
                self.game_over = True
                terminated = True
                reward += 100  # Win bonus
                self.score += 100
            
            if self.steps >= self.MAX_STEPS:
                self.game_over = True
                terminated = True

            if self.game_over and not terminated: # Game over from block placement
                terminated = True
                reward = -100 # Loss penalty
                self.score += -100
        
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        if self.falling_block is None:
            return

        # Horizontal Movement
        if movement == 3:  # Left
            self.falling_block['x'] -= self.MOVE_SPEED
        elif movement == 4:  # Right
            self.falling_block['x'] += self.MOVE_SPEED
        
        # Clamp to screen bounds
        half_w = self.falling_block['w'] / 2
        self.falling_block['x'] = max(half_w, min(self.SCREEN_WIDTH - half_w, self.falling_block['x']))
        
        # Soft Drop
        if movement == 2: # Down
            self.falling_block['y'] += self.FALL_SPEED * 1.5
        
        # Hard Drop
        if space_pressed:
            self._hard_drop()

    def _update_game_state(self):
        if self.falling_block is None:
            return 0

        # Normal fall
        self.falling_block['y'] += self.FALL_SPEED
        
        # Check for landing
        support_block = self.stack[-1]
        if self.falling_block['y'] + self.BLOCK_HEIGHT >= support_block['y']:
            return self._process_landing()
        
        return 0

    def _hard_drop(self):
        if self.falling_block is None:
            return
        support_block = self.stack[-1]
        self.falling_block['y'] = support_block['y'] - self.BLOCK_HEIGHT
        # The landing will be processed in the next _update_game_state call

    def _process_landing(self):
        reward = 0
        new_block = self.falling_block
        support_block = self.stack[-1]

        # Snap Y position
        new_block['y'] = support_block['y'] - self.BLOCK_HEIGHT

        # Check for stability
        new_left = new_block['x'] - new_block['w'] / 2
        new_right = new_block['x'] + new_block['w'] / 2
        support_left = support_block['x'] - support_block['w'] / 2
        support_right = support_block['x'] + support_block['w'] / 2
        
        overlap = max(0, min(new_right, support_right) - max(new_left, support_left))

        if overlap <= 0:
            self.game_over = True
            # Sound: failure_sound.wav
            return 0 # Terminal reward handled in step()
        
        # Calculate reward
        reward += 0.1  # Base reward for placing a block
        center_diff = abs(new_block['x'] - support_block['x'])
        if center_diff < self.PERFECT_THRESHOLD:
            reward += 1.0
            self.last_perfect_landing = True
            # Sound: perfect_place.wav
            self._create_particles(new_block['x'], new_block['y'], new_block['color'])
        else:
            self.last_perfect_landing = False

        # Trim the block based on the overlap (makes game harder and more realistic)
        new_block['x'] = (max(new_left, support_left) + min(new_right, support_right)) / 2
        new_block['w'] = overlap
        
        self.stack.append(new_block)
        # Sound: place_block.wav
        self._spawn_new_block()
        return reward

    def _spawn_new_block(self):
        width = self.np_random.choice([80, 100, 120, 140])
        color, glow = random.choice(self.BLOCK_COLORS)
        
        self.falling_block = {
            'x': self.SCREEN_WIDTH / 2,
            'y': -self.BLOCK_HEIGHT,
            'w': width,
            'h': self.BLOCK_HEIGHT,
            'color': color,
            'glow': glow,
        }

    def _get_stack_height(self):
        if len(self.stack) <= 1:
            return 0
        top_block_y = self.stack[-1]['y']
        return self.base_y - top_block_y

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_stack()
        if self.falling_block:
            self._render_falling_block()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        # Grid lines
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        
        # Target height line
        target_y = self.base_y - self.TARGET_HEIGHT
        pygame.gfxdraw.hline(self.screen, 0, self.SCREEN_WIDTH, int(target_y), self.COLOR_TARGET_LINE)

    def _render_stack(self):
        for block in self.stack:
            self._draw_block(block, self.screen)

    def _render_falling_block(self):
        self._draw_block(self.falling_block, self.screen)
        
        # Draw projection line
        if not self.game_over:
            support_block = self.stack[-1]
            landing_y = support_block['y']
            start_pos = (int(self.falling_block['x']), int(self.falling_block['y'] + self.BLOCK_HEIGHT))
            end_pos = (int(self.falling_block['x']), int(landing_y))
            
            # Dashed line
            dash_length = 5
            for y in range(start_pos[1], end_pos[1], dash_length * 2):
                pygame.draw.line(self.screen, (*self.falling_block['glow'], 100), (start_pos[0], y), (start_pos[0], y + dash_length))

    def _draw_block(self, block, surface):
        x, y, w, h = block['x'], block['y'], block['w'], block['h']
        color, glow = block['color'], block['glow']
        
        # Glow effect
        glow_rect = pygame.Rect(x - w / 2 - 4, y - 4, w + 8, h + 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*glow, 60), glow_surf.get_rect(), border_radius=5)
        surface.blit(glow_surf, glow_rect.topleft)
        
        # Main block
        rect = pygame.Rect(x - w / 2, y, w, h)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        pygame.draw.rect(surface, (255,255,255), rect, width=1, border_radius=3)

    def _create_particles(self, x, y, color):
        for _ in range(20):
            self.particles.append({
                'pos': [x, y + self.BLOCK_HEIGHT / 2],
                'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 0)],
                'alpha': 255,
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['alpha'] -= 8
        self.particles = [p for p in self.particles if p['alpha'] > 0]

    def _render_particles(self):
        for p in self.particles:
            s = pygame.Surface((p['size'] * 2, p['size'] * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 220, p['alpha']), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Height
        height_text = self.font_main.render(f"HEIGHT: {int(self._get_stack_height())}/{self.TARGET_HEIGHT}", True, self.COLOR_TEXT)
        self.screen.blit(height_text, (self.SCREEN_WIDTH - height_text.get_width() - 10, 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_GAMEOVER)
            self.screen.blit(overlay, (0, 0))
            
            win = self._get_stack_height() >= self.TARGET_HEIGHT
            msg = "YOU WIN!" if win else "GAME OVER"
            gameover_text = self.font_gameover.render(msg, True, self.COLOR_TEXT)
            text_rect = gameover_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(gameover_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stack_height": self._get_stack_height(),
            "stack_size": len(self.stack) - 1,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # To run validation
    # env.validate_implementation()

    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0 # unused
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        done = terminated or truncated
        
        clock.tick(30) # 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    
    # Keep the window open for a bit after game over
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        clock.tick(30)

    env.close()