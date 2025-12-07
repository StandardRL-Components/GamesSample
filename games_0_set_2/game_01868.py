
# Generated: 2025-08-27T18:32:41.464909
# Source Brief: brief_01868.md
# Brief Index: 1868

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, hold Shift for speed. Hold Space to drop faster."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Stack falling blocks to build the tallest tower. Reach 20 blocks to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.BLOCK_HEIGHT = 15
        self.BASE_BLOCK_WIDTH = 200
        self.FALLING_BLOCK_WIDTH = 100
        self.TARGET_HEIGHT = 20
        self.MAX_STEPS = 2000

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_BASE = (100, 100, 110)
        self.COLOR_TARGET_LINE = (0, 255, 128, 100) # Green with alpha
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.BLOCK_COLORS = [
            (255, 95, 87), (255, 179, 61), (255, 229, 89),
            (148, 224, 68), (87, 201, 255), (153, 114, 230),
            (255, 133, 209), (255, 82, 161), (46, 214, 195),
            (255, 255, 255)
        ]

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
        try:
            self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
            self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_game_over = pygame.font.Font(None, 50)
        
        # Game state (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stacked_blocks = []
        self.falling_block = None
        self.particles = []
        self.base_fall_speed = 0.0
        self.fall_speed = 0.0
        self.max_height = 0
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.max_height = 0
        self.game_over = False

        # Create base platform
        base_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.BASE_BLOCK_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.BLOCK_HEIGHT,
            self.BASE_BLOCK_WIDTH,
            self.BLOCK_HEIGHT
        )
        self.stacked_blocks = [(base_rect, self.COLOR_BASE)]

        self.particles = []
        self.base_fall_speed = 1.5
        self.fall_speed = self.base_fall_speed
        
        self._spawn_new_block()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(30) # Maintain 30 FPS
        self.steps += 1
        reward = -0.01 # Small penalty for time passing
        terminated = False

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # --- Action Handling ---
        move_speed = 10 if shift_held else 5
        if movement == 3: # Left
            self.falling_block['rect'].x -= move_speed
        elif movement == 4: # Right
            self.falling_block['rect'].x += move_speed

        # Clamp to screen
        self.falling_block['rect'].x = max(0, min(self.SCREEN_WIDTH - self.FALLING_BLOCK_WIDTH, self.falling_block['rect'].x))

        # --- Game Logic ---
        # Apply gravity
        drop_speed = self.fall_speed * 10 if space_held else self.fall_speed
        self.falling_block['y_float'] += drop_speed
        self.falling_block['rect'].y = int(self.falling_block['y_float'])

        # --- Collision and Placement ---
        highest_support_y = 0
        support_surface = None
        
        for block_rect, _ in self.stacked_blocks:
            if self.falling_block['rect'].colliderect(block_rect):
                if block_rect.y > highest_support_y:
                    highest_support_y = block_rect.y
                    support_surface = block_rect
        
        if support_surface:
            # Block has landed
            self.falling_block['rect'].bottom = support_surface.top
            clipped_rect = self.falling_block['rect'].clip(support_surface)
            
            if clipped_rect.width <= 1: # Block missed the stack
                reward = -100
                terminated = True
                # Sound: fail
                self._create_particles(self.falling_block['rect'].center, (255, 50, 50), 50)
            else:
                # Successful placement
                reward += 0.1
                
                overhang = self.falling_block['rect'].width - clipped_rect.width
                if overhang > 2: # Imperfect placement
                    reward -= 5
                else: # Perfect placement
                    reward += 1
                    clipped_rect = self.falling_block['rect'].copy()
                    clipped_rect.bottom = support_surface.top

                # Add new block to the stack
                self.stacked_blocks.append((clipped_rect, self.falling_block['color']))
                self.score += 1
                
                # New height record
                if self.score > self.max_height:
                    reward += 10
                    self.max_height = self.score

                # Sound: place_block
                self._create_particles(clipped_rect.midbottom, self.falling_block['color'], 20)
                
                # Check for win condition
                if self.score >= self.TARGET_HEIGHT:
                    reward += 100
                    terminated = True
                    # Sound: win
                else:
                    self._spawn_new_block()
        
        # --- Termination Checks ---
        if not terminated and self.falling_block['rect'].top > self.SCREEN_HEIGHT:
            reward = -100
            terminated = True
            # Sound: fail
            self._create_particles(self.falling_block['rect'].center, (255, 50, 50), 50)

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_new_block(self):
        # Increase difficulty every 5 blocks
        difficulty_level = self.score // 5
        self.fall_speed = self.base_fall_speed + difficulty_level * 0.25

        start_x = self.np_random.integers(0, self.SCREEN_WIDTH - self.FALLING_BLOCK_WIDTH + 1)
        rect = pygame.Rect(start_x, -self.BLOCK_HEIGHT, self.FALLING_BLOCK_WIDTH, self.BLOCK_HEIGHT)
        color = random.choice(self.BLOCK_COLORS)
        self.falling_block = {'rect': rect, 'color': color, 'y_float': float(rect.y)}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw target height line
        target_y = self.SCREEN_HEIGHT - self.TARGET_HEIGHT * self.BLOCK_HEIGHT
        if target_y > 0:
            pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (0, target_y), (self.SCREEN_WIDTH, target_y), 2)

        # Draw stacked blocks
        for rect, color in self.stacked_blocks:
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c * 0.7) for c in color), rect, 1)

        # Draw ghost block (landing preview)
        if not self.game_over and self.falling_block:
            ghost_rect = self.falling_block['rect'].copy()
            highest_y = self.SCREEN_HEIGHT
            for block_rect, _ in self.stacked_blocks:
                if ghost_rect.colliderect(block_rect.x, 0, block_rect.width, self.SCREEN_HEIGHT):
                    highest_y = min(highest_y, block_rect.top)
            
            ghost_rect.bottom = highest_y
            ghost_color = self.falling_block['color'] + (100,)
            s = pygame.Surface(ghost_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, ghost_color, s.get_rect())
            self.screen.blit(s, ghost_rect.topleft)

        # Draw falling block
        if not self.game_over and self.falling_block:
            rect = self.falling_block['rect']
            color = self.falling_block['color']
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c * 0.7) for c in color), rect, 1)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), int(p['size']))

    def _render_ui(self):
        score_text = self.font_ui.render(f"Height: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.TARGET_HEIGHT else "GAME OVER"
            color = (0, 255, 128) if self.score >= self.TARGET_HEIGHT else (255, 50, 50)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'size': random.uniform(2, 5),
                'life': random.randint(20, 40),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.05)
            if p['life'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.font.quit()
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
    # To run with a window, comment out the os.environ line
    # and ensure 'pygame' is installed.
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # Create a window to display the game
    window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()

    obs, info = env.reset()
    running = True
    while running:
        # User input mapping for human play
        mov, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [mov, space, shift]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000) # Pause on game over
            obs, info = env.reset()
        
        clock.tick(30)

    env.close()