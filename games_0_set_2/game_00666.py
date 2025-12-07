
# Generated: 2025-08-27T14:23:37.237594
# Source Brief: brief_00666.md
# Brief Index: 666

        
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
    user_guide = "Controls: ←→ to move paddle (slow/fast). Destroy all blocks."

    # Must be a short, user-facing description of the game:
    game_description = "A minimalist block breaker. Position the paddle to destroy falling blocks and score points."

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_Y = self.HEIGHT - 40
        self.BLOCK_SIZE = 20
        self.MAX_BLOCKS = 15
        self.BLOCK_SPAWN_CHANCE = 0.02
        self.WIN_CONDITION = 50
        self.MAX_STEPS = 2500
        self.RISKY_EDGE_THRESHOLD = 10

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 55)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 90, 90), (90, 255, 90), (90, 90, 255),
            (255, 255, 90), (90, 255, 255), (255, 90, 255)
        ]

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
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Initialize state variables to be populated in reset()
        self.paddle_rect = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.blocks_cleared = 0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) // 2,
            self.PADDLE_Y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.blocks_cleared = 0
        self.game_over = False
        self.win = False
        
        # Spawn initial blocks
        for _ in range(5):
            self._spawn_block()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02  # Time penalty to encourage efficiency
        
        if not self.game_over:
            self._handle_input(action)
            step_reward = self._update_game_state()
            reward += step_reward

        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100.0  # Goal-oriented reward for winning
            elif self.game_over: # Loss condition (block hit bottom or max steps)
                reward += -100.0 # Goal-oriented reward for losing
        
        self.score += reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        # Action mapping from brief section 7
        if movement == 1:   # Custom action: Move left slow
            self.paddle_rect.x -= 3
        elif movement == 2: # Custom action: Move right slow
            self.paddle_rect.x += 3
        elif movement == 3: # Custom action: Move left fast
            self.paddle_rect.x -= 10
        elif movement == 4: # Custom action: Move right fast
            self.paddle_rect.x += 10
        
        # Clamp paddle to screen boundaries
        self.paddle_rect.x = max(0, min(self.paddle_rect.x, self.WIDTH - self.PADDLE_WIDTH))

    def _update_game_state(self):
        step_reward = 0.0

        # Update and check collisions for all blocks
        for block in self.blocks[:]:
            block['rect'].y += block['speed']
            
            # Check for paddle collision
            if self.paddle_rect.colliderect(block['rect']):
                # sfx_block_break.play()
                is_risky = self.paddle_rect.left <= self.RISKY_EDGE_THRESHOLD or \
                           self.paddle_rect.right >= self.WIDTH - self.RISKY_EDGE_THRESHOLD
                
                # Event-based rewards
                step_reward += 2.0 if is_risky else -0.2
                
                self._spawn_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)
                self.blocks_cleared += 1

            # Check for hitting the bottom of the screen
            elif block['rect'].top > self.HEIGHT:
                # sfx_game_over.play()
                self.game_over = True
                self.blocks.remove(block) # Remove to clean up screen

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            p['radius'] -= 0.1
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

        # Spawn new blocks based on chance
        if self.np_random.random() < self.BLOCK_SPAWN_CHANCE and len(self.blocks) < self.MAX_BLOCKS:
            self._spawn_block()
        
        return step_reward

    def _spawn_block(self):
        x = self.np_random.integers(0, self.WIDTH - self.BLOCK_SIZE)
        speed = self.np_random.choice([2, 3, 4])
        color = random.choice(self.BLOCK_COLORS)
        rect = pygame.Rect(x, -self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
        self.blocks.append({'rect': rect, 'speed': speed, 'color': color})

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            radius = self.np_random.random() * 3 + 2
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'radius': radius, 'color': color})

    def _check_termination(self):
        if self.blocks_cleared >= self.WIN_CONDITION:
            self.win = True
            self.game_over = True
            return True
        if self.game_over: # This flag is set on loss by block hitting bottom
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True # Treat max steps as a loss condition
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid for aesthetic
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

        # Draw particles first (behind other elements)
        for p in self.particles:
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos_int, max(0, int(p['radius'])))

        # Draw blocks with a simple 3D effect
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            inner_color = tuple(max(0, c - 40) for c in block['color'])
            pygame.draw.rect(self.screen, inner_color, block['rect'], 2)

        # Draw paddle with a glow effect for visibility
        glow_rect = self.paddle_rect.inflate(6, 6)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PADDLE, 30), glow_surface.get_rect(), border_radius=5)
        self.screen.blit(glow_surface, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)

    def _render_ui(self):
        # Score and Blocks Cleared display
        score_text = self.font_small.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        blocks_text = self.font_small.render(f"CLEARED: {self.blocks_cleared}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(blocks_text, (self.WIDTH - blocks_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (150, 255, 150) if self.win else (255, 150, 150)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            # Add a subtle shadow for readability
            shadow_text = self.font_large.render(msg, True, (0,0,0,100))
            self.screen.blit(shadow_text, text_rect.move(2, 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "blocks_cleared": self.blocks_cleared,
            "game_over": self.game_over,
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