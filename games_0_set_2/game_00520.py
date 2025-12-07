
# Generated: 2025-08-27T13:53:59.676818
# Source Brief: brief_00520.md
# Brief Index: 520

        
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
        "Controls: ←→ to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block breaker. Deflect falling blocks with your paddle to score points. Clear 50 blocks to win, but let one touch the bottom and you lose."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 8.0
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 40, 20
        self.MAX_BLOCKS = 7
        self.WIN_CONDITION = 50
        self.MAX_STEPS = 1000
        self.INITIAL_BLOCK_SPEED = 2.0
        self.SPEED_INCREASE_INTERVAL = 10
        self.SPEED_INCREASE_AMOUNT = 0.5 # Increased for more noticeable difficulty ramp

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BORDER = (100, 100, 120)
        self.COLOR_PADDLE = (240, 240, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_SPECS = {
            "red": {"color": (255, 80, 80), "value": 1},
            "green": {"color": (80, 255, 80), "value": 2},
            "blue": {"color": (80, 150, 255), "value": 3},
        }
        self.BLOCK_TYPES = list(self.BLOCK_SPECS.keys())

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
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.paddle = None
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.cleared_blocks_count = 0
        self.current_block_speed = self.INITIAL_BLOCK_SPEED
        self.game_over = False

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.cleared_blocks_count = 0
        self.current_block_speed = self.INITIAL_BLOCK_SPEED
        self.game_over = False
        
        # Pre-populate with a few blocks
        for _ in range(3):
            self._spawn_block(initial_spawn=True)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # Base reward for surviving a step
        reward = -0.01

        # --- Update Game Logic ---
        self._handle_input(movement)
        
        reward += self._update_blocks()
        self._update_particles()
        self._spawn_blocks()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self.game_over
        if not terminated:
            if self.cleared_blocks_count >= self.WIN_CONDITION:
                reward += 100  # Win reward
                terminated = True
                self.game_over = True
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen bounds
        self.paddle.x = max(2, min(self.WIDTH - self.PADDLE_WIDTH - 2, self.paddle.x))

    def _update_blocks(self):
        step_reward = 0
        for block in self.blocks[:]:
            block['rect'].y += self.current_block_speed

            # Check for collision with paddle
            if self.paddle.colliderect(block['rect']):
                # SFX: Block deflect sound
                self.score += block['value']
                step_reward += block['value']
                self.cleared_blocks_count += 1
                
                self._create_particles(block['rect'].center, block['color'])
                
                self.blocks.remove(block)
                
                # Increase difficulty
                if self.cleared_blocks_count > 0 and self.cleared_blocks_count % self.SPEED_INCREASE_INTERVAL == 0:
                    self.current_block_speed += self.SPEED_INCREASE_AMOUNT

            # Check for reaching bottom
            elif block['rect'].bottom >= self.HEIGHT:
                # SFX: Game over sound
                self.game_over = True
                step_reward -= 100  # Loss penalty
                self.blocks.remove(block)

        return step_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'][0] += p['vel'][0]
                p['pos'][1] += p['vel'][1]
                p['vel'][1] += 0.1 # gravity
                p['radius'] -= 0.2

    def _spawn_block(self, initial_spawn=False):
        if len(self.blocks) < self.MAX_BLOCKS:
            block_type = self.np_random.choice(self.BLOCK_TYPES)
            spec = self.BLOCK_SPECS[block_type]
            
            x_pos = self.np_random.integers(2, self.WIDTH - self.BLOCK_WIDTH - 2)
            y_pos = self.np_random.integers(-100, -self.BLOCK_HEIGHT) if not initial_spawn else self.np_random.integers(50, 200)

            new_block = {
                'rect': pygame.Rect(x_pos, y_pos, self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                'color': spec['color'],
                'value': spec['value']
            }
            self.blocks.append(new_block)

    def _spawn_blocks(self):
        if self.np_random.random() < 0.02: # Probability-based spawning
             self._spawn_block()

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'radius': self.np_random.uniform(4, 8),
                'life': self.np_random.integers(20, 40),
                'color': color
            })

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
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']),
                    (*p['color'], alpha)
                )

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            # Add a subtle inner highlight for depth
            highlight_color = tuple(min(255, c + 50) for c in block['color'])
            pygame.draw.rect(self.screen, highlight_color, block['rect'].inflate(-8, -8), border_radius=2)


        # Draw paddle with a glow effect
        paddle_glow_rect = self.paddle.inflate(6, 6)
        paddle_glow_surf = pygame.Surface(paddle_glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(paddle_glow_surf, (*self.COLOR_PADDLE, 50), paddle_glow_surf.get_rect(), border_radius=8)
        self.screen.blit(paddle_glow_surf, paddle_glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        cleared_text = self.font.render(f"CLEARED: {self.cleared_blocks_count}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(cleared_text, (self.WIDTH - cleared_text.get_width() - 15, 10))
        
        if self.game_over:
            if self.cleared_blocks_count >= self.WIN_CONDITION:
                end_text_str = "YOU WIN!"
                end_color = (100, 255, 100)
            else:
                end_text_str = "GAME OVER"
                end_color = (255, 100, 100)
            
            end_font = pygame.font.SysFont("monospace", 60, bold=True)
            end_text = end_font.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cleared_blocks": self.cleared_blocks_count,
            "block_speed": self.current_block_speed,
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}, expected {(self.HEIGHT, self.WIDTH, 3)}"
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
    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(60) # Control the frame rate for human play
        
    env.close()