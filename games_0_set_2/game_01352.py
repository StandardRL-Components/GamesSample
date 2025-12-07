
# Generated: 2025-08-27T16:52:03.310140
# Source Brief: brief_01352.md
# Brief Index: 1352

        
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
        "Controls: ←→ to move the paddle. Catch the falling blocks to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block catcher. Position your paddle to intercept falling "
        "blocks, clear lines for points, and survive as long as you can."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game Constants
        self.color_bg = (20, 20, 30)
        self.color_paddle = (255, 255, 0)
        self.color_paddle_glow = (255, 255, 0, 50)
        self.block_colors = {
            1: (255, 50, 50),   # Red
            2: (50, 255, 50),   # Green
            5: (50, 150, 255)   # Blue
        }
        self.color_ui = (200, 200, 220)

        # Game Parameters
        self.paddle_width = 100
        self.paddle_height = 15
        self.paddle_speed = 10
        self.block_size = 32
        self.initial_fall_speed = 2.0
        self.fall_speed_increase = 0.1
        self.blocks_per_line = self.screen_width // self.block_size
        self.win_condition_lines = 20
        self.max_lives = 5
        self.max_steps = 5000
        self.spawn_interval = 30  # frames

        # Pygame Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # State variables (initialized in reset)
        self.paddle_x = 0
        self.lives = 0
        self.score = 0
        self.lines_cleared = 0
        self.blocks_destroyed_total = 0
        self.fall_speed = 0.0
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_spawn_step = 0
        self.np_random = None

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_x = (self.screen_width - self.paddle_width) / 2
        self.lives = self.max_lives
        self.score = 0
        self.lines_cleared = 0
        self.blocks_destroyed_total = 0
        self.fall_speed = self.initial_fall_speed
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_spawn_step = 0

        # Pre-populate the screen with some blocks
        for i in range(5):
            self._spawn_blocks(initial_y_offset=i * 100)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.02  # Small penalty for each frame to encourage action

        # 1. Handle Action
        movement = action[0]
        if movement == 3:  # Left
            self.paddle_x -= self.paddle_speed
        elif movement == 4:  # Right
            self.paddle_x += self.paddle_speed
        
        # Clamp paddle position
        self.paddle_x = max(0, min(self.paddle_x, self.screen_width - self.paddle_width))

        # 2. Update Game State
        # Spawn new blocks periodically
        if self.steps - self.last_spawn_step >= self.spawn_interval:
            self._spawn_blocks()
            self.last_spawn_step = self.steps

        # Move blocks and check for collisions
        paddle_rect = pygame.Rect(self.paddle_x, self.screen_height - self.paddle_height - 10, self.paddle_width, self.paddle_height)
        
        blocks_to_remove = []
        for block in self.blocks:
            block['y'] += self.fall_speed
            block_rect = pygame.Rect(block['x'], block['y'], self.block_size, self.block_size)

            # Check for paddle collision
            if paddle_rect.colliderect(block_rect):
                # SFX: Block hit sound
                blocks_to_remove.append(block)
                self.score += block['points']
                reward += 0.1  # Reward for hitting a block
                
                self._create_particles(block['x'] + self.block_size / 2, block['y'] + self.block_size / 2, self.block_colors[block['points']])
                
                self.blocks_destroyed_total += 1
                new_lines_cleared = self.blocks_destroyed_total // self.blocks_per_line
                
                if new_lines_cleared > self.lines_cleared:
                    lines_gained = new_lines_cleared - self.lines_cleared
                    prev_lines_for_speedup = self.lines_cleared // 2
                    self.lines_cleared = new_lines_cleared
                    reward += 1.0 * lines_gained  # Reward for clearing a "line"
                    # SFX: Line clear fanfare

                    # Increase difficulty every 2 lines
                    current_lines_for_speedup = self.lines_cleared // 2
                    if current_lines_for_speedup > prev_lines_for_speedup:
                        self.fall_speed += self.fall_speed_increase
                        # SFX: Speed up sound

            # Check for bottom collision (miss)
            elif block['y'] > self.screen_height:
                blocks_to_remove.append(block)
                self.lives -= 1
                reward -= 1.0  # Penalty for losing a life
                # SFX: Life lost sound
        
        self.blocks = [b for b in self.blocks if b not in blocks_to_remove]
        
        self._update_particles()
        
        # 3. Check for termination
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100  # Large penalty for losing
            # SFX: Game over music
        elif self.lines_cleared >= self.win_condition_lines:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100  # Large reward for winning
            # SFX: Victory music
        elif self.steps >= self.max_steps:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_blocks(self, initial_y_offset=0):
        num_blocks_to_spawn = self.np_random.integers(3, 7)
        available_slots = list(range(self.blocks_per_line))
        self.np_random.shuffle(available_slots)
        
        for i in range(num_blocks_to_spawn):
            if not available_slots: break
            slot = available_slots.pop()
            points = self.np_random.choice(list(self.block_colors.keys()))
            new_block = {
                'x': slot * self.block_size,
                'y': -self.block_size - self.np_random.uniform(0, 50) - initial_y_offset,
                'points': points
            }
            self.blocks.append(new_block)

    def _create_particles(self, x, y, color):
        for _ in range(15):
            particle = {
                'x': x,
                'y': y,
                'vx': self.np_random.uniform(-3, 3),
                'vy': self.np_random.uniform(-3, 3),
                'lifespan': self.np_random.integers(15, 30),
                'max_lifespan': 30,
                'color': color
            }
            self.particles.append(particle)
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifespan'] -= 1

    def _get_observation(self):
        self.screen.fill(self.color_bg)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            rect = pygame.Rect(int(block['x']), int(block['y']), self.block_size, self.block_size)
            color = self.block_colors[block['points']]
            border_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, border_color, rect, border_radius=3)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)

        # Render particles
        for p in self.particles:
            alpha = p['lifespan'] / p['max_lifespan']
            color = p['color']
            radius = int(alpha * 5)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), radius, (*color, int(alpha * 200)))

        # Render paddle
        paddle_y = self.screen_height - self.paddle_height - 10
        paddle_rect = pygame.Rect(int(self.paddle_x), paddle_y, self.paddle_width, self.paddle_height)
        
        # Glow effect
        glow_rect = paddle_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.color_paddle_glow, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Main paddle
        pygame.draw.rect(self.screen, self.color_paddle, paddle_rect, border_radius=5)
        
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.color_ui)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.color_ui)
        lives_rect = lives_text.get_rect(centerx=self.screen_width / 2, y=10)
        self.screen.blit(lives_text, lives_rect)
        
        # Lines
        lines_text = self.font_small.render(f"LINES: {self.lines_cleared}/{self.win_condition_lines}", True, self.color_ui)
        lines_rect = lines_text.get_rect(right=self.screen_width - 10, y=10)
        self.screen.blit(lines_text, lines_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            shadow_text = self.font_large.render(message, True, (0,0,0))
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(shadow_text, end_rect.move(2,2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "lines_cleared": self.lines_cleared,
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

# Example of how to run the environment for visualization
if __name__ == '__main__':
    # Set dummy video driver for headless execution
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # To visualize, you'd need a different setup not using the dummy driver
    # This block is for demonstrating the environment runs without a display
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    
    print("Running a short headless episode...")
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"Episode finished. Final Info: {info}, Total Reward: {total_reward:.2f}")

    # To actually play or watch the game, you would need to set up a Pygame window
    # and render the 'rgb_array' observation to the screen.
    # Example (requires a display):
    #
    # import os
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    #
    # env = GameEnv()
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((640, 400))
    # pygame.display.set_caption("Block Breaker")
    # clock = pygame.time.Clock()
    #
    # running = True
    # while running:
    #     action = [0, 0, 0] # Default no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]:
    #         action[0] = 3
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated:
    #         print(f"Game Over! Score: {info['score']}")
    #         pygame.time.wait(2000)
    #         obs, info = env.reset()
    #
    #     # Render the observation from the environment
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #     clock.tick(30) # Maintain 30 FPS
    #
    # pygame.quit()