
# Generated: 2025-08-28T05:26:16.765823
# Source Brief: brief_02625.md
# Brief Index: 2625

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use ← and → to move the paddle."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block-breaker. Deflect the ball to destroy all blocks. "
        "Brighter blocks are worth more points. Clear the screen to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Colors and Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (30, 10, 30)
    COLOR_PADDLE = (230, 230, 255)
    COLOR_BALL = (180, 255, 180)
    COLOR_WALL = (80, 80, 120)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (255, 200, 100)

    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 5.0
    MAX_EPISODE_STEPS = 10000
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
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
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.blocks = []
        self.blocks_destroyed_count = 0
        self.particles = []
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        # Paddle
        paddle_y = self.SCREEN_HEIGHT - self.PADDLE_HEIGHT * 2
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) // 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        # Ball
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1], dtype=np.float64)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25) # Upwards angle
        self.ball_speed = self.INITIAL_BALL_SPEED
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.ball_speed
        
        # Blocks
        self._generate_blocks()
        self.blocks_destroyed_count = 0
        
        # Effects
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.1  # Continuous reward for staying in the game

        # --- 1. Handle Input ---
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        if moved:
            reward -= 0.01 # Penalty for movement to encourage efficiency
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle.x))

        # --- 2. Update Game Logic ---
        self._update_ball()
        
        # Handle collisions and get event-based rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- 3. Check Termination Conditions ---
        self.steps += 1
        terminated = False
        if self.lives <= 0:
            terminated = True
            self.game_over = True
        elif not self.blocks:
            reward += 100  # Goal-oriented reward for clearing the level
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += collision_reward # Only add block points to score

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball(self):
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(
            self.ball_pos[0] - self.BALL_RADIUS,
            self.ball_pos[1] - self.BALL_RADIUS,
            self.BALL_RADIUS * 2,
            self.BALL_RADIUS * 2
        )

        # Wall collisions
        if ball_rect.left <= 0:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -1
        if ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -1
        if ball_rect.top <= 0:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -1

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # Sound effect placeholder: # pygame.mixer.Sound('paddle_hit.wav').play()
            
            # Change ball angle based on where it hit the paddle
            relative_intersect = (self.paddle.centerx - self.ball_pos[0]) / (self.PADDLE_WIDTH / 2)
            bounce_angle = math.radians(relative_intersect * 60) # Max 60 degree angle change
            
            new_vel_x = -self.ball_speed * math.sin(bounce_angle)
            new_vel_y = -self.ball_speed * math.cos(bounce_angle)
            
            self.ball_vel = np.array([new_vel_x, new_vel_y])
            
            # Ensure ball is pushed out of paddle to prevent sticking
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # Sound effect placeholder: # pygame.mixer.Sound('block_break.wav').play()
                
                # Determine collision side to correctly reflect
                prev_ball_pos = self.ball_pos - self.ball_vel
                
                # Check horizontal collision
                if (prev_ball_pos[0] <= block['rect'].left or prev_ball_pos[0] >= block['rect'].right):
                    self.ball_vel[0] *= -1
                # Check vertical collision
                else:
                    self.ball_vel[1] *= -1

                reward += block['points']
                self._spawn_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)
                self.blocks_destroyed_count += 1
                
                # Difficulty scaling
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 50 == 0:
                    self.ball_speed += 0.05
                    norm = np.linalg.norm(self.ball_vel)
                    if norm > 0:
                        self.ball_vel = (self.ball_vel / norm) * self.ball_speed
                break

        # Bottom wall / Lose life
        if ball_rect.top >= self.SCREEN_HEIGHT:
            # Sound effect placeholder: # pygame.mixer.Sound('lose_life.wav').play()
            self.lives -= 1
            reward -= 10
            if self.lives > 0:
                self._reset_ball()

        return reward

    def _reset_ball(self):
        self.ball_pos = np.array([self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 5], dtype=np.float64)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.ball_speed

    def _generate_blocks(self):
        self.blocks = []
        num_rows = 6
        num_cols = 12
        block_width = self.SCREEN_WIDTH / num_cols
        block_height = 20
        top_offset = 50
        
        for i in range(num_rows):
            for j in range(num_cols):
                # Add some spacing
                if self.np_random.random() < 0.1:
                    continue
                
                is_special = self.np_random.random() < 0.2
                
                # Procedural colors based on row
                hue = (i / num_rows) * 120  # Green to Red
                saturation = 0.6 if not is_special else 0.9
                value = 0.8 if not is_special else 1.0
                
                color = pygame.Color(0)
                color.hsva = (hue, saturation * 100, value * 100, 100)
                
                block_rect = pygame.Rect(
                    j * block_width,
                    top_offset + i * block_height,
                    block_width,
                    block_height
                )
                self.blocks.append({
                    "rect": block_rect,
                    "color": color,
                    "points": 2 if is_special else 1,
                    "special": is_special
                })

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'color': color})

    def _get_observation(self):
        # --- Render all game elements ---
        self._render_background()
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            # Simple vertical gradient
            interp = y / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['lifetime'] / 30))))
                size = max(1, int(self.BALL_RADIUS * 0.5 * (p['lifetime'] / 30)))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), size, (*p['color'][:3], alpha)
                )

        # Blocks
        for block in self.blocks:
            r = block['rect']
            color = block['color']
            # Flashing effect for special blocks
            if block['special'] and (self.steps // 5) % 2 == 0:
                color = color.lerp((255, 255, 255), 0.5)
            
            pygame.draw.rect(self.screen, color, r)
            pygame.draw.rect(self.screen, self.COLOR_BG_BOTTOM, r, 1) # Border

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        x, y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, x, y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_icon_width = self.PADDLE_WIDTH / 4
        life_icon_height = self.PADDLE_HEIGHT / 2
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - (i + 1) * (life_icon_width + 5) - 5
            y = 15
            pygame.draw.rect(
                self.screen, 
                self.COLOR_PADDLE, 
                (x, y, life_icon_width, life_icon_height),
                border_radius=2
            )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }
        
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)
    
    terminated = False
    total_reward = 0
    
    for i in range(2000):
        action = env.action_space.sample() # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (i+1) % 500 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")

        if terminated:
            print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            
    env.close()
    print("Environment closed.")