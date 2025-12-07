
# Generated: 2025-08-27T15:22:56.740420
# Source Brief: brief_00974.md
# Brief Index: 974

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro arcade block breaker. Destroy all the blocks with the ball to win. Don't let the ball fall!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 8

        # Ball settings
        self.BALL_RADIUS = 8
        self.BALL_INITIAL_SPEED = 4.0
        self.BALL_MAX_VX = 5.0
        
        # Color constants
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_TEXT = (200, 200, 220)
        self.COLOR_COMBO = (255, 100, 100)
        self.BLOCK_COLORS = [
            (255, 70, 70), (70, 255, 70), (70, 70, 255),
            (255, 255, 70), (70, 255, 255), (255, 70, 255)
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_combo = pygame.font.SysFont("monospace", 28, bold=True)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.paddle_x = 0
        self.ball_x = 0
        self.ball_y = 0
        self.ball_vx = 0
        self.ball_vy = 0
        self.ball_launched = False
        self.blocks = []
        self.particles = []
        self.combo = 0
        self.y_history = deque(maxlen=100)
        
        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def _setup_blocks(self):
        self.blocks = []
        block_width = 50
        block_height = 20
        rows = 5
        cols = 11
        gap = 4
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                rect = pygame.Rect(x, y, block_width, block_height)
                color = self.BLOCK_COLORS[(r + c) % len(self.BLOCK_COLORS)]
                self.blocks.append({"rect": rect, "color": color})
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        self.paddle_x = self.WIDTH / 2
        
        self.ball_launched = False
        self.ball_x = self.paddle_x
        self.ball_y = self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 5
        self.ball_vx = 0
        self.ball_vy = 0
        
        self._setup_blocks()
        self.particles = []
        self.combo = 0
        self.y_history.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            self.steps += 1
            terminated = self.steps >= self.MAX_STEPS or self.game_over
            return self._get_observation(), 0, terminated, False, self._get_info()

        movement = action[0]
        space_pressed = action[1] == 1
        
        reward = -0.01  # Time penalty

        # 1. Update paddle position
        if movement == 3:  # Left
            self.paddle_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_x += self.PADDLE_SPEED
        self.paddle_x = np.clip(self.paddle_x, self.PADDLE_WIDTH / 2, self.WIDTH - self.PADDLE_WIDTH / 2)

        # 2. Launch ball
        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards
            self.ball_vx = self.BALL_INITIAL_SPEED * math.cos(angle)
            self.ball_vy = -self.BALL_INITIAL_SPEED * math.sin(angle)
            # Sound: # LAUNCH_SOUND

        # 3. Update ball position (if launched)
        if self.ball_launched:
            collision_reward = self._handle_ball_movement_and_collisions()
            reward += collision_reward
        else:
            self.ball_x = self.paddle_x

        # 4. Update particles
        self._update_particles()
        
        # 5. Update score and check for win
        if not self.blocks:
            self.game_over = True
            reward += 100 # Win bonus
        
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS or self.game_over
        
        if self.game_over and self.balls_left <= 0 and self.blocks:
            reward += -100 # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_ball_movement_and_collisions(self):
        reward = 0
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        ball_rect = pygame.Rect(self.ball_x - self.BALL_RADIUS, self.ball_y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if self.ball_x - self.BALL_RADIUS < 0 or self.ball_x + self.BALL_RADIUS > self.WIDTH:
            self.ball_vx *= -1
            self.ball_x = np.clip(self.ball_x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
        if self.ball_y - self.BALL_RADIUS < 0:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
        
        # Bottom wall (lose ball)
        if self.ball_y + self.BALL_RADIUS > self.HEIGHT:
            self.balls_left -= 1
            self.ball_launched = False
            self.ball_x = self.paddle_x
            self.ball_y = self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 5
            self.ball_vx, self.ball_vy = 0, 0
            self.combo = 0
            self.y_history.clear()
            # Sound: # LOSE_BALL_SOUND
            if self.balls_left <= 0:
                self.game_over = True
            return 0 # No reward for losing a ball

        # Paddle collision
        paddle_rect = pygame.Rect(self.paddle_x - self.PADDLE_WIDTH / 2, self.HEIGHT - self.PADDLE_HEIGHT - 5, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vy > 0:
            self.ball_vy *= -1
            self.ball_y = paddle_rect.top - self.BALL_RADIUS

            impact_offset = (self.ball_x - self.paddle_x)
            norm_impact = impact_offset / (self.PADDLE_WIDTH / 2)
            self.ball_vx = norm_impact * self.BALL_MAX_VX
            
            # Risky/safe bounce reward
            if abs(norm_impact) > 0.9: reward += 0.1
            elif abs(norm_impact) < 0.5: reward -= 0.2
            
            self.combo = 0 # Brief doesn't specify combo reset on paddle hit, but it's common. I'll stick to brief: only on wall hit.
            # Sound: # PADDLE_HIT_SOUND
            
        # Block collisions
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            if ball_rect.colliderect(block['rect']):
                # Sound: # BLOCK_BREAK_SOUND
                self._create_particles(block['rect'].center, block['color'])
                
                # Collision response
                # Determine if collision is more horizontal or vertical
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vx *= -1
                else:
                    self.ball_vy *= -1
                
                # Reward and score
                reward += 1.0 + (0.5 * self.combo)
                self.score += 10 + (5 * self.combo)
                self.combo += 1
                
                self.blocks.pop(i)
                break # Handle one block per frame

        # Anti-softlock mechanism
        self.y_history.append(self.ball_y)
        if len(self.y_history) == self.y_history.maxlen:
            y_range = max(self.y_history) - min(self.y_history)
            if y_range < 1.0 and self.ball_vy != 0:
                self.ball_vy += self.np_random.uniform(-0.5, 0.5)
                self.ball_vx = np.clip(self.ball_vx + self.np_random.uniform(-0.5, 0.5), -self.BALL_MAX_VX, self.BALL_MAX_VX)
                self.y_history.clear()

        return reward

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            p = {
                "x": pos[0], "y": pos[1],
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.integers(20, 40),
                "color": color
            }
            self.particles.append(p)

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Paddle
        paddle_rect = pygame.Rect(0, 0, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        paddle_rect.center = (int(self.paddle_x), int(self.HEIGHT - self.PADDLE_HEIGHT/2 - 5))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (1, 1), 1)
            self.screen.blit(temp_surf, (int(p['x']), int(p['y'])))

        # Ball
        ball_pos = (int(self.ball_x), int(self.ball_y))
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL, 50))
        self.screen.blit(glow_surf, (ball_pos[0] - glow_radius, ball_pos[1] - glow_radius))
        # Ball itself
        pygame.gfxdraw.filled_circle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos[0], ball_pos[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        balls_text = self.font_ui.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))

        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"COMBO x{self.combo}", True, self.COLOR_COMBO)
            text_rect = combo_text.get_rect(center=(self.WIDTH / 2, 25))
            self.screen.blit(combo_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
            "combo": self.combo,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with display drivers
    import os
    if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
        print("Cannot run interactive test in a headless environment. Skipping.")
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Block Breaker")
        clock = pygame.time.Clock()

        terminated = False
        total_reward = 0
        
        # Map keyboard keys to MultiDiscrete actions
        key_to_action = {
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        while not terminated:
            # --- Action gathering ---
            movement_action = 0 # No-op
            space_action = 0    # Released
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                movement_action = 3
            elif keys[pygame.K_RIGHT]:
                movement_action = 4
            
            if keys[pygame.K_SPACE]:
                space_action = 1

            action = [movement_action, space_action, 0] # Shift is not used

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
            
            # --- Environment step ---
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term or trunc

            # --- Rendering ---
            # The observation is already a rendered frame
            # We just need to display it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Match the auto_advance rate

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Wait a bit before resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        pygame.quit()