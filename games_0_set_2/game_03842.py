
# Generated: 2025-08-28T00:36:05.679486
# Source Brief: brief_03842.md
# Brief Index: 3842

        
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
        "Controls: Use ← and → to move the paddle. Press Space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where strategic paddle positioning and risky plays are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        
        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_GLOW = (200, 200, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (0, 255, 128),  # Green
            20: (0, 128, 255),  # Blue
            30: (255, 50, 50),   # Red
        }

        # Game element properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 15
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 2.0
        self.BALL_SPEED_INCREMENT = 0.05
        
        self.BLOCK_ROWS, self.BLOCK_COLS = 10, 10
        self.BLOCK_WIDTH, self.BLOCK_HEIGHT = 60, 20
        self.BLOCK_GAP = 4
        self.TOTAL_BLOCKS = self.BLOCK_ROWS * self.BLOCK_COLS

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.np_random = None
        self.paddle_pos_x = 0
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.ball_speed = 0
        self.ball_launched = False
        self.blocks = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.chain_reaction_timer = 0
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        self.blocks_destroyed_count = 0
        self.chain_reaction_timer = 0
        
        self.particles = []
        self._create_blocks()
        self._reset_ball_and_paddle()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02  # Time penalty to encourage faster play
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        self._handle_input(movement, space_pressed)
        
        if self.ball_launched:
            reward += self._update_ball()
        else:
            # Keep ball on paddle before launch
            self.ball_pos[0] = self.paddle_pos_x + self.PADDLE_WIDTH / 2
            self.ball_pos[1] = self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 1

        self._update_particles()
        
        if self.chain_reaction_timer > 0:
            self.chain_reaction_timer -= 1
            
        # Paddle alignment reward
        if self.ball_launched and self.ball_vel[1] > 0:
            try:
                time_to_paddle = (self.HEIGHT - self.PADDLE_HEIGHT - self.ball_pos[1]) / self.ball_vel[1]
                if 0 < time_to_paddle < 200: # Only if ball is reasonably close
                    projected_x = self.ball_pos[0] + self.ball_vel[0] * time_to_paddle
                    paddle_center_x = self.paddle_pos_x + self.PADDLE_WIDTH / 2
                    if abs(projected_x - paddle_center_x) < self.PADDLE_WIDTH / 2:
                        reward += 0.1
            except ZeroDivisionError:
                pass

        self.steps += 1
        terminated = self._check_termination()
        
        if self.game_over:
            if self.lives <= 0:
                reward += -100
            elif self.blocks_destroyed_count == self.TOTAL_BLOCKS:
                reward += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle_pos_x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos_x += self.PADDLE_SPEED
        self.paddle_pos_x = np.clip(self.paddle_pos_x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # Launch Ball
        if space_pressed and not self.ball_launched and self.lives > 0:
            self.ball_launched = True
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            self.ball_vel = np.array([math.cos(angle), math.sin(angle)]) * self.ball_speed
            # sfx: launch_ball

    def _update_ball(self):
        reward = 0
        
        # Store old position
        old_pos = self.ball_pos.copy()
        
        # Update position
        self.ball_pos += self.ball_vel
        
        ball_rect = pygame.Rect(
            int(self.ball_pos[0] - self.BALL_RADIUS),
            int(self.ball_pos[1] - self.BALL_RADIUS),
            int(self.BALL_RADIUS * 2),
            int(self.BALL_RADIUS * 2)
        )
        
        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce
            
        # Bottom wall (lose life)
        if self.ball_pos[1] >= self.HEIGHT:
            self.lives -= 1
            reward -= 20
            if self.lives > 0:
                self._reset_ball_and_paddle()
            else:
                self.game_over = True
            # sfx: lose_life
            return reward

        # Paddle collision
        paddle_rect = pygame.Rect(int(self.paddle_pos_x), self.HEIGHT - self.PADDLE_HEIGHT, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vel[1] > 0:
            self.ball_pos[1] = self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 1
            
            offset = (self.ball_pos[0] - (self.paddle_pos_x + self.PADDLE_WIDTH / 2)) / (self.PADDLE_WIDTH / 2)
            offset = np.clip(offset, -0.9, 0.9) # Prevent extreme horizontal angles
            
            new_vx = self.ball_speed * offset
            new_vy_sq = self.ball_speed**2 - new_vx**2
            new_vy = -math.sqrt(max(0.1, new_vy_sq)) # Ensure it always has some upward velocity
            
            self.ball_vel = np.array([new_vx, new_vy], dtype=np.float32)
            # sfx: paddle_hit
            
        # Block collisions
        for block in self.blocks:
            if block['alive'] and ball_rect.colliderect(block['rect']):
                # sfx: block_break
                block['alive'] = False
                reward += block['value']
                self._spawn_particles(ball_rect.center, block['color'])
                self.blocks_destroyed_count += 1
                
                # Update ball speed
                speed_milestones = self.blocks_destroyed_count // 20
                prev_milestones = (self.blocks_destroyed_count - 1) // 20
                if speed_milestones > prev_milestones:
                    self.ball_speed += self.BALL_SPEED_INCREMENT
                
                # Chain reaction bonus
                if self.chain_reaction_timer > 0:
                    reward += 5
                self.chain_reaction_timer = 15 # Reset timer
                
                # Bounce logic
                self.ball_pos = old_pos # Revert to pre-collision position
                
                # Determine if it was a horizontal or vertical collision
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel[0] *= -1
                else:
                    self.ball_vel[1] *= -1
                
                # Apply one step of movement to exit the block
                self.ball_pos += self.ball_vel
                break
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self.blocks_destroyed_count == self.TOTAL_BLOCKS:
            self.game_over = True
            return True
        return False

    def _create_blocks(self):
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_GAP) - self.BLOCK_GAP
        start_x = (self.WIDTH - grid_width) / 2
        start_y = 50

        for r in range(self.BLOCK_ROWS):
            for c in range(self.BLOCK_COLS):
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_GAP)
                y = start_y + r * (self.BLOCK_HEIGHT + self.BLOCK_GAP)
                
                if r < 2:
                    value, color = 30, self.BLOCK_COLORS[30]
                elif r < 6:
                    value, color = 20, self.BLOCK_COLORS[20]
                else:
                    value, color = 10, self.BLOCK_COLORS[10]

                self.blocks.append({
                    'rect': pygame.Rect(int(x), int(y), self.BLOCK_WIDTH, self.BLOCK_HEIGHT),
                    'color': color,
                    'value': value,
                    'alive': True
                })

    def _reset_ball_and_paddle(self):
        self.paddle_pos_x = self.WIDTH / 2 - self.PADDLE_WIDTH / 2
        self.ball_launched = False
        self.ball_speed = self.INITIAL_BALL_SPEED + (self.blocks_destroyed_count // 20) * self.BALL_SPEED_INCREMENT
        self.ball_pos = np.array([
            self.paddle_pos_x + self.PADDLE_WIDTH / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - self.BALL_RADIUS - 1
        ], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.chain_reaction_timer = 0

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            vel = self.np_random.uniform(-2.5, 2.5, size=2)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(pos), 'vel': list(vel), 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw blocks
        for block in self.blocks:
            if block['alive']:
                pygame.draw.rect(self.screen, block['color'], block['rect'])
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Draw paddle
        paddle_rect = pygame.Rect(int(self.paddle_pos_x), self.HEIGHT - self.PADDLE_HEIGHT, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        
        # Draw ball
        ball_pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, (*self.COLOR_GLOW, 60))
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 40))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 2, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            pos_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS*2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 22, self.BALL_RADIUS, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 22, self.BALL_RADIUS, self.COLOR_PADDLE)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_or_lose_text = "LEVEL CLEARED!" if self.blocks_destroyed_count == self.TOTAL_BLOCKS else "GAME OVER"
            text_surface = self.font_large.render(win_or_lose_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text_surface, text_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, (200, 200, 200))
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": self.TOTAL_BLOCKS - self.blocks_destroyed_count,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Match the intended frame rate
        
    env.close()