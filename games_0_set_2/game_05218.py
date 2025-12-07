
# Generated: 2025-08-28T04:21:00.793045
# Source Brief: brief_05218.md
# Brief Index: 5218

        
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
        "Controls: ↑/↓ to rotate paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block breaker. Aim your paddle, launch the ball, and clear the screen to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Game Constants ---
        self.MAX_STEPS = 1000
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 10
        self.PADDLE_ROTATION_SPEED = 3.0  # degrees per step
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 6.0
        self.MAX_BALL_SPEED = 10.0
        self.MIN_PADDLE_ANGLE = 15
        self.MAX_PADDLE_ANGLE = 165
        self.NUM_BLOCK_ROWS = 5
        self.NUM_BLOCK_COLS = 15
        self.BLOCK_WIDTH = 38
        self.BLOCK_HEIGHT = 18
        self.BLOCK_SPACING = 4
        
        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 80, 80)
        self.COLOR_BALL_GLOW = (255, 0, 0)
        self.BLOCK_COLORS = [
            (60, 120, 180),  # Row 0 (bottom)
            (70, 140, 200),
            (80, 160, 220),
            (90, 180, 240),
            (100, 200, 255)  # Row 4 (top)
        ]
        self.TEXT_COLOR = (220, 220, 220)
        
        # --- Fonts ---
        self.FONT_UI = pygame.font.Font(None, 28)
        self.FONT_BALL_ICON = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.paddle_angle = 90.0
        self.paddle_pos = None
        self.blocks = []
        self.particles = []
        self.np_random = None

        # --- Initialize state variables and validate ---
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.ball_launched = False
        
        self.paddle_angle = 90.0
        self.paddle_pos = pygame.Vector2(self.screen_width // 2, self.screen_height - 30)
        
        self._reset_ball()
        self._generate_blocks()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle_pos.x, self.paddle_pos.y - self.PADDLE_HEIGHT - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)

    def _generate_blocks(self):
        self.blocks = []
        grid_width = self.NUM_BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.screen_width - grid_width) // 2
        start_y = 50
        
        for row in range(self.NUM_BLOCK_ROWS):
            for col in range(self.NUM_BLOCK_COLS):
                x = start_x + col * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = start_y + row * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                points = (row + 1) * 10
                color = self.BLOCK_COLORS[row]
                self.blocks.append({'rect': rect, 'points': points, 'color': color})

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        
        blocks_hit_this_step = 0
        ball_lost_this_step = False
        
        # --- Handle Input ---
        if not self.ball_launched:
            if movement == 1: # Up -> rotate counter-clockwise
                self.paddle_angle -= self.PADDLE_ROTATION_SPEED
            elif movement == 2: # Down -> rotate clockwise
                self.paddle_angle += self.PADDLE_ROTATION_SPEED
            self.paddle_angle = np.clip(self.paddle_angle, self.MIN_PADDLE_ANGLE, self.MAX_PADDLE_ANGLE)

            if space_pressed and not self.ball_launched:
                self.ball_launched = True
                launch_angle_rad = math.radians(self.paddle_angle - 90)
                self.ball_vel = pygame.Vector2(
                    math.cos(launch_angle_rad) * self.BALL_SPEED,
                    math.sin(launch_angle_rad) * self.BALL_SPEED
                )
                # sfx: ball_launch

        # --- Update Game State ---
        if self.ball_launched:
            self.ball_pos += self.ball_vel
            blocks_hit_this_step, ball_lost_this_step = self._update_ball_collisions()
        
        self._update_particles()
        
        # --- Calculate Reward ---
        reward = 0
        if blocks_hit_this_step > 0:
            reward += 1.0 * blocks_hit_this_step
        else:
            reward -= 0.02
        
        # --- Check Termination ---
        terminated = False
        if not self.blocks: # Win
            self.score += 1000 # Bonus for winning
            reward += 100
            terminated = True
            self.game_over = True
        
        if ball_lost_this_step:
            if self.balls_left <= 0: # Lose
                reward -= 100
                terminated = True
                self.game_over = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_ball_collisions(self):
        blocks_hit = 0
        ball_lost = False

        # Wall collisions
        if self.ball_pos.x - self.BALL_RADIUS < 0 or self.ball_pos.x + self.BALL_RADIUS > self.screen_width:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.screen_width - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos.y - self.BALL_RADIUS < 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.screen_height - self.BALL_RADIUS)
            # sfx: wall_bounce

        # Bottom edge (lose ball)
        if self.ball_pos.y - self.BALL_RADIUS > self.screen_height:
            self.balls_left -= 1
            ball_lost = True
            if self.balls_left > 0:
                self._reset_ball()
                # sfx: lose_ball
            else:
                # sfx: game_over
                pass
            return blocks_hit, ball_lost

        # Paddle collision
        paddle_angle_rad = math.radians(self.paddle_angle)
        half_paddle = self.PADDLE_WIDTH / 2
        p1 = self.paddle_pos + pygame.Vector2(-half_paddle * math.cos(paddle_angle_rad), half_paddle * math.sin(paddle_angle_rad))
        p2 = self.paddle_pos + pygame.Vector2(half_paddle * math.cos(paddle_angle_rad), -half_paddle * math.sin(paddle_angle_rad))
        
        line_vec = p2 - p1
        p_to_ball = self.ball_pos - p1
        t = p_to_ball.dot(line_vec) / line_vec.length_squared()
        t = np.clip(t, 0, 1)
        closest_point = p1 + t * line_vec

        if self.ball_pos.distance_to(closest_point) < self.BALL_RADIUS:
            # Reflect velocity
            normal = pygame.Vector2(math.sin(paddle_angle_rad), -math.cos(paddle_angle_rad))
            self.ball_vel = self.ball_vel.reflect(normal)
            
            # Add a slight spin based on impact point
            spin = (t - 0.5) * 0.5
            self.ball_vel.x += spin
            
            # Ensure minimum speed and cap max speed
            current_speed = self.ball_vel.length()
            if current_speed < self.BALL_SPEED:
                self.ball_vel.scale_to_length(self.BALL_SPEED)
            elif current_speed > self.MAX_BALL_SPEED:
                self.ball_vel.scale_to_length(self.MAX_BALL_SPEED)
                
            # Move ball out of paddle to prevent sticking
            self.ball_pos = closest_point + normal * self.BALL_RADIUS
            # sfx: paddle_bounce

        # Block collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                self.score += block['points']
                blocks_hit += 1
                self._create_particles(block['rect'].center, block['color'])
                self.blocks.remove(block)
                # sfx: block_break

                # Simple bounce logic
                # Determine if collision was more horizontal or vertical
                overlap = ball_rect.clip(block['rect'])
                if overlap.width < overlap.height:
                    self.ball_vel.x *= -1
                else:
                    self.ball_vel.y *= -1
                
                break # Handle one block collision per frame for simplicity

        return blocks_hit, ball_lost

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            size = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': lifespan, 'color': color, 'size': size})
    
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['size'] *= 0.95
            if p['life'] <= 0 or p['size'] < 1:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Render particles
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 30))
            p_color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p_color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Render paddle
        paddle_angle_rad = math.radians(self.paddle_angle)
        half_paddle = self.PADDLE_WIDTH / 2
        p1 = self.paddle_pos + pygame.Vector2(-half_paddle * math.cos(paddle_angle_rad), half_paddle * math.sin(paddle_angle_rad))
        p2 = self.paddle_pos + pygame.Vector2(half_paddle * math.cos(paddle_angle_rad), -half_paddle * math.sin(paddle_angle_rad))
        pygame.draw.line(self.screen, self.COLOR_PADDLE, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), self.PADDLE_HEIGHT)
        
        # Render ball
        ball_center = (int(self.ball_pos.x), int(self.ball_pos.y))
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 50))
        self.screen.blit(glow_surf, (ball_center[0] - glow_radius, ball_center[1] - glow_radius))
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_center[0], ball_center[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score}", True, self.TEXT_COLOR)
        self.screen.blit(score_text, (10, 10))
        
        # Balls left
        ball_text = self.FONT_UI.render("BALLS:", True, self.TEXT_COLOR)
        self.screen.blit(ball_text, (self.screen_width - 150, 10))
        for i in range(self.balls_left):
            pygame.gfxdraw.filled_circle(self.screen, self.screen_width - 65 + i * 20, 18, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.screen_width - 65 + i * 20, 18, 6, self.COLOR_BALL)

        if not self.ball_launched:
            # Aiming indicator
            launch_angle_rad = math.radians(self.paddle_angle - 90)
            end_pos = self.ball_pos + pygame.Vector2(math.cos(launch_angle_rad), math.sin(launch_angle_rad)) * 40
            pygame.draw.line(self.screen, (255, 255, 255, 100), self.ball_pos, end_pos, 1)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
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

if __name__ == "__main__":
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset()
    terminated = False
    
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    action = env.action_space.sample()
    action = [0, 0, 0] # Start with no-op

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]

        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        # Space
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, term, trunc, info = env.step(action)
        terminated = term
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(60) # Control the frame rate for human play

    print(f"Game Over. Final Score: {info['score']}")
    env.close()