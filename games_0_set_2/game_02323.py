
# Generated: 2025-08-28T04:26:14.294803
# Source Brief: brief_02323.md
# Brief Index: 2323

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded. Clear all the blocks to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAY_AREA_Y_START = 50
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_msg = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_BALL = (255, 0, 100)
        self.COLOR_WALL = (150, 150, 170)
        self.BLOCK_COLORS = [
            (255, 220, 0), (0, 255, 150), (255, 100, 0),
            (200, 0, 255), (0, 200, 50)
        ]
        self.COLOR_TEXT = (255, 255, 255)

        # Game parameters
        self.PADDLE_SPEED = 10
        self.BALL_BASE_SPEED = 6
        self.MAX_BALL_SPEED_X = 7
        self.MAX_STEPS = 5000
        self.INITIAL_LIVES = 3
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_attached = True
        
        self.blocks = []
        self.block_colors = []
        
        self.particles = []
        self.just_hit_paddle_edge = False
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False
        self.particles.clear()
        
        # Paddle
        paddle_w, paddle_h = 100, 15
        paddle_x = (self.WIDTH - paddle_w) / 2
        paddle_y = self.HEIGHT - 30
        self.paddle = pygame.Rect(paddle_x, paddle_y, paddle_w, paddle_h)
        
        # Ball
        ball_size = 12
        self.ball = pygame.Rect(0, 0, ball_size, ball_size)
        self.ball_vel = [0, 0]
        self.ball_attached = True
        self._attach_ball_to_paddle()

        # Blocks
        self.blocks.clear()
        self.block_colors.clear()
        num_cols, num_rows = 10, 5
        block_w, block_h = 58, 20
        gap = 6
        start_x = (self.WIDTH - (num_cols * (block_w + gap) - gap)) / 2
        start_y = self.PLAY_AREA_Y_START + 30
        for r in range(num_rows):
            for c in range(num_cols):
                color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                block = pygame.Rect(
                    start_x + c * (block_w + gap),
                    start_y + r * (block_h + gap),
                    block_w, block_h
                )
                self.blocks.append(block)
                self.block_colors.append(color)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, no-op but still return observation
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = self._update_game_logic(action)
        self.steps += 1
        
        # Check termination conditions
        terminated = False
        if self.lives <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        elif not self.blocks:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_game_logic(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle Paddle Movement
        paddle_moved = self._handle_paddle_movement(movement)
        
        # 2. Handle Ball Launch
        if self.ball_attached and space_held:
            # sfx: ball_launch.wav
            self.ball_attached = False
            launch_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [
                self.BALL_BASE_SPEED * math.cos(launch_angle),
                self.BALL_BASE_SPEED * math.sin(launch_angle)
            ]
            self.just_hit_paddle_edge = False

        # 3. Update Ball and Particles
        collision_reward = 0
        if not self.ball_attached:
            collision_reward = self._handle_ball_movement()
        else:
            self._attach_ball_to_paddle()
            
        self._update_particles()
        
        # 4. Calculate step reward
        reward = collision_reward
        if movement == 0:
            reward -= 0.02
        if not self.ball_attached and not paddle_moved:
            reward -= 0.2
            
        return reward

    def _handle_paddle_movement(self, movement):
        moved = False
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
            moved = True
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
            moved = True
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.paddle.width, self.paddle.x))
        return moved
        
    def _handle_ball_movement(self):
        reward = 0
        self.ball.move_ip(self.ball_vel)

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.WIDTH, self.ball.right)
            # sfx: wall_bounce.wav
        if self.ball.top <= self.PLAY_AREA_Y_START:
            self.ball_vel[1] *= -1
            self.ball.top = max(self.PLAY_AREA_Y_START, self.ball.top)
            # sfx: wall_bounce.wav

        # Bottom wall (lose life)
        if self.ball.top >= self.HEIGHT:
            self.lives -= 1
            self.ball_attached = True
            self._attach_ball_to_paddle()
            # sfx: lose_life.wav
        
        # Paddle collision
        if not self.ball_attached and self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit.wav
            self.ball.bottom = self.paddle.top
            
            # Calculate hit position on paddle (-1 for left edge, 1 for right edge)
            hit_pos = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            hit_pos = max(-1, min(1, hit_pos))
            
            # Change horizontal velocity based on hit position
            self.ball_vel[0] = hit_pos * self.MAX_BALL_SPEED_X
            self.ball_vel[1] *= -1
            
            # Check for edge hit for bonus reward
            self.just_hit_paddle_edge = abs(hit_pos) > 0.8
        
        # Block collision
        collided_idx = self.ball.collidelist(self.blocks)
        if collided_idx != -1:
            block = self.blocks[collided_idx]
            
            # Determine collision side
            prev_ball_rect = self.ball.copy()
            prev_ball_rect.move_ip(-self.ball_vel[0], -self.ball_vel[1])

            if prev_ball_rect.bottom <= block.top or prev_ball_rect.top >= block.bottom:
                self.ball_vel[1] *= -1
            else:
                self.ball_vel[0] *= -1
            
            # Pop block and its color
            color = self.block_colors.pop(collided_idx)
            self.blocks.pop(collided_idx)
            
            # Create particles and update score
            self._create_particles(block.center, color)
            self.score += 1
            reward += 1
            # sfx: block_break.wav
            
            # Check for skill shot bonus
            if self.just_hit_paddle_edge:
                reward += 5
                # sfx: skill_shot.wav
            
            # Reset edge hit flag after one use
            self.just_hit_paddle_edge = False

        return reward

    def _attach_ball_to_paddle(self):
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top - 2

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
        # Draw play area boundary
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.PLAY_AREA_Y_START, self.WIDTH, self.HEIGHT - self.PLAY_AREA_Y_START), 2)
        
        # Draw blocks
        for i, block in enumerate(self.blocks):
            pygame.draw.rect(self.screen, self.block_colors[i], block)
            
        # Draw particles
        self._render_particles()

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Draw ball with glow
        center = (int(self.ball.centerx), int(self.ball.centery))
        radius = self.ball.width // 2
        for i in range(radius, 0, -2):
            alpha = int(150 * (i / radius)**2)
            color = (*self.COLOR_BALL, alpha)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], i + 3, color)
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 15))

        # Lives
        lives_text_surf = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text_surf, (self.WIDTH - 150, 15))
        for i in range(self.lives):
            pos = (self.WIDTH - 80 + i * 25, 23)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_BALL)

        # Game over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 120) if self.win else (255, 50, 50)
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)
        elif self.ball_attached:
            msg_surf = self.font_ui.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 70))
            self.screen.blit(msg_surf, msg_rect)
            
    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # a little gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(3 * (p['life'] / 30)))
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.draw.rect(self.screen, p['color'], (*pos, size, size))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a display window
    pygame.display.set_caption("Block Breaker")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Map keyboard keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # Default action is no-op
        movement = 0
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_ESCAPE]:
            terminated = True
            
        action = np.array([movement, space_held, 0]) # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        env.clock.tick(30)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
    pygame.quit()
    sys.exit()