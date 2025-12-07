
# Generated: 2025-08-28T01:20:23.847368
# Source Brief: brief_04080.md
# Brief Index: 4080

        
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
    """
    A fast-paced, procedurally generated block breaker where risky plays are rewarded.
    The player controls a paddle to bounce a ball, breaking colored blocks.
    The episode ends when all blocks are broken, all balls are lost, or a step limit is reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade block breaker. Break all the blocks with your 3 balls. "
        "Risky edge-shots are rewarded!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 10
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 4.5
        self.BALL_MAX_X_VEL = 5.5
        self.WALL_THICKNESS = 10

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_WALL = (40, 40, 60)
        self.COLOR_PADDLE = (220, 220, 220)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (200, 200, 255)
        self.BLOCK_COLORS = {
            1: (217, 87, 99),   # Red
            2: (99, 217, 87),   # Green
            3: (87, 99, 217),   # Blue
        }
        self.COLOR_TEXT = (240, 240, 240)

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
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_launched = False
        self.balls_left = 0
        self.blocks = []
        self.particles = []
        self.blocks_destroyed_count = 0
        
        # Call reset to set up the initial state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        self.blocks_destroyed_count = 0
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._spawn_ball()
        self._create_blocks()
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Using pressed, not held, for launch
        
        # Initialize reward for this step
        reward = -0.02  # Small penalty per step to encourage speed

        # --- Handle Input ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle position
        self.paddle.x = max(self.WALL_THICKNESS, min(self.paddle.x, self.WIDTH - self.PADDLE_WIDTH - self.WALL_THICKNESS))

        # Launch ball
        if space_pressed and not self.ball_launched:
            self.ball_launched = True
            # sfx: launch_ball
            initial_x_vel = self.np_random.uniform(-0.5, 0.5)
            self.ball_vel = pygame.Vector2(initial_x_vel, -self.INITIAL_BALL_SPEED)
            self.ball_vel.scale_to_length(self.INITIAL_BALL_SPEED)

        # --- Update Game Logic ---
        physics_reward = self._handle_ball_movement_and_collisions()
        reward += physics_reward
        self._update_particles()
        
        # --- Check Termination Conditions ---
        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if not self.blocks and not self.game_over:
            reward += 100  # Win bonus
            self.game_over = True
            terminated = True
        
        if self.game_over and self.balls_left < 0:
            reward -= 100 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_ball(self):
        self.ball_launched = False
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        # Increase ball speed for the next life to avoid getting stuck
        self.INITIAL_BALL_SPEED = min(8, self.INITIAL_BALL_SPEED + 0.2)

    def _create_blocks(self):
        self.blocks = []
        block_width, block_height = 40, 15
        gap = 5
        rows, cols = 5, 12
        start_x = self.WALL_THICKNESS + 25
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                points = self.np_random.integers(1, 4)
                block_rect = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "points": points, "color": self.BLOCK_COLORS[points]})

    def _handle_ball_movement_and_collisions(self):
        if not self.ball_launched:
            self.ball_pos.x = self.paddle.centerx
            return 0

        reward = 0
        
        # Move ball
        self.ball_pos += self.ball_vel

        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left < self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.left = self.WALL_THICKNESS
            # sfx: wall_bounce
        if ball_rect.right > self.WIDTH - self.WALL_THICKNESS:
            self.ball_vel.x *= -1
            ball_rect.right = self.WIDTH - self.WALL_THICKNESS
            # sfx: wall_bounce
        if ball_rect.top < 0:
            self.ball_vel.y *= -1
            ball_rect.top = 0
            # sfx: wall_bounce

        # Bottom edge - lose ball
        if ball_rect.top > self.HEIGHT:
            self.balls_left -= 1
            reward -= 5
            # sfx: lose_ball
            if self.balls_left < 0:
                self.game_over = True
            else:
                self._spawn_ball()
            return reward

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # sfx: paddle_bounce
            offset = self.ball_pos.x - self.paddle.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            
            # Reward for being close to impact point
            if abs(offset) < 10:
                reward += 0.1
            # Reward for risky edge hits
            if abs(normalized_offset) > 0.9:
                reward += 2.0
            
            self.ball_vel.x = self.BALL_MAX_X_VEL * normalized_offset
            self.ball_vel.y *= -1
            self.ball_vel.scale_to_length(self.INITIAL_BALL_SPEED * (1 + 0.05 * (self.blocks_destroyed_count // 10)))
            
            ball_rect.bottom = self.paddle.top
            self.ball_pos.y = ball_rect.centery

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block["rect"]):
                # sfx: block_break
                reward += block["points"]
                self.score += block["points"]
                self._create_particles(block["rect"].center, block["color"])
                
                # Determine collision side to reflect correctly
                prev_ball_rect = pygame.Rect(ball_rect)
                prev_ball_rect.center = (self.ball_pos - self.ball_vel)
                
                if prev_ball_rect.bottom <= block["rect"].top or prev_ball_rect.top >= block["rect"].bottom:
                    self.ball_vel.y *= -1
                else:
                    self.ball_vel.x *= -1
                
                self.blocks.remove(block)
                self.blocks_destroyed_count += 1
                
                # Speed up ball every 10 blocks
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                    current_speed = self.ball_vel.length()
                    self.ball_vel.scale_to_length(current_speed * 1.05)
                
                break # Only handle one block collision per frame

        self.ball_pos.x = ball_rect.centerx
        self.ball_pos.y = ball_rect.centery
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            size = self.np_random.uniform(2, 6)
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'size': size, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            p['size'] -= 0.1
            if p['life'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))
        
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block["color"], block["rect"])
            pygame.draw.rect(self.screen, self.COLOR_BG, block["rect"], 1)

        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p['color'], (p['pos'].x, p['pos'].y, max(0, p['size']), max(0, p['size'])))

        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)

        # Draw ball with glow
        if self.ball_pos:
            ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
            pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS + 3, (*self.COLOR_BALL_GLOW, 80))
            pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
            
    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 15, 10))

        balls_text = self.font_main.render(f"BALLS: {max(0, self.balls_left)}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - self.WALL_THICKNESS - 15, 10))

        if self.game_over:
            if not self.blocks:
                end_text = "YOU WIN!"
            else:
                end_text = "GAME OVER"
            
            end_surf = self.font_main.render(end_text, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_surf, end_rect)

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Game loop
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if info.get('score', 0) != 0:
            pass # You can print reward/score here for debugging if you want
        
        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()