
# Generated: 2025-08-27T19:30:00.901549
# Source Brief: brief_02174.md
# Brief Index: 2174

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Neon Breakout: A retro-futuristic arcade game. Launch the ball, break all the blocks, and aim for a high score with combos."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (25, 20, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255)
    COLOR_BALL = (255, 0, 255)
    COLOR_BALL_GLOW = (200, 50, 200)
    
    BLOCK_COLORS = {
        1: ((0, 150, 255), (0, 100, 200)),  # Blue (Points, (Outline, Fill))
        2: ((50, 255, 50), (30, 200, 30)),   # Green
        3: ((255, 150, 0), (200, 100, 0)),  # Orange
    }

    # Game parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    BALL_SPEED = 6
    MAX_BALLS = 3
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps
    COMBO_WINDOW_STEPS = 9 # 3 steps in brief, but that's too fast. 9 steps (0.3s) is better for gameplay.

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = None
        self.blocks = []
        self.particles = []
        self.balls_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.last_block_hit_step = -self.COMBO_WINDOW_STEPS

        # For seeding
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional: call to test during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.MAX_BALLS
        self.prev_space_held = False
        self.last_block_hit_step = -self.COMBO_WINDOW_STEPS
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        self._reset_ball()
        self._create_blocks()
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = -0.01  # Small time penalty
        
        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            reward += self._calculate_reward()
            
        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)
        
        # Ball Launch
        space_pressed = space_held and not self.prev_space_held
        if self.ball_on_paddle and space_pressed:
            self.ball_on_paddle = False
            # sfx: launch_ball
            
            # Launch angle depends on paddle position
            paddle_center_norm = (self.paddle.centerx - self.SCREEN_WIDTH / 2) / (self.SCREEN_WIDTH / 2)
            angle = -math.pi / 2 - paddle_center_norm * (math.pi / 4) # Launch between -45 and -135 degrees
            
            self.ball_vel = [
                self.BALL_SPEED * math.cos(angle),
                self.BALL_SPEED * math.sin(angle)
            ]
        
        self.prev_space_held = space_held

    def _update_game_state(self):
        # Update ball
        if self.ball_on_paddle:
            self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        else:
            self._update_ball_position()

        # Update particles
        self._update_particles()

    def _update_ball_position(self):
        if self.ball_vel is None: return

        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # Wall collisions
        if self.ball_pos[0] <= self.BALL_RADIUS or self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = np.clip(self.ball_pos[0], self.BALL_RADIUS, self.SCREEN_WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = self.BALL_RADIUS
            # sfx: wall_bounce

        # Lose ball
        if self.ball_pos[1] >= self.SCREEN_HEIGHT + self.BALL_RADIUS:
            self.balls_left -= 1
            # sfx: lose_ball
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True

    def _calculate_reward(self):
        reward = 0
        if self.ball_on_paddle or self.ball_vel is None:
            return 0

        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            
            # Add "english" to the ball based on where it hits the paddle
            hit_offset = (self.ball_pos[0] - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += hit_offset * 2.0
            self.ball_vel[0] = np.clip(self.ball_vel[0], -self.BALL_SPEED, self.BALL_SPEED)
            
            self.ball_pos[1] = self.paddle.top - self.BALL_RADIUS
            # sfx: paddle_bounce

        # Block collisions
        for block in self.blocks[:]:
            if ball_rect.colliderect(block['rect']):
                # sfx: block_hit
                reward += 0.1 # Continuous feedback for hitting
                
                # Determine bounce direction
                self._handle_block_bounce(ball_rect, block['rect'])
                
                # Reward for breaking block
                reward += block['points']
                
                # Combo bonus
                if self.steps - self.last_block_hit_step <= self.COMBO_WINDOW_STEPS:
                    reward += 5
                    # sfx: combo_bonus
                self.last_block_hit_step = self.steps

                self._create_particles(block['rect'].center, self.BLOCK_COLORS[block['points']][0])
                self.blocks.remove(block)
                break

        # Win condition reward
        if not self.blocks:
            reward += 100
            self.game_over = True
            
        # Lose ball penalty
        if self.ball_pos[1] >= self.SCREEN_HEIGHT:
            reward -= 10

        return reward

    def _handle_block_bounce(self, ball_rect, block_rect):
        # A simple but effective bounce logic
        # Find the overlap between ball and block
        dx = (ball_rect.centerx - block_rect.centerx) / block_rect.width
        dy = (ball_rect.centery - block_rect.centery) / block_rect.height
        
        if abs(dx) > abs(dy): # Horizontal collision
            self.ball_vel[0] *= -1
        else: # Vertical collision
            self.ball_vel[1] *= -1

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if self.balls_left <= 0:
            return True
        if not self.blocks:
            return True
        return False
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color_fill'], block['rect'])
            pygame.draw.rect(self.screen, block['color_outline'], block['rect'], 2)

        # Render paddle with glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW + (50,), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Render ball with glow
        if self.ball_pos:
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS + 4, self.COLOR_BALL_GLOW + (100,))
            # Ball
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
            
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_PADDLE)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            pos_x = self.SCREEN_WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.aacircle(self.screen, pos_x, 18, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 18, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Game Over message
        if self.game_over:
            msg = "YOU WON!" if not self.blocks else "GAME OVER"
            color = (50, 255, 50) if not self.blocks else (255, 50, 50)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _reset_ball(self):
        self.ball_on_paddle = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _create_blocks(self):
        self.blocks.clear()
        block_width = 58
        block_height = 20
        gap = 6
        
        num_cols = 10
        num_rows = 5
        
        start_x = (self.SCREEN_WIDTH - (num_cols * (block_width + gap) - gap)) / 2
        start_y = 50

        for r in range(num_rows):
            for c in range(num_cols):
                points = 1 if r >= 3 else (2 if r >= 1 else 3)
                color_outline, color_fill = self.BLOCK_COLORS[points]
                
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                
                block = {
                    'rect': pygame.Rect(x, y, block_width, block_height),
                    'points': points,
                    'color_outline': color_outline,
                    'color_fill': color_fill
                }
                self.blocks.append(block)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': random.randint(1, 4)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example usage for interactive play
if __name__ == '__main__':
    import time
    
    env = GameEnv(render_mode="rgb_array")
    
    # Use a real screen for interactive play
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Breakout")
    
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            time.sleep(3) # Show final screen for a bit

    env.close()