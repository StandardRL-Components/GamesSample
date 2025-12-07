
# Generated: 2025-08-28T01:55:20.915685
# Source Brief: brief_04280.md
# Brief Index: 4280

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Neon Breakout is a fast-paced, grid-based arcade game where the player controls a paddle to bounce a glowing ball and break neon bricks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000

    # Colors
    COLOR_BG = (20, 10, 40)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GRID = (40, 30, 60)
    BRICK_COLORS = [
        (255, 0, 128),   # Pink
        (0, 255, 255),   # Cyan
        (128, 0, 255),   # Purple
        (0, 255, 128),   # Green
        (255, 128, 0),   # Orange
        (255, 255, 0),   # Yellow
    ]

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_RADIUS = 7
    INITIAL_LIVES = 3
    
    BRICK_ROWS = 6
    BRICK_COLS = 10
    BRICK_WIDTH = 58
    BRICK_HEIGHT = 18
    BRICK_SPACING = 4
    BRICK_AREA_TOP = 50
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_multiplier = pygame.font.SysFont('Consolas', 28, bold=True)

        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.bricks = None
        self.particles = None
        self.lives = None
        self.score = None
        self.multiplier = None
        self.steps = None
        self.game_over = None
        self.total_bricks = 0
        self.ball_speed_base = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.multiplier = 1
        self.game_over = False
        self.particles = []

        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self._create_bricks()
        self._attach_ball()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        
        self._handle_input(movement, space_held)
        reward += self._update_game_state()
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated:
            if self.lives == 0:
                reward -= 100 # Lose penalty
            elif len(self.bricks) == 0:
                reward += 100 # Win bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.PADDLE_WIDTH)

        # Launch Ball
        if self.ball_attached and space_held:
            # sfx: launch_ball.wav
            self.ball_attached = False
            launch_angle = self.np_random.uniform(-0.5, 0.5)
            self.ball_vel = [self.ball_speed_base * launch_angle, -self.ball_speed_base]

    def _update_game_state(self):
        reward = 0
        if self.ball_attached:
            self.ball_pos[0] = self.paddle.centerx
        else:
            reward += self._move_ball()

        self._update_particles()
        return reward

    def _move_ball(self):
        reward = 0
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collision
        if ball_rect.left <= 0:
            ball_rect.left = 0
            self.ball_vel[0] *= -1
            # sfx: wall_bounce.wav
        if ball_rect.right >= self.SCREEN_WIDTH:
            ball_rect.right = self.SCREEN_WIDTH
            self.ball_vel[0] *= -1
            # sfx: wall_bounce.wav
        if ball_rect.top <= 0:
            ball_rect.top = 0
            self.ball_vel[1] *= -1
            # sfx: wall_bounce.wav

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # sfx: paddle_hit.wav
            self.ball_vel[1] *= -1
            ball_rect.bottom = self.paddle.top
            
            # Change horizontal velocity based on where it hit the paddle
            offset = (ball_rect.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] = self.ball_speed_base * offset * 1.5
            self._normalize_ball_velocity()
            
            self.multiplier = 1 # Reset multiplier on paddle hit
            
            # Anti-stuck mechanism
            if abs(self.ball_vel[0]) < 0.1:
                self.ball_vel[0] += self.np_random.uniform(-0.2, 0.2)


        # Brick collision
        hit_brick = ball_rect.collidelist(self.bricks)
        if hit_brick != -1:
            brick = self.bricks[hit_brick]
            # sfx: brick_break.wav
            
            # Determine bounce direction
            # A simple approach: assume vertical bounce is most common
            self.ball_vel[1] *= -1
            
            # Handle edge cases for more natural bounces
            overlap = ball_rect.clip(brick)
            if overlap.width > overlap.height: # Hit top/bottom
                self.ball_vel[1] = abs(self.ball_vel[1]) * np.sign(ball_rect.centery - brick.centery)
            else: # Hit sides
                self.ball_vel[0] = abs(self.ball_vel[0]) * np.sign(ball_rect.centerx - brick.centerx)
            
            reward += 1.0 + (0.5 * self.multiplier)
            self.score += 10 * self.multiplier
            self.multiplier += 1
            
            brick_color = self.brick_colors_map[hit_brick]
            self._spawn_particles(brick.center, brick_color)
            
            self.bricks.pop(hit_brick)
            self.brick_colors_map.pop(hit_brick)

            # Check for speed increase
            bricks_destroyed = self.total_bricks - len(self.bricks)
            if bricks_destroyed > 0 and bricks_destroyed % 20 == 0:
                self.ball_speed_base += 0.2
                self._normalize_ball_velocity()


        # Bottom wall / Lose life
        if ball_rect.top >= self.SCREEN_HEIGHT:
            # sfx: lose_life.wav
            self.lives -= 1
            self.multiplier = 1
            if self.lives > 0:
                self._attach_ball()
            else:
                self.game_over = True

        self.ball_pos = [ball_rect.centerx, ball_rect.centery]
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _render_game(self):
        # Bricks
        for i, brick in enumerate(self.bricks):
            color = self.brick_colors_map[i]
            # Glow effect for bricks
            glow_rect = brick.inflate(4, 4)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, 50), s.get_rect(), border_radius=5)
            self.screen.blit(s, glow_rect.topleft)
            pygame.draw.rect(self.screen, color, brick, border_radius=3)
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Ball
        if self.lives > 0:
            # Glow effect
            for i in range(4, 0, -1):
                alpha = 60 - i * 12
                pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS + i, (*self.COLOR_BALL, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos[0]), int(self.ball_pos[1]), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Lives
        for i in range(self.lives):
            x = self.SCREEN_WIDTH - 25 - (i * (self.BALL_RADIUS * 2 + 10))
            pygame.gfxdraw.aacircle(self.screen, x, 22, self.BALL_RADIUS, self.COLOR_TEXT)
            pygame.gfxdraw.filled_circle(self.screen, x, 22, self.BALL_RADIUS, self.COLOR_TEXT)
            
        # Multiplier
        if self.multiplier > 1:
            mult_text = self.font_multiplier.render(f"x{self.multiplier}", True, self.COLOR_TEXT)
            text_rect = mult_text.get_rect(center=self.paddle.center)
            text_rect.y -= 40
            self.screen.blit(mult_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
            "multiplier": self.multiplier,
        }

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        if self.lives <= 0:
            self.game_over = True
        if not self.bricks:
            self.game_over = True
        return self.game_over

    def _create_bricks(self):
        self.bricks = []
        self.brick_colors_map = []
        total_brick_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_SPACING) - self.BRICK_SPACING
        start_x = (self.SCREEN_WIDTH - total_brick_width) / 2
        
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                x = start_x + c * (self.BRICK_WIDTH + self.BRICK_SPACING)
                y = self.BRICK_AREA_TOP + r * (self.BRICK_HEIGHT + self.BRICK_SPACING)
                brick = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                self.bricks.append(brick)
                self.brick_colors_map.append(self.BRICK_COLORS[r % len(self.BRICK_COLORS)])
        self.total_bricks = len(self.bricks)

    def _attach_ball(self):
        self.ball_attached = True
        self.ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]
        bricks_destroyed = self.total_bricks - len(self.bricks)
        speed_tiers = bricks_destroyed // 20
        self.ball_speed_base = 3.5 + (speed_tiers * 0.2)

    def _normalize_ball_velocity(self):
        current_speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if current_speed == 0: return
        scale = self.ball_speed_base / current_speed
        self.ball_vel[0] *= scale
        self.ball_vel[1] *= scale
        # Ensure minimum vertical speed to avoid boring loops
        if abs(self.ball_vel[1]) < 0.2 * self.ball_speed_base:
            self.ball_vel[1] = np.sign(self.ball_vel[1]) * 0.2 * self.ball_speed_base

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(1, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy', 'directfb', etc. as needed

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Neon Breakout")
    
    terminated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    running = True
    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space, shift]

        # --- Handle Pygame Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        if terminated:
            # Display Game Over message
            font_game_over = pygame.font.SysFont('Consolas', 50, bold=True)
            win_condition = info.get("bricks_left", -1) == 0
            msg = "YOU WIN!" if win_condition else "GAME OVER"
            color = (100, 255, 100) if win_condition else (255, 100, 100)
            
            text = font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 20))
            screen.blit(text, text_rect)
            
            font_restart = pygame.font.SysFont('Consolas', 20)
            text = font_restart.render("Press 'R' to restart", True, (200, 200, 200))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 30))
            screen.blit(text, text_rect)


        pygame.display.flip()
        
        # --- Frame Rate Control ---
        env.clock.tick(30)
        
    pygame.quit()