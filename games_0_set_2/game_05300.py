
# Generated: 2025-08-28T04:34:22.238413
# Source Brief: brief_05300.md
# Brief Index: 5300

        
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
        "Control a paddle to deflect a ball and destroy all bricks in a visually stunning, procedurally generated neon arena. Collect power-ups for an advantage!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors (Neon Palette)
    COLOR_BG = (10, 5, 25)
    COLOR_WALL = (40, 20, 80)
    COLOR_PADDLE = (0, 255, 255)
    COLOR_PADDLE_GLOW = (0, 150, 255, 50)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 200, 0, 100)
    COLOR_TEXT = (220, 220, 255)
    BRICK_COLORS = [
        (255, 0, 128),  # Magenta
        (0, 255, 0),    # Green
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Blue
        (255, 0, 0),    # Red
    ]

    # Game Parameters
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10
    BALL_RADIUS = 8
    BALL_SPEED_MIN = 5
    BALL_SPEED_MAX = 10
    BRICK_ROWS = 5
    BRICK_COLS = 10
    BRICK_WIDTH = 58
    BRICK_HEIGHT = 18
    BRICK_SPACING = 4
    INITIAL_LIVES = 3
    MAX_STEPS = 2500
    POWERUP_CHANCE = 0.2
    POWERUP_SPEED = 2
    POWERUP_RADIUS = 10
    POWERUP_DURATION = 450 # 15 seconds at 30fps

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 50)
        
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = None
        self.bricks = None
        self.particles = None
        self.powerups = None
        self.active_powerups = None
        self.rng = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_attached = True
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        
        self._generate_bricks()
        
        self.particles = []
        self.powerups = []
        self.active_powerups = {}
        
        return self._get_observation(), self._get_info()

    def _generate_bricks(self):
        self.bricks = []
        total_brick_width = self.BRICK_COLS * (self.BRICK_WIDTH + self.BRICK_SPACING) - self.BRICK_SPACING
        start_x = (self.SCREEN_WIDTH - total_brick_width) / 2
        start_y = 50

        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                x = start_x + j * (self.BRICK_WIDTH + self.BRICK_SPACING)
                y = start_y + i * (self.BRICK_HEIGHT + self.BRICK_SPACING)
                brick_rect = pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT)
                color = self.BRICK_COLORS[i % len(self.BRICK_COLORS)]
                self.bricks.append({"rect": brick_rect, "color": color, "points": (self.BRICK_ROWS - i) * 10})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.001 # Small time penalty

        self._handle_input(movement, space_held)
        self._update_game_state()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if not self.bricks:
                reward = 100 # Win bonus
            elif self.lives <= 0:
                reward = -100 # Lose penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
        if movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = max(10, min(self.paddle.x, self.SCREEN_WIDTH - self.paddle.width - 10))

        if self.ball_attached and space_held:
            self.ball_attached = False
            self.ball_vel = pygame.Vector2(self.rng.choice([-1, 1]) * self.BALL_SPEED_MIN / 2, -self.BALL_SPEED_MIN)
            # // SFX: Ball launch

    def _update_game_state(self):
        # Update paddle size from powerups
        if 'wide_paddle' in self.active_powerups:
            self.paddle.width = self.PADDLE_WIDTH * 1.5
            self.active_powerups['wide_paddle'] -= 1
            if self.active_powerups['wide_paddle'] <= 0:
                del self.active_powerups['wide_paddle']
        else:
            self.paddle.width = self.PADDLE_WIDTH

        # Update ball
        if self.ball_attached:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Update powerups
        for p in self.powerups[:]:
            p['pos'].y += self.POWERUP_SPEED
            p['rect'].center = p['pos']
            if p['pos'].y > self.SCREEN_HEIGHT:
                self.powerups.remove(p)

    def _handle_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Walls
        if ball_rect.left <= 10 or ball_rect.right >= self.SCREEN_WIDTH - 10:
            self.ball_vel.x *= -1
            ball_rect.left = max(10, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH - 10, ball_rect.right)
            self.ball_pos.x = ball_rect.centerx
            # // SFX: Wall hit
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            # // SFX: Wall hit

        # Bottom (lose life)
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            self.ball_attached = True
            if self.lives > 0:
                reward -= 10 # Penalty for losing a life
            self.active_powerups.clear() # Lose powerups on death
            # // SFX: Life lost

        # Paddle
        if not self.ball_attached and ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.paddle.width / 2)
            self.ball_vel.x += offset * 2
            
            # Clamp ball speed
            speed = self.ball_vel.length()
            if speed > self.BALL_SPEED_MAX:
                self.ball_vel.scale_to_length(self.BALL_SPEED_MAX)
            if speed < self.BALL_SPEED_MIN:
                self.ball_vel.scale_to_length(self.BALL_SPEED_MIN)

            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking
            # // SFX: Paddle hit

        # Bricks
        hit_index = ball_rect.collidelist([b['rect'] for b in self.bricks])
        if hit_index != -1:
            hit_brick = self.bricks.pop(hit_index)
            reward += 0.1
            self.score += hit_brick['points']
            self._create_particles(hit_brick['rect'].center, hit_brick['color'])
            # // SFX: Brick break
            
            # Simple collision response
            prev_pos = self.ball_pos - self.ball_vel
            if (prev_pos.y < hit_brick['rect'].top or prev_pos.y > hit_brick['rect'].bottom):
                self.ball_vel.y *= -1
            else:
                self.ball_vel.x *= -1
            
            # Spawn powerup
            if self.rng.random() < self.POWERUP_CHANCE:
                self._spawn_powerup(hit_brick['rect'].center)

        # Powerups
        for p in self.powerups[:]:
            if self.paddle.colliderect(p['rect']):
                self.active_powerups['wide_paddle'] = self.POWERUP_DURATION
                self.powerups.remove(p)
                reward += 1.0
                self.score += 50
                # // SFX: Powerup collect

        return reward

    def _spawn_powerup(self, pos):
        powerup_rect = pygame.Rect(0, 0, self.POWERUP_RADIUS*2, self.POWERUP_RADIUS*2)
        powerup_rect.center = pos
        self.powerups.append({'pos': pygame.Vector2(pos), 'rect': powerup_rect, 'type': 'wide_paddle'})

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.rng.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'color': color})

    def _check_termination(self):
        if self.lives <= 0 or not self.bricks or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_walls()
        self._render_particles()
        self._render_powerups()
        self._render_bricks()
        self._render_paddle()
        self._render_ball()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        # A simple vertical gradient
        for y in range(self.SCREEN_HEIGHT):
            color_val = 10 + int(y / self.SCREEN_HEIGHT * 20)
            pygame.draw.line(self.screen, (color_val, 5, color_val + 15), (0, y), (self.SCREEN_WIDTH, y))

    def _render_walls(self):
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, 10, self.SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT))

    def _render_bricks(self):
        for brick in self.bricks:
            r = brick['rect']
            c = brick['color']
            # Glow effect
            glow_rect = r.inflate(6, 6)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*c, 40), glow_surf.get_rect(), border_radius=5)
            self.screen.blit(glow_surf, glow_rect.topleft)
            # Main brick
            pygame.draw.rect(self.screen, c, r, border_radius=3)
            # Highlight
            pygame.draw.rect(self.screen, (255, 255, 255), r.inflate(-4,-4), 1, border_radius=2)

    def _render_paddle(self):
        # Glow
        glow_rect = self.paddle.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW, glow_surf.get_rect(), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        # Main paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)

    def _render_ball(self):
        # Trail
        if not self.ball_attached and self.ball_vel.length() > 0:
            trail_start = self.ball_pos - self.ball_vel.normalize() * self.BALL_RADIUS * 2
            pygame.draw.line(self.screen, self.COLOR_BALL, trail_start, self.ball_pos, 4)
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS + 3, self.COLOR_BALL_GLOW)
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            alpha = max(0, min(255, alpha))
            color = (*p['color'], alpha)
            size = int(p['life'] / 10) + 1
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,size,size))
            self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(size, size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_powerups(self):
        for p in self.powerups:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = self.POWERUP_RADIUS
            # Pulsing glow
            pulse = abs(math.sin(self.steps * 0.1))
            glow_radius = int(radius * (1.5 + pulse * 0.5))
            glow_alpha = int(80 + pulse * 40)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_PADDLE, glow_alpha))
            # Core
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (255,255,255))
            # Letter 'W' for Wide
            w_text = self.font_ui.render("W", True, self.COLOR_BG)
            self.screen.blit(w_text, (pos[0] - w_text.get_width() / 2, pos[1] - w_text.get_height() / 2))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))
        
        # Lives display
        for i in range(self.lives):
            life_rect = pygame.Rect(self.SCREEN_WIDTH - 30 - (i * 35), 15, 30, 8)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=3)

        if self.game_over:
            msg = "YOU WIN!" if not self.bricks else "GAME OVER"
            msg_text = self.font_msg.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bricks_left": len(self.bricks),
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set `render_mode="human"` in a hypothetical scenario to display the window
    # For this headless setup, we'll just step through and save a final frame
    
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)

    # --- Manual Control Mapping ---
    # To simulate playing, you can map keyboard keys to actions
    # This is a simplified mapping for demonstration
    key_to_action = {
        pygame.K_LEFT: np.array([3, 0, 0]),
        pygame.K_RIGHT: np.array([4, 0, 0]),
        pygame.K_SPACE: np.array([0, 1, 0]),
    }
    no_op_action = np.array([0, 0, 0])

    # Pygame setup for human play
    pygame.display.set_caption("Neon Breakout")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    # Game loop for human play
    while running:
        action = no_op_action.copy()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        if keys[pygame.K_RIGHT]:
            action[0] = 4
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()