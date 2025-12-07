
# Generated: 2025-08-28T04:09:54.145934
# Source Brief: brief_05158.md
# Brief Index: 5158

        
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

    user_guide = (
        "Controls: Use arrow keys to adjust launch angle. Press Space to launch the ball."
    )

    game_description = (
        "A retro arcade brick-breaker. Aim your shots to strategically clear the "
        "field and rack up points. Clear all bricks to win, but lose if you run out of balls."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 15, 40)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_GRID = (30, 30, 60)
    BRICK_COLORS = [
        (217, 87, 99), (217, 142, 87), (197, 217, 87),
        (87, 217, 134), (87, 160, 217), (136, 87, 217)
    ]
    
    BALL_RADIUS = 7
    BALL_SPEED = 7
    LAUNCHER_Y = 380
    
    BRICK_ROWS = 6
    BRICK_COLS = 15
    BRICK_W, BRICK_H = 38, 18
    BRICK_GAP = 4
    BRICK_OFFSET_X = (WIDTH - (BRICK_COLS * (BRICK_W + BRICK_GAP) - BRICK_GAP)) // 2
    BRICK_OFFSET_Y = 50

    ANGLE_ADJUST_SPEED = 2.5
    MIN_ANGLE = 15
    MAX_ANGLE = 165

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # This will be initialized in reset
        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        self.game_phase = "aiming"
        self.balls_left = 0
        self.launcher_angle = 90
        self.ball_pos = pygame.Vector2(0, 0)
        self.ball_vel = pygame.Vector2(0, 0)
        self.bricks = []
        self.particles = []
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Call after reset to ensure all vars are init'd

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over_message = ""
        self.game_phase = "aiming"
        self.balls_left = 5
        self.launcher_angle = 90.0
        
        self.ball_pos = pygame.Vector2(self.WIDTH // 2, self.LAUNCHER_Y - self.BALL_RADIUS - 5)
        self.ball_vel = pygame.Vector2(0, 0)
        
        self._create_bricks()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False
        self.steps += 1

        if self.game_over_message:
            # Game is over, no more actions, just wait for reset
            terminated = True
            return self._get_observation(), 0, terminated, False, self._get_info()

        # --- Handle player input ---
        if self.game_phase == "aiming":
            angle_change = 0
            if movement in [3, 1]:  # Left or Up
                angle_change = -self.ANGLE_ADJUST_SPEED
            elif movement in [4, 2]:  # Right or Down
                angle_change = self.ANGLE_ADJUST_SPEED
            
            self.launcher_angle = np.clip(self.launcher_angle + angle_change, self.MIN_ANGLE, self.MAX_ANGLE)

            if space_held and self.balls_left > 0:
                self.balls_left -= 1
                self.game_phase = "ball_moving"
                angle_rad = math.radians(self.launcher_angle)
                self.ball_vel = pygame.Vector2(
                    -math.cos(angle_rad) * self.BALL_SPEED,
                    -math.sin(angle_rad) * self.BALL_SPEED
                )
                # SFX: Ball launch

        # --- Update game state ---
        if self.game_phase == "ball_moving":
            self.ball_pos += self.ball_vel
            
            ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Wall collision
            if self.ball_pos.x <= self.BALL_RADIUS or self.ball_pos.x >= self.WIDTH - self.BALL_RADIUS:
                self.ball_vel.x *= -1
                self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                reward -= 0.02
                # SFX: Wall bounce
            if self.ball_pos.y <= self.BALL_RADIUS:
                self.ball_vel.y *= -1
                self.ball_pos.y = np.clip(self.ball_pos.y, self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                reward -= 0.02
                # SFX: Wall bounce

            # Bottom boundary (lose ball)
            if self.ball_pos.y > self.HEIGHT:
                self.game_phase = "aiming"
                self.ball_pos = pygame.Vector2(self.WIDTH // 2, self.LAUNCHER_Y - self.BALL_RADIUS - 5)
                self.ball_vel = pygame.Vector2(0, 0)
                if self.balls_left <= 0:
                    terminated = True
                    reward -= 100
                    self.game_over_message = "GAME OVER"
                    
            # Brick collision
            hit_brick_this_frame = False
            for i in range(len(self.bricks) - 1, -1, -1):
                brick_rect, brick_color, row = self.bricks[i]
                if not hit_brick_this_frame and ball_rect.colliderect(brick_rect):
                    hit_brick_this_frame = True
                    
                    # Bounce logic
                    diff_x = self.ball_pos.x - brick_rect.centerx
                    diff_y = self.ball_pos.y - brick_rect.centery
                    
                    if abs(diff_x / brick_rect.width) > abs(diff_y / brick_rect.height):
                        self.ball_vel.x *= -1
                    else:
                        self.ball_vel.y *= -1

                    # Reward and score
                    reward += 0.1
                    self.score += 10

                    # Particles
                    self._create_particles(brick_rect.center, brick_color)
                    # SFX: Brick break

                    # Remove brick and check for row clear
                    self.bricks.pop(i)
                    if self._is_row_clear(row):
                        reward += 1.0
                        self.score += 100 # Row clear bonus
                        # SFX: Row clear bonus
                    
                    # Check for win condition
                    if not self.bricks:
                        terminated = True
                        reward += 100
                        self.game_over_message = "YOU WIN!"

        # Update particles
        self._update_particles()

        # Check for max steps termination
        if self.steps >= 1000 and not terminated:
            terminated = True
            self.game_over_message = "TIME UP"
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_bricks(self):
        self.bricks = []
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                x = self.BRICK_OFFSET_X + c * (self.BRICK_W + self.BRICK_GAP)
                y = self.BRICK_OFFSET_Y + r * (self.BRICK_H + self.BRICK_GAP)
                rect = pygame.Rect(x, y, self.BRICK_W, self.BRICK_H)
                color = self.BRICK_COLORS[r % len(self.BRICK_COLORS)]
                self.bricks.append((rect, color, r))

    def _is_row_clear(self, row_index):
        for _, _, r in self.bricks:
            if r == row_index:
                return False
        return True

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Bricks
        for rect, color, _ in self.bricks:
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, shadow, rect.move(2, 2))
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.topright, 1)
            pygame.draw.line(self.screen, highlight, rect.topleft, rect.bottomleft, 1)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            size = int(max(1, p['life'] / 5))
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*p['color'], alpha))
            self.screen.blit(s, (int(p['pos'].x - size/2), int(p['pos'].y - size/2)))

        # Launcher and aiming line
        if self.game_phase == "aiming":
            launcher_pos = pygame.Vector2(self.WIDTH // 2, self.LAUNCHER_Y)
            pygame.draw.polygon(self.screen, self.COLOR_UI_TEXT, [
                (launcher_pos.x - 20, launcher_pos.y),
                (launcher_pos.x + 20, launcher_pos.y),
                (launcher_pos.x, launcher_pos.y - 10)
            ])
            
            angle_rad = math.radians(self.launcher_angle)
            end_pos = launcher_pos + pygame.Vector2(-math.cos(angle_rad), -math.sin(angle_rad)) * 80
            self._draw_dashed_line(self.screen, self.COLOR_UI_TEXT, launcher_pos, end_pos, 5, 3)

        # Ball
        if self.game_phase == "ball_moving" or self.game_phase == "aiming":
            # Glow effect
            glow_radius = int(self.BALL_RADIUS * 1.8)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 50))
            self.screen.blit(s, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)))
            # Main ball
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left):
            pos_x = self.WIDTH - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)

        # Game Over message
        if self.game_over_message:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_game_over.render(self.game_over_message, True, self.COLOR_BALL)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "bricks_left": len(self.bricks),
            "game_phase": self.game_phase,
        }

    def _draw_dashed_line(self, surf, color, start_pos, end_pos, width=1, dash_length=10, gap_length=5):
        origin = pygame.Vector2(start_pos)
        target = pygame.Vector2(end_pos)
        displacement = target - origin
        length = displacement.length()
        if length == 0: return
        
        direction = displacement.normalize()
        
        current_pos = 0
        while current_pos < length:
            start = origin + direction * current_pos
            end = origin + direction * min(current_pos + dash_length, length)
            pygame.draw.line(surf, color, start, end, width)
            current_pos += dash_length + gap_length

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
    env.validate_implementation()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Brick Breaker")
    
    terminated = False
    clock = pygame.time.Clock()
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_UP]:
            movement = 3 # map to left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_DOWN]:
            movement = 4 # map to right
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(30) # Limit frame rate for human play
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()