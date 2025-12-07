
# Generated: 2025-08-28T00:28:05.469206
# Source Brief: brief_03796.md
# Brief Index: 3796

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "An isometric brick-breaker. Bounce the ball to break all the bricks. "
        "Chain hits for a score multiplier, but don't miss the ball!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_multiplier = pygame.font.Font(None, 48)

        # World constants
        self.LOGICAL_WIDTH = 240
        self.LOGICAL_HEIGHT = 320
        self.ISO_ORIGIN_X = self.screen_width // 2
        self.ISO_ORIGIN_Y = 60
        self.BRICK_ROWS = 5
        self.BRICK_COLS = 8
        self.BRICK_WIDTH = self.LOGICAL_WIDTH / self.BRICK_COLS
        self.BRICK_HEIGHT = (self.LOGICAL_HEIGHT / 2.5) / self.BRICK_ROWS
        self.BRICK_ISO_DEPTH = 12

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (35, 40, 60)
        self.COLOR_PADDLE = (0, 150, 255)
        self.COLOR_PADDLE_GLOW = (0, 150, 255, 50)
        self.COLOR_BALL = (255, 255, 255)
        self.COLOR_BALL_GLOW = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.BRICK_COLORS = [
            ((255, 80, 80), (200, 50, 50), (150, 30, 30)), # Red
            ((255, 160, 80), (200, 120, 50), (150, 90, 30)), # Orange
            ((255, 255, 80), (200, 200, 50), (150, 150, 30)), # Yellow
            ((80, 255, 80), (50, 200, 50), (30, 150, 30)), # Green
            ((80, 160, 255), (50, 120, 200), (30, 90, 150)), # Blue
        ]

        # Game entity parameters
        self.PADDLE_WIDTH = 55
        self.PADDLE_HEIGHT = 10
        self.PADDLE_SPEED = 6.0
        self.BALL_RADIUS = 5
        self.BALL_SPEED = 4.0

        # Game state variables are initialized in reset()
        self.paddle = {}
        self.ball = {}
        self.bricks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.multiplier = 0
        self.game_over = False
        self.win = False
        self.ball_on_paddle = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 3
        self.multiplier = 1
        self.game_over = False
        self.win = False
        self.particles.clear()

        self.paddle = {
            'x': self.LOGICAL_WIDTH / 2 - self.PADDLE_WIDTH / 2,
            'y': self.LOGICAL_HEIGHT - 30,
            'w': self.PADDLE_WIDTH,
            'h': self.PADDLE_HEIGHT,
            'vx': 0.0,
        }

        self.ball_on_paddle = True
        self._reset_ball()

        self.bricks = []
        for r in range(self.BRICK_ROWS):
            for c in range(self.BRICK_COLS):
                brick_rect = pygame.Rect(
                    c * self.BRICK_WIDTH,
                    r * self.BRICK_HEIGHT + 40,
                    self.BRICK_WIDTH,
                    self.BRICK_HEIGHT,
                )
                self.bricks.append({
                    'rect': brick_rect,
                    'colors': self.BRICK_COLORS[r % len(self.BRICK_COLORS)],
                    'value': (self.BRICK_ROWS - r) * 10
                })

        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball = {
            'x': self.paddle['x'] + self.paddle['w'] / 2,
            'y': self.paddle['y'] - self.BALL_RADIUS,
            'vx': 0.0,
            'vy': 0.0,
            'radius': self.BALL_RADIUS,
        }
        self.ball_on_paddle = True

    def step(self, action):
        reward = -0.01  # Time penalty
        
        movement = action[0]
        space_held = action[1] == 1

        if not self.game_over:
            self._handle_input(movement, space_held)
            event_reward = self._update_game_state()
            reward += event_reward

        self.steps += 1
        terminated = (self.lives <= 0) or (len(self.bricks) == 0) or (self.steps >= 2000)

        if terminated and not self.game_over:
            self.game_over = True
            if len(self.bricks) == 0:
                self.win = True
                reward += 100
            else:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, space_held):
        if movement == 3:  # Left
            self.paddle['vx'] = -self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle['vx'] = self.PADDLE_SPEED
        else:
            self.paddle['vx'] = 0.0

        if space_held and self.ball_on_paddle:
            self.ball_on_paddle = False
            self.ball['vy'] = -self.BALL_SPEED
            self.ball['vx'] = self.np_random.uniform(-0.5, 0.5)
            # Sound: launch.wav

    def _update_game_state(self):
        event_reward = 0.0
        
        # Update paddle
        self.paddle['x'] += self.paddle['vx']
        self.paddle['x'] = max(0, min(self.paddle['x'], self.LOGICAL_WIDTH - self.paddle['w']))

        if self.ball_on_paddle:
            self.ball['x'] = self.paddle['x'] + self.paddle['w'] / 2
            self.ball['y'] = self.paddle['y'] - self.ball['radius']
        else:
            # Update ball position
            self.ball['x'] += self.ball['vx']
            self.ball['y'] += self.ball['vy']

            # Wall collisions
            if self.ball['x'] - self.ball['radius'] < 0:
                self.ball['x'] = self.ball['radius']
                self.ball['vx'] *= -1
            elif self.ball['x'] + self.ball['radius'] > self.LOGICAL_WIDTH:
                self.ball['x'] = self.LOGICAL_WIDTH - self.ball['radius']
                self.ball['vx'] *= -1
            
            if self.ball['y'] - self.ball['radius'] < 0:
                self.ball['y'] = self.ball['radius']
                self.ball['vy'] *= -1

            # Paddle miss (bottom wall)
            if self.ball['y'] + self.ball['radius'] > self.LOGICAL_HEIGHT:
                self.lives -= 1
                event_reward -= 1.0
                self.multiplier = 1
                self._reset_ball()
                # Sound: lose_life.wav
                return event_reward # Stop processing this frame

            # Paddle collision
            paddle_rect = pygame.Rect(self.paddle['x'], self.paddle['y'], self.paddle['w'], self.paddle['h'])
            if self.ball['vy'] > 0 and paddle_rect.colliderect(
                self.ball['x'] - self.ball['radius'], self.ball['y'] - self.ball['radius'],
                self.ball['radius'] * 2, self.ball['radius'] * 2):
                
                self.ball['y'] = self.paddle['y'] - self.ball['radius']
                self.ball['vy'] *= -1
                
                # Influence angle based on hit position
                hit_pos = (self.ball['x'] - (self.paddle['x'] + self.paddle['w'] / 2)) / (self.paddle['w'] / 2)
                self.ball['vx'] += hit_pos * 2.0
                self.ball['vx'] = max(-self.BALL_SPEED, min(self.ball['vx'], self.BALL_SPEED))

                self.multiplier = 1 # Reset multiplier on paddle hit
                # Sound: paddle_hit.wav

            # Brick collisions
            for i in range(len(self.bricks) - 1, -1, -1):
                brick = self.bricks[i]
                
                # AABB check first for performance
                brick_aabb = brick['rect']
                ball_aabb = pygame.Rect(self.ball['x'] - self.ball['radius'], self.ball['y'] - self.ball['radius'], self.ball['radius']*2, self.ball['radius']*2)
                if not brick_aabb.colliderect(ball_aabb):
                    continue

                # More precise circle-rect collision
                closest_x = max(brick_aabb.left, min(self.ball['x'], brick_aabb.right))
                closest_y = max(brick_aabb.top, min(self.ball['y'], brick_aabb.bottom))
                dist_sq = (self.ball['x'] - closest_x)**2 + (self.ball['y'] - closest_y)**2
                
                if dist_sq < self.ball['radius']**2:
                    event_reward += 1.0 + (0.1 * self.multiplier)
                    self.score += brick['value'] * self.multiplier
                    self.multiplier += 1

                    # Create particles
                    for _ in range(15):
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(1, 4)
                        self.particles.append({
                            'x': closest_x, 'y': closest_y,
                            'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                            'life': self.np_random.integers(15, 30),
                            'color': random.choice(brick['colors'])
                        })
                    
                    # Bounce logic
                    dx = self.ball['x'] - closest_x
                    dy = self.ball['y'] - closest_y

                    # If ball center is inside, push it out
                    if dx == 0 and dy == 0:
                        self.ball['x'] -= self.ball['vx']
                        self.ball['y'] -= self.ball['vy']
                    
                    # Reflect velocity based on collision side
                    if abs(dx) > abs(dy):
                        self.ball['vx'] *= -1
                        self.ball['x'] += self.ball['vx'] # Push out
                    else:
                        self.ball['vy'] *= -1
                        self.ball['y'] += self.ball['vy'] # Push out

                    self.bricks.pop(i)
                    # Sound: brick_break.wav
                    break # One brick break per frame

        # Update particles
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        return event_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_bricks()
        self._render_paddle()
        self._render_ball()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives, "multiplier": self.multiplier}

    def _to_screen(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y)
        screen_y = self.ISO_ORIGIN_Y + (x + y) * 0.5
        return int(screen_x), int(screen_y)

    def _draw_iso_rect(self, surface, colors, l_rect, depth):
        face_color, side_color, top_color = colors
        x, y, w, h = l_rect.x, l_rect.y, l_rect.width, l_rect.height

        p = [
            self._to_screen(x, y + h),      # 0 bottom-left
            self._to_screen(x + w, y + h),  # 1 bottom-right
            self._to_screen(x + w, y),      # 2 top-right
            self._to_screen(x, y),          # 3 top-left
        ]
        p_top = [(px, py - depth) for px, py in p]

        # Draw sides
        pygame.gfxdraw.filled_polygon(surface, [p[0], p[1], p_top[1], p_top[0]], side_color)
        pygame.gfxdraw.filled_polygon(surface, [p[3], p[0], p_top[0], p_top[3]], side_color)
        
        # Draw top face
        pygame.gfxdraw.filled_polygon(surface, p_top, top_color)
        pygame.gfxdraw.aapolygon(surface, p_top, top_color) # Anti-alias outline

    def _render_grid(self):
        for i in range(0, self.LOGICAL_WIDTH + 1, 20):
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._to_screen(i, 0), self._to_screen(i, self.LOGICAL_HEIGHT))
        for i in range(0, self.LOGICAL_HEIGHT + 1, 20):
            pygame.draw.aaline(self.screen, self.COLOR_GRID, self._to_screen(0, i), self._to_screen(self.LOGICAL_WIDTH, i))

    def _render_bricks(self):
        for brick in self.bricks:
            self._draw_iso_rect(self.screen, brick['colors'], brick['rect'], self.BRICK_ISO_DEPTH)

    def _render_paddle(self):
        paddle_rect = pygame.Rect(self.paddle['x'], self.paddle['y'], self.paddle['w'], self.paddle['h'])
        self._draw_iso_rect(self.screen, (self.COLOR_PADDLE, self.COLOR_PADDLE, self.COLOR_PADDLE), paddle_rect, self.PADDLE_HEIGHT)

    def _render_ball(self):
        screen_pos = self._to_screen(self.ball['x'], self.ball['y'])
        
        # Glow effect
        glow_radius = int(self.ball['radius'] * 2.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius - self.ball['radius']))
        
        # Ball itself
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1] - self.ball['radius'], self.ball['radius'], self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1] - self.ball['radius'], self.ball['radius'], self.COLOR_BALL)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30))))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pos = self._to_screen(p['x'], p['y'])
            size = int(p['life'] / 10 + 1)
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Lives
        lives_text = self.font_ui.render(f"BALLS: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.screen_width - lives_text.get_width() - 20, 10))

        # Multiplier
        if self.multiplier > 1:
            multi_text = self.font_multiplier.render(f"x{self.multiplier}", True, self.BRICK_COLORS[2][0])
            text_rect = multi_text.get_rect(center=(self.screen_width // 2, 30))
            self.screen.blit(multi_text, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Pygame event handling
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Map keyboard inputs to action space
        movement = 0 # no-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        # Pygame uses (width, height), numpy uses (height, width)
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((env.screen_width, env.screen_height))
            display_surf.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Control frame rate
        env.clock.tick(30)
        
    env.close()