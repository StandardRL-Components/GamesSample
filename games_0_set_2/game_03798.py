
# Generated: 2025-08-28T00:27:41.342960
# Source Brief: brief_03798.md
# Brief Index: 3798

        
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
        "Controls: ←→ to aim the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric Breakout-style game. Clear all the gems by bouncing the ball off your paddle. "
        "More extreme angles give more points but are riskier!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Colors
        self.COLOR_BG = (25, 20, 35)
        self.COLOR_GRID = (45, 40, 55)
        self.COLOR_PADDLE = (0, 200, 255)
        self.COLOR_PADDLE_OUTLINE = (100, 240, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.GEM_COLORS = [
            (255, 80, 120), (80, 255, 120), (80, 120, 255),
            (255, 255, 80), (255, 80, 255), (80, 255, 255)
        ]
        self.COLOR_TEXT = (220, 220, 240)
        
        # Fonts
        try:
            self.FONT_UI = pygame.font.SysFont("Consolas", 24)
            self.FONT_GAME_OVER = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.FONT_UI = pygame.font.Font(None, 28)
            self.FONT_GAME_OVER = pygame.font.Font(None, 60)

        # Game constants
        self.PADDLE_Y = self.HEIGHT - 50
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 12
        self.PADDLE_ROTATION_SPEED = 4.0
        self.PADDLE_MAX_ANGLE = 60
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 6.0
        self.GEM_RADIUS = 14
        self.NUM_GEMS = 10
        self.MAX_STEPS = 2000
        
        # Initialize state variables
        self.paddle = {}
        self.ball = {}
        self.gems = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.game_over = False
        self.last_shot_reward = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_left = 3
        self.game_over = False

        self.paddle = {
            'x': self.WIDTH / 2,
            'angle': 0,
        }
        
        self._reset_ball()

        self.gems = []
        while len(self.gems) < self.NUM_GEMS:
            new_gem_pos = (
                self.np_random.integers(self.GEM_RADIUS * 2, self.WIDTH - self.GEM_RADIUS * 2),
                self.np_random.integers(self.GEM_RADIUS * 2, self.HEIGHT // 2)
            )
            # Ensure gems don't overlap
            too_close = False
            for gem in self.gems:
                dist = math.hypot(new_gem_pos[0] - gem['pos'][0], new_gem_pos[1] - gem['pos'][1])
                if dist < self.GEM_RADIUS * 2.5:
                    too_close = True
                    break
            if not too_close:
                self.gems.append({
                    'pos': new_gem_pos,
                    'color': random.choice(self.GEM_COLORS),
                    'sparkle_timer': self.np_random.random() * math.pi * 2
                })

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball = {
            'x': self.paddle['x'],
            'y': self.PADDLE_Y - self.PADDLE_HEIGHT / 2 - self.BALL_RADIUS,
            'vx': 0,
            'vy': 0,
            'is_launched': False,
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01  # Cost of living
        
        # Handle paddle movement
        if movement == 3:  # Left
            self.paddle['angle'] -= self.PADDLE_ROTATION_SPEED
        elif movement == 4:  # Right
            self.paddle['angle'] += self.PADDLE_ROTATION_SPEED
        self.paddle['angle'] = max(-self.PADDLE_MAX_ANGLE, min(self.PADDLE_MAX_ANGLE, self.paddle['angle']))

        # Handle ball launch
        if space_held and not self.ball['is_launched']:
            self.ball['is_launched'] = True
            angle_rad = math.radians(90 - self.paddle['angle'])
            self.ball['vx'] = self.BALL_SPEED * math.cos(angle_rad)
            self.ball['vy'] = -self.BALL_SPEED * math.sin(angle_rad)
            # Sound: BallLaunch.wav
            
            # Risky/safe shot reward
            if abs(self.paddle['angle']) > 45:
                reward += 2.0  # Risky shot
            elif abs(self.paddle['angle']) < 15:
                reward -= 0.2  # Safe shot

        # Update game logic
        step_reward = self._update_game_state()
        reward += step_reward
        self.score += step_reward

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if len(self.gems) == 0:
                reward += 100 # Win bonus
                self.score += 100
            else:
                reward += -100 # Loss penalty
                self.score += -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self):
        reward = 0
        
        # Update ball
        if self.ball['is_launched']:
            self.ball['x'] += self.ball['vx']
            self.ball['y'] += self.ball['vy']
        else:
            # Ball follows paddle before launch
            self.ball['x'] = self.paddle['x']
        
        # Ball collisions
        # Walls
        if self.ball['x'] - self.BALL_RADIUS < 0 or self.ball['x'] + self.BALL_RADIUS > self.WIDTH:
            self.ball['vx'] *= -1
            self.ball['x'] = max(self.BALL_RADIUS, min(self.WIDTH - self.BALL_RADIUS, self.ball['x']))
            # Sound: WallBounce.wav
        if self.ball['y'] - self.BALL_RADIUS < 0:
            self.ball['vy'] *= -1
            self.ball['y'] = self.BALL_RADIUS
            # Sound: WallBounce.wav

        # Bottom (lose ball)
        if self.ball['y'] - self.BALL_RADIUS > self.HEIGHT:
            self.balls_left -= 1
            reward -= 5
            self._create_particles((self.ball['x'], self.HEIGHT), self.COLOR_BALL, 50, 5)
            # Sound: LoseBall.wav
            if self.balls_left > 0:
                self._reset_ball()
        
        # Paddle
        paddle_rect = pygame.Rect(self.paddle['x'] - self.PADDLE_WIDTH / 2, self.PADDLE_Y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if self.ball['is_launched'] and self.ball['vy'] > 0 and paddle_rect.collidepoint(self.ball['x'], self.ball['y'] + self.BALL_RADIUS):
            self.ball['y'] = paddle_rect.top - self.BALL_RADIUS
            
            angle_rad = math.radians(90 - self.paddle['angle'])
            self.ball['vx'] = self.BALL_SPEED * math.cos(angle_rad)
            self.ball['vy'] = -self.BALL_SPEED * math.sin(angle_rad)
            
            # Add some horizontal influence from impact point
            impact_factor = (self.ball['x'] - self.paddle['x']) / (self.PADDLE_WIDTH / 2)
            self.ball['vx'] += impact_factor * 1.5
            self.ball['vx'] = max(-self.BALL_SPEED, min(self.BALL_SPEED, self.ball['vx']))
            # Sound: PaddleHit.wav

        # Gems
        for gem in self.gems[:]:
            dist = math.hypot(self.ball['x'] - gem['pos'][0], self.ball['y'] - gem['pos'][1])
            if dist < self.BALL_RADIUS + self.GEM_RADIUS:
                reward += 10
                self._create_particles(gem['pos'], gem['color'], 30, 3)
                self.gems.remove(gem)
                # Sound: GemCollect.wav

                # Reflect ball velocity
                nx = self.ball['x'] - gem['pos'][0]
                ny = self.ball['y'] - gem['pos'][1]
                n_mag = math.hypot(nx, ny)
                if n_mag == 0: continue
                nx /= n_mag
                ny /= n_mag
                
                dot = self.ball['vx'] * nx + self.ball['vy'] * ny
                self.ball['vx'] -= 2 * dot * nx
                self.ball['vy'] -= 2 * dot * ny
                break

        # Update particles
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                
        # Update gem animations
        for gem in self.gems:
            gem['sparkle_timer'] += 0.1
        
        return reward

    def _check_termination(self):
        if len(self.gems) == 0:
            self.game_over = True
            return True
        if self.balls_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_gems()
        self._render_particles()
        self._render_paddle()
        self._render_ball()

    def _render_grid(self):
        # Simple isometric grid
        for i in range(-10, 25):
            # Lines going one way
            start_pos1 = (i * 40, 0)
            end_pos1 = (self.WIDTH + i * 40, self.HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos1, end_pos1)
            # Lines going the other way
            start_pos2 = (-i * 40, 0)
            end_pos2 = (self.WIDTH - i * 40, self.HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos2, end_pos2)
            
    def _render_gems(self):
        for gem in self.gems:
            pos = (int(gem['pos'][0]), int(gem['pos'][1]))
            color = gem['color']
            
            # Sparkle effect
            sparkle = (math.sin(gem['sparkle_timer']) + 1) / 2
            radius = int(self.GEM_RADIUS * (0.9 + sparkle * 0.2))
            
            # Shadow
            shadow_pos = (pos[0], pos[1] + 8)
            pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], radius, int(radius * 0.4), (0,0,0,50))

            # Main gem body with glow
            for i in range(radius, 0, -2):
                alpha = 80 * (1 - i / radius)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (color[0], color[1], color[2], int(alpha)))
            
            # Highlight
            highlight_angle = gem['sparkle_timer']
            hx = int(pos[0] + radius * 0.5 * math.cos(highlight_angle))
            hy = int(pos[1] + radius * 0.5 * math.sin(highlight_angle))
            pygame.gfxdraw.filled_circle(self.screen, hx, hy, 2, (255, 255, 255, 200))

    def _render_paddle(self):
        center_x, center_y = self.paddle['x'], self.PADDLE_Y
        angle_rad = math.radians(self.paddle['angle'])
        
        w, h = self.PADDLE_WIDTH / 2, self.PADDLE_HEIGHT / 2
        
        points = [(-w, -h), (w, -h), (w, h), (-w, h)]
        rotated_points = []
        for x, y in points:
            rx = center_x + x * math.cos(angle_rad) - y * math.sin(angle_rad)
            ry = center_y + x * math.sin(angle_rad) + y * math.cos(angle_rad)
            rotated_points.append((int(rx), int(ry)))
            
        # Draw shadow
        shadow_points = [(p[0], p[1] + 8) for p in rotated_points]
        pygame.gfxdraw.filled_polygon(self.screen, shadow_points, (0,0,0,50))

        # Draw paddle
        pygame.gfxdraw.filled_polygon(self.screen, rotated_points, self.COLOR_PADDLE)
        pygame.gfxdraw.aapolygon(self.screen, rotated_points, self.COLOR_PADDLE_OUTLINE)

    def _render_ball(self):
        pos = (int(self.ball['x']), int(self.ball['y']))
        
        # Shadow
        shadow_pos = (pos[0], pos[1] + 8)
        shadow_radius = self.BALL_RADIUS
        shadow_height = int(shadow_radius * 0.4)
        pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], shadow_radius, shadow_height, (0,0,0,50))
        
        # Glow
        for i in range(self.BALL_RADIUS, 0, -1):
            alpha = 100 * (1 - i / self.BALL_RADIUS)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (255, 255, 255, int(alpha)))

        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_particles(self):
        for p in self.particles:
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)

    def _render_ui(self):
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Balls left
        balls_text = self.FONT_UI.render("BALLS:", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - 180, 10))
        for i in range(self.balls_left):
            pos = (self.WIDTH - 80 + i * 25, 22)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS // 2, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS // 2, self.COLOR_BALL)
            
        # Game Over message
        if self.game_over:
            msg = "YOU WIN!" if len(self.gems) == 0 else "GAME OVER"
            color = (100, 255, 100) if len(self.gems) == 0 else (255, 100, 100)
            
            text_surf = self.FONT_GAME_OVER.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            # Text shadow
            shadow_surf = self.FONT_GAME_OVER.render(msg, True, (0,0,0,100))
            self.screen.blit(shadow_surf, text_rect.move(3, 3))
            
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "gems_left": len(self.gems),
        }

    def _create_particles(self, pos, color, count, max_life):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(max_life // 2, max_life),
                'max_life': max_life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })
            
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
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Gem Breaker")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print(env.user_guide)
    
    while not done:
        # Action defaults
        movement = 0  # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the intended frame rate
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
    
    env.close()