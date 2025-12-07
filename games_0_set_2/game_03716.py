import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ↑ and ↓ to move the paddle vertically. Your goal is to hit the ball past the coral on the right."
    )

    game_description = (
        "An isometric underwater sports game. Control a paddle to hit a bouncing ball past swaying seaweed obstacles and score goals. Miss three times and you lose!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG_DARK = (10, 20, 40)
        self.COLOR_BG_LIGHT = (20, 40, 80)
        self.COLOR_PADDLE = (50, 200, 50)
        self.COLOR_PADDLE_HL = (150, 255, 150)
        self.COLOR_BALL = (255, 80, 80)
        self.COLOR_BALL_HL = (255, 200, 200)
        self.COLOR_SEAWEED = (200, 100, 30)
        self.COLOR_CORAL = (230, 120, 90)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_MISS = (255, 0, 0)
        self.COLOR_SPARK = (255, 255, 100)

        # Game constants
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 12, 70
        self.PADDLE_SPEED = 7.0
        self.PADDLE_X_POS = self.WIDTH * 0.1
        self.BALL_RADIUS = 9
        self.INITIAL_BALL_SPEED = 4.0
        self.MIN_BALL_SPEED = 2.0
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 3
        self.LOSE_MISSES = 3

        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.missed_balls = 0
        self.game_over = False
        self.paddle_y = 0
        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.ball_speed_multiplier = 1.0
        self.obstacles = []
        self.particles = []
        self.bubbles = []
        self.obstacle_sway_speed_increase = 0.0
        
        # This call to reset() will fail if the environment is not correctly initialized
        # We need to ensure all variables used in rendering are set up.
        # The error happens because `reset` calls `_get_observation` which calls `_render_game`,
        # but the `y` key for obstacles is only set in `step`. We must set it in `reset` too.
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.missed_balls = 0
        self.game_over = False
        
        self.paddle_y = self.HEIGHT / 2
        
        self._reset_ball()
        
        self.obstacles = [
            {'x': self.WIDTH * 0.4, 'y_center': self.HEIGHT * 0.25, 'amp': 40, 'freq': 0.02, 'phase': 0},
            {'x': self.WIDTH * 0.55, 'y_center': self.HEIGHT * 0.5, 'amp': 50, 'freq': 0.015, 'phase': math.pi},
            {'x': self.WIDTH * 0.7, 'y_center': self.HEIGHT * 0.75, 'amp': 40, 'freq': 0.025, 'phase': math.pi/2},
        ]
        self.obstacle_sway_speed_increase = 0.0
        
        # FIX: Initialize the 'y' coordinate for each obstacle before the first render.
        # This calculation is also done in step() to update their positions.
        for obs in self.obstacles:
            obs['y'] = obs['y_center'] + obs['amp'] * math.sin(self.steps * (obs['freq'] + self.obstacle_sway_speed_increase) + obs['phase'])

        self.particles = []
        self.bubbles = [self._create_bubble() for _ in range(30)]
        
        return self._get_observation(), self._get_info()
    
    def _reset_ball(self):
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        angle = self.np_random.uniform(-math.pi / 8, math.pi / 8)
        direction = 1 if self.score > self.missed_balls else self.np_random.choice([-1, 1])
        base_speed = self.INITIAL_BALL_SPEED
        self.ball_vel = [direction * base_speed * math.cos(angle), base_speed * math.sin(angle)]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = -0.01  # Small penalty for each step to encourage faster play

        # 1. Update Paddle
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        self.paddle_y = np.clip(self.paddle_y, self.PADDLE_HEIGHT / 2, self.HEIGHT - self.PADDLE_HEIGHT / 2)

        # 2. Update Ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        # 3. Update Obstacles and Difficulty
        self.obstacle_sway_speed_increase = (self.steps // 100) * 0.02
        for obs in self.obstacles:
            obs['y'] = obs['y_center'] + obs['amp'] * math.sin(self.steps * (obs['freq'] + self.obstacle_sway_speed_increase) + obs['phase'])

        # 4. Handle Collisions
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.top <= 0 or ball_rect.bottom >= self.HEIGHT:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # Sfx: wall_bounce

        # Paddle collision
        paddle_rect = pygame.Rect(self.PADDLE_X_POS - self.PADDLE_WIDTH / 2, self.paddle_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        if ball_rect.colliderect(paddle_rect) and self.ball_vel[0] < 0:
            self.ball_vel[0] *= -1.05  # Reverse and slightly speed up
            
            # Add "spin" based on hit location
            hit_offset = (self.ball_pos[1] - self.paddle_y) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[1] += hit_offset * 2.0
            
            # Increase ball speed after successful hit
            speed = math.hypot(*self.ball_vel)
            new_speed = min(speed + 0.05, self.INITIAL_BALL_SPEED * 2.5) # Cap speed
            if speed > 0:
                self.ball_vel = [v * new_speed / speed for v in self.ball_vel]

            self._create_sparks(self.ball_pos)
            reward += 0.1
            # Sfx: paddle_hit

        # Obstacle collisions
        for obs in self.obstacles:
            # Approximate seaweed as a vertical line segment
            seaweed_length = 120
            if obs['x'] - 10 < self.ball_pos[0] < obs['x'] + 10 and \
               obs['y'] - seaweed_length/2 < self.ball_pos[1] < obs['y'] + seaweed_length/2:
                if (self.ball_vel[0] > 0 and self.ball_pos[0] < obs['x']) or \
                   (self.ball_vel[0] < 0 and self.ball_pos[0] > obs['x']):
                    self.ball_vel[0] *= -1
                    speed = math.hypot(*self.ball_vel)
                    new_speed = max(speed * 0.8, self.MIN_BALL_SPEED) # Reduce speed by 20%
                    if speed > 0:
                        self.ball_vel = [v * new_speed / speed for v in self.ball_vel]
                    self._create_sparks(self.ball_pos, count=5)
                    # Sfx: obstacle_hit

        # 5. Check for Score or Miss
        if self.ball_pos[0] > self.WIDTH:
            self.score += 1
            reward += 10
            self._reset_ball()
            # Sfx: score
        elif self.ball_pos[0] < 0:
            self.missed_balls += 1
            reward -= 5
            self._reset_ball()
            # Sfx: miss

        # 6. Update Particles and Bubbles
        self._update_particles()
        self._update_bubbles()

        # 7. Update step counter and check for termination
        self.steps += 1
        terminated = self.score >= self.WIN_SCORE or self.missed_balls >= self.LOSE_MISSES or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                # Sfx: win
            elif self.missed_balls >= self.LOSE_MISSES:
                reward -= 100
                # Sfx: lose

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_DARK[0] * (1 - ratio) + self.COLOR_BG_LIGHT[0] * ratio),
                int(self.COLOR_BG_DARK[1] * (1 - ratio) + self.COLOR_BG_LIGHT[1] * ratio),
                int(self.COLOR_BG_DARK[2] * (1 - ratio) + self.COLOR_BG_LIGHT[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        for bubble in self.bubbles:
            pygame.gfxdraw.aacircle(self.screen, int(bubble['pos'][0]), int(bubble['pos'][1]), int(bubble['radius']), (255, 255, 255, bubble['alpha']))

    def _render_game(self):
        # Render Coral Goal
        self._draw_coral(self.WIDTH - 20, self.HEIGHT / 2, 120, 10)
        
        # Render Seaweed Obstacles
        for obs in self.obstacles:
            self._draw_seaweed(obs['x'], obs['y'], 120, 15)

        # Render Particles
        for p in self.particles:
            p_type = p.get('type', 'spark')
            if p_type == 'trail':
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), (*self.COLOR_BALL, p['alpha']))
            else: # spark
                pygame.draw.line(self.screen, (*self.COLOR_SPARK, p['alpha']), p['pos'], p['end_pos'], int(p['size']))

        # Render Ball
        bx, by = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, bx, by, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, bx, by, self.BALL_RADIUS, self.COLOR_BALL)
        # Highlight
        h_off = self.BALL_RADIUS * 0.4
        pygame.gfxdraw.filled_circle(self.screen, int(bx - h_off), int(by - h_off), int(self.BALL_RADIUS * 0.4), self.COLOR_BALL_HL)

        # Render Paddle
        px, py = int(self.PADDLE_X_POS), int(self.paddle_y)
        paddle_rect = pygame.Rect(px - self.PADDLE_WIDTH / 2, py - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_HL, paddle_rect.inflate(-4, -4), border_radius=2, width=1)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Misses
        miss_text = self.font_ui.render("MISSES:", True, self.COLOR_TEXT)
        self.screen.blit(miss_text, (20, 45))
        for i in range(self.LOSE_MISSES):
            color = self.COLOR_MISS if i < self.missed_balls else (80, 80, 80)
            pygame.gfxdraw.filled_circle(self.screen, 130 + i * 25, 60, 8, color)
            pygame.gfxdraw.aacircle(self.screen, 130 + i * 25, 60, 8, color)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surface = self.font_game_over.render(message, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_balls": self.missed_balls
        }
        
    def _create_bubble(self):
        return {
            'pos': [self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)],
            'radius': self.np_random.uniform(1, 5),
            'speed': self.np_random.uniform(0.2, 0.8),
            'alpha': int(self.np_random.uniform(30, 100))
        }

    def _update_bubbles(self):
        for bubble in self.bubbles:
            bubble['pos'][1] -= bubble['speed']
            if bubble['pos'][1] < -bubble['radius']:
                bubble['pos'] = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + bubble['radius']]

    def _create_sparks(self, pos, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'type': 'spark',
                'pos': list(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 20),
                'size': self.np_random.uniform(1, 3)
            })

    def _update_particles(self):
        # Add ball trail
        if self.steps % 2 == 0:
            self.particles.append({
                'type': 'trail',
                'pos': list(self.ball_pos),
                'vel': [0, 0],
                'lifespan': 15,
                'size': self.BALL_RADIUS,
            })

        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['lifespan'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['alpha'] = int(255 * (p['lifespan'] / 15)) if p.get('type') == 'trail' else int(255 * (p['lifespan'] / 20))
            if p.get('type') == 'trail':
                p['size'] *= 0.9
            else: # spark
                p['end_pos'] = [p['pos'][0] - p['vel'][0] * 0.5, p['pos'][1] - p['vel'][1] * 0.5]


    def _draw_seaweed(self, x, y_base, length, segments):
        points = []
        for i in range(segments + 1):
            progress = i / segments
            lx = x + math.sin(progress * 3 + self.steps * 0.05) * 10 * (1 - progress)
            ly = y_base - (length / 2) + progress * length
            radius = max(1, (8 - progress * 7))
            points.append({'pos': (lx, ly), 'radius': radius})
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            pygame.draw.line(self.screen, self.COLOR_SEAWEED, p1['pos'], p2['pos'], int(p1['radius'] + p2['radius']))
        for p in points:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), self.COLOR_SEAWEED)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), self.COLOR_SEAWEED)
            
    def _draw_coral(self, x, y, size, branches):
        for i in range(branches):
            angle = (i / branches) * math.pi * 1.5 - math.pi * 1.25
            length1 = size * self.np_random.uniform(0.4, 0.8)
            length2 = length1 * self.np_random.uniform(0.5, 0.8)
            angle2 = angle + self.np_random.uniform(-0.5, 0.5)
            
            p1 = (x, y)
            p2 = (x + math.cos(angle) * length1, y + math.sin(angle) * length1)
            p3 = (p2[0] + math.cos(angle2) * length2, p2[1] + math.sin(angle2) * length2)
            
            pygame.draw.line(self.screen, self.COLOR_CORAL, p1, p2, 10)
            pygame.draw.line(self.screen, self.COLOR_CORAL, p2, p3, 6)
            pygame.gfxdraw.filled_circle(self.screen, int(p1[0]), int(p1[1]), 5, self.COLOR_CORAL)
            pygame.gfxdraw.filled_circle(self.screen, int(p2[0]), int(p2[1]), 5, self.COLOR_CORAL)
            pygame.gfxdraw.filled_circle(self.screen, int(p3[0]), int(p3[1]), 3, self.COLOR_CORAL)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over_display = False
    
    # For human play, we need a real display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    
    # Override Pygame screen for display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Underwater Paddleball")
    
    total_reward = 0.0
    
    while running:
        action = env.action_space.sample() # Start with a random action
        action[0] = 0 # Default to no movement
        action[1] = 0
        action[2] = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                game_over_display = False

        if not game_over_display:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
                game_over_display = True
        
        # The environment already renders to its internal surface in _get_observation
        # which is called by step. We just need to blit it to the display screen.
        display_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(display_surface, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(60) # Run at 60 FPS for smoother human play

    env.close()