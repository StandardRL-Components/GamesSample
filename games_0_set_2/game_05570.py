
# Generated: 2025-08-28T05:24:53.530757
# Source Brief: brief_05570.md
# Brief Index: 5570

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball. Hold shift to use a collected power-up."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro block-breaking game. Clear all blocks to win. Collect power-ups to help you. Risky edge shots give bonus points!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1800 # 60 seconds at 30 FPS

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
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Consolas", 24)
            self.font_powerup = pygame.font.SysFont("Consolas", 14, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 30)
            self.font_powerup = pygame.font.Font(None, 20)

        # Colors
        self.COLOR_BG_TOP = (10, 10, 30)
        self.COLOR_BG_BOTTOM = (40, 20, 50)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [(50, 205, 50), (65, 105, 225), (220, 20, 60)] # Green, Blue, Red
        self.POWERUP_COLOR = (255, 215, 0) # Gold

        # Game entity properties
        self.PADDLE_WIDTH_NORMAL = 100
        self.PADDLE_WIDTH_EXTENDED = 150
        self.PADDLE_HEIGHT = 12
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 6
        self.BALL_SPEED_MAX = 12
        self.POWERUP_CHANCE = 0.25
        self.POWERUP_DURATION = 10 * self.FPS # 10 seconds

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paddle = None
        self.balls = []
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.lives = 0
        self.collected_powerup = None
        self.powerup_timers = {}
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3

        # Player paddle
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH_NORMAL) / 2, 
            self.HEIGHT - 40, 
            self.PADDLE_WIDTH_NORMAL, 
            self.PADDLE_HEIGHT
        )

        # Ball(s)
        self.balls = []
        self._spawn_ball(stuck=True)

        # Blocks
        self._generate_blocks()
        
        # Effects and items
        self.particles = []
        self.powerups = []
        self.collected_powerup = None
        self.powerup_timers = {
            'extend': 0,
            'sticky': 0,
        }
        
        return self._get_observation(), self._get_info()

    def _generate_blocks(self):
        self.blocks = []
        num_cols = 10
        num_rows = 5
        block_width = self.WIDTH // num_cols
        block_height = 20
        y_offset = 50

        for i in range(num_rows):
            for j in range(num_cols):
                block_rect = pygame.Rect(
                    j * block_width,
                    y_offset + i * block_height,
                    block_width - 2,
                    block_height - 2
                )
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                self.blocks.append({'rect': block_rect, 'color': color})

    def _spawn_ball(self, pos=None, vel=None, stuck=False):
        if pos is None:
            pos = self.paddle.midtop
        
        if vel is None:
            angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL
        
        ball_rect = pygame.Rect(pos[0] - self.BALL_RADIUS, pos[1] - self.BALL_RADIUS * 2, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        
        self.balls.append({
            'rect': ball_rect,
            'vel': vel,
            'stuck_to_paddle': stuck,
            'trail': []
        })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Unpack factorized action
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        space_pressed = action[1] == 1
        shift_pressed = action[2] == 1
        
        # Update game logic
        self.steps += 1
        reward = self._update_game_state(movement, space_pressed, shift_pressed)
        
        # Check termination conditions
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_state(self, movement, space_pressed, shift_pressed):
        step_reward = 0

        # 1. Handle Input
        self._handle_input(movement, space_pressed, shift_pressed)

        # 2. Update Power-up Timers
        self._update_powerups()

        # 3. Move Balls and Handle Collisions
        for ball in self.balls[:]:
            if ball['stuck_to_paddle']:
                ball['rect'].midbottom = self.paddle.midtop
                continue

            ball['trail'].append(ball['rect'].center)
            if len(ball['trail']) > 10:
                ball['trail'].pop(0)

            ball['rect'].x += ball['vel'].x
            ball['rect'].y += ball['vel'].y

            # Wall collisions
            if ball['rect'].left < 0 or ball['rect'].right > self.WIDTH:
                ball['vel'].x *= -1
                ball['rect'].left = max(0, ball['rect'].left)
                ball['rect'].right = min(self.WIDTH, ball['rect'].right)
                # sfx: wall_bounce

            if ball['rect'].top < 0:
                ball['vel'].y *= -1
                ball['rect'].top = max(0, ball['rect'].top)
                # sfx: wall_bounce

            if ball['rect'].top > self.HEIGHT:
                self.balls.remove(ball)
                continue

            # Paddle collision
            if self.paddle.colliderect(ball['rect']) and ball['vel'].y > 0:
                # sfx: paddle_hit
                if self.powerup_timers['sticky'] > 0:
                    ball['stuck_to_paddle'] = True
                    ball['vel'] = pygame.Vector2(0,0)
                else:
                    offset = (ball['rect'].centerx - self.paddle.centerx) / (self.paddle.width / 2)
                    normalized_offset = max(-1, min(1, offset))
                    
                    if abs(normalized_offset) > 0.8: # Edge 20%
                        step_reward += 0.1
                    elif abs(normalized_offset) < 0.6: # Center 60%
                        step_reward -= 0.02

                    rebound_angle = math.pi/2 - normalized_offset * (math.pi/3)
                    speed = ball['vel'].magnitude()
                    ball['vel'].x = -math.cos(rebound_angle) * speed
                    ball['vel'].y = -abs(math.sin(rebound_angle) * speed)

                    # Ensure ball is pushed out of paddle
                    ball['rect'].bottom = self.paddle.top

            # Block collisions
            for block in self.blocks[:]:
                if block['rect'].colliderect(ball['rect']):
                    # sfx: block_break
                    self.blocks.remove(block)
                    step_reward += 1
                    self.score += 10
                    self._create_particles(block['rect'].center, block['color'])
                    
                    if self.np_random.random() < self.POWERUP_CHANCE:
                        self._spawn_powerup(block['rect'].center)
                    
                    # Determine bounce direction
                    ball_center = pygame.Vector2(ball['rect'].center)
                    block_center = pygame.Vector2(block['rect'].center)
                    dx = ball_center.x - block_center.x
                    dy = ball_center.y - block_center.y
                    
                    if abs(dx / block['rect'].width) > abs(dy / block['rect'].height):
                        ball['vel'].x *= -1
                    else:
                        ball['vel'].y *= -1
                    break
        
        # 4. Handle losing a life
        if not self.balls:
            self.lives -= 1
            if self.lives > 0:
                self._spawn_ball(stuck=True)
                # sfx: lose_life
            else:
                self.game_over = True
                step_reward -= 100
                # sfx: game_over
        
        # 5. Move falling powerups
        for pu in self.powerups[:]:
            pu['rect'].y += 3
            if pu['rect'].colliderect(self.paddle):
                # sfx: powerup_collect
                self.collected_powerup = pu['type']
                self.powerups.remove(pu)
                step_reward += 5
            elif pu['rect'].top > self.HEIGHT:
                self.powerups.remove(pu)
        
        # 6. Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        self.score += step_reward
        return step_reward

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Paddle Movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        # Launch Ball
        if space_pressed:
            for ball in self.balls:
                if ball['stuck_to_paddle']:
                    ball['stuck_to_paddle'] = False
                    angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
                    ball['vel'] = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL
                    # sfx: ball_launch

        # Use Power-up
        if shift_pressed and self.collected_powerup:
            self._activate_powerup()

    def _activate_powerup(self):
        # sfx: powerup_activate
        pu_type = self.collected_powerup
        self.collected_powerup = None

        if pu_type == 'multi-ball':
            if self.balls:
                original_ball = self.balls[0]
                pos = original_ball['rect'].center
                vel = original_ball['vel']
                
                # Create two new balls with slightly altered trajectories
                if vel.length() > 0:
                    angle1 = math.atan2(vel.y, vel.x) + 0.3
                    angle2 = math.atan2(vel.y, vel.x) - 0.3
                    speed = vel.length()
                    vel1 = pygame.Vector2(math.cos(angle1), math.sin(angle1)) * speed
                    vel2 = pygame.Vector2(math.cos(angle2), math.sin(angle2)) * speed
                    self._spawn_ball(pos, vel1)
                    self._spawn_ball(pos, vel2)

        elif pu_type == 'extend':
            self.powerup_timers['extend'] = self.POWERUP_DURATION
        
        elif pu_type == 'sticky':
            self.powerup_timers['sticky'] = self.POWERUP_DURATION

    def _update_powerups(self):
        # Extend Paddle
        if self.powerup_timers['extend'] > 0:
            self.powerup_timers['extend'] -= 1
            self.paddle.width = self.PADDLE_WIDTH_EXTENDED
        else:
            self.paddle.width = self.PADDLE_WIDTH_NORMAL
        
        # Sticky Paddle
        if self.powerup_timers['sticky'] > 0:
            self.powerup_timers['sticky'] -= 1

    def _spawn_powerup(self, pos):
        pu_type = self.np_random.choice(['multi-ball', 'extend', 'sticky'])
        pu_rect = pygame.Rect(pos[0] - 10, pos[1] - 10, 20, 20)
        self.powerups.append({'rect': pu_rect, 'type': pu_type})

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2))
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'life': life, 'color': color})

    def _check_termination(self):
        if not self.blocks:
            self.game_over = True
            self.score += 100 # Win bonus
            return True
        if self.lives <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "active_powerup": self.collected_powerup,
        }

    def _render_all(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block['color']), block['rect'], 2)

        # Draw falling powerups
        for pu in self.powerups:
            pygame.gfxdraw.filled_circle(self.screen, pu['rect'].centerx, pu['rect'].centery, 10, self.POWERUP_COLOR)
            pygame.gfxdraw.aacircle(self.screen, pu['rect'].centerx, pu['rect'].centery, 10, self.POWERUP_COLOR)
            pu_char = pu['type'][0].upper()
            text_surf = self.font_powerup.render(pu_char, True, self.COLOR_BG_BOTTOM)
            text_rect = text_surf.get_rect(center=pu['rect'].center)
            self.screen.blit(text_surf, text_rect)

        # Draw particles
        for p in self.particles:
            size = int(p['life'] / 5)
            if size > 0:
                pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y), size, size))

        # Draw ball trails
        for ball in self.balls:
            if len(ball['trail']) > 1:
                points = [(int(p[0]), int(p[1])) for p in ball['trail']]
                pygame.draw.aalines(self.screen, self.COLOR_BALL, False, points, 1)

        # Draw paddle
        paddle_color = self.COLOR_PADDLE
        if self.powerup_timers['sticky'] > 0:
            paddle_color = (100, 100, 255) # Blue tint for sticky
        pygame.draw.rect(self.screen, paddle_color, self.paddle, border_radius=3)
        
        # Draw balls
        for ball in self.balls:
            pygame.gfxdraw.filled_circle(self.screen, int(ball['rect'].centerx), int(ball['rect'].centery), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(ball['rect'].centerx), int(ball['rect'].centery), self.BALL_RADIUS, self.COLOR_BALL)

        # Draw UI
        self._render_ui()

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            pos = (self.WIDTH - 30 - i * 25, 15)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_PADDLE)
        
        # Collected Power-up
        if self.collected_powerup:
            pu_rect = pygame.Rect(self.WIDTH // 2 - 20, 5, 40, 30)
            pygame.draw.rect(self.screen, (30,30,60), pu_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.POWERUP_COLOR, pu_rect, 2, border_radius=5)
            pu_char = self.collected_powerup[0].upper()
            text_surf = self.font_powerup.render(pu_char, True, self.POWERUP_COLOR)
            text_rect = text_surf.get_rect(center=pu_rect.center)
            self.screen.blit(text_surf, text_rect)
    
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
    
    running = True
    total_reward = 0
    
    while running:
        # Player input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()