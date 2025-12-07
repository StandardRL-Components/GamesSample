
# Generated: 2025-08-28T00:37:14.144789
# Source Brief: brief_03844.md
# Brief Index: 3844

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball. "
        "Hold shift to activate a collected power-up."
    )

    game_description = (
        "A retro arcade Breakout game. Destroy all bricks, collect power-ups, "
        "and aim for a high score before you run out of lives."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PADDLE_BASE_WIDTH, self.PADDLE_HEIGHT = 80, 12
        self.BALL_RADIUS = 6
        self.BRICK_WIDTH, self.BRICK_HEIGHT = 56, 20
        self.POWERUP_SIZE = 12
        self.MAX_STEPS = 30 * 100 # Approx 100 seconds at 30fps
        self.WIN_SCORE = 500

        # --- Colors ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BORDER = (80, 80, 100)
        self.COLOR_PADDLE = (200, 200, 255)
        self.COLOR_BALL = (255, 255, 255)
        self.BRICK_COLORS = {
            10: (50, 200, 50),  # Green
            20: (50, 100, 220), # Blue
            30: (220, 50, 50),  # Red
        }
        self.POWERUP_COLORS = {
            'multi_ball': (255, 255, 0),  # Yellow
            'extended_paddle': (200, 0, 255), # Purple
            'fireball': (255, 128, 0),    # Orange
        }

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.balls = []
        self.bricks = []
        self.falling_powerups = []
        self.active_powerups = {}
        self.collected_powerup = None
        self.particles = []
        self.last_space_held = False
        self.last_shift_held = False
        self.reward_this_step = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False

        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_BASE_WIDTH) / 2,
            self.HEIGHT - 40,
            self.PADDLE_BASE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.balls = []
        self._spawn_ball(held=True)

        self._create_bricks()
        self.falling_powerups = []
        self.active_powerups = {}
        self.collected_powerup = None
        self.particles = []
        
        self.last_space_held = False
        self.last_shift_held = False
        self.reward_this_step = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = -0.02 # Small penalty for time passing

        self._handle_input(action)
        self._update_paddle(action)
        self._update_balls()
        self._update_powerups()
        self._update_particles()

        self.steps += 1
        reward = self.reward_this_step
        terminated = self._check_termination()
        
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 10 # Lose penalty (already applied in _lose_life)
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Launch ball on space press
        if space_held and not self.last_space_held:
            for ball in self.balls:
                if ball['state'] == 'held':
                    ball['state'] = 'moving'
                    # sfx: ball_launch
        
        # Activate power-up on shift press
        if shift_held and not self.last_shift_held and self.collected_powerup:
            self._activate_powerup(self.collected_powerup)
            self.collected_powerup = None
            # sfx: powerup_activate

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_paddle(self, action):
        movement = action[0]
        paddle_speed = 8
        if movement == 3: # Left
            self.paddle.x -= paddle_speed
        elif movement == 4: # Right
            self.paddle.x += paddle_speed
        
        self.paddle.x = max(10, min(self.WIDTH - 10 - self.paddle.width, self.paddle.x))

    def _update_balls(self):
        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            if ball['state'] == 'held':
                ball['rect'].centerx = self.paddle.centerx
                ball['rect'].bottom = self.paddle.top
                continue

            ball['rect'].x += ball['vel'][0]
            ball['rect'].y += ball['vel'][1]

            # Wall collisions
            if ball['rect'].left <= 10 or ball['rect'].right >= self.WIDTH - 10:
                ball['vel'][0] *= -1
                ball['rect'].x = max(10, min(self.WIDTH - 10 - ball['rect'].width, ball['rect'].x))
                # sfx: wall_bounce
            if ball['rect'].top <= 10:
                ball['vel'][1] *= -1
                # sfx: wall_bounce
            
            # Paddle collision
            if ball['rect'].colliderect(self.paddle) and ball['vel'][1] > 0:
                ball['vel'][1] *= -1
                
                # Change horizontal velocity based on hit location
                offset = (ball['rect'].centerx - self.paddle.centerx) / (self.paddle.width / 2)
                ball['vel'][0] += offset * 3
                ball['vel'][0] = max(-8, min(8, ball['vel'][0])) # Cap horizontal speed
                ball['rect'].bottom = self.paddle.top # Prevent sticking
                # sfx: paddle_hit

            # Brick collision
            hit_brick = False
            for brick in self.bricks:
                if brick['active'] and ball['rect'].colliderect(brick['rect']):
                    hit_brick = True
                    self.reward_this_step += 0.1
                    is_fireball = self.active_powerups.get('fireball', 0) > 0
                    
                    if not is_fireball:
                        self._handle_ball_brick_collision(ball, brick)
                    
                    self._destroy_brick(brick)
                    break # Only one brick per step per ball
            
            if hit_brick and not self.active_powerups.get('fireball', 0) > 0:
                # sfx: brick_hit
                pass
            
            # Bottom of screen (lose life)
            if ball['rect'].top >= self.HEIGHT:
                balls_to_remove.append(i)

        for i in sorted(balls_to_remove, reverse=True):
            del self.balls[i]

        if not self.balls:
            self._lose_life()

    def _handle_ball_brick_collision(self, ball, brick):
        ball_rect, brick_rect = ball['rect'], brick['rect']
        
        # Determine collision side to correctly flip velocity
        overlap = ball_rect.clip(brick_rect)
        
        if overlap.width > overlap.height:
            # Vertical collision (top/bottom of brick)
            ball['vel'][1] *= -1
            # Move ball out of collision
            if ball_rect.centery < brick_rect.centery:
                ball_rect.bottom = brick_rect.top
            else:
                ball_rect.top = brick_rect.bottom
        else:
            # Horizontal collision (left/right of brick)
            ball['vel'][0] *= -1
            # Move ball out of collision
            if ball_rect.centerx < brick_rect.centerx:
                ball_rect.right = brick_rect.left
            else:
                ball_rect.left = brick_rect.right

    def _destroy_brick(self, brick):
        brick['active'] = False
        self.score += brick['points']
        self.reward_this_step += brick['points']
        self._spawn_particles(brick['rect'].center, self.BRICK_COLORS[brick['points']], 20)
        # sfx: brick_destroy
        
        if self.np_random.random() < 0.2: # 20% chance to spawn power-up
            self._spawn_powerup(brick['rect'].center)

    def _update_powerups(self):
        # Update falling power-ups
        powerups_to_remove = []
        for i, pu in enumerate(self.falling_powerups):
            pu['rect'].y += 2
            if pu['rect'].top > self.HEIGHT:
                powerups_to_remove.append(i)
            elif pu['rect'].colliderect(self.paddle) and not self.collected_powerup:
                self.collected_powerup = pu['type']
                powerups_to_remove.append(i)
                self.reward_this_step += 1
                # sfx: powerup_collect
        
        for i in sorted(powerups_to_remove, reverse=True):
            del self.falling_powerups[i]
            
        # Update active power-ups
        expired_powerups = []
        for key, timer in self.active_powerups.items():
            self.active_powerups[key] = timer - 1
            if self.active_powerups[key] <= 0:
                expired_powerups.append(key)
        
        for key in expired_powerups:
            del self.active_powerups[key]
            if key == 'extended_paddle':
                self.paddle.width = self.PADDLE_BASE_WIDTH
                self.paddle.x += (self.PADDLE_BASE_WIDTH) / 2 # Center it
            # sfx: powerup_expire

    def _activate_powerup(self, powerup_type):
        duration = 10 * 30 # 10 seconds
        self.active_powerups[powerup_type] = duration
        
        if powerup_type == 'multi_ball':
            if self.balls:
                original_ball = self.balls[0]
                for angle in [-30, 30]:
                    self._spawn_ball(
                        pos=original_ball['rect'].center,
                        vel=pygame.math.Vector2(original_ball['vel']).rotate(angle)
                    )
        elif powerup_type == 'extended_paddle':
            self.paddle.x -= self.PADDLE_BASE_WIDTH / 2
            self.paddle.width = self.PADDLE_BASE_WIDTH * 2

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 10)

        # Bricks
        for brick in self.bricks:
            if brick['active']:
                pygame.draw.rect(self.screen, self.BRICK_COLORS[brick['points']], brick['rect'])
                pygame.draw.rect(self.screen, self.COLOR_BG, brick['rect'], 1)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Falling Power-ups
        for pu in self.falling_powerups:
            pygame.draw.rect(self.screen, self.POWERUP_COLORS[pu['type']], pu['rect'], border_radius=3)
            # Simple icon drawing
            if pu['type'] == 'multi_ball':
                pygame.draw.circle(self.screen, self.COLOR_BG, pu['rect'].center, 2)
            elif pu['type'] == 'extended_paddle':
                pygame.draw.line(self.screen, self.COLOR_BG, (pu['rect'].left+2, pu['rect'].centery), (pu['rect'].right-2, pu['rect'].centery), 2)
            elif pu['type'] == 'fireball':
                pygame.gfxdraw.filled_circle(self.screen, pu['rect'].centerx, pu['rect'].centery, 3, self.COLOR_BG)

        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, p['pos'])

        # Balls
        is_fireball = self.active_powerups.get('fireball', 0) > 0
        for ball in self.balls:
            if is_fireball:
                # Fireball trail effect
                for i in range(5):
                    alpha = 150 - i * 30
                    radius = self.BALL_RADIUS * (1 - i*0.1)
                    pos = (
                        ball['rect'].centerx - ball['vel'][0] * i * 0.2,
                        ball['rect'].centery - ball['vel'][1] * i * 0.2
                    )
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), (*self.POWERUP_COLORS['fireball'], alpha))
            
            pygame.gfxdraw.filled_circle(self.screen, ball['rect'].centerx, ball['rect'].centery, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, ball['rect'].centerx, ball['rect'].centery, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:05d}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 15))
        
        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 30 - i * 25, 30, 8, self.COLOR_PADDLE)

        # Collected Power-up
        if self.collected_powerup:
            pu_rect = pygame.Rect(self.paddle.centerx - 12, self.paddle.bottom + 5, 24, 24)
            pygame.draw.rect(self.screen, self.POWERUP_COLORS[self.collected_powerup], pu_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_BG, pu_rect, 2, border_radius=4)
            
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _lose_life(self):
        self.lives -= 1
        self.reward_this_step -= 10
        # sfx: life_lost
        if self.lives > 0:
            self._spawn_ball(held=True)
            # Reset active powerups on life loss for balance
            self.active_powerups = {}
            self.paddle.width = self.PADDLE_BASE_WIDTH
            self.paddle.centerx = self.WIDTH / 2


    def _create_bricks(self):
        self.bricks = []
        brick_points_layout = [30, 30, 20, 20, 10]
        total_brick_width = 10 * self.BRICK_WIDTH + 9 * 4
        start_x = (self.WIDTH - total_brick_width) / 2
        start_y = 60
        for r in range(5):
            for c in range(10):
                x = start_x + c * (self.BRICK_WIDTH + 4)
                y = start_y + r * (self.BRICK_HEIGHT + 4)
                points = brick_points_layout[r]
                self.bricks.append({
                    'rect': pygame.Rect(x, y, self.BRICK_WIDTH, self.BRICK_HEIGHT),
                    'points': points,
                    'active': True
                })

    def _spawn_ball(self, held=False, pos=None, vel=None):
        if pos is None:
            pos = (self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        if vel is None:
            angle = self.np_random.uniform(-45, 45)
            vel_vec = pygame.math.Vector2(0, -6).rotate(angle)
            vel = [vel_vec.x, vel_vec.y]

        self.balls.append({
            'rect': pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2),
            'vel': vel,
            'state': 'held' if held else 'moving'
        })
        self.balls[-1]['rect'].center = pos
        
    def _spawn_powerup(self, pos):
        pu_type = self.np_random.choice(list(self.POWERUP_COLORS.keys()))
        self.falling_powerups.append({
            'rect': pygame.Rect(pos[0] - self.POWERUP_SIZE/2, pos[1], self.POWERUP_SIZE, self.POWERUP_SIZE),
            'type': pu_type
        })
        
    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(math.radians(angle)) * speed, math.sin(math.radians(angle)) * speed],
                'life': life,
                'max_life': life,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Player Input ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Game Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        clock.tick(30) # Run at 30 FPS

    env.close()