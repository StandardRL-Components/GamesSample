
# Generated: 2025-08-28T02:43:45.679216
# Source Brief: brief_01793.md
# Brief Index: 1793

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced block breaker where aggressive play and combos are rewarded. "
        "Catch falling power-ups to gain an advantage, but don't lose the ball!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.width, self.height = 640, 400
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game Constants ---
        self.MAX_STEPS = 2500
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG_TOP = (10, 5, 20)
        self.COLOR_BG_BOTTOM = (30, 15, 50)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_WALL = (60, 40, 90)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (0, 150, 255), (0, 200, 200), (100, 220, 100), 
            (255, 200, 0), (255, 100, 100), (200, 100, 255)
        ]

        # Paddle
        self.PADDLE_H = 16
        self.PADDLE_W_NORMAL = 100
        self.PADDLE_W_WIDE = 150
        self.PADDLE_SPEED = 12
        self.PADDLE_Y = self.height - 30

        # Ball
        self.BALL_RADIUS = 7
        self.BALL_SPEED_INITIAL = 6.0
        self.BALL_SPEED_MAX = 10.0

        # Power-ups
        self.POWERUP_SIZE = 12
        self.POWERUP_SPEED = 2.0
        self.POWERUP_CHANCE = 0.25 # 25% chance for a block to drop a power-up
        self.POWERUP_DURATION_STEPS = 450 # 15 seconds at 30fps
        self.POWERUP_TYPES = {
            "extra_life": {"color": (100, 255, 100)},
            "wide_paddle": {"color": (100, 100, 255)},
            "fireball": {"color": (255, 100, 0)},
        }

        # --- State Variables (initialized in reset) ---
        self.paddle = None
        self.ball = None
        self.ball_stuck = True
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.active_powerups = {}
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        
        self.combo_count = 0
        self.last_block_hit_step = 0
        self.last_paddle_hit_angle_change = 0.0

        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        # Reset paddle
        paddle_w = self.PADDLE_W_NORMAL
        self.paddle = pygame.Rect(
            (self.width - paddle_w) // 2, self.PADDLE_Y, paddle_w, self.PADDLE_H
        )

        # Reset ball
        self.ball = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_velocity = pygame.Vector2(0, 0)
        self.ball_stuck = True

        # Reset game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False

        # Reset lists
        self.particles.clear()
        self.powerups.clear()
        self.active_powerups.clear()

        # Reset combo
        self.combo_count = 0
        self.last_block_hit_step = 0
        
        # Generate blocks
        self._generate_blocks()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.02  # Time penalty
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Paddle movement
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.width - self.paddle.width)

        # Launch ball
        if self.ball_stuck and space_held:
            self._launch_ball()

        # --- Game Logic ---
        self._update_ball()
        self._update_powerups()
        self._update_particles()
        
        reward += self._handle_collisions()

        # Update active power-up timers
        ended_powerups = [k for k, v in self.active_powerups.items() if self.steps >= v]
        for p_type in ended_powerups:
            self._deactivate_powerup(p_type)

        # Update combo
        if self.steps - self.last_block_hit_step > 30: # 1-second combo window
            self.combo_count = 0

        self.steps += 1
        terminated = self._check_termination()
        
        # --- Final Reward Adjustments ---
        if self.last_paddle_hit_angle_change < 0.1 and self.last_paddle_hit_angle_change != 0:
            reward -= 0.2 # Penalty for "safe" play
        self.last_paddle_hit_angle_change = 0 # Reset for next step

        if terminated:
            if not self.blocks: # Win condition
                reward += 100
            elif self.lives <= 0: # Loss condition
                reward -= 50
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _launch_ball(self):
        self.ball_stuck = False
        angle = self.np_random.uniform(math.radians(225), math.radians(315))
        self.ball_velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.BALL_SPEED_INITIAL
        # sfx: launch_ball

    def _update_ball(self):
        if self.ball_stuck:
            self.ball.x = self.paddle.centerx
            self.ball.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball += self.ball_velocity

    def _update_powerups(self):
        for p in self.powerups[:]:
            p['rect'].y += self.POWERUP_SPEED
            if p['rect'].top > self.height:
                self.powerups.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        if self.ball_stuck:
            return reward

        # Ball vs Walls
        if self.ball.x - self.BALL_RADIUS < 0 or self.ball.x + self.BALL_RADIUS > self.width:
            self.ball_velocity.x *= -1
            self.ball.x = np.clip(self.ball.x, self.BALL_RADIUS, self.width - self.BALL_RADIUS)
            # sfx: wall_bounce
        if self.ball.y - self.BALL_RADIUS < 0:
            self.ball_velocity.y *= -1
            self.ball.y = np.clip(self.ball.y, self.BALL_RADIUS, self.height - self.BALL_RADIUS)
            # sfx: wall_bounce
        
        # Ball vs Bottom (lose life)
        if self.ball.y + self.BALL_RADIUS > self.height:
            self._lose_life()
            return reward

        # Ball vs Paddle
        if self.paddle.collidepoint(self.ball.x, self.ball.y + self.BALL_RADIUS) and self.ball_velocity.y > 0:
            self.ball.bottom = self.paddle.top
            
            # "English" on the ball
            offset = (self.ball.x - self.paddle.centerx) / (self.paddle.width / 2)
            old_vx = self.ball_velocity.x
            self.ball_velocity.x = self.BALL_SPEED_INITIAL * offset * 1.5
            self.ball_velocity.y *= -1
            
            # Normalize and scale speed
            speed = self.ball_velocity.length()
            if speed > 0:
                self.ball_velocity = self.ball_velocity.normalize() * min(speed * 1.02, self.BALL_SPEED_MAX)

            self.last_paddle_hit_angle_change = abs(self.ball_velocity.x - old_vx)
            self.combo_count = 0 # Reset combo on paddle hit
            # sfx: paddle_bounce
        
        # Ball vs Blocks
        is_fireball = 'fireball' in self.active_powerups
        for block in self.blocks[:]:
            if block['rect'].collidepoint(self.ball):
                reward += 0.1 # Hit reward
                if not is_fireball:
                    # Simple collision response
                    dx = self.ball.x - block['rect'].centerx
                    dy = self.ball.y - block['rect'].centery
                    if abs(dx / block['rect'].width) > abs(dy / block['rect'].height):
                        self.ball_velocity.x *= -1
                    else:
                        self.ball_velocity.y *= -1
                
                # Destruction logic
                self.blocks.remove(block)
                # sfx: block_destroy
                
                # Rewards
                reward += 1.0 # Destruction reward
                self.combo_count += 1
                if self.combo_count > 1:
                    reward += 10 * self.combo_count
                
                self.last_block_hit_step = self.steps
                self.score += 10 * self.combo_count
                
                self._spawn_particles(block['rect'].center, block['color'])
                
                if self.np_random.random() < self.POWERUP_CHANCE:
                    self._spawn_powerup(block['rect'].center)
                
                if not is_fireball:
                    break # Only break one block at a time unless fireball
        
        # Paddle vs Powerups
        for p in self.powerups[:]:
            if self.paddle.colliderect(p['rect']):
                self.powerups.remove(p)
                reward += 5.0
                self._activate_powerup(p['type'])
                # sfx: powerup_collect

        return reward

    def _lose_life(self):
        self.lives -= 1
        self.ball_stuck = True
        self.combo_count = 0
        self._deactivate_powerup('fireball') # Lose fireball on life loss
        # sfx: lose_life

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.height):
            ratio = y / self.height
            color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM))
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

    def _render_game_elements(self):
        # Particles (drawn first)
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'].x), int(p['pos'].y)))

        # Blocks
        for block in self.blocks:
            r = block['rect']
            pygame.draw.rect(self.screen, block['color'], r)
            # Simple 3D effect
            pygame.draw.rect(self.screen, tuple(min(255, c+30) for c in block['color']), (r.x, r.y, r.width, 3))
            pygame.draw.rect(self.screen, tuple(max(0, c-30) for c in block['color']), (r.x, r.bottom - 3, r.width, 3))

        # Powerups
        for p in self.powerups:
            pygame.draw.rect(self.screen, p['color'], p['rect'])
            pygame.draw.rect(self.screen, self.COLOR_TEXT, p['rect'], 1)

        # Paddle
        is_wide = 'wide_paddle' in self.active_powerups
        timer_frac = (self.active_powerups.get('wide_paddle', self.steps) - self.steps) / self.POWERUP_DURATION_STEPS
        paddle_color = self.COLOR_PADDLE
        if is_wide and timer_frac < 0.2: # Flash when timer is low
            if self.steps % 10 < 5:
                paddle_color = self.POWERUP_TYPES['wide_paddle']['color']
        
        pygame.draw.rect(self.screen, paddle_color, self.paddle, border_radius=4)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in paddle_color), self.paddle.inflate(-6, -6), border_radius=4)

        # Ball
        ball_color = self.COLOR_BALL
        is_fireball = 'fireball' in self.active_powerups
        if is_fireball:
            ball_color = self.POWERUP_TYPES['fireball']['color']
            # Fireball particle trail
            p_pos = self.ball.copy()
            p_vel = -self.ball_velocity.normalize() * 2
            p_lifespan = 15
            self.particles.append({'pos': p_pos, 'vel': p_vel, 'color': (255, 150, 0), 'lifespan': p_lifespan, 'max_lifespan': p_lifespan})

        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS, ball_color)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.x), int(self.ball.y), self.BALL_RADIUS, ball_color)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        for i in range(self.lives):
            pos_x = self.width - 20 - (i * (self.BALL_RADIUS * 2 + 5))
            pygame.gfxdraw.filled_circle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos_x, 20, self.BALL_RADIUS, self.COLOR_BALL)
            
        # Combo
        if self.combo_count > 1:
            combo_text = f"COMBO x{self.combo_count}!"
            combo_surf = self.font_ui.render(combo_text, True, self.COLOR_TEXT)
            pos = combo_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(combo_surf, pos)

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "YOU WIN!" if not self.blocks else "GAME OVER"
        text_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
            "combo": self.combo_count
        }

    def _generate_blocks(self):
        self.blocks.clear()
        cols, rows = 8, 6
        block_w, block_h = 64, 20
        gap = 4
        total_grid_w = cols * (block_w + gap) - gap
        start_x = (self.width - total_grid_w) // 2
        start_y = 60
        
        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() > 0.1: # 10% chance for a gap
                    x = start_x + c * (block_w + gap)
                    y = start_y + r * (block_h + gap)
                    rect = pygame.Rect(x, y, block_w, block_h)
                    color = self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)]
                    points = (rows - r) * 10
                    self.blocks.append({'rect': rect, 'color': color, 'points': points})
    
    def _spawn_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'color': color, 'lifespan': lifespan, 'max_lifespan': lifespan})
    
    def _spawn_powerup(self, pos):
        p_type = self.np_random.choice(list(self.POWERUP_TYPES.keys()))
        rect = pygame.Rect(pos[0] - self.POWERUP_SIZE//2, pos[1] - self.POWERUP_SIZE//2, self.POWERUP_SIZE, self.POWERUP_SIZE)
        self.powerups.append({
            'rect': rect,
            'type': p_type,
            'color': self.POWERUP_TYPES[p_type]['color']
        })

    def _activate_powerup(self, p_type):
        if p_type == 'extra_life':
            if self.lives < 5: # Max 5 lives
                self.lives += 1
        elif p_type == 'wide_paddle':
            self.paddle.width = self.PADDLE_W_WIDE
            self.active_powerups['wide_paddle'] = self.steps + self.POWERUP_DURATION_STEPS
        elif p_type == 'fireball':
            self.active_powerups['fireball'] = self.steps + self.POWERUP_DURATION_STEPS

    def _deactivate_powerup(self, p_type):
        if p_type in self.active_powerups:
            del self.active_powerups[p_type]
        
        if p_type == 'wide_paddle':
            self.paddle.width = self.PADDLE_W_NORMAL
        # Fireball deactivates automatically by checking the dict

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example usage:
if __name__ == '__main__':
    # To run with display, you might need to unset the dummy video driver
    # import os
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.width, env.height))
    running = True
    total_reward = 0
    
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # move, space, shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # obs, info = env.reset() # Uncomment to auto-reset
            # total_reward = 0

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate for playability

    env.close()