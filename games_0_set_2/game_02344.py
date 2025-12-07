
# Generated: 2025-08-27T20:05:11.289414
# Source Brief: brief_02344.md
# Brief Index: 2344

        
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

    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    game_description = (
        "A fast-paced, top-down block breaker where risky plays are rewarded. "
        "Break all the blocks to win, but lose all your balls and you lose the game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.PADDLE_SPEED = 10
        self.BALL_BASE_SPEED = 5
        self.PADDLE_SPIN_FACTOR = 3.0
        self.POWERUP_SPEED = 2
        self.POWERUP_CHANCE = 0.25 # 25% chance to spawn a powerup

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_big = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_big = pygame.font.SysFont("arial", 24)
            self.font_small = pygame.font.SysFont("arial", 16)


        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_GRID = (30, 30, 60)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (200, 200, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.BLOCK_COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (80, 255, 255), (255, 80, 255)
        ]

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_attached = False
        self.blocks = []
        self.particles = []
        self.powerups = []
        self.paddle_expand_timer = 0
        self.ball_speed_timer = 0
        self.current_ball_speed = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 3
        
        # Powerup timers
        self.paddle_expand_timer = 0
        self.ball_speed_timer = 0
        
        # Paddle
        paddle_width = 100
        paddle_height = 15
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - paddle_width) // 2,
            self.HEIGHT - 30,
            paddle_width,
            paddle_height
        )

        # Ball
        self._reset_ball()

        # Blocks
        self.blocks = []
        num_blocks_x = 10
        num_blocks_y = 5
        block_width = self.WIDTH // num_blocks_x
        block_height = 20
        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                block_rect = pygame.Rect(
                    j * block_width,
                    i * block_height + 50,
                    block_width - 2,
                    block_height - 2
                )
                color = self.BLOCK_COLORS[(i + j) % len(self.BLOCK_COLORS)]
                self.blocks.append({'rect': block_rect, 'color': color})

        self.particles = []
        self.powerups = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_attached = True
        self.ball_pos = pygame.Vector2(self.paddle_rect.centerx, self.paddle_rect.top - 10)
        self.ball_vel = pygame.Vector2(0, 0)
        self.current_ball_speed = self.BALL_BASE_SPEED

    def step(self, action):
        self.clock.tick(30) # Maintain 30 FPS for interpolation
        reward = -0.02 # Small penalty per step to encourage speed

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Paddle movement
        if movement == 3: # Left
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4: # Right
            self.paddle_rect.x += self.PADDLE_SPEED
        self.paddle_rect.clamp_ip(self.screen.get_rect())

        # Launch ball
        if self.ball_attached and space_held:
            # sfx: ball_launch
            self.ball_attached = False
            angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
            self.ball_vel = pygame.Vector2(math.sin(angle), -math.cos(angle)) * self.current_ball_speed

        # --- Game Logic ---
        self._update_powerup_timers()

        if self.ball_attached:
            self.ball_pos.x = self.paddle_rect.centerx
        else:
            reward += self._update_ball()

        self._update_powerups()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        # --- Final Rewards ---
        if terminated:
            if len(self.blocks) == 0:
                reward += 100 # Win bonus
            elif self.balls_left <= 0:
                reward -= 100 # Lose penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        reward = 0
        self.ball_pos += self.ball_vel

        ball_radius = 8
        ball_rect = pygame.Rect(self.ball_pos.x - ball_radius, self.ball_pos.y - ball_radius, ball_radius*2, ball_radius*2)

        # Wall collisions
        if self.ball_pos.x <= ball_radius or self.ball_pos.x >= self.WIDTH - ball_radius:
            self.ball_vel.x *= -1
            self.ball_pos.x = max(ball_radius, min(self.WIDTH - ball_radius, self.ball_pos.x))
            # sfx: wall_bounce
        if self.ball_pos.y <= ball_radius:
            self.ball_vel.y *= -1
            self.ball_pos.y = max(ball_radius, self.ball_pos.y)
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball_vel.y > 0:
            # sfx: paddle_hit
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle_rect.top - ball_radius

            offset = self.ball_pos.x - self.paddle_rect.centerx
            normalized_offset = offset / (self.paddle_rect.width / 2)
            
            # Reward for precise hits
            if abs(normalized_offset) < 0.25:
                reward += 0.1
            else:
                reward -= 0.1

            self.ball_vel.x += normalized_offset * self.PADDLE_SPIN_FACTOR
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.current_ball_speed

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            # sfx: block_break
            block = self.blocks.pop(hit_block_idx)
            reward += 10
            self.score += 10
            self.ball_vel.y *= -1 # Simple vertical bounce
            self._spawn_particles(block['rect'].center, block['color'])
            if self.np_random.random() < self.POWERUP_CHANCE:
                self._spawn_powerup(block['rect'].center)

        # Lose ball
        if self.ball_pos.y >= self.HEIGHT:
            # sfx: lose_ball
            self.balls_left -= 1
            if self.balls_left > 0:
                self._reset_ball()
            else:
                self.game_over = True
        
        return reward
        
    def _update_powerups(self):
        for powerup in self.powerups[:]:
            powerup['rect'].y += self.POWERUP_SPEED
            powerup['phase'] = (powerup['phase'] + 0.2) % (2 * math.pi)

            if self.paddle_rect.colliderect(powerup['rect']):
                # sfx: powerup_collect
                self.powerups.remove(powerup)
                if powerup['type'] == 'expand':
                    self.paddle_expand_timer = 300 # 10 seconds
                    self.score += 2
                    # self.reward += 2 # Reward is handled by caller
                elif powerup['type'] == 'speed':
                    self.ball_speed_timer = 300 # 10 seconds
                    self.score += 5
                    # self.reward += 5
            elif powerup['rect'].top > self.HEIGHT:
                self.powerups.remove(powerup)

    def _update_powerup_timers(self):
        # Paddle expand
        if self.paddle_expand_timer > 0:
            self.paddle_expand_timer -= 1
            self.paddle_rect.width = 150
        else:
            self.paddle_rect.width = 100
        
        # Ball speed
        if self.ball_speed_timer > 0:
            self.ball_speed_timer -= 1
            self.current_ball_speed = self.BALL_BASE_SPEED * 1.5
        else:
            self.current_ball_speed = self.BALL_BASE_SPEED
        
        # Update ball velocity if it's in motion
        if not self.ball_attached and self.ball_vel.length() > 0:
            self.ball_vel.normalize_ip()
            self.ball_vel *= self.current_ball_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] -= 0.2
            if p['lifetime'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifetime': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })

    def _spawn_powerup(self, pos):
        ptype = self.np_random.choice(['expand', 'speed'])
        rect = pygame.Rect(pos[0] - 10, pos[1] - 10, 20, 20)
        self.powerups.append({'rect': rect, 'type': ptype, 'phase': 0})

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or len(self.blocks) == 0

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "balls": self.balls_left}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])

        # Powerups
        for p in self.powerups:
            r = int(127 + 127 * math.sin(p['phase']))
            g = int(127 + 127 * math.sin(p['phase'] + 2 * math.pi / 3))
            b = int(127 + 127 * math.sin(p['phase'] + 4 * math.pi / 3))
            color = (r, g, b)
            pygame.draw.rect(self.screen, color, p['rect'], border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, p['rect'], width=1, border_radius=5)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        
        # Ball
        ball_pos_int = (int(self.ball_pos.x), int(self.ball_pos.y))
        ball_radius = 8
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], ball_radius + 4, (*self.COLOR_BALL_GLOW, 100))
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], ball_radius + 4, (*self.COLOR_BALL_GLOW, 100))
        # Main ball
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_big.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls
        balls_text = self.font_big.render(f"BALLS: {self.balls_left}", True, self.COLOR_TEXT)
        self.screen.blit(balls_text, (self.WIDTH - balls_text.get_width() - 10, 10))

        # Blocks Remaining
        blocks_text = self.font_small.render(f"BLOCKS: {len(self.blocks)}", True, self.COLOR_TEXT)
        self.screen.blit(blocks_text, ((self.WIDTH - blocks_text.get_width()) // 2, self.HEIGHT - 20))

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value to run headless
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    total_reward = 0
    
    print("--- Playing Game ---")
    print(env.user_guide)

    while running:
        # Action defaults
        movement, space, shift = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000)

    env.close()