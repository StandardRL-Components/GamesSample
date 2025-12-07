
# Generated: 2025-08-28T01:40:17.317537
# Source Brief: brief_04190.md
# Brief Index: 4190

        
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
        "A retro arcade block-breaker. Clear all the blocks by bouncing the ball off your paddle. "
        "Some blocks drop power-ups when destroyed. Don't lose all your lives!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 80, 12
        self.PADDLE_SPEED = 8
        self.BALL_RADIUS = 6
        self.INITIAL_BALL_SPEED = 3.0
        self.BLOCK_ROWS, self.BLOCK_COLS = 5, 10
        
        # Reward structure
        self.REWARD_HIT_BLOCK = 1
        self.REWARD_POWERUP = 10
        self.REWARD_LAST_BLOCK = 50
        self.REWARD_WIN = 100
        self.REWARD_LOSE = -100
        self.REWARD_MOVE = -0.01 # Encourage efficiency

        # Colors
        self.COLOR_BG = (20, 20, 35)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = {
            10: (0, 200, 100), # Green
            20: (0, 150, 220), # Blue
            30: (220, 50, 100), # Red
        }
        self.POWERUP_COLORS = {
            "multi_ball": (255, 255, 0), # Yellow
            "extended_paddle": (200, 0, 255), # Purple
        }

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle = None
        self.balls = []
        self.blocks = []
        self.powerups = []
        self.particles = []
        self.active_effects = {}
        self.ball_launched = False
        self.base_ball_speed = self.INITIAL_BALL_SPEED
        self.blocks_destroyed_count = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.balls = []
        self._reset_ball()

        self._create_blocks()
        
        self.powerups = []
        self.particles = []
        self.active_effects = {}
        self.ball_launched = False
        self.base_ball_speed = self.INITIAL_BALL_SPEED
        self.blocks_destroyed_count = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Update paddle
        if movement == 3: # Left
            self.paddle.x -= self.PADDLE_SPEED
            reward += self.REWARD_MOVE
        elif movement == 4: # Right
            self.paddle.x += self.PADDLE_SPEED
            reward += self.REWARD_MOVE
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.paddle.width)

        # Launch ball
        if space_held and not self.ball_launched:
            self.ball_launched = True
            # sfx: launch_ball
            self.balls[0]['vel'] = [
                self.np_random.uniform(-0.5, 0.5), 
                -1
            ]
            self._normalize_ball_velocity(0)

        # Update game objects
        self._update_balls()
        reward += self._handle_ball_collisions()
        self._update_powerups()
        reward += self._handle_powerup_collisions()
        self._update_effects()
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.lives <= 0:
                reward += self.REWARD_LOSE
            elif not self.blocks:
                reward += self.REWARD_WIN + self.REWARD_LAST_BLOCK

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_blocks(self):
        self.blocks = []
        block_w = self.WIDTH // self.BLOCK_COLS
        block_h = 20
        points_map = [30, 30, 20, 20, 10]
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                points = points_map[i]
                block_rect = pygame.Rect(
                    j * block_w,
                    i * block_h + 50,
                    block_w,
                    block_h
                )
                self.blocks.append({'rect': block_rect, 'points': points, 'color': self.BLOCK_COLORS[points]})

    def _reset_ball(self):
        self.ball_launched = False
        ball_pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS]
        self.balls = [{'pos': ball_pos, 'vel': [0, 0]}]

    def _normalize_ball_velocity(self, ball_index):
        vel = self.balls[ball_index]['vel']
        norm = math.sqrt(vel[0]**2 + vel[1]**2)
        if norm > 0:
            vel[0] = (vel[0] / norm) * self.base_ball_speed
            vel[1] = (vel[1] / norm) * self.base_ball_speed

    def _update_balls(self):
        if not self.ball_launched:
            self.balls[0]['pos'][0] = self.paddle.centerx
            return

        for ball in self.balls:
            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]

    def _handle_ball_collisions(self):
        reward = 0
        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            ball_rect = pygame.Rect(ball['pos'][0] - self.BALL_RADIUS, ball['pos'][1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            # Wall collisions
            if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
                ball['vel'][0] *= -1
                ball['pos'][0] = np.clip(ball['pos'][0], self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # sfx: wall_bounce
            if ball_rect.top <= 0:
                ball['vel'][1] *= -1
                ball['pos'][1] = np.clip(ball['pos'][1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
                # sfx: wall_bounce

            # Paddle collision
            if ball_rect.colliderect(self.paddle) and ball['vel'][1] > 0:
                # sfx: paddle_hit
                ball['vel'][1] *= -1
                ball_rect.bottom = self.paddle.top
                ball['pos'][1] = ball_rect.centery
                
                # Influence angle based on hit position
                offset = (ball_rect.centerx - self.paddle.centerx) / (self.paddle.width / 2)
                ball['vel'][0] = offset * self.base_ball_speed * 1.2
                self._normalize_ball_velocity(i)

            # Block collisions
            collided_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
            if collided_block_idx != -1:
                # sfx: block_explode
                block = self.blocks.pop(collided_block_idx)
                reward += self.REWARD_HIT_BLOCK
                self.score += block['points']
                self.blocks_destroyed_count += 1
                
                # Increase ball speed every 10 blocks
                if self.blocks_destroyed_count % 10 == 0:
                    self.base_ball_speed += 0.5

                # Create particles
                self._create_particles(block['rect'].center, block['color'], 20)

                # Determine bounce direction
                if abs(ball_rect.centerx - block['rect'].centerx) > abs(ball_rect.centery - block['rect'].centery):
                    ball['vel'][0] *= -1
                else:
                    ball['vel'][1] *= -1

                # Spawn powerup
                if self.np_random.random() < 0.2: # 20% chance
                    self._spawn_powerup(block['rect'].center)

            # Ball lost
            if ball_rect.top >= self.HEIGHT:
                balls_to_remove.append(ball)
        
        for ball in balls_to_remove:
            self.balls.remove(ball)

        if not self.balls and self.ball_launched:
            # sfx: lose_life
            self.lives -= 1
            if self.lives > 0:
                self._reset_ball()
                self._deactivate_effects()

        return reward

    def _spawn_powerup(self, pos):
        ptype = self.np_random.choice(["multi_ball", "extended_paddle"])
        powerup_rect = pygame.Rect(pos[0] - 10, pos[1] - 10, 20, 20)
        self.powerups.append({'rect': powerup_rect, 'type': ptype, 'vel_y': 2})

    def _update_powerups(self):
        for p in self.powerups[:]:
            p['rect'].y += p['vel_y']
            if p['rect'].top > self.HEIGHT:
                self.powerups.remove(p)

    def _handle_powerup_collisions(self):
        reward = 0
        for p in self.powerups[:]:
            if self.paddle.colliderect(p['rect']):
                # sfx: powerup_collect
                self._activate_powerup(p['type'])
                self.powerups.remove(p)
                reward += self.REWARD_POWERUP
        return reward

    def _activate_powerup(self, ptype):
        if ptype == "extended_paddle":
            self.active_effects['extended_paddle'] = 300 # 10 seconds at 30fps
            self.paddle.width = self.PADDLE_WIDTH * 2
            self.paddle.x -= self.PADDLE_WIDTH / 2
        elif ptype == "multi_ball" and self.balls:
            original_ball = self.balls[0]
            for _ in range(2):
                new_vel = [
                    original_ball['vel'][0] * self.np_random.uniform(0.8, 1.2) + self.np_random.uniform(-1, 1),
                    original_ball['vel'][1] * self.np_random.uniform(0.8, 1.2)
                ]
                new_ball = {'pos': list(original_ball['pos']), 'vel': new_vel}
                self.balls.append(new_ball)
                self._normalize_ball_velocity(len(self.balls)-1)


    def _update_effects(self):
        if 'extended_paddle' in self.active_effects:
            self.active_effects['extended_paddle'] -= 1
            if self.active_effects['extended_paddle'] <= 0:
                self.paddle.width = self.PADDLE_WIDTH
                del self.active_effects['extended_paddle']

    def _deactivate_effects(self):
        self.active_effects = {}
        self.paddle.width = self.PADDLE_WIDTH

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifetime': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        return self.lives <= 0 or not self.blocks or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

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
            pygame.draw.rect(self.screen, self.COLOR_BG, block['rect'], 1)

        # Paddle
        paddle_color = self.COLOR_PADDLE
        if 'extended_paddle' in self.active_effects:
            paddle_color = self.POWERUP_COLORS['extended_paddle']
        pygame.draw.rect(self.screen, paddle_color, self.paddle, border_radius=3)
        
        # Balls
        for ball in self.balls:
            pos = (int(ball['pos'][0]), int(ball['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Powerups
        for p in self.powerups:
            color = self.POWERUP_COLORS[p['type']]
            pygame.draw.rect(self.screen, color, p['rect'], border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_BG, p['rect'], 2, border_radius=4)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'][0])-2, int(p['pos'][1])-2))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_icon_surf = pygame.Surface((self.PADDLE_WIDTH / 3, self.PADDLE_HEIGHT / 2), pygame.SRCALPHA)
        pygame.draw.rect(life_icon_surf, self.COLOR_PADDLE, life_icon_surf.get_rect(), border_radius=2)
        for i in range(self.lives):
            self.screen.blit(life_icon_surf, (self.WIDTH - 40 - i * 35, 15))
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER" if self.lives <= 0 else "YOU WIN!"
            msg_text = self.font_main.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, msg_rect.inflate(20, 20))
            self.screen.blit(msg_text, msg_rect)

        # Launch prompt
        if not self.ball_launched and not self.game_over:
            prompt_text = self.font_small.render("PRESS SPACE TO LAUNCH", True, self.COLOR_TEXT)
            prompt_rect = prompt_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 70))
            self.screen.blit(prompt_text, prompt_rect)

    def close(self):
        pygame.quit()

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    import os
    # Set a dummy video driver for headless execution
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # Test reset
    obs, info = env.reset()
    print("Reset successful. Initial state:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Info: {info}")

    # Test a few steps with random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action taken: {action}")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Info: {info}")
        if terminated:
            print("Episode ended.")
            break
            
    env.close()
    print("\nEnvironment closed.")