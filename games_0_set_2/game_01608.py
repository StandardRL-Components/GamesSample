
# Generated: 2025-08-27T17:40:39.269800
# Source Brief: brief_01608.md
# Brief Index: 1608

        
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
        "Controls: ↑/↓ to aim, SPACE to fire the ball. Clear all the blocks to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist block-breaker puzzle. Aim your shots carefully to clear the board with a limited number of balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.INITIAL_SHOTS = 10
        self.CANNON_POS = (self.WIDTH // 2, self.HEIGHT - 20)
        self.BALL_RADIUS = 8
        self.BALL_SPEED = 7
        self.FRICTION = 0.998
        self.MIN_VELOCITY_SQ = 0.05 ** 2

        # --- Colors ---
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (40, 44, 52)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_CANNON = (150, 155, 165)
        self.COLOR_AIM_LINE = (90, 95, 105)
        self.COLOR_BALL = (255, 255, 255)
        self.BLOCK_COLORS = [
            (97, 218, 251),  # Health 1: Bright Blue
            (224, 108, 117), # Health 2: Bright Red
            (233, 196, 107)  # Health 3: Bright Yellow
        ]
        self.BLOCK_BORDER_COLOR = self.COLOR_BG

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.avg_block_health = 1.0
        self.consecutive_failures = 0
        self.last_win = False

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        # Handle difficulty progression
        if self.last_win:
            self.avg_block_health = min(3.0, self.avg_block_health + 0.2)
            self.consecutive_failures = 0
        else: # On loss
            self.consecutive_failures += 1
        
        if self.consecutive_failures >= 3:
            self.avg_block_health = 1.0
            self.consecutive_failures = 0

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shots_remaining = self.INITIAL_SHOTS
        self.aim_angle = -math.pi / 2 # Straight up
        self.game_phase = 'AIMING' # 'AIMING' or 'PROJECTILE_MOVING'
        self.ball = None
        self.particles = []
        self._generate_blocks()
        
        self.last_win = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0
        terminated = False

        if self.game_phase == 'AIMING':
            # --- Handle Aiming ---
            if movement == 1: # Up in brief, interpreted as left rotation
                self.aim_angle -= 0.05
            elif movement == 2: # Down in brief, interpreted as right rotation
                self.aim_angle += 0.05
            self.aim_angle = np.clip(self.aim_angle, -math.pi + 0.1, -0.1)

            # --- Handle Firing ---
            if space_held and self.shots_remaining > 0:
                self.shots_remaining -= 1
                self.ball = {
                    'pos': list(self.CANNON_POS),
                    'vel': [self.BALL_SPEED * math.cos(self.aim_angle), self.BALL_SPEED * math.sin(self.aim_angle)]
                }
                self.game_phase = 'PROJECTILE_MOVING'
                # Sound: pew.wav
            else:
                # Small penalty for inaction to encourage firing
                reward -= 0.1

        # --- Simulate Projectile Motion ---
        if self.game_phase == 'PROJECTILE_MOVING':
            shot_reward, combo = self._simulate_ball_path()
            reward += shot_reward
            self.game_phase = 'AIMING'
        
        # --- Check Termination Conditions ---
        if not self.blocks: # Win condition
            terminated = True
            reward += 100
            self.score += 1000 # Win bonus
            self.last_win = True
            self.game_over = True
        elif self.shots_remaining <= 0 and self.game_phase == 'AIMING': # Loss condition
            terminated = True
            reward -= 100
            self.last_win = False
            self.game_over = True

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _simulate_ball_path(self):
        # This internal loop simulates the entire path of the ball until it stops.
        total_reward = 0
        combo_count = 0
        
        # Max sub-steps to prevent infinite loops
        for _ in range(2000):
            if not self.ball: break
            
            # Update position
            self.ball['pos'][0] += self.ball['vel'][0]
            self.ball['pos'][1] += self.ball['vel'][1]

            # Apply friction
            self.ball['vel'][0] *= self.FRICTION
            self.ball['vel'][1] *= self.FRICTION

            # Check for stopping
            if self.ball['vel'][0]**2 + self.ball['vel'][1]**2 < self.MIN_VELOCITY_SQ:
                self.ball = None
                break

            # Wall collisions
            if self.ball['pos'][0] <= self.BALL_RADIUS:
                self.ball['pos'][0] = self.BALL_RADIUS
                self.ball['vel'][0] *= -1
            elif self.ball['pos'][0] >= self.WIDTH - self.BALL_RADIUS:
                self.ball['pos'][0] = self.WIDTH - self.BALL_RADIUS
                self.ball['vel'][0] *= -1
            if self.ball['pos'][1] <= self.BALL_RADIUS:
                self.ball['pos'][1] = self.BALL_RADIUS
                self.ball['vel'][1] *= -1
            if self.ball['pos'][1] >= self.HEIGHT - self.BALL_RADIUS:
                # Let ball go off bottom of screen
                self.ball = None
                break
            
            # Block collisions
            hit_block = False
            for block in reversed(self.blocks):
                if self._check_ball_block_collision(block):
                    hit_block = True
                    # Sound: hit.wav
                    total_reward += 1 # Reward for any hit
                    
                    if combo_count == 0: total_reward += 10 # Combo start reward
                    else: total_reward += 5 # Combo continue reward
                    combo_count += 1
                    
                    block['health'] -= 1
                    if block['health'] <= 0:
                        total_reward += 5 # Reward for destruction
                        self.score += 10 + 5 * (block['initial_health'] - 1)
                        self._create_particles(block['rect'].center, self.BLOCK_COLORS[block['initial_health'] - 1])
                        self.blocks.remove(block)
                        # Sound: destroy.wav
                    break # Handle one collision per frame
            
        return total_reward, combo_count

    def _check_ball_block_collision(self, block):
        if not self.ball: return False
        
        rect = block['rect']
        ball_pos = self.ball['pos']
        
        closest_x = np.clip(ball_pos[0], rect.left, rect.right)
        closest_y = np.clip(ball_pos[1], rect.top, rect.bottom)
        
        dist_x = ball_pos[0] - closest_x
        dist_y = ball_pos[1] - closest_y
        
        if (dist_x**2 + dist_y**2) < self.BALL_RADIUS**2:
            # Collision occurred, calculate bounce
            if abs(dist_x) > abs(dist_y):
                self.ball['vel'][0] *= -1
                self.ball['pos'][0] += np.sign(dist_x) * 2
            else:
                self.ball['vel'][1] *= -1
                self.ball['pos'][1] += np.sign(dist_y) * 2
            return True
        return False

    def _generate_blocks(self):
        self.blocks = []
        block_width, block_height = 50, 20
        gap = 5
        rows = 6
        cols = 10
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) // 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                if self.np_random.random() > 0.3: # 70% chance of a block spawning
                    x = start_x + c * (block_width + gap)
                    y = start_y + r * (block_height + gap)
                    
                    # Determine health based on average difficulty
                    health_roll = self.np_random.random()
                    if health_roll < (self.avg_block_health - 2.0): health = 3
                    elif health_roll < (self.avg_block_health - 1.0): health = 2
                    else: health = 1
                    health = int(np.clip(health, 1, 3))
                    
                    self.blocks.append({
                        'rect': pygame.Rect(x, y, block_width, block_height),
                        'health': health,
                        'initial_health': health
                    })

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "shots_remaining": self.shots_remaining,
            "avg_block_health": self.avg_block_health
        }

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_blocks()
        self._update_and_render_particles()
        self._render_cannon_and_aim()
        if self.ball:
            pygame.gfxdraw.aacircle(self.screen, int(self.ball['pos'][0]), int(self.ball['pos'][1]), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(self.ball['pos'][0]), int(self.ball['pos'][1]), self.BALL_RADIUS, self.COLOR_BALL)
        self._render_ui()

    def _render_grid(self):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

    def _render_blocks(self):
        for block in self.blocks:
            color = self.BLOCK_COLORS[block['health'] - 1]
            pygame.draw.rect(self.screen, color, block['rect'], border_radius=3)
            pygame.draw.rect(self.screen, self.BLOCK_BORDER_COLOR, block['rect'], width=2, border_radius=3)

    def _render_cannon_and_aim(self):
        # Cannon
        pygame.draw.circle(self.screen, self.COLOR_CANNON, self.CANNON_POS, 15)
        pygame.draw.circle(self.screen, self.COLOR_BG, self.CANNON_POS, 10)
        
        # Aiming line
        if self.game_phase == 'AIMING':
            end_x = self.CANNON_POS[0] + 80 * math.cos(self.aim_angle)
            end_y = self.CANNON_POS[1] + 80 * math.sin(self.aim_angle)
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, self.CANNON_POS, (end_x, end_y), 2)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [speed * math.cos(angle), speed * math.sin(angle)],
                'life': self.np_random.integers(15, 30),
                'color': color,
            })

    def _update_and_render_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Particle friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                size = max(0, int(p['life'] / 6))
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Shots remaining
        shots_text = self.font_large.render(f"{self.shots_remaining}", True, self.COLOR_UI_TEXT)
        shots_label = self.font_small.render("BALLS", True, self.COLOR_UI_TEXT)
        self.screen.blit(shots_text, (20, 10))
        self.screen.blit(shots_label, (shots_text.get_width() + 25, 22))

        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(score_text, score_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''Call this at the end of __init__ to verify implementation.'''
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
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # Unused in this env

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        # Space press needs to be event-based for turn-based games
        if env.game_phase == 'AIMING':
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_SPACE:
                    space_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(60) # Limit frame rate for human play

    env.close()