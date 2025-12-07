
# Generated: 2025-08-27T20:08:35.194635
# Source Brief: brief_02362.md
# Brief Index: 2362

        
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
    """
    A fast-paced, top-down block breaker where strategic paddle positioning and
    risk-taking are key to maximizing your score. Destroy all blocks to win,
    but lose all your balls and the game is over.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Try to destroy all the blocks with the ball."
    )

    game_description = (
        "A fast-paced arcade block breaker. Destroy all blocks to win. "
        "Some blocks release power-ups when destroyed."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        self.width, self.height = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG = (20, 20, 40)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BOUNDARY = (80, 80, 90)
        self.COLOR_TEXT = (220, 220, 220)
        self.BLOCK_COLORS = [
            (255, 71, 87), (255, 165, 2), (46, 213, 115),
            (30, 144, 255), (125, 95, 255)
        ]
        self.POWERUP_COLORS = [
            (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
            (0, 0, 255), (75, 0, 130), (148, 0, 211)
        ]

        # Game constants
        self.PADDLE_SPEED = 8.0
        self.PADDLE_WIDTH_NORMAL = 100
        self.PADDLE_HEIGHT = 15
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 4.0
        self.MAX_BALL_SPEED = 8.0
        self.POWERUP_SPEED = 2.0
        self.POWERUP_DURATION = 600  # 20 seconds at 30fps
        self.MAX_STEPS = 10000
        self.NUM_BLOCKS = 50

        # Game state variables are initialized in reset()
        self.paddle = None
        self.balls = []
        self.blocks = []
        self.particles = []
        self.powerups_on_screen = []
        self.active_powerups = {}
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_total = 0
        self.blocks_destroyed_since_speedup = 0
        self.current_ball_speed = 0.0
        self.np_random = None
        
        self.reset()
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback to a default or unseeded generator if seed is None
            self.np_random = np.random.default_rng()


        # Initialize game state
        self.paddle = pygame.Rect(
            self.width // 2 - self.PADDLE_WIDTH_NORMAL // 2,
            self.height - 40,
            self.PADDLE_WIDTH_NORMAL,
            self.PADDLE_HEIGHT
        )
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self._spawn_ball()
        
        self._create_blocks()

        self.particles = []
        self.powerups_on_screen = []
        self.active_powerups = {}

        self.lives = 3
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.blocks_destroyed_total = 0
        self.blocks_destroyed_since_speedup = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Time penalty to encourage speed
        self.steps += 1

        # 1. Handle player input
        self._move_paddle(action[0])

        # 2. Update game elements
        self._apply_powerup_effects()
        
        ball_reward = self._update_balls()
        reward += ball_reward
        
        powerup_reward = self._update_powerups_on_screen()
        reward += powerup_reward
        
        self._update_particles()
        
        # 3. Check for termination conditions
        terminated = False
        if not self.blocks: # Win condition
            reward += 100
            terminated = True
            self.game_over = True
        elif self.lives <= 0: # Lose condition
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS: # Max steps reached
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            np.clip(reward, -100, 100),
            terminated,
            False,
            self._get_info()
        )

    def _spawn_ball(self, pos=None, vel=None):
        if pos is None:
            pos = [self.paddle.centerx, self.paddle.top - self.BALL_RADIUS - 1]
        if vel is None:
            angle = self.np_random.uniform(-math.pi * 3/4, -math.pi * 1/4)
            vel = [self.current_ball_speed * math.cos(angle), self.current_ball_speed * math.sin(angle)]

        self.balls.append({
            'pos': pos,
            'vel': vel,
            'rect': pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2),
            'stuck_timer': 0
        })

    def _create_blocks(self):
        self.blocks = []
        block_width = 58
        block_height = 20
        gap = 6
        rows = 5
        cols = 10
        
        # Power-up assignment
        num_powerups = 5
        powerup_indices = self.np_random.choice(self.NUM_BLOCKS, num_powerups, replace=False)
        powerup_types = ['large_paddle', 'multi_ball']
        
        block_idx = 0
        for r in range(rows):
            for c in range(cols):
                powerup_type = None
                if block_idx in powerup_indices:
                    powerup_type = self.np_random.choice(powerup_types)

                self.blocks.append({
                    'rect': pygame.Rect(
                        c * (block_width + gap) + gap,
                        r * (block_height + gap) + 50,
                        block_width,
                        block_height
                    ),
                    'color': self.BLOCK_COLORS[r % len(self.BLOCK_COLORS)],
                    'powerup': powerup_type
                })
                block_idx += 1

    def _move_paddle(self, movement_action):
        if movement_action == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement_action == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen bounds
        self.paddle.x = max(0, min(self.width - self.paddle.width, self.paddle.x))

    def _update_balls(self):
        reward = 0
        
        for ball in self.balls[:]:
            # Anti-softlock mechanism
            if abs(ball['vel'][1]) < 0.2:
                ball['stuck_timer'] += 1
                if ball['stuck_timer'] > 100:
                    ball['vel'][1] += self.np_random.choice([-0.5, 0.5])
                    ball['stuck_timer'] = 0
            else:
                ball['stuck_timer'] = 0

            ball['pos'][0] += ball['vel'][0]
            ball['pos'][1] += ball['vel'][1]
            ball['rect'].center = ball['pos']

            # Wall collisions
            if ball['rect'].left <= 0 or ball['rect'].right >= self.width:
                ball['vel'][0] *= -1
                ball['rect'].left = max(0, ball['rect'].left)
                ball['rect'].right = min(self.width, ball['rect'].right)
                # sfx: wall_bounce
            if ball['rect'].top <= 0:
                ball['vel'][1] *= -1
                ball['rect'].top = max(0, ball['rect'].top)
                # sfx: wall_bounce

            # Paddle collision
            if ball['rect'].colliderect(self.paddle) and ball['vel'][1] > 0:
                # sfx: paddle_hit
                offset = (ball['rect'].centerx - self.paddle.centerx) / (self.paddle.width / 2)
                
                # Reward for paddle hits
                if abs(offset) < 0.1: # Center 20%
                    reward += 0.1
                elif abs(offset) > 0.6: # Edge 40%
                    reward += 0.5

                # Recalculate velocity
                new_angle = -math.pi/2 + offset * (math.pi / 2.5) # Map offset to angle
                ball['vel'][0] = self.current_ball_speed * math.cos(new_angle)
                ball['vel'][1] = self.current_ball_speed * math.sin(new_angle)
                ball['rect'].bottom = self.paddle.top - 1

            # Block collisions
            collided_block_idx = ball['rect'].collidelist([b['rect'] for b in self.blocks])
            if collided_block_idx != -1:
                # sfx: block_destroy
                reward += 1
                collided_block = self.blocks.pop(collided_block_idx)
                
                self._spawn_particles(collided_block['rect'].center, collided_block['color'])
                
                # Handle power-ups
                if collided_block['powerup']:
                    self.powerups_on_screen.append({
                        'rect': pygame.Rect(collided_block['rect'].centerx - 10, collided_block['rect'].centery, 20, 20),
                        'type': collided_block['powerup']
                    })

                # Bounce logic
                ball['vel'][1] *= -1
                
                # Update difficulty
                self.blocks_destroyed_total += 1
                self.blocks_destroyed_since_speedup += 1
                if self.blocks_destroyed_since_speedup >= 10:
                    self.current_ball_speed = min(self.MAX_BALL_SPEED, self.current_ball_speed + 0.4)
                    self.blocks_destroyed_since_speedup = 0
                    # Update existing balls speed
                    for b in self.balls:
                        speed = math.sqrt(b['vel'][0]**2 + b['vel'][1]**2)
                        if speed > 0:
                            b['vel'][0] = (b['vel'][0] / speed) * self.current_ball_speed
                            b['vel'][1] = (b['vel'][1] / speed) * self.current_ball_speed


            # Ball loss
            if ball['rect'].top >= self.height:
                self.balls.remove(ball)
                # sfx: lose_ball
                if not self.balls: # Last ball lost
                    self.lives -= 1
                    reward -= 10
                    if self.lives > 0:
                        self._spawn_ball()

        return reward

    def _update_powerups_on_screen(self):
        reward = 0
        for pu in self.powerups_on_screen[:]:
            pu['rect'].y += self.POWERUP_SPEED
            if pu['rect'].top > self.height:
                self.powerups_on_screen.remove(pu)
            elif pu['rect'].colliderect(self.paddle):
                # sfx: powerup_get
                reward += 5
                self.active_powerups[pu['type']] = self.POWERUP_DURATION
                self.powerups_on_screen.remove(pu)
        return reward

    def _apply_powerup_effects(self):
        # Handle 'large_paddle'
        if 'large_paddle' in self.active_powerups:
            self.paddle.width = self.PADDLE_WIDTH_NORMAL * 1.5
            self.active_powerups['large_paddle'] -= 1
            if self.active_powerups['large_paddle'] <= 0:
                del self.active_powerups['large_paddle']
                self.paddle.width = self.PADDLE_WIDTH_NORMAL
        else:
            self.paddle.width = self.PADDLE_WIDTH_NORMAL
        
        # Handle 'multi_ball'
        if 'multi_ball' in self.active_powerups:
            if self.active_powerups['multi_ball'] == self.POWERUP_DURATION: # Activate only once
                if len(self.balls) > 0:
                    original_ball = self.balls[0]
                    self._spawn_ball(pos=original_ball['pos'][:], vel=[-original_ball['vel'][0], original_ball['vel'][1]])
                    self._spawn_ball(pos=original_ball['pos'][:], vel=[original_ball['vel'][0] * 0.8, original_ball['vel'][1] * 1.2])
            
            self.active_powerups['multi_ball'] -= 1
            if self.active_powerups['multi_ball'] <= 0:
                del self.active_powerups['multi_ball']

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Drag
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw boundaries
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.width, self.height), 2)
        
        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'])
            if block['powerup']: # Add a visual cue for power-up blocks
                pygame.draw.circle(self.screen, (255, 255, 255), block['rect'].center, 5)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (1, 1), 1)
            self.screen.blit(temp_surf, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw power-ups
        for i, pu in enumerate(self.powerups_on_screen):
            color = self.POWERUP_COLORS[(self.steps + i*5) % len(self.POWERUP_COLORS)]
            pygame.draw.rect(self.screen, color, pu['rect'])
            pygame.draw.rect(self.screen, self.COLOR_TEXT, pu['rect'], 1)

        # Draw paddle with rounded corners
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Draw balls with a glow effect
        for ball in self.balls:
            center = (int(ball['pos'][0]), int(ball['pos'][1]))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS, (*self.COLOR_BALL, 100))
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.BALL_RADIUS-2, self.COLOR_BALL)


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"BALLS: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.width - lives_text.get_width() - 10, 10))
        
        # Game Over / Win Text
        if self.game_over:
            if not self.blocks:
                msg = "YOU WIN!"
                color = (0, 255, 128)
            else:
                msg = "GAME OVER"
                color = (255, 60, 60)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
            "active_powerups": list(self.active_powerups.keys())
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Map keys to MultiDiscrete action
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()