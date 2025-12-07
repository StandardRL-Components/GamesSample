
# Generated: 2025-08-28T03:02:11.211256
# Source Brief: brief_04648.md
# Brief Index: 4648

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class Particle:
    """A simple class for a single particle in an effect."""
    def __init__(self, x, y, color, min_vel=-1.5, max_vel=1.5, lifespan=25):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(min_vel, max_vel), random.uniform(min_vel, max_vel))
        self.lifespan = random.randint(lifespan - 10, lifespan + 10)
        self.max_lifespan = self.lifespan
        self.color = color
        self.size = random.randint(2, 4)

    def update(self):
        """Update particle position and lifespan."""
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, surface):
        """Draw the particle with a fade-out effect."""
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            alpha = max(0, min(255, alpha))
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, (*self.color, alpha), temp_surf.get_rect())
            surface.blit(temp_surf, (int(self.pos.x), int(self.pos.y)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move the paddle. Break all the blocks to win."
    )

    game_description = (
        "A minimalist block breaker. Control the paddle to bounce the ball, "
        "destroying the block grid. Don't let the ball fall!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_PADDLE = (230, 230, 230)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 150, 0)
        self.COLOR_BLOCK_1HP = (100, 150, 255)
        self.COLOR_BLOCK_2HP = (60, 90, 200)
        self.COLOR_UI = (200, 200, 200)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.blocks = []
        self.particles = []
        self.blocks_destroyed_count = 0
        self.initial_ball_speed = 0
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            # Note: np.random is not used here, but this is where you'd seed it
            # self.np_random = np.random.default_rng(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = 3
        
        # Paddle
        paddle_width, paddle_height = 100, 15
        self.paddle = pygame.Rect(
            (self.WIDTH - paddle_width) / 2,
            self.HEIGHT - paddle_height - 10,
            paddle_width,
            paddle_height
        )
        
        # Ball
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - 15)
        self.initial_ball_speed = 7.0
        angle = random.uniform(math.pi * 1.25, math.pi * 1.75) # Upwards angle
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.initial_ball_speed
        
        # Blocks
        self.blocks = self._create_blocks()
        self.blocks_destroyed_count = 0
        
        # Effects
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        blocks = []
        block_width, block_height = 60, 20
        gap = 4
        rows, cols = 5, 10
        total_block_width = cols * (block_width + gap) - gap
        start_x = (self.WIDTH - total_block_width) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                hp = 2 if r < 2 and random.random() < 0.4 else 1 # Top rows can be tougher
                color = self.COLOR_BLOCK_2HP if hp == 2 else self.COLOR_BLOCK_1HP
                rect = pygame.Rect(
                    start_x + c * (block_width + gap),
                    start_y + r * (block_height + gap),
                    block_width,
                    block_height
                )
                blocks.append({'rect': rect, 'hp': hp, 'color': color, 'base_color': color})
        # Ensure exactly 50 blocks
        return blocks[:50]

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        paddle_speed = 10
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4:  # Right
            self.paddle.x += paddle_speed
        
        # Clamp paddle to screen
        self.paddle.x = max(0, min(self.WIDTH - self.paddle.width, self.paddle.x))

        # --- Game Logic Update ---
        step_reward = -0.02  # Small penalty for each step
        
        # Update ball position
        self.ball_pos += self.ball_vel

        # --- Collisions ---
        ball_radius = 8
        ball_rect = pygame.Rect(self.ball_pos.x - ball_radius, self.ball_pos.y - ball_radius, ball_radius * 2, ball_radius * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel.x *= -1
            ball_rect.left = max(1, ball_rect.left)
            ball_rect.right = min(self.WIDTH - 1, ball_rect.right)
            self.ball_pos.x = ball_rect.centerx
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            ball_rect.top = 1
            self.ball_pos.y = ball_rect.centery

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            # Influence ball's horizontal velocity based on impact point
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.paddle.width / 2)
            max_influence = 0.6
            self.ball_vel.x += offset * self.initial_ball_speed * max_influence
            # Prevent ball from getting stuck in paddle
            ball_rect.bottom = self.paddle.top - 1
            self.ball_pos.y = ball_rect.centery
            # // Sound effect: paddle_hit

        # Block collisions
        hit_block = None
        for block in self.blocks:
            if ball_rect.colliderect(block['rect']):
                hit_block = block
                break
        
        if hit_block:
            step_reward += 0.1 # Reward for any hit
            # // Sound effect: block_hit
            
            # Collision resolution
            prev_ball_pos = self.ball_pos - self.ball_vel
            if (prev_ball_pos.y - ball_radius > hit_block['rect'].bottom or
                prev_ball_pos.y + ball_radius < hit_block['rect'].top):
                self.ball_vel.y *= -1
            else:
                self.ball_vel.x *= -1

            hit_block['hp'] -= 1
            if hit_block['hp'] <= 0:
                step_reward += 2.0 if hit_block['base_color'] == self.COLOR_BLOCK_2HP else 1.0
                self.score += 20 if hit_block['base_color'] == self.COLOR_BLOCK_2HP else 10
                self.blocks.remove(hit_block)
                self.blocks_destroyed_count += 1
                # // Sound effect: block_destroy
                # Particle effect
                for _ in range(20):
                    self.particles.append(Particle(hit_block['rect'].centerx, hit_block['rect'].centery, hit_block['base_color']))
                
                # Difficulty scaling
                if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                    current_speed = self.ball_vel.length()
                    new_speed = current_speed + 0.5 # Increase speed slightly
                    self.ball_vel.scale_to_length(new_speed)
            else:
                # Flash effect for multi-hit block
                hit_block['color'] = self.COLOR_PADDLE

        # Reset block colors after flash
        for block in self.blocks:
            if block['hp'] == 1 and block['color'] != block['base_color']:
                 block['color'] = block['base_color']

        # Floor collision (lose life)
        if ball_rect.top >= self.HEIGHT:
            self.lives -= 1
            # // Sound effect: lose_life
            if self.lives <= 0:
                self.game_over = True
                step_reward -= 100 # Large penalty for losing
            else:
                # Reset ball position
                self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - 15)
                angle = random.uniform(math.pi * 1.25, math.pi * 1.75)
                self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.initial_ball_speed

        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
        # --- Termination ---
        self.steps += 1
        terminated = self.game_over
        
        if not self.blocks: # Win condition
            terminated = True
            self.game_over = True
            step_reward += 100 # Large reward for winning
            self.score += 1000 # Bonus score for winning
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.score += step_reward
        
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
            
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        # Ball Glow
        glow_radius = 12
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*self.COLOR_BALL_GLOW, 80))
        self.screen.blit(glow_surf, (int(self.ball_pos.x - glow_radius), int(self.ball_pos.y - glow_radius)))

        # Ball
        ball_radius = 8
        pygame.gfxdraw.aacircle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), ball_radius, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball_pos.x), int(self.ball_pos.y), ball_radius, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        life_size = 12
        life_gap = 5
        for i in range(self.lives):
            life_rect = pygame.Rect(self.WIDTH - (i + 1) * (life_size + life_gap), 15, life_size, life_size)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, life_rect, border_radius=2)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
        pygame.font.quit()
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
    # This block allows you to play the game manually for testing
    import os
    # Set a dummy video driver to run headless if not rendering to screen
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for manual play ---
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    total_reward = 0

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()