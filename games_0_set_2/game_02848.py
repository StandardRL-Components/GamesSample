
# Generated: 2025-08-27T21:37:02.524648
# Source Brief: brief_02848.md
# Brief Index: 2848

        
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
        "Controls: Use ← and → to move the paddle. Break all the blocks to win!"
    )

    game_description = (
        "A fast-paced, top-down block breaker where risk-taking is rewarded. "
        "Chain hits for bonus points and watch the ball speed up as you clear the screen."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True, though clock isn't used in step
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PADDLE = (255, 255, 255)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (255, 255, 0, 50)
        self.BLOCK_COLORS = [
            (255, 0, 128), (0, 255, 255), (0, 255, 0),
            (255, 128, 0), (128, 0, 255)
        ]
        self.COLOR_TEXT = (220, 220, 220)

        # Paddle settings
        self.PADDLE_WIDTH = 100
        self.PADDLE_HEIGHT = 15
        self.PADDLE_SPEED = 12
        self.PADDLE_SPIN_FACTOR = 1.5

        # Ball settings
        self.BALL_RADIUS = 8
        self.INITIAL_BALL_SPEED = 4.0
        self.BALL_SPEED_INCREMENT = 0.5 # Per 10 blocks

        # Block settings
        self.NUM_BLOCKS_X = 10
        self.NUM_BLOCKS_Y = 5
        self.BLOCK_WIDTH = 58
        self.BLOCK_HEIGHT = 20
        self.BLOCK_SPACING = 6
        self.BLOCK_AREA_TOP = 50

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.blocks = []
        self.particles = []
        self.blocks_destroyed_count = 0
        self.consecutive_hits_in_volley = 0
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.blocks_destroyed_count = 0
        self.consecutive_hits_in_volley = 0
        
        # Paddle
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT * 2
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            paddle_y,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        
        # Ball
        self.ball_speed = self.INITIAL_BALL_SPEED
        self._reset_ball()

        # Blocks
        self.blocks = []
        total_block_width = self.NUM_BLOCKS_X * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - total_block_width) / 2
        for i in range(self.NUM_BLOCKS_Y):
            for j in range(self.NUM_BLOCKS_X):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                block_rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                self.blocks.append({"rect": block_rect, "color": color})

        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_pos = pygame.math.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
        self.ball_vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * self.ball_speed
        self.consecutive_hits_in_volley = 0

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = -0.01 # Small penalty to encourage speed

        # 1. Update paddle position
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        # 2. Update ball position
        self.ball_pos += self.ball_vel

        # 3. Handle collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Walls
        if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
            self.ball_vel.x *= -1
            self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # sfx: wall_bounce.wav
        if self.ball_pos.y - self.BALL_RADIUS <= 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.BALL_RADIUS
            # sfx: wall_bounce.wav

        # Paddle
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            self.ball_vel.y *= -1
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS

            # Add "spin" based on hit location
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += offset * self.PADDLE_SPIN_FACTOR
            
            # Anti-softlock: prevent purely vertical bounces
            if abs(self.ball_vel.x) < 0.1 * self.ball_speed:
                self.ball_vel.x += self.np_random.uniform(-0.2, 0.2) * self.ball_speed

            # Normalize to maintain constant speed
            self.ball_vel = self.ball_vel.normalize() * self.ball_speed
            self.consecutive_hits_in_volley = 0
            # sfx: paddle_hit.wav

        # Blocks
        hit_block_idx = ball_rect.collidelist([b['rect'] for b in self.blocks])
        if hit_block_idx != -1:
            hit_block = self.blocks[hit_block_idx]
            self._spawn_particles(hit_block['rect'].center, hit_block['color'])
            
            # Determine bounce direction
            # A simple but effective method: check which side is penetrated the most
            dx = self.ball_pos.x - hit_block['rect'].centerx
            dy = self.ball_pos.y - hit_block['rect'].centery
            w, h = hit_block['rect'].width / 2, hit_block['rect'].height / 2
            
            if abs(dx / w) > abs(dy / h): # Horizontal collision
                self.ball_vel.x *= -1
            else: # Vertical collision
                self.ball_vel.y *= -1

            del self.blocks[hit_block_idx]
            
            # Update score and rewards
            self.score += 1 + self.consecutive_hits_in_volley * 2
            reward += 1.0 + self.consecutive_hits_in_volley * 2.0
            self.consecutive_hits_in_volley += 1
            self.blocks_destroyed_count += 1
            # sfx: block_break.wav

            # Increase ball speed every 10 blocks
            if self.blocks_destroyed_count > 0 and self.blocks_destroyed_count % 10 == 0:
                self.ball_speed += self.BALL_SPEED_INCREMENT
                self.ball_vel = self.ball_vel.normalize() * self.ball_speed
                # sfx: speed_up.wav

        # Miss / Life lost
        terminated = False
        if self.ball_pos.y - self.BALL_RADIUS > self.HEIGHT:
            self.lives -= 1
            # sfx: life_lost.wav
            if self.lives <= 0:
                self.game_over = True
                terminated = True
                reward = -100.0
            else:
                self._reset_ball()

        # 4. Check for termination
        if not self.blocks:
            self.game_over = True
            terminated = True
            reward = 100.0
            # sfx: win_game.wav
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True

        # 5. Update particles
        self._update_particles()
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_particles(self, pos, color):
        for _ in range(15):
            vel = pygame.math.Vector2(
                self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)
            )
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': self.np_random.integers(10, 20),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.9 # friction
            p['life'] -= 1
            p['radius'] -= 0.2
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'], block['rect'], border_radius=3)
        
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['radius']), int(p['pos'].y - p['radius'])))

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        
        # Ball
        ball_x, ball_y = int(self.ball_pos.x), int(self.ball_pos.y)
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_BALL_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (ball_x - glow_radius, ball_y - glow_radius))
        # Solid ball
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            if not self.blocks: # Win
                msg = "YOU WIN!"
            else: # Lose
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_BALL)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks)
        }
        
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time
    
    # Set this to 'human' to see the game being played.
    # Note: Gymnasium's render function is not used here for simplicity.
    # We are directly blitting to a display screen.
    render_mode = "human" # or "rgb_array"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = None
    if render_mode == "human":
        pygame.display.set_caption(env.game_description)
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    # This loop simulates a human player or a simple agent.
    # For human play, we map keyboard keys to actions.
    # For an agent, you would replace this with `agent.predict(obs)`.
    
    running = True
    while running:
        # For human play
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # --- Action Selection ---
        # Simple agent: always try to follow the ball
        action = np.array([0, 0, 0]) # Default: no-op
        if env.ball_pos.x < env.paddle.centerx - env.PADDLE_WIDTH / 4:
            action[0] = 3 # Move left
        elif env.ball_pos.x > env.paddle.centerx + env.PADDLE_WIDTH / 4:
            action[0] = 4 # Move right

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            # In human mode, pause for a moment before restarting
            if render_mode == "human":
                time.sleep(2)
        
        # --- Rendering (for human mode) ---
        if render_mode == "human":
            # Convert the observation (which is a numpy array) back to a Pygame surface
            # The observation is (H, W, C), but pygame needs (W, H) surface
            # and surfarray.make_surface expects (W, H, C)
            render_obs = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(render_obs)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Control the frame rate
            env.clock.tick(env.FPS)

    env.close()