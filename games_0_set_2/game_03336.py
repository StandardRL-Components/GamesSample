
# Generated: 2025-08-27T23:04:06.437058
# Source Brief: brief_03336.md
# Brief Index: 3336

        
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
    A fast-paced, retro-neon block breaker Gymnasium environment. The player
    controls a paddle to bounce a ball and destroy a grid of blocks. The game
    rewards breaking blocks and penalizes losing lives. Visuals are prioritized,
    with neon colors, particle effects, and smooth motion.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-neon block breaker. Clear all the blocks to win, but don't lose the ball!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000 * 5 # Allow for longer games
    LIVES = 3

    # Colors (Neon/Retro Theme)
    COLOR_BG = (10, 5, 30)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255, 50)
    COLOR_BALL = (255, 0, 128)
    COLOR_BALL_GLOW = (255, 100, 200, 100)
    COLOR_TEXT = (220, 220, 220)
    BLOCK_COLORS = {
        1: (0, 255, 0),    # Green: 1 point
        2: (0, 128, 255),  # Blue: 2 points
        3: (255, 255, 0),  # Yellow: 3 points
        5: (255, 0, 0),    # Red: 5 points
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)

        # Game state variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.paddle = None
        self.ball = None
        self.ball_vel = [0, 0]
        self.ball_launched = False
        self.blocks = []
        self.particles = []
        self.ball_trail = []
        self.last_block_hit_step = 0
        self.rng = None

        # Initialize state
        self.reset()

        # Self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.lives = self.LIVES
        self.game_over = False
        self.last_block_hit_step = 0
        self.particles = []
        self.ball_trail = []

        # Paddle setup
        paddle_width, paddle_height = 100, 15
        self.paddle = pygame.Rect(
            (self.SCREEN_WIDTH - paddle_width) // 2,
            self.SCREEN_HEIGHT - paddle_height - 10,
            paddle_width,
            paddle_height
        )

        # Ball setup
        self._reset_ball()

        # Block setup
        self.blocks = []
        block_width, block_height = 58, 20
        rows, cols = 5, 10
        for r in range(rows):
            for c in range(cols):
                block_val = [1, 1, 2, 3, 5][r]
                block_rect = pygame.Rect(
                    c * (block_width + 6) + 10,
                    r * (block_height + 6) + 40,
                    block_width,
                    block_height
                )
                self.blocks.append({"rect": block_rect, "value": block_val})

        return self._get_observation(), self._get_info()

    def _reset_ball(self):
        self.ball_launched = False
        ball_size = 12
        self.ball = pygame.Rect(
            self.paddle.centerx - ball_size // 2,
            self.paddle.top - ball_size,
            ball_size,
            ball_size
        )
        self.ball_vel = [0, 0]
        self.ball_trail = []

    def _launch_ball(self):
        if not self.ball_launched:
            self.ball_launched = True
            # Sound: Ball launch
            angle = self.rng.uniform(-math.pi / 4, math.pi / 4)
            speed = 6
            self.ball_vel = [speed * math.sin(angle), -speed * math.cos(angle)]
            self.last_block_hit_step = self.steps

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        paddle_speed = 8
        if movement == 3:  # Left
            self.paddle.x -= paddle_speed
        elif movement == 4: # Right
            self.paddle.x += paddle_speed
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.SCREEN_WIDTH - self.paddle.width)

        if space_held:
            self._launch_ball()

        # --- Update Game Logic ---
        self.steps += 1
        
        if self.ball_launched:
            self._update_ball()
            reward -= 0.02 # Small penalty for time passing
        else:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top

        self._update_particles()
        
        # --- Check for Rewards & Termination ---
        ball_reward, life_lost = self._handle_collisions()
        reward += ball_reward

        if life_lost:
            self.lives -= 1
            # Sound: Life lost
            if self.lives > 0:
                self._reset_ball()
            else:
                self.game_over = True
                reward -= 100 # Terminal penalty for losing all lives
        
        if not self.blocks:
            self.game_over = True
            reward += 100 # Terminal reward for winning

        # Anti-softlock mechanism
        if self.ball_launched and (self.steps - self.last_block_hit_step > 300):
             # Sound: Ball reset sfx
             self._reset_ball()
             self.last_block_hit_step = self.steps

        terminated = self.game_over or self.steps >= self.MAX_STEPS
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_trail.append(self.ball.copy())
        if len(self.ball_trail) > 5:
            self.ball_trail.pop(0)
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]

    def _handle_collisions(self):
        reward = 0
        life_lost = False

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(0, self.ball.left)
            self.ball.right = min(self.SCREEN_WIDTH, self.ball.right)
            # Sound: Wall bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(0, self.ball.top)
            # Sound: Wall bounce
        if self.ball.top >= self.SCREEN_HEIGHT:
            life_lost = True

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            # Sound: Paddle bounce
            offset = (self.ball.centerx - self.paddle.centerx) / (self.paddle.width / 2)
            max_vel_x = 7
            self.ball_vel[0] = max_vel_x * offset
            self.ball_vel[1] *= -1
            
            speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            min_speed_y = 4
            if abs(self.ball_vel[1]) < min_speed_y:
                 self.ball_vel[1] = -min_speed_y

        # Block collisions
        hit_index = self.ball.collidelist([b["rect"] for b in self.blocks])
        if hit_index != -1:
            block_hit = self.blocks.pop(hit_index)
            # Sound: Block break
            reward += block_hit["value"] + 0.1
            self.score += block_hit["value"]
            self.last_block_hit_step = self.steps
            self._create_particles(block_hit["rect"].center, block_hit["value"])
            self.ball_vel[1] *= -1
            
        return reward, life_lost
        
    def _create_particles(self, pos, block_value):
        color = self.BLOCK_COLORS[block_value]
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel,
                "lifetime": self.rng.integers(15, 30),
                "color": color, "size": self.rng.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for block_data in self.blocks:
            r = block_data["rect"]
            c = self.BLOCK_COLORS[block_data["value"]]
            pygame.draw.rect(self.screen, tuple(x * 0.6 for x in c), r)
            pygame.draw.rect(self.screen, c, r.inflate(-4, -4))

        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

        glow_rect = self.paddle.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PADDLE_GLOW, (0,0, *glow_rect.size), border_radius=8)
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        
        for i, trail_ball in enumerate(self.ball_trail):
            alpha = int(100 * (i / len(self.ball_trail)))
            pygame.gfxdraw.filled_circle(self.screen, int(trail_ball.centerx), int(trail_ball.centery), int(self.ball.width/2), (*self.COLOR_BALL, alpha))

        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), int(self.ball.width/2 + 3), self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), int(self.ball.width/2), self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), int(self.ball.width/2), self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        life_icon_width, life_icon_height = 25, 5
        for i in range(self.lives):
            icon_rect = pygame.Rect(self.SCREEN_WIDTH - (i + 1) * (life_icon_width + 5) - 5, 15, life_icon_width, life_icon_height)
            pygame.draw.rect(self.screen, self.COLOR_PADDLE, icon_rect, border_radius=2)

        if self.game_over:
            result_text = "YOU WIN!" if not self.blocks else "GAME OVER"
            color = (0, 255, 0) if not self.blocks else (255, 0, 0)
            text_surf = self.font_main.render(result_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_left": len(self.blocks),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(60)

    env.close()