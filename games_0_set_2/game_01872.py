import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro block breaker. Destroy all blocks to win, but lose all your balls and it's game over. Risky edge-of-paddle hits are rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and rendering setup
        self.WIDTH, self.HEIGHT = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PADDLE = (230, 230, 255)
        self.COLOR_PADDLE_OUTLINE = (180, 180, 220)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.BLOCK_COLORS = [
            (255, 80, 80), (255, 160, 80), (255, 240, 80),
            (160, 255, 80), (80, 255, 160), (80, 160, 255)
        ]

        # Game parameters
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 6
        self.MAX_BALL_SPEED = 12
        self.MAX_STEPS = 30 * 60 # 60 seconds at 30fps
        self.NUM_BALLS = 5
        
        # State variables (initialized in reset)
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_launched = None
        self.blocks = []
        self.particles = []
        self.ball_trail = deque(maxlen=10)
        
        self.score = 0
        self.steps = 0
        self.balls_left = 0
        self.game_over = False
        self.last_space_held = False
        
        # self.reset() is called in the first call to the user
        # self.validate_implementation() is for debugging
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.score = 0
        self.steps = 0
        self.balls_left = self.NUM_BALLS
        self.game_over = False
        self.last_space_held = False
        self.particles.clear()
        self.ball_trail.clear()
        
        self._create_blocks()
        self._reset_player_ball()
        
        return self._get_observation(), self._get_info()

    def _create_blocks(self):
        self.blocks.clear()
        block_width, block_height = 40, 20
        gap = 4
        rows = 6
        cols = 14
        start_x = (self.WIDTH - (cols * (block_width + gap) - gap)) / 2
        start_y = 50
        for r in range(rows):
            for c in range(cols):
                x = start_x + c * (block_width + gap)
                y = start_y + r * (block_height + gap)
                block_rect = pygame.Rect(x, y, block_width, block_height)
                color_idx = r % len(self.BLOCK_COLORS)
                self.blocks.append({'rect': block_rect, 'color_idx': color_idx})

    def _reset_player_ball(self):
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.center = self.paddle.center
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]
        self.ball_launched = False
        self.ball_trail.clear()

    def step(self, action):
        reward = -0.02  # Time penalty
        self.steps += 1
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input & Paddle Movement ---
        if movement == 3:  # Left
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle.x += self.PADDLE_SPEED
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)

        # --- Handle Ball Launch ---
        just_pressed_space = space_held and not self.last_space_held
        if not self.ball_launched and just_pressed_space:
            self.ball_launched = True
            # SFX: Ball launch
            angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
            self.ball_vel = [self.INITIAL_BALL_SPEED * math.sin(angle), -self.INITIAL_BALL_SPEED * math.cos(angle)]
        self.last_space_held = space_held
        
        # --- Update Game State ---
        if self.ball_launched:
            reward += 0.1  # Reward for keeping ball in play
            self._update_ball()
            collision_reward = self._handle_collisions()
            reward += collision_reward
        else:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top

        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if self.game_over:  # Lost all balls
            terminated = True
            reward -= 100
        elif not self.blocks:  # Won
            terminated = True
            reward += 100
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
             terminated = True # Gymnasium standard
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_ball(self):
        self.ball_trail.append(self.ball.center)
        
        self.ball.x += self.ball_vel[0]
        self.ball.y += self.ball_vel[1]
        
        # Anti-softlock: prevent ball from getting stuck horizontally
        if abs(self.ball_vel[1]) < 1.0:
            self.ball_vel[1] = 1.0 * np.sign(self.ball_vel[1] or 1.0)
        
        # Clamp speed
        speed = math.hypot(*self.ball_vel)
        if speed > self.MAX_BALL_SPEED:
            self.ball_vel = [(v / speed) * self.MAX_BALL_SPEED for v in self.ball_vel]

    def _handle_collisions(self):
        reward = 0
        
        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.left = max(1, self.ball.left)
            self.ball.right = min(self.WIDTH - 1, self.ball.right)
            # SFX: Wall bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            self.ball.top = max(1, self.ball.top)
            # SFX: Wall bounce

        # Bottom wall (lose ball)
        if self.ball.top >= self.HEIGHT:
            self.balls_left -= 1
            # SFX: Ball lost
            if self.balls_left > 0:
                self._reset_player_ball()
            else:
                self.game_over = True
            return -10 # Immediate penalty for losing a ball

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            # SFX: Paddle hit
            offset = self.ball.centerx - self.paddle.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            
            self.ball_vel[0] += normalized_offset * 4
            self.ball_vel[1] *= -1
            self.ball.bottom = self.paddle.top - 1

            if abs(normalized_offset) > 0.7:
                reward += 5  # Risky bounce
            else:
                reward -= 1  # Safe bounce

        # Block collisions
        block_rects = [b['rect'] for b in self.blocks]
        hit_index = self.ball.collidelist(block_rects)
        if hit_index != -1:
            # SFX: Block break
            block_data = self.blocks.pop(hit_index)
            block_rect = block_data['rect']
            color_idx = block_data['color_idx']

            reward += 1 + color_idx # Reward based on color/row
            self.score += 10 * (color_idx + 1)
            
            self._create_particles(block_rect.center, self.BLOCK_COLORS[color_idx])

            # Simple but effective bounce logic
            self.ball_vel[1] *= -1
            self.ball_vel[0] += self.np_random.uniform(-0.2, 0.2) # Prevent loops
        
        return reward

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Damping
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
        # Background Grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Blocks
        for block_data in self.blocks:
            rect = block_data['rect']
            color_idx = block_data['color_idx']
            color = self.BLOCK_COLORS[color_idx]
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            # 3D effect
            pygame.draw.line(self.screen, tuple(min(255, c+30) for c in color), rect.topleft, rect.topright, 2)
            pygame.draw.line(self.screen, tuple(max(0, c-30) for c in color), rect.bottomleft, rect.bottomright, 2)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color']
            radius = int(self.BALL_RADIUS / 2 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*color, alpha))

        # Ball Trail
        if len(self.ball_trail) > 1:
            for i, pos in enumerate(self.ball_trail):
                alpha = int(100 * (i / len(self.ball_trail)))
                radius = int(self.BALL_RADIUS * (i / len(self.ball_trail)))
                if radius > 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (*self.COLOR_BALL, alpha))

        # Ball
        pygame.gfxdraw.filled_circle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(self.ball.centerx), int(self.ball.centery), self.BALL_RADIUS, self.COLOR_BALL)
        
        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_OUTLINE, self.paddle, 2, border_radius=5)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Balls left
        ball_icon_radius = 6
        for i in range(self.balls_left):
            x = self.WIDTH - 20 - (i * (ball_icon_radius * 2 + 5))
            y = 22
            pygame.gfxdraw.filled_circle(self.screen, x, y, ball_icon_radius, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, x, y, ball_icon_radius, self.COLOR_BALL)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "blocks_left": len(self.blocks),
        }

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, _ = self.reset()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # In this mode, we need a real display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Block Breaker")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*40)
    print("      BLOCK BREAKER - MANUAL TEST")
    print("="*40)
    print(env.game_description)
    print(env.user_guide)
    print("="*40 + "\n")

    while not terminated:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated:
            terminated = True
        
        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    env.close()