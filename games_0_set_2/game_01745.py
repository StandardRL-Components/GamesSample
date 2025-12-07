import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move the paddle (different arrow keys for different speeds). Press Space to launch the ball."
    )

    game_description = (
        "A fast-paced, procedurally generated block-breaking game where strategic aiming and risk-taking are rewarded. Destroy all blocks to advance to the next level."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (40, 15, 30)
    COLOR_PADDLE = (230, 230, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    BLOCK_COLORS = [
        (255, 80, 80), (255, 160, 80), (255, 255, 80),
        (80, 255, 80), (80, 160, 255), (160, 80, 255)
    ]

    # Screen
    WIDTH, HEIGHT = 640, 400

    # Game parameters
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    BALL_RADIUS = 7
    MAX_BALL_SPEED = 8
    PADDLE_SPEED_SLOW = 5
    PADDLE_SPEED_FAST = 10
    INITIAL_BALLS = 3
    MAX_EPISODE_STEPS = 2000
    STUCK_BALL_THRESHOLD = 600 # steps without hitting a block

    # Helper class for blocks to store color
    class Block(pygame.Rect):
        def __init__(self, left, top, width, height, color):
            super().__init__(left, top, width, height)
            self.color = color

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 20, bold=True)

        # Initialize state variables in reset
        self.paddle = None
        self.ball = None
        self.ball_vel = None
        self.ball_stuck = None
        self.blocks = None
        self.particles = None
        self.ball_trail = None
        
        self.steps = 0
        self.score = 0
        self.balls_left = 0
        self.level = 0
        self.chain_hits = 0
        self.steps_since_last_hit = 0
        self.reward_this_step = 0
        self.game_over = False

        # self.reset() is called here to initialize the state,
        # but the validator in the original code was also calling it.
        # It's fine to call it here to ensure the env is ready.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.balls_left = self.INITIAL_BALLS
        self.level = 1
        self.game_over = False
        
        self.paddle = pygame.Rect(
            self.WIDTH / 2 - self.PADDLE_WIDTH / 2,
            self.HEIGHT - 40,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self._reset_ball()
        self._generate_blocks()
        
        self.particles = []
        self.ball_trail = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        self.reward_this_step = -0.02 # Continuous penalty for time passing
        
        self._handle_input(movement, space_held)
        self._update_game_state()
        
        self.steps += 1
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        
        if not self.blocks and not self.game_over:
            self.reward_this_step += 50
            self._next_level()

        reward = np.clip(self.reward_this_step, -50, 50)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Map MultiDiscrete to paddle movement
        if movement == 1: # fast left
            self.paddle.x -= self.PADDLE_SPEED_FAST
        elif movement == 2: # fast right
            self.paddle.x += self.PADDLE_SPEED_FAST
        elif movement == 3: # slow left
            self.paddle.x -= self.PADDLE_SPEED_SLOW
        elif movement == 4: # slow right
            self.paddle.x += self.PADDLE_SPEED_SLOW
            
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if self.ball_stuck and space_held:
            # Sound: Ball Launch
            self.ball_stuck = False
            self.ball_vel = [self.np_random.uniform(-2, 2), -5]
            self.steps_since_last_hit = 0

    def _update_game_state(self):
        if self.ball_stuck:
            self.ball.centerx = self.paddle.centerx
            self.ball.bottom = self.paddle.top
            return

        self.ball_trail.append(self.ball.copy())
        if len(self.ball_trail) > 10:
            self.ball_trail.pop(0)

        self.ball.move_ip(self.ball_vel)
        self.steps_since_last_hit += 1

        # Anti-softlock mechanism
        if self.steps_since_last_hit > self.STUCK_BALL_THRESHOLD:
            self._reset_ball_velocity()
            # Sound: Warp/Reset

        # Wall collisions
        if self.ball.left <= 0 or self.ball.right >= self.WIDTH:
            self.ball_vel[0] *= -1
            self.ball.x = np.clip(self.ball.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
            # Sound: Wall Bounce
        if self.ball.top <= 0:
            self.ball_vel[1] *= -1
            # Sound: Wall Bounce

        # Ball lost
        if self.ball.top >= self.HEIGHT:
            self.balls_left -= 1
            self.chain_hits = 0
            # Sound: Lose Life
            if self.balls_left <= 0:
                self.game_over = True
                self.reward_this_step -= 50
            else:
                self._reset_ball()
            return

        # Paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_vel[1] > 0:
            self.ball.bottom = self.paddle.top
            self.ball_vel[1] *= -1
            
            # Add spin based on hit location
            offset = (self.ball.centerx - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2
            self.ball_vel[0] = np.clip(self.ball_vel[0], -self.MAX_BALL_SPEED, self.MAX_BALL_SPEED)
            
            self.chain_hits = 0 # Reset chain on paddle hit
            self.steps_since_last_hit = 0
            # Sound: Paddle Hit

        # Block collision
        hit_block_idx = self.ball.collidelist(self.blocks)
        if hit_block_idx != -1:
            self.reward_this_step += 0.1
            self.chain_hits += 1
            
            block = self.blocks.pop(hit_block_idx)
            
            # Create particles
            for _ in range(15):
                self.particles.append(self._create_particle(block.center, block.color))
            
            # Reward
            block_reward = 1 + 0.5 * self.chain_hits
            self.reward_this_step += block_reward
            self.score += int(block_reward * 10)

            # Determine bounce direction
            prev_ball_center = (self.ball.centerx - self.ball_vel[0], self.ball.centery - self.ball_vel[1])
            if prev_ball_center[1] < block.top or prev_ball_center[1] > block.bottom:
                 self.ball_vel[1] *= -1
            else:
                 self.ball_vel[0] *= -1

            self.steps_since_last_hit = 0
            # Sound: Block Break

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _reset_ball(self):
        self.ball_stuck = True
        self.ball = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        self.ball.centerx = self.paddle.centerx
        self.ball.bottom = self.paddle.top
        self.ball_vel = [0, 0]
        self.ball_trail = []

    def _reset_ball_velocity(self):
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        speed = 5
        self.ball_vel = [speed * math.sin(angle), -speed * math.cos(angle)]
        self.steps_since_last_hit = 0

    def _next_level(self):
        self.level += 1
        self.score += self.level * 100
        self._generate_blocks()
        self._reset_ball()
        # Sound: Level Clear

    def _generate_blocks(self):
        self.blocks = []
        block_w = max(self.WIDTH / 10 * (0.95 ** (self.level-1)), 20)
        block_h = max(30 * (0.95 ** (self.level-1)), 20)
        
        rows = int(self.np_random.integers(3, 8))
        cols = int(self.np_random.integers(5, 12))

        for i in range(rows):
            for j in range(cols):
                if self.np_random.random() < 0.7: # Chance to spawn a block
                    x = j * (block_w + 4) + (self.WIDTH - cols * (block_w + 4)) / 2
                    y = i * (block_h + 4) + 50
                    color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
                    block = self.Block(int(x), int(y), int(block_w), int(block_h), color)
                    self.blocks.append(block)

    def _create_particle(self, pos, color):
        return {
            'pos': list(pos),
            'vel': [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, 1)],
            'life': self.np_random.integers(20, 40),
            'color': color,
            'max_life': 40
        }

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block.color, block, border_radius=3)
            # Add a subtle 3D effect
            highlight = tuple(min(255, c + 40) for c in block.color)
            shadow = tuple(max(0, c - 40) for c in block.color)
            pygame.draw.line(self.screen, highlight, block.topleft, block.topright, 2)
            pygame.draw.line(self.screen, highlight, block.topleft, block.bottomleft, 2)
            pygame.draw.line(self.screen, shadow, block.bottomright, block.topright, 2)
            pygame.draw.line(self.screen, shadow, block.bottomright, block.bottomleft, 2)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=5)
        pygame.draw.rect(self.screen, (150, 150, 180), self.paddle.inflate(-6, -6), border_radius=5)

        # Ball Trail
        if self.ball_trail:
            for i, trail_part in enumerate(self.ball_trail):
                alpha = int(255 * (i / len(self.ball_trail)) * 0.5)
                # Using a temporary surface for alpha blending
                temp_surf = pygame.Surface((self.BALL_RADIUS*2, self.BALL_RADIUS*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, self.BALL_RADIUS, self.BALL_RADIUS, self.BALL_RADIUS, (*self.COLOR_BALL, alpha))
                self.screen.blit(temp_surf, (trail_part.x, trail_part.y))
            
        # Ball
        pygame.gfxdraw.filled_circle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, self.ball.centerx, self.ball.centery, self.BALL_RADIUS, self.COLOR_BALL)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            radius = int(5 * (p['life'] / p['max_life']))
            if radius > 0:
                # Using a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*p['color'], alpha))
                self.screen.blit(temp_surf, (int(p['pos'][0] - radius), int(p['pos'][1] - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Balls left
        for i in range(self.balls_left -1): # Don't draw the one in play
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 20 - i * 20, 22, 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, self.WIDTH - 20 - i * 20, 22, 6, self.COLOR_BALL)
        
        # Game Over Message
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_PADDLE)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "level": self.level,
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires a graphical display. If you are running this in a headless environment,
    # comment out this block or run with a virtual display like Xvfb.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "macOS" depending on your OS
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Block Breaker")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    truncated = False
    total_reward = 0
    
    # Mapping from keyboard to MultiDiscrete action
    key_to_action = {
        pygame.K_a: 1,  # fast left
        pygame.K_d: 2,  # fast right
        pygame.K_LEFT: 3,  # slow left
        pygame.K_RIGHT: 4, # slow right
    }

    while not terminated and not truncated:
        movement_action = 0 # no-op
        space_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Check for multiple movement keys, prioritize one
        if keys[pygame.K_a]:
            movement_action = key_to_action[pygame.K_a]
        elif keys[pygame.K_d]:
            movement_action = key_to_action[pygame.K_d]
        elif keys[pygame.K_LEFT]:
            movement_action = key_to_action[pygame.K_LEFT]
        elif keys[pygame.K_RIGHT]:
            movement_action = key_to_action[pygame.K_RIGHT]

        if keys[pygame.K_SPACE]:
            space_action = 1
        
        # The environment uses MultiDiscrete, so we form the action array
        action = [movement_action, space_action, 0] # shift is not used

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(60) # Limit to 60 FPS for playable speed

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")

    env.close()