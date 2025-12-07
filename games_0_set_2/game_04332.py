
# Generated: 2025-08-28T02:04:44.887983
# Source Brief: brief_04332.md
# Brief Index: 4332

        
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
        "Controls: ←→ to move the paddle. Press space to launch the ball."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "A fast-paced, top-down block breaker. Clear all blocks to advance through stages, but lose all your balls and it's game over."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps
    MAX_STAGES = 3
    INITIAL_BALLS = 3

    # Colors
    COLOR_BG = (15, 15, 40)
    COLOR_GRID = (25, 25, 60)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_PADDLE_GLOW = (200, 200, 255)
    COLOR_BALL = (255, 255, 0)
    COLOR_BALL_GLOW = (255, 255, 150)
    COLOR_TEXT = (220, 220, 255)
    BLOCK_COLORS = {
        'G': ((0, 255, 100), (100, 255, 150), 10, 1.0),   # Green
        'B': ((0, 150, 255), (100, 200, 255), 25, 2.5),   # Blue
        'R': ((255, 50, 50), (255, 120, 120), 50, 5.0),    # Red
    }

    # Physics & Gameplay
    PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
    PADDLE_SPEED = 12
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 5.0
    BALL_SPEED_INCREMENT = 0.5
    MAX_BOUNCE_ANGLE = math.pi / 3 # 60 degrees

    # Stage Layouts
    STAGE_LAYOUTS = [
        [
            "        ",
            " RRRRRR ",
            " RRRRRR ",
            " BBBBBB ",
            " BBBBBB ",
            " GGGGGG ",
            " GGGGGG ",
            "        ",
        ],
        [
            "R B G  ",
            " B G R ",
            "G R B  ",
            " R B G ",
            "  G R B",
            " R B G ",
            "B G R  ",
            " G R B ",
        ],
        [
            "R      R",
            " B    B ",
            "  G  G  ",
            "   RR   ",
            "   GG   ",
            "  B  B  ",
            " R    R ",
            "B      B",
        ],
    ]


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = 0
        self.stage = 0
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.ball_stuck = True
        self.blocks = []
        self.particles = []
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_left = self.INITIAL_BALLS
        self.stage = 1
        self.paddle = pygame.Rect(
            self.WIDTH // 2 - self.PADDLE_WIDTH // 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self._load_stage(self.stage)
        self._reset_ball()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Remap movement actions: 1=left, 2=right
        if movement == 1: # Mapped from 'up'
            self.paddle.x -= self.PADDLE_SPEED
        elif movement == 2: # Mapped from 'down'
            self.paddle.x += self.PADDLE_SPEED
        
        self.paddle.x = np.clip(self.paddle.x, 0, self.WIDTH - self.PADDLE_WIDTH)

        if space_held and self.ball_stuck:
            self.ball_stuck = False
            # Sound: Ball launch
            self.ball_vel = pygame.Vector2(self.rng.choice([-1, 1]), -1).normalize() * self.ball_speed

        # --- Game Logic ---
        self._update_ball()
        reward += self._handle_collisions()
        self._update_particles()
        
        if not self.ball_stuck:
            reward -= 0.02 # Time penalty for efficiency

        # --- Stage & Termination Check ---
        if not self.blocks:
            reward += 50 # Stage clear bonus
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                terminated = True
                reward += 100 # Game win bonus
            else:
                # Sound: Stage clear
                self._load_stage(self.stage)
                self._reset_ball()
                self.ball_speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * self.BALL_SPEED_INCREMENT

        if self.balls_left <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Game over penalty

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_ball(self):
        if self.ball_stuck:
            self.ball_pos.x = self.paddle.centerx
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS
        else:
            self.ball_pos += self.ball_vel
            # Wall collisions
            if self.ball_pos.x - self.BALL_RADIUS <= 0 or self.ball_pos.x + self.BALL_RADIUS >= self.WIDTH:
                self.ball_vel.x *= -1
                self.ball_pos.x = np.clip(self.ball_pos.x, self.BALL_RADIUS, self.WIDTH - self.BALL_RADIUS)
                # Sound: Wall bounce
            if self.ball_pos.y - self.BALL_RADIUS <= 0:
                self.ball_vel.y *= -1
                self.ball_pos.y = self.BALL_RADIUS
                # Sound: Wall bounce
            # Bottom wall (lose ball)
            if self.ball_pos.y + self.BALL_RADIUS >= self.HEIGHT:
                self.balls_left -= 1
                # Sound: Lose ball
                if self.balls_left > 0:
                    self._reset_ball()

    def _handle_collisions(self):
        reward = 0
        if self.ball_stuck:
            return reward

        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Paddle collision
        if ball_rect.colliderect(self.paddle) and self.ball_vel.y > 0:
            # Sound: Paddle hit
            offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            bounce_angle = offset * self.MAX_BOUNCE_ANGLE
            
            new_vel = pygame.Vector2(math.sin(bounce_angle), -math.cos(bounce_angle))
            self.ball_vel = new_vel.normalize() * self.ball_speed
            self.ball_pos.y = self.paddle.top - self.BALL_RADIUS # Prevent sticking

        # Block collisions
        hit_block_idx = ball_rect.collidelist([b[0] for b in self.blocks])
        if hit_block_idx != -1:
            # Sound: Block break
            block_rect, color_key = self.blocks[hit_block_idx]
            
            # Determine bounce direction
            # A simple but effective method: check overlap
            prev_ball_pos = self.ball_pos - self.ball_vel
            prev_ball_rect = pygame.Rect(prev_ball_pos.x - self.BALL_RADIUS, prev_ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

            if prev_ball_rect.bottom <= block_rect.top or prev_ball_rect.top >= block_rect.bottom:
                self.ball_vel.y *= -1
            else:
                self.ball_vel.x *= -1

            # Reward and effects
            block_info = self.BLOCK_COLORS[color_key]
            self.score += block_info[2]
            reward += 0.1 + block_info[3]
            self._create_particles(block_rect.center, block_info[0])
            
            del self.blocks[hit_block_idx]

        return reward

    def _reset_ball(self):
        self.ball_stuck = True
        self.ball_pos = pygame.Vector2(self.paddle.centerx, self.paddle.top - self.BALL_RADIUS)
        self.ball_vel = pygame.Vector2(0, 0)
        self.ball_speed = self.INITIAL_BALL_SPEED + (self.stage - 1) * self.BALL_SPEED_INCREMENT

    def _load_stage(self, stage_num):
        self.blocks.clear()
        layout = self.STAGE_LAYOUTS[stage_num - 1]
        num_rows = len(layout)
        num_cols = len(layout[0])
        
        block_width = self.WIDTH // num_cols
        block_height = 20
        top_offset = 60

        for r, row_str in enumerate(layout):
            for c, block_char in enumerate(row_str):
                if block_char in self.BLOCK_COLORS:
                    rect = pygame.Rect(c * block_width, top_offset + r * block_height, block_width, block_height)
                    # Add small gap between blocks
                    rect.inflate_ip(-4, -4)
                    self.blocks.append((rect, block_char))

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            particle = {
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'radius': self.rng.uniform(2, 5),
                'lifespan': self.rng.integers(15, 30),
                'color': color
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)

    def _render_game(self):
        # Render blocks
        for rect, color_key in self.blocks:
            main_color, glow_color = self.BLOCK_COLORS[color_key][0], self.BLOCK_COLORS[color_key][1]
            glow_rect = rect.inflate(4, 4)
            pygame.draw.rect(self.screen, glow_color, glow_rect, border_radius=4)
            pygame.draw.rect(self.screen, main_color, rect, border_radius=3)

        # Render paddle
        glow_paddle = self.paddle.inflate(6, 6)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_GLOW, glow_paddle, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=6)

        # Render ball
        pos = (int(self.ball_pos.x), int(self.ball_pos.y))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 2, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS + 2, self.COLOR_BALL_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BALL_RADIUS, self.COLOR_BALL)

        # Render particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _render_ui(self):
        # Score
        score_surf = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Stage
        stage_surf = self.font_small.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_surf, (15, 45))

        # Balls left
        for i in range(self.balls_left):
            pos = (self.WIDTH - 25 - i * 25, 25)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_PADDLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_PADDLE)
            
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            msg = "GAME OVER" if self.balls_left <= 0 else "YOU WIN!"
            end_surf = self.font_main.render(msg, True, self.COLOR_TEXT)
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(end_surf, end_rect)

            final_score_surf = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20))
            self.screen.blit(final_score_surf, final_score_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "balls_left": self.balls_left
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


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set render_mode to "human" to see the pygame window
    # Note: The provided class is designed for headless "rgb_array" rendering.
    # To run in "human" mode, you'd need to modify the rendering logic slightly.
    # The following code demonstrates the headless "rgb_array" mode.
    
    import cv2 # pip install opencv-python

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Key mapping for human play ---
    # We'll use a dictionary to simulate the MultiDiscrete action space
    # actions[0]: Movement (0=none, 1=up(left), 2=down(right), 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    
    key_map = {
        pygame.K_LEFT: 1,
        pygame.K_RIGHT: 2
    }
    
    terminated = False
    total_reward = 0
    
    print("Starting game. Controls: Left/Right arrows, Space to launch.")
    print("Close the 'Game Preview' window to exit.")

    while not terminated:
        # Create a default action (no-op)
        action = [0, 0, 0]
        
        # Poll for pygame events to get keyboard input and window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Get pressed keys
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT]:
            action[0] = 1 # Remapped to paddle left
        elif keys[pygame.K_RIGHT]:
            action[0] = 2 # Remapped to paddle right

        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space held
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the RGB array observation using OpenCV
        # Convert RGB (from pygame) to BGR (for OpenCV)
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("Game Preview", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Control the frame rate to be playable for humans
        env.clock.tick(30)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()
    cv2.destroyAllWindows()