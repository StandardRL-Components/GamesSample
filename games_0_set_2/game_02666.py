
# Generated: 2025-08-27T21:04:32.400180
# Source Brief: brief_02666.md
# Brief Index: 2666

        
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

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, neon-drenched block-breaking game. Destroy all blocks before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (30, 20, 60)
    COLOR_PADDLE = (0, 150, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (150, 200, 255)
    NEON_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Lime
    ]
    COLOR_UI_TEXT = (220, 220, 220)

    # Paddle
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8

    # Ball
    BALL_RADIUS = 8
    BALL_SPEED = 6
    MAX_BOUNCE_ANGLE = math.radians(75)

    # Blocks
    BLOCK_ROWS = 5
    BLOCK_COLS = 10
    BLOCK_WIDTH = 60
    BLOCK_HEIGHT = 20
    BLOCK_SPACING = 4
    BLOCK_AREA_TOP_MARGIN = 50

    # Game State
    INITIAL_BALLS = 3

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        self.paddle_rect = None
        self.ball = {}
        self.blocks = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.balls_left = 0
        self.game_over = False
        self.win_status = ""

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = ""
        self.time_left = self.TIME_LIMIT_SECONDS * self.FPS
        self.balls_left = self.INITIAL_BALLS

        self._init_paddle()
        self._init_ball()
        self._init_blocks()
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_pressed = action[1] == 1

        reward = -0.01  # Time penalty to encourage speed
        self.time_left -= 1
        self.steps += 1

        if not self.game_over:
            self._update_paddle(movement)
            self._handle_launch(space_pressed)
            ball_lost, blocks_broken = self._update_ball()

            reward += blocks_broken * 1.0

            if ball_lost:
                self.balls_left -= 1
                if self.balls_left > 0:
                    self._init_ball()  # Respawn ball
                else:
                    self.game_over = True
                    self.win_status = "YOU LOSE"
                    reward -= 50 # Big penalty for losing all balls
            
            self._update_particles()

        # Check termination conditions
        terminated = self._check_termination()
        if terminated and not self.win_status: # Game ended by time or win
             if self.time_left <= 0:
                 self.win_status = "TIME UP"
                 reward -= 10
             else: # All blocks destroyed
                 self.win_status = "YOU WIN!"
                 reward += 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _init_paddle(self):
        self.paddle_rect = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

    def _init_ball(self):
        self.ball = {
            "pos": np.array([self.paddle_rect.centerx, self.paddle_rect.top - self.BALL_RADIUS], dtype=float),
            "vel": np.array([0.0, 0.0], dtype=float),
            "launched": False
        }
    
    def _init_blocks(self):
        self.blocks = []
        grid_width = self.BLOCK_COLS * (self.BLOCK_WIDTH + self.BLOCK_SPACING) - self.BLOCK_SPACING
        start_x = (self.WIDTH - grid_width) / 2
        
        for i in range(self.BLOCK_ROWS):
            for j in range(self.BLOCK_COLS):
                x = start_x + j * (self.BLOCK_WIDTH + self.BLOCK_SPACING)
                y = self.BLOCK_AREA_TOP_MARGIN + i * (self.BLOCK_HEIGHT + self.BLOCK_SPACING)
                rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                color = self.NEON_COLORS[self.np_random.integers(0, len(self.NEON_COLORS))]
                self.blocks.append({"rect": rect, "color": color, "alive": True})

    def _update_paddle(self, movement):
        # 3=left, 4=right
        if movement == 3:
            self.paddle_rect.x -= self.PADDLE_SPEED
        elif movement == 4:
            self.paddle_rect.x += self.PADDLE_SPEED
        
        self.paddle_rect.left = max(0, self.paddle_rect.left)
        self.paddle_rect.right = min(self.WIDTH, self.paddle_rect.right)

    def _handle_launch(self, space_pressed):
        if space_pressed and not self.ball["launched"]:
            self.ball["launched"] = True
            angle = (self.np_random.random() - 0.5) * math.pi / 4 # -22.5 to +22.5 deg
            self.ball["vel"] = np.array([
                self.BALL_SPEED * math.sin(angle),
                -self.BALL_SPEED * math.cos(angle)
            ])
            # sfx: launch_sound

    def _update_ball(self):
        if not self.ball["launched"]:
            self.ball["pos"][0] = self.paddle_rect.centerx
            return False, 0

        self.ball["pos"] += self.ball["vel"]
        ball_rect = pygame.Rect(0, 0, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
        ball_rect.center = tuple(self.ball["pos"])

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball["vel"][0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball["vel"][1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce
        if ball_rect.top >= self.HEIGHT:
            return True, 0 # Ball lost

        # Paddle collision
        if ball_rect.colliderect(self.paddle_rect) and self.ball["vel"][1] > 0:
            offset = ball_rect.centerx - self.paddle_rect.centerx
            normalized_offset = offset / (self.PADDLE_WIDTH / 2)
            bounce_angle = normalized_offset * self.MAX_BOUNCE_ANGLE

            self.ball["vel"][0] = self.BALL_SPEED * math.sin(bounce_angle)
            self.ball["vel"][1] = -self.BALL_SPEED * math.cos(bounce_angle)
            
            # Ensure ball is pushed out of paddle to prevent sticking
            ball_rect.bottom = self.paddle_rect.top
            self.ball["pos"] = np.array(ball_rect.center, dtype=float)
            # sfx: paddle_hit

        # Block collisions
        blocks_broken = 0
        for block in self.blocks:
            if block["alive"] and ball_rect.colliderect(block["rect"]):
                block["alive"] = False
                self.score += 1
                blocks_broken += 1
                self._spawn_particles(ball_rect.center, block["color"])
                # sfx: block_break
                
                # Collision response
                prev_ball_rect = pygame.Rect(0,0,ball_rect.width, ball_rect.height)
                prev_ball_rect.center = self.ball["pos"] - self.ball["vel"]

                # Horizontal collision
                if prev_ball_rect.right <= block["rect"].left or prev_ball_rect.left >= block["rect"].right:
                    self.ball["vel"][0] *= -1
                # Vertical collision
                else:
                    self.ball["vel"][1] *= -1
                break # Only one block per frame
        
        return False, blocks_broken

    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "lifespan": lifespan, "color": color})

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        all_blocks_destroyed = all(not b["alive"] for b in self.blocks)
        if self.balls_left <= 0 or self.time_left <= 0 or all_blocks_destroyed:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "balls_left": self.balls_left,
            "time_left": self.time_left // self.FPS
        }

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)

        # Background grid
        for x in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Blocks
        for block in self.blocks:
            if block["alive"]:
                pygame.draw.rect(self.screen, block["color"], block["rect"], border_radius=3)
                # Inner darker rect for 3D effect
                inner_rect = block["rect"].inflate(-6, -6)
                darker_color = tuple(c * 0.6 for c in block["color"])
                pygame.draw.rect(self.screen, darker_color, inner_rect, border_radius=3)

        # Paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=5)
        
        # Ball
        ball_pos_int = (int(self.ball["pos"][0]), int(self.ball["pos"][1]))
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*self.COLOR_BALL_GLOW, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (ball_pos_int[0] - glow_radius, ball_pos_int[1] - glow_radius))
        # Ball itself
        pygame.gfxdraw.aacircle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, ball_pos_int[0], ball_pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
        # Particles
        for p in self.particles:
            alpha = max(0, int(255 * (p["lifespan"] / 30)))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0,0,4,4))
            self.screen.blit(temp_surf, (int(p["pos"][0]-2), int(p["pos"][1]-2)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_str = f"TIME: {self.time_left // self.FPS}"
        time_text = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Ball count
        ball_display_y = self.HEIGHT - 25
        for i in range(self.balls_left):
            x_pos = self.WIDTH/2 - (self.balls_left-1)*15/2 + i*25 - self.BALL_RADIUS
            pygame.gfxdraw.aacircle(self.screen, int(x_pos), int(ball_display_y), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(x_pos), int(ball_display_y), self.BALL_RADIUS, self.COLOR_BALL)
            
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,180))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_game_over.render(self.win_status, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Reset to initialize state for observation
        self.reset()
        
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different screen for display if running directly
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(GameEnv.user_guide)

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Rendering ---
        # The observation is already a rendered frame. We just need to display it.
        # It's (H, W, C), but pygame blit wants (W, H) surface.
        # So we convert it back to a surface.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()