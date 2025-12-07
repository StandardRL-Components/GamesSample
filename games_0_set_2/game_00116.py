
# Generated: 2025-08-27T12:38:47.261930
# Source Brief: brief_00116.md
# Brief Index: 116

        
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
        "A fast-paced, top-down block breaker. Break all the blocks to win, but don't let the ball fall past your paddle!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000
    INITIAL_LIVES = 3

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (35, 35, 50)
    COLOR_PADDLE = (255, 255, 255)
    COLOR_BALL = (255, 255, 255)
    COLOR_BALL_GLOW = (200, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    BLOCK_COLORS = {
        10: (50, 205, 50),   # Green
        25: (65, 105, 225),  # Blue
        50: (220, 20, 60),   # Red
        100: (255, 215, 0)   # Gold
    }
    BLOCK_BORDER_COLOR = (10, 10, 15)

    # Game Object Properties
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 10.0
    BALL_RADIUS = 7
    INITIAL_BALL_SPEED = 5.0
    BLOCK_WIDTH = 58
    BLOCK_HEIGHT = 20
    BLOCK_GAP = 4

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_main = pygame.font.SysFont("monospace", 50, bold=True)

        # Initialize state variables
        self.paddle_pos = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_on_paddle = True
        self.blocks = []
        self.blocks_rects = []
        self.particles = []
        self.lives = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.blocks_broken_count = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.paddle_pos = pygame.Rect(
            (self.SCREEN_WIDTH - self.PADDLE_WIDTH) / 2,
            self.SCREEN_HEIGHT - self.PADDLE_HEIGHT - 10,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT,
        )
        self.ball_on_paddle = True
        self._reset_ball()

        self._generate_blocks()

        self.particles = []
        self.lives = self.INITIAL_LIVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.blocks_broken_count = 0
        self.current_ball_speed = self.INITIAL_BALL_SPEED

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        self._handle_input(movement, space_held)
        if movement in [3, 4]:
            reward -= 0.02 # Small penalty for movement to encourage efficiency

        if not self.ball_on_paddle:
            reward += 0.01 # Small reward for keeping ball in play
            reward += self._update_ball()
        else:
            self._reset_ball() # Keep ball centered on paddle

        self._update_particles()

        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100.0
            elif self.lives <= 0:
                reward -= 50.0 # Game over penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, movement, space_held):
        # Move paddle
        if movement == 3:  # Left
            self.paddle_pos.x -= self.PADDLE_SPEED
        elif movement == 4:  # Right
            self.paddle_pos.x += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_pos.x = max(0, min(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.paddle_pos.x))

        # Launch ball
        if self.ball_on_paddle and space_held:
            self.ball_on_paddle = False
            # sfx: launch_ball
            launch_angle = self.np_random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            self.ball_vel = [
                math.cos(launch_angle) * self.current_ball_speed,
                math.sin(launch_angle) * self.current_ball_speed,
            ]

    def _update_ball(self):
        reward = 0.0
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.SCREEN_WIDTH:
            self.ball_vel[0] *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.SCREEN_WIDTH, ball_rect.right)
            # sfx: wall_bounce
        if ball_rect.top <= 0:
            self.ball_vel[1] *= -1
            ball_rect.top = max(0, ball_rect.top)
            # sfx: wall_bounce

        # Paddle collision
        if ball_rect.colliderect(self.paddle_pos) and self.ball_vel[1] > 0:
            self.ball_vel[1] *= -1
            offset = (ball_rect.centerx - self.paddle_pos.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel[0] += offset * 2.5
            
            # Normalize velocity to maintain speed
            speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
            if speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / speed) * self.current_ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / speed) * self.current_ball_speed
            
            ball_rect.bottom = self.paddle_pos.top
            # sfx: paddle_hit
        
        # Block collision
        collided_idx = ball_rect.collidelist(self.blocks_rects)
        if collided_idx != -1:
            block_data = self.blocks[collided_idx]
            block_rect = block_data["rect"]
            
            reward += block_data["points"]
            self.score += block_data["points"]
            
            self._create_particles(block_rect.center, block_data["color"])
            
            del self.blocks[collided_idx]
            del self.blocks_rects[collided_idx]
            
            # Simple bounce logic
            self.ball_vel[1] *= -1
            
            self.blocks_broken_count += 1
            if self.blocks_broken_count > 0 and self.blocks_broken_count % 50 == 0:
                self.current_ball_speed += 0.5 # Per brief, increase speed
            
            # sfx: block_break

        # Miss / lose life
        if ball_rect.top >= self.SCREEN_HEIGHT:
            self.lives -= 1
            reward -= 50.0
            self.ball_on_paddle = True
            # sfx: lose_life
            if self.lives <= 0:
                self.game_over = True

        self.ball_pos = [ball_rect.centerx, ball_rect.centery]
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if not self.blocks:
            self.game_over = True
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _reset_ball(self):
        self.ball_pos = [self.paddle_pos.centerx, self.paddle_pos.top - self.BALL_RADIUS]
        self.ball_vel = [0, 0]

    def _generate_blocks(self):
        self.blocks = []
        self.blocks_rects = []
        num_cols = self.SCREEN_WIDTH // (self.BLOCK_WIDTH + self.BLOCK_GAP)
        start_x = (self.SCREEN_WIDTH - num_cols * (self.BLOCK_WIDTH + self.BLOCK_GAP) + self.BLOCK_GAP) / 2
        
        for r in range(5):
            for c in range(num_cols):
                points = self.np_random.choice([10, 10, 10, 25, 25, 50, 100], p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
                color = self.BLOCK_COLORS[points]
                
                x = start_x + c * (self.BLOCK_WIDTH + self.BLOCK_GAP)
                y = 50 + r * (self.BLOCK_HEIGHT + self.BLOCK_GAP)
                
                rect = pygame.Rect(x, y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                block_data = {"rect": rect, "color": color, "points": points}
                self.blocks.append(block_data)
                self.blocks_rects.append(rect)

    def _create_particles(self, pos, color):
        for _ in range(15):
            vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "color": color})
    
    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_particles()
        self._render_blocks()
        self._render_paddle()
        self._render_ball()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_blocks(self):
        for block in self.blocks:
            pygame.draw.rect(self.screen, self.BLOCK_BORDER_COLOR, block["rect"])
            inner_rect = block["rect"].inflate(-2, -2)
            pygame.draw.rect(self.screen, block["color"], inner_rect)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_pos, border_radius=3)

    def _render_ball(self):
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_BALL_GLOW, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        # Main ball
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        
    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20))))
            color = (*p["color"], alpha)
            pos_int = (int(p["pos"][0]), int(p["pos"][1]))
            pygame.draw.circle(self.screen, color, pos_int, int(p["life"] / 5))

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))

        message = "YOU WIN!" if self.win else "GAME OVER"
        text_surf = self.font_main.render(message, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "blocks_remaining": len(self.blocks),
        }

    def close(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Transpose the observation back for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            
        clock.tick(30) # Run at 30 FPS

    env.close()