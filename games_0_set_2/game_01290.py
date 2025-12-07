
# Generated: 2025-08-27T16:39:58.821586
# Source Brief: brief_01290.md
# Brief Index: 1290

        
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
        "Controls: Use Arrow Keys (↑↓←→) to move your paddle. Intercept the ball to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Speed Pong: Intercept the fast-moving ball to score points. Clear 3 stages by scoring 15 points in each before time runs out or you run out of lives. Risky edge-hits grant bonus rewards."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30  # Logic FPS
        self.MAX_STAGES = 3
        self.SCORE_TO_WIN_STAGE = 15
        self.MAX_MISSES = 5
        self.TIME_PER_STAGE_SECONDS = 60

        # Colors
        self.COLOR_BG = (26, 26, 46) # #1A1A2E
        self.COLOR_PADDLE = (0, 255, 255) # Cyan
        self.COLOR_BALL = (255, 255, 0) # Yellow
        self.COLOR_WALL = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_SUCCESS = (0, 255, 127) # SpringGreen
        self.COLOR_FAIL = (255, 69, 0) # OrangeRed

        # Entity properties
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 100, 15
        self.PADDLE_SPEED = 12
        self.BALL_RADIUS = 8
        self.BASE_BALL_SPEED = 7.0
        self.BALL_SPEED_INCREMENT = 1.0 # Per stage

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables (initialized in reset)
        self.stage = 0
        self.score = 0
        self.misses = 0
        self.time_remaining = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = 0
        self.particles = []
        self.last_hit_feedback = 0
        self.last_miss_feedback = 0
        self.stage_message_timer = 0
        self.stage_message_text = ""
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def _start_stage(self, stage_num):
        """Initializes state for a new stage."""
        self.stage = stage_num
        self.score = 0
        self.misses = 0
        self.time_remaining = self.TIME_PER_STAGE_SECONDS * self.FPS
        
        self.paddle = pygame.Rect(
            (self.WIDTH - self.PADDLE_WIDTH) / 2,
            self.HEIGHT - self.PADDLE_HEIGHT * 3,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.ball_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        
        # Set ball speed based on stage
        self.ball_speed = self.BASE_BALL_SPEED + (self.stage - 1) * self.BALL_SPEED_INCREMENT

        # Random initial velocity
        angle = self.np_random.uniform(math.pi * 1.25, math.pi * 1.75)
        self.ball_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.ball_speed
        
        self.particles.clear()
        
        # Display stage message
        self.stage_message_text = f"STAGE {self.stage}"
        self.stage_message_timer = 2 * self.FPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.game_over = False
        self.win = False
        
        self._start_stage(1)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        reward = 0
        terminated = False
        
        # -- UPDATE GAME LOGIC --
        
        # 1. Handle Input & Move Paddle
        if self.stage_message_timer <= 0:
            moved = False
            if movement == 1: # Up
                self.paddle.y -= self.PADDLE_SPEED
                moved = True
            elif movement == 2: # Down
                self.paddle.y += self.PADDLE_SPEED
                moved = True
            elif movement == 3: # Left
                self.paddle.x -= self.PADDLE_SPEED
                moved = True
            elif movement == 4: # Right
                self.paddle.x += self.PADDLE_SPEED
                moved = True
            
            if moved:
                reward -= 0.01 # Small penalty for movement to encourage efficiency

        # Clamp paddle to screen
        self.paddle.left = max(0, self.paddle.left)
        self.paddle.right = min(self.WIDTH, self.paddle.right)
        self.paddle.top = max(self.HEIGHT / 2, self.paddle.top) # Can't go past halfway
        self.paddle.bottom = min(self.HEIGHT, self.paddle.bottom)

        # 2. Update Ball
        if self.stage_message_timer <= 0:
            self.ball_pos += self.ball_vel

        # 3. Handle Collisions
        ball_rect = pygame.Rect(self.ball_pos.x - self.BALL_RADIUS, self.ball_pos.y - self.BALL_RADIUS, self.BALL_RADIUS*2, self.BALL_RADIUS*2)

        # Wall collisions
        if ball_rect.left <= 0 or ball_rect.right >= self.WIDTH:
            self.ball_vel.x *= -1
            ball_rect.left = max(0, ball_rect.left)
            ball_rect.right = min(self.WIDTH, ball_rect.right)
            self.ball_pos.x = ball_rect.centerx
            self._create_particles(pygame.Vector2(ball_rect.centerx, self.ball_pos.y), 5)
            # sfx: wall_bounce.wav
        if ball_rect.top <= 0:
            self.ball_vel.y *= -1
            ball_rect.top = max(0, ball_rect.top)
            self.ball_pos.y = ball_rect.centery
            self._create_particles(pygame.Vector2(self.ball_pos.x, ball_rect.centery), 5)
            # sfx: wall_bounce.wav

        # Paddle collision
        if self.ball_vel.y > 0 and self.paddle.colliderect(ball_rect):
            # sfx: paddle_hit.wav
            self.score += 1
            self.last_hit_feedback = 15 # frames
            reward += 10.1 # +10 for point, +0.1 for hit
            
            # Reverse y velocity
            self.ball_vel.y *= -1
            
            # Influence x velocity based on hit location
            hit_offset = (self.ball_pos.x - self.paddle.centerx) / (self.PADDLE_WIDTH / 2)
            self.ball_vel.x += hit_offset * 3.0
            
            # Normalize to maintain constant speed
            self.ball_vel = self.ball_vel.normalize() * self.ball_speed
            
            # Risky play reward
            if abs(hit_offset) > 0.6: # Hit near the edge
                reward += 1.0
            else: # Hit near the center
                reward -= 0.2
                
            self._create_particles(self.ball_pos, 20)
            
            # Stage Clear Check
            if self.score >= self.SCORE_TO_WIN_STAGE:
                if self.stage >= self.MAX_STAGES:
                    # Game Win
                    self.win = True
                    self.game_over = True
                    terminated = True
                    reward += 100
                    self.stage_message_text = "YOU WIN!"
                    self.stage_message_timer = 3 * self.FPS
                else:
                    # Next Stage
                    self._start_stage(self.stage + 1)
        
        # Miss
        elif ball_rect.top > self.HEIGHT:
            # sfx: miss.wav
            self.misses += 1
            self.last_miss_feedback = 15 # frames
            reward -= 10
            self._start_stage(self.stage) # Reset ball and paddle for the current stage
            
            if self.misses >= self.MAX_MISSES:
                self.game_over = True
                terminated = True
                reward -= 100
                self.stage_message_text = "GAME OVER"
                self.stage_message_timer = 3 * self.FPS

        # 4. Update Timers & Particles
        if self.stage_message_timer > 0:
            self.stage_message_timer -= 1
        else:
            self.time_remaining -= 1

        if self.last_hit_feedback > 0: self.last_hit_feedback -= 1
        if self.last_miss_feedback > 0: self.last_miss_feedback -= 1

        self._update_particles()
        
        # 5. Check Termination Conditions
        if not terminated and self.time_remaining <= 0 and self.stage_message_timer <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
            self.stage_message_text = "TIME'S UP!"
            self.stage_message_timer = 3 * self.FPS

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, position, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': lifetime, 'max_life': lifetime})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1

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
        # Draw playfield boundaries
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)
        
        # Draw particles
        for p in self.particles:
            alpha = 255 * (p['life'] / p['max_life'])
            size = 1 + int(2 * (p['life'] / p['max_life']))
            color = (*self.COLOR_PARTICLE, alpha)
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'].x - size), int(p['pos'].y - size)))

        # Draw ball with glow
        self._draw_glow_circle(self.screen, self.COLOR_BALL, self.ball_pos, self.BALL_RADIUS, 10)
        
        # Draw paddle
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle, border_radius=3)
        # Add a subtle glow to the paddle
        glow_paddle = self.paddle.inflate(6, 6)
        glow_surf = pygame.Surface(glow_paddle.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PADDLE, 50), (0, 0, *glow_paddle.size), border_radius=5)
        self.screen.blit(glow_surf, glow_paddle.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        if self.last_hit_feedback > 0:
            scale = 1 + 0.2 * math.sin(self.last_hit_feedback * math.pi / 15)
            scaled_text = pygame.transform.smoothscale_by(score_text, scale)
            score_rect = scaled_text.get_rect(topright=(self.WIDTH - 10, 10))
            self.screen.blit(scaled_text, score_rect)
        else:
            self.screen.blit(score_text, score_rect)

        # Misses
        misses_color = self.COLOR_FAIL if self.last_miss_feedback > 0 else self.COLOR_TEXT
        misses_text = self.font_medium.render(f"MISSES: {self.misses}/{self.MAX_MISSES}", True, misses_color)
        self.screen.blit(misses_text, (score_rect.left, score_rect.bottom + 5))

        # Time
        time_str = f"{int(self.time_remaining / self.FPS):02d}"
        time_color = self.COLOR_FAIL if self.time_remaining / self.FPS < 10 else self.COLOR_TEXT
        time_text = self.font_large.render(time_str, True, time_color)
        time_rect = time_text.get_rect(midtop=(self.WIDTH / 2, 5))
        self.screen.blit(time_text, time_rect)
        
        # Stage
        stage_text = self.font_medium.render(f"STAGE: {self.stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))
        
        # Stage/Game Over Message
        if self.stage_message_timer > 0:
            msg_color = self.COLOR_SUCCESS if self.win else (self.COLOR_FAIL if self.game_over else self.COLOR_TEXT)
            msg_text = self.font_large.render(self.stage_message_text, True, msg_color)
            
            alpha = 255
            if self.stage_message_timer < self.FPS / 2:
                alpha = int(255 * (self.stage_message_timer / (self.FPS / 2)))
            
            msg_text.set_alpha(alpha)
            msg_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 50))
            self.screen.blit(msg_text, msg_rect)

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength):
        center_x, center_y = int(center.x), int(center.y)
        for i in range(glow_strength, 0, -1):
            alpha = int(80 * (1 - (i / glow_strength))**2)
            pygame.gfxdraw.filled_circle(
                surface, center_x, center_y, radius + i, (*color, alpha)
            )
        pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)
        pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, color)

    def _get_info(self):
        # The brief is strict about the return keys.
        return {
            "score": self.score,
            "steps": self.steps,
            # "misses": self.misses,
            # "stage": self.stage,
            # "time_remaining": int(self.time_remaining / self.FPS)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert "score" in info and "steps" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "score" in info and "steps" in info
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # To map keyboard keys to actions:
    # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
    # actions[1]: Space button (0=released, 1=held)
    # actions[2]: Shift button (0=released, 1=held)
    action = np.array([0, 0, 0])
    
    # Use a separate pygame screen for human rendering
    pygame.display.set_caption("Speed Pong")
    human_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action[0] = 0 # No movement
        
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # The brief doesn't use space/shift, but we still need to handle them
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-facing screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()