
# Generated: 2025-08-27T21:59:04.058273
# Source Brief: brief_02970.md
# Brief Index: 2970

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: ↑↓ to move the paddle. Hit the ball into the goal."

    # Must be a short, user-facing description of the game:
    game_description = "Pixel Pong: A retro arcade sports game. Score 7 points to win, but miss 3 balls and you lose. The ball gets faster as you score!"

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PADDLE = (50, 150, 255)
    COLOR_BALL = (255, 80, 80)
    COLOR_GOAL = (80, 200, 120)
    COLOR_WALL = (200, 200, 200)
    COLOR_PARTICLE = (255, 255, 100)
    COLOR_TEXT = (255, 255, 255)

    # Game parameters
    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 80
    PADDLE_SPEED = 8
    PADDLE_X_POS = 40
    
    BALL_RADIUS = 8
    INITIAL_BALL_SPEED = 5.0
    BALL_SPEED_INCREMENT = 0.5

    MAX_SCORE = 7
    MAX_MISSES = 3
    MAX_STEPS = 2000

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 24)

        # Initialize state variables
        self.player_paddle = None
        self.ball_pos = None
        self.ball_vel = None
        self.ball_speed = None
        self.score = 0
        self.misses = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.rng = None
        
        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.ball_speed = self.INITIAL_BALL_SPEED
        
        self.player_paddle = pygame.Rect(
            self.PADDLE_X_POS,
            self.SCREEN_HEIGHT // 2 - self.PADDLE_HEIGHT // 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        
        self.particles = []
        self._spawn_ball()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Small penalty for each frame to encourage action
        
        # --- Handle Input ---
        self._handle_input(action)
        
        # --- Update Game Logic ---
        self._update_ball()
        
        # --- Check Collisions and Game Events ---
        collision_reward = self._check_collisions()
        reward += collision_reward

        self._update_particles()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.MAX_SCORE:
                reward += 10 # Win bonus
            elif self.misses >= self.MAX_MISSES:
                reward -= 10 # Loss penalty
        
        # --- Frame Rate ---
        if self.auto_advance:
            self.clock.tick(30)
            
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.player_paddle.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.player_paddle.y += self.PADDLE_SPEED
        
        # Clamp paddle position to stay within screen bounds
        self.player_paddle.y = max(0, min(self.player_paddle.y, self.SCREEN_HEIGHT - self.PADDLE_HEIGHT))

    def _update_ball(self):
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

    def _check_collisions(self):
        reward = 0
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        # Paddle collision
        if self.player_paddle.colliderect(ball_rect) and self.ball_vel[0] < 0:
            # SFX: paddle_hit.wav
            reward += 0.1 # Reward for hitting the ball
            self.ball_vel[0] *= -1
            
            # Prevent ball from getting stuck in paddle
            self.ball_pos[0] = self.player_paddle.right + self.BALL_RADIUS

            # Add spin based on where the ball hits the paddle
            offset = (self.player_paddle.centery - self.ball_pos[1]) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[1] = -offset * self.ball_speed * 0.8
            self._normalize_ball_velocity()
            
            self._create_particles(self.ball_pos, 20)

        # Wall collisions (top/bottom)
        if self.ball_pos[1] - self.BALL_RADIUS <= 0 or self.ball_pos[1] + self.BALL_RADIUS >= self.SCREEN_HEIGHT:
            # SFX: wall_bounce.wav
            self.ball_vel[1] *= -1
            self.ball_pos[1] = max(self.BALL_RADIUS, min(self.ball_pos[1], self.SCREEN_HEIGHT - self.BALL_RADIUS))

        # Goal (right wall)
        if self.ball_pos[0] + self.BALL_RADIUS >= self.SCREEN_WIDTH:
            # SFX: score.wav
            self.score += 1
            reward += 1
            if self.score % 2 == 0 and self.score > 0:
                self.ball_speed += self.BALL_SPEED_INCREMENT
            if not self._check_termination():
                self._spawn_ball()

        # Miss (left wall)
        if self.ball_pos[0] - self.BALL_RADIUS <= 0:
            # SFX: miss.wav
            self.misses += 1
            reward -= 1
            if not self._check_termination():
                self._spawn_ball()
            
        return reward

    def _spawn_ball(self):
        self.ball_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        angle = self.rng.uniform(math.pi * 0.75, math.pi * 1.25) # Aim towards player
        self.ball_vel = [math.cos(angle) * self.ball_speed, math.sin(angle) * self.ball_speed]

    def _normalize_ball_velocity(self):
        magnitude = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if magnitude > 0:
            self.ball_vel[0] = (self.ball_vel[0] / magnitude) * self.ball_speed
            self.ball_vel[1] = (self.ball_vel[1] / magnitude) * self.ball_speed

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.rng.integers(10, 20)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime})
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        return (
            self.score >= self.MAX_SCORE or
            self.misses >= self.MAX_MISSES or
            self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw playfield boundaries (center line)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.SCREEN_WIDTH // 2, 0), (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT), 2)
        
        # Draw goal area
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (self.SCREEN_WIDTH - 10, 0, 10, self.SCREEN_HEIGHT))
        
        # Draw player paddle with a slight glow
        glow_rect = self.player_paddle.inflate(6, 6)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, glow_rect, border_radius=5)
        pygame.draw.rect(self.screen, (200, 220, 255), self.player_paddle, border_radius=3)

        # Draw ball with antialiasing and glow
        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, (*self.COLOR_BALL, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*self.COLOR_PARTICLE, alpha)
            size = int(max(1, p['life'] / 5))
            # Use a surface with SRCALPHA for proper blending
            particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, (size, size), size)
            self.screen.blit(particle_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 4 - score_text.get_width() // 2, 20))
        
        # Render misses
        misses_label = self.font_small.render("Misses:", True, self.COLOR_TEXT)
        self.screen.blit(misses_label, (self.SCREEN_WIDTH - 150, 20))
        for i in range(self.MAX_MISSES):
            color = self.COLOR_BALL if i < self.misses else (50, 50, 50)
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 50 + i * 20, 32), 6)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.score >= self.MAX_SCORE else "GAME OVER"
            message_surf = self.font_large.render(message, True, self.COLOR_TEXT)
            message_rect = message_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(message_surf, message_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "misses": self.misses,
            "steps": self.steps,
            "ball_speed": self.ball_speed,
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

# Example of how to run the environment for human play
if __name__ == '__main__':
    import time
    import os
    
    # Set SDL to a real video driver to see the window
    # Use "x11", "windows", "mac", etc. depending on your OS
    # This check is to prevent errors in environments without a display
    try:
        os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        os.environ["SDL_VIDEODRIVER"] = "x11"

    env = GameEnv()
    
    # Setup display window
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Pong")
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Key mapping for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    print(GameEnv.user_guide)
    print(GameEnv.game_description)

    # Main game loop
    running = True
    while running:
        # --- Human Input ---
        action = [0, 0, 0] # Default action: no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize first key found

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            time.sleep(2) # Pause before resetting
            obs, info = env.reset()
            total_reward = 0
            
    env.close()