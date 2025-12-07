
# Generated: 2025-08-27T14:25:10.953223
# Source Brief: brief_00676.md
# Brief Index: 676

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Particle class for visual effects
class Particle:
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = np_random.integers(20, 40)
        self.color = color
        self.size = np_random.integers(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha = max(0, min(255, int(255 * (self.life / 40))))
        try:
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (self.size, self.size), self.size)
            surface.blit(temp_surf, (int(self.x) - self.size, int(self.y) - self.size))
        except (ValueError, TypeError):
            # Fallback if alpha or color is invalid, though it shouldn't be
            pass

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to move the paddle. Deflect the ball to score points."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced cyberpunk arcade game. Deflect the energy ball to score 10 points before you lose all 3 lives."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PADDLE_WIDTH, PADDLE_HEIGHT = 12, 80
    PADDLE_SPEED = 12
    BALL_RADIUS = 8
    BASE_BALL_SPEED = 8.0
    SPEED_INCREMENT = 1.0
    MAX_STEPS = 1500
    WIN_SCORE = 10
    MAX_LIVES = 3
    GRID_SPACING = 40

    # --- Colors (Cyberpunk Neon) ---
    COLOR_BG = (10, 10, 30)
    COLOR_GRID = (0, 50, 100)
    COLOR_PADDLE = (255, 0, 128)
    COLOR_BALL = (255, 150, 0)
    COLOR_SCORE = (50, 255, 50)
    COLOR_LIVES = (255, 50, 50)
    COLOR_PARTICLE = (255, 200, 100)

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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.paddle_rect = None
        self.ball_pos = None
        self.ball_vel = None
        self.current_ball_speed = self.BASE_BALL_SPEED
        self.ball_trail = []
        self.particles = []
        
        self.reset()
        # self.validate_implementation() # For development
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        # Paddle state
        paddle_x = self.SCREEN_WIDTH - 20 - self.PADDLE_WIDTH
        paddle_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) / 2
        self.paddle_rect = pygame.Rect(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)

        # Ball state
        self._reset_ball()

        # Visuals
        self.ball_trail = []
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _reset_ball(self, launch_towards_player=False):
        self.ball_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float64)
        
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        if not launch_towards_player:
             angle += math.pi # Launch towards left wall
        
        self.current_ball_speed = self.BASE_BALL_SPEED + (self.score // 2) * self.SPEED_INCREMENT
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * self.current_ball_speed
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0
        
        if self.game_over:
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        
        if movement == 1:  # Up
            self.paddle_rect.y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_rect.y += self.PADDLE_SPEED
        
        # Clamp paddle to screen
        self.paddle_rect.y = max(0, min(self.SCREEN_HEIGHT - self.PADDLE_HEIGHT, self.paddle_rect.y))

        # Update game logic
        self._update_ball()
        self._update_particles()
        
        # Continuous reward for when the ball is coming towards the player
        if self.ball_vel[0] > 0:
            reward += 0.01

        # Reward & State Changes from Ball Logic
        hit_info = self._handle_collisions()
        
        if hit_info["event"] == "hit":
            reward += 1.0
            self.score += 1
            if hit_info["risky"]:
                reward += 2.0
            # sfx: paddle_hit
            self._create_particle_burst(self.ball_pos[0], self.ball_pos[1])
        elif hit_info["event"] == "miss":
            reward -= 1.0
            self.lives -= 1
            # sfx: miss
            if self.lives > 0:
                self._reset_ball()
        elif hit_info["event"] == "wall_bounce":
            # sfx: wall_bounce
            pass

        self.steps += 1
        
        # Termination Check
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0
            elif self.lives <= 0:
                reward -= 100.0
            
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_ball(self):
        self.ball_trail.append(self.ball_pos.copy())
        if len(self.ball_trail) > 15:
            self.ball_trail.pop(0)
            
        self.ball_pos += self.ball_vel

    def _handle_collisions(self):
        ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)

        if self.ball_pos[1] <= self.BALL_RADIUS or self.ball_pos[1] >= self.SCREEN_HEIGHT - self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
            return {"event": "wall_bounce", "risky": False}

        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.BALL_RADIUS
            return {"event": "wall_bounce", "risky": False}

        if self.ball_pos[0] >= self.paddle_rect.left - self.BALL_RADIUS:
            if self.paddle_rect.colliderect(ball_rect):
                self.ball_vel[0] *= -1
                self.ball_pos[0] = self.paddle_rect.left - self.BALL_RADIUS

                offset = (self.paddle_rect.centery - self.ball_pos[1]) / (self.PADDLE_HEIGHT / 2)
                self.ball_vel[1] -= offset * 5
                
                self.current_ball_speed = self.BASE_BALL_SPEED + ((self.score + 1) // 2) * self.SPEED_INCREMENT
                norm = np.linalg.norm(self.ball_vel)
                if norm > 0: self.ball_vel = (self.ball_vel / norm) * self.current_ball_speed

                risky_zone = self.PADDLE_HEIGHT * 0.1
                is_risky = abs(self.paddle_rect.centery - self.ball_pos[1]) > (self.PADDLE_HEIGHT / 2 - risky_zone)
                
                return {"event": "hit", "risky": is_risky}
            
            elif self.ball_pos[0] > self.SCREEN_WIDTH:
                return {"event": "miss", "risky": False}

        return {"event": "none", "risky": False}

    def _check_termination(self):
        return self.lives <= 0 or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        for i, pos in enumerate(self.ball_trail):
            alpha = int(150 * (i / len(self.ball_trail)))
            radius = int(self.BALL_RADIUS * (i / len(self.ball_trail)))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, (*self.COLOR_BALL, alpha))

        ball_x, ball_y = int(self.ball_pos[0]), int(self.ball_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 5, (*self.COLOR_BALL, 50))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS + 2, (*self.COLOR_BALL, 100))
        pygame.gfxdraw.filled_circle(self.screen, ball_x, ball_y, self.BALL_RADIUS, self.COLOR_BALL)

        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=3)
        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in self.COLOR_PADDLE), self.paddle_rect, width=2, border_radius=3)

        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (20, 10))

        lives_surf = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_LIVES)
        self.screen.blit(lives_surf, (self.SCREEN_WIDTH - lives_surf.get_width() - 20, 10))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_SCORE if self.score >= self.WIN_SCORE else self.COLOR_LIVES
            
            game_over_surf = self.font_game_over.render(msg, True, color)
            pos = (self.SCREEN_WIDTH / 2 - game_over_surf.get_width() / 2, self.SCREEN_HEIGHT / 2 - game_over_surf.get_height() / 2)
            self.screen.blit(game_over_surf, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _create_particle_burst(self, x, y, count=20):
        for _ in range(count):
            self.particles.append(Particle(x, y, self.COLOR_PARTICLE, self.np_random))

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3) and isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    import os
    # Set the video driver to a working one on your system ('x11', 'windows', 'cocoa', etc.)
    # Or 'dummy' for headless operation.
    if os.name == 'posix':
        os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
    
    env = GameEnv()
    env.reset()
    
    try:
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Cyber Deflect")
    except pygame.error as e:
        print(f"Pygame display could not be initialized: {e}")
        print("Running in headless mode. No window will be shown.")
        display_screen = None

    terminated = False
    action = env.action_space.sample()
    action.fill(0)

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    running = True
    while running:
        if display_screen:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            action.fill(0)
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
        else: # No display, run with random actions for a bit
            action = env.action_space.sample()
            if env.steps > 2000: running = False

        obs, reward, terminated, truncated, info = env.step(action)
        
        if display_screen:
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated:
            print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")
            if display_screen:
                pygame.time.wait(3000)
            running = False

    env.close()