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
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    user_guide = (
        "Controls: Use ↑ to move your paddle up and ↓ to move it down. "
        "Try to survive as the ball gets faster."
    )

    game_description = (
        "A retro arcade game of grid-based Pong. The ball's speed constantly "
        "increases. Hitting the ball with the edges of your paddle gives a score "
        "bonus, but hitting it with the center is safer but incurs a small penalty. "
        "Survive for 60 seconds to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and play area dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # Grid dimensions
        self.GRID_W = 20
        self.GRID_H = 15
        
        # Sizing and offsets to center the play area
        self.CELL_W = 30
        self.CELL_H = 25
        self.PLAY_AREA_WIDTH = self.GRID_W * self.CELL_W
        self.PLAY_AREA_HEIGHT = self.GRID_H * self.CELL_H
        self.X_OFFSET = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.Y_OFFSET = (self.SCREEN_HEIGHT - self.PLAY_AREA_HEIGHT) // 2

        # Game constants
        self.PADDLE_H = 4  # in cells
        self.BALL_RADIUS = self.CELL_W // 2
        self.INITIAL_BALL_SPEED = 4.0 / 30.0 # cells per step
        self.BALL_SPEED_INCREASE = 0.75 / 30.0 # cells per step, every 5 seconds
        self.SPEED_INCREASE_INTERVAL = 5 * 30  # 5 seconds at 30fps
        self.MAX_STEPS = 60 * 30  # 60 seconds at 30fps

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_GRID = (20, 30, 80)
        self.COLOR_BALL = (255, 255, 0)
        self.COLOR_BALL_GLOW = (200, 200, 0, 64)
        self.COLOR_PADDLE_SAFE = (50, 255, 50)
        self.COLOR_PADDLE_RISKY = (255, 50, 50)
        self.COLOR_PADDLE_GLOW = (200, 200, 200, 48)
        self.COLOR_TEXT = (220, 220, 220)

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # For human rendering
        self.human_screen = None
        if render_mode == "human":
            pygame.display.set_caption("Grid Pong")
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        # Initialize the game state by calling reset(). This is necessary
        # because the validation logic below needs a valid state to run.
        self.reset()
        
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.paddle_y = (self.GRID_H - self.PADDLE_H) // 2
        self.ball_pos = np.array([self.GRID_W / 2, self.GRID_H / 2], dtype=float)
        
        self.ball_speed = self.INITIAL_BALL_SPEED
        angle = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        initial_vx = -self.ball_speed * math.cos(angle)
        initial_vy = self.ball_speed * math.sin(angle)
        self.ball_vel = np.array([initial_vx, initial_vy], dtype=float)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.01  # Survival reward

        # 1. Handle player input
        movement = action[0]
        if movement == 1:  # Up
            self.paddle_y -= 1
        elif movement == 2:  # Down
            self.paddle_y += 1
        
        # Clamp paddle position
        self.paddle_y = np.clip(self.paddle_y, 0, self.GRID_H - self.PADDLE_H)

        # 2. Update game state
        event_reward = self._update_ball()
        reward += event_reward
        self._update_particles()
        
        self.steps += 1
        self.score += reward

        # 3. Increase difficulty
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.ball_speed += self.BALL_SPEED_INCREASE
            self._normalize_ball_velocity()

        # 4. Check for termination
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        truncated = False
        if self.game_over:
            reward = -10.0 # Override reward on loss
            self.score += reward - event_reward # Adjust score to reflect final penalty
        elif self.steps >= self.MAX_STEPS:
            reward += 100.0 # Win bonus
            self.score += 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_ball(self):
        reward = 0.0
        
        # Store old position for collision checks
        old_pos = self.ball_pos.copy()
        self.ball_pos += self.ball_vel

        # Wall bounces (top/bottom)
        if self.ball_pos[1] < 0 or self.ball_pos[1] > self.GRID_H - 1:
            self.ball_pos[1] = np.clip(self.ball_pos[1], 0, self.GRID_H - 1)
            self.ball_vel[1] *= -1
            self._create_particles(self.ball_pos, self.COLOR_GRID, 10)
            # sfx: wall_bounce

        # Opponent wall bounce (right)
        if self.ball_pos[0] > self.GRID_W - 1:
            self.ball_pos[0] = self.GRID_W - 1
            self.ball_vel[0] *= -1
            self._create_particles(self.ball_pos, self.COLOR_GRID, 10)
            # sfx: wall_bounce

        # Paddle collision check
        # Check if ball crossed the paddle's line
        if old_pos[0] >= 1.0 and self.ball_pos[0] < 1.0:
            # Interpolate Y position at the paddle line (x=1.0)
            if (self.ball_pos[0] - old_pos[0]) != 0:
                intersect_y = old_pos[1] + (self.ball_pos[1] - old_pos[1]) * (1.0 - old_pos[0]) / (self.ball_pos[0] - old_pos[0])
            else:
                intersect_y = self.ball_pos[1]
            
            if self.paddle_y <= intersect_y < self.paddle_y + self.PADDLE_H:
                # Collision detected
                self.ball_pos[0] = 1.0
                
                # Determine hit zone and reward
                hit_pos_on_paddle = intersect_y - self.paddle_y
                if hit_pos_on_paddle < 1.0 or hit_pos_on_paddle >= self.PADDLE_H - 1.0:
                    reward += 2.0  # Risky return
                    particle_color = self.COLOR_PADDLE_RISKY
                    # sfx: risky_hit
                else:
                    reward -= 0.2  # Safe return
                    particle_color = self.COLOR_PADDLE_SAFE
                    # sfx: safe_hit

                # Change ball angle based on impact point
                relative_impact = (intersect_y - (self.paddle_y + self.PADDLE_H / 2)) / (self.PADDLE_H / 2)
                self.ball_vel[0] *= -1
                self.ball_vel[1] += relative_impact * 0.5 # Add some "spin"
                self._normalize_ball_velocity()
                
                self._create_particles(self.ball_pos, particle_color, 25)
            else:
                # Missed the paddle
                self.game_over = True
                # sfx: game_over
        
        # Check for loss
        if self.ball_pos[0] < 0:
            self.game_over = True
            # sfx: game_over
            
        return reward

    def _normalize_ball_velocity(self):
        current_speed = np.linalg.norm(self.ball_vel)
        if current_speed > 0:
            self.ball_vel = (self.ball_vel / current_speed) * self.ball_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'][1] += 0.01 # a little gravity

    def _create_particles(self, pos_grid, color, count):
        pos_pixels = self._grid_to_pixels(pos_grid)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos_pixels), 'vel': vel, 'life': life, 'color': color})

    def _grid_to_pixels(self, grid_pos):
        x = self.X_OFFSET + grid_pos[0] * self.CELL_W
        y = self.Y_OFFSET + grid_pos[1] * self.CELL_H
        return np.array([x, y])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for i in range(self.GRID_W + 1):
            x = self.X_OFFSET + i * self.CELL_W
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.Y_OFFSET), (x, self.Y_OFFSET + self.PLAY_AREA_HEIGHT))
        for i in range(self.GRID_H + 1):
            y = self.Y_OFFSET + i * self.CELL_H
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, y), (self.X_OFFSET + self.PLAY_AREA_WIDTH, y))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*color, alpha))

        # Render paddle
        paddle_x_px = self.X_OFFSET
        paddle_y_px = self.Y_OFFSET + self.paddle_y * self.CELL_H
        
        # Glow
        glow_rect = pygame.Rect(paddle_x_px, paddle_y_px, self.CELL_W, self.PADDLE_H * self.CELL_H)
        glow_rect.inflate_ip(12, 12)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_PADDLE_GLOW, s.get_rect(), border_radius=5)
        self.screen.blit(s, glow_rect.topleft)

        # Risky zones
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_RISKY, (paddle_x_px, paddle_y_px, self.CELL_W, self.CELL_H))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_RISKY, (paddle_x_px, paddle_y_px + (self.PADDLE_H - 1) * self.CELL_H, self.CELL_W, self.CELL_H))
        # Safe zones
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_SAFE, (paddle_x_px, paddle_y_px + self.CELL_H, self.CELL_W, (self.PADDLE_H - 2) * self.CELL_H))

        # Render ball
        if self.ball_pos is not None:
            ball_px, ball_py = self._grid_to_pixels(self.ball_pos)
            ball_px += self.CELL_W / 2
            ball_py += self.CELL_H / 2
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(ball_px), int(ball_py), int(self.BALL_RADIUS * 1.8), self.COLOR_BALL_GLOW)
            # Ball
            pygame.gfxdraw.aacircle(self.screen, int(ball_px), int(ball_py), self.BALL_RADIUS, self.COLOR_BALL)
            pygame.gfxdraw.filled_circle(self.screen, int(ball_px), int(ball_py), self.BALL_RADIUS, self.COLOR_BALL)

    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))

        speed_val = self.ball_speed * self.metadata['render_fps']
        speed_text = self.font_small.render(f"SPEED: {speed_val:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 15, 45))
        
        if self.game_over:
            status_text_str = "YOU WIN!" if self.steps >= self.MAX_STEPS else "GAME OVER"
            status_text = self.font_large.render(status_text_str, True, self.COLOR_PADDLE_RISKY if status_text_str == "GAME OVER" else self.COLOR_PADDLE_SAFE)
            text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(status_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ball_speed": self.ball_speed * self.metadata['render_fps'], # cells per second
        }

    def render(self):
        if self.metadata["render_modes"][0] == "rgb_array":
            return self._get_observation()
        elif self.metadata["render_modes"][0] == "human":
            obs = self._get_observation()
            if self.human_screen is not None:
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                self.human_screen.blit(surf, (0, 0))
                pygame.display.flip()
            return obs

    def close(self):
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


if __name__ == "__main__":
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Simple keyboard agent for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }
    
    while not done:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        action = [movement, 0, 0] # Space and shift are not used
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()
        env.clock.tick(env.metadata['render_fps'])

    print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
    env.close()