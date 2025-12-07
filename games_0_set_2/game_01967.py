
# Generated: 2025-08-28T03:14:40.880425
# Source Brief: brief_01967.md
# Brief Index: 1967

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: ↑/↓ to move the paddle."
    )

    game_description = (
        "Survive 60 seconds of escalating paddle-ball chaos. Hit the ball with your paddle, "
        "collect power-ups, and don't let the ball get past you. The ball gets faster over time!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and rendering setup
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        
        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame initialization
        pygame.init()
        pygame.font.init()

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_WALL = (80, 80, 100)
        self.COLOR_PADDLE = (200, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.POWERUP_COLORS = {
            "slow_ball": (50, 150, 255),
            "enlarge_paddle": (200, 50, 255),
            "extra_life": (255, 180, 50),
        }

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_end = pygame.font.SysFont("monospace", 48, bold=True)

        # Game constants
        self.MAX_STEPS = 3600  # 60 seconds at 60 FPS
        self.PADDLE_WIDTH = 12
        self.PADDLE_X_POS = 20
        self.PADDLE_SPEED = 7
        self.DEFAULT_PADDLE_HEIGHT = 70
        self.BALL_RADIUS = 7
        self.INITIAL_BALL_SPEED = 3.5
        self.POWERUP_SIZE = 15
        self.POWERUP_SPAWN_PROB = 0.006
        self.POWERUP_DURATION = 480  # 8 seconds

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.paddle_y = 0
        self.paddle_height = 0
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.ball_speed_multiplier = 1.0
        self.particles = []
        self.powerups = []
        self.active_powerup_effects = {}
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.lives = 1
        self.game_over = False
        self.win = False

        self.paddle_y = self.HEIGHT / 2
        self.paddle_height = self.DEFAULT_PADDLE_HEIGHT

        self.ball_pos = np.array([self.WIDTH / 3, self.HEIGHT / 2], dtype=float)
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)], dtype=float) * self.INITIAL_BALL_SPEED
        self.ball_speed_multiplier = 1.0

        self.particles = []
        self.powerups = []
        self.active_powerup_effects = {
            "slow_ball": 0,
            "enlarge_paddle": 0,
        }

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # 1. Handle player input
            self._handle_input(action)

            # 2. Update game state
            reward += self._update_ball()
            self._update_powerups()
            self._update_particles()
            self._update_active_effects()

            # 3. Increase difficulty over time
            if self.steps > 0 and self.steps % 600 == 0:
                self.ball_speed_multiplier += 0.1
                self._update_ball_speed()
            
            self.steps += 1
        
        # 4. Check for termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over: # Win condition
            self.win = True
            reward += 100

        # 5. Return standard Gymnasium 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement = action[0]
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        
        self.paddle_y = np.clip(self.paddle_y, self.paddle_height / 2, self.HEIGHT - self.paddle_height / 2)

    def _update_ball(self):
        reward = 0
        self.ball_pos += self.ball_vel

        # Wall collisions
        if self.ball_pos[1] <= self.BALL_RADIUS or self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.HEIGHT - self.BALL_RADIUS)
            # sfx: wall_bounce.wav

        if self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            # sfx: wall_bounce.wav

        # Paddle collision
        paddle_rect = pygame.Rect(self.PADDLE_X_POS, self.paddle_y - self.paddle_height / 2, self.PADDLE_WIDTH, self.paddle_height)
        if self.ball_vel[0] < 0 and self.ball_pos[0] - self.BALL_RADIUS < paddle_rect.right:
            ball_rect = pygame.Rect(self.ball_pos[0] - self.BALL_RADIUS, self.ball_pos[1] - self.BALL_RADIUS, self.BALL_RADIUS * 2, self.BALL_RADIUS * 2)
            if paddle_rect.colliderect(ball_rect):
                self.ball_vel[0] *= -1
                self.ball_pos[0] = paddle_rect.right + self.BALL_RADIUS
                
                # Add "spin" based on impact point
                offset = (self.ball_pos[1] - self.paddle_y) / (self.paddle_height / 2)
                self.ball_vel[1] += offset * 2.5 # More spin makes game more dynamic
                
                self._update_ball_speed()
                self._create_particles(self.ball_pos, self._get_ball_color())
                self.score += 1
                reward += 0.1
                # sfx: paddle_hit.wav

        # Ball miss (failure)
        if self.ball_pos[0] < 0:
            self.lives -= 1
            if self.lives < 0:
                self.game_over = True
                reward -= 2
                # sfx: game_over.wav
            else:
                self.score = max(0, self.score - 10)
                self.reset_ball()
                # sfx: lose_life.wav
        
        return reward

    def _update_powerups(self):
        # Spawn new powerups
        if self.np_random.random() < self.POWERUP_SPAWN_PROB and len(self.powerups) < 3:
            ptype = self.np_random.choice(list(self.POWERUP_COLORS.keys()))
            pos = [self.np_random.integers(self.WIDTH // 4, self.WIDTH * 3 // 4), self.np_random.integers(50, self.HEIGHT - 50)]
            self.powerups.append({"pos": pos, "type": ptype, "life": 720}) # 12 seconds to disappear
        
        paddle_rect = pygame.Rect(self.PADDLE_X_POS, self.paddle_y - self.paddle_height / 2, self.PADDLE_WIDTH, self.paddle_height)

        # Update and check for collection
        for p in self.powerups[:]:
            p["life"] -= 1
            powerup_rect = pygame.Rect(p["pos"][0] - self.POWERUP_SIZE/2, p["pos"][1] - self.POWERUP_SIZE/2, self.POWERUP_SIZE, self.POWERUP_SIZE)
            if paddle_rect.colliderect(powerup_rect):
                self._activate_powerup(p["type"])
                self.score += 25
                self.powerups.remove(p)
                # sfx: powerup_collect.wav
            elif p["life"] <= 0:
                self.powerups.remove(p)

    def _activate_powerup(self, ptype):
        if ptype == "extra_life":
            self.lives += 1
        elif ptype == "slow_ball":
            self.active_powerup_effects["slow_ball"] = self.POWERUP_DURATION
            self._update_ball_speed()
        elif ptype == "enlarge_paddle":
            self.active_powerup_effects["enlarge_paddle"] = self.POWERUP_DURATION
            self.paddle_height = self.DEFAULT_PADDLE_HEIGHT * 1.5
    
    def _update_active_effects(self):
        # Slow ball
        if self.active_powerup_effects["slow_ball"] > 0:
            self.active_powerup_effects["slow_ball"] -= 1
            if self.active_powerup_effects["slow_ball"] == 0:
                self._update_ball_speed()
        # Enlarge paddle
        if self.active_powerup_effects["enlarge_paddle"] > 0:
            self.active_powerup_effects["enlarge_paddle"] -= 1
            if self.active_powerup_effects["enlarge_paddle"] == 0:
                self.paddle_height = self.DEFAULT_PADDLE_HEIGHT

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(20, 40),
                "color": color,
                "radius": self.np_random.uniform(1, 3)
            })

    def reset_ball(self):
        self.ball_pos = np.array([self.WIDTH / 3, self.paddle_y], dtype=float)
        angle = self.np_random.uniform(-math.pi / 6, math.pi / 6)
        self.ball_vel = np.array([math.cos(angle), math.sin(angle)])
        self.ball_speed_multiplier = max(1.0, self.ball_speed_multiplier - 0.2) # Penalty
        self._update_ball_speed()

    def _update_ball_speed(self):
        current_speed = np.linalg.norm(self.ball_vel)
        if current_speed == 0: return

        target_speed = self.INITIAL_BALL_SPEED * self.ball_speed_multiplier
        if self.active_powerup_effects["slow_ball"] > 0:
            target_speed *= 0.6

        self.ball_vel = self.ball_vel * (target_speed / current_speed)

    def _get_ball_color(self):
        if self.ball_speed_multiplier > 1.4:
            return (255, 50, 50)  # Red
        elif self.ball_speed_multiplier > 1.15:
            return (255, 255, 50)  # Yellow
        return (50, 255, 50)  # Green

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, 0), (self.WIDTH, 0), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (self.WIDTH - 1, 0), (self.WIDTH - 1, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, self.HEIGHT - 1), (self.WIDTH, self.HEIGHT - 1), 2)

        # Draw powerups
        for p in self.powerups:
            color = self.POWERUP_COLORS[p["type"]]
            alpha = int(255 * (p["life"] / 720)) if p["life"] < 120 else 255
            glow_color = (*color, alpha)
            rect = pygame.Rect(p["pos"][0] - self.POWERUP_SIZE / 2, p["pos"][1] - self.POWERUP_SIZE / 2, self.POWERUP_SIZE, self.POWERUP_SIZE)
            pygame.draw.rect(self.screen, glow_color, rect, border_radius=3)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, 255 * (p["life"] / 40))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw paddle
        paddle_rect = pygame.Rect(self.PADDLE_X_POS, int(self.paddle_y - self.paddle_height / 2), self.PADDLE_WIDTH, int(self.paddle_height))
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, paddle_rect, border_radius=3)

        # Draw ball with glow
        ball_color = self._get_ball_color()
        self._render_glow_circle(self.screen, self.ball_pos, self.BALL_RADIUS, ball_color)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 60)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 15, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH / 2 - lives_text.get_width() / 2, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            color = (255, 50, 50)
        elif self.win:
            msg = "YOU SURVIVED!"
            color = (50, 255, 50)
        else:
            return

        end_text = self.font_end.render(msg, True, color)
        text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(end_text, text_rect)

    def _render_glow_circle(self, surf, center, radius, color):
        center_int = (int(center[0]), int(center[1]))
        
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 40 - i * 8
            glow_radius = radius + i * 2.5
            glow_color = (*color, alpha)
            
            temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            surf.blit(temp_surf, (center_int[0] - glow_radius, center_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main ball
        pygame.gfxdraw.aacircle(surf, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surf, center_int[0], center_int[1], int(radius), color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.lives}

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # Override Pygame screen for human rendering
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Paddle Chaos")

    terminated = False
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op for movement
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True

        # Human controls
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render to the display
        rendered_frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(env.screen, rendered_frame)
        pygame.display.flip()

        env.clock.tick(60) # Run at 60 FPS

    env.close()