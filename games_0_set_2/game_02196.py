
# Generated: 2025-08-27T19:36:06.982632
# Source Brief: brief_02196.md
# Brief Index: 2196

        
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
        "Controls: Use ↑ and ↓ to move the paddle. Use ← and → to cycle through colors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you must match your paddle's color to the incoming ball. "
        "Score points for correct matches and try to reach 7 points before you miss 3 balls."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_TEXT = (240, 240, 240)
    COLORS = {
        "red": (255, 80, 80),
        "green": (80, 255, 80),
        "blue": (80, 120, 255),
    }
    COLOR_LIST = list(COLORS.values())

    PADDLE_WIDTH, PADDLE_HEIGHT = 15, 80
    PADDLE_SPEED = 12
    PADDLE_X = 40
    
    BALL_RADIUS = 10
    INITIAL_BALL_SPEED = 3.0
    MAX_BALL_SPEED = 8.0
    BALL_SPEED_INCREASE_INTERVAL = 50
    BALL_SPEED_INCREASE_AMOUNT = 0.35

    MAX_STEPS = 1500
    WIN_SCORE = 7
    LOSE_MISSES = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 60)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.paddle_y = 0
        self.paddle_color_index = 0
        self.color_change_cooldown = 0
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.ball_color_index = 0
        self.ball_speed = 0
        self.particles = []
        self.hit_effect = {"timer": 0, "color": (0, 0, 0)}
        self.paddle_flash_effect = 0

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.paddle_y = self.SCREEN_HEIGHT // 2
        self.paddle_color_index = self.np_random.integers(0, len(self.COLOR_LIST))
        self.color_change_cooldown = 0
        
        self.ball_speed = self.INITIAL_BALL_SPEED
        self._spawn_ball()
        
        self.particles = []
        self.hit_effect = {"timer": 0, "color": (0,0,0)}
        self.paddle_flash_effect = 0
        
        return self._get_observation(), self._get_info()
    
    def _spawn_ball(self):
        self.ball_pos = [self.SCREEN_WIDTH * 0.6, self.np_random.uniform(self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)]
        angle = self.np_random.uniform(-math.pi / 5, math.pi / 5)
        self.ball_vel = [-self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)]
        self.ball_color_index = self.np_random.integers(0, len(self.COLOR_LIST))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        self._handle_input(action)
        reward += self._update_game_state()
        
        terminated = self._check_termination()
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 10
            elif self.misses >= self.LOSE_MISSES:
                reward += -10
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 1:  # Up
            self.paddle_y -= self.PADDLE_SPEED
        elif movement == 2:  # Down
            self.paddle_y += self.PADDLE_SPEED
        
        self.paddle_y = np.clip(self.paddle_y, self.PADDLE_HEIGHT / 2, self.SCREEN_HEIGHT - self.PADDLE_HEIGHT / 2)

        if self.color_change_cooldown == 0:
            if movement == 3:  # Left
                self.paddle_color_index = (self.paddle_color_index - 1) % len(self.COLOR_LIST)
                self.color_change_cooldown = 5
                self.paddle_flash_effect = 5
            elif movement == 4:  # Right
                self.paddle_color_index = (self.paddle_color_index + 1) % len(self.COLOR_LIST)
                self.color_change_cooldown = 5
                self.paddle_flash_effect = 5
        
        if self.color_change_cooldown > 0:
            self.color_change_cooldown -= 1

    def _update_game_state(self):
        step_reward = 0
        
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        self._update_particles()
        if self.hit_effect["timer"] > 0: self.hit_effect["timer"] -= 1
        if self.paddle_flash_effect > 0: self.paddle_flash_effect -= 1

        # Ball-wall collision
        if self.ball_pos[1] <= self.BALL_RADIUS or self.ball_pos[1] >= self.SCREEN_HEIGHT - self.BALL_RADIUS:
            self.ball_vel[1] *= -1
            self.ball_pos[1] = np.clip(self.ball_pos[1], self.BALL_RADIUS, self.SCREEN_HEIGHT - self.BALL_RADIUS)
            # // Play wall_bounce.wav
        
        # Ball-right wall collision
        if self.ball_pos[0] >= self.SCREEN_WIDTH - self.BALL_RADIUS:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = self.SCREEN_WIDTH - self.BALL_RADIUS
            # // Play wall_bounce.wav

        # Ball-paddle collision
        paddle_rect = pygame.Rect(self.PADDLE_X - self.PADDLE_WIDTH / 2, self.paddle_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        
        if self.ball_vel[0] < 0 and paddle_rect.clipline(self.ball_pos, (self.ball_pos[0] + self.ball_vel[0], self.ball_pos[1] + self.ball_vel[1])):
            self.ball_pos[0] = paddle_rect.right + self.BALL_RADIUS
            self.ball_vel[0] *= -1
            
            offset = (self.ball_pos[1] - self.paddle_y) / (self.PADDLE_HEIGHT / 2)
            self.ball_vel[1] += offset * 2.5
            
            # Normalize velocity
            current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
            if current_speed > 0:
                self.ball_vel[0] = (self.ball_vel[0] / current_speed) * self.ball_speed
                self.ball_vel[1] = (self.ball_vel[1] / current_speed) * self.ball_speed

            if self.paddle_color_index == self.ball_color_index:
                self.score += 1
                step_reward += 1.0
                self._create_particles(self.ball_pos, (255, 255, 255), 30)
                self.hit_effect = {"timer": 6, "color": (255, 255, 255)}
                # // Play score.wav
            else:
                step_reward += -0.1
                self._create_particles(self.ball_pos, (100, 100, 100), 15)
                self.hit_effect = {"timer": 6, "color": (100, 100, 100)}
                # // Play mismatch.wav
        
        # Ball miss
        if self.ball_pos[0] < 0:
            self.misses += 1
            step_reward += -1.0
            self._spawn_ball()
            self.hit_effect = {"timer": 8, "color": self.COLORS["red"]}
            # // Play miss.wav

        # Difficulty scaling
        if self.steps > 0 and self.steps % self.BALL_SPEED_INCREASE_INTERVAL == 0:
            new_speed = min(self.MAX_BALL_SPEED, self.ball_speed + self.BALL_SPEED_INCREASE_AMOUNT)
            if new_speed > self.ball_speed:
                current_speed = math.hypot(self.ball_vel[0], self.ball_vel[1])
                if current_speed > 0:
                    self.ball_vel[0] = (self.ball_vel[0] / current_speed) * new_speed
                    self.ball_vel[1] = (self.ball_vel[1] / current_speed) * new_speed
                self.ball_speed = new_speed
        
        return step_reward

    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.misses >= self.LOSE_MISSES or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_effects()
        self._render_game_elements()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_effects(self):
        if self.hit_effect["timer"] > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            flash_surface.set_alpha(self.hit_effect["timer"] * 25)
            flash_surface.fill(self.hit_effect["color"])
            self.screen.blit(flash_surface, (0, 0))

        for p in self.particles:
            color = p['color']
            alpha_color = (*color, p['lifespan'] * (255 / 20))
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_game_elements(self):
        # Draw ball with glow
        ball_color = self.COLOR_LIST[self.ball_color_index]
        self._draw_glow_circle(self.screen, self.ball_pos, self.BALL_RADIUS, ball_color)

        # Draw paddle with glow
        paddle_color = self.COLOR_LIST[self.paddle_color_index]
        paddle_rect = pygame.Rect(self.PADDLE_X - self.PADDLE_WIDTH / 2, self.paddle_y - self.PADDLE_HEIGHT / 2, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        self._draw_glow_rect(self.screen, paddle_rect, paddle_color, 4)
        
        if self.paddle_flash_effect > 0:
            s = pygame.Surface((self.PADDLE_WIDTH, self.PADDLE_HEIGHT), pygame.SRCALPHA)
            alpha = self.paddle_flash_effect * 50
            pygame.draw.rect(s, (255, 255, 255, alpha), s.get_rect(), border_radius=4)
            self.screen.blit(s, (paddle_rect.x, paddle_rect.y))

    def _render_ui(self):
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        misses_text = "● " * self.misses + "○ " * (self.LOSE_MISSES - self.misses)
        misses_surf = self.font.render(f"MISSES: {misses_text}", True, self.COLORS["red"])
        self.screen.blit(misses_surf, (self.SCREEN_WIDTH - misses_surf.get_width() - 20, 15))

        # Color indicators
        for i, color in enumerate(self.COLOR_LIST):
            rect = pygame.Rect(self.PADDLE_X - 20 + i * 25, self.SCREEN_HEIGHT - 30, 15, 15)
            self._draw_glow_rect(self.screen, rect, color, 2)
            if i == self.paddle_color_index:
                pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(4, 4), 2, border_radius=3)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLORS["green"] if self.score >= self.WIN_SCORE else self.COLORS["red"]
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "ball_speed": self.ball_speed,
        }

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = self.np_random.uniform(2, 5)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'color': color, 'lifespan': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['vel'][0] *= 0.98
            p['vel'][1] *= 0.98
            p['radius'] *= 0.96
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0.5]

    def _draw_glow_circle(self, surface, pos, radius, color):
        pos_int = (int(pos[0]), int(pos[1]))
        max_glow = int(radius * 1.5)
        for i in range(max_glow, 0, -2):
            alpha = int(100 * (1 - (i / max_glow))**2)
            if alpha > 0:
                glow_color = (*color, alpha)
                temp_surf = pygame.Surface((radius*2 + i*2, radius*2 + i*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, glow_color, (radius+i, radius+i), radius + i)
                surface.blit(temp_surf, (pos_int[0]-radius-i, pos_int[1]-radius-i))
        
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
    
    def _draw_glow_rect(self, surface, rect, color, border_radius):
        for i in range(5, 0, -1):
            alpha = 80 - i * 15
            glow_rect = rect.inflate(i * 2, i * 2)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*color, alpha), (0, 0, *glow_rect.size), border_radius=border_radius + i)
            surface.blit(shape_surf, glow_rect.topleft)
        pygame.draw.rect(surface, color, rect, border_radius=border_radius)
        
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # --- Action mapping for human play ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(env.FPS)

    env.close()