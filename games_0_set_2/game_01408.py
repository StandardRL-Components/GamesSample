
# Generated: 2025-08-27T17:02:39.263852
# Source Brief: brief_01408.md
# Brief Index: 1408

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to position your paddle. "
        "Survive the bouncing balls for 60 seconds."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Position your paddle to deflect an increasing "
        "number of speeding balls. Earn points and bonuses for successful deflections. "
        "You have 3 lives. Survive for 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 60
        
        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 55)
        self.COLOR_PADDLE = (0, 170, 255)
        self.COLOR_PADDLE_ACCENT = (100, 210, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        self.BALL_COLORS = [(255, 255, 0), (0, 255, 100), (255, 50, 50)] # Yellow, Green, Red
        
        # Play Area
        self.BORDER = 20
        self.PLAY_AREA_RECT = pygame.Rect(
            self.BORDER, self.BORDER,
            self.WIDTH - 2 * self.BORDER, self.HEIGHT - 2 * self.BORDER
        )
        
        # Grid
        self.GRID_DIMS = (10, 8)
        self.CELL_WIDTH = self.PLAY_AREA_RECT.width / self.GRID_DIMS[0]
        self.CELL_HEIGHT = self.PLAY_AREA_RECT.height / self.GRID_DIMS[1]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Verdana", 48, bold=True)
        self.font_bonus = pygame.font.SysFont("Verdana", 36, bold=True)
        
        # RNG
        self.np_random = None
        
        # Initialize state variables by calling reset
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.win = False
        self.bonus_multiplier = 1
        self.time_limit_steps = self.TIME_LIMIT_SECONDS * self.FPS
        
        # Paddle
        self.paddle_grid_pos = [self.GRID_DIMS[0] // 2, self.GRID_DIMS[1] - 1]
        self.paddle_rect = self._get_paddle_rect_from_grid()
        
        # Balls
        self.balls = []
        self.ball_spawn_timer = 0
        self.ball_spawn_interval = 2 * self.FPS # every 2 seconds
        self.max_balls = 10
        self.base_ball_speed = 3.0
        self.max_ball_speed = 8.0
        self.ball_speed_increase_per_frame = 0.005 / self.FPS
        
        # Effects
        self.particles = []
        
        # Initial setup
        self._spawn_ball()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
             return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        reward = 0.01 # Small reward for surviving a frame

        # 1. Handle Action
        self._handle_action(action)
        
        # 2. Update Game State
        event_reward = self._update_game_logic()
        reward += event_reward
        
        # 3. Check Termination
        terminated = self.lives <= 0 or self.steps >= self.time_limit_steps
        if terminated:
            self.game_over = True
            if self.steps >= self.time_limit_steps and self.lives > 0:
                self.win = True
                reward += 100.0 # Win bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement = action[0]
        
        if movement == 1: # Up
            self.paddle_grid_pos[1] -= 1
        elif movement == 2: # Down
            self.paddle_grid_pos[1] += 1
        elif movement == 3: # Left
            self.paddle_grid_pos[0] -= 1
        elif movement == 4: # Right
            self.paddle_grid_pos[0] += 1
            
        # Clamp paddle position to grid boundaries
        self.paddle_grid_pos[0] = max(0, min(self.GRID_DIMS[0] - 1, self.paddle_grid_pos[0]))
        self.paddle_grid_pos[1] = max(0, min(self.GRID_DIMS[1] - 1, self.paddle_grid_pos[1]))
        
        self.paddle_rect = self._get_paddle_rect_from_grid()

    def _update_game_logic(self):
        reward = 0.0
        reward += self._update_balls()
        self._update_particles()
        self._spawn_new_balls_if_needed()
        return reward

    def _update_balls(self):
        event_reward = 0.0
        current_ball_speed = min(self.max_ball_speed, self.base_ball_speed + self.steps * self.ball_speed_increase_per_frame)
        
        for ball in self.balls[:]:
            ball['trail'].append(ball['pos'].copy())
            
            # Update position
            normalized_vel = ball['vel'].normalize() if ball['vel'].length() > 0 else pygame.Vector2(0, 0)
            ball['pos'] += normalized_vel * current_ball_speed
            
            ball_rect = pygame.Rect(ball['pos'].x - ball['radius'], ball['pos'].y - ball['radius'], ball['radius']*2, ball['radius']*2)

            # Paddle collision
            if self.paddle_rect.colliderect(ball_rect):
                # sfx: paddle_hit
                event_reward += 1.0 + (0.1 * self.bonus_multiplier)
                self.score += 10 * self.bonus_multiplier
                self.bonus_multiplier += 1
                self._create_particles(ball['pos'], 20)

                # Reflect based on collision point
                center_dist = ball['pos'].x - self.paddle_rect.centerx
                influence = center_dist / (self.paddle_rect.width / 2)
                ball['vel'].y *= -1
                ball['vel'].x += influence * 2.0
                ball['vel'] = ball['vel'].normalize() if ball['vel'].length() > 0 else pygame.Vector2(self.np_random.uniform(-1, 1), -1).normalize()

                # Eject ball from paddle to prevent sticking
                if ball['vel'].y > 0: # Moving down
                    ball['pos'].y = self.paddle_rect.top - ball['radius'] - 1
                else: # Moving up
                    ball['pos'].y = self.paddle_rect.bottom + ball['radius'] + 1
            
            # Wall collisions
            if ball['pos'].x - ball['radius'] < self.PLAY_AREA_RECT.left:
                ball['pos'].x = self.PLAY_AREA_RECT.left + ball['radius']
                ball['vel'].x *= -1
            if ball['pos'].x + ball['radius'] > self.PLAY_AREA_RECT.right:
                ball['pos'].x = self.PLAY_AREA_RECT.right - ball['radius']
                ball['vel'].x *= -1
            if ball['pos'].y - ball['radius'] < self.PLAY_AREA_RECT.top:
                ball['pos'].y = self.PLAY_AREA_RECT.top + ball['radius']
                ball['vel'].y *= -1
                
            # Miss condition
            if ball['pos'].y + ball['radius'] > self.PLAY_AREA_RECT.bottom:
                # sfx: miss_ball
                self.lives -= 1
                event_reward -= 1.0
                self.bonus_multiplier = 1
                self._create_particles(ball['pos'], 30, (255, 80, 80))
                self.balls.remove(ball)
        
        return event_reward

    def _spawn_new_balls_if_needed(self):
        self.ball_spawn_timer += 1
        if self.ball_spawn_timer >= self.ball_spawn_interval and len(self.balls) < self.max_balls:
            self.ball_spawn_timer = 0
            self._spawn_ball()

    def _spawn_ball(self):
        pos_x = self.np_random.uniform(self.PLAY_AREA_RECT.left + 20, self.PLAY_AREA_RECT.right - 20)
        pos_y = self.np_random.uniform(self.PLAY_AREA_RECT.top + 20, self.PLAY_AREA_RECT.top + 50)
        
        angle = self.np_random.uniform(math.pi * 0.25, math.pi * 0.75) # Downward cone
        vel = pygame.Vector2(math.cos(angle), math.sin(angle))
        
        self.balls.append({
            'pos': pygame.Vector2(pos_x, pos_y),
            'vel': vel,
            'radius': 8,
            'trail': deque(maxlen=15)
        })

    def _create_particles(self, pos, count, color=None):
        if color is None:
            color = (255, 200, 0)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.uniform(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_paddle_rect_from_grid(self):
        px = self.PLAY_AREA_RECT.left + self.paddle_grid_pos[0] * self.CELL_WIDTH
        py = self.PLAY_AREA_RECT.top + self.paddle_grid_pos[1] * self.CELL_HEIGHT
        return pygame.Rect(px, py, self.CELL_WIDTH, self.CELL_HEIGHT)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_trails()
        self._render_balls()
        self._render_paddle()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        for x in range(self.GRID_DIMS[0] + 1):
            start_pos = (self.PLAY_AREA_RECT.left + x * self.CELL_WIDTH, self.PLAY_AREA_RECT.top)
            end_pos = (self.PLAY_AREA_RECT.left + x * self.CELL_WIDTH, self.PLAY_AREA_RECT.bottom)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for y in range(self.GRID_DIMS[1] + 1):
            start_pos = (self.PLAY_AREA_RECT.left, self.PLAY_AREA_RECT.top + y * self.CELL_HEIGHT)
            end_pos = (self.PLAY_AREA_RECT.right, self.PLAY_AREA_RECT.top + y * self.CELL_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.PLAY_AREA_RECT, 2)
    
    def _render_trails(self):
        for ball in self.balls:
            if len(ball['trail']) > 1:
                for i in range(len(ball['trail']) - 1):
                    alpha = (i / len(ball['trail'])) * 255
                    color = (*self.COLOR_PADDLE_ACCENT, int(alpha))
                    start_pos = (int(ball['trail'][i].x), int(ball['trail'][i].y))
                    end_pos = (int(ball['trail'][i+1].x), int(ball['trail'][i+1].y))
                    
                    temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(temp_surf, color, start_pos, end_pos, width=max(1, int(ball['radius'] * (i / len(ball['trail'])))))
                    self.screen.blit(temp_surf, (0,0))


    def _render_balls(self):
        current_ball_speed = min(self.max_ball_speed, self.base_ball_speed + self.steps * self.ball_speed_increase_per_frame)
        speed_ratio = (current_ball_speed - self.base_ball_speed) / max(1, self.max_ball_speed - self.base_ball_speed)
        
        if speed_ratio < 0.5:
            c1, c2 = self.BALL_COLORS[0], self.BALL_COLORS[1]
            interp_ratio = speed_ratio * 2
        else:
            c1, c2 = self.BALL_COLORS[1], self.BALL_COLORS[2]
            interp_ratio = (speed_ratio - 0.5) * 2
            
        color = tuple(int(c1[i] + (c2[i] - c1[i]) * interp_ratio) for i in range(3))

        for ball in self.balls:
            pos = (int(ball['pos'].x), int(ball['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], ball['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], ball['radius'], color)

    def _render_paddle(self):
        pygame.draw.rect(self.screen, self.COLOR_PADDLE, self.paddle_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PADDLE_ACCENT, self.paddle_rect.inflate(-6, -6), border_radius=4)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] / 6))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), size)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", (10, 5), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_str = f"TIME: {time_left:.1f}"
        text_width = self.font_ui.size(time_str)[0]
        self._render_text(time_str, (self.WIDTH - text_width - 10, 5), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Lives
        lives_str = "LIVES: " + "♥ " * self.lives
        lives_width = self.font_ui.size(lives_str)[0]
        self._render_text(lives_str, ((self.WIDTH - lives_width) // 2, self.HEIGHT - 25), self.font_ui, (255, 80, 80), self.COLOR_TEXT_SHADOW)

        # Bonus Multiplier
        if self.bonus_multiplier > 1:
            bonus_str = f"x{self.bonus_multiplier}"
            size_w, size_h = self.font_bonus.size(bonus_str)
            
            alpha = min(255, int(255 * ((self.bonus_multiplier - 1) / 10)))
            color = (255, 200, 0, alpha)

            temp_surf = pygame.Surface((size_w, size_h), pygame.SRCALPHA)
            text_surf = self.font_bonus.render(bonus_str, True, color)
            temp_surf.blit(text_surf, (0,0))
            
            self.screen.blit(temp_surf, ((self.WIDTH - size_w) // 2, (self.HEIGHT - size_h) // 2))

    def _render_text(self, text, pos, font, color, shadow_color):
        shadow_pos = (pos[0] + 2, pos[1] + 2)
        text_surf_shadow = font.render(text, True, shadow_color)
        self.screen.blit(text_surf_shadow, shadow_pos)
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        msg = "YOU WIN!" if self.win else "GAME OVER"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        text_surf = self.font_big.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "bonus": self.bonus_multiplier
        }
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Setup Pygame for display
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # Default to no-op
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Get key presses for this frame
        keys = pygame.key.get_pressed()
        
        # Reset movement action
        action[0] = 0
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize one key press per frame
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()
    print(f"Game Over! Final Info: {info}")