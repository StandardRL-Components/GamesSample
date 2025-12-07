import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:57:25.732487
# Source Brief: brief_02013.md
# Brief Index: 2013
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a skill-based arcade game.
    The player must juggle three balls with varying gravities to hit a series of
    recursively spawning targets against a time limit.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = "Juggle three balls with different physics to hit a series of moving targets before time runs out."
    user_guide = "Use ↑/↓ arrows to select a ball. Use ←/→ arrows to move it. Press space to give the selected ball an upward boost."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    DT = 1.0 / FPS

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_WALL = (40, 40, 60)
    COLOR_BALL_R = (255, 80, 80)
    COLOR_BALL_G = (80, 255, 80)
    COLOR_BALL_B = (80, 80, 255)
    COLOR_TARGET = (255, 255, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)

    # Physics
    GRAVITY = 980.0
    BALL_RADIUS_START = 15
    BALL_BOUNCE_DAMPING = 0.85
    IMPULSE_HORIZONTAL = 400.0
    IMPULSE_VERTICAL = 600.0

    # Game Rules
    MAX_EPISODE_SECONDS = 60
    MAX_STEPS = MAX_EPISODE_SECONDS * FPS
    TARGET_HIT_TIMEOUT_SECONDS = 3.0
    WIN_SCORE = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- Internal State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = 0.0
        self.target_hit_timeout = 0.0
        self.selected_ball_idx = 0
        self.balls = []
        self.targets = []
        self.particles = []
        self.last_action_ball_select = 0 # To make ball selection a discrete press event
        
        # --- Run Validation ---
        # self.validate_implementation() # Optional validation call

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_timer = self.MAX_EPISODE_SECONDS
        self.target_hit_timeout = self.TARGET_HIT_TIMEOUT_SECONDS
        self.selected_ball_idx = 0
        self.last_action_ball_select = 0

        # Initialize Balls
        self.balls = [
            self._create_ball(self.SCREEN_WIDTH * 0.25, self.COLOR_BALL_R, 1.0),
            self._create_ball(self.SCREEN_WIDTH * 0.50, self.COLOR_BALL_G, 0.5),
            self._create_ball(self.SCREEN_WIDTH * 0.75, self.COLOR_BALL_B, 2.0),
        ]

        # Initialize Target
        self.targets = [self._create_target()]
        
        # Initialize Particles
        self.particles = []

        return self._get_observation(), self._get_info()

    def _create_ball(self, x_pos, color, gravity_mult):
        return {
            "pos": pygame.Vector2(x_pos, self.SCREEN_HEIGHT / 2),
            "vel": pygame.Vector2(0, 0),
            "radius": self.BALL_RADIUS_START,
            "color": color,
            "gravity_mult": gravity_mult,
        }

    def _create_target(self, parent_target=None):
        if parent_target:
            size = max(10, parent_target["size"] * 0.8)
            speed_mult = 1.2
            pos = pygame.Vector2(parent_target["pos"])
        else: # Initial target
            size = 40
            speed_mult = 1.0
            pos = pygame.Vector2(self.np_random.uniform(size, self.SCREEN_WIDTH - size),
                                 self.np_random.uniform(size, self.SCREEN_HEIGHT * 0.6))
        
        return {
            "pos": pos,
            "size": size,
            "vel_x": self.np_random.choice([-1, 1]) * self.np_random.uniform(50, 80) * speed_mult,
            "sin_amplitude": self.np_random.uniform(20, 50),
            "sin_frequency": self.np_random.uniform(0.5, 1.5) * speed_mult,
            "sin_offset": self.np_random.uniform(0, 2 * math.pi),
            "initial_y": pos.y,
        }

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Timers ---
        self.steps += 1
        self.game_timer -= self.DT
        self.target_hit_timeout -= self.DT
        
        # --- Apply Actions ---
        self._apply_actions(action)

        # --- Update Game Logic ---
        self._update_balls()
        self._update_targets()
        self._update_particles()

        # --- Handle Collisions and Rewards ---
        reward = self._handle_collisions_and_rewards()

        # --- Check Termination Conditions ---
        terminated, term_reward = self._check_termination()
        if terminated:
            self.game_over = True
            reward += term_reward
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _apply_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Action 0: Ball selection (Up/Down) - discrete press
        current_ball_select_action = 0
        if movement == 1: current_ball_select_action = -1 # Up
        if movement == 2: current_ball_select_action = 1  # Down
        
        if current_ball_select_action != 0 and self.last_action_ball_select == 0:
            self.selected_ball_idx = (self.selected_ball_idx + current_ball_select_action) % len(self.balls)
            # sfx: ball_select_sound()
        self.last_action_ball_select = current_ball_select_action

        # Actions 1-4: Apply impulses to selected ball
        selected_ball = self.balls[self.selected_ball_idx]
        
        if movement == 3:  # Left
            selected_ball["vel"].x -= self.IMPULSE_HORIZONTAL * self.DT
        if movement == 4:  # Right
            selected_ball["vel"].x += self.IMPULSE_HORIZONTAL * self.DT
        if space_held:     # Upward impulse
            selected_ball["vel"].y -= self.IMPULSE_VERTICAL * self.DT

    def _update_balls(self):
        for ball in self.balls:
            # Apply gravity
            ball["vel"].y += self.GRAVITY * ball["gravity_mult"] * self.DT
            
            # Update position
            ball["pos"] += ball["vel"] * self.DT
            
            # Boundary collisions
            if ball["pos"].x - ball["radius"] < 0:
                ball["pos"].x = ball["radius"]
                ball["vel"].x *= -self.BALL_BOUNCE_DAMPING
            if ball["pos"].x + ball["radius"] > self.SCREEN_WIDTH:
                ball["pos"].x = self.SCREEN_WIDTH - ball["radius"]
                ball["vel"].x *= -self.BALL_BOUNCE_DAMPING
            if ball["pos"].y + ball["radius"] > self.SCREEN_HEIGHT:
                ball["pos"].y = self.SCREEN_HEIGHT - ball["radius"]
                ball["vel"].y *= -self.BALL_BOUNCE_DAMPING
            if ball["pos"].y - ball["radius"] < 0:
                ball["pos"].y = ball["radius"]
                ball["vel"].y *= -self.BALL_BOUNCE_DAMPING

    def _update_targets(self):
        for target in self.targets:
            target["pos"].x += target["vel_x"] * self.DT
            target["pos"].y = target["initial_y"] + target["sin_amplitude"] * math.sin(
                target["sin_frequency"] * self.steps * self.DT + target["sin_offset"]
            )
            
            # Screen wrap
            if target["pos"].x > self.SCREEN_WIDTH:
                target["pos"].x = 0
            if target["pos"].x < 0:
                target["pos"].x = self.SCREEN_WIDTH

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"] * self.DT
            p["lifespan"] -= 1

    def _handle_collisions_and_rewards(self):
        reward = 0
        
        # --- Ball-Target Collision ---
        for ball in self.balls:
            for target in self.targets[:]: # Iterate over a copy for safe removal
                ball_pos = ball["pos"]
                target_pos = target["pos"]
                target_half_size = target["size"] / 2
                
                # Simple AABB collision check
                if (target_pos.x - target_half_size < ball_pos.x < target_pos.x + target_half_size and
                    target_pos.y - target_half_size < ball_pos.y < target_pos.y + target_half_size):
                    
                    # sfx: target_hit_sound()
                    reward += 10.0 # Event-based reward
                    self.score += 1
                    self.target_hit_timeout = self.TARGET_HIT_TIMEOUT_SECONDS
                    
                    self._create_explosion(target["pos"], self.COLOR_TARGET)
                    
                    self.targets.remove(target)
                    if self.score < self.WIN_SCORE:
                        self.targets.append(self._create_target(parent_target=target))
                    break # A ball can only hit one target per frame
        
        # --- Proximity Reward ---
        min_dist = float('inf')
        if self.targets:
            for ball in self.balls:
                for target in self.targets:
                    dist = ball["pos"].distance_to(target["pos"])
                    if dist < min_dist:
                        min_dist = dist
            
            # Reward for being close, scaled by distance
            proximity_threshold = 100
            if min_dist < proximity_threshold:
                reward += 0.1 * (1 - min_dist / proximity_threshold)

        return reward

    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 200)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(10, 20),
                "max_lifespan": 20,
                "color": color,
            })

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            # sfx: win_sound()
            return True, 100.0 # Win
        if self.game_timer <= 0 or self.target_hit_timeout <= 0:
            # sfx: lose_sound()
            return True, -100.0 # Loss by timeout
        return False, 0.0

    def _get_observation(self):
        # --- Render to Pygame Surface ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        # --- Convert to Numpy Array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            color = (p["color"][0], p["color"][1], p["color"][2], alpha)
            size = int(5 * (p["lifespan"] / p["max_lifespan"]))
            if size > 0:
                rect = pygame.Rect(int(p["pos"].x - size/2), int(p["pos"].y - size/2), size, size)
                pygame.draw.rect(self.screen, color, rect)

        # Draw targets
        for target in self.targets:
            size = int(target["size"])
            rect = pygame.Rect(int(target["pos"].x - size/2), int(target["pos"].y - size/2), size, size)
            pygame.draw.rect(self.screen, self.COLOR_TARGET, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 2) # Outline

        # Draw balls
        for i, ball in enumerate(self.balls):
            pos = (int(ball["pos"].x), int(ball["pos"].y))
            radius = int(ball["radius"])
            
            # Glow effect
            glow_radius = int(radius * 1.8)
            glow_alpha = 60
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*ball["color"], glow_alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))
            
            # Main ball
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, ball["color"])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, ball["color"])
        
        # Draw selected ball indicator
        if self.balls:
            selected_ball = self.balls[self.selected_ball_idx]
            pos = (int(selected_ball["pos"].x), int(selected_ball["pos"].y))
            
            # Pulsating ring
            pulse_rad = selected_ball["radius"] + 8 + 4 * math.sin(self.steps * self.DT * 8)
            pulse_alpha = 150 + 100 * math.sin(self.steps * self.DT * 8)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(pulse_rad), (*self.COLOR_WHITE, int(pulse_alpha)))

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"{self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer display
        timer_text = self.font_large.render(f"{self.game_timer:.1f}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Target hit timeout bar
        timeout_ratio = max(0, self.target_hit_timeout / self.TARGET_HIT_TIMEOUT_SECONDS)
        bar_width = 200
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 20
        
        pygame.draw.rect(self.screen, self.COLOR_WALL, (bar_x, bar_y, bar_width, bar_height))
        fill_color = self.COLOR_TARGET if timeout_ratio > 0.25 else self.COLOR_BALL_R
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * timeout_ratio, bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_timer": self.game_timer,
            "target_hit_timeout": self.target_hit_timeout,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    # Set the video driver to a real one for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Juggler")
    clock = pygame.time.Clock()

    while running:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to Screen ---
        # The observation is already a rendered image, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for reset
            
        clock.tick(GameEnv.FPS)

    env.close()